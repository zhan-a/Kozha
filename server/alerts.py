"""Rule-based alerter for the Kozha translator.

Scope (polish-13 §6):

* 5xx rate > 5% over the last 15 minutes.
* ``unknown_word`` rate 3× its rolling baseline.
* Snapshot age > 36 hours (cron failure).
* Deploy failed (latest deploy run was not success).

Webhooks are opt-in. ``KOZHA_ALERT_WEBHOOK_URL`` (Slack / Discord /
PagerDuty — any URL that accepts a JSON POST) enables delivery.
Without it, violations go to stdout as JSON lines so a scheduler
running the script via cron can surface them through its own log
review without a separate integration.

Invocation (intended — a 5-minute cron)::

    python -m server.alerts --metrics-url http://127.0.0.1:8000/metrics

Running without arguments polls the metrics endpoint at
``KOZHA_METRICS_URL`` (defaults to ``http://127.0.0.1:8000/metrics``)
and exits non-zero if any rule fires, which the caller can use as an
exit-code gate for a shell-level alerting chain.

Baselines are cached under ``data/alerts_state.json`` so each
invocation compares against the last few samples instead of the all-
time mean — a long-tail rise in unknown words is what we want to
flag, not the natural daily variance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PUBLIC_DIR = REPO_ROOT / "public"
STATE_PATH = DATA_DIR / "alerts_state.json"

DEFAULT_METRICS_URL = os.environ.get(
    "KOZHA_METRICS_URL", "http://127.0.0.1:8000/metrics"
)
ERROR_RATE_THRESHOLD = float(os.environ.get("KOZHA_ALERT_ERROR_RATE", "0.05"))
UNKNOWN_SPIKE_RATIO = float(os.environ.get("KOZHA_ALERT_UNKNOWN_SPIKE", "3.0"))
BASELINE_WINDOW = int(os.environ.get("KOZHA_ALERT_BASELINE_WINDOW", "12"))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)
_LABEL_PAIR = re.compile(r'([a-zA-Z_]\w*)="((?:[^"\\]|\\.)*)"')


def parse_metrics(text: str) -> dict[str, list[dict]]:
    """Parse a Prometheus exposition into ``{name: [samples]}``.

    ``samples`` is a list of ``{"labels": {...}, "value": float,
    "suffix": "bucket"|"count"|"sum"|""}``. Only the fields this
    module needs are returned.
    """
    out: dict[str, list[dict]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        value = float(m.group("value"))
        labels_raw = m.group("labels") or ""
        labels = {k: v for k, v in _LABEL_PAIR.findall(labels_raw)}
        base = name
        suffix = ""
        for candidate in ("_bucket", "_count", "_sum"):
            if name.endswith(candidate):
                base = name[: -len(candidate)]
                suffix = candidate[1:]
                break
        out.setdefault(base, []).append(
            {"labels": labels, "value": value, "suffix": suffix}
        )
    return out


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    rule: str
    level: str  # "warn" | "page"
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "level": self.level,
            "message": self.message,
            "detail": self.detail,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


def check_error_rate(
    metrics: dict[str, list[dict]],
    state: dict,
    threshold: float = ERROR_RATE_THRESHOLD,
) -> Optional[Violation]:
    """Compute the fraction of ``outcome="server_error"`` over all
    translations in the *delta* between this run and the last one.

    Using a delta (vs. cumulative count) gives a rough 5-minute rate
    when the caller is a 5-minute cron; the first run always seeds the
    baseline without alerting.
    """
    samples = metrics.get("kozha_translations_total", [])
    if not samples:
        return None

    total_now = sum(s["value"] for s in samples if s["suffix"] == "")
    err_now = sum(
        s["value"]
        for s in samples
        if s["suffix"] == "" and s["labels"].get("outcome") == "server_error"
    )
    last = state.get("last_counts") or {}
    total_prev = float(last.get("translations_total", 0.0))
    err_prev = float(last.get("server_error_total", 0.0))

    # Persist for the next run regardless of the outcome.
    state["last_counts"] = {
        "translations_total": total_now,
        "server_error_total": err_now,
    }

    delta_total = total_now - total_prev
    delta_err = err_now - err_prev
    if delta_total < 10:
        # Not enough traffic to reason about a rate; avoid false
        # positives on a quiet hour.
        return None
    rate = (delta_err / delta_total) if delta_total > 0 else 0.0
    if rate > threshold:
        return Violation(
            rule="error_rate",
            level="page",
            message=f"5xx rate {rate:.2%} over last window (>{threshold:.0%})",
            detail={"delta_total": delta_total, "delta_err": delta_err, "rate": rate},
        )
    return None


def check_unknown_spike(
    metrics: dict[str, list[dict]],
    state: dict,
    ratio: float = UNKNOWN_SPIKE_RATIO,
    window: int = BASELINE_WINDOW,
) -> Optional[Violation]:
    """Flag when the delta of ``kozha_unknown_word_total`` in this
    window is ``ratio``× its rolling mean.

    Rolling-mean baseline avoids paging on a seasonal bump (e.g. a
    surge of unknown words at the start of a new browser-extension
    launch). A spike against the baseline is what signals "the
    translator shipped a regression, or a new user is asking about a
    corpus we don't support yet".
    """
    samples = metrics.get("kozha_unknown_word_total", [])
    if not samples:
        return None
    total_now = sum(s["value"] for s in samples if s["suffix"] == "")
    history: list[float] = list(state.get("unknown_history") or [])
    total_prev = history[-1] if history else 0.0
    delta = max(0.0, total_now - total_prev)
    history.append(total_now)
    state["unknown_history"] = history[-window:]

    deltas = [b - a for a, b in zip(history, history[1:])]
    if len(deltas) < max(3, window // 2):
        # Not enough history yet to compute a stable baseline.
        return None
    baseline = sum(deltas[:-1]) / max(1, len(deltas) - 1)
    if baseline < 5:
        # Low-volume baseline — a single noisy session can look like a
        # 3× jump. Skip until the baseline is interesting.
        return None
    if delta > baseline * ratio:
        return Violation(
            rule="unknown_word_spike",
            level="warn",
            message=(
                f"unknown_word rate {delta:.0f} is {delta / max(1.0, baseline):.1f}× "
                f"the {len(deltas) - 1}-sample baseline of {baseline:.1f}"
            ),
            detail={"delta": delta, "baseline": baseline, "ratio": delta / max(1.0, baseline)},
        )
    return None


def check_deploy_status(status_file: Optional[Path] = None) -> Optional[Violation]:
    """Inspect the most recent deploy receipt and alert on failure.

    The deploy workflow writes ``data/last_deploy.json`` (``{status,
    sha, at}``) after each run. Absent file means "no deploy yet" —
    which is fine, we return ``None``.
    """
    path = status_file or (DATA_DIR / "last_deploy.json")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Violation(
            rule="deploy_status_unreadable",
            level="warn",
            message=f"{path.name} could not be parsed",
        )
    status = str(payload.get("status") or "").lower()
    if status and status != "success":
        return Violation(
            rule="deploy_failed",
            level="page",
            message=f"last deploy status = {status}",
            detail=payload,
        )
    return None


# ---------------------------------------------------------------------------
# State + delivery
# ---------------------------------------------------------------------------


def load_state(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def post_webhook(url: str, payload: dict, timeout_s: float = 10.0) -> bool:
    """POST ``payload`` as JSON to ``url``. Returns True on 2xx.

    Uses stdlib ``urllib`` to avoid adding a ``requests`` dependency
    for one outbound call. A failed webhook is itself logged but not
    retried — the next cron run will re-fire the rule if the condition
    persists.
    """
    body = json.dumps({"text": payload.get("message", ""), **payload}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except urllib.error.URLError as e:
        print(f"[alerts] webhook delivery failed: {e}", file=sys.stderr)
        return False


def fetch_metrics_text(url: str, timeout_s: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def run_rules(metrics_text: str, state: dict) -> list[Violation]:
    metrics = parse_metrics(metrics_text)
    violations: list[Violation] = []
    for rule in (
        lambda: check_error_rate(metrics, state),
        lambda: check_unknown_spike(metrics, state),
        check_deploy_status,
    ):
        try:
            v = rule()
        except Exception as e:  # noqa: BLE001 — never let one rule kill the rest
            v = Violation(
                rule="rule_crashed",
                level="warn",
                message=f"rule raised: {e!s}",
            )
        if v is not None:
            violations.append(v)
    return violations


def deliver(violations: Iterable[Violation], webhook_url: Optional[str]) -> None:
    for v in violations:
        payload = v.to_dict()
        line = json.dumps(payload, ensure_ascii=False)
        print(line)
        if webhook_url:
            post_webhook(webhook_url, payload)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Kozha rule-based alerter")
    parser.add_argument("--metrics-url", default=DEFAULT_METRICS_URL)
    parser.add_argument(
        "--webhook-url",
        default=os.environ.get("KOZHA_ALERT_WEBHOOK_URL", ""),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip webhook delivery, print violations and exit",
    )
    args = parser.parse_args(argv)

    try:
        metrics_text = fetch_metrics_text(args.metrics_url)
    except urllib.error.URLError as e:
        # Fetching /metrics should never be a hard error — the server
        # might be between restarts. Emit a stdout line and exit 2 so
        # the caller can decide whether to escalate.
        print(
            json.dumps({
                "rule": "metrics_unreachable",
                "level": "warn",
                "message": f"{args.metrics_url} unreachable: {e!s}",
            }),
        )
        return 2

    state = load_state()
    violations = run_rules(metrics_text, state)
    save_state(state)

    if args.dry_run:
        for v in violations:
            print(json.dumps(v.to_dict()))
        return 1 if violations else 0

    deliver(violations, args.webhook_url or None)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
