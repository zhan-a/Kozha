"""Observability primitives for the Kozha translator.

Small, dependency-free module covering:

* Structured JSON logging — an ``Extra`` context dict per record that
  includes request id, source/target language, latency, and outcome.
  No user-supplied text is logged unless ``KOZHA_LOG_VERBOSE=1`` in the
  environment (disabled in production).
* Prometheus text exposition for the metrics names documented in
  ``prompts/prompt-polish-13.md`` §3. Implemented inline to avoid
  adding a ``prometheus_client`` runtime dependency — the exposition
  format is stable and tiny.
* IP hashing with a per-process salt so logs cannot be joined back to
  a raw IP across restarts. The salt is overrideable via
  ``KOZHA_IP_HASH_SALT`` for multi-replica deployments that want
  stable hashes.
* A ``readiness_probe`` function the ``/health/ready`` handler calls.

The module is import-safe: loading it does not open any sockets, read
any data files, or install logging handlers. Callers (server.py, the
test suite, the CLI benchmark) opt in explicitly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Fixed set of outcome labels. Kept small and closed so the metric's
# cardinality stays bounded (source_lang × sign_lang × outcome is the
# worst case; anything unbounded here breaks Prometheus query plans).
OUTCOMES: tuple[str, ...] = (
    "success",
    "missing_gloss",
    "validation_error",
    "server_error",
)


# ---------------------------------------------------------------------------
# IP hashing
# ---------------------------------------------------------------------------

_IP_SALT: str = os.environ.get("KOZHA_IP_HASH_SALT") or secrets.token_hex(16)


def hash_ip(raw: Optional[str]) -> str:
    """Return a short hex digest of ``raw`` IP salted with a process secret.

    Never returns the raw IP. An empty / ``None`` input produces an
    empty string so log lines don't carry a placeholder hash that looks
    meaningful. Digest truncated to 12 hex chars — long enough for
    abuse detection, short enough to keep log lines readable.
    """
    if not raw:
        return ""
    return hashlib.sha256((_IP_SALT + raw).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


_LOG_VERBOSE = os.environ.get("KOZHA_LOG_VERBOSE", "").strip() in {"1", "true", "yes"}


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a one-line JSON object.

    The record's ``extra={"ctx": {...}}`` dict (set by ``log_request``)
    is merged into the top-level payload so operators can filter by
    ``request_id`` or ``outcome`` without parsing a nested object.
    User-supplied text fields in ``ctx`` are dropped unless
    ``KOZHA_LOG_VERBOSE=1`` — the default is privacy-preserving.
    """

    _SENSITIVE_KEYS = frozenset({"source_text", "raw", "text"})

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(record.created),
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            for k, v in ctx.items():
                if not _LOG_VERBOSE and k in self._SENSITIVE_KEYS:
                    continue
                payload[k] = v
        if record.exc_info and record.exc_info[0]:
            # Include the exception class name (bounded cardinality);
            # the stack trace itself stays out of the log line because
            # it can carry contextual user data embedded in locals.
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str)


_LOGGING_CONFIGURED = False
_LOGGING_LOCK = threading.Lock()


def configure_logging(level: str = "INFO", stream=None) -> None:
    """Attach a single JSON handler to the root logger. Idempotent.

    Callers: ``server.server`` on import, and the test suite via
    ``reset_for_tests`` (see ``conftest``). Multiple calls replace
    handlers instead of stacking them, which keeps the test output
    clean.
    """
    global _LOGGING_CONFIGURED
    with _LOGGING_LOCK:
        root = logging.getLogger()
        # Remove previous handlers we own to avoid duplicate lines.
        for h in list(root.handlers):
            if getattr(h, "_kozha_obs_handler", False):
                root.removeHandler(h)
        handler = logging.StreamHandler(stream=stream or sys.stdout)
        handler.setFormatter(JsonFormatter())
        setattr(handler, "_kozha_obs_handler", True)
        root.addHandler(handler)
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        _LOGGING_CONFIGURED = True


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    path: str,
    source_lang: str,
    target_sign_lang: str,
    source_text_length: int,
    cache_hit: bool,
    latency_ms: float,
    outcome: str,
    source_text: Optional[str] = None,
    ip_hash: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Emit one ``/translate``-style request line.

    ``source_text`` is only included in the payload when the operator
    has flipped ``KOZHA_LOG_VERBOSE=1``. That flag is off by default
    and remains off in production; the log line carries
    ``source_text_length`` instead so post-hoc analysis can reason
    about payload size without exposing user content.
    """
    ctx: dict[str, object] = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "source_lang": source_lang,
        "target_sign_lang": target_sign_lang,
        "source_text_length": source_text_length,
        "cache_hit": cache_hit,
        "latency_ms": round(latency_ms, 2),
        "outcome": outcome,
    }
    if ip_hash:
        ctx["ip_hash"] = ip_hash
    if source_text is not None:
        # Filtered inside JsonFormatter unless verbose.
        ctx["source_text"] = source_text
    logger.log(level, "request", extra={"ctx": ctx})


def new_request_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Metrics — minimal Prometheus text exposition
# ---------------------------------------------------------------------------


@dataclass
class _MetricSample:
    value: float = 0.0
    # Histogram-only:
    bucket_counts: dict[float, int] = field(default_factory=dict)
    count: int = 0
    sum_: float = 0.0


def _escape_label_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
    )


def _format_labels(names: tuple[str, ...], values: tuple[str, ...]) -> str:
    if not names:
        return ""
    pairs = ",".join(
        f'{n}="{_escape_label_value(str(v))}"' for n, v in zip(names, values)
    )
    return "{" + pairs + "}"


def _with_extra_label(existing: str, extra: str) -> str:
    """Insert an extra ``key="value"`` into an existing label block."""
    if not existing:
        return "{" + extra + "}"
    # existing is "{a="1",b="2"}" — insert before the closing brace.
    return existing[:-1] + "," + extra + "}"


class _Metric:
    """Thread-safe counter / gauge / histogram. Private — use registry API."""

    def __init__(
        self,
        name: str,
        help_text: str,
        type_: str,
        labels: tuple[str, ...] = (),
        buckets: Optional[tuple[float, ...]] = None,
    ) -> None:
        self.name = name
        self.help = help_text
        self.type = type_
        self.labels = labels
        # Sorted to make exposition order deterministic across workers.
        self.buckets = tuple(sorted(buckets)) if buckets else ()
        self._lock = threading.Lock()
        self._samples: dict[tuple[str, ...], _MetricSample] = {}

    def _child(self, label_values: tuple[str, ...]) -> _MetricSample:
        sample = self._samples.get(label_values)
        if sample is not None:
            return sample
        with self._lock:
            sample = self._samples.get(label_values)
            if sample is None:
                sample = _MetricSample()
                if self.type == "histogram":
                    sample.bucket_counts = {b: 0 for b in self.buckets}
                self._samples[label_values] = sample
            return sample

    def _resolve_labels(self, labels: dict[str, object]) -> tuple[str, ...]:
        return tuple(str(labels.get(name, "")) for name in self.labels)

    def inc(self, amount: float = 1.0, **labels: object) -> None:
        if self.type not in {"counter", "gauge"}:
            raise RuntimeError(f"inc() not supported on {self.type}")
        sample = self._child(self._resolve_labels(labels))
        with self._lock:
            sample.value += amount

    def set(self, value: float, **labels: object) -> None:
        if self.type != "gauge":
            raise RuntimeError(f"set() not supported on {self.type}")
        sample = self._child(self._resolve_labels(labels))
        with self._lock:
            sample.value = float(value)

    def observe(self, value: float, **labels: object) -> None:
        if self.type != "histogram":
            raise RuntimeError(f"observe() not supported on {self.type}")
        sample = self._child(self._resolve_labels(labels))
        with self._lock:
            sample.count += 1
            sample.sum_ += value
            for b in self.buckets:
                if value <= b:
                    sample.bucket_counts[b] += 1

    def render_lines(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} {self.type}"
        with self._lock:
            # Snapshot for deterministic output; copies are cheap at
            # this scale.
            items = sorted(self._samples.items(), key=lambda kv: kv[0])
            items_copy = [
                (
                    values,
                    _MetricSample(
                        value=s.value,
                        bucket_counts=dict(s.bucket_counts),
                        count=s.count,
                        sum_=s.sum_,
                    ),
                )
                for values, s in items
            ]
        for values, sample in items_copy:
            label_block = _format_labels(self.labels, values)
            if self.type == "histogram":
                cumulative = 0
                for b in self.buckets:
                    cumulative = sample.bucket_counts[b]
                    bucket_label = _with_extra_label(label_block, f'le="{_format_bucket(b)}"')
                    yield f"{self.name}_bucket{bucket_label} {cumulative}"
                inf_label = _with_extra_label(label_block, 'le="+Inf"')
                yield f"{self.name}_bucket{inf_label} {sample.count}"
                yield f"{self.name}_count{label_block} {sample.count}"
                yield f"{self.name}_sum{label_block} {_format_value(sample.sum_)}"
            else:
                yield f"{self.name}{label_block} {_format_value(sample.value)}"


def _format_bucket(b: float) -> str:
    # Render whole numbers without trailing ".0" for readability.
    return f"{int(b)}" if float(b).is_integer() else f"{b}"


def _format_value(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.6f}".rstrip("0").rstrip(".")


class _Registry:
    def __init__(self) -> None:
        self._metrics: list[_Metric] = []
        self._lock = threading.Lock()

    def register(self, metric: _Metric) -> _Metric:
        with self._lock:
            self._metrics.append(metric)
        return metric

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            metrics = list(self._metrics)
        for m in metrics:
            lines.extend(m.render_lines())
        # Prometheus requires a trailing newline after the last line.
        return "\n".join(lines) + "\n"

    def reset_for_tests(self) -> None:
        """Zero every metric in place. Used by the test fixture so a
        fresh test observes starting counts, not whatever the previous
        test left behind.
        """
        with self._lock:
            for m in self._metrics:
                with m._lock:
                    m._samples.clear()


registry = _Registry()

# --- Instrumented metrics ---------------------------------------------------

# Default latency buckets in milliseconds. Covers a fast cache hit
# (~5ms) through a slow translation (~5s). 10 buckets is the sweet
# spot: Prometheus scrape/series cardinality stays small, and the
# p95/p99 summary still comes out accurate.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
)

TRANSLATIONS_TOTAL = registry.register(
    _Metric(
        "kozha_translations_total",
        "Total translation requests by source language, target sign language, and outcome.",
        "counter",
        labels=("source_lang", "target_sign_lang", "outcome"),
    )
)
TRANSLATION_LATENCY_MS = registry.register(
    _Metric(
        "kozha_translation_latency_ms",
        "Translation request latency in milliseconds.",
        "histogram",
        labels=("source_lang", "target_sign_lang", "outcome"),
        buckets=LATENCY_BUCKETS_MS,
    )
)
FINGERSPELL_FALLBACK_TOTAL = registry.register(
    _Metric(
        "kozha_fingerspell_fallback_total",
        "Count of glosses rendered as fingerspelling fallback (no sign in database).",
        "counter",
        labels=("target_sign_lang",),
    )
)
UNKNOWN_WORD_TOTAL = registry.register(
    _Metric(
        "kozha_unknown_word_total",
        "Tokens a user requested that the target sign language's database does not cover.",
        "counter",
        labels=("target_sign_lang",),
    )
)
DATABASE_SIZE_TOTAL = registry.register(
    _Metric(
        "kozha_database_size_total",
        "Number of kept signs in the database per language (set by the snapshot script).",
        "gauge",
        labels=("language",),
    )
)
REVIEWED_SIGNS_TOTAL = registry.register(
    _Metric(
        "kozha_reviewed_signs_total",
        "Number of Deaf-native-reviewed signs per language (set by the snapshot script).",
        "gauge",
        labels=("language",),
    )
)
CONTRIBUTIONS_RECEIVED_TOTAL = registry.register(
    _Metric(
        "kozha_contributions_received_total",
        "Total community contribution submissions received.",
        "counter",
    )
)
CONTRIBUTIONS_VALIDATED_TOTAL = registry.register(
    _Metric(
        "kozha_contributions_validated_total",
        "Total community contributions that reached validated state.",
        "counter",
    )
)


def record_translation(
    *,
    source_lang: str,
    target_sign_lang: str,
    outcome: str,
    latency_ms: float,
) -> None:
    """Hot-path helper — one function call instead of two."""
    o = outcome if outcome in OUTCOMES else "server_error"
    TRANSLATIONS_TOTAL.inc(
        source_lang=source_lang,
        target_sign_lang=target_sign_lang,
        outcome=o,
    )
    TRANSLATION_LATENCY_MS.observe(
        latency_ms,
        source_lang=source_lang,
        target_sign_lang=target_sign_lang,
        outcome=o,
    )


def refresh_database_gauges_from_snapshot(snapshot: dict) -> None:
    """Set the per-language size / reviewed gauges from a snapshot dict.

    Called at server start-up (and by the snapshot's in-process caller)
    so the ``/metrics`` view matches the public dashboard without a
    round-trip through disk.
    """
    langs = snapshot.get("languages") or []
    for lang in langs:
        if not isinstance(lang, dict):
            continue
        code = (lang.get("code") or "").strip().lower()
        if not code:
            continue
        total = lang.get("total")
        reviewed = lang.get("reviewed")
        if isinstance(total, (int, float)):
            DATABASE_SIZE_TOTAL.set(total, language=code)
        if isinstance(reviewed, (int, float)):
            REVIEWED_SIGNS_TOTAL.set(reviewed, language=code)


# ---------------------------------------------------------------------------
# Readiness probe
# ---------------------------------------------------------------------------


@dataclass
class ReadinessReport:
    ok: bool
    checks: dict[str, bool]
    detail: dict[str, str] = field(default_factory=dict)


def readiness_probe() -> ReadinessReport:
    """Return whether the server can actually serve traffic.

    Checks (each one optional — we don't want a missing argos install
    to indefinitely 503 the server when the translator can still serve
    /api/plan without external translation):

    * ``data_dir`` — ``data/`` exists and is readable.
    * ``meta_files`` — at least one ``*.meta.json`` is parseable.
    * ``spacy_en`` — the English spaCy model is loaded (the translator
      falls back to a blank pipeline if not, but we surface the state
      so operators know).
    * ``argostranslate`` — the package imports (installation of per-
      pair packages happens lazily on first use).
    """
    checks: dict[str, bool] = {}
    detail: dict[str, str] = {}

    # data/
    try:
        ok = DATA_DIR.exists() and DATA_DIR.is_dir()
        checks["data_dir"] = ok
        if not ok:
            detail["data_dir"] = f"missing or not a directory: {DATA_DIR}"
    except Exception as e:
        checks["data_dir"] = False
        detail["data_dir"] = f"error: {e!s}"

    # meta files
    try:
        any_meta = next(DATA_DIR.glob("*.meta.json"), None)
        if any_meta is None:
            checks["meta_files"] = False
            detail["meta_files"] = "no *.meta.json files under data/"
        else:
            json.loads(any_meta.read_text(encoding="utf-8"))
            checks["meta_files"] = True
    except Exception as e:
        checks["meta_files"] = False
        detail["meta_files"] = f"parse error: {e!s}"

    # spaCy
    try:
        import spacy  # noqa: F401
        try:
            spacy.load("en_core_web_sm", disable=["ner"])
            checks["spacy_en"] = True
        except Exception as e:
            checks["spacy_en"] = False
            detail["spacy_en"] = f"model not loadable: {e!s}"
    except Exception as e:
        checks["spacy_en"] = False
        detail["spacy_en"] = f"import failed: {e!s}"

    # argostranslate
    try:
        import argostranslate  # noqa: F401
        checks["argostranslate"] = True
    except Exception as e:
        checks["argostranslate"] = False
        detail["argostranslate"] = f"import failed: {e!s}"

    # Ready when the data invariants hold. spaCy and argos are warned
    # about but not required — the translator has fallbacks for both,
    # and an LB that flaps on a spacy-model download would be noisier
    # than the degradation is worth.
    ok = checks.get("data_dir", False) and checks.get("meta_files", False)
    return ReadinessReport(ok=ok, checks=checks, detail=detail)
