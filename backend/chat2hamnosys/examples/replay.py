"""Replay recorded sessions through the live API.

Each fixture in this directory is a captured event log plus the expected
``SignEntry`` the system produced when the session was originally recorded.
``replay`` does two things:

1. **Stream events at the API.** For demo use: ``python -m examples.replay
   --example electron`` posts the recorded ``described`` / ``answer`` /
   ``correct`` / ``accept`` calls to the running backend at the pace they
   happened (or instantly with ``--no-realtime``). The avatar in the
   browser plays the sign exactly as it did during the recording. When the
   conference Wi-Fi dies, this is the deterministic stand-in.

2. **Verify the result.** For golden-test use: after the replay completes,
   the resulting ``SignEntry`` is fetched and compared field-by-field
   against the fixture's ``expected_sign_entry``. A non-empty diff signals
   prompt drift, validator drift, or vocab-map drift — the replay prints
   it and exits non-zero.

Usage
-----

::

    python -m examples.replay --example electron
    python -m examples.replay --example all --no-realtime --base-url http://localhost:8000
    python -m examples.replay --list

The script intentionally has no dependency on the OpenAI key. The recorded
events already contain the LLM outputs; the backend has a deterministic
``replay`` mode that re-uses them instead of re-calling the API. (See
``api/router.py`` for the ``X-Replay-Mode: fixture`` header path.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from . import EXAMPLE_NAMES, EXAMPLES_DIR, example_path


@dataclass(frozen=True)
class ReplayConfig:
    base_url: str
    realtime: bool
    verify: bool


def _post(url: str, body: dict[str, Any], session_token: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "X-Replay-Mode": "fixture"}
    if session_token is not None:
        headers["X-Session-Token"] = session_token
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"POST {url} failed: HTTP {exc.code}\n{body_text}") from exc
    except error.URLError as exc:
        raise SystemExit(
            f"POST {url} failed: {exc.reason}. Is the backend running on {url}?"
        ) from exc


def _get(url: str, session_token: str | None = None) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if session_token is not None:
        headers["X-Session-Token"] = session_token
    req = request.Request(url, headers=headers, method="GET")
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_until(prev_ts: str | None, this_ts: str, *, realtime: bool) -> None:
    if not realtime or prev_ts is None:
        return
    from datetime import datetime

    delta = (
        datetime.fromisoformat(this_ts) - datetime.fromisoformat(prev_ts)
    ).total_seconds()
    if delta > 0:
        time.sleep(min(delta, 30.0))


def _diff_entry(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    for key, want in expected.items():
        if key in {"id", "created_at", "updated_at"}:
            continue
        got = actual.get(key)
        if got != want:
            diffs.append(f"  {key}: expected {want!r}, got {got!r}")
    return diffs


def replay_one(name: str, cfg: ReplayConfig) -> bool:
    fixture = json.loads(example_path(name).read_text(encoding="utf-8"))
    sess = fixture["session"]
    events = fixture["events"]
    expected = fixture.get("expected_sign_entry", {})

    base = cfg.base_url.rstrip("/") + "/api/chat2hamnosys"
    print(f"[{name}] starting (sign_language={sess['sign_language']})")

    create_url = f"{base}/sessions"
    response = _post(create_url, {"sign_language": sess["sign_language"]})
    session_id = response["session_id"]
    session_token = response["session_token"]
    print(f"[{name}] session created id={session_id}")

    final_entry_id: str | None = None
    prev_ts: str | None = None
    for ev in events:
        _wait_until(prev_ts, ev["timestamp"], realtime=cfg.realtime)
        prev_ts = ev["timestamp"]

        kind = ev["type"]
        if kind == "described":
            _post(
                f"{base}/sessions/{session_id}/describe",
                {"prose": ev["prose"]},
                session_token=session_token,
            )
            print(f"[{name}] described → gaps_found={ev['gaps_found']}")
        elif kind == "clarification_asked":
            continue
        elif kind == "clarification_answered":
            _post(
                f"{base}/sessions/{session_id}/answer",
                {
                    "question_field": ev["question_field"],
                    "answer": ev["answer"],
                    "resolved_value": ev["resolved_value"],
                },
                session_token=session_token,
            )
            print(f"[{name}] answered {ev['question_field']!r}")
        elif kind == "generated":
            _post(
                f"{base}/sessions/{session_id}/generate",
                {
                    "expected_hamnosys": ev.get("hamnosys"),
                    "expected_sigml": ev.get("sigml"),
                },
                session_token=session_token,
            )
            print(
                f"[{name}] generated success={ev['success']}"
                f" confidence={ev['confidence']:.2f}"
                f" llm_fallback={ev['used_llm_fallback']}"
            )
        elif kind == "correction_requested":
            _post(
                f"{base}/sessions/{session_id}/correct",
                {
                    "raw_text": ev["raw_text"],
                    "target_time_ms": ev.get("target_time_ms"),
                    "target_region": ev.get("target_region"),
                },
                session_token=session_token,
            )
            print(f"[{name}] correction submitted: {ev['raw_text']!r}")
        elif kind == "correction_applied":
            continue
        elif kind == "accepted":
            result = _post(
                f"{base}/sessions/{session_id}/accept",
                {},
                session_token=session_token,
            )
            final_entry_id = result["sign_entry_id"]
            print(f"[{name}] accepted → sign_entry_id={final_entry_id}")
        elif kind == "rejected":
            _post(
                f"{base}/sessions/{session_id}/reject",
                {"reason": ev.get("reason", "")},
                session_token=session_token,
            )
        elif kind == "abandoned":
            print(f"[{name}] (abandoned event in fixture; skipping)")
        else:
            print(f"[{name}] unknown event type {kind!r}; skipping", file=sys.stderr)

    if not cfg.verify or final_entry_id is None or not expected:
        return True

    actual = _get(f"{base}/signs/{final_entry_id}")
    diffs = _diff_entry(actual, expected)
    if diffs:
        print(f"[{name}] DIVERGENCE from expected sign entry:", file=sys.stderr)
        for line in diffs:
            print(line, file=sys.stderr)
        return False
    print(f"[{name}] verified ok")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="examples.replay", description=__doc__)
    parser.add_argument("--example", default=None, help='example name, or "all"')
    parser.add_argument("--list", action="store_true", help="list known examples")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="backend base URL"
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="ignore the recorded inter-event delays and replay as fast as possible",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="do not diff the resulting SignEntry against the fixture",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in EXAMPLE_NAMES:
            path = EXAMPLES_DIR / f"{name}.json"
            marker = "" if path.exists() else " (missing)"
            print(f"  {name}{marker}")
        return 0

    if args.example is None:
        parser.error("--example is required (or pass --list)")

    cfg = ReplayConfig(
        base_url=args.base_url,
        realtime=not args.no_realtime,
        verify=not args.no_verify,
    )

    targets = list(EXAMPLE_NAMES) if args.example == "all" else [args.example]
    failed: list[str] = []
    for name in targets:
        try:
            ok = replay_one(name, cfg)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] crashed: {exc!r}", file=sys.stderr)
            ok = False
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nfailed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nall examples ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
