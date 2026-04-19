"""Correction-loop metrics.

Writes one JSONL line per event to ``<repo>/logs/corrections/<YYYY-MM-DD>.jsonl``.
Two event kinds are emitted:

- ``correction_applied`` — one per call to :func:`apply_correction` with the
  interpreter's intent, the number of field changes, and the outcome
  (``ok`` / ``validation_failed`` / ``noop``).
- ``session_accepted`` — one per accepted sign with the
  ``corrections_count`` on the session's draft at accept time.

The dashboard metric "average corrections per accepted sign" is computed
offline by averaging the ``corrections_count`` field across
``session_accepted`` records over the date range of interest. Per-event
``correction_applied`` records are kept for drilldowns.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR: Path = _REPO_ROOT / "logs" / "corrections"

_LOCK = threading.Lock()


def _path_for_today(log_dir: Path) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return log_dir / f"{today}.jsonl"


def _append(log_dir: Path | None, record: dict[str, Any]) -> Path:
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    path = _path_for_today(directory)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    return path


def log_correction_applied(
    *,
    session_id: UUID | str,
    intent: str,
    field_count: int,
    outcome: str,
    corrections_count: int,
    log_dir: Path | None = None,
) -> Path:
    """Append one ``correction_applied`` record.

    ``outcome`` is one of ``"ok"``, ``"validation_failed"``, ``"noop"``;
    ``noop`` covers the elaborate / contradiction / vague / restart
    intents where no diff was applied.
    """
    return _append(
        log_dir,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "correction_applied",
            "session_id": str(session_id),
            "intent": intent,
            "field_count": int(field_count),
            "outcome": outcome,
            "corrections_count": int(corrections_count),
        },
    )


def log_session_accepted(
    *,
    session_id: UUID | str,
    sign_entry_id: UUID | str | None,
    corrections_count: int,
    log_dir: Path | None = None,
) -> Path:
    """Append one ``session_accepted`` record.

    Callers should invoke this right after :func:`session.orchestrator.on_accept`
    so the dashboard can compute corrections-per-accepted-sign as
    ``avg(corrections_count)`` over ``session_accepted`` records.
    """
    return _append(
        log_dir,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "session_accepted",
            "session_id": str(session_id),
            "sign_entry_id": str(sign_entry_id) if sign_entry_id is not None else None,
            "corrections_count": int(corrections_count),
        },
    )


__all__ = [
    "DEFAULT_LOG_DIR",
    "log_correction_applied",
    "log_session_accepted",
]
