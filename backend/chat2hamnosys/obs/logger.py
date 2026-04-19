"""Structured JSON event logger with daily rotation and a recent-events ring.

Every call to :func:`emit_event` produces one JSON object on the
configured sink (file in dev, stdout in production) **and** is appended
to an in-process ring buffer the dashboard reads for its live feed.

Top-level fields on every record (always present, even when ``None``):
``ts``, ``level``, ``event``, ``session_id``, ``request_id``,
``user_hash``. Event-specific fields are merged in next to them.

Sinks
-----
``CHAT2HAMNOSYS_LOG_SINK`` selects the sink:

- ``file`` (default) — writes to ``<log_dir>/<YYYY-MM-DD>.jsonl``;
  the directory is ``CHAT2HAMNOSYS_LOG_DIR`` or ``<repo>/logs``.
- ``stdout`` — one JSON object per line on stdout, suitable for the
  Docker / Cloud Logging path. No file is opened.
- ``both`` — file *and* stdout.

Retention
---------
``CHAT2HAMNOSYS_LOG_RETENTION_DAYS`` (default 30) drives a best-effort
cleanup that runs the first time the date rolls over inside the
process. Old ``YYYY-MM-DD.jsonl`` files in ``log_dir`` are deleted in
place; nothing else is touched, so the directory may safely contain
non-log files alongside.

PII
---
Never log API keys, never log full user descriptions unless the
``log_content=True`` flag is passed at the call site (intended for
local debugging only). The :func:`emit_event` API takes whatever fields
the caller supplies — the caller is responsible for redaction. To make
that easier the helper :func:`hash_user_id` is exposed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .events import ALL_EVENTS, level_for


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = _REPO_ROOT / "logs"
DEFAULT_RETENTION_DAYS = 30
DEFAULT_RING_SIZE = 200

LOG_DIR_ENV_VAR = "CHAT2HAMNOSYS_LOG_DIR"
LOG_SINK_ENV_VAR = "CHAT2HAMNOSYS_LOG_SINK"
LOG_RETENTION_ENV_VAR = "CHAT2HAMNOSYS_LOG_RETENTION_DAYS"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UnknownEventError(ValueError):
    """Caller emitted a name not declared in :data:`events.ALL_EVENTS`.

    The closed taxonomy is what makes the dashboard's filter list useful
    and stops typos like ``session.creted`` from quietly polluting the
    feed. Bypass by adding the constant to ``events.py`` first.
    """


# ---------------------------------------------------------------------------
# Recent-events ring
# ---------------------------------------------------------------------------


@dataclass
class StructuredEvent:
    """One emitted event, suitable for the dashboard feed."""

    ts: datetime
    level: str
    event: str
    session_id: str | None
    request_id: str | None
    user_hash: str | None
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts": self.ts.isoformat(),
            "level": self.level,
            "event": self.event,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "user_hash": self.user_hash,
        }
        out.update(self.fields)
        return out


# ---------------------------------------------------------------------------
# The logger
# ---------------------------------------------------------------------------


class EventLogger:
    """Append-only JSON event sink with rotation + a ring buffer.

    Thread-safe. One instance per process is the norm — see the module
    singleton accessors :func:`get_logger` / :func:`reset_logger`.
    """

    def __init__(
        self,
        log_dir: Path | str | None = None,
        *,
        sink: str | None = None,
        retention_days: int | None = None,
        ring_size: int = DEFAULT_RING_SIZE,
        clock: "type[datetime] | None" = None,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else _resolve_log_dir()
        self.sink = (sink or os.environ.get(LOG_SINK_ENV_VAR) or "file").lower()
        if self.sink not in ("file", "stdout", "both"):
            raise ValueError(
                f"sink must be one of file/stdout/both; got {self.sink!r}"
            )
        self.retention_days = (
            retention_days
            if retention_days is not None
            else _resolve_retention_days()
        )
        if self.retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        self._ring: deque[StructuredEvent] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._clock = clock or datetime
        self._last_purge_date: date | None = None
        # Per-day counter snapshot — exposed to the dashboard for hourly
        # spark visuals without re-reading files. Buckets are keyed by
        # (date, hour_int). Old buckets fall out via the ring buffer's
        # natural eviction when the dashboard refreshes.

    # -- public API --------------------------------------------------------

    def emit(
        self,
        event: str,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        user_hash: str | None = None,
        level: str | None = None,
        **fields: Any,
    ) -> StructuredEvent:
        """Emit one event. Returns the structured record so callers can chain."""
        if event not in ALL_EVENTS:
            raise UnknownEventError(
                f"event {event!r} is not in events.ALL_EVENTS — declare it first"
            )
        ts = self._clock.now(timezone.utc)
        record = StructuredEvent(
            ts=ts,
            level=level or level_for(event),
            event=event,
            session_id=session_id,
            request_id=request_id,
            user_hash=user_hash,
            fields=dict(fields),
        )
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self._ring.append(record)
            self._maybe_purge(ts.date())
            if self.sink in ("file", "both"):
                self._write_file(line, ts)
            if self.sink in ("stdout", "both"):
                sys.stdout.write(line)
                sys.stdout.flush()
        return record

    def recent(self, limit: int | None = None) -> list[StructuredEvent]:
        """Return the most-recent events (newest last) in time order."""
        with self._lock:
            items = list(self._ring)
        if limit is not None:
            return items[-limit:]
        return items

    def recent_for_session(self, session_id: str) -> list[StructuredEvent]:
        """Return ring-buffer events tagged with ``session_id`` (oldest first)."""
        with self._lock:
            return [e for e in self._ring if e.session_id == session_id]

    def clear_ring(self) -> None:
        """Empty the in-process recent-events ring (test helper)."""
        with self._lock:
            self._ring.clear()

    # -- internals ---------------------------------------------------------

    def _path_for(self, ts: datetime) -> Path:
        return self.log_dir / f"{ts.strftime('%Y-%m-%d')}.jsonl"

    def _write_file(self, line: str, ts: datetime) -> None:
        path = self._path_for(ts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def _maybe_purge(self, today: date) -> None:
        if self._last_purge_date == today:
            return
        self._last_purge_date = today
        if not self.log_dir.exists():
            return
        cutoff = today - timedelta(days=self.retention_days)
        for child in self.log_dir.iterdir():
            if not child.is_file() or child.suffix != ".jsonl":
                continue
            stem = child.stem
            try:
                file_date = date.fromisoformat(stem)
            except ValueError:
                continue
            if file_date < cutoff:
                try:
                    child.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Module singletons
# ---------------------------------------------------------------------------


_logger: EventLogger | None = None
_singleton_lock = threading.Lock()


def get_logger() -> EventLogger:
    """Return the process-wide :class:`EventLogger`."""
    global _logger
    if _logger is None:
        with _singleton_lock:
            if _logger is None:
                _logger = EventLogger()
    return _logger


def set_logger(logger: EventLogger) -> None:
    """Override the process-wide logger (test helper)."""
    global _logger
    with _singleton_lock:
        _logger = logger


def reset_logger() -> None:
    """Drop the singleton — next :func:`get_logger` rebuilds from env."""
    global _logger
    with _singleton_lock:
        _logger = None


def emit_event(
    event: str,
    *,
    session_id: str | None = None,
    request_id: str | None = None,
    user_hash: str | None = None,
    level: str | None = None,
    **fields: Any,
) -> StructuredEvent:
    """Top-level convenience: emit through the singleton logger."""
    return get_logger().emit(
        event,
        session_id=session_id,
        request_id=request_id,
        user_hash=user_hash,
        level=level,
        **fields,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_user_id(value: str | None, *, salt: str = "obs") -> str | None:
    """Deterministic SHA-256 hash for the ``user_hash`` field.

    Sessions already get a hashed signer via :func:`security.hash_signer_id`;
    this helper exists for callers that don't want to pull the security
    module just to log a user reference.
    """
    if not value:
        return None
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; return all parseable rows. Missing file → []."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def iter_log_files(log_dir: Path) -> Iterable[Path]:
    """Yield ``YYYY-MM-DD.jsonl`` files in ``log_dir`` newest-first."""
    if not log_dir.exists():
        return []
    files = []
    for child in log_dir.iterdir():
        if child.is_file() and child.suffix == ".jsonl":
            try:
                date.fromisoformat(child.stem)
            except ValueError:
                continue
            files.append(child)
    files.sort(reverse=True)
    return files


# ---------------------------------------------------------------------------
# Env resolution
# ---------------------------------------------------------------------------


def _resolve_log_dir() -> Path:
    raw = os.environ.get(LOG_DIR_ENV_VAR, "").strip()
    return Path(raw) if raw else DEFAULT_LOG_DIR


def _resolve_retention_days() -> int:
    raw = os.environ.get(LOG_RETENTION_ENV_VAR)
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{LOG_RETENTION_ENV_VAR}={raw!r} is not a valid int"
        ) from exc


__all__ = [
    "DEFAULT_LOG_DIR",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_RING_SIZE",
    "EventLogger",
    "LOG_DIR_ENV_VAR",
    "LOG_RETENTION_ENV_VAR",
    "LOG_SINK_ENV_VAR",
    "StructuredEvent",
    "UnknownEventError",
    "emit_event",
    "get_logger",
    "hash_user_id",
    "iter_log_files",
    "read_jsonl",
    "reset_logger",
    "set_logger",
]
