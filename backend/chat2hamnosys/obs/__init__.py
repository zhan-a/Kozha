"""Observability surface: events, metrics, dashboard, alerts.

Public API
----------
- :func:`emit_event` — write one structured JSON event through the
  process-wide logger; mirrored into the in-process ring buffer that
  backs the dashboard feed.
- :func:`hash_user_id` — one-way hash helper for the ``user_hash`` field.
- :mod:`events` — canonical event-name constants. Never build event
  names from string concatenation at a call site.
- :mod:`metrics` — Prometheus counter/gauge/histogram/summary singletons
  plus :func:`metrics.render_text`.
- :mod:`dashboard` — HTML renderers and pure aggregators.
- :mod:`alerts` — rule-based alerter + default rule set.

The session / LLM / pipeline modules depend on :func:`emit_event` and
the metric objects directly. Nothing imports ``dashboard``/``alerts``
at runtime except the router.
"""

from __future__ import annotations

from . import alerts, dashboard, events, metrics
from .events import ALL_EVENTS, EVENT_LEVEL, level_for
from .logger import (
    DEFAULT_LOG_DIR,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_RING_SIZE,
    EventLogger,
    LOG_DIR_ENV_VAR,
    LOG_RETENTION_ENV_VAR,
    LOG_SINK_ENV_VAR,
    StructuredEvent,
    UnknownEventError,
    emit_event,
    get_logger,
    hash_user_id,
    iter_log_files,
    read_jsonl,
    reset_logger,
    set_logger,
)


__all__ = [
    "ALL_EVENTS",
    "DEFAULT_LOG_DIR",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_RING_SIZE",
    "EVENT_LEVEL",
    "EventLogger",
    "LOG_DIR_ENV_VAR",
    "LOG_RETENTION_ENV_VAR",
    "LOG_SINK_ENV_VAR",
    "StructuredEvent",
    "UnknownEventError",
    "alerts",
    "dashboard",
    "emit_event",
    "events",
    "get_logger",
    "hash_user_id",
    "iter_log_files",
    "level_for",
    "metrics",
    "read_jsonl",
    "reset_logger",
    "set_logger",
]
