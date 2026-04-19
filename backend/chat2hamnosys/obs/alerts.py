"""Rule-based alerter over recent events and in-memory metrics.

The alerter scans the log ring-buffer + on-disk JSONL files produced by
:mod:`obs.logger` plus the :class:`obs.metrics` registry, evaluates a
small set of rules, and publishes triggered alerts to a configurable
sink (a webhook URL by default).

Design notes
------------
- **Pull, not push.** The alerter never blocks the hot path; it runs on
  a scheduler (see :func:`run_alerter_loop`) or on demand from a cron
  job. Every pipeline module just emits events and increments counters;
  alert logic lives here.
- **Rules are plain functions**: ``(AlerterContext) -> Alert | None``.
  Each rule reads the window of events and the metric registry, returns
  a single :class:`Alert` if it fires, ``None`` otherwise. Defaults are
  registered in :data:`DEFAULT_RULES`.
- **Silencing.** A rule that fires twice within the silence window is
  suppressed the second time. This avoids paging storms when a failure
  mode persists. Silence state is in-memory — restarts clear it, which
  is the desired behavior: a restart is a legitimate reason to hear
  about the same condition again.
- **Delivery.** :class:`WebhookSink` posts JSON to a configured URL
  (Slack incoming-webhook compatible shape). :class:`RecordingSink`
  accumulates alerts in memory, used by tests.

Environment variables
---------------------
- ``CHAT2HAMNOSYS_ALERT_WEBHOOK_URL`` — default destination URL.
- ``CHAT2HAMNOSYS_ALERT_INTERVAL_S`` — loop period (default 300s).
- ``CHAT2HAMNOSYS_ALERT_SILENCE_S`` — suppression window (default 1800s).
- ``CHAT2HAMNOSYS_DAILY_BUDGET_USD`` — cap used by the cost-projection
  rule; optional, nothing fires if unset.
- ``CHAT2HAMNOSYS_PENDING_REVIEW_STALE_H`` — hours a pending review may
  wait before triggering the queue-depth rule (default 24).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

from . import events as evs
from . import metrics as metrics_mod
from .logger import (
    DEFAULT_LOG_DIR,
    EventLogger,
    get_logger,
    iter_log_files,
    read_jsonl,
)


logger = logging.getLogger(__name__)


DEFAULT_INTERVAL_S = 300
DEFAULT_SILENCE_S = 1800
DEFAULT_PENDING_REVIEW_STALE_H = 24


# ---------------------------------------------------------------------------
# Alert model + sinks
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """One alerting event — keys the dedup bucket by :attr:`kind`."""

    kind: str
    severity: str  # "warning" | "error" | "info"
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "summary": self.summary,
            "details": dict(self.details),
            "fired_at": self.fired_at.isoformat(),
        }


class AlertSink(Protocol):
    """Delivery target for one :class:`Alert`."""

    def publish(self, alert: Alert) -> None: ...


class RecordingSink:
    """In-memory sink — tests use this to inspect fired alerts."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []
        self._lock = threading.Lock()

    def publish(self, alert: Alert) -> None:
        with self._lock:
            self.alerts.append(alert)


class WebhookSink:
    """POST one JSON payload per alert to a webhook URL.

    The shape matches Slack's incoming-webhook format (``{"text": …}``)
    but carries the full alert under ``attachments[0].alert`` for
    consumers that want to parse it. Network failures are logged and
    swallowed — the alerter must not crash the server if Slack is down.
    """

    def __init__(self, url: str, *, timeout: float = 5.0) -> None:
        if not url:
            raise ValueError("webhook url must be non-empty")
        self.url = url
        self.timeout = timeout

    def publish(self, alert: Alert) -> None:
        payload = {
            "text": f"[{alert.severity.upper()}] {alert.summary}",
            "attachments": [
                {
                    "color": _color_for(alert.severity),
                    "title": alert.kind,
                    "text": alert.summary,
                    "alert": alert.to_dict(),
                }
            ],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read(0)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("alert webhook failed for %s: %s", alert.kind, exc)


def _color_for(severity: str) -> str:
    return {"error": "#b91c1c", "warning": "#b45309"}.get(severity, "#2050c0")


# ---------------------------------------------------------------------------
# Alerter context — snapshot passed to each rule
# ---------------------------------------------------------------------------


@dataclass
class AlerterContext:
    """Snapshot read at the top of one evaluation pass.

    Rules read from this; they never touch the logger / metrics
    registries directly. Keeping it as data makes rules trivially
    unit-testable without spinning up the logger.
    """

    now: datetime
    events_recent_hour: list[dict[str, Any]]
    events_recent_day: list[dict[str, Any]]
    metrics_registry: metrics_mod.MetricsRegistry
    daily_budget_usd: Optional[float]
    pending_review_stale_hours: float


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


Rule = Callable[[AlerterContext], Optional[Alert]]


def rule_llm_failure_rate(ctx: AlerterContext) -> Optional[Alert]:
    """Fire when >20% of LLM calls in the last 15 minutes failed."""
    cutoff = ctx.now - timedelta(minutes=15)
    window = [e for e in ctx.events_recent_hour if _ts(e) and _ts(e) >= cutoff]
    calls = [e for e in window if e.get("event") in (evs.LLM_CALL_SUCCEEDED, evs.LLM_CALL_FAILED)]
    if len(calls) < 10:
        return None
    failures = [e for e in calls if e.get("event") == evs.LLM_CALL_FAILED]
    rate = len(failures) / len(calls)
    if rate <= 0.20:
        return None
    return Alert(
        kind="llm_failure_rate_high",
        severity="error",
        summary=(
            f"LLM failure rate {rate:.0%} over last 15m "
            f"({len(failures)}/{len(calls)} calls failed)"
        ),
        details={
            "window_minutes": 15,
            "calls": len(calls),
            "failures": len(failures),
            "rate": round(rate, 3),
        },
    )


def rule_session_abandonment(ctx: AlerterContext) -> Optional[Alert]:
    """Fire when >50% of completed sessions in the last hour abandoned."""
    created = 0
    abandoned = 0
    accepted = 0
    for e in ctx.events_recent_hour:
        if e.get("event") == evs.SESSION_CREATED:
            created += 1
        elif e.get("event") == evs.SESSION_ABANDONED:
            abandoned += 1
        elif e.get("event") == evs.SESSION_ACCEPTED:
            accepted += 1
    completed = abandoned + accepted
    if completed < 5:
        return None
    rate = abandoned / completed
    if rate <= 0.50:
        return None
    return Alert(
        kind="session_abandonment_high",
        severity="warning",
        summary=(
            f"Session abandonment {rate:.0%} over last hour "
            f"({abandoned}/{completed} completed sessions abandoned)"
        ),
        details={
            "completed": completed,
            "abandoned": abandoned,
            "accepted": accepted,
            "created": created,
            "rate": round(rate, 3),
        },
    )


def rule_daily_cost_projection(ctx: AlerterContext) -> Optional[Alert]:
    """Fire when the daily cost projection is set to exceed the budget cap."""
    if ctx.daily_budget_usd is None or ctx.daily_budget_usd <= 0:
        return None
    now = ctx.now
    today_spend = 0.0
    for e in ctx.events_recent_day:
        ts = _ts(e)
        if ts is None or ts.date() != now.date():
            continue
        if e.get("event") in (evs.LLM_CALL_SUCCEEDED, evs.LLM_CALL_FAILED):
            today_spend += float(e.get("cost_usd", 0.0) or 0.0)
    elapsed_hours = now.hour + now.minute / 60.0
    if elapsed_hours < 1.0:
        return None
    projected = today_spend * (24.0 / elapsed_hours)
    if projected <= ctx.daily_budget_usd:
        return None
    return Alert(
        kind="daily_cost_projected_over_budget",
        severity="error",
        summary=(
            f"Today's cost projection ${projected:.2f} exceeds cap "
            f"${ctx.daily_budget_usd:.2f} (spent ${today_spend:.2f} so far)"
        ),
        details={
            "today_spend_usd": round(today_spend, 4),
            "projected_usd": round(projected, 4),
            "cap_usd": ctx.daily_budget_usd,
            "elapsed_hours": round(elapsed_hours, 2),
        },
    )


def rule_pending_review_queue_deep(ctx: AlerterContext) -> Optional[Alert]:
    """Fire when pending reviews stay queued longer than the stale threshold."""
    depth = int(ctx.metrics_registry.get("pending_reviews_queue_depth").get())  # type: ignore[attr-defined]
    if depth <= 50:
        return None
    # Look for the oldest still-pending ``review.queued`` event that has
    # not been approved/rejected within the stale window.
    cutoff = ctx.now - timedelta(hours=ctx.pending_review_stale_hours)
    queued_before_cutoff: set[str] = set()
    resolved: set[str] = set()
    for e in ctx.events_recent_day:
        ts = _ts(e)
        sid = e.get("session_id") or ""
        if e.get("event") == evs.REVIEW_QUEUED and ts and ts <= cutoff:
            queued_before_cutoff.add(sid)
        if e.get("event") in (
            evs.REVIEW_APPROVED,
            evs.REVIEW_REJECTED,
            evs.REVIEW_REVISION_REQUESTED,
        ):
            resolved.add(sid)
    still_stale = queued_before_cutoff - resolved
    if not still_stale:
        return None
    return Alert(
        kind="pending_review_queue_stale",
        severity="warning",
        summary=(
            f"{len(still_stale)} sign(s) pending review for > "
            f"{ctx.pending_review_stale_hours:.0f}h; queue depth {depth}"
        ),
        details={
            "queue_depth": depth,
            "stale_ids": sorted(still_stale)[:20],
            "stale_threshold_hours": ctx.pending_review_stale_hours,
        },
    )


def rule_injection_detected(ctx: AlerterContext) -> Optional[Alert]:
    """Fire on ANY injection detection in the last 15 minutes."""
    cutoff = ctx.now - timedelta(minutes=15)
    hits = [
        e
        for e in ctx.events_recent_hour
        if e.get("event") == evs.SECURITY_INJECTION_DETECTED
        and _ts(e)
        and _ts(e) >= cutoff
    ]
    if not hits:
        return None
    return Alert(
        kind="injection_detected",
        severity="error",
        summary=f"{len(hits)} prompt-injection attempt(s) in last 15m",
        details={
            "count": len(hits),
            "sample": [
                {
                    "ts": h.get("ts"),
                    "verdict": h.get("verdict"),
                    "field": h.get("field"),
                    "client_ip": h.get("client_ip"),
                }
                for h in hits[:5]
            ],
        },
    )


DEFAULT_RULES: tuple[Rule, ...] = (
    rule_llm_failure_rate,
    rule_session_abandonment,
    rule_daily_cost_projection,
    rule_pending_review_queue_deep,
    rule_injection_detected,
)


# ---------------------------------------------------------------------------
# Alerter
# ---------------------------------------------------------------------------


class Alerter:
    """Evaluates rules and publishes alerts with per-kind silencing."""

    def __init__(
        self,
        *,
        sink: AlertSink,
        rules: Iterable[Rule] = DEFAULT_RULES,
        silence_seconds: int = DEFAULT_SILENCE_S,
        logger_obj: Optional[EventLogger] = None,
        log_dir: Optional[Path] = None,
        daily_budget_usd: Optional[float] = None,
        pending_review_stale_hours: float = DEFAULT_PENDING_REVIEW_STALE_H,
        registry: Optional[metrics_mod.MetricsRegistry] = None,
    ) -> None:
        self.sink = sink
        self.rules = list(rules)
        if silence_seconds < 0:
            raise ValueError("silence_seconds must be >= 0")
        self.silence_seconds = silence_seconds
        self._last_fired: dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._logger = logger_obj or get_logger()
        self._log_dir = log_dir or DEFAULT_LOG_DIR
        self.daily_budget_usd = daily_budget_usd
        self.pending_review_stale_hours = pending_review_stale_hours
        self._registry = registry or metrics_mod.registry()

    def evaluate(self, *, now: Optional[datetime] = None) -> list[Alert]:
        """Run every rule once, return fired alerts, publish non-silenced ones."""
        now = now or datetime.now(timezone.utc)
        events_hour, events_day = self._collect_events(now)
        ctx = AlerterContext(
            now=now,
            events_recent_hour=events_hour,
            events_recent_day=events_day,
            metrics_registry=self._registry,
            daily_budget_usd=self.daily_budget_usd,
            pending_review_stale_hours=self.pending_review_stale_hours,
        )
        fired: list[Alert] = []
        for rule in self.rules:
            try:
                alert = rule(ctx)
            except Exception as exc:  # rule bugs must not crash the loop
                logger.exception("alert rule %s raised: %s", rule.__name__, exc)
                continue
            if alert is None:
                continue
            fired.append(alert)
            if self._should_publish(alert, now):
                self.sink.publish(alert)
                self._record_fire(alert, now)
        return fired

    def _should_publish(self, alert: Alert, now: datetime) -> bool:
        with self._lock:
            last = self._last_fired.get(alert.kind)
        if last is None:
            return True
        elapsed = (now - last).total_seconds()
        return elapsed >= self.silence_seconds

    def _record_fire(self, alert: Alert, now: datetime) -> None:
        with self._lock:
            self._last_fired[alert.kind] = now

    def _collect_events(
        self, now: datetime
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        hour_cutoff = now - timedelta(hours=1)
        day_cutoff = now - timedelta(days=1)
        events_hour: list[dict[str, Any]] = []
        events_day: list[dict[str, Any]] = []
        # Ring buffer first — always the freshest data.
        for ev in self._logger.recent():
            row = ev.to_dict()
            ts = _ts(row)
            if ts is None:
                continue
            if ts >= hour_cutoff:
                events_hour.append(row)
            if ts >= day_cutoff:
                events_day.append(row)
        # Then the day's on-disk file for events that rolled out of the ring.
        today_path = self._log_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
        if today_path.exists():
            for row in read_jsonl(today_path):
                ts = _ts(row)
                if ts is None:
                    continue
                if ts >= hour_cutoff:
                    events_hour.append(row)
                if ts >= day_cutoff:
                    events_day.append(row)
        # And yesterday's file for day-window tail.
        yday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        yday_path = self._log_dir / f"{yday}.jsonl"
        if yday_path.exists():
            for row in read_jsonl(yday_path):
                ts = _ts(row)
                if ts is None or ts < day_cutoff:
                    continue
                events_day.append(row)
        return events_hour, events_day


# ---------------------------------------------------------------------------
# Runner — drives the evaluate loop on a schedule
# ---------------------------------------------------------------------------


def build_default_alerter(
    *,
    sink: Optional[AlertSink] = None,
    logger_obj: Optional[EventLogger] = None,
) -> Alerter:
    """Return an :class:`Alerter` wired from environment variables."""
    resolved_sink: AlertSink
    if sink is not None:
        resolved_sink = sink
    else:
        url = os.environ.get("CHAT2HAMNOSYS_ALERT_WEBHOOK_URL", "").strip()
        resolved_sink = WebhookSink(url) if url else RecordingSink()

    budget_raw = os.environ.get("CHAT2HAMNOSYS_DAILY_BUDGET_USD", "").strip()
    budget = float(budget_raw) if budget_raw else None

    stale_raw = os.environ.get("CHAT2HAMNOSYS_PENDING_REVIEW_STALE_H", "").strip()
    stale_hours = float(stale_raw) if stale_raw else DEFAULT_PENDING_REVIEW_STALE_H

    silence_raw = os.environ.get("CHAT2HAMNOSYS_ALERT_SILENCE_S", "").strip()
    silence = int(silence_raw) if silence_raw else DEFAULT_SILENCE_S

    return Alerter(
        sink=resolved_sink,
        daily_budget_usd=budget,
        pending_review_stale_hours=stale_hours,
        silence_seconds=silence,
        logger_obj=logger_obj,
    )


def run_alerter_loop(
    alerter: Alerter,
    *,
    interval_seconds: Optional[int] = None,
    stop: Optional[threading.Event] = None,
) -> None:
    """Evaluate the alerter every ``interval_seconds`` until ``stop`` is set.

    Intended for a dedicated background thread. Exceptions from individual
    rules are swallowed inside :meth:`Alerter.evaluate`; a broken sink
    is logged and the loop continues.
    """
    raw = interval_seconds if interval_seconds is not None else int(
        os.environ.get("CHAT2HAMNOSYS_ALERT_INTERVAL_S", str(DEFAULT_INTERVAL_S))
    )
    if raw <= 0:
        raise ValueError("interval_seconds must be > 0")
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            alerter.evaluate()
        except Exception as exc:
            logger.exception("alerter.evaluate raised: %s", exc)
        if stop.wait(raw):
            break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(row: dict[str, Any]) -> Optional[datetime]:
    raw = row.get("ts")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


__all__ = [
    "Alert",
    "AlertSink",
    "Alerter",
    "AlerterContext",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_PENDING_REVIEW_STALE_H",
    "DEFAULT_RULES",
    "DEFAULT_SILENCE_S",
    "RecordingSink",
    "Rule",
    "WebhookSink",
    "build_default_alerter",
    "rule_daily_cost_projection",
    "rule_injection_detected",
    "rule_llm_failure_rate",
    "rule_pending_review_queue_deep",
    "rule_session_abandonment",
    "run_alerter_loop",
]
