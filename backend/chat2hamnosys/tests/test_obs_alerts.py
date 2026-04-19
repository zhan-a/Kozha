"""Tests for the rule-based alerter and its default rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from obs import events as evs
from obs import metrics as _metrics
from obs.alerts import (
    Alert,
    Alerter,
    AlerterContext,
    RecordingSink,
    WebhookSink,
    build_default_alerter,
    rule_daily_cost_projection,
    rule_injection_detected,
    rule_llm_failure_rate,
    rule_pending_review_queue_deep,
    rule_session_abandonment,
)
from obs.logger import EventLogger, reset_logger, set_logger


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 19, 15, 0, 0, tzinfo=timezone.utc)


def _row(event: str, *, ts: datetime, **fields) -> dict:
    out = {
        "ts": ts.isoformat(),
        "level": "info",
        "event": event,
        "session_id": fields.pop("session_id", None),
    }
    out.update(fields)
    return out


def _ctx(*, now: datetime, events_hour=None, events_day=None, **kw) -> AlerterContext:
    return AlerterContext(
        now=now,
        events_recent_hour=list(events_hour or []),
        events_recent_day=list(events_day or []),
        metrics_registry=kw.pop("registry", _metrics.registry()),
        daily_budget_usd=kw.pop("daily_budget_usd", None),
        pending_review_stale_hours=kw.pop("pending_review_stale_hours", 24.0),
    )


def test_llm_failure_rate_fires_above_threshold(now: datetime) -> None:
    events = [
        _row(evs.LLM_CALL_SUCCEEDED, ts=now - timedelta(minutes=5))
        for _ in range(7)
    ] + [
        _row(evs.LLM_CALL_FAILED, ts=now - timedelta(minutes=3))
        for _ in range(5)
    ]
    ctx = _ctx(now=now, events_hour=events)
    alert = rule_llm_failure_rate(ctx)
    assert alert is not None
    assert alert.kind == "llm_failure_rate_high"
    assert alert.severity == "error"


def test_llm_failure_rate_silent_below_min_calls(now: datetime) -> None:
    events = [_row(evs.LLM_CALL_FAILED, ts=now) for _ in range(3)]
    assert rule_llm_failure_rate(_ctx(now=now, events_hour=events)) is None


def test_session_abandonment_fires(now: datetime) -> None:
    events = [
        _row(evs.SESSION_CREATED, ts=now - timedelta(minutes=30))
        for _ in range(10)
    ] + [
        _row(evs.SESSION_ABANDONED, ts=now - timedelta(minutes=5))
        for _ in range(6)
    ] + [
        _row(evs.SESSION_ACCEPTED, ts=now - timedelta(minutes=5))
        for _ in range(2)
    ]
    alert = rule_session_abandonment(_ctx(now=now, events_hour=events))
    assert alert is not None
    assert alert.kind == "session_abandonment_high"


def test_daily_cost_projection_fires_over_budget(now: datetime) -> None:
    # At 15:00 UTC (elapsed 15h), $1.50 spent → projection ≈ 2.40
    events_day = [
        _row(evs.LLM_CALL_SUCCEEDED, ts=now - timedelta(hours=3), cost_usd=0.5),
        _row(evs.LLM_CALL_SUCCEEDED, ts=now - timedelta(hours=2), cost_usd=1.0),
    ]
    ctx = _ctx(now=now, events_day=events_day, daily_budget_usd=2.0)
    alert = rule_daily_cost_projection(ctx)
    assert alert is not None
    assert alert.kind == "daily_cost_projected_over_budget"
    assert alert.severity == "error"


def test_daily_cost_projection_silent_without_budget(now: datetime) -> None:
    events_day = [
        _row(evs.LLM_CALL_SUCCEEDED, ts=now - timedelta(hours=3), cost_usd=10.0),
    ]
    ctx = _ctx(now=now, events_day=events_day, daily_budget_usd=None)
    assert rule_daily_cost_projection(ctx) is None


def test_pending_review_queue_stale(now: datetime) -> None:
    _metrics.pending_reviews_queue_depth.set(value=55.0)
    events_day = [
        _row(
            evs.REVIEW_QUEUED,
            ts=now - timedelta(hours=30),
            session_id="s1",
        ),
        _row(
            evs.REVIEW_QUEUED,
            ts=now - timedelta(hours=26),
            session_id="s2",
        ),
    ]
    ctx = _ctx(now=now, events_day=events_day, pending_review_stale_hours=24.0)
    alert = rule_pending_review_queue_deep(ctx)
    assert alert is not None
    assert alert.kind == "pending_review_queue_stale"
    _metrics.pending_reviews_queue_depth.set(value=0.0)


def test_pending_review_queue_silent_when_depth_low(now: datetime) -> None:
    _metrics.pending_reviews_queue_depth.set(value=5.0)
    events_day = [
        _row(evs.REVIEW_QUEUED, ts=now - timedelta(hours=30), session_id="s1"),
    ]
    ctx = _ctx(now=now, events_day=events_day)
    assert rule_pending_review_queue_deep(ctx) is None


def test_injection_detected_fires(now: datetime) -> None:
    events = [
        _row(
            evs.SECURITY_INJECTION_DETECTED,
            ts=now - timedelta(minutes=3),
            verdict="INSTRUCTION",
            client_ip="1.2.3.4",
            field="description",
        )
    ]
    alert = rule_injection_detected(_ctx(now=now, events_hour=events))
    assert alert is not None
    assert alert.kind == "injection_detected"
    assert alert.severity == "error"


def test_alerter_publishes_and_silences(now: datetime, tmp_path: Path) -> None:
    lg = EventLogger(log_dir=tmp_path, sink="file")
    set_logger(lg)
    try:
        for _ in range(12):
            lg.emit(evs.LLM_CALL_FAILED, model="gpt-4o")
        sink = RecordingSink()
        alerter = Alerter(
            sink=sink,
            rules=(rule_llm_failure_rate,),
            silence_seconds=3600,
            logger_obj=lg,
            log_dir=tmp_path,
        )
        fired_first = alerter.evaluate(now=now)
        assert len(fired_first) == 1
        # Second evaluate within silence window — no additional publish.
        alerter.evaluate(now=now + timedelta(seconds=60))
        assert len(sink.alerts) == 1
        # After silence expires — re-publishes.
        alerter.evaluate(now=now + timedelta(seconds=3700))
        assert len(sink.alerts) == 2
    finally:
        reset_logger()


def test_alerter_swallows_rule_exceptions(now: datetime, tmp_path: Path) -> None:
    def broken_rule(ctx: AlerterContext) -> Alert | None:
        raise RuntimeError("boom")

    lg = EventLogger(log_dir=tmp_path, sink="file")
    set_logger(lg)
    try:
        sink = RecordingSink()
        alerter = Alerter(
            sink=sink,
            rules=(broken_rule,),
            logger_obj=lg,
            log_dir=tmp_path,
        )
        fired = alerter.evaluate(now=now)
        assert fired == []
        assert sink.alerts == []
    finally:
        reset_logger()


def test_webhook_sink_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        WebhookSink("")


def test_build_default_alerter_reads_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_DAILY_BUDGET_USD", "12.34")
    monkeypatch.setenv("CHAT2HAMNOSYS_ALERT_SILENCE_S", "900")
    monkeypatch.delenv("CHAT2HAMNOSYS_ALERT_WEBHOOK_URL", raising=False)
    lg = EventLogger(log_dir=tmp_path, sink="file")
    set_logger(lg)
    try:
        alerter = build_default_alerter(logger_obj=lg)
        assert alerter.daily_budget_usd == pytest.approx(12.34)
        assert alerter.silence_seconds == 900
        assert isinstance(alerter.sink, RecordingSink)
    finally:
        reset_logger()
