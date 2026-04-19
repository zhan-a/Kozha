"""Tests for dashboard aggregators and HTML rendering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from obs import events as evs
from obs.dashboard import (
    compute_cost_breakdown,
    compute_error_rates,
    compute_funnel,
    compute_hourly_activity,
    compute_llm_breakdown,
    compute_today_numbers,
    render_cost_report,
    render_dashboard,
    render_session_trace,
)
from obs.logger import EventLogger, reset_logger, set_logger


def _ev(event: str, ts: datetime, **fields) -> dict:
    out = {
        "ts": ts.isoformat(),
        "level": "info",
        "event": event,
        "session_id": fields.pop("session_id", None),
        "request_id": fields.pop("request_id", None),
        "user_hash": fields.pop("user_hash", None),
    }
    out.update(fields)
    return out


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 19, 15, 0, 0, tzinfo=timezone.utc)


def test_funnel_counts_each_stage(now: datetime) -> None:
    events = [
        _ev(evs.SESSION_CREATED, now - timedelta(minutes=60), session_id="a"),
        _ev(evs.SESSION_CREATED, now - timedelta(minutes=50), session_id="b"),
        _ev(evs.PARSE_DESCRIPTION_COMPLETED, now - timedelta(minutes=40), session_id="a"),
        _ev(evs.GENERATE_HAMNOSYS_VALIDATED, now - timedelta(minutes=30), session_id="a"),
    ]
    stages = compute_funnel(events)
    by_stage = {s["stage"]: s["count"] for s in stages}
    # Stage labels come from FUNNEL_STAGES, not the raw event constants.
    assert by_stage["session.created"] == 2
    assert by_stage["description.submitted"] == 1
    assert by_stage["generation.succeeded"] == 1


def test_llm_breakdown_reports_calls_cost_failures(now: datetime) -> None:
    events = [
        _ev(
            evs.LLM_CALL_SUCCEEDED,
            now,
            model="gpt-4o",
            latency_ms=500,
            cost_usd=0.01,
            input_tokens=100,
            output_tokens=50,
        ),
        _ev(
            evs.LLM_CALL_SUCCEEDED,
            now,
            model="gpt-4o",
            latency_ms=1000,
            cost_usd=0.02,
            input_tokens=200,
            output_tokens=100,
        ),
        _ev(
            evs.LLM_CALL_FAILED,
            now,
            model="gpt-4o",
            error_class="RateLimitError",
        ),
    ]
    rows = compute_llm_breakdown(events)
    assert len(rows) >= 1
    total_calls = sum(r.get("calls", 0) for r in rows)
    total_failures = sum(r.get("failures", 0) for r in rows)
    total_cost = sum(r.get("cost_usd", 0.0) for r in rows)
    assert total_calls == 3
    assert total_failures == 1
    assert total_cost == pytest.approx(0.03, abs=1e-6)


def test_error_rates_from_failure_events(now: datetime) -> None:
    events = [
        _ev(evs.LLM_CALL_SUCCEEDED, now, model="gpt-4o"),
        _ev(evs.LLM_CALL_SUCCEEDED, now, model="gpt-4o"),
        _ev(evs.LLM_CALL_FAILED, now, model="gpt-4o"),
        _ev(evs.RENDER_PREVIEW_FAILED, now),
        _ev(evs.GENERATE_HAMNOSYS_GAVE_UP, now),
    ]
    rates = compute_error_rates(events, now=now)
    assert isinstance(rates, dict)
    # Some aggregator for llm failures should be present
    assert any("llm" in k.lower() or "fail" in k.lower() for k in rates)


def test_hourly_activity_buckets_by_hour(now: datetime) -> None:
    events = [
        _ev(evs.SESSION_CREATED, now - timedelta(hours=2)),
        _ev(evs.SESSION_CREATED, now - timedelta(hours=2)),
        _ev(evs.SESSION_CREATED, now - timedelta(hours=1)),
        _ev(evs.SESSION_CREATED, now),
    ]
    buckets = compute_hourly_activity(events, now=now)
    assert isinstance(buckets, list)
    assert len(buckets) > 0
    total = sum(b.get("count", 0) for b in buckets)
    assert total == 4


def test_today_numbers_counts_sessions(now: datetime) -> None:
    events = [
        _ev(evs.SESSION_CREATED, now - timedelta(minutes=10)),
        _ev(evs.SESSION_CREATED, now - timedelta(minutes=20)),
        _ev(evs.SESSION_ACCEPTED, now - timedelta(minutes=5)),
    ]
    nums = compute_today_numbers(events, now=now)
    assert isinstance(nums, dict)
    # At least some of: sessions_started, sessions_accepted, llm_calls, cost
    lower = {k.lower(): v for k, v in nums.items()}
    assert any("session" in k for k in lower)


def test_cost_breakdown_reads_jsonl_files(tmp_path: Path, now: datetime) -> None:
    date_str = now.strftime("%Y-%m-%d")
    path = tmp_path / f"{date_str}.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"ts":"' + now.isoformat() + '","event":"llm.call.succeeded","model":"gpt-4o","cost_usd":0.01}',
                '{"ts":"' + now.isoformat() + '","event":"llm.call.succeeded","model":"gpt-4o","cost_usd":0.02}',
            ]
        )
        + "\n"
    )
    result = compute_cost_breakdown(tmp_path, now=now, days=7)
    assert isinstance(result, dict)
    # Aggregate cost reflected somewhere
    flat = str(result)
    assert "0.03" in flat or "0.030" in flat or "0.0300" in flat or 0.03 in [v for v in result.values() if isinstance(v, (int, float))]


def test_render_dashboard_returns_html(tmp_path: Path, now: datetime) -> None:
    lg = EventLogger(log_dir=tmp_path, sink="file", retention_days=30)
    set_logger(lg)
    try:
        lg.emit(evs.SESSION_CREATED, session_id="s1")
        html = render_dashboard(logger=lg, log_dir=tmp_path, now=now)
        assert "<html" in html.lower() or "<!doctype" in html.lower()
        assert "session" in html.lower()
    finally:
        reset_logger()


def test_render_session_trace_includes_events(tmp_path: Path) -> None:
    lg = EventLogger(log_dir=tmp_path, sink="file", retention_days=30)
    set_logger(lg)
    try:
        lg.emit(evs.SESSION_CREATED, session_id="abc")
        lg.emit(evs.PARSE_DESCRIPTION_COMPLETED, session_id="abc", gaps_found=0)
        html = render_session_trace("abc", logger=lg, log_dir=tmp_path)
        assert "abc" in html
        assert evs.SESSION_CREATED in html
    finally:
        reset_logger()


def test_render_cost_report_returns_html(tmp_path: Path, now: datetime) -> None:
    html = render_cost_report(log_dir=tmp_path, now=now)
    assert isinstance(html, str)
    assert "<html" in html.lower() or "<!doctype" in html.lower()
