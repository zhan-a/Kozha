"""HTML rendering for ``/admin/dashboard``, per-session traces, and cost reports.

The admin pages are deliberately framework-free: a few inline-styled
HTML snippets stitched together by Python ``str.format``. There is no
JavaScript dependency, no template engine, no asset pipeline. Adding
Grafana later does not require touching this module — Grafana would
scrape ``/metrics`` directly, and these pages would remain the
zero-dependency fallback.

Public functions
----------------
- :func:`render_dashboard` — the top-level overview HTML.
- :func:`render_session_trace` — single-session timeline view.
- :func:`render_cost_report` — 30-day cost breakdown + projection.
- :func:`compute_funnel` / :func:`compute_llm_breakdown` /
  :func:`compute_error_rates` / :func:`compute_hourly_activity` —
  pure-Python aggregators tested separately from rendering.

All pure aggregators read from :mod:`obs.logger`'s in-process ring or
the on-disk JSONL files, so no LLM/network calls happen during a
dashboard request.
"""

from __future__ import annotations

import html
from collections import Counter as _Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from . import events as evs
from . import metrics as metrics_mod
from .logger import EventLogger, StructuredEvent, iter_log_files, read_jsonl


# ---------------------------------------------------------------------------
# Aggregators (pure)
# ---------------------------------------------------------------------------


# The funnel must read in this order; each downstream stage's count is
# the number of sessions that produced its event at any point.
FUNNEL_STAGES: tuple[tuple[str, str], ...] = (
    ("session.created", evs.SESSION_CREATED),
    ("description.submitted", evs.PARSE_DESCRIPTION_COMPLETED),
    ("generation.succeeded", evs.GENERATE_HAMNOSYS_VALIDATED),
    ("review.approved", evs.REVIEW_APPROVED),
    ("export.succeeded", evs.EXPORT_SUCCEEDED),
)


def compute_funnel(events: Iterable[dict[str, Any] | StructuredEvent]) -> list[dict[str, Any]]:
    """Return per-stage absolute counts and drop-off percentages.

    ``events`` may be a mix of dicts (from JSONL) and :class:`StructuredEvent`
    (from the in-process ring). Sessions are deduped by ``session_id``
    so a single session contributing two ``parse.description.completed``
    events still counts once for that stage.
    """
    seen_per_stage: dict[str, set[str]] = {name: set() for name, _ in FUNNEL_STAGES}
    for raw in events:
        ev = _to_dict(raw)
        sid = ev.get("session_id")
        if not sid:
            continue
        for stage_name, stage_event in FUNNEL_STAGES:
            if ev.get("event") == stage_event:
                seen_per_stage[stage_name].add(sid)
    out: list[dict[str, Any]] = []
    prev = None
    for stage_name, _ in FUNNEL_STAGES:
        count = len(seen_per_stage[stage_name])
        if prev is None or prev == 0:
            drop_pct = 0.0
        else:
            drop_pct = max(0.0, 100.0 * (1 - count / prev))
        out.append(
            {
                "stage": stage_name,
                "count": count,
                "drop_pct": round(drop_pct, 1),
            }
        )
        prev = count
    return out


def compute_llm_breakdown(
    events: Iterable[dict[str, Any] | StructuredEvent],
) -> list[dict[str, Any]]:
    """Return per-prompt-id totals — calls, tokens, cost.

    The LLM telemetry from :mod:`llm.telemetry` is the authoritative
    source; this aggregator works over the unified event stream so the
    dashboard can mix in events the dashboard logger emitted directly.
    """
    by_prompt: dict[str, dict[str, float]] = {}
    for raw in events:
        ev = _to_dict(raw)
        if ev.get("event") not in (evs.LLM_CALL_SUCCEEDED, evs.LLM_CALL_FAILED):
            continue
        pid = ev.get("prompt_id") or "(no prompt_id)"
        bucket = by_prompt.setdefault(
            pid,
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "failures": 0,
            },
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += int(ev.get("input_tokens", 0) or 0)
        bucket["output_tokens"] += int(ev.get("output_tokens", 0) or 0)
        bucket["cost_usd"] += float(ev.get("cost_usd", 0.0) or 0.0)
        if ev.get("event") == evs.LLM_CALL_FAILED:
            bucket["failures"] += 1
    rows = []
    for pid, b in by_prompt.items():
        rows.append({"prompt_id": pid, **b, "cost_usd": round(b["cost_usd"], 4)})
    rows.sort(key=lambda r: r["cost_usd"], reverse=True)
    return rows


def compute_error_rates(
    events: Iterable[dict[str, Any] | StructuredEvent],
    *,
    now: datetime | None = None,
) -> dict[str, dict[str, int]]:
    """Return ``{kind: {hour, day, week}}`` failure counts.

    Three kinds: ``llm`` (``llm.call.failed``), ``validator``
    (``generate.hamnosys.gave_up``), ``render`` (``render.preview.failed``).
    """
    now = now or datetime.now(timezone.utc)
    windows = {
        "hour": now - timedelta(hours=1),
        "day": now - timedelta(days=1),
        "week": now - timedelta(weeks=1),
    }
    rates = {
        "llm": {"hour": 0, "day": 0, "week": 0},
        "validator": {"hour": 0, "day": 0, "week": 0},
        "render": {"hour": 0, "day": 0, "week": 0},
    }
    target = {
        evs.LLM_CALL_FAILED: "llm",
        evs.GENERATE_HAMNOSYS_GAVE_UP: "validator",
        evs.RENDER_PREVIEW_FAILED: "render",
    }
    for raw in events:
        ev = _to_dict(raw)
        kind = target.get(ev.get("event") or "")
        if not kind:
            continue
        ts = _parse_ts(ev.get("ts"))
        if ts is None:
            continue
        for window_name, threshold in windows.items():
            if ts >= threshold:
                rates[kind][window_name] += 1
    return rates


def compute_hourly_activity(
    events: Iterable[dict[str, Any] | StructuredEvent],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return last-24h hourly buckets of ``session.created`` counts.

    Buckets are returned oldest-first so the dashboard sparkline plots
    naturally left-to-right.
    """
    now = now or datetime.now(timezone.utc)
    floor = now.replace(minute=0, second=0, microsecond=0)
    bucket_starts = [floor - timedelta(hours=23 - i) for i in range(24)]
    counts = [0] * 24
    for raw in events:
        ev = _to_dict(raw)
        if ev.get("event") != evs.SESSION_CREATED:
            continue
        ts = _parse_ts(ev.get("ts"))
        if ts is None:
            continue
        for i, start in enumerate(bucket_starts):
            end = start + timedelta(hours=1)
            if start <= ts < end:
                counts[i] += 1
                break
    return [
        {"hour": start.isoformat(), "count": count}
        for start, count in zip(bucket_starts, counts, strict=True)
    ]


def compute_today_numbers(
    events: Iterable[dict[str, Any] | StructuredEvent],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Top-row counters: started, accepted, pending reviews, cost today."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    started = 0
    accepted = 0
    cost_today = 0.0
    for raw in events:
        ev = _to_dict(raw)
        ts = _parse_ts(ev.get("ts"))
        if ts is None or ts.date() != today:
            continue
        if ev.get("event") == evs.SESSION_CREATED:
            started += 1
        elif ev.get("event") == evs.SESSION_ACCEPTED:
            accepted += 1
        if ev.get("event") in (evs.LLM_CALL_SUCCEEDED, evs.LLM_CALL_FAILED):
            cost_today += float(ev.get("cost_usd", 0.0) or 0.0)
    pending = int(metrics_mod.pending_reviews_queue_depth.get())
    return {
        "sessions_started_today": started,
        "signs_authored_today": accepted,
        "pending_reviews": pending,
        "cost_today_usd": round(cost_today, 4),
    }


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------


def compute_cost_breakdown(
    log_dir: Path,
    *,
    now: datetime | None = None,
    days: int = 30,
) -> dict[str, Any]:
    """30-day breakdown by date / prompt id / model + month projection.

    Reads ``logs/<YYYY-MM-DD>.jsonl`` directly so the report is the
    single source of truth even after the in-process ring has rolled.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now.date() - timedelta(days=days - 1)
    by_date: dict[str, float] = {}
    by_prompt: dict[str, float] = {}
    by_model: dict[str, float] = {}
    per_session: dict[str, float] = {}
    for path in iter_log_files(log_dir):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if file_date < cutoff:
            continue
        for row in read_jsonl(path):
            if row.get("event") not in (evs.LLM_CALL_SUCCEEDED, evs.LLM_CALL_FAILED):
                continue
            cost = float(row.get("cost_usd", 0.0) or 0.0)
            ds = row.get("ts", "")[:10]
            by_date[ds] = by_date.get(ds, 0.0) + cost
            by_prompt[row.get("prompt_id") or "(no prompt_id)"] = (
                by_prompt.get(row.get("prompt_id") or "(no prompt_id)", 0.0) + cost
            )
            by_model[row.get("model") or "(unknown)"] = (
                by_model.get(row.get("model") or "(unknown)", 0.0) + cost
            )
            sid = row.get("session_id")
            if sid:
                per_session[sid] = per_session.get(sid, 0.0) + cost

    sorted_dates = sorted(by_date.items())
    daily_costs = [v for _, v in sorted_dates]
    median_daily = _median(daily_costs)
    anomalies = [
        {"date": d, "cost_usd": round(v, 4)}
        for d, v in sorted_dates
        if median_daily > 0 and v > 2 * median_daily
    ]
    expensive_sessions = sorted(
        ({"session_id": s, "cost_usd": round(c, 4)} for s, c in per_session.items() if c > 1.0),
        key=lambda r: r["cost_usd"],
        reverse=True,
    )

    today_iso = now.date().isoformat()
    today_cost = by_date.get(today_iso, 0.0)
    days_this_month = now.day
    month_to_date_total = sum(
        v for d, v in sorted_dates if d[:7] == now.strftime("%Y-%m")
    )
    if days_this_month > 0:
        avg_per_day = month_to_date_total / days_this_month
    else:
        avg_per_day = 0.0
    days_in_month = _days_in_month(now)
    projected_month = round(avg_per_day * days_in_month, 2)

    return {
        "by_date": [{"date": d, "cost_usd": round(v, 4)} for d, v in sorted_dates],
        "by_prompt": sorted(
            ({"prompt_id": k, "cost_usd": round(v, 4)} for k, v in by_prompt.items()),
            key=lambda r: r["cost_usd"],
            reverse=True,
        ),
        "by_model": sorted(
            ({"model": k, "cost_usd": round(v, 4)} for k, v in by_model.items()),
            key=lambda r: r["cost_usd"],
            reverse=True,
        ),
        "today_cost_usd": round(today_cost, 4),
        "month_to_date_usd": round(month_to_date_total, 4),
        "projection_month_usd": projected_month,
        "median_daily_usd": round(median_daily, 4),
        "anomalous_days": anomalies,
        "expensive_sessions": expensive_sessions,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_BASE_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
       background: #f7f7fa; color: #1d1d1f; }
h1 { margin: 0 0 16px 0; font-size: 22px; }
h2 { margin: 24px 0 8px 0; font-size: 16px; color: #444; }
.section { background: white; border: 1px solid #e0e0e6; border-radius: 8px;
           padding: 16px; margin-bottom: 16px; }
.row { display: flex; gap: 12px; flex-wrap: wrap; }
.kpi { flex: 1 1 180px; background: white; border: 1px solid #e0e0e6;
       border-radius: 8px; padding: 16px; min-width: 160px; }
.kpi-label { font-size: 12px; color: #666; text-transform: uppercase;
             letter-spacing: 0.04em; }
.kpi-value { font-size: 28px; font-weight: 600; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }
th { background: #fafafc; }
tr:hover td { background: #fafbff; }
.spark { display: flex; align-items: flex-end; gap: 2px; height: 40px;
         margin: 8px 0; }
.spark-bar { flex: 1; background: #4a7cff; min-height: 1px; border-radius: 1px; }
.feed-item { padding: 6px 8px; border-bottom: 1px solid #eee;
             font-family: ui-monospace, monospace; font-size: 12px; }
.feed-info { color: #555; }
.feed-warning { color: #b45309; }
.feed-error { color: #b91c1c; font-weight: 600; }
.muted { color: #888; }
.funnel-bar { height: 14px; background: #4a7cff; border-radius: 3px; }
a { color: #2050c0; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def _kpi(label: str, value: Any) -> str:
    return (
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        "</div>"
    )


def _spark(values: list[int]) -> str:
    if not values:
        return '<div class="spark"></div>'
    high = max(values) or 1
    bars = "".join(
        f'<div class="spark-bar" style="height:{max(1, int(40 * v / high))}px"></div>'
        for v in values
    )
    return f'<div class="spark">{bars}</div>'


def _funnel_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No funnel data yet.</p>"
    top = max(r["count"] for r in rows) or 1
    parts = ["<table><tr><th>Stage</th><th>Count</th><th>Drop-off</th><th></th></tr>"]
    for r in rows:
        width_pct = max(2, int(100 * r["count"] / top))
        parts.append(
            "<tr>"
            f"<td>{html.escape(r['stage'])}</td>"
            f"<td>{r['count']}</td>"
            f"<td class='muted'>{r['drop_pct']:.1f}%</td>"
            f'<td><div class="funnel-bar" style="width:{width_pct}%"></div></td>'
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _llm_table_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No LLM calls recorded yet.</p>"
    parts = [
        "<table><tr>"
        "<th>Prompt id</th><th>Calls</th><th>Failures</th>"
        "<th>Input tokens</th><th>Output tokens</th><th>Cost (USD)</th>"
        "</tr>"
    ]
    for r in rows:
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(r['prompt_id']))}</td>"
            f"<td>{r['calls']}</td>"
            f"<td>{r['failures']}</td>"
            f"<td>{r['input_tokens']}</td>"
            f"<td>{r['output_tokens']}</td>"
            f"<td>${r['cost_usd']:.4f}</td>"
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _error_rates_html(rates: dict[str, dict[str, int]]) -> str:
    parts = [
        "<table><tr><th>Surface</th><th>Last hour</th><th>Last day</th><th>Last week</th></tr>"
    ]
    for kind, windows in rates.items():
        parts.append(
            "<tr>"
            f"<td>{html.escape(kind)}</td>"
            f"<td>{windows['hour']}</td>"
            f"<td>{windows['day']}</td>"
            f"<td>{windows['week']}</td>"
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _feed_html(events: list[StructuredEvent], *, prefix: str = "") -> str:
    if not events:
        return "<p class='muted'>No recent events.</p>"
    parts = []
    for ev in reversed(events):
        cls = f"feed-{ev.level}"
        sid = ev.session_id or ""
        link = (
            f' <a href="{prefix}/admin/sessions/{html.escape(sid)}">[trace]</a>'
            if sid
            else ""
        )
        when = ev.ts.strftime("%H:%M:%S")
        parts.append(
            f'<div class="feed-item {cls}">'
            f"{when} <strong>{html.escape(ev.event)}</strong>"
            f" <span class='muted'>session={html.escape(sid[:8] if sid else '—')}</span>"
            f"{link}</div>"
        )
    return "\n".join(parts)


def render_dashboard(
    *,
    logger: EventLogger,
    log_dir: Path,
    now: datetime | None = None,
    prefix: str = "",
) -> str:
    """Top-level dashboard HTML."""
    ring = logger.recent()
    file_events: list[dict[str, Any]] = []
    for path in iter_log_files(log_dir):
        file_events.extend(read_jsonl(path))
    combined = file_events + [e.to_dict() for e in ring]

    today = compute_today_numbers(combined, now=now)
    funnel = compute_funnel(combined)
    llm_rows = compute_llm_breakdown(combined)
    rates = compute_error_rates(combined, now=now)
    hourly = compute_hourly_activity(combined, now=now)
    spark_values = [b["count"] for b in hourly]

    body = f"""
    <h1>chat2hamnosys — operations dashboard</h1>

    <div class="row">
      {_kpi("Sessions today", today["sessions_started_today"])}
      {_kpi("Signs authored today", today["signs_authored_today"])}
      {_kpi("Pending reviews", today["pending_reviews"])}
      {_kpi("Cost today", f'${today["cost_today_usd"]:.4f}')}
    </div>

    <div class="section">
      <h2>Sessions per hour (last 24h)</h2>
      {_spark(spark_values)}
    </div>

    <div class="section">
      <h2>Funnel</h2>
      {_funnel_html(funnel)}
    </div>

    <div class="section">
      <h2>LLM breakdown by prompt id</h2>
      {_llm_table_html(llm_rows)}
    </div>

    <div class="section">
      <h2>Error rates</h2>
      {_error_rates_html(rates)}
    </div>

    <div class="section">
      <h2>Recent events</h2>
      {_feed_html(ring[-50:], prefix=prefix)}
    </div>

    <p class="muted"><a href="{prefix}/admin/cost">Cost report &rarr;</a> ·
       <a href="{prefix}/metrics">/metrics</a></p>
    """
    return _wrap_html("chat2hamnosys ops", body)


def render_session_trace(
    session_id: str,
    *,
    logger: EventLogger,
    log_dir: Path,
) -> str:
    """Per-session timeline HTML.

    Reads from the in-memory ring first (always-fresh) and falls back to
    on-disk JSONL files for older sessions. Records are merged and
    de-duped on ``ts`` + ``event``.
    """
    ring_events = [e.to_dict() for e in logger.recent_for_session(session_id)]
    file_events: list[dict[str, Any]] = []
    for path in iter_log_files(log_dir):
        for row in read_jsonl(path):
            if row.get("session_id") == session_id:
                file_events.append(row)
    seen: set[tuple[str, str]] = set()
    rows = []
    for ev in file_events + ring_events:
        key = (ev.get("ts", ""), ev.get("event", ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(ev)
    rows.sort(key=lambda r: r.get("ts", ""))

    if not rows:
        body = f"<h1>session {html.escape(session_id)}</h1><p>No events recorded.</p>"
        return _wrap_html(f"session {session_id}", body)

    parts = [
        f"<h1>session <code>{html.escape(session_id)}</code></h1>",
        "<table><tr><th>Time</th><th>Event</th><th>Fields</th></tr>",
    ]
    for r in rows:
        ts = r.get("ts", "")
        event = r.get("event", "")
        fields = {k: v for k, v in r.items() if k not in ("ts", "event", "session_id", "level")}
        cell = ", ".join(
            f"{html.escape(str(k))}={html.escape(str(v))}"
            for k, v in fields.items()
            if v not in (None, "", [])
        )
        parts.append(
            "<tr>"
            f"<td class='muted'>{html.escape(ts)}</td>"
            f"<td><strong>{html.escape(event)}</strong></td>"
            f"<td><code>{cell}</code></td>"
            "</tr>"
        )
    parts.append("</table>")
    return _wrap_html(f"session {session_id}", "\n".join(parts))


def render_cost_report(
    *,
    log_dir: Path,
    now: datetime | None = None,
    prefix: str = "",
) -> str:
    """30-day cost breakdown HTML with month projection + anomaly callouts."""
    now = now or datetime.now(timezone.utc)
    data = compute_cost_breakdown(log_dir, now=now)
    by_date_rows = "".join(
        f"<tr><td>{html.escape(r['date'])}</td><td>${r['cost_usd']:.4f}</td></tr>"
        for r in data["by_date"]
    )
    by_prompt_rows = "".join(
        f"<tr><td>{html.escape(r['prompt_id'])}</td><td>${r['cost_usd']:.4f}</td></tr>"
        for r in data["by_prompt"]
    )
    by_model_rows = "".join(
        f"<tr><td>{html.escape(r['model'])}</td><td>${r['cost_usd']:.4f}</td></tr>"
        for r in data["by_model"]
    )
    anomaly_rows = "".join(
        f"<tr><td>{html.escape(r['date'])}</td><td>${r['cost_usd']:.4f}</td></tr>"
        for r in data["anomalous_days"]
    )
    expensive_rows = "".join(
        f"<tr><td><a href='{prefix}/admin/sessions/{html.escape(r['session_id'])}'>{html.escape(r['session_id'])}</a></td>"
        f"<td>${r['cost_usd']:.4f}</td></tr>"
        for r in data["expensive_sessions"]
    )

    body = f"""
    <h1>Cost report</h1>
    <p>Today: <strong>${data["today_cost_usd"]:.4f}</strong>
       · Month-to-date: <strong>${data["month_to_date_usd"]:.4f}</strong>
       · Projected this month: <strong>${data["projection_month_usd"]:.2f}</strong>
       · 30-day median daily: <strong>${data["median_daily_usd"]:.4f}</strong>.</p>

    <div class="section">
      <h2>By date (last 30 days)</h2>
      <table><tr><th>Date</th><th>Cost</th></tr>{by_date_rows or '<tr><td colspan=2 class="muted">no data</td></tr>'}</table>
    </div>

    <div class="section">
      <h2>By prompt id</h2>
      <table><tr><th>Prompt id</th><th>Cost</th></tr>{by_prompt_rows or '<tr><td colspan=2 class="muted">no data</td></tr>'}</table>
    </div>

    <div class="section">
      <h2>By model</h2>
      <table><tr><th>Model</th><th>Cost</th></tr>{by_model_rows or '<tr><td colspan=2 class="muted">no data</td></tr>'}</table>
    </div>

    <div class="section">
      <h2>Anomalous days (cost &gt; 2x 30-day median)</h2>
      <table><tr><th>Date</th><th>Cost</th></tr>{anomaly_rows or '<tr><td colspan=2 class="muted">none</td></tr>'}</table>
    </div>

    <div class="section">
      <h2>Expensive sessions (cost &gt; $1.00)</h2>
      <table><tr><th>Session</th><th>Cost</th></tr>{expensive_rows or '<tr><td colspan=2 class="muted">none</td></tr>'}</table>
    </div>

    <p class="muted"><a href="{prefix}/admin/dashboard">&larr; Back to dashboard</a></p>
    """
    return _wrap_html("Cost report", body)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _wrap_html(title: str, body: str) -> str:
    return (
        "<!doctype html>"
        f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_BASE_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def _to_dict(raw: dict[str, Any] | StructuredEvent) -> dict[str, Any]:
    if isinstance(raw, StructuredEvent):
        return raw.to_dict()
    return raw


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    mid = len(sorted_v) // 2
    if len(sorted_v) % 2:
        return sorted_v[mid]
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2


def _days_in_month(now: datetime) -> int:
    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1)
    else:
        next_month = now.replace(month=now.month + 1, day=1)
    last_day = (next_month - timedelta(days=1)).day
    return last_day


__all__ = [
    "FUNNEL_STAGES",
    "compute_cost_breakdown",
    "compute_error_rates",
    "compute_funnel",
    "compute_hourly_activity",
    "compute_llm_breakdown",
    "compute_today_numbers",
    "render_cost_report",
    "render_dashboard",
    "render_session_trace",
]
