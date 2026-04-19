#!/usr/bin/env python3
"""scripts/loadtest.py — concurrent-session load test for chat2hamnosys.

Simulates ``--sessions N`` concurrent authoring flows against a running
backend. Each virtual user runs the canonical happy path:

    POST /sessions
    POST /sessions/{id}/describe
    POST /sessions/{id}/answer        (×2 — one per pending question)
    POST /sessions/{id}/generate
    POST /sessions/{id}/accept

For every endpoint we record per-call latency. At the end the script
prints p50 / p95 / p99 latency, error rate, throughput, and total
estimated cost (when the server returns Prometheus metrics).

Usage
-----

::

    # Local — 10 sessions, no rate limit (tight loop)
    python scripts/loadtest.py --sessions 10

    # Staging — 50 sessions, 5 sessions per second ramp-up
    python scripts/loadtest.py \
        --base-url https://kozha-staging.fly.dev \
        --sessions 50 --rate 5

    # Production cap-check — 100 sessions, ramp at 2/s, JSON output for CI
    python scripts/loadtest.py \
        --base-url https://kozha.example \
        --sessions 100 --rate 2 \
        --output loadtest-prod-2026-04-19.json

Dependencies: ``httpx`` (already in
``backend/chat2hamnosys/requirements.txt``).

Notes
-----
- This script is intentionally **not** locust — locust would add a
  heavyweight dependency and a UI we don't need. asyncio + httpx is
  enough for the load levels documented in
  ``docs/chat2hamnosys/19-prod-checklist.md`` (10 / 50 / 100).
- Cost is *estimated* by scraping the
  ``chat2hamnosys_llm_cost_usd_total`` Prometheus counter before / after
  the run. If the metric is unavailable, cost reports as ``unknown``.
- The script never sends a real OpenAI key — it relies on the server's
  configured key. Run against staging unless you have explicit
  permission to load-test production.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Per-call recording
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    endpoint: str
    duration_s: float
    status: int
    ok: bool
    error: str = ""


@dataclass
class SessionResult:
    session_id: str = ""
    calls: list[CallResult] = field(default_factory=list)
    completed: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# A single virtual-user flow
# ---------------------------------------------------------------------------


CANONICAL_DESCRIPTION = (
    "Right hand starts open at chest height with palm facing forward, "
    "moves outward in a small arc while the fingers gradually close to "
    "a fist."
)
CANONICAL_GLOSS = "loadtest_sign"


async def run_one_session(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    idx: int,
) -> SessionResult:
    out = SessionResult()
    api = f"{base_url.rstrip('/')}/api/chat2hamnosys"

    async def call(method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any] | None]:
        url = f"{api}{path}"
        start = time.perf_counter()
        status = 0
        error = ""
        body: dict[str, Any] | None = None
        try:
            r = await client.request(method, url, timeout=60.0, **kwargs)
            status = r.status_code
            if r.headers.get("content-type", "").startswith("application/json"):
                body = r.json()
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        ok = 200 <= status < 300 and not error
        out.calls.append(
            CallResult(endpoint=f"{method} {path}", duration_s=elapsed,
                       status=status, ok=ok, error=error)
        )
        return status, body

    # 1. Create session
    status, body = await call(
        "POST",
        "/sessions",
        json={
            "signer_id": f"loadtest_{idx}",
            "display_name": f"loadtest-{idx}",
            "sign_language": "bsl",
            "domain": "general",
            "gloss": CANONICAL_GLOSS,
        },
    )
    if status != 201 or not body:
        out.error = f"create failed: {status}"
        return out
    out.session_id = body["session_id"]
    token = body["session_token"]
    headers = {"X-Session-Token": token}

    # 2. Describe
    status, body = await call(
        "POST",
        f"/sessions/{out.session_id}/describe",
        headers=headers,
        json={"prose": CANONICAL_DESCRIPTION},
    )
    if status >= 400:
        out.error = f"describe failed: {status}"
        return out

    # 3. Answer up to 2 clarifications. The server returns the queue;
    #    pick the first option (or "yes" / "1" as a freeform fallback).
    for _ in range(2):
        if not body:
            break
        next_action = body.get("next_action", {})
        if next_action.get("kind") != "answer_questions":
            break
        questions = next_action.get("questions") or []
        if not questions:
            break
        q = questions[0]
        if q.get("options"):
            answer_value = q["options"][0]["value"]
        else:
            answer_value = "yes"
        status, body = await call(
            "POST",
            f"/sessions/{out.session_id}/answer",
            headers=headers,
            json={"question_id": q["question_id"], "answer": str(answer_value)},
        )
        if status >= 400:
            out.error = f"answer failed: {status}"
            return out

    # 4. Force a generate if we're still in CLARIFYING
    if body and body.get("state") == "CLARIFYING":
        status, body = await call(
            "POST",
            f"/sessions/{out.session_id}/generate",
            headers=headers,
        )
        if status >= 400:
            out.error = f"generate failed: {status}"
            return out

    # 5. Accept (will fail if review-required policy is on; that's
    #    expected and counted as a non-2xx — production usually rejects
    #    a single approver, which is the point of the test).
    status, body = await call(
        "POST",
        f"/sessions/{out.session_id}/accept",
        headers=headers,
    )
    if status < 400:
        out.completed = True
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def fetch_metric(base_url: str, name: str) -> float | None:
    """Best-effort scrape of one Prometheus counter before/after the run."""
    url = f"{base_url.rstrip('/')}/api/chat2hamnosys/metrics"
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(url)
            r.raise_for_status()
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            m = re.match(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)", line)
            if m:
                return float(m.group(1))
    except (httpx.HTTPError, ValueError):
        return None
    return None


def percentiles(values: list[float], pcts: tuple[int, ...]) -> dict[int, float]:
    if not values:
        return {p: float("nan") for p in pcts}
    sv = sorted(values)
    out: dict[int, float] = {}
    for p in pcts:
        # nearest-rank percentile so the integer indexing is well-defined
        k = max(0, min(len(sv) - 1, math.ceil(p / 100.0 * len(sv)) - 1))
        out[p] = sv[k]
    return out


def summarize(results: list[SessionResult], duration_s: float) -> dict[str, Any]:
    by_endpoint: dict[str, list[CallResult]] = {}
    for r in results:
        for c in r.calls:
            by_endpoint.setdefault(c.endpoint, []).append(c)

    summary: dict[str, Any] = {
        "sessions_started": len(results),
        "sessions_completed": sum(1 for r in results if r.completed),
        "sessions_errored": sum(1 for r in results if r.error),
        "wall_time_s": round(duration_s, 3),
        "throughput_sessions_per_s": round(len(results) / duration_s, 3) if duration_s > 0 else 0,
        "endpoints": {},
    }
    for ep, calls in sorted(by_endpoint.items()):
        durations = [c.duration_s for c in calls]
        errors = sum(1 for c in calls if not c.ok)
        ps = percentiles(durations, (50, 95, 99))
        summary["endpoints"][ep] = {
            "calls": len(calls),
            "errors": errors,
            "error_rate": round(errors / len(calls), 4) if calls else 0.0,
            "p50_ms": round(ps[50] * 1000, 1),
            "p95_ms": round(ps[95] * 1000, 1),
            "p99_ms": round(ps[99] * 1000, 1),
            "mean_ms": round(statistics.fmean(durations) * 1000, 1) if durations else 0.0,
        }
    return summary


def render_table(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"sessions: {summary['sessions_started']} started, "
        f"{summary['sessions_completed']} completed, "
        f"{summary['sessions_errored']} errored"
    )
    lines.append(
        f"wall time: {summary['wall_time_s']}s, "
        f"throughput: {summary['throughput_sessions_per_s']} sessions/s"
    )
    if "cost_usd" in summary:
        lines.append(f"estimated cost: ${summary['cost_usd']:.4f}")
    lines.append("-" * 78)
    lines.append(f"{'endpoint':<40} {'calls':>6} {'err%':>6} {'p50':>7} {'p95':>7} {'p99':>7}")
    for ep, s in summary["endpoints"].items():
        lines.append(
            f"{ep:<40} {s['calls']:>6} {s['error_rate']*100:>5.1f}% "
            f"{s['p50_ms']:>6.0f}ms {s['p95_ms']:>6.0f}ms {s['p99_ms']:>6.0f}ms"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


async def run_loadtest(
    *,
    base_url: str,
    sessions: int,
    rate: float,
    output: str | None,
) -> int:
    print(f"warm-up: GET {base_url}/api/chat2hamnosys/health")
    async with httpx.AsyncClient(timeout=10.0) as warm:
        try:
            r = await warm.get(f"{base_url}/api/chat2hamnosys/health")
            r.raise_for_status()
            print(f"  -> {r.status_code} {r.json()}")
        except httpx.HTTPError as exc:
            print(f"ERROR: backend not reachable at {base_url}: {exc}", file=sys.stderr)
            return 2

    cost_before = await fetch_metric(base_url, "chat2hamnosys_llm_cost_usd_total")

    semaphore_interval = (1.0 / rate) if rate > 0 else 0.0
    results: list[SessionResult] = []
    started = time.perf_counter()

    async with httpx.AsyncClient(http2=False) as client:
        async def runner(idx: int) -> None:
            if semaphore_interval > 0:
                await asyncio.sleep(idx * semaphore_interval)
            results.append(await run_one_session(client=client, base_url=base_url, idx=idx))

        await asyncio.gather(*(runner(i) for i in range(sessions)))

    duration = time.perf_counter() - started
    cost_after = await fetch_metric(base_url, "chat2hamnosys_llm_cost_usd_total")

    summary = summarize(results, duration)
    if cost_before is not None and cost_after is not None:
        summary["cost_usd"] = round(max(0.0, cost_after - cost_before), 4)
    summary["base_url"] = base_url
    summary["concurrency_target"] = sessions
    summary["ramp_rate_per_s"] = rate

    print(render_table(summary))

    if output:
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote machine-readable summary -> {output}")

    # Exit non-zero if any endpoint had >0.5 % errors — handy for CI gates.
    bad = [
        ep for ep, s in summary["endpoints"].items()
        if s["error_rate"] > 0.005
    ]
    if bad:
        print(f"FAIL: error rate above 0.5% on: {', '.join(bad)}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000",
                   help="Backend base URL (default: %(default)s)")
    p.add_argument("--sessions", type=int, default=10,
                   help="Total virtual users to launch (default: %(default)s)")
    p.add_argument("--rate", type=float, default=0,
                   help="Ramp-up rate (sessions per second). 0 = launch all at once.")
    p.add_argument("--output", default=None,
                   help="Optional JSON output path for the run summary.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(run_loadtest(
        base_url=ns.base_url,
        sessions=ns.sessions,
        rate=ns.rate,
        output=ns.output,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
