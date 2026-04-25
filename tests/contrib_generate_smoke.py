"""End-to-end smoke test for the contribute pipeline's SSE chain.

Locks in the fix from prompt 04 of the contrib-fix run: a
``POST /sessions/<id>/describe`` must reliably produce a ``generation``
event on the SSE stream within 30 seconds, regardless of whether the
LLM-backed parser succeeds (real ``OPENAI_API_KEY`` set) or fails (no
key, no BYO key — the deterministic-stub / failed-parse path).

Both terminal frames count as a pass:
  - ``event: generated`` with ``success: true``  (LLM produced HamNoSys)
  - ``event: generated`` with ``success: false`` (parser failed gracefully)

What we care about is that the SSE channel delivered a ``generated``
frame at all — the audit at ``docs/contrib-fix/01-audit.md`` § 3
documents the prior failure mode where the channel emitted only
``: keep-alive`` comments forever and the session row stayed at
``state=awaiting_description`` with empty history.

Driven by ``httpx`` + ``httpx_sse``. Configurable via:

  - ``KOZHA_BACKEND`` (default ``http://127.0.0.1:8000``) — base URL.
    Local default points at a uvicorn started from
    ``backend/chat2hamnosys`` with no api_prefix, so routes are at
    the root. The deployed domain proxies the same router under
    ``/api/chat2hamnosys`` (see ``deploy/nginx/kozha.conf``); the
    prefix is auto-detected by probing the public ``hamnosys/symbols``
    endpoint.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import httpx
import pytest
from httpx_sse import connect_sse


SMOKE_TIMEOUT_S = 30.0
PROBE_TIMEOUT_S = 10.0


def _backend_base() -> str:
    return os.environ.get("KOZHA_BACKEND", "http://127.0.0.1:8000").rstrip("/")


def _detect_api_prefix(base: str) -> str:
    """Return the path prefix the chat2hamnosys router is mounted at.

    Local uvicorn (``uvicorn backend.chat2hamnosys.api.app:app``) mounts
    everything at the root. The deployed nginx proxy mounts the same
    sub-app under ``/api/chat2hamnosys``. Probe the read-only
    ``/hamnosys/symbols`` endpoint to figure out which one we're talking
    to before committing to a request prefix.
    """
    override = os.environ.get("KOZHA_API_PREFIX")
    if override is not None:
        return override.rstrip("/")
    candidates = ("/api/chat2hamnosys", "")
    for prefix in candidates:
        try:
            r = httpx.get(
                f"{base}{prefix}/hamnosys/symbols",
                timeout=PROBE_TIMEOUT_S,
                headers={"If-None-Match": "smoke-probe"},
            )
        except httpx.HTTPError:
            continue
        if r.status_code in (200, 304):
            return prefix
    raise RuntimeError(
        f"could not locate /hamnosys/symbols under {base!r}; tried {candidates}"
    )


def test_describe_emits_generation_event_within_30s() -> None:
    base = _backend_base()
    prefix = _detect_api_prefix(base)
    api = f"{base}{prefix}"

    sse_lines: list[str] = []

    with httpx.Client(timeout=PROBE_TIMEOUT_S, follow_redirects=False) as client:
        # 1. Create a session.
        r = client.post(
            f"{api}/sessions",
            json={
                "sign_language": "bsl",
                "display_name": "smoke-test",
                "author_is_deaf_native": False,
                "gloss": "HELLO",
            },
        )
        assert r.status_code == 201, (
            f"create session failed: {r.status_code} {r.text[:500]}"
        )
        body = r.json()
        session_id = body["session_id"]
        token = body["session_token"]

        # 2. Submit prose. Even on a parser failure (no LLM key) the
        #    backend now records a DescribedEvent + failed
        #    GeneratedEvent so the SSE channel has frames to deliver.
        describe = client.post(
            f"{api}/sessions/{session_id}/describe",
            json={"prose": "the sign for hello"},
            headers={"X-Session-Token": token},
            timeout=SMOKE_TIMEOUT_S,
        )
        # Accept any status code: the success path returns 200, the
        # parser-fail path now also returns 200 thanks to the audit
        # fix. A 5xx here would mean we regressed — note it but still
        # let the SSE assertion below speak the final word.
        if describe.status_code >= 500:
            print(
                f"describe returned {describe.status_code}: "
                f"{describe.text[:500]}",
                file=sys.stderr,
            )

        # 3. Open SSE and wait for a "generat*" event.
        deadline = time.monotonic() + SMOKE_TIMEOUT_S
        sse_url = f"{api}/sessions/{session_id}/events?token={token}"
        seen_event_type: Optional[str] = None
        seen_payload: Optional[dict] = None
        try:
            sse_timeout = max(SMOKE_TIMEOUT_S + 5.0, 35.0)
            with httpx.Client(timeout=sse_timeout) as sse_client, connect_sse(
                sse_client, "GET", sse_url
            ) as event_source:
                for sse in event_source.iter_sse():
                    if time.monotonic() >= deadline:
                        break
                    line = (
                        f"event={sse.event!r} id={sse.id!r} data={sse.data[:200]!r}"
                    )
                    sse_lines.append(line)
                    if "generat" in (sse.event or "").lower():
                        seen_event_type = sse.event
                        try:
                            seen_payload = json.loads(sse.data) if sse.data else None
                        except json.JSONDecodeError:
                            seen_payload = None
                        break
        except httpx.HTTPError as exc:
            sse_lines.append(f"<httpx error: {type(exc).__name__}: {exc}>")

    if seen_event_type is None:
        print("--- captured SSE frames ---", file=sys.stderr)
        for line in sse_lines:
            print(line, file=sys.stderr)
        print("--- end captured SSE frames ---", file=sys.stderr)
        pytest.fail(
            f"no event with type containing 'generat' arrived within "
            f"{SMOKE_TIMEOUT_S}s on {sse_url}"
        )

    # Either success=True (LLM produced HamNoSys) or success=False
    # (parser/generator gave up gracefully) is acceptable; the
    # assertion is "the SSE chain delivered the event."
    assert seen_event_type, "expected a non-empty SSE event type"
    if isinstance(seen_payload, dict) and "success" in seen_payload:
        assert isinstance(seen_payload["success"], bool), (
            f"unexpected payload shape: {seen_payload!r}"
        )
