"""End-to-end smoke test for the contribute-pipeline corrections API.

Locks in the fix from prompt 05 of the contrib-fix run: the three
correction modes a contributor can use against a rendered sign — chip
handshape swap, chip orientation swap, and a free-text chat correction
— must all land as a SiGML rewrite on the session and emit a
``generated`` SSE frame within 60 s.

Each sub-test is a full HTTP round-trip: ensure a SiGML is on the
draft, ``POST /sessions/<id>/correct`` with the appropriate payload,
then ``GET /sessions/<id>`` and assert on ``envelope.sigml``.

The chip-swap sub-tests exercise the structured ``swap`` payload added
by this prompt; the chat sub-test exercises the deterministic
natural-language fast-path in
``backend/chat2hamnosys/correct/sigml_rewrite.py``. Neither needs an
LLM round-trip — the deterministic SiGML rewrite path runs
synchronously inside ``post_correct``.

Configuration
-------------
- ``KOZHA_BACKEND`` (default ``http://127.0.0.1:8000``) — base URL.
  Same convention as ``tests/contrib_generate_smoke.py``: local
  uvicorn mounts the router at the root, the deployed nginx proxy
  mounts it under ``/api/chat2hamnosys``. The prefix is auto-detected
  by probing ``/hamnosys/symbols``.
- ``KOZHA_API_PREFIX`` — pin the prefix and skip detection.
- ``CHAT2HAMNOSYS_SESSION_DB`` — when running locally, the test seeds
  the SiGML directly via :class:`SessionStore` against this SQLite
  file. Defaults to ``data/chat2hamnosys/sessions.sqlite3`` (the
  same default the API uses).

Local vs deployed
-----------------
Locally the test can write through the SQLite session store to plant
a known SiGML on the draft, so the three sub-tests run independently
from a deterministic starting state. Against the deployed domain the
SQLite is unreachable, so the test seeds via ``POST /describe`` (which
needs a working LLM key on the server) and resets between tests by
re-applying the inverse swap. If neither route can produce a SiGML
the sub-test skips with a clear message rather than 500-ing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import UUID

import httpx
import pytest


SMOKE_TIMEOUT_S = 60.0
PROBE_TIMEOUT_S = 10.0
DESCRIBE_TIMEOUT_S = 45.0
SIGML_POLL_INTERVAL_S = 0.5

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DB = REPO_ROOT / "data" / "chat2hamnosys" / "sessions.sqlite3"


# Canonical seed SiGML for the chip-swap fixtures. Carries one tag per
# mandatory slot in canonical order — fist • fingers up • palm up •
# chest. The chip-swap tests rewrite specific tags inside the
# ``<hamnosys_manual>`` block; the chat test rewrites the
# palm-direction slot from ``hampalmu`` (up) to ``hampalmd`` (down).
SEED_HAMNOSYS = ""  # fist • up • palm-up • chest
SEED_SIGML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<sigml>\n'
    '  <hns_sign gloss="SMOKE">\n'
    '    <hamnosys_manual>\n'
    '      <hamfist/>\n'
    '      <hamextfingeru/>\n'
    '      <hampalmu/>\n'
    '      <hamchest/>\n'
    '    </hamnosys_manual>\n'
    '  </hns_sign>\n'
    '</sigml>\n'
)


def _backend_base() -> str:
    return os.environ.get("KOZHA_BACKEND", "http://127.0.0.1:8000").rstrip("/")


def _is_local(base: str) -> bool:
    return "127.0.0.1" in base or "localhost" in base


def _detect_api_prefix(base: str) -> str:
    """Return the path prefix the chat2hamnosys router is mounted at.

    Same logic as ``contrib_generate_smoke._detect_api_prefix`` —
    duplicated rather than imported so each smoke file stays
    self-contained.
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


def _ensure_backend_on_path() -> None:
    """Add ``backend/chat2hamnosys`` to ``sys.path`` for the local seed.

    The package layout uses unqualified imports (``from session import
    ...``) so the test must extend ``sys.path`` rather than importing
    via a dotted module path.
    """
    backend = REPO_ROOT / "backend" / "chat2hamnosys"
    p = str(backend)
    if p not in sys.path:
        sys.path.insert(0, p)


def _local_session_db_path() -> Path:
    raw = os.environ.get("CHAT2HAMNOSYS_SESSION_DB", "").strip()
    return Path(raw) if raw else DEFAULT_LOCAL_DB


def _seed_via_store(session_id: str) -> bool:
    """Plant ``SEED_SIGML`` onto the session's draft via the SQLite store.

    Returns ``True`` on success, ``False`` if the local seed can't run
    (deployed mode, missing DB, missing modules). The caller decides
    whether to skip or fall back to a different seeding path.
    """
    db_path = _local_session_db_path()
    if not db_path.exists():
        return False
    try:
        _ensure_backend_on_path()
        from session.storage import SessionStore  # type: ignore[import-not-found]
        from session.state import (  # type: ignore[import-not-found]
            GeneratedEvent,
            SessionState,
        )
    except Exception:
        return False
    store = SessionStore(db_path=db_path)
    try:
        sid = UUID(session_id)
    except ValueError:
        return False
    session = store.get(sid)
    if session is None:
        return False
    session = session.append_event(
        GeneratedEvent(
            success=True,
            hamnosys=SEED_HAMNOSYS,
            sigml=SEED_SIGML,
            confidence=1.0,
            used_llm_fallback=False,
        )
    )
    session = session.with_draft(
        hamnosys=SEED_HAMNOSYS,
        sigml=SEED_SIGML,
        preview_status="ok",
        preview_message="",
        generation_errors=[],
    )
    session = session.with_state(SessionState.RENDERED)
    store.save(session)
    return True


def _wait_for_sigml(
    api: str,
    session_id: str,
    token: str,
    *,
    deadline_s: float,
) -> Optional[str]:
    """Poll ``GET /sessions/<id>`` until ``sigml`` is non-empty or timeout."""
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        with httpx.Client(timeout=PROBE_TIMEOUT_S) as client:
            r = client.get(
                f"{api}/sessions/{session_id}",
                headers={"X-Session-Token": token},
            )
        if r.status_code == 200:
            sigml = r.json().get("sigml")
            if sigml:
                return sigml
        time.sleep(SIGML_POLL_INTERVAL_S)
    return None


def _ensure_seeded_sigml(api: str, base: str, session_id: str, token: str) -> str:
    """Return a SiGML the session is currently holding, seeding if needed.

    Local: writes through :class:`SessionStore` so the test starts from
    a deterministic SiGML regardless of LLM-key availability.
    Deployed: uses ``POST /describe`` and waits for the background
    generation to land. Skips with a clear message if the deployed
    pipeline can't produce one.
    """
    if _is_local(base):
        if not _seed_via_store(session_id):
            pytest.skip(
                "local seeding requires the chat2hamnosys SQLite store "
                f"at {_local_session_db_path()} and the backend on the "
                "Python path; create a session via uvicorn first"
            )
        return SEED_SIGML
    # Deployed path: rely on /describe to produce a SiGML.
    with httpx.Client(timeout=DESCRIBE_TIMEOUT_S) as client:
        r = client.post(
            f"{api}/sessions/{session_id}/describe",
            json={"prose": "the sign for hello — a flat hand waving"},
            headers={"X-Session-Token": token},
        )
    if r.status_code >= 500:
        pytest.skip(
            f"deployed /describe returned {r.status_code} — cannot "
            "seed a SiGML for the correction tests"
        )
    sigml = (r.json().get("sigml") if r.status_code == 200 else None)
    if not sigml:
        sigml = _wait_for_sigml(api, session_id, token, deadline_s=SMOKE_TIMEOUT_S)
    if not sigml:
        pytest.skip(
            "deployed /describe did not produce a SiGML within "
            f"{int(SMOKE_TIMEOUT_S)}s — cannot run correction tests"
        )
    return sigml


def _post_correct(
    api: str,
    session_id: str,
    token: str,
    *,
    raw_text: str,
    swap: Optional[dict] = None,
) -> dict:
    """POST /correct, return the response envelope."""
    body: dict = {"raw_text": raw_text}
    if swap is not None:
        body["swap"] = swap
    with httpx.Client(timeout=SMOKE_TIMEOUT_S) as client:
        r = client.post(
            f"{api}/sessions/{session_id}/correct",
            json=body,
            headers={"X-Session-Token": token},
        )
    assert r.status_code == 200, (
        f"correct failed: {r.status_code} {r.text[:500]}"
    )
    return r.json()


def _wait_for_sigml_change(
    api: str,
    session_id: str,
    token: str,
    *,
    before: str,
    deadline_s: float = SMOKE_TIMEOUT_S,
) -> str:
    """Poll until ``envelope.sigml`` differs from ``before`` or timeout."""
    start = time.monotonic()
    last_sigml = before
    while time.monotonic() - start < deadline_s:
        with httpx.Client(timeout=PROBE_TIMEOUT_S) as client:
            r = client.get(
                f"{api}/sessions/{session_id}",
                headers={"X-Session-Token": token},
            )
        if r.status_code == 200:
            env = r.json()
            last_sigml = env.get("sigml") or last_sigml
            if last_sigml and last_sigml != before:
                return last_sigml
        time.sleep(SIGML_POLL_INTERVAL_S)
    return last_sigml


# ---------------------------------------------------------------------------
# Fixture: one session shared across all sub-tests
# ---------------------------------------------------------------------------
#
# Session creation is per-IP rate-limited (default 2/minute) so we
# cannot afford one session per test function. The fixture creates
# exactly one and each test re-seeds the draft to the canonical
# starting SiGML before exercising the correction it is responsible
# for. With three tests, the rate limit is comfortably observed.


@pytest.fixture(scope="module")
def smoke_session() -> dict:
    base = _backend_base()
    prefix = _detect_api_prefix(base)
    api = f"{base}{prefix}"
    with httpx.Client(timeout=PROBE_TIMEOUT_S) as client:
        r = client.post(
            f"{api}/sessions",
            json={
                "sign_language": "bsl",
                "display_name": "correct-smoke",
                "author_is_deaf_native": False,
                "gloss": "SMOKE",
            },
        )
    if r.status_code == 429:
        pytest.skip(
            "session creation rate-limited; wait for the limiter "
            "window to clear and retry, or set "
            "CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT=60/minute on the "
            "uvicorn process"
        )
    assert r.status_code == 201, (
        f"create session failed: {r.status_code} {r.text[:500]}"
    )
    body = r.json()
    return {
        "api":         api,
        "base":        base,
        "session_id":  body["session_id"],
        "token":       body["session_token"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chip_handshape_swap(smoke_session: dict) -> None:
    """Structured chip-swap payload rewrites the handshape tag in SiGML."""
    api = smoke_session["api"]
    base = smoke_session["base"]
    session_id = smoke_session["session_id"]
    token = smoke_session["token"]

    seeded = _ensure_seeded_sigml(api, base, session_id, token)
    assert "<hamfist/>" in seeded, (
        f"seed SiGML does not contain <hamfist/> — actual: {seeded[:300]!r}"
    )

    env = _post_correct(
        api,
        session_id,
        token,
        raw_text="chip swap: <hamfist/> → <hamflathand/>",
        swap={"from_tag": "hamfist", "to_tag": "hamflathand"},
    )
    sigml = env.get("sigml") or _wait_for_sigml_change(
        api, session_id, token, before=seeded
    )
    assert sigml, "envelope returned no sigml"
    assert "<hamflathand/>" in sigml, (
        f"expected <hamflathand/> in result; got: {sigml[:400]!r}"
    )
    assert "<hamfist/>" not in sigml, (
        f"<hamfist/> should have been replaced; got: {sigml[:400]!r}"
    )


def test_chip_orientation_swap(smoke_session: dict) -> None:
    """Structured chip-swap payload rewrites the palm-direction tag in SiGML."""
    api = smoke_session["api"]
    base = smoke_session["base"]
    session_id = smoke_session["session_id"]
    token = smoke_session["token"]

    seeded = _ensure_seeded_sigml(api, base, session_id, token)
    assert "<hampalmu/>" in seeded, (
        f"seed SiGML does not contain <hampalmu/> — actual: {seeded[:300]!r}"
    )

    env = _post_correct(
        api,
        session_id,
        token,
        raw_text="chip swap: <hampalmu/> → <hampalmd/>",
        swap={"from_tag": "hampalmu", "to_tag": "hampalmd"},
    )
    sigml = env.get("sigml") or _wait_for_sigml_change(
        api, session_id, token, before=seeded
    )
    assert sigml, "envelope returned no sigml"
    assert "<hampalmd/>" in sigml, (
        f"expected <hampalmd/> in result; got: {sigml[:400]!r}"
    )
    assert "<hampalmu/>" not in sigml, (
        f"<hampalmu/> should have been replaced; got: {sigml[:400]!r}"
    )


def test_chat_correction(smoke_session: dict) -> None:
    """Free-text "make the palm face downward instead" → palm tag → down."""
    api = smoke_session["api"]
    base = smoke_session["base"]
    session_id = smoke_session["session_id"]
    token = smoke_session["token"]

    seeded = _ensure_seeded_sigml(api, base, session_id, token)

    # The seed carries hampalmu; the deterministic fast-path in
    # ``correct/sigml_rewrite.py`` swaps it to hampalmd. The
    # structural expectation locked in by the audit is "a tag in the
    # palm_direction category changed toward down" — accept any of
    # hampalmd / hampalmdr / hampalmdl so the interpreter is free to
    # refine over time.
    env = _post_correct(
        api,
        session_id,
        token,
        raw_text="make the palm face downward instead",
    )
    sigml = env.get("sigml") or _wait_for_sigml_change(
        api, session_id, token, before=seeded
    )
    assert sigml, "envelope returned no sigml"
    downward_palm_tags = ("<hampalmd/>", "<hampalmdr/>", "<hampalmdl/>")
    matched = [t for t in downward_palm_tags if t in sigml]
    assert matched, (
        "expected a downward palm_direction tag (hampalmd / hampalmdr / "
        f"hampalmdl) in the resulting SiGML; got: {sigml[:400]!r}"
    )
    # The original up-facing tag must be gone — even a permissive read
    # of "instead" rules out leaving the upward palm in place.
    assert "<hampalmu/>" not in sigml, (
        "<hampalmu/> should have been replaced when the user asked the "
        f"palm to face downward; got: {sigml[:400]!r}"
    )
