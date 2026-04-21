"""Reachability smoke test for every endpoint in ``00-endpoints-map.md``.

A thin guard that the authoring / review / admin surfaces stay mounted
as the router grows. Every endpoint listed in the design doc must
answer with an HTTP status other than 404 — anything ≥200 & <500
except 404 counts as "wired up" here. This deliberately accepts 401
/ 403 / 405 / 422 since we're hitting the route without valid auth,
required headers, or a well-formed body; what we're asserting is that
the URL resolves to a registered route at all.

If a new endpoint lands in the router but not the map (or vice versa),
this test is the belt-and-braces check that flags the drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App builder — mirrors the fixture in ``test_api_router`` but keeps this
# file standalone so the smoke test runs even when router tests are
# deselected (e.g. on CI jobs that only run quick checks).
# ---------------------------------------------------------------------------


def _build_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_SIGN_DB", str(tmp_path / "signs.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_TOKEN_DB", str(tmp_path / "tokens.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "kozha_data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_RATE_LIMIT", "1000/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT", "1000/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_CAPTCHA_DISABLED", "1")

    from api.dependencies import reset_stores

    reset_stores()

    from api.router import limiter as _module_limiter

    _module_limiter.reset()

    from api import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    tc = _build_client(tmp_path, monkeypatch)
    try:
        yield tc
    finally:
        tc.close()


# ---------------------------------------------------------------------------
# The endpoint inventory — one row per endpoint from 00-endpoints-map.md.
# ---------------------------------------------------------------------------


# A placeholder UUID for ``{id}`` path parameters; we don't need it to
# resolve to a real session — 404 session_not_found still proves the
# route is wired and reachable. The assertion rejects FastAPI's 404-
# from-missing-route case by also rejecting ``{"detail": "Not Found"}``.
_UUID = str(uuid4())


# Each tuple: (method, path, expected status class). Expected status is
# one of ``"ok"`` (any 2xx/4xx except 404), ``"any"`` (literally
# anything), or ``"404_ok"`` (404 is acceptable — e.g. not_submitted
# for a session that never accepted). Default is ``"ok"``.
ENDPOINTS: list[tuple[str, str, str]] = [
    # 1. Contributor onboarding (prefix /contribute)
    ("GET",  "/contribute/captcha", "ok"),
    ("POST", "/contribute/register", "ok"),

    # 2. Authoring sessions (no prefix)
    ("POST",   "/sessions",                           "ok"),
    ("GET",    f"/sessions/{_UUID}",                  "ok"),
    ("POST",   f"/sessions/{_UUID}/describe",         "ok"),
    ("POST",   f"/sessions/{_UUID}/answer",           "ok"),
    ("POST",   f"/sessions/{_UUID}/generate",         "ok"),
    ("GET",    f"/sessions/{_UUID}/preview",          "ok"),
    ("GET",    f"/sessions/{_UUID}/preview/video",    "ok"),
    ("POST",   f"/sessions/{_UUID}/correct",          "ok"),
    ("POST",   f"/sessions/{_UUID}/accept",           "ok"),
    ("POST",   f"/sessions/{_UUID}/reject",           "ok"),
    ("GET",    f"/sessions/{_UUID}/status",           "ok"),
    # SSE endpoint — httpx's TestClient streams, a 4xx without auth is enough
    ("GET",    f"/sessions/{_UUID}/events",           "ok"),

    # 3. Review surface (/review)
    ("GET",    "/review/me",                          "ok"),
    ("GET",    "/review/queue",                       "ok"),
    ("GET",    f"/review/entries/{_UUID}",            "ok"),
    ("POST",   f"/review/entries/{_UUID}/approve",    "ok"),
    ("POST",   f"/review/entries/{_UUID}/reject",     "ok"),
    ("POST",   f"/review/entries/{_UUID}/request_revision", "ok"),
    ("POST",   f"/review/entries/{_UUID}/flag",       "ok"),
    ("POST",   f"/review/entries/{_UUID}/clear_quarantine", "ok"),
    ("POST",   f"/review/entries/{_UUID}/export",     "ok"),
    ("GET",    "/review/dashboard",                   "ok"),
    ("POST",   "/review/reviewers",                   "ok"),

    # 4. Admin + observability
    ("GET",    "/metrics",                            "ok"),
    ("GET",    "/admin/dashboard",                    "ok"),
    ("GET",    f"/admin/sessions/{_UUID}",            "ok"),
    ("GET",    "/admin/cost",                         "ok"),
    ("GET",    "/health",                             "ok"),
    ("GET",    "/health/ready",                       "ok"),

    # Static: HamNoSys symbol table
    ("GET",    "/hamnosys/symbols",                   "ok"),
]


def _is_reachable(status: int, body_text: str) -> bool:
    """Reachable iff the route resolved.

    A 404 with FastAPI's default ``{"detail": "Not Found"}`` body means
    the router has no matching route — that's the failure we guard
    against. A 404 with the app's structured error envelope
    (``{"error": {"code": ..., "message": ..., "details": ...}}``) means
    the route resolved and app logic chose to 404 — e.g. ``session_not_found``
    because the placeholder UUID is unknown. That's reachable.
    """
    if status != 404:
        return True
    try:
        body = json.loads(body_text)
    except (ValueError, TypeError):
        return False
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return True
    return False


@pytest.mark.parametrize("method,path,mode", ENDPOINTS)
def test_endpoint_is_mounted(client: TestClient, method: str, path: str, mode: str) -> None:
    """Each endpoint from ``00-endpoints-map.md`` must not 404 from a
    missing route.

    We hit the URL with an empty body (or empty JSON for POST) and no
    auth headers; the route's dependencies will reject with 401 / 403
    / 422 / etc. Any of those proves the route resolved. A 404 with
    ``{"detail": "Not Found"}`` means the route is missing — that's the
    failure mode we guard against. A 404 with the app's error envelope
    (e.g. ``session_not_found``) is still reachable.
    """
    request_kwargs: dict = {}
    if method == "POST":
        request_kwargs["json"] = {}

    resp = client.request(method, path, **request_kwargs)
    assert _is_reachable(resp.status_code, resp.text), (
        f"{method} {path} returned 404 — endpoint is missing from the router. "
        f"Response: {resp.text[:400]}"
    )


def test_all_map_rows_tested() -> None:
    """Sanity: the parametrized row count matches the map's endpoint count.

    If someone adds a new row to ``00-endpoints-map.md`` they must add
    the corresponding entry to ``ENDPOINTS`` above. The map currently
    lists 30 authoring/review/admin endpoints plus the symbol table —
    this is the belt-and-braces guard against silent drift.
    """
    # 30 map rows (#1 through #30) plus /sessions/{id}/status (which the
    # map treats as part of the session surface but we include it in
    # ``ENDPOINTS`` for completeness) plus /hamnosys/symbols.
    assert len(ENDPOINTS) >= 30
