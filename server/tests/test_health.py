"""Regression tests for the health endpoints (polish-13 §7).

Separate routes by design:

* ``/health`` returns 200 whenever the process is alive. The load
  balancer uses this as a liveness probe — a flake here means the LB
  should kill the container, not that the container should deregister
  traffic.
* ``/health/ready`` returns 200 only when the server can actually
  serve traffic (data readable, meta parses). This is the readiness
  probe — the LB should drop traffic on 503 but leave the container
  alive during warm-up.

``/api/health`` is kept as a backwards-compatible alias for older
clients that pinned the old path; the contract is the same.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

from server import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_live_returns_200(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


def test_health_live_carries_request_id(client: TestClient) -> None:
    r = client.get("/health")
    assert r.headers.get("x-request-id"), "health responses must carry X-Request-ID"


def test_health_live_does_no_external_checks(client: TestClient) -> None:
    """The liveness probe's body must not reflect any external state —
    otherwise a transient failure would take the pod down."""
    r = client.get("/health")
    body = r.json()
    assert set(body.keys()) == {"status"}


def test_health_ready_returns_200_in_this_env(client: TestClient) -> None:
    """The data/ + meta files are present in the test repo, so
    readiness should pass."""
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    checks = body.get("checks") or {}
    for required in ("data_dir", "meta_files"):
        assert checks.get(required) is True, (
            f"readiness check {required} did not pass — detail={body.get('detail')}"
        )


def test_health_ready_returns_503_when_data_dir_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping the data dir to a bogus path must drop readiness to
    503 without crashing the endpoint."""
    import kozha_obs as obs

    monkeypatch.setattr(obs, "DATA_DIR", Path("/nonexistent/path/probably-not-a-real-dir"))
    r = client.get("/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body.get("status") == "not_ready"
    checks = body.get("checks") or {}
    assert checks.get("data_dir") is False
    assert "data_dir" in (body.get("detail") or {})


def test_api_health_still_works(client: TestClient) -> None:
    """Older clients (and the CI smoke test) pin the path; don't break
    that contract."""
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_health_ready_reports_all_four_checks(client: TestClient) -> None:
    r = client.get("/health/ready")
    body = r.json()
    checks = body.get("checks") or {}
    assert {"data_dir", "meta_files", "spacy_en", "argostranslate"}.issubset(checks.keys()), (
        f"readiness is missing checks: {checks}"
    )


def test_5xx_body_carries_request_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trigger a real 5xx by making the translator raise, then confirm
    the body carries the request_id and does not leak internals.

    We monkey-patch ``plan_from_text`` — not add a test-only route —
    because the app mounts ``StaticFiles`` at ``/`` and routes added
    after module import never reach the dispatcher.
    """
    import server as srv

    def _explode(*args, **kwargs):
        raise RuntimeError("intentional test failure - should not leak")

    monkeypatch.setattr(srv, "plan_from_text", _explode)
    # TestClient re-raises server exceptions by default — flip that so
    # the exception flows through Starlette's ExceptionMiddleware to
    # our registered ``Exception`` handler, which is the production
    # path.
    safe_client = TestClient(app, raise_server_exceptions=False)
    r = safe_client.post(
        "/api/plan",
        json={"text": "boom", "language": "en", "sign_language": "bsl"},
    )
    assert r.status_code == 500, f"expected 500, got {r.status_code}: {r.text}"
    body = r.json()
    assert "request_id" in body, "5xx body must expose request_id"
    assert body.get("request_id"), "request_id should not be empty"
    assert r.headers.get("x-request-id") == body["request_id"], (
        "header and body request_id should match"
    )
    raw = json.dumps(body)
    assert "intentional test failure" not in raw, (
        "5xx body must not leak the raw exception message"
    )
    assert "RuntimeError" not in raw, "5xx body must not leak the exception type"
