"""Tests for the contributor on-ramp (captcha + registration).

The captcha is stateless — the signed challenge carries the SHA-256 of
the expected answer — so these tests don't need a session/token store.
They just round-trip the challenge through the two endpoints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _answer_for(challenge: str) -> str:
    """Extract the expected answer from the signed challenge.

    Only safe inside tests — real clients re-derive it from the
    human-readable ``question`` field.
    """
    import base64

    raw_b64 = challenge.split(".", 1)[0]
    padded = raw_b64 + "=" * (-len(raw_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
    ans_hash = payload["ans_sha256"]
    # Brute-force 2..18 since the captcha is single-digit addition.
    for candidate in range(2, 19):
        if hashlib.sha256(str(candidate).encode()).hexdigest() == ans_hash:
            return str(candidate)
    raise AssertionError("could not recover captcha answer for test")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_CAPTCHA_SECRET", "test-captcha-secret")
    monkeypatch.setenv("CHAT2HAMNOSYS_CONTRIBUTOR_SECRET", "test-contrib-secret")

    from api.dependencies import reset_stores

    reset_stores()

    from api import create_app

    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Captcha
# ---------------------------------------------------------------------------


def test_captcha_issues_a_signed_challenge(client: TestClient) -> None:
    resp = client.get("/contribute/captcha")
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"].startswith("What is ")
    assert "+" in body["question"]
    # challenge is <b64_json>.<b64_sig>
    assert body["challenge"].count(".") == 1
    assert body["expires_in"] > 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_happy_path(client: TestClient) -> None:
    captcha = client.get("/contribute/captcha").json()
    answer = _answer_for(captcha["challenge"])

    resp = client.post(
        "/contribute/register",
        json={
            "name": "Anna Novak",
            "contact": "anna@example.com",
            "captcha_challenge": captcha["challenge"],
            "captcha_answer": answer,
            "website": "",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["contributor_id"]
    assert body["contributor_token"].count(".") == 1
    assert body["expires_at"] > 0


def test_register_rejects_wrong_captcha_answer(client: TestClient) -> None:
    captcha = client.get("/contribute/captcha").json()
    resp = client.post(
        "/contribute/register",
        json={
            "name": "Wrong Answer",
            "contact": "x@y.com",
            "captcha_challenge": captcha["challenge"],
            "captcha_answer": "999",  # definitely not the sum of two digits
            "website": "",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "captcha_failed"


def test_register_rejects_honeypot_fill(client: TestClient) -> None:
    captcha = client.get("/contribute/captcha").json()
    answer = _answer_for(captcha["challenge"])
    resp = client.post(
        "/contribute/register",
        json={
            "name": "Spambot 3000",
            "contact": "bot@example.com",
            "captcha_challenge": captcha["challenge"],
            "captcha_answer": answer,
            "website": "http://spam.example",  # honeypot tripped
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "captcha_failed"


def test_register_rejects_invalid_contact(client: TestClient) -> None:
    captcha = client.get("/contribute/captcha").json()
    answer = _answer_for(captcha["challenge"])
    resp = client.post(
        "/contribute/register",
        json={
            "name": "Anna",
            "contact": "not-an-email-or-phone",
            "captcha_challenge": captcha["challenge"],
            "captcha_answer": answer,
            "website": "",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_contributor_input"


# ---------------------------------------------------------------------------
# Session gate
# ---------------------------------------------------------------------------


def test_session_gate_blocks_without_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR", "1")
    # Reset the enabled flag read (it's env-driven, so no singleton reset).
    resp = client.post("/sessions", json={})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "contributor_required"


def test_session_gate_accepts_valid_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR", "1")
    captcha = client.get("/contribute/captcha").json()
    answer = _answer_for(captcha["challenge"])
    reg = client.post(
        "/contribute/register",
        json={
            "name": "Anna Novak",
            "contact": "anna@example.com",
            "captcha_challenge": captcha["challenge"],
            "captcha_answer": answer,
            "website": "",
        },
    ).json()

    resp = client.post(
        "/sessions",
        json={"sign_language": "bsl"},
        headers={"X-Contributor-Token": reg["contributor_token"]},
    )
    assert resp.status_code == 201, resp.text


def test_session_gate_off_by_default(client: TestClient) -> None:
    # No CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR set → anonymous sessions still work.
    resp = client.post("/sessions", json={})
    assert resp.status_code == 201, resp.text


# ---------------------------------------------------------------------------
# Captcha disable — pre-launch scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture
def captcha_disabled_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """Fresh app with the captcha bypass flag on."""
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_CAPTCHA_SECRET", "test-captcha-secret")
    monkeypatch.setenv("CHAT2HAMNOSYS_CONTRIBUTOR_SECRET", "test-contrib-secret")
    monkeypatch.setenv("CHAT2HAMNOSYS_CAPTCHA_DISABLED", "1")

    from api.dependencies import reset_stores

    reset_stores()

    from api import create_app

    app = create_app()
    return TestClient(app)


def test_captcha_endpoint_advertises_disabled(
    captcha_disabled_client: TestClient,
) -> None:
    body = captcha_disabled_client.get("/contribute/captcha").json()
    assert body["disabled"] is True
    assert body["challenge"] == "disabled"
    assert body["expires_in"] == 0


def test_register_accepts_any_answer_when_disabled(
    captcha_disabled_client: TestClient,
) -> None:
    resp = captcha_disabled_client.post(
        "/contribute/register",
        json={
            "name": "Anna Novak",
            "contact": "anna@example.com",
            "captcha_challenge": "disabled",
            "captcha_answer": "",
            "website": "",
        },
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["contributor_token"].count(".") == 1


def test_honeypot_still_enforced_when_captcha_disabled(
    captcha_disabled_client: TestClient,
) -> None:
    # The honeypot is the only spam defence left in pre-launch mode —
    # it MUST keep rejecting bot fills even with the captcha bypass on.
    resp = captcha_disabled_client.post(
        "/contribute/register",
        json={
            "name": "Spambot",
            "contact": "bot@example.com",
            "captcha_challenge": "disabled",
            "captcha_answer": "",
            "website": "http://spam.example",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "captcha_failed"


# ---------------------------------------------------------------------------
# BYO OpenAI key — X-OpenAI-Api-Key plumbed through to LLMClient
# ---------------------------------------------------------------------------


def test_byo_openai_key_header_sets_contextvar(client: TestClient) -> None:
    """The middleware parks the header in a contextvar so
    :class:`LLMClient` picks it up in preference to the env var."""
    from fastapi import APIRouter

    from llm.client import _REQUEST_OPENAI_API_KEY

    captured: dict[str, str] = {}

    probe = APIRouter()

    @probe.get("/_byo_probe")
    def _byo_probe() -> dict:
        captured["key"] = _REQUEST_OPENAI_API_KEY.get("")
        return {"ok": True}

    client.app.include_router(probe)

    resp = client.get("/_byo_probe", headers={"X-OpenAI-Api-Key": "sk-from-browser"})
    assert resp.status_code == 200
    assert captured["key"] == "sk-from-browser"

    # No header → contextvar resets to empty on the next call.
    captured.clear()
    client.get("/_byo_probe")
    assert captured["key"] == ""


def test_byo_openai_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both the env var and the request contextvar are set, the
    contextvar wins — so users can supply a personal key even when a
    project key is configured."""
    from llm.client import (
        LLMClient,
        reset_request_openai_api_key,
        set_request_openai_api_key,
    )
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENAI_API_KEY", "sk-project")
    token = set_request_openai_api_key("sk-user")
    try:
        c = LLMClient(client=MagicMock())
        assert c._api_key == "sk-user"
    finally:
        reset_request_openai_api_key(token)

    # After reset, the env var wins again.
    c2 = LLMClient(client=MagicMock())
    assert c2._api_key == "sk-project"
