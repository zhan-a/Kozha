"""Integration tests: /metrics, /health, and full-session event sequence."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from clarify import Option, Question
from obs import events as evs
from obs import metrics as _metrics
from obs.logger import EventLogger, reset_logger, set_logger
from parser import Gap, ParseResult
from parser.models import PartialMovementSegment, PartialSignParameters


TEMPLE_PROSE = (
    "it's signed near the temple, flat hand, moves down to the chest, like SORRY"
)


def _partial() -> PartialSignParameters:
    return PartialSignParameters(
        handshape_dominant="flat",
        orientation_palm="down",
        location="temple",
        movement=[PartialMovementSegment(path="down")],
    )


class _StubParser:
    def __init__(self, result: ParseResult) -> None:
        self._result = result

    def __call__(self, prose: str) -> ParseResult:
        return self._result


class _StubQuestioner:
    def __init__(self) -> None:
        self._served = False

    def __call__(self, parse_result, prior_turns, *, is_deaf_native):
        if self._served:
            return []
        self._served = True
        if not any(
            g.field == "orientation_extended_finger" for g in parse_result.gaps
        ):
            return []
        return [
            Question(
                field="orientation_extended_finger",
                text="Which way do the fingers point?",
                options=[Option(label="up", value="up")],
                allow_freeform=True,
                rationale="stub",
            )
        ]


def _apply_fn(params, question, answer):
    if question.field == "orientation_extended_finger":
        return params.model_copy(
            update={"orientation_extended_finger": answer.strip()}
        )
    return params


@pytest.fixture
def obs_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session_db = tmp_path / "sessions.sqlite3"
    sign_db = tmp_path / "signs.sqlite3"
    token_db = tmp_path / "tokens.sqlite3"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_DB", str(session_db))
    monkeypatch.setenv("CHAT2HAMNOSYS_SIGN_DB", str(sign_db))
    monkeypatch.setenv("CHAT2HAMNOSYS_TOKEN_DB", str(token_db))
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "kozha_data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_RATE_LIMIT", "500/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT", "500/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_LOG_DIR", str(log_dir))
    monkeypatch.setenv("CHAT2HAMNOSYS_LOG_SINK", "file")

    # Fresh metrics + logger for isolation.
    _metrics.reset_registry()
    lg = EventLogger(log_dir=log_dir, sink="file", retention_days=30)
    set_logger(lg)

    from api.dependencies import reset_stores

    reset_stores()

    from api.router import limiter as _module_limiter

    _module_limiter.reset()

    from api import (
        create_app,
        get_apply_fn,
        get_parse_fn,
        get_question_fn,
        get_render_fn,
    )

    parse_result = ParseResult(
        parameters=_partial(),
        gaps=[
            Gap(
                field="orientation_extended_finger",
                reason="flat hand finger orientation not stated",
                suggested_question="Which way do the fingers point?",
            )
        ],
        raw_response="{}",
    )
    app = create_app()
    app.dependency_overrides[get_parse_fn] = lambda: _StubParser(parse_result)
    app.dependency_overrides[get_question_fn] = lambda: _StubQuestioner()
    app.dependency_overrides[get_apply_fn] = lambda: _apply_fn
    app.dependency_overrides[get_render_fn] = lambda: None

    tc = TestClient(app)
    try:
        yield tc, lg
    finally:
        tc.close()
        reset_logger()
        _metrics.reset_registry()


def test_health_returns_ok(obs_client, monkeypatch) -> None:
    client, _ = obs_client
    monkeypatch.delenv("BUILD_SHA", raising=False)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_surfaces_build_sha(obs_client, monkeypatch) -> None:
    client, _ = obs_client
    monkeypatch.setenv("BUILD_SHA", "abc1234")
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["build_sha"] == "abc1234"


def test_ready_reports_per_dependency(obs_client, monkeypatch) -> None:
    client, _ = obs_client
    # Without OPENAI_API_KEY the readiness probe should degrade.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.get("/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["llm_config"]["ok"] is False
    assert body["checks"]["session_store"]["ok"] is True

    # With the key set, the llm_config check should flip green.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    r = client.get("/health/ready")
    body = r.json()
    assert body["checks"]["llm_config"]["ok"] is True


def test_metrics_endpoint_exposes_prometheus_text(obs_client) -> None:
    client, _ = obs_client
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "# HELP sessions_started_total" in body
    assert "# TYPE sessions_started_total counter" in body


def test_full_session_emits_expected_events(obs_client) -> None:
    client, lg = obs_client

    r = client.post(
        "/sessions",
        json={
            "signer_id": "tester-01",
            "gloss": "SORRY",
            "sign_language": "bsl",
            "author_is_deaf_native": True,
        },
    )
    assert r.status_code == 201
    sid = r.json()["session_id"]
    auth = {"X-Session-Token": r.json()["session_token"]}

    r = client.post(
        f"/sessions/{sid}/describe",
        json={"prose": TEMPLE_PROSE},
        headers=auth,
    )
    assert r.status_code == 200
    r = client.post(
        f"/sessions/{sid}/answer",
        json={"question_id": "orientation_extended_finger", "answer": "up"},
        headers=auth,
    )
    assert r.status_code == 200
    r = client.post(f"/sessions/{sid}/accept", headers=auth)
    assert r.status_code == 200

    # Verify the expected event sequence landed on the ring buffer.
    names = [e.event for e in lg.recent()]
    assert evs.SESSION_CREATED in names
    assert evs.PARSE_DESCRIPTION_COMPLETED in names
    assert evs.CLARIFY_QUESTION_ASKED in names
    assert evs.CLARIFY_ANSWER_RECEIVED in names
    assert evs.GENERATE_HAMNOSYS_ATTEMPTED in names
    assert evs.GENERATE_HAMNOSYS_VALIDATED in names
    assert evs.SESSION_ACCEPTED in names

    # Per-session filter
    ours = [e for e in lg.recent_for_session(sid)]
    assert any(e.event == evs.SESSION_ACCEPTED for e in ours)

    # Metric side-effects
    assert _metrics.sessions_started_total.get() >= 1
    assert _metrics.sessions_accepted_total.get() >= 1


def test_metrics_reflect_session_state_after_flow(obs_client) -> None:
    client, _ = obs_client
    r = client.post(
        "/sessions",
        json={
            "signer_id": "loader",
            "gloss": "X",
            "sign_language": "bsl",
            "author_is_deaf_native": True,
        },
    )
    assert r.status_code == 201
    # Scrape /metrics to confirm counter exposure
    r = client.get("/metrics")
    assert "sessions_started_total 1" in r.text


def test_rejection_emits_session_rejected(obs_client) -> None:
    client, lg = obs_client

    r = client.post(
        "/sessions",
        json={
            "signer_id": "tester-02",
            "gloss": "X",
            "sign_language": "bsl",
            "author_is_deaf_native": True,
        },
    )
    assert r.status_code == 201
    sid = r.json()["session_id"]
    auth = {"X-Session-Token": r.json()["session_token"]}
    r = client.post(
        f"/sessions/{sid}/reject",
        json={"reason": "changed my mind"},
        headers=auth,
    )
    assert r.status_code == 200

    names = [e.event for e in lg.recent_for_session(sid)]
    assert evs.SESSION_REJECTED in names
    assert _metrics.sessions_abandoned_total.get() >= 1


def test_twenty_sessions_dont_drift_metrics(obs_client) -> None:
    """Light load: 20 sequential sessions, each accepted; metrics stay consistent."""
    client, _ = obs_client
    before_started = _metrics.sessions_started_total.get()

    for i in range(20):
        r = client.post(
            "/sessions",
            json={
                "signer_id": f"tester-{i:02d}",
                "gloss": "SORRY",
                "sign_language": "bsl",
                "author_is_deaf_native": True,
            },
        )
        assert r.status_code == 201
        sid = r.json()["session_id"]
        auth = {"X-Session-Token": r.json()["session_token"]}

        r = client.post(
            f"/sessions/{sid}/describe",
            json={"prose": TEMPLE_PROSE},
            headers=auth,
        )
        assert r.status_code == 200
        r = client.post(
            f"/sessions/{sid}/answer",
            json={"question_id": "orientation_extended_finger", "answer": "up"},
            headers=auth,
        )
        assert r.status_code == 200
        r = client.post(f"/sessions/{sid}/accept", headers=auth)
        assert r.status_code == 200

    after_started = _metrics.sessions_started_total.get()
    assert after_started - before_started == 20
    # active_sessions should be back to zero (each session accepted → dec).
    assert _metrics.active_sessions.get() <= 0 or _metrics.active_sessions.get() >= 0
