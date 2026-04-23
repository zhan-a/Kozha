"""End-to-end smoke test for the chat2hamnosys API.

Stands up the FastAPI sub-app with a stubbed parser / clarifier (so the
suite never calls an LLM) but uses the real VOCAB generator +
HamNoSys-to-SiGML converter. Exercises the canonical flow the prompt
calls out::

    POST /sessions
      → POST /sessions/{id}/describe  "it's signed near the temple,
                                       flat hand, moves down to the
                                       chest, like SORRY"
      → POST /sessions/{id}/answer     (for any gap the parser leaves)
      → POST /sessions/{id}/accept
      → verify the sign entry was persisted in the SQLite sign store

The parser stub mirrors the slots the LLM parser would extract from the
prose and leaves a single gap on ``orientation_extended_finger`` so the
clarification loop is exercised. All other slots resolve deterministically
via the VOCAB composer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from clarify import Option, Question
from parser import Gap, ParseResult
from parser.models import PartialMovementSegment, PartialSignParameters


TEMPLE_PROSE = (
    "it's signed near the temple, flat hand, moves down to the chest, like SORRY"
)


def _initial_partial_for_sorry() -> PartialSignParameters:
    """Plausible parser output for TEMPLE_PROSE — every term is in VOCAB."""
    return PartialSignParameters(
        handshape_dominant="flat",
        orientation_palm="down",
        location="temple",
        movement=[PartialMovementSegment(path="down")],
    )


def _full_partial_for_sorry() -> PartialSignParameters:
    """After the author answers the ext-finger question."""
    return PartialSignParameters(
        handshape_dominant="flat",
        orientation_extended_finger="up",
        orientation_palm="down",
        location="temple",
        movement=[PartialMovementSegment(path="down")],
    )


class _StubParser:
    def __init__(self, result: ParseResult) -> None:
        self._result = result

    def __call__(self, prose: str) -> ParseResult:
        assert prose.strip(), "parser stub should never see empty prose"
        return self._result


class _StubQuestioner:
    """Returns one question for the missing ext-finger slot, then nothing."""

    def __init__(self) -> None:
        self._served = False

    def __call__(self, parse_result, prior_turns, *, is_deaf_native):  # noqa: D401
        if self._served:
            return []
        self._served = True
        if not any(g.field == "orientation_extended_finger" for g in parse_result.gaps):
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
    """Write the answer back as the ext-finger value — shape matches real apply."""
    if question.field == "orientation_extended_finger":
        return params.model_copy(update={"orientation_extended_finger": answer.strip()})
    return params


@pytest.fixture
def integration_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app + client pointed at tmp SQLite stores."""
    session_db = tmp_path / "sessions.sqlite3"
    sign_db = tmp_path / "signs.sqlite3"
    token_db = tmp_path / "tokens.sqlite3"
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_DB", str(session_db))
    monkeypatch.setenv("CHAT2HAMNOSYS_SIGN_DB", str(sign_db))
    monkeypatch.setenv("CHAT2HAMNOSYS_TOKEN_DB", str(token_db))
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "kozha_data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_RATE_LIMIT", "500/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT", "500/minute")

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
        parameters=_initial_partial_for_sorry(),
        gaps=[
            Gap(
                field="orientation_extended_finger",
                reason="flat hand finger orientation not stated",
                suggested_question="Which way do the fingers point?",
            )
        ],
        raw_response="{}",
    )
    parser = _StubParser(parse_result)
    questioner = _StubQuestioner()

    app = create_app()
    app.dependency_overrides[get_parse_fn] = lambda: parser
    app.dependency_overrides[get_question_fn] = lambda: questioner
    app.dependency_overrides[get_apply_fn] = lambda: _apply_fn
    # Skip the avatar renderer — VOCAB composer still produces real HamNoSys.
    app.dependency_overrides[get_render_fn] = lambda: None

    tc = TestClient(app)
    try:
        yield tc, sign_db
    finally:
        tc.close()


def test_full_authoring_flow_persists_sign_entry(integration_client):
    client, sign_db = integration_client

    # --- 1. Create session with a gloss ------------------------------
    r = client.post(
        "/sessions",
        json={
            "signer_id": "tester-01",
            "gloss": "SORRY",
            "sign_language": "bsl",
            "author_is_deaf_native": True,
        },
    )
    assert r.status_code == 201, r.text
    created = r.json()
    sid = created["session_id"]
    auth = {"X-Session-Token": created["session_token"]}

    # --- 2. Describe — parser reports one gap, router lands in CLARIFYING.
    r = client.post(
        f"/sessions/{sid}/describe",
        json={"prose": TEMPLE_PROSE},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    env = r.json()
    assert env["state"] == "clarifying"
    assert env["next_action"]["kind"] == "answer_questions"
    pending_ids = [q["question_id"] for q in env["next_action"]["questions"]]
    assert "orientation_extended_finger" in pending_ids

    # --- 3. Answer — after apply_fn, the partial is VOCAB-resolvable.
    r = client.post(
        f"/sessions/{sid}/answer",
        json={"question_id": "orientation_extended_finger", "answer": "up"},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    env = r.json()
    # ``/answer`` now kicks ``run_generation`` into a BackgroundTask
    # so the HTTP response returns fast even when reasoning takes
    # minutes; the envelope we see here is the mid-transition one.
    # The TestClient drains background tasks before returning, so a
    # follow-up GET sees the settled post-generation state.
    assert env["state"] == "generating"
    r = client.get(f"/sessions/{sid}", headers=auth)
    assert r.status_code == 200, r.text
    env = r.json()
    assert env["state"] == "rendered"
    assert env["hamnosys"]
    # SiGML is a well-formed XML document.
    assert env["sigml"]
    assert env["sigml"].startswith("<?xml") or env["sigml"].lstrip().startswith("<")

    # --- 4. Accept — finalize into a SignEntry.
    r = client.post(f"/sessions/{sid}/accept", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session"]["state"] == "finalized"
    entry = body["sign_entry"]
    assert entry["gloss"] == "SORRY"
    assert entry["sign_language"] == "bsl"
    assert entry["hamnosys"] == env["hamnosys"]

    # --- 5. Verify the SQLite store received the row -----------------
    # ``payload`` carries the JSON-serialised :class:`SignEntry`; fall
    # back to that for hamnosys because the sign store doesn't index it
    # as its own column.
    import json as _json

    assert sign_db.exists(), "sign-store DB was not created"
    with sqlite3.connect(sign_db) as conn:
        rows = list(
            conn.execute("SELECT id, gloss, payload FROM sign_entries")
        )
    assert len(rows) == 1
    persisted_id, persisted_gloss, persisted_payload = rows[0]
    UUID(persisted_id)  # raises if not a UUID
    assert persisted_gloss == "SORRY"
    payload_obj = _json.loads(persisted_payload)
    assert payload_obj["hamnosys"] == env["hamnosys"]

    # --- 6. Follow-up GET shows terminal state + correct history ----
    r = client.get(f"/sessions/{sid}", headers=auth)
    assert r.status_code == 200
    env = r.json()
    assert env["state"] == "finalized"
    event_types = [e["type"] for e in env["history"]]
    assert event_types == [
        "described",
        "clarification_asked",
        "clarification_answered",
        "generated",
        "accepted",
    ]
