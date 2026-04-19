"""Unit tests for the chat2hamnosys FastAPI router.

Covers the transport layer only — the orchestration callables are
overridden via ``app.dependency_overrides`` so the suite never touches
an LLM. The underlying VOCAB composer is still real (``generate`` runs
deterministically when the stubbed parser returns VOCAB-resolvable
partial parameters) so the HTTP → orchestrator → generator → HamNoSys
pipeline is exercised end to end, just without LLM round-trips.

Scope
-----
- Happy path: create → describe (no gaps) → accept.
- Clarification loop: create → describe (one gap) → answer → accept.
- 404 ``session_not_found`` for unknown UUID.
- 403 ``session_forbidden`` for wrong / missing ``X-Session-Token``.
- 409 ``invalid_transition`` (accept before RENDERED).
- 422 ``validation_error`` for schema violations.
- 429 ``rate_limited`` when the per-IP limit trips.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from clarify import Option, Question
from parser import Gap, ParseResult
from parser.models import PartialMovementSegment, PartialSignParameters


# ---------------------------------------------------------------------------
# Fixtures + stubs
# ---------------------------------------------------------------------------


SOLID_HAMNOSYS = "\uE000\uE020\uE03C\uE049\uE084"


def _solid_partial() -> PartialSignParameters:
    """A VOCAB-resolvable parameter bundle (→ ``SOLID_HAMNOSYS``)."""
    return PartialSignParameters(
        handshape_dominant="fist",
        orientation_extended_finger="up",
        orientation_palm="down",
        location="temple",
        movement=[PartialMovementSegment(path="down")],
    )


def _gappy_partial() -> PartialSignParameters:
    return PartialSignParameters(handshape_dominant="fist", location="temple")


class _StubParser:
    """Scripted ``parse_fn``: returns one :class:`ParseResult` per call."""

    def __init__(self, *results: ParseResult) -> None:
        self._results = list(results)
        self.calls: list[str] = []

    def __call__(self, prose: str) -> ParseResult:
        self.calls.append(prose)
        if not self._results:
            raise AssertionError(f"parser stub exhausted; extra call: {prose!r}")
        return self._results.pop(0)


class _StubQuestioner:
    def __init__(self, *batches: list[Question]) -> None:
        self._batches = [list(b) for b in batches]
        self.calls = 0

    def __call__(self, parse_result, prior_turns, *, is_deaf_native):  # noqa: D401
        self.calls += 1
        if not self._batches:
            return []
        return self._batches.pop(0)


def _q(field: str, value: str, *, text: str = "question?") -> Question:
    return Question(
        field=field,
        text=text,
        options=[Option(label=value, value=value)],
        allow_freeform=True,
        rationale="stub",
    )


def _script_apply(seq: list[PartialSignParameters]):
    pending = list(seq)

    def _apply(params, question, answer):
        if not pending:
            raise AssertionError("apply_fn stub exhausted")
        return pending.pop(0)

    return _apply


# ---------------------------------------------------------------------------
# App builder — fresh app per test + env-isolated SQLite paths
# ---------------------------------------------------------------------------


def _build_app(
    tmp_path: Path,
    *,
    parse_fn=None,
    question_fn=None,
    apply_fn=None,
    rate_limit: str = "30/minute",
    monkeypatch: pytest.MonkeyPatch,
):
    """Build a fresh sub-app + TestClient pointed at tmp SQLite paths.

    Each test gets a brand-new :class:`SessionStore`,
    :class:`TokenStore`, and :class:`SQLiteSignStore` so persistence
    doesn't leak between cases. Rate-limit value comes from the
    ``rate_limit`` argument, which maps to the ``CHAT2HAMNOSYS_RATE_LIMIT``
    env var so :func:`api.app.create_app` picks it up.
    """
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_SIGN_DB", str(tmp_path / "signs.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_TOKEN_DB", str(tmp_path / "tokens.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "kozha_data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_RATE_LIMIT", rate_limit)

    # Reset the singleton store caches so every test gets a fresh DB
    # connection pool pointed at its own tmp path.
    from api.dependencies import reset_stores

    reset_stores()

    from api import (
        create_app,
        get_apply_fn,
        get_parse_fn,
        get_question_fn,
        get_render_fn,
    )

    app = create_app()

    if parse_fn is not None:
        app.dependency_overrides[get_parse_fn] = lambda: parse_fn
    if question_fn is not None:
        app.dependency_overrides[get_question_fn] = lambda: question_fn
    if apply_fn is not None:
        app.dependency_overrides[get_apply_fn] = lambda: apply_fn
    # Always skip the real renderer in unit tests.
    app.dependency_overrides[get_render_fn] = lambda: None

    return app


@pytest.fixture
def client_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a factory so tests can pick their stubs and rate-limit."""
    created: list[TestClient] = []

    def _factory(**kwargs) -> TestClient:
        app = _build_app(tmp_path, monkeypatch=monkeypatch, **kwargs)
        tc = TestClient(app)
        created.append(tc)
        return tc

    yield _factory
    for tc in created:
        tc.close()


# ---------------------------------------------------------------------------
# Happy path — create, describe (no gaps), accept
# ---------------------------------------------------------------------------


def test_create_session_returns_token_and_envelope(client_factory):
    client = client_factory()
    resp = client.post(
        "/sessions",
        json={"signer_id": "alice", "gloss": "TEMPLE"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "session_id" in body
    assert body["state"] == "awaiting_description"
    assert len(body["session_token"]) > 20
    assert body["session"]["next_action"]["kind"] == "await_description"


def test_describe_then_accept_happy_path(client_factory):
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    question_fn = _StubQuestioner([])
    client = client_factory(parse_fn=parse_fn, question_fn=question_fn)

    # 1. create
    r = client.post("/sessions", json={"signer_id": "alice", "gloss": "TEMPLE"})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    token = r.json()["session_token"]
    auth = {"X-Session-Token": token}

    # 2. describe — no gaps means orchestrator runs generation inline.
    r = client.post(
        f"/sessions/{sid}/describe",
        json={"prose": "fist at temple moving down"},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    env = r.json()
    assert env["state"] == "rendered"
    assert env["hamnosys"] == SOLID_HAMNOSYS
    assert env["next_action"]["kind"] == "preview_ready"

    # 3. preview metadata echoes the same HamNoSys.
    r = client.get(f"/sessions/{sid}/preview", headers=auth)
    assert r.status_code == 200
    assert r.json()["hamnosys"] == SOLID_HAMNOSYS

    # 4. accept → FINALIZED + sign entry persisted.
    r = client.post(f"/sessions/{sid}/accept", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session"]["state"] == "finalized"
    assert body["sign_entry"]["gloss"] == "TEMPLE"
    assert body["sign_entry"]["hamnosys"] == SOLID_HAMNOSYS
    assert body["sign_entry"]["parameters"]["handshape_dominant"] == "\uE000"


# ---------------------------------------------------------------------------
# Clarification loop
# ---------------------------------------------------------------------------


def test_describe_with_gap_then_answer_advances_to_rendered(client_factory):
    parse_fn = _StubParser(
        ParseResult(
            parameters=_gappy_partial(),
            gaps=[
                Gap(
                    field="orientation_extended_finger",
                    reason="not described",
                    suggested_question="Finger direction?",
                ),
            ],
            raw_response="{}",
        )
    )
    question_fn = _StubQuestioner(
        [_q("orientation_extended_finger", "up")],
        [],  # second call (after answer) returns empty → GENERATING
    )
    # After answer, apply_fn returns a solid partial so generation succeeds.
    apply_fn = _script_apply([_solid_partial()])

    client = client_factory(
        parse_fn=parse_fn, question_fn=question_fn, apply_fn=apply_fn
    )
    r = client.post("/sessions", json={"signer_id": "alice", "gloss": "TEMPLE"})
    sid = r.json()["session_id"]
    auth = {"X-Session-Token": r.json()["session_token"]}

    # describe → CLARIFYING with one pending question
    r = client.post(
        f"/sessions/{sid}/describe",
        json={"prose": "fist at temple"},
        headers=auth,
    )
    assert r.status_code == 200
    env = r.json()
    assert env["state"] == "clarifying"
    assert env["next_action"]["kind"] == "answer_questions"
    assert len(env["next_action"]["questions"]) == 1
    assert env["next_action"]["questions"][0]["field"] == "orientation_extended_finger"

    # answer → GENERATING → RENDERED (auto-run in the router)
    r = client.post(
        f"/sessions/{sid}/answer",
        json={
            "question_id": "orientation_extended_finger",
            "answer": "up",
        },
        headers=auth,
    )
    assert r.status_code == 200, r.text
    env = r.json()
    assert env["state"] == "rendered"
    assert env["hamnosys"] == SOLID_HAMNOSYS


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_get_unknown_session_returns_404(client_factory):
    client = client_factory()
    bogus = uuid4()
    r = client.get(f"/sessions/{bogus}", headers={"X-Session-Token": "anything"})
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "session_not_found"


def test_wrong_token_returns_403(client_factory):
    client = client_factory()
    r = client.post("/sessions", json={"signer_id": "alice"})
    sid = r.json()["session_id"]
    # Wrong token → 403
    r = client.get(f"/sessions/{sid}", headers={"X-Session-Token": "not-the-real-one"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "session_forbidden"


def test_missing_token_returns_403(client_factory):
    client = client_factory()
    r = client.post("/sessions", json={"signer_id": "alice"})
    sid = r.json()["session_id"]
    # No header at all → 403
    r = client.get(f"/sessions/{sid}")
    assert r.status_code == 403


def test_accept_before_rendered_returns_409(client_factory):
    parse_fn = _StubParser(
        ParseResult(
            parameters=_gappy_partial(),
            gaps=[
                Gap(field="orientation_palm", reason="r", suggested_question="Q?"),
            ],
            raw_response="{}",
        )
    )
    question_fn = _StubQuestioner([_q("orientation_palm", "down")])
    client = client_factory(parse_fn=parse_fn, question_fn=question_fn)

    r = client.post("/sessions", json={"signer_id": "alice", "gloss": "TEMPLE"})
    sid = r.json()["session_id"]
    auth = {"X-Session-Token": r.json()["session_token"]}

    # describe → CLARIFYING
    client.post(
        f"/sessions/{sid}/describe",
        json={"prose": "fist at temple"},
        headers=auth,
    )
    # accept from CLARIFYING → 409
    r = client.post(f"/sessions/{sid}/accept", headers=auth)
    assert r.status_code == 409
    body = r.json()
    assert body["error"]["code"] == "invalid_transition"


def test_describe_with_empty_prose_returns_422(client_factory):
    client = client_factory()
    r = client.post("/sessions", json={"signer_id": "alice"})
    sid = r.json()["session_id"]
    auth = {"X-Session-Token": r.json()["session_token"]}

    r = client.post(f"/sessions/{sid}/describe", json={"prose": ""}, headers=auth)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_extra_field_in_request_body_returns_422(client_factory):
    client = client_factory()
    # CreateSessionRequest has extra="forbid" — unknown fields rejected.
    r = client.post(
        "/sessions",
        json={"signer_id": "alice", "unknown_field": "nope"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429(client_factory):
    # 2 requests per minute — the third from the same IP should trip.
    client = client_factory(rate_limit="2/minute")
    r1 = client.post("/sessions", json={"signer_id": "alice"})
    r2 = client.post("/sessions", json={"signer_id": "bob"})
    r3 = client.post("/sessions", json={"signer_id": "carol"})
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r3.status_code == 429
    body = r3.json()
    assert body["error"]["code"] == "rate_limited"


# ---------------------------------------------------------------------------
# Reject + preview-video
# ---------------------------------------------------------------------------


def test_reject_sets_abandoned(client_factory):
    client = client_factory()
    r = client.post("/sessions", json={"signer_id": "alice"})
    sid = r.json()["session_id"]
    auth = {"X-Session-Token": r.json()["session_token"]}
    r = client.post(
        f"/sessions/{sid}/reject",
        json={"reason": "not the right sign"},
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["state"] == "abandoned"


def test_preview_video_404_when_no_render(client_factory):
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    question_fn = _StubQuestioner([])
    client = client_factory(parse_fn=parse_fn, question_fn=question_fn)
    r = client.post("/sessions", json={"signer_id": "alice", "gloss": "TEMPLE"})
    sid = r.json()["session_id"]
    auth = {"X-Session-Token": r.json()["session_token"]}
    client.post(
        f"/sessions/{sid}/describe",
        json={"prose": "fist at temple"},
        headers=auth,
    )
    r = client.get(f"/sessions/{sid}/preview/video", headers=auth)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "preview_not_available"
