"""Tests for :mod:`backend.chat2hamnosys.session.orchestrator`.

Drives the state machine end-to-end with stubbed ``parse_fn`` and
``question_fn`` callables so the suite does not depend on the LLM
parser / clarifier. The generator leaf is exercised for real because
VOCAB covers every term we use in the fixture bundle (``fist``,
``up``, ``down``, ``temple``, ``down``), and the renderer leaf is
stubbed via ``render_fn=None`` so the tests do not shell out.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from clarify import Option, Question
from hamnosys import validate
from parser import Gap, ParseResult
from parser.models import (
    PartialMovementSegment,
    PartialSignParameters,
)
from session.orchestrator import (
    Correction,
    InvalidTransitionError,
    apply_correction_diff,
    check_timeout,
    on_accept,
    on_clarification_answer,
    on_correction,
    on_description,
    on_reject,
    run_generation,
    start_session,
)
from session.state import (
    INACTIVITY_TIMEOUT,
    AbandonedEvent,
    AcceptedEvent,
    AuthoringSession,
    ClarificationAnsweredEvent,
    ClarificationAskedEvent,
    CorrectionAppliedEvent,
    CorrectionRequestedEvent,
    DescribedEvent,
    GeneratedEvent,
    RejectedEvent,
    SessionState,
)
from storage import SQLiteSignStore


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


SOLID_HAMNOSYS = "\uE000\uE020\uE03C\uE049\uE084"


def _solid_partial() -> PartialSignParameters:
    """VOCAB-resolvable parameter bundle (produces SOLID_HAMNOSYS)."""
    return PartialSignParameters(
        handshape_dominant="fist",
        orientation_extended_finger="up",
        orientation_palm="down",
        location="temple",
        movement=[PartialMovementSegment(path="down")],
    )


def _gappy_partial() -> PartialSignParameters:
    """A partial where orientation and movement are missing."""
    return PartialSignParameters(
        handshape_dominant="fist",
        location="temple",
    )


def _q(
    field: str,
    value: str,
    *,
    text: str = "question?",
) -> Question:
    return Question(
        field=field,
        text=text,
        options=[Option(label=value, value=value)],
        allow_freeform=True,
        rationale="stub",
    )


class _StubParser:
    """Scripted ``parse_fn`` — returns a ``ParseResult`` per invocation.

    ``parse_description`` is called once per :func:`on_description` call,
    so the test supplies one result per expected invocation.
    """

    def __init__(self, *results: ParseResult) -> None:
        self._results = list(results)
        self.calls: list[str] = []

    def __call__(self, prose: str) -> ParseResult:
        self.calls.append(prose)
        if not self._results:
            raise AssertionError(
                f"parser stub exhausted; extra call with prose={prose!r}"
            )
        return self._results.pop(0)


class _StubQuestioner:
    """Scripted ``question_fn`` — returns a programmed batch per call."""

    def __init__(self, *batches: list[Question]) -> None:
        self._batches = [list(b) for b in batches]
        self.calls = 0

    def __call__(self, parse_result, prior_turns, *, is_deaf_native):  # noqa: D401
        self.calls += 1
        if not self._batches:
            # No more scripted batches — return empty so the flow proceeds.
            return []
        return self._batches.pop(0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_start_session_requires_signer_id():
    with pytest.raises(ValueError, match="signer_id"):
        start_session(signer_id="")


def test_start_session_seeds_draft_fields():
    s = start_session(
        signer_id="alice",
        gloss="TEMPLE",
        sign_language="bsl",
        is_deaf_native=True,
        display_name="Alice",
        domain="anatomy",
    )
    assert s.state == SessionState.AWAITING_DESCRIPTION
    assert s.draft.author_signer_id == "alice"
    assert s.draft.author_display_name == "Alice"
    assert s.draft.author_is_deaf_native is True
    assert s.draft.gloss == "TEMPLE"
    assert s.draft.domain == "anatomy"


# ---------------------------------------------------------------------------
# on_description: routing
# ---------------------------------------------------------------------------


def test_on_description_routes_to_generating_when_parser_and_clarifier_agree():
    # Parser returns no gaps; clarifier returns nothing — straight to GENERATING.
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    question_fn = _StubQuestioner([])  # empty batch

    s = start_session(signer_id="alice", gloss="TEMPLE")
    s = on_description(s, "a closed fist at the temple moving down", parse_fn=parse_fn, question_fn=question_fn)

    assert s.state == SessionState.GENERATING
    assert parse_fn.calls == ["a closed fist at the temple moving down"]
    assert question_fn.calls == 1
    # history contains described, no clarification_asked
    types = [e.type for e in s.history]
    assert types == ["described"]
    described = s.history[0]
    assert isinstance(described, DescribedEvent)
    assert described.gaps_found == 0
    # Draft now carries the parser's output.
    assert s.draft.description_prose == "a closed fist at the temple moving down"
    assert s.draft.parameters_partial is not None
    assert s.draft.parameters_partial.handshape_dominant == "fist"


def test_on_description_routes_to_clarifying_when_questions_returned():
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
    q = _q("orientation_extended_finger", "up", text="Which finger direction?")
    question_fn = _StubQuestioner([q])

    s = start_session(signer_id="alice", gloss="TEMPLE")
    s = on_description(s, "fist at temple", parse_fn=parse_fn, question_fn=question_fn)

    assert s.state == SessionState.CLARIFYING
    assert [q.field for q in s.draft.pending_questions] == [
        "orientation_extended_finger"
    ]
    assert len(s.draft.clarifications) == 2  # author + assistant turn
    assert s.draft.clarifications[0].role == "author"
    assert s.draft.clarifications[1].role == "assistant"
    types = [e.type for e in s.history]
    assert types == ["described", "clarification_asked"]


def test_on_description_rejects_empty_prose():
    s = start_session(signer_id="alice")
    with pytest.raises(ValueError, match="prose"):
        on_description(s, "   ", parse_fn=_StubParser(), question_fn=_StubQuestioner())


def test_on_description_invalid_from_rendered():
    s = start_session(signer_id="alice").with_state(SessionState.RENDERED)
    with pytest.raises(InvalidTransitionError) as excinfo:
        on_description(
            s, "anything",
            parse_fn=_StubParser(
                ParseResult(parameters=PartialSignParameters(), gaps=[], raw_response="{}")
            ),
            question_fn=_StubQuestioner([]),
        )
    assert excinfo.value.current == SessionState.RENDERED
    assert excinfo.value.transition == "on_description"


# ---------------------------------------------------------------------------
# Clarify loop — happy path, "clarify twice" simulation
# ---------------------------------------------------------------------------


def _script_apply_answer(param_sequence: list[PartialSignParameters]):
    """Stubbed ``apply_fn`` — pops the next params snapshot per call."""

    pending = list(param_sequence)

    def _apply(params, question, answer):
        if not pending:
            raise AssertionError("apply_fn stub exhausted")
        return pending.pop(0)

    return _apply


def test_happy_path_description_clarify_twice_generate_accept(tmp_path: Path):
    # Two gaps on first parse; clarifier returns one question per turn.
    first_gap = Gap(
        field="orientation_extended_finger",
        reason="not described",
        suggested_question="Finger direction?",
    )
    second_gap = Gap(
        field="orientation_palm",
        reason="not described",
        suggested_question="Palm direction?",
    )
    parse_fn = _StubParser(
        ParseResult(
            parameters=_gappy_partial(),
            gaps=[first_gap, second_gap],
            raw_response="{}",
        )
    )
    # First batch asks about the extended finger; after the answer,
    # orchestrator re-asks question_fn and gets the palm batch; after
    # that, no more batches — the session progresses to GENERATING.
    q1 = _q("orientation_extended_finger", "up")
    q2 = _q("orientation_palm", "down")
    question_fn = _StubQuestioner([q1], [q2], [])  # 3 potential calls

    # apply_fn returns a params snapshot that progressively fills slots.
    after_q1 = PartialSignParameters(
        handshape_dominant="fist",
        orientation_extended_finger="up",
        location="temple",
    )
    after_q2 = _solid_partial()
    apply_fn = _script_apply_answer([after_q1, after_q2])

    s = start_session(signer_id="alice", gloss="TEMPLE")
    s = on_description(s, "fist at temple", parse_fn=parse_fn, question_fn=question_fn)
    assert s.state == SessionState.CLARIFYING
    assert s.draft.gaps == [first_gap, second_gap]

    # First clarification answer — still one gap left, so loops into CLARIFYING.
    s = on_clarification_answer(
        s, "orientation_extended_finger", "up",
        apply_fn=apply_fn, question_fn=question_fn,
    )
    assert s.state == SessionState.CLARIFYING
    assert [g.field for g in s.draft.gaps] == ["orientation_palm"]

    # Second answer — no gaps left, so we advance to GENERATING.
    s = on_clarification_answer(
        s, "orientation_palm", "down",
        apply_fn=apply_fn, question_fn=question_fn,
    )
    assert s.state == SessionState.GENERATING

    # Run the real generator (VOCAB resolves every slot).
    s = run_generation(s, render_fn=None)
    assert s.state == SessionState.RENDERED
    assert s.draft.hamnosys == SOLID_HAMNOSYS
    assert validate(s.draft.hamnosys).ok is True
    # Generated event appended in success shape.
    assert any(
        isinstance(e, GeneratedEvent) and e.success and e.hamnosys == SOLID_HAMNOSYS
        for e in s.history
    )

    # Accept — finalize into a real SignEntry and persist.
    store = SQLiteSignStore(db_path=tmp_path / "signs.sqlite3")
    s = on_accept(s, store=store)
    assert s.state == SessionState.FINALIZED
    accepted = [e for e in s.history if isinstance(e, AcceptedEvent)]
    assert len(accepted) == 1
    entry = store.get(accepted[0].sign_entry_id)
    assert entry.gloss == "TEMPLE"
    assert entry.hamnosys == SOLID_HAMNOSYS
    assert entry.parameters.handshape_dominant == "\uE000"

    # Events logged in order (no DESCRIBED after the first).
    types = [e.type for e in s.history]
    assert types == [
        "described",
        "clarification_asked",
        "clarification_answered",
        "clarification_asked",
        "clarification_answered",
        "generated",
        "accepted",
    ]


def test_on_clarification_answer_stays_clarifying_while_pending_remain():
    # Parser returns two gaps. The orchestrator emits clarifications
    # one at a time, so the first round asks q1 and the second round
    # asks q2 after the user answers q1. State must stay in CLARIFYING
    # across both rounds.
    q1 = _q("orientation_extended_finger", "up")
    q2 = _q("orientation_palm", "down")
    parse_fn = _StubParser(
        ParseResult(
            parameters=_gappy_partial(),
            gaps=[
                Gap(field="orientation_extended_finger", reason="r", suggested_question="Qa?"),
                Gap(field="orientation_palm", reason="r", suggested_question="Qb?"),
            ],
            raw_response="{}",
        )
    )
    question_fn = _StubQuestioner([q1], [q2])

    s = start_session(signer_id="alice", gloss="TEMPLE")
    s = on_description(s, "fist at temple", parse_fn=parse_fn, question_fn=question_fn)
    assert [q.field for q in s.draft.pending_questions] == ["orientation_extended_finger"]

    after_q1 = PartialSignParameters(
        handshape_dominant="fist",
        orientation_extended_finger="up",
        location="temple",
    )
    apply_fn = _script_apply_answer([after_q1])

    s = on_clarification_answer(
        s, "orientation_extended_finger", "up",
        apply_fn=apply_fn, question_fn=question_fn,
    )
    assert s.state == SessionState.CLARIFYING
    assert [q.field for q in s.draft.pending_questions] == ["orientation_palm"]
    types = [e.type for e in s.history]
    assert types == [
        "described",
        "clarification_asked",
        "clarification_answered",
        "clarification_asked",
    ]


def test_on_clarification_answer_requires_pending_question_for_field():
    # Put the session into CLARIFYING with one pending question.
    parse_fn = _StubParser(
        ParseResult(
            parameters=_gappy_partial(),
            gaps=[Gap(field="orientation_palm", reason="r", suggested_question="Q?")],
            raw_response="{}",
        )
    )
    question_fn = _StubQuestioner([_q("orientation_palm", "down")])
    s = start_session(signer_id="alice", gloss="TEMPLE")
    s = on_description(s, "fist at temple", parse_fn=parse_fn, question_fn=question_fn)
    with pytest.raises(ValueError, match="no pending question"):
        on_clarification_answer(
            s, "location", "temple",
            apply_fn=_script_apply_answer([_solid_partial()]),
            question_fn=question_fn,
        )


def test_on_clarification_answer_requires_clarifying_state():
    s = start_session(signer_id="alice")
    with pytest.raises(InvalidTransitionError) as excinfo:
        on_clarification_answer(
            s, "orientation_palm", "down",
            apply_fn=_script_apply_answer([_solid_partial()]),
            question_fn=_StubQuestioner(),
        )
    assert excinfo.value.transition == "on_clarification_answer"


# ---------------------------------------------------------------------------
# run_generation — failure and success
# ---------------------------------------------------------------------------


def _build_generating_session(
    parse_fn: _StubParser,
    question_fn: _StubQuestioner,
) -> AuthoringSession:
    """Drive a session to GENERATING via on_description (solid partial)."""
    s = start_session(signer_id="alice", gloss="TEMPLE")
    return on_description(
        s, "fist at temple",
        parse_fn=parse_fn, question_fn=question_fn,
    )


def test_run_generation_requires_generating_state():
    s = start_session(signer_id="alice")
    with pytest.raises(InvalidTransitionError):
        run_generation(s, render_fn=None)


def test_run_generation_requires_populated_partial():
    # Force state without running on_description first.
    s = start_session(signer_id="alice").with_state(SessionState.GENERATING)
    with pytest.raises(RuntimeError, match="parameters_partial"):
        run_generation(s, render_fn=None)


def test_run_generation_failure_stays_in_generating():
    # parse_fn feeds a bundle that VOCAB can't resolve; with no LLM
    # client the generator returns hamnosys=None, and the session must
    # stay in GENERATING with generation_errors populated.
    impossible = PartialSignParameters(
        handshape_dominant="zzzz-not-a-term",
        orientation_extended_finger="zzzz",
        orientation_palm="zzzz",
        location="zzzz",
        movement=[PartialMovementSegment(path="zzzz")],
    )
    parse_fn = _StubParser(
        ParseResult(parameters=impossible, gaps=[], raw_response="{}")
    )
    question_fn = _StubQuestioner([])
    s = _build_generating_session(parse_fn, question_fn)
    assert s.state == SessionState.GENERATING
    s = run_generation(s, render_fn=None)
    assert s.state == SessionState.GENERATING
    assert s.draft.hamnosys is None
    assert s.draft.generation_errors  # non-empty
    gen_events = [e for e in s.history if isinstance(e, GeneratedEvent)]
    assert gen_events and gen_events[-1].success is False


# ---------------------------------------------------------------------------
# Correction path
# ---------------------------------------------------------------------------


def test_on_correction_records_request_and_moves_to_applying():
    # Fast path to RENDERED: single-shot description with a solid partial.
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    question_fn = _StubQuestioner([])
    s = _build_generating_session(parse_fn, question_fn)
    s = run_generation(s, render_fn=None)
    assert s.state == SessionState.RENDERED

    s = on_correction(s, Correction(raw_text="move down, not up"))
    assert s.state == SessionState.APPLYING_CORRECTION
    req_events = [e for e in s.history if isinstance(e, CorrectionRequestedEvent)]
    assert len(req_events) == 1
    assert req_events[0].raw_text == "move down, not up"
    assert s.draft.corrections_count == 1


def test_on_correction_rejects_empty_text():
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    s = run_generation(
        _build_generating_session(parse_fn, _StubQuestioner([])),
        render_fn=None,
    )
    with pytest.raises(ValueError, match="raw_text"):
        on_correction(s, Correction(raw_text="   "))


def test_on_correction_invalid_from_awaiting_description():
    s = start_session(signer_id="alice")
    with pytest.raises(InvalidTransitionError):
        on_correction(s, Correction(raw_text="anything"))


def test_apply_correction_diff_then_regenerate_accepts(tmp_path: Path):
    # Initial run: gappy partial would fail, so start from solid.
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    question_fn = _StubQuestioner([])
    s = _build_generating_session(parse_fn, question_fn)
    s = run_generation(s, render_fn=None)

    # User flags a correction; interpreter (Prompt 10) will call
    # apply_correction_diff with the updated params. Here we simulate
    # by swapping the palm orientation — still VOCAB-resolvable so
    # re-running the generator produces a fresh HamNoSys string.
    s = on_correction(s, Correction(raw_text="palm up, not down"))
    updated = _solid_partial().model_copy(update={"orientation_palm": "up"})
    s = apply_correction_diff(
        s,
        field_changes=[{"field": "orientation_palm", "from": "down", "to": "up"}],
        summary="palm direction flipped",
        updated_params=updated,
    )
    assert s.state == SessionState.GENERATING
    applied = [e for e in s.history if isinstance(e, CorrectionAppliedEvent)]
    assert len(applied) == 1
    assert applied[0].summary == "palm direction flipped"

    s = run_generation(s, render_fn=None)
    assert s.state == SessionState.RENDERED
    # Palm orientation swapped — new HamNoSys should reflect that.
    assert s.draft.hamnosys != SOLID_HAMNOSYS
    assert validate(s.draft.hamnosys).ok is True

    store = SQLiteSignStore(db_path=tmp_path / "signs.sqlite3")
    s = on_accept(s, store=store)
    assert s.state == SessionState.FINALIZED


def test_apply_correction_diff_requires_applying_state():
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    s = run_generation(
        _build_generating_session(parse_fn, _StubQuestioner([])),
        render_fn=None,
    )
    # In RENDERED — apply_correction_diff should refuse.
    with pytest.raises(InvalidTransitionError):
        apply_correction_diff(s, field_changes=[], summary="nope")


# ---------------------------------------------------------------------------
# Accept / reject / timeout
# ---------------------------------------------------------------------------


def test_on_accept_invalid_from_awaiting_description(tmp_path: Path):
    s = start_session(signer_id="alice", gloss="TEMPLE")
    store = SQLiteSignStore(db_path=tmp_path / "signs.sqlite3")
    with pytest.raises(InvalidTransitionError) as excinfo:
        on_accept(s, store=store)
    assert excinfo.value.current == SessionState.AWAITING_DESCRIPTION


def test_on_reject_from_non_terminal_state_goes_to_abandoned():
    s = start_session(signer_id="alice")
    s = on_reject(s, reason="user cancelled")
    assert s.state == SessionState.ABANDONED
    assert any(isinstance(e, RejectedEvent) and e.reason == "user cancelled"
               for e in s.history)


def test_on_reject_from_terminal_raises():
    s = start_session(signer_id="alice").with_state(SessionState.FINALIZED)
    with pytest.raises(InvalidTransitionError):
        on_reject(s, reason="too late")


def test_check_timeout_idempotent_for_fresh_session():
    s = start_session(signer_id="alice")
    assert check_timeout(s) is s  # unchanged object


def test_check_timeout_marks_stale_session_abandoned():
    ancient = datetime.now(timezone.utc) - INACTIVITY_TIMEOUT - timedelta(seconds=1)
    s = AuthoringSession(last_activity_at=ancient)
    s = check_timeout(s)
    assert s.state == SessionState.ABANDONED
    assert any(isinstance(e, AbandonedEvent) for e in s.history)


def test_check_timeout_no_op_on_terminal_session():
    s = AuthoringSession(state=SessionState.FINALIZED)
    assert check_timeout(s) is s


# ---------------------------------------------------------------------------
# Purity: transitions never mutate the input
# ---------------------------------------------------------------------------


def test_on_description_does_not_mutate_input_session():
    parse_fn = _StubParser(
        ParseResult(parameters=_solid_partial(), gaps=[], raw_response="{}")
    )
    question_fn = _StubQuestioner([])
    s = start_session(signer_id="alice", gloss="TEMPLE")
    original_history = list(s.history)
    original_state = s.state
    t = on_description(s, "fist at temple", parse_fn=parse_fn, question_fn=question_fn)
    # Input left alone.
    assert s.history == original_history
    assert s.state == original_state
    # Output progressed.
    assert t is not s
    assert t.state != original_state
