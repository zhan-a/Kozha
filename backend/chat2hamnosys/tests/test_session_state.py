"""Tests for :mod:`backend.chat2hamnosys.session.state`.

Covers the state container mechanics (immutability of transitions,
event-log append, staleness), event-union serialisation round-trips,
and the draft-to-SignEntry promotion path used by ``on_accept``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from clarify import Option, Question
from parser.models import (
    PartialMovementSegment,
    PartialSignParameters,
)
from session.state import (
    INACTIVITY_TIMEOUT,
    AcceptedEvent,
    AuthoringSession,
    ClarificationAnsweredEvent,
    ClarificationAskedEvent,
    CorrectionRequestedEvent,
    DescribedEvent,
    DraftFinalizationError,
    GeneratedEvent,
    RejectedEvent,
    SessionState,
    SignEntryDraft,
    TERMINAL_STATES,
)


# A reusable, VOCAB-resolvable parameter bundle.
def _solid_partial() -> PartialSignParameters:
    return PartialSignParameters(
        handshape_dominant="fist",
        orientation_extended_finger="up",
        orientation_palm="down",
        location="temple",
        movement=[PartialMovementSegment(path="down")],
    )


# Per the smoke-check run in the repo: these are the codepoints the
# composer emits for the ``_solid_partial`` bundle above.
SOLID_CHUNKS: dict[str, str] = {
    "handshape_dominant": "\uE000",
    "orientation_extended_finger": "\uE020",
    "orientation_palm": "\uE03C",
    "location": "\uE049",
    "movement[0].path": "\uE084",
}
SOLID_HAMNOSYS = "\uE000\uE020\uE03C\uE049\uE084"


# ---------------------------------------------------------------------------
# SessionState / terminal set
# ---------------------------------------------------------------------------


def test_session_state_values_are_stable_strings():
    # The value-side of each enum is what storage serialises, so
    # renaming any of these is a schema-breaking change.
    assert SessionState.AWAITING_DESCRIPTION.value == "awaiting_description"
    assert SessionState.CLARIFYING.value == "clarifying"
    assert SessionState.RENDERED.value == "rendered"
    assert SessionState.FINALIZED.value == "finalized"
    assert SessionState.ABANDONED.value == "abandoned"


def test_terminal_states_contains_only_finalized_and_abandoned():
    assert TERMINAL_STATES == frozenset(
        {SessionState.FINALIZED, SessionState.ABANDONED}
    )


# ---------------------------------------------------------------------------
# AuthoringSession mechanics
# ---------------------------------------------------------------------------


def test_default_session_is_awaiting_description():
    s = AuthoringSession()
    assert s.state == SessionState.AWAITING_DESCRIPTION
    assert s.history == []
    assert isinstance(s.id, UUID)
    assert s.created_at <= s.last_activity_at


def test_with_state_returns_new_session_with_updated_activity():
    s = AuthoringSession()
    original_activity = s.last_activity_at
    # Sleep a microsecond via re-set to force progression.
    t = s.with_state(SessionState.CLARIFYING)
    assert t is not s  # copy, not mutation
    assert t.state == SessionState.CLARIFYING
    assert s.state == SessionState.AWAITING_DESCRIPTION  # unchanged
    assert t.last_activity_at >= original_activity


def test_append_event_is_non_mutating():
    s = AuthoringSession()
    ev = DescribedEvent(prose="hello", gaps_found=0)
    t = s.append_event(ev)
    assert s.history == []
    assert len(t.history) == 1
    assert t.history[0].type == "described"


def test_with_draft_patches_only_specified_fields():
    s = AuthoringSession()
    s = s.with_draft(gloss="WATER", domain="natural elements")
    assert s.draft.gloss == "WATER"
    assert s.draft.domain == "natural elements"
    # Unrelated defaults untouched.
    assert s.draft.description_prose == ""
    assert s.draft.sign_language == "bsl"


def test_is_stale_respects_timeout_and_terminal_exclusion():
    now = datetime.now(timezone.utc)
    ancient = now - INACTIVITY_TIMEOUT - timedelta(seconds=1)

    fresh = AuthoringSession(last_activity_at=now)
    assert fresh.is_stale(now=now) is False

    stale = AuthoringSession(last_activity_at=ancient)
    assert stale.is_stale(now=now) is True

    # Terminal sessions never count as stale even when old.
    terminal = AuthoringSession(
        state=SessionState.FINALIZED, last_activity_at=ancient
    )
    assert terminal.is_stale(now=now) is False


# ---------------------------------------------------------------------------
# Event-union serialisation
# ---------------------------------------------------------------------------


def test_session_round_trips_through_json_with_mixed_events():
    q = Question(
        field="handshape_dominant",
        text="Which handshape?",
        options=[Option(label="Fist", value="fist")],
        allow_freeform=True,
        rationale="test",
    )
    events = [
        DescribedEvent(prose="closed fist at the temple", gaps_found=1),
        ClarificationAskedEvent(questions=[q]),
        ClarificationAnsweredEvent(
            question_field="handshape_dominant",
            answer="fist",
            resolved_value="fist",
        ),
        GeneratedEvent(
            success=True,
            hamnosys=SOLID_HAMNOSYS,
            sigml="<sigml/>",
            confidence=1.0,
            used_llm_fallback=False,
        ),
        CorrectionRequestedEvent(raw_text="move down, not up"),
        AcceptedEvent(sign_entry_id=uuid4(), status="draft"),
        RejectedEvent(reason="user cancelled"),
    ]
    s = AuthoringSession(history=events)
    raw = s.model_dump_json()
    restored = AuthoringSession.model_validate_json(raw)

    assert len(restored.history) == len(events)
    # Discriminated union restored to the concrete subclasses.
    types = [e.type for e in restored.history]
    assert types == [
        "described",
        "clarification_asked",
        "clarification_answered",
        "generated",
        "correction_requested",
        "accepted",
        "rejected",
    ]
    asked = restored.history[1]
    assert isinstance(asked, ClarificationAskedEvent)
    assert asked.questions[0].field == "handshape_dominant"


def test_session_round_trip_preserves_draft_and_partial_params():
    draft = SignEntryDraft(
        gloss="TEMPLE",
        author_signer_id="bob-007",
        parameters_partial=_solid_partial(),
        slot_codepoints=SOLID_CHUNKS,
        hamnosys=SOLID_HAMNOSYS,
    )
    s = AuthoringSession(draft=draft)
    raw = s.model_dump_json()
    restored = AuthoringSession.model_validate_json(raw)
    assert restored.draft.gloss == "TEMPLE"
    assert restored.draft.parameters_partial is not None
    assert restored.draft.parameters_partial.handshape_dominant == "fist"
    assert restored.draft.slot_codepoints == SOLID_CHUNKS


# ---------------------------------------------------------------------------
# Draft → SignEntry finalisation
# ---------------------------------------------------------------------------


def test_draft_to_sign_entry_happy_path():
    draft = SignEntryDraft(
        gloss="TEMPLE",
        author_signer_id="alice",
        author_is_deaf_native=True,
        parameters_partial=_solid_partial(),
        slot_codepoints=SOLID_CHUNKS,
        hamnosys=SOLID_HAMNOSYS,
        sigml="<sigml/>",
    )
    entry = draft.to_sign_entry()
    assert entry.gloss == "TEMPLE"
    assert entry.status == "draft"
    assert entry.hamnosys == SOLID_HAMNOSYS
    assert entry.parameters.handshape_dominant == "\uE000"
    assert entry.parameters.location == "\uE049"
    assert entry.parameters.movement[0].path == "\uE084"
    assert entry.author.signer_id == "alice"
    assert entry.author.is_deaf_native is True


def test_draft_to_sign_entry_requires_gloss():
    draft = SignEntryDraft(
        author_signer_id="alice",
        parameters_partial=_solid_partial(),
        slot_codepoints=SOLID_CHUNKS,
        hamnosys=SOLID_HAMNOSYS,
    )
    with pytest.raises(DraftFinalizationError, match="no gloss"):
        draft.to_sign_entry()


def test_draft_to_sign_entry_requires_author_signer_id():
    draft = SignEntryDraft(
        gloss="TEMPLE",
        parameters_partial=_solid_partial(),
        slot_codepoints=SOLID_CHUNKS,
        hamnosys=SOLID_HAMNOSYS,
    )
    with pytest.raises(DraftFinalizationError, match="author"):
        draft.to_sign_entry()


def test_draft_to_sign_entry_requires_hamnosys():
    draft = SignEntryDraft(
        gloss="TEMPLE",
        author_signer_id="alice",
        parameters_partial=_solid_partial(),
        slot_codepoints=SOLID_CHUNKS,
        hamnosys=None,
    )
    with pytest.raises(DraftFinalizationError, match="hamnosys"):
        draft.to_sign_entry()


def test_draft_to_sign_entry_requires_mandatory_slot_chunks():
    # Drop the orientation_palm chunk — finalize should refuse rather
    # than build a half-populated SignParameters that would later fail
    # the pydantic validator.
    chunks = {k: v for k, v in SOLID_CHUNKS.items() if k != "orientation_palm"}
    draft = SignEntryDraft(
        gloss="TEMPLE",
        author_signer_id="alice",
        parameters_partial=_solid_partial(),
        slot_codepoints=chunks,
        hamnosys=SOLID_HAMNOSYS,
    )
    with pytest.raises(DraftFinalizationError, match="orientation_palm"):
        draft.to_sign_entry()
