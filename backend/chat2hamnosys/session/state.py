"""Session state, event log, and in-progress sign draft.

An :class:`AuthoringSession` is the persistent state of one
prose-to-HamNoSys dialogue. It carries a :class:`SessionState` (the
phase the dialogue is in), a progressively-populated
:class:`SignEntryDraft` (the sign being built), and an append-only
:class:`history` of :class:`SessionEvent` records — one per transition.

The draft is a loose mirror of :class:`backend.chat2hamnosys.models.SignEntry`
with every slot optional: the authoring loop populates fields step by
step, and only at :data:`SessionState.FINALIZED` do we promote the
draft into a validated :class:`SignEntry`. The event log is the
authoritative record of what happened — state fields are derived,
events are not.

Pure state only
---------------
This module holds no I/O. Transitions live in
:mod:`.orchestrator`; persistence lives in :mod:`.storage`. Keeping the
state object Pydantic-serialisable is the whole reason sessions can
round-trip through SQLite (see :func:`resume_session`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from clarify import Question
from models import (
    AuthorInfo,
    ClarificationTurn,
    MovementSegment,
    NonManualFeatures,
    SignEntry,
    SignLanguage,
    SignParameters,
    SignStatus,
)
from parser import Gap, PartialMovementSegment, PartialSignParameters


# Inactivity after which a session is marked ABANDONED (but still
# retained for the retention window below).
INACTIVITY_TIMEOUT: timedelta = timedelta(hours=24)

# Retention window after last activity. Beyond this, sessions are
# candidates for hard deletion by a cleanup job.
RETENTION_WINDOW: timedelta = timedelta(days=30)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Session phases
# ---------------------------------------------------------------------------


class SessionState(str, Enum):
    """The coarse phase of an authoring session.

    Transitions are restricted — see :mod:`.orchestrator` for the full
    matrix. ``str`` base so the enum value is a stable serialisable
    token across storage round-trips.
    """

    AWAITING_DESCRIPTION = "awaiting_description"
    PARSING = "parsing"
    CLARIFYING = "clarifying"
    GENERATING = "generating"
    RENDERED = "rendered"
    AWAITING_CORRECTION = "awaiting_correction"
    APPLYING_CORRECTION = "applying_correction"
    FINALIZED = "finalized"
    ABANDONED = "abandoned"


TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.FINALIZED, SessionState.ABANDONED}
)


# ---------------------------------------------------------------------------
# Event types — append-only history
# ---------------------------------------------------------------------------


class _EventBase(BaseModel):
    """Common fields every session event carries."""

    model_config = ConfigDict(extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utcnow)


class DescribedEvent(_EventBase):
    """Author supplied (or revised) the prose description."""

    type: Literal["described"] = "described"
    prose: str
    gaps_found: int = 0


class ClarificationAskedEvent(_EventBase):
    """System posed one or more clarification questions to the author."""

    type: Literal["clarification_asked"] = "clarification_asked"
    questions: List[Question] = Field(default_factory=list)


class ClarificationAnsweredEvent(_EventBase):
    """Author answered a previously-posed question.

    ``question_field`` identifies which slot was targeted; ``answer`` is
    the user's raw reply; ``resolved_value`` is the canonical value
    written back into the partial parameters.
    """

    type: Literal["clarification_answered"] = "clarification_answered"
    question_field: str
    answer: str
    resolved_value: str


class GeneratedEvent(_EventBase):
    """Generator ran — either produced HamNoSys+SiGML or reported errors."""

    type: Literal["generated"] = "generated"
    success: bool
    hamnosys: Optional[str] = None
    sigml: Optional[str] = None
    confidence: float = 0.0
    used_llm_fallback: bool = False
    errors: List[str] = Field(default_factory=list)


class CorrectionRequestedEvent(_EventBase):
    """Author flagged something wrong with the rendered preview.

    ``raw_text``/``target_time_ms``/``target_region`` mirror the
    :class:`Correction` shape introduced in Prompt 10; kept as plain
    fields here so the state machine has no compile-time dependency on
    the interpreter module.
    """

    type: Literal["correction_requested"] = "correction_requested"
    raw_text: str
    target_time_ms: Optional[int] = None
    target_region: Optional[str] = None


class CorrectionAppliedEvent(_EventBase):
    """Correction interpreter resolved a correction into a parameter diff."""

    type: Literal["correction_applied"] = "correction_applied"
    summary: str = ""
    field_changes: List[dict] = Field(default_factory=list)


class AcceptedEvent(_EventBase):
    """Author accepted the rendered sign; persisted as a draft SignEntry."""

    type: Literal["accepted"] = "accepted"
    sign_entry_id: UUID
    status: SignStatus = "draft"


class RejectedEvent(_EventBase):
    """Author rejected the rendered sign or cancelled the session."""

    type: Literal["rejected"] = "rejected"
    reason: str = ""


class AbandonedEvent(_EventBase):
    """System marked the session abandoned (usually timeout)."""

    type: Literal["abandoned"] = "abandoned"
    reason: str = "inactivity timeout"


SessionEvent = Annotated[
    Union[
        DescribedEvent,
        ClarificationAskedEvent,
        ClarificationAnsweredEvent,
        GeneratedEvent,
        CorrectionRequestedEvent,
        CorrectionAppliedEvent,
        AcceptedEvent,
        RejectedEvent,
        AbandonedEvent,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Draft — the in-progress sign being authored
# ---------------------------------------------------------------------------


class SignEntryDraft(BaseModel):
    """Progressively-populated mirror of :class:`SignEntry`.

    Every field is optional because the authoring flow populates them
    piecewise. At :data:`SessionState.FINALIZED` the draft is frozen
    into a real :class:`SignEntry` via
    :meth:`SignEntryDraft.to_sign_entry`.

    ``slot_codepoints`` records the per-slot HamNoSys PUA chunks the
    generator emitted — needed at finalization so we can build a
    :class:`SignParameters` whose slots carry codepoints, not
    plain-English terms.
    """

    model_config = ConfigDict(extra="forbid")

    gloss: Optional[str] = None
    sign_language: SignLanguage = "bsl"
    domain: Optional[str] = None
    regional_variant: Optional[str] = None

    description_prose: str = ""
    parameters_partial: Optional[PartialSignParameters] = None
    gaps: List[Gap] = Field(default_factory=list)
    pending_questions: List[Question] = Field(default_factory=list)
    clarifications: List[ClarificationTurn] = Field(default_factory=list)

    hamnosys: Optional[str] = None
    sigml: Optional[str] = None
    video_path: Optional[str] = None
    preview_status: Optional[str] = None
    preview_message: str = ""
    generation_confidence: Optional[float] = None
    generation_errors: List[str] = Field(default_factory=list)

    # Per-slot HamNoSys PUA chunks — populated by on_generate so
    # on_accept can construct a valid SignParameters without re-running
    # the composer.
    slot_codepoints: dict[str, str] = Field(default_factory=dict)

    author_signer_id: str = ""
    author_is_deaf_native: Optional[bool] = None
    author_display_name: Optional[str] = None

    corrections_count: int = 0

    def has_unresolved_gaps(self) -> bool:
        """Is at least one parser gap still unresolved?"""
        return bool(self.gaps)

    def has_pending_questions(self) -> bool:
        """Does at least one question await an answer?"""
        return bool(self.pending_questions)

    def to_sign_entry(self) -> SignEntry:
        """Promote the draft to a validated :class:`SignEntry`.

        Raises :class:`DraftFinalizationError` if the draft is missing
        any of the fields required by :class:`SignEntry` (gloss,
        author, hamnosys, mandatory parameter slots).
        """
        if not self.gloss:
            raise DraftFinalizationError("draft has no gloss; cannot finalize")
        if not self.author_signer_id:
            raise DraftFinalizationError(
                "draft has no author signer id; cannot finalize"
            )
        if not self.hamnosys:
            raise DraftFinalizationError(
                "draft has no hamnosys string; run on_generate first"
            )
        parameters = _build_sign_parameters(self)
        author = AuthorInfo(
            signer_id=self.author_signer_id,
            is_deaf_native=self.author_is_deaf_native,
            display_name=self.author_display_name,
        )
        return SignEntry(
            gloss=self.gloss,
            sign_language=self.sign_language,
            domain=self.domain,
            regional_variant=self.regional_variant,
            hamnosys=self.hamnosys,
            sigml=self.sigml,
            description_prose=self.description_prose,
            clarifications=list(self.clarifications),
            parameters=parameters,
            status="draft",
            author=author,
        )


class DraftFinalizationError(ValueError):
    """Raised when a draft lacks data required to build a SignEntry."""


def _build_sign_parameters(draft: SignEntryDraft) -> SignParameters:
    """Assemble a :class:`SignParameters` from a draft's slot codepoints.

    Relies entirely on ``draft.slot_codepoints``, which the orchestrator
    populates at generate time. The partial ``parameters_partial``
    structure is still read to preserve movement-segment ordering and
    any non-manual features the renderer would consume.
    """
    chunks = draft.slot_codepoints
    partial = draft.parameters_partial or PartialSignParameters()

    required = (
        "handshape_dominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
    )
    missing = [k for k in required if not chunks.get(k)]
    if missing:
        raise DraftFinalizationError(
            "draft slot_codepoints missing mandatory slots: "
            + ", ".join(missing)
        )

    movement: list[MovementSegment] = []
    for i, seg in enumerate(partial.movement or []):
        path = chunks.get(f"movement[{i}].path")
        if not path:
            raise DraftFinalizationError(
                f"draft slot_codepoints missing movement[{i}].path"
            )
        movement.append(
            MovementSegment(
                path=path,
                size_mod=chunks.get(f"movement[{i}].size_mod") or None,
                speed_mod=chunks.get(f"movement[{i}].speed_mod") or None,
                repeat=chunks.get(f"movement[{i}].repeat") or None,
            )
        )

    non_manual: Optional[NonManualFeatures] = None
    pnm = partial.non_manual
    if pnm is not None and any(
        getattr(pnm, k)
        for k in (
            "mouth_picture",
            "eye_gaze",
            "head_movement",
            "eyebrows",
            "facial_expression",
        )
    ):
        non_manual = NonManualFeatures(
            mouth_picture=pnm.mouth_picture,
            eye_gaze=pnm.eye_gaze,
            head_movement=pnm.head_movement,
            eyebrows=pnm.eyebrows,
            facial_expression=pnm.facial_expression,
        )

    return SignParameters(
        handshape_dominant=chunks["handshape_dominant"],
        handshape_nondominant=chunks.get("handshape_nondominant") or None,
        orientation_extended_finger=chunks["orientation_extended_finger"],
        orientation_palm=chunks["orientation_palm"],
        location=chunks["location"],
        contact=chunks.get("contact") or None,
        movement=movement,
        non_manual=non_manual,
    )


# ---------------------------------------------------------------------------
# The session itself
# ---------------------------------------------------------------------------


class AuthoringSession(BaseModel):
    """One prose-to-HamNoSys authoring dialogue.

    Pydantic-serialisable end-to-end so the storage layer can round-trip
    the entire session through a single SQLite BLOB.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    state: SessionState = SessionState.AWAITING_DESCRIPTION
    draft: SignEntryDraft = Field(default_factory=SignEntryDraft)
    history: List[SessionEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    last_activity_at: datetime = Field(default_factory=_utcnow)

    def with_state(self, new_state: SessionState) -> "AuthoringSession":
        """Return a copy of the session with ``state`` replaced."""
        return self.model_copy(
            update={
                "state": new_state,
                "last_activity_at": _utcnow(),
            }
        )

    def with_draft(self, **updates: Any) -> "AuthoringSession":
        """Return a copy with the draft's fields replaced.

        ``updates`` is passed directly to
        :meth:`SignEntryDraft.model_copy`; keys must match existing
        draft field names.
        """
        new_draft = self.draft.model_copy(update=updates)
        return self.model_copy(
            update={
                "draft": new_draft,
                "last_activity_at": _utcnow(),
            }
        )

    def append_event(self, event: Any) -> "AuthoringSession":
        """Return a copy with ``event`` appended to the history.

        ``event`` is any of the :data:`SessionEvent` union members.
        """
        new_history = list(self.history) + [event]
        return self.model_copy(
            update={
                "history": new_history,
                "last_activity_at": _utcnow(),
            }
        )

    def is_stale(
        self,
        *,
        now: datetime | None = None,
        timeout: timedelta = INACTIVITY_TIMEOUT,
    ) -> bool:
        """Has the session exceeded the inactivity timeout?

        Terminal sessions (FINALIZED / ABANDONED) never count as stale —
        staleness only drives the abandonment transition.
        """
        if self.state in TERMINAL_STATES:
            return False
        current = now if now is not None else _utcnow()
        return (current - self.last_activity_at) >= timeout

    def is_expired(
        self,
        *,
        now: datetime | None = None,
        retention: timedelta = RETENTION_WINDOW,
    ) -> bool:
        """Has the session passed the 30-day retention window?

        Used by the cleanup job to pick sessions safe to hard-delete.
        """
        current = now if now is not None else _utcnow()
        return (current - self.last_activity_at) >= retention


__all__ = [
    "AbandonedEvent",
    "AcceptedEvent",
    "AuthoringSession",
    "ClarificationAnsweredEvent",
    "ClarificationAskedEvent",
    "CorrectionAppliedEvent",
    "CorrectionRequestedEvent",
    "DescribedEvent",
    "DraftFinalizationError",
    "GeneratedEvent",
    "INACTIVITY_TIMEOUT",
    "RETENTION_WINDOW",
    "RejectedEvent",
    "SessionEvent",
    "SessionState",
    "SignEntryDraft",
    "TERMINAL_STATES",
]
