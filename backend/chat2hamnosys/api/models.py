"""Request / response Pydantic models for the chat2hamnosys HTTP surface.

Every endpoint in :mod:`.router` takes a typed body and returns a typed
:class:`SessionEnvelope` or a simpler specialised model. OpenAPI is
generated straight from these shapes, so anything a frontend needs to
know about the protocol is visible at ``/docs`` without reading router
code.

Payload shapes are intentionally loose on nested parsers — we serialize
the parser / clarifier models via ``model_dump(mode="json")`` rather
than re-declaring them here, so the authoritative schema stays in the
backend modules they belong to.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Body for ``POST /sessions`` (all fields optional)."""

    model_config = ConfigDict(extra="forbid")

    signer_id: Optional[str] = None
    display_name: Optional[str] = None
    author_is_deaf_native: Optional[bool] = None
    sign_language: Literal["bsl", "asl", "dgs"] = "bsl"
    regional_variant: Optional[str] = None
    domain: Optional[str] = None
    gloss: Optional[str] = None


class DescribeRequest(BaseModel):
    """Body for ``POST /sessions/{id}/describe``."""

    model_config = ConfigDict(extra="forbid")

    prose: str = Field(min_length=1)
    gloss: Optional[str] = None


class AnswerRequest(BaseModel):
    """Body for ``POST /sessions/{id}/answer``.

    ``question_id`` is the question's slot ``field`` (see
    :class:`clarify.Question`) — questions are identified by the slot
    they target, so no separate id field is needed.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer: Union[str, int]


class CorrectRequest(BaseModel):
    """Body for ``POST /sessions/{id}/correct``."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)
    target_time_ms: Optional[int] = None
    target_region: Optional[str] = None


class RejectRequest(BaseModel):
    """Body for ``POST /sessions/{id}/reject``."""

    model_config = ConfigDict(extra="forbid")

    reason: str = ""


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


class OptionOut(BaseModel):
    """Mirror of :class:`clarify.Option` for transport."""

    model_config = ConfigDict(extra="forbid")

    label: str
    value: str


class QuestionOut(BaseModel):
    """Mirror of :class:`clarify.Question` for transport."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    field: str
    text: str
    options: Optional[List[OptionOut]] = None
    allow_freeform: bool = True
    rationale: str = ""


class GapOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    reason: str
    suggested_question: str


class PreviewOut(BaseModel):
    """Preview metadata for ``GET /sessions/{id}/preview``."""

    model_config = ConfigDict(extra="forbid")

    status: Optional[str] = None
    message: str = ""
    video_url: Optional[str] = None
    sigml: Optional[str] = None
    hamnosys: Optional[str] = None


class NextAction(BaseModel):
    """What the frontend should do next with this session."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "await_description",
        "answer_questions",
        "await_generation",
        "preview_ready",
        "await_correction",
        "finalized",
        "abandoned",
    ]
    questions: List[QuestionOut] = Field(default_factory=list)
    preview: Optional[PreviewOut] = None


class SessionEnvelope(BaseModel):
    """The session-state payload returned by most endpoints."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    state: str
    gloss: Optional[str] = None
    sign_language: str = "bsl"
    domain: Optional[str] = None
    regional_variant: Optional[str] = None

    description_prose: str = ""
    parameters: Optional[dict[str, Any]] = None
    gaps: List[GapOut] = Field(default_factory=list)
    pending_questions: List[QuestionOut] = Field(default_factory=list)
    clarifications: List[dict[str, Any]] = Field(default_factory=list)

    hamnosys: Optional[str] = None
    sigml: Optional[str] = None
    preview: PreviewOut = Field(default_factory=PreviewOut)

    generation_confidence: Optional[float] = None
    generation_errors: List[str] = Field(default_factory=list)
    corrections_count: int = 0

    history: List[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    last_activity_at: datetime

    next_action: NextAction


class CreateSessionResponse(BaseModel):
    """Body of ``POST /sessions``."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    state: str
    session_token: str
    session: SessionEnvelope


class SignEntryOut(BaseModel):
    """Finalized sign entry — returned by ``POST /sessions/{id}/accept``."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    gloss: str
    sign_language: str
    domain: Optional[str] = None
    hamnosys: str
    sigml: Optional[str] = None
    status: str
    parameters: dict[str, Any]
    regional_variant: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AcceptResponse(BaseModel):
    """Body of ``POST /sessions/{id}/accept``."""

    model_config = ConfigDict(extra="forbid")

    sign_entry: SignEntryOut
    session: SessionEnvelope


__all__ = [
    "AcceptResponse",
    "AnswerRequest",
    "CorrectRequest",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "DescribeRequest",
    "GapOut",
    "NextAction",
    "OptionOut",
    "PreviewOut",
    "QuestionOut",
    "RejectRequest",
    "SessionEnvelope",
    "SignEntryOut",
]
