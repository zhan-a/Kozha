"""Parser-side Pydantic models for prose → phonological-features extraction.

These models mirror the *shape* of :class:`backend.chat2hamnosys.models.SignParameters`
but deliberately hold **plain-English vocabulary**, not HamNoSys PUA codepoints.
The parser's job is to surface what a natural-language description asserts
about each phonological slot; a later step maps the English vocabulary to
actual HamNoSys symbols.

- ``PartialSignParameters``: every slot is optional so the LLM can leave
  fields ``null`` when the prose is silent or ambiguous.
- ``Gap``: a machine-readable record of a slot the parser could not fill
  with confidence, plus a reviewer-facing follow-up question.
- ``ParseResult``: wraps the parsed parameters, the gap list, and the raw
  LLM output (kept for debugging and for writing fixture recordings).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Slot names we permit in a ``Gap.field``. Kept as a module-level tuple so
# the system prompt and the validator agree on the universe of slot names.
ALLOWED_GAP_FIELDS: tuple[str, ...] = (
    "handshape_dominant",
    "handshape_nondominant",
    "orientation_extended_finger",
    "orientation_palm",
    "location",
    "contact",
    "movement",
    "non_manual.mouth_picture",
    "non_manual.eye_gaze",
    "non_manual.head_movement",
    "non_manual.eyebrows",
    "non_manual.facial_expression",
)


class PartialMovementSegment(BaseModel):
    """One descriptive movement step.

    All fields accept plain-English phrases (e.g. ``path="small circular"``,
    ``size_mod="small"``). A later mapping step converts these to HamNoSys
    movement codepoints.
    """

    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    size_mod: Optional[str] = None
    speed_mod: Optional[str] = None
    repeat: Optional[str] = None


class PartialNonManualFeatures(BaseModel):
    """Non-manual (face / head / body) features described in prose."""

    model_config = ConfigDict(extra="forbid")

    mouth_picture: Optional[str] = None
    eye_gaze: Optional[str] = None
    head_movement: Optional[str] = None
    eyebrows: Optional[str] = None
    facial_expression: Optional[str] = None


class PartialSignParameters(BaseModel):
    """Optional-valued mirror of :class:`SignParameters` for parser output.

    Unlike ``SignParameters`` (which enforces HamNoSys PUA codepoints per
    slot), every field here is a plain-English phrase or ``None``. The LLM
    must leave a slot ``None`` rather than invent a codepoint.
    """

    model_config = ConfigDict(extra="forbid")

    handshape_dominant: Optional[str] = None
    handshape_nondominant: Optional[str] = None
    orientation_extended_finger: Optional[str] = None
    orientation_palm: Optional[str] = None
    location: Optional[str] = None
    contact: Optional[str] = None
    movement: List[PartialMovementSegment] = Field(default_factory=list)
    non_manual: Optional[PartialNonManualFeatures] = None


class Gap(BaseModel):
    """A slot the parser could not fill with confidence.

    ``field`` names the slot (dot-path for nested fields, e.g.
    ``"non_manual.eyebrows"``), ``reason`` explains briefly why the parser
    left it empty, and ``suggested_question`` is a follow-up the clarifier
    layer can ask the author in the next turn.
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    suggested_question: str = Field(min_length=1)


class ParseResult(BaseModel):
    """Outcome of :func:`parse_description`.

    ``raw_response`` is the unparsed JSON string the LLM returned — kept so
    test fixtures can record exact byte-level replies and so debugging a
    schema-conformance failure doesn't require re-running the API call.
    """

    model_config = ConfigDict(extra="forbid")

    parameters: PartialSignParameters
    gaps: List[Gap] = Field(default_factory=list)
    raw_response: str = ""
