"""Authored-sign data contract.

Pydantic v2 models for the chat2hamnosys authoring loop. ``SignEntry`` is the
canonical record produced by an LLM-mediated dialogue with a domain expert;
``SignParameters`` is its phonological decomposition; ``ClarificationTurn``
and ``ReviewRecord`` capture the dialogue and review provenance.

The ``hamnosys`` field is validated against the bundled HamNoSys 4.0 grammar
(:mod:`backend.chat2hamnosys.hamnosys`); the ``sigml`` field is derived from
``hamnosys`` at export time and never treated as primary truth. Phonological
slot strings are validated against the symbol inventory but only loosely
typed (multi-codepoint compounds such as flathand+thumbmod are allowed).

Conventions worth knowing before adding new fields:

- ``id`` is a stable UUIDv4 generated client-side; a re-``put`` overwrites.
- ``created_at`` / ``updated_at`` are timezone-aware UTC; the storage layer
  refreshes ``updated_at`` on mutation.
- ``status`` gates downstream behavior — only ``"validated"`` entries may be
  exported into the live Kozha sign library (see ``storage.py``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hamnosys import SYMBOLS, SymClass, classify, normalize, validate


SignLanguage = Literal["bsl", "asl", "dgs"]
SignStatus = Literal[
    "draft",
    "pending_review",
    "validated",
    "rejected",
    "quarantined",
]
ReviewVerdict = Literal[
    "approved",
    "changes_requested",
    "rejected",
    "flagged",
]
RejectionCategory = Literal[
    "inaccurate",
    "culturally_inappropriate",
    "regional_mismatch",
    "poor_quality",
    "other",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_pua_chunk(value: str, *, allowed_classes: set[SymClass] | None) -> str:
    """Check that ``value`` is one or more known HamNoSys codepoints.

    If ``allowed_classes`` is given, the *first* codepoint must belong to one
    of those classes — modifiers may follow. ``value`` must be non-empty.
    """
    if not value:
        raise ValueError("must be a non-empty HamNoSys codepoint sequence")
    for i, ch in enumerate(value):
        if ord(ch) not in SYMBOLS:
            raise ValueError(
                f"position {i}: U+{ord(ch):04X} is not a known HamNoSys 4.0 symbol"
            )
    if allowed_classes is not None:
        head_cls = classify(value[0])
        if head_cls not in allowed_classes:
            allowed = ", ".join(sorted(c.value for c in allowed_classes))
            raise ValueError(
                f"first symbol U+{ord(value[0]):04X} has class "
                f"{head_cls.value if head_cls else '?'}; expected one of: {allowed}"
            )
    return value


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


class ClarificationTurn(BaseModel):
    """One exchange in the authoring dialogue."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["author", "assistant", "reviewer"]
    text: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=_utcnow)


class ReviewRecord(BaseModel):
    """A single reviewer's verdict on a sign entry.

    The append-only nature of ``SignEntry.reviewers`` means every action a
    reviewer takes — approve, reject, request revision, or flag — is
    preserved. The Deaf governance board can audit who said what and when.

    ``allow_non_native`` records the explicit override flag a non-native
    reviewer must supply when approving (with justification in ``comment``);
    it is ``None`` for native reviewers and never silently true.
    """

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(min_length=1)
    is_deaf_native: bool | None = None
    verdict: ReviewVerdict
    notes: str = ""
    comment: str = ""
    category: RejectionCategory | None = None
    regional_background: str | None = None
    signs: list[str] = Field(default_factory=list)
    allow_non_native: bool | None = None
    fields_to_revise: list[str] = Field(default_factory=list)
    reviewed_at: datetime = Field(default_factory=_utcnow)
    # Prompt 6: the reviewer's language match against the entry's target
    # sign_language. True when the reviewer self-identifies as a native
    # signer of this entry's sign_language (i.e. entry.sign_language is
    # present in reviewer.signs). None when the record predates prompt 6
    # or the information wasn't captured at review time. The live
    # metadata export uses this to set reviewer_language_match on the
    # per-sign ``.meta.json`` override.
    reviewer_language_match: bool | None = None


class AuthorInfo(BaseModel):
    """Authoring provenance, optionally anonymized."""

    model_config = ConfigDict(extra="forbid")

    signer_id: str = Field(min_length=1)
    is_deaf_native: bool | None = None
    display_name: str | None = None


class MovementSegment(BaseModel):
    """One movement step within a sign.

    ``path`` is a HamNoSys movement codepoint (e.g. ``\\uE089`` =
    ``hammoveo``). ``size_mod`` and ``speed_mod`` accept their respective
    HamNoSys modifier codepoints. ``repeat`` accepts a HamNoSys repeat marker
    such as ``hamrepeatfromstart`` (U+E0D8).
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    size_mod: str | None = None
    speed_mod: str | None = None
    repeat: str | None = None

    @field_validator("path")
    @classmethod
    def _check_path(cls, v: str) -> str:
        return _validate_pua_chunk(
            v,
            allowed_classes={
                SymClass.MOVE_STRAIGHT,
                SymClass.MOVE_CIRCLE,
                SymClass.MOVE_ACTION,
                SymClass.MOVE_CLOCK,
                SymClass.MOVE_ARC,
                SymClass.MOVE_WAVY,
                SymClass.MOVE_ELLIPSE,
            },
        )

    @field_validator("size_mod")
    @classmethod
    def _check_size_mod(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_pua_chunk(v, allowed_classes={SymClass.SIZE_MOD})

    @field_validator("speed_mod")
    @classmethod
    def _check_speed_mod(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_pua_chunk(v, allowed_classes={SymClass.SPEED_MOD})

    @field_validator("repeat")
    @classmethod
    def _check_repeat(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_pua_chunk(v, allowed_classes={SymClass.REPEAT})


class NonManualFeatures(BaseModel):
    """Optional facial / body features carried alongside the manual stream.

    ``mouth_picture`` is the SAMPA-ish lip-shape string the SiGML renderer
    consumes via ``<hnm_mouthpicture picture="…"/>``. The other slots are
    free-text descriptions today — they're not yet tag-encoded — but exist
    so the dialogue layer has somewhere to store them without losing data.
    """

    model_config = ConfigDict(extra="forbid")

    mouth_picture: str | None = None
    eye_gaze: str | None = None
    head_movement: str | None = None
    eyebrows: str | None = None
    facial_expression: str | None = None


class SignParameters(BaseModel):
    """Explicit phonological slots for a sign.

    These slots mirror the canonical HamNoSys ordering: handshape →
    orientation → location → contact → movement, with optional non-manual
    overlay. Each slot is a HamNoSys PUA string; the full ``hamnosys`` field
    on :class:`SignEntry` is the assembled rendering.
    """

    model_config = ConfigDict(extra="forbid")

    handshape_dominant: str
    handshape_nondominant: str | None = None
    orientation_extended_finger: str
    orientation_palm: str
    location: str
    contact: str | None = None
    movement: list[MovementSegment] = Field(default_factory=list)
    non_manual: NonManualFeatures | None = None

    @field_validator("handshape_dominant", "handshape_nondominant")
    @classmethod
    def _check_handshape(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_pua_chunk(v, allowed_classes={SymClass.HANDSHAPE_BASE})

    @field_validator("orientation_extended_finger")
    @classmethod
    def _check_ext_finger(cls, v: str) -> str:
        return _validate_pua_chunk(v, allowed_classes={SymClass.EXT_FINGER_DIR})

    @field_validator("orientation_palm")
    @classmethod
    def _check_palm(cls, v: str) -> str:
        return _validate_pua_chunk(v, allowed_classes={SymClass.PALM_DIR})

    @field_validator("location")
    @classmethod
    def _check_location(cls, v: str) -> str:
        return _validate_pua_chunk(
            v,
            allowed_classes={
                SymClass.LOC_HEAD,
                SymClass.LOC_TORSO,
                SymClass.LOC_NEUTRAL,
                SymClass.LOC_ARM,
                SymClass.LOC_HAND_ZONE,
                SymClass.LOC_FINGER,
            },
        )

    @field_validator("contact")
    @classmethod
    def _check_contact(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_pua_chunk(v, allowed_classes={SymClass.CONTACT})


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


class SignEntry(BaseModel):
    """The canonical authored-sign record.

    A ``SignEntry`` is what the chat2hamnosys authoring loop produces and what
    the storage layer round-trips. Validation happens in three places: field
    validators on the phonological slots, the ``hamnosys`` field validator
    that runs the bundled HamNoSys 4.0 grammar, and the storage layer's
    status gate on export.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)
    gloss: str = Field(min_length=1, max_length=128)
    sign_language: SignLanguage = "bsl"
    domain: str | None = None

    hamnosys: str = Field(min_length=1)
    sigml: str | None = None

    description_prose: str = ""
    clarifications: list[ClarificationTurn] = Field(default_factory=list)
    parameters: SignParameters

    status: SignStatus = "draft"
    author: AuthorInfo
    reviewers: list[ReviewRecord] = Field(default_factory=list)
    regional_variant: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("gloss")
    @classmethod
    def _check_gloss(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("gloss must not be blank")
        # Keep gloss safe to embed as an XML attribute later — reject the
        # punctuation that would force an escape we'd rather not generate.
        for bad in ("<", ">", "&", '"'):
            if bad in v:
                raise ValueError(f"gloss may not contain {bad!r}")
        return v

    @field_validator("hamnosys")
    @classmethod
    def _check_hamnosys(cls, v: str) -> str:
        v = normalize(v)
        result = validate(v)
        if not result.ok:
            joined = "; ".join(str(e) for e in result.errors)
            raise ValueError(f"hamnosys failed HamNoSys 4.0 validation: {joined}")
        return v

    @model_validator(mode="after")
    def _touch_updated_at(self) -> "SignEntry":
        # ``created_at`` defaulted before ``updated_at`` did — when both
        # default-fire on construction the timestamps can be a few microseconds
        # apart. Pin updated_at >= created_at on construction to keep the
        # storage layer's "updated_at increases monotonically" check honest.
        if self.updated_at < self.created_at:
            object.__setattr__(self, "updated_at", self.created_at)
        return self

    def model_dump_json_safe(self) -> dict[str, Any]:
        """JSON-serialisable dict with ISO datetimes and string UUID."""
        return self.model_dump(mode="json")
