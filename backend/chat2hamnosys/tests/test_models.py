"""Tests for ``chat2hamnosys.models``.

Covers schema validation failures (one per slot) plus the round-trip
serialization invariant: ``SignEntry`` → JSON → ``SignEntry`` is the
identity, including UUID, datetimes, and nested phonology.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

import pytest
from pydantic import ValidationError as PydanticValidationError

from models import (
    AuthorInfo,
    ClarificationTurn,
    MovementSegment,
    NonManualFeatures,
    ReviewRecord,
    SignEntry,
    SignParameters,
)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_json_preserves_all_fields(valid_entry: SignEntry) -> None:
    valid_entry.clarifications = [
        ClarificationTurn(role="author", text="It's a chemistry sign for the orbital."),
        ClarificationTurn(role="assistant", text="Two-handed, both flat?"),
        ClarificationTurn(role="author", text="Yes, dominant moves clockwise."),
    ]
    valid_entry.reviewers = [
        ReviewRecord(
            reviewer_id="bsl-mod-7",
            is_deaf_native=True,
            verdict="approved",
            notes="Matches BANZSL convention.",
        )
    ]
    valid_entry.parameters.non_manual = NonManualFeatures(
        mouth_picture="V", facial_expression="neutral"
    )
    valid_entry.regional_variant = "BSL-London"

    payload = valid_entry.model_dump_json_safe()
    text = json.dumps(payload)
    restored = SignEntry.model_validate_json(text)

    assert restored.id == valid_entry.id
    assert restored.gloss == valid_entry.gloss
    assert restored.hamnosys == valid_entry.hamnosys
    assert restored.parameters == valid_entry.parameters
    assert restored.clarifications == valid_entry.clarifications
    assert restored.reviewers == valid_entry.reviewers
    assert restored.regional_variant == "BSL-London"
    assert restored.created_at == valid_entry.created_at
    assert restored.updated_at == valid_entry.updated_at


def test_default_factories_produce_distinct_uuid_and_timezone_aware_timestamps(
    valid_entry_factory: Callable[..., SignEntry],
) -> None:
    a = valid_entry_factory()
    b = valid_entry_factory()
    assert a.id != b.id
    assert a.created_at.tzinfo is timezone.utc
    assert a.updated_at >= a.created_at


# ---------------------------------------------------------------------------
# Schema validation failures
# ---------------------------------------------------------------------------


def test_blank_gloss_rejected(valid_entry_factory: Callable[..., SignEntry]) -> None:
    with pytest.raises(PydanticValidationError, match="gloss"):
        valid_entry_factory(gloss="   ")


def test_xml_unsafe_gloss_rejected(valid_entry_factory: Callable[..., SignEntry]) -> None:
    with pytest.raises(PydanticValidationError, match="may not contain"):
        valid_entry_factory(gloss='ELECTRON<script>')


def test_invalid_hamnosys_rejected(valid_entry_factory: Callable[..., SignEntry]) -> None:
    # Latin "X" is not a HamNoSys 4.0 codepoint.
    with pytest.raises(PydanticValidationError, match="HamNoSys 4.0 validation"):
        valid_entry_factory(hamnosys="XX")


def test_handshape_must_be_handshape_base_class(
    valid_entry_factory: Callable[..., SignEntry],
) -> None:
    # U+E020 = hamextfingeru — orientation, not a handshape base.
    bad_params = dict(
        handshape_dominant="\uE020",
        orientation_extended_finger="\uE020",
        orientation_palm="\uE03C",
        location="\uE040",
    )
    with pytest.raises(PydanticValidationError, match="expected one of"):
        SignParameters(**bad_params)


def test_movement_path_must_be_movement_class() -> None:
    with pytest.raises(PydanticValidationError):
        # U+E001 = hamflathand (handshape, not movement).
        MovementSegment(path="\uE001")


def test_unknown_status_rejected(valid_entry_factory: Callable[..., SignEntry]) -> None:
    with pytest.raises(PydanticValidationError):
        valid_entry_factory(status="archived")


def test_extra_fields_forbidden(valid_entry_factory: Callable[..., SignEntry]) -> None:
    # Pydantic v2 with ``extra="forbid"`` rejects unknown keys at construction.
    with pytest.raises(PydanticValidationError, match="Extra inputs"):
        valid_entry_factory(unknown_field=42)


def test_assignment_revalidates(valid_entry: SignEntry) -> None:
    # ``validate_assignment=True`` should re-run field validators on set.
    with pytest.raises(PydanticValidationError):
        valid_entry.hamnosys = "not-hamnosys"


# ---------------------------------------------------------------------------
# Phonological slot acceptance
# ---------------------------------------------------------------------------


def test_handshape_accepts_compound_with_thumb_modifier() -> None:
    # hamflathand + hamthumboutmod is a perfectly normal compound shape.
    p = SignParameters(
        handshape_dominant="\uE001\uE00C",
        orientation_extended_finger="\uE020",
        orientation_palm="\uE03C",
        location="\uE040",
    )
    assert p.handshape_dominant == "\uE001\uE00C"


def test_two_handed_sign_accepts_nondominant() -> None:
    p = SignParameters(
        handshape_dominant="\uE001",
        handshape_nondominant="\uE000",
        orientation_extended_finger="\uE020",
        orientation_palm="\uE03C",
        location="\uE040",
    )
    assert p.handshape_nondominant == "\uE000"
