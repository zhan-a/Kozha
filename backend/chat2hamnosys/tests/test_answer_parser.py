"""Tests for ``backend.chat2hamnosys.clarify.answer_parser``.

Covers all three answer shapes the module handles — ordinal word
references, numeric indices, and freeform text — plus the field-path
routing into :class:`PartialSignParameters` (top-level scalars,
``movement``, nested ``non_manual.*``).
"""

from __future__ import annotations

import pytest

from clarify import AnswerParseError, Option, Question, apply_answer, template_for
from parser import (
    PartialMovementSegment,
    PartialNonManualFeatures,
    PartialSignParameters,
)


def _palm_question() -> Question:
    return Question(
        field="orientation_palm",
        text="Which way does the palm face?",
        options=[
            Option(label="Down (toward the floor)", value="down"),
            Option(label="Up (toward the ceiling)", value="up"),
            Option(label="Toward signer", value="toward signer"),
            Option(label="Away from signer", value="away from signer"),
        ],
        allow_freeform=True,
        rationale="discrete",
    )


def _brow_question_strict() -> Question:
    """Options-only; no freeform escape hatch."""
    return Question(
        field="non_manual.eyebrows",
        text="Eyebrows?",
        options=[
            Option(label="Raised", value="raised"),
            Option(label="Furrowed", value="furrowed"),
            Option(label="Neutral", value="neutral"),
        ],
        allow_freeform=False,
        rationale="small closed class",
    )


def _freeform_location_question() -> Question:
    return Question(
        field="location",
        text="Where on the body?",
        options=None,
        allow_freeform=True,
        rationale="large vocabulary",
    )


# ---------------------------------------------------------------------------
# Ordinal / index / label resolution
# ---------------------------------------------------------------------------


class TestOrdinalWords:
    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("first", "down"),
            ("the first", "down"),
            ("the first one", "down"),
            ("first option", "down"),
            ("second", "up"),
            ("the second option", "up"),
            ("2nd", "up"),
            ("third", "toward signer"),
            ("fourth", "away from signer"),
        ],
    )
    def test_resolves(self, answer: str, expected: str) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _palm_question(), answer)
        assert r.orientation_palm == expected

    def test_freeform_containing_ordinal_word_does_not_misfire(self) -> None:
        # "bent-five claw" contains "five" but should go to freeform,
        # not be interpreted as "option 5".
        q = Question(
            field="handshape_dominant",
            text="Q",
            options=[
                Option(label="Fist", value="fist"),
                Option(label="Flat", value="flat"),
            ],
            allow_freeform=True,
            rationale="",
        )
        p = PartialSignParameters()
        r = apply_answer(p, q, "bent-five claw")
        assert r.handshape_dominant == "bent-five claw"


class TestIndexPatterns:
    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("1", "down"),
            ("2", "up"),
            ("#2", "up"),
            ("# 2", "up"),
            ("option 2", "up"),
            ("Option 2", "up"),
            ("choice 3", "toward signer"),
            ("number 4", "away from signer"),
        ],
    )
    def test_resolves(self, answer: str, expected: str) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _palm_question(), answer)
        assert r.orientation_palm == expected

    def test_out_of_range_raises(self) -> None:
        p = PartialSignParameters()
        with pytest.raises(AnswerParseError, match=r"option #99"):
            apply_answer(p, _palm_question(), "99")

    def test_zero_out_of_range(self) -> None:
        p = PartialSignParameters()
        with pytest.raises(AnswerParseError, match=r"option #0"):
            apply_answer(p, _palm_question(), "0")

    def test_no_options_raises(self) -> None:
        p = PartialSignParameters()
        with pytest.raises(AnswerParseError, match="no options"):
            apply_answer(p, _freeform_location_question(), "2")


class TestLabelAndValueMatch:
    def test_exact_value_match(self) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _palm_question(), "down")
        assert r.orientation_palm == "down"

    def test_case_insensitive_value(self) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _palm_question(), "DOWN")
        assert r.orientation_palm == "down"

    def test_exact_label_match(self) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _palm_question(), "Down (toward the floor)")
        assert r.orientation_palm == "down"


class TestFreeform:
    def test_allows_freeform_phrase(self) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _palm_question(), "slightly rotated upward and out")
        assert r.orientation_palm == "slightly rotated upward and out"

    def test_no_freeform_unmapped_raises(self) -> None:
        p = PartialSignParameters()
        with pytest.raises(AnswerParseError, match="could not map"):
            apply_answer(p, _brow_question_strict(), "purple")

    def test_no_freeform_mapped_still_works(self) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _brow_question_strict(), "raised")
        assert p.non_manual is None  # original not mutated
        assert r.non_manual is not None
        assert r.non_manual.eyebrows == "raised"

    def test_empty_answer_raises(self) -> None:
        p = PartialSignParameters()
        with pytest.raises(AnswerParseError, match="non-empty"):
            apply_answer(p, _palm_question(), "")
        with pytest.raises(AnswerParseError, match="non-empty"):
            apply_answer(p, _palm_question(), "   ")

    def test_freeform_trims_whitespace(self) -> None:
        p = PartialSignParameters()
        r = apply_answer(p, _freeform_location_question(), "  under the chin  ")
        assert r.location == "under the chin"


# ---------------------------------------------------------------------------
# Field-path routing
# ---------------------------------------------------------------------------


class TestFieldRouting:
    def test_top_level_scalar(self) -> None:
        p = PartialSignParameters()
        q = Question(field="handshape_dominant", text="?", allow_freeform=True)
        r = apply_answer(p, q, "flat")
        assert r.handshape_dominant == "flat"

    def test_movement_appends_when_empty(self) -> None:
        p = PartialSignParameters()
        q = template_for("movement")
        r = apply_answer(p, q, "arc")
        assert len(r.movement) == 1
        assert r.movement[0].path == "arc"

    def test_movement_updates_first_when_present(self) -> None:
        p = PartialSignParameters(
            movement=[PartialMovementSegment(path="unknown", size_mod="small")]
        )
        q = template_for("movement")
        r = apply_answer(p, q, "circular")
        assert len(r.movement) == 1
        assert r.movement[0].path == "circular"
        # Other modifiers on the existing segment are preserved.
        assert r.movement[0].size_mod == "small"

    def test_non_manual_creates_when_none(self) -> None:
        p = PartialSignParameters()
        q = template_for("non_manual.eyebrows")
        r = apply_answer(p, q, "raised")
        assert r.non_manual is not None
        assert r.non_manual.eyebrows == "raised"

    def test_non_manual_updates_when_present(self) -> None:
        p = PartialSignParameters(
            non_manual=PartialNonManualFeatures(mouth_picture="open")
        )
        q = template_for("non_manual.eyebrows")
        r = apply_answer(p, q, "raised")
        assert r.non_manual is not None
        assert r.non_manual.eyebrows == "raised"
        # Prior mouth_picture must not be lost.
        assert r.non_manual.mouth_picture == "open"

    def test_unknown_field_raises(self) -> None:
        p = PartialSignParameters()
        q = Question(field="made_up", text="?", allow_freeform=True)
        with pytest.raises(AnswerParseError, match="unknown field path"):
            apply_answer(p, q, "something")

    def test_unknown_non_manual_sub_field_raises(self) -> None:
        p = PartialSignParameters()
        q = Question(field="non_manual.nope", text="?", allow_freeform=True)
        with pytest.raises(AnswerParseError, match="unknown non_manual"):
            apply_answer(p, q, "something")


class TestImmutability:
    def test_input_not_mutated_on_success(self) -> None:
        p = PartialSignParameters(handshape_dominant="initial")
        q = _palm_question()
        r = apply_answer(p, q, "down")
        assert p.orientation_palm is None
        assert p.handshape_dominant == "initial"
        assert r.orientation_palm == "down"
        assert r.handshape_dominant == "initial"

    def test_input_not_mutated_on_raise(self) -> None:
        p = PartialSignParameters(orientation_palm="initial")
        q = _brow_question_strict()
        with pytest.raises(AnswerParseError):
            apply_answer(p, q, "purple")
        assert p.orientation_palm == "initial"
