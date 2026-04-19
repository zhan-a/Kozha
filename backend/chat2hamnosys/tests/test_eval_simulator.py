"""Tests for ``eval.simulator.SimulatedAnswerer``.

The simulator turns a :class:`clarify.Question` + the fixture's
``expected_parameters`` into a plausible user answer. These tests cover
the three branches (option-by-value, option-by-label, freeform
fallback) plus the explicit :class:`NoAnswerAvailable` path.
"""

from __future__ import annotations

import pytest

from clarify import Option, Question
from eval.simulator import NoAnswerAvailable, SimulatedAnswerer


@pytest.fixture
def expected_params() -> dict:
    return {
        "handshape_dominant": "flat",
        "orientation_palm": "away-from-signer",
        "location": "temple",
        "movement": [{"path": "arc", "size_mod": "small"}],
        "non_manual": {"eye_gaze": "forward"},
    }


def test_answers_by_option_value(expected_params: dict) -> None:
    q = Question(
        field="handshape_dominant",
        text="Which handshape is the dominant hand using?",
        options=[
            Option(label="Fist (A)", value="fist"),
            Option(label="Flat hand (B)", value="flat"),
        ],
        allow_freeform=False,
    )
    simulator = SimulatedAnswerer(expected_params)
    assert simulator.answer(q) == "flat"


def test_answers_by_option_label_when_value_differs(
    expected_params: dict,
) -> None:
    q = Question(
        field="orientation_palm",
        text="Which way does the palm face?",
        options=[
            Option(label="away_from_signer", value="AWAY"),
            Option(label="toward_signer", value="TOWARD"),
        ],
        allow_freeform=False,
    )
    simulator = SimulatedAnswerer(expected_params)
    # Value doesn't match any option's value, but the label matches the
    # expected term. The simulator returns the option's value.
    assert simulator.answer(q) == "AWAY"


def test_movement_field_drills_into_first_segment(
    expected_params: dict,
) -> None:
    q = Question(
        field="movement",
        text="What path?",
        options=[Option(label="arc", value="arc"), Option(label="line", value="line")],
        allow_freeform=False,
    )
    simulator = SimulatedAnswerer(expected_params)
    assert simulator.answer(q) == "arc"


def test_non_manual_dotted_field(expected_params: dict) -> None:
    q = Question(
        field="non_manual.eye_gaze",
        text="Which way do the eyes look?",
        options=None,
        allow_freeform=True,
    )
    simulator = SimulatedAnswerer(expected_params)
    assert simulator.answer(q) == "forward"


def test_freeform_fallback_when_options_miss() -> None:
    q = Question(
        field="location",
        text="Where?",
        options=[Option(label="chin", value="chin"), Option(label="nose", value="nose")],
        allow_freeform=True,
    )
    simulator = SimulatedAnswerer({"location": "temple"})
    assert simulator.answer(q) == "temple"


def test_no_answer_raises_when_missing_and_strict() -> None:
    q = Question(
        field="contact",
        text="Does the hand touch?",
        options=[Option(label="touch", value="touch")],
        allow_freeform=False,
    )
    simulator = SimulatedAnswerer({})  # no expected contact at all
    with pytest.raises(NoAnswerAvailable) as exc_info:
        simulator.answer(q)
    assert exc_info.value.field == "contact"


def test_unspecified_placeholder_when_freeform_and_missing() -> None:
    q = Question(
        field="handshape_nondominant",
        text="Non-dominant handshape?",
        options=None,
        allow_freeform=True,
    )
    simulator = SimulatedAnswerer({})
    # No expected value, but freeform is allowed — simulator returns a
    # marker rather than fabricating a lie.
    assert simulator.answer(q) == "unspecified"
