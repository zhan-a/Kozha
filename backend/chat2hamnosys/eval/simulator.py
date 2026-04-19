"""Simulated clarification answerer.

Given a :class:`Question` and the fixture's ``expected_parameters``,
produce a plausible user answer so the end-to-end flow can run
unattended. For multiple-choice questions we pick the option whose
``value`` matches the expected slot; for freeform questions we hand
back the expected term verbatim. When neither the option list nor
freeform yields a plausible answer, the simulator signals
:class:`NoAnswerAvailable` so the runner records the miss rather than
injecting a lie.

This is a deliberate simplification: a real human would sometimes
give a wrong or imprecise answer, which would drive additional
clarification turns. Modeling that noise would require a second
dataset of simulated-noise-rates per slot and we don't have one yet.
The harness is conservative — clean simulator answers give an
optimistic ceiling, and the live-human eval (Prompt 16 §9) is what
grounds us to reality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clarify import Question


class NoAnswerAvailable(Exception):
    """Raised when the simulator has nothing plausible to say.

    Carries ``field`` so the runner can log which slot it stalled on.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass
class SimulatedAnswerer:
    """Maps ``(Question, expected_parameters)`` → answer string.

    Stateless aside from holding the expected parameters for the
    fixture currently under evaluation. Instantiate once per fixture
    and call :meth:`answer` per question.

    Design rule — option preference order (first match wins):

    1. Options whose ``value`` equals (case-insensitive) the expected
       term. Matches the way the clarifier templates canonicalize
       their options (``Option.value`` is the plain-English phrase
       that will be written back to the slot).
    2. Options whose ``label`` equals the expected term. Rarely
       needed but catches templates whose label is the canonical form.
    3. Freeform fallback: return the expected term verbatim. Only
       used when :attr:`Question.allow_freeform` is true.
    """

    expected_parameters: dict[str, Any]

    def answer(self, question: Question) -> str:
        expected_value = _expected_value_for_field(
            self.expected_parameters, question.field
        )
        if expected_value is None:
            if question.allow_freeform:
                # No truth for this slot, but the question lets us
                # pass freeform. Return an empty-ish marker so the
                # answer parser still resolves without guessing.
                return "unspecified"
            raise NoAnswerAvailable(
                question.field,
                "no expected value in fixture and question forbids freeform",
            )

        expected_norm = _normalize(expected_value)
        options = question.options or []
        for opt in options:
            if _normalize(opt.value) == expected_norm:
                return opt.value
        for opt in options:
            if _normalize(opt.label) == expected_norm:
                return opt.value

        if question.allow_freeform:
            return expected_value

        # Out-of-vocabulary option list and freeform forbidden — pick
        # the closest option by substring match rather than lying with
        # a fabricated value.
        for opt in options:
            if expected_norm and expected_norm in _normalize(opt.value):
                return opt.value
        raise NoAnswerAvailable(
            question.field,
            f"expected {expected_value!r} matches no option and freeform forbidden",
        )


def _normalize(value: str) -> str:
    """Canonical form for option / label matching. Lowercase, punct-light."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def _expected_value_for_field(
    params: dict[str, Any], field_path: str
) -> str | None:
    """Drill into ``params`` to fetch the expected plain-English term.

    Handles the three shapes the clarifier asks about:

    - Scalar slot (``handshape_dominant`` et al.) → the string value.
    - ``movement`` → the first segment's ``path`` (good enough for v1;
      multi-segment fixtures set the leading path to the canonical
      term).
    - ``non_manual.<leaf>`` → the value at that leaf.

    Returns ``None`` when the field is not populated in the fixture,
    so the simulator can fall back to freeform or bail out loudly.
    """
    if field_path in (
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    ):
        val = params.get(field_path)
        return _coerce_scalar(val)

    if field_path == "movement":
        movement = params.get("movement") or []
        if not movement:
            return None
        first = movement[0] if isinstance(movement, list) else None
        if not isinstance(first, dict):
            return None
        return _coerce_scalar(first.get("path"))

    if field_path.startswith("non_manual."):
        leaf = field_path.split(".", 1)[1]
        nm = params.get("non_manual")
        if not isinstance(nm, dict):
            return None
        return _coerce_scalar(nm.get(leaf))

    return None


def _coerce_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


__all__ = ["NoAnswerAvailable", "SimulatedAnswerer"]
