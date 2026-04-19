"""Apply a user's answer to a :class:`Question` back into partial params.

User responses to a clarification question come in three shapes:

1. **Ordinal word** — ``"second option"``, ``"the first one"``,
   ``"third"``. Resolved to a 1-based index into ``question.options``.
2. **Bare / prefixed index** — ``"2"``, ``"#2"``, ``"option 2"``,
   ``"choice 3"``. Same 1-based resolution.
3. **Freeform phrase** — ``"a bent-five claw shape"``, ``"upward"``.
   Exact label / value matches first; then, if
   ``question.allow_freeform`` is true, the trimmed text is used as-is.

:func:`apply_answer` returns a **new** :class:`PartialSignParameters`
with the relevant slot populated; the input is never mutated.

:class:`AnswerParseError` is raised when the answer cannot be resolved
— typically the caller should re-ask rather than silently swallow an
ambiguous reply.
"""

from __future__ import annotations

import re
from typing import Optional

from parser import (
    PartialMovementSegment,
    PartialNonManualFeatures,
    PartialSignParameters,
)

from .models import Option, Question


class AnswerParseError(Exception):
    """Raised when a user answer cannot be mapped to a canonical value."""


# Ordinal words (incl. a few cardinals like "five" when used as "option five").
_ORDINAL_MAP: dict[str, int] = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "fifth": 5,
    "5th": 5,
    "five": 5,
}


# Matches ordinal-only answers: "second", "the second", "option two",
# "the first one", "number 2". Requires the ordinal word to be the
# whole response, optionally wrapped by "the/an", "option", etc. — so a
# freeform phrase like "bent-five claw" does *not* match.
_ORDINAL_RE = re.compile(
    r"^\s*"
    r"(?:(?:the|an?)\s+)?"
    r"(?:option\s+|choice\s+|number\s+|answer\s+|item\s+)?"
    r"(?P<ord>first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|one|two|three|four|five)"
    r"(?:\s+(?:one|option|choice|answer))?"
    r"\s*$",
    re.IGNORECASE,
)


# Matches bare / prefixed digits: "2", "#2", "option 2", "choice 3".
_INDEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*#\s*(\d+)\s*$"),
    re.compile(r"^\s*(\d+)\s*$"),
    re.compile(r"^\s*option\s+(\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*choice\s+(\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*number\s+(\d+)\s*$", re.IGNORECASE),
)


_NON_MANUAL_LEAVES: frozenset[str] = frozenset(
    {"mouth_picture", "eye_gaze", "head_movement", "eyebrows", "facial_expression"}
)


_SCALAR_SLOTS: frozenset[str] = frozenset(
    {
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_answer(
    parameters: PartialSignParameters,
    question: Question,
    user_answer: str,
) -> PartialSignParameters:
    """Merge ``user_answer`` into ``parameters`` and return a new copy.

    The input ``parameters`` is not mutated. The question's ``field``
    determines which slot gets written; the answer is first resolved to
    a canonical value via :func:`_resolve_value` (ordinal / index / label
    / freeform — see module docstring).

    Raises :class:`AnswerParseError` on empty input, an out-of-range
    index, or an unresolvable answer on a ``allow_freeform=False``
    question.
    """
    if not isinstance(user_answer, str) or not user_answer.strip():
        raise AnswerParseError("user_answer must be a non-empty string")

    value = _resolve_value(question, user_answer)
    return _set_field(parameters, question.field, value)


# ---------------------------------------------------------------------------
# Answer → canonical value
# ---------------------------------------------------------------------------


def _resolve_value(question: Question, answer: str) -> str:
    """Map a user-typed answer to a canonical value, or raise.

    Resolution order (first match wins):

    1. Whole-answer ordinal word ("first", "the second", "option two").
    2. Whole-answer bare / prefixed index ("2", "#2", "option 2").
    3. Case-insensitive exact match on an option ``label`` or ``value``.
    4. Freeform fallback — if ``question.allow_freeform`` is ``True``,
       the trimmed answer itself is returned.
    5. Otherwise raise :class:`AnswerParseError`.
    """
    cleaned = answer.strip()
    lowered = cleaned.lower()
    options = list(question.options or [])

    ordinal_idx = _match_ordinal(cleaned)
    if ordinal_idx is not None:
        return _option_at(options, ordinal_idx, cleaned)

    index_idx = _match_index(cleaned)
    if index_idx is not None:
        return _option_at(options, index_idx, cleaned)

    for opt in options:
        if opt.label.strip().lower() == lowered:
            return opt.value
        if opt.value.strip().lower() == lowered:
            return opt.value

    if question.allow_freeform:
        return cleaned

    raise AnswerParseError(
        f"could not map answer {cleaned!r} to any of the "
        f"{len(options)} listed options for field {question.field!r}"
    )


def _match_ordinal(text: str) -> Optional[int]:
    m = _ORDINAL_RE.match(text)
    if m is None:
        return None
    return _ORDINAL_MAP[m.group("ord").lower()]


def _match_index(text: str) -> Optional[int]:
    for pat in _INDEX_PATTERNS:
        m = pat.match(text)
        if m:
            return int(m.group(1))
    return None


def _option_at(options: list[Option], one_based_idx: int, raw: str) -> str:
    if not options:
        raise AnswerParseError(
            f"answer {raw!r} references an option by index, but the question "
            "has no options to choose from"
        )
    if one_based_idx < 1 or one_based_idx > len(options):
        raise AnswerParseError(
            f"answer {raw!r} references option #{one_based_idx}, but only "
            f"{len(options)} options are available"
        )
    return options[one_based_idx - 1].value


# ---------------------------------------------------------------------------
# Writing into PartialSignParameters
# ---------------------------------------------------------------------------


def _set_field(
    parameters: PartialSignParameters,
    field: str,
    value: str,
) -> PartialSignParameters:
    """Return a new :class:`PartialSignParameters` with ``field`` updated."""
    if field in _SCALAR_SLOTS:
        return parameters.model_copy(update={field: value})

    if field == "movement":
        segments = list(parameters.movement)
        if segments:
            segments[0] = segments[0].model_copy(update={"path": value})
        else:
            segments = [PartialMovementSegment(path=value)]
        return parameters.model_copy(update={"movement": segments})

    if field.startswith("non_manual."):
        leaf = field.split(".", 1)[1]
        if leaf not in _NON_MANUAL_LEAVES:
            raise AnswerParseError(f"unknown non_manual sub-field {leaf!r}")
        nm = parameters.non_manual or PartialNonManualFeatures()
        nm = nm.model_copy(update={leaf: value})
        return parameters.model_copy(update={"non_manual": nm})

    raise AnswerParseError(f"unknown field path {field!r}")


__all__ = ["AnswerParseError", "apply_answer"]
