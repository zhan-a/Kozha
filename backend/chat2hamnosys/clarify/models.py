"""Clarifier-side Pydantic models.

A :class:`Question` is a single clarification the authoring UI shows the
signer. Produced by :func:`generate_questions`, applied back via
:func:`apply_answer`.

- :class:`Option` — one multiple-choice entry. ``label`` is user-facing,
  ``value`` is the canonical plain-English phrase that will be written
  into the :class:`PartialSignParameters` slot when the option is picked.
- :class:`Question` — ``field``, ``text``, ``options`` (or ``None`` for
  open-ended slots), ``allow_freeform``, and ``rationale`` for the debug
  UI.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Option(BaseModel):
    """One multiple-choice option within a :class:`Question`."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    value: str = Field(min_length=1)


class Question(BaseModel):
    """One clarification request targeting a single phonological slot."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    text: str = Field(min_length=1)
    options: Optional[List[Option]] = None
    allow_freeform: bool = True
    rationale: str = ""


__all__ = ["Option", "Question"]
