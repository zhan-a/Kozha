"""Clarification-question generator and answer resolution.

After :func:`parse_description` returns a :class:`ParseResult` with
gaps, :func:`generate_questions` turns them into up to three user-facing
:class:`Question`s (LLM rewrite with a deterministic template fallback).
When the author answers, :func:`apply_answer` maps the response back
into the structured :class:`PartialSignParameters`.

Public API:

- :class:`Question`, :class:`Option` — the question data contract.
- :func:`generate_questions` — LLM-assisted generator with fallback.
- :func:`apply_answer` — answer → updated PartialSignParameters.
- :func:`template_for`, :data:`TEMPLATES` — deterministic question
  templates per slot, shared with the LLM as rewrite hints.
- :data:`SYSTEM_PROMPT`, :data:`GENERATOR_RESPONSE_SCHEMA` — exposed for
  tests and the eval doc to inspect what the LLM actually sees.
"""

from .answer_parser import AnswerParseError, apply_answer
from .models import Option, Question
from .question_generator import (
    GENERATOR_RESPONSE_SCHEMA,
    MAX_QUESTIONS_PER_TURN,
    SYSTEM_PROMPT,
    build_generator_response_schema,
    generate_questions,
)
from .templates import TEMPLATES, Template, template_for

__all__ = [
    "AnswerParseError",
    "GENERATOR_RESPONSE_SCHEMA",
    "MAX_QUESTIONS_PER_TURN",
    "Option",
    "Question",
    "SYSTEM_PROMPT",
    "TEMPLATES",
    "Template",
    "apply_answer",
    "build_generator_response_schema",
    "generate_questions",
    "template_for",
]
