"""Prose → phonological-features parser.

Given a natural-language description of a sign, :func:`parse_description`
returns a :class:`ParseResult` containing a (partially populated)
:class:`PartialSignParameters` plus a list of :class:`Gap` records for
slots the LLM could not fill with confidence.

The parser deliberately stays in plain-English vocabulary: it never emits
HamNoSys Private-Use-Area codepoints. A downstream step handles the
English-to-HamNoSys mapping.
"""

from .description_parser import (
    PARSER_RESPONSE_SCHEMA,
    ParserError,
    SYSTEM_PROMPT,
    build_parser_response_schema,
    parse_description,
)
from .models import (
    ALLOWED_GAP_FIELDS,
    Gap,
    ParseResult,
    PartialMovementSegment,
    PartialNonManualFeatures,
    PartialSignParameters,
)

__all__ = [
    "ALLOWED_GAP_FIELDS",
    "Gap",
    "PARSER_RESPONSE_SCHEMA",
    "ParseResult",
    "ParserError",
    "PartialMovementSegment",
    "PartialNonManualFeatures",
    "PartialSignParameters",
    "SYSTEM_PROMPT",
    "build_parser_response_schema",
    "parse_description",
]
