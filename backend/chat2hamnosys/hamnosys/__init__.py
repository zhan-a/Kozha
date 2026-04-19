"""HamNoSys 4.0 grammar, symbol table, and validator.

Public API:
    - validate(s)    -> ValidationResult
    - normalize(s)   -> str
    - SYMBOLS        -> dict[int, Symbol] (codepoint -> Symbol)
    - classify(ch)   -> SymClass | None
    - HamNoSysGrammar -> Lark grammar object
"""

from .symbols import (
    SYMBOLS,
    Symbol,
    SymClass,
    Slot,
    classify,
    is_known,
    codepoints_in_class,
)
from .validator import (
    validate,
    normalize,
    ValidationResult,
    ValidationError,
    ValidationWarning,
)
from .grammar import HamNoSysGrammar, parse_tree

__all__ = [
    "SYMBOLS",
    "Symbol",
    "SymClass",
    "Slot",
    "classify",
    "is_known",
    "codepoints_in_class",
    "validate",
    "normalize",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    "HamNoSysGrammar",
    "parse_tree",
]
