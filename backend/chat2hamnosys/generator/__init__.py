"""Parameters → HamNoSys generator.

Public surface:

- :func:`generate` — main entry; returns a :class:`GenerateResult`.
- :class:`GenerateResult` — dataclass with the HamNoSys string,
  validation result, LLM-fallback bookkeeping, and confidence.
- :data:`VOCAB` / :class:`VocabMap` — the authored plain-English →
  HamNoSys mapping table (handy for tests and the eval harness).
"""

from .params_to_hamnosys import GenerateResult, generate
from .vocab import VOCAB, VOCAB_PATH, VocabEntry, VocabMap, normalize_term

__all__ = [
    "GenerateResult",
    "generate",
    "VOCAB",
    "VOCAB_PATH",
    "VocabEntry",
    "VocabMap",
    "normalize_term",
]
