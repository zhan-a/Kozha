"""Ablation toggles for the end-to-end eval.

Three knobs, each disables exactly one layer of the pipeline so we can
measure what that layer contributes. The runner wraps every fixture
run in :func:`apply_ablation`; the context manager restores the
original behavior on exit, so a single process can execute multiple
ablations sequentially.

- ``no_clarification`` — skip the clarifier entirely. The runner
  hands the parser's partial parameters straight to the generator, so
  missing slots fall to the generator's LLM fallback (or cause the
  run to fail at that slot).
- ``no_validator_feedback`` — force the generator's validator-guided
  repair loop to zero iterations. If the deterministic composition
  produces an invalid string, the generator gives up.
- ``no_deterministic_map`` — patch the vocab lookup to always miss,
  which routes every slot through the LLM fallback. Exposes what the
  system looks like without a curated term → codepoint table.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

from .models import AblationName


@dataclass(frozen=True)
class AblationConfig:
    """Which ablation a run is operating under.

    Only ``name`` is semantically meaningful — the boolean flags are
    derived conveniences so call sites can read ``cfg.skip_clarifier``
    without dispatching on the string every time.
    """

    name: AblationName

    @property
    def skip_clarifier(self) -> bool:
        return self.name == "no_clarification"

    @property
    def disable_repair(self) -> bool:
        return self.name == "no_validator_feedback"

    @property
    def disable_vocab(self) -> bool:
        return self.name == "no_deterministic_map"


def ablation_presets() -> list[AblationConfig]:
    """Every ablation the CLI cycles through by default."""
    return [
        AblationConfig("full"),
        AblationConfig("no_clarification"),
        AblationConfig("no_validator_feedback"),
        AblationConfig("no_deterministic_map"),
    ]


@contextlib.contextmanager
def apply_ablation(cfg: AblationConfig) -> Iterator[None]:
    """Temporarily install the ablation on module-level generator state.

    The repair-loop toggle patches the module-level
    ``_MAX_VALIDATION_RETRIES`` constant in
    :mod:`generator.params_to_hamnosys`; the vocab toggle replaces
    :meth:`VocabMap.lookup` with a stub returning ``None`` for every
    slot. Both patches are cheap and local — we prefer them to adding
    ablation flags to the production generator API because the eval
    harness is the only caller that ever wants them.
    """
    if cfg.name == "full":
        yield
        return

    if cfg.disable_repair:
        with _patch_repair_disabled():
            yield
        return

    if cfg.disable_vocab:
        with _patch_vocab_miss():
            yield
        return

    # ``skip_clarifier`` is enforced at the runner level (we just skip
    # calling the clarifier) rather than patching. No monkeying here.
    yield


@contextlib.contextmanager
def _patch_repair_disabled() -> Iterator[None]:
    from generator import params_to_hamnosys as g

    original = g._MAX_VALIDATION_RETRIES
    g._MAX_VALIDATION_RETRIES = 0
    try:
        yield
    finally:
        g._MAX_VALIDATION_RETRIES = original


@contextlib.contextmanager
def _patch_vocab_miss() -> Iterator[None]:
    from generator.vocab import VOCAB

    original = VOCAB.lookup
    # ``lookup`` returns None for missing terms; the generator takes
    # that as "LLM fallback" signal (see params_to_hamnosys._vocab_resolve).
    VOCAB.lookup = lambda slot, term: None  # type: ignore[method-assign]
    try:
        yield
    finally:
        VOCAB.lookup = original  # type: ignore[method-assign]


__all__ = [
    "AblationConfig",
    "ablation_presets",
    "apply_ablation",
]
