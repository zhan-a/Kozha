"""End-to-end evaluation harness for chat2hamnosys.

Measures the full prose → HamNoSys pipeline against a golden fixture
dataset at four levels:

1. Parser — prose → plain-English phonological parameters + gap list.
2. Generator — parameters → HamNoSys string (deterministic + LLM
   fallback + validator-guided repair).
3. End-to-end — the full flow with a :class:`SimulatedAnswerer` in the
   clarifier seat, so gaps are filled without a human in the loop.
4. Cost — tokens, USD, and p95 latency extracted from the JSONL
   telemetry logger.

Three CLIs under :mod:`eval.cli`:

- ``run`` — execute the suite and write a timestamped JSON result.
- ``diff`` — compare two result JSONs, print per-fixture regressions.
- ``report`` — pretty-print a stdout table + write an HTML report.

Ablations (``--no-clarification``, ``--no-validator-feedback``,
``--no-deterministic-map``) run the same fixtures with a single piece
of the pipeline disabled so we can measure what each layer contributes.

Never invoked at import time. All expensive work lives behind the CLI
and the :func:`run_suite` entry point.
"""

from .ablations import AblationConfig, ablation_presets
from .metrics import (
    CostMetrics,
    EndToEndMetrics,
    GeneratorMetrics,
    OverallMetrics,
    ParserMetrics,
)
from .models import (
    AblationName,
    Difficulty,
    EvalResult,
    FixtureResult,
    GoldenFixture,
    HumanRating,
    SuiteReport,
)
from .runner import run_suite
from .simulator import SimulatedAnswerer

__all__ = [
    "AblationConfig",
    "AblationName",
    "CostMetrics",
    "Difficulty",
    "EndToEndMetrics",
    "EvalResult",
    "FixtureResult",
    "GeneratorMetrics",
    "GoldenFixture",
    "HumanRating",
    "OverallMetrics",
    "ParserMetrics",
    "SimulatedAnswerer",
    "SuiteReport",
    "ablation_presets",
    "run_suite",
]
