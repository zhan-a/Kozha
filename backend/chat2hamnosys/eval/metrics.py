"""Metric computation across fixture results.

Four layers of metrics, all computed from the list of
:class:`FixtureResult` rows produced by :mod:`eval.runner`:

- :class:`ParserMetrics` — per-field accuracy plus precision / recall
  for gap detection. ``parser_populated_*`` on each fixture carries
  the raw set membership.
- :class:`GeneratorMetrics` — exact-match rate, symbol-level PRF, and
  validity rate when the generator is fed gold parameters.
- :class:`EndToEndMetrics` — the same three numbers when gaps are
  filled by the :class:`SimulatedAnswerer`, plus the mean number of
  clarification turns used.
- :class:`CostMetrics` — total tokens, total USD, mean USD per sign,
  and p95 latency. The tokens / cost come from the JSONL telemetry
  log; we sum per-fixture rather than re-read the log, because the
  runner stores the aggregated numbers on each :class:`FixtureResult`.

:class:`OverallMetrics` bundles the four so a single object represents
a slice (e.g. "all hard DGS fixtures under the no-clarification
ablation").
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from .models import FixtureResult, GoldenFixture


# Slot names whose accuracy we track per-field (Prompt 16 §3 parser
# metrics). Matches the public surface of PartialSignParameters with
# one aggregated ``non_manual`` bucket — five separate sub-fields
# would swamp the report for no gain in signal.
_PARSER_FIELD_NAMES: tuple[str, ...] = (
    "handshape_dominant",
    "handshape_nondominant",
    "orientation_extended_finger",
    "orientation_palm",
    "location",
    "contact",
    "movement",
    "non_manual",
)


@dataclass
class ParserMetrics:
    """Parser-stage metrics aggregated across a set of fixtures."""

    n: int = 0
    # Per-field "did we get the value right?" counts.
    field_correct: dict[str, int] = field(default_factory=dict)
    field_total: dict[str, int] = field(default_factory=dict)
    # Gap-detection TP/FP/FN at the set level.
    gap_tp: int = 0
    gap_fp: int = 0
    gap_fn: int = 0
    # Populated-set TP/FP/FN (did we populate the right slots?).
    pop_tp: int = 0
    pop_fp: int = 0
    pop_fn: int = 0

    def field_accuracy(self, field_name: str) -> float:
        total = self.field_total.get(field_name, 0)
        if total == 0:
            return 0.0
        return self.field_correct.get(field_name, 0) / total

    def gap_precision(self) -> float:
        return _safe_div(self.gap_tp, self.gap_tp + self.gap_fp)

    def gap_recall(self) -> float:
        return _safe_div(self.gap_tp, self.gap_tp + self.gap_fn)

    def gap_f1(self) -> float:
        return _f1(self.gap_precision(), self.gap_recall())

    def populated_precision(self) -> float:
        return _safe_div(self.pop_tp, self.pop_tp + self.pop_fp)

    def populated_recall(self) -> float:
        return _safe_div(self.pop_tp, self.pop_tp + self.pop_fn)

    def populated_f1(self) -> float:
        return _f1(self.populated_precision(), self.populated_recall())

    def to_json_safe(self) -> dict:
        return {
            "n": self.n,
            "field_correct": dict(self.field_correct),
            "field_total": dict(self.field_total),
            "field_accuracy": {
                f: self.field_accuracy(f) for f in _PARSER_FIELD_NAMES
            },
            "gap_tp": self.gap_tp,
            "gap_fp": self.gap_fp,
            "gap_fn": self.gap_fn,
            "gap_precision": self.gap_precision(),
            "gap_recall": self.gap_recall(),
            "gap_f1": self.gap_f1(),
            "populated_precision": self.populated_precision(),
            "populated_recall": self.populated_recall(),
            "populated_f1": self.populated_f1(),
        }


@dataclass
class GeneratorMetrics:
    """Generator-stage metrics (gold-parameter input)."""

    n: int = 0
    exact_matches: int = 0
    valid_outputs: int = 0
    symbol_tp: int = 0
    symbol_fp: int = 0
    symbol_fn: int = 0

    def exact_match_rate(self) -> float:
        return _safe_div(self.exact_matches, self.n)

    def validity_rate(self) -> float:
        return _safe_div(self.valid_outputs, self.n)

    def symbol_precision(self) -> float:
        return _safe_div(self.symbol_tp, self.symbol_tp + self.symbol_fp)

    def symbol_recall(self) -> float:
        return _safe_div(self.symbol_tp, self.symbol_tp + self.symbol_fn)

    def symbol_f1(self) -> float:
        return _f1(self.symbol_precision(), self.symbol_recall())

    def to_json_safe(self) -> dict:
        return {
            "n": self.n,
            "exact_matches": self.exact_matches,
            "exact_match_rate": self.exact_match_rate(),
            "valid_outputs": self.valid_outputs,
            "validity_rate": self.validity_rate(),
            "symbol_tp": self.symbol_tp,
            "symbol_fp": self.symbol_fp,
            "symbol_fn": self.symbol_fn,
            "symbol_precision": self.symbol_precision(),
            "symbol_recall": self.symbol_recall(),
            "symbol_f1": self.symbol_f1(),
        }


@dataclass
class EndToEndMetrics:
    """End-to-end metrics (prose → parse → simulate → generate)."""

    n: int = 0
    exact_matches: int = 0
    valid_outputs: int = 0
    symbol_tp: int = 0
    symbol_fp: int = 0
    symbol_fn: int = 0
    total_clarification_turns: int = 0
    total_questions_asked: int = 0

    def exact_match_rate(self) -> float:
        return _safe_div(self.exact_matches, self.n)

    def validity_rate(self) -> float:
        return _safe_div(self.valid_outputs, self.n)

    def symbol_precision(self) -> float:
        return _safe_div(self.symbol_tp, self.symbol_tp + self.symbol_fp)

    def symbol_recall(self) -> float:
        return _safe_div(self.symbol_tp, self.symbol_tp + self.symbol_fn)

    def symbol_f1(self) -> float:
        return _f1(self.symbol_precision(), self.symbol_recall())

    def mean_clarification_turns(self) -> float:
        return _safe_div(self.total_clarification_turns, self.n)

    def mean_questions_asked(self) -> float:
        return _safe_div(self.total_questions_asked, self.n)

    def to_json_safe(self) -> dict:
        return {
            "n": self.n,
            "exact_matches": self.exact_matches,
            "exact_match_rate": self.exact_match_rate(),
            "valid_outputs": self.valid_outputs,
            "validity_rate": self.validity_rate(),
            "symbol_tp": self.symbol_tp,
            "symbol_fp": self.symbol_fp,
            "symbol_fn": self.symbol_fn,
            "symbol_precision": self.symbol_precision(),
            "symbol_recall": self.symbol_recall(),
            "symbol_f1": self.symbol_f1(),
            "total_clarification_turns": self.total_clarification_turns,
            "mean_clarification_turns": self.mean_clarification_turns(),
            "total_questions_asked": self.total_questions_asked,
            "mean_questions_asked": self.mean_questions_asked(),
        }


@dataclass
class CostMetrics:
    """Per-run cost + latency stats."""

    n: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    latencies_ms: list[int] = field(default_factory=list)
    total_llm_calls: int = 0

    def mean_cost_per_sign(self) -> float:
        return _safe_div(self.total_cost_usd, self.n)

    def p95_latency_ms(self) -> int:
        if not self.latencies_ms:
            return 0
        # statistics.quantiles needs >=2 points; handle the n==1 edge.
        if len(self.latencies_ms) == 1:
            return self.latencies_ms[0]
        # Use the inclusive method so the p95 is a real observed value.
        quants = statistics.quantiles(
            self.latencies_ms, n=20, method="inclusive"
        )
        return int(quants[18])  # 19th cutpoint ≈ 95th percentile

    def mean_latency_ms(self) -> int:
        if not self.latencies_ms:
            return 0
        return int(statistics.fmean(self.latencies_ms))

    def to_json_safe(self) -> dict:
        return {
            "n": self.n,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "mean_cost_per_sign": round(self.mean_cost_per_sign(), 6),
            "total_llm_calls": self.total_llm_calls,
            "p95_latency_ms": self.p95_latency_ms(),
            "mean_latency_ms": self.mean_latency_ms(),
        }


@dataclass
class OverallMetrics:
    """Parser + generator + e2e + cost bundle for one slice of fixtures."""

    parser: ParserMetrics = field(default_factory=ParserMetrics)
    generator: GeneratorMetrics = field(default_factory=GeneratorMetrics)
    end_to_end: EndToEndMetrics = field(default_factory=EndToEndMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)

    def to_json_safe(self) -> dict:
        return {
            "parser": self.parser.to_json_safe(),
            "generator": self.generator.to_json_safe(),
            "end_to_end": self.end_to_end.to_json_safe(),
            "cost": self.cost.to_json_safe(),
        }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _f1(p: float, r: float) -> float:
    if not p or not r:
        return 0.0
    return 2 * p * r / (p + r)


def aggregate(
    fixture_results: Iterable[FixtureResult],
    fixtures_by_id: dict[str, GoldenFixture] | None = None,
) -> OverallMetrics:
    """Sum up fixture-level numbers into :class:`OverallMetrics`.

    ``fixtures_by_id`` is only needed if the caller wants parser
    field-accuracy rows to be weighted against the *full* expected
    field set (e.g. a slot the parser skipped entirely still counts
    as "total=1, correct=0"). Pass ``None`` when aggregating a slice
    that already has its fixtures resolved.
    """
    out = OverallMetrics()
    for fr in fixture_results:
        out.parser.n += 1
        for fname, correct in fr.parser_field_accuracy.items():
            out.parser.field_correct[fname] = (
                out.parser.field_correct.get(fname, 0) + (1 if correct else 0)
            )
            out.parser.field_total[fname] = out.parser.field_total.get(fname, 0) + 1
        out.parser.pop_tp += len(fr.parser_populated_correct)
        out.parser.pop_fn += len(
            fr.parser_populated_expected - fr.parser_populated_actual
        )
        out.parser.pop_fp += len(
            fr.parser_populated_actual - fr.parser_populated_expected
        )
        out.parser.gap_tp += len(
            fr.parser_gaps_actual & fr.parser_gaps_expected
        )
        out.parser.gap_fp += len(
            fr.parser_gaps_actual - fr.parser_gaps_expected
        )
        out.parser.gap_fn += len(
            fr.parser_gaps_expected - fr.parser_gaps_actual
        )

        out.generator.n += 1
        if fr.gen_exact_match:
            out.generator.exact_matches += 1
        if fr.gen_valid:
            out.generator.valid_outputs += 1
        out.generator.symbol_tp += fr.gen_symbol_tp
        out.generator.symbol_fp += fr.gen_symbol_fp
        out.generator.symbol_fn += fr.gen_symbol_fn

        out.end_to_end.n += 1
        if fr.e2e_exact_match:
            out.end_to_end.exact_matches += 1
        if fr.e2e_valid:
            out.end_to_end.valid_outputs += 1
        out.end_to_end.symbol_tp += fr.e2e_symbol_tp
        out.end_to_end.symbol_fp += fr.e2e_symbol_fp
        out.end_to_end.symbol_fn += fr.e2e_symbol_fn
        out.end_to_end.total_clarification_turns += fr.e2e_clarification_turns
        out.end_to_end.total_questions_asked += fr.e2e_questions_asked

        out.cost.n += 1
        out.cost.total_tokens += fr.total_tokens
        out.cost.total_prompt_tokens += fr.prompt_tokens
        out.cost.total_completion_tokens += fr.completion_tokens
        out.cost.total_cost_usd += fr.cost_usd
        out.cost.total_llm_calls += fr.llm_calls
        if fr.latency_ms > 0:
            out.cost.latencies_ms.append(fr.latency_ms)
    return out


# ---------------------------------------------------------------------------
# Slicing (Prompt 16 §8)
# ---------------------------------------------------------------------------


# Each slicer returns a short string bucket name for the fixture.
# The runner asks each slicer for a key and appends the FixtureResult
# to ``slices[slice_name][bucket]``.
SLICERS: dict[str, Callable[[GoldenFixture], str]] = {
    "difficulty": lambda fx: fx.difficulty,
    "sign_language": lambda fx: fx.sign_language,
    "non_manual": lambda fx: "with_nm" if fx.has_non_manual() else "no_nm",
    "handedness": lambda fx: "two_handed" if fx.is_two_handed() else "one_handed",
}


def compute_slices(
    fixture_results: list[FixtureResult],
    fixtures_by_id: dict[str, GoldenFixture],
) -> dict[str, dict[str, OverallMetrics]]:
    """Group results by slicing key and aggregate each group."""
    out: dict[str, dict[str, OverallMetrics]] = {}
    for slice_name, slicer in SLICERS.items():
        groups: dict[str, list[FixtureResult]] = {}
        for fr in fixture_results:
            fx = fixtures_by_id.get(fr.fixture_id)
            if fx is None:
                continue
            key = slicer(fx)
            groups.setdefault(key, []).append(fr)
        out[slice_name] = {
            bucket: aggregate(rows) for bucket, rows in groups.items()
        }
    return out


__all__ = [
    "CostMetrics",
    "EndToEndMetrics",
    "GeneratorMetrics",
    "OverallMetrics",
    "ParserMetrics",
    "SLICERS",
    "aggregate",
    "compute_slices",
]
