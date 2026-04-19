"""Tests for ``eval.metrics`` — aggregation math + slicing."""

from __future__ import annotations

from eval.metrics import (
    EndToEndMetrics,
    OverallMetrics,
    ParserMetrics,
    aggregate,
    compute_slices,
)
from eval.models import FixtureResult, GoldenFixture


def _fixture(fid: str, **overrides) -> GoldenFixture:
    base: dict = {
        "id": fid,
        "prose_description": "sample",
        "gloss": "SAMPLE",
        "sign_language": "dgs",
        "expected_parameters": {"handshape_dominant": "flat"},
        "expected_hamnosys": "\ue001",
        "acceptable_hamnosys_variants": [],
        "source": "unit-test",
        "difficulty": "easy",
    }
    base.update(overrides)
    return GoldenFixture.from_dict(base)


def _result(fid: str, ablation: str = "full", **overrides) -> FixtureResult:
    fr = FixtureResult(fixture_id=fid, ablation=ablation)  # type: ignore[arg-type]
    for key, value in overrides.items():
        setattr(fr, key, value)
    return fr


def test_aggregate_parser_accuracy_and_gaps() -> None:
    results = [
        _result(
            "a",
            parser_field_accuracy={"handshape_dominant": True, "location": False},
            parser_populated_expected={"handshape_dominant", "location"},
            parser_populated_actual={"handshape_dominant"},
            parser_populated_correct={"handshape_dominant"},
            parser_gaps_actual={"location"},
            parser_gaps_expected={"location"},
        ),
        _result(
            "b",
            parser_field_accuracy={"handshape_dominant": True, "location": True},
            parser_populated_expected={"handshape_dominant", "location"},
            parser_populated_actual={"handshape_dominant", "location"},
            parser_populated_correct={"handshape_dominant", "location"},
            parser_gaps_actual=set(),
            parser_gaps_expected=set(),
        ),
    ]
    agg = aggregate(results)
    assert agg.parser.n == 2
    assert agg.parser.field_accuracy("handshape_dominant") == 1.0
    assert agg.parser.field_accuracy("location") == 0.5
    assert agg.parser.gap_tp == 1
    assert agg.parser.gap_fp == 0
    assert agg.parser.gap_fn == 0
    assert agg.parser.gap_precision() == 1.0
    assert agg.parser.gap_recall() == 1.0
    # 3 correctly populated slots across both fixtures, 1 missed (fixture
    # a left "location" unpopulated despite being expected), so recall
    # = 3 / (3 + 1) = 0.75.
    assert agg.parser.populated_precision() == 1.0
    assert agg.parser.populated_recall() == 0.75


def test_aggregate_generator_exact_match_and_symbols() -> None:
    results = [
        _result(
            "a",
            gen_exact_match=True,
            gen_valid=True,
            gen_symbol_tp=5,
            gen_symbol_fp=0,
            gen_symbol_fn=0,
        ),
        _result(
            "b",
            gen_exact_match=False,
            gen_valid=True,
            gen_symbol_tp=3,
            gen_symbol_fp=1,
            gen_symbol_fn=2,
        ),
    ]
    agg = aggregate(results)
    assert agg.generator.n == 2
    assert agg.generator.exact_matches == 1
    assert agg.generator.exact_match_rate() == 0.5
    assert agg.generator.validity_rate() == 1.0
    assert agg.generator.symbol_tp == 8
    assert agg.generator.symbol_fp == 1
    assert agg.generator.symbol_fn == 2
    # F1 should be in (0, 1).
    assert 0.0 < agg.generator.symbol_f1() < 1.0


def test_aggregate_end_to_end_with_clarification_turns() -> None:
    results = [
        _result(
            "a",
            e2e_exact_match=True,
            e2e_valid=True,
            e2e_symbol_tp=4,
            e2e_clarification_turns=1,
            e2e_questions_asked=2,
        ),
        _result(
            "b",
            e2e_exact_match=False,
            e2e_valid=False,
            e2e_symbol_tp=2,
            e2e_symbol_fp=1,
            e2e_clarification_turns=2,
            e2e_questions_asked=3,
        ),
    ]
    agg = aggregate(results)
    assert agg.end_to_end.n == 2
    assert agg.end_to_end.exact_match_rate() == 0.5
    assert agg.end_to_end.validity_rate() == 0.5
    assert agg.end_to_end.total_clarification_turns == 3
    assert agg.end_to_end.mean_clarification_turns() == 1.5
    assert agg.end_to_end.total_questions_asked == 5


def test_aggregate_cost_sums_tokens_and_latency() -> None:
    results = [
        _result(
            "a",
            total_tokens=100,
            prompt_tokens=60,
            completion_tokens=40,
            cost_usd=0.02,
            latency_ms=400,
            llm_calls=3,
        ),
        _result(
            "b",
            total_tokens=200,
            prompt_tokens=120,
            completion_tokens=80,
            cost_usd=0.04,
            latency_ms=800,
            llm_calls=5,
        ),
    ]
    agg = aggregate(results)
    assert agg.cost.total_tokens == 300
    assert agg.cost.total_prompt_tokens == 180
    assert agg.cost.total_completion_tokens == 120
    assert abs(agg.cost.total_cost_usd - 0.06) < 1e-9
    assert agg.cost.total_llm_calls == 8
    assert agg.cost.latencies_ms == [400, 800]
    assert agg.cost.p95_latency_ms() > 0


def test_overall_metrics_json_safe_shape() -> None:
    agg = aggregate([_result("a", gen_exact_match=True)])
    data = agg.to_json_safe()
    assert set(data.keys()) == {"parser", "generator", "end_to_end", "cost"}
    assert data["generator"]["exact_match_rate"] == 1.0
    assert data["end_to_end"]["n"] == 1


def test_compute_slices_groups_by_difficulty_and_handedness() -> None:
    fixtures = {
        "easy-1": _fixture("easy-1", difficulty="easy"),
        "hard-1": _fixture(
            "hard-1",
            difficulty="hard",
            expected_parameters={
                "handshape_dominant": "flat",
                "handshape_nondominant": "flat",
            },
        ),
    }
    results = [
        _result("easy-1", gen_exact_match=True),
        _result("hard-1", gen_exact_match=False),
    ]
    slices = compute_slices(results, fixtures)
    assert set(slices.keys()) == {
        "difficulty",
        "sign_language",
        "non_manual",
        "handedness",
    }
    assert slices["difficulty"]["easy"].generator.exact_match_rate() == 1.0
    assert slices["difficulty"]["hard"].generator.exact_match_rate() == 0.0
    assert set(slices["handedness"].keys()) == {"one_handed", "two_handed"}


def test_compute_slices_skips_unknown_fixture_ids() -> None:
    # A FixtureResult whose id is not in the fixtures_by_id dict should be
    # silently skipped (defensive — the report should not crash on stale data).
    fixtures = {"known": _fixture("known")}
    results = [
        _result("known", gen_exact_match=True),
        _result("orphan", gen_exact_match=True),
    ]
    slices = compute_slices(results, fixtures)
    assert slices["difficulty"]["easy"].generator.n == 1
