"""Tests for ``eval.regression`` — baseline I/O + guard trip conditions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.metrics import (
    EndToEndMetrics,
    OverallMetrics,
)
from eval.regression import (
    Baseline,
    GuardReport,
    check_regression,
    load_baseline,
    save_baseline,
)


def _metrics_with_f1(tp: int, fp: int, fn: int, exact_matches: int = 0, n: int = 10) -> OverallMetrics:
    m = OverallMetrics()
    m.end_to_end.n = n
    m.end_to_end.exact_matches = exact_matches
    m.end_to_end.valid_outputs = n
    m.end_to_end.symbol_tp = tp
    m.end_to_end.symbol_fp = fp
    m.end_to_end.symbol_fn = fn
    return m


def test_save_and_load_baseline_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    metrics = _metrics_with_f1(tp=90, fp=10, fn=10, exact_matches=7)
    save_baseline(metrics, commit="abc1234", model="stub-llm", suite="golden_signs", path=path)
    assert path.is_file()

    loaded = load_baseline(path)
    assert loaded.commit == "abc1234"
    assert loaded.model == "stub-llm"
    assert loaded.suite == "golden_signs"
    # F1 = 2*P*R / (P+R) where P=R=0.9 → 0.9
    assert abs(loaded.e2e_symbol_f1() - 0.9) < 1e-9
    assert abs(loaded.e2e_exact_match_rate() - 0.7) < 1e-9


def test_load_baseline_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "does-not-exist.json")


def test_guard_passes_when_within_tolerance() -> None:
    # Baseline: F1 ≈ 0.80; current: F1 ≈ 0.78 (drop of 0.02 < 0.05 threshold)
    baseline = Baseline.from_dict(
        {
            "model": "stub-llm",
            "suite": "golden_signs",
            "metrics": {
                "end_to_end": {
                    "symbol_f1": 0.80,
                    "exact_match_rate": 0.50,
                    "validity_rate": 1.0,
                }
            },
        }
    )
    current = _metrics_with_f1(tp=78, fp=22, fn=22, exact_matches=5, n=10)
    guard = check_regression(current, baseline)
    assert isinstance(guard, GuardReport)
    assert guard.ok is True
    assert "within tolerance" in guard.message


def test_guard_trips_on_f1_drop_beyond_threshold() -> None:
    baseline = Baseline.from_dict(
        {
            "model": "stub-llm",
            "suite": "golden_signs",
            "metrics": {
                "end_to_end": {
                    "symbol_f1": 0.90,
                    "exact_match_rate": 0.50,
                    "validity_rate": 1.0,
                }
            },
        }
    )
    # Current F1 = 0.5, drop = 0.4 >> 0.05
    current = _metrics_with_f1(tp=50, fp=50, fn=50, exact_matches=5, n=10)
    guard = check_regression(current, baseline)
    assert guard.ok is False
    assert "dropped" in guard.message.lower()
    assert guard.delta < -0.05


def test_guard_trips_on_exact_match_drop_even_when_f1_steady() -> None:
    baseline = Baseline.from_dict(
        {
            "model": "stub-llm",
            "suite": "golden_signs",
            "metrics": {
                "end_to_end": {
                    "symbol_f1": 0.80,
                    "exact_match_rate": 0.80,
                    "validity_rate": 1.0,
                }
            },
        }
    )
    # F1 holds at ~0.80, but exact matches crater: 0.80 → 0.10
    current = _metrics_with_f1(tp=80, fp=20, fn=20, exact_matches=1, n=10)
    guard = check_regression(current, baseline)
    assert guard.ok is False
    assert "exact-match" in guard.message.lower()


def test_guard_passes_on_improvement() -> None:
    baseline = Baseline.from_dict(
        {
            "model": "stub-llm",
            "suite": "golden_signs",
            "metrics": {
                "end_to_end": {
                    "symbol_f1": 0.70,
                    "exact_match_rate": 0.40,
                    "validity_rate": 1.0,
                }
            },
        }
    )
    current = _metrics_with_f1(tp=90, fp=10, fn=10, exact_matches=7, n=10)
    guard = check_regression(current, baseline)
    assert guard.ok is True
    assert guard.delta > 0


def test_shipped_placeholder_baseline_is_loadable() -> None:
    """The placeholder shipped at baselines/current.json must load cleanly.

    CI will call ``load_baseline()`` without an explicit path — if the
    file is absent or structurally wrong, the guard crashes before
    running the suite. This test is the cheap early warning.
    """
    from eval.regression import BASELINE_FILE

    if not BASELINE_FILE.is_file():
        pytest.skip("no shipped baseline")
    baseline = load_baseline()
    # Placeholder is all zeros; that's intentional (guard can't trip
    # against a zero floor). But the *shape* must be right.
    assert baseline.suite  # non-empty
    assert "end_to_end" in baseline.metrics
    assert "symbol_f1" in baseline.metrics["end_to_end"]
