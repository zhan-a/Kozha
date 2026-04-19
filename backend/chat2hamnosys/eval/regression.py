"""Regression guard for CI.

Runs the eval against the 10-fixture ``smoke`` subset and compares the
result against the baseline stored in
``backend/chat2hamnosys/eval/baselines/current.json``. Fails the
build when end-to-end symbol F1 drops by more than
:data:`MAX_F1_DROP` (default 0.05 — 5 percentage points).

The baseline is intentionally *not* auto-updated. When we deliberately
change behavior in a way that lowers F1, the team updates
``current.json`` by hand in the same PR so the guard doesn't fail
later unrelated PRs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import OverallMetrics


BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
BASELINE_FILE = BASELINES_DIR / "current.json"
MAX_F1_DROP = 0.05


# Each baseline file records the full-system metrics for the ``smoke``
# subset on the day it was recorded. New baselines are just a flat
# snapshot of :meth:`OverallMetrics.to_json_safe()` plus the sign
# language / model / commit context, so a human reviewer can see at a
# glance what was signed off.
@dataclass
class Baseline:
    """Saved reference metrics for the regression guard."""

    recorded_at: str
    commit: str
    model: str
    suite: str
    metrics: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Baseline":
        return cls(
            recorded_at=d.get("recorded_at", ""),
            commit=d.get("commit", ""),
            model=d.get("model", ""),
            suite=d.get("suite", "smoke"),
            metrics=d.get("metrics", {}),
        )

    def e2e_symbol_f1(self) -> float:
        return float(self.metrics.get("end_to_end", {}).get("symbol_f1", 0.0))

    def e2e_exact_match_rate(self) -> float:
        return float(
            self.metrics.get("end_to_end", {}).get("exact_match_rate", 0.0)
        )

    def validity_rate(self) -> float:
        return float(self.metrics.get("end_to_end", {}).get("validity_rate", 0.0))


@dataclass
class GuardReport:
    """Result of one regression check."""

    ok: bool
    message: str
    current_f1: float
    baseline_f1: float
    delta: float

    def render(self) -> str:
        icon = "OK" if self.ok else "FAIL"
        return (
            f"[{icon}] regression guard: "
            f"current e2e symbol F1 = {self.current_f1:.3f}, "
            f"baseline = {self.baseline_f1:.3f}, "
            f"delta = {self.delta:+.3f}. "
            f"{self.message}"
        )


def load_baseline(path: Path | None = None) -> Baseline:
    """Load the pinned baseline for the regression guard.

    When ``path`` is omitted, reads :data:`BASELINE_FILE`. A missing
    or empty file is an explicit configuration error — the guard
    should never silently pass because someone deleted the baseline.
    """
    baseline_path = path or BASELINE_FILE
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"baseline file not found at {baseline_path} — "
            f"run `python -m eval run --suite smoke --update-baseline` "
            f"to record one"
        )
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    return Baseline.from_dict(data)


def save_baseline(
    metrics: OverallMetrics,
    *,
    commit: str,
    model: str,
    suite: str,
    path: Path | None = None,
) -> Path:
    """Persist a fresh baseline. Manual-only — the guard never writes."""
    baseline_path = path or BASELINE_FILE
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "model": model,
        "suite": suite,
        "metrics": metrics.to_json_safe(),
    }
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return baseline_path


def check_regression(
    current: OverallMetrics,
    baseline: Baseline,
    *,
    max_drop: float = MAX_F1_DROP,
) -> GuardReport:
    """Return a :class:`GuardReport` for the current-vs-baseline delta.

    ``current`` is the aggregate metrics for the ``full`` ablation
    on the same subset that produced the baseline. The guard trips
    when end-to-end symbol F1 has dropped by more than ``max_drop``.
    Exact-match regressions also flag, because a no-F1-change but
    10% lower exact-match rate still represents a meaningful
    regression in the kind of output users see.
    """
    current_f1 = current.end_to_end.symbol_f1()
    baseline_f1 = baseline.e2e_symbol_f1()
    delta = current_f1 - baseline_f1

    current_exact = current.end_to_end.exact_match_rate()
    baseline_exact = baseline.e2e_exact_match_rate()
    exact_delta = current_exact - baseline_exact

    if delta < -max_drop:
        return GuardReport(
            ok=False,
            message=(
                f"e2e symbol F1 dropped {-delta:.3f} (> {max_drop:.3f} threshold). "
                "Review the failing fixtures or update the baseline if intentional."
            ),
            current_f1=current_f1,
            baseline_f1=baseline_f1,
            delta=delta,
        )
    if exact_delta < -max_drop:
        return GuardReport(
            ok=False,
            message=(
                f"e2e exact-match rate dropped {-exact_delta:.3f} "
                f"(> {max_drop:.3f} threshold). F1 was steady, but the "
                "whole-string match rate fell, which users will notice."
            ),
            current_f1=current_f1,
            baseline_f1=baseline_f1,
            delta=delta,
        )
    return GuardReport(
        ok=True,
        message="within tolerance.",
        current_f1=current_f1,
        baseline_f1=baseline_f1,
        delta=delta,
    )


__all__ = [
    "BASELINE_FILE",
    "Baseline",
    "GuardReport",
    "MAX_F1_DROP",
    "check_regression",
    "load_baseline",
    "save_baseline",
]
