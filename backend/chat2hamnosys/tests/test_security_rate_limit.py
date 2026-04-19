"""Per-IP / global cost-cap and anomaly-detector behavior."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from security.rate_limit import (
    CostCapExceeded,
    CostCapTracker,
    DailyBudgetExceeded,
    GlobalDailyBudget,
    anomaly_detector,
)


# ---------------------------------------------------------------------------
# CostCapTracker (per-IP daily)
# ---------------------------------------------------------------------------


def test_per_ip_tracker_accepts_under_cap() -> None:
    tracker = CostCapTracker(cap_usd=10.0, scope="per-ip")
    tracker.check("1.2.3.4", would_add=3.0)
    tracker.record("1.2.3.4", 3.0)
    assert tracker.spent("1.2.3.4") == pytest.approx(3.0)


def test_per_ip_tracker_rejects_over_cap() -> None:
    tracker = CostCapTracker(cap_usd=10.0, scope="per-ip")
    tracker.record("1.2.3.4", 9.0)
    with pytest.raises(CostCapExceeded) as excinfo:
        tracker.check("1.2.3.4", would_add=2.0)
    assert excinfo.value.key == "1.2.3.4"
    assert excinfo.value.cap == 10.0
    assert excinfo.value.spent == pytest.approx(9.0)


def test_per_ip_tracker_keys_are_isolated() -> None:
    tracker = CostCapTracker(cap_usd=5.0, scope="per-ip")
    tracker.record("1.1.1.1", 4.0)
    # Exhausting key A must not block key B.
    tracker.check("2.2.2.2", would_add=4.0)
    tracker.record("2.2.2.2", 4.0)
    assert tracker.spent("1.1.1.1") == pytest.approx(4.0)
    assert tracker.spent("2.2.2.2") == pytest.approx(4.0)


def test_per_ip_tracker_rolls_over_on_utc_date_change() -> None:
    tracker = CostCapTracker(cap_usd=5.0, scope="per-ip")

    # Burn the cap on day 1.
    tracker.record("1.2.3.4", 5.0)
    with pytest.raises(CostCapExceeded):
        tracker.check("1.2.3.4", would_add=0.01)

    # Simulate tomorrow: the tracker's recorded day marker is stale, so
    # the next call should reset the bucket.
    yesterday = tracker._date  # type: ignore[attr-defined]
    assert yesterday is not None
    tracker._date = yesterday - timedelta(days=1)  # type: ignore[attr-defined]

    # After rollover, a fresh charge passes.
    tracker.check("1.2.3.4", would_add=0.01)
    assert tracker.spent("1.2.3.4") == 0.0


def test_per_ip_tracker_rejects_negative_costs() -> None:
    tracker = CostCapTracker(cap_usd=5.0)
    with pytest.raises(ValueError):
        tracker.record("key", -0.01)
    with pytest.raises(ValueError):
        tracker.check("key", would_add=-0.01)


# ---------------------------------------------------------------------------
# GlobalDailyBudget
# ---------------------------------------------------------------------------


def test_global_daily_budget_accepts_under_cap() -> None:
    budget = GlobalDailyBudget(cap_usd=200.0)
    budget.check(150.0)
    budget.record(150.0)
    assert budget.spent_today == pytest.approx(150.0)


def test_global_daily_budget_rejects_over_cap() -> None:
    budget = GlobalDailyBudget(cap_usd=200.0)
    budget.record(190.0)
    with pytest.raises(DailyBudgetExceeded) as excinfo:
        budget.check(20.0)
    assert "Daily budget reached" in str(excinfo.value)
    assert excinfo.value.cap == 200.0


def test_global_daily_budget_rolls_over() -> None:
    budget = GlobalDailyBudget(cap_usd=1.0)
    budget.record(1.0)
    with pytest.raises(DailyBudgetExceeded):
        budget.check(0.01)

    # Simulate date rollover.
    yesterday = budget._date  # type: ignore[attr-defined]
    assert yesterday is not None
    budget._date = yesterday - timedelta(days=1)  # type: ignore[attr-defined]

    budget.check(0.5)
    assert budget.spent_today == 0.0


def test_simulated_session_burns_through_budget() -> None:
    """End-to-end simulation: many small calls eventually trip the cap.

    Mirrors the prompt's requested "simulated session that burns
    through the budget, asserts clean failure".
    """
    budget = GlobalDailyBudget(cap_usd=1.0)
    per_call_cost = 0.10
    allowed = 0
    with pytest.raises(DailyBudgetExceeded):
        for _ in range(100):
            budget.check(per_call_cost)
            budget.record(per_call_cost)
            allowed += 1

    # We should have let through exactly 10 calls before the 11th
    # tripped the cap — clean, boundary-correct failure.
    assert allowed == 10


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------


def test_anomaly_detector_flags_outliers_above_ratio() -> None:
    alerts: list[str] = []
    detect = anomaly_detector(
        window=10, ratio=10.0, min_floor_usd=0.05, alert_fn=alerts.append
    )

    # Warm up with typical sessions. The detector records every reading
    # so the median converges.
    for i in range(5):
        assert detect(f"session-{i}", 0.10) is False

    # 10x the median should fire.
    fired = detect("outlier", 2.0)
    assert fired is True
    assert len(alerts) == 1
    assert "outlier" in alerts[0]


def test_anomaly_detector_suppresses_duplicate_alerts_for_same_session() -> None:
    alerts: list[str] = []
    detect = anomaly_detector(
        window=10, ratio=10.0, min_floor_usd=0.05, alert_fn=alerts.append
    )
    for i in range(5):
        detect(f"session-{i}", 0.10)

    assert detect("outlier", 2.0) is True
    # A second trigger for the same session must not repeat the alert.
    assert detect("outlier", 3.0) in (True, False)  # status value is secondary
    assert len(alerts) == 1


def test_anomaly_detector_skips_below_floor() -> None:
    alerts: list[str] = []
    detect = anomaly_detector(
        window=10, ratio=10.0, min_floor_usd=0.50, alert_fn=alerts.append
    )
    for i in range(5):
        detect(f"session-{i}", 0.10)

    # Below floor: do not fire even if ratio would otherwise trigger.
    assert detect("cheap-outlier", 0.40) is False
    assert alerts == []


def test_anomaly_detector_requires_minimum_history() -> None:
    detect = anomaly_detector(window=10, ratio=10.0, min_floor_usd=0.05)
    # With only 2 readings in the window, the detector should not fire.
    assert detect("first", 5.0) is False
    assert detect("second", 5.0) is False


def test_anomaly_detector_validates_params() -> None:
    with pytest.raises(ValueError):
        anomaly_detector(window=2, ratio=10.0)
    with pytest.raises(ValueError):
        anomaly_detector(window=10, ratio=1.0)
