"""Per-IP and global daily cost caps, plus anomaly detection.

The :class:`~chat2hamnosys.llm.BudgetGuard` from Prompt 4 caps spend
*per session*. That is the right unit for individual-author abuse but
leaves two holes:

1. An attacker who enumerates sessions (cheap — `POST /sessions` has
   no auth beyond rate limiting) can chain session-sized budgets into
   arbitrary total spend from one IP.
2. A bug in the retry loop or a runaway loop anywhere in the backend
   could burn the whole OpenAI monthly allowance in minutes.

:class:`CostCapTracker` layers a per-IP daily cap on top of the
session guard. :class:`GlobalDailyBudget` is a single-process
ceiling on daily spend across **all** sessions and IPs — when hit,
the server rejects further LLM calls with :class:`DailyBudgetExceeded`
("Daily budget reached, please try again tomorrow.").

Both reset at the UTC date boundary. Both are thread-safe. Both are
**process-local** — they are a first line of defense, not a
distributed budget. A multi-replica deployment will need Redis-backed
counters; this module documents where to extend.

The :func:`anomaly_detector` helper tracks a rolling median of
per-session spend and flags any single session whose running total
exceeds ``10× median``. It does not block; it calls the supplied
``alert_fn`` (Slack webhook, email, etc.) so an operator can
investigate.
"""

from __future__ import annotations

import logging
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CostCapExceeded(Exception):
    """Raised when a per-scope daily cost cap would be breached."""

    def __init__(self, scope: str, key: str, spent: float, would_add: float, cap: float) -> None:
        super().__init__(
            f"{scope} daily cost cap exceeded for {key!r}: spent=${spent:.4f}, "
            f"would_add=${would_add:.4f}, cap=${cap:.2f}"
        )
        self.scope = scope
        self.key = key
        self.spent = spent
        self.would_add = would_add
        self.cap = cap


class DailyBudgetExceeded(Exception):
    """Raised when the global daily budget would be breached."""

    def __init__(self, spent: float, would_add: float, cap: float) -> None:
        super().__init__(
            f"global daily budget exceeded: spent=${spent:.4f}, "
            f"would_add=${would_add:.4f}, cap=${cap:.2f}. "
            "Daily budget reached, please try again tomorrow."
        )
        self.spent = spent
        self.would_add = would_add
        self.cap = cap


# ---------------------------------------------------------------------------
# Per-IP tracker
# ---------------------------------------------------------------------------


@dataclass
class CostCapTracker:
    """Tracks cumulative USD spend per key (IP, session, user, etc.).

    The ``scope`` tag is used only in error messages; it lets one class
    stand in for "per-IP" and "per-user" (or other) caps. Counters reset
    on UTC date change — we keep a ``_date`` marker and zero out when
    the current UTC date differs.
    """

    cap_usd: float
    scope: str = "per-key"
    _totals: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _date: date | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def _rollover_if_new_day(self, now: datetime) -> None:
        today = now.date()
        if self._date != today:
            self._totals.clear()
            self._date = today

    def spent(self, key: str) -> float:
        with self._lock:
            now = datetime.now(timezone.utc)
            self._rollover_if_new_day(now)
            return self._totals.get(key, 0.0)

    def check(self, key: str, would_add: float) -> None:
        """Raise :class:`CostCapExceeded` if ``would_add`` would breach the cap."""
        if would_add < 0:
            raise ValueError("would_add must be non-negative")
        with self._lock:
            now = datetime.now(timezone.utc)
            self._rollover_if_new_day(now)
            spent = self._totals.get(key, 0.0)
            if spent + would_add > self.cap_usd:
                raise CostCapExceeded(
                    scope=self.scope,
                    key=key,
                    spent=spent,
                    would_add=would_add,
                    cap=self.cap_usd,
                )

    def record(self, key: str, cost: float) -> None:
        """Add ``cost`` to the running total for ``key``. Negative amounts rejected."""
        if cost < 0:
            raise ValueError("cost must be non-negative")
        with self._lock:
            now = datetime.now(timezone.utc)
            self._rollover_if_new_day(now)
            self._totals[key] = self._totals.get(key, 0.0) + cost


# ---------------------------------------------------------------------------
# Global budget
# ---------------------------------------------------------------------------


@dataclass
class GlobalDailyBudget:
    """Single-process daily spend ceiling across all sessions/IPs."""

    cap_usd: float
    _spent: float = field(default=0.0, init=False, repr=False)
    _date: date | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def _rollover_if_new_day(self, now: datetime) -> None:
        today = now.date()
        if self._date != today:
            self._spent = 0.0
            self._date = today

    @property
    def spent_today(self) -> float:
        with self._lock:
            now = datetime.now(timezone.utc)
            self._rollover_if_new_day(now)
            return self._spent

    def check(self, would_add: float) -> None:
        if would_add < 0:
            raise ValueError("would_add must be non-negative")
        with self._lock:
            now = datetime.now(timezone.utc)
            self._rollover_if_new_day(now)
            if self._spent + would_add > self.cap_usd:
                raise DailyBudgetExceeded(
                    spent=self._spent, would_add=would_add, cap=self.cap_usd
                )

    def record(self, cost: float) -> None:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        with self._lock:
            now = datetime.now(timezone.utc)
            self._rollover_if_new_day(now)
            self._spent += cost


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def anomaly_detector(
    *,
    window: int = 50,
    ratio: float = 10.0,
    min_floor_usd: float = 0.05,
    alert_fn: Callable[[str], None] | None = None,
) -> Callable[[str, float], bool]:
    """Factory returning a session-spend anomaly detector closure.

    The returned callable takes ``(session_id, current_spend_usd)`` and
    returns ``True`` when ``current_spend >= ratio * median(window)``
    and ``current_spend >= min_floor_usd``. When ``True``, and an
    ``alert_fn`` was supplied, the alert function is invoked once per
    anomalous session id (tracked internally so alerts are not
    spammed).

    ``min_floor_usd`` prevents noise in the early minutes — a median
    of ``$0.001`` would trip on almost any session.
    """
    if window < 3:
        raise ValueError("window must be >= 3 for a meaningful median")
    if ratio <= 1.0:
        raise ValueError("ratio must be > 1.0")

    recent: deque[float] = deque(maxlen=window)
    alerted: set[str] = set()
    lock = threading.Lock()

    def detect(session_id: str, current_spend: float) -> bool:
        with lock:
            if current_spend < min_floor_usd:
                recent.append(current_spend)
                return False
            if len(recent) < 3:
                recent.append(current_spend)
                return False
            median = statistics.median(recent)
            threshold = max(min_floor_usd, ratio * median)
            recent.append(current_spend)
            if current_spend >= threshold and session_id not in alerted:
                alerted.add(session_id)
                message = (
                    f"cost anomaly: session {session_id} spent "
                    f"${current_spend:.4f} which is {current_spend / max(median, 1e-9):.1f}× "
                    f"the rolling median of ${median:.4f} over last {len(recent)} sessions"
                )
                logger.warning(message)
                if alert_fn is not None:
                    try:
                        alert_fn(message)
                    except Exception as exc:  # alert delivery must never raise
                        logger.warning("anomaly alert_fn failed: %s", exc)
                return True
            return False

    return detect


__all__ = [
    "CostCapExceeded",
    "CostCapTracker",
    "DailyBudgetExceeded",
    "GlobalDailyBudget",
    "anomaly_detector",
]
