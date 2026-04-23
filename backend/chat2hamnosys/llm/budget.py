"""Per-session spend cap for LLM calls.

:class:`BudgetGuard` is a mutable, thread-safe tracker of cumulative USD
spend. It is consulted *before* each call fires via
:meth:`BudgetGuard.estimate_and_check` using a worst-case estimate
(approximate input tokens + the caller's ``max_tokens`` as output). If
the worst case would push past the cap, :class:`BudgetExceeded` is
raised and no network call leaves the process. After a successful call
the client invokes :meth:`BudgetGuard.record` with the actual cost to
update the running total.

The default cap is ``$20`` per session (bumped 10× from the original
$2 to give the SiGML-direct retry loop, slot-by-slot fallback, and
correction passes plenty of headroom on a single contributor session
before the guard fires). Override via the
``CHAT2HAMNOSYS_SESSION_BUDGET_USD`` env var or by passing
``max_usd_per_session=`` to the constructor.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from .pricing import compute_cost


DEFAULT_SESSION_BUDGET_USD = 20.0
SESSION_BUDGET_ENV_VAR = "CHAT2HAMNOSYS_SESSION_BUDGET_USD"


class BudgetExceeded(Exception):
    """A call would push cumulative session spend past the configured cap.

    Carries ``spent`` (USD billed so far this session), ``would_add``
    (the worst-case USD the blocked call would have added) and ``cap``
    (the configured limit) so callers can surface an informative error.
    """

    def __init__(self, spent: float, would_add: float, cap: float) -> None:
        super().__init__(
            f"session budget exceeded: ${spent:.4f} spent; "
            f"next call would add up to ${would_add:.4f}; cap is ${cap:.2f}"
        )
        self.spent = spent
        self.would_add = would_add
        self.cap = cap


def _default_cap_from_env() -> float:
    raw = os.environ.get(SESSION_BUDGET_ENV_VAR)
    if raw is None:
        return DEFAULT_SESSION_BUDGET_USD
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{SESSION_BUDGET_ENV_VAR}={raw!r} is not a valid float"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"{SESSION_BUDGET_ENV_VAR}={value} must be a positive number"
        )
    return value


@dataclass
class BudgetGuard:
    """Thread-safe cumulative-cost tracker with a hard session cap."""

    max_usd_per_session: float = field(default_factory=_default_cap_from_env)
    _spent: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.max_usd_per_session - self._spent)

    def estimate_and_check(
        self,
        *,
        model: str,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> float:
        """Raise :class:`BudgetExceeded` if worst-case cost would breach the cap.

        Returns the computed worst-case USD on success, so callers may
        log it alongside the pre-flight decision.
        """
        worst_case = compute_cost(model, estimated_input_tokens, max_output_tokens)
        with self._lock:
            if self._spent + worst_case > self.max_usd_per_session:
                raise BudgetExceeded(
                    spent=self._spent,
                    would_add=worst_case,
                    cap=self.max_usd_per_session,
                )
        return worst_case

    def record(self, cost_usd: float) -> None:
        """Add actual cost to the running total."""
        with self._lock:
            self._spent += cost_usd
