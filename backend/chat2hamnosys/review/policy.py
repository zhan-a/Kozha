"""Reviewer-policy configuration for the Deaf-review workflow.

The :class:`ReviewPolicy` object captures every knob the legitimacy
backbone exposes — how many native-Deaf approvals are needed before a
sign can graduate from ``pending_review`` to ``validated``, whether
single-approval bootstrap mode is permitted (and warned about), and the
session-fatigue threshold that the frontend uses to nudge breaks.

Policy values are read from environment variables on construction so an
operator can re-tune the workflow without code changes — but the
defaults match the feasibility-study recommendation: two independent
native-Deaf approvers per sign, single-approval mode disabled.

Environment variables (all optional)
-----------------------------------
- ``CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS``     — integer, default ``2``.
- ``CHAT2HAMNOSYS_REVIEW_REQUIRE_NATIVE``    — ``true``/``false``, default ``true``.
- ``CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE``      — ``true``/``false``, default ``false``;
  enabling it logs a warning on every approval that lands a sign on this path.
- ``CHAT2HAMNOSYS_REVIEW_FATIGUE_THRESHOLD`` — integer, default ``25``.
- ``CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH`` — ``true``/``false``, default ``true``;
  if true, approvals only count when the reviewer's regional background matches
  the sign's ``regional_variant`` (or the sign has no variant set).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ReviewPolicy:
    """Frozen configuration for the reviewer workflow.

    The dataclass is frozen so a policy is value-typed — comparing two
    policies is structural equality and tests can build ad-hoc policies
    without monkeypatching env vars.
    """

    min_approvals: int = 2
    require_native_deaf: bool = True
    allow_single_approval: bool = False
    fatigue_threshold: int = 25
    require_region_match: bool = True

    @classmethod
    def from_env(cls) -> "ReviewPolicy":
        """Read every knob from the environment, applying the defaults."""
        return cls(
            min_approvals=max(1, _env_int("CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS", 2)),
            require_native_deaf=_env_bool("CHAT2HAMNOSYS_REVIEW_REQUIRE_NATIVE", True),
            allow_single_approval=_env_bool(
                "CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE", False
            ),
            fatigue_threshold=max(
                1, _env_int("CHAT2HAMNOSYS_REVIEW_FATIGUE_THRESHOLD", 25)
            ),
            require_region_match=_env_bool(
                "CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH", True
            ),
        )

    def effective_min_approvals(self) -> int:
        """Threshold actually enforced — accounts for single-approval mode.

        When ``allow_single_approval`` is set we drop the floor to 1 even
        if the operator left ``min_approvals`` higher; that combination
        is always a bootstrap setup and we shouldn't pretend otherwise.
        """
        if self.allow_single_approval:
            return 1
        return self.min_approvals

    def warn_if_bootstrap(self) -> None:
        """Emit a warning when single-approval mode is in use.

        Called every time a single approval lands a sign in ``validated``;
        the recurring log entry is the audit trail that nudges operators
        back toward two-reviewer mode once they have enough native Deaf
        reviewers onboarded.
        """
        if self.allow_single_approval:
            logger.warning(
                "review policy is in single-approval bootstrap mode "
                "(CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE=true) — "
                "this is not the legitimacy default; switch off "
                "once two native-Deaf reviewers are available"
            )


__all__ = ["ReviewPolicy"]
