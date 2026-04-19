"""Security hardening for the chat2hamnosys authoring surface.

Four concerns live here, in layers applied at the request boundary:

1. :mod:`.sanitize` — length caps, control-character stripping, Unicode
   normalization, and explicit ``<tag>``-wrapping of untrusted content.
2. :mod:`.injection` — regex fast-path plus optional LLM classifier that
   screens descriptions for prompt-injection payloads before they reach
   the main parser.
3. :mod:`.moderation` — OpenAI Moderation-endpoint filter applied to
   user-facing LLM outputs (clarification questions, correction
   explanations).
4. :mod:`.rate_limit` — per-IP and global daily cost caps plus the
   anomaly-alert hook.

Import the high-level helpers directly from the package; the submodules
remain importable for tests and for the occasional bespoke call.
"""

from __future__ import annotations

from .config import PIIPolicy, SecurityConfig, load_security_config
from .injection import (
    InjectionClassifier,
    InjectionResult,
    InjectionVerdict,
    RegexInjectionScreen,
    screen_description,
)
from .moderation import ModerationResult, moderate_output
from .pii import hash_signer_id, strip_signer_identifiers
from .rate_limit import (
    CostCapExceeded,
    CostCapTracker,
    DailyBudgetExceeded,
    GlobalDailyBudget,
    anomaly_detector,
)
from .sanitize import (
    InputTooLong,
    sanitize_for_prompt,
    wrap_user_content,
)


__all__ = [
    "CostCapExceeded",
    "CostCapTracker",
    "DailyBudgetExceeded",
    "GlobalDailyBudget",
    "InjectionClassifier",
    "InjectionResult",
    "InjectionVerdict",
    "InputTooLong",
    "ModerationResult",
    "PIIPolicy",
    "RegexInjectionScreen",
    "SecurityConfig",
    "anomaly_detector",
    "hash_signer_id",
    "load_security_config",
    "moderate_output",
    "sanitize_for_prompt",
    "screen_description",
    "strip_signer_identifiers",
    "wrap_user_content",
]
