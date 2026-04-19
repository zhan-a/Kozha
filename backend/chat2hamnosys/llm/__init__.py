"""LLM client wrapper.

Every LLM call in chat2hamnosys flows through :class:`LLMClient`. The
client adds three concerns on top of the bare OpenAI SDK:

1. Retries with exponential backoff + jitter on 429 / 5xx / connection
   errors; falls back to a cheaper model once the primary has been
   rate-limited past the retry budget.
2. A per-session :class:`BudgetGuard` that rejects a call *before* it
   fires if the worst-case cost would breach the configured cap.
3. A :class:`TelemetryLogger` that appends a structured JSONL record
   per call to ``logs/llm/<YYYY-MM-DD>.jsonl`` — without prompt or
   completion content unless explicitly enabled.

Never hardcode API keys. The client resolves them from (explicit arg →
``OPENAI_API_KEY`` env var) and raises :class:`LLMConfigError` when
both are missing. Keys are never logged.
"""

from .budget import (
    DEFAULT_SESSION_BUDGET_USD,
    SESSION_BUDGET_ENV_VAR,
    BudgetExceeded,
    BudgetGuard,
)
from .client import (
    ChatResult,
    LLMClient,
    LLMConfigError,
    LLMError,
    ToolCall,
)
from .pricing import PRICING_PER_1M_TOKENS, compute_cost
from .telemetry import DEFAULT_LOG_DIR, TelemetryLogger

__all__ = [
    "BudgetExceeded",
    "BudgetGuard",
    "ChatResult",
    "DEFAULT_LOG_DIR",
    "DEFAULT_SESSION_BUDGET_USD",
    "LLMClient",
    "LLMConfigError",
    "LLMError",
    "PRICING_PER_1M_TOKENS",
    "SESSION_BUDGET_ENV_VAR",
    "TelemetryLogger",
    "ToolCall",
    "compute_cost",
]
