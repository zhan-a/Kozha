"""OpenAI per-model pricing, in USD per 1M tokens.

The table is populated at module import and read by :func:`compute_cost`
to translate raw usage counts into dollar cost. Rates here are a
snapshot of OpenAI's published pricing — update when OpenAI revises the
price list. Unknown model ids fall back to ``gpt-4o`` rates so a typo
in a model name can't silently record ``$0.00`` in the session budget.
"""

from __future__ import annotations


PRICING_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50,  "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":       {"input": 10.00, "output": 30.00},
    "gpt-4":             {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
    "o1":                {"input": 15.00, "output": 60.00},
    "o1-mini":           {"input": 3.00,  "output": 12.00},
    "o1-preview":        {"input": 15.00, "output": 60.00},
}

_FALLBACK_MODEL_FOR_UNKNOWN = "gpt-4o"


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a single call given usage counts.

    An unknown ``model`` falls back to ``gpt-4o`` rates. The alternative
    (raising) would break :class:`LLMClient` for any newly-released
    model before the pricing table is updated — the wrong failure mode
    for a cost-tracking concern, whose job is to err on the side of
    over-charging rather than silently under-charging.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    rates = (
        PRICING_PER_1M_TOKENS.get(model)
        or PRICING_PER_1M_TOKENS[_FALLBACK_MODEL_FOR_UNKNOWN]
    )
    return (
        (input_tokens / 1_000_000) * rates["input"]
        + (output_tokens / 1_000_000) * rates["output"]
    )
