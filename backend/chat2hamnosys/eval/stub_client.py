"""Deterministic stub for the :class:`LLMClient` used in CI smoke runs.

Not a mock — this is a real object the runner calls ``chat()`` on. It
returns JSON content that parses cleanly under each stage's strict
schema but contributes nothing new (empty params, no gaps, empty
repair). That way the harness machinery — suite loading, runner
threading, metric aggregation, report rendering — is exercised end-to-
end without talking to OpenAI, and without requiring an ``OPENAI_API_KEY``
in the CI environment.

Semantics
---------
- **Parser** requests return ``{"parameters": {}, "gaps": []}`` so
  :func:`parser.description_parser.parse_description` succeeds. The
  downstream metrics show the parser as populating no slots, which
  surfaces as low accuracy — the *expected* CI shape, since we want the
  harness to run, not a pretend-real eval.
- **Clarifier** requests return ``{"questions": []}`` so
  :func:`clarify.generate_questions` finishes without simulated
  dialogue. (``run_suite`` still records zero clarification turns.)
- **Generator slot fallback** returns an empty codepoint with low
  confidence so :func:`generator.params_to_hamnosys.generate` treats
  the slot as unresolvable and falls back to the vocab/validator path.
- **Generator repair** returns an empty string so the repair loop
  gives up immediately. Combined with the ablation toggles this is a
  safe no-op.

Dispatch is based on substrings in ``request_id`` — the runner already
uses stable suffixes (``:parse``, ``:clarify``, ``:fallback``,
``:repair``) so reading that tail costs nothing. Any unrecognized
request returns ``{}`` — callers that happen to accept empty JSON keep
going, the rest raise and are handled by their own fallback paths.
"""

from __future__ import annotations

import json
from typing import Any

from llm import ChatResult


class StubLLMClient:
    """A minimal ``client.chat()`` implementation for eval smoke runs.

    Attributes
    ----------
    model :
        Matches :class:`LLMClient.model` — read by the runner when
        snapshotting the report's model string.
    budget / telemetry :
        Duck-typed stubs. The runner's :class:`_TallyingClient` forwards
        attribute access to these when the wrapped code (e.g.
        ``client.budget.record(...)``) happens to reach them.
    """

    def __init__(self, model: str = "stub-llm") -> None:
        self.model = model
        self.budget = _NullBudget()
        self.telemetry = _NullTelemetry()
        self._counter = 0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        *,
        request_id: str,
        prompt_metadata: Any = None,
    ) -> ChatResult:
        self._counter += 1
        content = self._route(request_id, messages)
        return ChatResult(
            content=content,
            tool_calls=[],
            model_used=self.model,
            input_tokens=0,
            output_tokens=len(content) // 4,
            total_tokens=len(content) // 4,
            cost_usd=0.0,
            latency_ms=1,
            request_id=request_id,
            fallback_used=False,
            raw=None,
        )

    def _route(self, request_id: str, _messages: list[dict]) -> str:
        rid = request_id or ""
        if ":repair:" in rid:
            return json.dumps(
                {"hamnosys_hex": "", "rationale": "stub: repair disabled"}
            )
        if ":fallback:" in rid:
            return json.dumps(
                {
                    "codepoint_hex": "",
                    "confidence": 0.0,
                    "rationale": "stub: llm fallback disabled",
                }
            )
        if ":clarify" in rid:
            return json.dumps({"questions": []})
        if rid.endswith(":parse") or ":e2e:parse" in rid:
            return json.dumps({"parameters": {}, "gaps": []})
        return "{}"


class _NullBudget:
    """No-op stand-in for :class:`BudgetGuard` inside the stub client."""

    def estimate_and_check(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None

    def record(self, _cost: float) -> None:
        return None


class _NullTelemetry:
    """No-op stand-in for :class:`TelemetryLogger`."""

    log_content = False

    def log_call(self, *_args: Any, **_kwargs: Any) -> None:
        return None


__all__ = ["StubLLMClient"]
