"""OpenAI chat-completions wrapper with retries, budget, and telemetry.

:class:`LLMClient` is the single point of contact for LLM calls in
chat2hamnosys. All prompt-engineering code (authoring dialogue,
validation hinting, tool-call flows) must go through it so spend
tracking and failure handling stay centralized.

Retry policy
------------
- Retries on :class:`openai.RateLimitError` (HTTP 429), HTTP 5xx
  server errors, and transient connection errors.
- Up to :attr:`LLMClient.max_retries` attempts on the primary ``model``
  with exponential backoff (``base_backoff * 2**attempt``) plus up to
  25% random jitter.
- If the primary still returns 429 after exhausting retries, one extra
  attempt is made against ``fallback_model``. On success the resulting
  :class:`ChatResult` has ``fallback_used=True``.
- 5xx / connection errors that exhaust retries propagate to the caller
  without triggering the fallback — the fallback is a rate-limit
  escape hatch, not a general retry.

Budget policy
-------------
- :class:`BudgetGuard` is consulted *before* each call fires, using a
  worst-case estimate (approx input tokens + ``max_tokens``). If the
  worst case would push past the cap, :class:`BudgetExceeded` is raised
  and no network call is made.
- After a successful call the actual cost is added to the running total
  via :meth:`BudgetGuard.record`.

Telemetry
---------
Every call (including failures) produces one JSONL record in
``logs/llm/<YYYY-MM-DD>.jsonl`` with ``request_id``, ``model``, token
counts, ``latency_ms``, ``cost_usd``, ``temperature``, ``success``
bool, and ``error_class`` if any. Prompt and completion content are
*not* written unless the :class:`TelemetryLogger` is constructed with
``log_content=True``.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        AsyncOpenAI,
        OpenAI,
        RateLimitError,
    )
except ImportError as _exc:  # pragma: no cover - openai is a hard dependency
    raise ImportError(
        "chat2hamnosys.llm requires the openai package — "
        "`pip install openai`."
    ) from _exc

from .budget import BudgetExceeded, BudgetGuard
from .pricing import compute_cost
from .telemetry import TelemetryLogger


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for LLM client errors surfaced by this package."""


class LLMConfigError(LLMError):
    """Raised when the client is misconfigured (missing key, bad arg, etc.)."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One tool invocation emitted by the model.

    ``arguments`` is the raw JSON string exactly as returned by the API
    — not parsed here, because the caller typically wants to validate
    it against its own schema before use.
    """

    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    """Outcome of one chat completion."""

    content: str
    tool_calls: list[ToolCall]
    model_used: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    request_id: str
    fallback_used: bool = False
    raw: Any = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_input_tokens(
    messages: Iterable[dict], tools: list[dict] | None
) -> int:
    """Rough input-token estimate for the pre-flight budget check.

    OpenAI's rule of thumb is ~4 characters per token for English. We
    add a floor and a small overhead so the budget guard errs on the
    side of blocking borderline calls — under-estimating input tokens
    here defeats the purpose of the guard.
    """
    total_chars = 0
    for m in messages:
        total_chars += len(json.dumps(m, ensure_ascii=False, default=str))
    if tools:
        for t in tools:
            total_chars += len(json.dumps(t, ensure_ascii=False, default=str))
    return max(10, total_chars // 4) + 10


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    raw = getattr(message, "tool_calls", None) or []
    out: list[ToolCall] = []
    for tc in raw:
        function = getattr(tc, "function", None)
        if function is None:
            continue
        out.append(
            ToolCall(
                id=getattr(tc, "id", "") or "",
                name=getattr(function, "name", "") or "",
                arguments=getattr(function, "arguments", "") or "",
            )
        )
    return out


def _is_server_error(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError):
        return 500 <= getattr(exc, "status_code", 0) < 600
    return False


def _is_retryable(exc: Exception) -> bool:
    return (
        isinstance(exc, RateLimitError)
        or isinstance(exc, APIConnectionError)
        or _is_server_error(exc)
    )


def _prompt_fields(prompt_metadata: Any) -> dict[str, Any]:
    """Extract telemetry fields from a ``PromptMetadata`` (or equivalent).

    Accepts a duck-typed object exposing ``id`` / ``version`` / ``hash``
    attributes — typically a :class:`prompts.PromptMetadata`. Returning
    ``{}`` when nothing is provided keeps call sites that don't use the
    versioned library free of noise.
    """
    if prompt_metadata is None:
        return {}
    return {
        "prompt_id": getattr(prompt_metadata, "id", None),
        "prompt_version": getattr(prompt_metadata, "version", None),
        "prompt_hash": getattr(prompt_metadata, "hash", None),
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """OpenAI chat-completions wrapper.

    Parameters
    ----------
    api_key:
        Explicit OpenAI key. Defaults to the ``OPENAI_API_KEY`` env
        var; raises :class:`LLMConfigError` if neither is set. Keys are
        never logged or echoed.
    model:
        Primary model id. Cost is computed against this model's entry
        in the pricing table; unknown ids fall back to ``gpt-4o``.
    fallback_model:
        Model to try once when the primary has been rate-limited past
        the retry budget.
    budget:
        Optional caller-supplied :class:`BudgetGuard`. If ``None``, a
        new guard is constructed with the default cap (or env
        override).
    telemetry:
        Optional caller-supplied :class:`TelemetryLogger`. If ``None``,
        a default logger writes to ``<repo>/logs/llm/<date>.jsonl``.
    client / async_client:
        Test seams — inject a fake SDK client. If unset, real
        ``OpenAI`` / ``AsyncOpenAI`` instances are built from the
        resolved API key.
    max_retries / base_backoff:
        Retry-loop tuning. Backoff delay is
        ``base_backoff * 2**attempt`` plus up to 25% jitter.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        fallback_model: str = "gpt-4o-mini",
        *,
        budget: BudgetGuard | None = None,
        telemetry: TelemetryLogger | None = None,
        client: Any = None,
        async_client: Any = None,
        max_retries: int = 3,
        base_backoff: float = 1.0,
    ) -> None:
        resolved = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not resolved:
            raise LLMConfigError(
                "no OpenAI API key: pass api_key= or set the OPENAI_API_KEY env var"
            )
        self._api_key = resolved
        self.model = model
        self.fallback_model = fallback_model
        self.budget = budget if budget is not None else BudgetGuard()
        self.telemetry = telemetry if telemetry is not None else TelemetryLogger()
        self._client = client if client is not None else OpenAI(api_key=resolved)
        self._async_client = async_client
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.last_stream_result: ChatResult | None = None

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.model!r}, fallback_model={self.fallback_model!r}, "
            f"max_retries={self.max_retries})"
        )

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        *,
        request_id: str,
        prompt_metadata: Any = None,
    ) -> ChatResult:
        if not request_id:
            raise LLMConfigError("request_id is required and must be non-empty")

        approx_input = _approx_input_tokens(messages, tools)
        self.budget.estimate_and_check(
            model=self.model,
            estimated_input_tokens=approx_input,
            max_output_tokens=max_tokens,
        )

        kwargs = self._build_kwargs(
            messages, tools, response_format, temperature, max_tokens
        )
        response, model_used, latency_ms, fallback_used = self._invoke_with_retry(
            kwargs=kwargs,
            request_id=request_id,
            temperature=temperature,
            prompt_metadata=prompt_metadata,
        )

        result = self._build_result(
            response=response,
            model_used=model_used,
            latency_ms=latency_ms,
            request_id=request_id,
            fallback_used=fallback_used,
        )

        prompt_fields = _prompt_fields(prompt_metadata)
        self.budget.record(result.cost_usd)
        self.telemetry.log_call(
            request_id=request_id,
            model=result.model_used,
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            temperature=temperature,
            success=True,
            fallback_used=fallback_used,
            prompt_content=messages if self.telemetry.log_content else None,
            completion_content=result.content if self.telemetry.log_content else None,
            **prompt_fields,
        )
        return result

    async def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        *,
        request_id: str,
        prompt_metadata: Any = None,
    ) -> AsyncIterator[str]:
        """Yield content-delta strings as they arrive.

        Budget pre-check happens at start and telemetry is written once
        the stream completes. Retries only apply to the initial
        handshake — mid-stream failures propagate to the caller. The
        final aggregated :class:`ChatResult` is exposed via
        ``self.last_stream_result`` for post-hoc inspection.
        """
        if not request_id:
            raise LLMConfigError("request_id is required and must be non-empty")

        approx_input = _approx_input_tokens(messages, tools)
        self.budget.estimate_and_check(
            model=self.model,
            estimated_input_tokens=approx_input,
            max_output_tokens=max_tokens,
        )

        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)

        kwargs = self._build_kwargs(
            messages, tools, response_format, temperature, max_tokens, stream=True
        )
        kwargs["model"] = self.model

        start = time.perf_counter()
        prompt_fields = _prompt_fields(prompt_metadata)
        try:
            stream = await self._async_client.chat.completions.create(**kwargs)
        except Exception as exc:
            self.telemetry.log_call(
                request_id=request_id,
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=int((time.perf_counter() - start) * 1000),
                cost_usd=0.0,
                temperature=temperature,
                success=False,
                error_class=type(exc).__name__,
                **prompt_fields,
            )
            raise

        content_parts: list[str] = []
        usage = None
        model_used = self.model
        async for chunk in stream:
            if getattr(chunk, "model", None):
                model_used = chunk.model
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    content_parts.append(piece)
                    yield piece
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

        latency_ms = int((time.perf_counter() - start) * 1000)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
        )
        cost = compute_cost(model_used, input_tokens, output_tokens)

        self.last_stream_result = ChatResult(
            content="".join(content_parts),
            tool_calls=[],
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            request_id=request_id,
        )

        self.budget.record(cost)
        self.telemetry.log_call(
            request_id=request_id,
            model=model_used,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            temperature=temperature,
            success=True,
            prompt_content=messages if self.telemetry.log_content else None,
            completion_content=self.last_stream_result.content
            if self.telemetry.log_content
            else None,
            **prompt_fields,
        )

    # -- internals ---------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        response_format: dict | None,
        temperature: float,
        max_tokens: int,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _invoke_with_retry(
        self,
        *,
        kwargs: dict[str, Any],
        request_id: str,
        temperature: float,
        prompt_metadata: Any = None,
    ) -> tuple[Any, str, int, bool]:
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.model, **kwargs
                )
                latency_ms = int((time.perf_counter() - start) * 1000)
                return response, self.model, latency_ms, False
            except Exception as exc:
                if not _is_retryable(exc):
                    self._log_failure(
                        request_id, self.model, temperature, exc, prompt_metadata
                    )
                    raise
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self._backoff(attempt))

        if isinstance(last_exc, RateLimitError):
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.fallback_model, **kwargs
                )
                latency_ms = int((time.perf_counter() - start) * 1000)
                return response, self.fallback_model, latency_ms, True
            except Exception as exc:
                self._log_failure(
                    request_id,
                    self.fallback_model,
                    temperature,
                    exc,
                    prompt_metadata,
                )
                raise

        assert last_exc is not None
        self._log_failure(
            request_id, self.model, temperature, last_exc, prompt_metadata
        )
        raise last_exc

    def _build_result(
        self,
        *,
        response: Any,
        model_used: str,
        latency_ms: int,
        request_id: str,
        fallback_used: bool,
    ) -> ChatResult:
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", "") or ""
        tool_calls = _extract_tool_calls(message)
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
        )
        cost = compute_cost(model_used, input_tokens, output_tokens)
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            request_id=request_id,
            fallback_used=fallback_used,
            raw=response,
        )

    def _backoff(self, attempt: int) -> float:
        base = self.base_backoff * (2 ** attempt)
        return base + random.uniform(0, base * 0.25)

    def _log_failure(
        self,
        request_id: str,
        model: str,
        temperature: float,
        exc: Exception,
        prompt_metadata: Any = None,
    ) -> None:
        self.telemetry.log_call(
            request_id=request_id,
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
            cost_usd=0.0,
            temperature=temperature,
            success=False,
            error_class=type(exc).__name__,
            **_prompt_fields(prompt_metadata),
        )
