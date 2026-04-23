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

import contextvars
import json
import os
import random
import re
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

from obs import events as _evs
from obs import metrics as _metrics
from obs.logger import emit_event

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
# Request-scoped OpenAI API key
# ---------------------------------------------------------------------------
#
# The API layer can set a per-request key via ``set_request_openai_api_key``
# — typically from an ``X-OpenAI-Api-Key`` header. :class:`LLMClient`
# then uses it in preference to the ``OPENAI_API_KEY`` env var, so
# contributors can provide a personal key on the website when the
# project's own secret hasn't been provisioned yet. An explicit
# ``api_key=`` argument still wins over both — that path is used by
# tests and by the parser/generator fixture recorders.
# ---------------------------------------------------------------------------


_REQUEST_OPENAI_API_KEY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chat2hamnosys_request_openai_api_key", default=""
)


def set_request_openai_api_key(key: str) -> "contextvars.Token[str]":
    """Set the per-request OpenAI key. Return a token for :func:`reset`."""
    return _REQUEST_OPENAI_API_KEY.set(key or "")


def reset_request_openai_api_key(token: "contextvars.Token[str]") -> None:
    _REQUEST_OPENAI_API_KEY.reset(token)


def _resolve_api_key(explicit: str | None) -> str:
    """Pick the OpenAI key to use: explicit > request contextvar > env."""
    if explicit is not None:
        return explicit
    byo = _REQUEST_OPENAI_API_KEY.get("")
    if byo:
        return byo
    return os.environ.get("OPENAI_API_KEY", "")


# Strongest general-purpose model as of this package's pricing table
# snapshot. Override at deploy time with ``CHAT2HAMNOSYS_MODEL`` if the
# project key lacks access to the default. The fallback model is what
# the retry path switches to when the primary is rate-limited past its
# retry budget — stay on a widely-available tier there.
_DEFAULT_MODEL_ENV = "CHAT2HAMNOSYS_MODEL"
_DEFAULT_FALLBACK_MODEL_ENV = "CHAT2HAMNOSYS_FALLBACK_MODEL"
# Default to the current strongest general-purpose model. ``gpt-5.4``
# is the live frontier model (reasoning.effort defaults to ``none``, so
# ``temperature`` and other sampling params remain valid); ``gpt-4o``
# stays as the fallback model that :meth:`_invoke_with_retry` swaps in
# when the primary is unavailable or rate-limited past its budget.
_BUILTIN_DEFAULT_MODEL = "gpt-5.4"
_BUILTIN_DEFAULT_FALLBACK_MODEL = "gpt-4o"

# Per-request timeout in seconds for the OpenAI SDK. The previous 30s
# cap was a hard ceiling tuned for plain chat completions — it
# routinely cut off ``gpt-5.4`` reasoning mid-thought, surfacing as
# the user-visible "AI had trouble generating a follow-up" error that
# disappeared as soon as the page was reloaded (the retry on resume
# completed within the new, generous budget). Reasoning models can
# legitimately think for several minutes on a complex SiGML emit —
# 5 minutes per call is the right ceiling: enough that "think hard
# and produce a good sign" wins, low enough that a genuinely hung
# request still surfaces as an error before the contributor leaves.
# Override at deploy time with ``CHAT2HAMNOSYS_REQUEST_TIMEOUT_S`` if
# the model needs even more rope. We still disable the SDK's own
# retry loop (``max_retries=0``) so ``_invoke_with_retry`` is the
# single owner of retry policy and the model-unavailable fallback
# path.
_OPENAI_REQUEST_TIMEOUT_S: float = float(
    os.environ.get("CHAT2HAMNOSYS_REQUEST_TIMEOUT_S", "300.0")
)
# Reasoning-capable models go through the Responses API rather than the
# legacy Chat Completions endpoint. The Responses API is OpenAI's
# recommended surface for gpt-5.x and the o-series — it accepts
# ``reasoning.effort``, returns reasoning summaries, and uses the
# correct ``max_output_tokens`` parameter. The legacy chat endpoint
# rejects some of those and silently degrades others, which surfaced as
# the BadRequestError storm in the contribute pipeline. Match anything
# that looks like a reasoning model so additions like ``gpt-5.4-pro``
# or ``o4-mini`` route correctly without further touch.
_REASONING_MODEL_RE = re.compile(
    r"^(?:gpt-5(?:\.\d+)?(?:-mini|-nano|-pro)?|o[1-4](?:-mini|-pro)?)(?:-\d+.*)?$",
    re.IGNORECASE,
)


def is_reasoning_model(name: str) -> bool:
    """Return True if ``name`` is a reasoning-capable OpenAI model."""
    return bool(name) and bool(_REASONING_MODEL_RE.match(name))


def _default_model() -> str:
    return os.environ.get(_DEFAULT_MODEL_ENV, _BUILTIN_DEFAULT_MODEL)


def _default_fallback_model() -> str:
    return os.environ.get(
        _DEFAULT_FALLBACK_MODEL_ENV, _BUILTIN_DEFAULT_FALLBACK_MODEL
    )


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


def _is_model_unavailable(exc: Exception) -> bool:
    """Heuristic: did OpenAI reject this request because the model is not
    available on the active API key?

    Covers the 404 ``model_not_found`` path and the 400 ``invalid_model``
    path — both shapes OpenAI has used at various points. We never want
    to retry blindly on arbitrary 4xx errors (that path is a bug to
    surface), so we look specifically for the "model" substring in the
    error body. Used to trigger the one-shot fallback in
    :meth:`LLMClient._invoke_with_retry`.
    """
    if not isinstance(exc, APIStatusError):
        return False
    status = getattr(exc, "status_code", 0) or 0
    if status not in (400, 403, 404):
        return False
    body = str(exc).lower()
    return (
        "model" in body
        and (
            "not_found" in body
            or "not found" in body
            or "does not exist" in body
            or "invalid_model" in body
            or "unsupported_model" in body
            or "does not have access" in body
        )
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
        model: str | None = None,
        fallback_model: str | None = None,
        *,
        budget: BudgetGuard | None = None,
        telemetry: TelemetryLogger | None = None,
        client: Any = None,
        async_client: Any = None,
        max_retries: int = 3,
        base_backoff: float = 1.0,
    ) -> None:
        resolved = _resolve_api_key(api_key)
        if not resolved:
            raise LLMConfigError(
                "no OpenAI API key: pass api_key= or set the OPENAI_API_KEY env var"
            )
        self._api_key = resolved
        self.model = model if model is not None else _default_model()
        self.fallback_model = (
            fallback_model
            if fallback_model is not None
            else _default_fallback_model()
        )
        self.budget = budget if budget is not None else BudgetGuard()
        self.telemetry = telemetry if telemetry is not None else TelemetryLogger()
        self._client = (
            client
            if client is not None
            else OpenAI(
                api_key=resolved,
                timeout=_OPENAI_REQUEST_TIMEOUT_S,
                max_retries=0,
            )
        )
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
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        if not request_id:
            raise LLMConfigError("request_id is required and must be non-empty")

        # Reasoning models go through the Responses API. We translate to
        # the legacy ChatResult so every caller stays unchanged.
        if is_reasoning_model(self.model) and tools is None:
            return self._respond(
                messages=messages,
                response_format=response_format,
                max_output_tokens=max_tokens,
                request_id=request_id,
                prompt_metadata=prompt_metadata,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )

        approx_input = _approx_input_tokens(messages, tools)
        self.budget.estimate_and_check(
            model=self.model,
            estimated_input_tokens=approx_input,
            max_output_tokens=max_tokens,
        )

        emit_event(
            _evs.LLM_CALL_STARTED,
            request_id=request_id,
            model=self.model,
            temperature=temperature,
            approx_input_tokens=approx_input,
            max_output_tokens=max_tokens,
        )

        kwargs = self._build_kwargs(
            messages, tools, response_format, temperature, max_tokens
        )
        try:
            response, model_used, latency_ms, fallback_used = (
                self._invoke_with_retry(
                    kwargs=kwargs,
                    request_id=request_id,
                    temperature=temperature,
                    prompt_metadata=prompt_metadata,
                )
            )
        except Exception as exc:
            emit_event(
                _evs.LLM_CALL_FAILED,
                request_id=request_id,
                model=self.model,
                error_class=type(exc).__name__,
                temperature=temperature,
            )
            _metrics.llm_calls_total.inc(self.model, "failure")
            raise

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
        _metrics.llm_calls_total.inc(result.model_used, "success")
        _metrics.llm_call_latency_ms.observe(
            result.model_used, value=float(result.latency_ms)
        )
        _metrics.llm_call_cost_usd.observe(value=float(result.cost_usd))
        _metrics.daily_cost_usd.observe(value=float(result.cost_usd))
        if fallback_used:
            emit_event(
                _evs.LLM_CALL_FALLBACK_USED,
                request_id=request_id,
                model=result.model_used,
                primary_model=self.model,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
            )
        emit_event(
            _evs.LLM_CALL_SUCCEEDED,
            request_id=request_id,
            model=result.model_used,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            fallback_used=fallback_used,
        )
        return result

    # -- Responses API path (reasoning models) -----------------------------

    def _respond(
        self,
        *,
        messages: list[dict],
        response_format: dict | None,
        max_output_tokens: int,
        request_id: str,
        prompt_metadata: Any,
        reasoning_effort: str | None,
        temperature: float,
    ) -> ChatResult:
        """Reasoning-model path via the Responses API.

        Translates the legacy ``messages`` + ``response_format`` shape
        used elsewhere in this codebase to the Responses API's
        ``instructions`` + ``input`` + ``text.format`` shape, runs the
        call, and returns a :class:`ChatResult` so the caller can stay
        completely API-agnostic.

        Reasoning effort defaults to ``medium`` for structured-output
        calls (the slot resolver, the SiGML-direct generator) and to
        ``low`` otherwise — the bias is toward correctness on the paths
        that actually need to follow a schema, with one cache miss
        amortised across the response. Pass an explicit
        ``reasoning_effort`` to override.
        """
        # Split out the system prompt as ``instructions`` — every call
        # site in this codebase uses at most one system message at the
        # start. Anything else stays in the input array as-is.
        instructions: str | None = None
        input_items: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system" and instructions is None:
                instructions = msg.get("content") or ""
                continue
            input_items.append({"role": role or "user", "content": msg.get("content") or ""})

        approx_input = _approx_input_tokens(messages, None)
        self.budget.estimate_and_check(
            model=self.model,
            estimated_input_tokens=approx_input,
            max_output_tokens=max_output_tokens,
        )

        effort = reasoning_effort or ("medium" if response_format else "low")
        emit_event(
            _evs.LLM_CALL_STARTED,
            request_id=request_id,
            model=self.model,
            temperature=temperature,
            approx_input_tokens=approx_input,
            max_output_tokens=max_output_tokens,
        )

        # The Responses API moves structured output under ``text.format``
        # with the schema fields hoisted to top level — Chat Completions
        # nests them inside ``json_schema``. Translate so call sites can
        # pass either shape interchangeably.
        text_format: dict[str, Any] | None = None
        if response_format and response_format.get("type") == "json_schema":
            schema_block = response_format.get("json_schema") or {}
            text_format = {
                "format": {
                    "type": "json_schema",
                    "name": schema_block.get("name") or "response",
                    "strict": bool(schema_block.get("strict", True)),
                    "schema": schema_block.get("schema") or {},
                }
            }

        # Reasoning models burn output tokens on the hidden chain-of-
        # thought before they emit a single visible character — the
        # OpenAI guide recommends reserving "at least 25,000 tokens"
        # per request when starting out. We pick a roomy floor of
        # 32,768 plus the caller-requested visible budget so the model
        # never gets cut off mid-thought, which was the silent cause
        # of the "AI had trouble generating a follow-up" errors that
        # vanished after a reload (the resume call ran with fresh
        # budget and finished cleanly).
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max(int(max_output_tokens) + 4096, 32768),
            "reasoning": {"effort": effort},
        }
        if instructions:
            kwargs["instructions"] = instructions
        if text_format is not None:
            kwargs["text"] = text_format

        last_exc: Exception | None = None
        response: Any = None
        model_used = self.model
        latency_ms = 0
        fallback_used = False
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                response = self._client.responses.create(**kwargs)
                latency_ms = int((time.perf_counter() - start) * 1000)
                break
            except Exception as exc:
                if _is_model_unavailable(exc) and self.fallback_model != self.model:
                    # Fall back to the legacy chat completions API on
                    # the fallback model — that path doesn't require
                    # reasoning support.
                    return self._respond_via_chat_fallback(
                        messages=messages,
                        response_format=response_format,
                        max_output_tokens=max_output_tokens,
                        request_id=request_id,
                        prompt_metadata=prompt_metadata,
                        temperature=temperature,
                    )
                if not _is_retryable(exc):
                    self._log_failure(
                        request_id, self.model, temperature, exc, prompt_metadata
                    )
                    emit_event(
                        _evs.LLM_CALL_FAILED,
                        request_id=request_id,
                        model=self.model,
                        error_class=type(exc).__name__,
                        temperature=temperature,
                    )
                    _metrics.llm_calls_total.inc(self.model, "failure")
                    raise
                last_exc = exc
                if attempt < self.max_retries - 1:
                    emit_event(
                        _evs.LLM_CALL_RETRIED,
                        request_id=request_id,
                        model=self.model,
                        attempt=attempt + 1,
                        error_class=type(exc).__name__,
                    )
                    time.sleep(self._backoff(attempt))
        if response is None:
            assert last_exc is not None
            self._log_failure(
                request_id, self.model, temperature, last_exc, prompt_metadata
            )
            raise last_exc

        # Extract the textual content (Responses API exposes it as
        # ``output_text`` for convenience). Reasoning items are skipped
        # — the caller only sees the model's final message.
        content = (getattr(response, "output_text", None) or "").strip()
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
        )
        cost = compute_cost(model_used, input_tokens, output_tokens)
        result = ChatResult(
            content=content,
            tool_calls=[],
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
            fallback_used=False,
            prompt_content=messages if self.telemetry.log_content else None,
            completion_content=result.content if self.telemetry.log_content else None,
            **prompt_fields,
        )
        _metrics.llm_calls_total.inc(result.model_used, "success")
        _metrics.llm_call_latency_ms.observe(
            result.model_used, value=float(result.latency_ms)
        )
        _metrics.llm_call_cost_usd.observe(value=float(result.cost_usd))
        _metrics.daily_cost_usd.observe(value=float(result.cost_usd))
        emit_event(
            _evs.LLM_CALL_SUCCEEDED,
            request_id=request_id,
            model=result.model_used,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            fallback_used=False,
        )
        return result

    def _respond_via_chat_fallback(
        self,
        *,
        messages: list[dict],
        response_format: dict | None,
        max_output_tokens: int,
        request_id: str,
        prompt_metadata: Any,
        temperature: float,
    ) -> ChatResult:
        """Last-resort: route the call through Chat Completions on the
        fallback model. Used when the primary reasoning model isn't
        available on the active API key (404 model_not_found etc.)."""
        kwargs = self._build_kwargs(
            messages, None, response_format, temperature, max_output_tokens
        )
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.fallback_model, **kwargs
            )
        except Exception as exc:
            self._log_failure(
                request_id,
                self.fallback_model,
                temperature,
                exc,
                prompt_metadata,
            )
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        return self._build_result(
            response=response,
            model_used=self.fallback_model,
            latency_ms=latency_ms,
            request_id=request_id,
            fallback_used=True,
        )

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
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=_OPENAI_REQUEST_TIMEOUT_S,
                max_retries=0,
            )

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
                if _is_model_unavailable(exc) and self.fallback_model != self.model:
                    # Primary model isn't usable on this API key — fall
                    # back to the known-available model immediately
                    # rather than burning retries we'll lose anyway.
                    start = time.perf_counter()
                    try:
                        response = self._client.chat.completions.create(
                            model=self.fallback_model, **kwargs
                        )
                        latency_ms = int((time.perf_counter() - start) * 1000)
                        return response, self.fallback_model, latency_ms, True
                    except Exception as fallback_exc:
                        self._log_failure(
                            request_id,
                            self.fallback_model,
                            temperature,
                            fallback_exc,
                            prompt_metadata,
                        )
                        raise
                if not _is_retryable(exc):
                    self._log_failure(
                        request_id, self.model, temperature, exc, prompt_metadata
                    )
                    raise
                last_exc = exc
                if attempt < self.max_retries - 1:
                    emit_event(
                        _evs.LLM_CALL_RETRIED,
                        request_id=request_id,
                        model=self.model,
                        attempt=attempt + 1,
                        error_class=type(exc).__name__,
                    )
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
