"""Tests for ``chat2hamnosys.llm``.

All OpenAI SDK calls are mocked — these tests never touch the network.
Retry / fallback behavior is exercised by injecting a stub client whose
``chat.completions.create`` walks a programmed sequence of errors and
responses. Backoff sleeps are either patched out or the client is
configured with ``base_backoff=0`` so the whole suite runs instantly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from llm import (
    BudgetExceeded,
    BudgetGuard,
    ChatResult,
    LLMClient,
    LLMConfigError,
    TelemetryLogger,
    compute_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    content: str = "hello",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list[dict] | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    if tool_calls is not None:
        message.tool_calls = [
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["name"], arguments=tc["arguments"]
                ),
            )
            for tc in tool_calls
        ]
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _rate_limit_error(msg: str = "rate limited") -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(msg, response=response, body=None)


def _api_status_error(status: int, msg: str = "err") -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return APIStatusError(msg, response=response, body=None)


def _stub_client(sequence: list[Any]) -> MagicMock:
    """MagicMock whose ``chat.completions.create`` walks a fixed sequence.

    Each entry is either a response object (returned) or an Exception
    (raised). The stub raises ``StopIteration`` if called beyond the
    end of the sequence — tests must provide enough entries.
    """
    client = MagicMock()
    it = iter(sequence)

    def side_effect(**_: Any) -> Any:
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    client.chat.completions.create.side_effect = side_effect
    return client


def _read_all_telemetry(log_dir: Path) -> list[dict]:
    if not log_dir.exists():
        return []
    out: list[dict] = []
    for f in sorted(log_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def telemetry(tmp_path: Path) -> TelemetryLogger:
    return TelemetryLogger(log_dir=tmp_path / "llm")


@pytest.fixture
def budget() -> BudgetGuard:
    # Generous cap so the guard only fires when a test deliberately
    # pushes spend past it.
    return BudgetGuard(max_usd_per_session=10.0)


@pytest.fixture
def client_factory(
    telemetry: TelemetryLogger,
    budget: BudgetGuard,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., LLMClient]:
    """Build an ``LLMClient`` with an injectable fake SDK client.

    Sets ``OPENAI_API_KEY`` so the config check passes without a real
    secret, and pins ``base_backoff=0`` so retry tests run instantly.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    def _make(
        stub: Any,
        *,
        model: str = "gpt-4o",
        fallback_model: str = "gpt-4o-mini",
    ) -> LLMClient:
        return LLMClient(
            model=model,
            fallback_model=fallback_model,
            budget=budget,
            telemetry=telemetry,
            client=stub,
            base_backoff=0.0,
        )

    return _make


# ---------------------------------------------------------------------------
# Config / API key
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        LLMClient()


def test_empty_explicit_key_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Env is set but caller passed "" — still rejected, does not fall
    # back to the env value.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with pytest.raises(LLMConfigError):
        LLMClient(api_key="")


def test_explicit_key_preferred_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    c = LLMClient(api_key="sk-explicit", client=MagicMock())
    assert c._api_key == "sk-explicit"


def test_repr_omits_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-never-leak")
    c = LLMClient(client=MagicMock())
    assert "sk-secret-should-never-leak" not in repr(c)


def test_request_id_required(client_factory: Callable[..., LLMClient]) -> None:
    client = client_factory(_stub_client([_mock_response()]))
    with pytest.raises(LLMConfigError, match="request_id"):
        client.chat(messages=[{"role": "user", "content": "x"}], request_id="")


# ---------------------------------------------------------------------------
# Successful call
# ---------------------------------------------------------------------------


def test_successful_call_returns_chat_result(
    client_factory: Callable[..., LLMClient],
) -> None:
    stub = _stub_client([
        _mock_response(content="hi there", prompt_tokens=12, completion_tokens=4),
    ])
    client = client_factory(stub)
    result = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        request_id="req-001",
    )
    assert isinstance(result, ChatResult)
    assert result.content == "hi there"
    assert result.tool_calls == []
    assert result.model_used == "gpt-4o"
    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert result.total_tokens == 16
    assert result.cost_usd == pytest.approx(compute_cost("gpt-4o", 12, 4))
    assert result.request_id == "req-001"
    assert result.fallback_used is False
    assert result.latency_ms >= 0


def test_tool_calls_are_extracted(
    client_factory: Callable[..., LLMClient],
) -> None:
    stub = _stub_client([
        _mock_response(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ],
        )
    ])
    client = client_factory(stub)
    result = client.chat(
        messages=[{"role": "user", "content": "weather?"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {}},
            }
        ],
        request_id="req-tools",
    )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"city": "London"}'
    assert result.tool_calls[0].id == "call_1"


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


def test_retry_on_429_then_succeeds(
    client_factory: Callable[..., LLMClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    stub = _stub_client([
        _rate_limit_error(),
        _rate_limit_error(),
        _mock_response(content="finally"),
    ])
    client = client_factory(stub)
    result = client.chat(
        messages=[{"role": "user", "content": "x"}],
        request_id="req-retry",
    )
    assert result.content == "finally"
    assert result.model_used == "gpt-4o"
    assert result.fallback_used is False
    assert stub.chat.completions.create.call_count == 3


def test_retry_on_5xx_then_succeeds(
    client_factory: Callable[..., LLMClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    stub = _stub_client([
        _api_status_error(503, "svc unavail"),
        _mock_response(content="ok"),
    ])
    client = client_factory(stub)
    result = client.chat(
        messages=[{"role": "user", "content": "x"}],
        request_id="req-5xx",
    )
    assert result.content == "ok"
    assert stub.chat.completions.create.call_count == 2


def test_all_retries_fail_with_429_triggers_fallback_model(
    client_factory: Callable[..., LLMClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    stub = _stub_client([
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
        _mock_response(content="cheap response", model="gpt-4o-mini"),
    ])
    client = client_factory(stub)
    result = client.chat(
        messages=[{"role": "user", "content": "x"}],
        request_id="req-fallback",
    )
    assert result.content == "cheap response"
    assert result.model_used == "gpt-4o-mini"
    assert result.fallback_used is True
    assert stub.chat.completions.create.call_count == 4

    # The fallback model was actually passed to the SDK on the 4th call.
    last_call_kwargs = stub.chat.completions.create.call_args_list[-1].kwargs
    assert last_call_kwargs["model"] == "gpt-4o-mini"


def test_5xx_exhausted_does_not_trigger_fallback(
    client_factory: Callable[..., LLMClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    stub = _stub_client([
        _api_status_error(502),
        _api_status_error(502),
        _api_status_error(502),
    ])
    client = client_factory(stub)
    with pytest.raises(APIStatusError):
        client.chat(
            messages=[{"role": "user", "content": "x"}],
            request_id="req-5xx-exhaust",
        )
    # Exactly 3 primary attempts — no extra fallback call.
    assert stub.chat.completions.create.call_count == 3


def test_non_retryable_error_propagates_immediately(
    client_factory: Callable[..., LLMClient],
    telemetry: TelemetryLogger,
) -> None:
    bad_request = _api_status_error(400, "bad req")
    stub = _stub_client([bad_request])
    client = client_factory(stub)
    with pytest.raises(APIStatusError):
        client.chat(
            messages=[{"role": "user", "content": "x"}],
            request_id="req-badreq",
        )
    # One call only — no retries for 4xx.
    assert stub.chat.completions.create.call_count == 1
    records = _read_all_telemetry(telemetry.log_dir)
    assert any(not r["success"] for r in records)


def test_fallback_failure_after_exhausted_retries_raises(
    client_factory: Callable[..., LLMClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    stub = _stub_client([
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),  # fallback also rate limited
    ])
    client = client_factory(stub)
    with pytest.raises(RateLimitError):
        client.chat(
            messages=[{"role": "user", "content": "x"}],
            request_id="req-totalfail",
        )


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


def test_budget_exceeded_blocks_call_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stub = MagicMock()
    tiny_budget = BudgetGuard(max_usd_per_session=0.00001)
    client = LLMClient(
        model="gpt-4o",
        budget=tiny_budget,
        telemetry=TelemetryLogger(log_dir=tmp_path / "llm"),
        client=stub,
        base_backoff=0.0,
    )
    with pytest.raises(BudgetExceeded):
        client.chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2000,
            request_id="req-budget",
        )
    stub.chat.completions.create.assert_not_called()


def test_budget_records_actual_cost(
    client_factory: Callable[..., LLMClient], budget: BudgetGuard
) -> None:
    stub = _stub_client([
        _mock_response(prompt_tokens=1000, completion_tokens=2000),
    ])
    client = client_factory(stub)
    result = client.chat(
        messages=[{"role": "user", "content": "x"}],
        request_id="req-record",
    )
    assert budget.spent == pytest.approx(result.cost_usd)
    assert budget.remaining == pytest.approx(10.0 - result.cost_usd)


def test_env_var_sets_budget_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_BUDGET_USD", "0.5")
    g = BudgetGuard()
    assert g.max_usd_per_session == 0.5


def test_env_var_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_BUDGET_USD", "not-a-number")
    with pytest.raises(ValueError, match="CHAT2HAMNOSYS_SESSION_BUDGET_USD"):
        BudgetGuard()


def test_env_var_nonpositive_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_BUDGET_USD", "0")
    with pytest.raises(ValueError, match="positive"):
        BudgetGuard()


def test_default_session_cap_is_two_dollars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHAT2HAMNOSYS_SESSION_BUDGET_USD", raising=False)
    assert BudgetGuard().max_usd_per_session == 2.0


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_telemetry_records_successful_call(
    client_factory: Callable[..., LLMClient],
    telemetry: TelemetryLogger,
) -> None:
    stub = _stub_client([
        _mock_response(prompt_tokens=42, completion_tokens=7),
    ])
    client = client_factory(stub)
    client.chat(
        messages=[{"role": "user", "content": "x"}],
        temperature=0.4,
        request_id="req-telemetry",
    )
    records = _read_all_telemetry(telemetry.log_dir)
    assert len(records) == 1
    r = records[0]
    assert r["request_id"] == "req-telemetry"
    assert r["success"] is True
    assert r["prompt_tokens"] == 42
    assert r["completion_tokens"] == 7
    assert r["temperature"] == 0.4
    assert r["fallback_used"] is False
    assert r["error_class"] is None
    assert r["model"] == "gpt-4o"
    assert r["cost_usd"] > 0
    assert "ts" in r


def test_telemetry_omits_content_by_default(
    client_factory: Callable[..., LLMClient],
    telemetry: TelemetryLogger,
) -> None:
    stub = _stub_client([_mock_response(content="secret answer")])
    client = client_factory(stub)
    client.chat(
        messages=[{"role": "user", "content": "secret question"}],
        request_id="req-quiet",
    )
    records = _read_all_telemetry(telemetry.log_dir)
    blob = json.dumps(records)
    assert "secret question" not in blob
    assert "secret answer" not in blob
    assert "prompt" not in records[0]
    assert "completion" not in records[0]


def test_telemetry_includes_content_when_flag_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    telemetry = TelemetryLogger(log_dir=tmp_path / "llm", log_content=True)
    stub = _stub_client([_mock_response(content="loud answer")])
    # Pin to a non-reasoning model so the call goes through Chat
    # Completions and the chat-completions stub is exercised — the
    # default model now routes reasoning-capable ids to the Responses
    # API, which would need a separate stub on `responses.create`.
    client = LLMClient(
        model="gpt-4o",
        budget=BudgetGuard(max_usd_per_session=10.0),
        telemetry=telemetry,
        client=stub,
        base_backoff=0.0,
    )
    client.chat(
        messages=[{"role": "user", "content": "loud question"}],
        request_id="req-loud",
    )
    records = _read_all_telemetry(telemetry.log_dir)
    assert "prompt" in records[0]
    assert "completion" in records[0]
    assert records[0]["completion"] == "loud answer"
    blob = json.dumps(records)
    assert "loud question" in blob
    assert "loud answer" in blob


def test_telemetry_records_failed_call(
    client_factory: Callable[..., LLMClient],
    telemetry: TelemetryLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    stub = _stub_client([
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
    ])
    client = client_factory(stub)
    with pytest.raises(RateLimitError):
        client.chat(
            messages=[{"role": "user", "content": "x"}],
            request_id="req-fail",
        )
    records = _read_all_telemetry(telemetry.log_dir)
    assert any(
        r["success"] is False and r["error_class"] == "RateLimitError"
        for r in records
    )


def test_telemetry_records_fallback_success(
    client_factory: Callable[..., LLMClient],
    telemetry: TelemetryLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    stub = _stub_client([
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
        _mock_response(content="cheap", model="gpt-4o-mini"),
    ])
    client = client_factory(stub)
    client.chat(
        messages=[{"role": "user", "content": "x"}],
        request_id="req-fb",
    )
    records = _read_all_telemetry(telemetry.log_dir)
    succeeded = [r for r in records if r["success"]]
    assert len(succeeded) == 1
    assert succeeded[0]["fallback_used"] is True
    assert succeeded[0]["model"] == "gpt-4o-mini"


def test_telemetry_path_is_date_bucketed(
    client_factory: Callable[..., LLMClient],
    telemetry: TelemetryLogger,
) -> None:
    stub = _stub_client([_mock_response()])
    client = client_factory(stub)
    client.chat(
        messages=[{"role": "user", "content": "x"}],
        request_id="req-dated",
    )
    files = list(telemetry.log_dir.glob("*.jsonl"))
    assert len(files) == 1
    # Looks like YYYY-MM-DD.jsonl.
    name = files[0].name
    assert name.endswith(".jsonl")
    assert len(name) == len("YYYY-MM-DD.jsonl")


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_pricing_unknown_model_falls_back_to_gpt4o() -> None:
    # Safety: an unknown model must not silently bill $0.
    c = compute_cost("totally-new-model", 1000, 1000)
    assert c > 0
    assert c == compute_cost("gpt-4o", 1000, 1000)


def test_pricing_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError):
        compute_cost("gpt-4o", -1, 0)
    with pytest.raises(ValueError):
        compute_cost("gpt-4o", 0, -1)


def test_pricing_output_tokens_more_expensive_than_input() -> None:
    input_only = compute_cost("gpt-4o", 10_000, 0)
    output_only = compute_cost("gpt-4o", 0, 10_000)
    assert output_only > input_only
