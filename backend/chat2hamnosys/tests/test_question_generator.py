"""Tests for ``backend.chat2hamnosys.clarify.question_generator``.

Two layers:

1. Mechanics — the strict-JSON schema, the system prompt contents, the
   prior-turns de-duplication heuristic, the three-question cap, and the
   fallback paths (no API key, LLM exception, malformed output).
2. Fixture replay — the JSON fixtures under ``fixtures/clarify/``
   feed prose + gaps + prior turns + a recorded LLM response, and the
   test asserts the oracle (``expected_fields``, ``expected_count``,
   ``expect_fallback``) matches what :func:`generate_questions` returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from clarify import (
    GENERATOR_RESPONSE_SCHEMA,
    MAX_QUESTIONS_PER_TURN,
    Option,
    Question,
    SYSTEM_PROMPT,
    build_generator_response_schema,
    generate_questions,
    template_for,
)
from clarify.question_generator import _fields_already_asked, _gap_targets
from llm import ChatResult, LLMConfigError
from llm.budget import BudgetExceeded
from models import ClarificationTurn
from parser import ALLOWED_GAP_FIELDS, Gap, ParseResult, PartialSignParameters


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "clarify"


# ---------------------------------------------------------------------------
# Fake LLM clients
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Replays a recorded ``content`` on each ``chat()`` call."""

    def __init__(self, content: str, model: str = "gpt-4o") -> None:
        self.content = content
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        *,
        request_id: str,
        prompt_metadata: Any = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "response_format": response_format,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "request_id": request_id,
                "prompt_metadata": prompt_metadata,
            }
        )
        return ChatResult(
            content=self.content,
            tool_calls=[],
            model_used=self.model,
            input_tokens=50,
            output_tokens=50,
            total_tokens=100,
            cost_usd=0.0,
            latency_ms=1,
            request_id=request_id,
        )


class RaisingLLMClient:
    """Raises ``exc`` on every call. Useful for testing the fallback path."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def chat(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise self.exc


# ---------------------------------------------------------------------------
# Mechanics: schema, prompt, helpers
# ---------------------------------------------------------------------------


class TestSchema:
    def test_top_level_shape(self) -> None:
        schema = build_generator_response_schema()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["questions"]
        assert schema["properties"]["questions"]["type"] == "array"

    def test_question_item_is_strict(self) -> None:
        item = GENERATOR_RESPONSE_SCHEMA["properties"]["questions"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"].keys())
        # Core fields are present.
        for key in ("field", "text", "options", "allow_freeform", "rationale"):
            assert key in item["properties"]

    def test_no_unsupported_keywords(self) -> None:
        forbidden = {"default", "title", "minLength", "pattern", "examples"}

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for bad in forbidden:
                    assert bad not in node, f"forbidden keyword {bad!r}"
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(GENERATOR_RESPONSE_SCHEMA)


class TestSystemPrompt:
    def test_mentions_hard_rules(self) -> None:
        assert "at most" in SYSTEM_PROMPT.lower()
        assert "three" in SYSTEM_PROMPT.lower()
        assert "multiple-choice" in SYSTEM_PROMPT.lower() or "multiple choice" in SYSTEM_PROMPT.lower()

    def test_mentions_deaf_native(self) -> None:
        assert "is_deaf_native" in SYSTEM_PROMPT

    def test_includes_allowed_fields(self) -> None:
        for field in ALLOWED_GAP_FIELDS:
            assert f"`{field}`" in SYSTEM_PROMPT, (
                f"field {field!r} missing from prompt"
            )

    def test_includes_template_library(self) -> None:
        # The LLM sees every template as a starting point.
        for field in ALLOWED_GAP_FIELDS:
            assert f"**{field}**" in SYSTEM_PROMPT

    def test_no_pua_codepoints(self) -> None:
        for ch in SYSTEM_PROMPT:
            assert not (0xE000 <= ord(ch) <= 0xF8FF), (
                f"prompt contains PUA U+{ord(ch):04X}"
            )


class TestFieldsAlreadyAsked:
    def test_empty_turns(self) -> None:
        assert _fields_already_asked([]) == set()

    def test_asks_are_only_from_assistant(self) -> None:
        turns = [
            ClarificationTurn(role="author", text="Which handshape is used?"),
        ]
        # An author turn is not an assistant ask — don't treat it as such.
        assert _fields_already_asked(turns) == set()

    def test_handshape_detection(self) -> None:
        turns = [
            ClarificationTurn(
                role="assistant",
                text="Which handshape is used? A closed fist or a flat palm?",
            ),
        ]
        asked = _fields_already_asked(turns)
        assert "handshape_dominant" in asked

    def test_handshape_nondominant_does_not_trigger_dominant(self) -> None:
        turns = [
            ClarificationTurn(
                role="assistant",
                text="What is the non-dominant handshape doing?",
            ),
        ]
        asked = _fields_already_asked(turns)
        assert "handshape_nondominant" in asked
        assert "handshape_dominant" not in asked

    def test_multiple_slots(self) -> None:
        turns = [
            ClarificationTurn(
                role="assistant",
                text="Which way does the palm face? And where on the body?",
            ),
        ]
        asked = _fields_already_asked(turns)
        assert "orientation_palm" in asked
        assert "location" in asked


class TestGapTargets:
    def test_caps_at_three(self) -> None:
        gaps = [
            Gap(field=f, reason="r", suggested_question="q")
            for f in (
                "handshape_dominant",
                "orientation_palm",
                "orientation_extended_finger",
                "location",
                "contact",
            )
        ]
        pr = ParseResult(parameters=PartialSignParameters(), gaps=gaps)
        targets = _gap_targets(pr, [])
        assert len(targets) == MAX_QUESTIONS_PER_TURN

    def test_dedupes_repeated_fields(self) -> None:
        gaps = [
            Gap(field="handshape_dominant", reason="a", suggested_question="q1"),
            Gap(field="handshape_dominant", reason="b", suggested_question="q2"),
        ]
        pr = ParseResult(parameters=PartialSignParameters(), gaps=gaps)
        targets = _gap_targets(pr, [])
        assert len(targets) == 1

    def test_filters_prior_turns(self) -> None:
        gaps = [
            Gap(field="handshape_dominant", reason="r", suggested_question="q"),
            Gap(field="location", reason="r", suggested_question="q"),
        ]
        pr = ParseResult(parameters=PartialSignParameters(), gaps=gaps)
        prior = [
            ClarificationTurn(
                role="assistant",
                text="Which handshape is used?",
            )
        ]
        targets = _gap_targets(pr, prior)
        assert [g.field for g in targets] == ["location"]


# ---------------------------------------------------------------------------
# End-to-end generate_questions behavior
# ---------------------------------------------------------------------------


class TestGenerateQuestions:
    def _single_gap_result(self) -> ParseResult:
        return ParseResult(
            parameters=PartialSignParameters(),
            gaps=[
                Gap(
                    field="handshape_dominant",
                    reason="not specified",
                    suggested_question="which handshape?",
                )
            ],
        )

    def _three_gaps_result(self) -> ParseResult:
        return ParseResult(
            parameters=PartialSignParameters(),
            gaps=[
                Gap(field="handshape_dominant", reason="r", suggested_question="q"),
                Gap(field="orientation_palm", reason="r", suggested_question="q"),
                Gap(field="orientation_extended_finger", reason="r", suggested_question="q"),
            ],
        )

    def test_empty_gaps_returns_empty(self) -> None:
        pr = ParseResult(parameters=PartialSignParameters(), gaps=[])
        fake = FakeLLMClient(content='{"questions": []}')
        assert generate_questions(pr, client=fake) == []
        # LLM should not even be called when there are no targets.
        assert fake.calls == []

    def test_routes_through_llm_client(self) -> None:
        pr = self._single_gap_result()
        recorded = json.dumps(
            {
                "questions": [
                    {
                        "field": "handshape_dominant",
                        "text": "Which handshape?",
                        "options": [{"label": "Flat", "value": "flat"}],
                        "allow_freeform": True,
                        "rationale": "discrete slot",
                    }
                ]
            }
        )
        fake = FakeLLMClient(content=recorded)
        qs = generate_questions(pr, client=fake, request_id="req-test-clarify")
        assert len(qs) == 1
        assert qs[0].text == "Which handshape?"

        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["request_id"] == "req-test-clarify"
        assert call["temperature"] == 0.3
        rf = call["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][0]["content"] == SYSTEM_PROMPT

    def test_generates_request_id_when_missing(self) -> None:
        pr = self._single_gap_result()
        recorded = json.dumps(
            {
                "questions": [
                    {
                        "field": "handshape_dominant",
                        "text": "Q?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    }
                ]
            }
        )
        fake = FakeLLMClient(content=recorded)
        generate_questions(pr, client=fake)
        assert fake.calls[0]["request_id"].startswith("clarify-")

    def test_prose_grounded_from_prior_author_turn(self) -> None:
        pr = self._single_gap_result()
        recorded = json.dumps(
            {
                "questions": [
                    {
                        "field": "handshape_dominant",
                        "text": "Q?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    }
                ]
            }
        )
        fake = FakeLLMClient(content=recorded)
        prose = "A sign at the temple with a claw-like shape."
        turns = [
            ClarificationTurn(role="author", text=prose),
        ]
        generate_questions(pr, turns, client=fake)
        user_msg = fake.calls[0]["messages"][1]["content"]
        payload = json.loads(user_msg)
        assert payload["prose"] == prose

    def test_single_gap_produces_one_question(self) -> None:
        pr = self._single_gap_result()
        recorded = json.dumps(
            {
                "questions": [
                    {
                        "field": "handshape_dominant",
                        "text": "Q?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    }
                ]
            }
        )
        fake = FakeLLMClient(content=recorded)
        qs = generate_questions(pr, client=fake)
        assert len(qs) == 1
        assert qs[0].field == "handshape_dominant"

    def test_three_gaps_produces_at_most_three(self) -> None:
        # Four input gaps; only three should come back.
        pr = ParseResult(
            parameters=PartialSignParameters(),
            gaps=[
                Gap(field="handshape_dominant", reason="r", suggested_question="q"),
                Gap(field="orientation_palm", reason="r", suggested_question="q"),
                Gap(field="orientation_extended_finger", reason="r", suggested_question="q"),
                Gap(field="location", reason="r", suggested_question="q"),
            ],
        )
        # LLM echoes back four questions; generator caps to three.
        recorded = json.dumps(
            {
                "questions": [
                    {
                        "field": "handshape_dominant",
                        "text": "Q?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                    {
                        "field": "orientation_palm",
                        "text": "Q?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                    {
                        "field": "orientation_extended_finger",
                        "text": "Q?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                    {
                        "field": "location",
                        "text": "Q?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                ]
            }
        )
        fake = FakeLLMClient(content=recorded)
        qs = generate_questions(pr, client=fake)
        assert len(qs) == MAX_QUESTIONS_PER_TURN
        # The dropped one is the fourth input gap (location), since the
        # generator capped _before_ calling the LLM.
        payload = json.loads(fake.calls[0]["messages"][1]["content"])
        target_fields = {g["field"] for g in payload["gaps"]}
        assert "location" not in target_fields

    def test_prior_turns_deduplicate(self) -> None:
        pr = self._three_gaps_result()
        prior = [
            ClarificationTurn(
                role="assistant",
                text="Which handshape is used?",
            )
        ]
        recorded = json.dumps(
            {
                "questions": [
                    {
                        "field": "orientation_palm",
                        "text": "Palm?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                    {
                        "field": "orientation_extended_finger",
                        "text": "Fingers?",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                ]
            }
        )
        fake = FakeLLMClient(content=recorded)
        qs = generate_questions(pr, prior, client=fake)
        fields = [q.field for q in qs]
        assert "handshape_dominant" not in fields
        assert "orientation_palm" in fields
        assert "orientation_extended_finger" in fields

    def test_drops_questions_with_unknown_field(self) -> None:
        pr = self._single_gap_result()
        recorded = json.dumps(
            {
                "questions": [
                    {
                        "field": "handshape_dominant",
                        "text": "real",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                    {
                        "field": "made_up_slot",
                        "text": "invented",
                        "options": None,
                        "allow_freeform": True,
                        "rationale": "",
                    },
                ]
            }
        )
        fake = FakeLLMClient(content=recorded)
        qs = generate_questions(pr, client=fake)
        assert len(qs) == 1
        assert qs[0].field == "handshape_dominant"


class TestFallbackPaths:
    def _one_gap(self) -> ParseResult:
        return ParseResult(
            parameters=PartialSignParameters(),
            gaps=[
                Gap(
                    field="handshape_dominant",
                    reason="claw-like",
                    suggested_question="?",
                )
            ],
        )

    def test_fallback_on_malformed_json(self) -> None:
        pr = self._one_gap()
        fake = FakeLLMClient(content="this is not json at all")
        qs = generate_questions(pr, client=fake)
        # Falls back to template — content matches template_for() verbatim.
        expected = template_for("handshape_dominant", reason="claw-like")
        assert len(qs) == 1
        assert qs[0].text == expected.text
        assert qs[0].options and len(qs[0].options) == len(expected.options or [])

    def test_fallback_on_missing_key(self) -> None:
        pr = self._one_gap()
        fake = FakeLLMClient(content=json.dumps({"not_questions": []}))
        qs = generate_questions(pr, client=fake)
        assert qs[0].field == "handshape_dominant"
        # Fallback template has options.
        assert qs[0].options is not None

    def test_fallback_on_empty_questions_list(self) -> None:
        pr = self._one_gap()
        fake = FakeLLMClient(content=json.dumps({"questions": []}))
        qs = generate_questions(pr, client=fake)
        # No valid questions → fallback fires.
        assert len(qs) == 1
        assert qs[0].field == "handshape_dominant"

    def test_fallback_on_llm_exception(self) -> None:
        pr = self._one_gap()
        raising = RaisingLLMClient(RuntimeError("network down"))
        qs = generate_questions(pr, client=raising)
        assert len(qs) == 1
        assert qs[0].field == "handshape_dominant"

    def test_fallback_on_schema_violation(self) -> None:
        pr = self._one_gap()
        # Question with empty field string — Pydantic min_length rejects.
        fake = FakeLLMClient(
            content=json.dumps(
                {
                    "questions": [
                        {
                            "field": "",
                            "text": "bad",
                            "options": None,
                            "allow_freeform": True,
                            "rationale": "",
                        }
                    ]
                }
            )
        )
        qs = generate_questions(pr, client=fake)
        assert len(qs) == 1
        assert qs[0].field == "handshape_dominant"

    def test_fallback_on_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No client passed AND no OPENAI_API_KEY env var → LLMConfigError
        # on construction → fall back to templates.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        pr = self._one_gap()
        qs = generate_questions(pr)
        assert len(qs) == 1
        assert qs[0].field == "handshape_dominant"

    def test_budget_exceeded_propagates(self) -> None:
        pr = self._one_gap()
        raising = RaisingLLMClient(BudgetExceeded(spent=1.0, would_add=0.5, cap=1.0))
        with pytest.raises(BudgetExceeded):
            generate_questions(pr, client=raising)

    def test_fallback_uses_deaf_native_phrasing(self) -> None:
        pr = self._one_gap()
        fake = FakeLLMClient(content="malformed")
        qs = generate_questions(pr, client=fake, is_deaf_native=True)
        # Deaf-native variant from the template.
        expected = template_for("handshape_dominant", is_deaf_native=True, reason="claw-like")
        assert qs[0].text == expected.text


# ---------------------------------------------------------------------------
# Fixture replay
# ---------------------------------------------------------------------------


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_id(path: Path) -> str:
    return path.stem


FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.json"))
assert FIXTURE_FILES, f"no fixtures found in {FIXTURES_DIR}"


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_FILES,
    ids=[_fixture_id(p) for p in FIXTURE_FILES],
)
def test_fixture_replay(fixture_path: Path) -> None:
    fixture = _load_fixture(fixture_path)
    pr = ParseResult.model_validate(fixture["parse_result"])
    prior = [
        ClarificationTurn.model_validate(t) for t in fixture.get("prior_turns", [])
    ]
    is_deaf_native = fixture.get("is_deaf_native")

    if fixture.get("expect_fallback"):
        content = fixture["recorded_response_raw"]
    else:
        content = json.dumps(fixture["recorded_response"])

    fake = FakeLLMClient(content=content)
    qs = generate_questions(
        pr, prior, client=fake, is_deaf_native=is_deaf_native
    )

    assert len(qs) == fixture["expected_count"], (
        f"[{fixture['id']}] expected {fixture['expected_count']} questions, "
        f"got {len(qs)}"
    )

    expected_fields = fixture["expected_fields"]
    actual_fields = [q.field for q in qs]
    assert actual_fields == expected_fields, (
        f"[{fixture['id']}] expected fields {expected_fields}, got {actual_fields}"
    )

    # Every returned field must be in the allowed universe.
    for q in qs:
        assert q.field in ALLOWED_GAP_FIELDS, (
            f"[{fixture['id']}] question names unknown field {q.field!r}"
        )

    if fixture.get("expect_fallback"):
        # Fallback questions must match template_for() exactly.
        for q, field in zip(qs, expected_fields):
            expected = template_for(
                field,
                is_deaf_native=bool(is_deaf_native),
                reason=next(
                    g["reason"]
                    for g in fixture["parse_result"]["gaps"]
                    if g["field"] == field
                ),
            )
            assert q.text == expected.text, (
                f"[{fixture['id']}] fallback text mismatch for {field}"
            )
