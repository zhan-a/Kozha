"""Tests for ``backend.chat2hamnosys.parser.description_parser``.

Two layers of coverage:

1. Mechanics — tests for :data:`SYSTEM_PROMPT`, the strict-schema builder,
   error paths in :func:`parse_description`.
2. Fixture replay — the 25 JSON fixtures under ``fixtures/parser/``
   each become a parametrised test case. A fake ``LLMClient`` returns the
   fixture's ``recorded_response`` verbatim, so the suite does not touch
   the real OpenAI API.

The per-fixture asserts check the **oracle** (``expected_populated`` and
``expected_gap_fields``), not the recording. A recording that no longer
matches the oracle will fail the test — re-recording against the real API
may therefore surface LLM drift, which is the intended signal. The eval
doc (``docs/chat2hamnosys/05-description-parser-eval.md``) measures the
overall population / gap hit-rate across the 25 fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm import ChatResult
from parser import (
    ALLOWED_GAP_FIELDS,
    PARSER_RESPONSE_SCHEMA,
    Gap,
    ParserError,
    ParseResult,
    PartialSignParameters,
    SYSTEM_PROMPT,
    build_parser_response_schema,
    parse_description,
)
from parser.description_parser import _build_parse_result, _reject_pua


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "parser"


# ---------------------------------------------------------------------------
# Fake LLM client (recorded-response pattern)
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Stand-in for :class:`LLMClient` that replays a recorded ``content``.

    Captures the messages / response_format / temperature / max_tokens the
    caller passed so tests can inspect them. Records latency as 1 ms and
    cost as 0 — we're not exercising the budget path here.
    """

    def __init__(self, content: str, model: str = "gpt-4o") -> None:
        self.content = content
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        *,
        request_id: str,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "response_format": response_format,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "request_id": request_id,
            }
        )
        return ChatResult(
            content=self.content,
            tool_calls=[],
            model_used=self.model,
            input_tokens=100,
            output_tokens=100,
            total_tokens=200,
            cost_usd=0.0,
            latency_ms=1,
            request_id=request_id,
        )


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_fixtures() -> list[Path]:
    files = sorted(FIXTURES_DIR.glob("*.json"))
    return files


# ---------------------------------------------------------------------------
# Schema / prompt mechanics
# ---------------------------------------------------------------------------


class TestSchema:
    def test_top_level_strict_shape(self) -> None:
        schema = build_parser_response_schema()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"parameters", "gaps"}

    def test_parameters_required_includes_every_property(self) -> None:
        # OpenAI strict mode demands every property be listed in required,
        # even nullable ones — the value may be null but the key must appear.
        params_schema = PARSER_RESPONSE_SCHEMA["properties"]["parameters"]
        assert set(params_schema["required"]) == set(params_schema["properties"].keys())
        assert params_schema["additionalProperties"] is False

    def test_gap_schema_fields(self) -> None:
        gap_items = PARSER_RESPONSE_SCHEMA["properties"]["gaps"]["items"]
        assert set(gap_items["required"]) == {"field", "reason", "suggested_question"}
        assert gap_items["additionalProperties"] is False

    def test_no_unsupported_keywords(self) -> None:
        forbidden = {
            "default",
            "title",
            "format",
            "minLength",
            "maxLength",
            "pattern",
            "examples",
        }

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for bad in forbidden:
                    assert bad not in node, f"forbidden keyword {bad!r} in schema"
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(PARSER_RESPONSE_SCHEMA)

    def test_defs_hoisted_to_top(self) -> None:
        assert "$defs" in PARSER_RESPONSE_SCHEMA
        defs = PARSER_RESPONSE_SCHEMA["$defs"]
        assert "PartialMovementSegment" in defs
        assert "PartialNonManualFeatures" in defs
        for d in defs.values():
            assert d["additionalProperties"] is False
            assert set(d["required"]) == set(d["properties"].keys())


class TestSystemPrompt:
    def test_mentions_no_pua(self) -> None:
        assert "NEVER invent HamNoSys" in SYSTEM_PROMPT
        # Prompt must not itself contain PUA codepoints — that would
        # partially defeat the prohibition.
        for ch in SYSTEM_PROMPT:
            assert not (0xE000 <= ord(ch) <= 0xF8FF), (
                f"system prompt contains PUA codepoint U+{ord(ch):04X}"
            )

    def test_includes_schema(self) -> None:
        # The prompt embeds PartialSignParameters.model_json_schema() so
        # the LLM sees the exact output contract.
        assert "PartialSignParameters schema" in SYSTEM_PROMPT
        assert '"handshape_dominant"' in SYSTEM_PROMPT
        assert '"non_manual"' in SYSTEM_PROMPT

    def test_includes_vocabulary_tables(self) -> None:
        for required in ("Handshapes", "Locations", "Palm direction", "Non-manuals"):
            assert required in SYSTEM_PROMPT

    def test_mentions_gap_fields(self) -> None:
        assert "Gap object" in SYSTEM_PROMPT
        for f in ("handshape_dominant", "non_manual.eyebrows"):
            assert f in SYSTEM_PROMPT

    def test_mandatory_slots_rule(self) -> None:
        assert "Mandatory" in SYSTEM_PROMPT
        for mandatory in (
            "handshape_dominant",
            "orientation_extended_finger",
            "orientation_palm",
            "location",
        ):
            assert mandatory in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# parse_description (routing + error handling)
# ---------------------------------------------------------------------------


class TestParseDescription:
    def test_routes_through_llm_client(self) -> None:
        recorded = json.dumps(
            {
                "parameters": {
                    "handshape_dominant": "flat",
                    "handshape_nondominant": None,
                    "orientation_extended_finger": "up",
                    "orientation_palm": "away from signer",
                    "location": "temple",
                    "contact": None,
                    "movement": [],
                    "non_manual": None,
                },
                "gaps": [],
            }
        )
        fake = FakeLLMClient(content=recorded)
        result = parse_description(
            "flat hand at temple, palm out, fingers up",
            client=fake,
            request_id="req-test-01",
        )
        assert isinstance(result, ParseResult)
        assert result.parameters.handshape_dominant == "flat"

        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["request_id"] == "req-test-01"
        assert call["temperature"] == 0.1
        # Response-format is pinned to strict json_schema.
        rf = call["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        # System prompt is the first message.
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][0]["content"] == SYSTEM_PROMPT
        assert call["messages"][1]["role"] == "user"

    def test_empty_prose_raises(self) -> None:
        with pytest.raises(ParserError, match="non-empty"):
            parse_description("", client=FakeLLMClient(content="{}"))
        with pytest.raises(ParserError, match="non-empty"):
            parse_description("   ", client=FakeLLMClient(content="{}"))

    def test_generates_request_id_when_missing(self) -> None:
        recorded = json.dumps({"parameters": _empty_params_dict(), "gaps": []})
        fake = FakeLLMClient(content=recorded)
        result = parse_description("test prose", client=fake)
        assert result.raw_response
        assert fake.calls[0]["request_id"].startswith("parse-")

    def test_malformed_json_raises(self) -> None:
        fake = FakeLLMClient(content="not json at all")
        with pytest.raises(ParserError, match="not valid JSON"):
            parse_description("x", client=fake)

    def test_missing_top_level_keys_raises(self) -> None:
        fake = FakeLLMClient(content=json.dumps({"parameters": {}}))
        with pytest.raises(ParserError, match="missing required keys"):
            parse_description("x", client=fake)

    def test_pua_in_output_rejected(self) -> None:
        bad = {
            "parameters": _empty_params_dict(),
            "gaps": [],
        }
        bad["parameters"]["handshape_dominant"] = "flat \uE001"  # smuggled PUA
        fake = FakeLLMClient(content=json.dumps(bad))
        with pytest.raises(ParserError, match="PUA codepoint"):
            parse_description("x", client=fake)

    def test_schema_violation_rejected(self) -> None:
        # Gap with empty string field — Pydantic min_length=1 should reject.
        bad = {
            "parameters": _empty_params_dict(),
            "gaps": [{"field": "", "reason": "", "suggested_question": ""}],
        }
        fake = FakeLLMClient(content=json.dumps(bad))
        with pytest.raises(ParserError, match="schema validation"):
            parse_description("x", client=fake)


def _empty_params_dict() -> dict[str, Any]:
    return {
        "handshape_dominant": None,
        "handshape_nondominant": None,
        "orientation_extended_finger": None,
        "orientation_palm": None,
        "location": None,
        "contact": None,
        "movement": [],
        "non_manual": None,
    }


# ---------------------------------------------------------------------------
# Fixture replay (oracle checks, 25 cases)
# ---------------------------------------------------------------------------


def _fixture_id(path: Path) -> str:
    return path.stem


FIXTURE_FILES = _all_fixtures()
assert len(FIXTURE_FILES) == 25, (
    f"expected 25 fixture files in {FIXTURES_DIR}, found {len(FIXTURE_FILES)}"
)


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_FILES,
    ids=[_fixture_id(p) for p in FIXTURE_FILES],
)
def test_fixture_replay(fixture_path: Path) -> None:
    """Replay each fixture through parse_description and check the oracle."""
    fixture = _load_fixture(fixture_path)
    expected_populated: set[str] = set(fixture["expected_populated"])
    expected_gaps: set[str] = set(fixture["expected_gap_fields"])

    recorded_content = json.dumps(fixture["recorded_response"])
    fake = FakeLLMClient(content=recorded_content)

    result = parse_description(fixture["prose"], client=fake)

    populated = _populated_fields(result.parameters)
    actual_gaps = {g.field for g in result.gaps}

    # Each expected populated slot must be non-null in the result.
    missing = expected_populated - populated
    assert not missing, (
        f"[{fixture['id']}] expected populated fields {missing} were empty; "
        f"got populated={sorted(populated)}"
    )

    # Each expected gap must appear in result.gaps.
    missing_gaps = expected_gaps - actual_gaps
    assert not missing_gaps, (
        f"[{fixture['id']}] expected gaps {missing_gaps} not flagged; "
        f"got gaps={sorted(actual_gaps)}"
    )

    # Every gap the parser emits must name a known slot.
    for g in result.gaps:
        assert g.field in ALLOWED_GAP_FIELDS, (
            f"[{fixture['id']}] gap names unknown slot {g.field!r}"
        )


def _populated_fields(params: PartialSignParameters) -> set[str]:
    """Return the set of slot names that are populated (non-null / non-empty)."""
    out: set[str] = set()
    for name in (
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    ):
        if getattr(params, name) is not None:
            out.add(name)
    if params.movement:
        out.add("movement")
    if params.non_manual is not None:
        out.add("non_manual")
    return out


# ---------------------------------------------------------------------------
# PUA rejection helper (exercised directly for quick feedback)
# ---------------------------------------------------------------------------


class TestRejectPua:
    def test_accepts_plain_english(self) -> None:
        p = PartialSignParameters(
            handshape_dominant="flat",
            location="temple",
        )
        # Should not raise.
        _reject_pua(p)

    def test_rejects_pua_in_handshape(self) -> None:
        p = PartialSignParameters(handshape_dominant="\uE001")
        with pytest.raises(ParserError, match="PUA"):
            _reject_pua(p)

    def test_rejects_pua_in_movement(self) -> None:
        from parser import PartialMovementSegment

        p = PartialSignParameters(
            movement=[PartialMovementSegment(path="circular \uE089")]
        )
        with pytest.raises(ParserError, match="PUA"):
            _reject_pua(p)

    def test_rejects_pua_in_non_manual(self) -> None:
        from parser import PartialNonManualFeatures

        p = PartialSignParameters(
            non_manual=PartialNonManualFeatures(eyebrows="raised \uE000")
        )
        with pytest.raises(ParserError, match="PUA"):
            _reject_pua(p)


# ---------------------------------------------------------------------------
# Models / Gap validation
# ---------------------------------------------------------------------------


class TestModels:
    def test_gap_requires_all_fields(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Gap(field="", reason="x", suggested_question="y")

    def test_partial_params_allow_all_nulls(self) -> None:
        p = PartialSignParameters()
        assert p.handshape_dominant is None
        assert p.movement == []
        assert p.non_manual is None
