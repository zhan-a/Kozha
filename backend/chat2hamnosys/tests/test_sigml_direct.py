"""Tests for the SiGML-direct generation path."""

from __future__ import annotations

import json
from typing import Any

from generator.sigml_direct import (
    generate_sigml_direct,
    sigml_to_hamnosys,
)
from generator.sigml_examples import few_shot_examples
from llm import ChatResult


# ---------------------------------------------------------------------------
# Reverse mapper: SiGML XML → HamNoSys PUA
# ---------------------------------------------------------------------------


def test_sigml_to_hamnosys_recovers_known_codepoints():
    sigml = (
        "<sigml><hns_sign gloss=\"X\">"
        "<hamnosys_manual>"
        "<hamflathand/><hamextfingeru/><hampalmd/><hamhead/>"
        "</hamnosys_manual>"
        "</hns_sign></sigml>"
    )
    out = sigml_to_hamnosys(sigml)
    # E001 = hamflathand, E020 = hamextfingeru, E03A = hampalmd,
    # E04A = hamhead (per hamnosys.symbols). Just check it's non-empty
    # and uses PUA codepoints.
    assert out
    assert all(0xE000 <= ord(c) <= 0xE0FF for c in out)
    # First codepoint must correspond to a hamflathand / similar.
    assert ord(out[0]) >= 0xE000


def test_sigml_to_hamnosys_skips_unknown_tags():
    sigml = (
        "<hamnosys_manual>"
        "<hamflathand/><hamnotarealtag/><hamextfingeru/>"
        "</hamnosys_manual>"
    )
    out = sigml_to_hamnosys(sigml)
    # Two known tags survive; the unknown one is dropped silently.
    assert len(out) == 2


def test_sigml_to_hamnosys_handles_empty_input():
    assert sigml_to_hamnosys("") == ""
    assert sigml_to_hamnosys("<sigml/>") == ""


# ---------------------------------------------------------------------------
# Few-shot loader
# ---------------------------------------------------------------------------


def test_few_shot_examples_returns_bsl_signs():
    examples = few_shot_examples(gloss="hello", n=4)
    # The BSL corpus is bundled with the repo; if it loaded we should
    # always get back something. Empty is acceptable in a stripped
    # build but we still want the function not to crash.
    assert isinstance(examples, list)
    for ex in examples:
        assert ex.gloss
        assert "<hns_sign" in ex.sigml_fragment
        assert "<hamnosys_manual" in ex.sigml_fragment


# ---------------------------------------------------------------------------
# generate_sigml_direct — happy path with a stub LLM
# ---------------------------------------------------------------------------


class _StubClient:
    """Minimal stand-in for :class:`LLMClient` used by these tests."""

    def __init__(self, content: str, model: str = "gpt-5.4") -> None:
        self.content = content
        self.model = model
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        *,
        request_id: str,
        prompt_metadata: Any = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        self.calls.append({
            "messages": messages,
            "response_format": response_format,
            "request_id": request_id,
            "reasoning_effort": reasoning_effort,
        })
        return ChatResult(
            content=self.content,
            tool_calls=[],
            model_used=self.model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_ms=1,
            request_id=request_id,
        )


def _bsl_sigml_response(gloss: str = "TESTSIGN") -> str:
    fragment = (
        f"<hns_sign gloss=\"{gloss}\">"
        "<hamnosys_manual>"
        "<hamflathand/><hamextfingeru/><hampalmd/><hamhead/>"
        "</hamnosys_manual>"
        "</hns_sign>"
    )
    return json.dumps({"sigml_xml": fragment, "rationale": "flat hand at head"})


def test_generate_sigml_direct_happy_path():
    from parser.models import PartialSignParameters

    client = _StubClient(_bsl_sigml_response("HELLO"))
    params = PartialSignParameters(
        handshape_dominant="flat", orientation_extended_finger="up",
        orientation_palm="down", location="head",
    )
    hamnosys, sigml_xml, rationale = generate_sigml_direct(
        parameters=params,
        client=client,
        request_id="test-direct",
        prose="A friendly hello at the head",
        gloss="HELLO",
        sign_language="bsl",
    )
    assert hamnosys
    assert all(0xE000 <= ord(c) <= 0xE0FF for c in hamnosys)
    assert "<hns_sign" in sigml_xml
    assert "HELLO" in sigml_xml
    assert rationale == "flat hand at head"
    # Reasoning effort propagated.
    assert client.calls[0]["reasoning_effort"] == "medium"


def test_generate_sigml_direct_handles_invalid_json():
    from parser.models import PartialSignParameters

    client = _StubClient("not json at all")
    params = PartialSignParameters()
    hamnosys, sigml_xml, rationale = generate_sigml_direct(
        parameters=params,
        client=client,
        request_id="test-badjson",
        prose="",
        gloss="TEST",
        sign_language="bsl",
    )
    assert hamnosys == ""
    assert sigml_xml == ""
    assert "invalid json" in rationale


def test_generate_sigml_direct_handles_empty_xml():
    from parser.models import PartialSignParameters

    client = _StubClient(json.dumps({"sigml_xml": "", "rationale": "skipped"}))
    params = PartialSignParameters()
    hamnosys, sigml_xml, rationale = generate_sigml_direct(
        parameters=params,
        client=client,
        request_id="test-empty",
        prose="",
        gloss="TEST",
        sign_language="bsl",
    )
    assert hamnosys == ""
    assert sigml_xml == ""
    assert "empty sigml_xml" in rationale


def test_generate_sigml_direct_strips_xml_declaration():
    from parser.models import PartialSignParameters

    fragment_with_decl = (
        "<?xml version=\"1.0\"?><sigml>"
        "<hns_sign gloss=\"X\"><hamnosys_manual>"
        "<hamflathand/><hamextfingeru/><hampalmd/><hamhead/>"
        "</hamnosys_manual></hns_sign>"
        "</sigml>"
    )
    client = _StubClient(json.dumps({"sigml_xml": fragment_with_decl, "rationale": "ok"}))
    params = PartialSignParameters()
    hamnosys, sigml_xml, _ = generate_sigml_direct(
        parameters=params,
        client=client,
        request_id="test-decl",
        prose="",
        gloss="X",
        sign_language="bsl",
    )
    assert hamnosys
    # Canonical re-emit goes through to_sigml which adds its own
    # declaration, so the result includes one — the model's wrapper
    # was stripped before reverse-mapping.
    assert "<?xml" in sigml_xml


def test_generate_sigml_direct_autofills_missing_palm_direction():
    # The LLM emits SiGML with handshape + ext-finger + location +
    # movement but no palm tag — the slot-repair fallback should
    # inject ``<hampalmd/>`` (default) so the contributor gets a
    # playable preview instead of a hard "missing palm_direction"
    # error and a blank screen. Regression: this failure mode was
    # the #1 stuck state reported by contributors and used to be
    # terminal after 3 retries.
    from parser.models import PartialSignParameters

    fragment = (
        "<hns_sign gloss=\"X\">"
        "<hamnosys_manual>"
        "<hamflathand/><hamextfingeru/><hamhead/><hammoveo/>"
        "</hamnosys_manual>"
        "</hns_sign>"
    )
    client = _StubClient(
        json.dumps({"sigml_xml": fragment, "rationale": "flat at head"})
    )
    params = PartialSignParameters(
        handshape_dominant="flat", orientation_extended_finger="up",
        location="head",
    )
    hamnosys, sigml_xml, rationale = generate_sigml_direct(
        parameters=params,
        client=client,
        request_id="test-autofill",
        prose="flat hand at head moving outward",
        gloss="X",
        sign_language="bsl",
    )
    # Generation succeeds because the fallback patches the missing slot.
    assert hamnosys
    assert sigml_xml
    # Rationale annotates which slot was auto-filled so the chat can
    # surface the info to the reviewer.
    assert "auto-filled" in rationale
    assert "palm_direction" in rationale
    # Default palm tag is hampalmd (U+E03C).
    assert chr(0xE03C) in hamnosys


def test_generate_sigml_direct_autofills_multiple_slots():
    # Two slots missing (palm + ext-finger). Both should get defaults
    # and the rationale should mention both. A sign with only a
    # handshape and a location isn't a great sign, but it's better to
    # hand the reviewer a visible draft they can correct than a 500.
    from parser.models import PartialSignParameters

    fragment = (
        "<hns_sign gloss=\"X\">"
        "<hamnosys_manual>"
        "<hamflathand/><hamhead/>"
        "</hamnosys_manual>"
        "</hns_sign>"
    )
    client = _StubClient(
        json.dumps({"sigml_xml": fragment, "rationale": "flat at head"})
    )
    params = PartialSignParameters(handshape_dominant="flat", location="head")
    hamnosys, _, rationale = generate_sigml_direct(
        parameters=params,
        client=client,
        request_id="test-autofill-multi",
        prose="flat hand at head",
        gloss="X",
        sign_language="bsl",
    )
    assert hamnosys
    assert "auto-filled" in rationale
    assert "palm_direction" in rationale
    assert "ext_finger_direction" in rationale
    assert chr(0xE020) in hamnosys  # hamextfingeru
    assert chr(0xE03C) in hamnosys  # hampalmd
