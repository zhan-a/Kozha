"""Tests for ``backend.chat2hamnosys.generator.params_to_hamnosys``.

Three layers of coverage:

1. **Vocab / composer mechanics** — deterministic slot resolution, alias
   normalization, two-handed symmetry prefix, unresolved-slot handling.
2. **LLM fallback** — fake :class:`LLMClient` stand-in exercising the
   per-slot resolver and the whole-string repair loop, plus the negative
   paths (malformed JSON, wrong-class codepoints, budget exhaustion).
3. **Gold-fixture replay** — each JSON under ``fixtures/generator/`` is a
   parametrised test case asserting the deterministic composer exactly
   reproduces the recorded ``expected_hamnosys_hex``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from generator import VOCAB, GenerateResult, generate
from generator.params_to_hamnosys import (
    _compose_pieces,
    _llm_repair,
    _llm_resolve_slot,
    _Piece,
)
from hamnosys import SymClass, validate
from llm import ChatResult
from llm.budget import BudgetExceeded
from parser.models import PartialMovementSegment, PartialSignParameters


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "generator"


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Programmable stand-in for :class:`LLMClient`.

    Constructed with a queue of ``content`` strings — each :meth:`chat`
    call pops the next one. A ``raise_exc`` entry is re-raised instead.
    Records each call so tests can inspect arguments.
    """

    def __init__(self, contents: list[Any], model: str = "gpt-4o") -> None:
        self._queue = list(contents)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        *,
        request_id: str,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": messages,
                "response_format": response_format,
                "temperature": temperature,
                "request_id": request_id,
            }
        )
        if not self._queue:
            raise AssertionError("FakeLLMClient received an unexpected call")
        payload = self._queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return ChatResult(
            content=payload,
            tool_calls=[],
            model_used=self.model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_ms=1,
            request_id=request_id,
        )


# ---------------------------------------------------------------------------
# 1. Composer mechanics
# ---------------------------------------------------------------------------


def _basic_params(**overrides) -> PartialSignParameters:
    base: dict[str, Any] = {
        "handshape_dominant": "flat",
        "orientation_extended_finger": "up",
        "orientation_palm": "right",
        "location": "chest",
    }
    base.update(overrides)
    base.setdefault("movement", [])
    return PartialSignParameters(**base)


def test_basic_one_handed_exact_sequence():
    result = generate(_basic_params())
    assert result.hamnosys == "\uE001\uE020\uE03A\uE052"
    assert result.validation.ok
    assert result.used_llm_fallback is False
    assert result.llm_fallback_fields == []
    assert result.confidence == 1.0


def test_basic_one_handed_passes_validator_for_real():
    result = generate(_basic_params())
    # Generator's validation result is populated from the real validator.
    assert result.validation.ok
    assert result.validation.errors == []
    # Double-check: a fresh validate() call agrees.
    fresh = validate(result.hamnosys or "")
    assert fresh.ok


def test_compound_handshape_emits_base_plus_modifier():
    result = generate(_basic_params(handshape_dominant="claw"))
    # claw → E005 E011 per vocab
    assert result.hamnosys is not None
    assert result.hamnosys[0:2] == "\uE005\uE011"


def test_contact_goes_between_location_and_movement():
    params = _basic_params(
        contact="touch",
        movement=[PartialMovementSegment(path="down")],
    )
    result = generate(params)
    # Expected: flat, up, right, chest, touch, down
    assert result.hamnosys == "\uE001\uE020\uE03A\uE052\uE0D1\uE084"


def test_movement_modifiers_ordered_size_speed_repeat():
    params = _basic_params(
        movement=[
            PartialMovementSegment(
                path="down", size_mod="small", speed_mod="fast", repeat="twice"
            )
        ],
    )
    result = generate(params)
    # After chest (E052) → down (E084), small (E0C6), fast (E0C8), twice (E0D8)
    assert result.hamnosys == "\uE001\uE020\uE03A\uE052\uE084\uE0C6\uE0C8\uE0D8"


def test_multiple_movement_segments_kept_in_order():
    params = _basic_params(
        movement=[
            PartialMovementSegment(path="down"),
            PartialMovementSegment(path="up"),
        ],
    )
    result = generate(params)
    assert result.hamnosys == "\uE001\uE020\uE03A\uE052\uE084\uE080"


def test_two_handed_prefixes_hamsymmpar():
    params = _basic_params(
        handshape_nondominant="flat",
        orientation_palm="toward_signer",
    )
    result = generate(params)
    # hamsymmpar is U+E0E8
    assert result.hamnosys is not None
    assert result.hamnosys[0] == "\uE0E8"


def test_no_op_term_emits_nothing():
    # ``repeat: once`` is a legal no-op in the YAML (HamNoSys default is
    # a single production). It must not corrupt the output.
    params = _basic_params(
        movement=[PartialMovementSegment(path="down", repeat="once")]
    )
    result = generate(params)
    assert result.hamnosys == "\uE001\uE020\uE03A\uE052\uE084"


# ---- Alias / normalization ----


@pytest.mark.parametrize(
    "term, expected_codepoint",
    [
        ("flat", 0xE001),
        ("B", 0xE001),               # ASL-LEX letter alias
        ("flat-hand", 0xE001),       # hyphen
        ("Flat Hand", 0xE001),       # space + case
    ],
)
def test_handshape_aliases_resolve(term: str, expected_codepoint: int):
    params = _basic_params(handshape_dominant=term)
    result = generate(params)
    assert result.hamnosys is not None
    assert ord(result.hamnosys[0]) == expected_codepoint


def test_hyphen_bent_5_equals_claw():
    a = generate(_basic_params(handshape_dominant="bent-5"))
    b = generate(_basic_params(handshape_dominant="claw"))
    assert a.hamnosys == b.hamnosys


# ---- Unresolved slots without LLM client ----


def test_unresolved_slot_without_client_returns_none():
    params = _basic_params(handshape_dominant="zebra_handshape")
    result = generate(params)
    assert result.hamnosys is None
    assert result.confidence == 0.0
    assert any("handshape_dominant" in e for e in result.errors)
    assert result.used_llm_fallback is False


# ---------------------------------------------------------------------------
# 2. LLM fallback mechanics
# ---------------------------------------------------------------------------


def _slot_payload(hex_cp: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "codepoint_hex": hex_cp,
            "confidence": confidence,
            "rationale": "test",
        }
    )


def _repair_payload(hex_tokens: str, rationale: str = "repair") -> str:
    return json.dumps({"hamnosys_hex": hex_tokens, "rationale": rationale})


def test_llm_fallback_resolves_missing_slot():
    # An unknown handshape term — the LLM should be asked to pick a
    # codepoint; we return E001 (flat) which is a valid handshape base.
    client = FakeLLMClient([_slot_payload("E001", confidence=0.8)])
    params = _basic_params(handshape_dominant="mystery_shape")
    result = generate(params, client=client, request_id="test-fallback")

    assert result.hamnosys == "\uE001\uE020\uE03A\uE052"
    assert result.used_llm_fallback is True
    assert "handshape_dominant" in result.llm_fallback_fields
    assert result.confidence == pytest.approx(0.8)
    # Request id propagated to the fallback call.
    assert len(client.calls) == 1
    assert client.calls[0]["request_id"].startswith("test-fallback:fallback:")


def test_llm_fallback_rejects_wrong_class_codepoint():
    # Return an ext-finger codepoint for a handshape slot — the generator
    # must reject it and leave the slot unresolved.
    client = FakeLLMClient([_slot_payload("E020", confidence=0.9)])
    params = _basic_params(handshape_dominant="mystery_shape")
    result = generate(params, client=client, request_id="test-wrongclass")

    assert result.hamnosys is None
    assert result.confidence == 0.0
    assert any("not valid for slot" in e for e in result.errors)


def test_llm_fallback_rejects_unknown_codepoint():
    client = FakeLLMClient([_slot_payload("E7FF", confidence=1.0)])
    params = _basic_params(handshape_dominant="mystery_shape")
    result = generate(params, client=client, request_id="test-unknown")

    assert result.hamnosys is None
    assert any("unknown codepoint" in e for e in result.errors)


def test_llm_fallback_invalid_json_unresolved():
    client = FakeLLMClient(["not json at all"])
    params = _basic_params(handshape_dominant="mystery_shape")
    result = generate(params, client=client, request_id="test-badjson")

    assert result.hamnosys is None
    assert any("invalid json" in e for e in result.errors)


def test_budget_exceeded_bubbles_up():
    client = FakeLLMClient([BudgetExceeded(spent=1.0, would_add=0.5, cap=1.0)])
    params = _basic_params(handshape_dominant="mystery_shape")
    with pytest.raises(BudgetExceeded):
        generate(params, client=client, request_id="test-budget")


def test_llm_fallback_fields_ordered_by_slot_sequence():
    # Two unknown slots in different positions — both should be logged
    # in the order the composer visited them.
    client = FakeLLMClient(
        [
            _slot_payload("E001", confidence=0.7),  # handshape
            _slot_payload("E04A", confidence=0.8),  # location
        ]
    )
    params = _basic_params(
        handshape_dominant="unknown_hs",
        location="unknown_loc",
    )
    result = generate(params, client=client, request_id="test-order")
    assert result.llm_fallback_fields == ["handshape_dominant", "location"]
    # Product of confidences.
    assert result.confidence == pytest.approx(0.56, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Validation repair loop
# ---------------------------------------------------------------------------


class _AlwaysInvalidComposeClient(FakeLLMClient):
    """A client that returns repair candidates which still fail validation.

    Used to verify the generator gives up after ``_MAX_VALIDATION_RETRIES``
    repair attempts rather than looping forever.
    """


def test_repair_loop_succeeds_within_two_attempts(monkeypatch):
    # Force the composer to emit an invalid string and let the first
    # repair attempt fix it. We do this by patching ``_assemble`` to
    # inject garbage, then letting the client return the correct string.
    from generator import params_to_hamnosys as mod

    real_assemble = mod._assemble
    fake_invocations: list[int] = []

    def broken_assemble(pieces):
        fake_invocations.append(1)
        if len(fake_invocations) == 1:
            return "ABC"
        return real_assemble(pieces)

    monkeypatch.setattr(mod, "_assemble", broken_assemble)

    client = FakeLLMClient(
        [_repair_payload("E001 E020 E03A E052", rationale="drop garbage")]
    )
    params = _basic_params()
    result = generate(params, client=client, request_id="test-repair")

    assert result.hamnosys == "\uE001\uE020\uE03A\uE052"
    assert result.used_llm_fallback is True
    assert "_repair" in result.llm_fallback_fields


def test_repair_loop_gives_up_after_max_attempts(monkeypatch):
    from generator import params_to_hamnosys as mod

    monkeypatch.setattr(mod, "_assemble", lambda pieces: "ABC")
    # Both repair attempts return strings that still fail validation.
    client = FakeLLMClient([_repair_payload("ABCD"), _repair_payload("DEFG")])

    params = _basic_params()
    result = generate(params, client=client, request_id="test-repair-give-up")

    assert result.hamnosys is None
    assert not result.validation.ok
    # Two repair calls attempted.
    assert len(client.calls) == 2


def test_no_client_and_invalid_string_returns_none(monkeypatch):
    from generator import params_to_hamnosys as mod

    monkeypatch.setattr(mod, "_assemble", lambda pieces: "ABC")
    params = _basic_params()
    result = generate(params)  # no client
    assert result.hamnosys is None
    assert any("no LLM client available to repair" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 4. Piece-level composer (internal API)
# ---------------------------------------------------------------------------


def test_compose_pieces_orders_slots_canonically():
    params = _basic_params(
        contact="touch",
        movement=[PartialMovementSegment(path="down", size_mod="small")],
    )
    pieces = _compose_pieces(params)
    fields = [p.field_name for p in pieces]
    # Expected canonical order.
    assert fields == [
        "handshape_dominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
        "movement[0].path",
        "movement[0].size_mod",
    ]


def test_compose_pieces_handles_empty_optional_slots():
    params = _basic_params()
    pieces = _compose_pieces(params)
    fields = [p.field_name for p in pieces]
    assert "contact" not in fields
    assert not any(f.startswith("movement[") for f in fields)


# ---------------------------------------------------------------------------
# 5. Gold-fixture replay
# ---------------------------------------------------------------------------


def _fixture_ids() -> list[str]:
    return sorted(p.stem for p in FIXTURES_DIR.glob("*.json"))


@pytest.mark.parametrize("fixture_id", _fixture_ids())
def test_gold_fixture_deterministic_reproduction(fixture_id: str):
    """Each gold fixture must be reproducible by the deterministic composer alone."""
    fx_path = FIXTURES_DIR / f"{fixture_id}.json"
    fx = json.loads(fx_path.read_text(encoding="utf-8"))
    params_dict = fx["parameters"]
    segments = [
        PartialMovementSegment(**s) for s in params_dict.get("movement", []) or []
    ]
    params = PartialSignParameters(
        handshape_dominant=params_dict.get("handshape_dominant"),
        handshape_nondominant=params_dict.get("handshape_nondominant"),
        orientation_extended_finger=params_dict.get("orientation_extended_finger"),
        orientation_palm=params_dict.get("orientation_palm"),
        location=params_dict.get("location"),
        contact=params_dict.get("contact"),
        movement=segments,
    )
    result = generate(params)
    assert result.hamnosys is not None, (
        f"{fixture_id}: generator returned None "
        f"(errors={result.errors}, validation={result.validation.summary()})"
    )
    produced_hex = " ".join(f"{ord(c):04X}" for c in result.hamnosys)
    assert produced_hex == fx["expected_hamnosys_hex"], (
        f"{fixture_id}: produced {produced_hex!r} "
        f"vs gold {fx['expected_hamnosys_hex']!r}"
    )
    assert result.used_llm_fallback is False, (
        f"{fixture_id}: gold pair unexpectedly required LLM fallback"
    )
    assert result.validation.ok, (
        f"{fixture_id}: produced string failed validation: "
        f"{result.validation.summary()}"
    )


# ---------------------------------------------------------------------------
# 6. Vocab sanity checks — ensure the YAML author didn't regress.
# ---------------------------------------------------------------------------


def test_vocab_has_all_required_slots():
    for slot in (
        "handshape",
        "orientation_ext_finger",
        "orientation_palm",
        "location",
        "contact",
        "movement_path",
        "size_mod",
        "speed_mod",
        "repeat",
    ):
        assert slot in VOCAB.slots, f"missing vocab slot: {slot}"


def test_vocab_handshape_covers_asl_lex_minimum():
    # ASL-LEX 2.0 minimum set: we require at least these well-known
    # handshapes to be authored.
    for term in ("flat", "fist", "index", "five", "claw", "cee_12", "pinch_all"):
        assert VOCAB.has_term("handshape", term), f"handshape term missing: {term}"


def test_vocab_lookup_respects_normalization():
    for variant in ("flat", "FLAT", "Flat Hand", "flat-hand"):
        assert VOCAB.has_term("handshape", variant), variant
        entry = VOCAB.lookup("handshape", variant)
        assert entry is not None
        assert entry.codepoints == (0xE001,)
