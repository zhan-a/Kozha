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
        prompt_metadata: Any = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": messages,
                "response_format": response_format,
                "temperature": temperature,
                "request_id": request_id,
                "prompt_metadata": prompt_metadata,
                "reasoning_effort": reasoning_effort,
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
# 2. LLM fallback mechanics — SiGML-direct is the only LLM path
# ---------------------------------------------------------------------------
#
# Earlier revisions of this file exercised ``_llm_resolve_slot`` (per-
# slot codepoint pick) and ``_llm_repair`` (rewrite a HamNoSys string).
# Both paths asked the model to emit HamNoSys PUA codepoints directly
# — per the user's feedback, unreliable. ("Make sure it generates
# written SiGML and only then maybe converts to regular HamNoSys.")
# Those paths are gone. The only LLM path now is
# ``generate_sigml_direct`` (tested in ``test_sigml_direct.py``); the
# tests below cover the non-LLM and integration behaviour of
# ``generate``.


def _sigml_fail_payload() -> str:
    """A sigml-direct response that's well-formed JSON but yields no
    SiGML. Two of these in a row exhaust the SiGML-direct retry loop."""
    return json.dumps({"sigml_xml": "", "rationale": "skip for test"})


def test_sigml_direct_failure_surfaces_unresolved_slots():
    # SiGML-direct exhausts its 2-attempt internal retry; the
    # generator falls through and returns a failure envelope listing
    # each still-unresolved slot by name so the chat panel can show
    # the contributor what was left open.
    client = FakeLLMClient([_sigml_fail_payload(), _sigml_fail_payload()])
    params = _basic_params(handshape_dominant="mystery_shape")
    result = generate(params, client=client, request_id="test-unresolved")
    assert result.hamnosys is None
    assert result.confidence == 0.0
    assert any("handshape_dominant" in e for e in result.errors)


def test_budget_exceeded_bubbles_up():
    # BudgetExceeded fires on the first SiGML-direct LLM call; it must
    # propagate rather than being swallowed by the retry loop.
    client = FakeLLMClient([BudgetExceeded(spent=1.0, would_add=0.5, cap=1.0)])
    params = _basic_params(handshape_dominant="mystery_shape")
    with pytest.raises(BudgetExceeded):
        generate(params, client=client, request_id="test-budget")


# ---------------------------------------------------------------------------
# 3. Graceful degrade without a client
# ---------------------------------------------------------------------------


def test_no_client_and_unresolved_slot_returns_none(monkeypatch):
    from generator import params_to_hamnosys as mod

    # Force the auto-construct path to return None regardless of
    # whether the developer's shell has OPENAI_API_KEY set — this
    # test covers the graceful-degrade branch when no LLM client is
    # available. The composer leaves a slot unresolved (zebra
    # handshape isn't in VOCAB) and ``generate`` must return None
    # with an actionable error, not raise.
    monkeypatch.setattr(mod, "_maybe_construct_client", lambda c: None)
    params = _basic_params(handshape_dominant="zebra_handshape")
    result = generate(params)  # no client
    assert result.hamnosys is None
    assert any("no LLM client available" in e for e in result.errors)


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


def test_vocab_movement_path_includes_waving_aliases():
    # Regression guard — polish-14 gate. Contributors describing a
    # greeting wave as "waving" previously bounced into the LLM fallback
    # and surfaced a user-facing BadRequestError. "waving" / "wave" and
    # their side-to-side phrasings are aliases for HamNoSys pendulum
    # swing (U+E0A6).
    for term in (
        "waving",
        "wave",
        "waves",
        "hand wave",
        "hand waving",
        "side to side",
        "side-to-side",
        "side to side wave",
        "wave side to side",
        "pendulum",
        "pendulum swing",
    ):
        assert VOCAB.has_term("movement_path", term), term
        entry = VOCAB.lookup("movement_path", term)
        assert entry is not None
        assert entry.codepoints == (0xE0A6,)


def test_generate_resolves_waving_without_llm():
    # Authored "dominant hand up and waving" parse — the downstream
    # movement path is "waving". Prior bug: this slot went to the LLM
    # fallback and could fail with BadRequestError. Now the deterministic
    # composer must resolve it without any LLM involvement.
    params = _basic_params(
        orientation_palm="down",
        movement=[PartialMovementSegment(path="waving")],
    )
    result = generate(params, client=None)
    assert result.hamnosys is not None
    assert result.used_llm_fallback is False
    assert result.llm_fallback_fields == []
    # trailing codepoint is the swing/waving pendulum
    assert result.hamnosys.endswith(""), result.hamnosys
