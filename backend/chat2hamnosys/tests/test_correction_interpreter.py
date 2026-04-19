"""Tests for :mod:`backend.chat2hamnosys.correct.correction_interpreter`.

Two layers:

1. Mechanics — the restart fast path, the strict-JSON schema shape, the
   system prompt contents, the LLM-exception fallback, the malformed-JSON
   fallback, budget propagation, and the direct ``_apply_field_changes``
   writer.
2. Fixture replay — the 20 JSON fixtures under ``fixtures/correct/`` feed
   parameter bundles + prose + a correction + a recorded LLM response,
   and the test asserts the oracle (``expected_intent``, ``expected_paths``,
   ``expected_needs_confirmation``, ``expected_follow_up_present``,
   ``expected_updated_slots``) matches what
   :func:`interpret_correction` returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from correct import (
    ApplyOutcome,
    Correction,
    CorrectionApplyError,
    CorrectionIntent,
    CorrectionPlan,
    FieldChange,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    apply_correction,
    interpret_correction,
    log_correction_applied,
    log_session_accepted,
)
from correct.correction_interpreter import (
    _apply_field_changes,
    _apply_single_change,
    _is_empty_segment,
    _looks_like_restart,
    _vague_plan,
)
from generator import GenerateResult
from hamnosys.validator import ValidationResult
from llm import ChatResult
from llm.budget import BudgetExceeded
from parser.models import (
    PartialMovementSegment,
    PartialNonManualFeatures,
    PartialSignParameters,
)
from rendering.preview import PreviewResult, PreviewStatus
from session.orchestrator import on_correction, run_generation, start_session
from session.state import (
    AuthoringSession,
    CorrectionAppliedEvent,
    GeneratedEvent,
    SessionState,
    SignEntryDraft,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "correct"
SOLID_HAMNOSYS = "\uE000\uE020\uE03C\uE049"


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
        temperature: float = 0.2,
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
    """Raises ``exc`` on every call."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def chat(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise self.exc


# ---------------------------------------------------------------------------
# Session builders
# ---------------------------------------------------------------------------


def _solid_partial() -> PartialSignParameters:
    return PartialSignParameters(
        handshape_dominant="fist",
        orientation_extended_finger="up",
        orientation_palm="down",
        location="temple",
    )


def _session_in_applying_correction(
    *,
    params: PartialSignParameters | None = None,
    hamnosys: str = SOLID_HAMNOSYS,
    prose: str = "A closed fist at the temple.",
) -> AuthoringSession:
    """Build a session already parked in APPLYING_CORRECTION for tests."""
    draft = SignEntryDraft(
        gloss="TEMPLE",
        sign_language="bsl",
        author_signer_id="alice",
        description_prose=prose,
        parameters_partial=params or _solid_partial(),
        hamnosys=hamnosys,
    )
    return AuthoringSession(draft=draft, state=SessionState.APPLYING_CORRECTION)


def _drive_session_to_applying(correction_text: str) -> AuthoringSession:
    """Drive a real state machine to APPLYING_CORRECTION for end-to-end tests."""
    s = start_session(signer_id="alice", gloss="TEMPLE")
    s = s.with_draft(
        description_prose="A closed fist at the temple.",
        parameters_partial=_solid_partial().model_copy(
            update={"movement": [PartialMovementSegment(path="down")]}
        ),
    )
    s = s.with_state(SessionState.GENERATING)
    s = run_generation(s, render_fn=None)
    assert s.state is SessionState.RENDERED
    s = on_correction(s, Correction(raw_text=correction_text))
    assert s.state is SessionState.APPLYING_CORRECTION
    return s


# ---------------------------------------------------------------------------
# Stubs for apply_correction's regeneration chain
# ---------------------------------------------------------------------------


def _ok_generate(params: PartialSignParameters) -> GenerateResult:
    return GenerateResult(
        hamnosys="\uE001\uE020\uE03C\uE048",
        validation=ValidationResult(ok=True, errors=[], warnings=[]),
        used_llm_fallback=False,
        llm_fallback_fields=[],
        confidence=1.0,
        errors=[],
    )


def _failing_generate(params: PartialSignParameters) -> GenerateResult:
    return GenerateResult(
        hamnosys=None,
        validation=ValidationResult(
            ok=False, errors=["unresolvable slot"], warnings=[]
        ),
        used_llm_fallback=False,
        llm_fallback_fields=[],
        confidence=0.0,
        errors=["unresolvable slot"],
    )


def _noop_sigml(hamnosys: str, *, gloss: str, non_manual=None) -> str:
    return f"<sigml gloss='{gloss}'>{hamnosys}</sigml>"


# ---------------------------------------------------------------------------
# Mechanics: restart fast path
# ---------------------------------------------------------------------------


class TestRestartFastPath:
    def test_looks_like_restart_detects_start_over(self) -> None:
        assert _looks_like_restart("let me start over") is True

    def test_looks_like_restart_detects_whole_wrong(self) -> None:
        assert _looks_like_restart("the whole sign is wrong") is True

    def test_looks_like_restart_detects_scrap(self) -> None:
        assert _looks_like_restart("scrap this") is True

    def test_looks_like_restart_detects_bare_restart(self) -> None:
        assert _looks_like_restart("restart") is True
        assert _looks_like_restart("  Reset.  ") is True

    def test_looks_like_restart_rejects_minor_correction(self) -> None:
        assert _looks_like_restart("the handshape is wrong") is False

    def test_interpret_correction_restart_skips_llm(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(content="should-not-be-called")
        plan = interpret_correction(
            session,
            Correction(raw_text="the whole sign is wrong, let me start over"),
            client=fake,
        )
        assert plan.intent is CorrectionIntent.RESTART
        assert plan.field_changes == []
        assert plan.needs_user_confirmation is True
        assert plan.follow_up_question is None
        assert fake.calls == []  # LLM never invoked


# ---------------------------------------------------------------------------
# Mechanics: schema / prompt
# ---------------------------------------------------------------------------


class TestSchema:
    def test_top_level_shape(self) -> None:
        assert RESPONSE_SCHEMA["type"] == "json_schema"
        inner = RESPONSE_SCHEMA["json_schema"]
        assert inner["strict"] is True
        schema = inner["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {
            "intent",
            "explanation",
            "needs_user_confirmation",
            "follow_up_question",
            "field_changes",
        }

    def test_intent_enum_matches_python_enum(self) -> None:
        intents = RESPONSE_SCHEMA["json_schema"]["schema"]["properties"][
            "intent"
        ]["enum"]
        assert set(intents) == {i.value for i in CorrectionIntent}

    def test_field_change_item_is_strict(self) -> None:
        item = RESPONSE_SCHEMA["json_schema"]["schema"]["properties"][
            "field_changes"
        ]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == {
            "path",
            "old_value",
            "new_value",
            "confidence",
        }
        conf = item["properties"]["confidence"]
        assert conf["minimum"] == 0.0
        assert conf["maximum"] == 1.0

    def test_no_unsupported_keywords(self) -> None:
        forbidden = {"default", "title", "examples", "pattern", "minLength"}

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for bad in forbidden:
                    assert bad not in node, f"forbidden keyword {bad!r}"
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(RESPONSE_SCHEMA)


class TestSystemPrompt:
    def test_mentions_minimal_diff_rule(self) -> None:
        assert "MINIMAL" in SYSTEM_PROMPT
        assert "smallest" in SYSTEM_PROMPT.lower()

    def test_mentions_every_intent(self) -> None:
        for intent in CorrectionIntent:
            assert intent.value in SYSTEM_PROMPT

    def test_includes_allowed_scalar_slots(self) -> None:
        for slot in (
            "handshape_dominant",
            "handshape_nondominant",
            "orientation_palm",
            "orientation_extended_finger",
            "location",
            "contact",
        ):
            assert slot in SYSTEM_PROMPT

    def test_includes_non_manual_leaves(self) -> None:
        for leaf in (
            "non_manual.mouth_picture",
            "non_manual.eye_gaze",
            "non_manual.head_movement",
            "non_manual.eyebrows",
            "non_manual.facial_expression",
        ):
            assert leaf in SYSTEM_PROMPT

    def test_includes_movement_subfield_shape(self) -> None:
        assert "movement[i].path" in SYSTEM_PROMPT
        assert "movement[i].repeat" in SYSTEM_PROMPT

    def test_no_pua_codepoints(self) -> None:
        for ch in SYSTEM_PROMPT:
            assert not (0xE000 <= ord(ch) <= 0xF8FF), (
                f"prompt contains PUA U+{ord(ch):04X}"
            )


# ---------------------------------------------------------------------------
# Mechanics: fallback paths
# ---------------------------------------------------------------------------


class TestInterpretFallbacks:
    def test_empty_raw_text_vague_without_llm(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(content="unused")
        plan = interpret_correction(
            session, Correction(raw_text="   "), client=fake
        )
        assert plan.intent is CorrectionIntent.VAGUE
        assert plan.follow_up_question
        assert fake.calls == []

    def test_malformed_json_falls_back_to_vague(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(content="this is not json")
        plan = interpret_correction(
            session, Correction(raw_text="swap the handshape"), client=fake
        )
        assert plan.intent is CorrectionIntent.VAGUE
        assert plan.needs_user_confirmation is True
        assert plan.follow_up_question
        assert len(fake.calls) == 1

    def test_unknown_intent_falls_back_to_vague(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(
            content=json.dumps(
                {
                    "intent": "bogus_intent",
                    "explanation": "",
                    "needs_user_confirmation": True,
                    "follow_up_question": None,
                    "field_changes": [],
                }
            )
        )
        plan = interpret_correction(
            session, Correction(raw_text="swap the handshape"), client=fake
        )
        assert plan.intent is CorrectionIntent.VAGUE
        assert plan.follow_up_question

    def test_apply_diff_with_no_changes_falls_back_to_vague(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(
            content=json.dumps(
                {
                    "intent": "apply_diff",
                    "explanation": "",
                    "needs_user_confirmation": False,
                    "follow_up_question": None,
                    "field_changes": [],
                }
            )
        )
        plan = interpret_correction(
            session, Correction(raw_text="change something"), client=fake
        )
        assert plan.intent is CorrectionIntent.VAGUE
        assert plan.follow_up_question

    def test_apply_diff_with_bad_path_falls_back_to_vague(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(
            content=json.dumps(
                {
                    "intent": "apply_diff",
                    "explanation": "bogus",
                    "needs_user_confirmation": False,
                    "follow_up_question": None,
                    "field_changes": [
                        {
                            "path": "not_a_slot",
                            "old_value": None,
                            "new_value": "x",
                            "confidence": 0.9,
                        }
                    ],
                }
            )
        )
        plan = interpret_correction(
            session, Correction(raw_text="do something"), client=fake
        )
        assert plan.intent is CorrectionIntent.VAGUE

    def test_llm_exception_falls_back_to_vague(self) -> None:
        session = _session_in_applying_correction()
        raising = RaisingLLMClient(RuntimeError("network down"))
        plan = interpret_correction(
            session,
            Correction(raw_text="swap the handshape"),
            client=raising,
        )
        assert plan.intent is CorrectionIntent.VAGUE
        assert plan.follow_up_question

    def test_budget_exceeded_propagates(self) -> None:
        session = _session_in_applying_correction()
        raising = RaisingLLMClient(
            BudgetExceeded(spent=1.0, would_add=0.5, cap=1.0)
        )
        with pytest.raises(BudgetExceeded):
            interpret_correction(
                session,
                Correction(raw_text="swap the handshape"),
                client=raising,
            )

    def test_missing_partial_returns_vague(self) -> None:
        draft = SignEntryDraft(
            gloss="TEMPLE",
            sign_language="bsl",
            author_signer_id="alice",
            parameters_partial=None,
        )
        session = AuthoringSession(
            draft=draft, state=SessionState.APPLYING_CORRECTION
        )
        fake = FakeLLMClient(content="unused")
        plan = interpret_correction(
            session,
            Correction(raw_text="swap handshape"),
            client=fake,
        )
        assert plan.intent is CorrectionIntent.VAGUE
        assert fake.calls == []

    def test_rejects_non_correction_type(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(content="unused")
        with pytest.raises(TypeError, match="Correction"):
            interpret_correction(session, "not a Correction", client=fake)  # type: ignore[arg-type]


class TestInterpretWiring:
    def test_sends_parameters_and_correction_to_llm(self) -> None:
        session = _session_in_applying_correction()
        recorded = json.dumps(
            {
                "intent": "apply_diff",
                "explanation": "switch handshape",
                "needs_user_confirmation": False,
                "follow_up_question": None,
                "field_changes": [
                    {
                        "path": "handshape_dominant",
                        "old_value": "fist",
                        "new_value": "flat-O",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        fake = FakeLLMClient(content=recorded)
        correction = Correction(
            raw_text="make it a flat-O",
            target_time_ms=1500,
            target_region="hand",
        )
        plan = interpret_correction(
            session, correction, client=fake, request_id="req-corr-1"
        )
        assert plan.intent is CorrectionIntent.APPLY_DIFF
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["request_id"] == "req-corr-1"
        assert call["temperature"] == 0.1
        assert call["response_format"] is RESPONSE_SCHEMA
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][0]["content"] == SYSTEM_PROMPT
        payload = json.loads(call["messages"][1]["content"])
        assert payload["correction"]["raw_text"] == "make it a flat-O"
        assert payload["correction"]["target_time_ms"] == 1500
        assert payload["correction"]["target_region"] == "hand"
        assert payload["parameters"]["handshape_dominant"] == "fist"
        assert payload["hamnosys"] == SOLID_HAMNOSYS

    def test_generates_request_id_when_missing(self) -> None:
        session = _session_in_applying_correction()
        fake = FakeLLMClient(content="{}")
        interpret_correction(
            session, Correction(raw_text="change something"), client=fake
        )
        assert fake.calls[0]["request_id"].startswith("correct-")


# ---------------------------------------------------------------------------
# Mechanics: _apply_field_changes writer
# ---------------------------------------------------------------------------


class TestApplyFieldChanges:
    def test_empty_changes_is_identity(self) -> None:
        params = _solid_partial()
        out = _apply_field_changes(params, [])
        assert out is params  # short-circuits to the same object

    def test_scalar_slot_swap(self) -> None:
        params = _solid_partial()
        change = FieldChange(
            path="handshape_dominant",
            old_value="fist",
            new_value="flat-O",
            confidence=0.9,
        )
        out = _apply_field_changes(params, [change])
        assert out.handshape_dominant == "flat-O"
        # Original untouched.
        assert params.handshape_dominant == "fist"

    def test_scalar_slot_to_none(self) -> None:
        params = _solid_partial().model_copy(update={"handshape_nondominant": "flat"})
        change = FieldChange(
            path="handshape_nondominant",
            old_value="flat",
            new_value=None,
            confidence=0.9,
        )
        out = _apply_field_changes(params, [change])
        assert out.handshape_nondominant is None

    def test_scalar_slot_empty_string_becomes_none(self) -> None:
        params = _solid_partial()
        change = FieldChange(
            path="handshape_dominant",
            old_value="fist",
            new_value="   ",
            confidence=0.9,
        )
        out = _apply_field_changes(params, [change])
        assert out.handshape_dominant is None

    def test_movement_path_swap(self) -> None:
        params = _solid_partial().model_copy(
            update={"movement": [PartialMovementSegment(path="up")]}
        )
        change = FieldChange(
            path="movement[0].path",
            old_value="up",
            new_value="down",
            confidence=0.95,
        )
        out = _apply_field_changes(params, [change])
        assert len(out.movement) == 1
        assert out.movement[0].path == "down"

    def test_movement_repeat_add(self) -> None:
        params = _solid_partial().model_copy(
            update={"movement": [PartialMovementSegment(path="straight")]}
        )
        change = FieldChange(
            path="movement[0].repeat",
            old_value=None,
            new_value="twice",
            confidence=0.9,
        )
        out = _apply_field_changes(params, [change])
        assert out.movement[0].repeat == "twice"

    def test_movement_extends_list_for_high_index(self) -> None:
        params = _solid_partial()
        assert params.movement == []
        change = FieldChange(
            path="movement[1].path",
            old_value=None,
            new_value="up",
            confidence=0.8,
        )
        out = _apply_field_changes(params, [change])
        # Index 0 is empty-trimmed back off, index 1 is kept because it's
        # the last populated segment.
        assert len(out.movement) == 2
        assert out.movement[0] == PartialMovementSegment()
        assert out.movement[1].path == "up"

    def test_movement_emptying_trims_trailing_segments(self) -> None:
        params = _solid_partial().model_copy(
            update={
                "movement": [
                    PartialMovementSegment(path="straight"),
                    PartialMovementSegment(path="up"),
                ]
            }
        )
        change = FieldChange(
            path="movement[1].path",
            old_value="up",
            new_value=None,
            confidence=0.9,
        )
        out = _apply_field_changes(params, [change])
        assert len(out.movement) == 1
        assert out.movement[0].path == "straight"

    def test_non_manual_add_eyebrows(self) -> None:
        params = _solid_partial()
        change = FieldChange(
            path="non_manual.eyebrows",
            old_value=None,
            new_value="raised",
            confidence=0.95,
        )
        out = _apply_field_changes(params, [change])
        assert out.non_manual is not None
        assert out.non_manual.eyebrows == "raised"

    def test_non_manual_updates_existing(self) -> None:
        params = _solid_partial().model_copy(
            update={
                "non_manual": PartialNonManualFeatures(eyebrows="raised")
            }
        )
        change = FieldChange(
            path="non_manual.mouth_picture",
            old_value=None,
            new_value="pah",
            confidence=0.9,
        )
        out = _apply_field_changes(params, [change])
        assert out.non_manual is not None
        assert out.non_manual.eyebrows == "raised"
        assert out.non_manual.mouth_picture == "pah"

    def test_non_manual_collapses_to_none_when_fully_empty(self) -> None:
        params = _solid_partial().model_copy(
            update={
                "non_manual": PartialNonManualFeatures(eyebrows="raised")
            }
        )
        change = FieldChange(
            path="non_manual.eyebrows",
            old_value="raised",
            new_value=None,
            confidence=0.9,
        )
        out = _apply_field_changes(params, [change])
        assert out.non_manual is None

    def test_unknown_path_raises(self) -> None:
        params = _solid_partial()
        change = FieldChange(
            path="not_a_slot",
            old_value=None,
            new_value="x",
            confidence=0.9,
        )
        with pytest.raises(CorrectionApplyError):
            _apply_single_change(params, change)

    def test_unknown_non_manual_leaf_raises(self) -> None:
        params = _solid_partial()
        change = FieldChange(
            path="non_manual.bogus",
            old_value=None,
            new_value="x",
            confidence=0.9,
        )
        with pytest.raises(CorrectionApplyError):
            _apply_single_change(params, change)

    def test_is_empty_segment(self) -> None:
        assert _is_empty_segment(PartialMovementSegment()) is True
        assert _is_empty_segment(PartialMovementSegment(path="up")) is False


# ---------------------------------------------------------------------------
# Mechanics: apply_correction state transitions
# ---------------------------------------------------------------------------


class TestApplyCorrection:
    def test_requires_applying_correction_state(self, tmp_path: Path) -> None:
        s = start_session(signer_id="alice", gloss="TEMPLE")
        # AWAITING_DESCRIPTION is not APPLYING_CORRECTION.
        plan = _vague_plan("nope", "what specifically?")
        with pytest.raises(RuntimeError, match="APPLYING_CORRECTION"):
            apply_correction(s, plan, log_dir=tmp_path)

    def test_apply_diff_regenerates_and_goes_to_rendered(self, tmp_path: Path) -> None:
        session = _drive_session_to_applying("switch handshape")
        updated = session.draft.parameters_partial.model_copy(
            update={"handshape_dominant": "flat"}
        )
        plan = CorrectionPlan(
            intent=CorrectionIntent.APPLY_DIFF,
            field_changes=[
                FieldChange(
                    path="handshape_dominant",
                    old_value="fist",
                    new_value="flat",
                    confidence=0.9,
                )
            ],
            explanation="swap to flat hand",
            needs_user_confirmation=False,
            follow_up_question=None,
            updated_params=updated,
        )
        outcome = apply_correction(
            session,
            plan,
            generate_fn=_ok_generate,
            to_sigml_fn=_noop_sigml,
            render_fn=None,
            log_dir=tmp_path,
        )
        assert outcome.outcome == "ok"
        assert outcome.session.state is SessionState.RENDERED
        applied = [
            e for e in outcome.session.history
            if isinstance(e, CorrectionAppliedEvent)
        ]
        assert len(applied) == 1
        assert applied[0].summary == "swap to flat hand"
        assert applied[0].field_changes[0]["path"] == "handshape_dominant"

    def test_apply_diff_validation_failure_parks_in_awaiting_correction(
        self, tmp_path: Path
    ) -> None:
        session = _drive_session_to_applying("break everything")
        updated = session.draft.parameters_partial.model_copy(
            update={"handshape_dominant": "zzz-unresolvable"}
        )
        plan = CorrectionPlan(
            intent=CorrectionIntent.APPLY_DIFF,
            field_changes=[
                FieldChange(
                    path="handshape_dominant",
                    old_value="fist",
                    new_value="zzz-unresolvable",
                    confidence=0.5,
                )
            ],
            explanation="swap to garbage",
            needs_user_confirmation=False,
            follow_up_question=None,
            updated_params=updated,
        )
        outcome = apply_correction(
            session,
            plan,
            generate_fn=_failing_generate,
            to_sigml_fn=_noop_sigml,
            render_fn=None,
            log_dir=tmp_path,
        )
        assert outcome.outcome == "validation_failed"
        assert outcome.session.state is SessionState.AWAITING_CORRECTION
        assert outcome.session.draft.generation_errors

    def test_apply_diff_with_no_updated_params_is_noop(
        self, tmp_path: Path
    ) -> None:
        session = _drive_session_to_applying("something")
        plan = CorrectionPlan(
            intent=CorrectionIntent.APPLY_DIFF,
            field_changes=[
                FieldChange(
                    path="handshape_dominant",
                    old_value="fist",
                    new_value="flat",
                    confidence=0.5,
                )
            ],
            explanation="partial plan",
            needs_user_confirmation=False,
            follow_up_question=None,
            updated_params=None,
        )
        outcome = apply_correction(
            session,
            plan,
            generate_fn=_ok_generate,
            to_sigml_fn=_noop_sigml,
            render_fn=None,
            log_dir=tmp_path,
        )
        assert outcome.outcome == "noop"
        assert outcome.session.state is SessionState.AWAITING_CORRECTION

    def test_restart_clears_draft_and_awaits_description(
        self, tmp_path: Path
    ) -> None:
        session = _drive_session_to_applying("whole thing wrong")
        plan = CorrectionPlan(
            intent=CorrectionIntent.RESTART,
            field_changes=[],
            explanation="discard and start over",
            needs_user_confirmation=True,
            follow_up_question=None,
            updated_params=None,
        )
        outcome = apply_correction(session, plan, log_dir=tmp_path)
        assert outcome.outcome == "noop"
        assert outcome.session.state is SessionState.AWAITING_DESCRIPTION
        draft = outcome.session.draft
        assert draft.parameters_partial is None
        assert draft.hamnosys is None
        assert draft.description_prose == ""

    @pytest.mark.parametrize(
        "intent",
        [CorrectionIntent.ELABORATE, CorrectionIntent.CONTRADICTION, CorrectionIntent.VAGUE],
    )
    def test_nondiff_intents_park_in_awaiting_correction(
        self, intent: CorrectionIntent, tmp_path: Path
    ) -> None:
        session = _drive_session_to_applying("it looks off")
        plan = CorrectionPlan(
            intent=intent,
            field_changes=[],
            explanation="not enough info",
            needs_user_confirmation=True,
            follow_up_question="Which slot?",
            updated_params=None,
        )
        outcome = apply_correction(session, plan, log_dir=tmp_path)
        assert outcome.outcome == "noop"
        assert outcome.session.state is SessionState.AWAITING_CORRECTION
        applied = [
            e for e in outcome.session.history
            if isinstance(e, CorrectionAppliedEvent)
        ]
        assert applied
        # Nondiff intents write an empty field_changes list.
        assert applied[-1].field_changes == []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_apply_correction_writes_metric(self, tmp_path: Path) -> None:
        session = _drive_session_to_applying("small fix")
        updated = session.draft.parameters_partial.model_copy(
            update={"handshape_dominant": "flat"}
        )
        plan = CorrectionPlan(
            intent=CorrectionIntent.APPLY_DIFF,
            field_changes=[
                FieldChange(
                    path="handshape_dominant",
                    old_value="fist",
                    new_value="flat",
                    confidence=0.9,
                )
            ],
            explanation="swap",
            needs_user_confirmation=False,
            follow_up_question=None,
            updated_params=updated,
        )
        apply_correction(
            session,
            plan,
            generate_fn=_ok_generate,
            to_sigml_fn=_noop_sigml,
            render_fn=None,
            log_dir=tmp_path,
        )
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "correction_applied"
        assert record["intent"] == "apply_diff"
        assert record["outcome"] == "ok"
        assert record["field_count"] == 1
        assert record["corrections_count"] == session.draft.corrections_count
        assert record["session_id"] == str(session.id)

    def test_log_correction_applied_direct(self, tmp_path: Path) -> None:
        path = log_correction_applied(
            session_id="abc",
            intent="apply_diff",
            field_count=2,
            outcome="ok",
            corrections_count=3,
            log_dir=tmp_path,
        )
        assert path.exists()
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["event"] == "correction_applied"
        assert record["field_count"] == 2
        assert record["corrections_count"] == 3

    def test_log_session_accepted_direct(self, tmp_path: Path) -> None:
        path = log_session_accepted(
            session_id="sess-1",
            sign_entry_id="entry-9",
            corrections_count=5,
            log_dir=tmp_path,
        )
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["event"] == "session_accepted"
        assert record["sign_entry_id"] == "entry-9"
        assert record["corrections_count"] == 5

    def test_log_session_accepted_nullable_sign_entry(self, tmp_path: Path) -> None:
        path = log_session_accepted(
            session_id="sess-1",
            sign_entry_id=None,
            corrections_count=0,
            log_dir=tmp_path,
        )
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["sign_entry_id"] is None


# ---------------------------------------------------------------------------
# Fixture replay
# ---------------------------------------------------------------------------


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.json"))
assert FIXTURE_FILES, f"no fixtures found in {FIXTURES_DIR}"


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_FILES,
    ids=[p.stem for p in FIXTURE_FILES],
)
def test_fixture_replay(fixture_path: Path) -> None:
    fixture = _load_fixture(fixture_path)
    params = PartialSignParameters.model_validate(fixture["current_params"])
    draft = SignEntryDraft(
        gloss="TEST",
        sign_language="bsl",
        author_signer_id="alice",
        description_prose=fixture["original_prose"],
        parameters_partial=params,
        hamnosys=fixture["current_hamnosys"],
    )
    session = AuthoringSession(
        draft=draft, state=SessionState.APPLYING_CORRECTION
    )
    correction = Correction(
        raw_text=fixture["correction"]["raw_text"],
        target_time_ms=fixture["correction"]["target_time_ms"],
        target_region=fixture["correction"]["target_region"],
    )
    fake = FakeLLMClient(content=json.dumps(fixture["recorded_response"]))
    plan = interpret_correction(session, correction, client=fake)

    if fixture["expect_llm_called"]:
        assert len(fake.calls) == 1, (
            f"[{fixture['id']}] expected one LLM call, got {len(fake.calls)}"
        )
    else:
        assert fake.calls == [], (
            f"[{fixture['id']}] expected no LLM call, got {len(fake.calls)}"
        )

    assert plan.intent.value == fixture["expected_intent"], (
        f"[{fixture['id']}] expected intent {fixture['expected_intent']!r}, "
        f"got {plan.intent.value!r}"
    )

    actual_paths = [c.path for c in plan.field_changes]
    assert actual_paths == fixture["expected_paths"], (
        f"[{fixture['id']}] expected paths {fixture['expected_paths']}, "
        f"got {actual_paths}"
    )

    assert plan.needs_user_confirmation == fixture["expected_needs_confirmation"], (
        f"[{fixture['id']}] needs_confirmation mismatch"
    )

    if fixture["expected_follow_up_present"]:
        assert plan.follow_up_question, (
            f"[{fixture['id']}] expected a follow-up question, got none"
        )
    else:
        assert not plan.follow_up_question, (
            f"[{fixture['id']}] did not expect a follow-up question, "
            f"got {plan.follow_up_question!r}"
        )

    # For APPLY_DIFF plans, assert the updated_params reflect the slots.
    expected_slots = fixture["expected_updated_slots"]
    if plan.intent is CorrectionIntent.APPLY_DIFF:
        assert plan.updated_params is not None
        dumped = plan.updated_params.model_dump(mode="json")
        for key, expected in expected_slots.items():
            assert dumped[key] == expected, (
                f"[{fixture['id']}] slot {key!r} expected {expected!r}, "
                f"got {dumped[key]!r}"
            )
    else:
        assert plan.updated_params is None
