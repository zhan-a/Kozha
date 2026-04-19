"""Correction interpreter — point-and-fix semantics for rendered signs.

After the author sees the rendered avatar they may say things like
"the handshape at 2 seconds is wrong; it should be a flat-O, not a
fisted hand", "location is too low; move it up to the temple", or
"the movement should repeat twice, not once". This module translates
those free-form corrections into **minimal diffs** on the session's
:class:`PartialSignParameters` — never a full restart — and feeds the
result through the generator + validator + renderer chain so the user
sees an updated avatar with only the intended parts changed.

Public API
----------
- :class:`Correction` — the user's request shape (raw text + optional
  timeline scrubber timestamp + optional clicked body region).
- :class:`FieldChange` — one parameter edit with confidence.
- :class:`CorrectionIntent` — coarse classification of what the user
  wants (apply a diff, start over, elaborate, resolve a contradiction,
  or answer a follow-up).
- :class:`CorrectionPlan` — the interpreter's output: list of
  :class:`FieldChange`, plain-English explanation, whether confirmation
  is required, and — when an edit is being applied — the updated
  :class:`PartialSignParameters`.
- :func:`interpret_correction` — session + correction →
  :class:`CorrectionPlan`.
- :func:`apply_correction` — session + plan → new session via
  :func:`apply_correction_diff` + :func:`run_generation`; validator
  failures transition to :data:`SessionState.AWAITING_CORRECTION` with
  the error recorded on the draft.

Edge cases the interpreter handles explicitly
---------------------------------------------
- **Restart** ("the whole sign is wrong"): intent is ``RESTART``; the
  caller is expected to route the user back to the prose-description
  prompt.
- **Unencoded feature** ("facial expression should be happier" with no
  non-manuals populated): intent is ``ELABORATE`` with a follow-up
  question asking for specifics.
- **Contradiction with original prose**: intent is ``CONTRADICTION``
  with a follow-up question surfacing the two options.
- **Vague** ("it looks off"): intent is ``VAGUE`` with a follow-up
  question; no edits are guessed.

LLM call shape
--------------
- Model: ``gpt-4o`` (spec-pinned; overridable via ``model`` kwarg).
- Temperature: 0.1 — we want determinism for minimal-diff behavior.
- ``response_format``: strict JSON schema wrapping a
  :class:`CorrectionPlan`-ish object with an explicit ``intent`` enum.
- Budget / telemetry flow through :class:`LLMClient` as usual.

Fallback policy
---------------
If the LLM call raises (except :class:`BudgetExceeded`, which is a hard
stop), the interpreter returns a ``VAGUE`` plan asking the user to
rephrase — the authoring UI never hard-fails on a flaky upstream.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from generator import GenerateResult, generate
from llm import ChatResult, LLMClient, LLMError
from llm.budget import BudgetExceeded
from parser import (
    PartialMovementSegment,
    PartialNonManualFeatures,
    PartialSignParameters,
)
from prompts import PromptMetadata, load_prompt
from rendering.hamnosys_to_sigml import to_sigml
from rendering.preview import PreviewResult, PreviewStatus, render_preview
from session.orchestrator import (
    Correction,
    apply_correction_diff,
    run_generation,
)
from session.state import AuthoringSession, SessionState

from .metrics import log_correction_applied


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CorrectionIntent(str, Enum):
    """Coarse classification of what the user wants from the correction.

    ``APPLY_DIFF``
        Standard minimal-diff edit — ``field_changes`` is populated and
        ``updated_params`` is the new parameter bundle.
    ``RESTART``
        User asked to start over (e.g. "the whole sign is wrong"). The
        caller should route the user back to prose-description input.
    ``ELABORATE``
        Correction refers to a feature not currently encoded (common:
        facial expression on a sign with ``non_manual=None``). Ask a
        follow-up to collect specifics.
    ``CONTRADICTION``
        Correction contradicts the original prose. Surface both options
        and let the user pick which is authoritative.
    ``VAGUE``
        Correction is too ambiguous to guess at (e.g. "looks off"). Ask
        a targeted follow-up — **do not** fabricate edits.
    """

    APPLY_DIFF = "apply_diff"
    RESTART = "restart"
    ELABORATE = "elaborate"
    CONTRADICTION = "contradiction"
    VAGUE = "vague"


@dataclass(frozen=True)
class FieldChange:
    """One edit on a :class:`PartialSignParameters` slot.

    ``path`` is a dotted path into the parameters structure:
    scalar slots (``"handshape_dominant"``), movement subfields
    (``"movement[0].path"``, ``"movement[0].repeat"``), and non-manual
    sub-slots (``"non_manual.eyebrows"``). ``old_value`` / ``new_value``
    are the plain-English values before and after — the parser stage
    stays in plain English, the generator handles codepoint mapping.
    ``confidence`` is the model's self-reported confidence in the edit
    (0..1).
    """

    path: str
    old_value: Any
    new_value: Any
    confidence: float


@dataclass(frozen=True)
class CorrectionPlan:
    """Interpreter output — the proposed response to one correction.

    Only ``APPLY_DIFF`` plans carry ``updated_params``. Non-apply
    intents leave ``field_changes`` empty and set
    ``needs_user_confirmation`` + ``follow_up_question`` so the caller
    can surface the follow-up to the user.
    """

    intent: CorrectionIntent
    field_changes: list[FieldChange]
    explanation: str
    needs_user_confirmation: bool
    follow_up_question: Optional[str] = None
    updated_params: Optional[PartialSignParameters] = None


# Dotted-path regex for movement segments — e.g. "movement[0].path".
_MOVEMENT_PATH_RE = re.compile(r"^movement\[(\d+)\]\.(path|size_mod|speed_mod|repeat)$")

# Scalar slots we allow a correction to target.
_SCALAR_SLOTS: frozenset[str] = frozenset(
    {
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    }
)

_NON_MANUAL_LEAVES: frozenset[str] = frozenset(
    {"mouth_picture", "eye_gaze", "head_movement", "eyebrows", "facial_expression"}
)


# Deterministic restart-phrase detector. The LLM path also returns
# ``RESTART`` in these cases, but matching before the call saves a
# round-trip and keeps the tests deterministic.
_RESTART_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(start\s+over|start\s+again|from\s+scratch)\b", re.IGNORECASE),
    re.compile(r"\b(whole|entire|everything)\b.*\b(wrong|bad|off|incorrect)\b", re.IGNORECASE),
    re.compile(r"\b(scrap|discard|throw\s+out)\b", re.IGNORECASE),
    re.compile(r"^\s*(redo|restart|reset)\s*[.!?]*\s*$", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


PROMPT_ID: str = "interpret_correction"


def _prompt_context() -> dict[str, Any]:
    """Assemble the Jinja2 render context for the interpreter prompt."""
    allowed_scalars = ", ".join(sorted(_SCALAR_SLOTS))
    allowed_nm = ", ".join(
        f"non_manual.{leaf}" for leaf in sorted(_NON_MANUAL_LEAVES)
    )
    intents = ", ".join(f'"{i.value}"' for i in CorrectionIntent)
    return {
        "allowed_scalars": allowed_scalars,
        "allowed_nm": allowed_nm,
        "intents": intents,
    }


def _render_system_prompt() -> tuple[str, PromptMetadata]:
    """Render the interpreter system prompt with its :class:`PromptMetadata`."""
    pt = load_prompt(PROMPT_ID)
    return pt.render(**_prompt_context()), pt.metadata


SYSTEM_PROMPT: str
_PROMPT_METADATA: PromptMetadata
SYSTEM_PROMPT, _PROMPT_METADATA = _render_system_prompt()


# ---------------------------------------------------------------------------
# Strict-JSON response schema
# ---------------------------------------------------------------------------


def _build_response_schema() -> dict[str, Any]:
    """Strict JSON-schema for the interpreter response."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "correction_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "intent",
                    "explanation",
                    "needs_user_confirmation",
                    "follow_up_question",
                    "field_changes",
                ],
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [i.value for i in CorrectionIntent],
                    },
                    "explanation": {"type": "string"},
                    "needs_user_confirmation": {"type": "boolean"},
                    "follow_up_question": {"type": ["string", "null"]},
                    "field_changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "path",
                                "old_value",
                                "new_value",
                                "confidence",
                            ],
                            "properties": {
                                "path": {"type": "string"},
                                "old_value": {"type": ["string", "null"]},
                                "new_value": {"type": ["string", "null"]},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


RESPONSE_SCHEMA: dict[str, Any] = _build_response_schema()


# ---------------------------------------------------------------------------
# Public API — interpret_correction
# ---------------------------------------------------------------------------


class CorrectionApplyError(ValueError):
    """Raised when a :class:`CorrectionPlan` cannot be written onto params.

    Signals that the LLM returned a path / value the slot writer cannot
    accept. The interpreter catches this and downgrades the plan to
    ``VAGUE`` with a follow-up so callers do not see a raw exception.
    """


def interpret_correction(
    session: AuthoringSession,
    correction: Correction,
    *,
    client: LLMClient | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 1000,
    request_id: str | None = None,
) -> CorrectionPlan:
    """Interpret ``correction`` against ``session`` and return a plan.

    Never raises for LLM-side failures — returns a ``VAGUE`` plan with a
    follow-up question instead, so the authoring UI can always surface
    something to the user. :class:`BudgetExceeded` propagates.
    """
    if not isinstance(correction, Correction):
        raise TypeError("correction must be a session.orchestrator.Correction")
    raw = (correction.raw_text or "").strip()
    if not raw:
        return _vague_plan(
            "Empty correction — nothing to interpret.",
            "What specifically would you like to change?",
        )

    # Fast path: obvious restart phrasings skip the LLM call entirely.
    if _looks_like_restart(raw):
        return CorrectionPlan(
            intent=CorrectionIntent.RESTART,
            field_changes=[],
            explanation=(
                "You asked to start over. The current sign will be discarded "
                "and you can re-describe it from scratch."
            ),
            needs_user_confirmation=True,
            follow_up_question=None,
            updated_params=None,
        )

    params = session.draft.parameters_partial
    if params is None:
        return _vague_plan(
            "No parameters are populated yet; run parse → generate first.",
            "Could you re-describe the sign so I can start from scratch?",
        )

    req_id = request_id or f"correct-{uuid.uuid4().hex[:12]}"

    # If the caller did not inject a client, try to build a default one.
    try:
        llm = client if client is not None else LLMClient(model=model)
    except LLMError:
        return _vague_plan(
            "Correction interpreter is unavailable (no LLM client).",
            "Could you restate the correction as an explicit slot change, "
            "e.g. 'palm should face up' or 'move the location to the chin'?",
        )

    user_payload = {
        "parameters": params.model_dump(mode="json"),
        "hamnosys": session.draft.hamnosys or "",
        "original_prose": session.draft.description_prose or "",
        "correction": {
            "raw_text": raw,
            "target_time_ms": correction.target_time_ms,
            "target_region": correction.target_region,
        },
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    try:
        chat_result: ChatResult = llm.chat(
            messages=messages,
            response_format=RESPONSE_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=req_id,
            prompt_metadata=_PROMPT_METADATA,
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning("correction interpreter LLM call failed: %s", exc)
        return _vague_plan(
            "Could not reach the correction interpreter.",
            "Could you restate the correction as a specific slot change?",
        )

    return _plan_from_llm(chat_result.content, params)


# ---------------------------------------------------------------------------
# Public API — apply_correction
# ---------------------------------------------------------------------------


@dataclass
class ApplyOutcome:
    """Bookkeeping returned alongside the new session by :func:`apply_correction`.

    ``outcome`` mirrors the metric label written by
    :func:`log_correction_applied`. ``regeneration`` is the
    :class:`GenerateResult` from the regeneration pass, or ``None`` for
    non-``APPLY_DIFF`` intents.
    """

    session: AuthoringSession
    plan: CorrectionPlan
    outcome: str
    regeneration: Optional[GenerateResult] = None


def apply_correction(
    session: AuthoringSession,
    plan: CorrectionPlan,
    *,
    generate_fn: Callable[..., GenerateResult] = generate,
    to_sigml_fn: Callable[..., str] = to_sigml,
    render_fn: Optional[Callable[..., PreviewResult]] = render_preview,
    log_dir: Any = None,
) -> ApplyOutcome:
    """Apply a :class:`CorrectionPlan` to ``session``.

    Expects ``session.state == APPLYING_CORRECTION`` (set by
    :func:`session.orchestrator.on_correction`). Behavior by intent:

    - ``APPLY_DIFF`` — call :func:`apply_correction_diff` with the plan
      and then :func:`run_generation`. On validator failure the session
      is parked in :data:`SessionState.AWAITING_CORRECTION` with
      ``generation_errors`` on the draft so the UI can show the error
      and let the user re-correct.
    - ``RESTART`` — clear draft parameter state and return the session
      in :data:`SessionState.AWAITING_DESCRIPTION`. The caller is
      expected to prompt the user for a fresh description and call
      :func:`session.orchestrator.on_description`.
    - ``ELABORATE`` / ``CONTRADICTION`` / ``VAGUE`` — record the
      :class:`CorrectionAppliedEvent` with an empty diff, transition to
      :data:`SessionState.AWAITING_CORRECTION`, and return the plan's
      ``follow_up_question`` on the outcome for the UI to surface.

    Always writes one ``correction_applied`` metrics record regardless
    of intent.
    """
    if session.state is not SessionState.APPLYING_CORRECTION:
        raise RuntimeError(
            f"apply_correction requires state APPLYING_CORRECTION, got "
            f"{session.state.value!r}"
        )
    session_id = session.id
    corrections_count = session.draft.corrections_count

    if plan.intent is CorrectionIntent.APPLY_DIFF:
        outcome = _apply_diff_plan(
            session,
            plan,
            generate_fn=generate_fn,
            to_sigml_fn=to_sigml_fn,
            render_fn=render_fn,
        )
    elif plan.intent is CorrectionIntent.RESTART:
        outcome = _apply_restart_plan(session, plan)
    else:
        outcome = _apply_nondiff_plan(session, plan)

    log_correction_applied(
        session_id=session_id,
        intent=plan.intent.value,
        field_count=len(plan.field_changes),
        outcome=outcome.outcome,
        corrections_count=corrections_count,
        log_dir=log_dir,
    )
    return outcome


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------


def _apply_diff_plan(
    session: AuthoringSession,
    plan: CorrectionPlan,
    *,
    generate_fn: Callable[..., GenerateResult],
    to_sigml_fn: Callable[..., str],
    render_fn: Optional[Callable[..., PreviewResult]],
) -> ApplyOutcome:
    """APPLY_DIFF path — rewrite params, regenerate, classify outcome."""
    if plan.updated_params is None:
        # Interpreter emitted APPLY_DIFF with no params bundle; safest
        # fallback is a no-op transition to AWAITING_CORRECTION.
        session = apply_correction_diff(
            session,
            field_changes=_plain_changes(plan.field_changes),
            summary=plan.explanation,
            updated_params=None,
        ).with_state(SessionState.AWAITING_CORRECTION)
        return ApplyOutcome(
            session=session, plan=plan, outcome="noop", regeneration=None
        )

    session = apply_correction_diff(
        session,
        field_changes=_plain_changes(plan.field_changes),
        summary=plan.explanation,
        updated_params=plan.updated_params,
    )
    session = run_generation(
        session,
        generate_fn=generate_fn,
        to_sigml_fn=to_sigml_fn,
        render_fn=render_fn,
    )
    gen_events = [e for e in session.history if e.type == "generated"]
    last_gen = gen_events[-1] if gen_events else None
    regeneration: Optional[GenerateResult] = None

    if session.state is SessionState.RENDERED:
        return ApplyOutcome(
            session=session, plan=plan, outcome="ok", regeneration=regeneration
        )

    # run_generation failed — park in AWAITING_CORRECTION so the UI can
    # surface the validator error and let the user re-correct.
    session = session.with_state(SessionState.AWAITING_CORRECTION)
    return ApplyOutcome(
        session=session,
        plan=plan,
        outcome="validation_failed",
        regeneration=None,
    )


def _apply_restart_plan(
    session: AuthoringSession,
    plan: CorrectionPlan,
) -> ApplyOutcome:
    """RESTART path — log the CorrectionAppliedEvent and reset the draft."""
    session = apply_correction_diff(
        session,
        field_changes=_plain_changes(plan.field_changes),
        summary=plan.explanation,
        updated_params=None,
    )
    # apply_correction_diff sets state to GENERATING; override to
    # AWAITING_DESCRIPTION so the caller re-prompts the user.
    session = session.with_draft(
        description_prose="",
        parameters_partial=None,
        gaps=[],
        pending_questions=[],
        hamnosys=None,
        sigml=None,
        video_path=None,
        preview_status=None,
        preview_message="",
        generation_confidence=None,
        generation_errors=[],
        slot_codepoints={},
    )
    session = session.with_state(SessionState.AWAITING_DESCRIPTION)
    return ApplyOutcome(
        session=session, plan=plan, outcome="noop", regeneration=None
    )


def _apply_nondiff_plan(
    session: AuthoringSession,
    plan: CorrectionPlan,
) -> ApplyOutcome:
    """ELABORATE / CONTRADICTION / VAGUE path — park in AWAITING_CORRECTION."""
    session = apply_correction_diff(
        session,
        field_changes=_plain_changes(plan.field_changes),
        summary=plan.explanation,
        updated_params=None,
    )
    session = session.with_state(SessionState.AWAITING_CORRECTION)
    return ApplyOutcome(
        session=session, plan=plan, outcome="noop", regeneration=None
    )


# ---------------------------------------------------------------------------
# LLM response → CorrectionPlan
# ---------------------------------------------------------------------------


def _plan_from_llm(
    raw_content: str, params: PartialSignParameters
) -> CorrectionPlan:
    """Parse the LLM's raw JSON text into a :class:`CorrectionPlan`.

    Catches all non-budget failures (malformed JSON, unknown intent,
    bad path) and downgrades to ``VAGUE`` with a follow-up — the UI
    never sees an exception from a flaky model.
    """
    stripped = (raw_content or "").strip()
    if not stripped:
        return _vague_plan(
            "Interpreter returned no content.",
            "Could you restate the correction as an explicit slot change?",
        )
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.warning("interpreter returned invalid JSON: %s", exc)
        return _vague_plan(
            "Interpreter response was not valid JSON.",
            "Could you restate the correction as an explicit slot change?",
        )

    if not isinstance(payload, dict):
        return _vague_plan(
            "Interpreter response was not a JSON object.",
            "Could you restate the correction as an explicit slot change?",
        )
    try:
        intent_raw = str(payload.get("intent") or "").strip()
        intent = CorrectionIntent(intent_raw)
    except ValueError:
        return _vague_plan(
            f"Interpreter returned unknown intent {payload.get('intent')!r}.",
            "Could you restate the correction as an explicit slot change?",
        )

    explanation = str(payload.get("explanation") or "").strip() or (
        "No explanation provided."
    )
    needs_conf = bool(payload.get("needs_user_confirmation", False))
    follow_up_raw = payload.get("follow_up_question")
    follow_up = str(follow_up_raw).strip() if follow_up_raw else None

    raw_changes = payload.get("field_changes") or []
    if not isinstance(raw_changes, list):
        raw_changes = []
    changes: list[FieldChange] = []
    for rc in raw_changes:
        if not isinstance(rc, dict):
            continue
        path = str(rc.get("path") or "").strip()
        if not path:
            continue
        try:
            confidence = float(rc.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        changes.append(
            FieldChange(
                path=path,
                old_value=rc.get("old_value"),
                new_value=rc.get("new_value"),
                confidence=max(0.0, min(1.0, confidence)),
            )
        )

    if intent is CorrectionIntent.APPLY_DIFF:
        if not changes:
            return _vague_plan(
                "Interpreter returned apply_diff but no field changes.",
                "Which slot should I change — handshape, location, orientation, "
                "movement, or a non-manual feature?",
            )
        try:
            updated = _apply_field_changes(params, changes)
        except CorrectionApplyError as exc:
            logger.warning("failed to apply interpreter field changes: %s", exc)
            return _vague_plan(
                f"Could not apply the proposed diff: {exc}",
                "Could you restate the correction as an explicit slot change?",
            )
        return CorrectionPlan(
            intent=CorrectionIntent.APPLY_DIFF,
            field_changes=changes,
            explanation=explanation,
            needs_user_confirmation=needs_conf,
            follow_up_question=follow_up,
            updated_params=updated,
        )

    # Non-diff intents — keep the follow-up question if one was provided
    # and require confirmation unconditionally.
    if intent in {
        CorrectionIntent.ELABORATE,
        CorrectionIntent.CONTRADICTION,
        CorrectionIntent.VAGUE,
    } and not follow_up:
        follow_up = (
            "Could you give a bit more detail on what you'd like to change?"
        )
    return CorrectionPlan(
        intent=intent,
        field_changes=[],
        explanation=explanation,
        needs_user_confirmation=True,
        follow_up_question=follow_up,
        updated_params=None,
    )


# ---------------------------------------------------------------------------
# FieldChange → PartialSignParameters writer
# ---------------------------------------------------------------------------


def _apply_field_changes(
    params: PartialSignParameters,
    changes: list[FieldChange],
) -> PartialSignParameters:
    """Return a new ``PartialSignParameters`` with all edits applied.

    Raises :class:`CorrectionApplyError` on any unsupported path or
    out-of-range movement index.
    """
    if not changes:
        return params
    working = params.model_copy(deep=True)
    for change in changes:
        working = _apply_single_change(working, change)
    return working


def _apply_single_change(
    params: PartialSignParameters,
    change: FieldChange,
) -> PartialSignParameters:
    path = change.path
    new_value = change.new_value
    if isinstance(new_value, str):
        new_value = new_value.strip()
        if not new_value:
            new_value = None

    if path in _SCALAR_SLOTS:
        return params.model_copy(update={path: new_value})

    match = _MOVEMENT_PATH_RE.match(path)
    if match:
        idx = int(match.group(1))
        sub = match.group(2)
        segments = list(params.movement)
        while idx >= len(segments):
            segments.append(PartialMovementSegment())
        seg = segments[idx].model_copy(update={sub: new_value})
        segments[idx] = seg
        # If the segment is fully empty, trim trailing empties to keep
        # the movement list tight — but never drop a segment that is
        # still referenced by another FieldChange.
        while segments and _is_empty_segment(segments[-1]):
            segments.pop()
        return params.model_copy(update={"movement": segments})

    if path.startswith("non_manual."):
        leaf = path.split(".", 1)[1]
        if leaf not in _NON_MANUAL_LEAVES:
            raise CorrectionApplyError(
                f"unknown non_manual sub-field {leaf!r} in path {path!r}"
            )
        nm = params.non_manual or PartialNonManualFeatures()
        nm = nm.model_copy(update={leaf: new_value})
        # If every non-manual slot ended up empty, collapse back to None.
        if not any(
            getattr(nm, k) for k in _NON_MANUAL_LEAVES
        ):
            return params.model_copy(update={"non_manual": None})
        return params.model_copy(update={"non_manual": nm})

    raise CorrectionApplyError(f"unsupported correction path {path!r}")


def _is_empty_segment(seg: PartialMovementSegment) -> bool:
    return not any(
        getattr(seg, k) for k in ("path", "size_mod", "speed_mod", "repeat")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plain_changes(changes: list[FieldChange]) -> list[dict]:
    """Flatten :class:`FieldChange` instances for the event log."""
    return [
        {
            "path": c.path,
            "old_value": c.old_value,
            "new_value": c.new_value,
            "confidence": c.confidence,
        }
        for c in changes
    ]


def _vague_plan(explanation: str, follow_up: str) -> CorrectionPlan:
    return CorrectionPlan(
        intent=CorrectionIntent.VAGUE,
        field_changes=[],
        explanation=explanation,
        needs_user_confirmation=True,
        follow_up_question=follow_up,
        updated_params=None,
    )


def _looks_like_restart(text: str) -> bool:
    for pat in _RESTART_PATTERNS:
        if pat.search(text):
            return True
    return False


__all__ = [
    "ApplyOutcome",
    "Correction",
    "CorrectionApplyError",
    "CorrectionIntent",
    "CorrectionPlan",
    "FieldChange",
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
    "apply_correction",
    "interpret_correction",
]
