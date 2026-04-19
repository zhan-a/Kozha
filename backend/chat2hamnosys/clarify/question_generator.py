"""Clarification-question generator.

Given a :class:`ParseResult` with gaps and the authoring dialogue so
far, produce up to three :class:`Question` records for the next turn.
Tries an LLM rewrite first (grounded in the original prose) and falls
back to deterministic :mod:`templates` on any failure — the authoring
UI never blocks on an LLM issue.

Two pieces the caller will touch:

- :func:`generate_questions` — one call, returns a list of
  :class:`Question`.
- :data:`SYSTEM_PROMPT` — exposed so tests and the eval doc can inspect
  the exact text sent to the LLM.

LLM call shape
--------------
- Model: ``gpt-4o`` (spec-pinned; overridable via the ``model`` kwarg).
- Temperature: 0.3 — slightly higher than the parser because we want
  phrasing variety, not determinism.
- ``response_format``: ``{"type": "json_schema", "strict": True, ...}``
  wrapping a list of :class:`Question` records.
- Routed through :class:`LLMClient` so retries, budget-guarding, and
  JSONL telemetry all apply.

Fallback policy
---------------
If any of the following happens, :func:`generate_questions` returns
template questions directly (never raises to the caller):

- No API key available when constructing a default :class:`LLMClient`.
- The LLM call raises any exception *except* :class:`BudgetExceeded`
  (which is a hard-stop configuration error the caller must handle).
- The LLM returns malformed JSON, missing keys, or no valid questions.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from llm import ChatResult, LLMClient, LLMError
from llm.budget import BudgetExceeded

from models import ClarificationTurn
from parser import ALLOWED_GAP_FIELDS, Gap, ParseResult

from .models import Question
from .templates import TEMPLATES, template_for


# Mobile UX cap. Don't raise — a turn with four questions has
# measurably worse completion rates than three.
MAX_QUESTIONS_PER_TURN: int = 3


# Per-field keyword lists used by :func:`_fields_already_asked` to detect
# when a prior assistant turn has already covered a given slot. Keywords
# are deliberately *specific phrases*, not bare slot nouns — the
# handshape template mentions "a flat palm" as an example, so using
# "palm" alone would incorrectly mark ``orientation_palm`` as already
# asked. Phrases like "palm direction" / "palm face" only appear when
# the assistant actually asked about the orientation slot.
_FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "handshape_dominant": ("which handshape", "what handshape"),
    "handshape_nondominant": (
        "non-dominant",
        "nondominant",
        "helper hand",
        "second hand",
        "other hand",
    ),
    "orientation_extended_finger": (
        "fingertips point",
        "fingers point",
        "extended-finger",
        "extended finger direction",
        "finger direction",
        "which direction do the fingertips",
        "which direction do the fingers",
    ),
    "orientation_palm": (
        "palm direction",
        "palm face",
        "palm faces",
        "palm facing",
        "which way does the palm",
    ),
    "location": ("where on the body", "where is the sign"),
    "contact": (
        "does the hand touch",
        "does the hand contact",
        "make contact",
        "brief tap",
        "brush or graze",
    ),
    "movement": ("how does the hand move", "movement shape"),
    "non_manual.mouth_picture": (
        "mouth shape",
        "mouthing",
        "particular mouth",
    ),
    "non_manual.eye_gaze": (
        "gaze directed",
        "gaze target",
        "signer's gaze",
    ),
    "non_manual.head_movement": ("head moving", "head movement"),
    "non_manual.eyebrows": ("eyebrows doing", "eyebrow", "brow —", "brow?"),
    "non_manual.facial_expression": (
        "facial expression",
        "overall expression",
    ),
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _render_templates_block() -> str:
    lines = []
    for field, t in TEMPLATES.items():
        if t.options:
            opts = ", ".join(f'"{o.label}"→"{o.value}"' for o in t.options)
        else:
            opts = "(freeform)"
        lines.append(
            f"- **{field}**\n"
            f"    Plain: {t.text}\n"
            f"    Deaf-native: {t.deaf_native_text}\n"
            f"    Options: {opts}"
        )
    return "\n".join(lines)


def _build_system_prompt() -> str:
    """Compose the system prompt used on every generator call.

    Kept as a single stable string so recorded test fixtures remain
    valid across minor refactors — invalidating the prompt invalidates
    every recording.
    """
    templates_block = _render_templates_block()
    allowed_fields_list = "\n- ".join(f"`{f}`" for f in ALLOWED_GAP_FIELDS)

    return f"""You are a sign-language authoring assistant. A previous parser step read a natural-language description of a sign and flagged slots it could not fill (``gaps``). Your job is to ask the author focused follow-up questions so the structured phonological parameters can be completed.

## Input

A JSON object with these keys:

- ``prose`` — the author's original description, verbatim (may be empty if not yet provided).
- ``gaps`` — a list of ``{{field, reason, suggested_question}}`` records the parser flagged. Ask one question per listed gap; ignore fields not in this list.
- ``is_deaf_native`` — ``true`` / ``false`` / ``null``. When ``true``, adjust phrasing: less hand-holding, more direct phonological terminology (e.g. "palm direction?" instead of "which way does the palm face?").
- ``template_hints`` — optional starting-point phrasings you may adapt.

## Output

Return a JSON object ``{{"questions": [...]}}`` where each question has:

- ``field`` — the slot name, drawn from this exact list:
- {allowed_fields_list}
- ``text`` — the actual question to ask, plain English, 1–2 sentences.
- ``options`` — a list of 2–5 ``{{label, value}}`` objects when the slot has a small closed vocabulary; ``null`` when genuinely open-ended. ``label`` is shown to the user; ``value`` is the canonical phrase the downstream code will record.
- ``allow_freeform`` — ``true`` when the user may type a freeform answer outside the options; ``false`` only when the options truly exhaust the space (rare).
- ``rationale`` — one clause for the debug UI explaining why this question was chosen; **not** shown to the user.

## Hard rules

1. Ask **at most three** questions per turn — mobile UX degrades past that. If more than three gaps are given, pick the three highest-impact ones (handshape > location > orientation > movement > contact > non-manual).
2. One question per distinct ``field``. Never ask two questions about the same slot.
3. **Prefer multiple-choice.** Offer 2–5 concrete options whenever the slot has a small closed vocabulary (handshape, palm direction, eyebrows, movement shape). Use pure freeform only for slots with unbounded vocabularies (unusual locations, specific mouthings).
4. **Ground in the prose.** If the author mentioned "near the temple", say "You mentioned the temple — does the hand touch it or hover nearby?" instead of a generic "Where is the sign made?". If ``prose`` is empty, use the generic template phrasing.
5. Only emit ``field`` values from the allowed list above. Silently drop gap records whose ``field`` is not one of these.
6. Plain English by default. A signer with no phonology training should understand every question. When ``is_deaf_native=true``, you may use direct terminology; when ``false`` or ``null``, prefer the longer hand-holding form.
7. ``allow_freeform`` defaults to ``true``. The small-vocabulary slots (palm direction, eyebrows) may set it ``false``; everything else should allow a freeform escape hatch.

## Template library (starting points — you may rewrite)

{templates_block}
"""


SYSTEM_PROMPT: str = _build_system_prompt()


# ---------------------------------------------------------------------------
# Strict-JSON response schema
# ---------------------------------------------------------------------------


_UNSUPPORTED_KEYWORDS = {
    "default",
    "title",
    "format",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "examples",
    "example",
    "readOnly",
    "writeOnly",
    "const",
}


def _strictify(node: Any) -> Any:
    """Massage a Pydantic-emitted schema into OpenAI strict form.

    Mirrors ``parser.description_parser._strictify`` — kept local rather
    than imported so the clarifier module has no lateral coupling.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _UNSUPPORTED_KEYWORDS:
                continue
            out[k] = _strictify(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    return node


def build_generator_response_schema() -> dict[str, Any]:
    """Return the top-level schema sent as ``response_format.json_schema``.

    Wraps a list of :class:`Question` records in a single object
    ``{questions: [...]}`` so the LLM's response is self-contained.
    """
    question_schema = _strictify(Question.model_json_schema())
    defs = question_schema.pop("$defs", {})
    wrapper: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["questions"],
        "properties": {
            "questions": {"type": "array", "items": question_schema},
        },
    }
    if defs:
        wrapper["$defs"] = defs
    return wrapper


GENERATOR_RESPONSE_SCHEMA: dict[str, Any] = build_generator_response_schema()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_questions(
    parse_result: ParseResult,
    prior_turns: list[ClarificationTurn] | None = None,
    *,
    is_deaf_native: bool | None = None,
    client: LLMClient | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    request_id: str | None = None,
) -> list[Question]:
    """Produce up to :data:`MAX_QUESTIONS_PER_TURN` clarification questions.

    Flow:

    1. Filter ``parse_result.gaps`` against ``prior_turns`` and cap at
       :data:`MAX_QUESTIONS_PER_TURN`. Empty target list → return ``[]``.
    2. Build the LLM payload (system prompt + per-gap target list +
       grounding prose taken from the most recent author turn).
    3. Call :class:`LLMClient`. On success, validate the response into
       :class:`Question` records; drop any record whose ``field`` is not
       in the target list; de-duplicate same-field questions.
    4. On any LLM/validation failure, return :func:`templates.template_for`
       for each target gap so the UI still has something to show.

    :class:`BudgetExceeded` is *not* caught — a budget breach is a
    configuration failure, not a soft failure, and must surface to the
    caller.
    """
    prior_turns = prior_turns or []
    targets = _gap_targets(parse_result, prior_turns)
    if not targets:
        return []

    req_id = request_id or f"clarify-{uuid.uuid4().hex[:12]}"

    try:
        llm = client if client is not None else LLMClient(model=model)
    except LLMError:
        return _fallback_questions(targets, is_deaf_native=bool(is_deaf_native))

    user_payload = _build_user_payload(
        parse_result, targets, prior_turns, is_deaf_native
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "clarification_questions",
            "strict": True,
            "schema": GENERATOR_RESPONSE_SCHEMA,
        },
    }

    try:
        chat_result: ChatResult = llm.chat(
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=req_id,
        )
    except BudgetExceeded:
        raise
    except Exception:
        return _fallback_questions(targets, is_deaf_native=bool(is_deaf_native))

    try:
        return _parse_llm_questions(chat_result.content, targets)
    except Exception:
        return _fallback_questions(targets, is_deaf_native=bool(is_deaf_native))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gap_targets(
    parse_result: ParseResult,
    prior_turns: list[ClarificationTurn],
) -> list[Gap]:
    """Filter gaps by prior-turns coverage and cap at the mobile-UX ceiling."""
    already_asked = _fields_already_asked(prior_turns)
    targets: list[Gap] = []
    seen_fields: set[str] = set()
    for gap in parse_result.gaps:
        if gap.field in already_asked:
            continue
        if gap.field in seen_fields:
            continue
        seen_fields.add(gap.field)
        targets.append(gap)
        if len(targets) >= MAX_QUESTIONS_PER_TURN:
            break
    return targets


def _fields_already_asked(prior_turns: list[ClarificationTurn]) -> set[str]:
    """Loose substring scan of prior assistant turns.

    A keyword list per field catches the common phrasings produced by
    either the LLM path or the template fallback. False positives are
    preferable to false negatives — asking the same question twice is a
    worse UX failure than silently skipping a marginal re-ask.
    """
    asked: set[str] = set()
    for turn in prior_turns:
        if turn.role != "assistant":
            continue
        text = turn.text.lower()
        # "handshape" appears in both handshape_dominant and non-dominant
        # phrasings; resolve by checking for the non-dominant markers first.
        nondom_hit = any(
            kw in text for kw in _FIELD_KEYWORDS["handshape_nondominant"]
        )
        for field, keywords in _FIELD_KEYWORDS.items():
            if field == "handshape_dominant":
                if "handshape" in text and not nondom_hit:
                    asked.add(field)
                continue
            if any(kw in text for kw in keywords):
                asked.add(field)
    return asked


def _build_user_payload(
    parse_result: ParseResult,
    targets: list[Gap],
    prior_turns: list[ClarificationTurn],
    is_deaf_native: bool | None,
) -> dict[str, Any]:
    """Assemble the JSON payload the LLM consumes as the user message.

    ``prose`` is drawn from the most recent author turn — ParseResult
    itself only carries ``raw_response`` (the LLM's JSON reply), not the
    original description. When no author turn is present the field is
    left empty and the LLM falls back to generic template phrasing.
    """
    prose = ""
    for turn in reversed(prior_turns):
        if turn.role == "author":
            prose = turn.text
            break

    return {
        "prose": prose,
        "gaps": [
            {
                "field": g.field,
                "reason": g.reason,
                "suggested_question": g.suggested_question,
            }
            for g in targets
        ],
        "is_deaf_native": is_deaf_native,
        "template_hints": [
            {
                "field": g.field,
                "default_text": template_for(
                    g.field, is_deaf_native=bool(is_deaf_native)
                ).text,
            }
            for g in targets
        ],
    }


def _parse_llm_questions(raw: str, targets: list[Gap]) -> list[Question]:
    """Validate the LLM's raw text into Question records or raise.

    Caller catches any exception and falls back to templates — so this
    function is deliberately strict: unparseable JSON, missing keys,
    schema violations, or zero valid questions all propagate.
    """
    stripped = (raw or "").strip()
    if not stripped:
        raise ValueError("empty LLM response")

    payload = json.loads(stripped)
    if not isinstance(payload, dict) or "questions" not in payload:
        raise ValueError("LLM response missing 'questions' key")

    raw_qs = payload["questions"]
    if not isinstance(raw_qs, list):
        raise ValueError("'questions' must be a list")

    target_fields = {g.field for g in targets}
    out: list[Question] = []
    seen: set[str] = set()
    for rq in raw_qs:
        q = Question.model_validate(rq)
        if q.field not in target_fields:
            # LLM invented a slot we didn't ask about — drop it.
            continue
        if q.field in seen:
            continue
        seen.add(q.field)
        out.append(q)
        if len(out) >= MAX_QUESTIONS_PER_TURN:
            break

    if not out:
        raise ValueError("LLM returned no valid questions matching target gaps")
    return out


def _fallback_questions(
    gaps: list[Gap],
    *,
    is_deaf_native: bool,
) -> list[Question]:
    return [
        template_for(g.field, is_deaf_native=is_deaf_native, reason=g.reason)
        for g in gaps[:MAX_QUESTIONS_PER_TURN]
    ]


__all__ = [
    "GENERATOR_RESPONSE_SCHEMA",
    "MAX_QUESTIONS_PER_TURN",
    "SYSTEM_PROMPT",
    "build_generator_response_schema",
    "generate_questions",
]
