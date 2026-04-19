"""Prose → phonological-features parser.

Converts an author's natural-language description (e.g. *"signed near the
temple, bent-5 handshape, small circular motion, eyebrows raised"*) into a
populated-but-partial :class:`PartialSignParameters` object plus a list of
:class:`Gap` records flagging slots the LLM could not fill with confidence.

Two pieces the caller will actually touch:

- :func:`parse_description` — one call, returns a :class:`ParseResult`.
- :data:`SYSTEM_PROMPT` — exposed so tests and the eval doc can inspect
  the exact text sent to the LLM.

The parser **does not** emit HamNoSys PUA codepoints — it intentionally
stays in plain-English phonological vocabulary and lets a later mapping
step translate strings like ``"bent-5"`` into ``hamfinger23456+hamthumboutmod``
or equivalent. Keeping the two concerns separate means the clarifier layer
can run gap-detection against structured data that is still legible to a
human reviewer.

LLM call shape
--------------
- Model: ``gpt-4o`` (spec-pinned; overridable via the ``model`` kwarg).
- Temperature: 0.1 (spec-pinned).
- ``response_format``: ``{"type": "json_schema", "strict": True, ...}``;
  the schema is derived from :class:`ParseResult` via
  :func:`build_parser_response_schema`.
- Routed through :class:`LLMClient`, so retries, budget guarding and the
  JSONL telemetry log from Prompt 4 all apply.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import ValidationError

from llm import ChatResult, LLMClient

from .models import (
    ALLOWED_GAP_FIELDS,
    Gap,
    ParseResult,
    PartialSignParameters,
)


# ---------------------------------------------------------------------------
# Reference vocabulary (grounds the system prompt)
# ---------------------------------------------------------------------------


# Kept as plain dicts so we can render them into the system prompt as a
# tight table. Values are short comments that remind the LLM what each term
# means — *not* an exhaustive enumeration.
HANDSHAPE_VOCAB: dict[str, str] = {
    "fist": "closed fist, all fingers curled",
    "flat": "flat hand, all fingers extended and together (B-hand)",
    "5": "spread hand, all five fingers extended and apart",
    "bent-5": "spread hand with all fingers bent at knuckles (claw / bent-5 / cee-5)",
    "flat-O": "fingertips touching thumb tip, hand flat (O with extended base)",
    "baby-O": "thumb-tip touching index-tip only, other fingers curled",
    "index": "index finger extended, others in fist (G / 1)",
    "V": "index and middle extended in V shape",
    "bent-V": "V shape with both fingers bent at knuckles",
    "ILY": "pinky, index and thumb extended (I-love-you)",
    "A": "closed fist with thumb alongside (A handshape)",
    "B": "flat hand with thumb tucked across palm",
    "C": "curved hand forming a C shape",
    "O": "all fingers curved to touch thumb, forming an O",
    "S": "closed fist with thumb across fingers",
    "Y": "thumb and pinky extended, others curled",
    "F": "thumb-tip and index-tip touching, other three fingers extended",
    "L": "thumb and index extended at right angles",
    "3": "thumb, index, middle extended",
    "4": "four fingers extended (no thumb)",
    "W": "thumb, pinky curled; index, middle, ring extended and spread",
    "claw": "synonym for bent-5",
}

LOCATION_VOCAB: dict[str, str] = {
    "forehead": "mid-forehead, above the eyebrows",
    "temple": "side of the head beside the eye",
    "eye": "at / near the eye",
    "nose": "at / near the nose",
    "cheek": "cheekbone area",
    "mouth / lips": "at the mouth",
    "chin": "at the chin / under the lower lip",
    "ear": "at / near the ear",
    "neck": "front of the neck / throat",
    "chest": "mid-chest / sternum",
    "ipsilateral shoulder": "shoulder on the signing-hand side",
    "contralateral shoulder": "shoulder on the non-dominant side",
    "upper arm": "on the non-dominant upper arm",
    "forearm": "on the non-dominant forearm",
    "back of hand": "dorsum of the non-dominant hand",
    "palm": "palm of the non-dominant hand",
    "neutral space": "in front of the body, no body contact",
    "belly / waist": "lower torso",
}

ORIENTATION_EXT_FINGER_VOCAB: dict[str, str] = {
    "up": "fingertips point upward",
    "down": "fingertips point downward",
    "forward": "fingertips point away from signer",
    "toward signer": "fingertips point toward the signer's body",
    "ipsilateral": "fingertips point to the signing-hand side",
    "contralateral": "fingertips point across the body",
    "up-forward": "diagonal upward-forward",
    "down-forward": "diagonal downward-forward",
}

ORIENTATION_PALM_VOCAB: dict[str, str] = {
    "down": "palm faces the ground",
    "up": "palm faces the ceiling",
    "toward signer": "palm faces the signer's body",
    "away from signer": "palm faces away from the signer",
    "ipsilateral": "palm faces the signing-hand side",
    "contralateral": "palm faces across the body",
}

MOVEMENT_VOCAB: dict[str, str] = {
    "straight": "straight-line movement along a single axis",
    "arc": "curved path (partial arc, less than a full circle)",
    "circular": "full circle in the vertical or horizontal plane",
    "zigzag / wavy": "repeated side-to-side undulation",
    "wiggling fingers": "finger-play without hand displacement",
    "tapping": "repeated short contact against a location",
    "nodding": "small wrist flexion / extension",
    "twisting": "forearm rotation in place",
    "stirring": "small continuous circles in neutral space",
    "opening": "handshape change from closed to open",
    "closing": "handshape change from open to closed",
}

SIZE_VOCAB = ("small", "normal", "large")
SPEED_VOCAB = ("slow", "normal", "fast", "tense", "relaxed")
REPEAT_VOCAB = (
    "once",
    "twice",
    "three-plus times",
    "continuous",
    "alternating",
)

NON_MANUAL_VOCAB: dict[str, str] = {
    "eyebrows raised": "lifted brows — often marks yes/no questions or topics",
    "eyebrows furrowed": "drawn-together brows — often marks wh-questions",
    "mouth open": "jaw dropped",
    "mouth closed / pursed": "lips pressed",
    "tongue out": "tongue protrudes",
    "cheeks puffed": "air held in cheeks",
    "head tilt (left / right / forward / back)": "head off-axis",
    "head shake": "repeated lateral head movement",
    "head nod": "repeated vertical head movement",
    "eye gaze to location": "eyes look toward a spatial referent",
}


def _vocab_table(title: str, vocab: dict[str, str]) -> str:
    lines = [f"### {title}"]
    for term, gloss in vocab.items():
        lines.append(f"- **{term}** — {gloss}")
    return "\n".join(lines)


def _list_vocab(title: str, items: tuple[str, ...]) -> str:
    return f"### {title}\n- " + "\n- ".join(items)


def _build_system_prompt() -> str:
    """Compose the system prompt sent on every parser call.

    Includes the :class:`PartialSignParameters` JSON schema verbatim and
    the four reference-vocabulary tables. Kept as a single string (not a
    template) because the prompt is stable across requests — invalidating
    it invalidates every recorded fixture.
    """
    schema = PartialSignParameters.model_json_schema()
    schema_block = json.dumps(schema, indent=2)
    gap_fields_block = "\n- ".join(ALLOWED_GAP_FIELDS)

    return f"""You are a sign-language phonology parser. You are given a natural-language description of a single sign (in English) and you extract structured phonological parameters from it.

## What to produce

Return a JSON object with exactly two top-level fields:

1. ``parameters`` — a populated (possibly partial) ``PartialSignParameters`` object. Its JSON schema is shown below for reference.
2. ``gaps`` — a list of ``Gap`` objects flagging slots you could not fill with confidence.

### PartialSignParameters schema

```json
{schema_block}
```

Every slot is optional. When the description does not unambiguously specify a slot, leave the slot ``null`` (or omit from ``movement`` / ``non_manual`` entirely) **and** add a corresponding entry to ``gaps``.

## Gap object

Each ``Gap`` has three fields:

- ``field`` — the slot name, using one of these exact values (dot-paths for nested slots):
- {gap_fields_block}
- ``reason`` — a short English clause explaining why the slot was left empty (e.g. "description says 'claw-like' which could be bent-5 or bent-V").
- ``suggested_question`` — a single follow-up question the author could answer to disambiguate (e.g. "Is the handshape bent-5 (all five fingers bent at the knuckles) or bent-V (only index and middle)?").

## Hard rules

1. **NEVER invent HamNoSys codepoints.** Do not emit characters in the Unicode Private Use Area (U+E000..U+F8FF). Use only plain-English phrases.
2. When multiple plausible interpretations exist, leave the slot ``null`` and open a gap — do not guess.
3. When the description flags a regional variant ("the London variant", "Northern British"), capture that *in the gap's reason* and flag the affected slot(s). Do not silently pick a variant.
4. Populate ``non_manual`` only if the description mentions face / head / body features. If no non-manual is mentioned, leave ``non_manual: null`` — do not emit an empty ``non_manual`` object and do not open gaps for non-manuals.
5. If a slot is explicitly described, populate it. Do not flag an over-specified sign as "ambiguous" just because a HamNoSys specialist might phrase it differently.
6. Keep field values short — 1–3 tokens where possible (e.g. ``location: "temple"``, not ``location: "near the right temple"``).

## Mandatory vs. optional slots

These four **mandatory** slots must have either a value or a gap entry — never both empty. If the prose does not specify one, leave it ``null`` and open a corresponding gap:

- ``handshape_dominant``
- ``orientation_extended_finger``
- ``orientation_palm``
- ``location``

These slots are **optional** — populate them only if the prose mentions them, no gap needed when silent:

- ``handshape_nondominant`` — only for two-handed signs
- ``contact`` — only when contact / touching / tapping is mentioned
- ``movement`` — empty list when no movement is described
- ``non_manual`` — ``null`` when no face / head / body features are described

## Vocabulary reference

Use these plain-English terms when possible. Listed terms are examples, not an exhaustive enumeration — prefer a listed term over a synonym.

{_vocab_table("Handshapes", HANDSHAPE_VOCAB)}

{_vocab_table("Locations", LOCATION_VOCAB)}

{_vocab_table("Extended-finger direction", ORIENTATION_EXT_FINGER_VOCAB)}

{_vocab_table("Palm direction", ORIENTATION_PALM_VOCAB)}

{_vocab_table("Movement paths & actions", MOVEMENT_VOCAB)}

{_list_vocab("Size modifiers", SIZE_VOCAB)}

{_list_vocab("Speed modifiers", SPEED_VOCAB)}

{_list_vocab("Repeat modifiers", REPEAT_VOCAB)}

{_vocab_table("Non-manuals", NON_MANUAL_VOCAB)}

## Worked example

Input: "Bent-5 handshape near the temple, palm facing the signer, small circular motion repeated twice, eyebrows raised."

Output:
```json
{{
  "parameters": {{
    "handshape_dominant": "bent-5",
    "handshape_nondominant": null,
    "orientation_extended_finger": null,
    "orientation_palm": "toward signer",
    "location": "temple",
    "contact": null,
    "movement": [
      {{"path": "circular", "size_mod": "small", "speed_mod": null, "repeat": "twice"}}
    ],
    "non_manual": {{
      "mouth_picture": null,
      "eye_gaze": null,
      "head_movement": null,
      "eyebrows": "raised",
      "facial_expression": null
    }}
  }},
  "gaps": [
    {{
      "field": "orientation_extended_finger",
      "reason": "description does not specify where the fingertips point",
      "suggested_question": "Which direction do the fingertips point — up, forward, or toward the signer?"
    }}
  ]
}}
```
"""


SYSTEM_PROMPT: str = _build_system_prompt()


# ---------------------------------------------------------------------------
# Response schema (OpenAI strict JSON-schema)
# ---------------------------------------------------------------------------


# Keywords OpenAI strict mode does not accept (either silently ignored or
# causes a 400). Stripped recursively before sending.
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
    """Recursively massage a Pydantic-emitted schema into OpenAI strict form.

    - Drops unsupported keywords (``default``, ``title``, etc.).
    - On every ``type: object`` node: sets ``additionalProperties: false``
      and forces ``required`` to list every property (including
      nullable-typed ones — OpenAI's strict mode requires the *key* to
      always be present in the response even if the value is ``null``).
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


def build_parser_response_schema() -> dict[str, Any]:
    """Return the top-level schema sent as ``response_format.json_schema``.

    Wraps :class:`PartialSignParameters` and :class:`Gap` into a single
    object ``{parameters, gaps}`` so the LLM's response is self-contained.
    """
    partial_schema = _strictify(PartialSignParameters.model_json_schema())
    gap_schema = _strictify(Gap.model_json_schema())

    # Pydantic nests $defs under each model's schema. Hoist them to the
    # top level so both ``parameters`` and ``gaps`` can $ref them.
    defs: dict[str, Any] = {}
    for model_schema in (partial_schema, gap_schema):
        defs.update(model_schema.pop("$defs", {}))

    wrapper: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["parameters", "gaps"],
        "properties": {
            "parameters": partial_schema,
            "gaps": {"type": "array", "items": gap_schema},
        },
    }
    if defs:
        wrapper["$defs"] = defs
    return wrapper


PARSER_RESPONSE_SCHEMA: dict[str, Any] = build_parser_response_schema()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ParserError(Exception):
    """Raised when the LLM's response cannot be parsed into a ParseResult."""


def parse_description(
    prose: str,
    *,
    client: LLMClient | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 1500,
    request_id: str | None = None,
) -> ParseResult:
    """Parse a prose description into a :class:`ParseResult`.

    Parameters
    ----------
    prose:
        Natural-language description of a single sign. Passed verbatim
        as the user message.
    client:
        An :class:`LLMClient` to route the call through. One is
        constructed with default settings if omitted — the caller should
        usually inject a shared client so budget / telemetry is unified.
    model / temperature / max_tokens:
        Call-shape knobs. ``model="gpt-4o"`` and ``temperature=0.1`` are
        spec defaults; override sparingly and update the eval doc when
        you do.
    request_id:
        Auto-generated UUID if omitted. Surfaces in the telemetry log.

    Raises
    ------
    ParserError
        When the LLM returns malformed JSON or the JSON fails
        :class:`ParseResult` validation. The original LLM content is
        included in the exception message so fixture authors can inspect
        what actually came back.
    """
    if not isinstance(prose, str) or not prose.strip():
        raise ParserError("prose must be a non-empty string")

    req_id = request_id or f"parse-{uuid.uuid4().hex[:12]}"
    llm = client if client is not None else LLMClient(model=model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prose.strip()},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "parse_description_result",
            "strict": True,
            "schema": PARSER_RESPONSE_SCHEMA,
        },
    }

    chat_result: ChatResult = llm.chat(
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        request_id=req_id,
    )
    return _build_parse_result(chat_result.content)


def _build_parse_result(raw_content: str) -> ParseResult:
    """Validate the LLM's raw text into a :class:`ParseResult`.

    Separated from :func:`parse_description` so test fixtures — which
    replay a recorded ``raw_content`` rather than calling the API — can
    exercise the exact validation path.
    """
    stripped = (raw_content or "").strip()
    if not stripped:
        raise ParserError("LLM returned empty content")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ParserError(
            f"LLM response was not valid JSON: {exc}. "
            f"Raw content (truncated): {stripped[:200]!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise ParserError(
            f"LLM response must be a JSON object, got {type(payload).__name__}"
        )
    if "parameters" not in payload or "gaps" not in payload:
        raise ParserError(
            "LLM response missing required keys — expected 'parameters' and 'gaps'"
        )

    try:
        parameters = PartialSignParameters.model_validate(payload["parameters"])
        gaps_raw = payload["gaps"] or []
        gaps = [Gap.model_validate(g) for g in gaps_raw]
    except ValidationError as exc:
        raise ParserError(f"LLM response failed schema validation: {exc}") from exc

    # Hygiene check: reject codepoints in the Private Use Area anywhere in
    # the parsed fields. The system prompt forbids them explicitly, but a
    # defense-in-depth check keeps a misbehaving LLM from leaking HamNoSys
    # characters into what is supposed to be a plain-English stage.
    _reject_pua(parameters)

    return ParseResult(
        parameters=parameters,
        gaps=gaps,
        raw_response=stripped,
    )


def _reject_pua(params: PartialSignParameters) -> None:
    def _check(value: Any, path: str) -> None:
        if isinstance(value, str):
            for ch in value:
                if 0xE000 <= ord(ch) <= 0xF8FF:
                    raise ParserError(
                        f"parser output contained HamNoSys PUA codepoint "
                        f"U+{ord(ch):04X} in field {path!r}"
                    )

    for fname, fvalue in params.model_dump().items():
        if fname == "movement" and isinstance(fvalue, list):
            for i, seg in enumerate(fvalue):
                if isinstance(seg, dict):
                    for sk, sv in seg.items():
                        _check(sv, f"movement[{i}].{sk}")
        elif fname == "non_manual" and isinstance(fvalue, dict):
            for nk, nv in fvalue.items():
                _check(nv, f"non_manual.{nk}")
        else:
            _check(fvalue, fname)


__all__ = [
    "HANDSHAPE_VOCAB",
    "LOCATION_VOCAB",
    "MOVEMENT_VOCAB",
    "NON_MANUAL_VOCAB",
    "ORIENTATION_EXT_FINGER_VOCAB",
    "ORIENTATION_PALM_VOCAB",
    "PARSER_RESPONSE_SCHEMA",
    "ParserError",
    "SYSTEM_PROMPT",
    "build_parser_response_schema",
    "parse_description",
]
