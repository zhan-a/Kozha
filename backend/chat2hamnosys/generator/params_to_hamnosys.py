"""Parameters → HamNoSys generator.

Given a :class:`~parser.models.PartialSignParameters` populated with
plain-English slot values, produce a validated HamNoSys 4.0 string.

Two-layer design
----------------
1. **Deterministic composer** (primary). Every plain-English term is
   normalized and looked up in :data:`generator.vocab.VOCAB`, the
   authored YAML map. Hits are concatenated in canonical HamNoSys order:

       [symmetry] handshape ext_finger palm location
                 [contact] [movement [size/speed/timing] [repeat]]

2. **LLM fallback** (secondary). Slots the composer could not resolve
   are referred one-by-one to :class:`llm.LLMClient`, which is asked to
   pick a HamNoSys codepoint from the slot's class and return a
   confidence score. Each fallback is logged with the slot name, the
   original term, and the chosen codepoint.

Every candidate string is passed through :func:`hamnosys.validate`. If
the Lark grammar or semantic pass rejects the output, up to two repair
passes are made with the validator's error list fed back to the LLM. If
none of those succeed, the generator returns ``hamnosys=None`` with the
failing :class:`ValidationResult` and a confidence of 0.

Two-handed signs
----------------
Version 1 handles only the **symmetric** case: when
``handshape_nondominant`` is set we prefix the output with
``hamsymmpar`` (U+E0E8) so the renderer mirrors the dominant block.
Fully independent two-hand specs (``hamnondominant`` + separate
handshape/orientation/location blocks) are listed as a known gap in the
Prompt-7 eval doc and deferred to a later prompt.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from hamnosys import SYMBOLS, SymClass, ValidationResult, classify, normalize, validate
from hamnosys.symbols import symbols_in_classes
from llm import ChatResult, LLMClient, LLMError
from llm.budget import BudgetExceeded
from parser.models import PartialMovementSegment, PartialSignParameters
from prompts import PromptMetadata, load_prompt

from .vocab import VOCAB, VocabEntry, normalize_term


logger = logging.getLogger(__name__)


_MAX_VALIDATION_RETRIES = 2
_SYMMETRY_PAR: str = chr(0xE0E8)  # hamsymmpar — mirror both hands


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GenerateResult:
    """Outcome of :func:`generate`.

    Attributes
    ----------
    hamnosys:
        The assembled HamNoSys string on success, else ``None``.
    validation:
        :class:`ValidationResult` for the (attempted) string. Always
        populated so callers can inspect errors/warnings even on
        success.
    used_llm_fallback:
        ``True`` iff at least one slot (or a repair pass) was resolved
        by the LLM rather than by the deterministic composer.
    llm_fallback_fields:
        Ordered list of slot names for which the LLM was consulted.
        Repairs are recorded under the pseudo-field ``"_repair"``.
    confidence:
        Product of per-slot confidences. Deterministic hits contribute
        ``1.0``; LLM hits contribute whatever the model self-reports.
        ``0.0`` when the final string still fails validation.
    errors:
        Human-readable explanations of any unresolved slots (e.g.
        ``"no LLM client available to resolve handshape_dominant"``).
    """

    hamnosys: str | None
    validation: ValidationResult
    used_llm_fallback: bool = False
    llm_fallback_fields: list[str] = field(default_factory=list)
    confidence: float = 1.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


@dataclass
class _Piece:
    """One slot in the canonical sign order.

    ``chunk`` is the final HamNoSys characters emitted for this slot —
    either resolved deterministically from VOCAB, obtained from the LLM
    fallback, or empty if both paths failed.
    """

    field_name: str            # user-facing slot name (for logging / gap reports)
    vocab_slot: str            # key into VOCAB
    term: str | None           # normalized plain-English term, None for hardcoded
    chunk: str = ""
    resolved: bool = False
    via_llm: bool = False
    confidence: float = 1.0


def _vocab_resolve(slot: str, term: str | None) -> tuple[str, bool]:
    """Look up ``term`` in VOCAB slot ``slot``.

    Returns ``(chunk, resolved)``. ``resolved=True`` means the
    deterministic layer handled this slot — either with a codepoint
    chunk or a legitimate no-op (``""``). ``resolved=False`` means the
    term is non-empty but missing from VOCAB; the caller must route to
    the LLM fallback.
    """
    if term is None:
        return "", True
    stripped = term.strip()
    if not stripped:
        return "", True
    entry = VOCAB.lookup(slot, stripped)
    if entry is None:
        return "", False
    return entry.string, True


def _segment_pieces(seg: PartialMovementSegment, index: int) -> list[_Piece]:
    """Expand one movement segment into its canonical sub-pieces."""
    pieces: list[_Piece] = []
    # Path first — movement classes lead the action block.
    if seg.path:
        chunk, ok = _vocab_resolve("movement_path", seg.path)
        pieces.append(
            _Piece(
                field_name=f"movement[{index}].path",
                vocab_slot="movement_path",
                term=seg.path,
                chunk=chunk,
                resolved=ok,
            )
        )
    # Then size / speed / timing (in that order — matches validator's
    # expected modifier adjacency and keeps HamNoSys idiomatic).
    if seg.size_mod:
        chunk, ok = _vocab_resolve("size_mod", seg.size_mod)
        pieces.append(
            _Piece(
                field_name=f"movement[{index}].size_mod",
                vocab_slot="size_mod",
                term=seg.size_mod,
                chunk=chunk,
                resolved=ok,
            )
        )
    if seg.speed_mod:
        chunk, ok = _vocab_resolve("speed_mod", seg.speed_mod)
        pieces.append(
            _Piece(
                field_name=f"movement[{index}].speed_mod",
                vocab_slot="speed_mod",
                term=seg.speed_mod,
                chunk=chunk,
                resolved=ok,
            )
        )
    if seg.repeat:
        chunk, ok = _vocab_resolve("repeat", seg.repeat)
        pieces.append(
            _Piece(
                field_name=f"movement[{index}].repeat",
                vocab_slot="repeat",
                term=seg.repeat,
                chunk=chunk,
                resolved=ok,
            )
        )
    return pieces


def _compose_pieces(params: PartialSignParameters) -> list[_Piece]:
    """Assemble the ordered list of slot pieces for ``params``."""
    pieces: list[_Piece] = []

    # Symmetry prefix: fire only when the author explicitly wrote a
    # non-dominant handshape. We always use hamsymmpar — see module
    # docstring for why asymmetric two-handed is out of scope for v1.
    if params.handshape_nondominant and params.handshape_nondominant.strip():
        pieces.append(
            _Piece(
                field_name="symmetry",
                vocab_slot="_symmetry",
                term=None,
                chunk=_SYMMETRY_PAR,
                resolved=True,
            )
        )

    # Mandatory slots — four in canonical order.
    chunk, ok = _vocab_resolve("handshape", params.handshape_dominant)
    pieces.append(
        _Piece(
            field_name="handshape_dominant",
            vocab_slot="handshape",
            term=params.handshape_dominant,
            chunk=chunk,
            resolved=ok,
        )
    )
    chunk, ok = _vocab_resolve(
        "orientation_ext_finger", params.orientation_extended_finger
    )
    pieces.append(
        _Piece(
            field_name="orientation_extended_finger",
            vocab_slot="orientation_ext_finger",
            term=params.orientation_extended_finger,
            chunk=chunk,
            resolved=ok,
        )
    )
    chunk, ok = _vocab_resolve("orientation_palm", params.orientation_palm)
    pieces.append(
        _Piece(
            field_name="orientation_palm",
            vocab_slot="orientation_palm",
            term=params.orientation_palm,
            chunk=chunk,
            resolved=ok,
        )
    )
    chunk, ok = _vocab_resolve("location", params.location)
    pieces.append(
        _Piece(
            field_name="location",
            vocab_slot="location",
            term=params.location,
            chunk=chunk,
            resolved=ok,
        )
    )

    # Optional contact.
    if params.contact and params.contact.strip():
        chunk, ok = _vocab_resolve("contact", params.contact)
        pieces.append(
            _Piece(
                field_name="contact",
                vocab_slot="contact",
                term=params.contact,
                chunk=chunk,
                resolved=ok,
            )
        )

    # Movement segments.
    for i, seg in enumerate(params.movement or []):
        pieces.extend(_segment_pieces(seg, i))

    return pieces


# ---------------------------------------------------------------------------
# LLM fallback — per-slot codepoint resolution
# ---------------------------------------------------------------------------


_SLOT_CLASSES: dict[str, tuple[SymClass, ...]] = {
    "handshape": (SymClass.HANDSHAPE_BASE,),
    "orientation_ext_finger": (SymClass.EXT_FINGER_DIR,),
    "orientation_palm": (SymClass.PALM_DIR,),
    "location": (
        SymClass.LOC_HEAD,
        SymClass.LOC_TORSO,
        SymClass.LOC_NEUTRAL,
        SymClass.LOC_ARM,
        SymClass.LOC_HAND_ZONE,
        SymClass.LOC_FINGER,
    ),
    "contact": (SymClass.CONTACT,),
    "movement_path": (
        SymClass.MOVE_STRAIGHT,
        SymClass.MOVE_CIRCLE,
        SymClass.MOVE_ACTION,
        SymClass.MOVE_CLOCK,
        SymClass.MOVE_ARC,
        SymClass.MOVE_WAVY,
        SymClass.MOVE_ELLIPSE,
    ),
    "size_mod": (SymClass.SIZE_MOD,),
    "speed_mod": (SymClass.SPEED_MOD,),
    "timing": (SymClass.TIMING,),
    "repeat": (SymClass.REPEAT,),
}


_FALLBACK_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "hamnosys_slot_choice",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["codepoint_hex", "confidence", "rationale"],
            "properties": {
                "codepoint_hex": {
                    "type": "string",
                    "description": (
                        "A single 4-digit uppercase hex HamNoSys codepoint "
                        "like 'E001'. Must be exactly one of the codepoints "
                        "listed in the `allowed` field of the user message."
                    ),
                    "pattern": "^[0-9A-F]{4}$",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "One sentence explaining why the chosen codepoint "
                        "is the closest match for the term."
                    ),
                },
            },
        },
    },
}


PROMPT_ID_SLOT: str = "generate_hamnosys_fallback"
PROMPT_ID_REPAIR: str = "generate_hamnosys_repair"

_SLOT_PROMPT = load_prompt(PROMPT_ID_SLOT)
_SYSTEM_PROMPT_SLOT: str = _SLOT_PROMPT.render()
_SLOT_METADATA: PromptMetadata = _SLOT_PROMPT.metadata


def _allowed_codepoints(vocab_slot: str) -> list[dict[str, str]]:
    """Return the allowed codepoint table for a given slot.

    Each entry is a dict with ``hex``, ``name``, and ``class`` — the
    minimum the LLM needs to make an informed pick.
    """
    classes = _SLOT_CLASSES.get(vocab_slot)
    if classes is None:
        return []
    rows: list[dict[str, str]] = []
    for sym in symbols_in_classes(classes):
        rows.append(
            {
                "hex": f"{sym.codepoint:04X}",
                "name": sym.short_name,
                "class": sym.sym_class.value,
            }
        )
    return rows


def _vocab_excerpt_rows(vocab_slot: str, *, limit: int = 30) -> list[dict[str, str]]:
    """Return a small table of existing ``term → codepoints`` rows.

    Helps the LLM pick an adjacent concept rather than hallucinating.
    """
    if vocab_slot in VOCAB.slots:
        excerpt = VOCAB.excerpt(vocab_slot, limit=limit)
        return [{"term": t, "codepoints": h} for t, h in excerpt]
    return []


def _llm_resolve_slot(
    piece: _Piece,
    *,
    client: LLMClient,
    request_id: str,
) -> tuple[str, float, str]:
    """Ask the LLM for a codepoint to fill ``piece``.

    Returns ``(chunk, confidence, rationale)``. An empty ``chunk``
    signals the LLM did not return a usable codepoint.
    """
    slot = piece.vocab_slot
    allowed = _allowed_codepoints(slot)
    if not allowed:
        return "", 0.0, f"no allowed codepoints registered for slot {slot!r}"

    user_payload = {
        "slot": slot,
        "term": piece.term,
        "allowed": allowed,
        "vocab_examples": _vocab_excerpt_rows(slot),
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_SLOT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    try:
        result: ChatResult = client.chat(
            messages=messages,
            response_format=_FALLBACK_SCHEMA,
            temperature=0.0,
            max_tokens=200,
            request_id=f"{request_id}:fallback:{piece.field_name}",
            prompt_metadata=_SLOT_METADATA,
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning(
            "LLM fallback failed for slot=%s term=%r: %s",
            slot,
            piece.term,
            exc,
        )
        return "", 0.0, f"llm call failed: {type(exc).__name__}"

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM fallback returned invalid JSON for slot=%s term=%r: %s",
            slot,
            piece.term,
            exc,
        )
        return "", 0.0, "llm returned invalid json"

    hex_str = payload.get("codepoint_hex", "").strip().upper()
    confidence = float(payload.get("confidence", 0.0))
    rationale = str(payload.get("rationale", "")).strip()

    try:
        cp = int(hex_str, 16)
    except ValueError:
        return "", 0.0, f"llm returned non-hex codepoint {hex_str!r}"

    if cp not in SYMBOLS:
        return "", 0.0, f"llm returned unknown codepoint U+{cp:04X}"
    expected = _SLOT_CLASSES.get(slot, ())
    if expected and classify(chr(cp)) not in expected:
        return "", 0.0, (
            f"llm returned U+{cp:04X} (class "
            f"{classify(chr(cp)).value if classify(chr(cp)) else '?'}) "
            f"which is not valid for slot {slot!r}"
        )
    return chr(cp), confidence, rationale


# ---------------------------------------------------------------------------
# LLM fallback — whole-string repair
# ---------------------------------------------------------------------------


_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "hamnosys_repair",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["hamnosys_hex", "rationale"],
            "properties": {
                "hamnosys_hex": {
                    "type": "string",
                    "description": (
                        "A space-separated list of 4-digit uppercase hex "
                        "codepoints like 'E001 E020 E038 E040'. Must "
                        "encode a valid HamNoSys 4.0 sign string."
                    ),
                    "pattern": "^([0-9A-F]{4})(\\s+[0-9A-F]{4})*$",
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence on why the proposed repair fixes the errors.",
                },
            },
        },
    },
}


_REPAIR_PROMPT = load_prompt(PROMPT_ID_REPAIR)
_SYSTEM_PROMPT_REPAIR: str = _REPAIR_PROMPT.render()
_REPAIR_METADATA: PromptMetadata = _REPAIR_PROMPT.metadata


def _decode_hex_string(hex_str: str) -> str:
    """Decode ``'E001 E020'`` → ``chr(0xE001) + chr(0xE020)``."""
    parts = hex_str.strip().split()
    chars: list[str] = []
    for p in parts:
        cp = int(p, 16)
        chars.append(chr(cp))
    return "".join(chars)


def _llm_repair(
    candidate: str,
    vr: ValidationResult,
    *,
    client: LLMClient,
    request_id: str,
    attempt: int,
) -> tuple[str, str]:
    """Ask the LLM to repair ``candidate`` given its validation errors.

    Returns ``(repaired, rationale)``. ``repaired`` is empty on failure.
    """
    error_lines = [
        {
            "position": e.position,
            "symbol_hex": f"{ord(e.symbol):04X}" if e.symbol else "",
            "code": e.code,
            "message": e.message,
        }
        for e in vr.errors
    ]
    candidate_hex = " ".join(f"{ord(c):04X}" for c in candidate)
    user_payload = {
        "candidate_hex": candidate_hex,
        "errors": error_lines,
        "guidance": (
            "Return the fewest-edit repair. Do not change codepoints that "
            "are unrelated to the reported error positions."
        ),
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_REPAIR},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    try:
        result: ChatResult = client.chat(
            messages=messages,
            response_format=_REPAIR_SCHEMA,
            temperature=0.0,
            max_tokens=400,
            request_id=f"{request_id}:repair:{attempt}",
            prompt_metadata=_REPAIR_METADATA,
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning("LLM repair call failed: %s", exc)
        return "", f"llm call failed: {type(exc).__name__}"

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        return "", "llm returned invalid json"
    hex_str = payload.get("hamnosys_hex", "")
    rationale = str(payload.get("rationale", "")).strip()
    try:
        repaired = _decode_hex_string(hex_str)
    except ValueError:
        return "", "llm returned non-hex payload"
    return repaired, rationale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _assemble(pieces: list[_Piece]) -> str:
    return "".join(p.chunk for p in pieces)


def generate(
    parameters: PartialSignParameters,
    *,
    client: LLMClient | None = None,
    request_id: str | None = None,
) -> GenerateResult:
    """Produce a HamNoSys string from plain-English parameters.

    Parameters
    ----------
    parameters:
        A :class:`PartialSignParameters` with slot values filled in as
        plain-English terms. The generator does *not* attempt to fill in
        missing mandatory slots — callers should ensure the parser stage
        has resolved them first.
    client:
        An :class:`LLMClient` used only when the deterministic composer
        cannot resolve a slot, or when the initial assembly fails
        grammar validation. ``None`` disables the fallback; unresolved
        slots then cause the generator to return ``hamnosys=None``.
    request_id:
        Used as a prefix for telemetry ``request_id`` when the LLM
        fallback fires. A random UUID is generated if not provided.

    Returns
    -------
    :class:`GenerateResult` — see its docstring.
    """
    rid = request_id or f"gen-{uuid.uuid4().hex[:12]}"

    pieces = _compose_pieces(parameters)
    unresolved = [p for p in pieces if not p.resolved]

    fallback_fields: list[str] = []
    used_llm_fallback = False
    errors_out: list[str] = []
    running_confidence = 1.0

    # First pass — LLM fallback for slots the deterministic layer missed.
    for piece in unresolved:
        if client is None:
            errors_out.append(
                f"no LLM client available to resolve {piece.field_name!r} "
                f"(term={piece.term!r})"
            )
            continue
        chunk, conf, rationale = _llm_resolve_slot(
            piece, client=client, request_id=rid
        )
        if not chunk:
            errors_out.append(
                f"LLM fallback could not resolve {piece.field_name!r} "
                f"(term={piece.term!r}): {rationale}"
            )
            continue
        piece.chunk = chunk
        piece.resolved = True
        piece.via_llm = True
        piece.confidence = conf
        fallback_fields.append(piece.field_name)
        used_llm_fallback = True
        running_confidence *= max(conf, 0.0)
        logger.info(
            "llm fallback resolved %s term=%r → U+%04X (conf=%.2f): %s",
            piece.field_name,
            piece.term,
            ord(chunk[0]),
            conf,
            rationale,
        )

    # If any slot is still unresolved, we cannot assemble a string.
    still_missing = [p for p in pieces if not p.resolved]
    if still_missing:
        candidate = _assemble(pieces)
        vr = validate(candidate) if candidate else ValidationResult(
            ok=False,
            errors=[],
            warnings=[],
            tree=None,
        )
        return GenerateResult(
            hamnosys=None,
            validation=vr,
            used_llm_fallback=used_llm_fallback,
            llm_fallback_fields=fallback_fields,
            confidence=0.0,
            errors=errors_out,
        )

    candidate = normalize(_assemble(pieces))
    vr = validate(candidate)

    # Second pass — whole-string repair loop.
    for attempt in range(_MAX_VALIDATION_RETRIES):
        if vr.ok:
            break
        if client is None:
            errors_out.append(
                f"validation failed and no LLM client available to repair: "
                f"{vr.summary()}"
            )
            break
        repaired, rationale = _llm_repair(
            candidate, vr, client=client, request_id=rid, attempt=attempt
        )
        if not repaired:
            errors_out.append(
                f"LLM repair attempt {attempt} returned no candidate: {rationale}"
            )
            break
        used_llm_fallback = True
        fallback_fields.append("_repair")
        # Heuristic: repair confidence decays per attempt.
        running_confidence *= 0.7
        candidate = normalize(repaired)
        vr = validate(candidate)
        logger.info(
            "llm repair attempt=%d ok=%s rationale=%s",
            attempt,
            vr.ok,
            rationale,
        )

    if not vr.ok:
        return GenerateResult(
            hamnosys=None,
            validation=vr,
            used_llm_fallback=used_llm_fallback,
            llm_fallback_fields=fallback_fields,
            confidence=0.0,
            errors=errors_out,
        )

    return GenerateResult(
        hamnosys=candidate,
        validation=vr,
        used_llm_fallback=used_llm_fallback,
        llm_fallback_fields=fallback_fields,
        confidence=running_confidence,
        errors=errors_out,
    )


__all__ = ["GenerateResult", "generate"]
