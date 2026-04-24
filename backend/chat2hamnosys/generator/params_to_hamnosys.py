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
from llm.client import LLMConfigError
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
    last_candidate:
        The final HamNoSys string produced before giving up. On success
        this is the same as ``hamnosys``; on failure it is the best we
        had before the validator rejected it (possibly empty when the
        composer couldn't even assemble a candidate).
    """

    hamnosys: str | None
    validation: ValidationResult
    used_llm_fallback: bool = False
    llm_fallback_fields: list[str] = field(default_factory=list)
    confidence: float = 1.0
    errors: list[str] = field(default_factory=list)
    last_candidate: str = ""


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
PROMPT_ID_WHOLE_SIGN: str = "generate_hamnosys_whole_sign"

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


_WHOLE_SIGN_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "hamnosys_whole_sign",
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
                        "codepoints encoding the full sign."
                    ),
                    "pattern": "^([0-9A-F]{4})(\\s+[0-9A-F]{4})*$",
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence on the slot choices.",
                },
            },
        },
    },
}


_WHOLE_SIGN_PROMPT = load_prompt(PROMPT_ID_WHOLE_SIGN)
_SYSTEM_PROMPT_WHOLE_SIGN: str = _WHOLE_SIGN_PROMPT.render()
_WHOLE_SIGN_METADATA: PromptMetadata = _WHOLE_SIGN_PROMPT.metadata


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
# LLM fallback — whole-sign emergency
# ---------------------------------------------------------------------------


def _llm_whole_sign(
    parameters: PartialSignParameters,
    pieces: list[_Piece],
    *,
    client: LLMClient,
    request_id: str,
    previous_errors: list[str],
    prose: str = "",
    gloss: str = "",
    sign_language: str = "bsl",
) -> tuple[str, str]:
    """Last-resort: ask the LLM for the whole HamNoSys string at once.

    Used when the slot-by-slot composer + per-slot LLM fallback +
    repair loop have all failed to produce a validator-clean string.
    Returns ``(hamnosys, rationale)``; ``hamnosys`` is empty on failure.
    """
    resolved_chunks: dict[str, str] = {}
    unresolved_slots: list[str] = []
    for piece in pieces:
        if piece.field_name == "symmetry":
            continue
        if piece.resolved and piece.chunk:
            resolved_chunks[piece.field_name] = f"{ord(piece.chunk[0]):04X}"
        else:
            unresolved_slots.append(piece.field_name)

    user_payload = {
        "prose": prose or "",
        "gloss": gloss or "",
        "sign_language": sign_language,
        "partial_params": parameters.model_dump(mode="json", exclude_none=True),
        "resolved_chunks": resolved_chunks,
        "unresolved_slots": unresolved_slots,
        "previous_errors": list(previous_errors)[:5],
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_WHOLE_SIGN},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    try:
        result: ChatResult = client.chat(
            messages=messages,
            response_format=_WHOLE_SIGN_SCHEMA,
            temperature=0.1,
            max_tokens=400,
            request_id=f"{request_id}:whole_sign",
            prompt_metadata=_WHOLE_SIGN_METADATA,
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning("LLM whole-sign call failed: %s", exc)
        return "", f"llm call failed: {type(exc).__name__}"

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        return "", "llm returned invalid json"
    hex_str = payload.get("hamnosys_hex", "")
    rationale = str(payload.get("rationale", "")).strip()
    try:
        whole = _decode_hex_string(hex_str)
    except ValueError:
        return "", "llm returned non-hex payload"
    return whole, rationale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _assemble(pieces: list[_Piece]) -> str:
    return "".join(p.chunk for p in pieces)


def _maybe_construct_client(client: LLMClient | None) -> LLMClient | None:
    """Return ``client`` if given, else try to build a default one.

    The orchestrator and the dependency layer historically passed
    ``client=None`` and expected the generator to gracefully degrade.
    Auto-constructing here means the slot-fallback / repair / whole-sign
    paths fire whenever an OpenAI key (project secret OR per-request
    BYO header) is configured — which is the normal production case.
    Returns ``None`` if no key is available so legacy graceful-degrade
    callers still get the ``no LLM client available`` error path.
    """
    if client is not None:
        return client
    try:
        return LLMClient()
    except LLMConfigError as exc:
        logger.info(
            "no OpenAI key available; LLM fallback paths disabled (%s)", exc
        )
        return None


def generate(
    parameters: PartialSignParameters,
    *,
    client: LLMClient | None = None,
    request_id: str | None = None,
    prose: str = "",
    gloss: str = "",
    sign_language: str = "bsl",
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
        grammar validation. If ``None``, one is auto-constructed from
        the ambient OpenAI key (project secret or BYO header). Pass
        ``client=None`` and call without a configured key to opt into
        the legacy "no LLM" graceful-degrade path.
    request_id:
        Used as a prefix for telemetry ``request_id`` when the LLM
        fallback fires. A random UUID is generated if not provided.
    prose, gloss, sign_language:
        Optional context forwarded to the whole-sign emergency
        fallback. The slot-by-slot composer ignores these — they only
        matter when slot-and-repair both give up and we ask the LLM to
        compose the entire sign from the contributor's prose.

    Returns
    -------
    :class:`GenerateResult` — see its docstring.
    """
    rid = request_id or f"gen-{uuid.uuid4().hex[:12]}"
    client = _maybe_construct_client(client)

    pieces = _compose_pieces(parameters)
    unresolved = [p for p in pieces if not p.resolved]

    fallback_fields: list[str] = []
    used_llm_fallback = False
    errors_out: list[str] = []
    running_confidence = 1.0

    # ---- LLM path: SiGML-first (one-shot) --------------------------
    #
    # The generator used to have three LLM paths that emitted
    # HamNoSys PUA codepoints directly (``_llm_resolve_slot``,
    # ``_llm_repair``, ``_llm_whole_sign``). All three were brittle:
    # models routinely hallucinated non-existent codepoints or mixed
    # in decimal hex, and every attempt was one more fragile
    # parse-coerce round-trip. The user's own instruction crystallised
    # the issue: "make sure it generates written SiGML and only then
    # maybe converts to regular HamNoSys" — because the XML tag names
    # are concrete, named, easy to validate, and grounded in real
    # few-shot BSL examples.
    #
    # So the only LLM path now is ``generate_sigml_direct``, which
    # asks the model for a complete ``<hns_sign>...</hns_sign>``
    # fragment and reverse-maps it to HamNoSys PUA via the symbol
    # table. Failure → we surface errors; we do NOT fall back to
    # slot-by-slot LLM.
    if unresolved and client is not None:
        from .sigml_direct import generate_sigml_direct

        sigml_hamnosys, _sigml_xml, sigml_rationale = generate_sigml_direct(
            parameters=parameters,
            client=client,
            request_id=rid,
            prose=prose,
            gloss=gloss,
            sign_language=sign_language,
            previous_errors=errors_out,
        )
        if sigml_hamnosys:
            normalized = normalize(sigml_hamnosys)
            vr = validate(normalized)
            if vr.ok:
                fallback_fields.append("_sigml_direct")
                logger.info(
                    "sigml-direct fallback succeeded: %s",
                    sigml_rationale[:200],
                )
                return GenerateResult(
                    hamnosys=normalized,
                    validation=vr,
                    used_llm_fallback=True,
                    llm_fallback_fields=fallback_fields,
                    confidence=0.6,
                    errors=[],
                    last_candidate=normalized,
                )
            errors_out.append(
                f"sigml-direct produced invalid HamNoSys: {vr.summary()}"
            )
        else:
            errors_out.append(f"sigml-direct gave up: {sigml_rationale}")
        # SiGML-direct exhausted its internal retry loop. Fall through
        # to report the combined failure — the deterministic composer
        # may have resolved enough pieces that an approximate sign is
        # still worth handing back, but we don't make up the missing
        # slots via LLM-codepoint-pick.

    still_missing = [p for p in pieces if not p.resolved]
    if still_missing:
        candidate = _assemble(pieces)
        vr = validate(candidate) if candidate else ValidationResult(
            ok=False,
            errors=[],
            warnings=[],
            tree=None,
        )
        # Surface each still-unresolved slot by name so the chat panel
        # can show the contributor what was left open. SiGML-direct
        # (if it ran) already added its own failure reason to
        # errors_out; append the per-slot names on top.
        missing_msgs = [
            f"no LLM client available to resolve {p.field_name!r} "
            f"(term={p.term!r})"
            if client is None
            else f"slot {p.field_name!r} unresolved (term={p.term!r})"
            for p in still_missing
        ]
        return GenerateResult(
            hamnosys=None,
            validation=vr,
            used_llm_fallback=used_llm_fallback,
            llm_fallback_fields=fallback_fields,
            confidence=0.0,
            errors=errors_out + missing_msgs,
            last_candidate=candidate,
        )

    candidate = normalize(_assemble(pieces))
    vr = validate(candidate)

    # ---- Validation failure recovery: SiGML-direct, then repair ----
    if not vr.ok and client is not None:
        from .sigml_direct import generate_sigml_direct

        sigml_hamnosys, _sigml_xml, sigml_rationale = generate_sigml_direct(
            parameters=parameters,
            client=client,
            request_id=rid,
            prose=prose,
            gloss=gloss,
            sign_language=sign_language,
            previous_errors=list(errors_out) + [vr.summary()],
        )
        if sigml_hamnosys:
            normalized = normalize(sigml_hamnosys)
            wvr = validate(normalized)
            if wvr.ok:
                fallback_fields.append("_sigml_direct")
                logger.info(
                    "sigml-direct recovery succeeded post-validation-failure"
                )
                # Drop in-progress retry traces — see the equivalent
                # success-path comment in the unresolved-slots branch
                # above. The contributor sees a working sign without
                # stale "missing palm_direction" / "invalid json"
                # noise next to it.
                return GenerateResult(
                    hamnosys=normalized,
                    validation=wvr,
                    used_llm_fallback=True,
                    llm_fallback_fields=fallback_fields,
                    confidence=0.5,
                    errors=[],
                    last_candidate=normalized,
                )
            errors_out.append(
                f"sigml-direct produced invalid HamNoSys: {wvr.summary()}"
            )
        else:
            errors_out.append(f"sigml-direct gave up: {sigml_rationale}")

    # The legacy ``_llm_repair`` loop asked the model to rewrite a
    # HamNoSys codepoint string — the same kind of fragile LLM-picks-
    # PUA-hex path that the user asked us to kill. Removed. If SiGML-
    # direct couldn't rescue a validator failure, the caller surfaces
    # the error and offers Submit-as-is or a manual edit on the SiGML
    # tab. That's both simpler and more reliable than round-tripping
    # the model through a broken HamNoSys string.
    if not vr.ok:
        return GenerateResult(
            hamnosys=None,
            validation=vr,
            used_llm_fallback=used_llm_fallback,
            llm_fallback_fields=fallback_fields,
            confidence=0.0,
            errors=errors_out,
            last_candidate=candidate,
        )

    # Successful slot/repair path. Drop accumulated retry traces from
    # the SiGML-direct attempts and the per-slot fallback failures —
    # we have a working HamNoSys, the in-progress notes are stale.
    # Surface them only on the failure return paths above.
    return GenerateResult(
        hamnosys=candidate,
        validation=vr,
        used_llm_fallback=used_llm_fallback,
        llm_fallback_fields=fallback_fields,
        confidence=running_confidence,
        errors=[],
        last_candidate=candidate,
    )


__all__ = ["GenerateResult", "generate"]
