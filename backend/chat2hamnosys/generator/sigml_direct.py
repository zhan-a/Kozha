"""SiGML-direct generation path (Responses API + reasoning + BSL few-shot).

This module is the contribute pipeline's preferred path for resolving
a sign when the deterministic slot composer can't fill every mandatory
slot. Instead of asking the LLM to pick HamNoSys codepoints
slot-by-slot — a brittle dance that has historically blocked on
``BadRequestError`` for a single missing term — we ask the model to
emit a complete, validator-clean ``<hns_sign>...</hns_sign>`` SiGML
fragment in one shot.

Why SiGML, not HamNoSys
-----------------------
HamNoSys lives in the Unicode Private Use Area (U+E000..U+E0F1). The
model has to pick exact codepoints, in order, with no syntactic safety
net. SiGML expresses the same content as named XML elements
(``<hamflathand/>``, ``<hamextfingeru/>``, …) which the model handles
much more reliably. Once we have valid SiGML we recover the HamNoSys
PUA string by mapping each ``<ham*/>`` tag back to its codepoint via
``hamnosys.SYMBOLS``. Both representations are stored on the result so
the renderer (CWASA) and the reviewer (HamNoSys notation tab) each get
their preferred shape.

Few-shot grounding
------------------
Every prompt includes 4–8 BSL signs from
``data/hamnosys_bsl_version1.sigml`` chosen by gloss similarity. BSL
is the highest-quality corpus we ship — every entry is human-curated
and validates against the SiGML DTD. The model sees real, validator-
clean SiGML and is asked to produce the same shape for the new sign.

Reasoning effort
----------------
Picks ``medium`` by default — the model has to reason over slot
ordering, handshape geometry, and the sign-language conventions in the
few-shot examples. The chat client routes any ``gpt-5.x`` / ``o*``
model through the Responses API (see ``llm.client.is_reasoning_model``),
so ``reasoning.effort`` is honoured even though the prompt is shaped
like a regular chat.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from hamnosys import SYMBOLS, SymClass, classify, normalize, validate
from llm import ChatResult, LLMClient
from llm.budget import BudgetExceeded
from parser.models import PartialSignParameters
from prompts import PromptMetadata, load_prompt
from rendering.hamnosys_to_sigml import (
    HamNoSysConversionError,
    SigmlValidationError,
    to_sigml,
    validate_sigml,
)


# Locations span six SymClass values (head/torso/neutral/arm/hand_zone/
# finger); any one of them satisfies the location requirement. Palm
# orientation is a single class. Handshape and ext-finger are the base
# tags that must precede them.
_LOCATION_CLASSES = frozenset({
    SymClass.LOC_HEAD,
    SymClass.LOC_TORSO,
    SymClass.LOC_NEUTRAL,
    SymClass.LOC_ARM,
    SymClass.LOC_HAND_ZONE,
    SymClass.LOC_FINGER,
})


def _missing_required_slots(hamnosys: str) -> list[str]:
    """Return the list of required slot names absent from ``hamnosys``.

    The HamNoSys grammar treats each slot as optional, but CWASA's
    runtime needs handshape, ext-finger direction, palm direction, and
    location all present (in that canonical order) before any movement
    tag. A SiGML missing palm direction validates fine against the DTD
    but blows up at play time with
    ``TypeError: Cannot read properties of null (reading 'getName')``.
    Surface the gap here so the SiGML-direct retry loop can ask the
    model to add the missing tags.
    """
    has_handshape = False
    has_ext_finger = False
    has_palm = False
    has_location = False
    for ch in hamnosys:
        cls = classify(ch)
        if cls is None:
            continue
        if cls is SymClass.HANDSHAPE_BASE:
            has_handshape = True
        elif cls is SymClass.EXT_FINGER_DIR:
            has_ext_finger = True
        elif cls is SymClass.PALM_DIR:
            has_palm = True
        elif cls in _LOCATION_CLASSES:
            has_location = True
    missing: list[str] = []
    if not has_handshape:  missing.append("handshape")
    if not has_ext_finger: missing.append("ext_finger_direction")
    if not has_palm:       missing.append("palm_direction")
    if not has_location:   missing.append("location")
    return missing

from .sigml_examples import few_shot_examples, render_few_shot


logger = logging.getLogger(__name__)


PROMPT_ID_SIGML_DIRECT: str = "generate_sigml_direct"

# Cached short-name → codepoint map. Built lazily once because we use
# it on every SiGML→HamNoSys reverse pass.
_NAME_TO_CODEPOINT: dict[str, int] | None = None


def _name_to_codepoint() -> dict[str, int]:
    global _NAME_TO_CODEPOINT
    if _NAME_TO_CODEPOINT is None:
        _NAME_TO_CODEPOINT = {sym.short_name: sym.codepoint for sym in SYMBOLS.values()}
    return _NAME_TO_CODEPOINT


_HAM_TAG_RE = re.compile(r"<\s*(ham[a-z0-9_]+)\s*/?\s*>", re.IGNORECASE)
_OPEN_HAM_RE = re.compile(r"<\s*(ham[a-z0-9_]+)[^/>]*>", re.IGNORECASE)


def sigml_to_hamnosys(sigml_xml: str) -> str:
    """Reverse-map a SiGML XML fragment to its HamNoSys PUA string.

    Walks the ``<hamnosys_manual>`` block of the supplied SiGML and
    looks every ``<ham.../>`` tag up in :data:`hamnosys.SYMBOLS`. The
    returned string is the concatenation of those codepoints in
    document order — the same shape ``rendering.hamnosys_to_sigml.to_sigml``
    consumes, so a round-trip is loss-free.

    Returns ``""`` when no manual tags are found. Unknown tags are
    skipped with a warning rather than aborting; the caller can then
    re-validate the resulting HamNoSys to catch any structural issues
    introduced by the dropped tags.
    """
    name_to_cp = _name_to_codepoint()
    chars: list[str] = []
    # Restrict to <hamnosys_manual> ... </hamnosys_manual> if present;
    # otherwise scan the whole fragment.
    manual_match = re.search(
        r"<\s*hamnosys_manual\s*>(.*?)<\s*/\s*hamnosys_manual\s*>",
        sigml_xml,
        re.DOTALL | re.IGNORECASE,
    )
    body = manual_match.group(1) if manual_match else sigml_xml
    # Match self-closing <ham.../> first, then opening tags <ham...>.
    for match in re.finditer(r"<\s*(ham[a-z0-9_]+)\s*/?\s*>", body, re.IGNORECASE):
        name = match.group(1).lower()
        cp = name_to_cp.get(name)
        if cp is None:
            logger.debug("sigml_to_hamnosys: unknown tag <%s/> — skipping", name)
            continue
        chars.append(chr(cp))
    return "".join(chars)


_SIGML_DIRECT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "sigml_direct_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sigml_xml", "rationale"],
            "properties": {
                "sigml_xml": {
                    "type": "string",
                    "description": (
                        "A complete <hns_sign gloss=\"...\">...</hns_sign> "
                        "SiGML XML fragment (no XML declaration, no <sigml> "
                        "wrapper). Use only the <hamnosys_manual> and "
                        "optional <hamnosys_nonmanual> children, with "
                        "self-closing <ham*/> tags exactly as in the "
                        "examples. Tag names must be canonical HamNoSys "
                        "short names like <hamflathand/>, <hamextfingeru/>, "
                        "<hamhead/>, <hammoveo/>."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "One sentence explaining the dominant choices "
                        "(handshape, location, movement) so a reviewer can "
                        "audit the model's reasoning quickly."
                    ),
                },
            },
        },
    },
}


def _build_prompt(
    *,
    parameters: PartialSignParameters,
    prose: str,
    gloss: str,
    sign_language: str,
    previous_errors: list[str],
) -> tuple[str, str]:
    """Render the system + user message pair for the SiGML-direct call."""
    meta = load_prompt(PROMPT_ID_SIGML_DIRECT)
    system = meta.render(sign_language=sign_language or "bsl")

    examples = few_shot_examples(gloss=gloss, n=6)
    examples_block = render_few_shot(examples) if examples else "(no examples available)"

    user_payload = {
        "gloss": gloss or "(unspecified)",
        "sign_language": sign_language or "bsl",
        "description": prose or "",
        "partial_parameters": parameters.model_dump(mode="json", exclude_none=True),
        "previous_errors": previous_errors[:5],
    }
    user = (
        "Reference BSL signs (use the same XML shape — self-closing "
        "<ham*/> tags inside <hamnosys_manual>):\n\n"
        f"{examples_block}\n\n"
        "Now produce SiGML for the contributor's sign described below.\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )
    return system, user


# Number of times we retry the LLM with the explicit failure reason
# fed back in as a previous_error. One retry is usually enough — most
# shape failures (object-literal leak, unknown ham tag, missing manual
# block) recover when the prompt sees its own previous bad output.
_SIGML_DIRECT_MAX_ATTEMPTS = 2


def generate_sigml_direct(
    *,
    parameters: PartialSignParameters,
    client: LLMClient,
    request_id: str,
    prose: str = "",
    gloss: str = "",
    sign_language: str = "bsl",
    previous_errors: list[str] | None = None,
    reasoning_effort: str = "medium",
) -> tuple[str, str, str]:
    """Ask the LLM for a complete SiGML fragment for the sign.

    Returns ``(hamnosys, sigml_fragment, rationale)``. ``hamnosys`` is
    the PUA string recovered from the SiGML by
    :func:`sigml_to_hamnosys` — empty string on failure. ``sigml_fragment``
    is the validated ``<hns_sign>...</hns_sign>`` (without ``<?xml`` /
    ``<sigml>`` wrapping); empty on failure. ``rationale`` is human-
    readable diagnostic text suitable for the chat panel — the
    model's one-line explanation on success, or the failure reason on
    error.

    Self-correction loop
    --------------------
    On the first attempt, validation failures (object-literal leak,
    invalid ham tag, missing manual block, validator-reject HamNoSys)
    are fed back into ``previous_errors`` and the model is re-prompted
    once. The retry typically succeeds because the prompt now contains
    the failure reason verbatim — the model treats it as a constraint
    violation to repair rather than a fresh request.
    """
    errors_list = list(previous_errors or [])
    last_failure_reason = ""
    for attempt in range(_SIGML_DIRECT_MAX_ATTEMPTS):
        ham, sigml_xml, rationale = _generate_sigml_direct_once(
            parameters=parameters,
            client=client,
            request_id=f"{request_id}:attempt{attempt}",
            prose=prose,
            gloss=gloss,
            sign_language=sign_language,
            previous_errors=errors_list,
            reasoning_effort=reasoning_effort,
        )
        if ham:
            return ham, sigml_xml, rationale
        last_failure_reason = rationale or "unknown failure"
        # Feed the failure verbatim into the next attempt's prompt so
        # the model sees the exact constraint to satisfy.
        errors_list = errors_list + [
            f"previous attempt #{attempt + 1} failed: {last_failure_reason}"
        ]
        logger.info(
            "sigml-direct attempt %d failed (%s) — retrying with "
            "explicit error context",
            attempt + 1,
            last_failure_reason,
        )
    return "", "", last_failure_reason


def _generate_sigml_direct_once(
    *,
    parameters: PartialSignParameters,
    client: LLMClient,
    request_id: str,
    prose: str,
    gloss: str,
    sign_language: str,
    previous_errors: list[str],
    reasoning_effort: str,
) -> tuple[str, str, str]:
    """Single LLM round — see :func:`generate_sigml_direct` for the
    retry loop wrapper. Returns the same tuple shape."""
    system, user = _build_prompt(
        parameters=parameters,
        prose=prose,
        gloss=gloss or "",
        sign_language=sign_language or "bsl",
        previous_errors=previous_errors,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        result: ChatResult = client.chat(
            messages=messages,
            response_format=_SIGML_DIRECT_SCHEMA,
            temperature=0.2,
            # Generous visible-output budget — the SiGML response
            # plus rationale can run a few hundred tokens, but the
            # reasoning model also benefits from headroom so its
            # internal chain-of-thought isn't squeezed against the
            # ceiling. The Responses API path bumps this further
            # with a 32k-token reasoning floor (see llm/client.py).
            max_tokens=8192,
            request_id=f"{request_id}:sigml_direct",
            prompt_metadata=load_prompt(PROMPT_ID_SIGML_DIRECT).metadata,
            reasoning_effort=reasoning_effort,
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning("SiGML-direct LLM call failed: %s", exc)
        return "", "", f"llm call failed: {type(exc).__name__}: {exc}"

    content = (result.content or "").strip()
    if not content:
        return "", "", "llm returned empty content"

    payload: dict[str, Any]
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        # Reasoning models occasionally wrap JSON in ```json fences
        # under the strict-format contract; strip and retry once.
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return "", "", "llm returned invalid json"

    sigml_fragment = (payload.get("sigml_xml") or "").strip()
    rationale = (payload.get("rationale") or "").strip()
    if not sigml_fragment:
        return "", "", "llm returned empty sigml_xml"

    # The Hamburg renderer's grammar treats a literal "[object Object]"
    # as a parse error ("Ham4HMLGen.g: node from line 0:0 mismatched
    # input '[object Object]' expecting MICFG2"). The model occasionally
    # leaks this when its prompt context contained an unfilled JS object
    # placeholder. Reject the fragment up front so the caller can
    # repair (typically by re-running with a fresh prompt) instead of
    # shipping broken SiGML to CWASA.
    if "[object Object]" in sigml_fragment:
        return "", "", "llm sigml contains '[object Object]' literal — repair needed"

    # Strip any accidental <?xml ...?> declaration or outer <sigml>
    # wrapper so the fragment slots straight into to_sigml's output.
    sigml_fragment = re.sub(
        r"^<\?xml[^>]*\?>\s*", "", sigml_fragment
    )
    sigml_fragment = re.sub(
        r"^<\s*sigml\s*>\s*", "", sigml_fragment
    )
    sigml_fragment = re.sub(
        r"\s*<\s*/\s*sigml\s*>\s*$", "", sigml_fragment
    )

    # Reverse-map to HamNoSys PUA so the rest of the pipeline (storage,
    # display, validator) gets its expected shape.
    hamnosys = sigml_to_hamnosys(sigml_fragment)
    if not hamnosys:
        return "", "", "could not extract any HamNoSys codepoints from sigml"

    normalized = normalize(hamnosys)
    vr = validate(normalized)
    if not vr.ok:
        return "", "", f"validator rejected reverse-mapped HamNoSys: {vr.summary()}"

    # Phonological completeness check: the HamNoSys grammar accepts
    # signs missing a palm-orientation tag, but CWASA's renderer needs
    # one to compute the hand pose — without it the player throws
    # ``TypeError: Cannot read properties of null (reading 'getName')``
    # at play time. The user sees a SiGML that looks fine and a preview
    # that crashes silently. Catch it here so the retry loop can ask
    # the model to add the missing slot.
    missing = _missing_required_slots(normalized)
    if missing:
        return "", "", (
            "missing required HamNoSys slot(s): " + ", ".join(missing) +
            ". CWASA needs each of [handshape, ext_finger, palm, location] "
            "before any movement tag — re-emit the SiGML with all four "
            "present, in canonical order."
        )

    # Re-emit canonical SiGML through to_sigml so the renderer always
    # gets a DTD-clean document. We trust to_sigml's output over the
    # raw model fragment because to_sigml validates against the DTD;
    # the model-produced fragment may use shorthand that the renderer
    # doesn't accept.
    try:
        canonical_sigml = to_sigml(
            normalized, gloss=gloss or "SIGN"
        )
    except (HamNoSysConversionError, SigmlValidationError, ValueError) as exc:
        # Fall back to validating the model's raw fragment; if that
        # is itself DTD-clean we still surface the HamNoSys for the
        # reviewer.
        ok, dtd_errors = validate_sigml(
            f"<?xml version=\"1.0\" ?><sigml>{sigml_fragment}</sigml>"
        )
        if not ok:
            return "", "", (
                f"reverse SiGML failed canonical re-emit: {exc}; "
                f"model fragment also invalid: {dtd_errors[0] if dtd_errors else 'unknown'}"
            )
        canonical_sigml = f"<sigml>{sigml_fragment}</sigml>"

    return normalized, canonical_sigml, rationale or "ok"


__all__ = [
    "PROMPT_ID_SIGML_DIRECT",
    "generate_sigml_direct",
    "sigml_to_hamnosys",
]
