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

from hamnosys import SYMBOLS, normalize, validate
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
    """
    errors_list = list(previous_errors or [])
    system, user = _build_prompt(
        parameters=parameters,
        prose=prose,
        gloss=gloss or "",
        sign_language=sign_language or "bsl",
        previous_errors=errors_list,
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
            max_tokens=2400,
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
