"""Deterministic SiGML-level corrections — no LLM needed.

The correction interpreter in :mod:`.correction_interpreter` requires
an LLM to classify free-form corrections into a parameter diff. Two
classes of correction don't need that round-trip:

1. **Chip-swap from the annotated SiGML editor.** The contributor
   clicked a ``<ham*/>`` tag in the chip strip and picked an
   alternative from the same category. The desired edit is fully
   structured: ``swap <hamfist/> at index N → <hamflathand/>``.
2. **Common natural-language directional changes.** Phrases like
   "make the palm face downward instead" or "use a flat hand instead"
   uniquely determine a tag-level swap when the current SiGML carries
   exactly one tag in the targeted category.

For both, this module rewrites ``session.draft.sigml`` (and the
matching codepoint in ``session.draft.hamnosys``) directly, then
emits a ``CorrectionAppliedEvent`` + ``GeneratedEvent(success=True)``
so the SSE channel delivers a ``generated`` frame and the frontend
re-renders the avatar — same shape as a successful regeneration, but
with no LLM round-trip and no parameter-level reasoning.

The handlers run *before* the LLM-backed interpreter in
:func:`.correction_interpreter.interpret_correction`'s caller chain;
on a miss they fall through to the LLM path so anything ambiguous
still gets the full reasoning treatment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from generator.sigml_reference import BY_CATEGORY, BY_NAME
from session.orchestrator import Correction
from session.state import (
    AuthoringSession,
    CorrectionAppliedEvent,
    GeneratedEvent,
    SessionState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagSwap:
    """A single SiGML tag swap. ``index`` is the zero-based occurrence
    of ``from_tag`` inside ``<hamnosys_manual>``; ``None`` means "first
    occurrence" (the common case for a category that only appears
    once)."""

    from_tag: str
    to_tag: str
    index: Optional[int] = None


# ---------------------------------------------------------------------------
# SiGML rewriting
# ---------------------------------------------------------------------------


_MANUAL_BLOCK_RE = re.compile(
    r"(<\s*hamnosys_manual\s*>)([\s\S]*?)(<\s*/\s*hamnosys_manual\s*>)",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<\s*(ham[a-z0-9_]+)\s*/?\s*>", re.IGNORECASE)


def extract_manual_tags(sigml: str) -> list[str]:
    """Return the ``ham*`` tag names inside ``<hamnosys_manual>``, in order.

    Returns ``[]`` if the SiGML has no manual block — the caller can
    treat that as "nothing to swap".
    """
    if not sigml or not isinstance(sigml, str):
        return []
    m = _MANUAL_BLOCK_RE.search(sigml)
    if not m:
        return []
    return [match.group(1).lower() for match in _TAG_RE.finditer(m.group(2))]


def apply_tag_swap_to_sigml(sigml: str, swap: TagSwap) -> str:
    """Return a new SiGML string with ``swap`` applied.

    Raises :class:`ValueError` if the from-tag is not present at the
    requested index. The caller is expected to validate the swap
    against ``extract_manual_tags`` before calling so failures are
    user-actionable rather than 500s.
    """
    if not isinstance(sigml, str) or not sigml:
        raise ValueError("sigml must be a non-empty string")
    m = _MANUAL_BLOCK_RE.search(sigml)
    if not m:
        raise ValueError("SiGML has no <hamnosys_manual> block")
    before = sigml[: m.start() + len(m.group(1))]
    body = m.group(2)
    after = sigml[m.start() + len(m.group(1)) + len(m.group(2)) :]
    target_tag = swap.from_tag.lower()
    target_index = swap.index
    seen = 0
    new_body = ""
    cursor = 0
    rewrote = False
    for match in _TAG_RE.finditer(body):
        new_body += body[cursor : match.start()]
        cursor = match.end()
        tag_name = match.group(1).lower()
        if tag_name == target_tag and (target_index is None or seen == target_index):
            new_body += f"<{swap.to_tag}/>"
            rewrote = True
            # Once-only when index is None (first hit); for an explicit
            # index we also bail after the hit so a duplicate tag at a
            # later index isn't accidentally swapped.
            cursor_tail = body[cursor:]
            new_body += cursor_tail
            cursor = len(body)
            break
        if tag_name == target_tag:
            seen += 1
        new_body += match.group(0)
    if cursor < len(body):
        new_body += body[cursor:]
    if not rewrote:
        raise ValueError(
            f"tag {swap.from_tag!r} not found at index "
            f"{swap.index if swap.index is not None else 0}"
        )
    return before + new_body + after


def apply_tag_swap_to_hamnosys(hamnosys: Optional[str], swap: TagSwap) -> Optional[str]:
    """Mirror the swap into the HamNoSys PUA codepoint string.

    Tags map to specific codepoints via :data:`BY_NAME`. When a tag is
    not in the catalog (a custom SiGML tag, or a one-off non-PUA tag)
    we return ``hamnosys`` unchanged — the SiGML is the source of
    truth; the codepoint string just has to match the *first*
    occurrence on a best-effort basis. ``None`` in / ``None`` out so
    drafts that never had a hamnosys string stay coherent.
    """
    if not hamnosys:
        return hamnosys
    from_entry = BY_NAME.get(swap.from_tag.lower())
    to_entry = BY_NAME.get(swap.to_tag.lower())
    if not from_entry or not to_entry:
        return hamnosys
    from_cp = from_entry.get("codepoint")
    to_cp = to_entry.get("codepoint")
    if from_cp is None or to_cp is None:
        return hamnosys
    from_ch = chr(from_cp)
    to_ch = chr(to_cp)
    # Replace only the first occurrence so a sign that legitimately
    # carries the same codepoint twice (rare but possible) doesn't get
    # both clobbered when the user only swapped one chip.
    if from_ch in hamnosys:
        return hamnosys.replace(from_ch, to_ch, 1)
    return hamnosys


# ---------------------------------------------------------------------------
# Natural-language matchers — return a TagSwap or None
# ---------------------------------------------------------------------------


# Direction keywords → palm_direction tag suffix.
# The eight palm directions live in :data:`BY_CATEGORY['palm_direction']`
# (hampalmu, hampalmur, hampalmr, hampalmdr, hampalmd, hampalmdl,
# hampalml, hampalmul). We map the contributor's plain-English
# direction to the closest cardinal/diagonal — diagonals get a fallback
# to the cardinal axis when the user didn't qualify.
_PALM_DIRECTION_KEYWORDS: dict[str, str] = {
    "down":      "hampalmd",
    "downward":  "hampalmd",
    "downwards": "hampalmd",
    "up":        "hampalmu",
    "upward":    "hampalmu",
    "upwards":   "hampalmu",
    "left":      "hampalml",
    "right":     "hampalmr",
    "in":        "hampalml",   # toward signer ≈ left in BSL convention
    "out":       "hampalmr",   # away from signer ≈ right
    "inward":    "hampalml",
    "outward":   "hampalmr",
}


# "palm" mention is required to disambiguate from the ext_finger
# direction (which uses "fingers point ..." phrasing).
_PALM_INTENT_RE = re.compile(
    r"\bpalm(?:s)?\b",
    re.IGNORECASE,
)
_DIRECTION_TOKEN_RE = re.compile(
    r"\b(downwards|downward|down|upwards|upward|up|left|right|inward|outward|in|out)\b",
    re.IGNORECASE,
)


def _match_palm_direction_swap(text: str, current_sigml: str) -> Optional[TagSwap]:
    """Detect "make the palm face X" / "palm should be X" requests.

    Returns ``None`` on miss so the caller falls through to the LLM
    path. On hit the from-tag is whichever palm_direction tag the
    current SiGML carries; the to-tag is the catalog tag matching the
    direction keyword.
    """
    if not _PALM_INTENT_RE.search(text):
        return None
    match = _DIRECTION_TOKEN_RE.search(text)
    if not match:
        return None
    direction_token = match.group(1).lower()
    target_tag = _PALM_DIRECTION_KEYWORDS.get(direction_token)
    if not target_tag:
        return None
    # Find the existing palm_direction tag in the SiGML.
    palm_tags = {e["name"] for e in BY_CATEGORY.get("palm_direction", [])}
    tags = extract_manual_tags(current_sigml)
    current_palm = next((t for t in tags if t in palm_tags), None)
    if not current_palm:
        return None
    if current_palm == target_tag:
        # Already the requested direction — nothing to swap. The LLM
        # path can ask the user to clarify.
        return None
    return TagSwap(from_tag=current_palm, to_tag=target_tag)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


# Verb cues that disambiguate replacement vs. addition intent. The
# bracketed/bare HamNoSys-tag matcher fires on replacement intent
# only — "add/include/also/insert" indicates the user wants to
# coexist a tag with whatever is already in the same slot, which is a
# parameter-level edit best left to the LLM interpreter.
_ADD_VERB_RE = re.compile(
    r"\b(add|insert|include|also|append|alongside|coexist)\b",
    re.IGNORECASE,
)
_SWAP_VERB_RE = re.compile(
    r"(\b(change|swap|replace|use|make|set|switch|should\s+be|instead\s+of)\b|->|→)",
    re.IGNORECASE,
)


# Bracketed-form HamNoSys tag references — what the chip strip
# displays and what users tend to copy/paste back into the chat
# ("change <hampalmr/> to <hampalmd/>"). The trailing slash and inner
# whitespace are optional.
_HAM_BRACKET_TAG_RE = re.compile(
    r"<\s*(ham[a-z0-9_]+)\s*/?\s*>",
    re.IGNORECASE,
)
# Bare-word references — "use hampalmd instead". Looser than the
# bracketed form, so we additionally require the matched token to be
# a known catalog name before treating it as a tag reference.
_HAM_BARE_TAG_RE = re.compile(
    r"\b(ham[a-z0-9_]+)\b",
    re.IGNORECASE,
)


def _extract_ham_tag_refs(text: str) -> list[str]:
    """Return ham-tag names referenced in ``text``, in order of appearance.

    Bracketed references are gathered first because they're the form
    the chip strip displays. Bare-word references are only kept if
    they match a real catalog entry — that filter avoids false
    positives on stray strings that happen to start with "ham".
    """
    refs = [m.group(1).lower() for m in _HAM_BRACKET_TAG_RE.finditer(text)]
    if refs:
        return refs
    return [
        m.group(1).lower()
        for m in _HAM_BARE_TAG_RE.finditer(text)
        if m.group(1).lower() in BY_NAME
    ]


def _match_tag_syntax_swap(text: str, current_sigml: str) -> Optional[TagSwap]:
    """Match corrections that name HamNoSys tags directly.

    Examples this catches:

    - ``change <hampalmr/> to <hampalmd/>``
    - ``swap <hamfist/> for <hamflathand/>``
    - ``use <hampalmd/> instead``
    - ``<hampalml/> → <hampalmd/>``
    - ``make it <hampalmd/>``

    Strategy: identify the TARGET tag (the last ham-tag mentioned —
    every "change X to Y", "swap X for Y", "make it Y" phrasing puts
    the target last), look up its category in :data:`BY_NAME`, and
    find the current sign's tag in that same category. Emit a swap
    from the actual current tag to the target.

    Robust to the user naming a non-existent source tag — the chip
    strip they were reading from might disagree with what they typed.
    Production case that motivated this matcher: the user wrote
    ``change <hampalmr/> to <hampalmd/>`` while the sign carried
    ``<hampalml/>``; the LLM interpreter classified that as VAGUE
    because old_value couldn't be matched against current parameters,
    and the user got "did NOT change sigml/hamnosys" with no
    animation update. We trust the target intent and let the source
    claim be advisory.

    Returns ``None`` if no recognisable tag is found, the target tag
    isn't in the catalog, the current sign has no tag in that target
    category to swap (correction would be ELABORATE, not APPLY_DIFF),
    or the target tag is already the current tag (no-op).
    """
    if not text or not current_sigml:
        return None
    refs = _extract_ham_tag_refs(text)
    if not refs:
        return None
    # Verbs of intent. "change/swap/replace/use/make/should/->" ⇒ swap.
    # "add/include/also/insert" is intentionally NOT covered — adding
    # a tag in a category that already has one is ambiguous (replace
    # vs. coexist) and the LLM's parameter-level reasoning handles
    # additions better than a deterministic same-category swap.
    if _ADD_VERB_RE.search(text) and not _SWAP_VERB_RE.search(text):
        return None
    target_tag = refs[-1]
    target_entry = BY_NAME.get(target_tag)
    if not target_entry:
        return None
    target_category = target_entry.get("category")
    if not target_category:
        return None
    cat_tag_names = {e["name"] for e in BY_CATEGORY.get(target_category, [])}
    current_tags = extract_manual_tags(current_sigml)
    current_in_cat = next((t for t in current_tags if t in cat_tag_names), None)
    if not current_in_cat:
        return None
    if current_in_cat == target_tag:
        return None
    return TagSwap(from_tag=current_in_cat, to_tag=target_tag)


def match_deterministic_swap(
    correction: Correction,
    current_sigml: Optional[str],
) -> Optional[TagSwap]:
    """Attempt to interpret ``correction`` as a deterministic SiGML swap.

    Three routes, in order:

    - ``correction.target_region == "swap:from:to"`` — structured
      chip-swap payload encoded by the frontend's
      ``contribute-sigml-edit.js``. ``from`` and ``to`` are tag names
      (without angle brackets). An optional ``:index`` suffix targets
      a specific occurrence.
    - ``correction.raw_text`` references HamNoSys tags directly
      (``<hampalmd/>``, ``hampalmd``, …). The target tag's category
      identifies which slot to swap; the user's named source is
      advisory (a wrong source tag like ``<hampalmr/>`` when the sign
      carries ``<hampalml/>`` still produces the correct swap).
    - ``correction.raw_text`` matches a natural-language directional
      pattern (currently: palm-direction in plain English).

    Returns ``None`` on miss — caller should fall through to the
    LLM-backed interpreter.
    """
    if not current_sigml:
        return None
    region = (correction.target_region or "").strip()
    if region.startswith("swap:"):
        return _parse_structured_swap(region, current_sigml)
    text = (correction.raw_text or "").strip()
    if not text:
        return None
    swap = _match_tag_syntax_swap(text, current_sigml)
    if swap is not None:
        return swap
    return _match_palm_direction_swap(text, current_sigml)


def _parse_structured_swap(region: str, current_sigml: str) -> Optional[TagSwap]:
    """Parse a ``"swap:<from>:<to>[:<index>]"`` target_region payload.

    Validates that ``from`` actually appears in the SiGML (at the
    requested index when given). Returns ``None`` on any malformed
    payload so the caller can downgrade gracefully.
    """
    parts = region.split(":")
    if len(parts) < 3:
        return None
    _, from_tag, to_tag, *rest = parts
    from_tag = from_tag.strip().lower()
    to_tag = to_tag.strip().lower()
    if not from_tag.startswith("ham") or not to_tag.startswith("ham"):
        return None
    idx: Optional[int] = None
    if rest:
        try:
            idx = int(rest[0])
        except (TypeError, ValueError):
            idx = None
        if idx is not None and idx < 0:
            idx = None
    tags = extract_manual_tags(current_sigml)
    matching = [i for i, t in enumerate(tags) if t == from_tag]
    if not matching:
        return None
    if idx is not None and idx not in matching:
        # Caller asked for an occurrence that doesn't exist; fall back
        # to the first one rather than 4xx-ing the contributor.
        idx = None
    return TagSwap(from_tag=from_tag, to_tag=to_tag, index=idx)


# ---------------------------------------------------------------------------
# Session-level apply
# ---------------------------------------------------------------------------


def apply_swap_to_session(
    session: AuthoringSession,
    swap: TagSwap,
) -> AuthoringSession:
    """Rewrite ``session.draft.sigml`` (+ hamnosys) and emit events.

    Expects ``session.state == APPLYING_CORRECTION`` (set by
    ``on_correction``). Appends a ``CorrectionAppliedEvent`` recording
    the swap, then a ``GeneratedEvent(success=True)`` carrying the new
    SiGML/HamNoSys so the SSE channel delivers a ``generated`` frame
    the frontend re-renders on. Transitions the session to
    ``RENDERED``. Mirrors the success path of
    :func:`session.orchestrator.run_generation` but skips the
    LLM-backed generator pass.
    """
    if session.state is not SessionState.APPLYING_CORRECTION:
        raise RuntimeError(
            f"apply_swap_to_session requires state APPLYING_CORRECTION, "
            f"got {session.state.value!r}"
        )
    current_sigml = session.draft.sigml or ""
    new_sigml = apply_tag_swap_to_sigml(current_sigml, swap)
    new_hamnosys = apply_tag_swap_to_hamnosys(session.draft.hamnosys, swap)

    summary = f"chip swap: <{swap.from_tag}/> → <{swap.to_tag}/>"
    field_change = {
        "path": "sigml.tag_swap",
        "old_value": swap.from_tag,
        "new_value": swap.to_tag,
        "confidence": 1.0,
        "index": swap.index if swap.index is not None else 0,
    }
    session = session.append_event(
        CorrectionAppliedEvent(summary=summary, field_changes=[field_change])
    )
    session = session.append_event(
        GeneratedEvent(
            success=True,
            hamnosys=new_hamnosys or "",
            sigml=new_sigml,
            confidence=1.0,
            used_llm_fallback=False,
        )
    )
    session = session.with_draft(
        sigml=new_sigml,
        hamnosys=new_hamnosys,
        preview_status="ok",
        preview_message="",
        generation_errors=[],
    )
    return session.with_state(SessionState.RENDERED)


__all__ = [
    "TagSwap",
    "apply_swap_to_session",
    "apply_tag_swap_to_hamnosys",
    "apply_tag_swap_to_sigml",
    "extract_manual_tags",
    "match_deterministic_swap",
]
