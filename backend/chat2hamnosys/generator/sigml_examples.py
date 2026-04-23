"""BSL SiGML few-shot example loader.

The SiGML-direct LLM path leans on a handful of known-good signs from
the BSL corpus (``data/hamnosys_bsl_version1.sigml``) to anchor the
model's output. BSL is the highest-quality corpus we ship — every sign
has been validated against the SiGML DTD, so each example is
guaranteed to round-trip through ``rendering.hamnosys_to_sigml.to_sigml``.

Selection
---------
:func:`few_shot_examples` returns up to ``n`` examples chosen by:

1. Lexical match against the gloss when one is supplied (e.g.
   ``gloss="ELECTRON"`` prefers signs whose gloss contains ``electron``,
   ``electric``, etc., before falling back to neighbours).
2. A diversity heuristic to avoid picking, say, three near-duplicate
   variants of ``abroad`` — we sample one example per gloss prefix
   bucket so the model sees a range of handshapes/locations rather than
   a tight cluster.
3. A deterministic fallback set of common everyday signs (``hello``,
   ``yes``, ``no``, ``thank-you``) so the response is reproducible
   when no gloss is provided.

Caching
-------
The corpus is parsed once at import time and held in module memory.
~22k lines parses to ~881 entries in <100 ms with lxml; cheaper than
re-reading per request. The serialised string for each entry is the
``<hns_sign>...</hns_sign>`` fragment with a stable indent.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lxml import etree


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_BSL_PATH = _REPO_ROOT / "data" / "hamnosys_bsl_version1.sigml"


@dataclass(frozen=True)
class SigmlExample:
    """One BSL sign as a few-shot example."""

    gloss: str
    sigml_fragment: str  # <hns_sign>...</hns_sign> serialised


_CACHE: list[SigmlExample] | None = None


def _normalize_gloss(gloss: str) -> str:
    """Lowercase, strip variant tags (``(a)``, ``#1``) for matching."""
    base = re.sub(r"\([^)]*\)|#\d+", "", gloss).strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", base).strip()


def _load_corpus() -> list[SigmlExample]:
    """Parse the BSL SiGML file into per-sign examples.

    On any parse error we log and return an empty list so the
    SiGML-direct path degrades to "no few-shot" rather than crashing
    the contribute flow.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not _BSL_PATH.exists():
        logger.warning("BSL SiGML corpus not found at %s; few-shot disabled", _BSL_PATH)
        _CACHE = []
        return _CACHE
    try:
        tree = etree.parse(str(_BSL_PATH))
    except etree.XMLSyntaxError as exc:
        logger.warning("BSL SiGML corpus failed to parse (%s); few-shot disabled", exc)
        _CACHE = []
        return _CACHE
    out: list[SigmlExample] = []
    root = tree.getroot()
    for sign in root.iter("hns_sign"):
        gloss = sign.get("gloss") or ""
        if not gloss:
            continue
        # Serialise the single sign with pretty indenting; trim outer
        # whitespace so the prompt stays compact.
        fragment = etree.tostring(
            sign, pretty_print=True, encoding="unicode"
        ).strip()
        out.append(SigmlExample(gloss=gloss, sigml_fragment=fragment))
    _CACHE = out
    logger.info("Loaded %d BSL SiGML few-shot examples from %s", len(out), _BSL_PATH)
    return _CACHE


_FALLBACK_GLOSSES = (
    "hello",
    "yes",
    "no",
    "thankyou",
    "thank you",
    "good",
    "house",
    "water",
    "name",
    "home",
)


def _score(example: SigmlExample, target_terms: list[str]) -> int:
    if not target_terms:
        return 0
    gloss_norm = _normalize_gloss(example.gloss)
    return sum(1 for t in target_terms if t and t in gloss_norm)


def few_shot_examples(
    *,
    gloss: str = "",
    n: int = 6,
    seed: int | None = None,
) -> list[SigmlExample]:
    """Pick up to ``n`` BSL SiGML examples for the SiGML-direct prompt.

    ``gloss`` is the contributor's working gloss for the sign being
    authored. We prefer corpus entries whose gloss shares a substring
    with the target after normalization (e.g. ``GREETING`` matches
    ``greet``, ``greeting``); if no useful matches turn up we fill the
    rest with diverse signs from a stable seed.
    """
    corpus = _load_corpus()
    if not corpus:
        return []
    target_terms = [t for t in _normalize_gloss(gloss).split() if t]

    # Bucket by gloss-stem to keep variants of the same sign from
    # crowding out variety.
    seen_stems: set[str] = set()
    rng = random.Random(seed if seed is not None else (hash(gloss or "_") & 0xFFFFFFFF))

    scored = sorted(
        corpus,
        key=lambda ex: (-_score(ex, target_terms), rng.random()),
    )
    picked: list[SigmlExample] = []
    for ex in scored:
        stem = _normalize_gloss(ex.gloss).split(" ", 1)[0]
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        picked.append(ex)
        if len(picked) >= n:
            break

    if len(picked) < n:
        # Top up with the deterministic fallback set if available.
        gloss_index: dict[str, SigmlExample] = {}
        for ex in corpus:
            stem = _normalize_gloss(ex.gloss).split(" ", 1)[0]
            gloss_index.setdefault(stem, ex)
        for fallback in _FALLBACK_GLOSSES:
            stem = fallback.split(" ", 1)[0]
            if stem in seen_stems:
                continue
            ex = gloss_index.get(stem)
            if ex is None:
                continue
            picked.append(ex)
            seen_stems.add(stem)
            if len(picked) >= n:
                break

    return picked


def render_few_shot(examples: Iterable[SigmlExample]) -> str:
    """Render a list of examples as a single text block for the prompt."""
    chunks = []
    for i, ex in enumerate(examples, 1):
        chunks.append(f"Example {i} — gloss={ex.gloss!r}:\n{ex.sigml_fragment}")
    return "\n\n".join(chunks)


__all__ = ["SigmlExample", "few_shot_examples", "render_few_shot"]
