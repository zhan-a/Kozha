"""Generate .meta.json review-metadata files alongside each data/*.sigml.

Each ``data/<filename>.sigml`` gets a parallel ``data/<filename>.meta.json``
describing per-sign review status. Backfill is conservative: only signs whose
provenance is a Deaf-reviewed academic corpus get ``deaf_native_reviewed:
true``; everything else stays ``false`` with ``review_source: null``.

Run:
    python3 scripts/build_review_metadata.py

Re-running is idempotent — an existing ``signs`` dict in the meta file is
preserved, so manual edits (e.g. a governance board's explicit override on a
specific gloss) are never clobbered. Only the ``default_review`` block and the
sign index are refreshed from the ``.sigml`` source of truth.

Prompt 6 spec: docs/polish/06-review-metadata.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

META_VERSION = 1

# Fixed enum for the ``review_source`` field. Also enforced by the tests.
VALID_REVIEW_SOURCES: set[Optional[str]] = {
    None,
    "corpus_provenance",
    "deaf_reviewer_single",
    "deaf_reviewer_double",
    "governance_board",
}


# ---------------------------------------------------------------------------
# Provenance table — the ground truth behind the backfill.
#
# Only academic corpora whose public documentation states native-signer
# review at the source level get ``deaf_native_reviewed: true``. Community
# repositories where provenance is unclear stay ``false``.
# ---------------------------------------------------------------------------

DEAF_REVIEWED = {
    "deaf_native_reviewed": True,
    "reviewer_count": 1,
    "reviewer_language_match": True,
    "review_source": "corpus_provenance",
    "last_reviewed": None,
    "notes": None,
}

UNREVIEWED = {
    "deaf_native_reviewed": False,
    "reviewer_count": 0,
    "reviewer_language_match": False,
    "review_source": None,
    "last_reviewed": None,
    "notes": "community-contributed, pending review",
}

# language tag (ISO-ish) + human-readable source + default review record
PROVENANCE: dict[str, dict] = {
    # --- Deaf-reviewed corpora ---
    "hamnosys_bsl_version1.sigml": {
        "language": "bsl",
        "source": "DictaSign Corpus (Universitat Hamburg, CC BY-NC-SA 3.0)",
        "source_kind": "corpus",
        "default_review": DEAF_REVIEWED,
    },
    "German_SL_DGS.sigml": {
        "language": "dgs",
        "source": "DGS Lexicon via SignAvatars (Universitat Hamburg)",
        "source_kind": "corpus",
        "default_review": DEAF_REVIEWED,
    },
    "Polish_SL_PJM.sigml": {
        "language": "pjm",
        "source": "PJM Dictionary via SignAvatars (University of Warsaw)",
        "source_kind": "corpus",
        "default_review": DEAF_REVIEWED,
    },
    "Greek_SL_GSL.sigml": {
        "language": "gsl",
        "source": "DictaSign Corpus via SignAvatars (Universitat Hamburg)",
        "source_kind": "corpus",
        "default_review": DEAF_REVIEWED,
    },
    "French_SL_LSF.sigml": {
        "language": "lsf",
        "source": "DictaSign Corpus via SignAvatars (Universitat Hamburg)",
        "source_kind": "corpus",
        "default_review": DEAF_REVIEWED,
    },
    # --- Community sources (no corpus-level Deaf review documented) ---
    "Dutch_SL_NGT.sigml": {
        "language": "ngt",
        "source": "SignLanguageSynthesis by Lyke Esselink (community)",
        "source_kind": "community",
        "default_review": UNREVIEWED,
    },
    "Algerian_SL.sigml": {
        "language": "algerian",
        "source": "algerianSignLanguage-avatar by Taha Zerrouki (community)",
        "source_kind": "community",
        "default_review": UNREVIEWED,
    },
    "Bangla_SL.sigml": {
        "language": "bangla",
        "source": "bdsl-3d-animation by Devr Arif Khan (community)",
        "source_kind": "community",
        "default_review": UNREVIEWED,
    },
    "Indian_SL.sigml": {
        "language": "isl",
        "source": "Text-to-Sign-Language / text_to_isl (community)",
        "source_kind": "community",
        "default_review": UNREVIEWED,
    },
    "Kurdish_SL.sigml": {
        "language": "kurdish",
        "source": "KurdishSignLanguage by KurdishBLARK (community)",
        "source_kind": "community",
        "default_review": UNREVIEWED,
    },
    "Vietnamese_SL.sigml": {
        "language": "vsl",
        "source": "VSL by Raian Rido (community)",
        "source_kind": "community",
        "default_review": UNREVIEWED,
    },
    "Filipino_SL.sigml": {
        "language": "fsl",
        "source": "syntheticfsl / signtyper by Jennie Ablog (community)",
        "source_kind": "community",
        "default_review": UNREVIEWED,
    },
    # --- Fingerspelling alphabets ---
    # Alphabets are standardized community-taught letter shapes. We do not
    # claim corpus-level Deaf review; they're used as fallback only.
    "asl_alphabet_sigml.sigml": {
        "language": "asl",
        "source": "ASL manual alphabet (community-standard fingerspelling)",
        "source_kind": "alphabet",
        "default_review": UNREVIEWED,
    },
    "bsl_alphabet_sigml.sigml": {
        "language": "bsl",
        "source": "BSL manual alphabet (community-standard fingerspelling)",
        "source_kind": "alphabet",
        "default_review": UNREVIEWED,
    },
    "dgs_alphabet_sigml.sigml": {
        "language": "dgs",
        "source": "DGS manual alphabet (community-standard fingerspelling)",
        "source_kind": "alphabet",
        "default_review": UNREVIEWED,
    },
    "lsf_alphabet_sigml.sigml": {
        "language": "lsf",
        "source": "LSF manual alphabet (community-standard fingerspelling)",
        "source_kind": "alphabet",
        "default_review": UNREVIEWED,
    },
    "ngt_alphabet_sigml.sigml": {
        "language": "ngt",
        "source": "NGT manual alphabet (community-standard fingerspelling)",
        "source_kind": "alphabet",
        "default_review": UNREVIEWED,
    },
    "pjm_alphabet_sigml.sigml": {
        "language": "pjm",
        "source": "PJM manual alphabet (community-standard fingerspelling)",
        "source_kind": "alphabet",
        "default_review": UNREVIEWED,
    },
}


_GLOSS_RE = re.compile(r'<hns_sign\s+gloss="([^"]*)"', re.IGNORECASE)


def enumerate_glosses(sigml_path: Path) -> list[str]:
    """Return the lowercased gloss list from a .sigml file.

    Matches the frontend's glossToSign lookup (public/app.html: stripped +
    lowercased). Files in the hamgestural_sign format (Filipino_SL.sigml)
    have no glosses and return an empty list.
    """
    text = sigml_path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    seen: set[str] = set()
    for m in _GLOSS_RE.finditer(text):
        g = m.group(1).strip().lower()
        if not g or g in seen:
            continue
        seen.add(g)
        out.append(g)
    return out


def _load_existing(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_meta_for(sigml_name: str) -> dict:
    if sigml_name not in PROVENANCE:
        # New .sigml file dropped in without a provenance entry — be
        # conservative: treat it as community until a maintainer claims it.
        provenance = {
            "language": "unknown",
            "source": "unknown source — please classify in scripts/build_review_metadata.py",
            "source_kind": "community",
            "default_review": UNREVIEWED,
        }
    else:
        provenance = PROVENANCE[sigml_name]

    sigml_path = DATA_DIR / sigml_name
    glosses = enumerate_glosses(sigml_path)

    meta_path = DATA_DIR / f"{sigml_name}.meta.json"
    existing = _load_existing(meta_path)
    preserved_signs: dict[str, dict] = {}
    if isinstance(existing.get("signs"), dict):
        for k, v in existing["signs"].items():
            if (
                isinstance(v, dict)
                and v.get("review_source") in VALID_REVIEW_SOURCES
            ):
                preserved_signs[str(k).strip().lower()] = v

    # Keep only per-sign entries that correspond to an actual gloss in the
    # file — if a gloss was renamed upstream we don't want to carry dead
    # entries forward.
    gloss_set = set(glosses)
    preserved_signs = {k: v for k, v in preserved_signs.items() if k in gloss_set}

    return {
        "version": META_VERSION,
        "language": provenance["language"],
        "source": provenance["source"],
        "source_kind": provenance["source_kind"],
        "sigml_file": sigml_name,
        "default_review": provenance["default_review"],
        "sign_count": len(glosses),
        "signs": preserved_signs,
    }


def write_meta(sigml_name: str) -> Path:
    meta = build_meta_for(sigml_name)
    out_path = DATA_DIR / f"{sigml_name}.meta.json"
    out_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    sigmls = sorted(p.name for p in DATA_DIR.glob("*.sigml"))
    for name in sigmls:
        out = write_meta(name)
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
