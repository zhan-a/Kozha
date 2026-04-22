"""Load per-sign review metadata from ``data/*.meta.json`` files.

The review-metadata schema is documented in ``docs/polish/06-review-metadata.md``.
Each ``.sigml`` under ``data/`` has a parallel ``.meta.json`` that records the
source provenance and per-sign review status. This module exposes a small
read-only API the FastAPI layer uses to:

1. Look up the review record for a ``(sign_language, gloss)`` pair.
2. Tell the translator whether a gloss is reviewed enough to be served when
   the client passes ``reviewed_only: true``.
3. Summarize coverage numbers (total vs. reviewed vs. unreviewed) for the
   public progress dashboard.

The loader is conservative: a ``.meta.json`` that is missing, malformed, or
missing a sign it would otherwise cover falls back to ``deaf_native_reviewed:
False, review_source: None``. Metadata is additive — the translator continues
to serve every sign it currently serves whether or not this file exists.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

# Whitelisted values for ``review_source``. Exposed for test assertions.
VALID_REVIEW_SOURCES: frozenset[Optional[str]] = frozenset(
    {
        None,
        "corpus_provenance",
        "deaf_reviewer_single",
        "deaf_reviewer_double",
        "governance_board",
    }
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Defaults applied when a sign has no metadata entry (conservative — do not
# claim a review that isn't in the data).
DEFAULT_UNREVIEWED: dict = {
    "deaf_native_reviewed": False,
    "reviewer_count": 0,
    "reviewer_language_match": False,
    "review_source": None,
    "last_reviewed": None,
    "notes": None,
}


@dataclass(frozen=True)
class ReviewRecord:
    """Plain read-only projection of a per-sign metadata entry."""

    deaf_native_reviewed: bool
    reviewer_count: int
    reviewer_language_match: bool
    review_source: Optional[str]
    last_reviewed: Optional[str]
    notes: Optional[str]
    source: Optional[str]
    source_kind: Optional[str]
    in_database: bool = False

    def to_dict(self) -> dict:
        return {
            "deaf_native_reviewed": self.deaf_native_reviewed,
            "reviewer_count": self.reviewer_count,
            "reviewer_language_match": self.reviewer_language_match,
            "review_source": self.review_source,
            "last_reviewed": self.last_reviewed,
            "notes": self.notes,
            "source": self.source,
            "source_kind": self.source_kind,
            "in_database": self.in_database,
        }


_GLOSS_RE = re.compile(r'<hns_sign\s+gloss="([^"]*)"', re.IGNORECASE)


def _enumerate_glosses(sigml_path: Path) -> set[str]:
    """Case-insensitive set of non-empty glosses in a .sigml file."""
    if not sigml_path.exists():
        return set()
    try:
        text = sigml_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    out: set[str] = set()
    for m in _GLOSS_RE.finditer(text):
        g = m.group(1).strip().lower()
        if g:
            out.add(g)
    return out


# Port of public/app.html:glossBase(). The frontend strips POS tags like
# "(n)#1", trailing homophone markers, and normalizes punctuation so
# "fruit(n)#1" and "fruit" both reduce to the same base key. We mirror it
# here so the backend can look up the same sign the frontend resolves to.
_GLOSS_BASE_PATTERNS = [
    re.compile(r"\(.*?\)"),
    re.compile(r"#\d+$"),
    re.compile(r"\d+[a-z]?\^?$"),
    re.compile(r"^_num-"),
    re.compile(r"_\(.*?\)"),
]
_GLOSS_BASE_CLEAN = re.compile(
    r"[^a-z0-9À-ɏͰ-ϿЀ-ӿĀ-ſ]+"
)


def gloss_base(gloss: str) -> str:
    s = (gloss or "").strip().lower()
    for rx in _GLOSS_BASE_PATTERNS:
        s = rx.sub("", s)
    s = _GLOSS_BASE_CLEAN.sub(" ", s)
    return s.strip()


def _coerce_record(
    raw: dict,
    source: str,
    source_kind: str,
    *,
    in_database: bool = False,
) -> ReviewRecord:
    return ReviewRecord(
        deaf_native_reviewed=bool(raw.get("deaf_native_reviewed", False)),
        reviewer_count=int(raw.get("reviewer_count") or 0),
        reviewer_language_match=bool(raw.get("reviewer_language_match", False)),
        review_source=raw.get("review_source"),
        last_reviewed=raw.get("last_reviewed"),
        notes=raw.get("notes"),
        source=source,
        source_kind=source_kind,
        in_database=in_database,
    )


class ReviewMetadataIndex:
    """Merged (language → gloss → ReviewRecord) index across all meta files.

    Multiple ``.meta.json`` files may cover the same sign-language code (for
    example BSL has both the main DictaSign corpus and the alphabet). Later
    loads override the default at the *language* level only when their
    ``default_review`` claims a review source at least as strong as the
    existing one — alphabet files (typically unreviewed) must not downgrade
    the corpus default.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self._lock = Lock()
        # lang → default ReviewRecord
        self._language_defaults: dict[str, ReviewRecord] = {}
        # lang → {gloss_lower: ReviewRecord}
        self._per_sign: dict[str, dict[str, ReviewRecord]] = {}
        # lang → set of lowercased glosses actually present in a .sigml file.
        # Read from the backing .sigml on load so lookups for glosses outside
        # the database don't inherit the corpus default.
        self._known_glosses: dict[str, set[str]] = {}
        # lang → {base_form: canonical_gloss}. Lets the translator resolve
        # ``fruit`` → ``fruit(n)#1`` the same way the frontend does.
        self._base_index: dict[str, dict[str, str]] = {}
        # lang → aggregate coverage stats
        self._summary: dict[str, dict] = {}
        # filenames loaded (for diagnostics)
        self._loaded_files: list[str] = []
        self.reload()

    # -------------------- loading --------------------

    def reload(self) -> None:
        with self._lock:
            self._language_defaults.clear()
            self._per_sign.clear()
            self._known_glosses.clear()
            self._base_index.clear()
            self._summary.clear()
            self._loaded_files.clear()
            if not self._data_dir.exists():
                logger.warning("review-metadata: %s does not exist", self._data_dir)
                return
            for meta_path in sorted(self._data_dir.glob("*.meta.json")):
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "review-metadata: skipping malformed %s: %s",
                        meta_path.name,
                        exc,
                    )
                    continue
                self._ingest(payload, meta_path.name)

    def _ingest(self, payload: dict, filename: str) -> None:
        lang = str(payload.get("language") or "").strip().lower()
        source = str(payload.get("source") or "")
        source_kind = str(payload.get("source_kind") or "unknown")
        if not lang:
            logger.warning("review-metadata: %s missing 'language'", filename)
            return
        default_raw = payload.get("default_review") or {}
        default_record = _coerce_record(default_raw, source, source_kind)

        existing = self._language_defaults.get(lang)
        # Prefer the record whose default claims more review weight. This
        # prevents an alphabet file (always unreviewed) from clobbering the
        # main corpus default for the same sign language.
        if existing is None or _stronger(default_record, existing):
            self._language_defaults[lang] = default_record

        bucket = self._per_sign.setdefault(lang, {})
        signs = payload.get("signs") or {}
        if isinstance(signs, dict):
            for gloss, raw in signs.items():
                if not isinstance(raw, dict):
                    continue
                key = str(gloss).strip().lower()
                if not key:
                    continue
                bucket[key] = _coerce_record(
                    raw, source, source_kind, in_database=True
                )

        sigml_file = payload.get("sigml_file")
        if isinstance(sigml_file, str) and sigml_file:
            known = _enumerate_glosses(self._data_dir / sigml_file)
            self._known_glosses.setdefault(lang, set()).update(known)
            base_idx = self._base_index.setdefault(lang, {})
            for g in known:
                base = gloss_base(g)
                if base and base not in base_idx:
                    base_idx[base] = g

        self._loaded_files.append(filename)
        self._summary.setdefault(
            lang,
            {
                "total": 0,
                "reviewed": 0,
                "unreviewed": 0,
                "sources": [],
            },
        )
        total = int(payload.get("sign_count") or 0)
        self._summary[lang]["total"] += total
        default_reviewed = default_record.deaf_native_reviewed
        overrides_reviewed = sum(
            1
            for raw in (signs.values() if isinstance(signs, dict) else [])
            if isinstance(raw, dict) and raw.get("deaf_native_reviewed")
        )
        overrides_unreviewed = sum(
            1
            for raw in (signs.values() if isinstance(signs, dict) else [])
            if isinstance(raw, dict) and not raw.get("deaf_native_reviewed")
        )
        unspecified = max(0, total - overrides_reviewed - overrides_unreviewed)
        if default_reviewed:
            reviewed = overrides_reviewed + unspecified
            unreviewed = overrides_unreviewed
        else:
            reviewed = overrides_reviewed
            unreviewed = overrides_unreviewed + unspecified
        self._summary[lang]["reviewed"] += reviewed
        self._summary[lang]["unreviewed"] += unreviewed
        if source and source not in self._summary[lang]["sources"]:
            self._summary[lang]["sources"].append(source)

    # -------------------- read API --------------------

    def get_default(self, sign_language: str) -> ReviewRecord:
        lang = (sign_language or "").strip().lower()
        return self._language_defaults.get(
            lang,
            ReviewRecord(
                **DEFAULT_UNREVIEWED, source=None, source_kind=None
            ),
        )

    def get(self, sign_language: str, gloss: str) -> ReviewRecord:
        """Return the merged record for ``gloss`` within ``sign_language``.

        Precedence:

        1. Per-sign override from the meta file → use that (marked
           ``in_database=True``).
        2. Gloss is listed in the backing .sigml file → use the language
           default (marked ``in_database=True``).
        3. Gloss is not in any .sigml file for this language → conservative
           unreviewed fallback (marked ``in_database=False``). This is what
           prevents a made-up word like ``xyzzy`` from inheriting the BSL
           DictaSign corpus's "reviewed=true" default.
        """
        lang = (sign_language or "").strip().lower()
        key = (gloss or "").strip().lower()
        bucket = self._per_sign.get(lang, {})
        if key in bucket:
            return bucket[key]
        known = self._known_glosses.get(lang, set())
        # Try the raw key first, then the frontend's base form (so a lookup
        # for ``fruit`` resolves to the BSL entry ``fruit(n)#1``).
        resolved = key if key in known else None
        if resolved is None:
            base = gloss_base(key)
            if base:
                resolved = self._base_index.get(lang, {}).get(base)
        if resolved is not None:
            if resolved in bucket:
                return bucket[resolved]
            default = self.get_default(lang)
            # The default record is stored with in_database=False (the
            # language-level default record carries no per-sign context);
            # clone it with in_database=True for this gloss.
            return ReviewRecord(
                deaf_native_reviewed=default.deaf_native_reviewed,
                reviewer_count=default.reviewer_count,
                reviewer_language_match=default.reviewer_language_match,
                review_source=default.review_source,
                last_reviewed=default.last_reviewed,
                notes=default.notes,
                source=default.source,
                source_kind=default.source_kind,
                in_database=True,
            )
        # Unknown gloss — conservative fallback. ``source`` is None so the
        # UI can distinguish "no known source" from "not yet reviewed".
        return ReviewRecord(
            **DEFAULT_UNREVIEWED,
            source=None,
            source_kind=None,
            in_database=False,
        )

    def is_reviewed(self, sign_language: str, gloss: str) -> bool:
        return self.get(sign_language, gloss).deaf_native_reviewed

    def known_glosses(self, sign_language: str) -> set[str]:
        lang = (sign_language or "").strip().lower()
        return set(self._known_glosses.get(lang, set()))

    def language_summary(self, sign_language: str) -> dict:
        lang = (sign_language or "").strip().lower()
        default = self.get_default(lang).to_dict()
        summary = dict(self._summary.get(lang, {"total": 0, "reviewed": 0, "unreviewed": 0, "sources": []}))
        summary["default_review"] = default
        return summary

    def all_summaries(self) -> dict[str, dict]:
        return {lang: self.language_summary(lang) for lang in sorted(self._summary)}

    def known_languages(self) -> list[str]:
        return sorted(self._language_defaults.keys())

    def record_update(
        self,
        *,
        sign_language: str,
        gloss: str,
        record: ReviewRecord,
    ) -> None:
        """In-memory update. Used by the chat2hamnosys export hook so tests
        can observe the new state without re-reading the .meta.json file.
        Persistence is handled separately (see ``write_override``).
        """
        lang = (sign_language or "").strip().lower()
        key = (gloss or "").strip().lower()
        if not lang or not key:
            return
        with self._lock:
            self._per_sign.setdefault(lang, {})[key] = record

    def write_override(
        self,
        *,
        sigml_filename: str,
        gloss: str,
        record_dict: dict,
    ) -> Path:
        """Persist a per-sign override into ``data/<sigml>.meta.json``.

        Used by the chat2hamnosys review pipeline when a sign is exported
        into the live library with reviewer-match metadata. Creates the
        meta file if it doesn't exist; merges with the existing ``signs``
        dict otherwise.
        """
        meta_path = self._data_dir / f"{sigml_filename}.meta.json"
        payload: dict
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "version": 1,
                "language": "unknown",
                "source": "unknown",
                "source_kind": "authored",
                "sigml_file": sigml_filename,
                "default_review": DEFAULT_UNREVIEWED,
                "sign_count": 0,
                "signs": {},
            }
        signs = payload.setdefault("signs", {})
        key = (gloss or "").strip().lower()
        if not key:
            return meta_path
        signs[key] = {
            "deaf_native_reviewed": bool(record_dict.get("deaf_native_reviewed", False)),
            "reviewer_count": int(record_dict.get("reviewer_count") or 0),
            "reviewer_language_match": bool(record_dict.get("reviewer_language_match", False)),
            "review_source": record_dict.get("review_source"),
            "last_reviewed": record_dict.get("last_reviewed"),
            "notes": record_dict.get("notes"),
        }
        meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return meta_path


def _stronger(a: ReviewRecord, b: ReviewRecord) -> bool:
    """Is ``a`` at least as strong a default as ``b``?

    ``deaf_native_reviewed=True`` beats False. Among True defaults, corpus
    provenance is not preferred over reviewer-attested: we just keep the
    first True we see, since the alphabet/ancillary files are never True.
    """
    if a.deaf_native_reviewed and not b.deaf_native_reviewed:
        return True
    if a.deaf_native_reviewed == b.deaf_native_reviewed:
        return a.reviewer_count > b.reviewer_count
    return False


# ---------------------------------------------------------------------------
# Module-level singleton for the FastAPI server.
# ---------------------------------------------------------------------------


_index: Optional[ReviewMetadataIndex] = None
_index_lock = Lock()


def get_index(data_dir: Optional[Path] = None) -> ReviewMetadataIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = ReviewMetadataIndex(data_dir=data_dir)
        return _index


def reset_for_tests(data_dir: Optional[Path] = None) -> ReviewMetadataIndex:
    """Tests: force the singleton to rebuild from disk."""
    global _index
    with _index_lock:
        _index = ReviewMetadataIndex(data_dir=data_dir)
        return _index
