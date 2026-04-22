"""Precompute the public progress dashboard's JSON payload.

Reads every ``data/*.meta.json`` (source attribution, per-sign overrides,
default review), the backing ``.sigml`` files via
``server.review_metadata.ReviewMetadataIndex`` (gloss enumeration with the
same "real entry" guards the translator uses), the optional prompt-7
rollup at ``docs/polish/07-database-stats.json`` (post-quarantine kept
counts and alphabet coverage), and the optional
``data/progress_log.jsonl`` (chart series produced by prompt 13's cron).

Writes ``public/progress_snapshot.json`` — a small, static payload the
``/progress`` page fetches once per visit. Running this script is
idempotent and safe to run repeatedly.

Honest reporting rules (prompt 8 §7):

* If a stat cannot be computed (missing meta file, malformed default),
  the corresponding field is ``null`` and ``partial_data`` is ``true``.
  The frontend renders ``—`` (em-dash) rather than zero so visitors can
  tell "unknown" apart from "zero".
* Post-quarantine ``kept`` counts win over the raw ``sign_count`` in
  meta files whenever the prompt-7 rollup is available — that's the
  number of signs the translator can actually serve.
* The ``reviewed`` count is derived from the merged review metadata
  with the same precedence the translator's ``/api/plan`` uses, so the
  dashboard number matches what a user would see on a per-token chip.

Run::

    python -m server.progress_snapshot
    python server/progress_snapshot.py   # equivalent
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PUBLIC_DIR = REPO_ROOT / "public"
DOCS_DIR = REPO_ROOT / "docs" / "polish"
OUTPUT_PATH = PUBLIC_DIR / "progress_snapshot.json"
PROGRESS_LOG_PATH = DATA_DIR / "progress_log.jsonl"
DB_STATS_PATH = DOCS_DIR / "07-database-stats.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the same review-metadata loader the translator uses so the
# dashboard's counts always agree with what ``/api/plan`` would report.
# ``_enumerate_glosses`` here is the defence-in-depth gloss reader
# (rejects [object Object] bodies, quarantine sidecars, manual-less
# entries) — re-using it keeps the "alphabet letters covered" number
# honest rather than regex-scanning blindly.
from server.review_metadata import (  # noqa: E402
    ReviewMetadataIndex,
    _enumerate_glosses,
    gloss_base,
)

# Reuse prompt-7's curated top-500 word list and alphabet definitions so
# the dashboard's coverage numbers are the same numbers the audit
# markdown reports. If prompt 7's script ever moves or the list changes,
# both artefacts update together.
from scripts.database_health_audit import (  # noqa: E402
    EXPECTED_ALPHABETS,
    TOP_500_EN,
)

# Display names pulled from the contribute page's shared language JSON
# so the dashboard and the contribute picker can never drift.
LANGUAGES_JSON_PATH = PUBLIC_DIR / "contribute-languages.json"

# Fallback names for the language codes the audit covers that the
# contribute picker doesn't currently expose (community-only databases).
# Keys are sign-language codes; values are {"name": English name,
# "native": endonym or best available}. If a code appears in both this
# table and contribute-languages.json, the shared JSON wins.
_FALLBACK_LANG_NAMES: dict[str, dict[str, str]] = {
    "algerian": {
        "name": "Algerian Sign Language",
        "native": "لغة الإشارات الجزائرية",
    },
    "bangla": {
        "name": "Bangla Sign Language",
        "native": "বাংলা ইশারা ভাষা",
    },
    "fsl": {
        "name": "Filipino Sign Language",
        "native": "Filipino Sign Language",
    },
    "isl": {
        "name": "Indian Sign Language",
        "native": "Indian Sign Language",
    },
    "kurdish": {
        "name": "Kurdish Sign Language",
        "native": "زمانی ئاماژەیی کوردی",
    },
    "vsl": {
        "name": "Vietnamese Sign Language",
        "native": "Ngôn ngữ ký hiệu Việt Nam",
    },
    "rsl": {
        "name": "Russian Sign Language",
        "native": "Русский жестовый язык",
    },
    "lse": {
        "name": "Spanish Sign Language",
        "native": "Lengua de Signos Española",
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_language_names() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    try:
        raw = LANGUAGES_JSON_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        payload = {}
    for lang in (payload.get("languages") or []) if isinstance(payload, dict) else []:
        code = (lang.get("code") or "").strip().lower()
        if not code:
            continue
        out[code] = {
            "name": lang.get("english_name") or lang.get("name") or code.upper(),
            "native": lang.get("name") or code.upper(),
        }
    for code, names in _FALLBACK_LANG_NAMES.items():
        out.setdefault(code, names)
    return out


def _load_db_stats() -> dict:
    """Read prompt-7's rollup if present; return {} when absent/malformed."""
    try:
        return json.loads(DB_STATS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_progress_series() -> Optional[list[dict]]:
    """Read the JSONL growth log if present.

    Returns ``None`` when the file doesn't exist yet (prompt 13's cron
    hasn't written it) so the frontend can render "data unavailable"
    for the chart rather than an empty axis. Returns ``[]`` when the
    file exists but has no valid lines — also renders "data unavailable"
    on the frontend.
    """
    if not PROGRESS_LOG_PATH.exists():
        return None
    series: list[dict] = []
    try:
        lines = PROGRESS_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        date = obj.get("date")
        reviewed = obj.get("reviewed")
        if not isinstance(date, str) or not isinstance(reviewed, (int, float)):
            continue
        series.append({"date": date, "reviewed": int(reviewed)})
    series.sort(key=lambda x: x["date"])
    return series


def _load_meta_payloads() -> dict[str, list[tuple[Path, dict]]]:
    """Group ``(path, payload)`` by language code.

    Multiple meta files may cover the same code (e.g. BSL has both a
    main corpus and an alphabet); the caller picks among them by
    ``source_kind`` when a single citation is needed.
    """
    out: dict[str, list[tuple[Path, dict]]] = {}
    for mf in sorted(DATA_DIR.glob("*.meta.json")):
        try:
            payload = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        code = (payload.get("language") or "").strip().lower()
        if not code:
            continue
        out.setdefault(code, []).append((mf, payload))
    return out


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def _top500_unique() -> list[str]:
    """Ordered, deduplicated top-500 English word list."""
    seen: set[str] = set()
    words: list[str] = []
    for w in TOP_500_EN:
        lw = (w or "").strip().lower()
        if not lw or lw in seen:
            continue
        seen.add(lw)
        words.append(lw)
    return words


def _bases_for_language(index: ReviewMetadataIndex, code: str) -> set[str]:
    """Compute the set of ``gloss_base`` keys for one language's glosses.

    Matches what the translator considers a "coverage hit" — a lookup
    of ``fruit`` resolves through ``gloss_base`` the same way here.
    """
    bases: set[str] = set()
    for g in index.known_glosses(code):
        base = gloss_base(g)
        if base:
            bases.add(base)
    return bases


def _pick_main_source(payloads: list[tuple[Path, dict]]) -> tuple[str, str]:
    """Return the (citation, kind) to display in the per-language row.

    Preference: corpus > community > alphabet. The alphabet files'
    "source" is a generic "X manual alphabet" which carries no
    provenance signal, so we only fall back to it when nothing better
    exists.
    """
    for wanted in ("corpus", "community", "alphabet"):
        for _, payload in payloads:
            if (payload.get("source_kind") or "").strip() == wanted:
                return (
                    str(payload.get("source") or ""),
                    str(payload.get("source_kind") or ""),
                )
    # Nothing matched — fall back to whatever the first payload carries.
    if payloads:
        first = payloads[0][1]
        return (str(first.get("source") or ""), str(first.get("source_kind") or ""))
    return ("", "")


def _last_updated(payloads: list[tuple[Path, dict]]) -> Optional[str]:
    """YYYY-MM-DD of the most recent meta-file mtime for the language."""
    best: Optional[float] = None
    for path, _ in payloads:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best:
            best = mtime
    if best is None:
        return None
    return datetime.fromtimestamp(best, tz=timezone.utc).strftime("%Y-%m-%d")


def _alphabet_status(
    code: str,
    db_stats: dict,
) -> tuple[str, Optional[int], Optional[int]]:
    """Return ("full"|"partial"|"none"|"unknown", present, expected).

    Preference order for the counts:

    1. Prompt-7's rollup — already knows letters kept post-quarantine.
    2. Re-read the alphabet SIGML via ``_enumerate_glosses`` — the
       honest live count if the rollup hasn't been regenerated.
    3. "unknown" when we can't decide at all.
    """
    expected_meta = EXPECTED_ALPHABETS.get(code)
    if expected_meta is None:
        # Languages without a defined alphabet show "none" — the
        # translator has no fingerspelling fallback for unknown words,
        # and the dashboard should say so plainly.
        return ("none", 0, 0)

    # Prefer the prompt-7 rollup when available.
    by_lang = (db_stats.get("by_language") or {}) if isinstance(db_stats, dict) else {}
    entry = by_lang.get(code) or {}
    alph_block = entry.get("alphabet") if isinstance(entry, dict) else None
    present: Optional[int] = None
    expected: Optional[int] = None
    if isinstance(alph_block, dict):
        present = int(alph_block.get("present") or 0)
        expected = int(alph_block.get("expected") or 0)

    # Fallback: enumerate the alphabet file directly.
    if present is None or expected is None or expected == 0:
        alph_path = DATA_DIR / expected_meta["file"]
        expected_letters: list[str] = list(expected_meta["letters"])
        expected = len(expected_letters)
        if not alph_path.exists():
            present = 0
        else:
            present_letters = {g.upper() for g in _enumerate_glosses(alph_path)}
            present = sum(1 for L in expected_letters if L in present_letters)

    if expected == 0:
        return ("none", 0, 0)
    if present >= expected:
        return ("full", present, expected)
    if present > 0:
        return ("partial", present, expected)
    return ("none", present, expected)


def _language_total_signs(
    code: str,
    db_stats: dict,
    meta_payloads: list[tuple[Path, dict]],
    index: ReviewMetadataIndex,
) -> Optional[int]:
    """Prefer prompt-7's kept count; fall back to meta ``sign_count``.

    Returns ``None`` when neither source can supply a number — that
    pushes the frontend to render "—" rather than fabricating a zero.
    """
    by_lang = (db_stats.get("by_language") or {}) if isinstance(db_stats, dict) else {}
    entry = by_lang.get(code) or {}
    if isinstance(entry, dict) and "kept" in entry:
        try:
            return int(entry["kept"])
        except (TypeError, ValueError):
            pass

    # Fallback 1: sum sign_count across meta files for this language.
    total = 0
    found = False
    for _, payload in meta_payloads:
        sc = payload.get("sign_count")
        if isinstance(sc, int):
            total += sc
            found = True
    if found:
        return total

    # Fallback 2: count enumerated glosses from the backing .sigml.
    glosses = index.known_glosses(code)
    if glosses:
        return len(glosses)
    return None


def _reviewed_and_pending(
    code: str,
    total: Optional[int],
    meta_payloads: list[tuple[Path, dict]],
) -> tuple[Optional[int], Optional[int], Optional[int], bool]:
    """Return (reviewed, awaiting_review, community_pending, partial).

    ``reviewed`` is counted against the ``total`` from
    ``_language_total_signs`` — the honest post-quarantine number.
    When the meta file is missing or malformed, every number is ``None``
    and ``partial`` is ``True`` so the frontend shows "partial data".

    ``community_pending`` counts per-sign overrides with
    ``deaf_native_reviewed == False`` AND ``in_database == True``
    semantics — i.e. a community contribution waiting for a reviewer.
    When no per-sign overrides are present, this equals ``0`` (not
    unknown) because the meta schema explicitly carries a
    ``default_review`` that tells us every sign's state.
    """
    if total is None:
        return (None, None, None, True)

    if not meta_payloads:
        # No meta file at all — we know the total from the sigml scan
        # but can't say anything about review status.
        return (None, None, None, True)

    # Pick the "strongest" default_review across meta files for this
    # language. A corpus default with deaf_native_reviewed=true beats
    # an alphabet default with deaf_native_reviewed=false.
    best_default_reviewed = False
    best_default_score = -1
    any_default_seen = False
    override_reviewed = 0
    override_unreviewed = 0
    total_overrides = 0
    for _, payload in meta_payloads:
        default = payload.get("default_review")
        if isinstance(default, dict):
            any_default_seen = True
            reviewed_flag = bool(default.get("deaf_native_reviewed"))
            # score: prefer "reviewed=True + reviewer_count" > "reviewed=True"
            # > "reviewed=False".
            reviewer_count = int(default.get("reviewer_count") or 0)
            score = (2 if reviewed_flag else 0) + min(reviewer_count, 2)
            if score > best_default_score:
                best_default_score = score
                best_default_reviewed = reviewed_flag
        signs = payload.get("signs")
        if isinstance(signs, dict):
            for raw in signs.values():
                if not isinstance(raw, dict):
                    continue
                total_overrides += 1
                if raw.get("deaf_native_reviewed"):
                    override_reviewed += 1
                else:
                    override_unreviewed += 1

    if not any_default_seen:
        return (None, None, None, True)

    # Unspecified entries pick up the default — clamped against `total`
    # so a bogus meta with more overrides than kept signs doesn't
    # produce a negative "unspecified" bucket.
    unspecified = max(0, total - total_overrides)
    if best_default_reviewed:
        reviewed = override_reviewed + unspecified
        unreviewed = override_unreviewed
    else:
        reviewed = override_reviewed
        unreviewed = override_unreviewed + unspecified

    # Never let clamping produce a total higher than the kept count —
    # if per-sign overrides over-index against kept, prefer the
    # override numbers and mark the state as partial so the tooltip
    # explains the discrepancy on the frontend.
    partial = False
    if reviewed + unreviewed > total:
        partial = True
    return (reviewed, unreviewed, override_unreviewed, partial)


def _top500_coverage(
    index: ReviewMetadataIndex,
    code: str,
    unique_top500: list[str],
) -> tuple[Optional[float], Optional[int], int]:
    bases = _bases_for_language(index, code)
    if not bases:
        return (None, None, len(unique_top500))
    covered = sum(1 for w in unique_top500 if w in bases)
    pct = (covered * 100.0 / len(unique_top500)) if unique_top500 else None
    return (round(pct, 1) if pct is not None else None, covered, len(unique_top500))


def _coverage_gaps(
    index: ReviewMetadataIndex,
    a_code: str,
    b_code: str,
    limit: int = 20,
) -> list[str]:
    """Top-500 words present in ``a_code`` but missing from ``b_code``."""
    a_bases = _bases_for_language(index, a_code)
    b_bases = _bases_for_language(index, b_code)
    if not a_bases:
        return []
    out: list[str] = []
    for w in _top500_unique():
        if w in a_bases and w not in b_bases:
            out.append(w)
            if len(out) >= limit:
                break
    return out


def _recent_validations(limit: int = 20) -> list[dict]:
    """Last ``limit`` per-sign overrides that flipped to deaf_native_reviewed.

    Sources the events from per-sign ``signs`` entries across every
    meta file — exactly the channel prompt 14's export hook writes
    through when a submission reaches ``validated``. The feed is
    aggregate (gloss + language + reviewer count + timestamp) — it
    exposes no contributor identity.
    """
    events: list[dict] = []
    for meta_file in sorted(DATA_DIR.glob("*.meta.json")):
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        code = (payload.get("language") or "").strip().lower()
        signs = payload.get("signs") or {}
        if not isinstance(signs, dict):
            continue
        for gloss, raw in signs.items():
            if not isinstance(raw, dict):
                continue
            if not raw.get("deaf_native_reviewed"):
                continue
            ts = raw.get("last_reviewed")
            if not isinstance(ts, str) or not ts:
                continue
            events.append(
                {
                    "gloss": str(gloss).upper(),
                    "language": code,
                    "reviewer_count": int(raw.get("reviewer_count") or 0),
                    "timestamp": ts,
                }
            )
    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return events[:limit]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_snapshot() -> dict:
    index = ReviewMetadataIndex(data_dir=DATA_DIR)
    lang_names = _load_language_names()
    db_stats = _load_db_stats()
    meta_payloads_by_lang = _load_meta_payloads()
    unique_top500 = _top500_unique()

    # Discover every code the translator or the audit knows about. Using
    # the union lets us show a row for a language whose meta file is
    # missing (with partial_data=true) rather than silently hiding it.
    codes: set[str] = set(index.known_languages())
    codes.update(meta_payloads_by_lang.keys())
    codes.update(
        (db_stats.get("by_language") or {}).keys()
        if isinstance(db_stats, dict) else []
    )

    languages: list[dict] = []
    for code in sorted(codes):
        payloads = meta_payloads_by_lang.get(code, [])
        source, source_kind = _pick_main_source(payloads)
        total = _language_total_signs(code, db_stats, payloads, index)
        reviewed, unreviewed, community_pending, review_partial = _reviewed_and_pending(
            code, total, payloads,
        )
        alphabet_status, alph_present, alph_expected = _alphabet_status(code, db_stats)
        top500_pct, top500_covered, top500_total = _top500_coverage(
            index, code, unique_top500,
        )
        last_updated = _last_updated(payloads)
        partial = bool(review_partial) or total is None or not payloads
        names = lang_names.get(code) or {
            "name": code.upper(),
            "native": code.upper(),
        }
        languages.append(
            {
                "code": code,
                "name": names["name"],
                "native_name": names["native"],
                "source": source or None,
                "source_kind": source_kind or None,
                "total": total,
                "reviewed": reviewed,
                "awaiting_review": unreviewed,
                "community_pending": community_pending,
                "alphabet": alphabet_status,
                "alphabet_present": alph_present,
                "alphabet_expected": alph_expected,
                "top500_covered": top500_covered,
                "top500_total": top500_total,
                "top500_coverage_pct": top500_pct,
                "last_updated": last_updated,
                "partial_data": partial,
            }
        )

    # Default table order: total signs desc, language name asc.
    languages.sort(
        key=lambda e: (
            -(e["total"] or 0),
            (e["name"] or "").lower(),
        )
    )

    # Top-line summary. A "language supported" is one with at least one
    # kept sign; we don't inflate the count with empty meta files.
    total_signs = sum((e["total"] or 0) for e in languages)
    total_reviewed = sum((e["reviewed"] or 0) for e in languages)
    total_awaiting = sum((e["awaiting_review"] or 0) for e in languages)
    languages_supported = sum(1 for e in languages if (e["total"] or 0) > 0)

    # Honest reporting: if every language reports partial data, the
    # top-line numbers are unreliable — surface that.
    all_partial = languages and all(e["partial_data"] for e in languages)
    totals = {
        "signs": total_signs if not all_partial else None,
        "languages": languages_supported if not all_partial else None,
        "reviewed": total_reviewed if not all_partial else None,
        "awaiting_review": total_awaiting if not all_partial else None,
    }

    # Growth chart series.
    progress_series = _load_progress_series()

    # Recent activity feed.
    recent = _recent_validations()

    gaps = {
        "bsl_missing_from_asl": _coverage_gaps(index, "bsl", "asl"),
        "asl_missing_from_bsl": _coverage_gaps(index, "asl", "bsl"),
    }

    return {
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "generator": "server/progress_snapshot.py",
        "schema_version": 1,
        "totals": totals,
        "languages": languages,
        "progress_series": progress_series,
        "recent_activity": recent,
        "coverage_gaps": gaps,
    }


def write_snapshot(out_path: Path = OUTPUT_PATH) -> dict:
    snapshot = build_snapshot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return snapshot


def main(argv: Optional[Iterable[str]] = None) -> int:
    snapshot = write_snapshot()
    totals = snapshot["totals"]
    def _num(value: object) -> str:
        return "—" if value is None else str(value)
    print(
        f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: "
        f"{_num(totals['signs'])} signs across {_num(totals['languages'])} languages, "
        f"{_num(totals['reviewed'])} reviewed, {_num(totals['awaiting_review'])} awaiting"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
