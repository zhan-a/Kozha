#!/usr/bin/env python3
"""Stage 61 — bulk-assemble the ASL lexicon into active + quarantine SiGML.

Runs the stage-57 deterministic + heuristic converter (``asl_lex_to_sigml``)
over the full ASL-LEX 2.0 phonological export and splits the result into:

  data/American_SL_ASL.sigml            — renderable, plausible signs (served)
  data/American_SL_ASL.sigml.meta.json  — honest provenance for the active set
  data/American_SL_ASL_quarantine.sigml — approx-handshape / structurally
                                          infeasible / orientation-defaulted
                                          signs held for Deaf-native authoring
                                          (NO meta sidecar; not served)

FSW orientation enrichment (stage 58) is DEFERRED: the SignPuddle dictionary-
content license is unconfirmed and ``data/sources/signpuddle/LICENSE_CLEARED``
is absent, so no FSW-sourced field is written here. No LLM is used — this stage
is a deterministic assembly with zero token spend.

A sign is QUARANTINED when any of these hold (each recorded as a reason):
  - handshape label has no faithful HamNoSys base (approximated) or is unmapped
  - location did not map to a HamNoSys body tag
  - orientation fell back to the last-resort default (no location rule matched)
  - the sign is structurally infeasible for the single-hand citation encoder:
      * FlexionChange.2.0          — handshape changes mid-sign
      * HandshapeM2.2.0            — a distinct second (sequential) handshape
      * NumberOfMorphemes.2.0 > 1  — multi-morpheme
      * Compound.2.0               — compound sign
      * FingerspelledLoanSign.2.0  — fingerspelled loan
Otherwise it is ACTIVE.

Glosses are kept DISJOINT: a gloss that lands in the active file is never also
written to quarantine (active wins on duplicate EntryIDs).

Usage:
    python3 scripts/asl_integrate.py \
        --in data/sources/asl_lex/signdata.csv \
        [--validate]      # fail if any emitted tag is unknown to CWASA (default on)
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import quoteattr, escape

from asl_lex_to_sigml import (  # noqa: E402  (sibling module)
    col,
    convert_row,
    load_known_tags,
    lookup_handshape,
    _norm,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

ACTIVE_SIGML = REPO_ROOT / "data" / "American_SL_ASL.sigml"
ACTIVE_META = REPO_ROOT / "data" / "American_SL_ASL.sigml.meta.json"
QUARANTINE_SIGML = REPO_ROOT / "data" / "American_SL_ASL_quarantine.sigml"

_TRUTHY = {"1", "y", "yes", "true", "+"}


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in _TRUTHY


def classify(row: dict[str, str], fields: list[dict]) -> list[str]:
    """Return the list of quarantine reasons for ``row`` (empty => active)."""
    reasons: list[str] = []

    hs_label = (col(row, "Handshape") or "").strip()
    (_base, _mods, approx), found = lookup_handshape(hs_label)
    if not found:
        reasons.append(f"handshape {hs_label or 'n/a'!r} unmapped")
    elif approx:
        reasons.append(f"handshape {hs_label or 'n/a'!r} approximated (no faithful HamNoSys base)")

    loc_field = next((f for f in fields if f["slot"] == "location"), None)
    if loc_field is not None and not loc_field["tags"]:
        major = _norm(col(row, "MajorLocation")) or "n/a"
        minor = _norm(col(row, "MinorLocation")) or "n/a"
        reasons.append(f"location ({major}/{minor}) unmapped")

    ori_field = next((f for f in fields if f["slot"] == "orientation"), None)
    if ori_field is not None and ori_field["source"] == "default":
        reasons.append("orientation defaulted (no location rule matched)")

    # --- structural infeasibility for a single-hand citation encoder ---
    if _truthy(col(row, "FlexionChange")):
        reasons.append("flexion change (handshape changes mid-sign)")
    h2 = _norm(row.get("HandshapeM2.2.0") or "")
    if h2 and h2 not in {"na", "none"}:
        reasons.append("second sequential handshape (HandshapeM2)")
    try:
        n_morph = int(float((row.get("NumberOfMorphemes.2.0") or "1").strip() or 1))
    except (TypeError, ValueError):
        n_morph = 1
    if n_morph > 1:
        reasons.append("multi-morpheme")
    if _truthy(row.get("Compound.2.0")):
        reasons.append("compound sign")
    if _truthy(row.get("FingerspelledLoanSign.2.0")):
        reasons.append("fingerspelled loan")

    return reasons


def _reason_category(reason: str) -> str:
    """Collapse a per-sign reason string to a stable category for counting."""
    if reason.startswith("handshape") and "unmapped" in reason:
        return "handshape unmapped"
    if reason.startswith("handshape"):
        return "handshape approximated"
    if reason.startswith("location"):
        return "location unmapped"
    if reason.startswith("orientation"):
        return "orientation defaulted"
    return reason  # structural reasons are already category-shaped


def _sign_xml(gloss: str, tags: list[str]) -> str:
    body = "\n      ".join(f"<{t} />" for t in tags)
    return (
        f"  <hns_sign gloss={quoteattr(gloss)}>\n"
        f"    <hamnosys_manual>\n      {body}\n    </hamnosys_manual>\n"
        f"  </hns_sign>"
    )


def _quarantine_xml(gloss: str, tags: list[str], reasons: list[str]) -> str:
    body = "\n      ".join(f"<{t} />" for t in tags)
    reason_txt = escape("; ".join(reasons))
    return (
        f"  <!-- quarantined: {reason_txt} -->\n"
        f"  <hns_sign gloss={quoteattr(gloss)}>\n"
        f"    <hamnosys_manual>\n      {body}\n    </hamnosys_manual>\n"
        f"  </hns_sign>"
    )


def build_active_meta(active: list[dict], sign_count: int) -> dict:
    """Honest provenance summary computed over the ACTIVE set only."""
    prov_tags: dict[str, int] = {
        "asl_lex": 0, "heuristic": 0, "default": 0, "override": 0, "fsw": 0, "llm": 0,
    }
    ori_heur = ori_def = mv_heur = mv_def = 0
    for s in active:
        for fld in s["fields"]:
            prov_tags[fld["source"]] = prov_tags.get(fld["source"], 0) + len(fld["tags"])
        for fld in s["fields"]:
            if fld["slot"] == "orientation":
                if fld["source"] in {"heuristic", "override"}:
                    ori_heur += 1
                elif fld["source"] == "default":
                    ori_def += 1
            elif fld["slot"] == "movement":
                if fld["source"] in {"heuristic", "override"}:
                    mv_heur += 1
                elif fld["source"] == "default":
                    mv_def += 1
    total_tags = sum(prov_tags.values()) or 1
    provenance_summary = {
        "tags_by_source": prov_tags,
        "tags_total": sum(prov_tags.values()),
        "share": {k: round(v / total_tags, 4) for k, v in prov_tags.items()},
        "orientation_heuristic_signs": ori_heur,
        "orientation_default_signs": ori_def,
        "movement_dir_heuristic_signs": mv_heur,
        "movement_dir_default_signs": mv_def,
        "fsw_signs": 0,
        "llm_signs": 0,
    }
    return {
        "version": 1,
        "language": "asl",
        "iso_code": "ase",
        "display_name": "American Sign Language",
        "source": "ASL-LEX 2.0 phonological coding scheme (Sehyr, Caselli, "
                  "Cohen-Goldberg & Emmorey 2021), asl-lex.org. Handshape, location, "
                  "symmetry and movement-shape are derived deterministically from the "
                  "ASL-LEX 2.0 phonological columns; the raw ASL-LEX corpus is NOT "
                  "redistributed (gitignored under data/sources/).",
        "source_kind": "seed",
        "license": "CC BY-NC 4.0 (ASL-LEX 2.0 database, per asl-lex.org). Note: the "
                   "ASL-LEX OSF node is tagged CC BY 4.0; we adopt the stricter "
                   "non-commercial CC BY-NC 4.0 stated on asl-lex.org. Emitted SiGML "
                   "encodings are original derived works.",
        "data_completeness": "seed",
        "accepts_first_contributions": True,
        "review_required": True,
        "conversion_note": "Handshape/location/symmetry/movement-shape derived "
                           "deterministically from ASL-LEX 2.0 columns (provenance "
                           "asl_lex). Palm+finger orientation and movement direction "
                           "are rule-derived heuristics (provenance heuristic), NOT "
                           "measured by ASL-LEX and UNVERIFIED. SignWriting/FSW "
                           "orientation enrichment (stage 58) was NOT applied — the "
                           "SignPuddle dictionary-content license is unconfirmed "
                           "(data/sources/signpuddle/LICENSE_CLEARED absent), so no "
                           "fsw-sourced field is present (fsw=0). No LLM repair was "
                           "run (llm=0). Per-parameter provenance is in "
                           "provenance_summary.",
        "credits": [
            {
                "name": "ASL-LEX 2.0",
                "role": "phonological source (handshape, location, symmetry, movement shape)",
                "citation": "Sehyr ZS, Caselli N, Cohen-Goldberg AM, Emmorey K (2021). "
                            "The ASL-LEX 2.0 Project. Journal of Deaf Studies and Deaf "
                            "Education, 26(2), 263-277.",
                "url": "https://asl-lex.org",
                "license": "CC BY-NC 4.0",
            },
            {
                "name": "SignWriting / SignPuddle (ISWA 2010 / FSW)",
                "role": "orientation enrichment — EVALUATED but NOT used in this build "
                        "(license gate closed); attribution carried for the deferred "
                        "stage-58 path",
                "url": "https://signpuddle.net",
                "license": "dictionary-content license UNCONFIRMED — see "
                           "proposals/reports/signpuddle-license-request.md",
            },
        ],
        "provenance_summary": provenance_summary,
        "sigml_file": ACTIVE_SIGML.name,
        "quarantine_file": QUARANTINE_SIGML.name,
        "generated": date.today().isoformat(),
        "default_review": {
            "deaf_native_reviewed": False,
            "reviewer_count": 0,
            "reviewer_language_match": False,
            "review_source": None,
            "last_reviewed": None,
            "notes": "machine-converted seed from ASL-LEX 2.0; orientation/direction "
                     "are rule-derived heuristics; awaiting Deaf-native review",
        },
        "sign_count": sign_count,
        "signs": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="infile",
                    default="data/sources/asl_lex/signdata.csv")
    ap.add_argument("--no-validate", dest="validate", action="store_false",
                    help="skip CWASA tokenNameMap validation (validation is on by default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report counts but write nothing")
    ap.set_defaults(validate=True)
    args = ap.parse_args()

    in_path = (REPO_ROOT / args.infile) if not Path(args.infile).is_absolute() else Path(args.infile)
    if not in_path.exists():
        raise SystemExit(f"[error] corpus not found: {in_path}")

    # ASL-LEX export carries stray Latin-1 bytes; decode leniently (matches the
    # converter and the feasibility analysis) so the bulk run never dies.
    with in_path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    active: list[dict] = []           # {gloss, tags, fields, review}
    quar_pending: list[tuple] = []    # (gloss, key, tags, reasons)
    seen_active: set[str] = set()
    reason_counts: dict[str, int] = {}

    for row in rows:
        gloss, tags, fields, review = convert_row(row)
        if not gloss:
            continue
        reasons = classify(row, fields)
        key = gloss.strip().lower()
        if not reasons:
            if key in seen_active:
                continue
            seen_active.add(key)
            active.append({"gloss": gloss, "tags": tags, "fields": fields, "review": review})
        else:
            quar_pending.append((gloss, key, tags, reasons))

    # Resolve quarantine AFTER the full pass so active always wins a duplicate
    # gloss regardless of row order; dedup within quarantine too.
    quarantine: list[dict] = []
    seen_quar: set[str] = set()
    for gloss, key, tags, reasons in quar_pending:
        if key in seen_active or key in seen_quar:
            continue
        seen_quar.add(key)
        quarantine.append({"gloss": gloss, "tags": tags, "reasons": reasons})
        for cat in {_reason_category(r) for r in reasons}:
            reason_counts[cat] = reason_counts.get(cat, 0) + 1

    # --- validate every emitted tag against the CWASA tokenNameMap oracle ---
    if args.validate:
        known = load_known_tags() | {"hns_sign", "hamnosys_manual", "sigml_collection"}
        all_tags = (
            {t for s in active for t in s["tags"]}
            | {t for s in quarantine for t in s["tags"]}
        )
        bad = sorted(t for t in all_tags if t not in known)
        if bad:
            raise SystemExit(f"VALIDATION FAILED — tags not in CWASA tokenNameMap: {bad}")
        n_uses = (sum(len(s["tags"]) for s in active)
                  + sum(len(s["tags"]) for s in quarantine))
        print(f"[validate] OK — all {n_uses} tag uses are CWASA-renderable")

    meta = build_active_meta(active, len(active))

    active_doc = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<sigml_collection language="American_SL_ASL" count="{len(active)}">\n\n'
        + "\n\n".join(_sign_xml(s["gloss"], s["tags"]) for s in active)
        + "\n\n</sigml_collection>\n"
    )
    quar_doc = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<sigml_collection language="American_SL_ASL_quarantine" count="{len(quarantine)}">\n\n'
        + "\n\n".join(_quarantine_xml(s["gloss"], s["tags"], s["reasons"]) for s in quarantine)
        + "\n\n</sigml_collection>\n"
    )

    # --- report ---
    print(f"Rows read:           {len(rows)}")
    print(f"Active (served):     {len(active)}")
    print(f"Quarantine:          {len(quarantine)}")
    print(f"Active ∩ Quarantine: {len(seen_active & seen_quar)} (must be 0)")
    ps = meta["provenance_summary"]
    print("Active tag provenance: " + ", ".join(
        f"{k}={v}" for k, v in ps["tags_by_source"].items() if v))
    print(f"Active orientation: heuristic={ps['orientation_heuristic_signs']} "
          f"default={ps['orientation_default_signs']}  |  movement dir: "
          f"heuristic={ps['movement_dir_heuristic_signs']} default={ps['movement_dir_default_signs']}")
    if reason_counts:
        print("Quarantine reasons (by leading cause, signs may have several):")
        for r, c in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {c:>5}  {r}")

    if args.dry_run:
        print("[dry-run] wrote nothing")
        return

    ACTIVE_SIGML.write_text(active_doc, encoding="utf-8")
    ACTIVE_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    QUARANTINE_SIGML.write_text(quar_doc, encoding="utf-8")
    print(f"Wrote {ACTIVE_SIGML}  ({len(active)} signs)")
    print(f"Wrote {ACTIVE_META}")
    print(f"Wrote {QUARANTINE_SIGML}  ({len(quarantine)} signs)")


if __name__ == "__main__":
    main()
