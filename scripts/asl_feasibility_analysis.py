#!/usr/bin/env python3
"""ASL-LEX -> SiGML feasibility spike (proposal 56). DETERMINISTIC ANALYSIS ONLY.

Measures whether the orientation + movement-direction gap (the part of an ASL
sign that ASL-LEX does NOT encode) can be closed automatically, and for what
fraction of the 2,723-sign lexicon. No LLM calls; no data/*.sigml written.

Inputs (gitignored, license-gated raw corpora under data/sources/):
  - data/sources/asl_lex/signdata.csv      (ASL-LEX 2.0, 2723 signs)
  - data/sources/asl_lex/signdataKEY.csv   (codebook)
  - data/sources/signpuddle/sgn4.spml      (SignPuddle ASL dict, 12266 FSW entries)

Emits a stats blob (stdout, and --json) folded into
proposals/reports/asl-integration-feasibility.md by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASL_LEX = REPO / "data" / "sources" / "asl_lex" / "signdata.csv"
SPML = REPO / "data" / "sources" / "signpuddle" / "sgn4.spml"

# ---------------------------------------------------------------------------
# Gloss normalisation — mirrors extension/panel.js glossBase(): lowercase,
# drop parentheticals + trailing disambiguation digits, non-alnum -> space.
# ---------------------------------------------------------------------------
def gloss_base(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"#\d+$", "", s)
    s = re.sub(r"_\d+[a-z]?$", "", s)      # ASL-LEX disambiguators: candy_1, tall_2
    s = re.sub(r"\d+[a-z]?\^?$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)      # underscores/punct -> space (new_york -> new york)
    return s.strip()


# ---------------------------------------------------------------------------
# Handshape faithfulness: HamNoSys ships 12 base handshapes. An ASL-LEX label
# is "faithful" if it maps to a HamNoSys base using only standard thumb/bend/
# hook modifiers; "approx" if its selected-finger pattern (thumb-only, single
# non-index finger, non-contiguous fingers, claw/cee curls) has no faithful
# HamNoSys base and must be approximated. Keyed by the real lowercase labels.
# ---------------------------------------------------------------------------
FAITHFUL_HS = {
    # index family -> hamfinger2 (+bend/hook/thumbout)
    "1", "flat_1", "curved_1", "bent_1", "g", "d", "l", "bent_l", "curved_l",
    # flat hand -> hamflathand
    "b", "open_b", "closed_b", "flat_b",
    # spread five -> hamflathand/hamfinger2345 (spread approximated, base faithful)
    "5", "flatspread_5", "stacked_5", "curved_5",
    # fist -> hamfist (+thumbout)
    "s", "a",
    # cee -> hamceeall
    "c",
    # round/pinch -> hampinchall / hampinch12
    "o", "baby_o", "flat_o",
    # open pinch -> hampinch12open
    "f", "open_f",
    # index+middle spread -> hamfinger23spread
    "v", "curved_v", "bent_v", "flat_v",
    # index+middle adjacent -> hamfinger23
    "h", "flat_h", "open_h", "curved_h",
    # four fingers -> hamfinger2345
    "4", "flat_4", "curved_4",
}
# Everything else observed in the data is approximated (no faithful base):
#   y, ily, flat_ily, i(pinky), horns, flat_horns, 8, open_8, p, r, w, 3, k,
#   t, m, flat_m, flat_n, e, open_e, closed_e, spread_e, spread_open_e,
#   goody_goody, 7


def load_asl_lex() -> list[dict]:
    with ASL_LEX.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# FSW (Formal SignWriting in ASCII) parser.
# A sign string: [A <sorting-symbols>] <box M|B|L|R> WxH ( S<key> colXrow )*
# A symbol key is  S + 3 hex (base 0x100..0x38b) + 1 hex (fill 0..5) + 1 hex
# (rotation 0..15, 22.5deg steps). Categories by base value (ISWA 2010):
#   0x100..0x204 hand (handshape)   0x205..0x2f6 movement (arrows/contact/dyn)
#   0x2ff..0x36c head/face          0x36d..0x375 body
# ---------------------------------------------------------------------------
SYM_RE = re.compile(r"S([0-9a-f]{3})([0-5])([0-9a-f])(\d{3})x(\d{3})")
SORT_SYM_RE = re.compile(r"S([0-9a-f]{3})([0-5])([0-9a-f])")

HAND_LO, HAND_HI = 0x100, 0x204
MOVE_LO, MOVE_HI = 0x205, 0x2F6

# rotation digit (0..15) -> nearest 8-way compass (2 steps = 45deg).
# Documented best-effort per ISWA convention: rot 0 = fingers up; steps
# advance clockwise. Parity 8..15 selects the mirrored/other-plane variant.
DIR8 = ["u", "ur", "r", "dr", "d", "dl", "l", "ul"]


def rot_to_dir8(rot: int) -> str:
    return DIR8[round((rot % 16) / 2) % 8]


def fill_to_plane(fill: int) -> str:
    # ISWA hand fills: 0-2 wall plane (vertical, palm faces signer/away by
    # rotation), 3-5 floor plane (horizontal, palm up/down).
    return "wall" if fill <= 2 else "floor"


def parse_fsw(fsw: str) -> dict | None:
    """Return {hand:(base,fill,rot), moves:[(base,fill,rot)], n_hands, n_moves} or None."""
    fsw = fsw.strip()
    # strip the optional A...sorting prefix (symbols with no coordinates)
    body = fsw
    if body.startswith("A"):
        m = re.match(r"A((?:S[0-9a-f]{5})+)", body)
        if m:
            body = body[m.end():]
    syms = [(int(b, 16), int(fl), int(rt, 16)) for b, fl, rt, _x, _y in SYM_RE.findall(body)]
    if not syms:
        return None
    hands = [(b, fl, rt) for (b, fl, rt) in syms if HAND_LO <= b <= HAND_HI]
    moves = [(b, fl, rt) for (b, fl, rt) in syms if MOVE_LO <= b <= MOVE_HI]
    return {
        "hand": hands[0] if hands else None,
        "moves": moves,
        "n_hands": len(hands),
        "n_moves": len(moves),
    }


def fsw_orientation(parsed: dict) -> dict | None:
    """Decode dominant-hand symbol -> candidate HamNoSys extfidir/palmor/movedir."""
    if not parsed or not parsed["hand"]:
        return None
    base, fill, rot = parsed["hand"]
    extfidir = rot_to_dir8(rot)
    plane = fill_to_plane(fill)
    if plane == "floor":
        palmor = "d" if fill in (3, 4) else "u"
    else:
        # wall plane: palm faces signer (i)/away(o) -> map to nearest vertical palmor
        palmor = {"u": "l", "r": "d", "d": "r", "l": "u"}.get(extfidir, "l")
    movedir = None
    if parsed["moves"]:
        movedir = rot_to_dir8(parsed["moves"][0][2])
    return {"extfidir": extfidir, "palmor": palmor, "plane": plane, "movedir": movedir}


def load_fsw_dict() -> dict[str, list[str]]:
    """gloss_base -> list of FSW strings, from sgn4.spml."""
    txt = SPML.read_text(encoding="utf-8", errors="replace")
    out: dict[str, list[str]] = {}
    for ent in re.findall(r"<entry\b.*?</entry>", txt, re.DOTALL):
        terms = re.findall(r"<term>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</term>", ent, re.DOTALL)
        terms = [a or b for a, b in terms]
        if len(terms) < 2:
            continue
        fsw, gloss = terms[0].strip(), terms[1].strip()
        if not fsw.startswith("A") and not re.match(r"[MBLR]\d", fsw):
            continue
        gb = gloss_base(gloss)
        if gb:
            out.setdefault(gb, []).append(fsw)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    rows = load_asl_lex()
    N = len(rows)
    out: dict = {"n_signs": N}

    # ------------------------------------------------------------------ schema
    def col(c):
        return [(r.get(c) or "").strip() for r in rows]
    def dist(c):
        return Counter(col(c))

    SCHEMA_COLS = [
        "Handshape.2.0", "NonDominantHandshape.2.0", "SelectedFingers.2.0",
        "Flexion.2.0", "FlexionChange.2.0", "Spread.2.0", "SpreadChange.2.0",
        "ThumbPosition.2.0", "ThumbContact.2.0", "SignType.2.0", "Movement.2.0",
        "RepeatedMovement.2.0", "MajorLocation.2.0", "MinorLocation.2.0",
        "SecondMinorLocation.2.0", "Contact.2.0", "UlnarRotation.2.0",
    ]
    out["schema"] = {c: dict(dist(c).most_common()) for c in SCHEMA_COLS}

    # --------------------------------------------------- handshape faithfulness
    hs = dist("Handshape.2.0")
    faithful = sum(n for k, n in hs.items() if k in FAITHFUL_HS)
    approx = sum(n for k, n in hs.items() if k and k not in FAITHFUL_HS)
    approx_labels = sorted({k for k in hs if k and k not in FAITHFUL_HS})
    out["handshape"] = {
        "distinct": len([k for k in hs if k]),
        "faithful": faithful, "approx": approx,
        "approx_pct": round(100 * approx / N, 1),
        "approx_labels": approx_labels,
    }

    # -------------------------------------------------- structural infeasibility
    def is_int_gt(v, n):
        try:
            return int(v) > n
        except ValueError:
            return False
    infeasible_flags = {}
    for r in rows:
        eid = r["EntryID"]
        reasons = []
        if is_int_gt(r.get("NumberOfMorphemes.2.0", "1"), 1):
            reasons.append("multi-morpheme")
        if r.get("Compound.2.0") == "1":
            reasons.append("compound")
        if r.get("FingerspelledLoanSign.2.0") == "1":
            reasons.append("fs-loan")
        if r.get("FlexionChange.2.0") == "1":
            reasons.append("flexion-change")
        if (r.get("HandshapeM2.2.0") or "").strip() not in ("", "NA"):
            reasons.append("handshape-M2")
        if reasons:
            infeasible_flags[eid] = reasons
    infeasible = set(infeasible_flags)
    by_reason = Counter(rr for rs in infeasible_flags.values() for rr in rs)
    out["infeasible"] = {
        "total_union": len(infeasible),
        "pct": round(100 * len(infeasible) / N, 1),
        "by_reason": dict(by_reason),
    }

    # ------------------------------------------------------ gloss overlap w/ FSW
    fsw_dict = load_fsw_dict()
    out["fsw_entries_with_gloss"] = sum(len(v) for v in fsw_dict.values())
    out["fsw_distinct_glosses"] = len(fsw_dict)

    exact = norm = 0
    matched_eids: dict[str, str] = {}     # eid -> chosen FSW
    raw_fsw_glosses = set()
    txt = SPML.read_text(encoding="utf-8", errors="replace")
    for ent in re.findall(r"<entry\b.*?</entry>", txt, re.DOTALL):
        terms = re.findall(r"<term>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</term>", ent, re.DOTALL)
        terms = [a or b for a, b in terms]
        if len(terms) >= 2:
            raw_fsw_glosses.add(terms[1].strip().lower())

    for r in rows:
        eid = r["EntryID"]
        gb = gloss_base(eid)
        if eid.lower() in raw_fsw_glosses:
            exact += 1
        if gb in fsw_dict:
            norm += 1
            matched_eids[eid] = fsw_dict[gb][0]
    out["match"] = {
        "exact": exact, "exact_pct": round(100 * exact / N, 1),
        "normalized": norm, "normalized_pct": round(100 * norm / N, 1),
        "matched_signs": len(matched_eids),
    }

    # ------------------------------------ FSW orientation extractability (matched)
    orient_ok = move_ok = both_ok = 0
    samples = []
    for eid, fsw in matched_eids.items():
        p = parse_fsw(fsw)
        o = fsw_orientation(p) if p else None
        has_orient = bool(o)
        has_move = bool(o and o.get("movedir"))
        orient_ok += has_orient
        move_ok += has_move
        both_ok += has_orient and has_move
        if len(samples) < args.sample and has_orient:
            samples.append({"gloss": eid, "fsw": fsw[:40],
                            "extfidir": o["extfidir"], "palmor": o["palmor"],
                            "plane": o["plane"], "movedir": o["movedir"]})
    nm = max(len(matched_eids), 1)
    out["fsw_orientation"] = {
        "matched": len(matched_eids),
        "hand_parseable": orient_ok, "hand_parseable_pct": round(100 * orient_ok / nm, 1),
        "movedir_present": move_ok, "movedir_present_pct": round(100 * move_ok / nm, 1),
        "both": both_ok, "both_pct": round(100 * both_ok / nm, 1),
    }
    out["fsw_samples"] = samples

    # ----------------------------------------------------- A / B / C orientation
    # Bucket A: deterministic orientation HIGH-confidence. The citation
    # orientation is strongly anchored when the hand contacts a specific body
    # or hand location (extfidir toward it, palm facing it). Neutral-space,
    # no-contact signs are genuinely underdetermined -> NOT bucket A.
    A_eids = set()
    for r in rows:
        major = r.get("MajorLocation.2.0", "")
        contact = r.get("Contact.2.0") == "1"
        minor = r.get("MinorLocation.2.0", "")
        if contact and major in ("Head", "Body", "Arm", "Hand") and minor not in ("", "Other"):
            A_eids.add(r["EntryID"])
    A = len(A_eids)

    # Bucket B: signs NOT in A whose gloss matches FSW AND FSW yields a
    # parseable dominant-hand orientation.
    B_eids = set()
    for eid, fsw in matched_eids.items():
        if eid in A_eids:
            continue
        p = parse_fsw(fsw)
        if p and p["hand"]:
            B_eids.add(eid)
    B = len(B_eids)
    C = N - A - B
    out["buckets"] = {
        "A_deterministic": A, "A_pct": round(100 * A / N, 1),
        "B_fsw_crossref": B, "B_pct": round(100 * B / N, 1),
        "C_residual": C, "C_pct": round(100 * C / N, 1),
    }

    # ----------------------------------------------------------------- print
    def p(*a):
        print(*a)
    p(f"\n===== ASL-LEX feasibility spike =====  N={N} signs")
    p(f"\n[handshape] {out['handshape']['distinct']} distinct labels; "
      f"faithful HamNoSys base={faithful} ({round(100*faithful/N,1)}%), "
      f"approx/no-faithful={approx} ({out['handshape']['approx_pct']}%)")
    p(f"  approx labels: {', '.join(approx_labels)}")
    p(f"\n[structural infeasibility] union={len(infeasible)} ({out['infeasible']['pct']}%)  by-reason={dict(by_reason)}")
    p(f"\n[FSW dict] {out['fsw_entries_with_gloss']} glossed entries, "
      f"{out['fsw_distinct_glosses']} distinct normalized glosses")
    p(f"[gloss match ASL-LEX->FSW] exact={exact} ({out['match']['exact_pct']}%)  "
      f"normalized={norm} ({out['match']['normalized_pct']}%)")
    fo = out["fsw_orientation"]
    p(f"\n[FSW orientation extractable | matched={fo['matched']}] "
      f"hand-parseable={fo['hand_parseable']} ({fo['hand_parseable_pct']}%)  "
      f"movedir-present={fo['movedir_present']} ({fo['movedir_present_pct']}%)")
    p(f"\n[buckets over N={N}]")
    p(f"  A deterministic (anchored contact)  = {A:5d} ({out['buckets']['A_pct']}%)")
    p(f"  B FSW cross-ref (A-miss, FSW hand)  = {B:5d} ({out['buckets']['B_pct']}%)")
    p(f"  C residual (LLM/human)              = {C:5d} ({out['buckets']['C_pct']}%)")
    p(f"\n[FSW decode samples] (eyeball correctness):")
    for s in samples[:20]:
        p(f"  {s['gloss']:18s} ext={s['extfidir']:2s} palm={s['palmor']:2s} "
          f"plane={s['plane']:5s} move={s['movedir'] or '-':2s}  {s['fsw']}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        p(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
