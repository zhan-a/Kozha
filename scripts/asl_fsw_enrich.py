#!/usr/bin/env python3
"""Cross-reference ASL-LEX signs against SignWriting/FSW to fill the orientation
gap (proposal 58, stage 3 of the ASL-integration batch).

WHY THIS EXISTS
---------------
ASL-LEX (the deterministic source used by ``scripts/asl_lex_to_sigml.py``) does
NOT encode palm/finger orientation or movement *direction* — those are filled by
rule-based heuristics (provenance ``heuristic``/``default``). SignWriting *does*
encode them: a SignWriting hand glyph carries a full 3-D orientation via its
**fill** (plane + palm facing) and its **rotation** (in-plane angle), and the
movement arrows carry direction. SignPuddle's ASL dictionary (``sgn4.spml``,
~12k Formal-SignWriting/FSW entries) is therefore a *deterministic, zero-token*
source for exactly the two axes ASL-LEX is missing.

This script, for each ASL-LEX sign:
  1. normalises the gloss and finds candidate FSW entries (stage-56 normalisation),
  2. parses the dominant-hand FSW symbol -> candidate HamNoSys ``extfidir``/``palmor``,
  3. parses the movement arrow -> candidate movement *direction*,
  4. OVERRIDES only the orientation/direction fields the stage-57 converter filled
     by ``heuristic``/``default`` (never the deterministic ASL-LEX handshape/
     location), marking the overridden fields provenance ``"fsw"``,
  5. logs handshape disagreements (FSW vs ASL-LEX) for review without overriding.

LICENSE GATE (mandatory — proposal 55 §6)
-----------------------------------------
The SignPuddle *dictionary content* license is UNCONFIRMED. The MIT
``sutton-signwriting`` *tooling* is commercial-OK, but the entries are not.
Therefore this script is **dry-run by default**: it computes and reports coverage
but writes NOTHING under ``data/``. It will write the enriched ``data/American_SL_ASL*.sigml``
only when BOTH ``--write`` is passed AND the human-dropped clearance file
``data/sources/signpuddle/LICENSE_CLEARED`` exists. Without that file it refuses
to write and stays dry-run. (The clarification email is drafted at
``proposals/reports/signpuddle-license-request.md``.)

FSW -> HamNoSys MAPPING (the bulk of the work; documented + lossy points)
-------------------------------------------------------------------------
An FSW symbol key is ``S`` + 3 hex (base, 0x100..0x38b) + 1 hex (fill, 0..5) +
1 hex (rotation, 0..15 at 22.5deg). Categories by base (ISWA 2010):
  0x100..0x204 hands     0x205..0x2f6 movement/contact/dynamics
  0x2ff..0x36c head/face 0x36d..0x375 trunk     (rest: limb/loc/punctuation)

Hand FILL -> plane + palm facing (signer's own "expressive" viewpoint):
  fill 0,1,2 = WALL plane (vertical, frontal); fill 3,4,5 = FLOOR plane (horizontal).
  In each plane the three fills are palm-toward / palm-side / palm-away:
    wall  0 -> palm toward signer (i)   1 -> palm to the side (l*)   2 -> palm away (o)
    floor 3 -> palm up (u)              4 -> palm to the side (l*)   5 -> palm down (d)
  (*) the half-fill "side" cases are direction-ambiguous (which side?) -> LOW
      confidence -> NOT used to override (see CONFIDENCE below).

Hand ROTATION -> finger direction (absolute), 22.5deg steps, rot 0 = up (wall) /
out/away (floor); mapped to the nearest 8-way compass (2 steps = 45deg).

The absolute (extfidir, palm-facing) pair is then run through the *same* validated
geometry the stage-57 converter uses (``to_palmor``) to get the relative 8-value
HamNoSys ``palmor`` — guaranteeing enum-valid, stage-57-consistent output.

Movement arrow (base 0x205..0x24f) ROTATION -> straight-move direction; used only
to replace the *direction* of an ASL-LEX straight/arc path (never to invent a path
where ASL-LEX says "none", and never to change the ASL-LEX path *shape*).

KNOWN LOSSY POINTS (carried to the report)
  - 2-D spatial layout: FSW places symbols in a 2-D signbox; absolute body
    location is not recovered here (location stays ASL-LEX-deterministic).
  - exact rotation: the 0-7 vs 8-15 rotation parity can select a mirrored / other-
    plane variant; collapsed to an 8-way compass here.
  - half-fill "side": palm-side does not say which side -> treated low-confidence.
  - dominant-hand disambiguation: two-handed signs list both hands; the dominant
    one is picked by signbox x-coordinate (signer's-own view) -> imperfect.
  - non-manuals (head/face/trunk symbols) are ignored.

CONFIDENCE / OVERRIDE POLICY
  An orientation override fires only when (a) a normalised gloss match exists,
  (b) a dominant hand glyph is parseable, and (c) the fill is a *cardinal* palm
  fill (toward/away/up/down) — i.e. NOT a half-fill side case. Two-handed signs
  are accepted but flagged lower confidence. A movement-direction override fires
  only when a movement arrow is parseable AND the ASL-LEX path shape is straight/arc.

Usage:
    python3 scripts/asl_fsw_enrich.py --dry-run        # report coverage, write nothing
    python3 scripts/asl_fsw_enrich.py --write          # write ONLY if LICENSE_CLEARED exists
    python3 scripts/asl_fsw_enrich.py --json /tmp/x.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the stage-57 deterministic converter: its row->fields pipeline, the
# validated absolute->relative palm geometry, and the enum sets.
from scripts.asl_lex_to_sigml import (  # noqa: E402
    EXTFIDIR,
    PALMOR,
    col,
    convert_row,
    emit_sigml,
    load_known_tags,
    to_palmor,
)

ASL_LEX_CSV = REPO_ROOT / "data" / "sources" / "asl_lex" / "signdata.csv"
POC_CSV = REPO_ROOT / "data" / "asl_lex_poc_seed.csv"
SPML = REPO_ROOT / "data" / "sources" / "signpuddle" / "sgn4.spml"
LICENSE_CLEARED = REPO_ROOT / "data" / "sources" / "signpuddle" / "LICENSE_CLEARED"
DEFAULT_OUT = REPO_ROOT / "data" / "American_SL_ASL.sigml"

# ---------------------------------------------------------------------------
# Gloss normalisation — mirrors extension/panel.js glossBase() and the stage-56
# feasibility script: lowercase, drop parentheticals + trailing disambiguators,
# punctuation/underscores -> space (new_york -> "new york", candy_1 -> "candy").
# ---------------------------------------------------------------------------
def gloss_base(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"#\d+$", "", s)
    s = re.sub(r"_\d+[a-z]?$", "", s)
    s = re.sub(r"\d+[a-z]?\^?$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# FSW parser. A sign string is:
#   [A <sort-symbols>] <box M|B|L|R> <cx>x<cy> ( S<key> <x>x<y> )*
# Each placed symbol = S + 3 hex base + 1 hex fill(0-5) + 1 hex rot(0-15) + x x y.
# We keep the coordinates (needed for dominant-hand disambiguation) — unlike the
# feasibility prototype, which dropped them.
# ---------------------------------------------------------------------------
SYM_RE = re.compile(r"S([0-9a-f]{3})([0-5])([0-9a-f])(\d{3})x(\d{3})")
SORT_PREFIX_RE = re.compile(r"^A(?:S[0-9a-f]{5})+")

HAND_LO, HAND_HI = 0x100, 0x204
# Only *straight* movement arrows carry a single decodable direction (rotation):
#   wall-plane straight  0x221..0x22e  -> vertical compass (u/d/l/r...)
#   floor-plane straight 0x22f..0x238  -> horizontal compass (o/i/l/r...)
# Curves (0x239..) and circles (0x265..) have no single direction and are skipped
# (their *shape* already comes from ASL-LEX, so FSW adds nothing for them).
ARROW_WALL_LO, ARROW_WALL_HI = 0x221, 0x22E
ARROW_FLOOR_LO, ARROW_FLOOR_HI = 0x22F, 0x238
# Contact symbols (touch/grasp/strike/brush/rub) -> confirm hamlrat.
CONTACT_LO, CONTACT_HI = 0x205, 0x215


class Symbol:
    __slots__ = ("base", "fill", "rot", "x", "y")

    def __init__(self, base: int, fill: int, rot: int, x: int, y: int):
        self.base, self.fill, self.rot, self.x, self.y = base, fill, rot, x, y

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Sym(0x{self.base:03x} f{self.fill} r{self.rot} @{self.x}x{self.y})"


def parse_fsw(fsw: str):
    """Parse an FSW string into a dict, or None if it has no placed symbols.

    Returns {"hands": [Symbol...], "arrows": [Symbol...], "contacts": [Symbol...],
             "n_hands": int, "n_arrows": int}.
    """
    if not fsw:
        return None
    body = SORT_PREFIX_RE.sub("", fsw.strip())
    syms = [
        Symbol(int(b, 16), int(fl), int(rt, 16), int(x), int(y))
        for b, fl, rt, x, y in SYM_RE.findall(body)
    ]
    if not syms:
        return None
    hands = [s for s in syms if HAND_LO <= s.base <= HAND_HI]
    arrows = [s for s in syms
              if ARROW_WALL_LO <= s.base <= ARROW_WALL_HI
              or ARROW_FLOOR_LO <= s.base <= ARROW_FLOOR_HI]
    contacts = [s for s in syms if CONTACT_LO <= s.base <= CONTACT_HI]
    return {
        "hands": hands,
        "arrows": arrows,
        "contacts": contacts,
        "n_hands": len(hands),
        "n_arrows": len(arrows),
    }


def dominant_hand(parsed: dict):
    """Pick the dominant-hand glyph. One hand -> that hand. Two+ hands -> the one
    with the largest signbox x (signer's-own/expressive view: the right/dominant
    hand sits on the reader's right). Returns (Symbol, confidence_penalty)."""
    hands = parsed["hands"]
    if not hands:
        return None, 0.0
    if len(hands) == 1:
        return hands[0], 0.0
    dom = max(hands, key=lambda s: s.x)
    return dom, 0.15  # two-handed pick is imperfect -> small confidence penalty


# ---------------------------------------------------------------------------
# Orientation decode (the heart of the FSW->HamNoSys mapping).
# ---------------------------------------------------------------------------
# Wall plane: rot 0 = fingers up; advance clockwise in 22.5deg steps.
_WALL8 = ["u", "ur", "r", "dr", "d", "dl", "l", "ul"]
# Floor plane: rot 0 = fingers out/away; "up on the page" reads as out (o).
_FLOOR8 = ["o", "or", "r", "ir", "i", "il", "l", "ol"]


def rot_index8(rot: int) -> int:
    return round((rot % 16) / 2) % 8


def fill_plane(fill: int) -> str:
    return "wall" if fill <= 2 else "floor"


# fill -> absolute palm-facing component + whether it is a cardinal (confident)
# facing. The half-fill "side" cases (1, 4) are ambiguous about which side.
_FILL_PALM: dict[int, tuple[str, bool]] = {
    0: ("i", True),   # wall, palm toward signer
    1: ("l", False),  # wall, palm to the side (ambiguous)
    2: ("o", True),   # wall, palm away from signer
    3: ("u", True),   # floor, palm up
    4: ("l", False),  # floor, palm to the side (ambiguous)
    5: ("d", True),   # floor, palm down
}


def decode_orientation(hand: Symbol) -> dict:
    """Decode a dominant-hand glyph into candidate HamNoSys orientation.

    Returns {extfidir, palmor, palm_abs, plane, cardinal, note}. ``palmor`` is the
    relative 8-value HamNoSys code (via the stage-57 ``to_palmor`` geometry);
    ``cardinal`` is False for the ambiguous half-fill "side" facings.
    """
    plane = fill_plane(hand.fill)
    idx = rot_index8(hand.rot)
    extfidir = _WALL8[idx] if plane == "wall" else _FLOOR8[idx]
    palm_abs, cardinal = _FILL_PALM.get(hand.fill, ("l", False))
    palmor = to_palmor(extfidir, palm_abs)
    note = f"FSW base=0x{hand.base:03x} fill={hand.fill}({plane}) rot={hand.rot}"
    if palmor is None:  # extfidir ~parallel to palm normal -> degenerate
        palmor, cardinal = "l", False
        note += " [degenerate palm/finger -> palmor defaulted]"
    if extfidir not in EXTFIDIR:  # belt-and-braces; both tables are subsets
        extfidir, cardinal = "u", False
    return {
        "extfidir": extfidir,
        "palmor": palmor,
        "palm_abs": palm_abs,
        "plane": plane,
        "cardinal": cardinal,
        "note": note,
    }


# HamNoSys straight-move tags exist only for these 10 directions.
_MOVE_DIRS = {"u", "d", "l", "r", "o", "i", "ul", "ur", "dl", "dr"}


def decode_movement(parsed: dict) -> dict | None:
    """Decode the first straight movement arrow -> candidate hammove* direction.

    Wall-plane arrows read in the vertical compass (u/d/l/r); floor-plane arrows
    read in the horizontal compass (o/i/l/r). Directions HamNoSys has no straight
    tag for (e.g. floor diagonals ir/il/or/ol) are dropped (returns None)."""
    if not parsed["arrows"]:
        return None
    arrow = parsed["arrows"][0]
    idx = rot_index8(arrow.rot)
    if ARROW_FLOOR_LO <= arrow.base <= ARROW_FLOOR_HI:
        plane, d = "floor", _FLOOR8[idx]
    else:
        plane, d = "wall", _WALL8[idx]
    if d not in _MOVE_DIRS:
        return None
    return {
        "direction": d,
        "tag": "hammove" + d,
        "note": f"FSW arrow base=0x{arrow.base:03x} fill={arrow.fill}({plane}) rot={arrow.rot}",
    }


# ---------------------------------------------------------------------------
# Handshape disagreement check (LOG ONLY — handshape stays ASL-LEX-deterministic).
#
# We do NOT hardcode an FSW-base -> handshape table (the ISWA base->shape chart
# can't be verified offline, and guessed ranges produce noisy false mismatches).
# Instead the FSW handshape "family" is CALIBRATED empirically from the matched
# corpus: each FSW dominant-hand base is labelled with the HamNoSys handshape
# family it MOST OFTEN co-occurs with across matched ASL-LEX signs. A sign is
# then flagged "disagreement" only when its FSW base is well-supported (>= MIN
# occurrences) and reasonably pure, yet its own ASL-LEX family differs from that
# base's majority family — i.e. the FSW glyph is atypical for the stated shape.
# ---------------------------------------------------------------------------
HS_CALIB_MIN_SUPPORT = 8
HS_CALIB_MIN_PURITY = 0.6

# HamNoSys base handshape tag -> coarse family (mirrors HANDSHAPE_MAP outputs).
_HAM_HS_FAMILY: dict[str, str] = {
    "hamfinger2": "index",
    "hamflathand": "flat",
    "hamfinger2345": "flat",
    "hamfinger23spread": "two_finger",
    "hamfinger23": "two_finger",
    "hamceeall": "cee",
    "hamfist": "fist",
    "hampinchall": "round",
    "hampinch12": "round",
    "hampinch12open": "round",
}


def ham_handshape_family(hs_tags: list[str]) -> str | None:
    return _HAM_HS_FAMILY.get(hs_tags[0]) if hs_tags else None


# ---------------------------------------------------------------------------
# FSW dictionary load: normalised gloss -> list of raw FSW strings.
# ---------------------------------------------------------------------------
def load_fsw_dict(path: Path = SPML) -> dict[str, list[str]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
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


def _slot(fields: list[dict], name: str) -> dict | None:
    return next((f for f in fields if f["slot"] == name), None)


# Canonical HamNoSys slot order for reassembly (matches asl_lex_to_sigml).
_SLOT_ORDER = ["symmetry", "handshape", "orientation", "location", "movement"]


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def enrich_row(row: dict[str, str], fsw_dict: dict[str, list[str]],
               hs_calibration: dict[int, str] | None = None,
               min_confidence: str = "medium") -> dict:
    """Run stage-57 conversion, then override gap fields from FSW where confident.

    Returns a record describing the baseline, the FSW pick, what was overridden,
    any handshape disagreement, and the final reassembled tag list.
    """
    gloss, _base_tags, fields, review = convert_row(row)
    gb = gloss_base(gloss)
    rec: dict = {
        "gloss": gloss,
        "gloss_base": gb,
        "matched": False,
        "fsw": None,
        "overrode_orientation": False,
        "overrode_movement": False,
        "orientation_confidence": None,
        "handshape_disagreement": None,
        "contact_confirmed": False,
        "review": list(review),
        "fields": fields,
        "anchored": is_deterministic_anchor(row),
    }

    candidates = fsw_dict.get(gb) or []
    # Choose the FSW candidate with a parseable dominant hand (first wins).
    chosen = parsed = dom = None
    dom_penalty = 0.0
    for fsw in candidates:
        p = parse_fsw(fsw)
        if p and p["hands"]:
            d, pen = dominant_hand(p)
            if d is not None:
                chosen, parsed, dom, dom_penalty = fsw, p, d, pen
                break

    if candidates:
        rec["matched"] = True
        rec["fsw"] = (chosen or candidates[0])[:48]

    if dom is None:  # no match, or matched but no usable hand glyph
        rec["tags"] = _reassemble(fields)
        return rec

    # ---- orientation override (gap field only; cardinal fills only) ----------
    ori = _slot(fields, "orientation")
    deco = decode_orientation(dom)
    conf = "high"
    if dom_penalty:
        conf = "medium"  # two-handed dominant-hand pick
    if not deco["cardinal"]:
        conf = "low"     # half-fill side -> ambiguous, do not trust
    rec["orientation_confidence"] = conf
    # Only override a *gap* field (heuristic/default), never a curator override,
    # and only when confidence meets the requested floor (default: medium).
    if (ori is not None and ori["source"] in {"heuristic", "default"}
            and _CONF_RANK[conf] >= _CONF_RANK[min_confidence]):
        ori["tags"] = ["hamextfinger" + deco["extfidir"], "hampalm" + deco["palmor"]]
        ori["source"] = "fsw"
        ori["note"] = deco["note"] + f" -> extfidir={deco['extfidir']} palmor={deco['palmor']} ({conf})"
        rec["overrode_orientation"] = True
        rec["review"].append(f"orientation from FSW ({conf} confidence): {deco['note']}")

    # ---- movement-direction override (replace direction of a straight/arc) ----
    mv = _slot(fields, "movement")
    mvd = decode_movement(parsed)
    if mv is not None and mvd is not None and mv["source"] in {"heuristic", "default"}:
        # Only touch a straight move (single hammove* tag, optional repeat).
        core = [t for t in mv["tags"] if t not in ("hamrepeatfromstart",)]
        if len(core) == 1 and core[0].startswith("hammove"):
            new_tags = [mvd["tag"]]
            if "hamrepeatfromstart" in mv["tags"]:
                new_tags.append("hamrepeatfromstart")
            mv["tags"] = new_tags
            mv["source"] = "fsw"
            mv["note"] = mvd["note"] + f" -> {mvd['tag']}"
            rec["overrode_movement"] = True
            rec["review"].append(f"movement direction from FSW: {mvd['note']}")

    # ---- contact confirmation (does not change tags; ASL-LEX contact trusted) -
    if parsed["contacts"]:
        loc = _slot(fields, "location")
        rec["contact_confirmed"] = bool(loc and "hamlrat" in loc["tags"])

    # ---- handshape disagreement (log only; never override) -------------------
    if hs_calibration is not None:
        hs = _slot(fields, "handshape")
        fam_ham = ham_handshape_family(hs["tags"]) if hs else None
        fam_fsw = hs_calibration.get(dom.base)
        if fam_ham and fam_fsw and fam_ham != fam_fsw:
            rec["handshape_disagreement"] = {"asl_lex": fam_ham, "fsw": fam_fsw,
                                             "fsw_base": f"0x{dom.base:03x}"}
            rec["review"].append(
                f"handshape disagreement: ASL-LEX={fam_ham} vs FSW majority={fam_fsw} "
                f"(0x{dom.base:03x}); not overridden")

    rec["tags"] = _reassemble(fields)
    return rec


def _reassemble(fields: list[dict]) -> list[str]:
    """Flatten field tags back into canonical HamNoSys slot order."""
    tags: list[str] = []
    for name in _SLOT_ORDER:
        f = _slot(fields, name)
        if f:
            tags.extend(f["tags"])
    return tags


# ---------------------------------------------------------------------------
# Load ASL-LEX rows (real corpus if present, else the PoC seed).
# ---------------------------------------------------------------------------
def load_rows(in_path: Path | None) -> tuple[list[dict], Path]:
    import csv
    if in_path is None:
        in_path = ASL_LEX_CSV if ASL_LEX_CSV.exists() else POC_CSV
    with in_path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f)), in_path


def _gloss_of(row: dict) -> str:
    return (col(row, "Gloss") or row.get("EntryID") or row.get("LemmaID") or "").strip()


def is_deterministic_anchor(row: dict) -> bool:
    """Stage-57 'bucket A': orientation is strongly anchored when the hand
    contacts a specific body/hand location (mirrors the stage-56 feasibility
    definition). Such signs already have a defensible deterministic orientation
    even without FSW, so they are NOT residual for the LLM stage."""
    contact = (col(row, "Contact") or "").strip() in {"1", "y", "yes", "true", "+"}
    major = (col(row, "MajorLocation") or "").strip().lower()
    minor = (col(row, "MinorLocation") or "").strip().lower()
    return contact and major in {"head", "body", "arm", "hand"} and minor not in {"", "other"}


def build_hs_calibration(rows: list[dict], fsw_dict: dict[str, list[str]]) -> dict[int, str]:
    """Empirical FSW-base -> HamNoSys handshape family from matched pairs.

    For every matched sign with a parseable dominant hand, tally that hand's FSW
    base against the sign's ASL-LEX handshape family. Assign each base the family
    that is BOTH well-supported (>= HS_CALIB_MIN_SUPPORT) and dominant
    (>= HS_CALIB_MIN_PURITY of occurrences). Bases that fail either gate are left
    unlabelled (no disagreement will ever be raised for them)."""
    tally: dict[int, Counter] = {}
    for row in rows:
        gloss = _gloss_of(row)
        if not gloss:
            continue
        cands = fsw_dict.get(gloss_base(gloss)) or []
        for fsw in cands:
            p = parse_fsw(fsw)
            if p and p["hands"]:
                dom, _ = dominant_hand(p)
                _g, _t, fields, _r = convert_row(row)
                fam = ham_handshape_family((_slot(fields, "handshape") or {}).get("tags", []))
                if fam:
                    tally.setdefault(dom.base, Counter())[fam] += 1
                break
    calib: dict[int, str] = {}
    for base, c in tally.items():
        total = sum(c.values())
        fam, n = c.most_common(1)[0]
        if total >= HS_CALIB_MIN_SUPPORT and n / total >= HS_CALIB_MIN_PURITY:
            calib[base] = fam
    return calib


def run(rows: list[dict], fsw_dict: dict[str, list[str]],
        min_confidence: str = "medium") -> tuple[list[dict], dict]:
    hs_calibration = build_hs_calibration(rows, fsw_dict)
    records: list[dict] = []
    for row in rows:
        if not _gloss_of(row):
            continue
        records.append(enrich_row(row, fsw_dict, hs_calibration, min_confidence))

    n = len(records)
    matched = sum(1 for r in records if r["matched"])
    hand_parseable = sum(1 for r in records if r["orientation_confidence"])
    ori_over = sum(1 for r in records if r["overrode_orientation"])
    mv_over = sum(1 for r in records if r["overrode_movement"])
    # Confidence over signs with a parseable dominant hand; "low" = side-fill,
    # conservatively skipped (not overridden).
    conf = Counter(r["orientation_confidence"] for r in records if r["orientation_confidence"])
    disagreements = [r for r in records if r["handshape_disagreement"]]
    anchored = sum(1 for r in records if r["anchored"])
    # Auto-orientation coverage = stage-57 deterministic anchor (A) OR stage-58
    # FSW override (B). Residual (-> gated LLM stage 60) = neither.
    covered = sum(1 for r in records if r["overrode_orientation"] or r["anchored"])
    residual = [r for r in records if not r["overrode_orientation"] and not r["anchored"]]

    stats = {
        "n_signs": n,
        "min_confidence": min_confidence,
        "matched_gloss": matched,
        "matched_pct": round(100 * matched / n, 1) if n else 0.0,
        "fsw_hand_parseable": hand_parseable,
        "fsw_hand_parseable_pct": round(100 * hand_parseable / n, 1) if n else 0.0,
        "orientation_overridden": ori_over,
        "orientation_overridden_pct": round(100 * ori_over / n, 1) if n else 0.0,
        "orientation_skipped_low_conf": conf.get("low", 0),
        "movement_dir_overridden": mv_over,
        "movement_dir_overridden_pct": round(100 * mv_over / n, 1) if n else 0.0,
        "confidence_breakdown": dict(conf),
        "stage57_anchored": anchored,
        "stage57_anchored_pct": round(100 * anchored / n, 1) if n else 0.0,
        "auto_orientation_covered": covered,
        "auto_orientation_covered_pct": round(100 * covered / n, 1) if n else 0.0,
        "hs_calibrated_bases": len(hs_calibration),
        "handshape_disagreements": len(disagreements),
        "residual_needing_stage60": len(residual),
        "residual_pct": round(100 * len(residual) / n, 1) if n else 0.0,
    }
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default=None,
                    help="ASL-LEX CSV (default: data/sources/asl_lex/signdata.csv, else PoC seed)")
    ap.add_argument("--out", dest="outfile", default=str(DEFAULT_OUT),
                    help="enriched SiGML output (written ONLY with --write + LICENSE_CLEARED)")
    ap.add_argument("--spml", default=str(SPML), help="SignPuddle FSW dictionary")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="(default) compute + report coverage, write nothing")
    ap.add_argument("--write", action="store_true",
                    help="attempt to write enriched data/ (hard-gated on LICENSE_CLEARED)")
    ap.add_argument("--min-confidence", dest="min_confidence",
                    choices=["high", "medium"], default="medium",
                    help="orientation-override confidence floor: 'high' = one-handed cardinal "
                         "fills only; 'medium' (default) also accepts two-handed picks")
    ap.add_argument("--sample", type=int, default=20, help="how many enriched samples to print")
    ap.add_argument("--json", default=None, help="also dump full stats+samples JSON here")
    args = ap.parse_args()

    spml_path = Path(args.spml)
    if not spml_path.exists():
        raise SystemExit(f"FSW dictionary not found: {spml_path} (gitignored; re-download per report §1)")

    in_path = Path(args.infile) if args.infile else None
    rows, used_in = load_rows(in_path)
    fsw_dict = load_fsw_dict(spml_path)
    records, stats = run(rows, fsw_dict, args.min_confidence)

    # ---------------------------------------------------------------- report
    rel_in = used_in.relative_to(REPO_ROOT) if used_in.is_relative_to(REPO_ROOT) else used_in
    print(f"===== ASL-LEX <-> SignPuddle FSW cross-reference =====")
    print(f"input: {rel_in}  ({stats['n_signs']} signs)")
    print(f"FSW dict: {len(fsw_dict)} distinct normalised glosses")
    print(f"override confidence floor: {stats['min_confidence']}")
    print(f"gloss match:            {stats['matched_gloss']:5d} ({stats['matched_pct']}%)")
    print(f"FSW dominant hand parseable:{stats['fsw_hand_parseable']:4d} ({stats['fsw_hand_parseable_pct']}%)")
    print(f"orientation OVERRIDDEN:  {stats['orientation_overridden']:5d} ({stats['orientation_overridden_pct']}%)  "
          f"confidence={stats['confidence_breakdown']}")
    print(f"  (low-confidence side-fills skipped: {stats['orientation_skipped_low_conf']})")
    print(f"movement dir overridden: {stats['movement_dir_overridden']:4d} ({stats['movement_dir_overridden_pct']}%)")
    print(f"handshape disagreements: {stats['handshape_disagreements']:4d} "
          f"(over {stats['hs_calibrated_bases']} calibrated FSW bases)")
    print(f"stage-57 anchored (A):   {stats['stage57_anchored']:5d} ({stats['stage57_anchored_pct']}%)")
    print(f"AUTO orientation (A+FSW):{stats['auto_orientation_covered']:5d} ({stats['auto_orientation_covered_pct']}%)")
    print(f"residual (-> stage 60):  {stats['residual_needing_stage60']:5d} ({stats['residual_pct']}%)")
    print("\nsample enriched signs (gloss | ext/palm | conf | fsw):")
    shown = 0
    for r in records:
        if r["overrode_orientation"] and shown < args.sample:
            ori = _slot(r["fields"], "orientation")
            print(f"  {r['gloss']:18s} {'/'.join(t.replace('hamextfinger','ext=').replace('hampalm','palm=') for t in ori['tags'])}"
                  f"  {r['orientation_confidence']:6s}  {r['fsw']}")
            shown += 1

    if args.json:
        blob = {"stats": stats,
                "samples": [
                    {"gloss": r["gloss"], "tags": r["tags"], "fsw": r["fsw"],
                     "confidence": r["orientation_confidence"],
                     "overrode_orientation": r["overrode_orientation"],
                     "overrode_movement": r["overrode_movement"],
                     "handshape_disagreement": r["handshape_disagreement"]}
                    for r in records if r["matched"]][:200]}
        Path(args.json).write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    # ------------------------------------------------------- LICENSE GATE
    if not args.write:
        print("\n[dry-run] default mode — NO data/ files written. "
              "Pass --write (and clear the license) to emit enriched SiGML.")
        return

    if not LICENSE_CLEARED.exists():
        print(f"\n[REFUSED] --write requested but the license gate is CLOSED:\n"
              f"  missing {LICENSE_CLEARED.relative_to(REPO_ROOT)}\n"
              f"  SignPuddle dictionary-content license is UNCONFIRMED (proposal 55 §6).\n"
              f"  A human must drop that clearance file once the maintainers confirm reuse.\n"
              f"  See proposals/reports/signpuddle-license-request.md. Staying dry-run; wrote nothing.")
        return

    # License cleared: emit enriched SiGML + meta with provenance "fsw", validate.
    print(f"\n[LICENSE_CLEARED present] writing enriched SiGML...")
    _write_enriched(rows, records, Path(args.outfile), used_in)


def _write_enriched(rows: list[dict], records: list[dict], out_path: Path, used_in: Path) -> None:
    """Emit enriched SiGML using the stage-57 emitter shape, then fold in the
    FSW-overridden tags + provenance and validate against the CWASA oracle."""
    from xml.sax.saxutils import quoteattr

    # Build per-sign XML from the (possibly FSW-enriched) reassembled tags.
    signs_xml: list[str] = []
    prov: dict[str, int] = {"asl_lex": 0, "heuristic": 0, "default": 0, "override": 0, "fsw": 0}
    for r in records:
        if not r.get("tags") or not r["gloss"]:
            continue
        body = "\n      ".join(f"<{t} />" for t in r["tags"])
        signs_xml.append(
            f"  <hns_sign gloss={quoteattr(r['gloss'])}>\n"
            f"    <hamnosys_manual>\n      {body}\n    </hamnosys_manual>\n"
            f"  </hns_sign>"
        )
        for f in r.get("fields", []):
            prov[f["source"]] = prov.get(f["source"], 0) + len(f["tags"])

    # Validate every emitted tag is CWASA-renderable.
    known = load_known_tags() | {"hns_sign", "hamnosys_manual", "sigml_collection"}
    bad = sorted({t for r in records for t in r.get("tags", []) if t not in known})
    if bad:
        raise SystemExit(f"VALIDATION FAILED — tags not in CWASA tokenNameMap: {bad}")

    doc = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<sigml_collection language="American_SL_ASL" count="{len(signs_xml)}">\n\n'
        + "\n\n".join(signs_xml)
        + "\n\n</sigml_collection>\n"
    )
    out_path.write_text(doc, encoding="utf-8")

    total = sum(prov.values()) or 1
    fsw_signs = sum(1 for r in records if r["overrode_orientation"] or r["overrode_movement"])
    meta = {
        "version": 1,
        "language": "asl",
        "iso_code": "ase",
        "display_name": "American Sign Language",
        "source": "ASL-LEX 2.0 phonological coding (Sehyr et al. 2021) + SignPuddle ASL FSW "
                  "cross-reference (sutton-signwriting/ISWA-2010) for orientation/direction",
        "source_kind": "seed",
        "license": "ASL-LEX: CC BY-NC 4.0. SignPuddle FSW content: reuse cleared via "
                   "data/sources/signpuddle/LICENSE_CLEARED (see signpuddle-license-request.md).",
        "data_completeness": "seed",
        "accepts_first_contributions": True,
        "review_required": True,
        "conversion_note": "Handshape/location/symmetry/movement-shape deterministic from ASL-LEX "
                           "(provenance asl_lex). Orientation/movement-direction filled from SignWriting/FSW "
                           "where a confident match exists (provenance fsw), else stage-57 heuristics. "
                           "FSW decode is lossy (rotation collapsed to 8-way, 2-D layout dropped, "
                           "half-fill side cases skipped). NOT Deaf-reviewed.",
        "provenance_summary": {
            "tags_by_source": prov,
            "tags_total": sum(prov.values()),
            "share": {k: round(v / total, 4) for k, v in prov.items()},
            "signs_enriched_from_fsw": fsw_signs,
        },
        "sigml_file": out_path.name,
        "default_review": {
            "deaf_native_reviewed": False,
            "reviewer_count": 0,
            "reviewer_language_match": False,
            "review_source": None,
            "last_reviewed": None,
            "notes": "machine-converted seed; orientation/direction FSW-derived where possible, "
                     "still UNVERIFIED; awaiting Deaf-native review",
        },
        "sign_count": len(signs_xml),
        "signs": {},
    }
    meta_path = out_path.with_name(out_path.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[validate] OK — all emitted tags CWASA-renderable")
    print(f"wrote {out_path}  ({meta['sign_count']} signs; {fsw_signs} FSW-enriched)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
