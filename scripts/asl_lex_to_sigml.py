#!/usr/bin/env python3
"""Convert ASL-LEX-style phonological rows into Kozha H-SiGML.

This is the proof-of-concept converter for the ASL lexicon-sourcing task
(see ``notes/asl-lexicon-sourcing.md``). It takes a CSV whose columns mirror
the ASL-LEX 2.0 phonological coding scheme (Sehyr, Caselli, Cohen-Goldberg &
Emmorey 2021) and emits one ``<hns_sign>`` per row in the exact shape the rest
of ``data/*.sigml`` uses, validated against CWASA's ``tokenNameMap`` so the
JASigning/CWASA avatar can render every emitted tag.

WHAT CONVERTS DETERMINISTICALLY (faithful to the source coding, provenance
``asl_lex``):
  - handshape (+ thumb/flexion modifiers)   <- Handshape / SelectedFingers / Flexion / ThumbPosition
  - location                                 <- MajorLocation / MinorLocation
  - two-handed symmetry operator             <- SignType
  - movement *shape* and repetition          <- PathMovement / RepeatedMovement

WHAT THE SOURCE DOES NOT ENCODE -> filled by rule-based heuristics (provenance
``heuristic``), or as a last resort a constant (provenance ``default``):
  - extended-finger direction (orientation)  -> ORIENTATION_HEURISTICS
  - palm orientation                         -> ORIENTATION_HEURISTICS (relative-to-finger frame)
  - movement *direction*                     -> infer_movement (start location + SignType)

The heuristics encode well-known regularities of ASL citation form (e.g. a hand
at the chin points up with the palm toward the signer; a hand articulated on the
non-dominant base hand points outward palm-down). They are *plausible*, not
measured: ASL-LEX carries no orientation/direction column at all (verified over
the real 2,723-sign corpus). Every emitted parameter is therefore tagged with its
provenance (``asl_lex`` / ``heuristic`` / ``default``; ``override`` when a curator
column supplies it) so downstream stages know what is safe to override.

Because orientation and movement direction are never *measured* by ASL-LEX, NO
sign produced here may be marked ``deaf_native_reviewed``. Output is written with
``data_completeness: "seed"``.

Usage:
    python3 scripts/asl_lex_to_sigml.py \
        --in data/asl_lex_poc_seed.csv \
        --out data/American_SL_ASL.sigml \
        [--validate]            # fail if any emitted tag is unknown to CWASA
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Handshape:  ASL-LEX handshape label -> (base HamNoSys handshape, [modifiers])
#
# HamNoSys ships only 12 base handshapes; ASL uses ~40-50. The common core maps
# with thumb/flexion modifiers. Shapes with no faithful HamNoSys analogue
# (Y, ILY, I, 8, ...) are approximated and flagged `approx=True` so the caller
# can route them to review. This table is the single place to extend coverage.
# ---------------------------------------------------------------------------
HANDSHAPE_MAP: dict[str, tuple[str, list[str], bool]] = {
    # label:            (base,                [modifiers],                 approx)
    "1":                ("hamfinger2",        [],                          False),
    "G":                ("hamfinger2",        [],                          False),
    "D":                ("hamfinger2",        [],                          False),
    "5":                ("hamflathand",       ["hamthumboutmod"],          False),
    "4":                ("hamfinger2345",     [],                          False),
    "B":                ("hamflathand",       [],                          False),
    "B_flat":           ("hamflathand",       [],                          False),
    "open_B":           ("hamflathand",       ["hamthumboutmod"],          False),
    "bent_B":           ("hamflathand",       ["hamfingerbendmod"],        False),
    "flat_B":           ("hamflathand",       [],                          False),
    "A":                ("hamfist",           ["hamthumboutmod"],          False),
    "S":                ("hamfist",           [],                          False),
    "fist":             ("hamfist",           [],                          False),
    "C":                ("hamceeall",         [],                          False),
    "O":                ("hampinchall",       [],                          False),
    "flat_O":           ("hampinchall",       [],                          False),
    "baby_O":           ("hampinch12",        [],                          False),
    "F":                ("hampinch12open",    [],                          False),
    "L":                ("hamfinger2",        ["hamthumboutmod"],          False),
    "V":                ("hamfinger23spread", [],                          False),
    "2":                ("hamfinger23spread", [],                          False),
    "U":                ("hamfinger23",       [],                          False),
    "H":                ("hamfinger23",       [],                          False),
    "K":                ("hamfinger23spread", ["hamthumboutmod"],          True),
    "P":                ("hamfinger23spread", ["hamthumboutmod"],          True),
    "W":                ("hamfinger23spread", ["hamthumboutmod"],          True),
    "3":                ("hamfinger23spread", ["hamthumboutmod"],          True),
    "X":                ("hamfinger2",        ["hamfingerhookmod"],        False),
    "R":                ("hamfinger23",       [],                          True),
    "curved_5":         ("hamfinger2345",     ["hamfingerbendmod"],        False),
    "claw_5":           ("hamfinger2345",     ["hamfingerbendmod"],        False),
    "claw":             ("hamceeall",         ["hamfingerbendmod"],        True),
    "Y":                ("hamfist",           ["hamthumboutmod"],          True),
    "ILY":              ("hamfinger2",        ["hamthumboutmod"],          True),
    "I":                ("hamfist",           [],                          True),
    "8":                ("hampinch12open",    [],                          True),
    "open_8":           ("hampinch12open",    [],                          True),
    # --- full ASL-LEX 2.0 inventory (58 labels). The faithful/approx split
    #     below mirrors scripts/asl_feasibility_analysis.py::FAITHFUL_HS so the
    #     converter's approx count matches the stage-56 feasibility report. ---
    # index family -> hamfinger2 (+ thumb/bend modifiers), all faithful
    "flat_1":           ("hamfinger2",        [],                          False),
    "curved_1":         ("hamfinger2",        ["hamfingerbendmod"],        False),
    "bent_1":           ("hamfinger2",        ["hamfingerbendmod"],        False),
    "bent_l":           ("hamfinger2",        ["hamthumboutmod", "hamfingerbendmod"], False),
    "curved_l":         ("hamfinger2",        ["hamthumboutmod", "hamfingerbendmod"], False),
    # flat hand -> hamflathand, faithful
    "closed_b":         ("hamflathand",       [],                          False),
    # spread five -> hamflathand / hamfinger2345, faithful (spread approximated)
    "flatspread_5":     ("hamflathand",       ["hamthumboutmod"],          False),
    "stacked_5":        ("hamfinger2345",     [],                          False),
    # index+middle spread -> hamfinger23spread, faithful
    "curved_v":         ("hamfinger23spread", ["hamfingerbendmod"],        False),
    "bent_v":           ("hamfinger23spread", ["hamfingerbendmod"],        False),
    "flat_v":           ("hamfinger23spread", [],                          False),
    # index+middle adjacent -> hamfinger23, faithful
    "flat_h":           ("hamfinger23",       [],                          False),
    "open_h":           ("hamfinger23",       [],                          False),
    "curved_h":         ("hamfinger23",       ["hamfingerbendmod"],        False),
    # open pinch -> hampinch12open, faithful
    "open_f":           ("hampinch12open",    [],                          False),
    # four fingers -> hamfinger2345, faithful
    "flat_4":           ("hamfinger2345",     [],                          False),
    "curved_4":         ("hamfinger2345",     ["hamfingerbendmod"],        False),
    # --- no faithful HamNoSys base: best-effort, flagged approx (-> quarantine) ---
    "flat_ily":         ("hamfinger2",        ["hamthumboutmod"],          True),
    "horns":            ("hamfinger2",        ["hamthumboutmod"],          True),
    "flat_horns":       ("hamfinger2",        ["hamthumboutmod"],          True),
    "t":                ("hamfist",           ["hamthumboutmod"],          True),
    "m":                ("hamfist",           [],                          True),
    "flat_m":           ("hamfist",           [],                          True),
    "flat_n":           ("hamfist",           [],                          True),
    "e":                ("hamfist",           [],                          True),
    "open_e":           ("hamfist",           [],                          True),
    "closed_e":         ("hamfist",           [],                          True),
    "spread_e":         ("hamfinger2345",     ["hamfingerbendmod"],        True),
    "spread_open_e":    ("hamfinger2345",     ["hamfingerbendmod"],        True),
    "goody_goody":      ("hamflathand",       [],                          True),
    "7":                ("hampinchall",       [],                          True),
}

# ---------------------------------------------------------------------------
# 2. Location:  ASL-LEX (MajorLocation, MinorLocation) -> HamNoSys location tag
#    Major buckets: Head, Arm, Body, Hand (non-dominant), Neutral.
#    Minor lookup is tried first (more specific), then the major-bucket default.
# ---------------------------------------------------------------------------
MINOR_LOCATION_MAP: dict[str, str] = {
    "forehead":     "hamforehead",
    "top_head":     "hamheadtop",
    "eyes":         "hameyes",
    "eye":          "hameyes",
    "eyebrows":     "hameyebrows",
    "nose":         "hamnose",
    "cheek":        "hamcheek",
    "lips":         "hamlips",
    "mouth":        "hamlips",
    "chin":         "hamchin",
    "ear":          "hamear",
    "neck":         "hamneck",
    "chest":        "hamchest",
    "torso":        "hamchest",
    "stomach":      "hamstomach",
    "abdomen":      "hamstomach",
    "shoulder":     "hamshoulders",
    "upper_arm":    "hamUpperarm",
    "lower_arm":    "hamlowerarm",
    "elbow":        "hamelbow",
    "wrist":        "hamwristback",
    "palm":         "hampalm",
    "back_of_hand": "hamhandback",
    "fingers":      "hamfingertip",
    "fingertip":    "hamfingertip",
}
MAJOR_LOCATION_DEFAULT: dict[str, str] = {
    "head":    "hamhead",
    "arm":     "hamlowerarm",
    "body":    "hamchest",
    "trunk":   "hamchest",
    "hand":    "hampalm",
    "neutral": "hamneutralspace",
}

# ---------------------------------------------------------------------------
# 3. Movement:  ASL-LEX PathMovement -> HamNoSys movement tag(s)
#    ASL-LEX records the path SHAPE but not its DIRECTION. The path shape maps
#    deterministically; the direction is inferred from start location + SignType
#    (see ``infer_movement``). An optional curator ``MovementDirection`` column
#    can override the inference via the table below.
# ---------------------------------------------------------------------------
# Explicit direction override (MovementDirection column) -> straight-move tag.
DIRECTION_MOVE: dict[str, str] = {
    "u": "hammoveu", "d": "hammoved", "l": "hammovel", "r": "hammover",
    "o": "hammoveo", "i": "hammovei", "ul": "hammoveul", "ur": "hammoveur",
    "dl": "hammovedl", "dr": "hammovedr",
}

# Valid orientation enums (reference §4.4 / §4.5). palmor has only 8 values.
EXTFIDIR = {"u","ur","r","dr","d","dl","l","ul","ol","o","or","il","i","ir","ui","di","do","uo"}
PALMOR = {"u","ur","r","dr","d","dl","l","ul"}

SIGNTYPE_SYMMETRY: dict[str, str | None] = {
    "onehanded":                       None,
    "one_handed":                      None,
    "twohanded_symmetrical":           "hamsymmlr",
    "symmetrical":                     "hamsymmlr",
    "symmetricalalternating":          "hamsymmlr",
    "symmetrical_or_alternating":      "hamsymmlr",
    "symmetricaloralternating":        "hamsymmlr",  # real ASL-LEX 2.0 label
    "twohanded_asymmetrical_same":     "hamsymmpar",
    "asymmetrical_same_handshape":     "hamsymmpar",
    "asymmetricalsamehandshape":       "hamsymmpar",  # real ASL-LEX 2.0 label
    "twohanded_asymmetrical_different":"hamsymmpar",
    "asymmetrical_different_handshape":"hamsymmpar",
    "asymmetricaldifferenthandshape":  "hamsymmpar",  # real ASL-LEX 2.0 label
    "dominanceviolation":              "hamsymmpar",  # two-handed, non-dom is a base
    "symmetryviolation":               "hamsymmpar",
}

# SignType values that mean "two hands, symmetric" (mirror across the midline).
SYMMETRIC_SIGNTYPES = {
    "twohanded_symmetrical", "symmetrical", "symmetricalalternating",
    "symmetrical_or_alternating", "symmetricaloralternating",
}


def _norm(s: str) -> str:
    return re.sub(r"[\s/]+", "_", (s or "").strip().lower()).strip("_")


# ---------------------------------------------------------------------------
# Column access: the real ASL-LEX 2.0 export suffixes its phonological columns
# with ``.2.0`` (e.g. ``Handshape.2.0``). The PoC seed uses the bare names. Read
# either so the same converter runs over both.
# ---------------------------------------------------------------------------
def col(row: dict[str, str], name: str) -> str:
    for key in (name, f"{name}.2.0"):
        if key in row and row[key] is not None:
            return row[key]
    return ""


# Case-insensitive handshape lookup (real ASL-LEX labels are lowercase:
# ``open_b``, ``baby_o``, ``curved_5``; the table keys mix case: ``open_B``).
_HANDSHAPE_BY_NORM: dict[str, tuple[str, list[str], bool]] = {
    k.lower(): v for k, v in HANDSHAPE_MAP.items()
}


def lookup_handshape(label: str) -> tuple[tuple[str, list[str], bool], bool]:
    """Return ((base, mods, approx), found)."""
    key = (label or "").strip().lower()
    if key in _HANDSHAPE_BY_NORM:
        return _HANDSHAPE_BY_NORM[key], True
    return ("hamflathand", [], True), False


# ---------------------------------------------------------------------------
# Handshape articulatory class — only the *index/pointing* class changes
# orientation (the extended finger follows the movement/target); the rest take
# their finger direction from the location rule. Kept coarse and testable.
# ---------------------------------------------------------------------------
_HS_CLASS: dict[str, str] = {}
for _hs in ("1", "g", "d", "x", "l", "bent_1", "flat_1", "curved_1", "ily", "flat_ily"):
    _HS_CLASS[_hs] = "index"
for _hs in ("b", "b_flat", "open_b", "bent_b", "flat_b", "closed_b", "4", "flat_4",
            "5", "flatspread_5", "stacked_5"):
    _HS_CLASS[_hs] = "flat"
for _hs in ("curved_5", "claw_5", "claw", "curved_v", "curved_l", "curved_h",
            "curved_4"):
    _HS_CLASS[_hs] = "curved"
for _hs in ("a", "s", "fist", "t", "m", "e", "open_e", "closed_e", "spread_e",
            "spread_open_e", "flat_m", "flat_n"):
    _HS_CLASS[_hs] = "fist"
for _hs in ("c", "open_h", "flat_h", "curved_h"):
    _HS_CLASS[_hs] = "cee"
for _hs in ("o", "flat_o", "baby_o", "f", "open_f", "8", "open_8"):
    _HS_CLASS[_hs] = "pinch"
for _hs in ("v", "2", "u", "h", "k", "p", "w", "3", "r", "flat_v", "bent_v",
            "horns", "flat_horns", "y", "i", "7", "k"):
    _HS_CLASS[_hs] = "spread"


def handshape_class(label: str) -> str:
    return _HS_CLASS.get((label or "").strip().lower(), "other")


# ---------------------------------------------------------------------------
# Palm orientation is RELATIVE to the extended-finger direction and has only the
# 8 vertical-plane codes (u/ur/r/dr/d/dl/l/ul) — no out/in. The heuristic rules
# below are authored in the intuitive *absolute* frame ("at the chin the palm
# faces the signer"); ``to_palmor`` converts an absolute palm-facing direction
# into the relative 8-code palmor, given the finger direction.
#
# Convention (documented): read the palmor "clock" in the frame where the
# fingers are rotated (shortest path) to point toward the signer. The clock-up
# axis is world-out (away from signer) unless the fingers themselves point
# out/in, in which case it is world-up. Verified cardinal cases (fingers up):
# palm-away -> u, palm-toward-signer -> d, palm-left -> l, palm-right -> r.
# ---------------------------------------------------------------------------
_DIRVEC: dict[str, tuple[float, float, float]] = {
    # x: right(+)/left(-)   y: up(+)/down(-)   z: out/away(+)/in/toward-signer(-)
    "u": (0, 1, 0),  "d": (0, -1, 0), "l": (-1, 0, 0), "r": (1, 0, 0),
    "o": (0, 0, 1),  "i": (0, 0, -1),
    "ur": (1, 1, 0), "ul": (-1, 1, 0), "dr": (1, -1, 0), "dl": (-1, -1, 0),
    "uo": (0, 1, 1), "ui": (0, 1, -1), "do": (0, -1, 1), "di": (0, -1, -1),
    "or": (1, 0, 1), "ol": (-1, 0, 1), "ir": (1, 0, -1), "il": (-1, 0, -1),
}
_PALMOR_CLOCK = ["u", "ur", "r", "dr", "d", "dl", "l", "ul"]  # 0,45,...,315 deg


def _v3norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    import math
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def to_palmor(extfidir: str, palm_abs: str) -> str | None:
    """Absolute palm-facing -> relative HamNoSys palmor (8 codes), or None if the
    posture is degenerate (palm-facing ~parallel to fingers => not perpendicular)."""
    import math
    if extfidir not in _DIRVEC or palm_abs not in _DIRVEC:
        return None
    f = _v3norm(_DIRVEC[extfidir])
    p = _v3norm(_DIRVEC[palm_abs])
    if abs(_dot(f, p)) > 0.9:  # palm normal must be ~perpendicular to fingers
        return None
    out = (0.0, 0.0, 1.0)
    up = (0.0, 1.0, 0.0)
    ref = up if abs(_dot(f, out)) > 0.9 else out
    d = _dot(ref, f)
    cu = _v3norm((ref[0] - d * f[0], ref[1] - d * f[1], ref[2] - d * f[2]))
    cr = _cross(f, cu)
    ang = math.degrees(math.atan2(_dot(p, cr), _dot(p, cu))) % 360.0
    return _PALMOR_CLOCK[int((ang + 22.5) // 45) % 8]


# ---------------------------------------------------------------------------
# ORIENTATION_HEURISTICS — well-known ASL citation-form regularities.
# Keyed by normalized MinorLocation (covers both PoC seed labels like ``chin``
# and real ASL-LEX 2.0 labels like ``cheeknose``). Value = (extfidir, absolute
# palm-facing, rationale). Major-location fallback is ORIENTATION_MAJOR_DEFAULT.
# ---------------------------------------------------------------------------
ORIENTATION_HEURISTICS: dict[str, tuple[str, str, str]] = {
    # --- upper face: hand raised, fingers up, palm to the contralateral side ---
    "forehead":   ("u", "l", "upper face: fingers up, palm to the side"),
    "headtop":    ("u", "l", "top of head: fingers up, palm to the side"),
    "top_head":   ("u", "l", "top of head: fingers up, palm to the side"),
    "eyebrows":   ("u", "l", "at the brow: fingers up, palm to the side"),
    "eye":        ("u", "l", "at the eye: fingers up, palm to the side"),
    "eyes":       ("u", "l", "at the eye: fingers up, palm to the side"),
    # --- nose / cheek / ear: fingers up, palm toward the face ---
    "nose":       ("u", "i", "at the nose: fingers up, palm toward the face"),
    "cheek":      ("u", "i", "at the cheek: fingers up, palm toward the face"),
    "cheeknose":  ("u", "i", "cheek/nose: fingers up, palm toward the face"),
    "ear":        ("u", "i", "at the ear: fingers up, palm toward the head"),
    "earlobe":    ("u", "i", "at the ear: fingers up, palm toward the head"),
    # --- mouth / chin: fingers up, palm toward the signer ---
    "chin":       ("u", "i", "at the chin: fingers up, palm toward the signer"),
    "underchin":  ("u", "i", "under the chin: fingers up, palm toward the signer"),
    "mouth":      ("u", "i", "at the mouth: fingers up, palm toward the face"),
    "lips":       ("u", "i", "at the mouth: fingers up, palm toward the face"),
    "upperlip":   ("u", "i", "at the mouth: fingers up, palm toward the face"),
    "teeth":      ("u", "i", "at the mouth: fingers up, palm toward the face"),
    "tongue":     ("u", "i", "at the mouth: fingers up, palm toward the face"),
    # --- neck / upper chest: fingers up, palm toward the body ---
    "neck":       ("u", "i", "at the neck: fingers up, palm toward the body"),
    "clavicle":   ("u", "i", "upper chest: fingers up, palm toward the body"),
    "shoulder":   ("u", "i", "at the shoulder: fingers up, palm toward the body"),
    "shoulders":  ("u", "i", "at the shoulder: fingers up, palm toward the body"),
    "shouldertop":("u", "i", "at the shoulder: fingers up, palm toward the body"),
    # --- torso: fingers up, palm toward the body ---
    "chest":      ("u", "i", "on the torso: fingers up, palm toward the body"),
    "torso":      ("u", "i", "on the torso: fingers up, palm toward the body"),
    "torsotop":   ("u", "i", "upper torso: fingers up, palm toward the body"),
    "torsomid":   ("u", "i", "mid torso: fingers up, palm toward the body"),
    "torsobottom":("u", "i", "lower torso: fingers up, palm toward the body"),
    "stomach":    ("u", "i", "lower torso: fingers up, palm toward the body"),
    "abdomen":    ("u", "i", "lower torso: fingers up, palm toward the body"),
    "belowstomach":("u", "i", "lower torso: fingers up, palm toward the body"),
    "waist":      ("u", "i", "at the waist: fingers up, palm toward the body"),
    "hips":       ("u", "i", "at the hip: fingers up, palm toward the body"),
    # --- non-dominant hand as place of articulation: fingers out, palm down ---
    "palm":       ("o", "d", "on the base palm: fingers out, palm down onto the base hand"),
    "palmback":   ("o", "d", "on the back of the base hand: fingers out, palm down"),
    "back_of_hand":("o", "d", "on the back of the base hand: fingers out, palm down"),
    "handback":   ("o", "d", "on the back of the base hand: fingers out, palm down"),
    "heel":       ("o", "d", "on the heel of the base hand: fingers out, palm down"),
    "fingers":    ("o", "d", "at the base-hand fingers: fingers out, palm down"),
    "fingertip":  ("o", "d", "at the base-hand fingertips: fingers out, palm down"),
    "fingerfront":("o", "d", "at the base-hand fingers: fingers out, palm down"),
    "fingerback": ("o", "d", "at the base-hand fingers: fingers out, palm down"),
    "fingerradial":("o", "d", "at the base-hand fingers: fingers out, palm down"),
    "fingerulnar":("o", "d", "at the base-hand fingers: fingers out, palm down"),
    "wristback":  ("o", "d", "on the base wrist: fingers out, palm down"),
    "wristfront": ("o", "d", "on the base wrist: fingers out, palm down"),
    # --- non-dominant arm: fingers out, palm down ---
    "forearmback":("o", "d", "on the base forearm: fingers out, palm down"),
    "forearmfront":("o", "d", "on the base forearm: fingers out, palm down"),
    "forearmulnar":("o", "d", "on the base forearm: fingers out, palm down"),
    "lowerarm":   ("o", "d", "on the lower arm: fingers out, palm down"),
    "upperarm":   ("o", "d", "on the upper arm: fingers out, palm down"),
    "elbow":      ("o", "d", "at the elbow: fingers out, palm down"),
    "elbowback":  ("o", "d", "at the elbow: fingers out, palm down"),
    # --- signs held just off a location (…Away): keep the location's posture ---
    "headaway":   ("u", "l", "just off the head: fingers up, palm to the side"),
    "handaway":   ("o", "d", "just off the base hand: fingers out, palm down"),
    "bodyaway":   ("o", "i", "just off the body: fingers out, palm toward the body"),
    "armaway":    ("o", "d", "just off the arm: fingers out, palm down"),
    # --- neutral signing space: hand extended forward, palm down ---
    "neutral":    ("o", "d", "neutral space: fingers out, palm down"),
    "":           ("o", "d", "neutral space: fingers out, palm down"),
}

ORIENTATION_MAJOR_DEFAULT: dict[str, tuple[str, str, str]] = {
    "head":    ("u", "i", "head region: fingers up, palm toward the face"),
    "body":    ("u", "i", "body: fingers up, palm toward the body"),
    "trunk":   ("u", "i", "body: fingers up, palm toward the body"),
    "arm":     ("o", "d", "on the arm: fingers out, palm down"),
    "hand":    ("o", "d", "on the base hand: fingers out, palm down"),
    "neutral": ("o", "d", "neutral space: fingers out, palm down"),
}

# Last-resort orientation when no location rule matches (provenance ``default``).
ORIENTATION_FALLBACK = ("u", "l", "no location rule: defaulted, verify")

# Compass directions usable as an extfidir (the 18 valid codes already in EXTFIDIR).


def orientation_for(
    major: str, minor: str, hs_label: str, signtype: str, move_dir: str | None
) -> tuple[str, str, str, str]:
    """Return (extfidir, palmor, source, rationale).

    source is ``heuristic`` when a location/symmetry/handshape rule fired, else
    ``default``. palmor is always one of the 8 valid codes.
    """
    minor_n = _norm(minor)
    major_n = _norm(major)

    if minor_n in ORIENTATION_HEURISTICS:
        extfi, palm_abs, why = ORIENTATION_HEURISTICS[minor_n]
        source = "heuristic"
    elif major_n in ORIENTATION_MAJOR_DEFAULT:
        extfi, palm_abs, why = ORIENTATION_MAJOR_DEFAULT[major_n]
        why = f"{why} (major-location fallback)"
        source = "heuristic"
    else:
        extfi, palm_abs, why = ORIENTATION_FALLBACK
        source = "default"

    notes = [why]

    # Two-handed symmetric in neutral space: palms face each other (the dominant
    # hand faces the midline) rather than down.
    if signtype in SYMMETRIC_SIGNTYPES and (not minor_n or major_n == "neutral"):
        extfi, palm_abs = "o", "l"
        notes = ["two-handed symmetric, neutral space: fingers out, palms facing each other"]
        source = "heuristic"

    # Pointing/index handshapes: the extended finger follows the movement target.
    if handshape_class(hs_label) == "index" and move_dir in _DIRVEC:
        extfi = move_dir
        notes.append("index handshape: fingers follow the movement direction")
        source = "heuristic"

    palmor = to_palmor(extfi, palm_abs)
    if palmor is None:  # degenerate absolute combo -> safe perpendicular default
        palmor = "l"
        notes.append("palm/finger combo degenerate -> palmor defaulted")
    return extfi, palmor, source, "; ".join(notes)


# ---------------------------------------------------------------------------
# Movement direction inference. ASL-LEX gives the path SHAPE (Movement.2.0) but
# never its DIRECTION. Derive a plausible default from the start location and
# SignType: signs leaving a face/head location move outward; body-anchored signs
# move down/out; two-handed symmetric signs move apart (the dominant hand to the
# dominant side, mirrored by hamsymmlr). Where no principled default exists, fall
# back to a downward straight move and flag it ``default`` for review.
# ---------------------------------------------------------------------------
_FACE_MAJORS = {"head"}
_BODY_MAJORS = {"body", "trunk"}
_HAND_MAJORS = {"hand", "arm"}


def infer_straight_direction(
    major: str, minor: str, signtype: str
) -> tuple[str, str, str]:
    """Return (direction_code, source, rationale) for a straight/arc path."""
    major_n = _norm(major)
    if signtype in SYMMETRIC_SIGNTYPES:
        return "r", "heuristic", "two-handed symmetric: hands move apart (dominant to the side)"
    if major_n in _FACE_MAJORS:
        return "o", "heuristic", "leaves a face/head location: moves outward"
    if major_n in _BODY_MAJORS:
        return "o", "heuristic", "body-anchored: moves outward from the body"
    if major_n in _HAND_MAJORS:
        return "o", "heuristic", "articulated on the base hand: moves outward"
    return "d", "default", "neutral one-handed straight path: direction defaulted (down)"


def infer_movement(
    path: str, major: str, minor: str, signtype: str, repeated: bool
) -> tuple[list[str], str, str, str | None]:
    """Return (tags, source, rationale, inferred_direction)."""
    path_n = _norm(path)
    inferred_dir: str | None = None

    if path_n in {"none", ""}:
        tags, source, why = ["hamnomotion"], "asl_lex", "no path movement"
    elif path_n == "straight":
        d, source, why = infer_straight_direction(major, minor, signtype)
        tags = ["hammove" + d]
        inferred_dir = d
        why = f"straight path (asl_lex shape) + inferred direction: {why}"
    elif path_n in {"arc", "curved"}:
        d, source, why = infer_straight_direction(major, minor, signtype)
        # HamNoSys arcs are MODIFIERS on a straight move (cf. DGS "hammovedo hamarcd"):
        # emit the directional move first, then a perpendicular arc curve. A bare
        # <hamarcd/> is ungrammatical and makes Ham4HMLGen emit [object Object].
        curve = {"u": "r", "d": "r", "l": "u", "r": "u"}.get(d, "u")
        tags = ["hammove" + d, "hamarc" + curve]
        inferred_dir = d
        why = f"arc path (asl_lex shape): move {d} + arc curve {curve}; {why}"
    elif path_n in {"circular", "circle"}:
        tags, source, why = ["hamcircleo"], "asl_lex", "circular path (direction not encoded)"
    elif path_n in {"back_and_forth", "backandforth"}:
        tags = ["hammoveo", "hamrepeatfromstart"]
        source, why = "heuristic", "back-and-forth: out-and-in repeat (axis inferred)"
    elif path_n in {"z_shaped", "z-shaped", "zshaped"}:
        tags, source, why = ["hamzigzag"], "asl_lex", "z-shaped path"
    elif path_n in {"x_shaped", "x-shaped", "xshaped"}:
        tags, source, why = ["hamcross"], "asl_lex", "x-shaped path"
    elif path_n == "other":
        tags, source, why = ["hamnomotion"], "default", "unrecognized path shape -> no motion"
    else:
        tags, source, why = ["hamnomotion"], "default", f"unmapped path {path!r} -> no motion"

    if repeated and "hamrepeatfromstart" not in tags:
        tags.append("hamrepeatfromstart")
    return tags, source, why, inferred_dir


def load_known_tags() -> set[str]:
    """CWASA tokenNameMap = the set of tags the avatar will actually render."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.scan_unknown_hns_tags import extract_known_tags
    return extract_known_tags()


_TRUTHY = {"y", "yes", "true", "1", "+"}


def convert_row(row: dict[str, str]) -> tuple[str, list[str], list[dict], list[str]]:
    """Return (gloss, [hamnosys tags], [field provenance records], [review reasons]).

    Each field record is ``{"slot", "tags", "source", "note"}`` where source is
    ``asl_lex`` (deterministic from columns), ``heuristic`` (the rules in this
    file), ``default`` (last-resort), or ``override`` (a curator column).
    Tags are assembled in canonical HamNoSys slot order, but computed in
    dependency order (movement direction feeds index-handshape orientation).
    """
    review: list[str] = []
    fields: list[dict] = []

    # --- raw hand-authored escape hatch (curator RawTags column) ---------
    # For structurally complex signs (e.g. HOUSE = roof + walls) the rule
    # pipeline can't express the shape; a curator supplies the exact ham* tag
    # sequence verbatim. Still validated against CWASA's tokenNameMap.
    raw = (col(row, "RawTags") or "").strip()
    if raw:
        gl = (col(row, "Gloss") or row.get("EntryID") or "").strip()
        rtags = raw.split()
        return gl, rtags, [{"slot": "raw", "tags": list(rtags), "source": "override",
                            "note": "RawTags (hand-authored HamNoSys)"}], ["hand-authored RawTags; verify"]

    signtype = _norm(col(row, "SignType"))

    # --- symmetry (deterministic from SignType) -------------------------
    symm_tags: list[str] = []
    symm = SIGNTYPE_SYMMETRY.get(signtype)
    if symm:
        symm_tags.append(symm)
    fields.append({"slot": "symmetry", "tags": list(symm_tags), "source": "asl_lex",
                   "note": f"SignType={signtype or 'n/a'}"})
    if signtype.startswith(("twohanded_asymmetrical", "asymmetrical")) or \
            signtype in {"asymmetricaldifferenthandshape", "asymmetricalsamehandshape",
                         "dominanceviolation"}:
        review.append("two-handed asymmetrical: non-dominant hand not emitted (needs manual encoding)")

    # --- handshape (deterministic, with modifiers) ----------------------
    hs_label = (col(row, "Handshape") or "").strip()
    (base, mods, approx), found = lookup_handshape(hs_label)
    if not found:
        review.append(f"handshape {hs_label!r} unmapped -> fell back to hamflathand")
    elif approx:
        review.append(f"handshape {hs_label!r} has no faithful HamNoSys base -> approximated")
    hs_tags = [base, *mods]
    flex = _norm(col(row, "Flexion"))
    if flex in {"curved", "flat", "bent"} and "hamfingerbendmod" not in mods:
        hs_tags.append("hamfingerbendmod")
    fields.append({"slot": "handshape", "tags": list(hs_tags), "source": "asl_lex",
                   "note": f"Handshape={hs_label or 'n/a'}" + ("" if found else " (unmapped)")})

    # --- location (deterministic) ---------------------------------------
    minor = _norm(col(row, "MinorLocation"))
    major = _norm(col(row, "MajorLocation"))
    loc = MINOR_LOCATION_MAP.get(minor) or MAJOR_LOCATION_DEFAULT.get(major)
    loc_tags: list[str] = []
    if loc:
        loc_tags.append(loc)
        contact_val = _norm(col(row, "Contact"))
        if contact_val in _TRUTHY:
            loc_tags.append("hamlrat")            # touching the location
        elif contact_val in {"near", "beside", "close", "off"}:
            loc_tags.append("hamlrbeside")        # hovering beside (avoids clipping into the body)
        fields.append({"slot": "location", "tags": list(loc_tags), "source": "asl_lex",
                       "note": f"Major={major or 'n/a'} Minor={minor or 'n/a'}"})
    else:
        review.append(f"location ({major!r}/{minor!r}) unmapped -> omitted")
        fields.append({"slot": "location", "tags": [], "source": "asl_lex",
                       "note": f"Major={major or 'n/a'} Minor={minor or 'n/a'} (unmapped)"})

    # --- movement (shape deterministic; direction inferred) -------------
    repeated = _norm(col(row, "RepeatedMovement")) in _TRUTHY
    override_dir = (col(row, "MovementDirection") or "").strip().lower()
    path_kw = _norm(col(row, "Movement"))
    contact_move = {"cross": "hamcross", "touch": "hamtouch",
                    "interlock": "haminterlock", "brush": "hambrushing"}
    if path_kw in contact_move:            # curator contact movement (e.g. NAME hands cross)
        move_tags = [contact_move[path_kw]]
        if repeated:
            move_tags.append("hamrepeatfromstart")
        move_source, move_note, move_dir = "override", f"contact={path_kw}", ""
    elif override_dir in DIRECTION_MOVE:    # curator straight-direction override
        move_tags = [DIRECTION_MOVE[override_dir]]
        if repeated:
            move_tags.append("hamrepeatfromstart")
        move_source, move_note, move_dir = "override", f"MovementDirection={override_dir}", override_dir
    else:
        # Real ASL-LEX 2.0 names the path-shape column ``Movement`` (-> ``Movement.2.0``);
        # the PoC seed uses ``PathMovement``. Read whichever is present.
        path_shape = col(row, "Movement") or col(row, "PathMovement")
        move_tags, move_source, move_note, move_dir = infer_movement(
            path_shape, major, minor, signtype, repeated)
    # optional movement-size modifier (curator MovementSize column) -> attaches
    # to the preceding movement tag (e.g. EAT = small gentle taps to the mouth).
    size_mod = {"small": "hamsmallmod", "sm": "hamsmallmod",
                "large": "hamlargemod", "big": "hamlargemod", "lg": "hamlargemod"
                }.get(_norm(col(row, "MovementSize")))
    # A size modifier is only grammatical after a directional/circular move
    # (cf. DGS "hammoved hamsmallmod"); never after a bare contact (hamtouch)
    # or hamnomotion — that yields the [object Object] parse error.
    if size_mod and move_tags and move_tags[0].startswith(("hammove", "hamcircle")):
        move_tags.insert(1, size_mod)
    fields.append({"slot": "movement", "tags": list(move_tags), "source": move_source,
                   "note": move_note + (f" +{size_mod}" if size_mod else "")})
    if move_source == "default":
        review.append(f"movement direction: {move_note}; verify")

    # --- orientation (NOT in source -> heuristic, else default) ---------
    ori_extfi = (col(row, "FingerDirection") or "").strip().lower()
    ori_palm = (col(row, "PalmOrientation") or "").strip().lower()
    if ori_extfi in EXTFIDIR and ori_palm in PALMOR:  # curator override
        extfi, palmor, ori_source = ori_extfi, ori_palm, "override"
        ori_note = f"curator FingerDirection={ori_extfi} PalmOrientation={ori_palm}"
    else:
        extfi, palmor, ori_source, ori_note = orientation_for(
            major, minor, hs_label, signtype, move_dir)
    ori_tags = ["hamextfinger" + extfi, "hampalm" + palmor]
    fields.append({"slot": "orientation", "tags": list(ori_tags), "source": ori_source,
                   "note": ori_note})
    if ori_source == "default":
        review.append("orientation: no location rule matched -> defaulted; verify")
    else:
        review.append("orientation is rule-derived, not measured by ASL-LEX -> verify")

    # --- optional handshape change (curator HandshapeEnd column) ---------
    # ASL signs like NO close the selected fingers onto the thumb mid-sign.
    # HamNoSys expresses a handshape change as hamreplace + the new handshape.
    end_tags: list[str] = []
    hs_end_label = (col(row, "HandshapeEnd") or "").strip()
    if hs_end_label:
        (eb, em, _eapprox), efound = lookup_handshape(hs_end_label)
        end_tags = ["hamreplace", eb, *em]
        fields.append({"slot": "handshape_change", "tags": list(end_tags), "source": "override",
                       "note": f"HandshapeEnd={hs_end_label}" + ("" if efound else " (unmapped)")})

    # --- optional orientation change (curator *End columns) --------------
    # Signs like FINISH flip the palm during the movement (palms end facing us).
    pend = (col(row, "PalmOrientationEnd") or "").strip().lower()
    fend = (col(row, "FingerDirectionEnd") or "").strip().lower()
    if pend in PALMOR or fend in EXTFIDIR:
        ne = fend if fend in EXTFIDIR else extfi
        npp = pend if pend in PALMOR else palmor
        oc = ["hamreplace", "hamextfinger" + ne, "hampalm" + npp]
        end_tags += oc
        fields.append({"slot": "orientation_change", "tags": list(oc), "source": "override",
                       "note": f"end FingerDirection={ne} PalmOrientation={npp}"})

    # --- optional explicit non-dominant hand (asymmetric two-handed) -----
    # CWASA renders a visible second hand only when both hands are described:
    #   hamparbegin <dom hs+ori> hamplus hamnondominant <nondom hs+ori> hamparend
    # (cf. DGS "ARM2"). Without this block the non-dominant hand is invisible.
    nd_label = (col(row, "NonDominantHandshape") or "").strip()
    is_asym = signtype.startswith(("twohanded_asymmetrical", "asymmetrical")) or \
        signtype in {"asymmetricaldifferenthandshape", "asymmetricalsamehandshape", "dominanceviolation"}
    twohand_block: list[str] = []
    if is_asym and nd_label:
        if nd_label.lower() == hs_label.lower():
            # Same handshape on both hands: CWASA renders the second hand via
            # hametc. (Verified: an EXPLICIT different handshape after
            # hamnondominant wedges the JASigning renderer, and all 11
            # hamnondominant signs in the DGS corpus use hametc — never an
            # explicit handshape. So we only build the block in this case.)
            nd_extfi = (col(row, "NonDomFingerDirection") or "").strip().lower()
            nd_palm = (col(row, "NonDomPalmOrientation") or "").strip().lower()
            if nd_extfi not in EXTFIDIR:
                nd_extfi = extfi
            if nd_palm not in PALMOR:
                nd_palm = palmor
            nd_ori = ["hamextfinger" + nd_extfi, "hampalm" + nd_palm]
            twohand_block = ["hamparbegin", *hs_tags, *ori_tags, "hamplus",
                             "hamnondominant", "hametc", *nd_ori, "hamparend"]
            fields.append({"slot": "nondominant", "tags": ["hametc", *nd_ori], "source": "override",
                           "note": f"NonDominantHandshape={nd_label} (rendered as hametc)"})
            review = [r for r in review if "non-dominant hand not emitted" not in r]
        else:
            # Different non-dominant handshape is NOT renderable in CWASA's
            # H-SiGML -> emit dominant-only and flag rather than wedge the avatar.
            # Drop any symmetry operator so the dominant hand isn't mirrored into
            # a misleading two-hand parallel (it would show the wrong base hand).
            symm_tags = []
            review.append(
                f"non-dominant hand ({nd_label}) differs from dominant ({hs_label}); "
                "CWASA H-SiGML can't render a different second handshape -> dominant-only")

    # --- assemble in canonical HamNoSys slot order ----------------------
    if twohand_block:
        # parbegin block encodes both hands explicitly — no symmetry operator.
        tags = [*twohand_block, *loc_tags, *move_tags, *end_tags]
    else:
        tags = [*symm_tags, *hs_tags, *ori_tags, *loc_tags, *move_tags, *end_tags]

    gloss = (col(row, "Gloss") or row.get("EntryID") or "").strip()
    return gloss, tags, fields, review


def emit_sigml(rows: list[dict[str, str]], language: str) -> tuple[str, dict, list[dict]]:
    signs_xml: list[str] = []
    audit: list[dict] = []
    prov_tags: dict[str, int] = {"asl_lex": 0, "heuristic": 0, "default": 0, "override": 0}
    for row in rows:
        gloss, tags, fields, review = convert_row(row)
        if not gloss:
            continue
        body = "\n      ".join(f"<{t} />" for t in tags)
        signs_xml.append(
            f"  <hns_sign gloss={quoteattr(gloss)}>\n"
            f"    <hamnosys_manual>\n      {body}\n    </hamnosys_manual>\n"
            f"  </hns_sign>"
        )
        for fld in fields:
            prov_tags[fld["source"]] = prov_tags.get(fld["source"], 0) + len(fld["tags"])
        audit.append({"gloss": gloss, "tags": tags, "fields": fields, "review": review})

    total_tags = sum(prov_tags.values()) or 1
    provenance_summary = {
        "tags_by_source": prov_tags,
        "tags_total": sum(prov_tags.values()),
        "share": {k: round(v / total_tags, 4) for k, v in prov_tags.items()},
        # Of the two axes ASL-LEX never encodes, how many signs got them from a
        # rule (heuristic/override) vs the last-resort constant default.
        "orientation_heuristic_signs": sum(
            1 for s in audit for f in s["fields"]
            if f["slot"] == "orientation" and f["source"] in {"heuristic", "override"}),
        "orientation_default_signs": sum(
            1 for s in audit for f in s["fields"]
            if f["slot"] == "orientation" and f["source"] == "default"),
        "movement_dir_heuristic_signs": sum(
            1 for s in audit for f in s["fields"]
            if f["slot"] == "movement" and f["source"] in {"heuristic", "override"}),
        "movement_dir_default_signs": sum(
            1 for s in audit for f in s["fields"]
            if f["slot"] == "movement" and f["source"] == "default"),
    }
    doc = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<sigml_collection language={quoteattr(language)} count="{len(signs_xml)}">\n\n'
        + "\n\n".join(signs_xml)
        + "\n\n</sigml_collection>\n"
    )
    meta = {
        "version": 1,
        "language": "asl",
        "iso_code": "ase",
        "display_name": "American Sign Language",
        "source": "ASL-LEX 2.0 phonological coding scheme (Sehyr, Caselli, Cohen-Goldberg & Emmorey 2021); "
                  "PoC seed rows hand-curated from citation-form ASL, NOT redistributed ASL-LEX data",
        "source_kind": "seed",
        "license": "CC BY-NC 4.0 (ASL-LEX database, per asl-lex.org; OSF node tagged CC BY 4.0) "
                   "— non-commercial; PoC seed rows are original encodings",
        "data_completeness": "seed",
        "accepts_first_contributions": True,
        "review_required": True,
        "conversion_note": "Handshape/location/symmetry/movement-shape derived deterministically from "
                           "ASL-LEX columns (provenance asl_lex); palm+finger orientation and movement "
                           "direction are rule-derived heuristics (provenance heuristic), NOT measured by "
                           "ASL-LEX and UNVERIFIED. Per-parameter provenance is in provenance_summary.",
        "provenance_summary": provenance_summary,
        "sigml_file": Path(f"American_SL_ASL.sigml").name,
        "default_review": {
            "deaf_native_reviewed": False,
            "reviewer_count": 0,
            "reviewer_language_match": False,
            "review_source": None,
            "last_reviewed": None,
            "notes": "machine-converted seed; orientation/direction are rule-derived heuristics; "
                     "awaiting Deaf-native review",
        },
        "sign_count": len(signs_xml),
        "signs": {},
    }
    return doc, meta, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default="data/asl_lex_poc_seed.csv")
    ap.add_argument("--out", dest="outfile", default="data/American_SL_ASL.sigml")
    ap.add_argument("--language", default="American_SL_ASL")
    ap.add_argument("--validate", action="store_true",
                    help="fail if any emitted tag is unknown to CWASA tokenNameMap")
    ap.add_argument("--audit", dest="auditfile", default=None,
                    help="also write a per-sign, per-field provenance audit JSON here")
    args = ap.parse_args()

    in_path = (REPO_ROOT / args.infile) if not Path(args.infile).is_absolute() else Path(args.infile)
    out_path = (REPO_ROOT / args.outfile) if not Path(args.outfile).is_absolute() else Path(args.outfile)

    # The real ASL-LEX export carries a few Latin-1 bytes; decode leniently so
    # the bulk run never dies on a stray byte (matches asl_feasibility_analysis).
    with in_path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    doc, meta, audit = emit_sigml(rows, args.language)

    if args.validate:
        known = load_known_tags() | {"hns_sign", "hamnosys_manual", "sigml_collection"}
        bad = sorted({t for s in audit for t in s["tags"] if t not in known})
        if bad:
            raise SystemExit(f"VALIDATION FAILED — tags not in CWASA tokenNameMap: {bad}")
        print(f"[validate] OK — all {sum(len(s['tags']) for s in audit)} tag uses are CWASA-renderable")

    out_path.write_text(doc, encoding="utf-8")
    meta_path = out_path.with_name(out_path.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.auditfile:
        audit_path = (REPO_ROOT / args.auditfile) if not Path(args.auditfile).is_absolute() \
            else Path(args.auditfile)
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {audit_path}")

    ps = meta["provenance_summary"]
    n_review = sum(1 for s in audit if s["review"])
    print(f"Wrote {out_path}  ({meta['sign_count']} signs)")
    print(f"Wrote {meta_path}")
    print(f"Signs flagged needs-review: {n_review}/{len(audit)}")
    print("Tag provenance: " + ", ".join(
        f"{k}={v} ({ps['share'][k]*100:.0f}%)" for k, v in ps["tags_by_source"].items() if v))
    print(f"Orientation source: heuristic={ps['orientation_heuristic_signs']} "
          f"default={ps['orientation_default_signs']}  |  "
          f"Movement direction: heuristic={ps['movement_dir_heuristic_signs']} "
          f"default={ps['movement_dir_default_signs']}  (of {len(audit)} signs)")


if __name__ == "__main__":
    main()
