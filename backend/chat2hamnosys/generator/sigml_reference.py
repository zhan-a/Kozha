"""Structured HamNoSys + SiGML reference catalog.

Source: ``hamnosys-sigml-reference.md`` at the repo root (derived from the
HamNoSys LaTeX package v1.0.3 + UEA SiGML schemas). Every entry is
machine-readable and shared by:

- The LLM prompt builder (``sigml_direct._build_prompt``) — gives the
  model authoritative tag-name → semantic-role pairs so it stops
  hallucinating non-existent tags.
- The HTTP API (``GET /reference/sigml``) — frontend pulls the catalog
  to render the interactive annotated-SiGML editor where each tag is
  click-swappable for an alternative in the same category.

Categories follow the slot order from the reference:
``[symmetry] handshape [handshape_modifier]* ext_finger palm location
[location_modifier]* movement_block``.

Each entry has the shape::

    {
      "name":        "hamflathand",     # canonical SiGML tag name
      "codepoint":   0xE001,            # PUA Unicode point (None for non-PUA)
      "category":    "handshape",       # see CATEGORY_ORDER
      "role":        "flat hand, all fingers extended and adjacent",
      "label":       "flat hand",       # short human label for pickers
    }

The catalog is intentionally hand-curated rather than parsed from the
markdown at runtime so we get type-safety + zero startup cost. To
update: edit the markdown reference, then update this file.
"""

from __future__ import annotations

from typing import Any


# Canonical slot order. Used by the frontend to render category-grouped
# pickers, and by the LLM prompt to remind the model of the order.
CATEGORY_ORDER: list[str] = [
    "symmetry",
    "handshape",
    "handshape_modifier",
    "ext_finger_direction",
    "palm_direction",
    "location",
    "location_modifier",
    "movement",
    "movement_modifier",
    "structural",
    "punctuation",
]

# Plain-English category labels for the picker UI.
CATEGORY_LABELS: dict[str, str] = {
    "symmetry":            "Two-handed symmetry",
    "handshape":           "Handshape",
    "handshape_modifier":  "Handshape modifier",
    "ext_finger_direction": "Extended-finger direction",
    "palm_direction":      "Palm direction",
    "location":            "Location",
    "location_modifier":   "Location modifier",
    "movement":            "Movement",
    "movement_modifier":   "Movement size",
    "structural":          "Structural (groups, repeats)",
    "punctuation":         "Punctuation",
}


def _e(name: str, cp: int | None, category: str, role: str, label: str | None = None) -> dict[str, Any]:
    return {
        "name":      name,
        "codepoint": cp,
        "category":  category,
        "role":      role,
        "label":     label or role.split(",")[0].split("(")[0].strip(),
    }


# ---------------------------------------------------------------------------
# 1. Handshapes (12) — base finger configuration. Required.
# ---------------------------------------------------------------------------
HANDSHAPES: list[dict[str, Any]] = [
    _e("hamfist",            0xE000, "handshape", "closed fist", "fist"),
    _e("hamflathand",        0xE001, "handshape", "flat hand, all fingers extended and adjacent", "flat hand"),
    _e("hamfinger2",         0xE002, "handshape", "index finger extended only", "index"),
    _e("hamfinger23",        0xE003, "handshape", "index and middle fingers extended, adjacent", "index + middle"),
    _e("hamfinger23spread",  0xE004, "handshape", "index and middle fingers extended, spread (V-shape)", "V-shape"),
    _e("hamfinger2345",      0xE005, "handshape", "four fingers extended (no thumb)", "four fingers"),
    _e("hampinch12",         0xE006, "handshape", "pinch between thumb and index", "thumb-index pinch"),
    _e("hampinchall",        0xE007, "handshape", "all fingers pinched together with thumb", "all-finger pinch"),
    _e("hampinch12open",     0xE008, "handshape", "pinch shape but with thumb-index gap (open pinch)", "open pinch"),
    _e("hamcee12",           0xE009, "handshape", "C-shape between thumb and index", "small C"),
    _e("hamceeall",          0xE00A, "handshape", "C-shape using all fingers", "full C"),
    _e("hamceeopen",         0xE00B, "handshape", "open C-shape, wider aperture", "open C"),
]


# ---------------------------------------------------------------------------
# 2. Handshape modifiers (8) — optional diacritics on the handshape.
# ---------------------------------------------------------------------------
HANDSHAPE_MODIFIERS: list[dict[str, Any]] = [
    _e("hamthumboutmod",        0xE00C, "handshape_modifier", "thumb extended outward", "thumb out"),
    _e("hamthumbacrossmod",     0xE00D, "handshape_modifier", "thumb crosses over palm", "thumb across"),
    _e("hamthumbopenmod",       0xE00E, "handshape_modifier", "thumb away from fingers (open)", "thumb open"),
    _e("hamfingerstraightmod",  0xE010, "handshape_modifier", "fingers fully straight", "fingers straight"),
    _e("hamfingerbendmod",      0xE011, "handshape_modifier", "fingers bent at base joint", "fingers bent"),
    _e("hamfingerhookmod",      0xE012, "handshape_modifier", "fingers curled into hook", "fingers hooked"),
    _e("hamdoublebent",         0xE013, "handshape_modifier", "fingers bent at multiple joints", "double-bent"),
    _e("hamdoublehooked",       0xE014, "handshape_modifier", "doubly hooked finger configuration", "double-hooked"),
]


# ---------------------------------------------------------------------------
# 3. Extended-finger directions (18) — required.
# ---------------------------------------------------------------------------
EXT_FINGER_DIRECTIONS: list[dict[str, Any]] = [
    _e("hamextfingeru",  0xE020, "ext_finger_direction", "fingers point up", "up"),
    _e("hamextfingerur", 0xE021, "ext_finger_direction", "fingers point up-right", "up-right"),
    _e("hamextfingerr",  0xE022, "ext_finger_direction", "fingers point right", "right"),
    _e("hamextfingerdr", 0xE023, "ext_finger_direction", "fingers point down-right", "down-right"),
    _e("hamextfingerd",  0xE024, "ext_finger_direction", "fingers point down", "down"),
    _e("hamextfingerdl", 0xE025, "ext_finger_direction", "fingers point down-left", "down-left"),
    _e("hamextfingerl",  0xE026, "ext_finger_direction", "fingers point left", "left"),
    _e("hamextfingerul", 0xE027, "ext_finger_direction", "fingers point up-left", "up-left"),
    _e("hamextfingerol", 0xE028, "ext_finger_direction", "fingers point out-left (away and to left)", "out-left"),
    _e("hamextfingero",  0xE029, "ext_finger_direction", "fingers point out (away from signer)", "out"),
    _e("hamextfingeror", 0xE02A, "ext_finger_direction", "fingers point out-right", "out-right"),
    _e("hamextfingeril", 0xE02B, "ext_finger_direction", "fingers point in-left (toward signer and to left)", "in-left"),
    _e("hamextfingeri",  0xE02C, "ext_finger_direction", "fingers point in (toward signer)", "in"),
    _e("hamextfingerir", 0xE02D, "ext_finger_direction", "fingers point in-right", "in-right"),
    _e("hamextfingerui", 0xE02E, "ext_finger_direction", "fingers point up-in (up and toward signer)", "up-in"),
    _e("hamextfingerdi", 0xE02F, "ext_finger_direction", "fingers point down-in", "down-in"),
    _e("hamextfingerdo", 0xE030, "ext_finger_direction", "fingers point down-out", "down-out"),
    _e("hamextfingeruo", 0xE031, "ext_finger_direction", "fingers point up-out", "up-out"),
]


# ---------------------------------------------------------------------------
# 4. Palm directions (8) — required.
# ---------------------------------------------------------------------------
PALM_DIRECTIONS: list[dict[str, Any]] = [
    _e("hampalmu",  0xE038, "palm_direction", "palm faces up", "up"),
    _e("hampalmur", 0xE039, "palm_direction", "palm faces up-right", "up-right"),
    _e("hampalmr",  0xE03A, "palm_direction", "palm faces right", "right"),
    _e("hampalmdr", 0xE03B, "palm_direction", "palm faces down-right", "down-right"),
    _e("hampalmd",  0xE03C, "palm_direction", "palm faces down", "down"),
    _e("hampalmdl", 0xE03D, "palm_direction", "palm faces down-left", "down-left"),
    _e("hampalml",  0xE03E, "palm_direction", "palm faces left", "left"),
    _e("hampalmul", 0xE03F, "palm_direction", "palm faces up-left", "up-left"),
]


# ---------------------------------------------------------------------------
# 5. Locations (44) — required. One per sign.
# ---------------------------------------------------------------------------
LOCATIONS: list[dict[str, Any]] = [
    # Head and face
    _e("hamhead",        0xE040, "location", "head (general)", "head"),
    _e("hamheadtop",     0xE041, "location", "top of head", "head top"),
    _e("hamforehead",    0xE042, "location", "forehead", "forehead"),
    _e("hameyebrows",    0xE043, "location", "eyebrows", "eyebrows"),
    _e("hameyes",        0xE044, "location", "eyes", "eyes"),
    _e("hamnose",        0xE045, "location", "nose", "nose"),
    _e("hamnostrils",    0xE046, "location", "nostrils", "nostrils"),
    _e("hamear",         0xE047, "location", "ear", "ear"),
    _e("hamearlobe",     0xE048, "location", "earlobe", "earlobe"),
    _e("hamcheek",       0xE049, "location", "cheek", "cheek"),
    _e("hamlips",        0xE04A, "location", "lips", "lips"),
    _e("hamtongue",      0xE04B, "location", "tongue", "tongue"),
    _e("hamteeth",       0xE04C, "location", "teeth", "teeth"),
    _e("hamchin",        0xE04D, "location", "chin", "chin"),
    _e("hamunderchin",   0xE04E, "location", "under chin", "under chin"),
    # Torso
    _e("hamneck",        0xE04F, "location", "neck", "neck"),
    _e("hamshouldertop", 0xE050, "location", "top of shoulder", "shoulder top"),
    _e("hamshoulders",   0xE051, "location", "shoulders (front)", "shoulders"),
    _e("hamchest",       0xE052, "location", "chest", "chest"),
    _e("hamstomach",     0xE053, "location", "stomach", "stomach"),
    _e("hambelowstomach", 0xE054, "location", "below stomach", "below stomach"),
    _e("hamneutralspace", 0xE05F, "location", "neutral signing space (in front of body)", "neutral space"),
    # Arm
    _e("hamupperarm",    0xE060, "location", "upper arm", "upper arm"),
    _e("hamelbow",       0xE061, "location", "elbow (outside)", "elbow"),
    _e("hamelbowinside", 0xE062, "location", "inside of elbow", "elbow inside"),
    _e("hamlowerarm",    0xE063, "location", "lower arm", "lower arm"),
    _e("hamwristback",   0xE064, "location", "back of wrist", "wrist back"),
    _e("hamwristpulse",  0xE065, "location", "wrist pulse side (palm side)", "wrist pulse"),
    # Hand zones
    _e("hamthumbball",   0xE066, "location", "thumb ball (thenar eminence)", "thumb ball"),
    _e("hampalm",        0xE067, "location", "palm", "palm"),
    _e("hamhandback",    0xE068, "location", "back of hand", "hand back"),
    _e("hamthumbside",   0xE069, "location", "thumb side of hand (radial)", "thumb side"),
    _e("hampinkyside",   0xE06A, "location", "pinky side of hand (ulnar)", "pinky side"),
    # Specific fingers and finger zones
    _e("hamthumb",       0xE070, "location", "thumb", "thumb"),
    _e("hamindexfinger", 0xE071, "location", "index finger", "index finger"),
    _e("hammiddlefinger", 0xE072, "location", "middle finger", "middle finger"),
    _e("hamringfinger",  0xE073, "location", "ring finger", "ring finger"),
    _e("hampinky",       0xE074, "location", "pinky", "pinky"),
    _e("hamfingertip",   0xE075, "location", "fingertip", "fingertip"),
    _e("hamfingernail",  0xE076, "location", "fingernail", "fingernail"),
    _e("hamfingerpad",   0xE077, "location", "finger pad (volar tip)", "finger pad"),
    _e("hamfingermidjoint", 0xE078, "location", "finger middle joint (PIP)", "finger mid-joint"),
    _e("hamfingerbase",  0xE079, "location", "finger base joint (MCP)", "finger base"),
    _e("hamfingerside",  0xE07A, "location", "side of finger", "finger side"),
]


# ---------------------------------------------------------------------------
# 6. Location modifiers (4)
# ---------------------------------------------------------------------------
LOCATION_MODIFIERS: list[dict[str, Any]] = [
    _e("hamlrbeside",    0xE058, "location_modifier", "location is beside the body part (offset)", "beside"),
    _e("hamlrat",        0xE059, "location_modifier", "location is at/touching the body part", "at"),
    _e("hamcoreftag",    0xE05A, "location_modifier", "coreference tag (establishes a referent location)", "coref tag"),
    _e("hamcorefref",    0xE05B, "location_modifier", "coreference reference", "coref ref"),
]


# ---------------------------------------------------------------------------
# 7. Movements (subset — most-used). Optional but adds clarity.
# ---------------------------------------------------------------------------
MOVEMENTS: list[dict[str, Any]] = [
    # Straight directional movements
    _e("hammoveu",  0xE080, "movement", "move up", "up"),
    _e("hammoveur", 0xE081, "movement", "move up-right", "up-right"),
    _e("hammover",  0xE082, "movement", "move right", "right"),
    _e("hammovedr", 0xE083, "movement", "move down-right", "down-right"),
    _e("hammoved",  0xE084, "movement", "move down", "down"),
    _e("hammovedl", 0xE085, "movement", "move down-left", "down-left"),
    _e("hammovel",  0xE086, "movement", "move left", "left"),
    _e("hammoveul", 0xE087, "movement", "move up-left", "up-left"),
    _e("hammoveol", 0xE088, "movement", "move out-left (away from signer to left)", "out-left"),
    _e("hammoveo",  0xE089, "movement", "move out (away from signer)", "out"),
    _e("hammoveor", 0xE08A, "movement", "move out-right", "out-right"),
    _e("hammoveil", 0xE08B, "movement", "move in-left (toward signer to left)", "in-left"),
    _e("hammovei",  0xE08C, "movement", "move in (toward signer)", "in"),
    _e("hammoveir", 0xE08D, "movement", "move in-right", "in-right"),
    _e("hammoveui", 0xE08E, "movement", "move up-in", "up-in"),
    _e("hammovedi", 0xE08F, "movement", "move down-in", "down-in"),
    _e("hammovedo", 0xE090, "movement", "move down-out", "down-out"),
    _e("hammoveuo", 0xE091, "movement", "move up-out", "up-out"),
    # Full circles in cardinal planes
    _e("hamcircleo", 0xE092, "movement", "full circle starting outward", "circle out"),
    _e("hamcirclei", 0xE093, "movement", "full circle starting inward", "circle in"),
    _e("hamcircled", 0xE094, "movement", "full circle starting downward", "circle down"),
    _e("hamcircleu", 0xE095, "movement", "full circle starting upward", "circle up"),
    _e("hamcirclel", 0xE096, "movement", "full circle starting leftward", "circle left"),
    _e("hamcircler", 0xE097, "movement", "full circle starting rightward", "circle right"),
    # Local hand movements
    _e("hamfingerplay", 0xE0A4, "movement", "finger wiggle (sequential finger flexion)", "finger play"),
    _e("hamnodding",    0xE0A5, "movement", "hand nodding (wrist flexion)", "nodding"),
    _e("hamswinging",   0xE0A6, "movement", "hand swinging side-to-side at wrist", "swinging"),
    _e("hamtwisting",   0xE0A7, "movement", "forearm rotation (twisting)", "twisting"),
    _e("hamstircw",     0xE0A8, "movement", "stir clockwise (small circle at wrist)", "stir CW"),
    _e("hamstirccw",    0xE0A9, "movement", "stir counter-clockwise", "stir CCW"),
    # Special operators
    _e("hamreplace",    0xE0AA, "movement", "replace marker (handshape changes during movement)", "replace"),
    _e("hamnomotion",   0xE0AF, "movement", "explicit no-movement marker", "no motion"),
    # Arc movements
    _e("hamarcl",       0xE0B9, "movement", "arc curving leftward", "arc left"),
    _e("hamarcu",       0xE0BA, "movement", "arc curving upward", "arc up"),
    _e("hamarcr",       0xE0BB, "movement", "arc curving rightward", "arc right"),
    _e("hamarcd",       0xE0BC, "movement", "arc curving downward", "arc down"),
    # Path-shape modifiers
    _e("hamwavy",       0xE0BD, "movement", "wavy path", "wavy"),
    _e("hamzigzag",     0xE0BE, "movement", "zigzag path", "zigzag"),
    # Ellipses
    _e("hamellipseh",   0xE0C0, "movement", "horizontal ellipse", "ellipse horiz."),
    _e("hamellipsev",   0xE0C2, "movement", "vertical ellipse", "ellipse vert."),
    # Size dynamics
    _e("hamincreasing", 0xE0C4, "movement", "movement size increases over time", "increasing"),
    _e("hamdecreasing", 0xE0C5, "movement", "movement size decreases over time", "decreasing"),
    # Speed and tension
    _e("hamfast",       0xE0C8, "movement", "fast movement", "fast"),
    _e("hamslow",       0xE0C9, "movement", "slow movement", "slow"),
    _e("hamtense",      0xE0CA, "movement", "tense / forceful movement", "tense"),
    _e("hamrest",       0xE0CB, "movement", "relaxed movement", "relaxed"),
    _e("hamhalt",       0xE0CC, "movement", "abrupt halt at end of movement", "halt"),
    # Contact and proximity
    _e("hamclose",      0xE0D0, "movement", "hands move close", "close"),
    _e("hamtouch",      0xE0D1, "movement", "hands touch (light contact)", "touch"),
    _e("haminterlock",  0xE0D2, "movement", "hands interlock", "interlock"),
    _e("hamcross",      0xE0D3, "movement", "hands cross", "cross"),
    _e("hamarmextended", 0xE0D4, "movement", "arm fully extended", "arm extended"),
    _e("hambehind",     0xE0D5, "movement", "positioned behind reference", "behind"),
    _e("hambrushing",   0xE0D6, "movement", "brushing contact during movement", "brushing"),
]


# ---------------------------------------------------------------------------
# 8. Movement modifiers (2)
# ---------------------------------------------------------------------------
MOVEMENT_MODIFIERS: list[dict[str, Any]] = [
    _e("hamsmallmod",   0xE0C6, "movement_modifier", "movement is smaller than default", "smaller"),
    _e("hamlargemod",   0xE0C7, "movement_modifier", "movement is larger than default", "larger"),
]


# ---------------------------------------------------------------------------
# 9. Structural symbols (repetition + composition + symmetry)
# ---------------------------------------------------------------------------
STRUCTURAL: list[dict[str, Any]] = [
    # Repetition
    _e("hamrepeatfromstart",     0xE0D8, "structural", "repeat the entire sign from start", "repeat from start"),
    _e("hamrepeatfromstartseveral", 0xE0D9, "structural", "repeat from start several times", "repeat several"),
    _e("hamrepeatcontinue",      0xE0DA, "structural", "continue current motion repeatedly", "continue repeat"),
    _e("hamrepeatcontinueseveral", 0xE0DB, "structural", "continue motion several times", "continue several"),
    _e("hamrepeatreverse",       0xE0DC, "structural", "repeat in reverse direction", "repeat reverse"),
    _e("hamalternatingmotion",   0xE0DD, "structural", "hands alternate during repetition", "alternating"),
    # Composition brackets
    _e("hamseqbegin",   0xE0E0, "structural", "begin sequential composition group", "seq ["),
    _e("hamseqend",     0xE0E1, "structural", "end sequential composition group", "seq ]"),
    _e("hamparbegin",   0xE0E2, "structural", "begin parallel composition group (simultaneous actions)", "par ["),
    _e("hamparend",     0xE0E3, "structural", "end parallel composition group", "par ]"),
    _e("hamfusionbegin", 0xE0E4, "structural", "begin fusion group (movements blend)", "fusion ["),
    _e("hamfusionend",  0xE0E5, "structural", "end fusion group", "fusion ]"),
    _e("hambetween",    0xE0E6, "structural", "between operator (intermediate position)", "between"),
    _e("hamplus",       0xE0E7, "structural", "combine / additive operator", "plus"),
    # Two-handed symmetry
    _e("hamsymmpar",    0xE0E8, "symmetry", "parallel symmetry — both hands move in same direction", "parallel"),
    _e("hamsymmlr",     0xE0E9, "symmetry", "mirror symmetry — hands mirror across midline", "mirror"),
    # Hand designation
    _e("hamnondominant", 0xE0EA, "structural", "marks the non-dominant hand", "non-dominant"),
    _e("hamnonipsi",    0xE0EB, "structural", "contralateral side marker", "contralateral"),
]


# ---------------------------------------------------------------------------
# Master tables
# ---------------------------------------------------------------------------
ALL_ENTRIES: list[dict[str, Any]] = (
    HANDSHAPES
    + HANDSHAPE_MODIFIERS
    + EXT_FINGER_DIRECTIONS
    + PALM_DIRECTIONS
    + LOCATIONS
    + LOCATION_MODIFIERS
    + MOVEMENTS
    + MOVEMENT_MODIFIERS
    + STRUCTURAL
)

# name -> entry lookup
BY_NAME: dict[str, dict[str, Any]] = {e["name"]: e for e in ALL_ENTRIES}

# category -> entries lookup (for the picker UI)
BY_CATEGORY: dict[str, list[dict[str, Any]]] = {}
for e in ALL_ENTRIES:
    BY_CATEGORY.setdefault(e["category"], []).append(e)


def get_catalog() -> dict[str, Any]:
    """Return the full catalog as a JSON-serialisable dict."""
    return {
        "category_order":  CATEGORY_ORDER,
        "category_labels": CATEGORY_LABELS,
        "entries":         ALL_ENTRIES,
        "by_name":         BY_NAME,
        "by_category":     {k: v for k, v in BY_CATEGORY.items()},
        "slot_order_note": (
            "Canonical slot order: [symmetry?] handshape "
            "[handshape_modifier]* ext_finger_direction palm_direction "
            "location [location_modifier]* [movement_block]. "
            "Mandatory slots: handshape, ext_finger_direction, "
            "palm_direction, location."
        ),
    }


def render_prompt_catalog(max_per_category: int | None = None) -> str:
    """Render the catalog as a compact table for inclusion in the LLM prompt.

    The LLM needs the tag-name → semantic-role mapping so it stops
    hallucinating non-existent tags. We include every handshape /
    ext-finger / palm direction (small categories), and a curated subset
    of locations / movements (large categories) capped by
    ``max_per_category`` to keep the prompt under context limits.
    """
    lines: list[str] = []
    for cat in CATEGORY_ORDER:
        entries = BY_CATEGORY.get(cat) or []
        if not entries:
            continue
        # Apply per-category cap when set; small categories are kept whole.
        capped = entries if max_per_category is None or len(entries) <= max_per_category else entries[:max_per_category]
        lines.append(f"### {CATEGORY_LABELS.get(cat, cat)} ({len(entries)} options)")
        for e in capped:
            lines.append(f"  <{e['name']}/> — {e['role']}")
        if len(capped) < len(entries):
            lines.append(f"  … and {len(entries) - len(capped)} more in this category.")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "CATEGORY_ORDER",
    "CATEGORY_LABELS",
    "HANDSHAPES",
    "HANDSHAPE_MODIFIERS",
    "EXT_FINGER_DIRECTIONS",
    "PALM_DIRECTIONS",
    "LOCATIONS",
    "LOCATION_MODIFIERS",
    "MOVEMENTS",
    "MOVEMENT_MODIFIERS",
    "STRUCTURAL",
    "ALL_ENTRIES",
    "BY_NAME",
    "BY_CATEGORY",
    "get_catalog",
    "render_prompt_catalog",
]
