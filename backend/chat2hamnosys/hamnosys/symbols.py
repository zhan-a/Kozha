"""HamNoSys 4.0 symbol inventory.

Every codepoint used by the Hamburg Notation System 4.0 is enumerated here
with its class (what kind of role it plays in a sign) and its slot tags
(which positions in a sign it may legitimately appear in).

Source: the CTAN ``hamnosys`` LaTeX package v1.0.3 (Schulder & Hanke, 2022),
https://ctan.org/pkg/hamnosys — maintained by the authors of the Hamburg
notation and mirroring the upstream HamNoSys 4.0 font used by the DGS-Korpus
and SiGML toolchain at sign-lang.uni-hamburg.de. The ranges here cover the
entire Private Use Area block E000..E0F1 plus the ASCII punctuation block
documented in section 5.11 of that manual.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable


class SymClass(str, Enum):
    """Class of a HamNoSys symbol — what semantic role it plays."""

    SYMMETRY = "symmetry"            # hamsymmpar, hamsymmlr
    NONDOM = "nondom"                # hamnondominant, hamnonipsi
    HANDSHAPE_BASE = "handshape_base"
    THUMB_MOD = "thumb_mod"
    FINGER_MOD = "finger_mod"
    EXT_FINGER_DIR = "ext_finger_dir"
    ORI_RELATIVE = "ori_relative"    # hamorirelative
    PALM_DIR = "palm_dir"
    LOC_HEAD = "loc_head"
    LOC_TORSO = "loc_torso"
    LOC_NEUTRAL = "loc_neutral"
    LOC_ARM = "loc_arm"
    LOC_HAND_ZONE = "loc_hand_zone"
    LOC_FINGER = "loc_finger"
    COMBINER = "combiner"            # hamlrbeside, hamlrat
    COREF = "coref"                  # hamcoreftag, hamcorefref
    MOVE_STRAIGHT = "move_straight"
    MOVE_CIRCLE = "move_circle"
    MOVE_ACTION = "move_action"      # fingerplay, nodding, twisting, stir, etc.
    MOVE_CLOCK = "move_clock"
    MOVE_ARC = "move_arc"
    MOVE_WAVY = "move_wavy"
    MOVE_ELLIPSE = "move_ellipse"
    SIZE_MOD = "size_mod"
    SPEED_MOD = "speed_mod"
    TIMING = "timing"
    CONTACT = "contact"
    REPEAT = "repeat"
    SEQ_BEGIN = "seq_begin"
    SEQ_END = "seq_end"
    PAR_BEGIN = "par_begin"
    PAR_END = "par_end"
    FUSION_BEGIN = "fusion_begin"
    FUSION_END = "fusion_end"
    JOINER = "joiner"                # hambetween, hamplus, hametc
    MIME = "mime"                    # hammime
    VERSION = "version"              # hamversionfourzero
    ALT_BEGIN = "alt_begin"          # ASCII '{'
    ALT_END = "alt_end"              # ASCII '}'
    META_ALT = "meta_alt"            # ASCII '|'
    PUNCT = "punct"                  # space, !, comma, fullstop, query
    OBSOLETE = "obsolete"            # wristto*, movecross, moveX


class Slot(str, Enum):
    """Positions in a HamNoSys sign where a symbol may legitimately appear."""

    SIGN_HEAD = "sign_head"          # prefix: symmetry or nondominant markers
    HANDSHAPE = "handshape"          # handshape block
    ORIENTATION = "orientation"      # orientation block
    LOCATION = "location"            # location block
    ACTION = "action"                # action / movement / contact / repetition
    BRACKET = "bracket"              # structural grouping (seq/par/fusion/alt)
    JOINER = "joiner"                # cross-block or within-block combiner
    META = "meta"                    # version markers, mime
    COSMETIC = "cosmetic"            # punctuation the font ignores


@dataclass(frozen=True)
class Symbol:
    codepoint: int
    short_name: str          # canonical HamNoSys name (e.g. "hamfinger2")
    latex_command: str       # CTAN LaTeX macro (e.g. "hamfingertwo")
    sym_class: SymClass
    slots: FrozenSet[Slot]

    @property
    def char(self) -> str:
        return chr(self.codepoint)

    @property
    def hex(self) -> str:
        return f"U+{self.codepoint:04X}"

    def __str__(self) -> str:
        return f"{self.hex} {self.short_name} ({self.sym_class.value})"


# Slot groups reused for many symbols
_SL_HS = frozenset({Slot.HANDSHAPE})
_SL_OR = frozenset({Slot.ORIENTATION})
_SL_LOC = frozenset({Slot.LOCATION})
_SL_ACT = frozenset({Slot.ACTION})
_SL_HEAD = frozenset({Slot.SIGN_HEAD})
_SL_BRACKET = frozenset({Slot.BRACKET})
_SL_JOIN = frozenset({Slot.JOINER})
_SL_META = frozenset({Slot.META})
_SL_COSMETIC = frozenset({Slot.COSMETIC})
# Fingers and hand zones can serve as both touch-targets (locations) and as
# handshape position references; keep both slots open.
_SL_LOC_OR_HS = frozenset({Slot.LOCATION, Slot.HANDSHAPE})


def _entry(cp, name, latex, cls, slots):
    return Symbol(cp, name, latex, cls, slots)


# Section references are to Schulder & Hanke (2022), hamnosys CTAN package.
_ALL_SYMBOLS: tuple[Symbol, ...] = (
    # § 5.1 Handshapes
    _entry(0xE000, "hamfist",             "hamfist",                   SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE001, "hamflathand",         "hamflathand",               SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE002, "hamfinger2",          "hamfingertwo",              SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE003, "hamfinger23",         "hamfingertwothree",         SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE004, "hamfinger23spread",   "hamfingertwothreespread",   SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE005, "hamfinger2345",       "hamfingertwothreefourfive", SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE006, "hampinch12",          "hampinchonetwo",            SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE007, "hampinchall",         "hampinchall",               SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE008, "hampinch12open",      "hampinchonetwoopen",        SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE009, "hamcee12",            "hamceeonetwo",              SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE00A, "hamceeall",           "hamceeall",                 SymClass.HANDSHAPE_BASE, _SL_HS),
    _entry(0xE00B, "hamceeopen",          "hamceeopen",                SymClass.HANDSHAPE_BASE, _SL_HS),
    # § 5.2 Handshape modifiers
    _entry(0xE00C, "hamthumboutmod",      "hamthumboutmod",            SymClass.THUMB_MOD,  _SL_HS),
    _entry(0xE00D, "hamthumbacrossmod",   "hamthumbacrossmod",         SymClass.THUMB_MOD,  _SL_HS),
    _entry(0xE00E, "hamthumbopenmod",     "hamthumbopenmod",           SymClass.THUMB_MOD,  _SL_HS),
    _entry(0xE010, "hamfingerstraightmod","hamfingerstraightmod",      SymClass.FINGER_MOD, _SL_HS),
    _entry(0xE011, "hamfingerbendmod",    "hamfingerbendmod",          SymClass.FINGER_MOD, _SL_HS),
    _entry(0xE012, "hamfingerhookmod",    "hamfingerhookmod",          SymClass.FINGER_MOD, _SL_HS),
    _entry(0xE013, "hamdoublebent",       "hamdoublebent",             SymClass.FINGER_MOD, _SL_HS),
    _entry(0xE014, "hamdoublehooked",     "hamdoublehooked",           SymClass.FINGER_MOD, _SL_HS),
    # § 5.3 Extended finger directions
    _entry(0xE020, "hamextfingeru",  "hamextfingeru",  SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE021, "hamextfingerur", "hamextfingerur", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE022, "hamextfingerr",  "hamextfingerr",  SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE023, "hamextfingerdr", "hamextfingerdr", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE024, "hamextfingerd",  "hamextfingerd",  SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE025, "hamextfingerdl", "hamextfingerdl", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE026, "hamextfingerl",  "hamextfingerl",  SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE027, "hamextfingerul", "hamextfingerul", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE028, "hamextfingerol", "hamextfingerol", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE029, "hamextfingero",  "hamextfingero",  SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE02A, "hamextfingeror", "hamextfingeror", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE02B, "hamextfingeril", "hamextfingeril", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE02C, "hamextfingeri",  "hamextfingeri",  SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE02D, "hamextfingerir", "hamextfingerir", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE02E, "hamextfingerui", "hamextfingerui", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE02F, "hamextfingerdi", "hamextfingerdi", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE030, "hamextfingerdo", "hamextfingerdo", SymClass.EXT_FINGER_DIR, _SL_OR),
    _entry(0xE031, "hamextfingeruo", "hamextfingeruo", SymClass.EXT_FINGER_DIR, _SL_OR),
    # § 5.4 Palm orientation
    _entry(0xE038, "hampalmu",  "hampalmu",  SymClass.PALM_DIR, _SL_OR),
    _entry(0xE039, "hampalmur", "hampalmur", SymClass.PALM_DIR, _SL_OR),
    _entry(0xE03A, "hampalmr",  "hampalmr",  SymClass.PALM_DIR, _SL_OR),
    _entry(0xE03B, "hampalmdr", "hampalmdr", SymClass.PALM_DIR, _SL_OR),
    _entry(0xE03C, "hampalmd",  "hampalmd",  SymClass.PALM_DIR, _SL_OR),
    _entry(0xE03D, "hampalmdl", "hampalmdl", SymClass.PALM_DIR, _SL_OR),
    _entry(0xE03E, "hampalml",  "hampalml",  SymClass.PALM_DIR, _SL_OR),
    _entry(0xE03F, "hampalmul", "hampalmul", SymClass.PALM_DIR, _SL_OR),
    # § 5.5 Location — head
    _entry(0xE040, "hamhead",       "hamhead",       SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE041, "hamheadtop",    "hamheadtop",    SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE042, "hamforehead",   "hamforehead",   SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE043, "hameyebrows",   "hameyebrows",   SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE044, "hameyes",       "hameyes",       SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE045, "hamnose",       "hamnose",       SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE046, "hamnostrils",   "hamnostrils",   SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE047, "hamear",        "hamear",        SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE048, "hamearlobe",    "hamearlobe",    SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE049, "hamcheek",      "hamcheek",      SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE04A, "hamlips",       "hamlips",       SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE04B, "hamtongue",     "hamtongue",     SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE04C, "hamteeth",      "hamteeth",      SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE04D, "hamchin",       "hamchin",       SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE04E, "hamunderchin",  "hamunderchin",  SymClass.LOC_HEAD, _SL_LOC),
    _entry(0xE04F, "hamneck",       "hamneck",       SymClass.LOC_HEAD, _SL_LOC),
    # § 5.5 Location — torso
    _entry(0xE050, "hamshouldertop",  "hamshouldertop",  SymClass.LOC_TORSO, _SL_LOC),
    _entry(0xE051, "hamshoulders",    "hamshoulders",    SymClass.LOC_TORSO, _SL_LOC),
    _entry(0xE052, "hamchest",        "hamchest",        SymClass.LOC_TORSO, _SL_LOC),
    _entry(0xE053, "hamstomach",      "hamstomach",      SymClass.LOC_TORSO, _SL_LOC),
    _entry(0xE054, "hambelowstomach", "hambelowstomach", SymClass.LOC_TORSO, _SL_LOC),
    # § 5.6 Location modifiers (two-hand combiners + coreference)
    _entry(0xE058, "hamlrbeside", "hamlrbeside", SymClass.COMBINER, frozenset({Slot.HANDSHAPE, Slot.LOCATION, Slot.ORIENTATION})),
    _entry(0xE059, "hamlrat",     "hamlrat",     SymClass.COMBINER, frozenset({Slot.HANDSHAPE, Slot.LOCATION, Slot.ORIENTATION})),
    _entry(0xE05A, "hamcoreftag", "hamcoreftag", SymClass.COREF,    _SL_LOC),
    _entry(0xE05B, "hamcorefref", "hamcorefref", SymClass.COREF,    _SL_LOC),
    # § 5.5 Location — neutral space
    _entry(0xE05F, "hamneutralspace", "hamneutralspace", SymClass.LOC_NEUTRAL, _SL_LOC),
    # § 5.5 Location — arm
    _entry(0xE060, "hamupperarm",    "hamupperarm",    SymClass.LOC_ARM, _SL_LOC),
    _entry(0xE061, "hamelbow",       "hamelbow",       SymClass.LOC_ARM, _SL_LOC),
    _entry(0xE062, "hamelbowinside", "hamelbowinside", SymClass.LOC_ARM, _SL_LOC),
    _entry(0xE063, "hamlowerarm",    "hamlowerarm",    SymClass.LOC_ARM, _SL_LOC),
    _entry(0xE064, "hamwristback",   "hamwristback",   SymClass.LOC_ARM, _SL_LOC),
    _entry(0xE065, "hamwristpulse",  "hamwristpulse",  SymClass.LOC_ARM, _SL_LOC),
    # § 5.5 Location — hand zones
    _entry(0xE066, "hamthumbball", "hamthumbball", SymClass.LOC_HAND_ZONE, _SL_LOC_OR_HS),
    _entry(0xE067, "hampalm",      "hampalm",      SymClass.LOC_HAND_ZONE, _SL_LOC_OR_HS),
    _entry(0xE068, "hamhandback",  "hamhandback",  SymClass.LOC_HAND_ZONE, _SL_LOC_OR_HS),
    _entry(0xE069, "hamthumbside", "hamthumbside", SymClass.LOC_HAND_ZONE, _SL_LOC_OR_HS),
    _entry(0xE06A, "hampinkyside", "hampinkyside", SymClass.LOC_HAND_ZONE, _SL_LOC_OR_HS),
    # § 5.5 Location — fingers and finger parts
    _entry(0xE070, "hamthumb",          "hamthumb",          SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE071, "hamindexfinger",    "hamindexfinger",    SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE072, "hammiddlefinger",   "hammiddlefinger",   SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE073, "hamringfinger",     "hamringfinger",     SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE074, "hampinky",          "hampinky",          SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE075, "hamfingertip",      "hamfingertip",      SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE076, "hamfingernail",     "hamfingernail",     SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE077, "hamfingerpad",      "hamfingerpad",      SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE078, "hamfingermidjoint", "hamfingermidjoint", SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE079, "hamfingerbase",     "hamfingerbase",     SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    _entry(0xE07A, "hamfingerside",     "hamfingerside",     SymClass.LOC_FINGER, _SL_LOC_OR_HS),
    # § 5.12 Obsolete wrist spacing
    _entry(0xE07C, "hamwristtopulse", "hamwristtopulse", SymClass.OBSOLETE, _SL_LOC),
    _entry(0xE07D, "hamwristtoback",  "hamwristtoback",  SymClass.OBSOLETE, _SL_LOC),
    _entry(0xE07E, "hamwristtothumb", "hamwristtothumb", SymClass.OBSOLETE, _SL_LOC),
    _entry(0xE07F, "hamwristtopinky", "hamwristtopinky", SymClass.OBSOLETE, _SL_LOC),
    # § 5.7 Movement — straight
    _entry(0xE080, "hammoveu",  "hammoveu",  SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE081, "hammoveur", "hammoveur", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE082, "hammover",  "hammover",  SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE083, "hammovedr", "hammovedr", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE084, "hammoved",  "hammoved",  SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE085, "hammovedl", "hammovedl", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE086, "hammovel",  "hammovel",  SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE087, "hammoveul", "hammoveul", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE088, "hammoveol", "hammoveol", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE089, "hammoveo",  "hammoveo",  SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE08A, "hammoveor", "hammoveor", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE08B, "hammoveil", "hammoveil", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE08C, "hammovei",  "hammovei",  SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE08D, "hammoveir", "hammoveir", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE08E, "hammoveui", "hammoveui", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE08F, "hammovedi", "hammovedi", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE090, "hammovedo", "hammovedo", SymClass.MOVE_STRAIGHT, _SL_ACT),
    _entry(0xE091, "hammoveuo", "hammoveuo", SymClass.MOVE_STRAIGHT, _SL_ACT),
    # § 5.7 Movement — circle plane indicators
    _entry(0xE092, "hamcircleo",  "hamcircleo",  SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE093, "hamcirclei",  "hamcirclei",  SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE094, "hamcircled",  "hamcircled",  SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE095, "hamcircleu",  "hamcircleu",  SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE096, "hamcirclel",  "hamcirclel",  SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE097, "hamcircler",  "hamcircler",  SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE098, "hamcircleul", "hamcircleul", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE099, "hamcircledr", "hamcircledr", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE09A, "hamcircleur", "hamcircleur", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE09B, "hamcircledl", "hamcircledl", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE09C, "hamcircleol", "hamcircleol", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE09D, "hamcircleir", "hamcircleir", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE09E, "hamcircleor", "hamcircleor", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE09F, "hamcircleil", "hamcircleil", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE0A0, "hamcircleui", "hamcircleui", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE0A1, "hamcircledo", "hamcircledo", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE0A2, "hamcircleuo", "hamcircleuo", SymClass.MOVE_CIRCLE, _SL_ACT),
    _entry(0xE0A3, "hamcircledi", "hamcircledi", SymClass.MOVE_CIRCLE, _SL_ACT),
    # § 5.7 Movement — action modifiers
    _entry(0xE0A4, "hamfingerplay", "hamfingerplay", SymClass.MOVE_ACTION, _SL_ACT),
    _entry(0xE0A5, "hamnodding",    "hamnodding",    SymClass.MOVE_ACTION, _SL_ACT),
    _entry(0xE0A6, "hamswinging",   "hamswinging",   SymClass.MOVE_ACTION, _SL_ACT),
    _entry(0xE0A7, "hamtwisting",   "hamtwisting",   SymClass.MOVE_ACTION, _SL_ACT),
    _entry(0xE0A8, "hamstircw",     "hamstircw",     SymClass.MOVE_ACTION, _SL_ACT),
    _entry(0xE0A9, "hamstirccw",    "hamstirccw",    SymClass.MOVE_ACTION, _SL_ACT),
    _entry(0xE0AA, "hamreplace",    "hamreplace",    SymClass.MOVE_ACTION, _SL_ACT),
    _entry(0xE0AD, "hammovecross",  "hammovecross",  SymClass.OBSOLETE,    _SL_ACT),
    _entry(0xE0AE, "hammoveX",      "hammoveX",      SymClass.OBSOLETE,    _SL_ACT),
    _entry(0xE0AF, "hamnomotion",   "hamnomotion",   SymClass.MOVE_ACTION, _SL_ACT),
    # § 5.7 Movement — clock positions (for curved paths)
    _entry(0xE0B0, "hamclocku",    "hamclocku",    SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B1, "hamclockul",   "hamclockul",   SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B2, "hamclockl",    "hamclockl",    SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B3, "hamclockdl",   "hamclockdl",   SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B4, "hamclockd",    "hamclockd",    SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B5, "hamclockdr",   "hamclockdr",   SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B6, "hamclockr",    "hamclockr",    SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B7, "hamclockur",   "hamclockur",   SymClass.MOVE_CLOCK, _SL_ACT),
    _entry(0xE0B8, "hamclockfull", "hamclockfull", SymClass.MOVE_CLOCK, _SL_ACT),
    # § 5.7 Movement — arc, wavy, zigzag
    _entry(0xE0B9, "hamarcl",   "hamarcl",   SymClass.MOVE_ARC,  _SL_ACT),
    _entry(0xE0BA, "hamarcu",   "hamarcu",   SymClass.MOVE_ARC,  _SL_ACT),
    _entry(0xE0BB, "hamarcr",   "hamarcr",   SymClass.MOVE_ARC,  _SL_ACT),
    _entry(0xE0BC, "hamarcd",   "hamarcd",   SymClass.MOVE_ARC,  _SL_ACT),
    _entry(0xE0BD, "hamwavy",   "hamwavy",   SymClass.MOVE_WAVY, _SL_ACT),
    _entry(0xE0BE, "hamzigzag", "hamzigzag", SymClass.MOVE_WAVY, _SL_ACT),
    # § 5.7 Movement — ellipse shapes
    _entry(0xE0C0, "hamellipseh",  "hamellipseh",  SymClass.MOVE_ELLIPSE, _SL_ACT),
    _entry(0xE0C1, "hamellipseur", "hamellipseur", SymClass.MOVE_ELLIPSE, _SL_ACT),
    _entry(0xE0C2, "hamellipsev",  "hamellipsev",  SymClass.MOVE_ELLIPSE, _SL_ACT),
    _entry(0xE0C3, "hamellipseul", "hamellipseul", SymClass.MOVE_ELLIPSE, _SL_ACT),
    # § 5.7 Movement — size / speed / timing modifiers (apply to a movement)
    _entry(0xE0C4, "hamincreasing", "hamincreasing", SymClass.SIZE_MOD,  _SL_ACT),
    _entry(0xE0C5, "hamdecreasing", "hamdecreasing", SymClass.SIZE_MOD,  _SL_ACT),
    # § 5.8 Movement modifiers
    _entry(0xE0C6, "hamsmallmod",   "hamsmallmod",   SymClass.SIZE_MOD,  _SL_ACT),
    _entry(0xE0C7, "hamlargemod",   "hamlargemod",   SymClass.SIZE_MOD,  _SL_ACT),
    _entry(0xE0C8, "hamfast",       "hamfast",       SymClass.SPEED_MOD, _SL_ACT),
    _entry(0xE0C9, "hamslow",       "hamslow",       SymClass.SPEED_MOD, _SL_ACT),
    _entry(0xE0CA, "hamtense",      "hamtense",      SymClass.SPEED_MOD, _SL_ACT),
    _entry(0xE0CB, "hamrest",       "hamrest",       SymClass.TIMING,    _SL_ACT),
    _entry(0xE0CC, "hamhalt",       "hamhalt",       SymClass.TIMING,    _SL_ACT),
    # § 5.7 Contact operators
    _entry(0xE0D0, "hamclose",        "hamclose",        SymClass.CONTACT, _SL_ACT),
    _entry(0xE0D1, "hamtouch",        "hamtouch",        SymClass.CONTACT, _SL_ACT),
    _entry(0xE0D2, "haminterlock",    "haminterlock",    SymClass.CONTACT, _SL_ACT),
    _entry(0xE0D3, "hamcross",        "hamcross",        SymClass.CONTACT, _SL_ACT),
    _entry(0xE0D4, "hamarmextended",  "hamarmextended",  SymClass.CONTACT, _SL_ACT),
    _entry(0xE0D5, "hambehind",       "hambehind",       SymClass.CONTACT, _SL_ACT),
    _entry(0xE0D6, "hambrushing",     "hambrushing",     SymClass.CONTACT, _SL_ACT),
    # § 5.9 Other — repetitions
    _entry(0xE0D8, "hamrepeatfromstart",        "hamrepeatfromstart",        SymClass.REPEAT, _SL_ACT),
    _entry(0xE0D9, "hamrepeatfromstartseveral", "hamrepeatfromstartseveral", SymClass.REPEAT, _SL_ACT),
    _entry(0xE0DA, "hamrepeatcontinue",         "hamrepeatcontinue",         SymClass.REPEAT, _SL_ACT),
    _entry(0xE0DB, "hamrepeatcontinueseveral",  "hamrepeatcontinueseveral",  SymClass.REPEAT, _SL_ACT),
    _entry(0xE0DC, "hamrepeatreverse",          "hamrepeatreverse",          SymClass.REPEAT, _SL_ACT),
    _entry(0xE0DD, "hamalternatingmotion",      "hamalternatingmotion",      SymClass.REPEAT, _SL_ACT),
    # § 5.9 Other — brackets
    _entry(0xE0E0, "hamseqbegin",    "hamseqbegin",    SymClass.SEQ_BEGIN,    _SL_BRACKET),
    _entry(0xE0E1, "hamseqend",      "hamseqend",      SymClass.SEQ_END,      _SL_BRACKET),
    _entry(0xE0E2, "hamparbegin",    "hamparbegin",    SymClass.PAR_BEGIN,    _SL_BRACKET),
    _entry(0xE0E3, "hamparend",      "hamparend",      SymClass.PAR_END,      _SL_BRACKET),
    _entry(0xE0E4, "hamfusionbegin", "hamfusionbegin", SymClass.FUSION_BEGIN, _SL_BRACKET),
    _entry(0xE0E5, "hamfusionend",   "hamfusionend",   SymClass.FUSION_END,   _SL_BRACKET),
    # § 5.9 Other — joiners
    _entry(0xE0E6, "hambetween", "hambetween", SymClass.JOINER, _SL_JOIN),
    _entry(0xE0E7, "hamplus",    "hamplus",    SymClass.JOINER, _SL_JOIN),
    # § 5.9 Other — symmetry operators
    _entry(0xE0E8, "hamsymmpar",     "hamsymmpar",     SymClass.SYMMETRY, _SL_HEAD),
    _entry(0xE0E9, "hamsymmlr",      "hamsymmlr",      SymClass.SYMMETRY, _SL_HEAD),
    # § 5.9 Other — nondominant markers
    _entry(0xE0EA, "hamnondominant", "hamnondominant", SymClass.NONDOM,   _SL_HEAD),
    _entry(0xE0EB, "hamnonipsi",     "hamnonipsi",     SymClass.NONDOM,   _SL_HEAD),
    # § 5.9 Other — shortcut operators
    _entry(0xE0EC, "hametc",         "hametc",         SymClass.JOINER,   _SL_JOIN),
    _entry(0xE0ED, "hamorirelative", "hamorirelative", SymClass.ORI_RELATIVE, _SL_OR),
    # § 5.9 Other — mime marker
    _entry(0xE0F0, "hammime",        "hammime",        SymClass.MIME,     _SL_META),
    # § 5.10 Version marker
    _entry(0xE0F1, "hamversion40",   "hamversionfourzero", SymClass.VERSION, _SL_META),
    # § 5.11 Regular Unicode (ASCII) characters
    _entry(0x0020, "hamspace",     "hamspace",     SymClass.PUNCT,     _SL_COSMETIC),
    _entry(0x0021, "hamexclaim",   "hamexclaim",   SymClass.PUNCT,     _SL_COSMETIC),
    _entry(0x002C, "hamcomma",     "hamcomma",     SymClass.PUNCT,     _SL_COSMETIC),
    _entry(0x002E, "hamfullstop",  "hamfullstop",  SymClass.PUNCT,     _SL_COSMETIC),
    _entry(0x003F, "hamquery",     "hamquery",     SymClass.PUNCT,     _SL_COSMETIC),
    _entry(0x007B, "hamaltbegin",  "hamaltbegin",  SymClass.ALT_BEGIN, _SL_BRACKET),
    _entry(0x007C, "hammetaalt",   "hammetaalt",   SymClass.META_ALT,  _SL_BRACKET),
    _entry(0x007D, "hamaltend",    "hamaltend",    SymClass.ALT_END,   _SL_BRACKET),
)


SYMBOLS: dict[int, Symbol] = {s.codepoint: s for s in _ALL_SYMBOLS}


def classify(ch: str) -> SymClass | None:
    """Return the class of a single HamNoSys character, or None if unknown."""
    if len(ch) != 1:
        raise ValueError(f"classify() expects a single character, got len={len(ch)}")
    sym = SYMBOLS.get(ord(ch))
    return sym.sym_class if sym else None


def is_known(ch: str) -> bool:
    """True if ``ch`` is a recognised HamNoSys 4.0 codepoint."""
    return len(ch) == 1 and ord(ch) in SYMBOLS


def codepoints_in_class(cls: SymClass) -> frozenset[int]:
    """Return all codepoints belonging to a class."""
    return frozenset(cp for cp, s in SYMBOLS.items() if s.sym_class is cls)


def symbols_in_classes(classes: Iterable[SymClass]) -> tuple[Symbol, ...]:
    """Return all Symbol entries whose class is in ``classes``."""
    class_set = set(classes)
    return tuple(s for s in _ALL_SYMBOLS if s.sym_class in class_set)
