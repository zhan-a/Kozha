"""Lark grammar for the HamNoSys 4.0 notation.

The grammar here encodes the spine of a HamNoSys sign::

    sign      := prefix? handshape orientation? location? action*
    prefix    := (symmetry | nondominant)+
    handshape := handshape_base (thumb_mod | finger_mod)*  ... (combiner handshape)?
    action    := movement | contact | repetition | bracket_group
    bracket   := seq | par | fusion | alt

Since real-world HamNoSys in corpora (BSL dicta-sign, DGS-Korpus, etc.) shows
a great deal of block reordering, alternation, and modifier fusion, the
grammar is intentionally permissive in *ordering* of atoms and focuses on
catching the hard structural errors:

* only known HamNoSys 4.0 codepoints (Private Use Area + the eight ASCII
  escapes listed in the CTAN manual § 5.11) — everything else is rejected;
* bracket pairing — ``seq/par/fusion`` begin markers must be closed by the
  matching end marker; the ASCII alternative pair ``{...|...}`` must also
  balance;
* non-empty body — a sign of zero atoms is not a sign.

Everything else (symmetry at the head, modifier after parent, at-least-one
handshape, palm-after-extended-finger) is a semantic check performed by
``validator.py`` on top of the parse tree — keeping the CFG uncomplicated
while still reporting those concerns with position info.
"""

from __future__ import annotations

from typing import Any

from lark import Lark, Tree


# NOTE: Lark's regex terminals use Python's ``re`` module, which accepts
# ``\uHHHH`` escapes verbatim. Character ranges like ``[\uE000-\uE00B]``
# match a single PUA codepoint. We do NOT ``%ignore`` whitespace — the ASCII
# space (U+0020) is a legitimate (if cosmetic) HamNoSys glyph.

HAMNOSYS_GRAMMAR = r"""
start: sign

sign: prefix? body

prefix: head_marker+
head_marker: SYMMETRY | NONDOM

body: atom+

atom: HANDSHAPE_BASE
    | THUMB_MOD
    | FINGER_MOD
    | EXT_FINGER_DIR
    | ORI_RELATIVE
    | PALM_DIR
    | LOC_HEAD
    | LOC_TORSO
    | LOC_NEUTRAL
    | LOC_ARM
    | LOC_HAND_ZONE
    | LOC_FINGER
    | COMBINER
    | COREF
    | MOVE_STRAIGHT
    | MOVE_CIRCLE
    | MOVE_ACTION
    | MOVE_CLOCK
    | MOVE_ARC
    | MOVE_WAVY
    | MOVE_ELLIPSE
    | SIZE_MOD
    | SPEED_MOD
    | TIMING
    | CONTACT
    | REPEAT
    | JOINER
    | MIME
    | VERSION
    | PUNCT
    | OBSOLETE
    | SYMMETRY
    | NONDOM
    | seq_block
    | par_block
    | fusion_block
    | alt_block

seq_block: SEQ_BEGIN body SEQ_END
par_block: PAR_BEGIN body PAR_END
fusion_block: FUSION_BEGIN body FUSION_END
alt_block: ALT_BEGIN body (META_ALT body)+ ALT_END

// -- Terminals ------------------------------------------------------------
// Ranges mirror the class groupings in symbols.py. We use string literals
// and string-range syntax (".." between two one-char literals) rather than
// Lark's regex mode because several PUNCT/JOINER codepoints — notably the
// ASCII pipe U+007C — are regex metacharacters, and the Earley frontend
// rejects zero-width or ambiguous regexes.

SYMMETRY:       "\uE0E8" | "\uE0E9"
NONDOM:         "\uE0EA" | "\uE0EB"

HANDSHAPE_BASE: "\uE000".."\uE00B"
THUMB_MOD:      "\uE00C".."\uE00E"
FINGER_MOD:     "\uE010".."\uE014"

EXT_FINGER_DIR: "\uE020".."\uE031"
ORI_RELATIVE:   "\uE0ED"
PALM_DIR:       "\uE038".."\uE03F"

LOC_HEAD:       "\uE040".."\uE04F"
LOC_TORSO:      "\uE050".."\uE054"
LOC_NEUTRAL:    "\uE05F"
LOC_ARM:        "\uE060".."\uE065"
LOC_HAND_ZONE:  "\uE066".."\uE06A"
LOC_FINGER:     "\uE070".."\uE07A"

COMBINER:       "\uE058" | "\uE059"
COREF:          "\uE05A" | "\uE05B"

MOVE_STRAIGHT:  "\uE080".."\uE091"
MOVE_CIRCLE:    "\uE092".."\uE0A3"
MOVE_ACTION:    "\uE0A4".."\uE0AA" | "\uE0AF"
MOVE_CLOCK:     "\uE0B0".."\uE0B8"
MOVE_ARC:       "\uE0B9".."\uE0BC"
MOVE_WAVY:      "\uE0BD" | "\uE0BE"
MOVE_ELLIPSE:   "\uE0C0".."\uE0C3"

SIZE_MOD:       "\uE0C4".."\uE0C7"
SPEED_MOD:      "\uE0C8".."\uE0CA"
TIMING:         "\uE0CB" | "\uE0CC"
CONTACT:        "\uE0D0".."\uE0D6"
REPEAT:         "\uE0D8".."\uE0DD"

SEQ_BEGIN:      "\uE0E0"
SEQ_END:        "\uE0E1"
PAR_BEGIN:      "\uE0E2"
PAR_END:        "\uE0E3"
FUSION_BEGIN:   "\uE0E4"
FUSION_END:     "\uE0E5"

JOINER:         "\uE0E6" | "\uE0E7" | "\uE0EC"
MIME:           "\uE0F0"
VERSION:        "\uE0F1"

ALT_BEGIN:      "{"
META_ALT:       "|"
ALT_END:        "}"

PUNCT:          " " | "!" | "," | "." | "?"
OBSOLETE:       "\uE07C".."\uE07F" | "\uE0AD" | "\uE0AE"
"""


def _make_parser() -> Lark:
    # Earley handles the structural ambiguity better than LALR; the input
    # strings are short (median ~16 chars, max under 200) so speed is fine.
    # ``propagate_positions`` attaches .line/.column/.start_pos to each Tree
    # — used by validator.py for error reporting.
    return Lark(
        HAMNOSYS_GRAMMAR,
        start="start",
        parser="earley",
        propagate_positions=True,
        keep_all_tokens=True,
    )


# Module-level singleton — compiling the grammar is the expensive part.
HamNoSysGrammar: Lark = _make_parser()


def parse_tree(s: str) -> Tree[Any]:
    """Parse ``s`` and return the Lark parse tree. Raises on parse errors."""
    return HamNoSysGrammar.parse(s)
