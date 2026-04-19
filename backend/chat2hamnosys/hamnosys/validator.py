"""HamNoSys 4.0 validator.

Public API::

    validate(s: str) -> ValidationResult
    normalize(s: str) -> str

``validate`` reports errors and warnings with positional info. ``normalize``
applies cheap textual fixups (NFC, strip whitespace, repair the specific
latin-1/utf-8 double-encoding mojibake found in the bundled DGS/BSL CSVs)
but does NOT attempt to reorder HamNoSys symbols — the canonical ordering
in HamNoSys is semantics-sensitive and beyond a safe text-pass.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from lark import Token, Tree
from lark.exceptions import (
    LarkError,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)

from .grammar import HamNoSysGrammar
from .symbols import SYMBOLS, Slot, SymClass, Symbol


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    position: int
    symbol: str
    message: str
    code: str

    def __str__(self) -> str:
        sym_repr = _describe_char(self.symbol)
        return f"[{self.code}] pos={self.position} {sym_repr}: {self.message}"


class ValidationError(ValidationIssue):
    """A hard violation — the string is not valid HamNoSys 4.0."""


class ValidationWarning(ValidationIssue):
    """A soft issue — renderable but suspicious (obsolete glyph, odd order)."""


@dataclass
class ValidationResult:
    ok: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    tree: Tree | None = None

    def __bool__(self) -> bool:
        return self.ok

    def summary(self) -> str:
        lines = ["ok" if self.ok else "invalid"]
        for e in self.errors:
            lines.append(f"  E {e}")
        for w in self.warnings:
            lines.append(f"  W {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def _looks_like_latin1_utf8_mojibake(s: str) -> bool:
    # The DGS/BSL CSVs ship with strings that have been round-tripped through
    # latin-1→utf-8 twice, so a genuine U+E001 "hamflathand" shows up as the
    # three-character sequence U+00EE U+0080 U+0081. Diagnose by the presence
    # of the U+00EE prefix followed by two Latin-1-supplement bytes.
    for i, ch in enumerate(s[:-2]):
        if ord(ch) == 0x00EE and 0x0080 <= ord(s[i + 1]) <= 0x00BF and 0x0080 <= ord(s[i + 2]) <= 0x00BF:
            return True
    return False


def _fix_mojibake(s: str) -> str:
    # Some corpus rows (e.g. rows in data/hamnosys_bsl.csv) have a trailing
    # truncated UTF-8 triple — use ``errors='ignore'`` on decode to preserve
    # whatever prefix is recoverable rather than bailing on the whole string.
    try:
        raw = s.encode("latin-1", errors="replace")
    except UnicodeEncodeError:
        return s
    try:
        return raw.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        return s


def normalize(s: str) -> str:
    """Apply conservative textual fixups to a candidate HamNoSys string.

    Steps, in order:
      1. NFC Unicode normalization.
      2. Detect and repair latin-1/utf-8 double-encoding mojibake
         (U+00EE U+00XX U+00XX → proper U+E0XX triple).
      3. Strip leading/trailing *ASCII* whitespace — mojibake repair is
         intentionally run first because U+00A0 is a valid last byte in
         the latin-1/utf-8 triple that encodes e.g. U+E020, and the
         default ``str.strip()`` would otherwise eat it.
    """
    if not isinstance(s, str):
        raise TypeError(f"normalize() expects str, got {type(s).__name__}")
    s = unicodedata.normalize("NFC", s)
    if _looks_like_latin1_utf8_mojibake(s):
        fixed = _fix_mojibake(s)
        # only accept the fixup if it actually produced PUA codepoints.
        if any(0xE000 <= ord(c) <= 0xE0FF for c in fixed):
            s = unicodedata.normalize("NFC", fixed)
    # Only ASCII whitespace — U+0020 hamspace inside the body is intentional,
    # and we must not strip U+00A0 et al. because they are mojibake payload.
    return s.strip(" \t\n\r\f\v")


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


_BRACKET_PAIRS = {
    "seq_block": (SymClass.SEQ_BEGIN, SymClass.SEQ_END),
    "par_block": (SymClass.PAR_BEGIN, SymClass.PAR_END),
    "fusion_block": (SymClass.FUSION_BEGIN, SymClass.FUSION_END),
    "alt_block": (SymClass.ALT_BEGIN, SymClass.ALT_END),
}


def _describe_char(ch: str) -> str:
    if len(ch) != 1:
        return f"{ch!r}"
    cp = ord(ch)
    sym = SYMBOLS.get(cp)
    if sym is not None:
        return f"U+{cp:04X}/{sym.short_name}"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "?"
    return f"U+{cp:04X}/{name}"


def _lex_unknowns(s: str) -> list[ValidationError]:
    """Report every codepoint that isn't in the HamNoSys 4.0 inventory."""
    errors: list[ValidationError] = []
    for i, ch in enumerate(s):
        if ord(ch) in SYMBOLS:
            continue
        errors.append(
            ValidationError(
                position=i,
                symbol=ch,
                message=f"unknown codepoint; not part of HamNoSys 4.0 — {_describe_char(ch)}",
                code="unknown_symbol",
            )
        )
    return errors


def _parse_to_error(exc: UnexpectedInput, s: str) -> ValidationError:
    """Convert a Lark parse error into a ValidationError with position."""
    pos = getattr(exc, "pos_in_stream", None)
    if pos is None:
        pos = len(s)
    pos = min(pos, max(len(s) - 1, 0))
    ch = s[pos] if 0 <= pos < len(s) else ""
    if isinstance(exc, UnexpectedEOF):
        msg = "unexpected end of input — an opened bracket or group is not closed"
        code = "truncated"
    elif isinstance(exc, UnexpectedCharacters):
        # Lark reports UnexpectedCharacters for both truly unknown chars and
        # for valid tokens that are out of position (e.g. SEQ_END without a
        # matching SEQ_BEGIN). Disambiguate by consulting the symbol table.
        if 0 <= pos < len(s) and ord(s[pos]) in SYMBOLS:
            msg = "symbol is not valid in this context — likely a mismatched bracket or out-of-order atom"
            code = "bad_order"
        else:
            msg = "character is not a HamNoSys 4.0 symbol"
            code = "unknown_symbol"
    elif isinstance(exc, UnexpectedToken):
        expected = ", ".join(sorted(exc.expected)) if exc.expected else "?"
        msg = f"unexpected symbol here — grammar expected one of: {expected}"
        code = "bad_order"
    else:
        msg = f"parse error: {exc!s}"
        code = "parse_error"
    return ValidationError(position=pos, symbol=ch, message=msg, code=code)


def _collect_tokens(tree: Tree) -> list[Token]:
    """Return all Tokens (with position info) in the tree, in document order."""
    tokens: list[Token] = []

    def walk(node):
        if isinstance(node, Token):
            tokens.append(node)
            return
        for c in node.children:
            walk(c)

    walk(tree)
    tokens.sort(key=lambda t: (t.start_pos if t.start_pos is not None else 0))
    return tokens


def _semantic_checks(s: str, tree: Tree) -> tuple[list[ValidationError], list[ValidationWarning]]:
    """Run anatomical and ordering checks on top of the parse tree."""
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    tokens = _collect_tokens(tree)
    classes: list[tuple[int, str, SymClass]] = []
    for tok in tokens:
        ch = str(tok)
        if len(ch) != 1:
            continue
        sym = SYMBOLS.get(ord(ch))
        if sym is None:
            continue  # unknowns already reported by _lex_unknowns
        classes.append((tok.start_pos or 0, ch, sym.sym_class))

    # 1. Must contain at least one handshape base or a bracketed structure.
    has_handshape = any(c is SymClass.HANDSHAPE_BASE for _, _, c in classes)
    has_bracket_open = any(
        c in {SymClass.SEQ_BEGIN, SymClass.PAR_BEGIN, SymClass.FUSION_BEGIN, SymClass.ALT_BEGIN}
        for _, _, c in classes
    )
    if not has_handshape and not has_bracket_open:
        errors.append(
            ValidationError(
                position=0,
                symbol=s[0] if s else "",
                message="sign has no handshape base (at least one of U+E000–U+E00B is required, "
                "either directly or inside a bracketed alternation/parallel group)",
                code="missing_handshape",
            )
        )

    # 2. Symmetry and nondominant prefix markers should appear at the head.
    #    We allow the first N symmetry/nondom tokens to be a "prefix"; any
    #    such marker appearing after a non-prefix token is a warning.
    seen_non_prefix = False
    for pos, ch, cls in classes:
        if cls in (SymClass.SYMMETRY, SymClass.NONDOM):
            if seen_non_prefix:
                warnings.append(
                    ValidationWarning(
                        position=pos,
                        symbol=ch,
                        message=f"{cls.value} marker should normally appear at the start of a sign, "
                        "before any handshape or location",
                        code="symmetry_not_at_head",
                    )
                )
        else:
            seen_non_prefix = True

    # 3. Modifier tokens should follow a compatible parent.
    #    THUMB_MOD and FINGER_MOD need a handshape context (HANDSHAPE_BASE
    #    or another modifier, plus a finger-part or hand-zone symbol when
    #    used inside a handshape block to mark the affected finger);
    #    SIZE/SPEED/TIMING modifiers attach to a movement or contact;
    #    ORI_RELATIVE anchors to a preceding EXT_FINGER_DIR / PALM_DIR.
    #
    #    These are reported as *warnings* rather than errors because the
    #    DGS-Korpus and BSL dicta-sign datasets both contain canonical signs
    #    where the convention is relaxed (e.g. sign-level modifiers placed
    #    right after a symmetry operator before any handshape).
    parents_for_class: dict[SymClass, set[SymClass]] = {
        SymClass.THUMB_MOD: {
            SymClass.HANDSHAPE_BASE, SymClass.THUMB_MOD, SymClass.FINGER_MOD,
            SymClass.LOC_FINGER, SymClass.LOC_HAND_ZONE,
        },
        SymClass.FINGER_MOD: {
            SymClass.HANDSHAPE_BASE, SymClass.THUMB_MOD, SymClass.FINGER_MOD,
            SymClass.LOC_FINGER, SymClass.LOC_HAND_ZONE,
        },
        SymClass.SIZE_MOD: {
            SymClass.MOVE_STRAIGHT, SymClass.MOVE_CIRCLE, SymClass.MOVE_ACTION,
            SymClass.MOVE_CLOCK, SymClass.MOVE_ARC, SymClass.MOVE_WAVY, SymClass.MOVE_ELLIPSE,
            SymClass.CONTACT, SymClass.REPEAT, SymClass.SIZE_MOD, SymClass.SPEED_MOD,
            SymClass.TIMING, SymClass.SYMMETRY, SymClass.FINGER_MOD,
        },
        SymClass.SPEED_MOD: {
            SymClass.MOVE_STRAIGHT, SymClass.MOVE_CIRCLE, SymClass.MOVE_ACTION,
            SymClass.MOVE_CLOCK, SymClass.MOVE_ARC, SymClass.MOVE_WAVY, SymClass.MOVE_ELLIPSE,
            SymClass.CONTACT, SymClass.SIZE_MOD, SymClass.SPEED_MOD, SymClass.TIMING,
            SymClass.SYMMETRY, SymClass.FINGER_MOD,
        },
        SymClass.TIMING: {
            SymClass.MOVE_STRAIGHT, SymClass.MOVE_CIRCLE, SymClass.MOVE_ACTION,
            SymClass.MOVE_CLOCK, SymClass.MOVE_ARC, SymClass.MOVE_WAVY, SymClass.MOVE_ELLIPSE,
            SymClass.CONTACT, SymClass.SIZE_MOD, SymClass.SPEED_MOD, SymClass.TIMING,
        },
        SymClass.ORI_RELATIVE: {SymClass.EXT_FINGER_DIR, SymClass.PALM_DIR},
    }

    for i, (pos, ch, cls) in enumerate(classes):
        parents = parents_for_class.get(cls)
        if parents is None:
            continue
        # Look leftward for the nearest non-bracket, non-punctuation token.
        found_parent = False
        for j in range(i - 1, -1, -1):
            _, _, prev_cls = classes[j]
            if prev_cls in {
                SymClass.SEQ_BEGIN, SymClass.PAR_BEGIN, SymClass.FUSION_BEGIN,
                SymClass.ALT_BEGIN, SymClass.PUNCT,
            }:
                continue
            if prev_cls in {
                SymClass.SEQ_END, SymClass.PAR_END, SymClass.FUSION_END,
                SymClass.ALT_END, SymClass.META_ALT,
            }:
                break
            if prev_cls in parents:
                found_parent = True
            break
        if not found_parent:
            warnings.append(
                ValidationWarning(
                    position=pos,
                    symbol=ch,
                    message=f"{cls.value} has no matching preceding parent in its local region; "
                    f"expected one of: " + ", ".join(sorted(p.value for p in parents)),
                    code="orphan_modifier",
                )
            )

    # 4. Two consecutive handshape bases without an intervening COMBINER /
    #    JOINER / bracket are anatomically impossible (one hand cannot hold
    #    two handshapes simultaneously). This often indicates a missing
    #    modifier or combiner.
    prev_atom_cls: SymClass | None = None
    prev_atom_pos: int = 0
    prev_atom_ch: str = ""
    SEPARATORS = {
        SymClass.COMBINER, SymClass.JOINER, SymClass.SEQ_BEGIN, SymClass.SEQ_END,
        SymClass.PAR_BEGIN, SymClass.PAR_END, SymClass.FUSION_BEGIN, SymClass.FUSION_END,
        SymClass.ALT_BEGIN, SymClass.ALT_END, SymClass.META_ALT, SymClass.PUNCT,
    }
    for pos, ch, cls in classes:
        if cls in SEPARATORS:
            prev_atom_cls = None
            continue
        if cls is SymClass.HANDSHAPE_BASE and prev_atom_cls is SymClass.HANDSHAPE_BASE:
            errors.append(
                ValidationError(
                    position=pos,
                    symbol=ch,
                    message="two handshape bases in sequence without a combiner/joiner — "
                    "one hand cannot hold two shapes simultaneously",
                    code="double_handshape",
                )
            )
        prev_atom_cls = cls
        prev_atom_pos = pos
        prev_atom_ch = ch

    # 5. Obsolete glyphs — warn, don't fail.
    for pos, ch, cls in classes:
        if cls is SymClass.OBSOLETE:
            warnings.append(
                ValidationWarning(
                    position=pos,
                    symbol=ch,
                    message="obsolete symbol (still supported by the font, deprecated in HamNoSys 4.0)",
                    code="obsolete_symbol",
                )
            )

    return errors, warnings


def validate(s: str) -> ValidationResult:
    """Validate a candidate HamNoSys 4.0 string.

    Pipeline:
      1. Empty-string short-circuit.
      2. Lexical check: flag unknown codepoints.
      3. Grammar parse: catch bracket imbalance, unexpected-order issues.
      4. Semantic pass: anatomical and ordering rules (missing handshape,
         orphan modifiers, symmetry-not-at-head, obsolete glyph, etc.).
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    if not isinstance(s, str):
        raise TypeError(f"validate() expects str, got {type(s).__name__}")

    if s == "":
        errors.append(
            ValidationError(
                position=0,
                symbol="",
                message="empty string is not a valid HamNoSys sign",
                code="empty",
            )
        )
        return ValidationResult(ok=False, errors=errors, warnings=warnings, tree=None)

    # Pass 1: unknown codepoints (Latin letters, control chars, stray CJK).
    unknown_errors = _lex_unknowns(s)
    errors.extend(unknown_errors)

    # Pass 2: grammar parse. If there are unknown codepoints the parse will
    # also fail on them — we run it anyway to surface bracket-imbalance and
    # ordering errors beyond the first unknown.
    tree: Tree | None = None
    try:
        tree = HamNoSysGrammar.parse(s)
    except UnexpectedInput as exc:
        parse_err = _parse_to_error(exc, s)
        # Avoid double-reporting: if the parse error is just one of the
        # already-known unknown codepoints, skip it.
        already = {(e.position, e.code) for e in unknown_errors}
        if (parse_err.position, "unknown_symbol") not in already and parse_err.code != "unknown_symbol":
            errors.append(parse_err)
        elif parse_err.code == "unknown_symbol" and not unknown_errors:
            errors.append(parse_err)
    except LarkError as exc:  # pragma: no cover — defensive
        errors.append(
            ValidationError(
                position=0,
                symbol=s[0],
                message=f"grammar engine error: {exc}",
                code="parse_error",
            )
        )

    # Pass 3: semantic checks (only if we have a parse tree).
    if tree is not None:
        sem_errs, sem_warns = _semantic_checks(s, tree)
        errors.extend(sem_errs)
        warnings.extend(sem_warns)

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, tree=tree)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def validate_normalized(s: str) -> ValidationResult:
    """Run ``normalize`` then ``validate``."""
    return validate(normalize(s))


def iter_symbols(s: str) -> Iterable[tuple[int, str, Symbol | None]]:
    """Yield ``(position, char, Symbol_or_None)`` for each char in ``s``."""
    for i, ch in enumerate(s):
        yield i, ch, SYMBOLS.get(ord(ch))
