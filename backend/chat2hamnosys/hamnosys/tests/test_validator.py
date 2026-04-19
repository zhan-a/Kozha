"""Tests for the HamNoSys 4.0 validator.

The fixture block ``REAL_CORPUS_SIGNS`` seeds the suite with 36 signs taken
verbatim from six separate sign-language corpora; the SIGML collections
(DGS/LSF/NGT/GSL/PJM) are those bundled by the Dicta-Sign EU project
(https://dicta-sign.eu/) and the BSL rows come from the Hamburg dicta-sign
portal. Source URLs for each row are in the inline comments.

Each fixture was round-tripped through ``validate()`` when this file was
generated — if a recorded sign now fails, it means the validator changed;
investigate the failure in ``test_real_corpus_signs_validate_clean``.

Known-malformed corpus rows are kept as a separate fixture
(``CORPUS_MALFORMED``) with a brief diagnosis — the verdict for each is
"source is malformed, not a validator bug".
"""

from __future__ import annotations

import unicodedata

import pytest

from hamnosys import (
    SymClass,
    ValidationError,
    ValidationWarning,
    classify,
    codepoints_in_class,
    normalize,
    validate,
)


# ---------------------------------------------------------------------------
# Well-formed corpus fixtures — extracted from the bundled data/ directory
# ---------------------------------------------------------------------------
#
# Provenance:
#   BSL     — Hamburg dicta-sign portal CSV export. The per-row page URL
#             (sign-lang.uni-hamburg.de/dicta-sign/portal/...) is included
#             in the comment for each BSL entry.
#   DGS     — DGS-Korpus / meinedgs (https://www.sign-lang.uni-hamburg.de/meinedgs/).
#   LSF/NGT/GSL/PJM — SIGML collections from the Dicta-Sign 4 project
#             (https://dicta-sign.eu/).  LSF = French SL, NGT = Dutch SL,
#             GSL = Greek SL, PJM = Polish SL (also Korpus PJM —
#             https://www.korpuspjm.uw.edu.pl/en).
#
# Format: (corpus_tag, gloss, hamnosys_string)

REAL_CORPUS_SIGNS: list[tuple[str, str, str]] = [
    # BSL: abroad(a)#1
    #   https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_2.html
    ("BSL", "abroad(a)#1",
     "\uE001\uE00C\uE011\uE0E6\uE001\uE00C\uE020\uE03C\uE040\uE059\uE089\uE0C6\uE0D8"),
    # BSL: accept(v)#2
    #   https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_3.html
    ("BSL", "accept(v)#2",
     "\uE0E9\uE005\uE00E\uE028\uE0E6\uE029\uE038\uE0E2\uE08B\uE0AA\uE000\uE038\uE084\uE0E3\uE054\uE059\uE0D1"),
    # BSL: lodge(v)#4
    #   https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_5.html
    ("BSL", "lodge(v)#4",
     "\uE0E9\uE002\uE00E\uE011\uE0E6\uE002\uE00E\uE027\uE03C\uE042\uE059\uE0D1"
     "\uE0E2\uE084\uE0AA\uE028\uE0E6\uE029\uE051\uE0E3"),
    # BSL: actor(n)#1
    #   https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_9.html
    ("BSL", "actor(n)#1",
     "\uE0E9\uE0DD\uE000\uE00C\uE026\uE03E\uE0E6\uE03D\uE052\uE059\uE0D1"
     "\uE0E2\uE097\uE0AA\uE026\uE03C\uE0E6\uE03D\uE0E3\uE0D8"),
    # BSL: adapt(v)#1
    #   https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_10.html
    ("BSL", "adapt(v)#1",
     "\uE0E8\uE0E2\uE002\uE00D\uE012\uE020\uE03E\uE0E7\uE002\uE012\uE00D\uE020\uE03A\uE0E3"
     "\uE0D1\uE051\uE0E2\uE089\uE0C6\uE0E7\uE08C\uE0C6\uE0E3\uE0D8"),
    # BSL: address(n)#6
    #   https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_11.html
    ("BSL", "address(n)#6",
     "\uE001\uE00C\uE024\uE0E6\uE025\uE03E\uE0E2\uE0AA\uE026\uE03E\uE0E3\uE052\uE0D0\uE0D8"),

    # DGS: A-MOMENT-AGO1^
    #   https://www.sign-lang.uni-hamburg.de/meinedgs/
    ("DGS", "A-MOMENT-AGO1^",
     "\uE002\uE011\uE020\uE038\uE050\uE059\uE0D0\uE0AA\uE02E\uE03C\uE0D1\uE0D8"),
    # DGS: A-WHOLE-LOT1^
    #   https://www.sign-lang.uni-hamburg.de/meinedgs/
    ("DGS", "A-WHOLE-LOT1^",
     "\uE005\uE0E6\uE001\uE010\uE031\uE028\uE03E\uE052\uE059\uE0E2\uE084\uE0C6\uE0AA\uE030"
     "\uE028\uE03D\uE0E3\uE0D9"),
    # DGS: ABOVE1^
    #   https://www.sign-lang.uni-hamburg.de/meinedgs/
    ("DGS", "ABOVE1^",
     "\uE002\uE020\uE03D\uE0E6\uE03E\uE051\uE059\uE080"),
    # DGS: ABOVE2A^
    #   https://www.sign-lang.uni-hamburg.de/meinedgs/
    ("DGS", "ABOVE2A^",
     "\uE001\uE010\uE020\uE03E\uE051\uE059\uE080"),
    # DGS: ABOVE2B^
    #   https://www.sign-lang.uni-hamburg.de/meinedgs/
    ("DGS", "ABOVE2B^",
     "\uE0E2\uE001\uE010\uE020\uE03E\uE0E7\uE001\uE010\uE020\uE03A\uE0E3"
     "\uE0E2\uE076\uE0E7\uE077\uE0E3\uE0D0\uE080"),
    # DGS: ABSOLUTE1^
    #   https://www.sign-lang.uni-hamburg.de/meinedgs/
    ("DGS", "ABSOLUTE1^",
     "\uE0E8\uE001\uE031\uE03E\uE0E2\uE084\uE0CC\uE0AA\uE029\uE0E3"),

    # LSF (Dicta-Sign 4 French SL SIGML collection)
    #   https://dicta-sign.eu/
    ("LSF", "_NUM-QUATORZE",
     "\uE005\uE00D\uE025\uE03E\uE052\uE0AA\uE027\uE03E"),
    ("LSF", "_NUM-QUATRE",
     "\uE005\uE00D\uE027\uE03E\uE050\uE089\uE0C6"),
    ("LSF", "_NUM-QUINZE",
     "\uE005\uE00C\uE025\uE03E\uE050\uE0AA\uE027\uE03E"),
    ("LSF", "_NUM-SEIZE",
     "\uE0E9\uE0E2\uE000\uE00C\uE027\uE03E\uE0E7\uE005\uE00C\uE021\uE03A\uE0E3"
     "\uE050\uE0AA\uE0E2\uE025\uE0E7\uE023\uE0E3\uE052"),
    ("LSF", "_NUM-TREIZE",
     "\uE004\uE00C\uE025\uE03E\uE050\uE0AA\uE027\uE03E"),
    ("LSF", "_NUM-UN",
     "\uE000\uE00C\uE029\uE03E\uE050\uE089\uE0C6"),

    # NGT (Dicta-Sign 4 Dutch SL SIGML collection)
    #   https://dicta-sign.eu/
    ("NGT", "DAAROM",
     "\uE000\uE00D\uE028\uE0E6\uE031\uE03F\uE051\uE09C\uE0B2"),
    ("NGT", "DAT",
     "\uE002\uE00D\uE010\uE0E6\uE002\uE00D\uE031\uE03C\uE051\uE0E6\uE059\uE052"
     "\uE0E2\uE090\uE0C6\uE0AA\uE029\uE0E3"),
    ("NGT", "DOEL",
     "\uE0AF\uE002\uE00D\uE031\uE0E6\uE029\uE03E\uE04F\uE059\uE0D0"
     "\uE0E2\uE089\uE0AA\uE029\uE0E3"),
    ("NGT", "DOOF_(ZIJN)_01",
     "\uE002\uE00D\uE020\uE0E6\uE027\uE03C\uE0E6\uE03D\uE047\uE059"
     "\uE0E0\uE0D1\uE071\uE077\uE0E1\uE0AF"),
    ("NGT", "GEBAREN_B",
     "\uE0E9\uE0DD\uE005\uE00C\uE031\uE03E\uE0D0\uE051\uE0E6\uE052\uE0D0\uE096\uE0C6"),
    ("NGT", "GEMAKKELIJK_02",
     "\uE001\uE00C\uE020\uE0E6\uE027\uE03E\uE04D\uE0E0\uE0D0\uE072\uE077\uE0E1\uE08C\uE0C8"
     "\uE04D\uE0E0\uE0D1\uE072\uE077\uE0E1\uE0D9"),

    # GSL (Dicta-Sign 4 Greek SL SIGML collection)
    #   https://dicta-sign.eu/
    ("GSL", "0",
     "\uE008\uE010\uE071\uE020\uE03D\uE0E6\uE03E\uE050\uE059\uE0D0\uE089\uE0C6\uE0CC"),
    ("GSL", "1",
     "\uE002\uE00D\uE020\uE038\uE051\uE059\uE0D0\uE089\uE0C6\uE0CC"),
    ("GSL", "10",
     "\uE0E9\uE0E2\uE005\uE00C\uE0E7\uE005\uE00C\uE0E3\uE020\uE038\uE051\uE059"
     "\uE0D0\uE089\uE0C6\uE0CC"),
    ("GSL", "2",
     "\uE004\uE00D\uE020\uE038\uE051\uE059\uE0D0\uE089\uE0C6\uE0CC"),
    ("GSL", "3",
     "\uE005\uE00D\uE074\uE014\uE020\uE038\uE051\uE059\uE0D0\uE089\uE0C6\uE0CC"),
    ("GSL", "4",
     "\uE005\uE00D\uE020\uE038\uE051\uE059\uE0D0\uE089\uE0C6\uE0CC"),

    # PJM (Korpus PJM / Dicta-Sign 4 Polish SL SIGML collection)
    #   https://www.korpuspjm.uw.edu.pl/en  +  https://dicta-sign.eu/
    ("PJM", "aby",
     "\uE000\uE00C\uE026\uE03E\uE052\uE059\uE0D1\uE0E2\uE081\uE0AA\uE027\uE0E3"),
    ("PJM", "aby/\u017Ceby",
     "\uE0E9\uE000\uE00D\uE026\uE03E\uE052\uE059\uE0D1\uE081"),
    ("PJM", "adres",
     "\uE0E2\uE000\uE00D\uE029\uE03E\uE0E7\uE001\uE02A\uE038\uE0E3"
     "\uE067\uE0E0\uE0D1\uE06A\uE0E1\uE084\uE0C6\uE0CC\uE0D8"),
    ("PJM", "adwokat",
     "\uE0E9\uE000\uE00D\uE029\uE038\uE0AA\uE005\uE00C\uE011\uE0D8"),
    ("PJM", "Afryka",
     "\uE004\uE00C\uE020\uE03E\uE04D\uE0E0\uE0D1\uE077\uE0E1"
     "\uE0E2\uE089\uE0C6\uE0BA\uE0AA\uE029\uE0E3\uE0D8"),
    ("PJM", "albo/lub",
     "\uE0E9\uE010\uE0C7\uE001\uE029\uE038\uE080\uE0C6\uE0D8"),
]


# Corpus rows that FAIL validation. The verdict for each is that the source
# row is malformed — these are documented so future regressions that make any
# of these pass would be a red flag.
CORPUS_MALFORMED: list[tuple[str, str, str, str]] = [
    # DGS MONTH-CALENDAR11^ — the recorded sign has no handshape base at all,
    # only an extended-finger direction + location. HamNoSys requires at least
    # one of U+E000–U+E00B. (Source: meinedgs DGS-Korpus.)
    ("DGS", "MONTH-CALENDAR11^",
     "\uE020\uE03C\uE051\uE059\uE084",
     "no handshape base — source row is incomplete"),
    # DGS TO-BE-IN-ONES-BLOOD1A^ — a parbegin (U+E0E2) at pos 0 is closed by a
    # parend (U+E0E3) at pos 10, but then a stray seqend (U+E0E1) shows up at
    # pos 19 with no matching seqbegin. Bracket imbalance in the source.
    ("DGS", "TO-BE-IN-ONES-BLOOD1A^",
     "\uE0E2\uE002\uE011\uE026\uE03C\uE0E7\uE0EA\uE0EC\uE02A\uE038\uE0E3"
     "\uE063\uE067\uE0D1\uE0E2\uE08B\uE0C6\uE08A\uE0C6\uE0E1\uE0D8",
     "bracket imbalance — stray SEQ_END without a matching SEQ_BEGIN"),
]


# ---------------------------------------------------------------------------
# Basic API shape
# ---------------------------------------------------------------------------


def test_validate_returns_validation_result_shape():
    """The public surface contract: ok / errors / warnings / tree."""
    # A minimal well-formed sign: handshape "hamfist" (U+E000).
    res = validate("\uE000")
    assert res.ok is True
    assert bool(res) is True
    assert res.errors == []
    assert isinstance(res.warnings, list)
    assert res.tree is not None


def test_validate_empty_string_is_an_error():
    res = validate("")
    assert res.ok is False
    assert bool(res) is False
    assert len(res.errors) == 1
    err = res.errors[0]
    assert err.code == "empty"
    assert err.position == 0
    assert "empty" in err.message.lower()


def test_validate_rejects_non_string():
    with pytest.raises(TypeError):
        validate(None)  # type: ignore[arg-type]


def test_error_has_position_symbol_message():
    """Contract: every issue exposes .position, .symbol, .message."""
    res = validate("A")  # Latin letter
    assert res.ok is False
    err = res.errors[0]
    assert isinstance(err, ValidationError)
    assert err.position == 0
    assert err.symbol == "A"
    assert err.message  # non-empty string
    assert err.code == "unknown_symbol"


# ---------------------------------------------------------------------------
# Citation-form well-formed signs (minimal synthetic examples + ≥30 real)
# ---------------------------------------------------------------------------


def test_minimal_citation_sign_handshape_only():
    """A single handshape base is a legal HamNoSys citation sign."""
    for cp in codepoints_in_class(SymClass.HANDSHAPE_BASE):
        res = validate(chr(cp))
        assert res.ok, f"U+{cp:04X} alone should validate: {res.summary()}"


def test_citation_sign_with_orientation_and_location():
    """handshape (U+E002 flat hand) + ext-finger-dir + palm-dir + loc-head."""
    s = "\uE002\uE020\uE03C\uE040"
    res = validate(s)
    assert res.ok, res.summary()


def test_citation_sign_with_symmetry_prefix():
    """symmetry-parallel + handshape + finger-mod is valid."""
    s = "\uE0E8\uE005\uE010"
    res = validate(s)
    assert res.ok, res.summary()


@pytest.mark.parametrize(
    "corpus,gloss,sign",
    REAL_CORPUS_SIGNS,
    ids=[f"{c}-{g}" for c, g, _ in REAL_CORPUS_SIGNS],
)
def test_real_corpus_signs_validate_clean(corpus: str, gloss: str, sign: str):
    """Every seeded real-world corpus sign must validate without errors.

    If one of these fails, investigate whether:
      (a) the validator lost a feature (regression — fix the validator), or
      (b) the source row has been updated upstream (update the fixture).
    """
    res = validate(sign)
    assert res.ok, f"{corpus} {gloss} failed: {res.summary()}"


def test_at_least_thirty_real_corpus_signs_are_seeded():
    """Prompt 2 requires ≥30 real strings from the corpora. Guard the count."""
    assert len(REAL_CORPUS_SIGNS) >= 30, len(REAL_CORPUS_SIGNS)
    corpora = {c for c, _, _ in REAL_CORPUS_SIGNS}
    # cross-language coverage, not just DGS
    assert len(corpora) >= 4, corpora


# ---------------------------------------------------------------------------
# Corpus-malformed — documented "verdict: source is malformed"
# ---------------------------------------------------------------------------


def test_corpus_month_calendar_missing_handshape():
    """DGS MONTH-CALENDAR11^ lacks a handshape base (corpus bug)."""
    _, _, sign, _ = CORPUS_MALFORMED[0]
    res = validate(sign)
    assert not res.ok
    codes = {e.code for e in res.errors}
    assert "missing_handshape" in codes


def test_corpus_to_be_in_ones_blood_bracket_imbalance():
    """DGS TO-BE-IN-ONES-BLOOD1A^ has a stray SEQ_END (corpus bug)."""
    _, _, sign, _ = CORPUS_MALFORMED[1]
    res = validate(sign)
    assert not res.ok
    codes = {e.code for e in res.errors}
    # Either classed as bad_order (mismatched bracket) or truncated
    # (the grammar may also flag an unclosed group earlier). Both are
    # acceptable diagnostics for a real bracket-imbalance bug.
    assert codes & {"bad_order", "truncated"}


def test_corpus_empty_cell_from_bsl_csv():
    """The BSL empty(v)#1 row, after mojibake repair, collapses to empty.

    This documents a genuine corpus defect rather than a validator issue;
    see https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_278.html.
    """
    # The raw cell is a very short mojibake fragment whose repair yields "".
    # We simulate by feeding the empty string directly.
    res = validate(normalize(""))
    assert not res.ok
    assert any(e.code == "empty" for e in res.errors)


# ---------------------------------------------------------------------------
# Malformed ordering / structural errors
# ---------------------------------------------------------------------------


def test_unclosed_seq_block_is_truncated():
    """U+E0E0 (seqbegin) opened but never closed."""
    s = "\uE000\uE0E0\uE084"
    res = validate(s)
    assert not res.ok
    assert any(e.code == "truncated" for e in res.errors)


def test_unclosed_par_block_is_truncated():
    s = "\uE000\uE0E2\uE084"
    res = validate(s)
    assert not res.ok
    assert any(e.code == "truncated" for e in res.errors)


def test_unclosed_fusion_block_is_truncated():
    s = "\uE000\uE0E4\uE084"
    res = validate(s)
    assert not res.ok
    assert any(e.code == "truncated" for e in res.errors)


def test_unclosed_alt_block_is_truncated():
    # '{' without matching '}' / '|'
    s = "\uE000{\uE084"
    res = validate(s)
    assert not res.ok
    assert any(e.code in {"truncated", "bad_order"} for e in res.errors)


def test_seqend_without_seqbegin_is_rejected():
    """A SEQ_END at the top level with no opener is malformed."""
    s = "\uE000\uE0E1"
    res = validate(s)
    assert not res.ok
    codes = {e.code for e in res.errors}
    # Should be recognised as a structural mismatch, not a "wholly unknown".
    assert codes & {"bad_order", "truncated"}, codes


def test_parend_without_parbegin_is_rejected():
    s = "\uE000\uE0E3"
    res = validate(s)
    assert not res.ok
    assert {e.code for e in res.errors} & {"bad_order", "truncated"}


def test_double_handshape_base_is_an_error():
    """Two adjacent handshape bases without a combiner/joiner is anatomically impossible."""
    s = "\uE000\uE002"
    res = validate(s)
    assert not res.ok
    assert any(e.code == "double_handshape" for e in res.errors)


def test_double_handshape_separated_by_combiner_is_ok():
    """...but with a combiner (e.g. hamlrbeside U+E058) between them it's fine."""
    s = "\uE000\uE058\uE002"
    res = validate(s)
    assert res.ok, res.summary()


def test_no_handshape_at_all_is_an_error():
    """Only location + movement, no handshape. Missing-handshape error."""
    s = "\uE040\uE080"
    res = validate(s)
    assert not res.ok
    assert any(e.code == "missing_handshape" for e in res.errors)


# ---------------------------------------------------------------------------
# Unknown codepoints and mixed-script pollution
# ---------------------------------------------------------------------------


def test_single_latin_letter_is_unknown():
    res = validate("Q")
    assert not res.ok
    assert res.errors[0].code == "unknown_symbol"
    assert res.errors[0].symbol == "Q"
    assert res.errors[0].position == 0


def test_latin_letters_embedded_in_valid_sign():
    """'abc' interleaved with real HamNoSys — every Latin letter should be reported."""
    s = "\uE000a\uE010b\uE020c"
    res = validate(s)
    assert not res.ok
    unknown_positions = [
        e.position for e in res.errors if e.code == "unknown_symbol"
    ]
    # Each of the three Latin letters is at a distinct odd position.
    assert unknown_positions == [1, 3, 5]


def test_cjk_ideograph_is_unknown():
    """U+4E2D (Chinese 中) is outside HamNoSys."""
    s = "\uE000\u4E2D"
    res = validate(s)
    assert not res.ok
    assert any(e.symbol == "\u4E2D" for e in res.errors)


def test_unrelated_private_use_codepoint_is_unknown():
    """U+E100 is in the PUA but outside the HamNoSys 4.0 allocation."""
    s = "\uE000\uE100"
    res = validate(s)
    assert not res.ok
    assert any(e.code == "unknown_symbol" and e.symbol == "\uE100" for e in res.errors)


def test_control_character_is_unknown():
    """NUL is never a HamNoSys symbol."""
    s = "\uE000\x00"
    res = validate(s)
    assert not res.ok
    assert any(e.code == "unknown_symbol" for e in res.errors)


def test_emoji_is_unknown():
    s = "\uE000\U0001F600"  # grinning face
    res = validate(s)
    assert not res.ok
    assert any(e.code == "unknown_symbol" for e in res.errors)


# ---------------------------------------------------------------------------
# Truncated signs
# ---------------------------------------------------------------------------


def test_only_a_symmetry_marker_is_truncated():
    """A symmetry prefix alone, with no body, is not a sign."""
    s = "\uE0E8"
    res = validate(s)
    assert not res.ok


def test_only_a_handshape_modifier_is_truncated():
    """A thumb modifier with no handshape is an orphan — parse may still fail."""
    s = "\uE00C"
    res = validate(s)
    assert not res.ok


def test_nothing_but_a_bracket_opener():
    s = "\uE0E0"
    res = validate(s)
    assert not res.ok


def test_nothing_but_alt_opener():
    s = "{"
    res = validate(s)
    assert not res.ok


# ---------------------------------------------------------------------------
# Semantic warnings (parse is OK but something is suspicious)
# ---------------------------------------------------------------------------


def test_symmetry_after_body_emits_warning():
    """A symmetry operator should be at the head; mid-sign it's a warning."""
    s = "\uE000\uE0E8\uE010"
    res = validate(s)
    # Grammar accepts (prefix is optional-maximal-prefix, mid-string symm is
    # just an atom), but the semantic pass flags it.
    assert any(w.code == "symmetry_not_at_head" for w in res.warnings)


def test_obsolete_symbol_emits_warning():
    """U+E07C is an obsolete wrist-location glyph; still parseable, but warn."""
    s = "\uE000\uE07C"
    res = validate(s)
    assert res.ok, res.summary()
    assert any(w.code == "obsolete_symbol" for w in res.warnings)


def test_warning_instance_is_a_validation_warning():
    s = "\uE000\uE07C"
    res = validate(s)
    assert all(isinstance(w, ValidationWarning) for w in res.warnings)
    for w in res.warnings:
        assert w.position >= 0
        assert w.symbol
        assert w.message


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_is_idempotent_on_clean_input():
    for _, _, s in REAL_CORPUS_SIGNS[:5]:
        assert normalize(s) == normalize(normalize(s))


def test_normalize_strips_outer_whitespace():
    s = "  \n\t\uE000\uE010  \n"
    assert normalize(s) == "\uE000\uE010"


def test_normalize_preserves_internal_space_glyph():
    """U+0020 is the hamspace cosmetic glyph — do not trim it internally."""
    s = "\uE000 \uE002"  # fist, space, flathand (with a combiner-ish separator)
    # The internal space is not trimmed; it's the outer strip that runs.
    assert normalize(s) == s


def test_normalize_repairs_latin1_utf8_double_encoding():
    """The BSL dicta-sign CSV encodes each PUA char as a 3-byte Latin-1 triple.

    U+E001 (hamflathand) in UTF-8 is the byte sequence EE 80 81; after an
    extra Latin-1 decode it becomes the three codepoints U+00EE U+0080 U+0081.
    normalize() must spot this and round-trip it back to U+E001.
    """
    # Build the mojibake form from a known-good string.
    good = "\uE001\uE010\uE020"
    mojibake = good.encode("utf-8").decode("latin-1")
    assert mojibake != good  # sanity: the mojibake form is different
    assert normalize(mojibake) == good


def test_normalize_applies_nfc():
    """NFC joins a decomposed sequence. HamNoSys is flat ASCII/PUA so the
    effect is mostly defensive — we still want the property to hold."""
    s = "e\u0301"  # e + combining acute
    out = normalize(s)
    assert out == unicodedata.normalize("NFC", s.strip())


def test_normalize_refuses_non_string():
    with pytest.raises(TypeError):
        normalize(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Classification & inventory helpers
# ---------------------------------------------------------------------------


def test_classify_round_trips_for_every_real_sign_char():
    """Every char used in the real-corpus fixtures must classify."""
    for _, gloss, s in REAL_CORPUS_SIGNS:
        for ch in s:
            cls = classify(ch)
            assert cls is not None, f"unclassified char U+{ord(ch):04X} in {gloss}"


def test_classify_returns_none_for_unknown_char():
    assert classify("A") is None
    assert classify("\u4E2D") is None
    assert classify("\uE100") is None


def test_codepoints_in_class_has_all_handshape_bases():
    """HamNoSys 4.0 has 12 base handshapes (U+E000..U+E00B)."""
    cps = codepoints_in_class(SymClass.HANDSHAPE_BASE)
    assert set(cps) == set(range(0xE000, 0xE00C))


# ---------------------------------------------------------------------------
# Integration: normalize → validate on raw corpus cells
# ---------------------------------------------------------------------------


def test_normalize_then_validate_recovers_mojibake_cells():
    """A raw BSL-CSV-style mojibake cell should become validatable."""
    # Take a known-good BSL sign, mangle to mojibake, verify round-trip.
    _, _, clean = REAL_CORPUS_SIGNS[0]
    mojibake = clean.encode("utf-8").decode("latin-1")
    fixed = normalize(mojibake)
    assert fixed == clean
    assert validate(fixed).ok
