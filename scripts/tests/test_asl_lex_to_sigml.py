"""Unit tests for the ASL-LEX -> H-SiGML proof-of-concept converter.

Covers the deterministic mappings (handshape, location, symmetry, movement
shape) and the rule-based heuristics added for orientation and movement
direction, plus two hard invariants:

  * every emitted orientation code is a valid HamNoSys enum value
    (extfidir in the 18-set, palmor in the 8-set), and
  * every tag the converter emits over the PoC seed is in CWASA's
    ``tokenNameMap`` (i.e. ``--validate`` would pass).

``scripts`` is a namespace package (no ``__init__.py``), so the repo root is
put on ``sys.path`` and the module imported as ``scripts.asl_lex_to_sigml`` —
mirroring ``server/tests/test_database_health.py``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.asl_lex_to_sigml import (  # noqa: E402
    EXTFIDIR,
    PALMOR,
    convert_row,
    emit_sigml,
    handshape_class,
    infer_movement,
    load_known_tags,
    lookup_handshape,
    orientation_for,
    to_palmor,
)

SEED_CSV = REPO_ROOT / "data" / "asl_lex_poc_seed.csv"


def _row(**over: str) -> dict[str, str]:
    """A minimal one-handed neutral row, overridable per-field."""
    base = {
        "EntryID": "asl-test", "Gloss": "TEST", "SignType": "OneHanded",
        "Handshape": "B", "NonDominantHandshape": "", "SelectedFingers": "all",
        "Flexion": "straight", "ThumbPosition": "open", "MajorLocation": "Neutral",
        "MinorLocation": "", "Contact": "no", "PathMovement": "none",
        "RepeatedMovement": "no",
    }
    base.update(over)
    return base


def _orientation_field(row: dict[str, str]) -> dict:
    _, _, fields, _ = convert_row(row)
    return next(f for f in fields if f["slot"] == "orientation")


# --------------------------------------------------------------------------
# Handshape mapping
# --------------------------------------------------------------------------
def test_handshape_known_base_and_modifiers():
    (base, mods, approx), found = lookup_handshape("5")
    assert found is True
    assert base == "hamflathand"
    assert "hamthumboutmod" in mods
    assert approx is False


def test_handshape_lookup_is_case_insensitive():
    # real ASL-LEX labels are lowercase; the table key is "open_B".
    (base, mods, approx), found = lookup_handshape("open_b")
    assert found is True
    assert base == "hamflathand"
    assert "hamthumboutmod" in mods


def test_handshape_approximated_flag():
    (base, _mods, approx), found = lookup_handshape("Y")
    assert found is True
    assert approx is True


def test_handshape_unmapped_falls_back_and_flags_review():
    (base, mods, approx), found = lookup_handshape("ZZZ-nope")
    assert found is False
    assert base == "hamflathand"
    assert approx is True
    review = convert_row(_row(Handshape="ZZZ-nope"))[3]
    assert any("unmapped" in r for r in review)


def test_handshape_class_index_vs_flat():
    assert handshape_class("1") == "index"
    assert handshape_class("G") == "index"
    assert handshape_class("B") == "flat"
    assert handshape_class("totally-unknown") == "other"


# --------------------------------------------------------------------------
# Location mapping
# --------------------------------------------------------------------------
def test_location_minor_lookup_and_contact():
    _, tags, fields, _ = convert_row(
        _row(MajorLocation="Head", MinorLocation="chin", Contact="yes"))
    loc = next(f for f in fields if f["slot"] == "location")
    assert loc["tags"][0] == "hamchin"
    assert "hamlrat" in loc["tags"]  # Contact=yes -> touch tag


def test_location_major_fallback_when_minor_absent():
    _, _tags, fields, _ = convert_row(
        _row(MajorLocation="Head", MinorLocation="", Contact="no"))
    loc = next(f for f in fields if f["slot"] == "location")
    assert loc["tags"] == ["hamhead"]  # major-bucket default, no touch


def test_location_unmapped_is_flagged():
    review = convert_row(
        _row(MajorLocation="Nowhere", MinorLocation="void"))[3]
    assert any("unmapped" in r for r in review)


# --------------------------------------------------------------------------
# Symmetry from SignType
# --------------------------------------------------------------------------
def test_symmetry_one_handed_has_no_operator():
    _, tags, _fields, _ = convert_row(_row(SignType="OneHanded"))
    assert not any(t.startswith("hamsymm") for t in tags)


def test_symmetry_symmetrical_emits_symmlr():
    _, tags, _fields, _ = convert_row(_row(SignType="TwoHanded_Symmetrical"))
    assert tags[0] == "hamsymmlr"


def test_symmetry_real_asllex_label_symmetricaloralternating():
    _, tags, _fields, _ = convert_row(_row(SignType="SymmetricalOrAlternating"))
    assert tags[0] == "hamsymmlr"


def test_symmetry_asymmetrical_flags_review():
    review = convert_row(
        _row(SignType="TwoHanded_Asymmetrical_Different",
             NonDominantHandshape="B"))[3]
    assert any("asymmetrical" in r.lower() for r in review)


# --------------------------------------------------------------------------
# Orientation heuristics — representative (location -> extfidir/palmor) cases
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "major,minor,expected_extfi,expected_palmor",
    [
        ("Head", "chin", "u", "d"),      # at the chin: fingers up, palm to signer
        ("Head", "forehead", "u", "l"),  # upper face: fingers up, palm to the side
        ("Hand", "palm", "o", "d"),      # on base hand: fingers out, palm down
    ],
)
def test_orientation_heuristic_cases(major, minor, expected_extfi, expected_palmor):
    extfi, palmor, source, _why = orientation_for(major, minor, "B", "onehanded", None)
    assert extfi == expected_extfi
    assert palmor == expected_palmor
    assert source == "heuristic"


def test_orientation_two_handed_symmetric_neutral_palms_face_each_other():
    extfi, palmor, source, _why = orientation_for(
        "Neutral", "", "5", "twohanded_symmetrical", None)
    assert extfi == "o"          # fingers out
    assert palmor == "r"         # dominant palm to the midline (relative frame)
    assert source == "heuristic"


def test_orientation_index_handshape_follows_movement():
    # pointing handshape: finger direction tracks the inferred movement.
    extfi, _palmor, source, why = orientation_for(
        "Neutral", "", "1", "onehanded", "o")
    assert extfi == "o"
    assert source == "heuristic"
    assert "index" in why


def test_orientation_unknown_location_defaults():
    extfi, palmor, source, _why = orientation_for(
        "Nowhere", "void", "B", "onehanded", None)
    assert source == "default"
    assert extfi in EXTFIDIR
    assert palmor in PALMOR


def test_to_palmor_cardinal_invariants():
    # fingers up: palm away -> u, toward signer -> d, left -> l, right -> r.
    assert to_palmor("u", "o") == "u"
    assert to_palmor("u", "i") == "d"
    assert to_palmor("u", "l") == "l"
    assert to_palmor("u", "r") == "r"


def test_to_palmor_degenerate_returns_none():
    # palm parallel to the fingers is not a valid (perpendicular) posture.
    assert to_palmor("u", "u") is None


# --------------------------------------------------------------------------
# Movement shape + inferred direction
# --------------------------------------------------------------------------
def test_movement_none_is_nomotion_from_asllex():
    tags, source, _why, direction = infer_movement(
        "none", "neutral", "", "onehanded", False)
    assert tags == ["hamnomotion"]
    assert source == "asl_lex"
    assert direction is None


def test_movement_straight_symmetric_moves_apart():
    tags, source, _why, direction = infer_movement(
        "straight", "neutral", "", "twohanded_symmetrical", False)
    assert tags == ["hammover"]
    assert source == "heuristic"
    assert direction == "r"


def test_movement_straight_from_face_moves_outward():
    tags, source, _why, direction = infer_movement(
        "straight", "head", "forehead", "onehanded", False)
    assert tags == ["hammoveo"]
    assert source == "heuristic"
    assert direction == "o"


def test_movement_neutral_one_handed_direction_defaulted():
    tags, source, _why, _direction = infer_movement(
        "straight", "neutral", "", "onehanded", False)
    assert source == "default"  # no principled default -> flagged
    assert tags == ["hammoved"]


def test_movement_repeat_appends_tag():
    tags, _source, _why, _direction = infer_movement(
        "none", "neutral", "", "onehanded", True)
    assert "hamrepeatfromstart" in tags


# --------------------------------------------------------------------------
# Enum validity over the whole seed + canonical slot order
# --------------------------------------------------------------------------
def _seed_rows() -> list[dict[str, str]]:
    with SEED_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_every_emitted_orientation_code_is_a_valid_enum():
    for row in _seed_rows():
        ori = _orientation_field(row)
        extfi_tag, palmor_tag = ori["tags"]
        assert extfi_tag.startswith("hamextfinger")
        assert palmor_tag.startswith("hampalm")
        assert extfi_tag[len("hamextfinger"):] in EXTFIDIR
        assert palmor_tag[len("hampalm"):] in PALMOR


def test_orientation_is_heuristic_not_constant_default_over_seed():
    # Task 5 acceptance: PoC signs carry heuristic (not default) orientation.
    sources = [_orientation_field(r)["source"] for r in _seed_rows()]
    assert sources, "seed produced no signs"
    assert all(s == "heuristic" for s in sources)


# --------------------------------------------------------------------------
# --validate would pass: every emitted tag is in CWASA's tokenNameMap
# --------------------------------------------------------------------------
def test_all_emitted_tags_are_cwasa_renderable():
    rows = _seed_rows()
    _doc, _meta, audit = emit_sigml(rows, "American_SL_ASL")
    known = load_known_tags() | {"hns_sign", "hamnosys_manual", "sigml_collection"}
    bad = sorted({t for s in audit for t in s["tags"] if t not in known})
    assert bad == [], f"tags not in CWASA tokenNameMap: {bad}"


def test_emit_keeps_seed_completeness_and_unreviewed():
    _doc, meta, _audit = emit_sigml(_seed_rows(), "American_SL_ASL")
    assert meta["data_completeness"] == "seed"
    assert meta["default_review"]["deaf_native_reviewed"] is False
