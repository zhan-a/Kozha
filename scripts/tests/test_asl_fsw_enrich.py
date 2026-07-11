"""Unit tests for the ASL-LEX <-> SignWriting/FSW cross-reference enricher
(proposal 58, stage 3 of the ASL-integration batch).

Covers the FSW parser, dominant-hand disambiguation, the fill->plane and
rotation->finger-direction decode, the orientation/movement HamNoSys mapping,
two hard invariants (every decoded orientation code is a valid HamNoSys enum,
and FSW never overrides deterministic ASL-LEX handshape/location), the
``--min-confidence`` override gate, and — most importantly — the LICENSE GATE:
``--write`` must refuse and stay dry-run unless ``LICENSE_CLEARED`` exists.

Tests are hermetic: synthetic FSW strings + tiny temp fixtures, never the
~4MB gitignored ``sgn4.spml``. ``scripts`` is a namespace package (no
``__init__.py``), so the repo root is put on ``sys.path`` and the module
imported as ``scripts.asl_fsw_enrich`` — mirroring the stage-57 test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.asl_fsw_enrich as fsw  # noqa: E402
from scripts.asl_fsw_enrich import (  # noqa: E402
    EXTFIDIR,
    PALMOR,
    _slot,
    decode_movement,
    decode_orientation,
    dominant_hand,
    enrich_row,
    fill_plane,
    gloss_base,
    parse_fsw,
    rot_index8,
)


# A placed FSW symbol = S + 3hex base + 1 fill(0-5) + 1 rot(0-15) + xxx x yyy.
def _sym(base: int, fill: int, rot: int, x: int, y: int) -> str:
    return f"S{base:03x}{fill}{rot:x}{x:03d}x{y:03d}"


def _row(**over: str) -> dict[str, str]:
    """A minimal one-handed neutral ASL-LEX (PoC-schema) row, overridable."""
    base = {
        "EntryID": "asl-test", "Gloss": "TEST", "SignType": "OneHanded",
        "Handshape": "B", "NonDominantHandshape": "", "SelectedFingers": "all",
        "Flexion": "straight", "ThumbPosition": "open", "MajorLocation": "Neutral",
        "MinorLocation": "", "Contact": "no", "PathMovement": "none",
        "RepeatedMovement": "no",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# Gloss normalisation (mirrors panel.js glossBase / stage-56 feasibility)
# --------------------------------------------------------------------------
def test_gloss_base_strips_disambiguators_and_punctuation():
    assert gloss_base("CANDY_1") == "candy"
    assert gloss_base("NEW_YORK") == "new york"
    assert gloss_base("THANK-YOU") == "thank you"
    assert gloss_base("run (verb)") == "run"


# --------------------------------------------------------------------------
# FSW parser
# --------------------------------------------------------------------------
def test_parse_fsw_splits_categories_and_keeps_coords():
    # box + a hand (0x100), a wall arrow (0x221), and a contact (0x205).
    s = "M500x500" + _sym(0x100, 0, 0, 490, 490) + _sym(0x221, 0, 0, 490, 450) \
        + _sym(0x205, 0, 0, 500, 500)
    p = parse_fsw(s)
    assert p is not None
    assert p["n_hands"] == 1 and p["hands"][0].base == 0x100
    assert p["hands"][0].x == 490 and p["hands"][0].y == 490
    assert p["n_arrows"] == 1 and p["arrows"][0].base == 0x221
    assert len(p["contacts"]) == 1 and p["contacts"][0].base == 0x205


def test_parse_fsw_strips_a_sort_prefix():
    # The 'A...' sort prefix lists symbols WITHOUT coordinates; it must be
    # stripped so its symbols are not mistaken for placed glyphs.
    placed = "M500x500" + _sym(0x100, 0, 0, 490, 490)
    s = "AS10000S20600" + placed
    p = parse_fsw(s)
    assert p["n_hands"] == 1  # only the placed hand, not the prefix copy


def test_parse_fsw_empty_or_symbolless_is_none():
    assert parse_fsw("") is None
    assert parse_fsw("M500x500") is None  # box only, no symbols


# --------------------------------------------------------------------------
# Dominant-hand disambiguation
# --------------------------------------------------------------------------
def test_dominant_hand_single_no_penalty():
    p = parse_fsw("M500x500" + _sym(0x100, 0, 0, 490, 490))
    dom, pen = dominant_hand(p)
    assert dom.base == 0x100 and pen == 0.0


def test_dominant_hand_two_picks_largest_x_with_penalty():
    p = parse_fsw("M500x500" + _sym(0x100, 0, 0, 400, 490)
                  + _sym(0x100, 5, 0, 600, 490))
    dom, pen = dominant_hand(p)
    assert dom.x == 600 and pen == 0.15  # signer's-own view: right hand = larger x


def test_dominant_hand_none_when_no_hands():
    p = parse_fsw("M500x500" + _sym(0x221, 0, 0, 490, 450))  # arrow only
    dom, pen = dominant_hand(p)
    assert dom is None and pen == 0.0


# --------------------------------------------------------------------------
# Fill -> plane, rotation -> 8-way index
# --------------------------------------------------------------------------
def test_fill_plane_wall_vs_floor():
    assert [fill_plane(f) for f in (0, 1, 2)] == ["wall", "wall", "wall"]
    assert [fill_plane(f) for f in (3, 4, 5)] == ["floor", "floor", "floor"]


def test_rot_index8_rounds_to_nearest_eighth():
    assert rot_index8(0) == 0
    assert rot_index8(2) == 1   # 2 steps of 22.5deg = 45deg = 1/8 turn
    assert rot_index8(4) == 2
    assert rot_index8(16) == 0  # wraps


# --------------------------------------------------------------------------
# Orientation decode (the heart of the FSW->HamNoSys mapping)
# --------------------------------------------------------------------------
def test_decode_orientation_wall_palm_toward_fingers_up():
    # wall plane (fill 0 = palm toward signer), rot 0 = fingers up.
    hand = parse_fsw("M500x500" + _sym(0x100, 0, 0, 490, 490))["hands"][0]
    deco = decode_orientation(hand)
    assert deco["plane"] == "wall"
    assert deco["extfidir"] == "u"
    assert deco["palmor"] == "d"   # palm toward signer, fingers up -> hampalmd
    assert deco["cardinal"] is True


def test_decode_orientation_floor_palm_down_fingers_out():
    hand = parse_fsw("M500x500" + _sym(0x100, 5, 0, 490, 490))["hands"][0]
    deco = decode_orientation(hand)
    assert deco["plane"] == "floor"
    assert deco["extfidir"] == "o"   # floor rot 0 reads as out/away
    assert deco["cardinal"] is True


def test_decode_orientation_half_fill_side_is_not_cardinal():
    # fills 1 and 4 are palm-to-the-side: ambiguous about WHICH side -> low conf.
    for fill in (1, 4):
        hand = parse_fsw("M500x500" + _sym(0x100, fill, 0, 490, 490))["hands"][0]
        assert decode_orientation(hand)["cardinal"] is False


def test_every_decoded_orientation_is_a_valid_enum():
    # Hard invariant: across every fill x rotation, extfidir/palmor are valid
    # HamNoSys enum members (the belt-and-braces guard must hold).
    for fill in range(6):
        for rot in range(16):
            hand = parse_fsw("M500x500" + _sym(0x100, fill, rot, 490, 490))["hands"][0]
            deco = decode_orientation(hand)
            assert deco["extfidir"] in EXTFIDIR
            assert deco["palmor"] in PALMOR


# --------------------------------------------------------------------------
# Movement-direction decode
# --------------------------------------------------------------------------
def test_decode_movement_wall_arrow_up():
    p = parse_fsw("M500x500" + _sym(0x221, 0, 0, 490, 450))
    mvd = decode_movement(p)
    assert mvd is not None
    assert mvd["direction"] == "u" and mvd["tag"] == "hammoveu"


def test_decode_movement_floor_arrow_out():
    p = parse_fsw("M500x500" + _sym(0x22f, 0, 0, 490, 450))
    mvd = decode_movement(p)
    assert mvd["direction"] == "o" and mvd["tag"] == "hammoveo"


def test_decode_movement_no_arrow_is_none():
    p = parse_fsw("M500x500" + _sym(0x100, 0, 0, 490, 490))
    assert decode_movement(p) is None


def test_decode_movement_undecodable_diagonal_is_none():
    # floor diagonal "or" has no HamNoSys straight tag -> dropped.
    p = parse_fsw("M500x500" + _sym(0x22f, 0, 2, 490, 450))  # rot 2 -> idx 1 -> "or"
    assert decode_movement(p) is None


# --------------------------------------------------------------------------
# enrich_row: override gap fields only, mark provenance "fsw", confidence gate
# --------------------------------------------------------------------------
def test_enrich_overrides_orientation_and_marks_fsw():
    fsw_dict = {"test": ["M500x500" + _sym(0x100, 0, 0, 490, 490)]}
    rec = enrich_row(_row(Gloss="TEST"), fsw_dict)
    assert rec["matched"] is True
    assert rec["overrode_orientation"] is True
    ori = _slot(rec["fields"], "orientation")
    assert ori["source"] == "fsw"
    assert ori["tags"] == ["hamextfingeru", "hampalmd"]
    assert rec["orientation_confidence"] == "high"


def test_enrich_never_touches_deterministic_handshape_or_location():
    fsw_dict = {"test": ["M500x500" + _sym(0x100, 0, 0, 490, 490)]}
    rec = enrich_row(_row(Gloss="TEST", MajorLocation="Head", MinorLocation="chin",
                          Contact="yes"), fsw_dict)
    hs = _slot(rec["fields"], "handshape")
    loc = _slot(rec["fields"], "location")
    assert hs["source"] == "asl_lex"      # handshape stays deterministic
    assert loc["source"] == "asl_lex"     # location stays deterministic
    assert hs["source"] != "fsw" and loc["source"] != "fsw"


def test_enrich_no_match_no_override():
    rec = enrich_row(_row(Gloss="TEST"), {})  # empty FSW dict
    assert rec["matched"] is False
    assert rec["overrode_orientation"] is False
    ori = _slot(rec["fields"], "orientation")
    assert ori["source"] in {"heuristic", "default"}  # untouched stage-57 value
    assert rec["tags"]  # still reassembles a full tag list


def test_min_confidence_high_skips_two_handed_medium_pick():
    # Two hands -> 0.15 penalty -> "medium" confidence. The high floor skips it.
    two = "M500x500" + _sym(0x100, 0, 0, 400, 490) + _sym(0x100, 0, 0, 600, 490)
    fsw_dict = {"test": [two]}
    hi = enrich_row(_row(Gloss="TEST"), fsw_dict, min_confidence="high")
    assert hi["orientation_confidence"] == "medium"
    assert hi["overrode_orientation"] is False  # below the high floor
    med = enrich_row(_row(Gloss="TEST"), fsw_dict, min_confidence="medium")
    assert med["overrode_orientation"] is True


def test_min_confidence_skips_low_side_fill_always():
    # half-fill side -> "low" -> never overrides, even at the medium floor.
    fsw_dict = {"test": ["M500x500" + _sym(0x100, 1, 0, 490, 490)]}
    rec = enrich_row(_row(Gloss="TEST"), fsw_dict, min_confidence="medium")
    assert rec["orientation_confidence"] == "low"
    assert rec["overrode_orientation"] is False


def test_enrich_overrides_movement_direction_on_straight_path():
    # ASL-LEX straight path (no direction) + an FSW arrow -> direction filled.
    fsw_dict = {"test": ["M500x500" + _sym(0x100, 0, 0, 490, 490)
                         + _sym(0x221, 0, 0, 490, 450)]}
    rec = enrich_row(_row(Gloss="TEST", PathMovement="straight"), fsw_dict)
    mv = _slot(rec["fields"], "movement")
    assert rec["overrode_movement"] is True
    assert mv["source"] == "fsw"
    assert "hammoveu" in mv["tags"]


# --------------------------------------------------------------------------
# LICENSE GATE — the mandatory acceptance criterion
# --------------------------------------------------------------------------
def _tiny_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "EntryID,Gloss,SignType,Handshape,NonDominantHandshape,SelectedFingers,"
        "Flexion,ThumbPosition,MajorLocation,MinorLocation,Contact,PathMovement,"
        "RepeatedMovement\n"
        "asl-test,TEST,OneHanded,B,,all,straight,open,Neutral,,no,none,no\n",
        encoding="utf-8")
    spml_path = tmp_path / "tiny.spml"
    spml_path.write_text(
        "<entry><term>M500x500" + _sym(0x100, 0, 0, 490, 490)
        + "</term><term>TEST</term></entry>\n",
        encoding="utf-8")
    out_path = tmp_path / "out.sigml"
    return csv_path, spml_path, out_path


def test_committed_repo_has_no_license_cleared():
    # Default committed state must keep the script dry-run.
    assert not fsw.LICENSE_CLEARED.exists()


def test_write_refused_without_license_cleared(tmp_path, monkeypatch, capsys):
    csv_path, spml_path, out_path = _tiny_corpus(tmp_path)
    # Force the gate closed regardless of local developer state. The path must
    # stay under the repo root (the REFUSED message renders it relative).
    monkeypatch.setattr(
        fsw, "LICENSE_CLEARED",
        REPO_ROOT / "data" / "sources" / "signpuddle" / "NOPE_TEST_ABSENT")
    monkeypatch.setattr(sys, "argv", [
        "asl_fsw_enrich.py", "--in", str(csv_path), "--spml", str(spml_path),
        "--out", str(out_path), "--write",
    ])
    fsw.main()
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert not out_path.exists()                       # nothing written
    assert not out_path.with_name(out_path.name + ".meta.json").exists()


def test_write_succeeds_when_license_cleared(tmp_path, monkeypatch, capsys):
    csv_path, spml_path, out_path = _tiny_corpus(tmp_path)
    cleared = tmp_path / "LICENSE_CLEARED"
    cleared.write_text("test clearance\n", encoding="utf-8")
    monkeypatch.setattr(fsw, "LICENSE_CLEARED", cleared)
    monkeypatch.setattr(sys, "argv", [
        "asl_fsw_enrich.py", "--in", str(csv_path), "--spml", str(spml_path),
        "--out", str(out_path), "--write",
    ])
    fsw.main()
    out = capsys.readouterr().out
    assert "validate" in out.lower()
    assert out_path.exists()                           # SiGML written
    meta_path = out_path.with_name(out_path.name + ".meta.json")
    assert meta_path.exists()
    import json
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["default_review"]["deaf_native_reviewed"] is False
    assert meta["data_completeness"] == "seed"
    assert meta["provenance_summary"]["tags_by_source"].get("fsw", 0) >= 1


def test_dry_run_default_writes_nothing(tmp_path, monkeypatch, capsys):
    csv_path, spml_path, out_path = _tiny_corpus(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "asl_fsw_enrich.py", "--in", str(csv_path), "--spml", str(spml_path),
        "--out", str(out_path),  # no --write
    ])
    fsw.main()
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert not out_path.exists()
