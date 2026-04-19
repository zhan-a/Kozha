"""Author the 40 gold-pair fixtures for the Prompt-7 generator eval.

Not a pytest file — run once to regenerate the JSON fixtures in this
directory after editing the table below::

    cd backend/chat2hamnosys
    python tests/fixtures/generator/_build_fixtures.py

Each spec below is a tuple ``(id, category, gloss, source, params_dict,
expected_hex, notes)``. The script validates each spec by calling
:func:`generate` and comparing the result against ``expected_hex``; a
mismatch is surfaced so the human author can decide whether to tweak
the spec (a vocab gap) or keep the divergence as an intentional
legitimate-equivalent case for the eval doc.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running this file as a script from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
_REPO = _THIS_DIR.parent.parent.parent  # backend/chat2hamnosys
sys.path.insert(0, str(_REPO))

from generator import generate  # noqa: E402
from parser.models import PartialMovementSegment, PartialSignParameters  # noqa: E402


def _h(hex_str: str) -> str:
    """Decode a hex spec like 'E001 E020 E03A' into a Unicode string."""
    return "".join(chr(int(p, 16)) for p in hex_str.strip().split())


# id, category, gloss, source, parameters (dict), expected_hex, notes
FIXTURES: list[tuple] = [
    # ---------------------------------------------------------------------
    # basic_one_handed — single handshape + orientation + location, no contact,
    # one simple straight movement or none.
    # ---------------------------------------------------------------------
    (
        "aachen3-fist-shoulders-right",
        "basic_one_handed",
        "AACHEN3^",
        "DGS-Korpus",
        {
            "handshape_dominant": "fist",
            "orientation_extended_finger": "up",
            "orientation_palm": "down",
            "location": "shoulders",
            "movement": [{"path": "right"}],
        },
        "E000 E020 E03C E051 E082",
        "Closed fist at shoulder height, moving to the right.",
    ),
    (
        "to-grow-flat-stomach-up",
        "basic_one_handed",
        "TO-GROW2A^",
        "DGS-Korpus",
        {
            "handshape_dominant": "flat",
            "orientation_extended_finger": "out_left",
            "orientation_palm": "down",
            "location": "stomach",
            "movement": [{"path": "up"}],
        },
        "E001 E028 E03C E053 E080",
        "Flat hand rising up from the stomach.",
    ),
    (
        "year3a-index-shoulders-circle",
        "basic_one_handed",
        "YEAR3A^",
        "DGS-Korpus",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "up",
            "location": "shoulders",
            "movement": [{"path": "circle_out"}],
        },
        "E002 E020 E038 E051 E092",
        "Index finger tracing an outward circle at shoulder level.",
    ),
    (
        "hour2a-index-shoulders-circle",
        "basic_one_handed",
        "HOUR2A^",
        "DGS-Korpus",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "down_left",
            "location": "shoulders",
            "movement": [{"path": "circle_out"}],
        },
        "E002 E020 E03D E051 E092",
        "Index finger at shoulder, palm-angled, outward circle.",
    ),
    (
        "no1a-index-shoulders-swing",
        "basic_one_handed",
        "NO1A^",
        "DGS-Korpus",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "down_left",
            "location": "shoulders",
            "movement": [{"path": "swinging"}],
        },
        "E002 E020 E03D E051 E0A6",
        "Index finger swinging at shoulder — lexicalised NO.",
    ),
    (
        "luxury3-three-eyes-touch",
        "basic_one_handed",
        "LUXURY3^",
        "DGS-Korpus",
        {
            "handshape_dominant": "three_spread",
            "orientation_extended_finger": "left",
            "orientation_palm": "left",
            "location": "eyes",
            "contact": "touch",
        },
        "E004 E026 E03E E044 E0D1",
        "Three-spread hand touching the eyes.",
    ),
    (
        "no-clue-pinch-forehead-touch",
        "basic_one_handed",
        "NO-CLUE1^",
        "DGS-Korpus",
        {
            "handshape_dominant": "pinch_12_open",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "down",
            "location": "forehead",
            "contact": "touch",
        },
        "E008 E027 E03C E042 E0D1",
        "Open-pinch handshape touching forehead.",
    ),
    (
        "urine-pinch-stomach-touch",
        "basic_one_handed",
        "URINE1^",
        "DGS-Korpus",
        {
            "handshape_dominant": "pinch_12_open",
            "orientation_extended_finger": "down_left",
            "orientation_palm": "down",
            "location": "stomach",
            "contact": "touch",
        },
        "E008 E025 E03C E053 E0D1",
        "Open-pinch touching stomach.",
    ),
    # ---------------------------------------------------------------------
    # with_contact — includes an explicit contact operator (touch/close/brush).
    # ---------------------------------------------------------------------
    (
        "whistle3-pinch-lips-close",
        "with_contact",
        "WHISTLE3^",
        "DGS-Korpus",
        {
            "handshape_dominant": "pinch_12_open",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "up_left",
            "location": "lips",
            "contact": "close",
        },
        "E008 E027 E03F E04A E0D0",
        "Open pinch approaching lips — whistle.",
    ),
    (
        "fine4-index-chin-down",
        "with_contact",
        "FINE4^",
        "DGS-Korpus",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "left",
            "location": "chin",
            "contact": "touch",
            "movement": [{"path": "down"}],
        },
        "E002 E020 E03E E04D E0D1 E084",
        "Index finger touching chin and moving down.",
    ),
    (
        "style2-index-chin-down",
        "with_contact",
        "STYLE2^",
        "DGS-Korpus",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "left",
            "location": "chin",
            "contact": "touch",
            "movement": [{"path": "down"}],
        },
        "E002 E020 E03E E04D E0D1 E084",
        "Same phonology as FINE4 — different gloss, identical shape.",
    ),
    (
        "sweet1-index-chin-down",
        "with_contact",
        "SWEET1^",
        "DGS-Korpus",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "left",
            "location": "chin",
            "contact": "touch",
            "movement": [{"path": "down"}],
        },
        "E002 E020 E03E E04D E0D1 E084",
        "Index chin-stroke — many DGS phonotactic twins.",
    ),
    (
        "to-bear-fist-chest-down",
        "with_contact",
        "TO-BEAR1B^",
        "DGS-Korpus",
        {
            "handshape_dominant": "fist",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "down",
            "location": "chest",
            "contact": "touch",
            "movement": [{"path": "down"}],
        },
        "E000 E027 E03C E052 E0D1 E084",
        "Fist touching chest, moving downward.",
    ),
    (
        "question-pinch-chin-out",
        "with_contact",
        "QUESTION1^",
        "DGS-Korpus",
        {
            "handshape_dominant": "pinch_12_open",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "left",
            "location": "chin",
            "contact": "touch",
            "movement": [{"path": "out"}],
        },
        "E008 E027 E03E E04D E0D1 E089",
        "Open-pinch at chin, moving outward.",
    ),
    (
        "to-need-pinch-neck-down-out",
        "with_contact",
        "TO-NEED2^",
        "DGS-Korpus",
        {
            "handshape_dominant": "pinch_12_open",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "left",
            "location": "neck",
            "contact": "touch",
            "movement": [{"path": "down_out"}],
        },
        "E008 E027 E03E E04F E0D1 E090",
        "Open pinch at neck, moving down-and-out.",
    ),
    (
        "esophagus-cee-neck-down",
        "with_contact",
        "ESOPHAGUS-OR-TRACHEA1A^",
        "DGS-Korpus",
        {
            "handshape_dominant": "cee_12",
            "orientation_extended_finger": "in_left",
            "orientation_palm": "down_right",
            "location": "neck",
            "contact": "touch",
            "movement": [{"path": "down"}],
        },
        "E009 E02B E03B E04F E0D1 E084",
        "C-handshape at the neck, sliding down.",
    ),
    # ---------------------------------------------------------------------
    # compound_handshape — base + modifier composites in the vocab.
    # ---------------------------------------------------------------------
    (
        "depression3-claw-chest-down",
        "compound_handshape",
        "DEPRESSION3^",
        "DGS-Korpus",
        {
            "handshape_dominant": "claw",
            "orientation_extended_finger": "left",
            "orientation_palm": "left",
            "location": "chest",
            "contact": "touch",
            "movement": [{"path": "down"}],
        },
        "E005 E011 E026 E03E E052 E0D1 E084",
        "Claw (bent-5) at chest, sliding down.",
    ),
    (
        "dirty3c-five-under-chin-fingerplay",
        "compound_handshape",
        "DIRTY3C^",
        "DGS-Korpus",
        {
            "handshape_dominant": "five",
            "orientation_extended_finger": "left",
            "orientation_palm": "down",
            "location": "under_chin",
            "contact": "touch",
            "movement": [{"path": "fingerplay"}],
        },
        "E005 E026 E03C E04E E0D1 E0A4",
        "Open-5 under chin, wiggling fingers — iconic for DIRTY.",
    ),
    (
        "claw-forehead-down",
        "compound_handshape",
        "invented:CLAW-FOREHEAD",
        "authored",
        {
            "handshape_dominant": "bent_5",
            "orientation_extended_finger": "down",
            "orientation_palm": "left",
            "location": "forehead",
            "contact": "touch",
            "movement": [{"path": "down"}],
        },
        "E005 E011 E024 E03E E042 E0D1 E084",
        "Claw at forehead, stroking downward — authored for coverage.",
    ),
    (
        "bent-v-eyes-touch",
        "compound_handshape",
        "invented:BENT-V-EYES",
        "authored",
        {
            "handshape_dominant": "bent_v",
            "orientation_extended_finger": "down",
            "orientation_palm": "down",
            "location": "eyes",
            "contact": "touch",
        },
        "E003 E011 E024 E03C E044 E0D1",
        "Bent-V near eyes — authored for coverage of E003+E011.",
    ),
    # ---------------------------------------------------------------------
    # with_modifier — size, speed, and repeat markers.
    # ---------------------------------------------------------------------
    (
        "dizzy1a-five-head-circle-small-repeat",
        "with_modifier",
        "DIZZY1A^",
        "DGS-Korpus",
        {
            "handshape_dominant": "five",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "left",
            "location": "head",
            "movement": [{"path": "circle_in", "size_mod": "small", "repeat": "twice"}],
        },
        "E005 E027 E03E E040 E093 E0C6 E0D8",
        "Small inward circle at head, repeated — DIZZY.",
    ),
    (
        "crazy1a-fist-forehead-circle-small-repeat",
        "with_modifier",
        "CRAZY1A^",
        "DGS-Korpus",
        {
            "handshape_dominant": "fist",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "down",
            "location": "forehead",
            "contact": "close",
            "movement": [{"path": "circle_in", "size_mod": "small", "repeat": "twice"}],
        },
        "E000 E027 E03C E042 E0D0 E093 E0C6 E0D8",
        "Fist near forehead, small inward circle, repeated.",
    ),
    (
        "circulation-flat-chest-circle-repeat",
        "with_modifier",
        "CIRCULATION1B^",
        "DGS-Korpus",
        {
            "handshape_dominant": "flat",
            "orientation_extended_finger": "in_left",
            "orientation_palm": "right",
            "location": "chest",
            "contact": "close",
            "movement": [{"path": "circle_in", "repeat": "twice"}],
        },
        "E001 E02B E03A E052 E0D0 E093 E0D8",
        "Flat hand circling over the chest — CIRCULATION.",
    ),
    (
        "breathing-five-chest-out-small-repeat",
        "with_modifier",
        "BREATHING1^",
        "DGS-Korpus",
        {
            "handshape_dominant": "five",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "left",
            "location": "chest",
            "contact": "close",
            "movement": [
                {"path": "out", "size_mod": "small", "repeat": "twice"}
            ],
        },
        "E005 E027 E03E E052 E0D0 E089 E0C6 E0D8",
        "Five shape expanding slightly outward from chest, repeating.",
    ),
    (
        "hard4-cee-lips-out-fast",
        "with_modifier",
        "HARD4^",
        "DGS-Korpus",
        {
            "handshape_dominant": "cee_12",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "up_left",
            "location": "lips",
            "contact": "close",
            "movement": [{"path": "out", "speed_mod": "fast"}],
        },
        "E009 E027 E03F E04A E0D0 E089 E0C8",
        "C handshape near lips, fast outward motion.",
    ),
    (
        "slow-flat-neutral-down",
        "with_modifier",
        "invented:SLOW-FLAT",
        "authored",
        {
            "handshape_dominant": "flat",
            "orientation_extended_finger": "up",
            "orientation_palm": "down",
            "location": "neutral_space",
            "movement": [{"path": "down", "speed_mod": "slow"}],
        },
        "E001 E020 E03C E05F E084 E0C9",
        "Authored: flat hand descending slowly in neutral space.",
    ),
    # ---------------------------------------------------------------------
    # action_movement — fingerplay / twisting / nodding / swinging.
    # ---------------------------------------------------------------------
    (
        "brown47-pinchall-nose-twist",
        "action_movement",
        "BROWN47^",
        "DGS-Korpus (approx)",
        {
            "handshape_dominant": "pinch_all",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "down_left",
            "location": "nose",
            "contact": "touch",
            "movement": [{"path": "twisting"}],
        },
        "E007 E027 E03D E045 E0D1 E0A7",
        "Pinch-all at nose, twisting — BROWN.",
    ),
    (
        "curious-pinchall-nose-twist",
        "action_movement",
        "CURIOUS2^",
        "DGS-Korpus (approx)",
        {
            "handshape_dominant": "pinch_all",
            "orientation_extended_finger": "up_left",
            "orientation_palm": "down_left",
            "location": "nose",
            "contact": "touch",
            "movement": [{"path": "twisting"}],
        },
        "E007 E027 E03D E045 E0D1 E0A7",
        "Same phonology as BROWN47 — twin sign.",
    ),
    (
        "five-under-chin-wiggle",
        "action_movement",
        "invented:WIGGLE-BEARD",
        "authored",
        {
            "handshape_dominant": "five",
            "orientation_extended_finger": "down",
            "orientation_palm": "toward_signer",
            "location": "under_chin",
            "movement": [{"path": "fingerplay"}],
        },
        "E005 E024 E03E E04E E0A4",
        "Open-5 below chin, palm toward signer (left for RH), fingers wiggling — beard gesture.",
    ),
    (
        "index-shoulders-swing",
        "action_movement",
        "invented:SWING-INDEX",
        "authored",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "down",
            "location": "shoulders",
            "movement": [{"path": "swinging"}],
        },
        "E002 E020 E03C E051 E0A6",
        "Authored swinging-index for coverage.",
    ),
    # ---------------------------------------------------------------------
    # circular_movement — circle / arc / ellipse variants.
    # ---------------------------------------------------------------------
    (
        "flat-chest-circle-in",
        "circular_movement",
        "invented:CIRCLE-CHEST-IN",
        "authored",
        {
            "handshape_dominant": "flat",
            "orientation_extended_finger": "up",
            "orientation_palm": "toward_signer",
            "location": "chest",
            "movement": [{"path": "circle_in"}],
        },
        "E001 E020 E03E E052 E093",
        "Authored: flat hand circling inward at chest (palm toward body).",
    ),
    (
        "index-forehead-circle-out",
        "circular_movement",
        "invented:CIRCLE-FOREHEAD-OUT",
        "authored",
        {
            "handshape_dominant": "index",
            "orientation_extended_finger": "up",
            "orientation_palm": "down",
            "location": "forehead",
            "movement": [{"path": "circle_out"}],
        },
        "E002 E020 E03C E042 E092",
        "Authored index circle outward at forehead.",
    ),
    (
        "five-head-circle-in",
        "circular_movement",
        "invented:CIRCLE-HEAD-IN",
        "authored",
        {
            "handshape_dominant": "five",
            "orientation_extended_finger": "up",
            "orientation_palm": "down",
            "location": "head",
            "movement": [{"path": "circle_in"}],
        },
        "E005 E020 E03C E040 E093",
        "Authored five-hand circle-in at head.",
    ),
    (
        "flat-chest-arc-up",
        "circular_movement",
        "invented:ARC-UP-CHEST",
        "authored",
        {
            "handshape_dominant": "flat",
            "orientation_extended_finger": "up",
            "orientation_palm": "away_from_signer",
            "location": "chest",
            "movement": [{"path": "arc"}],
        },
        "E001 E020 E03A E052 E0BA",
        "Authored: flat hand tracing an upward arc (palm away).",
    ),
    # ---------------------------------------------------------------------
    # aliases — verify the vocab normalization handles synonyms.
    # ---------------------------------------------------------------------
    (
        "alias-b-forward-right-chest",
        "aliases",
        "invented:ALIAS-B",
        "authored",
        {
            "handshape_dominant": "B",           # alias for flat
            "orientation_extended_finger": "forward",  # alias for out
            "orientation_palm": "right",
            "location": "chest",
        },
        "E001 E029 E03A E052",
        "Tests 'B' → flat (E001), 'forward' → out (E029) aliases.",
    ),
    (
        "alias-one-up-left-chest",
        "aliases",
        "invented:ALIAS-ONE",
        "authored",
        {
            "handshape_dominant": "1",           # alias for index
            "orientation_extended_finger": "up",
            "orientation_palm": "left",
            "location": "chest",
        },
        "E002 E020 E03E E052",
        "Tests '1' → index (E002) alias.",
    ),
    (
        "alias-hyphen-bent-5",
        "aliases",
        "invented:HYPHEN-BENT-5",
        "authored",
        {
            "handshape_dominant": "bent-5",      # hyphen → underscore
            "orientation_extended_finger": "down",
            "orientation_palm": "down",
            "location": "eyes",
        },
        "E005 E011 E024 E03C E044",
        "Tests hyphenated term normalization.",
    ),
    (
        "alias-case-FLAT",
        "aliases",
        "invented:CASE-FLAT",
        "authored",
        {
            "handshape_dominant": "FLAT",        # uppercase
            "orientation_extended_finger": "UP",
            "orientation_palm": "Down",
            "location": "Chest",
        },
        "E001 E020 E03C E052",
        "Tests case-insensitive vocab lookup.",
    ),
    # ---------------------------------------------------------------------
    # two_handed — symmetric two-handed signs (we prefix with hamsymmpar).
    # ---------------------------------------------------------------------
    (
        "symm-two-flats-chest",
        "two_handed",
        "invented:SYMM-FLATS",
        "authored",
        {
            "handshape_dominant": "flat",
            "handshape_nondominant": "flat",
            "orientation_extended_finger": "up",
            "orientation_palm": "toward_signer",
            "location": "chest",
            "contact": "touch",
        },
        "E0E8 E001 E020 E03E E052 E0D1",
        "Symmetric two flats at chest, palms toward body, touching — hamsymmpar prefix.",
    ),
    (
        "symm-two-fists-up-out",
        "two_handed",
        "invented:SYMM-FISTS-PUSH",
        "authored",
        {
            "handshape_dominant": "fist",
            "handshape_nondominant": "fist",
            "orientation_extended_finger": "up",
            "orientation_palm": "away_from_signer",
            "location": "chest",
            "movement": [{"path": "out"}],
        },
        "E0E8 E000 E020 E03A E052 E089",
        "Two fists pushing outward, palms away — symmetric.",
    ),
    # ---------------------------------------------------------------------
    # no_movement — static holds without movement or contact.
    # ---------------------------------------------------------------------
    (
        "static-cee-lips",
        "no_movement",
        "invented:STATIC-C-LIPS",
        "authored",
        {
            "handshape_dominant": "cee_all",
            "orientation_extended_finger": "up",
            "orientation_palm": "left",
            "location": "lips",
        },
        "E00A E020 E03E E04A",
        "Static C near lips.",
    ),
]


def _to_dict(params_spec: dict) -> dict:
    seg_list = params_spec.get("movement", [])
    return {
        "handshape_dominant": params_spec.get("handshape_dominant"),
        "handshape_nondominant": params_spec.get("handshape_nondominant"),
        "orientation_extended_finger": params_spec.get("orientation_extended_finger"),
        "orientation_palm": params_spec.get("orientation_palm"),
        "location": params_spec.get("location"),
        "contact": params_spec.get("contact"),
        "movement": seg_list,
        "non_manual": params_spec.get("non_manual"),
    }


def _build_params(params_spec: dict) -> PartialSignParameters:
    seg_list = [PartialMovementSegment(**s) for s in params_spec.get("movement", [])]
    return PartialSignParameters(
        handshape_dominant=params_spec.get("handshape_dominant"),
        handshape_nondominant=params_spec.get("handshape_nondominant"),
        orientation_extended_finger=params_spec.get("orientation_extended_finger"),
        orientation_palm=params_spec.get("orientation_palm"),
        location=params_spec.get("location"),
        contact=params_spec.get("contact"),
        movement=seg_list,
    )


def _hex_of(s: str) -> str:
    return " ".join(f"{ord(c):04X}" for c in s)


def main() -> int:
    mismatches = 0
    matches = 0
    out_dir = _THIS_DIR
    for fx in FIXTURES:
        fid, category, gloss, source, params_spec, expected_hex, notes = fx
        params = _build_params(params_spec)
        result = generate(params)
        produced_hex = _hex_of(result.hamnosys or "")
        is_match = produced_hex == expected_hex
        if is_match:
            matches += 1
        else:
            mismatches += 1

        record = {
            "id": fid,
            "category": category,
            "gloss": gloss,
            "source": source,
            "notes": notes,
            "parameters": _to_dict(params_spec),
            "expected_hamnosys_hex": expected_hex,
            "produced_hamnosys_hex": produced_hex,
            "deterministic_match": is_match,
            "validation_ok": bool(result.validation.ok),
        }
        (out_dir / f"{fid}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"wrote {matches + mismatches} fixtures "
          f"({matches} deterministic matches, {mismatches} diverging).")
    for fx in FIXTURES:
        fid, _, _, _, params_spec, expected, _ = fx
        params = _build_params(params_spec)
        result = generate(params)
        produced = _hex_of(result.hamnosys or "")
        if produced != expected:
            print(f"  DIVERGE {fid}: want {expected!r}  got {produced!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
