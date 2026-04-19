"""One-shot generator for correct/ fixtures.

Produces 20 JSON fixtures covering the categories called out in
Prompt 10:

- handshape swap (4)
- location change (3)
- movement direction flip (2)
- movement repetition change (2)
- non-manual addition (2)
- two-handed ↔ one-handed conversion (2)
- ambiguous / vague (2)
- contradictory (2)
- restart (1)

Each fixture file is JSON with this shape::

    {
      "id": "...",
      "notes": "...",
      "current_params": {...},        # PartialSignParameters JSON
      "current_hamnosys": "...",
      "original_prose": "...",
      "correction": {
        "raw_text": "...",
        "target_time_ms": null|int,
        "target_region": null|str
      },
      "recorded_response": {...},     # JSON the LLM is hand-scripted to return
      "expect_llm_called": true|false,
      "expected_intent": "apply_diff"|"restart"|...,
      "expected_paths": ["..."],
      "expected_needs_confirmation": true|false,
      "expected_follow_up_present": true|false,
      "expected_updated_slots": {...} # subset of slots to assert on the new params
    }

Run from this directory::

    python _build_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def _scalar_params(**overrides: Any) -> dict[str, Any]:
    """Return a baseline PartialSignParameters dict."""
    base: dict[str, Any] = {
        "handshape_dominant": "fist",
        "handshape_nondominant": None,
        "orientation_extended_finger": "up",
        "orientation_palm": "down",
        "location": "temple",
        "contact": None,
        "movement": [],
        "non_manual": None,
    }
    base.update(overrides)
    return base


def _movement(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "path": "straight",
        "size_mod": None,
        "speed_mod": None,
        "repeat": None,
    }
    base.update(overrides)
    return base


def _apply_diff_response(
    *,
    explanation: str,
    field_changes: list[dict[str, Any]],
    needs_confirmation: bool = False,
    follow_up: str | None = None,
) -> dict[str, Any]:
    return {
        "intent": "apply_diff",
        "explanation": explanation,
        "needs_user_confirmation": needs_confirmation,
        "follow_up_question": follow_up,
        "field_changes": field_changes,
    }


def _nondiff_response(
    intent: str,
    *,
    explanation: str,
    follow_up: str,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "explanation": explanation,
        "needs_user_confirmation": True,
        "follow_up_question": follow_up,
        "field_changes": [],
    }


def _change(
    path: str,
    old: Any,
    new: Any,
    *,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "path": path,
        "old_value": old,
        "new_value": new,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


FIXTURES: list[dict[str, Any]] = []


# 1. Handshape swap — fist → flat-O
FIXTURES.append({
    "id": "01-handshape-swap-fist-to-flato",
    "notes": "Standard handshape swap with timestamp targeting; minimal diff.",
    "current_params": _scalar_params(),
    "current_hamnosys": "\uE000\uE020\uE03C\uE049",
    "original_prose": "A closed fist held at the temple, fingers up, palm down.",
    "correction": {
        "raw_text": "the handshape at 2 seconds is wrong; it should be a flat-O, not a fisted hand",
        "target_time_ms": 2000,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Changing the dominant handshape from fist to flat-O; everything else stays.",
        field_changes=[_change("handshape_dominant", "fist", "flat-O", confidence=0.95)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["handshape_dominant"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"handshape_dominant": "flat-O"},
})


# 2. Handshape swap — flat → bent-5
FIXTURES.append({
    "id": "02-handshape-swap-flat-to-bent5",
    "notes": "Handshape swap inferred from a freer phrasing.",
    "current_params": _scalar_params(handshape_dominant="flat", location="chest"),
    "current_hamnosys": "\uE001\uE020\uE03C\uE048",
    "original_prose": "A flat hand at the chest moving outward.",
    "correction": {
        "raw_text": "actually, fingers should be bent at the knuckles — make it bent-5",
        "target_time_ms": None,
        "target_region": "hand",
    },
    "recorded_response": _apply_diff_response(
        explanation="Switching the handshape from flat to bent-5; chest location stays.",
        field_changes=[_change("handshape_dominant", "flat", "bent-5", confidence=0.92)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["handshape_dominant"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"handshape_dominant": "bent-5"},
})


# 3. Handshape swap with low confidence — claw vs bent-V ambiguity
FIXTURES.append({
    "id": "03-handshape-swap-low-confidence",
    "notes": "Interpreter applies the diff but flags it for confirmation.",
    "current_params": _scalar_params(handshape_dominant="5", location="forehead"),
    "current_hamnosys": "\uE005\uE020\uE03C\uE040",
    "original_prose": "A spread five at the forehead.",
    "correction": {
        "raw_text": "it should be more claw-like — fingers bent",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Changing the handshape from spread-5 to bent-5 (claw). Confirm before regenerating.",
        field_changes=[_change("handshape_dominant", "5", "bent-5", confidence=0.55)],
        needs_confirmation=True,
        follow_up=None,
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["handshape_dominant"],
    "expected_needs_confirmation": True,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"handshape_dominant": "bent-5"},
})


# 4. Handshape swap — index → V (with click region)
FIXTURES.append({
    "id": "04-handshape-swap-index-to-v",
    "notes": "Region-clicked correction; LLM uses region to pin down which slot.",
    "current_params": _scalar_params(handshape_dominant="index", location="chin", movement=[_movement(path="down")]),
    "current_hamnosys": "\uE002\uE020\uE03C\uE044\uE08C",
    "original_prose": "Index finger at the chin moving down.",
    "correction": {
        "raw_text": "use two fingers, not one",
        "target_time_ms": None,
        "target_region": "hand",
    },
    "recorded_response": _apply_diff_response(
        explanation="Replacing the index handshape with a V (index + middle extended).",
        field_changes=[_change("handshape_dominant", "index", "V", confidence=0.93)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["handshape_dominant"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"handshape_dominant": "V"},
})


# 5. Location change — chin → temple, with timestamp
FIXTURES.append({
    "id": "05-location-change-chin-to-temple",
    "notes": "Pure location swap with timestamp; minimal diff.",
    "current_params": _scalar_params(handshape_dominant="index", location="chin"),
    "current_hamnosys": "\uE002\uE020\uE03C\uE044",
    "original_prose": "An index finger at the chin.",
    "correction": {
        "raw_text": "location is too low; move it up to the temple",
        "target_time_ms": 1500,
        "target_region": "head",
    },
    "recorded_response": _apply_diff_response(
        explanation="Moving the location from chin to temple. Handshape and orientation are unchanged.",
        field_changes=[_change("location", "chin", "temple", confidence=0.97)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["location"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"location": "temple"},
})


# 6. Location change — chest → neutral space
FIXTURES.append({
    "id": "06-location-change-chest-to-neutral",
    "notes": "Body-anchored to neutral-space location swap.",
    "current_params": _scalar_params(handshape_dominant="flat", location="chest"),
    "current_hamnosys": "\uE001\uE020\uE03C\uE048",
    "original_prose": "Flat hand at the chest.",
    "correction": {
        "raw_text": "the hand should be in front of the body, not touching the chest",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Moving the location from chest to neutral space.",
        field_changes=[_change("location", "chest", "neutral space", confidence=0.93)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["location"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"location": "neutral space"},
})


# 7. Location change — clarifies via region click
FIXTURES.append({
    "id": "07-location-change-region-click",
    "notes": "Click-on-mouth disambiguates a vague 'face' correction.",
    "current_params": _scalar_params(handshape_dominant="O", location="cheek"),
    "current_hamnosys": "\uE007\uE020\uE03C\uE042",
    "original_prose": "An O hand at the cheek.",
    "correction": {
        "raw_text": "no, more towards the mouth",
        "target_time_ms": None,
        "target_region": "mouth",
    },
    "recorded_response": _apply_diff_response(
        explanation="Moving the location from cheek to mouth/lips based on the clicked region.",
        field_changes=[_change("location", "cheek", "mouth / lips", confidence=0.88)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["location"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"location": "mouth / lips"},
})


# 8. Movement direction flip — straight up → straight down
FIXTURES.append({
    "id": "08-movement-flip-up-to-down",
    "notes": "Flip the movement path direction.",
    "current_params": _scalar_params(
        handshape_dominant="flat",
        location="chest",
        movement=[_movement(path="up")],
    ),
    "current_hamnosys": "\uE001\uE020\uE03C\uE048\uE088",
    "original_prose": "Flat hand at the chest moving upward.",
    "correction": {
        "raw_text": "the movement is going the wrong way — should be downward, not upward",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Flipping the movement direction from up to down.",
        field_changes=[_change("movement[0].path", "up", "down", confidence=0.95)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["movement[0].path"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {},
})


# 9. Movement direction flip — circular outward → circular inward
FIXTURES.append({
    "id": "09-movement-flip-out-to-in",
    "notes": "Flip a circular movement direction.",
    "current_params": _scalar_params(
        handshape_dominant="index",
        location="forehead",
        movement=[_movement(path="circular", size_mod="small")],
    ),
    "current_hamnosys": "\uE002\uE020\uE03C\uE040\uE092\uE0C6",
    "original_prose": "Index finger at the forehead, small circle clockwise outward.",
    "correction": {
        "raw_text": "the circle should go the other direction, inward",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Flipping the circle direction from outward to inward.",
        field_changes=[_change("movement[0].path", "circular", "circular inward", confidence=0.85)],
        needs_confirmation=True,
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["movement[0].path"],
    "expected_needs_confirmation": True,
    "expected_follow_up_present": False,
    "expected_updated_slots": {},
})


# 10. Movement repetition change — once → twice
FIXTURES.append({
    "id": "10-movement-repeat-once-to-twice",
    "notes": "Bump the repetition count.",
    "current_params": _scalar_params(
        handshape_dominant="flat",
        location="chest",
        movement=[_movement(path="straight", repeat="once")],
    ),
    "current_hamnosys": "\uE001\uE020\uE03C\uE048\uE085",
    "original_prose": "Flat hand at the chest moving once.",
    "correction": {
        "raw_text": "the movement should repeat twice, not once",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Changing the repetition from once to twice; everything else stays.",
        field_changes=[_change("movement[0].repeat", "once", "twice", confidence=0.97)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["movement[0].repeat"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {},
})


# 11. Movement repetition change — adds repeat to a previously-once movement
FIXTURES.append({
    "id": "11-movement-repeat-add-continuous",
    "notes": "User asks for continuous repetition where none was specified.",
    "current_params": _scalar_params(
        handshape_dominant="5",
        location="chest",
        movement=[_movement(path="circular")],
    ),
    "current_hamnosys": "\uE005\uE020\uE03C\uE048\uE092",
    "original_prose": "A spread-5 hand at the chest moving in a circle.",
    "correction": {
        "raw_text": "make it keep going — continuous circling, not just one loop",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Marking the movement as continuous so it loops instead of running once.",
        field_changes=[_change("movement[0].repeat", None, "continuous", confidence=0.92)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["movement[0].repeat"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {},
})


# 12. Non-manual addition — add raised eyebrows
FIXTURES.append({
    "id": "12-nonmanual-add-raised-eyebrows",
    "notes": "Add a non-manual feature (eyebrows) to a sign without one.",
    "current_params": _scalar_params(handshape_dominant="index", location="chin"),
    "current_hamnosys": "\uE002\uE020\uE03C\uE044",
    "original_prose": "An index finger at the chin.",
    "correction": {
        "raw_text": "raise the eyebrows on this — it's a yes/no question",
        "target_time_ms": None,
        "target_region": "face",
    },
    "recorded_response": _apply_diff_response(
        explanation="Adding raised eyebrows for the yes/no marker.",
        field_changes=[_change("non_manual.eyebrows", None, "raised", confidence=0.95)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["non_manual.eyebrows"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {},
})


# 13. Non-manual addition — mouth picture for breathing
FIXTURES.append({
    "id": "13-nonmanual-add-mouth-picture",
    "notes": "Adds a SAMPA-ish mouth picture to a sign.",
    "current_params": _scalar_params(
        handshape_dominant="5",
        location="chest",
        movement=[_movement(path="straight")],
    ),
    "current_hamnosys": "\uE005\uE020\uE03C\uE048\uE084",
    "original_prose": "Spread-5 hand at the chest moving outward.",
    "correction": {
        "raw_text": "add a quiet 'pah' mouth shape during the movement",
        "target_time_ms": None,
        "target_region": "mouth",
    },
    "recorded_response": _apply_diff_response(
        explanation="Adding the mouth picture 'pah' during the movement.",
        field_changes=[_change("non_manual.mouth_picture", None, "pah", confidence=0.9)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["non_manual.mouth_picture"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {},
})


# 14. Two-handed → one-handed conversion (drop nondom handshape)
FIXTURES.append({
    "id": "14-two-handed-to-one-handed",
    "notes": "Drop the non-dominant handshape to convert to one-handed.",
    "current_params": _scalar_params(
        handshape_dominant="fist",
        handshape_nondominant="flat",
        location="chest",
    ),
    "current_hamnosys": "\uE0E8\uE000\uE020\uE03C\uE048",
    "original_prose": "Both hands at the chest — dominant fist on top of the non-dominant flat hand.",
    "correction": {
        "raw_text": "this should be one-handed, drop the helper hand",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Removing the non-dominant handshape so the sign becomes one-handed.",
        field_changes=[_change("handshape_nondominant", "flat", None, confidence=0.95)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["handshape_nondominant"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"handshape_nondominant": None},
})


# 15. One-handed → two-handed conversion (add nondom handshape)
FIXTURES.append({
    "id": "15-one-handed-to-two-handed",
    "notes": "Add a non-dominant handshape to convert to two-handed.",
    "current_params": _scalar_params(
        handshape_dominant="fist",
        location="chest",
        movement=[_movement(path="down")],
    ),
    "current_hamnosys": "\uE000\uE020\uE03C\uE048\uE08C",
    "original_prose": "Closed fist at the chest moving down.",
    "correction": {
        "raw_text": "this is symmetric — both hands do the same thing, mirrored",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _apply_diff_response(
        explanation="Mirroring the dominant handshape on the non-dominant hand for a symmetric two-handed sign.",
        field_changes=[_change("handshape_nondominant", None, "fist", confidence=0.9)],
    ),
    "expect_llm_called": True,
    "expected_intent": "apply_diff",
    "expected_paths": ["handshape_nondominant"],
    "expected_needs_confirmation": False,
    "expected_follow_up_present": False,
    "expected_updated_slots": {"handshape_nondominant": "fist"},
})


# 16. Vague — "looks off"
FIXTURES.append({
    "id": "16-vague-looks-off",
    "notes": "Truly vague correction; interpreter must ask for specifics.",
    "current_params": _scalar_params(handshape_dominant="5", location="chest"),
    "current_hamnosys": "\uE005\uE020\uE03C\uE048",
    "original_prose": "Spread-5 at the chest.",
    "correction": {
        "raw_text": "it looks off",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _nondiff_response(
        "vague",
        explanation="The correction is too general to act on without more detail.",
        follow_up="What specifically looks off — the handshape, the location, the movement, or a non-manual feature?",
    ),
    "expect_llm_called": True,
    "expected_intent": "vague",
    "expected_paths": [],
    "expected_needs_confirmation": True,
    "expected_follow_up_present": True,
    "expected_updated_slots": {},
})


# 17. Elaborate — non-manual not encoded
FIXTURES.append({
    "id": "17-elaborate-happier-no-nm",
    "notes": "User asks for a 'happier' facial expression on a sign with no non-manuals.",
    "current_params": _scalar_params(handshape_dominant="flat", location="chest"),
    "current_hamnosys": "\uE001\uE020\uE03C\uE048",
    "original_prose": "Flat hand at the chest, palm down.",
    "correction": {
        "raw_text": "the facial expression should be happier",
        "target_time_ms": None,
        "target_region": "face",
    },
    "recorded_response": _nondiff_response(
        "elaborate",
        explanation="No facial features are currently encoded; need a concrete description before adding one.",
        follow_up="Which specific non-manual would express 'happier' — raised eyebrows, an open mouth, or something else?",
    ),
    "expect_llm_called": True,
    "expected_intent": "elaborate",
    "expected_paths": [],
    "expected_needs_confirmation": True,
    "expected_follow_up_present": True,
    "expected_updated_slots": {},
})


# 18. Contradiction — temple vs chin
FIXTURES.append({
    "id": "18-contradiction-temple-vs-chin",
    "notes": "Correction contradicts the original prose; surface both options.",
    "current_params": _scalar_params(handshape_dominant="index", location="temple"),
    "current_hamnosys": "\uE002\uE020\uE03C\uE040",
    "original_prose": "An index finger at the right temple.",
    "correction": {
        "raw_text": "no, this should be at the chin, not the temple",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": _nondiff_response(
        "contradiction",
        explanation="Original description said 'at the right temple' but you're now asking for the chin.",
        follow_up="You originally described the location as 'temple' but now want 'chin' — which is right?",
    ),
    "expect_llm_called": True,
    "expected_intent": "contradiction",
    "expected_paths": [],
    "expected_needs_confirmation": True,
    "expected_follow_up_present": True,
    "expected_updated_slots": {},
})


# 19. Contradiction — handshape conflict
FIXTURES.append({
    "id": "19-contradiction-handshape",
    "notes": "Correction contradicts the prose's handshape claim.",
    "current_params": _scalar_params(handshape_dominant="fist", location="chest"),
    "current_hamnosys": "\uE000\uE020\uE03C\uE048",
    "original_prose": "A closed fist on the chest.",
    "correction": {
        "raw_text": "actually it should be an open hand, not a fist",
        "target_time_ms": None,
        "target_region": "hand",
    },
    "recorded_response": _nondiff_response(
        "contradiction",
        explanation="Original prose specified 'closed fist' but the correction asks for an open hand.",
        follow_up="You originally said 'closed fist' but now want an open hand — which should I keep?",
    ),
    "expect_llm_called": True,
    "expected_intent": "contradiction",
    "expected_paths": [],
    "expected_needs_confirmation": True,
    "expected_follow_up_present": True,
    "expected_updated_slots": {},
})


# 20. Restart — deterministic fast path; LLM is never called
FIXTURES.append({
    "id": "20-restart-whole-sign-wrong",
    "notes": "Deterministic restart phrase; LLM short-circuited.",
    "current_params": _scalar_params(handshape_dominant="fist", location="chest"),
    "current_hamnosys": "\uE000\uE020\uE03C\uE048",
    "original_prose": "Closed fist at the chest.",
    "correction": {
        "raw_text": "the whole sign is wrong, let me start over",
        "target_time_ms": None,
        "target_region": None,
    },
    "recorded_response": {
        "intent": "restart",
        "explanation": "(unused — deterministic restart fast path)",
        "needs_user_confirmation": True,
        "follow_up_question": None,
        "field_changes": [],
    },
    "expect_llm_called": False,
    "expected_intent": "restart",
    "expected_paths": [],
    "expected_needs_confirmation": True,
    "expected_follow_up_present": False,
    "expected_updated_slots": {},
})


def _validate_fixtures() -> None:
    seen_ids: set[str] = set()
    for f in FIXTURES:
        if f["id"] in seen_ids:
            raise SystemExit(f"duplicate fixture id: {f['id']!r}")
        seen_ids.add(f["id"])
    if len(FIXTURES) != 20:
        raise SystemExit(
            f"expected exactly 20 fixtures, got {len(FIXTURES)}"
        )


def main() -> None:
    _validate_fixtures()
    HERE.mkdir(parents=True, exist_ok=True)
    for fixture in FIXTURES:
        path = HERE / f"{fixture['id']}.json"
        path.write_text(
            json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path.name}")
    print(f"total: {len(FIXTURES)} fixtures")


if __name__ == "__main__":
    main()
