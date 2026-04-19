"""Deterministic question templates per phonological slot.

Two roles:

1. **Primary source of question text** when the LLM is unavailable or
   returns malformed output — :func:`generate_questions` falls back to
   :func:`template_for` per gap field so the authoring UI is never
   blocked on an LLM failure.
2. **Reference the LLM may rewrite.** The generator's system prompt
   embeds every template so the LLM can use the phrasing as a starting
   point and adapt for the author's own prose.

Keys mostly match a slot from :data:`parser.ALLOWED_GAP_FIELDS`. Two
extra keys — ``two_handed_symmetry`` and ``regional_variant`` — are
contextual templates the LLM may draw on but :func:`generate_questions`
does not auto-emit, since those concepts span more than one slot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Option, Question


@dataclass(frozen=True)
class Template:
    """Frozen record of per-slot phrasings and option lists.

    ``text`` is the long plain-English phrasing (for general signers).
    ``deaf_native_text`` is the terser direct-terminology variant used
    when the signer has indicated they are a Deaf native signer.
    """

    text: str
    deaf_native_text: str
    options: tuple[Option, ...]
    allow_freeform: bool
    rationale: str


TEMPLATES: dict[str, Template] = {
    "handshape_dominant": Template(
        text=(
            "Which handshape is used? For example: a closed fist, a flat "
            "palm, a spread five, or a bent-five (claw) shape?"
        ),
        deaf_native_text=(
            "Which handshape — A (fist), B (flat), 5, bent-5, or another?"
        ),
        options=(
            Option(label="Closed fist (A / S)", value="fist"),
            Option(label="Flat palm (B / flat)", value="flat"),
            Option(label="Spread five (5)", value="5"),
            Option(label="Bent-5 / claw", value="bent-5"),
            Option(label="Index point (1 / G)", value="index"),
        ),
        allow_freeform=True,
        rationale=(
            "Handshape is the most discriminating phonological parameter; "
            "always ask with concrete visual options when gapped."
        ),
    ),
    "handshape_nondominant": Template(
        text=(
            "What is the non-dominant (helper) hand doing? The same shape "
            "as the dominant hand, a flat-palm base, or is this a one-"
            "handed sign?"
        ),
        deaf_native_text=(
            "Non-dominant handshape — mirror dominant, flat base, or "
            "one-handed?"
        ),
        options=(
            Option(label="Same as dominant hand", value="same"),
            Option(label="Flat palm (B) — passive base", value="flat"),
            Option(label="Closed fist (A / S)", value="fist"),
            Option(label="No second hand — one-handed sign", value="none"),
        ),
        allow_freeform=True,
        rationale=(
            "Non-dominant handshape often mirrors the dominant one or is a "
            "flat passive base; include a one-handed escape hatch."
        ),
    ),
    "orientation_extended_finger": Template(
        text=(
            "Which direction do the fingertips point? Up, down, forward "
            "(away from you), or toward your body?"
        ),
        deaf_native_text=(
            "Extended-finger direction — up, down, forward, toward "
            "signer, or to the side?"
        ),
        options=(
            Option(label="Up", value="up"),
            Option(label="Down", value="down"),
            Option(label="Forward (away from signer)", value="forward"),
            Option(label="Toward signer", value="toward signer"),
            Option(label="To the side", value="ipsilateral"),
        ),
        allow_freeform=True,
        rationale=(
            "Five cardinal directions cover the vast majority of signs."
        ),
    ),
    "orientation_palm": Template(
        text=(
            "Which way does the palm face? Down toward the floor, up "
            "toward the ceiling, toward you, or away from you?"
        ),
        deaf_native_text=(
            "Palm direction — down, up, toward signer, away, or to the "
            "side?"
        ),
        options=(
            Option(label="Down (toward the floor)", value="down"),
            Option(label="Up (toward the ceiling)", value="up"),
            Option(label="Toward signer (toward your body)", value="toward signer"),
            Option(label="Away from signer", value="away from signer"),
            Option(label="To the side", value="ipsilateral"),
        ),
        allow_freeform=True,
        rationale=(
            "Palm direction is a discrete slot with a small known "
            "vocabulary; always offer the six cardinal directions."
        ),
    ),
    "location": Template(
        text=(
            "Where on the body is the sign made? For example: at the "
            "temple, at the chin, on the chest, or in neutral space in "
            "front of you?"
        ),
        deaf_native_text="Location?",
        options=(
            Option(label="Forehead / temple", value="temple"),
            Option(label="Mouth / chin", value="chin"),
            Option(label="Chest", value="chest"),
            Option(label="Neutral space (in front of body)", value="neutral space"),
            Option(label="Non-dominant palm / hand", value="palm"),
        ),
        allow_freeform=True,
        rationale=(
            "Location vocabulary is large — show the most common anchors "
            "and always allow freeform for unusual spots."
        ),
    ),
    "contact": Template(
        text=(
            "Does the hand make contact with the body or the other hand? "
            "If so, is it a brief tap, a sustained touch, or a brush?"
        ),
        deaf_native_text="Contact — tap, hold, brush, or none?",
        options=(
            Option(label="No contact", value="no contact"),
            Option(label="Brief touch / tap", value="tap"),
            Option(label="Sustained contact / hold", value="hold"),
            Option(label="Brush / graze", value="brush"),
        ),
        allow_freeform=True,
        rationale="Contact type is discrete; four options cover most signs.",
    ),
    "movement": Template(
        text=(
            "How does the hand move? In a straight line, in an arc, in a "
            "full circle, with a tap, or not at all?"
        ),
        deaf_native_text=(
            "Movement shape — straight, arc, circle, tap, wiggle, or "
            "none?"
        ),
        options=(
            Option(label="Straight line", value="straight"),
            Option(label="Arc (curved path)", value="arc"),
            Option(label="Full circle", value="circular"),
            Option(label="Tapping / repeated contact", value="tapping"),
            Option(label="No movement", value="none"),
        ),
        allow_freeform=True,
        rationale=(
            "Movement path is a small discrete class. The 'none' option "
            "matters — absence of movement is a valid answer."
        ),
    ),
    "non_manual.mouth_picture": Template(
        text=(
            "Is there a particular mouth shape or mouthing? For example: "
            "mouth open, pursed lips, tongue out, or no specific mouth "
            "shape?"
        ),
        deaf_native_text="Mouth picture (SAMPA-ish)?",
        options=(
            Option(label="No specific mouth shape", value="none"),
            Option(label="Mouth open", value="open"),
            Option(label="Mouth closed / pursed", value="pursed"),
            Option(label="Tongue out", value="tongue out"),
        ),
        allow_freeform=True,
        rationale=(
            "Mouth shape is mostly a small discrete set; freeform covers "
            "specific mouthings like 'pah' or 'shh'."
        ),
    ),
    "non_manual.eye_gaze": Template(
        text=(
            "Where is the signer's gaze directed? At the person being "
            "addressed, toward the hand, at a spatial point, or neutral?"
        ),
        deaf_native_text="Gaze target?",
        options=(
            Option(label="At addressee", value="at addressee"),
            Option(label="At the hand", value="toward hand"),
            Option(label="To a spatial referent", value="to location"),
            Option(label="Down / neutral", value="neutral"),
        ),
        allow_freeform=True,
        rationale=(
            "Gaze often anchors reference — give a spatial-referent option."
        ),
    ),
    "non_manual.head_movement": Template(
        text=(
            "Is the head moving? For example: nodding, shaking side to "
            "side, or tilted?"
        ),
        deaf_native_text="Head movement?",
        options=(
            Option(label="Still", value="still"),
            Option(label="Nod (vertical)", value="nod"),
            Option(label="Shake (horizontal)", value="shake"),
            Option(label="Tilt to the side", value="tilt"),
        ),
        allow_freeform=True,
        rationale=(
            "Head movement is a small class tied to non-manual grammar."
        ),
    ),
    "non_manual.eyebrows": Template(
        text=(
            "What are the eyebrows doing? Raised (for yes/no questions), "
            "furrowed / drawn together (for wh-questions), or neutral?"
        ),
        deaf_native_text="Brow — raised, furrowed, or neutral?",
        options=(
            Option(label="Raised", value="raised"),
            Option(label="Furrowed", value="furrowed"),
            Option(label="Neutral", value="neutral"),
        ),
        allow_freeform=True,
        rationale=(
            "Brow position is grammatical in BSL/ASL — raised ~ Y/N, "
            "furrowed ~ wh-question."
        ),
    ),
    "non_manual.facial_expression": Template(
        text=(
            "What is the overall facial expression? For example: neutral, "
            "smiling, surprised, or puzzled?"
        ),
        deaf_native_text="Facial expression?",
        options=(
            Option(label="Neutral", value="neutral"),
            Option(label="Smile", value="smile"),
            Option(label="Surprise", value="surprise"),
            Option(label="Puzzled / uncertain", value="puzzled"),
        ),
        allow_freeform=True,
        rationale="Overall affect — freeform for anything unusual.",
    ),
    "two_handed_symmetry": Template(
        text=(
            "Both hands are involved — are they doing the same thing "
            "(mirrored), or is one hand a still base while the other "
            "moves?"
        ),
        deaf_native_text=(
            "Two-handed type — symmetric, asymmetric base, or "
            "one-handed?"
        ),
        options=(
            Option(label="Both hands move the same way (symmetric)", value="symmetric"),
            Option(label="Non-dominant acts as a still base", value="base"),
            Option(label="Both hands move differently (asymmetric)", value="asymmetric"),
            Option(label="Only the dominant hand — one-handed", value="one-handed"),
        ),
        allow_freeform=True,
        rationale=(
            "Used when handshape_nondominant is gapped AND the description "
            "implies a two-handed sign — one question resolves both the "
            "second handshape and the interaction type."
        ),
    ),
    "regional_variant": Template(
        text=(
            "The description mentions a regional variant. Which region "
            "should we record — for example, Southern British, Northern "
            "British, Scottish, or a specific city variant?"
        ),
        deaf_native_text="Regional variant — which region?",
        options=(
            Option(label="Southern British (SE England)", value="southern british"),
            Option(label="Northern British (Manchester, Leeds)", value="northern british"),
            Option(label="Scottish", value="scottish"),
            Option(label="London", value="london"),
        ),
        allow_freeform=True,
        rationale=(
            "Used when a gap's reason flags a regional variant we must "
            "pin down before filling the affected slots."
        ),
    ),
}


def template_for(
    field: str,
    *,
    is_deaf_native: bool = False,
    reason: str = "",
) -> Question:
    """Return a :class:`Question` for ``field`` from the template library.

    Picks ``deaf_native_text`` when ``is_deaf_native`` is truthy; otherwise
    uses the longer plain-English phrasing. If ``reason`` is non-empty, it
    is appended to the debug rationale so operators can trace why the
    parser flagged the slot. Unknown fields fall back to a generic
    freeform question.
    """
    t = TEMPLATES.get(field)
    if t is None:
        leaf = field.replace("_", " ").replace(".", " — ")
        return Question(
            field=field,
            text=f"Could you describe the {leaf} in more detail?",
            options=None,
            allow_freeform=True,
            rationale=f"No template for slot {field!r}; freeform fallback.",
        )
    text = t.deaf_native_text if is_deaf_native else t.text
    rationale = t.rationale
    if reason:
        rationale = f"{rationale} (parser note: {reason})"
    return Question(
        field=field,
        text=text,
        options=list(t.options) if t.options else None,
        allow_freeform=t.allow_freeform,
        rationale=rationale,
    )


__all__ = ["TEMPLATES", "Template", "template_for"]
