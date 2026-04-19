"""Tests for ``backend.chat2hamnosys.clarify.templates``.

The template library is the deterministic fallback path for the
question generator. Every slot the parser may flag must have a
template, the templates must be valid :class:`Question`s, and the
Deaf-native phrasing must actually be shorter than the plain version
(otherwise the "less hand-holding" contract is empty).
"""

from __future__ import annotations

import pytest

from clarify import Option, Question
from clarify.templates import TEMPLATES, Template, template_for
from parser import ALLOWED_GAP_FIELDS


class TestTemplateCoverage:
    def test_every_allowed_gap_field_has_a_template(self) -> None:
        missing = [f for f in ALLOWED_GAP_FIELDS if f not in TEMPLATES]
        assert not missing, f"templates missing for: {missing}"

    def test_contextual_templates_exist(self) -> None:
        # These aren't in ALLOWED_GAP_FIELDS but the module docstring
        # promises they are present so the LLM can draw on them.
        assert "two_handed_symmetry" in TEMPLATES
        assert "regional_variant" in TEMPLATES

    def test_template_shape(self) -> None:
        for name, t in TEMPLATES.items():
            assert isinstance(t, Template), name
            assert t.text.strip(), f"{name}: blank text"
            assert t.deaf_native_text.strip(), f"{name}: blank deaf_native_text"
            assert isinstance(t.options, tuple), name
            for opt in t.options:
                assert isinstance(opt, Option)
            assert isinstance(t.allow_freeform, bool), name
            assert t.rationale.strip(), f"{name}: blank rationale"


class TestDeafNativePhrasing:
    def test_deaf_native_is_shorter_on_average(self) -> None:
        # Not strictly shorter for every slot (e.g. location → "Location?"
        # vs. a long sentence), but the population mean must drop.
        plain_total = sum(len(t.text) for t in TEMPLATES.values())
        native_total = sum(len(t.deaf_native_text) for t in TEMPLATES.values())
        assert native_total < plain_total, (
            f"deaf-native phrasing should be terser overall: "
            f"{native_total=} vs {plain_total=}"
        )

    @pytest.mark.parametrize(
        "field",
        ["handshape_dominant", "orientation_palm", "location"],
    )
    def test_template_for_switches_text(self, field: str) -> None:
        plain = template_for(field, is_deaf_native=False)
        native = template_for(field, is_deaf_native=True)
        assert plain.text != native.text, f"{field}: phrasing identical"
        assert plain.field == native.field == field


class TestTemplateFor:
    def test_returns_valid_question(self) -> None:
        q = template_for("handshape_dominant")
        assert isinstance(q, Question)
        assert q.field == "handshape_dominant"
        assert q.text.strip()
        assert q.options is not None and len(q.options) >= 2

    def test_unknown_field_freeform_fallback(self) -> None:
        q = template_for("utterly_made_up_slot")
        assert q.field == "utterly_made_up_slot"
        assert q.options is None
        assert q.allow_freeform is True
        assert "utterly made up slot" in q.text.lower()

    def test_reason_appended_to_rationale(self) -> None:
        q = template_for("handshape_dominant", reason="claw is bent-5 or bent-V")
        assert "claw is bent-5 or bent-V" in q.rationale

    def test_reason_omitted_keeps_rationale_clean(self) -> None:
        q = template_for("handshape_dominant")
        assert "(parser note:" not in q.rationale

    def test_nested_non_manual_field(self) -> None:
        q = template_for("non_manual.eyebrows")
        assert q.field == "non_manual.eyebrows"
        assert q.options is not None
        values = {o.value for o in q.options}
        assert {"raised", "furrowed", "neutral"}.issubset(values)


class TestOptionsAreDistinct:
    def test_option_values_distinct_within_each_template(self) -> None:
        for name, t in TEMPLATES.items():
            if not t.options:
                continue
            values = [o.value for o in t.options]
            assert len(values) == len(set(values)), (
                f"{name}: duplicate option values {values}"
            )
