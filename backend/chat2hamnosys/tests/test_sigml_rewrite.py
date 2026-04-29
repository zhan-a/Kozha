"""Tests for the deterministic SiGML rewriter.

The rewriter short-circuits the LLM-backed correction interpreter for
two classes of correction:

  1. Structured chip swaps (``target_region`` payloads from the
     frontend chip strip).
  2. Free-text chat corrections that name HamNoSys tags directly or
     describe a palm direction in plain English.

These tests cover the third dispatch route — the HamNoSys tag-syntax
matcher added in response to a production miss where the user wrote
``change <hampalmr/> to <hampalmd/>`` (advisory wrong source tag —
the sign actually carried ``<hampalml/>``) and the LLM interpreter
classified the correction as VAGUE because old_value couldn't be
matched against current parameters.
"""
from __future__ import annotations

import pytest

from correct.sigml_rewrite import (
    TagSwap,
    apply_tag_swap_to_sigml,
    extract_manual_tags,
    match_deterministic_swap,
)
from session.orchestrator import Correction


HELLO_SIGML = (
    "<?xml version='1.0' encoding='UTF-8'?>\n"
    "<sigml>\n"
    "  <hns_sign gloss=\"HELLO\">\n"
    "    <hamnosys_manual>\n"
    "      <hamflathand/>\n"
    "      <hamextfingeru/>\n"
    "      <hampalml/>\n"
    "      <hamneutralspace/>\n"
    "      <hamswinging/>\n"
    "    </hamnosys_manual>\n"
    "  </hns_sign>\n"
    "</sigml>\n"
)


def _correction(text: str, *, region: str | None = None) -> Correction:
    return Correction(raw_text=text, target_time_ms=None, target_region=region)


# ---------------------------------------------------------------------------
# Tag-syntax matcher — production-miss regression
# ---------------------------------------------------------------------------


class TestTagSyntaxMatcher:
    def test_regression_advisory_wrong_source_tag(self) -> None:
        """The motivating production case: user names a source tag that
        does not appear in the sign (``<hampalmr/>``) but the target
        tag (``<hampalmd/>``) is unambiguous. The matcher should swap
        whatever palm-direction tag the sign actually has
        (``<hampalml/>``) to the target — not bail on the source
        mismatch.
        """
        swap = match_deterministic_swap(
            _correction("change <hampalmr/> to <hampalmd/>"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd")

    def test_correct_source_tag(self) -> None:
        swap = match_deterministic_swap(
            _correction("change <hampalml/> to <hampalmd/>"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd")

    def test_swap_phrasing(self) -> None:
        swap = match_deterministic_swap(
            _correction("swap <hamflathand/> for <hamfist/>"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hamflathand", to_tag="hamfist")

    def test_arrow_phrasing(self) -> None:
        swap = match_deterministic_swap(
            _correction("<hampalml/> -> <hampalmd/>"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd")

    def test_unicode_arrow_phrasing(self) -> None:
        swap = match_deterministic_swap(
            _correction("<hampalml/> → <hampalmd/>"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd")

    def test_use_instead_phrasing(self) -> None:
        swap = match_deterministic_swap(
            _correction("use <hampalmu/> instead"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmu")

    def test_make_it_phrasing(self) -> None:
        swap = match_deterministic_swap(
            _correction("make it <hamfist/>"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hamflathand", to_tag="hamfist")

    def test_bare_word_reference(self) -> None:
        # No angle brackets around the target — only fires when the
        # bare token matches a real catalog entry.
        swap = match_deterministic_swap(
            _correction("use hampalmd instead"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd")

    # ------- expected misses (fall through to LLM path) -------

    def test_target_tag_unknown_returns_none(self) -> None:
        swap = match_deterministic_swap(
            _correction("change <hampalmd/> to <hamnotreal/>"),
            HELLO_SIGML,
        )
        assert swap is None

    def test_target_already_current_is_noop(self) -> None:
        # Sign already carries <hampalml/>; "change to <hampalml/>"
        # has nothing to do.
        swap = match_deterministic_swap(
            _correction("change to <hampalml/>"),
            HELLO_SIGML,
        )
        assert swap is None

    def test_target_category_not_in_sign_returns_none(self) -> None:
        # HELLO has no contact slot in its sigml. A correction that
        # references a contact tag should fall through so the LLM
        # interpreter can route it to ELABORATE.
        swap = match_deterministic_swap(
            _correction("add <hamtouch/>"),
            HELLO_SIGML,
        )
        assert swap is None

    def test_no_ham_tag_falls_through(self) -> None:
        # No tag references at all — the natural-language palm matcher
        # gets a shot, and on its miss the caller drops to the LLM.
        swap = match_deterministic_swap(
            _correction("looks weird"),
            HELLO_SIGML,
        )
        assert swap is None

    def test_natural_language_palm_still_works(self) -> None:
        # Don't regress the existing palm-direction NL matcher.
        swap = match_deterministic_swap(
            _correction("the palm should face down"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd")


# ---------------------------------------------------------------------------
# Structured chip-swap matcher — production-miss regression
# ---------------------------------------------------------------------------


class TestStructuredSwap:
    """``target_region="swap:from:to[:index]"`` is the payload encoded
    by ``contribute-sigml-edit.js`` when the contributor clicks a chip
    in the annotated SiGML strip and picks an alternative. ``index`` is
    the chip's *overall* position among every ``<ham*/>`` tag in
    ``<hamnosys_manual>`` — not the occurrence-of-from_tag index that
    :class:`TagSwap` carries forward to ``apply_tag_swap_to_sigml``.
    The parser must convert between the two."""

    def test_picker_swap_at_nonzero_overall_index(self) -> None:
        # HELLO has hampalml at overall position 2 (after hamflathand
        # and hamextfingeru). The chip strip sends index=2 because
        # that's the chip's rank in the strip. There's only one
        # hampalml occurrence, so TagSwap.index must come back as 0
        # (or None) — index=2 would make apply_tag_swap_to_sigml look
        # for a nonexistent third hampalml and raise.
        swap = match_deterministic_swap(
            _correction("chip swap", region="swap:hampalml:hampalmd:2"),
            HELLO_SIGML,
        )
        assert swap is not None
        assert swap.from_tag == "hampalml"
        assert swap.to_tag == "hampalmd"
        # Either 0 or None is correct (both mean "first occurrence").
        assert swap.index in (0, None)
        # End-to-end: the swap the matcher produces must apply cleanly
        # to the source SiGML. Before the fix this raised ValueError
        # and the router rolled the session back, reverting the chip.
        rewritten = apply_tag_swap_to_sigml(HELLO_SIGML, swap)
        assert "<hampalmd/>" in rewritten
        assert "<hampalml/>" not in rewritten

    def test_picker_swap_without_index(self) -> None:
        swap = match_deterministic_swap(
            _correction("chip swap", region="swap:hampalml:hampalmd"),
            HELLO_SIGML,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd", index=None)

    def test_picker_swap_unknown_index_falls_back_to_first(self) -> None:
        # idx=4 isn't a hampalml position — parser drops the index and
        # the swap targets the first (and only) occurrence.
        swap = match_deterministic_swap(
            _correction("chip swap", region="swap:hampalml:hampalmd:4"),
            HELLO_SIGML,
        )
        assert swap is not None
        assert swap.index is None

    def test_picker_swap_from_tag_absent_returns_none(self) -> None:
        # No hamflatfist in HELLO — caller should fall through.
        swap = match_deterministic_swap(
            _correction("chip swap", region="swap:hamflatfist:hamfist:0"),
            HELLO_SIGML,
        )
        assert swap is None

    def test_picker_swap_second_occurrence_maps_to_index_one(self) -> None:
        # Pathological-but-legal sign with two palm-direction tags
        # (e.g. dominant + non-dominant slots). The chip at overall
        # position 4 is the *second* hampalml — TagSwap.index should
        # be 1 so apply_tag_swap_to_sigml swaps the right one.
        sigml_two_palms = (
            "<sigml><hns_sign><hamnosys_manual>"
            "<hamflathand/>"
            "<hamextfingeru/>"
            "<hampalml/>"
            "<hamneutralspace/>"
            "<hampalml/>"
            "</hamnosys_manual></hns_sign></sigml>"
        )
        swap = match_deterministic_swap(
            _correction("chip swap", region="swap:hampalml:hampalmd:4"),
            sigml_two_palms,
        )
        assert swap == TagSwap(from_tag="hampalml", to_tag="hampalmd", index=1)
        rewritten = apply_tag_swap_to_sigml(sigml_two_palms, swap)
        # First hampalml stays, second flips to hampalmd.
        assert rewritten.count("<hampalml/>") == 1
        assert rewritten.count("<hampalmd/>") == 1


# ---------------------------------------------------------------------------
# Sanity-check the helper used by the matcher
# ---------------------------------------------------------------------------


def test_extract_manual_tags_preserves_order() -> None:
    assert extract_manual_tags(HELLO_SIGML) == [
        "hamflathand",
        "hamextfingeru",
        "hampalml",
        "hamneutralspace",
        "hamswinging",
    ]
