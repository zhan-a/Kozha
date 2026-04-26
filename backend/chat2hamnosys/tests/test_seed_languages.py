"""Tests for the rare-SL seed languages (Kazakh, Russian, …).

When a contributor authors a sign in a sign language with no qualified
Deaf reviewer on the roster, the entry must:

- accept and store cleanly (the model layer must not reject the new
  language code),
- promote to ``pending_review`` on accept (so the contributor sees the
  submission landed),
- refuse to graduate to ``validated`` (no competent reviewer can approve),
- refuse to export into the live Kozha library (the export gate is the
  defense-in-depth backstop).

A reviewer registered for a *different* language must not be able to
unlock the entry — language scoping is enforced by
``review.actions._competent_or_raise``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from models import SignEntry
from review.actions import (
    ReviewerNotCompetent,
    approve,
    qualifying_approval_count,
)
from review.policy import ReviewPolicy
from review.storage import ReviewerStore
from storage import (
    ExportNotAllowedError,
    InsufficientApprovalsError,
    SQLiteSignStore,
)


# Codes we expect the model layer to accept end-to-end. KSL is the headline
# Kazakh case from the prompt; the rest exercise the same scaffolding for
# the other rare-SL seeds so a future broken-Literal regression would
# surface here rather than only on Kazakh.
RARE_SL_CODES = [
    "ksl",   # Kazakh Sign Language
    "rsl",   # Russian
    "usl",   # Ukrainian
    "tid",   # Turkish (Türk İşaret Dili)
    "jsl",   # Japanese
    "kvk",   # Korean
    "csl",   # Chinese
    "arsl",  # Arabic (umbrella)
    "msl",   # Mongolian
    "zei",   # Persian (ZEI)
]


@pytest.fixture
def sign_store(tmp_path: Path) -> SQLiteSignStore:
    return SQLiteSignStore(
        db_path=tmp_path / "signs.sqlite3",
        kozha_data_dir=tmp_path / "kozha",
    )


@pytest.fixture
def reviewer_store(tmp_path: Path) -> ReviewerStore:
    return ReviewerStore(db_path=tmp_path / "reviewers.sqlite3")


@pytest.fixture
def policy_default() -> ReviewPolicy:
    return ReviewPolicy(
        min_approvals=2,
        require_native_deaf=True,
        allow_single_approval=False,
        require_region_match=False,
    )


@pytest.mark.parametrize("sign_language", RARE_SL_CODES)
def test_rare_language_accepted_by_model_layer(
    valid_entry_factory: Callable[..., SignEntry], sign_language: str
) -> None:
    """The Pydantic model must accept the seed-language code without raising."""
    entry = valid_entry_factory(sign_language=sign_language)
    assert entry.sign_language == sign_language


def test_kazakh_entry_persists_and_round_trips(
    sign_store: SQLiteSignStore,
    valid_entry_factory: Callable[..., SignEntry],
) -> None:
    """A KSL draft must survive a write/read round-trip in the SQLite store."""
    entry = valid_entry_factory(
        sign_language="ksl",
        gloss="SALEM",  # Kazakh "hello" — placeholder gloss for the test
        status="draft",
    )
    sign_store.put(entry)
    fetched = sign_store.get(entry.id)
    assert fetched.sign_language == "ksl"
    assert fetched.gloss == "SALEM"


def test_kazakh_export_refused_without_validated_status(
    sign_store: SQLiteSignStore,
    valid_entry_factory: Callable[..., SignEntry],
    policy_default: ReviewPolicy,
) -> None:
    """A pending_review KSL entry must not be exportable.

    The status gate is the first of two independent checks. Without a
    qualified reviewer, the entry never reaches ``validated``, so this
    is the gate the seed-language path actually trips.
    """
    entry = valid_entry_factory(sign_language="ksl", status="pending_review")
    sign_store.put(entry)
    with pytest.raises(ExportNotAllowedError):
        sign_store.export_to_kozha_library(entry.id, policy=policy_default)


def test_kazakh_reviewer_for_other_language_cannot_approve(
    sign_store: SQLiteSignStore,
    reviewer_store: ReviewerStore,
    valid_entry_factory: Callable[..., SignEntry],
    policy_default: ReviewPolicy,
) -> None:
    """A BSL-only reviewer must not unlock a KSL entry.

    Language scoping (``review.actions._competent_or_raise``) is the
    line that keeps a reviewer trained in one signing community from
    speaking for another.
    """
    entry = valid_entry_factory(sign_language="ksl", status="pending_review")
    sign_store.put(entry)
    bsl_only, _ = reviewer_store.create(
        display_name="BSL-only reviewer",
        is_deaf_native=True,
        signs=["bsl"],
    )
    with pytest.raises(ReviewerNotCompetent):
        approve(
            sign_id=entry.id,
            reviewer=bsl_only,
            store=sign_store,
            policy=policy_default,
        )
    # Status untouched — no record, no transition.
    refetched = sign_store.get(entry.id)
    assert refetched.status == "pending_review"
    assert refetched.reviewers == []


def test_kazakh_export_gate_independently_enforces_approval_count(
    sign_store: SQLiteSignStore,
    valid_entry_factory: Callable[..., SignEntry],
    policy_default: ReviewPolicy,
) -> None:
    """Defense-in-depth: even if status is hand-mutated to ``validated``,
    the export gate refuses to write the entry without qualifying approvals.

    This is what protects the live Kozha library from a buggy admin flow,
    a hand-edited DB row, or any path that would otherwise let a seed
    sign with zero Deaf-native approvals leak into the published corpus.
    """
    entry = valid_entry_factory(sign_language="ksl", status="validated")
    sign_store.put(entry)
    with pytest.raises(InsufficientApprovalsError) as exc:
        sign_store.export_to_kozha_library(entry.id, policy=policy_default)
    assert exc.value.present == 0
    assert exc.value.required == 2


def test_kazakh_full_state_machine_with_qualified_reviewers(
    sign_store: SQLiteSignStore,
    reviewer_store: ReviewerStore,
    valid_entry_factory: Callable[..., SignEntry],
    policy_default: ReviewPolicy,
) -> None:
    """The happy path proves the rare-SL plumbing isn't permanently broken.

    Once two native-Deaf KSL reviewers exist on the roster, a KSL entry
    transits draft → pending_review → validated and the export gate
    permits writing it to the authored library file.
    """
    a, _ = reviewer_store.create(
        display_name="Aigerim",
        is_deaf_native=True,
        signs=["ksl"],
    )
    b, _ = reviewer_store.create(
        display_name="Berik",
        is_deaf_native=True,
        signs=["ksl"],
    )
    entry = valid_entry_factory(sign_language="ksl", status="pending_review")
    sign_store.put(entry)
    out = approve(sign_id=entry.id, reviewer=a, store=sign_store, policy=policy_default)
    assert out.status == "pending_review"
    assert qualifying_approval_count(out, policy_default) == 1
    out = approve(sign_id=entry.id, reviewer=b, store=sign_store, policy=policy_default)
    assert out.status == "validated"
    # Now the export gate accepts the write.
    sign_store.export_to_kozha_library(entry.id, policy=policy_default)
    target = sign_store.kozha_data_dir / "hamnosys_ksl_authored.sigml"
    assert target.exists()
