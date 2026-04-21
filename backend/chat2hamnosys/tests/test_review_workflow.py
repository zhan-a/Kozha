"""Tests for the Deaf-reviewer workflow and the export gate.

Covers:

- Reviewer-store CRUD + token authentication.
- Action functions: approve, reject, request_revision, flag,
  clear_quarantine — including the two-reviewer rule, region match,
  non-native overrides, and terminal-state guards.
- The export gate: status check + defense-in-depth approval count.
- Audit-log hash chain — both the happy path and tamper detection.
- The HTTP layer end-to-end via :class:`fastapi.testclient.TestClient`.

The store fixtures live in :mod:`tests.conftest` (``valid_entry_factory``,
``permissive_review_policy``); we add a few review-specific fixtures
locally (a registered native-Deaf reviewer pair, a board reviewer, etc).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from models import SignEntry
from review.actions import (
    CannotActOnTerminalEntry,
    NonNativeApprovalForbidden,
    ReviewActionError,
    ReviewerNotCompetent,
    approve,
    clear_quarantine,
    flag,
    qualifying_approval_count,
    qualifying_approval_ids,
    reject,
    request_revision,
)
from review.policy import ReviewPolicy
from review.storage import (
    ExportAuditLog,
    ReviewerAuthError,
    ReviewerStore,
    hash_token,
    new_reviewer_token,
)
from storage import (
    ExportNotAllowedError,
    InsufficientApprovalsError,
    SQLiteSignStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reviewer_store(tmp_path: Path) -> ReviewerStore:
    return ReviewerStore(db_path=tmp_path / "reviewers.sqlite3")


@pytest.fixture
def audit_log(tmp_path: Path) -> ExportAuditLog:
    return ExportAuditLog(log_path=tmp_path / "exports.jsonl")


@pytest.fixture
def policy_default() -> ReviewPolicy:
    """Production-like policy: two native-Deaf approvals, region-match on."""
    return ReviewPolicy(
        min_approvals=2,
        require_native_deaf=True,
        allow_single_approval=False,
        require_region_match=True,
    )


@pytest.fixture
def policy_single_approval() -> ReviewPolicy:
    return ReviewPolicy(
        min_approvals=2,
        require_native_deaf=True,
        allow_single_approval=True,
        require_region_match=True,
    )


@pytest.fixture
def sign_store(tmp_path: Path) -> SQLiteSignStore:
    return SQLiteSignStore(
        db_path=tmp_path / "signs.sqlite3",
        kozha_data_dir=tmp_path / "kozha",
    )


@pytest.fixture
def native_pair(reviewer_store: ReviewerStore):
    """Two native-Deaf BSL-London reviewers sharing the right region."""
    a, _ = reviewer_store.create(
        display_name="Alex",
        is_deaf_native=True,
        signs=["bsl"],
        regional_background="BSL-London",
    )
    b, _ = reviewer_store.create(
        display_name="Beth",
        is_deaf_native=True,
        signs=["bsl"],
        regional_background="BSL-London",
    )
    return a, b


@pytest.fixture
def hearing_reviewer(reviewer_store: ReviewerStore):
    r, _ = reviewer_store.create(
        display_name="Hearing-Helper",
        is_deaf_native=False,
        signs=["bsl"],
        regional_background="BSL-London",
    )
    return r


@pytest.fixture
def board_reviewer(reviewer_store: ReviewerStore):
    r, token = reviewer_store.create(
        display_name="Board-Member",
        is_deaf_native=True,
        is_board=True,
        signs=["bsl"],
        regional_background="BSL-London",
    )
    return r, token


# ---------------------------------------------------------------------------
# Reviewer store + auth
# ---------------------------------------------------------------------------


class TestReviewerStore:
    def test_create_returns_reviewer_and_raw_token(self, reviewer_store):
        r, token = reviewer_store.create(
            display_name="Alex", is_deaf_native=True, signs=["bsl"]
        )
        assert token  # raw token returned exactly once
        assert r.token_hash == hash_token(token)
        # Round-trip through the DB.
        same = reviewer_store.get(r.id)
        assert same.display_name == "Alex"
        assert same.is_deaf_native is True
        assert same.signs == ["bsl"]

    def test_authenticate_known_token_returns_reviewer(self, reviewer_store):
        r, token = reviewer_store.create(display_name="A", signs=["bsl"])
        out = reviewer_store.authenticate(token)
        assert out.id == r.id

    def test_authenticate_missing_token_raises(self, reviewer_store):
        with pytest.raises(ReviewerAuthError):
            reviewer_store.authenticate(None)
        with pytest.raises(ReviewerAuthError):
            reviewer_store.authenticate("")

    def test_authenticate_unknown_token_raises(self, reviewer_store):
        with pytest.raises(ReviewerAuthError):
            reviewer_store.authenticate(new_reviewer_token())

    def test_deactivated_reviewer_cannot_authenticate(self, reviewer_store):
        r, token = reviewer_store.create(display_name="A", signs=["bsl"])
        reviewer_store.deactivate(r.id)
        with pytest.raises(ReviewerAuthError):
            reviewer_store.authenticate(token)

    def test_list_only_active_by_default(self, reviewer_store):
        a, _ = reviewer_store.create(display_name="A", signs=["bsl"])
        b, _ = reviewer_store.create(display_name="B", signs=["bsl"])
        reviewer_store.deactivate(b.id)
        ids = [r.id for r in reviewer_store.list()]
        assert ids == [a.id]
        all_ids = [r.id for r in reviewer_store.list(only_active=False)]
        assert set(all_ids) == {a.id, b.id}

    def test_token_hash_is_unique_constraint(self, reviewer_store):
        # Two creates produce different tokens; collisions are astronomically
        # unlikely but the UNIQUE constraint is the backstop.
        a, _ = reviewer_store.create(display_name="A", signs=["bsl"])
        b, _ = reviewer_store.create(display_name="B", signs=["bsl"])
        assert a.token_hash != b.token_hash

    def test_db_reopen_preserves_reviewers(self, tmp_path):
        path = tmp_path / "rev.sqlite3"
        s1 = ReviewerStore(db_path=path)
        a, token = s1.create(display_name="Alex", signs=["bsl"])
        s2 = ReviewerStore(db_path=path)
        # Auth still works against the re-opened DB.
        assert s2.authenticate(token).id == a.id


# ---------------------------------------------------------------------------
# Action functions
# ---------------------------------------------------------------------------


def _put_pending(
    sign_store: SQLiteSignStore,
    valid_entry_factory: Callable[..., SignEntry],
    **overrides,
) -> SignEntry:
    """Convenience: store a pending_review entry and return it."""
    e = valid_entry_factory(status="pending_review", **overrides)
    sign_store.put(e)
    return e


class TestApprove:
    def test_one_approval_keeps_entry_pending(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _b = native_pair
        out = approve(
            sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default
        )
        assert out.status == "pending_review"
        assert qualifying_approval_count(out, policy_default) == 1
        assert len(out.reviewers) == 1
        assert out.reviewers[0].verdict == "approved"

    def test_two_native_approvals_validate(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, b = native_pair
        approve(sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default)
        out = approve(
            sign_id=e.id, reviewer=b, store=sign_store, policy=policy_default
        )
        assert out.status == "validated"
        assert qualifying_approval_count(out, policy_default) == 2

    def test_same_reviewer_twice_does_not_satisfy_threshold(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _ = native_pair
        approve(sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default)
        out = approve(
            sign_id=e.id,
            reviewer=a,
            store=sign_store,
            policy=policy_default,
            comment="changed mind, still ok",
        )
        # Two records appended (history is permanent), but only one
        # qualifying approval — distinct reviewers required.
        assert len(out.reviewers) == 2
        assert qualifying_approval_count(out, policy_default) == 1
        assert out.status == "pending_review"

    def test_non_native_blocked_without_override(
        self,
        sign_store,
        valid_entry_factory,
        hearing_reviewer,
        policy_default,
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        with pytest.raises(NonNativeApprovalForbidden):
            approve(
                sign_id=e.id,
                reviewer=hearing_reviewer,
                store=sign_store,
                policy=policy_default,
            )

    def test_non_native_override_requires_justification(
        self,
        sign_store,
        valid_entry_factory,
        hearing_reviewer,
        policy_default,
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        with pytest.raises(ReviewActionError, match="justification"):
            approve(
                sign_id=e.id,
                reviewer=hearing_reviewer,
                store=sign_store,
                policy=policy_default,
                allow_non_native=True,
                justification="   ",
            )

    def test_non_native_override_with_justification_qualifies(
        self,
        sign_store,
        valid_entry_factory,
        hearing_reviewer,
        policy_default,
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        out = approve(
            sign_id=e.id,
            reviewer=hearing_reviewer,
            store=sign_store,
            policy=policy_default,
            allow_non_native=True,
            justification="bootstrap reviewer; native pool not yet onboarded",
        )
        assert qualifying_approval_count(out, policy_default) == 1

    def test_region_mismatch_does_not_qualify(
        self, sign_store, valid_entry_factory, reviewer_store, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        scot, _ = reviewer_store.create(
            display_name="Scot",
            is_deaf_native=True,
            signs=["bsl"],
            regional_background="BSL-Scotland",
        )
        out = approve(
            sign_id=e.id, reviewer=scot, store=sign_store, policy=policy_default
        )
        # The record is appended but doesn't count toward validation.
        assert len(out.reviewers) == 1
        assert qualifying_approval_count(out, policy_default) == 0
        assert out.status == "pending_review"

    def test_competence_check_rejects_wrong_language(
        self, sign_store, valid_entry_factory, reviewer_store, policy_default
    ):
        e = _put_pending(
            sign_store,
            valid_entry_factory,
            sign_language="bsl",
            regional_variant="BSL-London",
        )
        asl, _ = reviewer_store.create(
            display_name="Asher", is_deaf_native=True, signs=["asl"]
        )
        with pytest.raises(ReviewerNotCompetent):
            approve(
                sign_id=e.id, reviewer=asl, store=sign_store, policy=policy_default
            )

    def test_single_approval_mode_validates_with_one(
        self,
        sign_store,
        valid_entry_factory,
        native_pair,
        policy_single_approval,
        caplog,
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _ = native_pair
        with caplog.at_level("WARNING", logger="review.policy"):
            out = approve(
                sign_id=e.id,
                reviewer=a,
                store=sign_store,
                policy=policy_single_approval,
            )
        assert out.status == "validated"
        assert any(
            "single-approval bootstrap mode" in r.message for r in caplog.records
        ), "single-approval mode must log a warning each time"

    def test_terminal_states_block_approval(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _ = native_pair
        for terminal in ("validated", "rejected", "quarantined"):
            e.status = terminal
            sign_store.put(e)
            with pytest.raises(CannotActOnTerminalEntry):
                approve(
                    sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default
                )

    def test_draft_promoted_to_pending_on_first_approve(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = valid_entry_factory(status="draft", regional_variant="BSL-London")
        sign_store.put(e)
        a, _ = native_pair
        out = approve(
            sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default
        )
        assert out.status == "pending_review"


class TestReject:
    def test_reject_terminates_with_category(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _ = native_pair
        out = reject(
            sign_id=e.id,
            reviewer=a,
            store=sign_store,
            policy=policy_default,
            reason="movement size is wrong",
            category="inaccurate",
        )
        assert out.status == "rejected"
        assert out.reviewers[-1].category == "inaccurate"

    def test_reject_requires_non_empty_reason(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _ = native_pair
        with pytest.raises(ReviewActionError, match="reason"):
            reject(
                sign_id=e.id,
                reviewer=a,
                store=sign_store,
                policy=policy_default,
                reason="   ",
                category="inaccurate",
            )

    def test_reject_blocked_on_terminal(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        e.status = "rejected"
        sign_store.put(e)
        a, _ = native_pair
        with pytest.raises(CannotActOnTerminalEntry):
            reject(
                sign_id=e.id,
                reviewer=a,
                store=sign_store,
                policy=policy_default,
                reason="x",
                category="inaccurate",
            )


class TestRequestRevision:
    def test_returns_to_draft_with_history(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _ = native_pair
        out = request_revision(
            sign_id=e.id,
            reviewer=a,
            store=sign_store,
            policy=policy_default,
            comment="please specify the orientation",
            fields_to_revise=["orientation_palm"],
        )
        assert out.status == "draft"
        assert out.reviewers[-1].verdict == "changes_requested"
        assert out.reviewers[-1].fields_to_revise == ["orientation_palm"]

    def test_requires_comment(
        self, sign_store, valid_entry_factory, native_pair, policy_default
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, _ = native_pair
        with pytest.raises(ReviewActionError, match="comment"):
            request_revision(
                sign_id=e.id,
                reviewer=a,
                store=sign_store,
                policy=policy_default,
                comment="   ",
                fields_to_revise=[],
            )


class TestFlagAndQuarantine:
    def test_flag_quarantines_from_pending(
        self, sign_store, valid_entry_factory, native_pair
    ):
        e = _put_pending(sign_store, valid_entry_factory)
        a, _ = native_pair
        out = flag(
            sign_id=e.id,
            reviewer=a,
            store=sign_store,
            reason="cultural appropriation concern",
        )
        assert out.status == "quarantined"
        assert out.reviewers[-1].verdict == "flagged"

    def test_flag_works_on_validated(
        self,
        sign_store,
        valid_entry_factory,
        native_pair,
        policy_default,
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, b = native_pair
        approve(sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default)
        approve(sign_id=e.id, reviewer=b, store=sign_store, policy=policy_default)
        # Now validated — but flag still works.
        out = flag(
            sign_id=e.id,
            reviewer=a,
            store=sign_store,
            reason="post-publication concern raised by community member",
        )
        assert out.status == "quarantined"

    def test_flag_requires_reason(self, sign_store, valid_entry_factory, native_pair):
        e = _put_pending(sign_store, valid_entry_factory)
        a, _ = native_pair
        with pytest.raises(ReviewActionError, match="reason"):
            flag(sign_id=e.id, reviewer=a, store=sign_store, reason="  ")

    def test_clear_quarantine_board_only(
        self,
        sign_store,
        valid_entry_factory,
        native_pair,
        board_reviewer,
    ):
        e = _put_pending(sign_store, valid_entry_factory)
        a, _ = native_pair
        flag(sign_id=e.id, reviewer=a, store=sign_store, reason="concern")
        # Non-board reviewer rejected.
        with pytest.raises(ReviewActionError, match="board"):
            clear_quarantine(
                sign_id=e.id,
                reviewer=a,
                store=sign_store,
                target_status="pending_review",
                comment="lift",
            )
        # Board member can clear.
        board, _ = board_reviewer
        out = clear_quarantine(
            sign_id=e.id,
            reviewer=board,
            store=sign_store,
            target_status="pending_review",
            comment="resolved with reporter; sign re-enters review",
        )
        assert out.status == "pending_review"

    def test_clear_quarantine_only_from_quarantined(
        self,
        sign_store,
        valid_entry_factory,
        board_reviewer,
    ):
        e = _put_pending(sign_store, valid_entry_factory)
        board, _ = board_reviewer
        with pytest.raises(CannotActOnTerminalEntry):
            clear_quarantine(
                sign_id=e.id,
                reviewer=board,
                store=sign_store,
                target_status="pending_review",
                comment="x",
            )

    def test_clear_quarantine_target_must_be_legal(
        self,
        sign_store,
        valid_entry_factory,
        native_pair,
        board_reviewer,
    ):
        e = _put_pending(sign_store, valid_entry_factory)
        a, _ = native_pair
        flag(sign_id=e.id, reviewer=a, store=sign_store, reason="x")
        board, _ = board_reviewer
        with pytest.raises(ReviewActionError, match="target_status"):
            clear_quarantine(
                sign_id=e.id,
                reviewer=board,
                store=sign_store,
                target_status="validated",  # not allowed
                comment="x",
            )


# ---------------------------------------------------------------------------
# Export gate
# ---------------------------------------------------------------------------


class TestExportGate:
    def test_export_blocks_unvalidated(
        self, sign_store, valid_entry_factory, policy_default, audit_log
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        with pytest.raises(ExportNotAllowedError):
            sign_store.export_to_kozha_library(
                e.id, policy=policy_default, audit_log=audit_log
            )
        assert audit_log.read_all() == []

    def test_export_blocks_status_validated_with_no_approvals(
        self, sign_store, valid_entry_factory, policy_default, audit_log
    ):
        # Defense-in-depth: even if status is somehow set to validated
        # without going through the workflow, the gate refuses.
        e = valid_entry_factory(status="validated", regional_variant="BSL-London")
        sign_store.put(e)
        with pytest.raises(InsufficientApprovalsError) as ei:
            sign_store.export_to_kozha_library(
                e.id, policy=policy_default, audit_log=audit_log
            )
        # The error carries the gap so callers can show it to the user.
        assert "0 qualifying approval" in str(ei.value)
        assert audit_log.read_all() == []

    def test_export_succeeds_after_two_native_approvals(
        self,
        sign_store,
        valid_entry_factory,
        native_pair,
        policy_default,
        audit_log,
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, b = native_pair
        approve(sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default)
        approve(sign_id=e.id, reviewer=b, store=sign_store, policy=policy_default)
        # Now validated — export goes through.
        sign_store.export_to_kozha_library(
            e.id, policy=policy_default, audit_log=audit_log
        )
        target = sign_store.kozha_data_dir / "hamnosys_bsl_authored.sigml"
        assert target.exists()
        rows = audit_log.read_all()
        assert len(rows) == 1
        assert rows[0]["sign_id"] == str(e.id)
        assert set(rows[0]["reviewer_ids"]) == {str(a.id), str(b.id)}

    def test_export_re_export_appends_audit_row(
        self,
        sign_store,
        valid_entry_factory,
        native_pair,
        policy_default,
        audit_log,
    ):
        e = _put_pending(sign_store, valid_entry_factory, regional_variant="BSL-London")
        a, b = native_pair
        approve(sign_id=e.id, reviewer=a, store=sign_store, policy=policy_default)
        approve(sign_id=e.id, reviewer=b, store=sign_store, policy=policy_default)
        sign_store.export_to_kozha_library(
            e.id, policy=policy_default, audit_log=audit_log
        )
        sign_store.export_to_kozha_library(
            e.id, policy=policy_default, audit_log=audit_log
        )
        # Two audit rows — the SiGML file is idempotent but each
        # publication event is a separate fact worth recording.
        assert len(audit_log.read_all()) == 2


# ---------------------------------------------------------------------------
# Audit-log tamper-evidence
# ---------------------------------------------------------------------------


class TestAuditChain:
    def test_genesis_chain_verifies_clean(self, audit_log):
        ok, errors = audit_log.verify()
        assert ok and errors == []

    def test_chain_verifies_after_appends(self, audit_log):
        for i in range(3):
            audit_log.append(
                sign_id=uuid4(),
                reviewer_ids=[str(uuid4())],
                hamnosys=f"\uE001\uE040{i}",
                sigml=f"<sigml>{i}</sigml>",
                sign_language="bsl",
                gloss=f"GLOSS_{i}",
            )
        ok, errors = audit_log.verify()
        assert ok, errors

    def test_record_hash_tamper_detected(self, audit_log):
        audit_log.append(
            sign_id=uuid4(),
            reviewer_ids=[],
            hamnosys="\uE001",
            sigml="<sigml/>",
            sign_language="bsl",
            gloss="X",
        )
        # Mutate the row payload — the recomputed record_hash will differ.
        path = audit_log.log_path
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows[0]["gloss"] = "TAMPERED"
        path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
        )
        ok, errors = audit_log.verify()
        assert not ok
        assert any("record_hash mismatch" in e for e in errors)

    def test_chain_link_tamper_detected(self, audit_log):
        # Append two rows, then drop the middle link by editing prev_hash.
        for i in range(2):
            audit_log.append(
                sign_id=uuid4(),
                reviewer_ids=[],
                hamnosys=f"\uE001{i}",
                sigml="<sigml/>",
                sign_language="bsl",
                gloss=f"G{i}",
            )
        path = audit_log.log_path
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows[1]["prev_hash"] = "0" * 64  # break the chain
        path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
        )
        ok, errors = audit_log.verify()
        assert not ok
        # Both checks fail for row 2: record_hash recomputes off the
        # tampered prev_hash too. We just care that the chain check fires.
        assert any("prev_hash chain broken" in e for e in errors)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@pytest.fixture
def review_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up the chat2hamnosys app with isolated stores for review tests."""
    # Per-test sqlite + audit paths.
    monkeypatch.setenv(
        "CHAT2HAMNOSYS_REVIEWER_DB", str(tmp_path / "reviewers.sqlite3")
    )
    monkeypatch.setenv(
        "CHAT2HAMNOSYS_EXPORT_AUDIT", str(tmp_path / "exports.jsonl")
    )
    monkeypatch.setenv(
        "CHAT2HAMNOSYS_SESSION_DB", str(tmp_path / "sessions.sqlite3")
    )
    monkeypatch.setenv("CHAT2HAMNOSYS_SIGN_DB", str(tmp_path / "signs.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_TOKEN_DB", str(tmp_path / "tokens.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "kozha_data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_RATE_LIMIT", "500/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT", "500/minute")
    # Force production-like policy regardless of host env.
    monkeypatch.setenv("CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS", "2")
    monkeypatch.setenv("CHAT2HAMNOSYS_REVIEW_REQUIRE_NATIVE", "true")
    monkeypatch.setenv("CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE", "false")
    monkeypatch.setenv("CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH", "true")

    from api.dependencies import reset_stores
    from review.dependencies import reset_review_stores

    reset_stores()
    reset_review_stores()

    from api.router import limiter as _module_limiter

    _module_limiter.reset()

    from api import create_app

    app = create_app()
    tc = TestClient(app)
    try:
        yield tc, tmp_path
    finally:
        tc.close()
        reset_stores()
        reset_review_stores()


def _bootstrap_reviewer(tmp_path: Path, **kwargs) -> tuple[str, str]:
    """Create a reviewer directly via the store — returns (id, raw_token)."""
    from review.storage import ReviewerStore

    store = ReviewerStore(db_path=tmp_path / "reviewers.sqlite3")
    r, token = store.create(**kwargs)
    return str(r.id), token


def _put_sign(
    tmp_path: Path, valid_entry_factory: Callable[..., SignEntry], **overrides
) -> SignEntry:
    """Drop a SignEntry directly into the SQLite store the API will read."""
    store = SQLiteSignStore(
        db_path=tmp_path / "signs.sqlite3",
        kozha_data_dir=tmp_path / "kozha_data",
    )
    e = valid_entry_factory(**overrides)
    store.put(e)
    return e


class TestHttpLayer:
    def test_queue_requires_token(self, review_client):
        client, _ = review_client
        r = client.get("/review/queue")
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "reviewer_forbidden"

    def test_queue_lists_pending_signs_for_competent_reviewer(
        self, review_client, valid_entry_factory
    ):
        client, tmp_path = review_client
        _id, token = _bootstrap_reviewer(
            tmp_path,
            display_name="Alex",
            is_deaf_native=True,
            signs=["bsl"],
            regional_background="BSL-London",
        )
        e = _put_sign(
            tmp_path,
            valid_entry_factory,
            status="pending_review",
            regional_variant="BSL-London",
        )
        r = client.get("/review/queue", headers={"X-Reviewer-Token": token})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == 1
        assert body["items"][0]["id"] == str(e.id)
        assert body["items"][0]["min_approvals_required"] == 2

    def test_queue_filters_by_competence(self, review_client, valid_entry_factory):
        client, tmp_path = review_client
        # ASL reviewer; the sign is BSL → should be filtered out.
        _id, token = _bootstrap_reviewer(
            tmp_path, display_name="Asher", is_deaf_native=True, signs=["asl"]
        )
        _put_sign(
            tmp_path, valid_entry_factory, status="pending_review", sign_language="bsl"
        )
        r = client.get("/review/queue", headers={"X-Reviewer-Token": token})
        assert r.json()["count"] == 0

    def test_full_lifecycle_two_approvals_then_export(
        self, review_client, valid_entry_factory
    ):
        client, tmp_path = review_client
        _ai, ta = _bootstrap_reviewer(
            tmp_path,
            display_name="Alex",
            is_deaf_native=True,
            signs=["bsl"],
            regional_background="BSL-London",
        )
        _bi, tb = _bootstrap_reviewer(
            tmp_path,
            display_name="Beth",
            is_deaf_native=True,
            signs=["bsl"],
            regional_background="BSL-London",
        )
        # Board reviewer for the export call.
        _ci, tboard = _bootstrap_reviewer(
            tmp_path,
            display_name="Board",
            is_deaf_native=True,
            is_board=True,
            signs=["bsl"],
            regional_background="BSL-London",
        )

        e = _put_sign(
            tmp_path,
            valid_entry_factory,
            status="pending_review",
            regional_variant="BSL-London",
        )

        # First approval — still pending.
        r = client.post(
            f"/review/entries/{e.id}/approve",
            json={"comment": "looks correct"},
            headers={"X-Reviewer-Token": ta},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "pending_review"

        # Second approval — validated.
        r = client.post(
            f"/review/entries/{e.id}/approve",
            json={"comment": "agreed"},
            headers={"X-Reviewer-Token": tb},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "validated"

        # Non-board approver cannot trigger export.
        r = client.post(
            f"/review/entries/{e.id}/export", headers={"X-Reviewer-Token": ta}
        )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "board_only"

        # Board member can.
        r = client.post(
            f"/review/entries/{e.id}/export",
            headers={"X-Reviewer-Token": tboard},
        )
        assert r.status_code == 200, r.text
        assert r.json()["exported"] is True

        # Audit log written.
        audit_path = tmp_path / "exports.jsonl"
        assert audit_path.exists()
        rows = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
        assert len(rows) == 1
        assert rows[0]["sign_id"] == str(e.id)

    def test_single_approval_blocked_from_export(
        self, review_client, valid_entry_factory
    ):
        client, tmp_path = review_client
        _ai, ta = _bootstrap_reviewer(
            tmp_path,
            display_name="Alex",
            is_deaf_native=True,
            is_board=True,  # board so they can call export
            signs=["bsl"],
            regional_background="BSL-London",
        )
        e = _put_sign(
            tmp_path,
            valid_entry_factory,
            status="pending_review",
            regional_variant="BSL-London",
        )
        # One approval — not enough.
        client.post(
            f"/review/entries/{e.id}/approve",
            json={},
            headers={"X-Reviewer-Token": ta},
        )
        r = client.post(
            f"/review/entries/{e.id}/export", headers={"X-Reviewer-Token": ta}
        )
        # The status is still pending_review, so the gate trips on
        # status — not approval count.
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "export_not_allowed"

    def test_reject_terminates_via_http(self, review_client, valid_entry_factory):
        client, tmp_path = review_client
        _id, token = _bootstrap_reviewer(
            tmp_path,
            display_name="Alex",
            is_deaf_native=True,
            signs=["bsl"],
            regional_background="BSL-London",
        )
        e = _put_sign(
            tmp_path,
            valid_entry_factory,
            status="pending_review",
            regional_variant="BSL-London",
        )
        r = client.post(
            f"/review/entries/{e.id}/reject",
            json={"reason": "wrong handshape", "category": "inaccurate"},
            headers={"X-Reviewer-Token": token},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"

    def test_dashboard_includes_audit_chain_status(
        self, review_client, valid_entry_factory
    ):
        client, tmp_path = review_client
        _id, token = _bootstrap_reviewer(
            tmp_path, display_name="X", is_deaf_native=True, signs=["bsl"]
        )
        r = client.get("/review/dashboard", headers={"X-Reviewer-Token": token})
        assert r.status_code == 200
        body = r.json()
        assert body["exports"]["audit_chain_ok"] is True
        assert "policy" in body
        assert body["policy"]["min_approvals"] == 2

    def test_unknown_sign_returns_404(self, review_client):
        client, tmp_path = review_client
        _id, token = _bootstrap_reviewer(
            tmp_path, display_name="X", is_deaf_native=True, signs=["bsl"]
        )
        r = client.get(
            f"/review/entries/{uuid4()}", headers={"X-Reviewer-Token": token}
        )
        assert r.status_code == 404
