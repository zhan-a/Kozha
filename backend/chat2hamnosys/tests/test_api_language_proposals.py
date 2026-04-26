"""Tests for the inbound rare-SL language-proposal endpoint (Prompt 02).

Three angles:

1. Pydantic validation — the proposal body rejects empty names,
   over-length fields, and malformed ISO 639-3 codes.
2. Endpoint integration — happy path, honeypot, unknown-ISO triage
   flag, and the per-signer rate limit (the eleventh submission in
   the same minute window trips ``429 rate_limited``).
3. Maintainer queue — board reviewers can list / accept / reject /
   mark-duplicate proposals, and the accept response carries the
   ready-to-paste seed-file snippet.

Notes on rate limiting:
    The router-level limiter is keyed off the contributor token (or
    falls back to client IP). Tests that need to hit the per-day cap
    set both the minute and day envs to a generous value so they can
    submit ``N`` and assert ``N+1`` is the trip without contending
    with the production-default ``1/minute``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Fixtures — fresh app per test pointed at tmp SQLite paths
# ---------------------------------------------------------------------------


def _build_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    minute_limit: str = "100/minute",
    day_limit: str = "100/day",
) -> TestClient:
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_PROPOSALS_DB", str(tmp_path / "proposals.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_REVIEWER_DB", str(tmp_path / "reviewers.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_SIGNER_ID_SALT", "test-salt-proposals")
    monkeypatch.setenv("CHAT2HAMNOSYS_PROPOSALS_RATE_LIMIT_MINUTE", minute_limit)
    monkeypatch.setenv("CHAT2HAMNOSYS_PROPOSALS_RATE_LIMIT_DAY", day_limit)
    # The router's session-create cap kicks in at 2/minute by default —
    # bump it out of the way so anonymous proposals don't get tangled
    # up with session-creation accounting (different routes, but the
    # SlowAPI default applies).
    monkeypatch.setenv("CHAT2HAMNOSYS_RATE_LIMIT", "1000/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT", "1000/minute")

    from api.dependencies import reset_stores
    from api.proposals import reset_proposals_store
    from api.router import limiter as _module_limiter
    from review.dependencies import reset_review_stores

    reset_stores()
    reset_proposals_store()
    reset_review_stores()
    _module_limiter.reset()

    from api import create_app

    return TestClient(create_app())


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _build_client(tmp_path, monkeypatch)


@pytest.fixture
def board_reviewer_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Mint a board-reviewer token so admin-only endpoints can be exercised."""
    monkeypatch.setenv("CHAT2HAMNOSYS_REVIEWER_DB", str(tmp_path / "reviewers.sqlite3"))
    from review.storage import ReviewerStore

    store = ReviewerStore(db_path=tmp_path / "reviewers.sqlite3")
    _, token = store.create(
        display_name="Maintainer", is_deaf_native=True, is_board=True, signs=["bsl"]
    )
    return token


def _valid_payload(**overrides: object) -> dict:
    body = {
        "name": "Vietnamese Sign Language",
        "endonym": "Ngôn ngữ Ký hiệu Việt Nam",
        "iso_639_3": "vie",
        "region": "Vietnam",
        "corpus_url": "https://example.org/vsl-corpus",
        "submitter_is_deaf": True,
        "motivation": "VSL is widely used and not yet listed.",
        "website": "",
    }
    body.update(overrides)  # type: ignore[arg-type]
    return body


# ---------------------------------------------------------------------------
# 1. Pydantic validation
# ---------------------------------------------------------------------------


def test_pydantic_rejects_empty_name() -> None:
    from api.proposals import LanguageProposalIn

    with pytest.raises(ValidationError):
        LanguageProposalIn(**_valid_payload(name=""))
    with pytest.raises(ValidationError):
        LanguageProposalIn(**_valid_payload(name="   "))


def test_pydantic_rejects_overlong_name() -> None:
    from api.proposals import LanguageProposalIn, NAME_MAX_LEN

    with pytest.raises(ValidationError):
        LanguageProposalIn(**_valid_payload(name="x" * (NAME_MAX_LEN + 1)))


def test_pydantic_rejects_overlong_motivation() -> None:
    from api.proposals import LanguageProposalIn, MOTIVATION_MAX_LEN

    with pytest.raises(ValidationError):
        LanguageProposalIn(
            **_valid_payload(motivation="m" * (MOTIVATION_MAX_LEN + 1))
        )


def test_pydantic_rejects_empty_motivation() -> None:
    from api.proposals import LanguageProposalIn

    with pytest.raises(ValidationError):
        LanguageProposalIn(**_valid_payload(motivation=""))


@pytest.mark.parametrize("bad_iso", ["", "vi", "vies", "1ie", "VIE_X", "v i"])
def test_pydantic_rejects_malformed_iso(bad_iso: str) -> None:
    from api.proposals import LanguageProposalIn

    with pytest.raises(ValidationError):
        LanguageProposalIn(**_valid_payload(iso_639_3=bad_iso))


def test_pydantic_normalizes_iso_to_lowercase() -> None:
    from api.proposals import LanguageProposalIn

    body = LanguageProposalIn(**_valid_payload(iso_639_3="VIE"))
    assert body.iso_639_3 == "vie"


# ---------------------------------------------------------------------------
# 2. Endpoint integration
# ---------------------------------------------------------------------------


def test_post_proposal_happy_path(client: TestClient) -> None:
    resp = client.post("/language-proposals", json=_valid_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["proposal_id"]
    assert "Thanks" in body["message"]
    assert "Deaf reviewer" in body["message"]


def test_post_proposal_honeypot_tripped(client: TestClient) -> None:
    resp = client.post(
        "/language-proposals",
        json=_valid_payload(website="http://spam.example"),
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "proposal_rejected"


def test_post_proposal_unknown_iso_flagged_for_triage(
    client: TestClient,
    board_reviewer_token: str,
) -> None:
    # ``zzz`` is not in the spaCy/Argos roster — the row should land
    # with ``triage_unknown_iso=true`` so the maintainer can confirm
    # the language exists before seating it.
    resp = client.post("/language-proposals", json=_valid_payload(iso_639_3="zzz"))
    assert resp.status_code == 201, resp.text
    listing = client.get(
        "/admin/language-proposals",
        headers={"X-Reviewer-Token": board_reviewer_token},
    ).json()
    assert listing["proposals"][0]["triage_unknown_iso"] is True


def test_post_proposal_known_iso_does_not_flag_triage(
    client: TestClient,
    board_reviewer_token: str,
) -> None:
    resp = client.post("/language-proposals", json=_valid_payload())
    assert resp.status_code == 201
    listing = client.get(
        "/admin/language-proposals",
        headers={"X-Reviewer-Token": board_reviewer_token},
    ).json()
    assert listing["proposals"][0]["triage_unknown_iso"] is False


def test_rate_limit_trips_at_eleventh_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eleven submissions in the same minute → eleventh returns 429."""
    client = _build_client(
        tmp_path,
        monkeypatch,
        minute_limit="10/minute",
        day_limit="100/day",
    )
    for i in range(10):
        resp = client.post(
            "/language-proposals",
            json=_valid_payload(name=f"VSL #{i}"),
        )
        assert resp.status_code == 201, f"expected 201 on {i}: {resp.text}"

    # The eleventh one should be rate-limited.
    resp = client.post(
        "/language-proposals", json=_valid_payload(name="VSL #11")
    )
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "rate_limited"


def test_post_proposal_signer_id_is_hashed(
    client: TestClient,
    board_reviewer_token: str,
) -> None:
    resp = client.post("/language-proposals", json=_valid_payload())
    assert resp.status_code == 201
    listing = client.get(
        "/admin/language-proposals",
        headers={"X-Reviewer-Token": board_reviewer_token},
    ).json()
    # Hashed via :func:`security.hash_signer_id` → the literal "h:" prefix.
    signer = listing["proposals"][0]["submitter_signer_id"]
    assert signer.startswith("h:")
    # And the raw IP / signer-id is not leaked.
    assert "ip:" not in signer
    assert "127.0.0.1" not in signer


# ---------------------------------------------------------------------------
# 3. Admin queue
# ---------------------------------------------------------------------------


def test_admin_queue_requires_board_token(client: TestClient) -> None:
    resp = client.get("/admin/language-proposals")
    assert resp.status_code == 403


def test_admin_accept_emits_seed_snippet_and_does_not_modify_data_dir(
    client: TestClient,
    board_reviewer_token: str,
    tmp_path: Path,
) -> None:
    # Submit a proposal and grab its id.
    resp = client.post(
        "/language-proposals",
        json=_valid_payload(name="Vietnamese Sign Language", iso_639_3="vie"),
    )
    assert resp.status_code == 201
    proposal_id = resp.json()["proposal_id"]

    # Snapshot the data dir so we can verify accept does not modify it.
    data_dir = tmp_path / "data"
    before = sorted(p.name for p in data_dir.rglob("*")) if data_dir.exists() else []

    accept = client.post(
        f"/admin/language-proposals/{proposal_id}/accept",
        headers={"X-Reviewer-Token": board_reviewer_token},
        json={"notes": "looks good"},
    )
    assert accept.status_code == 200, accept.text
    body = accept.json()
    assert body["proposal"]["status"] == "accepted"
    assert body["proposal"]["notes"] == "looks good"
    snippet = body["seed_snippet"]
    # Mentions the canonical file convention from prompt 01.
    assert "Vietnamese_Sign_Language_SL_VIE.sigml" in snippet
    assert "contribute-languages.json" in snippet
    assert "vie" in snippet  # ISO code carried through
    assert "license" in snippet.lower()

    # And — most importantly — accepting did not touch the data dir.
    after = sorted(p.name for p in data_dir.rglob("*")) if data_dir.exists() else []
    assert before == after, "accept must not modify any file under data/"


def test_admin_reject_sets_status_and_carries_notes(
    client: TestClient, board_reviewer_token: str
) -> None:
    resp = client.post("/language-proposals", json=_valid_payload())
    proposal_id = resp.json()["proposal_id"]

    reject = client.post(
        f"/admin/language-proposals/{proposal_id}/reject",
        headers={"X-Reviewer-Token": board_reviewer_token},
        json={"notes": "duplicate of bsl variant — see comment"},
    )
    assert reject.status_code == 200, reject.text
    assert reject.json()["status"] == "rejected"
    assert reject.json()["notes"] == "duplicate of bsl variant — see comment"


def test_admin_mark_duplicate_sets_status(
    client: TestClient, board_reviewer_token: str
) -> None:
    resp = client.post("/language-proposals", json=_valid_payload())
    proposal_id = resp.json()["proposal_id"]

    dup = client.post(
        f"/admin/language-proposals/{proposal_id}/duplicate",
        headers={"X-Reviewer-Token": board_reviewer_token},
        json={"notes": "see existing entry"},
    )
    assert dup.status_code == 200
    assert dup.json()["status"] == "dup"


def test_admin_status_filter(
    client: TestClient, board_reviewer_token: str
) -> None:
    # Two pending proposals, one accepted.
    p1 = client.post(
        "/language-proposals", json=_valid_payload(name="VSL 1")
    ).json()["proposal_id"]
    client.post("/language-proposals", json=_valid_payload(name="VSL 2"))
    client.post(
        f"/admin/language-proposals/{p1}/accept",
        headers={"X-Reviewer-Token": board_reviewer_token},
        json={},
    )

    pending = client.get(
        "/admin/language-proposals?status=pending",
        headers={"X-Reviewer-Token": board_reviewer_token},
    ).json()
    accepted = client.get(
        "/admin/language-proposals?status=accepted",
        headers={"X-Reviewer-Token": board_reviewer_token},
    ).json()
    assert len(pending["proposals"]) == 1
    assert pending["proposals"][0]["name"] == "VSL 2"
    assert len(accepted["proposals"]) == 1
    assert accepted["proposals"][0]["name"] == "VSL 1"
