"""Inbound rare-SL proposals — public submission + maintainer review queue.

Lets a contributor propose a sign language that isn't yet listed in
``public/contribute-languages.json``. Proposals land in a new SQLite
table next to the session store; a board reviewer accepts, rejects, or
marks them duplicate from the admin tab on ``/contribute.html``.

Acceptance does **not** modify the live language list or any file under
``data/`` — that step is explicit and human-driven (the admin endpoint
emits a ready-to-paste ``git``-friendly snippet so the maintainer can
seat the language by hand). The reasoning, in short: every corpus
needs a license claim and a Deaf reviewer; we never want the API to
auto-accept either.

Authentication / abuse defences mirror the rest of the chat2hamnosys
API:

- Honeypot field (``website``) — same shape as the contributor on-ramp.
- Captcha is currently disabled project-wide; this endpoint respects
  ``CHAT2HAMNOSYS_CAPTCHA_DISABLED`` rather than re-introducing one.
- Rate limit per signer-id hash to ``1/minute`` and ``10/day``. Authed
  contributors get keyed by their token; anonymous submissions fall
  back to client IP.
- Submitter signer-ids are hashed via the same
  ``CHAT2HAMNOSYS_SIGNER_ID_SALT`` as the rest of the system.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi.util import get_remote_address

from security import hash_signer_id

from .contributors import verify_contributor_token
from .errors import ApiError
from .router import limiter
from .security import get_security_config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROPOSALS_DB = _REPO_ROOT / "data" / "chat2hamnosys" / "language_proposals.sqlite3"

NAME_MAX_LEN = 200
ENDONYM_MAX_LEN = 200
REGION_MAX_LEN = 200
CORPUS_URL_MAX_LEN = 600
MOTIVATION_MAX_LEN = 1000
NOTES_MAX_LEN = 2000


# ISO 639-3 codes paired with the spaCy + Argos translation roster the
# project ships today. The list is intentionally short and conservative
# — the validator accepts any well-formed 3-letter code, and unknown
# codes set ``triage_unknown_iso=true`` on the stored row so a
# maintainer can confirm the language exists before seating it.
KNOWN_ISO_639_3: frozenset[str] = frozenset({
    "ara",  # Arabic
    "ben",  # Bengali
    "cmn",  # Mandarin Chinese
    "deu",  # German
    "ell",  # Greek
    "eng",  # English
    "fas",  # Persian
    "fra",  # French
    "hin",  # Hindi
    "ind",  # Indonesian
    "ita",  # Italian
    "jpn",  # Japanese
    "kaz",  # Kazakh
    "khk",  # Halh Mongolian
    "kor",  # Korean
    "kur",  # Kurdish (macro)
    "nld",  # Dutch
    "pol",  # Polish
    "por",  # Portuguese
    "rus",  # Russian
    "spa",  # Spanish
    "tha",  # Thai
    "tur",  # Turkish
    "ukr",  # Ukrainian
    "urd",  # Urdu
    "vie",  # Vietnamese
})


_ISO_RE = re.compile(r"^[a-z]{3}$")


def _proposals_rate_limit_minute() -> str:
    return os.environ.get(
        "CHAT2HAMNOSYS_PROPOSALS_RATE_LIMIT_MINUTE", "1/minute"
    )


def _proposals_rate_limit_day() -> str:
    return os.environ.get(
        "CHAT2HAMNOSYS_PROPOSALS_RATE_LIMIT_DAY", "10/day"
    )


def _proposals_db_path() -> Path:
    raw = os.environ.get("CHAT2HAMNOSYS_PROPOSALS_DB", "").strip()
    return Path(raw) if raw else DEFAULT_PROPOSALS_DB


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS language_proposals (
    id                     TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    endonym                TEXT,
    iso_639_3              TEXT NOT NULL,
    triage_unknown_iso     INTEGER NOT NULL DEFAULT 0,
    region                 TEXT,
    corpus_url             TEXT,
    submitter_is_deaf      INTEGER,
    motivation             TEXT NOT NULL,
    submitter_signer_id    TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'pending',
    notes                  TEXT,
    created_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON language_proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_signer ON language_proposals(submitter_signer_id);
"""


ProposalStatus = Literal["pending", "accepted", "rejected", "dup"]


class LanguageProposal(BaseModel):
    """One persisted row in ``language_proposals``.

    All free-text fields are sanitized at the request boundary; the
    submitter signer-id is already hashed by the time it lands here.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    name: str
    endonym: Optional[str] = None
    iso_639_3: str
    triage_unknown_iso: bool = False
    region: Optional[str] = None
    corpus_url: Optional[str] = None
    submitter_is_deaf: Optional[bool] = None
    motivation: str
    submitter_signer_id: str
    status: ProposalStatus = "pending"
    notes: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ProposalsStore:
    """SQLite-backed repository for :class:`LanguageProposal` rows.

    Mirrors :class:`session.SessionStore` — same idempotent schema
    creation, same default thread affinity. Multi-process access is not
    a goal (the admin tab is single-operator).
    """

    def __init__(self, *, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_proposal(row: sqlite3.Row) -> LanguageProposal:
        return LanguageProposal(
            id=UUID(row["id"]),
            name=row["name"],
            endonym=row["endonym"],
            iso_639_3=row["iso_639_3"],
            triage_unknown_iso=bool(row["triage_unknown_iso"]),
            region=row["region"],
            corpus_url=row["corpus_url"],
            submitter_is_deaf=(
                None
                if row["submitter_is_deaf"] is None
                else bool(row["submitter_is_deaf"])
            ),
            motivation=row["motivation"],
            submitter_signer_id=row["submitter_signer_id"],
            status=row["status"],
            notes=row["notes"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def insert(self, proposal: LanguageProposal) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO language_proposals (
                    id, name, endonym, iso_639_3, triage_unknown_iso,
                    region, corpus_url, submitter_is_deaf, motivation,
                    submitter_signer_id, status, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(proposal.id),
                    proposal.name,
                    proposal.endonym,
                    proposal.iso_639_3,
                    1 if proposal.triage_unknown_iso else 0,
                    proposal.region,
                    proposal.corpus_url,
                    None
                    if proposal.submitter_is_deaf is None
                    else (1 if proposal.submitter_is_deaf else 0),
                    proposal.motivation,
                    proposal.submitter_signer_id,
                    proposal.status,
                    proposal.notes,
                    proposal.created_at.isoformat(),
                ),
            )

    def get(self, proposal_id: UUID) -> Optional[LanguageProposal]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM language_proposals WHERE id = ?",
                (str(proposal_id),),
            ).fetchone()
        return self._row_to_proposal(row) if row else None

    def list(
        self,
        *,
        status: Optional[ProposalStatus] = None,
        limit: int = 200,
    ) -> list[LanguageProposal]:
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM language_proposals "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM language_proposals WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def update_status(
        self,
        proposal_id: UUID,
        *,
        status: ProposalStatus,
        notes: Optional[str] = None,
    ) -> Optional[LanguageProposal]:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE language_proposals SET status = ?, notes = ? "
                "WHERE id = ?",
                (status, notes, str(proposal_id)),
            )
            if cur.rowcount == 0:
                return None
        return self.get(proposal_id)


_proposals_store: Optional[ProposalsStore] = None


def get_proposals_store() -> ProposalsStore:
    """Process-wide :class:`ProposalsStore` at the configured path."""
    global _proposals_store
    if _proposals_store is None:
        _proposals_store = ProposalsStore(db_path=_proposals_db_path())
    return _proposals_store


def reset_proposals_store() -> None:
    """Test helper — clears the singleton so the next call rebinds."""
    global _proposals_store
    _proposals_store = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProposalRejected(ApiError):
    """Honeypot tripped — handled like a captcha failure."""

    status_code = 400
    code = "proposal_rejected"


class ProposalNotFound(ApiError):
    status_code = 404
    code = "proposal_not_found"


class ProposalForbidden(ApiError):
    status_code = 403
    code = "proposal_forbidden"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LanguageProposalIn(BaseModel):
    """Public submission body for ``POST /language-proposals``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=NAME_MAX_LEN)
    endonym: str = Field(default="", max_length=ENDONYM_MAX_LEN)
    iso_639_3: str = Field(..., min_length=3, max_length=3)
    region: str = Field(default="", max_length=REGION_MAX_LEN)
    corpus_url: str = Field(default="", max_length=CORPUS_URL_MAX_LEN)
    submitter_is_deaf: Optional[bool] = None
    motivation: str = Field(..., min_length=1, max_length=MOTIVATION_MAX_LEN)
    # Honeypot — same shape as ``RegisterIn``. Real users' browsers
    # leave this empty; bots that fill every field trip it.
    website: str = Field(default="", max_length=200)

    @field_validator("name", "motivation")
    @classmethod
    def _strip_required(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("must not be blank")
        return s

    @field_validator("endonym", "region", "corpus_url")
    @classmethod
    def _strip_optional(cls, v: str) -> str:
        return v.strip()

    @field_validator("iso_639_3")
    @classmethod
    def _normalize_iso(cls, v: str) -> str:
        s = v.strip().lower()
        if not _ISO_RE.match(s):
            raise ValueError(
                "iso_639_3 must be a 3-letter lowercase ISO 639-3 code"
            )
        return s


class LanguageProposalOut(BaseModel):
    """Admin-side view of one stored proposal."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    endonym: Optional[str] = None
    iso_639_3: str
    triage_unknown_iso: bool
    region: Optional[str] = None
    corpus_url: Optional[str] = None
    submitter_is_deaf: Optional[bool] = None
    motivation: str
    submitter_signer_id: str
    status: ProposalStatus
    notes: Optional[str] = None
    created_at: datetime

    @classmethod
    def from_model(cls, p: LanguageProposal) -> "LanguageProposalOut":
        return cls(
            id=p.id,
            name=p.name,
            endonym=p.endonym,
            iso_639_3=p.iso_639_3,
            triage_unknown_iso=p.triage_unknown_iso,
            region=p.region,
            corpus_url=p.corpus_url,
            submitter_is_deaf=p.submitter_is_deaf,
            motivation=p.motivation,
            submitter_signer_id=p.submitter_signer_id,
            status=p.status,
            notes=p.notes,
            created_at=p.created_at,
        )


class ProposalAcknowledgement(BaseModel):
    """Body of ``POST /language-proposals``.

    The frontend uses ``message`` verbatim; that wording is the
    plain-language register the rest of the contribute flow uses.
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: UUID
    status: ProposalStatus = "pending"
    message: str


class AdminUpdateBody(BaseModel):
    """Body for the admin accept / reject / duplicate endpoints."""

    model_config = ConfigDict(extra="forbid")

    notes: str = Field(default="", max_length=NOTES_MAX_LEN)


class AdminProposalListOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[LanguageProposalOut]


class AdminAcceptOut(BaseModel):
    """Response body for the accept endpoint.

    Carries the seed-file snippet the maintainer pastes to seat the
    language. The endpoint does **not** modify any file under
    ``data/`` — it just flips the status and renders the snippet.
    """

    model_config = ConfigDict(extra="forbid")

    proposal: LanguageProposalOut
    seed_snippet: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_submitter(raw: str) -> str:
    """Salted hash of the submitter's signer-id."""
    cfg = get_security_config()
    return hash_signer_id(raw or "anonymous", salt=cfg.signer_id_salt)


def _resolve_signer_key(request: Request) -> str:
    """Pick the rate-limit / dedup key for one submission.

    Authenticated contributors have a token in their headers; resolve
    it to the contributor_id so two anonymous tabs from the same IP
    are tracked together. Falls back to client IP when no token is
    present (matches the rest of the API).
    """
    token = request.headers.get("X-Contributor-Token")
    cid = verify_contributor_token(token) if token else None
    if cid:
        return f"cid:{cid}"
    return f"ip:{get_remote_address(request)}"


def _slug_from_name(name: str) -> str:
    """Best-effort filename slug for the seed snippet."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return cleaned or "Sign_Language"


def _build_seed_snippet(proposal: LanguageProposal) -> str:
    """One-paragraph maintainer cheat-sheet — paths, filenames, JSON.

    Mirrors the file convention from prompt 01: a SiGML seed under
    ``data/<slug>_SL_<CODE>.sigml`` plus a paired ``.meta.json``.
    Codes default to the ISO 639-3 base; the maintainer is expected to
    correct it to the canonical 3-letter SL code (e.g. ``KSL``,
    ``RSL``) before committing.
    """
    slug = _slug_from_name(proposal.name)
    code_guess = proposal.iso_639_3.upper()
    sigml_path = f"data/{slug}_SL_{code_guess}.sigml"
    meta_path = f"{sigml_path}.meta.json"
    quarantine = f"data/{slug}_SL_{code_guess}_quarantine.sigml"

    lines = [
        "Suggested seed-file layout (verify license + reviewer-pool first):",
        f"  - {sigml_path}",
        f"  - {meta_path}",
        f"  - {quarantine}   # empty placeholder until quarantine lands",
        "",
        "Suggested public/contribute-languages.json entry:",
        '  {',
        f'    "code":      "{proposal.iso_639_3.lower()}",',
        f'    "name":      "{proposal.endonym or proposal.name}",',
        f'    "english_name": "{proposal.name}",',
        '    "group":     "rare",',
        '    "has_reviewers": false,',
        '    "coverage_count": 0,',
        '    "data_completeness": "seed",',
        '    "accepts_first_contributions": true',
        '  }',
        "",
        "Reminder: do not commit until a license claim and a Deaf",
        "reviewer for this language are confirmed.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(tags=["chat2hamnosys-proposals"])


# ``key_func=_resolve_signer_key`` overrides the module-level limiter's
# default ``get_remote_address`` so an authenticated contributor's
# limit follows their token across IPs (and conversely, an anonymous
# submission still falls back to the source IP). Stacking two
# ``@limiter.limit`` decorators applies both rates.
@router.post(
    "/language-proposals",
    response_model=ProposalAcknowledgement,
    status_code=201,
    summary="Submit a sign-language suggestion for maintainer review",
)
@limiter.limit(_proposals_rate_limit_minute, key_func=_resolve_signer_key)
@limiter.limit(_proposals_rate_limit_day, key_func=_resolve_signer_key)
def post_language_proposal(
    request: Request,
    body: LanguageProposalIn,
    store: ProposalsStore = Depends(get_proposals_store),
) -> ProposalAcknowledgement:
    if body.website.strip():
        # Honeypot — same response shape as captcha so a scraper can't
        # tell which field tripped it. The honeypot still runs even
        # when the captcha is disabled (matches RegisterIn).
        logger.info("language proposal: honeypot tripped")
        raise ProposalRejected("Submission rejected.")

    signer_key = _resolve_signer_key(request)
    submitter_signer_id = _hash_submitter(signer_key)

    triage = body.iso_639_3 not in KNOWN_ISO_639_3
    proposal = LanguageProposal(
        name=body.name,
        endonym=body.endonym or None,
        iso_639_3=body.iso_639_3,
        triage_unknown_iso=triage,
        region=body.region or None,
        corpus_url=body.corpus_url or None,
        submitter_is_deaf=body.submitter_is_deaf,
        motivation=body.motivation,
        submitter_signer_id=submitter_signer_id,
    )
    store.insert(proposal)
    logger.info(
        "language proposal stored id=%s iso=%s triage=%s",
        proposal.id,
        proposal.iso_639_3,
        triage,
    )

    return ProposalAcknowledgement(
        proposal_id=proposal.id,
        status="pending",
        message=(
            "Thanks — a maintainer will review your suggestion. "
            "We do not auto-add languages because every corpus needs "
            "a license and a Deaf reviewer."
        ),
    )


# ---------------------------------------------------------------------------
# Admin queue — gated behind board-reviewer auth
# ---------------------------------------------------------------------------


def _require_board(x_reviewer_token: Optional[str]):
    # Lazy-import so this module is safe to import before the review
    # dependency singletons are primed (mirrors api/admin.py).
    from .admin import require_board_reviewer

    return require_board_reviewer(x_reviewer_token=x_reviewer_token)


@router.get(
    "/admin/language-proposals",
    response_model=AdminProposalListOut,
    summary="List language proposals (board reviewers only)",
)
def get_admin_proposals(
    status: Optional[str] = None,
    x_reviewer_token: Optional[str] = Header(default=None),
    store: ProposalsStore = Depends(get_proposals_store),
) -> AdminProposalListOut:
    _require_board(x_reviewer_token)
    status_filter: Optional[ProposalStatus] = None
    if status:
        if status not in ("pending", "accepted", "rejected", "dup"):
            raise ApiError(
                f"unknown status {status!r}",
                status_code=400,
                code="invalid_status",
            )
        status_filter = status  # type: ignore[assignment]
    rows = store.list(status=status_filter)
    return AdminProposalListOut(
        proposals=[LanguageProposalOut.from_model(p) for p in rows]
    )


def _admin_update(
    *,
    proposal_id: UUID,
    new_status: ProposalStatus,
    notes: Optional[str],
    store: ProposalsStore,
) -> LanguageProposal:
    proposal = store.update_status(
        proposal_id, status=new_status, notes=notes or None
    )
    if proposal is None:
        raise ProposalNotFound(f"proposal {proposal_id} not found")
    return proposal


@router.post(
    "/admin/language-proposals/{proposal_id}/accept",
    response_model=AdminAcceptOut,
    summary="Accept a proposal (does NOT seat the language; emits a snippet)",
)
def post_admin_accept(
    proposal_id: UUID,
    body: Optional[AdminUpdateBody] = None,
    x_reviewer_token: Optional[str] = Header(default=None),
    store: ProposalsStore = Depends(get_proposals_store),
) -> AdminAcceptOut:
    _require_board(x_reviewer_token)
    notes = (body.notes if body else "").strip() or None
    proposal = _admin_update(
        proposal_id=proposal_id,
        new_status="accepted",
        notes=notes,
        store=store,
    )
    snippet = _build_seed_snippet(proposal)
    return AdminAcceptOut(
        proposal=LanguageProposalOut.from_model(proposal),
        seed_snippet=snippet,
    )


@router.post(
    "/admin/language-proposals/{proposal_id}/reject",
    response_model=LanguageProposalOut,
    summary="Reject a proposal",
)
def post_admin_reject(
    proposal_id: UUID,
    body: Optional[AdminUpdateBody] = None,
    x_reviewer_token: Optional[str] = Header(default=None),
    store: ProposalsStore = Depends(get_proposals_store),
) -> LanguageProposalOut:
    _require_board(x_reviewer_token)
    notes = (body.notes if body else "").strip() or None
    proposal = _admin_update(
        proposal_id=proposal_id,
        new_status="rejected",
        notes=notes,
        store=store,
    )
    return LanguageProposalOut.from_model(proposal)


@router.post(
    "/admin/language-proposals/{proposal_id}/duplicate",
    response_model=LanguageProposalOut,
    summary="Mark a proposal as a duplicate of an existing entry",
)
def post_admin_duplicate(
    proposal_id: UUID,
    body: Optional[AdminUpdateBody] = None,
    x_reviewer_token: Optional[str] = Header(default=None),
    store: ProposalsStore = Depends(get_proposals_store),
) -> LanguageProposalOut:
    _require_board(x_reviewer_token)
    notes = (body.notes if body else "").strip() or None
    proposal = _admin_update(
        proposal_id=proposal_id,
        new_status="dup",
        notes=notes,
        store=store,
    )
    return LanguageProposalOut.from_model(proposal)


def all_iso_codes() -> Iterable[str]:
    """Helper for tests/CLI — exposed so callers don't import the constant."""
    return iter(KNOWN_ISO_639_3)


__all__ = [
    "AdminAcceptOut",
    "AdminProposalListOut",
    "AdminUpdateBody",
    "DEFAULT_PROPOSALS_DB",
    "KNOWN_ISO_639_3",
    "LanguageProposal",
    "LanguageProposalIn",
    "LanguageProposalOut",
    "ProposalAcknowledgement",
    "ProposalForbidden",
    "ProposalNotFound",
    "ProposalRejected",
    "ProposalStatus",
    "ProposalsStore",
    "all_iso_codes",
    "get_proposals_store",
    "reset_proposals_store",
    "router",
]
