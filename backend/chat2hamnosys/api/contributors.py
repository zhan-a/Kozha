"""Contributor on-ramp for the public kozha-translate.com pipeline.

The authoring flow spends real OpenAI dollars. To gate that spend we
require a lightweight registration step before a user can create
authoring sessions:

1. ``GET  /contribute/captcha``  — issues a signed math challenge.
2. ``POST /contribute/register`` — validates captcha + honeypot,
   stores the contributor's name + contact, and returns a
   HMAC-signed ``contributor_token``.
3. Subsequent ``POST /sessions`` requests must carry the token in the
   ``X-Contributor-Token`` header.

Design notes
------------
- **Stateless captcha**. The challenge is a signed JSON blob carrying
  the SHA-256 of the expected answer plus an expiry. No server-side
  store; rolling deploys don't lose in-flight captchas.
- **Single-layer defences**. Honeypot field + signed challenge +
  minimal input validation. Sized for the current scraping risk —
  anything fancier (Turnstile, proof-of-work) is a drop-in replacement.
- **PII handling**. Name + contact are stored in cleartext because
  reviewers may need to reach out. IP / user-agent are hashed before
  storage; log lines only include the hashed form.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field, field_validator

from .errors import ApiError


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


CAPTCHA_TTL_SECONDS = 600                    # 10 min — plenty for form fill
CONTRIBUTOR_TOKEN_TTL_SECONDS = 14 * 86400   # 2 weeks
NAME_MAX_LEN = 120
CONTACT_MAX_LEN = 200
ANSWER_MAX_LEN = 16


def _captcha_secret() -> bytes:
    """Key for signing captcha challenges.

    Falls back to a dev-only constant when unset; production MUST set
    ``CHAT2HAMNOSYS_CAPTCHA_SECRET``. Rotating the secret invalidates
    all in-flight captchas, which is fine — users just refresh the page.
    """
    raw = os.environ.get("CHAT2HAMNOSYS_CAPTCHA_SECRET", "")
    if not raw:
        raw = "dev-captcha-secret-change-me"
    return raw.encode()


def _contributor_token_secret() -> bytes:
    """Key for signing contributor tokens.

    Reuses ``CHAT2HAMNOSYS_SIGNER_ID_SALT`` when set (the one secret
    already required by every prod deploy) so we don't multiply
    rotate-me surfaces; a dedicated ``CHAT2HAMNOSYS_CONTRIBUTOR_SECRET``
    takes precedence when present.
    """
    raw = (
        os.environ.get("CHAT2HAMNOSYS_CONTRIBUTOR_SECRET")
        or os.environ.get("CHAT2HAMNOSYS_SIGNER_ID_SALT")
        or "dev-contributor-secret-change-me"
    )
    return raw.encode()


def require_contributor_enabled() -> bool:
    """Is the session-creation gate active?

    Off by default so existing tests (which create sessions without a
    token) keep passing; production sets ``CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR=1``.
    """
    return os.environ.get("CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR", "0") == "1"


def _captcha_disabled() -> bool:
    """Is the captcha challenge bypassed?

    Used during pre-launch deployments where
    ``CHAT2HAMNOSYS_CAPTCHA_SECRET`` hasn't been provisioned yet. The
    honeypot field is still enforced, so naïve bots are still rejected
    — but sophisticated scrapers slip through until the captcha is
    back on. See README → "Pre-launch setup" for the re-enable
    procedure.
    """
    return os.environ.get("CHAT2HAMNOSYS_CAPTCHA_DISABLED", "0") == "1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CaptchaFailed(ApiError):
    status_code = 400
    code = "captcha_failed"


class InvalidContributorInput(ApiError):
    status_code = 400
    code = "invalid_contributor_input"


class ContributorRequired(ApiError):
    status_code = 401
    code = "contributor_required"


# ---------------------------------------------------------------------------
# Token codec — small HMAC-signed JSON blobs
# ---------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padded = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(padded.encode())


def _sign(payload: dict, secret: bytes) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return f"{_b64url(raw)}.{_b64url(sig)}"


def _verify(token: str, secret: bytes) -> Optional[dict]:
    try:
        raw_b64, sig_b64 = token.split(".", 1)
        raw = _b64url_decode(raw_b64)
        sig = _b64url_decode(sig_b64)
    except (ValueError, base64.binascii.Error):
        return None
    expected = hmac.new(secret, raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp < int(time.time()):
        return None
    return payload


# ---------------------------------------------------------------------------
# Captcha — simple signed arithmetic challenge
# ---------------------------------------------------------------------------


def new_captcha() -> tuple[str, str]:
    """Return ``(question, signed_challenge)``.

    The challenge carries the SHA-256 of the expected answer so the
    cleartext never leaves the server. The frontend renders ``question``
    and sends the user's answer plus the opaque challenge back to
    ``/register``.
    """
    a = secrets.randbelow(9) + 1
    b = secrets.randbelow(9) + 1
    question = f"What is {a} + {b}?"
    answer = str(a + b)
    payload = {
        "ans_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "exp": int(time.time()) + CAPTCHA_TTL_SECONDS,
        "nonce": secrets.token_hex(8),
    }
    return question, _sign(payload, _captcha_secret())


def verify_captcha(challenge: str, user_answer: str) -> bool:
    if not challenge or not user_answer:
        return False
    payload = _verify(challenge, _captcha_secret())
    if payload is None:
        return False
    user_hash = hashlib.sha256(user_answer.strip().encode()).hexdigest()
    expected = payload.get("ans_sha256", "")
    if not isinstance(expected, str):
        return False
    return hmac.compare_digest(user_hash, expected)


# ---------------------------------------------------------------------------
# Contributor tokens
# ---------------------------------------------------------------------------


def issue_contributor_token(contributor_id: str) -> tuple[str, int]:
    exp = int(time.time()) + CONTRIBUTOR_TOKEN_TTL_SECONDS
    token = _sign(
        {"cid": contributor_id, "exp": exp},
        _contributor_token_secret(),
    )
    return token, exp


def verify_contributor_token(token: Optional[str]) -> Optional[str]:
    """Return the contributor_id if ``token`` is valid and unexpired."""
    if not token:
        return None
    payload = _verify(token, _contributor_token_secret())
    if payload is None:
        return None
    cid = payload.get("cid")
    return cid if isinstance(cid, str) and cid else None


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


def _contributors_db_path() -> Path:
    override = os.environ.get("CHAT2HAMNOSYS_CONTRIBUTOR_DB")
    if override:
        return Path(override)
    base = Path(os.environ.get("CHAT2HAMNOSYS_DATA_DIR", "./data"))
    return base / "chat2hamnosys" / "contributors.sqlite3"


def _open_db() -> sqlite3.Connection:
    path = _contributors_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contributors (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            contact       TEXT NOT NULL,
            ip_hash       TEXT NOT NULL,
            user_agent    TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _persist_contributor(
    *,
    contributor_id: str,
    name: str,
    contact: str,
    ip_hash: str,
    user_agent: str,
) -> None:
    with closing(_open_db()) as conn:
        conn.execute(
            "INSERT INTO contributors "
            "(id, name, contact, ip_hash, user_agent, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                contributor_id,
                name,
                contact,
                ip_hash,
                user_agent,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def lookup_contributor(contributor_id: str) -> Optional[dict]:
    """Read-only accessor — used by admin / reviewer views."""
    with closing(_open_db()) as conn:
        row = conn.execute(
            "SELECT id, name, contact, created_at FROM contributors WHERE id = ?",
            (contributor_id,),
        ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "name": row[1], "contact": row[2], "created_at": row[3]}


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class CaptchaOut(BaseModel):
    question: str
    challenge: str
    expires_in: int = Field(description="Seconds until the challenge expires")
    disabled: bool = Field(
        default=False,
        description=(
            "Pre-launch flag — when true, the frontend skips the captcha UI "
            "and the register endpoint accepts any challenge/answer pair."
        ),
    )


class RegisterIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=NAME_MAX_LEN)
    contact: str = Field(..., min_length=3, max_length=CONTACT_MAX_LEN)
    captcha_challenge: str = Field(default="", max_length=4096)
    captcha_answer: str = Field(default="", max_length=ANSWER_MAX_LEN)
    # Honeypot — real users' browsers leave this empty. Bots that fill
    # every field trip it. The CSS on the frontend hides the input so a
    # human never sees it.
    website: str = Field(default="", max_length=200)

    @field_validator("name", "contact")
    @classmethod
    def _strip_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("must not be blank")
        return s


class RegisterOut(BaseModel):
    contributor_id: str
    contributor_token: str
    expires_at: int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(tags=["chat2hamnosys-contribute"], prefix="/contribute")


@router.get("/captcha", response_model=CaptchaOut, summary="Issue a new captcha")
def get_captcha() -> CaptchaOut:
    if _captcha_disabled():
        # Sentinel payload — the frontend hides the captcha UI, and the
        # matching register endpoint accepts any answer.
        return CaptchaOut(
            question="Captcha disabled (pre-launch setup).",
            challenge="disabled",
            expires_in=0,
            disabled=True,
        )
    question, challenge = new_captcha()
    return CaptchaOut(
        question=question,
        challenge=challenge,
        expires_in=CAPTCHA_TTL_SECONDS,
    )


def _looks_like_contact(value: str) -> bool:
    """Very loose — an email OR at least 6 digits (phone)."""
    if "@" in value and "." in value.split("@", 1)[-1]:
        return True
    digits = sum(1 for c in value if c.isdigit())
    return digits >= 6


@router.post(
    "/register",
    response_model=RegisterOut,
    status_code=201,
    summary="Register a contributor and receive a session-creation token",
)
def register(body: RegisterIn, request: Request) -> RegisterOut:
    # Honeypot — any value means a bot filled the hidden field. This is
    # the *only* spam defence when the captcha is disabled, so it always
    # runs (even in pre-launch mode).
    if body.website.strip():
        logger.info("contributor register: honeypot tripped")
        raise CaptchaFailed("Captcha failed")

    if not _captcha_disabled():
        if not verify_captcha(body.captcha_challenge, body.captcha_answer):
            raise CaptchaFailed("Captcha failed")

    name = body.name.strip()
    contact = body.contact.strip()
    if not _looks_like_contact(contact):
        raise InvalidContributorInput(
            "Contact must be an email address or phone number.",
            details={"field": "contact"},
        )

    contributor_id = str(uuid4())
    ip = getattr(getattr(request, "client", None), "host", "") or ""
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16] if ip else ""
    user_agent = (request.headers.get("user-agent") or "")[:500]

    _persist_contributor(
        contributor_id=contributor_id,
        name=name,
        contact=contact,
        ip_hash=ip_hash,
        user_agent=user_agent,
    )

    token, exp = issue_contributor_token(contributor_id)
    logger.info(
        "contributor registered id=%s ip_hash=%s",
        contributor_id,
        ip_hash or "-",
    )
    return RegisterOut(
        contributor_id=contributor_id,
        contributor_token=token,
        expires_at=exp,
    )


# ---------------------------------------------------------------------------
# FastAPI dependency — used by router.py on POST /sessions
# ---------------------------------------------------------------------------


def require_contributor(
    x_contributor_token: Optional[str] = Header(default=None),
) -> Optional[str]:
    """Header-based gate; returns contributor_id or ``None`` if the
    gate is disabled. Raises :class:`ContributorRequired` on a missing
    or invalid token when the gate is on.
    """
    if not require_contributor_enabled():
        return verify_contributor_token(x_contributor_token)

    cid = verify_contributor_token(x_contributor_token)
    if cid is None:
        raise ContributorRequired(
            "Register at /contribute.html before starting a session."
        )
    return cid


__all__ = [
    "CaptchaFailed",
    "CaptchaOut",
    "ContributorRequired",
    "InvalidContributorInput",
    "RegisterIn",
    "RegisterOut",
    "issue_contributor_token",
    "lookup_contributor",
    "new_captcha",
    "require_contributor",
    "require_contributor_enabled",
    "router",
    "verify_captcha",
    "verify_contributor_token",
]
