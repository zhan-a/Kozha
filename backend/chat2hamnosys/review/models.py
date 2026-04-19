"""Pydantic models for the Deaf reviewer workflow.

The :class:`Reviewer` record is the canonical reviewer identity — opaque
UUID + display name + self-attested Deaf-native flag + the sign
languages they're competent in + an optional regional background. The
``token_hash`` field stores the SHA-256 of the bearer token minted at
creation; the raw token is shown to the operator once and never
persisted.

Request / response shapes for the API live here too so the router stays
thin and OpenAPI is generated from a single source.

Authentication note
-------------------
The bearer-token scheme is **prototype-grade**. It exists so review
actions are non-trivially authorized — the reviewer must have a token
in their possession — but it does **not** offer revocation, scope
limiting, expiry, or rate limiting per principal. Production use
requires lifting reviewer auth onto the same SSO surface the rest of
the Kozha service eventually adopts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


SignLanguageLit = Literal["bsl", "asl", "dgs"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Reviewer(BaseModel):
    """A registered reviewer.

    ``token_hash`` is the SHA-256 hex digest of the raw bearer token
    minted at creation. We never round-trip the raw token through the
    DB; verification recomputes the digest and constant-time compares.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    display_name: str = Field(min_length=1, max_length=128)
    is_deaf_native: bool = False
    is_board: bool = False
    signs: list[SignLanguageLit] = Field(default_factory=list)
    regional_background: Optional[str] = None
    token_hash: str = Field(min_length=32)
    active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("display_name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("display_name must not be blank")
        return v

    @field_validator("signs")
    @classmethod
    def _dedup_signs(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for s in v:
            s = s.strip().lower()
            if s and s not in seen:
                seen.append(s)
        return seen

    def can_review(self, sign_language: str, regional_variant: Optional[str]) -> bool:
        """Is this reviewer competent on the given sign + region?

        A reviewer must have the sign language in their ``signs`` list.
        Regional background match is enforced by the policy layer (see
        :class:`backend.chat2hamnosys.review.policy.ReviewPolicy`); this
        method only enforces the language.
        """
        if not self.active:
            return False
        if sign_language.lower() not in self.signs:
            return False
        return True


class ReviewerPublic(BaseModel):
    """Reviewer record without the token hash — safe to return from APIs."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    display_name: str
    is_deaf_native: bool
    is_board: bool
    signs: list[str]
    regional_background: Optional[str] = None
    active: bool
    created_at: datetime

    @classmethod
    def from_reviewer(cls, r: Reviewer) -> "ReviewerPublic":
        return cls(
            id=r.id,
            display_name=r.display_name,
            is_deaf_native=r.is_deaf_native,
            is_board=r.is_board,
            signs=list(r.signs),
            regional_background=r.regional_background,
            active=r.active,
            created_at=r.created_at,
        )


# ---------------------------------------------------------------------------
# Request / response bodies for the review API
# ---------------------------------------------------------------------------


class ApproveRequest(BaseModel):
    """Body for ``POST /review/entries/{sign_id}/approve``."""

    model_config = ConfigDict(extra="forbid")

    comment: Optional[str] = None
    allow_non_native: bool = False
    justification: Optional[str] = None


class RejectRequest(BaseModel):
    """Body for ``POST /review/entries/{sign_id}/reject``."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    category: Literal[
        "inaccurate",
        "culturally_inappropriate",
        "regional_mismatch",
        "poor_quality",
        "other",
    ]


class RequestRevisionRequest(BaseModel):
    """Body for ``POST /review/entries/{sign_id}/request_revision``."""

    model_config = ConfigDict(extra="forbid")

    comment: str = Field(min_length=1)
    fields_to_revise: list[str] = Field(default_factory=list)


class FlagRequest(BaseModel):
    """Body for ``POST /review/entries/{sign_id}/flag``."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)


class ClearQuarantineRequest(BaseModel):
    """Body for ``POST /review/entries/{sign_id}/clear_quarantine``."""

    model_config = ConfigDict(extra="forbid")

    target_status: Literal["draft", "pending_review", "rejected"] = "pending_review"
    comment: str = Field(min_length=1)


class ReviewerCreateRequest(BaseModel):
    """Body for the admin CLI / programmatic reviewer creation."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1)
    is_deaf_native: bool = False
    is_board: bool = False
    signs: list[str] = Field(default_factory=list)
    regional_background: Optional[str] = None


class ReviewerCredentials(BaseModel):
    """Returned by the admin CLI on reviewer creation — token shown once."""

    model_config = ConfigDict(extra="forbid")

    reviewer: ReviewerPublic
    token: str = Field(min_length=16)


__all__ = [
    "ApproveRequest",
    "ClearQuarantineRequest",
    "FlagRequest",
    "RejectRequest",
    "RequestRevisionRequest",
    "Reviewer",
    "ReviewerCreateRequest",
    "ReviewerCredentials",
    "ReviewerPublic",
    "SignLanguageLit",
]
