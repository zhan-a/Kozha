"""Shared fixtures for chat2hamnosys models / storage tests.

The ``valid_entry`` factory builds a real BSL sign (``abroad(a)#1`` from the
Hamburg dicta-sign portal) so each test starts from a record that already
passes HamNoSys 4.0 validation. Tests opt-out by passing kwargs to override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import pytest

from models import (
    AuthorInfo,
    MovementSegment,
    SignEntry,
    SignParameters,
)
from review.policy import ReviewPolicy
from storage import JSONFileSignStore, SignStore, SQLiteSignStore


# A sign known to validate (BSL "abroad(a)#1" — see hamnosys/tests fixtures).
ABROAD_HAMNOSYS = (
    "\uE001\uE00C\uE011\uE0E6\uE001\uE00C\uE020\uE03C\uE040\uE059\uE089\uE0C6\uE0D8"
)


@pytest.fixture
def valid_entry_factory() -> Callable[..., SignEntry]:
    """Return a factory that builds a fresh, schema-valid ``SignEntry``."""

    def _make(**overrides: Any) -> SignEntry:
        defaults: dict[str, Any] = dict(
            gloss="ABROAD",
            sign_language="bsl",
            domain="general",
            hamnosys=ABROAD_HAMNOSYS,
            description_prose="To travel to a place that is far away.",
            parameters=SignParameters(
                handshape_dominant="\uE001",            # hamflathand
                orientation_extended_finger="\uE020",   # hamextfingeru
                orientation_palm="\uE03C",              # hampalmd
                location="\uE040",                       # hamhead
                movement=[
                    MovementSegment(
                        path="\uE089",                   # hammoveo
                        size_mod="\uE0C6",               # hamsmallmod
                        repeat="\uE0D8",                 # hamrepeatfromstart
                    )
                ],
            ),
            author=AuthorInfo(signer_id="alice-001", is_deaf_native=True),
        )
        defaults.update(overrides)
        return SignEntry(**defaults)

    return _make


@pytest.fixture
def valid_entry(valid_entry_factory: Callable[..., SignEntry]) -> SignEntry:
    return valid_entry_factory()


@pytest.fixture
def permissive_review_policy() -> ReviewPolicy:
    """A policy that disables the export-gate approval check.

    Storage-layer tests exercise XML serialization, idempotency, and
    file-per-language separation — none of which care about the
    Deaf-reviewer approval count. Pass this policy to
    ``export_to_kozha_library`` so the defense-in-depth gate doesn't
    short-circuit the test before the code under test runs.
    """
    return ReviewPolicy(min_approvals=0, require_native_deaf=False)


@pytest.fixture(params=["json", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> SignStore:
    """Yield a freshly-initialized store of each backend type.

    ``kozha_data_dir`` is pinned to ``tmp_path`` so tests never touch the
    real ``data/`` directory.
    """
    if request.param == "json":
        return JSONFileSignStore(
            base_dir=tmp_path / "json_store",
            kozha_data_dir=tmp_path / "kozha_data",
        )
    return SQLiteSignStore(
        db_path=tmp_path / "store.sqlite3",
        kozha_data_dir=tmp_path / "kozha_data",
    )
