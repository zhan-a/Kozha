"""Tests for the storage backends and the Kozha library export path.

The ``store`` fixture in ``conftest.py`` parameterises every contract test
across both the JSON-file and SQLite backends, so the same suite proves
they implement the abstract interface identically.

SQLite-specific tests live at the bottom — they verify idempotency claims
that don't apply to the file backend (re-init of the same DB, ON CONFLICT
upsert, denormalized index columns staying in sync after status changes).
"""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable
from uuid import uuid4

import pytest

from models import SignEntry
from storage import (
    DEFAULT_KOZHA_DATA_DIR,
    ExportNotAllowedError,
    JSONFileSignStore,
    SignNotFoundError,
    SignStore,
    SQLiteSignStore,
    StorageError,
)


# ---------------------------------------------------------------------------
# Contract: CRUD on both backends
# ---------------------------------------------------------------------------


def test_put_then_get_round_trips(store: SignStore, valid_entry: SignEntry) -> None:
    store.put(valid_entry)
    fetched = store.get(valid_entry.id)
    assert fetched.id == valid_entry.id
    assert fetched.gloss == valid_entry.gloss
    assert fetched.hamnosys == valid_entry.hamnosys
    assert fetched.parameters == valid_entry.parameters


def test_get_missing_raises_sign_not_found(store: SignStore) -> None:
    with pytest.raises(SignNotFoundError):
        store.get(uuid4())


def test_list_returns_all_entries(
    store: SignStore, valid_entry_factory: Callable[..., SignEntry]
) -> None:
    a = valid_entry_factory(gloss="ABROAD")
    b = valid_entry_factory(gloss="ELECTRON")
    c = valid_entry_factory(gloss="ORBITAL")
    for e in (a, b, c):
        store.put(e)
    glosses = {e.gloss for e in store.list()}
    assert glosses == {"ABROAD", "ELECTRON", "ORBITAL"}


def test_list_empty_store_returns_empty_list(store: SignStore) -> None:
    assert store.list() == []


def test_update_status_persists(store: SignStore, valid_entry: SignEntry) -> None:
    store.put(valid_entry)
    store.update_status(valid_entry.id, "validated")
    assert store.get(valid_entry.id).status == "validated"


def test_search_by_gloss_is_case_insensitive(
    store: SignStore, valid_entry_factory: Callable[..., SignEntry]
) -> None:
    e = valid_entry_factory(gloss="ELECTRON")
    store.put(e)
    assert [x.id for x in store.search_by_gloss("electron")] == [e.id]
    assert [x.id for x in store.search_by_gloss("Electron")] == [e.id]
    assert store.search_by_gloss("notthere") == []


def test_search_by_gloss_returns_all_matches(
    store: SignStore, valid_entry_factory: Callable[..., SignEntry]
) -> None:
    a = valid_entry_factory(gloss="ELECTRON", regional_variant="BSL-London")
    b = valid_entry_factory(gloss="ELECTRON", regional_variant="BSL-Scotland")
    store.put(a)
    store.put(b)
    matches = {x.id for x in store.search_by_gloss("electron")}
    assert matches == {a.id, b.id}


def test_put_is_idempotent(store: SignStore, valid_entry: SignEntry) -> None:
    store.put(valid_entry)
    store.put(valid_entry)
    store.put(valid_entry)
    assert len(store.list()) == 1


# ---------------------------------------------------------------------------
# Contract: status-gated export
# ---------------------------------------------------------------------------


def _read_sigml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


@pytest.mark.parametrize("bad_status", ["draft", "pending_review", "rejected"])
def test_export_refuses_non_validated_status(
    store: SignStore,
    valid_entry_factory: Callable[..., SignEntry],
    bad_status: str,
) -> None:
    e = valid_entry_factory(status=bad_status)
    store.put(e)
    with pytest.raises(ExportNotAllowedError):
        store.export_to_kozha_library(e.id)
    assert not (store.kozha_data_dir / f"hamnosys_{e.sign_language}_authored.sigml").exists()


def test_export_validated_writes_canonical_sigml(
    store: SignStore, valid_entry: SignEntry
) -> None:
    valid_entry.status = "validated"
    store.put(valid_entry)
    store.export_to_kozha_library(valid_entry.id)

    target = store.kozha_data_dir / "hamnosys_bsl_authored.sigml"
    assert target.exists()

    root = _read_sigml(target)
    assert root.tag == "sigml"
    signs = list(root)
    assert len(signs) == 1
    sign = signs[0]
    assert sign.tag == "hns_sign"
    # Frontend lookup is lowercased — exporter must follow suit.
    assert sign.get("gloss") == "abroad"

    # Must contain both the nonmanual and manual sections.
    children = [c.tag for c in sign]
    assert "hamnosys_nonmanual" in children
    assert "hamnosys_manual" in children

    manual = sign.find("hamnosys_manual")
    assert manual is not None
    tag_names = [c.tag for c in manual]
    # Spot-check a few — every PUA codepoint becomes one tag.
    assert tag_names[0] == "hamflathand"      # U+E001
    assert tag_names[1] == "hamthumboutmod"   # U+E00C
    assert "hamhead" in tag_names              # U+E040
    assert "hamrepeatfromstart" in tag_names   # U+E0D8
    assert len(tag_names) == len(valid_entry.hamnosys)


def test_export_emits_mouth_picture_when_present(
    store: SignStore, valid_entry: SignEntry
) -> None:
    from models import NonManualFeatures

    valid_entry.parameters.non_manual = NonManualFeatures(mouth_picture="V")
    valid_entry.status = "validated"
    store.put(valid_entry)
    store.export_to_kozha_library(valid_entry.id)

    root = _read_sigml(store.kozha_data_dir / "hamnosys_bsl_authored.sigml")
    mouth = root.find(".//hnm_mouthpicture")
    assert mouth is not None
    assert mouth.get("picture") == "V"


def test_export_is_idempotent_no_duplicates(
    store: SignStore, valid_entry: SignEntry
) -> None:
    valid_entry.status = "validated"
    store.put(valid_entry)
    store.export_to_kozha_library(valid_entry.id)
    first = (store.kozha_data_dir / "hamnosys_bsl_authored.sigml").read_text()
    store.export_to_kozha_library(valid_entry.id)
    store.export_to_kozha_library(valid_entry.id)
    second = (store.kozha_data_dir / "hamnosys_bsl_authored.sigml").read_text()
    assert first == second
    root = _read_sigml(store.kozha_data_dir / "hamnosys_bsl_authored.sigml")
    assert len(list(root)) == 1


def test_export_replaces_same_gloss_in_place(
    store: SignStore, valid_entry_factory: Callable[..., SignEntry]
) -> None:
    a = valid_entry_factory(gloss="ABROAD", status="validated")
    b = valid_entry_factory(gloss="ABROAD", status="validated")  # different uuid, same gloss
    assert a.id != b.id
    store.put(a)
    store.put(b)
    store.export_to_kozha_library(a.id)
    store.export_to_kozha_library(b.id)
    root = _read_sigml(store.kozha_data_dir / "hamnosys_bsl_authored.sigml")
    glosses = [c.get("gloss") for c in root]
    assert glosses == ["abroad"]


def test_export_appends_distinct_glosses(
    store: SignStore, valid_entry_factory: Callable[..., SignEntry]
) -> None:
    a = valid_entry_factory(gloss="ABROAD", status="validated")
    b = valid_entry_factory(gloss="ELECTRON", status="validated")
    store.put(a)
    store.put(b)
    store.export_to_kozha_library(a.id)
    store.export_to_kozha_library(b.id)
    root = _read_sigml(store.kozha_data_dir / "hamnosys_bsl_authored.sigml")
    glosses = sorted(c.get("gloss") for c in root)
    assert glosses == ["abroad", "electron"]


def test_export_separates_files_per_language(
    store: SignStore, valid_entry_factory: Callable[..., SignEntry]
) -> None:
    bsl = valid_entry_factory(gloss="ABROAD", sign_language="bsl", status="validated")
    asl = valid_entry_factory(gloss="ABROAD", sign_language="asl", status="validated")
    store.put(bsl)
    store.put(asl)
    store.export_to_kozha_library(bsl.id)
    store.export_to_kozha_library(asl.id)
    assert (store.kozha_data_dir / "hamnosys_bsl_authored.sigml").exists()
    assert (store.kozha_data_dir / "hamnosys_asl_authored.sigml").exists()


def test_export_missing_sign_raises(store: SignStore) -> None:
    with pytest.raises(SignNotFoundError):
        store.export_to_kozha_library(uuid4())


def test_export_rejects_corrupt_existing_file(
    store: SignStore, valid_entry: SignEntry
) -> None:
    store.kozha_data_dir.mkdir(parents=True, exist_ok=True)
    (store.kozha_data_dir / "hamnosys_bsl_authored.sigml").write_text(
        "this is not xml at all", encoding="utf-8"
    )
    valid_entry.status = "validated"
    store.put(valid_entry)
    with pytest.raises(StorageError, match="malformed"):
        store.export_to_kozha_library(valid_entry.id)


# ---------------------------------------------------------------------------
# SQLite-specific idempotency / persistence
# ---------------------------------------------------------------------------


def test_sqlite_reopen_preserves_entries(
    tmp_path: Path, valid_entry: SignEntry
) -> None:
    db = tmp_path / "store.sqlite3"
    s1 = SQLiteSignStore(db_path=db, kozha_data_dir=tmp_path / "kozha")
    s1.put(valid_entry)

    # Re-open the same DB file: the schema migration is IF NOT EXISTS, so
    # this must not blow up and must surface the previously-stored entry.
    s2 = SQLiteSignStore(db_path=db, kozha_data_dir=tmp_path / "kozha")
    fetched = s2.get(valid_entry.id)
    assert fetched.id == valid_entry.id
    assert fetched.gloss == valid_entry.gloss


def test_sqlite_init_idempotent_no_duplicate_indexes(tmp_path: Path) -> None:
    db = tmp_path / "store.sqlite3"
    SQLiteSignStore(db_path=db)
    SQLiteSignStore(db_path=db)
    SQLiteSignStore(db_path=db)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
    names = sorted(r[0] for r in rows)
    assert names == ["idx_sign_entries_gloss_lower", "idx_sign_entries_status"]


def test_sqlite_put_is_upsert_not_insert(
    tmp_path: Path, valid_entry: SignEntry
) -> None:
    db = tmp_path / "store.sqlite3"
    store = SQLiteSignStore(db_path=db)

    valid_entry.gloss = "FIRST"
    store.put(valid_entry)
    valid_entry.gloss = "SECOND"
    store.put(valid_entry)
    valid_entry.gloss = "THIRD"
    store.put(valid_entry)

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT id, gloss, gloss_lower FROM sign_entries"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "THIRD"
    # Denormalized lower-case column must reflect the latest gloss.
    assert rows[0][2] == "third"


def test_sqlite_update_status_keeps_indexed_columns_in_sync(
    tmp_path: Path, valid_entry: SignEntry
) -> None:
    db = tmp_path / "store.sqlite3"
    store = SQLiteSignStore(db_path=db)
    store.put(valid_entry)
    store.update_status(valid_entry.id, "validated")
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT status FROM sign_entries WHERE id = ?", (str(valid_entry.id),)
        ).fetchone()
    assert row[0] == "validated"


# ---------------------------------------------------------------------------
# JSON-file specific
# ---------------------------------------------------------------------------


def test_json_file_path_uses_uuid(tmp_path: Path, valid_entry: SignEntry) -> None:
    store = JSONFileSignStore(base_dir=tmp_path / "j", kozha_data_dir=tmp_path / "k")
    store.put(valid_entry)
    assert (tmp_path / "j" / f"{valid_entry.id}.json").exists()


def test_json_file_corrupt_entry_raises(tmp_path: Path) -> None:
    base = tmp_path / "j"
    base.mkdir()
    (base / "00000000-0000-0000-0000-000000000000.json").write_text("not json")
    store = JSONFileSignStore(base_dir=base, kozha_data_dir=tmp_path / "k")
    with pytest.raises(StorageError, match="corrupt JSON entry"):
        store.list()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_kozha_data_dir_points_into_repo() -> None:
    # The default lives at <repo>/data — confirm we're not silently writing
    # somewhere unexpected when the constructor is called bare.
    assert DEFAULT_KOZHA_DATA_DIR.name == "data"
    assert DEFAULT_KOZHA_DATA_DIR.is_dir()
