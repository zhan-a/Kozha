"""Storage backends for authored sign entries.

Two backends share a single :class:`SignStore` contract:

- :class:`JSONFileSignStore` — one JSON file per entry, file name is the UUID.
  Convenient for development, easy to inspect with ``cat`` / ``jq``.
- :class:`SQLiteSignStore` — single-file SQLite database. Better for the
  authoring service in production: faster ``list``, atomic transactions,
  no million-tiny-files filesystem overhead.

Both backends share the export path: :meth:`SignStore.export_to_kozha_library`
transforms a ``SignEntry`` into the canonical Kozha SiGML format
(``<hns_sign gloss="…"><hamnosys_manual>…</hamnosys_manual></hns_sign>``)
and writes it into ``data/hamnosys_<lang>_authored.sigml``.

The export method refuses any entry whose ``status`` is not ``"validated"``
— this is the gate that keeps draft authoring work out of the live library.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from hamnosys import SYMBOLS
from models import SignEntry, SignStatus


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """Base class for storage-layer errors."""


class SignNotFoundError(StorageError, KeyError):
    """Requested sign id is not in the store."""

    def __init__(self, sign_id: UUID) -> None:
        super().__init__(f"sign id {sign_id} not found")
        self.sign_id = sign_id


class ExportNotAllowedError(StorageError):
    """Export refused because the entry is not in ``"validated"`` status."""

    def __init__(self, sign_id: UUID, status: SignStatus) -> None:
        super().__init__(
            f"sign id {sign_id} has status {status!r}; only 'validated' entries "
            "may be exported to the Kozha sign library"
        )
        self.sign_id = sign_id
        self.status = status


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KOZHA_DATA_DIR = _REPO_ROOT / "data"
DEFAULT_AUTHORED_JSON_DIR = DEFAULT_KOZHA_DATA_DIR / "authored_signs"
DEFAULT_AUTHORED_SQLITE_PATH = DEFAULT_KOZHA_DATA_DIR / "authored_signs.sqlite3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via tmpfile + ``os.replace``.

    Avoids partial-write corruption on crash mid-write — readers always see
    either the previous file or the complete new file, never a half-flushed
    intermediate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _hamnosys_to_manual_element(hamnosys: str) -> ET.Element:
    """Build a ``<hamnosys_manual>`` element from a HamNoSys PUA string.

    Each codepoint becomes one self-closing tag named after its
    ``Symbol.short_name`` (e.g. U+E001 → ``<hamflathand/>``). Codepoints not
    in the symbol table raise — the model layer should have rejected these
    already, but we re-check at the boundary.
    """
    manual = ET.Element("hamnosys_manual")
    for ch in hamnosys:
        sym = SYMBOLS.get(ord(ch))
        if sym is None:
            raise ValueError(
                f"cannot serialize: U+{ord(ch):04X} is not a known HamNoSys symbol"
            )
        ET.SubElement(manual, sym.short_name)
    return manual


def _build_hns_sign_element(entry: SignEntry) -> ET.Element:
    """Build the ``<hns_sign gloss="…">`` element for a sign entry."""
    # Frontend lowercases on lookup (public/app.html:1092) — emit the gloss
    # already lowercased so the authored file matches the lookup convention.
    gloss = entry.gloss.strip().lower()
    sign = ET.Element("hns_sign", {"gloss": gloss})

    nonmanual = ET.SubElement(sign, "hamnosys_nonmanual")
    nm = entry.parameters.non_manual
    if nm is not None and nm.mouth_picture:
        ET.SubElement(nonmanual, "hnm_mouthpicture", {"picture": nm.mouth_picture})

    sign.append(_hamnosys_to_manual_element(entry.hamnosys))
    return sign


def _read_or_init_sigml(path: Path) -> ET.Element:
    """Read an existing authored SiGML file or return an empty ``<sigml/>``."""
    if not path.exists():
        return ET.Element("sigml")
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise StorageError(
            f"existing authored SiGML file {path} is malformed: {exc}"
        ) from exc
    root = tree.getroot()
    if root.tag != "sigml":
        raise StorageError(
            f"existing authored SiGML file {path} root is <{root.tag}>, expected <sigml>"
        )
    return root


def _replace_or_append_sign(root: ET.Element, new_sign: ET.Element) -> None:
    """Replace an existing ``<hns_sign>`` with the same gloss, else append."""
    target_gloss = (new_sign.get("gloss") or "").lower()
    for i, child in enumerate(list(root)):
        if child.tag != "hns_sign":
            continue
        existing = (child.get("gloss") or "").lower()
        if existing == target_gloss:
            root.remove(child)
            root.insert(i, new_sign)
            return
    root.append(new_sign)


def _serialize_sigml(root: ET.Element) -> str:
    """Pretty-print a ``<sigml>`` document with tab indent + XML prolog.

    Mimics the upstream ``data/hamnosys_<lang>_version1.sigml`` formatting:
    one entry per line of children, tab indentation, self-closing voids.
    """
    ET.indent(root, space="\t")
    body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SignStore(ABC):
    """Storage contract for authored sign entries.

    The export path (:meth:`export_to_kozha_library`) is implemented here
    once and shared across backends — it is a pure function of the entry's
    contents and the ``kozha_data_dir`` configured at construction.
    """

    def __init__(self, *, kozha_data_dir: Path | str | None = None) -> None:
        self.kozha_data_dir = (
            Path(kozha_data_dir) if kozha_data_dir is not None else DEFAULT_KOZHA_DATA_DIR
        )

    # -- CRUD --------------------------------------------------------------

    @abstractmethod
    def put(self, entry: SignEntry) -> None:
        """Insert or replace ``entry`` (keyed by ``entry.id``)."""

    @abstractmethod
    def get(self, sign_id: UUID) -> SignEntry:
        """Fetch by id. Raises :class:`SignNotFoundError` if absent."""

    @abstractmethod
    def list(self) -> list[SignEntry]:
        """Return all entries in the store (no defined order)."""

    @abstractmethod
    def update_status(self, sign_id: UUID, new_status: SignStatus) -> None:
        """Change ``status`` and bump ``updated_at``."""

    @abstractmethod
    def search_by_gloss(self, gloss: str) -> list[SignEntry]:
        """Case-insensitive exact-match search on the ``gloss`` field."""

    # -- Export ------------------------------------------------------------

    def export_to_kozha_library(self, sign_id: UUID) -> None:
        """Write a validated entry into Kozha's authored SiGML library.

        Refuses any entry whose ``status`` is not ``"validated"`` —
        :class:`ExportNotAllowedError` is raised. Idempotent: re-exporting
        the same entry replaces its ``<hns_sign>`` in place rather than
        appending a duplicate.

        Target file: ``<kozha_data_dir>/hamnosys_<lang>_authored.sigml``.
        Per the audit (docs/chat2hamnosys/00-repo-audit.md §9e), authored
        entries go in a separate per-language file so the upstream
        DictaSign-licensed corpora are never mutated.
        """
        entry = self.get(sign_id)
        if entry.status != "validated":
            raise ExportNotAllowedError(sign_id, entry.status)

        target = self.kozha_data_dir / f"hamnosys_{entry.sign_language}_authored.sigml"
        root = _read_or_init_sigml(target)
        new_sign = _build_hns_sign_element(entry)
        _replace_or_append_sign(root, new_sign)
        _atomic_write_text(target, _serialize_sigml(root))


# ---------------------------------------------------------------------------
# JSON file backend
# ---------------------------------------------------------------------------


class JSONFileSignStore(SignStore):
    """One JSON file per entry. File name = ``<uuid>.json``.

    Good for dev — every write is human-readable and grep-able. Not great
    for thousands of entries: ``list()`` does a directory scan and reparses
    each file, and ``search_by_gloss`` is O(n).
    """

    def __init__(
        self,
        *,
        base_dir: Path | str | None = None,
        kozha_data_dir: Path | str | None = None,
    ) -> None:
        super().__init__(kozha_data_dir=kozha_data_dir)
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_AUTHORED_JSON_DIR

    def _path_for(self, sign_id: UUID) -> Path:
        return self.base_dir / f"{sign_id}.json"

    def put(self, entry: SignEntry) -> None:
        # Bump updated_at on every put so callers don't have to remember.
        entry.updated_at = _utcnow()
        payload = entry.model_dump_json_safe()
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        _atomic_write_text(self._path_for(entry.id), text)

    def get(self, sign_id: UUID) -> SignEntry:
        path = self._path_for(sign_id)
        if not path.exists():
            raise SignNotFoundError(sign_id)
        text = path.read_text(encoding="utf-8")
        return SignEntry.model_validate_json(text)

    def list(self) -> list[SignEntry]:
        if not self.base_dir.exists():
            return []
        out: list[SignEntry] = []
        for child in sorted(self.base_dir.iterdir()):
            if child.suffix != ".json":
                continue
            try:
                out.append(SignEntry.model_validate_json(child.read_text(encoding="utf-8")))
            except Exception as exc:
                raise StorageError(f"corrupt JSON entry at {child}: {exc}") from exc
        return out

    def update_status(self, sign_id: UUID, new_status: SignStatus) -> None:
        entry = self.get(sign_id)
        entry.status = new_status
        entry.updated_at = _utcnow()
        self.put(entry)

    def search_by_gloss(self, gloss: str) -> list[SignEntry]:
        needle = gloss.strip().lower()
        return [e for e in self.list() if e.gloss.strip().lower() == needle]


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


_SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS sign_entries (
    id            TEXT PRIMARY KEY,
    gloss         TEXT NOT NULL,
    gloss_lower   TEXT NOT NULL,
    sign_language TEXT NOT NULL,
    status        TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sign_entries_gloss_lower ON sign_entries(gloss_lower);
CREATE INDEX IF NOT EXISTS idx_sign_entries_status      ON sign_entries(status);
"""


class SQLiteSignStore(SignStore):
    """SQLite-backed store.

    The authoritative payload is the JSON serialization of the entry; the
    indexed columns (``gloss_lower``, ``status``, ``updated_at``) are
    denormalized copies maintained on every write.

    Idempotent: ``put`` is ``INSERT OR REPLACE``, so writing the same entry
    twice ends with one row, not two. The schema migration is also
    idempotent (``IF NOT EXISTS``), so re-opening the same DB file is safe.
    """

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        kozha_data_dir: Path | str | None = None,
    ) -> None:
        super().__init__(kozha_data_dir=kozha_data_dir)
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_AUTHORED_SQLITE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SQL_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def put(self, entry: SignEntry) -> None:
        entry.updated_at = _utcnow()
        payload = json.dumps(entry.model_dump_json_safe(), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sign_entries
                    (id, gloss, gloss_lower, sign_language, status, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    gloss=excluded.gloss,
                    gloss_lower=excluded.gloss_lower,
                    sign_language=excluded.sign_language,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    str(entry.id),
                    entry.gloss,
                    entry.gloss.strip().lower(),
                    entry.sign_language,
                    entry.status,
                    entry.updated_at.isoformat(),
                    payload,
                ),
            )

    def get(self, sign_id: UUID) -> SignEntry:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM sign_entries WHERE id = ?", (str(sign_id),)
            ).fetchone()
        if row is None:
            raise SignNotFoundError(sign_id)
        return SignEntry.model_validate_json(row["payload"])

    def list(self) -> list[SignEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM sign_entries ORDER BY updated_at DESC"
            ).fetchall()
        return [SignEntry.model_validate_json(r["payload"]) for r in rows]

    def update_status(self, sign_id: UUID, new_status: SignStatus) -> None:
        # Round-trip through SignEntry so the same validation rules apply
        # whether status changes via update_status or a re-put.
        entry = self.get(sign_id)
        entry.status = new_status
        entry.updated_at = _utcnow()
        self.put(entry)

    def search_by_gloss(self, gloss: str) -> list[SignEntry]:
        needle = gloss.strip().lower()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM sign_entries WHERE gloss_lower = ?",
                (needle,),
            ).fetchall()
        return [SignEntry.model_validate_json(r["payload"]) for r in rows]
