"""Loader for :file:`vocab_map.yaml`.

The YAML file is the single source of truth for plain-English →
HamNoSys 4.0 mappings. This module loads it once at import time, keeps a
normalized in-memory view, and exposes a small lookup API the composer
uses directly.

Design notes worth knowing before editing:

- The YAML values are hex codepoint *strings* (``"E001"``) or lists of
  hex strings. They are decoded into actual characters exactly once here
  and cached; the composer only handles ``str`` thereafter.
- ``null`` values in YAML encode a legal but no-op term (e.g. ``repeat:
  once`` emits nothing). ``lookup()`` returns ``""`` for these, which is
  distinct from a missing-key miss — callers use :func:`has_term` when
  they need to distinguish.
- Input normalization strips the quoting used in the YAML keys:
  lower-cased, ``-``/space folded to ``_``, trailing whitespace dropped.
  So ``"bent-5"``, ``"Bent 5"``, and ``"bent_5"`` all hit the same key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VOCAB_PATH: Path = Path(__file__).resolve().parent / "vocab_map.yaml"


_SLOT_KEYS: tuple[str, ...] = (
    "handshape",
    "thumb_mod",
    "finger_mod",
    "orientation_ext_finger",
    "orientation_palm",
    "location",
    "contact",
    "movement_path",
    "size_mod",
    "speed_mod",
    "timing",
    "repeat",
)


@dataclass(frozen=True)
class VocabEntry:
    """One resolved vocabulary row.

    ``string`` is the concatenated HamNoSys chunk (possibly empty for
    no-op entries such as ``repeat: once``). ``codepoints`` is the
    underlying hex codepoint list, preserved for diagnostics and for the
    eval harness which reports per-symbol accuracy.
    """

    term: str
    string: str
    codepoints: tuple[int, ...]


def _decode_value(value: Any) -> tuple[str, tuple[int, ...]]:
    """Decode a YAML value into ``(chunk, codepoints)``.

    Accepts a single hex string, a list of hex strings, or ``None`` for a
    legal no-op term. Raises ``ValueError`` on any other shape so a
    typo'd YAML entry fails loudly at import time rather than silently
    dropping coverage.
    """
    if value is None:
        return "", ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = list(value)
    else:
        raise ValueError(
            f"vocab value must be a hex string, list of hex strings, or null; "
            f"got {type(value).__name__}: {value!r}"
        )
    codepoints: list[int] = []
    for raw in items:
        if not isinstance(raw, str):
            raise ValueError(
                f"vocab codepoint entries must be hex strings, got {raw!r}"
            )
        stripped = raw.strip()
        if not stripped:
            raise ValueError("vocab codepoint entries must be non-empty")
        try:
            codepoints.append(int(stripped, 16))
        except ValueError as exc:
            raise ValueError(f"{stripped!r} is not a valid hex codepoint") from exc
    chunk = "".join(chr(cp) for cp in codepoints)
    return chunk, tuple(codepoints)


def normalize_term(term: str) -> str:
    """Canonicalize a user-facing vocabulary term for lookup."""
    if not isinstance(term, str):
        raise TypeError(f"term must be a string, got {type(term).__name__}")
    t = term.strip().lower()
    # Collapse all separators to the single form used in YAML keys.
    return t.replace("-", "_").replace(" ", "_")


class VocabMap:
    """In-memory view of :file:`vocab_map.yaml`.

    Constructed at module load time and referenced as :data:`VOCAB`.
    Tests and the LLM fallback may build a fresh instance from a
    different YAML path by calling :func:`load`.
    """

    def __init__(self, table: dict[str, dict[str, VocabEntry]]) -> None:
        self._table = table

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(self._table.keys())

    def terms(self, slot: str) -> list[str]:
        """Return every term registered under ``slot``, sorted alphabetically."""
        self._require_slot(slot)
        return sorted(self._table[slot].keys())

    def has_term(self, slot: str, term: str) -> bool:
        self._require_slot(slot)
        return normalize_term(term) in self._table[slot]

    def lookup(self, slot: str, term: str) -> VocabEntry | None:
        """Return the :class:`VocabEntry` for ``term`` in ``slot``, or ``None``."""
        self._require_slot(slot)
        return self._table[slot].get(normalize_term(term))

    def resolve(self, slot: str, term: str) -> str | None:
        """Convenience wrapper — return just the HamNoSys chunk, or ``None``.

        Note: a hit on a no-op term returns ``""`` (empty string), which
        is *not* the same as ``None`` (missing term). Use :meth:`has_term`
        if you need to disambiguate.
        """
        entry = self.lookup(slot, term)
        if entry is None:
            return None
        return entry.string

    def excerpt(
        self, slot: str, *, limit: int = 20
    ) -> list[tuple[str, str]]:
        """Return up to ``limit`` ``(term, hex-codepoints)`` rows for ``slot``.

        Used by the LLM fallback to show adjacent concepts — a little
        context per slot helps the model pick an existing codepoint
        instead of hallucinating a new one.
        """
        self._require_slot(slot)
        rows: list[tuple[str, str]] = []
        for term in sorted(self._table[slot].keys()):
            entry = self._table[slot][term]
            hex_repr = " ".join(f"U+{cp:04X}" for cp in entry.codepoints)
            rows.append((term, hex_repr or "(no-op)"))
            if len(rows) >= limit:
                break
        return rows

    def _require_slot(self, slot: str) -> None:
        if slot not in self._table:
            raise KeyError(
                f"unknown slot {slot!r}; expected one of: "
                f"{', '.join(self._table.keys())}"
            )


def load(path: Path | None = None) -> VocabMap:
    """Parse a vocab YAML file into a :class:`VocabMap`."""
    yaml_path = path if path is not None else VOCAB_PATH
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: top-level must be a mapping")
    table: dict[str, dict[str, VocabEntry]] = {}
    for slot in _SLOT_KEYS:
        if slot not in raw:
            table[slot] = {}
            continue
        section = raw[slot]
        if not isinstance(section, dict):
            raise ValueError(
                f"{yaml_path}: slot {slot!r} must be a mapping, "
                f"got {type(section).__name__}"
            )
        entries: dict[str, VocabEntry] = {}
        for raw_term, raw_value in section.items():
            term = normalize_term(str(raw_term))
            if term in entries:
                raise ValueError(
                    f"{yaml_path}: duplicate term {term!r} under slot {slot!r}"
                )
            chunk, codepoints = _decode_value(raw_value)
            entries[term] = VocabEntry(term=term, string=chunk, codepoints=codepoints)
        table[slot] = entries
    # Surface any unexpected top-level keys — a typo like "handhsape:" would
    # otherwise silently drop the entire section.
    extra = set(raw.keys()) - set(_SLOT_KEYS)
    if extra:
        raise ValueError(
            f"{yaml_path}: unexpected top-level keys {sorted(extra)}; "
            f"expected only: {list(_SLOT_KEYS)}"
        )
    return VocabMap(table)


# Module-level singleton. Loaded at import time.
VOCAB: VocabMap = load()


__all__ = ["VOCAB", "VocabEntry", "VocabMap", "VOCAB_PATH", "load", "normalize_term"]
