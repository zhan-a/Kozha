"""PII handling for authored sign entries.

Two operations, both keyed off :class:`.config.PIIPolicy`:

- :func:`hash_signer_id` deterministically hashes a plaintext signer
  identifier under a per-installation salt so two entries by the same
  author can be correlated without storing the raw id.
- :func:`strip_signer_identifiers` removes author-identifying fields
  from a dumped :class:`SignEntry` payload prior to export. The
  default policy strips, even when the policy allows plaintext
  storage — exports leave the system and cannot be retracted once
  downstream consumers cache them.
"""

from __future__ import annotations

import hashlib
import hmac
from copy import deepcopy
from typing import Any


_HASH_PREFIX: str = "h:"


def hash_signer_id(signer_id: str, *, salt: str) -> str:
    """Deterministic HMAC-SHA-256 of ``signer_id`` under ``salt``.

    Returns a string with the literal prefix ``"h:"`` followed by the
    hex digest truncated to 32 characters (128 bits). The prefix lets
    downstream code distinguish already-hashed values from plaintext
    ones, so double-hashing is avoided.

    The output is 34 characters total, below the 64-char SQL column
    width that some legacy stores use for ``signer_id``.
    """
    if not isinstance(signer_id, str) or not signer_id:
        raise ValueError("signer_id must be a non-empty string")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    if signer_id.startswith(_HASH_PREFIX):
        return signer_id
    digest = hmac.new(
        salt.encode("utf-8"), signer_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{_HASH_PREFIX}{digest[:32]}"


def strip_signer_identifiers(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``payload`` with author identifiers removed.

    Specifically: under any ``author`` sub-object, the fields
    ``signer_id`` and ``display_name`` are dropped. The rest of
    :class:`AuthorInfo` (``is_deaf_native``) is preserved because it
    is research-relevant but not identifying.

    Safe to call on payloads that do not contain an ``author`` field.
    """
    stripped = deepcopy(payload)
    _strip_in_place(stripped)
    return stripped


def _strip_in_place(node: Any) -> None:
    if isinstance(node, dict):
        author = node.get("author")
        if isinstance(author, dict):
            author.pop("signer_id", None)
            author.pop("display_name", None)
        for value in node.values():
            _strip_in_place(value)
    elif isinstance(node, list):
        for item in node:
            _strip_in_place(item)


__all__ = ["hash_signer_id", "strip_signer_identifiers"]
