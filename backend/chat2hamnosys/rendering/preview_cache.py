"""Content-hash cache for avatar render outputs.

Avatar rendering is expensive (every call spins up JASigning/CWASA and
writes a video file), so we memoize by SHA-256 of the normalised SiGML
document. A hit returns the cached ``<hash>.mp4`` and its JSON sidecar;
a miss yields a reservation so the caller can write a new video into
the slot atomically.

Layout
------
Given ``cache_dir = data/preview_cache/`` and a SiGML document whose
SHA-256 is ``abcdef…``::

    data/preview_cache/
        abcdef….mp4         # the rendered clip
        abcdef….json        # sidecar metadata (see :class:`CacheMetadata`)

The sidecar records the gloss, cache creation time, the sigml hash
itself (so a rename doesn't de-sync), and the byte size of the video.
Having both files makes orphan detection easy: a ``.mp4`` without a
``.json`` (or vice-versa) is flagged by :meth:`PreviewCache.integrity`.

Concurrency
-----------
Writes go through :func:`os.replace` over a ``.tmp`` file so
concurrent renderers can't read a half-written video. The cache itself
does no locking — higher layers should avoid rendering the same SiGML
twice in parallel, but if they do, the last writer wins cleanly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sigml_content_hash(sigml: str) -> str:
    """SHA-256 over the UTF-8 bytes of ``sigml``.

    We hash the raw string, not a canonicalised form: the converter is
    deterministic so identical inputs yield byte-identical outputs, and
    normalising here would hide whitespace bugs upstream.
    """
    return hashlib.sha256(sigml.encode("utf-8")).hexdigest()


@dataclass
class CacheMetadata:
    """Sidecar record describing a cached render."""

    sigml_hash: str
    gloss: str
    created_at: str           # ISO 8601 UTC
    video_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sigml_hash": self.sigml_hash,
            "gloss": self.gloss,
            "created_at": self.created_at,
            "video_bytes": self.video_bytes,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CacheMetadata":
        return cls(
            sigml_hash=d["sigml_hash"],
            gloss=d.get("gloss", ""),
            created_at=d.get("created_at", ""),
            video_bytes=int(d.get("video_bytes", 0)),
            extra=dict(d.get("extra", {}) or {}),
        )


@dataclass
class CacheEntry:
    """One (video, sidecar) pair in the cache."""

    video_path: Path
    sidecar_path: Path
    metadata: CacheMetadata


class PreviewCache:
    """Content-addressable store of avatar renders.

    Typical usage::

        cache = PreviewCache(Path("data/preview_cache"))
        entry = cache.get(sigml)
        if entry is None:
            video = render_somehow(sigml)
            entry = cache.put(sigml, video, gloss="ELECTRON")
        serve(entry.video_path)
    """

    VIDEO_EXT = ".mp4"
    SIDECAR_EXT = ".json"

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _video_path(self, sigml_hash: str) -> Path:
        return self.cache_dir / f"{sigml_hash}{self.VIDEO_EXT}"

    def _sidecar_path(self, sigml_hash: str) -> Path:
        return self.cache_dir / f"{sigml_hash}{self.SIDECAR_EXT}"

    def get(self, sigml: str) -> CacheEntry | None:
        """Return the cached entry for ``sigml`` or ``None`` if absent.

        A partial hit (video without sidecar, or vice-versa) is treated
        as a miss — let :meth:`put` overwrite it cleanly rather than
        serving a half-populated record.
        """
        h = sigml_content_hash(sigml)
        video = self._video_path(h)
        sidecar = self._sidecar_path(h)
        if not video.exists() or not sidecar.exists():
            return None
        try:
            meta = CacheMetadata.from_dict(
                json.loads(sidecar.read_text("utf-8"))
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
        return CacheEntry(video_path=video, sidecar_path=sidecar, metadata=meta)

    def put(
        self,
        sigml: str,
        video_source: Path | str,
        *,
        gloss: str = "",
        extra: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Copy ``video_source`` into the cache under ``sigml``'s hash.

        ``video_source`` is the path to a freshly-rendered ``.mp4``.
        The file is copied (not moved) to leave the caller's render
        artefact intact — they might want it for logging.
        """
        source = Path(video_source)
        if not source.exists():
            raise FileNotFoundError(f"video source does not exist: {source}")

        h = sigml_content_hash(sigml)
        video = self._video_path(h)
        sidecar = self._sidecar_path(h)

        tmp_video = video.with_suffix(video.suffix + ".tmp")
        tmp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")

        shutil.copyfile(source, tmp_video)
        os.replace(tmp_video, video)

        meta = CacheMetadata(
            sigml_hash=h,
            gloss=gloss,
            created_at=datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            video_bytes=video.stat().st_size,
            extra=dict(extra or {}),
        )
        tmp_sidecar.write_text(
            json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_sidecar, sidecar)

        return CacheEntry(video_path=video, sidecar_path=sidecar, metadata=meta)

    def invalidate(self, sigml: str) -> bool:
        """Remove the cache entry for ``sigml`` if present.

        Returns ``True`` if something was removed. Useful in tests and
        for recovering from a known-bad render without nuking the whole
        cache directory.
        """
        h = sigml_content_hash(sigml)
        removed = False
        for p in (self._video_path(h), self._sidecar_path(h)):
            if p.exists():
                p.unlink()
                removed = True
        return removed

    def integrity(self) -> list[str]:
        """Return human-readable complaints about the cache on disk.

        An empty list means the cache is consistent. Entries are
        flagged if they're missing their sibling file or if the
        sidecar's recorded hash disagrees with its filename.
        """
        complaints: list[str] = []
        seen: dict[str, dict[str, Path]] = {}
        for p in self.cache_dir.iterdir():
            if p.suffix not in (self.VIDEO_EXT, self.SIDECAR_EXT):
                continue
            stem = p.stem
            seen.setdefault(stem, {})[p.suffix] = p
        for stem, files in seen.items():
            if self.VIDEO_EXT not in files:
                complaints.append(f"{stem}: sidecar without video")
                continue
            if self.SIDECAR_EXT not in files:
                complaints.append(f"{stem}: video without sidecar")
                continue
            try:
                meta = json.loads(files[self.SIDECAR_EXT].read_text("utf-8"))
            except json.JSONDecodeError as exc:
                complaints.append(f"{stem}: sidecar unreadable ({exc})")
                continue
            if meta.get("sigml_hash") != stem:
                complaints.append(
                    f"{stem}: sidecar hash {meta.get('sigml_hash')!r} "
                    f"does not match filename"
                )
        return complaints


__all__ = [
    "CacheEntry",
    "CacheMetadata",
    "PreviewCache",
    "sigml_content_hash",
]
