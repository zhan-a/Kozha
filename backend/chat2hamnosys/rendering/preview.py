"""Avatar preview — wraps a renderer if one is available, degrades if not.

``render_preview`` is the seam between chat2hamnosys and whatever avatar
engine is installed on the host. Kozha's browser UI calls
``CWASA.playSiGMLText`` from JavaScript (see
``public/cwa/allcsa.js``), so there is no first-party CLI renderer to
shell out to. In practice the Python side uses this wrapper for two
purposes:

1. **Dev / CI** — confirm that a SiGML document is well-formed and
   DTD-valid without rendering. The returned :class:`PreviewResult`
   carries the SiGML and a clear "renderer not available" status.
2. **Production** — when a headless renderer is installed (JASigning's
   CLI ``jasigning-cli`` or any compatible wrapper that accepts
   ``--sigml <file> --output <file.mp4>``), we invoke it, cache the
   output by SiGML hash, and return the video path.

The renderer is discovered from:

- ``KOZHA_RENDERER_CMD`` (env var): a command template with
  ``{sigml}`` and ``{output}`` placeholders, e.g.
  ``jasigning --sigml {sigml} --out {output}``.
- ``shutil.which("jasigning-cli")``: bare-path fallback with default
  flags.

If neither resolves, the wrapper returns
``PreviewStatus.RENDERER_NOT_AVAILABLE`` — callers get the SiGML they
asked for and a human-readable explanation instead of an exception.

CLI
---
    python -m backend.chat2hamnosys.rendering.preview '<hamnosys>' --gloss ELECTRON

Prints the SiGML to stdout, then a single status line on stderr.
``--render`` actually runs the renderer (if found); without it the
command only produces the SiGML.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from .hamnosys_to_sigml import SigmlValidationError, to_sigml
from .preview_cache import CacheEntry, PreviewCache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from models import NonManualFeatures  # noqa: F401


class PreviewStatus(str, Enum):
    """Outcome categories for :func:`render_preview`."""

    OK = "ok"                               # rendered fresh
    CACHED = "cached"                       # served from PreviewCache
    RENDERER_NOT_AVAILABLE = "not_available"  # no renderer + we degraded cleanly
    RENDERER_FAILED = "renderer_failed"     # renderer ran but did not produce video


@dataclass
class PreviewResult:
    """What :func:`render_preview` returns.

    ``video_path`` is ``None`` whenever the renderer did not run (or
    failed) — callers that only need a valid SiGML document can still
    rely on ``sigml``. ``message`` is always a human-readable sentence,
    safe to surface in UI.
    """

    sigml: str
    status: PreviewStatus
    message: str
    video_path: Path | None = None
    cache_entry: CacheEntry | None = None
    renderer_stderr: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff a playable video is available (fresh or cached)."""
        return self.status in (PreviewStatus.OK, PreviewStatus.CACHED)


def _default_cache_dir() -> Path:
    """Locate ``data/preview_cache`` relative to the repo root.

    The repo layout puts ``data/`` at the repo root and
    ``backend/chat2hamnosys/…`` three directories down, so we walk up
    from this module. If the resolution fails (e.g., the package was
    installed elsewhere), fall back to a cwd-local path.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "data").is_dir():
            return ancestor / "data" / "preview_cache"
    return Path.cwd() / "data" / "preview_cache"


def _resolve_renderer_cmd() -> list[str] | None:
    """Return the configured renderer command as a list of argv tokens.

    Returns ``None`` if no renderer is configured — the caller should
    degrade gracefully. The template may use ``{sigml}`` and
    ``{output}`` placeholders; they're substituted per invocation.
    """
    import os

    template = os.environ.get("KOZHA_RENDERER_CMD", "").strip()
    if template:
        return shlex.split(template)
    bare = shutil.which("jasigning-cli")
    if bare:
        # Conventional flags — any CLI that speaks SiGML typically
        # accepts file-in and file-out positional/flag pairs.
        return [bare, "--sigml", "{sigml}", "--output", "{output}"]
    return None


def _fill_template(cmd: list[str], *, sigml_path: Path, output_path: Path) -> list[str]:
    """Substitute ``{sigml}`` / ``{output}`` placeholders in ``cmd``.

    Tokens that don't match either placeholder are left alone. If
    neither placeholder appears we append the two paths at the end —
    that matches the ``jasigning-cli`` convention of taking positional
    input/output file paths when no flags are used.
    """
    saw_sigml = any("{sigml}" in t for t in cmd)
    saw_output = any("{output}" in t for t in cmd)
    filled = [
        t.replace("{sigml}", str(sigml_path)).replace("{output}", str(output_path))
        for t in cmd
    ]
    if not saw_sigml:
        filled.append(str(sigml_path))
    if not saw_output:
        filled.append(str(output_path))
    return filled


def _invoke_renderer(
    cmd: list[str],
    sigml: str,
    *,
    timeout: float,
) -> tuple[Path | None, str, Path]:
    """Write SiGML to a temp file, run ``cmd``, return the output path.

    Returns ``(output_path, stderr, tmpdir)``. On failure
    ``output_path`` is ``None``. The caller owns ``tmpdir`` and must
    remove it after copying the video out.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="kozha_render_"))
    sigml_file = tmpdir / "input.sigml"
    sigml_file.write_text(sigml, encoding="utf-8")
    output_file = tmpdir / "output.mp4"
    full_cmd = _fill_template(cmd, sigml_path=sigml_file, output_path=output_file)
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return None, f"renderer invocation failed: {type(exc).__name__}: {exc}", tmpdir
    stderr = result.stderr or ""
    if result.returncode != 0:
        return None, f"renderer exited with status {result.returncode}: {stderr}", tmpdir
    if not output_file.exists() or output_file.stat().st_size == 0:
        return None, f"renderer produced no output at {output_file}: {stderr}", tmpdir
    return output_file, stderr, tmpdir


def render_preview(
    sigml: str,
    *,
    gloss: str = "",
    cache: PreviewCache | None = None,
    renderer_cmd: list[str] | None = None,
    timeout: float = 60.0,
) -> PreviewResult:
    """Render ``sigml`` to a video, caching by content hash.

    Parameters
    ----------
    sigml:
        A SiGML document (produced by :func:`to_sigml` or equivalent).
    gloss:
        Optional gloss tag to record in the cache sidecar — only used
        if a fresh render is written.
    cache:
        A :class:`PreviewCache` to consult / populate. When ``None``,
        a default cache at ``<repo>/data/preview_cache`` is used.
    renderer_cmd:
        Explicit argv template for the renderer. Overrides the env /
        PATH resolution done by :func:`_resolve_renderer_cmd`. Mostly
        for tests.
    timeout:
        Seconds to wait for the renderer subprocess before giving up.

    Returns
    -------
    :class:`PreviewResult` describing the outcome. Never raises — even
    if the renderer is missing or crashes, the caller gets a SiGML
    string and a status explaining what happened.
    """
    if cache is None:
        cache = PreviewCache(_default_cache_dir())

    hit = cache.get(sigml)
    if hit is not None:
        return PreviewResult(
            sigml=sigml,
            status=PreviewStatus.CACHED,
            message=f"served from cache ({hit.metadata.sigml_hash[:12]})",
            video_path=hit.video_path,
            cache_entry=hit,
        )

    cmd = renderer_cmd if renderer_cmd is not None else _resolve_renderer_cmd()
    if cmd is None:
        return PreviewResult(
            sigml=sigml,
            status=PreviewStatus.RENDERER_NOT_AVAILABLE,
            message=(
                "rendered preview not available, SiGML produced successfully"
            ),
        )

    produced_path, stderr, tmpdir = _invoke_renderer(cmd, sigml, timeout=timeout)
    try:
        if produced_path is None:
            return PreviewResult(
                sigml=sigml,
                status=PreviewStatus.RENDERER_FAILED,
                message=(
                    "renderer did not produce a video, "
                    "SiGML produced successfully"
                ),
                renderer_stderr=stderr,
            )
        entry = cache.put(sigml, produced_path, gloss=gloss)
        return PreviewResult(
            sigml=sigml,
            status=PreviewStatus.OK,
            message=f"rendered fresh ({entry.metadata.sigml_hash[:12]})",
            video_path=entry.video_path,
            cache_entry=entry,
            renderer_stderr=stderr,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entry point — `python -m backend.chat2hamnosys.rendering.preview …`
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.chat2hamnosys.rendering.preview",
        description=(
            "Convert a HamNoSys string to SiGML and optionally render a "
            "preview video. Useful for manual spot-checks."
        ),
    )
    p.add_argument(
        "hamnosys",
        help=(
            "HamNoSys codepoint sequence. Accepts raw PUA characters, "
            "'U+E001 U+E020 …' hex tokens, or just '\\uE001\\uE020 …'."
        ),
    )
    p.add_argument(
        "--gloss",
        default="SIGN",
        help="Gloss attribute for the <hns_sign> element (default: SIGN).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help=(
            "Attempt to invoke the avatar renderer (KOZHA_RENDERER_CMD "
            "or jasigning-cli). Without this flag only SiGML is emitted."
        ),
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip DTD validation (diagnostic only — not recommended).",
    )
    p.add_argument(
        "--mouth-picture",
        default=None,
        help="Optional SAMPA mouth-shape string for <hnm_mouthpicture>.",
    )
    return p


def _parse_hamnosys_input(raw: str) -> str:
    """Accept either raw PUA chars or a hex-token form and return PUA.

    Hex tokens are anything of the form ``U+XXXX`` or ``\\uXXXX``,
    whitespace-separated. If none match we assume the input is already
    raw HamNoSys.
    """
    stripped = raw.strip()
    if not stripped:
        return ""
    # Lightweight hex-token detection: if the string looks like hex
    # tokens only, decode them; otherwise return the input verbatim.
    tokens = stripped.split()
    looks_hex = all(
        (t.upper().startswith("U+") and len(t) == 6)
        or (t.lower().startswith("\\u") and len(t) == 6)
        or (len(t) == 4 and all(c in "0123456789abcdefABCDEF" for c in t))
        for t in tokens
    )
    if looks_hex and len(tokens) > 1:
        chars: list[str] = []
        for t in tokens:
            h = t
            if h.upper().startswith("U+") or h.lower().startswith("\\u"):
                h = h[2:]
            chars.append(chr(int(h, 16)))
        return "".join(chars)
    return stripped


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = _build_argparser().parse_args(argv)
    hamnosys = _parse_hamnosys_input(args.hamnosys)

    non_manual = None
    if args.mouth_picture:
        # Keep this import local — importing models at module load would
        # force pydantic on every `python -m … --help` invocation.
        from models import NonManualFeatures
        non_manual = NonManualFeatures(mouth_picture=args.mouth_picture)

    try:
        sigml = to_sigml(
            hamnosys,
            gloss=args.gloss,
            non_manual=non_manual,
            validate=not args.no_validate,
        )
    except SigmlValidationError as exc:
        print(f"ERROR: SiGML failed DTD validation: {exc}", file=sys.stderr)
        for err in exc.errors[:5]:
            print(f"  {err}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(sigml, end="")
    if args.render:
        result = render_preview(sigml, gloss=args.gloss)
        print(f"\n[status] {result.status.value}: {result.message}", file=sys.stderr)
        if result.video_path:
            print(f"[video] {result.video_path}", file=sys.stderr)
        if result.renderer_stderr:
            print(
                f"[renderer-stderr] {result.renderer_stderr.strip()}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PreviewResult",
    "PreviewStatus",
    "main",
    "render_preview",
]
