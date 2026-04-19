"""File-backed loader for the versioned Jinja2 prompt library.

File layout
-----------
Every prompt lives in ``prompts/<id>_v<n>.md.j2`` (e.g.
``parse_description_v1.md.j2``). Two hard rules:

1. The version suffix is baked into the file name — an immutable
   prompt is never edited once shipped; a modification ships as a
   new ``<id>_v<n+1>.md.j2`` so eval numbers remain comparable across
   versions.
2. Every file begins with a YAML frontmatter header delimited by
   ``---`` lines carrying ``id``, ``description``, ``model``,
   ``temperature``, ``response_format``, ``created``, ``owner``. The
   ``id`` in the frontmatter must match the file-name stem (minus the
   ``_v<n>`` suffix).

The loader parses the header once per file and caches both metadata
and a Jinja2 :class:`~jinja2.Template` keyed by ``(id, version)``.
Templates are rendered with :class:`jinja2.StrictUndefined` so every
missing context variable is a loud error, not a silent empty string.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import (
    BaseLoader,
    Environment,
    StrictUndefined,
    Template,
    TemplateNotFound,
    select_autoescape,
)


PROMPTS_DIR: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PromptError(Exception):
    """Base class for prompt-loading errors."""


class PromptNotFound(PromptError):
    """Raised when no prompt matches the requested id / version."""


class PromptFrontmatterError(PromptError):
    """Raised when a prompt file's frontmatter is missing or malformed."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "description",
    "model",
    "temperature",
    "response_format",
    "created",
    "owner",
)


@dataclass(frozen=True)
class PromptMetadata:
    """Structured view of a prompt file's frontmatter header.

    ``hash`` is the SHA-256 of the raw file bytes (header + body) —
    a stable fingerprint of the exact prompt in use, logged alongside
    every LLM call so telemetry entries can be correlated with the
    prompt version that produced them.
    """

    id: str
    version: str
    description: str
    model: str
    temperature: float
    response_format: str
    created: str
    owner: str
    hash: str
    path: Path


@dataclass(frozen=True)
class PromptTemplate:
    """One loaded prompt: metadata + rendered-on-demand Jinja2 template."""

    metadata: PromptMetadata
    template: Template

    def render(self, **context: Any) -> str:
        """Render the template. Missing context vars raise ``UndefinedError``."""
        return self.template.render(**context)


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------


class _FrontmatterStrippingLoader(BaseLoader):
    """Jinja2 loader that returns only the prompt body, not its header.

    The frontmatter YAML between the leading ``---`` lines is consumed
    by :func:`_parse_frontmatter` and must not reach the Jinja2 parser
    (otherwise literal YAML ends up in rendered prompts).
    """

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str, Any]:
        path = PROMPTS_DIR / template
        if not path.is_file():
            raise TemplateNotFound(template)
        _, body = _parse_frontmatter(path)
        mtime = path.stat().st_mtime

        def _uptodate() -> bool:
            try:
                return path.stat().st_mtime == mtime
            except OSError:
                return False

        return body, str(path), _uptodate


def _build_env() -> Environment:
    return Environment(
        loader=_FrontmatterStrippingLoader(),
        undefined=StrictUndefined,
        autoescape=select_autoescape(
            enabled_extensions=(),
            default_for_string=False,
            default=False,
        ),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


_ENV: Environment = _build_env()


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    """Split a prompt file into (frontmatter_dict, body_string).

    Enforces the three-line header shape: line 1 must be ``---``, the
    closing ``---`` must appear before the body, and the header must
    parse as a YAML mapping. Any deviation raises
    :class:`PromptFrontmatterError` so broken prompts fail at load
    time rather than silently producing empty renders.
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise PromptFrontmatterError(
            f"{path.name}: file must begin with a '---' frontmatter line"
        )
    try:
        close_idx = next(
            i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise PromptFrontmatterError(
            f"{path.name}: missing closing '---' for frontmatter"
        ) from exc

    header_text = "".join(lines[1:close_idx])
    body = "".join(lines[close_idx + 1:])
    try:
        meta = yaml.safe_load(header_text) or {}
    except yaml.YAMLError as exc:
        raise PromptFrontmatterError(
            f"{path.name}: frontmatter is not valid YAML: {exc}"
        ) from exc
    if not isinstance(meta, dict):
        raise PromptFrontmatterError(
            f"{path.name}: frontmatter must be a YAML mapping, got "
            f"{type(meta).__name__}"
        )
    missing = [f for f in _REQUIRED_FIELDS if f not in meta]
    if missing:
        raise PromptFrontmatterError(
            f"{path.name}: frontmatter missing required fields: {missing}"
        )
    return meta, body


def _version_from_filename(path: Path) -> str:
    """Extract ``v<n>`` from ``<id>_v<n>.md.j2``.

    The suffix is canonical: a prompt file without ``_v<digits>`` is
    a programmer error and fails loudly rather than defaulting to v1.
    """
    stem = path.name
    for suffix in (".md.j2", ".j2", ".md"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "_v" not in stem:
        raise PromptFrontmatterError(
            f"{path.name}: file name must contain '_v<n>' version suffix"
        )
    base, _, ver = stem.rpartition("_v")
    if not ver.isdigit() or not base:
        raise PromptFrontmatterError(
            f"{path.name}: version suffix must be numeric, got {ver!r}"
        )
    return f"v{int(ver)}"


def _id_from_filename(path: Path) -> str:
    stem = path.name
    for suffix in (".md.j2", ".j2", ".md"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    base, _, _ = stem.rpartition("_v")
    return base


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Index + public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _index() -> dict[str, dict[str, Path]]:
    """Map ``{prompt_id: {version: path}}`` built from the directory listing."""
    out: dict[str, dict[str, Path]] = {}
    for path in sorted(PROMPTS_DIR.iterdir()):
        if not path.is_file():
            continue
        if not path.name.endswith(".md.j2"):
            continue
        pid = _id_from_filename(path)
        ver = _version_from_filename(path)
        out.setdefault(pid, {})[ver] = path
    return out


def list_prompts() -> dict[str, list[str]]:
    """Return ``{prompt_id: [sorted versions]}`` — useful for the eval CLI."""
    return {pid: sorted(vers.keys(), key=_version_sort_key) for pid, vers in _index().items()}


def _version_sort_key(v: str) -> int:
    return int(v.lstrip("v"))


def _latest_version(prompt_id: str) -> str:
    vers = _index().get(prompt_id)
    if not vers:
        raise PromptNotFound(
            f"no prompt found with id {prompt_id!r} in {PROMPTS_DIR}"
        )
    return max(vers.keys(), key=_version_sort_key)


@lru_cache(maxsize=128)
def _load_cached(prompt_id: str, version: str) -> PromptTemplate:
    vers = _index().get(prompt_id)
    if not vers or version not in vers:
        raise PromptNotFound(
            f"no prompt found with id {prompt_id!r} version {version!r} in "
            f"{PROMPTS_DIR}"
        )
    path = vers[version]
    meta_dict, body = _parse_frontmatter(path)

    # Cross-check: the frontmatter id must agree with the filename id.
    if meta_dict["id"] != f"{prompt_id}_{version}" and meta_dict["id"] != prompt_id:
        # Accept either "<id>" or "<id>_v<n>" — the spec example in
        # Prompt 11 shows "parse_description_v1" (file-stem style),
        # but callers pass the bare id.
        raise PromptFrontmatterError(
            f"{path.name}: frontmatter id {meta_dict['id']!r} does not match "
            f"filename-derived id {prompt_id!r} (or {prompt_id}_{version!r})"
        )

    template = _ENV.get_template(path.name)
    metadata = PromptMetadata(
        id=prompt_id,
        version=version,
        description=str(meta_dict["description"]),
        model=str(meta_dict["model"]),
        temperature=float(meta_dict["temperature"]),
        response_format=str(meta_dict["response_format"]),
        created=str(meta_dict["created"]),
        owner=str(meta_dict["owner"]),
        hash=_hash_file(path),
        path=path,
    )
    return PromptTemplate(metadata=metadata, template=template)


def load_prompt(prompt_id: str, version: str = "latest") -> PromptTemplate:
    """Load a prompt by id and optional version.

    ``version="latest"`` picks the highest ``v<n>`` suffix present on
    disk. Explicit versions use the exact ``v<n>`` form (e.g. ``"v2"``).
    Raises :class:`PromptNotFound` when the prompt / version does not
    exist, and :class:`PromptFrontmatterError` on a malformed header.
    """
    if version == "latest":
        version = _latest_version(prompt_id)
    return _load_cached(prompt_id, version)


def render(prompt_id: str, version: str = "latest", **context: Any) -> str:
    """Shorthand for ``load_prompt(...).render(**context)``.

    Missing context variables raise :class:`jinja2.UndefinedError`
    because the environment uses :class:`StrictUndefined`.
    """
    return load_prompt(prompt_id, version).render(**context)


def _reset_cache() -> None:
    """Test helper — clears the module-level caches.

    Only used by eval harness tooling or tests that mutate files on
    disk; production code never needs to invalidate the cache.
    """
    _index.cache_clear()
    _load_cached.cache_clear()
