"""Guardrails for the versioned prompt library.

Two concerns:

1. **Every LLM call must go through the library.** A regression that
   inlined a raw system-prompt string at a ``.chat(...)`` call site
   would bypass the hash/version fingerprint and cause the eval harness
   to miss behavior drift. An AST walk flags any prompt-sized string
   constant passed as message ``content`` outside ``prompts/``.

2. **Every prompt file is loadable.** If frontmatter is malformed or a
   Jinja2 variable is missing the loader raises — we assert the four
   spec-required prompts all load.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from prompts import PROMPTS_DIR, list_prompts, load_prompt


_REPO_ROOT: Path = PROMPTS_DIR.parent  # backend/chat2hamnosys
_SKIP_DIRS: tuple[Path, ...] = (
    PROMPTS_DIR,
    _REPO_ROOT / "tests",
)
_INLINE_CONTENT_THRESHOLD = 80  # chars above which a literal looks prompt-sized

# Prompts the spec calls out explicitly — these four must always exist.
_REQUIRED_PROMPTS: tuple[str, ...] = (
    "parse_description",
    "generate_clarification",
    "generate_hamnosys_fallback",
    "interpret_correction",
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _python_modules() -> list[Path]:
    """All package .py files outside prompts/ and tests/."""
    files: list[Path] = []
    for path in _REPO_ROOT.rglob("*.py"):
        if any(_is_under(path, skip) for skip in _SKIP_DIRS):
            continue
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _iter_chat_calls(tree: ast.AST):
    """Yield every ``Call`` node whose function name is ``chat`` or ``stream_chat``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"chat", "stream_chat"}:
            yield node


def _messages_arg(call: ast.Call) -> ast.expr | None:
    """Locate the ``messages`` argument — positional first, then keyword."""
    for kw in call.keywords:
        if kw.arg == "messages":
            return kw.value
    if call.args:
        return call.args[0]
    return None


def _string_content_offenders(
    module_path: Path, tree: ast.AST
) -> list[str]:
    """Return pretty-printed lines flagging prompt-sized inline content strings."""
    offenders: list[str] = []
    for call in _iter_chat_calls(tree):
        msgs = _messages_arg(call)
        if not isinstance(msgs, ast.List):
            continue
        for element in msgs.elts:
            if not isinstance(element, ast.Dict):
                continue
            for key, value in zip(element.keys, element.values):
                if not (isinstance(key, ast.Constant) and key.value == "content"):
                    continue
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and len(value.value) > _INLINE_CONTENT_THRESHOLD
                ):
                    rel = module_path.relative_to(_REPO_ROOT)
                    offenders.append(f"{rel}:{value.lineno}")
    return offenders


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_inline_prompt_strings_outside_library() -> None:
    """AST-guardrail: every LLM prompt must come from ``prompts/``."""
    offenders: list[str] = []
    for path in _python_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(_string_content_offenders(path, tree))
    assert not offenders, (
        "Inline system/user prompt strings detected outside prompts/. "
        "Move them into a versioned prompt template and load via "
        f"prompts.load_prompt(). Offenders: {offenders}"
    )


@pytest.mark.parametrize("prompt_id", _REQUIRED_PROMPTS)
def test_required_prompts_are_loadable(prompt_id: str) -> None:
    """Each spec-required prompt must load with frontmatter intact."""
    pt = load_prompt(prompt_id)
    meta = pt.metadata
    assert meta.id == prompt_id
    assert meta.version.startswith("v") and meta.version[1:].isdigit()
    assert meta.model
    assert 0.0 <= meta.temperature <= 2.0
    assert meta.hash and len(meta.hash) == 64


def test_list_prompts_includes_required_ids() -> None:
    """``list_prompts`` advertises every required prompt id."""
    catalog = list_prompts()
    for pid in _REQUIRED_PROMPTS:
        assert pid in catalog, f"prompts library is missing required id {pid!r}"
