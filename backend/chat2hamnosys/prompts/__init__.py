"""Versioned prompt library for every LLM call in chat2hamnosys.

Each prompt is a Jinja2 template at ``prompts/<id>_v<n>.md.j2`` with a
strict YAML frontmatter header. The file name carries the version:
once shipped a prompt is immutable and a modification means creating
``<id>_v<n+1>.md.j2`` so eval numbers can be compared across versions.

Public API:

- :func:`load_prompt` — load one prompt by id (optionally at a pinned
  version); returns a :class:`PromptTemplate` with metadata + Jinja2
  template.
- :func:`render` — load + render in one call; missing context
  variables raise :class:`jinja2.UndefinedError` because the
  environment uses ``StrictUndefined``.
- :data:`PROMPTS_DIR` — absolute path to the template directory.
"""

from .loader import (
    PROMPTS_DIR,
    PromptError,
    PromptMetadata,
    PromptNotFound,
    PromptTemplate,
    list_prompts,
    load_prompt,
    render,
)

__all__ = [
    "PROMPTS_DIR",
    "PromptError",
    "PromptMetadata",
    "PromptNotFound",
    "PromptTemplate",
    "list_prompts",
    "load_prompt",
    "render",
]
