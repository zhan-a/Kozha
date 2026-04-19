"""OpenAI Moderation-endpoint wrapper for user-facing LLM output.

The parser and correction interpreter emit JSON that is structurally
constrained by schema. Two outputs are *natural language* shown to
users: the clarification questions from
:mod:`chat2hamnosys.clarify.question_generator` and the explanation
text from :mod:`chat2hamnosys.correct.correction_interpreter`. Those
are the narrow channel an LLM could use to emit harassment, hate,
self-harm instructions, etc. at a user.

:func:`moderate_output` calls OpenAI's free ``/moderations`` endpoint
(model ``omni-moderation-latest``) and returns a structured verdict.
If the endpoint is unavailable (no client, no network), the function
returns ``blocked=False`` and logs — moderation is defense in depth,
not an availability-critical gate.

We explicitly block the categories that would be serious in a
sign-authoring context: ``hate``, ``harassment``, ``self-harm``,
``sexual/minors``, and ``violence/graphic``. Generic NSFW content is
not blocked by default because some sign descriptions legitimately
touch on sexuality, body, and medical topics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable


logger = logging.getLogger(__name__)


DEFAULT_BLOCKED_CATEGORIES: tuple[str, ...] = (
    "hate",
    "hate/threatening",
    "harassment",
    "harassment/threatening",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual/minors",
    "violence/graphic",
)


@dataclass
class ModerationResult:
    """Outcome of one moderation call."""

    blocked: bool
    categories: tuple[str, ...] = ()
    reason: str = ""
    raw: Any = field(default=None, repr=False)


def moderate_output(
    text: str,
    *,
    client: Any = None,
    blocked_categories: Iterable[str] = DEFAULT_BLOCKED_CATEGORIES,
    model: str = "omni-moderation-latest",
) -> ModerationResult:
    """Run ``text`` through the moderation endpoint.

    Parameters
    ----------
    text:
        Model-generated natural-language output to be shown to a user.
        Empty strings short-circuit to ``blocked=False`` — there is
        nothing to moderate.
    client:
        An ``OpenAI()`` instance (or any object exposing
        ``.moderations.create(model=..., input=...)``). When ``None``,
        the function returns ``blocked=False`` without calling out —
        moderation is off unless wired.
    blocked_categories:
        Category names to block. Defaults to
        :data:`DEFAULT_BLOCKED_CATEGORIES`. Any flagged category
        matching this set causes ``blocked=True``.
    """
    if not text.strip() or client is None:
        return ModerationResult(blocked=False, reason="no-op (empty or no client)")

    try:
        response = client.moderations.create(model=model, input=text)
    except Exception as exc:  # moderation must never hard-fail the pipeline
        logger.warning("moderation call failed: %s", exc)
        return ModerationResult(blocked=False, reason=f"moderation error: {exc}")

    results = getattr(response, "results", None) or []
    if not results:
        return ModerationResult(blocked=False, reason="empty moderation response")

    first = results[0]
    flagged_obj = getattr(first, "categories", None)
    flagged: dict[str, bool] = {}
    if flagged_obj is not None:
        if hasattr(flagged_obj, "model_dump"):
            flagged = {str(k): bool(v) for k, v in flagged_obj.model_dump().items()}
        elif isinstance(flagged_obj, dict):
            flagged = {str(k): bool(v) for k, v in flagged_obj.items()}
        else:
            # Best-effort dict view of SDK objects
            flagged = {
                k: bool(getattr(flagged_obj, k))
                for k in dir(flagged_obj)
                if not k.startswith("_") and isinstance(getattr(flagged_obj, k, False), bool)
            }

    blocked_set = set(blocked_categories)
    hit = tuple(name for name, v in flagged.items() if v and name in blocked_set)
    return ModerationResult(
        blocked=bool(hit),
        categories=hit,
        reason=f"moderation flagged: {', '.join(hit)}" if hit else "clean",
        raw=response,
    )


__all__ = ["DEFAULT_BLOCKED_CATEGORIES", "ModerationResult", "moderate_output"]
