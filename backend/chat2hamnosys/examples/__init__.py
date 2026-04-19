"""Recorded example sessions and a replay CLI.

Each ``*.json`` file under this directory is a complete authoring session
captured as the orchestrator's append-only event log plus the final
:class:`SignEntry`. Two purposes:

1. **Demo fixtures.** ``python -m examples.replay --example electron``
   streams the events back through the live API so the avatar plays the
   sign exactly as it would have during a real session. Useful when the
   conference Wi-Fi dies and you need a deterministic stand-in.
2. **Golden tests.** Each fixture pins the expected ``hamnosys`` and
   ``sigml`` strings the system produced when the session was recorded.
   Re-running the events through the current pipeline must reproduce the
   same final entry; if it doesn't, either the LLM has drifted or a
   prompt has changed (see :mod:`prompts.eval`) or the validator's
   semantics have changed (see :mod:`hamnosys.validator`). The replay CLI
   prints a diff in either case.
"""

from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent

EXAMPLE_NAMES = (
    "electron",
    "catalyst",
    "photon",
    "derivative",
    "integral",
)


def example_path(name: str) -> Path:
    """Resolve an example name to its JSON path. Raises if missing."""
    if name not in EXAMPLE_NAMES:
        raise ValueError(
            f"unknown example {name!r}; valid: {', '.join(EXAMPLE_NAMES)}"
        )
    path = EXAMPLES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"example fixture missing: {path}")
    return path
