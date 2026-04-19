"""Correction interpreter — point-and-fix semantics for rendered signs.

After the author sees the rendered avatar they can flag what is wrong
("the handshape at 2 seconds is wrong; it should be a flat-O"). The
:mod:`.correction_interpreter` module translates those free-form
corrections into **minimal diffs** on the session's
:class:`PartialSignParameters` and re-runs the generator + validator +
renderer chain so only the intended parts change.

Public API:

- :class:`Correction` — the user's correction request shape.
- :class:`FieldChange` / :class:`CorrectionIntent` /
  :class:`CorrectionPlan` — the interpreter's output.
- :func:`interpret_correction` — session + correction → plan.
- :func:`apply_correction` — plan → new session (with regeneration).
- :func:`log_correction_applied` / :func:`log_session_accepted` —
  metric emitters for the "average corrections per accepted sign"
  dashboard.
"""

from .correction_interpreter import (
    ApplyOutcome,
    Correction,
    CorrectionApplyError,
    CorrectionIntent,
    CorrectionPlan,
    FieldChange,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    apply_correction,
    interpret_correction,
)
from .metrics import (
    DEFAULT_LOG_DIR,
    log_correction_applied,
    log_session_accepted,
)

__all__ = [
    "ApplyOutcome",
    "Correction",
    "CorrectionApplyError",
    "CorrectionIntent",
    "CorrectionPlan",
    "DEFAULT_LOG_DIR",
    "FieldChange",
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
    "apply_correction",
    "interpret_correction",
    "log_correction_applied",
    "log_session_accepted",
]
