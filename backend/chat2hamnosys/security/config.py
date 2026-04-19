"""Security configuration loaded from environment variables.

One :class:`SecurityConfig` object is built once per process via
:func:`load_security_config` and threaded through the API layer. All
fields have safe defaults, so the module is callable in tests without
setting anything.

Env var contract (all optional):

- ``CHAT2HAMNOSYS_MAX_INPUT_LEN`` — max characters per prose /
  correction / answer string. Default: ``2000``.
- ``CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD`` — per-IP daily spend cap.
  Default: ``10.0``.
- ``CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD`` — process-wide daily ceiling.
  Default: ``200.0``.
- ``CHAT2HAMNOSYS_PII_POLICY`` — ``"hashed"`` (default) or
  ``"plaintext"``. Plaintext is reserved for research studies with
  IRB approval.
- ``CHAT2HAMNOSYS_SIGNER_ID_SALT`` — HMAC salt used when the policy
  hashes. Default: ``"chat2hamnosys-default-salt"`` (unsafe for
  production; log a warning on startup if unchanged).
- ``CHAT2HAMNOSYS_ENABLE_INJECTION_CLASSIFIER`` — ``"0"`` disables the
  LLM-backed injection classifier, leaving only the regex screen.
  Default: ``"1"`` (enabled).
- ``CHAT2HAMNOSYS_ENABLE_OUTPUT_MODERATION`` — ``"0"`` disables the
  OpenAI moderation call. Default: ``"1"``.
- ``CHAT2HAMNOSYS_ANOMALY_ALERT_URL`` — optional webhook URL for
  anomaly alerts. Unset means log-only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal


logger = logging.getLogger(__name__)


PIIPolicy = Literal["hashed", "plaintext"]


DEFAULT_SIGNER_SALT = "chat2hamnosys-default-salt"


@dataclass(frozen=True)
class SecurityConfig:
    """Immutable snapshot of the security-relevant env config."""

    max_input_len: int = 2000
    per_ip_daily_cap_usd: float = 10.0
    global_daily_cap_usd: float = 200.0
    pii_policy: PIIPolicy = "hashed"
    signer_id_salt: str = DEFAULT_SIGNER_SALT
    enable_injection_classifier: bool = True
    enable_output_moderation: bool = True
    anomaly_alert_url: str | None = None


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid integer") from exc
    if value <= 0:
        raise ValueError(f"{name}={value} must be positive")
    return value


def _read_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid float") from exc
    if value <= 0:
        raise ValueError(f"{name}={value} must be positive")
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _read_pii_policy(name: str, default: PIIPolicy) -> PIIPolicy:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value not in ("hashed", "plaintext"):
        raise ValueError(
            f"{name}={raw!r} must be 'hashed' or 'plaintext'"
        )
    return value  # type: ignore[return-value]


def load_security_config() -> SecurityConfig:
    """Build a :class:`SecurityConfig` from the current env.

    Logs a warning if the HMAC salt is still the default, or if
    ``pii_policy="plaintext"`` is in effect — both are situations an
    operator should know about.
    """
    salt = os.environ.get("CHAT2HAMNOSYS_SIGNER_ID_SALT", DEFAULT_SIGNER_SALT)
    if salt == DEFAULT_SIGNER_SALT:
        logger.warning(
            "CHAT2HAMNOSYS_SIGNER_ID_SALT is unset; using built-in default. "
            "Set a per-deployment salt before handling real signer ids."
        )
    policy = _read_pii_policy("CHAT2HAMNOSYS_PII_POLICY", "hashed")
    if policy == "plaintext":
        logger.warning(
            "CHAT2HAMNOSYS_PII_POLICY=plaintext — signer identifiers will be "
            "stored in the clear. Ensure an IRB-approved study or equivalent "
            "authorization is in place."
        )

    return SecurityConfig(
        max_input_len=_read_int("CHAT2HAMNOSYS_MAX_INPUT_LEN", 2000),
        per_ip_daily_cap_usd=_read_float(
            "CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD", 10.0
        ),
        global_daily_cap_usd=_read_float(
            "CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD", 200.0
        ),
        pii_policy=policy,
        signer_id_salt=salt,
        enable_injection_classifier=_read_bool(
            "CHAT2HAMNOSYS_ENABLE_INJECTION_CLASSIFIER", True
        ),
        enable_output_moderation=_read_bool(
            "CHAT2HAMNOSYS_ENABLE_OUTPUT_MODERATION", True
        ),
        anomaly_alert_url=os.environ.get("CHAT2HAMNOSYS_ANOMALY_ALERT_URL"),
    )


__all__ = [
    "DEFAULT_SIGNER_SALT",
    "PIIPolicy",
    "SecurityConfig",
    "load_security_config",
]
