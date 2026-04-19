"""Structured JSONL telemetry for LLM calls.

Every successful or failed call writes one JSONL line to
``<log_dir>/<YYYY-MM-DD>.jsonl`` (UTC-date bucket). The default
``<log_dir>`` is ``<repo>/logs/llm``.

Prompt and completion content are *never* written by default — the
logger only records token counts, latency, cost, and success/error
metadata. Passing ``log_content=True`` opts into content logging; this
is intended for local debugging only and must never be enabled in
production.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = _REPO_ROOT / "logs" / "llm"


class TelemetryLogger:
    """Append-only JSONL writer for LLM-call records. Thread-safe."""

    def __init__(
        self,
        log_dir: Path | str | None = None,
        *,
        log_content: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        self.log_content = log_content
        self._lock = threading.Lock()

    def _path_for_today(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"{today}.jsonl"

    def log_call(
        self,
        *,
        request_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        cost_usd: float,
        temperature: float,
        success: bool,
        error_class: str | None = None,
        fallback_used: bool = False,
        prompt_content: Any = None,
        completion_content: Any = None,
    ) -> Path:
        """Append one record. Returns the file path it was written to."""
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "cost_usd": round(cost_usd, 6),
            "temperature": temperature,
            "success": success,
            "error_class": error_class,
            "fallback_used": fallback_used,
        }
        if self.log_content:
            record["prompt"] = prompt_content
            record["completion"] = completion_content

        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        path = self._path_for_today()

        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        return path
