"""One-shot CLI to sanity-check the LLM wrapper.

Usage::

    python -m backend.chat2hamnosys.llm.ping "hello"

Sends the argv-joined text as a single user message through
:class:`LLMClient` with defaults and prints the completion plus a
cost/token summary. Useful for confirming that ``OPENAI_API_KEY`` is
set and the wrapper is wired up correctly. Not a production entry
point.
"""

from __future__ import annotations

import sys
import uuid

from .budget import BudgetExceeded
from .client import ChatResult, LLMClient, LLMConfigError


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or any(a in {"-h", "--help"} for a in args):
        sys.stderr.write(__doc__ or "")
        sys.stderr.write("\n")
        return 0

    prompt = " ".join(args)

    try:
        client = LLMClient()
    except LLMConfigError as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return 2

    request_id = f"ping-{uuid.uuid4().hex[:8]}"
    try:
        result: ChatResult = client.chat(
            messages=[{"role": "user", "content": prompt}],
            request_id=request_id,
        )
    except BudgetExceeded as exc:
        sys.stderr.write(f"budget exceeded: {exc}\n")
        return 3
    except Exception as exc:  # noqa: BLE001 — ping is best-effort diagnostic
        sys.stderr.write(f"call failed: {type(exc).__name__}: {exc}\n")
        return 1

    sys.stdout.write(result.content)
    sys.stdout.write("\n")
    sys.stderr.write(
        f"-- model={result.model_used} "
        f"tokens=(in={result.input_tokens}, out={result.output_tokens}) "
        f"cost=${result.cost_usd:.5f} latency={result.latency_ms}ms "
        f"fallback={result.fallback_used} request_id={result.request_id}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
