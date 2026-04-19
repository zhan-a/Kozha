"""Re-record parser fixtures against the real OpenAI API.

Usage::

    # record all fixtures
    OPENAI_API_KEY=sk-... python -m parser.record_fixtures

    # record a single fixture
    OPENAI_API_KEY=sk-... python -m parser.record_fixtures 11-ambiguous-claw

Writes each fixture's ``recorded_response`` field in place with whatever
``gpt-4o`` returns for that ``prose``. The ``expected_populated`` and
``expected_gap_fields`` oracle fields are **not** touched — they remain
the author-defined ground truth. After a record run, re-run the tests:
any oracle miss surfaces as a failed ``test_fixture_replay`` case and
feeds into the eval numbers in
``docs/chat2hamnosys/05-description-parser-eval.md``.

A short summary (fixtures recorded, total cost) prints at the end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm import LLMClient

from .description_parser import parse_description


FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "parser"
)


def _pick_fixtures(only_id: str | None) -> list[Path]:
    files = sorted(FIXTURES_DIR.glob("*.json"))
    if only_id is None:
        return files
    match = [f for f in files if f.stem == only_id]
    if not match:
        raise SystemExit(
            f"no fixture matches id {only_id!r}. Available: "
            f"{[f.stem for f in files]}"
        )
    return match


def _record_one(path: Path, client: LLMClient) -> tuple[bool, float]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    prose = fixture["prose"]
    print(f"  [{fixture['id']}] calling LLM... ", end="", flush=True)
    result = parse_description(
        prose,
        client=client,
        request_id=f"record-{fixture['id']}",
    )
    # raw_response is the exact JSON string the LLM returned; we
    # re-decode it so the stored recording is pretty-printed.
    recorded = json.loads(result.raw_response)
    fixture["recorded_response"] = recorded
    path.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("ok")
    return True, client.budget.spent


def main() -> int:
    only_id = sys.argv[1] if len(sys.argv) > 1 else None
    files = _pick_fixtures(only_id)
    client = LLMClient(model="gpt-4o")
    print(f"Recording {len(files)} fixture(s) against {client.model!r}:")
    prev_spent = 0.0
    for fp in files:
        _record_one(fp, client)
    total = client.budget.spent - prev_spent
    print(f"\nDone. Session spend: ${client.budget.spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
