"""Prompt eval harness — fixture-replay mode.

Re-runs every recorded fixture against the production pipeline with a
replay LLM client (returns the fixture's ``recorded_response``), checks
the oracle assertions each fixture carries, and writes a JSON report
with per-fixture pass/fail, aggregate accuracy, and estimated token /
cost usage.

Token counts and dollar cost are **estimates** — real API calls are not
made in replay mode. The estimate is ``len(text) // 4`` tokens per
message, costed through :func:`llm.pricing.compute_cost` at the model
declared in the prompt's frontmatter.

Usage
-----
Run from ``backend/chat2hamnosys``::

    python3 -m prompts.eval parse_description
    python3 -m prompts.eval parse_description_v1          # explicit version
    python3 -m prompts.eval interpret_correction --version v1

Reports land at
``prompts/eval_results/<prompt_id>/<version>/<YYYY-MM-DDTHHMMSSZ>.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from llm import ChatResult
from llm.pricing import compute_cost

from .loader import PROMPTS_DIR, PromptMetadata, load_prompt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


_REPO_ROOT: Path = PROMPTS_DIR.parent  # backend/chat2hamnosys
_FIXTURES_ROOT: Path = _REPO_ROOT / "tests" / "fixtures"
EVAL_RESULTS_DIR: Path = PROMPTS_DIR / "eval_results"


# ---------------------------------------------------------------------------
# Replay LLM client
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token count from character length — one token ≈ 4 chars."""
    return max(1, len(text) // 4) if text else 0


class _ReplayClient:
    """Stand-in for :class:`LLMClient` that replays a recorded response.

    Mirrors the ``FakeLLMClient`` stubs used by the unit tests but also
    tracks per-call token estimates and costs so the harness can
    aggregate usage across a fixture suite.
    """

    def __init__(self, content: str, *, model: str = "gpt-4o") -> None:
        self.content = content
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self.results: list[ChatResult] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        *,
        request_id: str,
        prompt_metadata: Any = None,
    ) -> ChatResult:
        input_text = "\n".join(
            str(m.get("content", "")) for m in messages
        )
        in_tokens = _estimate_tokens(input_text)
        out_tokens = _estimate_tokens(self.content)
        cost = compute_cost(self.model, in_tokens, out_tokens)
        result = ChatResult(
            content=self.content,
            tool_calls=[],
            model_used=self.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            cost_usd=cost,
            latency_ms=0,
            request_id=request_id,
        )
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "response_format": response_format,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "request_id": request_id,
                "prompt_metadata": prompt_metadata,
            }
        )
        self.results.append(result)
        return result


# ---------------------------------------------------------------------------
# Eval records
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One fixture's outcome."""

    fixture_id: str
    passed: bool
    reason: str = ""
    llm_called: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class EvalReport:
    """Aggregate report written to disk."""

    prompt_id: str
    version: str
    prompt_hash: str
    model: str
    timestamp: str
    fixture_count: int
    pass_count: int
    fail_count: int
    accuracy: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    notes: str = ""
    cases: list[EvalCase] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-prompt runners
# ---------------------------------------------------------------------------


# Runner signature: (fixture_dict, metadata) -> (EvalCase without token fields, client)
# The caller augments EvalCase with token/cost from the client's results.
Runner = Callable[[dict[str, Any], PromptMetadata], tuple[EvalCase, _ReplayClient | None]]


def _run_parse_description(
    fixture: dict[str, Any], metadata: PromptMetadata
) -> tuple[EvalCase, _ReplayClient | None]:
    """Replay one parser fixture and verify the expected slots / gaps."""
    # Local import — avoids paying parser module init cost when this
    # runner isn't used by the selected prompt id.
    from parser import parse_description

    fid = fixture["id"]
    prose = fixture["prose"]
    expected_populated = set(fixture["expected_populated"])
    expected_gap_fields = set(fixture["expected_gap_fields"])

    content = json.dumps(fixture["recorded_response"])
    client = _ReplayClient(content=content, model=metadata.model)
    try:
        result = parse_description(prose, client=client)
    except Exception as exc:
        return (
            EvalCase(
                fixture_id=fid,
                passed=False,
                reason=f"raised {type(exc).__name__}: {exc}",
            ),
            client,
        )

    populated: set[str] = set()
    for key, val in result.parameters.model_dump().items():
        if val is None:
            continue
        if key == "movement" and not val:
            continue
        populated.add(key)
    actual_gaps = {g.field for g in result.gaps}

    problems: list[str] = []
    missing = expected_populated - populated
    if missing:
        problems.append(f"missing populated slots: {sorted(missing)}")
    extra = populated - expected_populated
    if extra:
        problems.append(f"unexpected populated slots: {sorted(extra)}")
    if actual_gaps != expected_gap_fields:
        problems.append(
            f"gap fields mismatch: expected {sorted(expected_gap_fields)}, "
            f"got {sorted(actual_gaps)}"
        )

    passed = not problems
    return (
        EvalCase(fixture_id=fid, passed=passed, reason="; ".join(problems)),
        client,
    )


def _run_generate_clarification(
    fixture: dict[str, Any], metadata: PromptMetadata
) -> tuple[EvalCase, _ReplayClient | None]:
    """Replay one clarifier fixture and verify the produced questions."""
    from clarify import generate_questions
    from models import ClarificationTurn
    from parser import ParseResult

    fid = fixture["id"]
    parse_result = ParseResult.model_validate(fixture["parse_result"])
    prior = [
        ClarificationTurn.model_validate(t) for t in fixture.get("prior_turns", [])
    ]
    is_deaf_native = fixture.get("is_deaf_native")

    if fixture.get("expect_fallback"):
        content = fixture["recorded_response_raw"]
    else:
        content = json.dumps(fixture["recorded_response"])

    client = _ReplayClient(content=content, model=metadata.model)
    try:
        questions = generate_questions(
            parse_result, prior, client=client, is_deaf_native=is_deaf_native
        )
    except Exception as exc:
        return (
            EvalCase(
                fixture_id=fid,
                passed=False,
                reason=f"raised {type(exc).__name__}: {exc}",
            ),
            client,
        )

    expected_fields = fixture["expected_fields"]
    expected_count = fixture["expected_count"]
    actual_fields = [q.field for q in questions]

    problems: list[str] = []
    if len(questions) != expected_count:
        problems.append(
            f"expected {expected_count} questions, got {len(questions)}"
        )
    if actual_fields != expected_fields:
        problems.append(
            f"expected fields {expected_fields}, got {actual_fields}"
        )

    passed = not problems
    return (
        EvalCase(fixture_id=fid, passed=passed, reason="; ".join(problems)),
        client,
    )


def _run_interpret_correction(
    fixture: dict[str, Any], metadata: PromptMetadata
) -> tuple[EvalCase, _ReplayClient | None]:
    """Replay one correction fixture and verify the produced plan."""
    from correct import CorrectionIntent, interpret_correction
    from parser.models import PartialSignParameters
    from session.orchestrator import Correction
    from session.state import AuthoringSession, SessionState, SignEntryDraft

    fid = fixture["id"]
    params = PartialSignParameters.model_validate(fixture["current_params"])
    draft = SignEntryDraft(
        gloss="TEST",
        sign_language="bsl",
        author_signer_id="alice",
        description_prose=fixture["original_prose"],
        parameters_partial=params,
        hamnosys=fixture["current_hamnosys"],
    )
    session = AuthoringSession(
        draft=draft, state=SessionState.APPLYING_CORRECTION
    )
    correction = Correction(
        raw_text=fixture["correction"]["raw_text"],
        target_time_ms=fixture["correction"]["target_time_ms"],
        target_region=fixture["correction"]["target_region"],
    )
    content = json.dumps(fixture["recorded_response"])
    client = _ReplayClient(content=content, model=metadata.model)
    try:
        plan = interpret_correction(session, correction, client=client)
    except Exception as exc:
        return (
            EvalCase(
                fixture_id=fid,
                passed=False,
                reason=f"raised {type(exc).__name__}: {exc}",
            ),
            client,
        )

    expected_intent = fixture["expected_intent"]
    expected_paths = fixture["expected_paths"]
    expected_slots = fixture["expected_updated_slots"]
    expected_follow_up = fixture["expected_follow_up_present"]
    expected_needs_conf = fixture["expected_needs_confirmation"]
    expect_llm = fixture["expect_llm_called"]

    problems: list[str] = []
    if expect_llm and not client.calls:
        problems.append("expected LLM call but none made")
    if not expect_llm and client.calls:
        problems.append("unexpected LLM call for a no-call fixture")
    if plan.intent.value != expected_intent:
        problems.append(
            f"expected intent {expected_intent!r}, got {plan.intent.value!r}"
        )
    actual_paths = [c.path for c in plan.field_changes]
    if actual_paths != expected_paths:
        problems.append(
            f"expected paths {expected_paths}, got {actual_paths}"
        )
    if plan.needs_user_confirmation != expected_needs_conf:
        problems.append(
            f"needs_user_confirmation mismatch: expected {expected_needs_conf}"
        )
    if bool(plan.follow_up_question) != expected_follow_up:
        problems.append(
            f"follow_up_question presence mismatch: "
            f"expected {expected_follow_up}, got {bool(plan.follow_up_question)}"
        )
    if plan.intent is CorrectionIntent.APPLY_DIFF:
        if plan.updated_params is None:
            problems.append("APPLY_DIFF plan missing updated_params")
        else:
            dumped = plan.updated_params.model_dump(mode="json")
            for key, exp in expected_slots.items():
                if dumped.get(key) != exp:
                    problems.append(
                        f"slot {key!r} expected {exp!r}, got {dumped.get(key)!r}"
                    )

    passed = not problems
    return (
        EvalCase(
            fixture_id=fid, passed=passed, reason="; ".join(problems)
        ),
        client,
    )


def _run_generate_hamnosys_fallback(
    fixture: dict[str, Any], metadata: PromptMetadata
) -> tuple[EvalCase, _ReplayClient | None]:
    """No dedicated fixtures exist for the fallback prompt.

    Fallback codepoint picks are exercised transitively by the generator
    fixtures only when the deterministic composer misses — the current
    generator fixtures are all deterministic hits. This runner returns a
    placeholder EvalCase so the harness still produces a well-formed
    report.
    """
    return (
        EvalCase(
            fixture_id=fixture.get("id", "n/a"),
            passed=True,
            reason="no fixtures exercise this prompt directly",
        ),
        None,
    )


def _run_generate_hamnosys_repair(
    fixture: dict[str, Any], metadata: PromptMetadata
) -> tuple[EvalCase, _ReplayClient | None]:
    """Same situation as the fallback prompt — repair path has no direct fixtures."""
    return (
        EvalCase(
            fixture_id=fixture.get("id", "n/a"),
            passed=True,
            reason="no fixtures exercise this prompt directly",
        ),
        None,
    )


# (fixture_dir_name_or_None, runner)
_RUNNERS: dict[str, tuple[str | None, Runner]] = {
    "parse_description": ("parser", _run_parse_description),
    "generate_clarification": ("clarify", _run_generate_clarification),
    "interpret_correction": ("correct", _run_interpret_correction),
    "generate_hamnosys_fallback": (None, _run_generate_hamnosys_fallback),
    "generate_hamnosys_repair": (None, _run_generate_hamnosys_repair),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_ID_VERSION_RE = re.compile(r"^(?P<id>.+?)_v(?P<ver>\d+)$")


def _split_id_and_version(
    spec: str, explicit_version: str | None
) -> tuple[str, str]:
    """Accept ``parse_description`` or ``parse_description_v1`` for the CLI arg."""
    match = _ID_VERSION_RE.match(spec)
    if match:
        pid = match.group("id")
        parsed_ver = f"v{int(match.group('ver'))}"
        if explicit_version and explicit_version != parsed_ver:
            raise ValueError(
                f"conflicting versions: spec says {parsed_ver}, "
                f"--version says {explicit_version}"
            )
        return pid, parsed_ver
    return spec, explicit_version or "latest"


def _load_fixtures(fixture_dir_name: str) -> list[dict[str, Any]]:
    directory = _FIXTURES_ROOT / fixture_dir_name
    if not directory.is_dir():
        raise FileNotFoundError(f"fixture directory not found: {directory}")
    files = sorted(p for p in directory.glob("*.json") if not p.name.startswith("_"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def run_eval(prompt_id: str, version: str = "latest") -> EvalReport:
    """Run every fixture for ``prompt_id`` and return an :class:`EvalReport`."""
    if prompt_id not in _RUNNERS:
        raise KeyError(
            f"no eval runner registered for prompt id {prompt_id!r}. "
            f"Known: {sorted(_RUNNERS)}"
        )
    prompt = load_prompt(prompt_id, version)
    resolved_version = prompt.metadata.version
    fixture_dir_name, runner = _RUNNERS[prompt_id]

    cases: list[EvalCase] = []
    notes = ""
    if fixture_dir_name is None:
        notes = (
            "no dedicated fixtures for this prompt — report contains no cases"
        )
    else:
        fixtures = _load_fixtures(fixture_dir_name)
        for fixture in fixtures:
            case, client = runner(fixture, prompt.metadata)
            if client is not None and client.results:
                case = EvalCase(
                    fixture_id=case.fixture_id,
                    passed=case.passed,
                    reason=case.reason,
                    llm_called=True,
                    input_tokens=sum(r.input_tokens for r in client.results),
                    output_tokens=sum(r.output_tokens for r in client.results),
                    cost_usd=sum(r.cost_usd for r in client.results),
                )
            cases.append(case)

    pass_count = sum(1 for c in cases if c.passed)
    fail_count = len(cases) - pass_count
    accuracy = (pass_count / len(cases)) if cases else 0.0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return EvalReport(
        prompt_id=prompt_id,
        version=resolved_version,
        prompt_hash=prompt.metadata.hash,
        model=prompt.metadata.model,
        timestamp=timestamp,
        fixture_count=len(cases),
        pass_count=pass_count,
        fail_count=fail_count,
        accuracy=accuracy,
        total_input_tokens=sum(c.input_tokens for c in cases),
        total_output_tokens=sum(c.output_tokens for c in cases),
        total_cost_usd=sum(c.cost_usd for c in cases),
        notes=notes,
        cases=cases,
    )


def save_report(report: EvalReport, base_dir: Path = EVAL_RESULTS_DIR) -> Path:
    """Persist a report to ``base_dir/<id>/<version>/<timestamp>.json``."""
    target_dir = base_dir / report.prompt_id / report.version
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{report.timestamp}.json"
    path.write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_summary(report: EvalReport) -> str:
    header = (
        f"{report.prompt_id} {report.version} "
        f"[hash {report.prompt_hash[:12]}…] — "
        f"{report.pass_count}/{report.fixture_count} passed "
        f"({report.accuracy:.1%}) · "
        f"~{report.total_input_tokens} in / ~{report.total_output_tokens} out tokens · "
        f"~${report.total_cost_usd:.4f}"
    )
    lines = [header]
    if report.notes:
        lines.append(f"  note: {report.notes}")
    for case in report.cases:
        marker = "PASS" if case.passed else "FAIL"
        line = f"  {marker}  {case.fixture_id}"
        if not case.passed and case.reason:
            line += f"  — {case.reason}"
        lines.append(line)
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompts.eval",
        description="Replay fixtures against a versioned prompt and report accuracy.",
    )
    parser.add_argument(
        "prompt_id",
        help=(
            "Prompt id to evaluate. Accepts either the bare id "
            "(e.g. 'parse_description') or the file-stem form "
            "(e.g. 'parse_description_v1')."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Pin a specific version (e.g. 'v1'). Defaults to 'latest'.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the JSON report to disk.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    prompt_id, version = _split_id_and_version(args.prompt_id, args.version)
    report = run_eval(prompt_id, version)
    print(_format_summary(report))
    if not args.no_save:
        path = save_report(report)
        print(f"saved report → {path.relative_to(_REPO_ROOT)}")
    return 0 if report.fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
