"""Fixture runner — drives one fixture through the pipeline.

Given a :class:`GoldenFixture`, :func:`run_fixture` returns a
:class:`FixtureResult` holding the parser / generator / end-to-end /
cost numbers for that single row. :func:`run_suite` glues the runner
to the CLI — iterates fixtures, dispatches ablations, aggregates
metrics, and emits a :class:`SuiteReport`.

Three seams the caller can tweak:

- ``make_client`` — factory for the :class:`LLMClient`. Tests inject a
  stub; the CLI defers to :func:`_default_make_client` which builds a
  real client from :envvar:`OPENAI_API_KEY`.
- ``ablations`` — list of :class:`AblationConfig`; defaults to every
  preset so a single run measures the full system *and* every
  ablation.
- ``on_fixture`` — optional progress callback; used by the CLI to
  print a dot per fixture.

Every LLM call made during a run carries ``request_id`` starting
``eval:<run_id>:<fixture_id>:<stage>`` so the JSONL telemetry log can
be joined back to the eval run after the fact. The runner also sums
per-call cost/latency on the fly so the caller doesn't need to read
the log.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from clarify import Option, Question, apply_answer, generate_questions
from clarify.answer_parser import AnswerParseError
from generator import GenerateResult, generate
from hamnosys import normalize, validate
from llm import ChatResult, LLMClient
from models import ClarificationTurn
from parser import (
    Gap,
    ParseResult,
    PartialMovementSegment,
    PartialNonManualFeatures,
    PartialSignParameters,
    parse_description,
)
from prompts import list_prompts

from .ablations import AblationConfig, ablation_presets, apply_ablation
from .metrics import OverallMetrics, aggregate, compute_slices
from .models import (
    EvalResult,
    FixtureResult,
    GoldenFixture,
    SuiteReport,
    _ALL_ABLATIONS,
)
from .simulator import NoAnswerAvailable, SimulatedAnswerer


logger = logging.getLogger(__name__)


LLMClientFactory = Callable[[], LLMClient]


# ---------------------------------------------------------------------------
# Per-call cost accounting
# ---------------------------------------------------------------------------


@dataclass
class _CallTally:
    """Accumulates cost/latency stats across the LLM calls of one fixture.

    The runner wraps the injected :class:`LLMClient` with a thin
    recorder that calls ``client.chat()`` and then appends to the
    tally. Saves a second pass over the telemetry JSONL.
    """

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    n_calls: int = 0

    def record(self, result: ChatResult) -> None:
        self.total_prompt_tokens += result.input_tokens
        self.total_completion_tokens += result.output_tokens
        self.total_tokens += result.total_tokens
        self.total_cost_usd += result.cost_usd
        self.total_latency_ms += result.latency_ms
        self.n_calls += 1


class _TallyingClient:
    """Wraps an :class:`LLMClient` to record per-call cost/latency.

    Implements :meth:`chat` / :meth:`stream_chat` by delegating to the
    wrapped client. Importantly, it also preserves public attributes
    the rest of the code reads (``budget``, ``telemetry``, ``model``)
    by ``__getattr__``-ing through to the delegate.
    """

    def __init__(self, delegate: LLMClient, tally: _CallTally) -> None:
        self._delegate = delegate
        self._tally = tally

    def chat(self, *args: Any, **kwargs: Any) -> ChatResult:
        result = self._delegate.chat(*args, **kwargs)
        self._tally.record(result)
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._delegate, item)


# ---------------------------------------------------------------------------
# Fixture I/O
# ---------------------------------------------------------------------------


def load_suite(suite_path: Path) -> list[GoldenFixture]:
    """Parse a ``.jsonl`` suite file into :class:`GoldenFixture` rows."""
    import json

    fixtures: list[GoldenFixture] = []
    with suite_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{suite_path}:{lineno}: invalid JSON: {exc}"
                ) from exc
            fixtures.append(GoldenFixture.from_dict(data))
    return fixtures


def resolve_suite_path(suite: str) -> Path:
    """Resolve a suite name like ``"golden_signs"`` to a fixture path."""
    if suite.endswith(".jsonl"):
        candidate = Path(suite)
        if candidate.is_file():
            return candidate
    base = Path(__file__).resolve().parent / "fixtures"
    candidate = base / f"{suite}.jsonl"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"suite {suite!r} not found at {candidate} — pass the absolute path "
            f"or place the .jsonl under backend/chat2hamnosys/eval/fixtures/"
        )
    return candidate


# ---------------------------------------------------------------------------
# Parameter expansion
# ---------------------------------------------------------------------------


def _build_partial_params(raw: dict[str, Any]) -> PartialSignParameters:
    """Materialize the fixture's ``expected_parameters`` dict.

    Mirrors :func:`generator.eval._build_params` but keeps the
    non-manual features too — the eval harness needs them to evaluate
    parser accuracy on non-manual fixtures.
    """
    movement = [
        PartialMovementSegment(**m) for m in raw.get("movement") or []
    ]
    nm = raw.get("non_manual")
    nm_obj = PartialNonManualFeatures(**nm) if isinstance(nm, dict) else None
    return PartialSignParameters(
        handshape_dominant=raw.get("handshape_dominant"),
        handshape_nondominant=raw.get("handshape_nondominant"),
        orientation_extended_finger=raw.get("orientation_extended_finger"),
        orientation_palm=raw.get("orientation_palm"),
        location=raw.get("location"),
        contact=raw.get("contact"),
        movement=movement,
        non_manual=nm_obj,
    )


def _populated_fields(params: PartialSignParameters | dict[str, Any]) -> set[str]:
    """Which slots carry a populated value — matches parser.eval bookkeeping."""
    if isinstance(params, PartialSignParameters):
        d = params.model_dump()
    else:
        d = params
    out: set[str] = set()
    for name in (
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    ):
        v = d.get(name)
        if v:
            out.add(name)
    if d.get("movement"):
        out.add("movement")
    nm = d.get("non_manual")
    if isinstance(nm, dict) and any(v for v in nm.values()):
        out.add("non_manual")
    return out


def _gap_fields(gaps: list[Gap] | list[dict]) -> set[str]:
    out: set[str] = set()
    for g in gaps or []:
        if isinstance(g, Gap):
            out.add(g.field)
        elif isinstance(g, dict) and "field" in g:
            out.add(g["field"])
    return out


def _parser_field_accuracy(
    expected: dict[str, Any],
    actual: PartialSignParameters,
) -> dict[str, bool]:
    """Per-field ``did we match?`` map for the parser stage.

    A slot counts as correct when (a) neither side populates it, or
    (b) both populate it with equal normalized values. For
    multi-term slots (movement, non_manual) we compare the canonical
    first-segment path or the set of populated leaves respectively —
    good enough for aggregate accuracy, and the exact comparisons
    that matter for downstream generation still surface in the
    generator metrics.
    """
    actual_dict = actual.model_dump()
    out: dict[str, bool] = {}
    for name in (
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    ):
        out[name] = _canon(expected.get(name)) == _canon(actual_dict.get(name))

    exp_mov = expected.get("movement") or []
    act_mov = actual_dict.get("movement") or []
    exp_path = exp_mov[0].get("path") if exp_mov else None
    act_path = act_mov[0].get("path") if act_mov else None
    out["movement"] = _canon(exp_path) == _canon(act_path)

    exp_nm = expected.get("non_manual") or {}
    act_nm = actual_dict.get("non_manual") or {}
    exp_keys = {k for k, v in (exp_nm or {}).items() if v}
    act_keys = {k for k, v in (act_nm or {}).items() if v}
    out["non_manual"] = exp_keys == act_keys
    return out


def _canon(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# HamNoSys comparison
# ---------------------------------------------------------------------------


def _symbol_counts(expected: str, actual: str) -> tuple[int, int, int]:
    """Multiset-based TP/FP/FN for HamNoSys symbol sequences.

    Mirrors :func:`generator.eval._align_tokens` so parser-stage and
    generator-stage numbers are directly comparable with the existing
    reports.
    """
    from collections import Counter

    exp_c = Counter(expected)
    act_c = Counter(actual)
    tp = sum((exp_c & act_c).values())
    fp = sum(act_c.values()) - tp
    fn = sum(exp_c.values()) - tp
    return tp, max(fp, 0), max(fn, 0)


def _exact_match(actual: str, accept_set: set[str]) -> bool:
    if not actual:
        return False
    actual_n = normalize(actual)
    return any(normalize(c) == actual_n for c in accept_set)


# ---------------------------------------------------------------------------
# Per-fixture run
# ---------------------------------------------------------------------------


def run_fixture(
    fixture: GoldenFixture,
    *,
    ablation: AblationConfig,
    client: LLMClient,
    run_id: str = "eval",
) -> FixtureResult:
    """Run a single fixture through parser → generator → e2e.

    All three stages run inside the same ablation context, so for
    instance ``no_deterministic_map`` applies to both the generator
    stage (gold params in) and the end-to-end stage. This matches
    Prompt 16 §7's intent — a single ablation should show the effect
    on *every* layer it touches.
    """
    tally = _CallTally()
    wrapped = _TallyingClient(client, tally)

    result = FixtureResult(
        fixture_id=fixture.id,
        ablation=ablation.name,
    )
    expected_params_dict = fixture.expected_parameters or {}
    expected_populated = _populated_fields(expected_params_dict)
    result.parser_populated_expected = expected_populated

    with apply_ablation(ablation):
        _run_parser_stage(fixture, wrapped, run_id, result, expected_populated)
        _run_generator_stage(fixture, wrapped, run_id, result)
        _run_e2e_stage(fixture, wrapped, run_id, result, ablation)

    result.total_tokens = tally.total_tokens
    result.prompt_tokens = tally.total_prompt_tokens
    result.completion_tokens = tally.total_completion_tokens
    result.cost_usd = tally.total_cost_usd
    result.latency_ms = tally.total_latency_ms
    result.llm_calls = tally.n_calls
    return result


def _run_parser_stage(
    fixture: GoldenFixture,
    client: Any,
    run_id: str,
    result: FixtureResult,
    expected_populated: set[str],
) -> None:
    """Run the parser and score field accuracy + gap-detection."""
    try:
        parse_result: ParseResult = parse_description(
            fixture.prose_description,
            client=client,
            request_id=f"{run_id}:{fixture.id}:parse",
        )
    except Exception as exc:
        result.notes.append(f"parser raised {type(exc).__name__}: {exc}")
        result.parser_ok = False
        return

    result.parser_ok = True
    populated_actual = _populated_fields(parse_result.parameters)
    result.parser_populated_actual = populated_actual
    result.parser_populated_correct = populated_actual & expected_populated
    result.parser_field_accuracy = _parser_field_accuracy(
        fixture.expected_parameters, parse_result.parameters
    )

    # Expected gap fields = expected-populated slots the parser did
    # *not* populate; the parser should flag those as gaps. (We skip
    # slots the fixture leaves unpopulated — gap expectations derive
    # from the gold, not from what the parser guessed.)
    expected_gaps = expected_populated - populated_actual
    result.parser_gaps_expected = expected_gaps
    result.parser_gaps_actual = _gap_fields(parse_result.gaps)


def _run_generator_stage(
    fixture: GoldenFixture,
    client: Any,
    run_id: str,
    result: FixtureResult,
) -> None:
    """Run the generator against the fixture's gold parameters."""
    params = _build_partial_params(fixture.expected_parameters or {})
    try:
        gen: GenerateResult = generate(
            params,
            client=client,
            request_id=f"{run_id}:{fixture.id}:gen",
        )
    except Exception as exc:
        result.notes.append(f"generator raised {type(exc).__name__}: {exc}")
        result.gen_valid = False
        return

    hamnosys = gen.hamnosys
    result.gen_hamnosys = hamnosys
    result.gen_valid = bool(hamnosys and gen.validation.ok)
    if hamnosys:
        result.gen_exact_match = _exact_match(hamnosys, fixture.accept_set())
        # Compare against the *first-listed* expected string when
        # computing symbol-level counts — the variants are phonological
        # equivalents we already credited via exact-match.
        tp, fp, fn = _symbol_counts(fixture.expected_hamnosys, hamnosys)
        result.gen_symbol_tp = tp
        result.gen_symbol_fp = fp
        result.gen_symbol_fn = fn
    if not gen.validation.ok:
        result.notes.append(
            f"generator validation: {gen.validation.summary()[:200]}"
        )


def _run_e2e_stage(
    fixture: GoldenFixture,
    client: Any,
    run_id: str,
    result: FixtureResult,
    ablation: AblationConfig,
) -> None:
    """Run prose → parse → simulate → generate and score it."""
    try:
        parse_result: ParseResult = parse_description(
            fixture.prose_description,
            client=client,
            request_id=f"{run_id}:{fixture.id}:e2e:parse",
        )
    except Exception as exc:
        result.notes.append(f"e2e parser raised {type(exc).__name__}: {exc}")
        return

    params = parse_result.parameters
    questions_asked = 0
    clarification_turns = 0

    if not ablation.skip_clarifier and parse_result.gaps:
        simulator = SimulatedAnswerer(fixture.expected_parameters or {})
        dialogue: list[ClarificationTurn] = [
            ClarificationTurn(role="author", text=fixture.prose_description),
        ]
        # Loop at most 3 turns so a parser that keeps surfacing new gaps
        # doesn't cause the runner to spin forever.
        for turn_idx in range(3):
            try:
                questions = generate_questions(
                    parse_result,
                    dialogue,
                    client=client,
                    request_id=f"{run_id}:{fixture.id}:e2e:clarify:{turn_idx}",
                )
            except Exception as exc:
                result.notes.append(
                    f"e2e clarifier raised {type(exc).__name__}: {exc}"
                )
                break
            if not questions:
                break
            clarification_turns += 1
            for question in questions:
                questions_asked += 1
                try:
                    answer = simulator.answer(question)
                except NoAnswerAvailable as exc:
                    result.notes.append(
                        f"simulator: no answer for {exc.field}: {exc.reason}"
                    )
                    continue
                try:
                    params = apply_answer(params, question, answer)
                except AnswerParseError as exc:
                    result.notes.append(
                        f"simulator: could not apply answer for {question.field}: {exc}"
                    )
                    continue
                dialogue.append(
                    ClarificationTurn(role="assistant", text=question.text)
                )
                dialogue.append(
                    ClarificationTurn(role="author", text=answer)
                )
            # Re-evaluate gaps using the updated params. No second
            # parse call — we rebuild a ParseResult with the gaps
            # that are still unfilled.
            populated = _populated_fields(params)
            remaining_gaps = [
                g for g in parse_result.gaps if g.field not in populated
            ]
            if not remaining_gaps:
                break
            parse_result = ParseResult(
                parameters=params,
                gaps=remaining_gaps,
                raw_response=parse_result.raw_response,
            )

    result.e2e_clarification_turns = clarification_turns
    result.e2e_questions_asked = questions_asked

    try:
        gen: GenerateResult = generate(
            params,
            client=client,
            request_id=f"{run_id}:{fixture.id}:e2e:gen",
        )
    except Exception as exc:
        result.notes.append(f"e2e generator raised {type(exc).__name__}: {exc}")
        return

    hamnosys = gen.hamnosys
    result.e2e_hamnosys = hamnosys
    result.e2e_valid = bool(hamnosys and gen.validation.ok)
    if hamnosys:
        result.e2e_exact_match = _exact_match(hamnosys, fixture.accept_set())
        tp, fp, fn = _symbol_counts(fixture.expected_hamnosys, hamnosys)
        result.e2e_symbol_tp = tp
        result.e2e_symbol_fp = fp
        result.e2e_symbol_fn = fn
    if not gen.validation.ok:
        result.notes.append(
            f"e2e validation: {gen.validation.summary()[:200]}"
        )


# ---------------------------------------------------------------------------
# Suite-level run
# ---------------------------------------------------------------------------


def _default_make_client() -> LLMClient:
    return LLMClient()


def _prompt_versions_snapshot() -> dict[str, str]:
    """Record the latest version of every prompt the pipeline touches.

    Written into each ``EvalResult`` so a regression diff can tell us
    "oh right, this run used parse_description_v2".
    """
    try:
        versions = list_prompts()
    except Exception:  # pragma: no cover — defensive against filesystem oddities
        return {}
    return {pid: vers[-1] for pid, vers in versions.items() if vers}


def run_suite(
    suite_path: Path,
    *,
    ablations: Iterable[AblationConfig] | None = None,
    make_client: LLMClientFactory | None = None,
    on_fixture: Callable[[GoldenFixture, FixtureResult], None] | None = None,
    fixture_filter: Callable[[GoldenFixture], bool] | None = None,
    run_id: str | None = None,
) -> SuiteReport:
    """Run every fixture in ``suite_path`` under every ablation.

    ``make_client`` is called once per ablation (so ablations don't
    share a :class:`BudgetGuard`). Passing a closure over a stub
    client is the canonical test pattern.
    """
    fixtures = load_suite(suite_path)
    if fixture_filter is not None:
        fixtures = [fx for fx in fixtures if fixture_filter(fx)]
    if not fixtures:
        raise ValueError(f"{suite_path}: no fixtures to run after filtering")

    ablation_list = list(ablations) if ablations is not None else ablation_presets()
    factory = make_client if make_client is not None else _default_make_client
    rid = run_id or f"eval-{uuid.uuid4().hex[:10]}"

    fixtures_by_id = {fx.id: fx for fx in fixtures}
    report = SuiteReport(results_by_ablation={}, metrics_by_ablation={})
    per_ablation_metrics: dict[str, OverallMetrics] = {}

    # Slices are built against the ``full`` ablation so the report
    # surfaces the full-system breakdown; individual ablations still
    # get their own top-line metrics.
    for cfg in ablation_list:
        client = factory()
        started = datetime.now(timezone.utc).isoformat()
        results: list[FixtureResult] = []
        for fx in fixtures:
            fr = run_fixture(fx, ablation=cfg, client=client, run_id=rid)
            results.append(fr)
            if on_fixture is not None:
                on_fixture(fx, fr)
        finished = datetime.now(timezone.utc).isoformat()

        eval_result = EvalResult(
            suite=suite_path.stem,
            ablation=cfg.name,
            run_id=rid,
            started_at=started,
            finished_at=finished,
            model=getattr(client, "model", "unknown"),
            prompt_versions=_prompt_versions_snapshot(),
            fixture_results=results,
        )
        report.results_by_ablation[cfg.name] = eval_result
        report.metrics_by_ablation[cfg.name] = aggregate(results)
        per_ablation_metrics[cfg.name] = report.metrics_by_ablation[cfg.name]

    # Slice only the full system; ablation slices would 4× the report.
    full = report.results_by_ablation.get("full")
    if full is not None:
        report.slices = compute_slices(full.fixture_results, fixtures_by_id)

    return report


__all__ = [
    "LLMClientFactory",
    "load_suite",
    "resolve_suite_path",
    "run_fixture",
    "run_suite",
]
