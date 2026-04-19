"""Data contracts for the evaluation harness.

The golden dataset is a JSONL file of :class:`GoldenFixture` rows.
One fixture drives one flow through the pipeline and yields one
:class:`FixtureResult`. The runner aggregates fixture results into a
:class:`SuiteReport` with :class:`OverallMetrics` and per-category
slices.

These are dataclasses rather than Pydantic models because we never
validate them at an API boundary — they're read from trusted local
JSON/JSONL files the eval harness ships and writes. Keeping the
serialization layer thin (``to_json_safe`` / ``from_dict``) matches
the style the parser/generator eval scripts already use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


# Per-fixture annotation of how much the prose specifies. The runner
# uses this to break down metrics by difficulty so we can see where
# the system breaks and where it shines (Prompt 16 §8).
Difficulty = Literal["easy", "medium", "hard"]


# Ablation names must stay in sync with :mod:`eval.ablations`. Encoded
# as a string Literal so the CLI and JSON round-trip cleanly; the
# runner dispatches on the name itself.
AblationName = Literal[
    "full",                 # full system; baseline for regression guard
    "no_clarification",     # skip clarifier Q&A — measure its value
    "no_validator_feedback",  # skip repair loop — measure its value
    "no_deterministic_map",  # force LLM fallback for every slot — measure VOCAB value
]


_ALL_ABLATIONS: tuple[AblationName, ...] = (
    "full",
    "no_clarification",
    "no_validator_feedback",
    "no_deterministic_map",
)


@dataclass(frozen=True)
class GoldenFixture:
    """One fixture row in the golden dataset.

    ``expected_hamnosys`` is *one* valid notation; HamNoSys permits
    equivalent forms (e.g. redundant modifiers, alternate symmetry
    encodings), so :attr:`acceptable_hamnosys_variants` lists other
    strings that should also count as a match. The runner treats any
    variant hit as an exact-match.

    Difficulty bucket guidance:

    - ``easy`` — prose specifies every mandatory slot unambiguously.
    - ``medium`` — prose is fluent but leaves 1 slot implicit.
    - ``hard`` — prose is missing 2+ parameters and/or uses ambiguous
      phrasing the parser must flag as gaps.
    """

    id: str
    prose_description: str
    gloss: str
    sign_language: Literal["bsl", "asl", "dgs"]
    expected_parameters: dict[str, Any]
    expected_hamnosys: str
    acceptable_hamnosys_variants: list[str]
    source: str
    difficulty: Difficulty

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldenFixture":
        return cls(
            id=d["id"],
            prose_description=d["prose_description"],
            gloss=d["gloss"],
            sign_language=d["sign_language"],
            expected_parameters=d["expected_parameters"],
            expected_hamnosys=d["expected_hamnosys"],
            acceptable_hamnosys_variants=list(
                d.get("acceptable_hamnosys_variants") or []
            ),
            source=d["source"],
            difficulty=d["difficulty"],
        )

    def accept_set(self) -> set[str]:
        """All HamNoSys strings considered a correct match."""
        return {self.expected_hamnosys, *self.acceptable_hamnosys_variants}

    # ---- category slicing helpers (Prompt 16 §8) ----

    def has_non_manual(self) -> bool:
        nm = self.expected_parameters.get("non_manual")
        return isinstance(nm, dict) and any(v for v in nm.values())

    def is_two_handed(self) -> bool:
        return bool(self.expected_parameters.get("handshape_nondominant"))


# ---------------------------------------------------------------------------
# Per-fixture result
# ---------------------------------------------------------------------------


@dataclass
class FixtureResult:
    """Outcome of running one fixture under one ablation.

    ``parser_populated_correct`` / ``parser_gaps_correct`` hold the
    slot-level evaluation of the parser stage (fields are the slot
    names the parser got right / wrong). ``generator_exact_match``
    and ``generator_symbol_tp/fp/fn`` capture the generator stage
    when it's fed the *gold* parameters. The end-to-end counterparts
    measure the pipeline when the simulator fills gaps.

    ``notes`` carries free-form diagnostic strings (validation errors,
    which ablation clipped which layer, etc.). Kept open-ended so
    future metrics can stuff extra context in without schema churn.
    """

    fixture_id: str
    ablation: AblationName

    # Parser stage
    parser_ok: bool = False
    parser_populated_correct: set[str] = field(default_factory=set)
    parser_populated_expected: set[str] = field(default_factory=set)
    parser_populated_actual: set[str] = field(default_factory=set)
    parser_gaps_actual: set[str] = field(default_factory=set)
    parser_gaps_expected: set[str] = field(default_factory=set)
    parser_field_accuracy: dict[str, bool] = field(default_factory=dict)

    # Generator stage (fed gold parameters)
    gen_hamnosys: str | None = None
    gen_valid: bool = False
    gen_exact_match: bool = False
    gen_symbol_tp: int = 0
    gen_symbol_fp: int = 0
    gen_symbol_fn: int = 0

    # End-to-end stage (parse → simulate answers → generate)
    e2e_hamnosys: str | None = None
    e2e_valid: bool = False
    e2e_exact_match: bool = False
    e2e_symbol_tp: int = 0
    e2e_symbol_fp: int = 0
    e2e_symbol_fn: int = 0
    e2e_clarification_turns: int = 0
    e2e_questions_asked: int = 0

    # Cost (aggregated from telemetry; harness fills after LLM calls)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    llm_calls: int = 0

    notes: list[str] = field(default_factory=list)

    def to_json_safe(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "parser_populated_correct",
            "parser_populated_expected",
            "parser_populated_actual",
            "parser_gaps_actual",
            "parser_gaps_expected",
        ):
            data[key] = sorted(data[key])
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FixtureResult":
        kwargs = dict(d)
        for key in (
            "parser_populated_correct",
            "parser_populated_expected",
            "parser_populated_actual",
            "parser_gaps_actual",
            "parser_gaps_expected",
        ):
            kwargs[key] = set(kwargs.get(key) or [])
        kwargs["parser_field_accuracy"] = dict(
            kwargs.get("parser_field_accuracy") or {}
        )
        kwargs["notes"] = list(kwargs.get("notes") or [])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Human ratings (Prompt 16 §9 — bridge to Huenerfauth protocol)
# ---------------------------------------------------------------------------


@dataclass
class HumanRating:
    """One Deaf reviewer rating on a single generated sign.

    Three 1–10 scales follow Huenerfauth's published protocol for
    evaluating signed output: grammaticality, naturalness,
    comprehensibility. The integers are deliberately raw — we correlate
    against automated metrics over time rather than averaging them up
    front, because the three axes disagree more often than they agree.
    """

    fixture_id: str
    ablation: AblationName
    reviewer_id: str
    grammaticality: int
    naturalness: int
    comprehensibility: int
    notes: str = ""
    rated_at: str = ""  # ISO-8601 UTC string

    def to_json_safe(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Suite-level result
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """One complete run of the suite (all fixtures × one ablation).

    One JSON file on disk per run; the ``diff`` command consumes two
    of these and produces a regression summary.
    """

    suite: str
    ablation: AblationName
    run_id: str
    started_at: str
    finished_at: str
    model: str
    prompt_versions: dict[str, str]  # {prompt_id: version}
    fixture_results: list[FixtureResult]
    human_ratings: list[HumanRating] = field(default_factory=list)

    def to_json_safe(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "ablation": self.ablation,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "model": self.model,
            "prompt_versions": dict(self.prompt_versions),
            "fixture_results": [fr.to_json_safe() for fr in self.fixture_results],
            "human_ratings": [hr.to_json_safe() for hr in self.human_ratings],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalResult":
        return cls(
            suite=d["suite"],
            ablation=d["ablation"],
            run_id=d["run_id"],
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            model=d.get("model", ""),
            prompt_versions=dict(d.get("prompt_versions") or {}),
            fixture_results=[
                FixtureResult.from_dict(fr) for fr in d.get("fixture_results") or []
            ],
            human_ratings=[
                HumanRating(**hr) for hr in d.get("human_ratings") or []
            ],
        )


# ---------------------------------------------------------------------------
# Aggregate report (multiple ablations, multiple slices)
# ---------------------------------------------------------------------------


@dataclass
class SuiteReport:
    """The top-level artifact `report` and `run` commands emit.

    Holds one :class:`EvalResult` per ablation plus the precomputed
    :class:`OverallMetrics` and the per-category slices (difficulty,
    sign language, non-manual, handedness).

    Computing metrics eagerly and storing them on the report means the
    ``diff`` and ``report`` commands never have to re-run any
    aggregation — they just read numbers off the dataclass.
    """

    results_by_ablation: dict[AblationName, "EvalResult"]
    metrics_by_ablation: dict[AblationName, "OverallMetrics"]
    slices: dict[str, dict[str, "OverallMetrics"]] = field(default_factory=dict)

    def to_json_safe(self) -> dict[str, Any]:
        return {
            "results_by_ablation": {
                k: v.to_json_safe() for k, v in self.results_by_ablation.items()
            },
            "metrics_by_ablation": {
                k: v.to_json_safe() for k, v in self.metrics_by_ablation.items()
            },
            "slices": {
                slice_name: {
                    bucket: m.to_json_safe() for bucket, m in buckets.items()
                }
                for slice_name, buckets in self.slices.items()
            },
        }


__all__ = [
    "AblationName",
    "Difficulty",
    "EvalResult",
    "FixtureResult",
    "GoldenFixture",
    "HumanRating",
    "SuiteReport",
    "_ALL_ABLATIONS",
]
