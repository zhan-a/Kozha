"""argparse entry points for the eval harness.

Three subcommands:

- ``run`` — execute the suite under one or more ablations and write a
  JSON report.  Flags mirror Prompt 16 §5 / §7: ``--suite``,
  ``--ablations``, ``--smoke``, ``--out``, plus single-ablation flags
  ``--no-clarification``, ``--no-validator-feedback``,
  ``--no-deterministic-map`` that overrule ``--ablations`` for ad-hoc
  one-off runs.
- ``diff`` — compare a baseline result JSON against a current one and
  print per-fixture regressions (wraps :func:`report.render_diff`).
- ``report`` — render a previously saved result to stdout and/or HTML.

Three auxiliary commands:

- ``smoke`` — run the regression guard against the pinned
  ``baselines/current.json``. Fails with exit code 1 when e2e F1 drops
  beyond :data:`MAX_F1_DROP`. CI wires this to a required check.
- ``update-baseline`` — manual-only baseline refresher. Writes the
  current ``smoke`` subset's full-ablation metrics to the baseline
  file. Never run automatically from CI.
- ``human-eval`` — produce the Huenerfauth 1-10 rating form HTML for
  a saved result JSON so a Deaf reviewer can rate the generated signs.

Design rule: the harness never silently uses a different suite or
ablation than the caller asked for. Any mismatch between ``--suite``
and the baseline's recorded suite raises an error the CLI surfaces to
stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .ablations import AblationConfig, ablation_presets
from .human_eval import render_rating_form
from .models import AblationName, GoldenFixture, _ALL_ABLATIONS
from .regression import (
    BASELINE_FILE,
    check_regression,
    load_baseline,
    save_baseline,
)
from .report import (
    read_result,
    render_diff,
    render_html,
    render_text,
    write_result,
)
from .runner import load_suite, resolve_suite_path, run_suite


logger = logging.getLogger(__name__)


SMOKE_FIXTURE_COUNT = 10


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description=(
            "End-to-end evaluation harness for chat2hamnosys. See "
            "docs/chat2hamnosys/16-eval-results-readme.md for the "
            "metrics reference."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run ---------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run fixtures through the pipeline and write a JSON report.",
    )
    p_run.add_argument(
        "--suite",
        default="golden_signs",
        help="Suite name (resolved under eval/fixtures/) or absolute path.",
    )
    p_run.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path for the JSON report. Default: "
            "eval_results/<timestamp>.json."
        ),
    )
    p_run.add_argument(
        "--html",
        type=Path,
        default=None,
        help=(
            "Optional output path for the HTML report. Default: same "
            "basename as --out with .html extension."
        ),
    )
    p_run.add_argument(
        "--ablations",
        nargs="*",
        choices=list(_ALL_ABLATIONS),
        default=None,
        help=(
            "Ablations to run. Default: all of them. Passing --ablations "
            "without values disables every ablation and runs the full "
            "system only."
        ),
    )
    p_run.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run only the first "
            f"{SMOKE_FIXTURE_COUNT} fixtures (CI guard subset)."
        ),
    )
    p_run.add_argument(
        "--fixture-filter",
        default=None,
        help=(
            "Only run fixtures whose id contains this substring (useful "
            "for debugging a single failure)."
        ),
    )
    # Single-ablation convenience flags (mutually exclusive).
    ablation_flags = p_run.add_mutually_exclusive_group()
    ablation_flags.add_argument(
        "--no-clarification",
        dest="single_ablation",
        action="store_const",
        const="no_clarification",
        help="Shortcut for --ablations no_clarification.",
    )
    ablation_flags.add_argument(
        "--no-validator-feedback",
        dest="single_ablation",
        action="store_const",
        const="no_validator_feedback",
        help="Shortcut for --ablations no_validator_feedback.",
    )
    ablation_flags.add_argument(
        "--no-deterministic-map",
        dest="single_ablation",
        action="store_const",
        const="no_deterministic_map",
        help="Shortcut for --ablations no_deterministic_map.",
    )
    p_run.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use a deterministic stub LLM client instead of hitting the "
            "live API. Useful for shape-checking the pipeline in CI "
            "without spending tokens."
        ),
    )
    p_run.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "After the run, overwrite baselines/current.json with the "
            "full-ablation metrics. Only valid with --smoke."
        ),
    )

    # -- diff --------------------------------------------------------------
    p_diff = sub.add_parser(
        "diff",
        help="Compare two saved result JSONs; show per-fixture regressions.",
    )
    p_diff.add_argument("baseline", type=Path, help="Earlier result JSON")
    p_diff.add_argument("current", type=Path, help="Later result JSON")

    # -- report ------------------------------------------------------------
    p_report = sub.add_parser(
        "report",
        help="Render a saved result JSON to stdout and/or HTML.",
    )
    p_report.add_argument("result", type=Path, help="Saved result JSON")
    p_report.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Write an HTML report here (optional).",
    )

    # -- smoke (regression guard) ------------------------------------------
    p_smoke = sub.add_parser(
        "smoke",
        help=(
            "Run the regression guard against the pinned baseline. "
            "Exit 1 when F1 drops beyond MAX_F1_DROP."
        ),
    )
    p_smoke.add_argument(
        "--suite",
        default="golden_signs",
        help="Suite name (defaults to golden_signs).",
    )
    p_smoke.add_argument(
        "--stub",
        action="store_true",
        help="Use a deterministic stub client (CI without API keys).",
    )
    p_smoke.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=f"Baseline file. Default: {BASELINE_FILE}.",
    )

    # -- update-baseline ---------------------------------------------------
    p_ub = sub.add_parser(
        "update-baseline",
        help=(
            "Rewrite baselines/current.json from a prior run's JSON. "
            "Manual-only; CI never calls this."
        ),
    )
    p_ub.add_argument("result", type=Path, help="Saved result JSON to pin.")
    p_ub.add_argument(
        "--commit",
        default="",
        help="Git commit hash to record alongside the baseline.",
    )
    p_ub.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=f"Baseline file. Default: {BASELINE_FILE}.",
    )

    # -- human-eval --------------------------------------------------------
    p_he = sub.add_parser(
        "human-eval",
        help=(
            "Render the Huenerfauth 1-10 rating form HTML for a saved "
            "result. Reviewers fill it out offline and the responses are "
            "folded back in with `ingest-ratings`."
        ),
    )
    p_he.add_argument("result", type=Path, help="Saved result JSON")
    p_he.add_argument(
        "--out",
        type=Path,
        required=True,
        help="HTML file to write (e.g. eval_results/<run>/ratings.html).",
    )
    p_he.add_argument(
        "--reviewer-id",
        default="",
        help=(
            "Optional reviewer id pre-filled on the form. Useful when "
            "shipping separate forms to different Deaf reviewers."
        ),
    )

    # -- ingest-ratings ----------------------------------------------------
    p_ir = sub.add_parser(
        "ingest-ratings",
        help=(
            "Merge a filled-out JSON rating payload into a saved eval "
            "result (mutates the file in place)."
        ),
    )
    p_ir.add_argument("result", type=Path, help="Saved eval result JSON")
    p_ir.add_argument(
        "ratings",
        type=Path,
        help="JSON file produced by the human-eval form's download button.",
    )

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    suite_path = resolve_suite_path(args.suite)
    ablations = _selected_ablations(args)
    out_path = args.out or _default_result_path()
    html_path = args.html or out_path.with_suffix(".html")
    fixture_filter = _compile_filter(args.fixture_filter, args.smoke)

    make_client = _stub_factory() if args.stub else None

    def _progress(fx: GoldenFixture, _fr) -> None:
        sys.stderr.write(".")
        sys.stderr.flush()

    report = run_suite(
        suite_path,
        ablations=ablations,
        make_client=make_client,
        on_fixture=_progress,
        fixture_filter=fixture_filter,
    )
    sys.stderr.write("\n")

    write_result(report, out_path)
    html_path.write_text(render_html(report), encoding="utf-8")
    text = render_text(report)
    print(text)
    print(f"\nWrote JSON: {out_path}")
    print(f"Wrote HTML: {html_path}")

    if args.update_baseline:
        if not args.smoke:
            raise SystemExit(
                "--update-baseline is only valid together with --smoke"
            )
        full_metrics = report.metrics_by_ablation.get("full")
        if full_metrics is None:
            raise SystemExit(
                "cannot update baseline: --ablations excluded the full run"
            )
        saved = save_baseline(
            full_metrics,
            commit=_current_git_commit(),
            model=_model_from_report(report),
            suite=suite_path.stem,
        )
        print(f"Updated baseline at {saved}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    baseline = read_result(args.baseline)
    current = read_result(args.current)
    print(render_diff(baseline, current))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = read_result(args.result)
    print(render_text(report))
    if args.html:
        args.html.write_text(render_html(report), encoding="utf-8")
        print(f"Wrote HTML: {args.html}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    suite_path = resolve_suite_path(args.suite)
    make_client = _stub_factory() if args.stub else None
    report = run_suite(
        suite_path,
        ablations=[AblationConfig("full")],
        make_client=make_client,
        fixture_filter=_smoke_filter(),
    )
    full_metrics = report.metrics_by_ablation["full"]
    baseline = load_baseline(args.baseline)
    guard = check_regression(full_metrics, baseline)
    print(guard.render())
    return 0 if guard.ok else 1


def cmd_update_baseline(args: argparse.Namespace) -> int:
    report = read_result(args.result)
    full_metrics = report.metrics_by_ablation.get("full")
    if full_metrics is None:
        raise SystemExit(
            f"{args.result} has no 'full' ablation; cannot update baseline"
        )
    full_eval = report.results_by_ablation.get("full")
    suite = full_eval.suite if full_eval is not None else "unknown"
    model = full_eval.model if full_eval is not None else "unknown"
    saved = save_baseline(
        full_metrics,
        commit=args.commit or _current_git_commit(),
        model=model,
        suite=suite,
        path=args.baseline,
    )
    print(f"Updated baseline at {saved}")
    return 0


def cmd_human_eval(args: argparse.Namespace) -> int:
    report = read_result(args.result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render_rating_form(report, reviewer_id=args.reviewer_id),
        encoding="utf-8",
    )
    print(f"Wrote human-eval form: {args.out}")
    return 0


def cmd_ingest_ratings(args: argparse.Namespace) -> int:
    from .models import EvalResult, HumanRating

    payload = json.loads(args.ratings.read_text(encoding="utf-8"))
    raw_ratings = payload.get("ratings") or []
    result_raw = json.loads(args.result.read_text(encoding="utf-8"))

    by_ablation_results = result_raw.get("results_by_ablation", {})
    n_added = 0
    for item in raw_ratings:
        rating = HumanRating(
            fixture_id=item["fixture_id"],
            ablation=item.get("ablation", "full"),
            reviewer_id=item.get("reviewer_id", payload.get("reviewer_id", "")),
            grammaticality=int(item.get("grammaticality", 0)),
            naturalness=int(item.get("naturalness", 0)),
            comprehensibility=int(item.get("comprehensibility", 0)),
            notes=item.get("notes", ""),
            rated_at=item.get("rated_at")
            or datetime.now(timezone.utc).isoformat(),
        )
        target = by_ablation_results.get(rating.ablation)
        if target is None:
            logger.warning(
                "skipping rating for unknown ablation %s", rating.ablation
            )
            continue
        target.setdefault("human_ratings", []).append(rating.to_json_safe())
        n_added += 1

    args.result.write_text(
        json.dumps(result_raw, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Merged {n_added} rating(s) into {args.result}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selected_ablations(args: argparse.Namespace) -> list[AblationConfig]:
    if args.single_ablation:
        return [AblationConfig("full"), AblationConfig(args.single_ablation)]
    if args.ablations is None:
        return ablation_presets()
    if not args.ablations:
        return [AblationConfig("full")]
    return [AblationConfig(name) for name in args.ablations]


def _compile_filter(substring: str | None, smoke: bool):
    if smoke:
        return _smoke_filter()
    if not substring:
        return None
    needle = substring.lower()
    return lambda fx: needle in fx.id.lower()


def _smoke_filter():
    counter = {"seen": 0}

    def _f(_fx) -> bool:
        counter["seen"] += 1
        return counter["seen"] <= SMOKE_FIXTURE_COUNT

    return _f


def _stub_factory():
    """Build a deterministic stub LLM client for CI / smoke runs."""
    from .stub_client import StubLLMClient

    return lambda: StubLLMClient()


def _default_result_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    base = Path.cwd() / "eval_results"
    return base / f"{stamp}.json"


def _current_git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def _model_from_report(report) -> str:
    full = report.results_by_ablation.get("full")
    return full.model if full is not None else "unknown"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_COMMANDS = {
    "run": cmd_run,
    "diff": cmd_diff,
    "report": cmd_report,
    "smoke": cmd_smoke,
    "update-baseline": cmd_update_baseline,
    "human-eval": cmd_human_eval,
    "ingest-ratings": cmd_ingest_ratings,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover — argparse guards this
        parser.error(f"unknown command {args.command!r}")
    return int(handler(args) or 0)


__all__ = ["build_parser", "main"]
