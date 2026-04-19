"""Pretty-print the :class:`SuiteReport` to stdout and HTML.

Two surfaces:

- :func:`render_text` — a Markdown-ish table block the ``report`` CLI
  prints to stdout. Same shape as the existing parser / generator
  eval reports so reviewers see familiar layout.
- :func:`render_html` — a self-contained HTML page with tables for
  every ablation and every slice, plus the human-eval form rendered
  by :mod:`eval.human_eval`. Writes alongside the JSON result under
  ``eval_results/<timestamp>.html``.

Zero external dependencies beyond the stdlib — the HTML is a hand-
written template string, not Jinja2. Keeps the eval harness runnable
from any environment without pulling extra packages.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .metrics import OverallMetrics
from .models import EvalResult, SuiteReport


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _money(value: float) -> str:
    return f"${value:.4f}"


def _render_metrics_block(title: str, m: OverallMetrics) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"- Parser: "
        f"populated F1={_pct(m.parser.populated_f1())} "
        f"(P={_pct(m.parser.populated_precision())}, "
        f"R={_pct(m.parser.populated_recall())}); "
        f"gap F1={_pct(m.parser.gap_f1())}",
        f"- Generator (gold params): "
        f"exact match={_pct(m.generator.exact_match_rate())}, "
        f"validity={_pct(m.generator.validity_rate())}, "
        f"symbol F1={_pct(m.generator.symbol_f1())}",
        f"- End-to-end: "
        f"exact match={_pct(m.end_to_end.exact_match_rate())}, "
        f"validity={_pct(m.end_to_end.validity_rate())}, "
        f"symbol F1={_pct(m.end_to_end.symbol_f1())}, "
        f"mean clarification turns={m.end_to_end.mean_clarification_turns():.2f}",
        f"- Cost: "
        f"total={_money(m.cost.total_cost_usd)}, "
        f"mean/sign={_money(m.cost.mean_cost_per_sign())}, "
        f"tokens={m.cost.total_tokens}, "
        f"calls={m.cost.total_llm_calls}, "
        f"p95 latency={m.cost.p95_latency_ms()}ms",
    ]
    return lines


def render_text(report: SuiteReport) -> str:
    """Stdout-friendly Markdown summary of the report."""
    lines: list[str] = []
    lines.append("# chat2hamnosys eval results")
    lines.append("")
    full = report.results_by_ablation.get("full")
    if full is not None:
        lines.append(f"- **Suite**: {full.suite}")
        lines.append(f"- **Run ID**: {full.run_id}")
        lines.append(f"- **Model**: {full.model}")
        lines.append(f"- **Started**: {full.started_at}")
        lines.append(f"- **Finished**: {full.finished_at}")
        if full.prompt_versions:
            joined = ", ".join(
                f"{k}={v}" for k, v in sorted(full.prompt_versions.items())
            )
            lines.append(f"- **Prompts**: {joined}")
    lines.append("")
    lines.append("## By ablation")
    lines.append("")
    for ablation_name, metrics in report.metrics_by_ablation.items():
        lines.extend(_render_metrics_block(f"Ablation: `{ablation_name}`", metrics))
        lines.append("")

    if report.slices:
        lines.append("## Breakdowns (full ablation)")
        lines.append("")
        for slice_name, buckets in report.slices.items():
            lines.append(f"### By {slice_name}")
            lines.append("")
            lines.append(
                "| Bucket | N | E2E exact | E2E F1 | Gen exact | Cost/sign |"
            )
            lines.append(
                "|--------|---|-----------|--------|-----------|-----------|"
            )
            for bucket, metrics in sorted(buckets.items()):
                lines.append(
                    f"| {bucket} | {metrics.end_to_end.n} | "
                    f"{_pct(metrics.end_to_end.exact_match_rate())} | "
                    f"{_pct(metrics.end_to_end.symbol_f1())} | "
                    f"{_pct(metrics.generator.exact_match_rate())} | "
                    f"{_money(metrics.cost.mean_cost_per_sign())} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       margin: 2em auto; max-width: 1100px; color: #222; }
h1, h2, h3 { color: #0b3b6b; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #eef4ff; }
tr:nth-child(even) { background: #fafafa; }
.ablation { border-left: 4px solid #1e6fc2; padding-left: 1em; margin: 1.5em 0; }
.kpi { display: inline-block; margin: 0.2em 1em 0.2em 0; }
.kpi b { color: #1e6fc2; }
.summary { background: #f2f7fd; padding: 1em; border-radius: 4px; }
.warn { background: #fff6e5; padding: 0.5em 1em; border-left: 4px solid #e59500;
        margin: 1em 0; }
.notes { font-family: ui-monospace, monospace; font-size: 0.85em; color: #555; }
"""


def render_html(report: SuiteReport) -> str:
    """Full HTML page — tables per ablation, slices, and links."""
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang=\"en\"><head>")
    parts.append("<meta charset=\"utf-8\">")
    parts.append("<title>chat2hamnosys eval results</title>")
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("</head><body>")

    full = report.results_by_ablation.get("full")
    parts.append("<h1>chat2hamnosys eval results</h1>")

    if full is not None:
        parts.append("<div class=\"summary\">")
        parts.append(
            f"<div class=\"kpi\"><b>Suite:</b> {html.escape(full.suite)}</div>"
        )
        parts.append(
            f"<div class=\"kpi\"><b>Run ID:</b> {html.escape(full.run_id)}</div>"
        )
        parts.append(
            f"<div class=\"kpi\"><b>Model:</b> {html.escape(full.model)}</div>"
        )
        parts.append(
            f"<div class=\"kpi\"><b>Finished:</b> {html.escape(full.finished_at)}</div>"
        )
        if full.prompt_versions:
            prompt_html = ", ".join(
                f"<code>{html.escape(k)}={html.escape(v)}</code>"
                for k, v in sorted(full.prompt_versions.items())
            )
            parts.append(
                f"<div class=\"kpi\"><b>Prompts:</b> {prompt_html}</div>"
            )
        parts.append("</div>")

    parts.append("<h2>Metrics by ablation</h2>")
    parts.append(_ablation_table_html(report))

    if report.slices:
        parts.append("<h2>Full-system breakdowns</h2>")
        for slice_name, buckets in report.slices.items():
            parts.append(f"<h3>By {html.escape(slice_name)}</h3>")
            parts.append(_slice_table_html(buckets))

    for ablation_name, result in report.results_by_ablation.items():
        parts.append(
            f"<h2 class=\"ablation\">Per-fixture details — "
            f"<code>{html.escape(ablation_name)}</code></h2>"
        )
        parts.append(_per_fixture_table_html(result))

    parts.append("</body></html>")
    return "\n".join(parts)


def _metric_cell(label: str, value: str) -> str:
    return f"<div class=\"kpi\"><b>{html.escape(label)}:</b> {html.escape(value)}</div>"


def _ablation_table_html(report: SuiteReport) -> str:
    rows: list[str] = []
    rows.append("<table>")
    rows.append(
        "<tr>"
        "<th>Ablation</th>"
        "<th>N</th>"
        "<th>E2E exact</th>"
        "<th>E2E F1</th>"
        "<th>Validity</th>"
        "<th>Gen exact</th>"
        "<th>Gen F1</th>"
        "<th>Parser pop F1</th>"
        "<th>Parser gap F1</th>"
        "<th>Mean turns</th>"
        "<th>Cost/sign</th>"
        "<th>p95 latency</th>"
        "</tr>"
    )
    for name, m in report.metrics_by_ablation.items():
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td>{m.end_to_end.n}</td>"
            f"<td>{_pct(m.end_to_end.exact_match_rate())}</td>"
            f"<td>{_pct(m.end_to_end.symbol_f1())}</td>"
            f"<td>{_pct(m.end_to_end.validity_rate())}</td>"
            f"<td>{_pct(m.generator.exact_match_rate())}</td>"
            f"<td>{_pct(m.generator.symbol_f1())}</td>"
            f"<td>{_pct(m.parser.populated_f1())}</td>"
            f"<td>{_pct(m.parser.gap_f1())}</td>"
            f"<td>{m.end_to_end.mean_clarification_turns():.2f}</td>"
            f"<td>{_money(m.cost.mean_cost_per_sign())}</td>"
            f"<td>{m.cost.p95_latency_ms()}ms</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _slice_table_html(buckets: dict[str, OverallMetrics]) -> str:
    rows: list[str] = []
    rows.append("<table>")
    rows.append(
        "<tr><th>Bucket</th><th>N</th>"
        "<th>E2E exact</th><th>E2E F1</th>"
        "<th>Gen exact</th><th>Cost/sign</th></tr>"
    )
    for bucket, m in sorted(buckets.items()):
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(bucket)}</code></td>"
            f"<td>{m.end_to_end.n}</td>"
            f"<td>{_pct(m.end_to_end.exact_match_rate())}</td>"
            f"<td>{_pct(m.end_to_end.symbol_f1())}</td>"
            f"<td>{_pct(m.generator.exact_match_rate())}</td>"
            f"<td>{_money(m.cost.mean_cost_per_sign())}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _per_fixture_table_html(result: EvalResult) -> str:
    rows: list[str] = []
    rows.append("<table>")
    rows.append(
        "<tr>"
        "<th>Fixture</th>"
        "<th>Parser OK</th>"
        "<th>Gen exact</th>"
        "<th>E2E exact</th>"
        "<th>E2E valid</th>"
        "<th>Turns</th>"
        "<th>Cost</th>"
        "<th>Notes</th>"
        "</tr>"
    )
    for fr in result.fixture_results:
        notes_html = "<br>".join(html.escape(n) for n in fr.notes[:3])
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(fr.fixture_id)}</code></td>"
            f"<td>{'yes' if fr.parser_ok else 'no'}</td>"
            f"<td>{'yes' if fr.gen_exact_match else 'no'}</td>"
            f"<td>{'yes' if fr.e2e_exact_match else 'no'}</td>"
            f"<td>{'yes' if fr.e2e_valid else 'no'}</td>"
            f"<td>{fr.e2e_clarification_turns}</td>"
            f"<td>{_money(fr.cost_usd)}</td>"
            f"<td class=\"notes\">{notes_html}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------


def render_diff(
    baseline: SuiteReport,
    current: SuiteReport,
) -> str:
    """Compare two reports, highlight per-fixture regressions.

    Operates only on the ``full`` ablation. A regression for this
    command is any fixture that flipped from exact_match=True to
    False on end-to-end, or from valid=True to False, plus any
    aggregate F1 drop greater than :data:`eval.regression.MAX_F1_DROP`.
    """
    from .regression import MAX_F1_DROP

    lines: list[str] = []
    lines.append("# eval diff")
    lines.append("")
    base_full = baseline.results_by_ablation.get("full")
    curr_full = current.results_by_ablation.get("full")
    if base_full is None or curr_full is None:
        return "Both reports must include a 'full' ablation to diff."

    lines.append(f"- Baseline run_id: {base_full.run_id}")
    lines.append(f"- Current run_id:  {curr_full.run_id}")
    lines.append("")

    base_metrics = baseline.metrics_by_ablation.get("full")
    curr_metrics = current.metrics_by_ablation.get("full")
    if base_metrics is not None and curr_metrics is not None:
        e2e_f1_delta = (
            curr_metrics.end_to_end.symbol_f1()
            - base_metrics.end_to_end.symbol_f1()
        )
        e2e_exact_delta = (
            curr_metrics.end_to_end.exact_match_rate()
            - base_metrics.end_to_end.exact_match_rate()
        )
        lines.append("## Aggregate deltas (full ablation)")
        lines.append(
            f"- E2E symbol F1: {_pct(base_metrics.end_to_end.symbol_f1())} "
            f"→ {_pct(curr_metrics.end_to_end.symbol_f1())} "
            f"({e2e_f1_delta:+.3f})"
        )
        lines.append(
            f"- E2E exact match: {_pct(base_metrics.end_to_end.exact_match_rate())} "
            f"→ {_pct(curr_metrics.end_to_end.exact_match_rate())} "
            f"({e2e_exact_delta:+.3f})"
        )
        lines.append(
            f"- Cost per sign: "
            f"{_money(base_metrics.cost.mean_cost_per_sign())} "
            f"→ {_money(curr_metrics.cost.mean_cost_per_sign())}"
        )
        if e2e_f1_delta < -MAX_F1_DROP:
            lines.append(
                f"- **REGRESSION** — F1 dropped more than "
                f"{MAX_F1_DROP:.2f}"
            )
        lines.append("")

    base_by_id = {fr.fixture_id: fr for fr in base_full.fixture_results}
    curr_by_id = {fr.fixture_id: fr for fr in curr_full.fixture_results}

    flips: list[str] = []
    for fid, curr_fr in curr_by_id.items():
        base_fr = base_by_id.get(fid)
        if base_fr is None:
            flips.append(f"+ new fixture {fid}")
            continue
        if base_fr.e2e_exact_match and not curr_fr.e2e_exact_match:
            flips.append(f"- regression on {fid}: e2e exact match lost")
        if base_fr.e2e_valid and not curr_fr.e2e_valid:
            flips.append(f"- regression on {fid}: e2e validity lost")
        if not base_fr.e2e_exact_match and curr_fr.e2e_exact_match:
            flips.append(f"+ improvement on {fid}: e2e exact match gained")
    for fid in base_by_id.keys() - curr_by_id.keys():
        flips.append(f"- dropped fixture {fid}")

    if flips:
        lines.append("## Per-fixture changes")
        lines.extend(flips)
    else:
        lines.append("## No per-fixture changes")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def write_result(report: SuiteReport, out_path: Path) -> Path:
    """Write a report to ``out_path`` (JSON). Returns the written path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_json_safe()
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out_path


def read_result(path: Path) -> SuiteReport:
    """Re-hydrate a :class:`SuiteReport` from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    from .metrics import ParserMetrics, GeneratorMetrics, EndToEndMetrics, CostMetrics

    results_by_ablation: dict[str, EvalResult] = {}
    for name, result_dict in raw.get("results_by_ablation", {}).items():
        results_by_ablation[name] = EvalResult.from_dict(result_dict)

    metrics_by_ablation: dict[str, OverallMetrics] = {}
    for name, result in results_by_ablation.items():
        # Re-aggregate from the stored fixture_results rather than
        # trust the raw metrics dict — keeps precision/recall math
        # in one place (metrics.aggregate) and avoids drift.
        from .metrics import aggregate

        metrics_by_ablation[name] = aggregate(result.fixture_results)

    # Slices are a little expensive to recompute; trust the raw dict
    # when present, otherwise leave empty.
    slices: dict[str, dict[str, OverallMetrics]] = {}
    raw_slices = raw.get("slices") or {}
    for slice_name, buckets in raw_slices.items():
        slices[slice_name] = {
            bucket: _overall_from_dict(m) for bucket, m in buckets.items()
        }

    return SuiteReport(
        results_by_ablation=results_by_ablation,
        metrics_by_ablation=metrics_by_ablation,
        slices=slices,
    )


def _overall_from_dict(d: dict[str, Any]) -> OverallMetrics:
    from .metrics import (
        CostMetrics,
        EndToEndMetrics,
        GeneratorMetrics,
        OverallMetrics,
        ParserMetrics,
    )

    p_raw = d.get("parser", {})
    g_raw = d.get("generator", {})
    e_raw = d.get("end_to_end", {})
    c_raw = d.get("cost", {})
    parser = ParserMetrics(
        n=p_raw.get("n", 0),
        field_correct=dict(p_raw.get("field_correct") or {}),
        field_total=dict(p_raw.get("field_total") or {}),
        gap_tp=p_raw.get("gap_tp", 0),
        gap_fp=p_raw.get("gap_fp", 0),
        gap_fn=p_raw.get("gap_fn", 0),
        pop_tp=p_raw.get("pop_tp", 0),
        pop_fp=p_raw.get("pop_fp", 0),
        pop_fn=p_raw.get("pop_fn", 0),
    )
    generator = GeneratorMetrics(
        n=g_raw.get("n", 0),
        exact_matches=g_raw.get("exact_matches", 0),
        valid_outputs=g_raw.get("valid_outputs", 0),
        symbol_tp=g_raw.get("symbol_tp", 0),
        symbol_fp=g_raw.get("symbol_fp", 0),
        symbol_fn=g_raw.get("symbol_fn", 0),
    )
    e2e = EndToEndMetrics(
        n=e_raw.get("n", 0),
        exact_matches=e_raw.get("exact_matches", 0),
        valid_outputs=e_raw.get("valid_outputs", 0),
        symbol_tp=e_raw.get("symbol_tp", 0),
        symbol_fp=e_raw.get("symbol_fp", 0),
        symbol_fn=e_raw.get("symbol_fn", 0),
        total_clarification_turns=e_raw.get("total_clarification_turns", 0),
        total_questions_asked=e_raw.get("total_questions_asked", 0),
    )
    cost = CostMetrics(
        n=c_raw.get("n", 0),
        total_tokens=c_raw.get("total_tokens", 0),
        total_prompt_tokens=c_raw.get("total_prompt_tokens", 0),
        total_completion_tokens=c_raw.get("total_completion_tokens", 0),
        total_cost_usd=c_raw.get("total_cost_usd", 0.0),
        total_llm_calls=c_raw.get("total_llm_calls", 0),
    )
    return OverallMetrics(
        parser=parser, generator=generator, end_to_end=e2e, cost=cost
    )


__all__ = [
    "read_result",
    "render_diff",
    "render_html",
    "render_text",
    "write_result",
]
