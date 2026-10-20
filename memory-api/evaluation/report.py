"""Markdown report generation for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_markdown_report(run_report: dict[str, Any], comparison: dict[str, Any] | None = None) -> str:
    lines: list[str] = []
    lines.append("# Memory Evaluation Report")
    lines.append("")
    run = run_report.get("run", {})
    lines.append(f"- Baseline: {run.get('baseline')}")
    lines.append(f"- Dataset: {run.get('dataset')}")
    lines.append(f"- Queries: {run.get('num_queries')}")
    lines.append(f"- Timestamp: {run.get('timestamp')}")
    lines.append("")
    lines.append("## Retrieval")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in (run_report.get("retrieval") or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in (run_report.get("decisions") or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Trajectory")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in (run_report.get("trajectory") or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in (run_report.get("performance") or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    if comparison:
        lines.append("## Comparison")
        lines.append("")
        lines.append("| Metric | " + " | ".join(comparison.get("baselines", [])) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(comparison.get("baselines", []))) + "|")
        for metric, vals in (comparison.get("metrics") or {}).items():
            lines.append("| " + metric + " | " + " | ".join(str(v) for v in vals) + " |")
        lines.append("")
    return "\n".join(lines)


def write_markdown_report(run_report: dict[str, Any], path: str | Path, comparison: dict[str, Any] | None = None) -> None:
    Path(path).write_text(generate_markdown_report(run_report, comparison), encoding="utf-8")
