"""
Side-by-side comparison report: BERT baseline vs LayoutLMv3.

Produces a Markdown table showing per-field F1 for both models
and highlights fields where layout features provide ≥ 5% absolute improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.label_schema import ENTITY_TYPES


def generate_comparison_report(
    layoutlmv3_results: dict,
    bert_results: dict,
    output_path: Path | None = None,
) -> str:
    """
    Generate a Markdown comparison table.

    Args:
        layoutlmv3_results: Output of compute_entity_metrics() for LayoutLMv3.
        bert_results:       Output of compute_entity_metrics() for BERT.
        output_path:        If provided, save the Markdown string here.

    Returns:
        Markdown string.
    """
    lv3_per  = layoutlmv3_results.get("per_entity", {})
    bert_per = bert_results.get("per_entity", {})

    THRESHOLD = 0.05  # 5 pp improvement to highlight

    lines = [
        "# BERT Baseline vs LayoutLMv3 — Entity-Level F1 on Real 41 Documents",
        "",
        f"**LayoutLMv3 macro F1**: {layoutlmv3_results.get('f1', 0):.4f}",
        f"**BERT macro F1**:       {bert_results.get('f1', 0):.4f}",
        "",
        "Fields marked with ★ show ≥ 5 pp absolute F1 improvement with LayoutLMv3.",
        "",
        "| Field | BERT F1 | LayoutLMv3 F1 | Δ F1 | Note |",
        "|-------|---------|---------------|------|------|",
    ]

    for entity in ENTITY_TYPES:
        bert_f1 = bert_per.get(entity, {}).get("f1", 0.0)
        lv3_f1  = lv3_per.get(entity, {}).get("f1", 0.0)
        delta   = lv3_f1 - bert_f1
        note    = "★ layout helps" if delta >= THRESHOLD else ""
        lines.append(
            f"| {entity} | {bert_f1:.4f} | {lv3_f1:.4f} | "
            f"{delta:+.4f} | {note} |"
        )

    lines += [
        "",
        "| **Macro Avg** | "
        f"**{bert_results.get('f1', 0):.4f}** | "
        f"**{layoutlmv3_results.get('f1', 0):.4f}** | "
        f"**{layoutlmv3_results.get('f1', 0) - bert_results.get('f1', 0):+.4f}** | |",
        "",
        "## Notes",
        "",
        "- Training set: 900 synthetic invoices (identical for both models)",
        "- Evaluation set: 41 real seafood trade invoices",
        "- BERT: text-only BIO tagging (no bounding boxes, no image)",
        "- LayoutLMv3: text + bounding boxes + image patches",
        "- Metric: seqeval macro entity-level F1 (spans, excluding O)",
    ]

    md = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        print(f"Comparison report saved → {output_path}")

    return md


def print_comparison_table(layoutlmv3_results: dict, bert_results: dict) -> None:
    """Print the comparison report to stdout."""
    print(generate_comparison_report(layoutlmv3_results, bert_results))
