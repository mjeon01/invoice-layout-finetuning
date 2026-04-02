"""
Multi-model comparison report for invoice NER.

Produces a Markdown table showing per-field F1 for all evaluated models
and highlights fields where LayoutLMv3 (full) provides ≥ 5 pp improvement
over BERT.
"""

from __future__ import annotations

from pathlib import Path

from ..data.label_schema import ENTITY_TYPES

# Display order for models (left → right in the table)
_MODEL_ORDER = ["BERT", "LMv3 (no bbox)", "LMv3 (no image)", "LayoutLMv3"]
_THRESHOLD = 0.05  # 5 pp improvement to highlight with ★


def generate_comparison_report(
    results: dict[str, dict],
    output_path: Path | None = None,
) -> str:
    """
    Generate a Markdown comparison table for all evaluated models.

    Args:
        results:     Dict mapping model name → compute_entity_metrics() output.
                     Expected keys: "BERT", "LMv3 (no bbox)", "LMv3 (no image)", "LayoutLMv3"
                     (subset is also accepted — missing models are omitted from the table).
        output_path: If provided, save the Markdown string here.

    Returns:
        Markdown string.
    """
    # Determine column order (only include models present in results)
    model_names = [m for m in _MODEL_ORDER if m in results]
    # Append any extra models not in the default order
    for m in results:
        if m not in model_names:
            model_names.append(m)

    # Header
    col_header = " | ".join(f"{m} F1" for m in model_names)
    col_sep    = " | ".join("-------" for _ in model_names)

    lines = [
        "# Invoice NER — Multi-Model Comparison (Entity-Level F1)",
        "",
        "평가 데이터: 실제 수산물 무역 인보이스 41건  ",
        "학습 데이터: 합성 인보이스 800건  ",
        "지표: seqeval 엔티티 단위 매크로 F1 (배경 토큰 제외)",
        "",
        "## 전체 요약",
        "",
        f"| 모델 | Precision | Recall | F1 |",
        f"|------|-----------|--------|----|",
    ]
    for m in model_names:
        r = results[m]
        lines.append(
            f"| {m} | {r.get('precision', 0):.4f} | "
            f"{r.get('recall', 0):.4f} | {r.get('f1', 0):.4f} |"
        )

    lines += [
        "",
        "## 필드별 F1",
        "",
        f"| 필드 | {col_header} | 비고 |",
        f"|------|{col_sep}|------|",
    ]

    bert_per = results.get("BERT", {}).get("per_entity", {})
    lv3_per  = results.get("LayoutLMv3", {}).get("per_entity", {})

    for entity in ENTITY_TYPES:
        cols = []
        for m in model_names:
            f1 = results[m].get("per_entity", {}).get(entity, {}).get("f1", 0.0)
            cols.append(f"{f1:.4f}")

        # Highlight if LayoutLMv3 improves ≥ THRESHOLD over BERT
        bert_f1 = bert_per.get(entity, {}).get("f1", 0.0)
        lv3_f1  = lv3_per.get(entity, {}).get("f1", 0.0)
        note = "★ 레이아웃 효과" if (lv3_f1 - bert_f1) >= _THRESHOLD else ""

        lines.append(f"| {entity} | {' | '.join(cols)} | {note} |")

    # Macro avg row
    macro_cols = []
    for m in model_names:
        macro_cols.append(f"**{results[m].get('f1', 0):.4f}**")
    lines.append(f"| **Macro Avg** | {' | '.join(macro_cols)} | |")

    # KIE summary table — per model: Precision / Recall / F1
    kie_col_header = " | ".join(f"{m} P | {m} R | {m} F1" for m in model_names)
    kie_col_sep    = " | ".join("------:| ------:| ------:" for _ in model_names)

    lines += [
        "",
        "## KIE 정확도 (필드값 일치율)",
        "",
        "모델이 추출한 텍스트 값과 정답값의 정확 일치 여부를 필드별로 측정합니다.",
        "",
        f"| 필드 | {kie_col_header} |",
        f"|------|{kie_col_sep}|",
    ]

    for entity in ENTITY_TYPES:
        kie_cols = []
        for m in model_names:
            ep = results[m].get("kie_metrics", {}).get("per_entity", {}).get(entity, {})
            p  = ep.get("precision", 0.0)
            r  = ep.get("recall",    0.0)
            f1 = ep.get("f1",        0.0)
            kie_cols.append(f"{p:.4f} | {r:.4f} | {f1:.4f}")
        lines.append(f"| {entity} | {' | '.join(kie_cols)} |")

    kie_macro_cols = []
    for m in model_names:
        km = results[m].get("kie_metrics", {})
        p  = km.get("kie_precision", 0.0)
        r  = km.get("kie_recall",    0.0)
        f1 = km.get("kie_f1",        0.0)
        kie_macro_cols.append(f"**{p:.4f}** | **{r:.4f}** | **{f1:.4f}**")
    lines.append(f"| **Macro Avg** | {' | '.join(kie_macro_cols)} |")

    lines += [
        "",
        "## 실험 조건",
        "",
        "| 모델 | 텍스트 | 위치(bbox) | 이미지 |",
        "|------|:------:|:----------:|:------:|",
        "| BERT | ✅ | ❌ | ❌ |",
        "| LMv3 (no bbox) | ✅ | ❌ | ✅ |",
        "| LMv3 (no image) | ✅ | ✅ | ❌ |",
        "| LayoutLMv3 | ✅ | ✅ | ✅ |",
        "",
        "> ★ : LayoutLMv3 전체 모델이 BERT 대비 F1이 5%p 이상 향상된 필드",
    ]

    md = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        print(f"Comparison report saved → {output_path}")

    return md


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print the comparison report to stdout."""
    print(generate_comparison_report(results))
