"""
Evaluation metrics for invoice NER.

Provides:
  - compute_entity_metrics: seqeval entity-level precision/recall/F1 (primary)
  - compute_token_metrics:  sklearn token-level metrics (diagnostic)
  - compute_per_field_f1:   per-entity-type F1 for comparison table
"""

from __future__ import annotations

import numpy as np

from ..data.label_schema import ENTITY_TYPES, id2label


def compute_entity_metrics(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> dict:
    """
    Compute entity-level (span-level) precision, recall, F1 using seqeval.

    seqeval ignores the O label and evaluates full named entity spans.
    A span is correct only if both the entity type AND all boundary tokens match.

    Args:
        true_labels: List of sequences; each sequence is word-level BIO label strings.
        pred_labels: Same structure, predicted labels.

    Returns:
        dict with keys: precision, recall, f1, per_entity (dict of entity → F1).
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall    = recall_score   (true_labels, pred_labels, average="macro", zero_division=0)
    f1        = f1_score       (true_labels, pred_labels, average="macro", zero_division=0)

    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    per_entity = {
        entity: {
            "precision": round(report.get(entity, {}).get("precision", 0.0), 4),
            "recall":    round(report.get(entity, {}).get("recall",    0.0), 4),
            "f1":        round(report.get(entity, {}).get("f1-score",  0.0), 4),
            "support":   int  (report.get(entity, {}).get("support",   0)),
        }
        for entity in ENTITY_TYPES
    }

    return {
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "per_entity": per_entity,
    }


def compute_token_metrics(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> dict:
    """
    Compute token-level precision/recall/F1 using sklearn, excluding O labels.

    Useful for diagnosing per-field confusion at the token level.

    Returns:
        dict with keys: precision, recall, f1, per_label (dict of label → F1).
    """
    from sklearn.metrics import classification_report

    flat_true = [l for seq in true_labels for l in seq]
    flat_pred = [l for seq in pred_labels for l in seq]

    # Filter O labels for macro average
    entity_labels = [l for l in set(flat_true) | set(flat_pred) if l != "O"]
    entity_labels.sort()

    report = classification_report(
        flat_true, flat_pred,
        labels=entity_labels,
        output_dict=True,
        zero_division=0,
    )

    macro = report.get("macro avg", {})
    return {
        "token_precision": round(macro.get("precision", 0.0), 4),
        "token_recall":    round(macro.get("recall",    0.0), 4),
        "token_f1":        round(macro.get("f1-score",  0.0), 4),
        "per_label":       {
            lbl: round(report.get(lbl, {}).get("f1-score", 0.0), 4)
            for lbl in entity_labels
        },
    }


def _extract_kie_spans(tokens: list[str], labels: list[str]) -> dict[str, list[str]]:
    """
    BIO 레이블 시퀀스에서 엔티티별 텍스트 값을 추출한다.

    예:
        tokens = ["PT.", "INDO", "PACIFIC", "9.76"]
        labels = ["B-EXPORTER_NAME", "I-EXPORTER_NAME", "I-EXPORTER_NAME", "B-ITEM_UNIT_PRICE"]
        →  {"EXPORTER_NAME": ["PT. INDO PACIFIC"], "ITEM_UNIT_PRICE": ["9.76"]}
    """
    spans: dict[str, list[str]] = {}
    current_entity: str | None = None
    current_tokens: list[str] = []

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity and current_tokens:
                spans.setdefault(current_entity, []).append(" ".join(current_tokens))
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_entity == label[2:]:
            current_tokens.append(token)
        else:
            if current_entity and current_tokens:
                spans.setdefault(current_entity, []).append(" ".join(current_tokens))
            current_entity = None
            current_tokens = []

    if current_entity and current_tokens:
        spans.setdefault(current_entity, []).append(" ".join(current_tokens))

    return spans


def compute_kie_metrics(
    tokens_list: list[list[str]],
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> dict:
    """
    KIE (Key Information Extraction) 필드값 수준 정확도 계산.

    엔티티 F1이 스팬 경계의 정확도를 측정하는 것과 달리,
    KIE는 모델이 추출한 텍스트 값이 정답값과 일치하는지를 측정한다.

    비교 방식: 소문자 변환 + 공백 정규화 후 정확 일치 (exact match)

    예:
        정답값: "PT. INDO PACIFIC FISHERIES"
        예측값: "PT. INDO PACIFIC FISHERIES"  → TP
        예측값: "INDO PACIFIC"               → FP (정답), FN (예측)

    Returns:
        dict with keys: kie_precision, kie_recall, kie_f1, per_entity
    """
    import re

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    entity_tp: dict[str, int] = {e: 0 for e in ENTITY_TYPES}
    entity_fp: dict[str, int] = {e: 0 for e in ENTITY_TYPES}
    entity_fn: dict[str, int] = {e: 0 for e in ENTITY_TYPES}

    for tokens, true_seq, pred_seq in zip(tokens_list, true_labels, pred_labels):
        true_spans = _extract_kie_spans(tokens, true_seq)
        pred_spans = _extract_kie_spans(tokens, pred_seq)

        for entity in ENTITY_TYPES:
            true_vals = set(_norm(v) for v in true_spans.get(entity, []))
            pred_vals = set(_norm(v) for v in pred_spans.get(entity, []))

            entity_tp[entity] += len(true_vals & pred_vals)
            entity_fp[entity] += len(pred_vals - true_vals)
            entity_fn[entity] += len(true_vals - pred_vals)

    per_entity: dict[str, dict] = {}
    for entity in ENTITY_TYPES:
        tp = entity_tp[entity]
        fp = entity_fp[entity]
        fn = entity_fn[entity]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_entity[entity] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        }

    macro_p  = sum(v["precision"] for v in per_entity.values()) / len(ENTITY_TYPES)
    macro_r  = sum(v["recall"]    for v in per_entity.values()) / len(ENTITY_TYPES)
    macro_f1 = sum(v["f1"]        for v in per_entity.values()) / len(ENTITY_TYPES)

    return {
        "kie_precision": round(macro_p,  4),
        "kie_recall":    round(macro_r,  4),
        "kie_f1":        round(macro_f1, 4),
        "per_entity":    per_entity,
    }


def compute_per_field_f1(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> dict[str, float]:
    """
    Return entity-level F1 for each of the 14 entity types.
    Convenience wrapper used for the comparison table.
    """
    metrics = compute_entity_metrics(true_labels, pred_labels)
    return {
        entity: metrics["per_entity"][entity]["f1"]
        for entity in ENTITY_TYPES
    }
