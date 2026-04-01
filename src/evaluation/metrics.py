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
