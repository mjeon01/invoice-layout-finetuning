"""
InvoiceNERTrainer — HuggingFace Trainer subclass with compute_metrics.

Injects entity-level F1 (seqeval) as the primary evaluation metric,
which is used for best-model selection and early stopping.
"""

from __future__ import annotations

import numpy as np
from transformers import Trainer, TrainingArguments

from ..data.label_schema import ENTITY_TYPES, LABELS, id2label


class InvoiceNERTrainer(Trainer):
    """
    Trainer that computes token-level and entity-level (seqeval) metrics
    after each evaluation epoch.

    Usage:
        trainer = InvoiceNERTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
        )
        trainer.train()
    """

    def compute_metrics(self, eval_pred) -> dict:
        """
        Called by Trainer after each eval pass.

        eval_pred.predictions: np.ndarray shape (N, seq_len, num_labels)
        eval_pred.label_ids:   np.ndarray shape (N, seq_len)
        """
        try:
            from seqeval.metrics import (
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )
        except ImportError:
            raise ImportError("seqeval is required: pip install seqeval")

        logits, label_ids = eval_pred.predictions, eval_pred.label_ids

        # argmax over num_labels dimension
        predictions = np.argmax(logits, axis=-1)

        true_seqs:  list[list[str]] = []
        pred_seqs:  list[list[str]] = []

        for pred_row, label_row in zip(predictions, label_ids):
            true_seq: list[str] = []
            pred_seq: list[str] = []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue  # skip padding / non-first subtokens
                true_seq.append(id2label.get(int(l), "O"))
                pred_seq.append(id2label.get(int(p), "O"))
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq)

        precision = precision_score(true_seqs, pred_seqs, average="macro", zero_division=0)
        recall    = recall_score   (true_seqs, pred_seqs, average="macro", zero_division=0)
        f1        = f1_score       (true_seqs, pred_seqs, average="macro", zero_division=0)

        # Per-entity F1 for diagnostics
        report = classification_report(
            true_seqs, pred_seqs, output_dict=True, zero_division=0
        )
        per_entity = {
            entity: round(report.get(entity, {}).get("f1-score", 0.0), 4)
            for entity in ENTITY_TYPES
        }

        return {
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1":           round(f1, 4),
            **{f"eval_f1_{e.lower()}": v for e, v in per_entity.items()},
        }


def build_training_args(cfg_dict: dict) -> TrainingArguments:
    """Build TrainingArguments from a config dict (from TrainingConfig.to_hf_training_args())."""
    return TrainingArguments(**cfg_dict)
