"""
BERT-base text-only NER baseline for invoice information extraction.

Identical label schema as LayoutLMv3 but receives only token text —
no bounding boxes, no image patches. Allows direct comparison to
measure the contribution of layout information.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
)

from ..data.label_schema import NUM_LABELS, id2label, label2id


class BertBaselineNERModel:
    """Wrapper for BertForTokenClassification (text-only NER)."""

    DEFAULT_MODEL = "bert-base-uncased"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        num_labels: int = NUM_LABELS,
        from_pretrained_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.from_pretrained_path = from_pretrained_path

    def get_model_and_tokenizer(
        self,
    ) -> tuple[BertForTokenClassification, BertTokenizerFast]:
        """
        Load model and tokenizer.

        Returns:
            (model, tokenizer) tuple.
        """
        source = self.from_pretrained_path or self.model_name

        tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

        model = BertForTokenClassification.from_pretrained(
            source,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        return model, tokenizer

    @staticmethod
    def predict(
        model: BertForTokenClassification,
        tokenizer: BertTokenizerFast,
        tokens: list[str],
        device: str = "cpu",
    ) -> list[str]:
        """
        Run inference on a list of word tokens.

        Returns:
            List of predicted label strings (one per original token).
        """
        model.eval()
        model.to(device)

        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()

        word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else None
        if word_ids is None:
            # BertTokenizerFast encoding: use .word_ids() from the BatchEncoding
            word_ids = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
            ).word_ids()

        word_preds: dict[int, str] = {}
        for subtoken_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_preds:
                word_preds[word_idx] = id2label.get(pred_ids[subtoken_idx], "O")

        return [word_preds.get(i, "O") for i in range(len(tokens))]
