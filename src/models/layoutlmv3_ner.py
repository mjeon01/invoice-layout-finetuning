"""
LayoutLMv3ForTokenClassification wrapper for invoice NER.

Key design decisions:
  - apply_ocr=False: pre-computed tokens and bboxes are passed directly.
  - The processor handles image patch extraction internally (ViT-style patches).
  - Bboxes must be normalized to 0-1000 integers (done in synthetic_generator).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoModelForTokenClassification,
    LayoutLMv3Processor,
)

from ..data.label_schema import NUM_LABELS, id2label, label2id


class LayoutLMv3NERModel:
    """Wrapper for LayoutLMv3ForTokenClassification."""

    DEFAULT_MODEL = "microsoft/layoutlmv3-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        num_labels: int = NUM_LABELS,
        from_pretrained_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.from_pretrained_path = from_pretrained_path

    def get_model_and_processor(
        self,
    ) -> tuple[AutoModelForTokenClassification, LayoutLMv3Processor]:
        """
        Load model and processor.

        If from_pretrained_path is set, loads fine-tuned weights from disk.
        Otherwise loads the base pre-trained model from HuggingFace Hub.

        Returns:
            (model, processor) tuple.
        """
        source = self.from_pretrained_path or self.model_name

        # Processor: apply_ocr=False so we provide our own tokens/bboxes
        processor = LayoutLMv3Processor.from_pretrained(
            self.model_name,   # always use base model tokenizer/feature-extractor
            apply_ocr=False,
        )

        model = AutoModelForTokenClassification.from_pretrained(
            source,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,  # classifier head size may differ
        )
        return model, processor

    @staticmethod
    def predict(
        model: AutoModelForTokenClassification,
        processor: LayoutLMv3Processor,
        sample: dict,
        device: str = "cpu",
    ) -> list[str]:
        """
        Run inference on a single sample dict.

        Args:
            sample: dict with keys tokens, bboxes, image (PIL.Image).
            device: "cpu", "cuda", or "mps".

        Returns:
            List of predicted label strings (one per original token).
            Uses first-subtoken rule: label of the first WordPiece subtoken
            is assigned to the whole word.
        """
        model.eval()
        model.to(device)

        tokens: list[str]        = sample["tokens"]
        bboxes: list[list[int]]  = sample["bboxes"]
        image:  Image.Image      = sample["image"]

        encoding = processor(
            image,
            text=tokens,
            boxes=bboxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        pred_ids = logits.argmax(dim=-1).tolist()

        # Align back to word tokens using word_ids()
        word_ids = encoding.word_ids()  # may not be available on all processor versions
        if word_ids is None:
            # Fallback: return raw subtoken predictions, truncated to token count
            return [id2label.get(p, "O") for p in pred_ids[1:len(tokens) + 1]]

        word_preds: dict[int, str] = {}
        for subtoken_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_preds:
                word_preds[word_idx] = id2label.get(pred_ids[subtoken_idx], "O")

        return [word_preds.get(i, "O") for i in range(len(tokens))]
