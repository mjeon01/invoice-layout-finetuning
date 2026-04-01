"""
Data collators for LayoutLMv3 and BERT NER fine-tuning.

Critical responsibility: align word-level BIO labels to WordPiece subtokens.
  - First subtoken of word i  → original label for word i
  - Subsequent subtokens      → -100 (ignored in cross-entropy loss)
  - Special tokens ([CLS],[SEP],padding) → -100
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image
from transformers import (
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    LayoutLMv3Processor,
)


# ---------------------------------------------------------------------------
# Shared label-alignment utility
# ---------------------------------------------------------------------------

def align_labels_with_tokens(
    word_ids: list[int | None],
    word_labels: list[int],
    label_pad_id: int = -100,
) -> list[int]:
    """
    Expand word-level label ids to subtoken-level label ids.

    Args:
        word_ids:    Output of tokenizer().word_ids() — maps each subtoken
                     position to its original word index (None for special tokens).
        word_labels: One label id per original word.
        label_pad_id: Label id assigned to non-first subtokens and specials.

    Returns:
        List of label ids, same length as word_ids.
    """
    aligned = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(label_pad_id)
        elif word_idx != prev_word_idx:
            # First subtoken of this word: use the actual label
            label = word_labels[word_idx] if word_idx < len(word_labels) else label_pad_id
            aligned.append(label)
        else:
            # Continuation subtoken: ignore
            aligned.append(label_pad_id)
        prev_word_idx = word_idx
    return aligned


# ---------------------------------------------------------------------------
# LayoutLMv3 collator
# ---------------------------------------------------------------------------

class LayoutLMv3DataCollator:
    """
    Collator for LayoutLMv3 token classification.

    Handles three modalities per sample:
      - text tokens + bboxes (padded to batch-max length)
      - pixel_values (fixed shape from processor, just stacked)
      - labels (aligned + padded with -100)
    """

    def __init__(
        self,
        processor: LayoutLMv3Processor,
        max_length: int = 512,
        label_pad_id: int = -100,
    ):
        self.processor = processor
        self.max_length = max_length
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch_tokens    = [f["tokens"]   for f in features]
        batch_bboxes    = [f["bboxes"]   for f in features]
        batch_ner_tags  = [f["ner_tags"] for f in features]
        batch_images    = [f["image"] if isinstance(f["image"], Image.Image)
                           else Image.fromarray(f["image"])
                           for f in features]

        # Encode each sample individually to get word_ids for label alignment
        all_input_ids       = []
        all_attention_masks = []
        all_token_type_ids  = []
        all_bboxes_enc      = []
        all_pixel_values    = []
        all_labels          = []

        for tokens, bboxes, ner_tags, image in zip(
            batch_tokens, batch_bboxes, batch_ner_tags, batch_images
        ):
            enc = self.processor(
                image,
                text=tokens,
                boxes=bboxes,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            word_ids = enc.word_ids()
            labels = align_labels_with_tokens(word_ids, ner_tags, self.label_pad_id)

            all_input_ids.append(enc["input_ids"][0])
            all_attention_masks.append(enc["attention_mask"][0])
            if "token_type_ids" in enc:
                all_token_type_ids.append(enc["token_type_ids"][0])
            all_bboxes_enc.append(enc["bbox"][0])
            all_pixel_values.append(enc["pixel_values"][0])
            all_labels.append(torch.tensor(labels, dtype=torch.long))

        # Pad sequences to max length in this batch
        max_len = max(t.size(0) for t in all_input_ids)

        def pad_1d(tensor: torch.Tensor, pad_val: int) -> torch.Tensor:
            pad_size = max_len - tensor.size(0)
            if pad_size == 0:
                return tensor
            return torch.cat([tensor, torch.full((pad_size,), pad_val, dtype=tensor.dtype)])

        def pad_2d_bbox(tensor: torch.Tensor) -> torch.Tensor:
            # tensor shape: (seq_len, 4)
            pad_size = max_len - tensor.size(0)
            if pad_size == 0:
                return tensor
            pad = torch.zeros(pad_size, 4, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0)

        padded_input_ids  = torch.stack([pad_1d(t, 0) for t in all_input_ids])
        padded_attn_masks = torch.stack([pad_1d(t, 0) for t in all_attention_masks])
        padded_bboxes     = torch.stack([pad_2d_bbox(t) for t in all_bboxes_enc])
        padded_labels     = torch.stack([pad_1d(t, self.label_pad_id) for t in all_labels])
        pixel_values      = torch.stack(all_pixel_values)

        batch = {
            "input_ids":      padded_input_ids,
            "attention_mask": padded_attn_masks,
            "bbox":           padded_bboxes,
            "pixel_values":   pixel_values,
            "labels":         padded_labels,
        }
        if all_token_type_ids:
            batch["token_type_ids"] = torch.stack([pad_1d(t, 0) for t in all_token_type_ids])

        return batch


# ---------------------------------------------------------------------------
# BERT collator
# ---------------------------------------------------------------------------

class BertDataCollator:
    """
    Collator for BERT token classification (text-only).
    """

    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        max_length: int = 512,
        label_pad_id: int = -100,
    ):
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch_tokens   = [f["tokens"]   for f in features]
        batch_ner_tags = [f["ner_tags"] for f in features]

        all_input_ids       = []
        all_attention_masks = []
        all_token_type_ids  = []
        all_labels          = []

        for tokens, ner_tags in zip(batch_tokens, batch_ner_tags):
            enc = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            word_ids = enc.word_ids()
            labels = align_labels_with_tokens(word_ids, ner_tags, self.label_pad_id)

            all_input_ids.append(enc["input_ids"][0])
            all_attention_masks.append(enc["attention_mask"][0])
            if "token_type_ids" in enc:
                all_token_type_ids.append(enc["token_type_ids"][0])
            all_labels.append(torch.tensor(labels, dtype=torch.long))

        max_len = max(t.size(0) for t in all_input_ids)

        def pad_1d(tensor: torch.Tensor, pad_val: int) -> torch.Tensor:
            pad_size = max_len - tensor.size(0)
            if pad_size == 0:
                return tensor
            return torch.cat([tensor, torch.full((pad_size,), pad_val, dtype=tensor.dtype)])

        batch = {
            "input_ids":      torch.stack([pad_1d(t, 0) for t in all_input_ids]),
            "attention_mask": torch.stack([pad_1d(t, 0) for t in all_attention_masks]),
            "labels":         torch.stack([pad_1d(t, self.label_pad_id) for t in all_labels]),
        }
        if all_token_type_ids:
            batch["token_type_ids"] = torch.stack([pad_1d(t, 0) for t in all_token_type_ids])

        return batch
