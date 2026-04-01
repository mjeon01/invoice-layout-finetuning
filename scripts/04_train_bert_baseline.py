#!/usr/bin/env python
"""
Step 4b — Fine-tune BERT baseline (text-only) for invoice NER.

Usage:
    python scripts/04_train_bert_baseline.py \
        --config configs/bert_baseline.yaml \
        --train-data data/processed/train \
        --val-data data/processed/val_synthetic \
        --output-dir models/bert-baseline-invoice

Uses identical training data and label schema as LayoutLMv3,
but provides only token text (no bboxes, no image).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT baseline for invoice NER")
    parser.add_argument("--config",      type=str, default="configs/bert_baseline.yaml")
    parser.add_argument("--train-data",  type=str, default="data/processed/train")
    parser.add_argument("--val-data",    type=str, default="data/processed/val_synthetic")
    parser.add_argument("--output-dir",  type=str, default=None)
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import TrainingArguments

    from src.models.bert_baseline_ner import BertBaselineNERModel
    from src.training.collator import BertDataCollator
    from src.training.config import TrainingConfig
    from src.training.trainer import InvoiceNERTrainer, build_training_args

    cfg = TrainingConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    print(f"Config: {cfg.model_name}, epochs={cfg.num_train_epochs}, lr={cfg.learning_rate}")
    print(f"Output: {cfg.output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        cfg.fp16 = False

    print(f"Loading datasets ...")
    train_ds = load_from_disk(args.train_data)
    val_ds   = load_from_disk(args.val_data)
    print(f"  train={len(train_ds)}, val={len(val_ds)}")

    print(f"Loading model: {cfg.model_name} ...")
    model_wrapper = BertBaselineNERModel(model_name=cfg.model_name)
    model, tokenizer = model_wrapper.get_model_and_tokenizer()
    model.to(device)

    collator = BertDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
    )

    training_args = TrainingArguments(**cfg.to_hf_training_args())

    trainer = InvoiceNERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("\nStarting training ...")
    trainer.train()

    print(f"\nTraining complete. Best model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
