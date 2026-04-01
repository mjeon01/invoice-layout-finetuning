#!/usr/bin/env python
"""
Step 4a — Fine-tune LayoutLMv3 for invoice NER.

Usage:
    python scripts/04_train_layoutlmv3.py \
        --config configs/layoutlmv3.yaml \
        --train-data data/processed/train \
        --val-data data/processed/val_synthetic \
        --output-dir models/layoutlmv3-invoice

The best checkpoint (by eval_f1 on val_synthetic) is saved to output_dir.
After training, evaluate on real documents with 05_evaluate.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 for invoice NER")
    parser.add_argument("--config",      type=str, default="configs/layoutlmv3.yaml")
    parser.add_argument("--train-data",  type=str, default="data/processed/train")
    parser.add_argument("--val-data",    type=str, default="data/processed/val_synthetic")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Override output_dir from config")
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import TrainingArguments

    from src.data.dataset_builder import load_dataset
    from src.models.layoutlmv3_ner import LayoutLMv3NERModel
    from src.training.collator import LayoutLMv3DataCollator
    from src.training.config import TrainingConfig
    from src.training.trainer import InvoiceNERTrainer, build_training_args

    # Load config
    cfg = TrainingConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    print(f"Config: {cfg.model_name}, epochs={cfg.num_train_epochs}, lr={cfg.learning_rate}")
    print(f"Output: {cfg.output_dir}")

    # Check hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow. Consider using a GPU.")
        cfg.fp16 = False

    # Load datasets
    print(f"Loading train dataset from {args.train_data} ...")
    train_ds = load_from_disk(args.train_data)
    print(f"  {len(train_ds)} training samples")

    print(f"Loading val dataset from {args.val_data} ...")
    val_ds = load_from_disk(args.val_data)
    print(f"  {len(val_ds)} validation samples")

    # Model and processor
    print(f"Loading model: {cfg.model_name} ...")
    model_wrapper = LayoutLMv3NERModel(model_name=cfg.model_name)
    model, processor = model_wrapper.get_model_and_processor()
    model.to(device)

    # Collator
    collator = LayoutLMv3DataCollator(
        processor=processor,
        max_length=cfg.max_seq_length,
    )

    # Training arguments
    training_args_dict = cfg.to_hf_training_args()
    training_args = TrainingArguments(**training_args_dict)

    # Trainer
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
    print("Run 05_evaluate.py to evaluate on real documents.")


if __name__ == "__main__":
    main()
