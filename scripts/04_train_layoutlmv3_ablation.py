#!/usr/bin/env python
"""
Step 4c — Fine-tune LayoutLMv3 with ablation (no_bbox or no_image).

The ablation mode is read from the config yaml (ablation: "no_bbox" | "no_image").
--ablation CLI flag overrides the config value if provided.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 with ablation")
    parser.add_argument("--config",     type=str, default="configs/layoutlmv3_no_bbox.yaml")
    parser.add_argument("--train-data", type=str, default="data/processed/final/train")
    parser.add_argument("--val-data",   type=str, default="data/processed/final/val_synth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ablation",   type=str, default=None,
                        choices=["no_bbox", "no_image"],
                        help="Override ablation mode from config")
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import TrainingArguments

    from src.models.layoutlmv3_ner import LayoutLMv3NERModel
    from src.training.collator import LayoutLMv3DataCollator
    from src.training.config import TrainingConfig
    from src.training.trainer import InvoiceNERTrainer, build_training_args

    cfg = TrainingConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.ablation:
        cfg.ablation = args.ablation

    if not cfg.ablation:
        raise ValueError("ablation must be 'no_bbox' or 'no_image'. "
                         "Set it in the yaml or via --ablation.")

    print(f"Config : {cfg.model_name}, epochs={cfg.num_train_epochs}, lr={cfg.learning_rate}")
    print(f"Ablation: {cfg.ablation}")
    print(f"Output  : {cfg.output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow.")
        cfg.fp16 = False

    print(f"Loading train dataset from {args.train_data} ...")
    train_ds = load_from_disk(args.train_data)
    print(f"  {len(train_ds)} training samples")

    print(f"Loading val dataset from {args.val_data} ...")
    val_ds = load_from_disk(args.val_data)
    print(f"  {len(val_ds)} validation samples")

    print(f"Loading model: {cfg.model_name} ...")
    model_wrapper = LayoutLMv3NERModel(model_name=cfg.model_name)
    model, processor = model_wrapper.get_model_and_processor()
    model.to(device)

    collator = LayoutLMv3DataCollator(
        processor=processor,
        max_length=cfg.max_seq_length,
        ablation=cfg.ablation,
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
