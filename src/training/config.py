"""
Training configuration dataclass.

Loaded from YAML via scripts/04_train_*.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainingConfig:
    # ---- Model ----
    model_name: str = "microsoft/layoutlmv3-base"
    model_type: str = "layoutlmv3"          # "layoutlmv3" | "bert"
    max_seq_length: int = 512

    # ---- Data ----
    train_dataset_path: str = "data/processed/train"
    val_dataset_path: str   = "data/processed/val_synthetic"

    # ---- Batch & gradient ----
    per_device_train_batch_size: int  = 2
    per_device_eval_batch_size: int   = 4
    gradient_accumulation_steps: int  = 4

    # ---- Optimization ----
    num_train_epochs: int   = 15
    learning_rate: float    = 2e-5
    weight_decay: float     = 0.01
    warmup_ratio: float     = 0.1
    lr_scheduler_type: str  = "linear"
    adam_epsilon: float     = 1e-8

    # ---- Checkpointing ----
    output_dir: str               = "models/layoutlmv3-invoice"
    save_strategy: str            = "epoch"
    evaluation_strategy: str      = "epoch"
    save_total_limit: int         = 3
    load_best_model_at_end: bool  = True
    metric_for_best_model: str    = "eval_f1"
    greater_is_better: bool       = True

    # ---- Ablation ----
    ablation: str = ""   # "" | "no_bbox" | "no_image"

    # ---- Hardware ----
    fp16: bool  = True
    dataloader_num_workers: int = 0   # 0 required on Windows

    # ---- Logging ----
    logging_steps: int = 10
    report_to: str     = "tensorboard"
    logging_dir: str   = "runs/layoutlmv3"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls()
        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    def to_hf_training_args(self) -> dict:
        """Return a dict suitable for passing to transformers.TrainingArguments."""
        return {
            "output_dir":                    self.output_dir,
            "num_train_epochs":              self.num_train_epochs,
            "per_device_train_batch_size":   self.per_device_train_batch_size,
            "per_device_eval_batch_size":    self.per_device_eval_batch_size,
            "gradient_accumulation_steps":   self.gradient_accumulation_steps,
            "learning_rate":                 self.learning_rate,
            "weight_decay":                  self.weight_decay,
            "warmup_ratio":                  self.warmup_ratio,
            "lr_scheduler_type":             self.lr_scheduler_type,
            "adam_epsilon":                  self.adam_epsilon,
            "save_strategy":                 self.save_strategy,
            "eval_strategy":                 self.evaluation_strategy,
            "remove_unused_columns":         False,
            "save_total_limit":              self.save_total_limit,
            "load_best_model_at_end":        self.load_best_model_at_end,
            "metric_for_best_model":         self.metric_for_best_model,
            "greater_is_better":             self.greater_is_better,
            "fp16":                          self.fp16,
            "dataloader_num_workers":        self.dataloader_num_workers,
            "logging_steps":                 self.logging_steps,
            "report_to":                     self.report_to,
            "logging_dir":                   self.logging_dir,
        }
