"""
Assembles HuggingFace Dataset objects from raw annotation JSON files.

Synthetic split: 900 train + 100 val_synthetic
Real split:      41 val_real  (all real documents used for evaluation)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import datasets
from PIL import Image

from .label_schema import LABELS, NUM_LABELS


# HuggingFace feature schema shared by both synthetic and real data
_FEATURES = datasets.Features({
    "id":       datasets.Value("string"),
    "tokens":   datasets.Sequence(datasets.Value("string")),
    "bboxes":   datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
    "ner_tags": datasets.Sequence(datasets.Value("int32")),   # label id (int)
    "image":    datasets.Image(),
})


def _load_annotation(ann_path: Path) -> dict | None:
    """Load one annotation JSON and attach its PIL image."""
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    img_path = Path(ann.get("image_path", ""))
    if not img_path.exists():
        # Try relative to annotation file's parent's parent
        img_path = ann_path.parent.parent / "images" / (ann_path.stem + ".png")

    if not img_path.exists():
        return None

    image = Image.open(img_path).convert("RGB")
    return {
        "id":       ann["id"],
        "tokens":   ann["tokens"],
        "bboxes":   ann["bboxes"],
        "ner_tags": ann["ner_tags"],
        "image":    image,
    }


def build_hf_dataset(annotation_dir: Path, image_dir: Path | None = None) -> datasets.Dataset:
    """
    Build a HuggingFace Dataset from a directory of annotation JSON files.

    Args:
        annotation_dir: Directory containing <id>.json annotation files.
        image_dir:      Directory containing <id>.png images.
                        If None, uses annotation_dir/../images/.

    Returns:
        datasets.Dataset with features matching _FEATURES.
    """
    annotation_dir = Path(annotation_dir)
    if image_dir is None:
        image_dir = annotation_dir.parent / "images"

    records = []
    for ann_path in sorted(annotation_dir.glob("*.json")):
        rec = _load_annotation(ann_path)
        if rec is not None:
            records.append(rec)

    if not records:
        raise ValueError(f"No valid annotations found in {annotation_dir}")

    return datasets.Dataset.from_list(records, features=_FEATURES)


def create_train_val_split(
    synthetic_ds: datasets.Dataset,
    real_ds: datasets.Dataset,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> datasets.DatasetDict:
    """
    Split synthetic data into train/val_synthetic and combine with real val set.

    Returns DatasetDict with keys:
      train           — 900 synthetic samples
      val_synthetic   — 100 synthetic samples
      val_real        — all real documents (41)
    """
    split = synthetic_ds.train_test_split(
        test_size=1.0 - train_ratio,
        seed=seed,
        shuffle=True,
    )
    return datasets.DatasetDict({
        "train":         split["train"],
        "val_synthetic": split["test"],
        "val_real":      real_ds,
    })


def save_dataset(dataset_dict: datasets.DatasetDict, output_dir: Path) -> None:
    """Save dataset splits to disk in Arrow format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"Saved dataset to {output_dir}")
    for split, ds in dataset_dict.items():
        print(f"  {split}: {len(ds)} samples")


def load_dataset(dataset_dir: Path) -> datasets.DatasetDict:
    """Load a previously saved DatasetDict from disk."""
    return datasets.load_from_disk(str(dataset_dir))
