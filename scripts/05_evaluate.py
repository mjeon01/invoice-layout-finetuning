#!/usr/bin/env python
"""
Step 5 — Evaluate all models on real documents and produce comparison report.

Usage:
    python scripts/05_evaluate.py \
        --layoutlmv3-model      models/layoutlmv3-invoice \
        --bert-model            models/bert-baseline-invoice \
        --lmv3-no-bbox-model    models/layoutlmv3-no-bbox \
        --lmv3-no-image-model   models/layoutlmv3-no-image \
        --test-data             data/processed/final/test_real \
        --output-dir            results

Outputs:
    results/layoutlmv3_eval.json
    results/bert_baseline_eval.json
    results/layoutlmv3_no_bbox_eval.json
    results/layoutlmv3_no_image_eval.json
    results/comparison_report.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_layoutlmv3_inference(
    model_path: str,
    dataset,
    device: str,
    ablation: str = "",
) -> tuple[list, list, list]:
    """Run LayoutLMv3 (optionally with ablation) on all samples.

    Returns (tokens_all, true_all, pred_all).
    """
    from src.data.label_schema import id2label
    from src.models.layoutlmv3_ner import LayoutLMv3NERModel
    from tqdm import tqdm
    import numpy as np
    from PIL import Image

    wrapper = LayoutLMv3NERModel(
        model_name="microsoft/layoutlmv3-base",
        from_pretrained_path=model_path,
    )
    model, processor = wrapper.get_model_and_processor()
    model.to(device)
    model.eval()

    tokens_all, true_all, pred_all = [], [], []

    for sample in tqdm(dataset, desc=f"LayoutLMv3{' (' + ablation + ')' if ablation else ''} inference"):
        true_ids = [t for t in sample["ner_tags"] if t != -100]
        true_labels = [id2label.get(t, "O") for t in true_ids]

        inf_sample = dict(sample)
        if ablation == "no_bbox":
            inf_sample["bboxes"] = [[0, 0, 0, 0]] * len(sample["bboxes"])
        elif ablation == "no_image":
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            inf_sample["image"] = Image.fromarray(
                np.zeros((img.height, img.width, 3), dtype=np.uint8)
            )

        pred_labels = LayoutLMv3NERModel.predict(model, processor, inf_sample, device=device)
        min_len = min(len(true_labels), len(pred_labels))
        tokens_all.append(list(sample["tokens"])[:min_len])
        true_all.append(true_labels[:min_len])
        pred_all.append(pred_labels[:min_len])

    return tokens_all, true_all, pred_all


def run_bert_inference(model_path: str, dataset, device: str) -> tuple[list, list, list]:
    """Run BERT on all samples.

    Returns (tokens_all, true_all, pred_all).
    """
    from src.data.label_schema import id2label
    from src.models.bert_baseline_ner import BertBaselineNERModel
    from tqdm import tqdm

    wrapper = BertBaselineNERModel(
        model_name="bert-base-uncased",
        from_pretrained_path=model_path,
    )
    model, tokenizer = wrapper.get_model_and_tokenizer()
    model.to(device)
    model.eval()

    tokens_all, true_all, pred_all = [], [], []

    for sample in tqdm(dataset, desc="BERT inference"):
        true_ids = [t for t in sample["ner_tags"] if t != -100]
        true_labels = [id2label.get(t, "O") for t in true_ids]

        pred_labels = BertBaselineNERModel.predict(
            model, tokenizer, sample["tokens"], device=device
        )
        min_len = min(len(true_labels), len(pred_labels))
        tokens_all.append(list(sample["tokens"])[:min_len])
        true_all.append(true_labels[:min_len])
        pred_all.append(pred_labels[:min_len])

    return tokens_all, true_all, pred_all


def evaluate_and_save(
    name: str,
    tokens_all: list,
    true_all: list,
    pred_all: list,
    output_dir: Path,
    filename: str,
) -> dict:
    from src.evaluation.metrics import compute_entity_metrics, compute_kie_metrics, compute_token_metrics

    results = compute_entity_metrics(true_all, pred_all)
    results.update({"token_metrics": compute_token_metrics(true_all, pred_all)})
    results.update({"kie_metrics": compute_kie_metrics(tokens_all, true_all, pred_all)})

    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    kie_f1 = results["kie_metrics"]["kie_f1"]
    print(f"  entity F1 = {results['f1']:.4f}  |  KIE F1 = {kie_f1:.4f}  →  {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all models on real documents")
    parser.add_argument("--layoutlmv3-model",    type=str, default="models/layoutlmv3-invoice")
    parser.add_argument("--bert-model",          type=str, default="models/bert-baseline-invoice")
    parser.add_argument("--lmv3-no-bbox-model",  type=str, default=None)
    parser.add_argument("--lmv3-no-image-model", type=str, default=None)
    parser.add_argument("--test-data",           type=str, default="data/processed/final/test_real")
    parser.add_argument("--output-dir",          type=str, default="results")
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from src.evaluation.comparison import generate_comparison_report

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test set: {args.test_data}")
    test_ds = load_from_disk(args.test_data)
    print(f"  {len(test_ds)} real documents\n")

    all_results: dict[str, dict] = {}

    # ----- BERT -----
    print(f"Evaluating BERT from {args.bert_model} ...")
    bert_tokens, bert_true, bert_pred = run_bert_inference(args.bert_model, test_ds, device)
    all_results["BERT"] = evaluate_and_save(
        "BERT", bert_tokens, bert_true, bert_pred, output_dir, "bert_baseline_eval.json"
    )

    # ----- LayoutLMv3 no bbox -----
    if args.lmv3_no_bbox_model:
        print(f"\nEvaluating LayoutLMv3 (no bbox) from {args.lmv3_no_bbox_model} ...")
        nb_tokens, nb_true, nb_pred = run_layoutlmv3_inference(
            args.lmv3_no_bbox_model, test_ds, device, ablation="no_bbox"
        )
        all_results["LMv3 (no bbox)"] = evaluate_and_save(
            "LMv3 (no bbox)", nb_tokens, nb_true, nb_pred, output_dir, "layoutlmv3_no_bbox_eval.json"
        )

    # ----- LayoutLMv3 no image -----
    if args.lmv3_no_image_model:
        print(f"\nEvaluating LayoutLMv3 (no image) from {args.lmv3_no_image_model} ...")
        ni_tokens, ni_true, ni_pred = run_layoutlmv3_inference(
            args.lmv3_no_image_model, test_ds, device, ablation="no_image"
        )
        all_results["LMv3 (no image)"] = evaluate_and_save(
            "LMv3 (no image)", ni_tokens, ni_true, ni_pred, output_dir, "layoutlmv3_no_image_eval.json"
        )

    # ----- LayoutLMv3 full -----
    print(f"\nEvaluating LayoutLMv3 (full) from {args.layoutlmv3_model} ...")
    lv3_tokens, lv3_true, lv3_pred = run_layoutlmv3_inference(args.layoutlmv3_model, test_ds, device)
    all_results["LayoutLMv3"] = evaluate_and_save(
        "LayoutLMv3", lv3_tokens, lv3_true, lv3_pred, output_dir, "layoutlmv3_eval.json"
    )

    # ----- Comparison report -----
    print("\nGenerating comparison report ...")
    report_path = output_dir / "comparison_report.md"
    generate_comparison_report(all_results, output_path=report_path)

    # ----- Summary -----
    print(f"\n{'='*75}")
    print(f"RESULTS SUMMARY  ({len(test_ds)} real documents)")
    print(f"{'='*75}")
    print(f"{'Model':<22} {'Entity F1':>10} {'KIE F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*65}")
    for name, res in all_results.items():
        kie_f1 = res.get("kie_metrics", {}).get("kie_f1", 0.0)
        print(f"{name:<22} {res['f1']:>10.4f} {kie_f1:>10.4f} "
              f"{res['precision']:>10.4f} {res['recall']:>10.4f}")
    print(f"\nFull comparison: {report_path}")


if __name__ == "__main__":
    main()
