#!/usr/bin/env python
"""
Step 5 — Evaluate both models on real documents and produce comparison report.

Usage:
    python scripts/05_evaluate.py \
        --layoutlmv3-model models/layoutlmv3-invoice \
        --bert-model models/bert-baseline-invoice \
        --test-data data/processed/val_real \
        --output-dir results

Outputs:
    results/layoutlmv3_eval.json
    results/bert_baseline_eval.json
    results/comparison_report.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_layoutlmv3_inference(model_path: str, dataset, device: str) -> tuple[list, list]:
    """Run LayoutLMv3 on all samples, return (true_labels, pred_labels)."""
    from src.data.label_schema import id2label
    from src.models.layoutlmv3_ner import LayoutLMv3NERModel
    from tqdm import tqdm

    wrapper = LayoutLMv3NERModel(
        model_name="microsoft/layoutlmv3-base",
        from_pretrained_path=model_path,
    )
    model, processor = wrapper.get_model_and_processor()
    model.to(device)
    model.eval()

    true_all, pred_all = [], []

    for sample in tqdm(dataset, desc="LayoutLMv3 inference"):
        true_ids = [t for t in sample["ner_tags"] if t != -100]
        true_labels = [id2label.get(t, "O") for t in true_ids]

        pred_labels = LayoutLMv3NERModel.predict(model, processor, sample, device=device)
        # Align lengths
        min_len = min(len(true_labels), len(pred_labels))
        true_all.append(true_labels[:min_len])
        pred_all.append(pred_labels[:min_len])

    return true_all, pred_all


def run_bert_inference(model_path: str, dataset, device: str) -> tuple[list, list]:
    """Run BERT on all samples, return (true_labels, pred_labels)."""
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

    true_all, pred_all = [], []

    for sample in tqdm(dataset, desc="BERT inference"):
        true_ids = [t for t in sample["ner_tags"] if t != -100]
        true_labels = [id2label.get(t, "O") for t in true_ids]

        pred_labels = BertBaselineNERModel.predict(
            model, tokenizer, sample["tokens"], device=device
        )
        min_len = min(len(true_labels), len(pred_labels))
        true_all.append(true_labels[:min_len])
        pred_all.append(pred_labels[:min_len])

    return true_all, pred_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on real documents")
    parser.add_argument("--layoutlmv3-model", type=str,
                        default="models/layoutlmv3-invoice")
    parser.add_argument("--bert-model",       type=str,
                        default="models/bert-baseline-invoice")
    parser.add_argument("--test-data",        type=str,
                        default="data/processed/final/test_real")
    parser.add_argument("--output-dir",       type=str, default="results")
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk

    from src.evaluation.comparison import generate_comparison_report
    from src.evaluation.metrics import compute_entity_metrics, compute_token_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test set: {args.test_data}")
    test_ds = load_from_disk(args.test_data)
    print(f"  {len(test_ds)} real documents")

    # ----- LayoutLMv3 -----
    print(f"\nEvaluating LayoutLMv3 from {args.layoutlmv3_model} ...")
    lv3_true, lv3_pred = run_layoutlmv3_inference(args.layoutlmv3_model, test_ds, device)
    lv3_results = compute_entity_metrics(lv3_true, lv3_pred)
    lv3_token   = compute_token_metrics(lv3_true, lv3_pred)
    lv3_results.update({"token_metrics": lv3_token})

    lv3_out = output_dir / "layoutlmv3_eval.json"
    with open(lv3_out, "w", encoding="utf-8") as f:
        json.dump(lv3_results, f, ensure_ascii=False, indent=2)
    print(f"  entity F1 = {lv3_results['f1']:.4f}  →  {lv3_out}")

    # ----- BERT -----
    print(f"\nEvaluating BERT baseline from {args.bert_model} ...")
    bert_true, bert_pred = run_bert_inference(args.bert_model, test_ds, device)
    bert_results = compute_entity_metrics(bert_true, bert_pred)
    bert_token   = compute_token_metrics(bert_true, bert_pred)
    bert_results.update({"token_metrics": bert_token})

    bert_out = output_dir / "bert_baseline_eval.json"
    with open(bert_out, "w", encoding="utf-8") as f:
        json.dump(bert_results, f, ensure_ascii=False, indent=2)
    print(f"  entity F1 = {bert_results['f1']:.4f}  →  {bert_out}")

    # ----- Comparison report -----
    print("\nGenerating comparison report ...")
    report_path = output_dir / "comparison_report.md"
    generate_comparison_report(lv3_results, bert_results, output_path=report_path)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (real {len(test_ds)} documents)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*50}")
    print(f"{'LayoutLMv3':<20} {lv3_results['precision']:>10.4f} "
          f"{lv3_results['recall']:>10.4f} {lv3_results['f1']:>10.4f}")
    print(f"{'BERT baseline':<20} {bert_results['precision']:>10.4f} "
          f"{bert_results['recall']:>10.4f} {bert_results['f1']:>10.4f}")
    print(f"\nFull comparison: {report_path}")


if __name__ == "__main__":
    main()
