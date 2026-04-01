#!/usr/bin/env python
"""
Step 2 — 실제 PDF + GT JSON → HuggingFace Dataset 변환

실제 PDF는 스캔 이미지이므로 Tesseract OCR로 토큰+bbox 추출 후
invoices_*_truth.json의 GT와 invoice_number로 매핑해 BIO 레이블 생성.

Usage:
    python scripts/02_build_real_dataset.py \
        --pdf-dir  data/input \
        --gt-dir   data/ground_truth \
        --out-dir  data/processed/real

산출물:
    data/processed/real/val/    (HF Dataset, ~14건)
    data/processed/real/test/   (HF Dataset, ~14건)

필수:
    pip install pytesseract
    Tesseract OCR 설치: https://github.com/tesseract-ocr/tesseract
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets
from tqdm import tqdm

from src.data.bio_aligner import align_gt_to_tokens, build_invoice_index
from src.data.dataset_builder import _FEATURES
from src.data.pdf_extractor import extract_real_page, pdf_page_count


def _flatten_truth(truth_data) -> list[dict]:
    """
    실제 GT JSON의 중첩 리스트를 flat list of dicts로 변환.
    중첩 깊이에 관계없이 dict가 나올 때까지 재귀적으로 풀어냅니다.
    """
    result = []
    def _recurse(item):
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, list):
            for sub in item:
                _recurse(sub)
    _recurse(truth_data)
    return result


def process_real_split(
    pdf_path: Path,
    gt_flat: list[dict],
    split_name: str,
    out_dir: Path,
    ocr_backend: str = "tesseract",
) -> None:
    """
    실제 PDF의 모든 페이지를 OCR 후 GT와 매핑해 Dataset으로 저장.

    GT가 있는 페이지만 BIO 레이블 부여, 나머지는 건너뜁니다.
    """
    n_pages = pdf_page_count(pdf_path)
    inv_idx = build_invoice_index(gt_flat)
    print(f"  PDF {pdf_path.name}: {n_pages}페이지 | GT: {len(gt_flat)}건")
    print(f"  GT invoice_numbers: {list(inv_idx.keys())[:5]} ...")

    records  = []
    matched  = 0
    skipped  = 0

    for page_idx in tqdm(range(n_pages), desc=f"OCR {split_name}"):
        sample = extract_real_page(pdf_path, page_idx, ocr_backend=ocr_backend)
        tokens = sample["tokens"]

        if not tokens:
            skipped += 1
            continue

        # invoice_number로 GT 매핑 시도
        matched_rec = None
        for inv_no, rec_idx in inv_idx.items():
            # 정확 매칭 우선, 이후 부분 문자열
            joined = " ".join(tokens)
            if inv_no in joined or any(t.strip() == inv_no for t in tokens):
                matched_rec = gt_flat[rec_idx]
                break

        if matched_rec is None:
            # GT 없는 페이지: 평가 불가 → 건너뜀
            skipped += 1
            continue

        ner_tags = align_gt_to_tokens(tokens, matched_rec["y_true"])
        inv_no_str = matched_rec.get("meta", {}).get("invoice_number", f"page_{page_idx}")

        records.append({
            "id":       f"real_{split_name}_{inv_no_str}",
            "tokens":   tokens,
            "bboxes":   sample["bboxes"],
            "ner_tags": ner_tags,
            "image":    sample["image"],
        })
        matched += 1

    print(f"  GT 매핑 성공: {matched}건, 건너뜀: {skipped}페이지")

    if not records:
        print(f"  경고: {split_name} 처리된 샘플 없음")
        return

    ds  = datasets.Dataset.from_list(records, features=_FEATURES)
    out = out_dir / split_name
    ds.save_to_disk(str(out))
    print(f"  저장 완료: {out} ({len(ds)}건)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir",     default="data/input")
    parser.add_argument("--gt-dir",      default="data/ground_truth")
    parser.add_argument("--out-dir",     default="data/processed/real")
    parser.add_argument("--ocr-backend", default="easyocr")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    gt_dir  = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, pdf_name, gt_name in [
        ("val",  "invoices_val.pdf",  "invoices_val_truth.json"),
        ("test", "invoices_test.pdf", "invoices_test_truth.json"),
    ]:
        pdf_path = pdf_dir / pdf_name
        truth    = json.load(open(gt_dir / gt_name, encoding="utf-8"))
        gt_flat  = _flatten_truth(truth)
        print(f"\n[{split}] {pdf_name} → {len(gt_flat)}건 GT")
        process_real_split(pdf_path, gt_flat, split, out_dir,
                           ocr_backend=args.ocr_backend)

    print("\n완료. 다음 단계: python scripts/04_train_layoutlmv3.py")


if __name__ == "__main__":
    main()
