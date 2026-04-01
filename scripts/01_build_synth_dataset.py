#!/usr/bin/env python
"""
Step 1 — 합성 PDF + GT JSON → HuggingFace Dataset 변환

합성 PDF의 각 페이지에서 PyMuPDF로 토큰+bbox 추출 후,
all_synth.json의 GT와 invoice_number로 매핑해 BIO 레이블을 생성합니다.

Usage:
    python scripts/01_build_synth_dataset.py \
        --pdf-dir  data/input/synth \
        --gt-dir   data/ground_truth \
        --out-dir  data/processed/synth \
        --split    all          # all | train | val

산출물:
    data/processed/synth/train/    (HF Dataset, ~800건)
    data/processed/synth/val/      (HF Dataset, ~200건)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bio_aligner import align_gt_to_tokens, build_invoice_index
from src.data.dataset_builder import _FEATURES, save_dataset
from src.data.pdf_extractor import extract_synth_page, pdf_page_count

import datasets
from tqdm import tqdm


def _load_gt(gt_dir: Path, split: str) -> list[dict]:
    """split에 맞는 GT JSON 로드."""
    name_map = {
        "all":   "all_synth.json",
        "train": "train_synth.json",
        "val":   "val_synth.json",
        "200":   "train_synth_200.json",
        "50":    "train_synth_50.json",
    }
    fname = name_map.get(split)
    if fname is None:
        raise ValueError(f"알 수 없는 split: {split}. 사용 가능: {list(name_map)}")
    path = gt_dir / fname
    print(f"GT 로드: {path}")
    return json.load(open(path, encoding="utf-8"))


def process_split(
    pdf_path: Path,
    gt_records: list[dict],
    split_name: str,
    out_dir: Path,
) -> None:
    """
    하나의 PDF + GT 세트를 처리해 HF Dataset으로 저장합니다.

    PDF 페이지 수 > GT 레코드 수인 경우(멀티페이지 인보이스 존재)
    invoice_number 역인덱스로 정확히 매핑합니다.
    """
    n_pages  = pdf_page_count(pdf_path)
    inv_idx  = build_invoice_index(gt_records)   # invoice_number → record index
    print(f"  PDF {pdf_path.name}: {n_pages}페이지 | GT: {len(gt_records)}건")

    records = []
    unmatched = 0

    for page_idx in tqdm(range(n_pages), desc=f"{split_name}"):
        sample = extract_synth_page(pdf_path, page_idx)
        tokens = sample["tokens"]
        if not tokens:
            continue

        # 이 페이지의 invoice_number 찾기 (토큰 중 정확히 매칭되는 것)
        matched_rec = None
        for inv_no, rec_idx in inv_idx.items():
            if any(t.strip() == inv_no or inv_no in " ".join(tokens) for t in tokens):
                matched_rec = gt_records[rec_idx]
                break

        if matched_rec is None:
            # invoice_number가 페이지에 없으면 BIO 없이 전부 O 레이블
            unmatched += 1
            ner_tags = [0] * len(tokens)
        else:
            ner_tags = align_gt_to_tokens(tokens, matched_rec["y_true"])

        inv_no_meta = (matched_rec or {}).get("meta", {}).get("invoice_number", f"page_{page_idx}")
        records.append({
            "id":       f"synth_{split_name}_{inv_no_meta}",
            "tokens":   tokens,
            "bboxes":   sample["bboxes"],
            "ner_tags": ner_tags,
            "image":    sample["image"],
        })

    if unmatched:
        print(f"  경고: {unmatched}페이지 GT 매핑 실패 (O레이블로 처리)")

    ds = datasets.Dataset.from_list(records, features=_FEATURES)
    out = out_dir / split_name
    ds.save_to_disk(str(out))
    print(f"  저장 완료: {out} ({len(ds)}건)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data/input/synth")
    parser.add_argument("--gt-dir",  default="data/ground_truth")
    parser.add_argument("--out-dir", default="data/processed/synth")
    parser.add_argument("--split",   default="all",
                        help="all | train | val | 200 | 50")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    gt_dir  = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "all":
        # train + val 모두 생성
        for sp, pdf_name, gt_name in [
            ("train", "train_synth.pdf", "train_synth.json"),
            ("val",   "val_synth.pdf",   "val_synth.json"),
        ]:
            pdf_path = pdf_dir / pdf_name
            gt_records = json.load(open(gt_dir / gt_name, encoding="utf-8"))
            process_split(pdf_path, gt_records, sp, out_dir)
    else:
        name_map = {
            "train": ("train_synth.pdf", "train_synth.json"),
            "val":   ("val_synth.pdf",   "val_synth.json"),
            "200":   ("train_synth.pdf", "train_synth_200.json"),
            "50":    ("train_synth.pdf", "train_synth_50.json"),
        }
        pdf_name, gt_name = name_map[args.split]
        pdf_path   = pdf_dir / pdf_name
        gt_records = json.load(open(gt_dir / gt_name, encoding="utf-8"))
        process_split(pdf_path, gt_records, args.split, out_dir)

    print("\n완료. 다음 단계: python scripts/02_build_real_dataset.py")


if __name__ == "__main__":
    main()
