#!/usr/bin/env python
"""
Step 3 — 합성 + 실제 Dataset을 최종 DatasetDict로 조립

01, 02 스크립트 실행 후 생성된 개별 Dataset을 합쳐
학습/검증 DatasetDict를 만듭니다.

최종 splits:
  train      : data/processed/synth/train  (합성 800건)
  val_synth  : data/processed/synth/val    (합성 200건)
  test_real  : data/processed/real/val + real/test 합산 (실제 ~41건)

Usage:
    python scripts/03_build_final_dataset.py \
        --synth-dir  data/processed/synth \
        --real-dir   data/processed/real \
        --out-dir    data/processed/final
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-dir", default="data/processed/synth")
    parser.add_argument("--real-dir",  default="data/processed/real")
    parser.add_argument("--out-dir",   default="data/processed/final")
    args = parser.parse_args()

    synth_dir = Path(args.synth_dir)
    real_dir  = Path(args.real_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}

    def _load(path: Path, name: str):
        if path.exists():
            ds = datasets.load_from_disk(str(path))
            print(f"  {name}: {len(ds)}건")
            return ds
        else:
            print(f"  {name}: 없음 (건너뜀) — 경로: {path}")
            return None

    print("로딩 중...")
    for key, path in [
        ("train",     synth_dir / "train"),
        ("val_synth", synth_dir / "val"),
    ]:
        ds = _load(path, key)
        if ds is not None:
            splits[key] = ds

    # 실제 문서 val + test를 하나의 test_real로 합산 (~41건)
    real_parts = []
    for label, path in [("val", real_dir / "val"), ("test", real_dir / "test")]:
        ds = _load(path, f"real/{label}")
        if ds is not None:
            real_parts.append(ds)
    if real_parts:
        splits["test_real"] = datasets.concatenate_datasets(real_parts)
        print(f"  test_real (합산): {len(splits['test_real'])}건")

    if not splits:
        raise SystemExit("처리된 데이터 없음. 01, 02 스크립트를 먼저 실행하세요.")

    dataset_dict = datasets.DatasetDict(splits)
    dataset_dict.save_to_disk(str(out_dir))

    print(f"\n최종 DatasetDict 저장: {out_dir}")
    for k, v in dataset_dict.items():
        print(f"  {k:12s}: {len(v):>5}건")

    print("\n다음 단계:")
    print("  python scripts/04_train_layoutlmv3.py  --train-data data/processed/final/train  --val-data data/processed/final/val_synth")
    print("  python scripts/04_train_bert_baseline.py  --train-data data/processed/final/train  --val-data data/processed/final/val_synth")


if __name__ == "__main__":
    main()
