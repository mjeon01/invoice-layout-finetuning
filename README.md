# Fishery Invoice Information Extraction — LayoutLMv3 Fine-Tuning

해외 수산물 무역 인보이스에서 구조화된 정보를 자동 추출하는 LayoutLMv3 파인튜닝 파이프라인입니다.

---

## 배경 및 문제

해외 수산물 무역 인보이스는 수출국·업체마다 레이아웃이 제각각입니다.  
기존 처리 방식의 한계:

| 방식 | 문제 |
|------|------|
| 수작업 | 비효율, 인적 오류 |
| LLM (GPT 등) | 수치 환각, 품목 누락, 느린 추론 |

**제안**: 합성 데이터로 LayoutLMv3을 파인튜닝해 레이아웃 인식 기반 경량 추출기를 만듭니다.

---

## 접근 방법 요약

```
[학습]  합성 인보이스 PDF 800건  ──► LayoutLMv3 파인튜닝
                                      (텍스트 + 바운딩박스 + 이미지)
[평가]  실제 인보이스 41건       ──► 일반화 성능 측정

[비교]  BERT (텍스트만)         ──► 레이아웃 정보의 기여도 정량화
```

### 왜 LayoutLMv3인가?

일반 NLP 모델(BERT 등)은 텍스트만 봅니다.  
인보이스에서 `9.76`이 **단가**인지 **중량**인지는 **어디에 위치하는지**로 결정됩니다.  
LayoutLMv3는 텍스트 + 바운딩박스(위치) + 문서 이미지를 동시에 처리합니다.

```
입력:  토큰 ["9.76"]  +  bbox [504, 210, 545, 220]  +  이미지 패치
           ↑ 텍스트           ↑ 페이지 내 위치              ↑ 시각 정보
출력:  B-ITEM_UNIT_PRICE   (단가 필드)
```

---

## 데이터셋

### 합성 데이터 (학습용)
- **출처**: 프로그래밍 방식으로 생성한 인보이스 PDF
- **규모**: 1,000건 (학습 800 / 검증 200)
- **특징**: PDF 텍스트 레이어에서 바운딩박스 정확 추출 (OCR 불필요)
- **파일**: `data/input/synth/`, `data/ground_truth/*_synth.json`

### 실제 데이터 (평가용)
- **출처**: 실제 해외 수산물 무역 인보이스 스캔본
- **규모**: 41건 (val 14 + test 14 + 추가 13건) — 최종 평가셋으로 통합
- **특징**: 이미지 기반 PDF → EasyOCR로 토큰 추출 (별도 바이너리 설치 불필요)
- **파일**: `data/input/invoices_*.pdf`, `data/ground_truth/invoices_*_truth.json`

### GT 포맷 (`ground_truth/*.json`)

```json
{
  "meta": { "invoice_number": "CHQ24PS01", "invoice_date": "2022-09-13" },
  "y_true": {
    "invoice_number":          "CHQ24PS01",
    "exporter.name":           "PT. INDO PACIFIC FISHERIES",
    "importer.name":           "SEOUL FRESH IMPORTS CO., LTD",
    "line_items[0].description": "FROZEN ROUND SCAD",
    "line_items[0].quantity":    161,
    "line_items[0].unit_price":  9.76,
    "totals.final_total":        46145.93
  }
}
```

---

## 추출 대상 필드 (22개 엔티티)

| 카테고리 | 엔티티 | 설명 |
|----------|--------|------|
| 인보이스 | `INVOICE_NUMBER`, `INVOICE_DATE` | 번호, 발행일 |
| 수출업체 | `EXPORTER_NAME`, `EXPORTER_ADDRESS` | 명칭, 주소 |
| 수입업체 | `IMPORTER_NAME`, `IMPORTER_ADDRESS` | 명칭, 주소 |
| 운송 | `PORT_LOADING`, `PORT_DESTINATION`, `INCOTERMS` | 선적항, 도착항, 거래조건 |
| 라인 아이템 | `ITEM_DESC`, `ITEM_QTY`, `ITEM_QTY_UNIT` | 품목, 수량, 단위 |
| | `ITEM_NET_WEIGHT`, `ITEM_NET_WEIGHT_UNIT` | 중량, 중량 단위 |
| | `ITEM_UNIT_PRICE`, `ITEM_CURRENCY`, `ITEM_AMOUNT` | 단가, 통화, 금액 |
| | `ITEM_SIZE_GRADE` | 규격·등급 |
| 합계 | `TOTAL_QTY`, `TOTAL_NET_WEIGHT` | 총 수량, 총 중량 |
| | `TOTAL_AMOUNT`, `TOTAL_CURRENCY` | 총 금액, 통화 |

레이블 체계: `O` + `B-{ENTITY}` + `I-{ENTITY}` = **45개** (BIO 방식)

---

## 파이프라인

```
data/input/synth/*.pdf          data/input/invoices_*.pdf
data/ground_truth/*_synth.json  data/ground_truth/*_truth.json
        │                                   │
        ▼ PyMuPDF (텍스트 레이어)           ▼ EasyOCR
   tokens + bboxes                     tokens + bboxes
        │                                   │
        └──────────┬────────────────────────┘
                   ▼ BIO Aligner (GT값 → 토큰 레이블)
             HuggingFace Dataset
             (tokens, bboxes, ner_tags, image)
                   │
          ┌────────┴────────┐
          ▼                 ▼
    LayoutLMv3-base     BERT-base
    (텍스트+bbox+이미지)   (텍스트만)
          │                 │
          └────────┬────────┘
                   ▼
          실제 41건 평가
          Entity-level F1 비교
```

---

## 프로젝트 구조

```
invoice-layout-finetuning/
├── data/
│   ├── input/
│   │   ├── synth/                   # 합성 인보이스 PDF
│   │   │   ├── train_synth.pdf      # 학습용 (850페이지)
│   │   │   └── val_synth.pdf        # 합성 검증용 (213페이지)
│   │   ├── invoices_val.pdf         # 실제 문서 val (20페이지)
│   │   └── invoices_test.pdf        # 실제 문서 test (21페이지)
│   ├── ground_truth/
│   │   ├── train_synth.json         # 합성 학습 GT (800건)
│   │   ├── val_synth.json           # 합성 검증 GT (200건)
│   │   ├── train_synth_200.json     # 소규모 실험용 (170건)
│   │   ├── train_synth_50.json      # 소규모 실험용 (40건)
│   │   ├── invoices_val_truth.json  # 실제 val GT (14건)
│   │   └── invoices_test_truth.json # 실제 test GT (14건)
│   └── processed/                   # 변환된 HF Dataset (스크립트 실행 후 생성)
│       ├── synth/train, val/
│       ├── real/val, test/
│       └── final/                   # 최종 학습/평가 DatasetDict
│
├── src/
│   ├── data/
│   │   ├── label_schema.py          # BIO 레이블 22종 정의, GT키 매핑
│   │   ├── pdf_extractor.py         # PDF → 토큰+bbox (합성: PyMuPDF, 실제: EasyOCR)
│   │   ├── bio_aligner.py           # GT 필드값 → BIO 토큰 레이블 자동 정렬
│   │   └── dataset_builder.py       # HuggingFace Dataset 피처 스키마
│   ├── models/
│   │   ├── layoutlmv3_ner.py        # LayoutLMv3 토큰분류 래퍼 + 추론
│   │   └── bert_baseline_ner.py     # BERT 베이스라인 래퍼 + 추론
│   ├── training/
│   │   ├── config.py                # TrainingConfig (YAML 로더)
│   │   ├── collator.py              # 서브토큰 BIO 정렬 + 멀티모달 패딩
│   │   └── trainer.py               # seqeval F1 compute_metrics 포함 Trainer
│   └── evaluation/
│       ├── metrics.py               # Entity-level / Token-level F1
│       └── comparison.py            # BERT vs LayoutLMv3 Markdown 비교표 생성
│
├── scripts/
│   ├── 01_build_synth_dataset.py    # 합성 PDF+GT → HF Dataset
│   ├── 02_build_real_dataset.py     # 실제 PDF+GT → EasyOCR → HF Dataset
│   ├── 03_build_final_dataset.py    # 두 Dataset 합쳐 최종 DatasetDict
│   ├── 04_train_layoutlmv3.py       # LayoutLMv3 파인튜닝
│   ├── 04_train_bert_baseline.py    # BERT 베이스라인 학습
│   └── 05_evaluate.py               # 실제 문서 평가 + 비교 리포트
│
├── configs/
│   ├── layoutlmv3.yaml              # 학습 하이퍼파라미터
│   └── bert_baseline.yaml
│
├── models/                          # 학습된 모델 가중치 (gitignored)
├── results/                         # 평가 결과 JSON + 비교 리포트
└── requirements.txt
```

---

## 실행 방법

### 환경 설정

```bash
pip install -r requirements.txt
```

> EasyOCR은 Python 패키지만으로 동작하며 별도 바이너리 설치가 불필요합니다.

### Step 1 — 합성 데이터 변환 (OCR 불필요, ~수 분)

```bash
python scripts/01_build_synth_dataset.py --pdf-dir data/input/synth --gt-dir data/ground_truth --out-dir data/processed/synth --split all
```

### Step 2 — 실제 문서 변환 (EasyOCR)

```bash
python scripts/02_build_real_dataset.py --pdf-dir data/input --gt-dir data/ground_truth --out-dir data/processed/real
```

### Step 3 — 최종 DatasetDict 조립

```bash
python scripts/03_build_final_dataset.py --synth-dir data/processed/synth --real-dir data/processed/real --out-dir data/processed/final
```

출력:
```
data/processed/final/
├── train/       합성 800건  (LayoutLMv3 + BERT 공통 학습)
├── val_synth/   합성 200건  (학습 중 모니터링)
└── test_real/   실제 41건   (최종 평가 — val + test 통합)
```

### Step 4 — 모델 학습

```bash
# LayoutLMv3 (~2-4시간, GPU 권장)
python scripts/04_train_layoutlmv3.py --config configs/layoutlmv3.yaml --train-data data/processed/final/train --val-data data/processed/final/val_synth --output-dir models/layoutlmv3-invoice

# BERT 베이스라인 (~30분)
python scripts/04_train_bert_baseline.py --config configs/bert_baseline.yaml --train-data data/processed/final/train --val-data data/processed/final/val_synth --output-dir models/bert-baseline-invoice
```

### Step 5 — 실제 문서 평가

```bash
python scripts/05_evaluate.py --layoutlmv3-model models/layoutlmv3-invoice --bert-model models/bert-baseline-invoice --test-data data/processed/final/test_real --output-dir results
```

출력:
- `results/layoutlmv3_eval.json` — 필드별 Precision/Recall/F1
- `results/bert_baseline_eval.json`
- `results/comparison_report.md` — 두 모델 비교표

---

## 학습 설정

### LayoutLMv3 (`configs/layoutlmv3.yaml`)

| 항목 | 값 | 비고 |
|------|----|------|
| 베이스 모델 | `microsoft/layoutlmv3-base` | 125M 파라미터 |
| 입력 | 토큰 + bbox + 이미지 | 멀티모달 |
| 유효 배치 | 8 (per_device=2 × accum=4) | GPU 메모리 절약 |
| 에폭 | 15 | early stop: eval_f1 기준 |
| Learning rate | 2e-5 (linear, warmup 10%) | |
| num_labels | 45 | O + B/I × 22 |

### BERT 베이스라인 (`configs/bert_baseline.yaml`)

| 항목 | 값 | 비고 |
|------|----|------|
| 베이스 모델 | `bert-base-uncased` | 110M 파라미터 |
| 입력 | 토큰만 | bbox·이미지 없음 |
| 유효 배치 | 16 | |
| 에폭 | 10 | |
| Learning rate | 3e-5 | |

---

## 평가 지표

**주요 지표**: seqeval **entity-level macro F1** (O 레이블 제외)

엔티티 스팬의 타입과 경계가 모두 정확해야 TP로 인정됩니다.

```
예시:
  GT:   [B-ITEM_UNIT_PRICE] [O]       [B-ITEM_AMOUNT]
  Pred: [B-ITEM_UNIT_PRICE] [O]       [B-ITEM_UNIT_PRICE]  ← 타입 오류 → FP
```

레이아웃 정보 효과가 기대되는 필드:

| 필드 쌍 | 혼동 원인 | 레이아웃으로 구분 |
|---------|-----------|-----------------|
| `ITEM_UNIT_PRICE` vs `ITEM_AMOUNT` | 둘 다 숫자 | 열(column) 위치 |
| `EXPORTER_ADDRESS` vs `IMPORTER_ADDRESS` | 유사한 주소 텍스트 | 페이지 상/하단 위치 |
| `PORT_LOADING` vs `PORT_DESTINATION` | 같은 형식 ("XXX PORT") | 레이블 행 위치 |

---

## 의존성

```
# ML
torch>=2.3.0
transformers>=4.44.0
datasets>=2.20.0
accelerate>=0.34.0
Pillow>=10.0.0

# 평가
seqeval>=1.2.2
scikit-learn>=1.5.0

# PDF 처리
pymupdf>=1.24.0    # 합성 PDF 텍스트 추출
easyocr>=1.7.1     # 실제 문서 OCR (별도 바이너리 설치 불필요)

# 유틸
pyyaml>=6.0.1
tqdm>=4.66.0
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.13.0
```

> **Windows**: `DataLoader`의 `num_workers`는 반드시 `0` — config에 이미 설정됨
