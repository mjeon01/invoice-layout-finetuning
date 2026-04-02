# Fishery Invoice Information Extraction — LayoutLMv3 Fine-Tuning

해외 수산물 무역 인보이스에서 구조화된 정보를 자동 추출하는 LayoutLMv3 파인튜닝 파이프라인입니다.

---

## 배경 및 문제

해외 수산물 무역 인보이스는 수출국·업체마다 레이아웃이 제각각입니다.  
동일한 필드(단가, 수량, 금액 등)라도 표의 위치나 헤더 표기 방식이 문서마다 다르기 때문에, 규칙 기반 자동화가 어렵습니다.

이를 해결하려고 LLM(GPT 등)을 활용한 자동화를 시도했으나 다음과 같은 문제가 반복적으로 발생했습니다:

| 문제 | 설명 |
|------|------|
| **수치 환각** | 단가·금액·중량 등 숫자 필드에서 실제 문서에 없는 값을 출력 |
| **품목 누락** | 라인 아이템이 여러 개일 때 일부 품목을 빠뜨리는 경우 발생 |
| **느린 처리** | API 호출 기반이라 대량 처리 시 비용·속도 문제 |

결국 LLM 출력을 사람이 다시 검토해야 해서 자동화의 이점이 반감되었습니다.

파인튜닝된 전용 모델을 쓰면 이 문제들을 해결할 수 있지만, **학습에 필요한 레이블 데이터를 구하기 어렵다**는 현실적인 장벽이 있습니다. 실제 인보이스는 거래처 정보·가격 등 민감한 정보를 담고 있어 대량 수집과 레이블링이 쉽지 않습니다.

**제안**: 실제 인보이스와 동일한 구조의 합성 PDF를 프로그래밍으로 생성해 학습 데이터를 확보하고, LayoutLMv3을 파인튜닝합니다.  
텍스트만 보는 모델(BERT)과 비교해 위치·이미지 정보가 실제로 얼마나 기여하는지도 함께 검증합니다.

---

## 접근 방법 요약

```
[학습]  합성 인보이스 PDF 800건  ──► 4개 모델 파인튜닝
[평가]  실제 인보이스 41건       ──► 일반화 성능 측정 및 비교
```

**4개 모델 비교 구성:**

| 모델 | 텍스트 | 위치(bbox) | 이미지 | 목적 |
|------|:------:|:----------:|:------:|------|
| BERT | ✅ | ❌ | ❌ | 텍스트만의 한계 측정 |
| LayoutLMv3 (bbox 제거) | ✅ | ❌ | ✅ | 이미지 단독 기여도 |
| LayoutLMv3 (image 제거) | ✅ | ✅ | ❌ | 위치 단독 기여도 |
| LayoutLMv3 전체 | ✅ | ✅ | ✅ | 모든 정보 활용 |

### 왜 LayoutLMv3인가?

일반 NLP 모델(BERT 등)은 텍스트만 봅니다.  
인보이스에서 `9.76`이 **단가**인지 **중량**인지는 **어디에 위치하는지**로 결정됩니다.  
LayoutLMv3는 텍스트 + 바운딩박스(위치) + 문서 이미지를 동시에 처리합니다.

```
입력:  토큰 ["9.76"]  +  위치 [504, 210, 545, 220]  +  이미지 패치
           ↑ 텍스트           ↑ 페이지 내 좌표              ↑ 시각 정보
출력:  "단가" 필드로 분류
```

---

## 데이터셋

### 합성 데이터 (학습용)
- **출처**: 절차적 생성(Procedural Generation) 방식으로 제작한 인보이스 PDF
- **규모**: 1,000건 (학습 800 / 검증 200)
- **특징**: 마스터 템플릿에 도메인 지식 기반 가상 데이터를 주입해 생성.
  PDF 텍스트 레이어에서 좌표를 직접 추출하므로 레이블링 오차가 없는 Pixel-perfect 학습셋 구성
- **파일**: `data/input/synth/`, `data/ground_truth/*_synth.json`

### 실제 데이터 (평가용)
- **출처**: 실제 해외 수산물 무역 인보이스 스캔본
- **규모**: 41건 — 최종 평가셋
- **한계 및 의의**: 수산물 무역 인보이스는 거래처 정보·가격 등 민감한 정보를 포함하고 있어 대량 수집에 제약이 있음.
  합성 데이터로만 학습한 모델이 한 번도 보지 못한 실제 레이아웃에서 얼마나 일반화되는지(Zero-shot generalization)를 검증하는 평가셋으로 활용됨.
- **특징**: 이미지 기반 PDF → EasyOCR로 토큰 추출 (별도 바이너리 설치 불필요)
- **파일**: `data/input/invoices_*.pdf`, `data/ground_truth/invoices_*_truth.json`

### 정답 데이터 포맷 (`ground_truth/*.json`)

각 인보이스마다 문서 메타 정보(`meta`)와 추출 대상 정답값(`y_true`)으로 구성됩니다.

```json
{
  "meta": { "invoice_number": "CHQ24PS01", "invoice_date": "2022-09-13" },
  "y_true": {
    "invoice_number":            "CHQ24PS01",          // 인보이스 번호
    "exporter.name":             "PT. INDO PACIFIC FISHERIES",  // 수출업체 명칭
    "importer.name":             "SEOUL FRESH IMPORTS CO., LTD", // 수입업체 명칭
    "line_items[0].description": "FROZEN ROUND SCAD",  // 첫 번째 품목 설명
    "line_items[0].quantity":    161,                   // 수량
    "line_items[0].unit_price":  9.76,                  // 단가
    "totals.final_total":        46145.93               // 최종 합계 금액
  }
}
```

---

## 추출 대상 필드 (22개)

| 카테고리 | 필드 (한국어) | 레이블 이름 |
|----------|--------------|-------------|
| 인보이스 | 인보이스 번호 | `INVOICE_NUMBER` |
| 인보이스 | 발행일 | `INVOICE_DATE` |
| 수출업체 | 명칭 | `EXPORTER_NAME` |
| 수출업체 | 주소 | `EXPORTER_ADDRESS` |
| 수입업체 | 명칭 | `IMPORTER_NAME` |
| 수입업체 | 주소 | `IMPORTER_ADDRESS` |
| 운송 정보 | 선적항 | `PORT_LOADING` |
| 운송 정보 | 도착항 | `PORT_DESTINATION` |
| 운송 정보 | 거래조건(Incoterms) | `INCOTERMS` |
| 품목 (라인별) | 품목 설명 | `ITEM_DESC` |
| 품목 (라인별) | 수량 | `ITEM_QTY` |
| 품목 (라인별) | 수량 단위 | `ITEM_QTY_UNIT` |
| 품목 (라인별) | 순중량 | `ITEM_NET_WEIGHT` |
| 품목 (라인별) | 중량 단위 | `ITEM_NET_WEIGHT_UNIT` |
| 품목 (라인별) | 단가 | `ITEM_UNIT_PRICE` |
| 품목 (라인별) | 통화 | `ITEM_CURRENCY` |
| 품목 (라인별) | 라인 금액 | `ITEM_AMOUNT` |
| 품목 (라인별) | 규격·등급 | `ITEM_SIZE_GRADE` |
| 합계 | 총 수량 | `TOTAL_QTY` |
| 합계 | 총 순중량 | `TOTAL_NET_WEIGHT` |
| 합계 | 총 금액 | `TOTAL_AMOUNT` |
| 합계 | 통화 | `TOTAL_CURRENCY` |

### BIO 레이블 체계

각 단어(토큰)에 BIO 태그를 붙여 엔티티 스팬을 표현합니다.

| 태그 | 의미 | 설명 |
|------|------|------|
| `B-{레이블}` | Begin | 엔티티의 **첫 번째** 단어 |
| `I-{레이블}` | Inside | 엔티티의 **이어지는** 단어 |
| `O` | Outside | 어떤 엔티티에도 속하지 않는 단어 |

예시 — 수출업체명(`EXPORTER_NAME`)과 단가(`ITEM_UNIT_PRICE`)가 같은 문서에 있을 때:

```
단어:    PT.              INDO             PACIFIC          FISHERIES        ...  9.76
태그:    B-EXPORTER_NAME  I-EXPORTER_NAME  I-EXPORTER_NAME  I-EXPORTER_NAME  ...  B-ITEM_UNIT_PRICE
```

총 **45개** 레이블: `O` × 1 + `B-{레이블}` × 22 + `I-{레이블}` × 22

---

## 방법론

### 연구 질문

> 인보이스 필드 추출에서 **위치 정보(bbox)** 와 **이미지** 는 각각 얼마나 기여하는가?

단순히 "LayoutLMv3가 BERT보다 좋다"를 보이는 게 아니라, 어떤 입력 정보가 성능 향상을 만드는지 분해해서 확인합니다.

---

### 실험 설계 — Ablation Study

같은 LayoutLMv3 구조에서 입력 모달리티를 하나씩 제거하며 비교합니다.

| 모델 | 텍스트 | 위치(bbox) | 이미지 | 측정 목적 |
|------|:------:|:----------:|:------:|-----------|
| BERT | ✅ | ❌ | ❌ | 텍스트만으로 가능한 성능 상한 |
| LayoutLMv3 (bbox 제거) | ✅ | ❌ | ✅ | 이미지 단독 기여도 |
| LayoutLMv3 (image 제거) | ✅ | ✅ | ❌ | 위치 정보 단독 기여도 |
| LayoutLMv3 전체 | ✅ | ✅ | ✅ | 두 정보를 함께 쓸 때의 효과 |

ablation은 모델 구조를 바꾸지 않고, 해당 입력을 0으로 채워 넣는 방식으로 구현했습니다.  
동일한 학습 절차로 훈련하므로 모달리티 이외의 변수는 통제됩니다.

---

### 학습 데이터 구성

실제 레이블된 인보이스 데이터를 확보하기 어렵기 때문에 합성 데이터로 학습합니다.  
합성 PDF는 실제 인보이스와 동일한 필드 구조를 갖도록 프로그래밍 방식으로 생성했습니다.

정답 파일에는 각 인보이스의 필드값만 있고 단어별 레이블이 없으므로, 단어 목록에서 정답값을 찾아 레이블을 자동으로 생성합니다.

```
정답 파일:  단가 = 9.76,  수출업체 = "PT. INDO PACIFIC FISHERIES"

페이지 단어 목록: ["PT.", "INDO", "PACIFIC", "FISHERIES", ..., "9.76", ...]

자동 레이블 부여:
  단어:   "PT."    "INDO"   "PACIFIC"  "FISHERIES"  ...  "9.76"
  레이블:  수출업체  수출업체  수출업체    수출업체      ...   단가
```

숫자 표기 차이(`9.76` / `9,76` / `9.760`)와 단어 겹침 방지 처리가 포함되어 있습니다.

---

### 서브토큰 정렬

LayoutLMv3는 내부에서 단어를 여러 조각(서브토큰)으로 쪼개 처리합니다(`FISHERIES` → `FISH` + `##ERIES`).  
레이블은 단어 단위이므로 첫 번째 조각에만 레이블을 부여하고, 나머지는 손실 계산에서 제외(`-100`)합니다.  
추론 시에도 첫 번째 조각의 예측값을 해당 단어의 결과로 사용합니다.

```
원래:  [ FISHERIES / B-EXPORTER_NAME ]
쪼갠 후: [ FISH / B-EXPORTER_NAME (학습함) ]  [ ##ERIES / -100 (무시) ]
```

**왜 첫 번째 조각에만 레이블을 주나요?**

인보이스 번호(`CHQ24PS01`) 같은 복잡한 단어는 서브토큰이 5~6개 이상 나올 수 있습니다.  
모든 조각에 동일한 레이블을 붙이면, 모델 입장에서는 같은 위치(bbox)에 정답이 여러 개 존재하는 것처럼 보여 학습이 불안정해집니다.  
대표 조각(첫 번째)에만 레이블을 집중시키면 **단어 하나 = 정답 하나**로 일관되어 학습 효율이 높아집니다.

---

## 파이프라인

```
  합성 인보이스 PDF                  실제 인보이스 스캔본
  (코드로 생성 → 텍스트 레이어 있음)  (스캐너 촬영 → 이미지만 있음)
  + 정답 파일 (800건)                + 정답 파일 (41건)
         │                                  │
         ▼ 텍스트 레이어 직접 읽기            ▼ EasyOCR로 텍스트 인식
    단어 + 위치 좌표 (오차 없음)         단어 + 위치 좌표 (인식 오류 가능)
         │                                  │
         └─────────────────┬────────────────┘
                           ▼
              [정답값 → 레이블 자동 부여]

              정답 파일:  단가 = 9.76
                         수출업체 = "PT. INDO PACIFIC FISHERIES"

              단어 목록에서 값을 찾아 레이블 부여:
                "PT."→수출업체  "INDO"→수출업체  "PACIFIC"→수출업체
                "FISHERIES"→수출업체  ...  "9.76"→단가

                           ▼
                  학습 데이터셋 구성
                           │
           ┌───────────────┼──────────────────┐
           ▼               ▼                  ▼               ▼
        BERT        LMv3(bbox 제거)    LMv3(image 제거)   LayoutLMv3
      (텍스트만)    (텍스트+이미지)     (텍스트+위치)      (텍스트+위치+이미지)
           └───────────────┼──────────────────┘
                           ▼
                실제 인보이스 41건 평가
                ├─ 엔티티 F1  : 스팬 경계 + 타입이 모두 맞아야 정답
                └─ KIE 정확도 : 추출한 필드값이 정답값과 일치하는 비율
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
│   ├── 04_train_layoutlmv3.py          # LayoutLMv3 전체 모델 학습
│   ├── 04_train_bert_baseline.py       # BERT 베이스라인 학습
│   ├── 04_train_layoutlmv3_ablation.py # Ablation 학습 (no_bbox / no_image)
│   └── 05_evaluate.py                  # 4개 모델 평가 + 비교 리포트
│
├── configs/
│   ├── layoutlmv3.yaml              # LayoutLMv3 전체 모델
│   ├── layoutlmv3_no_bbox.yaml      # Ablation — bbox 제거
│   ├── layoutlmv3_no_image.yaml     # Ablation — image 제거
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

### Step 4 — 모델 학습 (4개)

```bash
# LayoutLMv3 전체 (~2-4시간, GPU 권장)
python scripts/04_train_layoutlmv3.py \
    --config configs/layoutlmv3.yaml \
    --train-data data/processed/final/train \
    --val-data data/processed/final/val_synth \
    --output-dir models/layoutlmv3-invoice

# BERT 베이스라인 (~30분)
python scripts/04_train_bert_baseline.py \
    --config configs/bert_baseline.yaml \
    --train-data data/processed/final/train \
    --val-data data/processed/final/val_synth \
    --output-dir models/bert-baseline-invoice

# Ablation — bbox 제거 (텍스트 + 이미지만, ~2-4시간)
python scripts/04_train_layoutlmv3_ablation.py \
    --config configs/layoutlmv3_no_bbox.yaml \
    --train-data data/processed/final/train \
    --val-data data/processed/final/val_synth

# Ablation — image 제거 (텍스트 + 위치만, ~2-4시간)
python scripts/04_train_layoutlmv3_ablation.py \
    --config configs/layoutlmv3_no_image.yaml \
    --train-data data/processed/final/train \
    --val-data data/processed/final/val_synth
```

### Step 5 — 실제 문서 평가

```bash
python scripts/05_evaluate.py \
    --layoutlmv3-model    models/layoutlmv3-invoice \
    --bert-model          models/bert-baseline-invoice \
    --lmv3-no-bbox-model  models/layoutlmv3-no-bbox \
    --lmv3-no-image-model models/layoutlmv3-no-image \
    --test-data           data/processed/final/test_real \
    --output-dir          results
```

출력:
- `results/layoutlmv3_eval.json` — 필드별 Precision/Recall/F1
- `results/bert_baseline_eval.json`
- `results/layoutlmv3_no_bbox_eval.json`
- `results/layoutlmv3_no_image_eval.json`
- `results/comparison_report.md` — 4개 모델 비교표

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

두 가지 관점에서 평가합니다.

### 1. 엔티티 F1 (seqeval)

스팬의 종류(타입)와 범위(경계)가 모두 정확히 일치해야 정답으로 인정됩니다.  
토큰 하나라도 범위가 다르면 오답 처리됩니다.

```
예시:
  정답:   [단가↑] [기타]  [금액↑]
  예측:   [단가↑] [기타]  [단가↑]  ← 타입 오류 → 오탐(FP)
```

### 2. KIE 정확도 (Key Information Extraction)

모델이 예측한 스팬에서 실제 텍스트 값을 꺼내, 정답값과 일치하는지 비교합니다.  
"토큰 경계를 정확히 맞췄는가"가 아니라 "올바른 값을 뽑아냈는가"를 측정합니다.

```
예시:
  정답 파일:  단가 = "9.76"
  모델 예측 스팬에서 추출한 값: "9.76"  → 일치 ✅

  정답 파일:  수출업체 = "PT. INDO PACIFIC FISHERIES"
  모델 예측 스팬에서 추출한 값: "INDO PACIFIC"  → 불일치 ❌
```

| 지표 | 측정 내용 | 주요 용도 |
|------|-----------|-----------|
| 엔티티 F1 | 스팬 경계 + 타입 정확도 | 모델 성능 비교 |
| KIE 정확도 | 추출 값의 실제 일치 여부 | 실용적 활용 가능성 |

레이아웃 정보 효과가 기대되는 필드:

| 혼동되기 쉬운 필드 쌍 | 혼동 원인 | 레이아웃으로 구분하는 방법 |
|-----------------------|-----------|--------------------------|
| 단가 vs 라인 금액 | 둘 다 숫자값 | 표의 열(column) 위치 |
| 수출업체 주소 vs 수입업체 주소 | 유사한 주소 텍스트 | 페이지 상단/하단 위치 |
| 선적항 vs 도착항 | 같은 형식 ("XXX PORT") | 헤더 라벨 행 기준 위치 |

---

## 실험 결과 (학습 후 업데이트 예정)

| 모델 | Entity F1 | KIE F1 | 비고 |
|------|:---------:|:------:|------|
| BERT | — | — | 텍스트만 |
| LMv3 (bbox 제거) | — | — | 텍스트 + 이미지 |
| LMv3 (image 제거) | — | — | 텍스트 + 위치 |
| LayoutLMv3 전체 | — | — | 텍스트 + 위치 + 이미지 |

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
