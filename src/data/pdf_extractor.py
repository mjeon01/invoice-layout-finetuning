"""
PDF → 토큰 + 바운딩박스 추출기.

합성 PDF  : PyMuPDF 직접 텍스트 추출 (OCR 불필요, bbox 정확)
실제 PDF  : 페이지를 이미지로 렌더링 후 EasyOCR

출력 포맷 (샘플 dict):
  {
    "tokens":  List[str],
    "bboxes":  List[[x0, y0, x1, y1]],   # 0-1000 정규화 정수
    "image":   PIL.Image (RGB),
    "page_idx": int,
    "source":  "synth" | "real"
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz                      # PyMuPDF
from PIL import Image

RENDER_DPI = 150   # 실제 문서 렌더링 해상도


# ─────────────────────────────────────────────────────────────────────────────
# 좌표 정규화 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _norm_bbox(x0: float, y0: float, x1: float, y1: float,
               pw: float, ph: float) -> list[int]:
    """Point 좌표 → 0-1000 정수 (origin 좌상단)."""
    return [
        max(0, min(1000, int(x0 / pw * 1000))),
        max(0, min(1000, int(y0 / ph * 1000))),
        max(0, min(1000, int(x1 / pw * 1000))),
        max(0, min(1000, int(y1 / ph * 1000))),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 합성 PDF: PyMuPDF word 추출
# ─────────────────────────────────────────────────────────────────────────────

def extract_synth_page(
    pdf_path: str | Path,
    page_idx: int,
    min_word_len: int = 1,
) -> dict:
    """
    합성 PDF의 page_idx 페이지에서 단어 토큰과 bbox를 추출합니다.

    PyMuPDF의 get_text("words")는 단어별로 (x0,y0,x1,y1,word,...) 반환.
    좌표계: 좌상단 원점, 단위 pt. 이를 0-1000으로 정규화합니다.

    Args:
        pdf_path:     PDF 파일 경로
        page_idx:     0-indexed 페이지 번호
        min_word_len: 이 길이 미만의 단일 문자 토큰 필터 (0=전부 포함)

    Returns:
        dict(tokens, bboxes, image, page_idx, source)
    """
    doc  = fitz.open(str(pdf_path))
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height

    words = page.get_text("words")  # [(x0,y0,x1,y1,word,block,line,word_no), ...]

    tokens: list[str]      = []
    bboxes: list[list[int]] = []

    for w in words:
        x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
        word = word.strip()
        if not word:
            continue
        if len(word) < min_word_len:
            continue
        tokens.append(word)
        bboxes.append(_norm_bbox(x0, y0, x1, y1, pw, ph))

    # 이미지 렌더링 (LayoutLMv3 pixel_values용)
    mat   = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix   = page.get_pixmap(matrix=mat)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()
    return {
        "tokens":   tokens,
        "bboxes":   bboxes,
        "image":    image,
        "page_idx": page_idx,
        "source":   "synth",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 실제 PDF: 렌더 → OCR
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_reader = None  # 모듈 수준 캐시 (매 페이지마다 재초기화 방지)


def _ocr_with_easyocr(image: Image.Image) -> tuple[list[str], list[list[int]]]:
    """
    EasyOCR로 이미지에서 단어 토큰과 bbox 추출.

    EasyOCR 결과: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, conf), ...]
    bbox는 4점 폴리곤 → (min_x, min_y, max_x, max_y) 변환 후 0-1000 정규화.

    Returns (tokens, bboxes) — bboxes는 0-1000 정규화.
    """
    global _easyocr_reader
    try:
        import easyocr
    except ImportError:
        raise ImportError("easyocr 미설치: pip install easyocr")

    import numpy as np

    if _easyocr_reader is None:
        # gpu=False: CPU 전용 (GPU 있으면 True로 변경 가능)
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    img_w, img_h = image.size
    results = _easyocr_reader.readtext(np.array(image))

    tokens: list[str]       = []
    bboxes: list[list[int]] = []

    for (pts, text, conf) in results:
        text = text.strip()
        if not text or conf < 0.3:
            continue

        # 4점 폴리곤 → axis-aligned bbox
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        # 공백 기준으로 단어 분리 (EasyOCR는 문장 단위로 반환할 수 있음)
        words = text.split()
        if not words:
            continue

        word_w = (x1 - x0) / len(words)
        for i, word in enumerate(words):
            wx0 = x0 + i * word_w
            wx1 = wx0 + word_w
            tokens.append(word)
            bboxes.append([
                max(0, min(1000, int(wx0 / img_w * 1000))),
                max(0, min(1000, int(y0  / img_h * 1000))),
                max(0, min(1000, int(wx1 / img_w * 1000))),
                max(0, min(1000, int(y1  / img_h * 1000))),
            ])

    return tokens, bboxes


def extract_real_page(
    pdf_path: str | Path,
    page_idx: int,
    ocr_backend: str = "easyocr",
) -> dict:
    """
    실제 (이미지 기반) PDF 페이지에서 토큰과 bbox 추출.

    1. PyMuPDF로 페이지를 이미지로 렌더링
    2. OCR 실행 → 토큰 + bbox

    Args:
        pdf_path:    PDF 파일 경로
        page_idx:    0-indexed 페이지 번호
        ocr_backend: "tesseract" (현재 지원)

    Returns:
        dict(tokens, bboxes, image, page_idx, source)
    """
    doc  = fitz.open(str(pdf_path))
    page = doc[page_idx]

    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    if ocr_backend == "easyocr":
        tokens, bboxes = _ocr_with_easyocr(image)
    else:
        raise ValueError(f"지원하지 않는 OCR 백엔드: {ocr_backend}")

    return {
        "tokens":   tokens,
        "bboxes":   bboxes,
        "image":    image,
        "page_idx": page_idx,
        "source":   "real",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 전체 PDF 처리
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_pages(
    pdf_path: str | Path,
    source: str = "synth",
    page_indices: Optional[list[int]] = None,
    ocr_backend: str = "easyocr",
    verbose: bool = True,
) -> list[dict]:
    """
    PDF 전체(또는 지정 페이지)에서 샘플 추출.

    Args:
        pdf_path:     PDF 경로
        source:       "synth" | "real"
        page_indices: 처리할 페이지 목록 (None이면 전체)
        ocr_backend:  실제 문서 OCR 백엔드

    Returns:
        List[dict] — 각 dict는 extract_synth/real_page() 출력 포맷
    """
    doc = fitz.open(str(pdf_path))
    n_pages = doc.page_count
    doc.close()

    if page_indices is None:
        page_indices = list(range(n_pages))

    results = []
    fn = extract_synth_page if source == "synth" else extract_real_page

    try:
        from tqdm import tqdm
        it = tqdm(page_indices, desc=f"Extracting {Path(pdf_path).name}")
    except ImportError:
        it = page_indices

    for idx in it:
        try:
            if source == "synth":
                sample = extract_synth_page(pdf_path, idx)
            else:
                sample = extract_real_page(pdf_path, idx, ocr_backend=ocr_backend)
            results.append(sample)
        except Exception as e:
            if verbose:
                print(f"  page {idx} 오류: {e}")

    return results


def pdf_page_count(pdf_path: str | Path) -> int:
    doc = fitz.open(str(pdf_path))
    n   = doc.page_count
    doc.close()
    return n
