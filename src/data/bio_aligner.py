"""
GT 필드값 → BIO 토큰 레이블 자동 정렬.

GT의 y_true (flat key-value dict)에서 각 필드값을 OCR 토큰 시퀀스에서 찾아
BIO 레이블 id 시퀀스를 생성합니다.

핵심 알고리즘:
  1. 필드값을 공백 분리해 서브시퀀스(sub_tokens) 생성
  2. 토큰 리스트를 슬라이딩 윈도우로 스캔
  3. 정규화된 문자열 비교로 매칭 (대소문자·구두점 무시)
  4. 매칭된 위치에 BIO 레이블 부여 (먼저 매칭된 것 우선, 중복 없음)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .label_schema import GT_KEY_TO_ENTITY, bio_ids, gt_key_to_entity, label2id


# ─────────────────────────────────────────────────────────────────────────────
# 문자열 정규화
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """
    비교를 위한 문자열 정규화:
    - 유니코드 NFC 정규화
    - 소문자 변환
    - 앞뒤 공백 제거
    - 연속 공백 → 단일 공백
    """
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize_value(value: str) -> list[str]:
    """GT 필드값을 공백 기준으로 토큰 분리."""
    return [t for t in value.split() if t]


# ─────────────────────────────────────────────────────────────────────────────
# 숫자 정규화 (6.75 / 6,75 / 6.750 등 매칭)
# ─────────────────────────────────────────────────────────────────────────────

def _num_variants(value: str) -> list[str]:
    """
    숫자 값의 다양한 표현 형태 반환 (OCR 오류·포맷 차이 대응).
    예: 9.76 → ["9.76", "9,76", "9.760"]
    """
    variants = [str(value)]
    s = str(value)
    # 소수점 형태
    if "." in s:
        variants.append(s.replace(".", ","))
        # 정수 형태 (소수점 이하 0)
        try:
            f = float(s)
            if f == int(f):
                variants.append(str(int(f)))
        except ValueError:
            pass
    return variants


# ─────────────────────────────────────────────────────────────────────────────
# 서브시퀀스 매칭
# ─────────────────────────────────────────────────────────────────────────────

def _find_subsequence(
    tokens: list[str],
    sub_tokens: list[str],
    used: set[int],
    start: int = 0,
) -> Optional[tuple[int, int]]:
    """
    tokens[start:] 에서 sub_tokens와 일치하는 연속 구간을 탐색.

    Args:
        tokens:     전체 토큰 리스트
        sub_tokens: 찾을 서브토큰 리스트 (정규화됨)
        used:       이미 레이블이 할당된 토큰 인덱스 집합
        start:      탐색 시작 위치

    Returns:
        (start_idx, end_idx) exclusive, 또는 None
    """
    n_sub = len(sub_tokens)
    if n_sub == 0:
        return None

    norm_sub = [_normalize(t) for t in sub_tokens]

    for i in range(start, len(tokens) - n_sub + 1):
        # 이미 사용된 위치 건너뜀
        if any(i + j in used for j in range(n_sub)):
            continue
        norm_window = [_normalize(tokens[i + j]) for j in range(n_sub)]
        if norm_window == norm_sub:
            return (i, i + n_sub)
    return None


def _find_single_numeric(
    tokens: list[str],
    value,
    used: set[int],
    start: int = 0,
) -> Optional[tuple[int, int]]:
    """
    숫자 단일 토큰 매칭 (다양한 표현 시도).
    """
    variants = _num_variants(str(value))
    for i in range(start, len(tokens)):
        if i in used:
            continue
        t = _normalize(tokens[i])
        if any(t == _normalize(v) for v in variants):
            return (i, i + 1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 메인 정렬 함수
# ─────────────────────────────────────────────────────────────────────────────

def align_gt_to_tokens(
    tokens: list[str],
    y_true: dict,
) -> list[int]:
    """
    y_true 필드값들을 tokens에서 찾아 BIO 레이블 id 시퀀스를 반환합니다.

    처리 순서:
    1. 긴 필드값 먼저 처리 (단일 토큰 값이 더 긴 값의 일부와 충돌하는 것 방지)
    2. 각 필드값을 정규화 서브시퀀스로 분리 후 슬라이딩 윈도우 매칭
    3. 매칭된 위치에 BIO 레이블 부여, used 집합으로 중복 방지

    Args:
        tokens: 페이지에서 추출된 단어 토큰 리스트
        y_true: GT 딕셔너리 (flat key-value)

    Returns:
        len(tokens) 길이의 ner_tag id 리스트
    """
    n = len(tokens)
    ner_tags = [label2id["O"]] * n
    used: set[int] = set()

    # entity_type → [(value, is_numeric)] 리스트 구성
    entity_spans: list[tuple[str, str, bool]] = []
    for key, val in y_true.items():
        entity = gt_key_to_entity(key)
        if entity is None or val is None:
            continue
        is_numeric = isinstance(val, (int, float))
        entity_spans.append((entity, str(val), is_numeric))

    # 긴 값 먼저 (서브시퀀스 토큰 수 기준 내림차순)
    entity_spans.sort(key=lambda x: -len(x[1].split()))

    for entity, value_str, is_numeric in entity_spans:
        sub_tokens = _tokenize_value(value_str)
        if not sub_tokens:
            continue

        # 단일 숫자 토큰은 변형 매칭 사용
        if is_numeric and len(sub_tokens) == 1:
            match = _find_single_numeric(tokens, value_str, used)
        else:
            match = _find_subsequence(tokens, sub_tokens, used)

        if match is None:
            continue

        start_idx, end_idx = match
        ids = bio_ids(entity, end_idx - start_idx)
        for j, tag_id in enumerate(ids):
            ner_tags[start_idx + j] = tag_id
        used.update(range(start_idx, end_idx))

    return ner_tags


# ─────────────────────────────────────────────────────────────────────────────
# GT JSON → invoice_number 인덱스
# ─────────────────────────────────────────────────────────────────────────────

def build_invoice_index(gt_records: list[dict]) -> dict[str, int]:
    """
    invoice_number → GT record index 역방향 인덱스 생성.
    PDF 페이지와 GT 레코드를 invoice_number로 연결할 때 사용합니다.
    """
    index: dict[str, int] = {}
    for i, rec in enumerate(gt_records):
        inv_no = rec.get("y_true", {}).get("invoice_number") or \
                 rec.get("meta", {}).get("invoice_number")
        if inv_no:
            index[str(inv_no).strip()] = i
    return index


def find_invoice_number_in_tokens(tokens: list[str]) -> Optional[str]:
    """
    토큰 리스트에서 인보이스 번호로 추정되는 값을 찾습니다.
    - 형태: 영숫자 + 특수문자(/ - _)를 포함하는 5자 이상 토큰
    - 나중에 GT index에서 매핑
    """
    pattern = re.compile(r"^[A-Z0-9][A-Z0-9/_\-\.]{3,}$", re.IGNORECASE)
    candidates = [t for t in tokens if pattern.match(t)]
    return candidates[0] if candidates else None
