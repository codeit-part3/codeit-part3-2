"""
RFP 전처리 실행 스크립트

실행 (프로젝트 루트에서):
    uv run python ai/dh/run_preprocess.py

데이터:
    data/data_list.csv          메타데이터
    data/files/*.hwp, *.pdf  원본 문서 (하위 폴더가 있어도 됨)
산출물 (data/processed/):
    rfp_docs.parquet            문서 1행 (메타데이터 + 정제 텍스트 + 요구사항)
    rfp_chunks.jsonl            청크 1줄 (임베딩 -> 벡터DB 적재용)
    extraction_suspects.csv     추출 품질 의심 문서 목록

처리 순서:
    1. 메타데이터 정제        (metadata.py)
    2. 파일 목록 + 텍스트 추출 (extract.py)
    3. 요구사항 분리          (req_split.py)  <- 반드시 정제 '전' 원문에서
    4. 텍스트 정제            (text_clean.py)
    5. 메타데이터 + 추출 결과 병합
    6. 청킹                  (chunking.py)
    7. 저장
"""

import json
import re
import sys
from typing import Any  # 타입 힌트용

import numpy as np
import pandas as pd  # 타입 힌트(pd.DataFrame)용
from chunking import build_chunks, summarize_chunks
from config import FILES_DIR, META_PATH, OUT_DIR, SAVE_RAW_TEXT
from extract import add_quality, extract_all, list_files
from metadata import COLS, load_and_prepare_metadata
from req_split import split_all
from text_clean import clean_docs

# 윈도우 터미널(cp949)에서 한글/특수 기호를 print할 때 UnicodeEncodeError가 나지 않도록
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ################################################################
# 타입 힌트(type hint)
# ################################################################
# 함수가 받는 값과 돌려주는 값의 자료형을 적어 두는 표시
#   - 이름: 자료형          -> 파라미터의 자료형  예) meta: pd.DataFrame
#   - -> 자료형             -> 돌려주는 값의 자료형
#   - -> None              -> 돌려주는 값이 없음 (파일 저장, 전체 실행처럼 '일'만 하는 함수)
#   - Any                  -> 어떤 자료형이든 올 수 있음
#   - list[dict[str, Any]] -> '키는 글자, 값은 아무 자료형'인 딕셔너리들의 리스트 (청크 목록)


# ################################################################
# 병합 (노트북 6-2)
# ################################################################
def merge_meta_docs(meta: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    # meta  : load_and_prepare_metadata가 돌려준 메타데이터 표
    # docs  : 추출/요구사항 분리/정제가 끝난 문서 표
    # 반환값: 두 표를 파일명으로 합치고 doc_id 열을 맨 앞에 붙인 표 (_merge 열 포함)

    # ============================================================
    # 1단계: 메타데이터 복사본을 만들고 글자 열 정리
    # ============================================================
    # copy(): 원본 meta는 그대로 두고 복사본을 수정 (원본을 실수로 바꾸지 않도록)
    meta_clean = meta.copy()

    # COLS에 등록한 원래 CSV 열만 대상 (title_tags 같은 리스트 열은 건드리지 않음)
    for col_name in COLS.values():
        # dtype == object: 주로 글자(str)가 들어 있는 열 (숫자 열은 int64·float64)
        if meta_clean[col_name].dtype == object:
            # 앞뒤 공백 제거 -> 공백만 있던 칸은 빈 글자('')가 됨 -> 빈 글자를 빈 값(NaN)으로 통일
            meta_clean[col_name] = (
                meta_clean[col_name].str.strip().replace({"": np.nan})
            )

    # ============================================================
    # 2단계: 메타데이터 + 추출 결과 합치기
    # ============================================================
    # docs에서 용량이 크고 이후에 필요 없는 열은 빼고 합침
    #   pages(페이지별 원문) -> raw_text에 이미 합쳐져 있으므로 불필요
    #   raw_text(정제 전 원문) -> SAVE_RAW_TEXT=True(config.py)면 남김
    #     청크 크기를 바꿔 다시 청킹 할 때 요구사항 분리는 '정제 전 원문'에서 해야 함
    #     -> 저장해 두면 원본 HWP/PDF 파일을 다시 추출하지 않아도 됨 (대신 parquet 파일이 커짐)
    drop_cols = ["pages"] if SAVE_RAW_TEXT else ["pages", "raw_text"]
    docs_slim = docs.drop(columns=drop_cols)

    # how='left': 왼쪽(meta_clean)의 행은 모두 남기고, 맞는 파일이 없으면 오른쪽 칸은 빈 값
    # indicator=True: '_merge' 열을 추가해서 각 행이 어떻게 합쳐졌는지 표시
    #     both       -> 양쪽 모두 있음 (정상)
    #     left_only  -> 메타에만 있고 파일이 없음
    merged = meta_clean.merge(
        docs_slim,
        left_on="파일명_norm",
        right_on="파일명_실제",
        how="left",
        indicator=True,
    )
    print("병합 결과:", merged["_merge"].value_counts().to_dict())

    # 파일과 연결되지 않은 행은 청킹에서 에러가 발생하므로 미리 알림
    not_matched = merged["_merge"] != "both"
    if not_matched.any():
        print(
            "[경고] 파일을 찾지 못한 메타데이터 행:",
            merged.loc[not_matched, COLS["filename"]].tolist(),
        )

    # ============================================================
    # 3단계: 문서 ID 붙이기
    # ============================================================
    # reset_index(drop=True): 행 번호를 0, 1, 2, ...로 새로 매김 (기존 번호는 버림)
    merged = merged.reset_index(drop=True)

    # f'doc_{i:03d}': 숫자를 3자리로 맞추고 빈자리는 0으로 채움  예) 0 -> 'doc_000', 12 -> 'doc_012'
    doc_ids = []
    for i in range(len(merged)):
        doc_ids.append(f"doc_{i:03d}")

    # insert(위치, 열 이름, 값): 원하는 위치에 열 추가 (0 = 맨 앞)
    merged.insert(0, "doc_id", doc_ids)
    return merged


# ============================================================
# 최종 문서 표에 남길 열 목록 (노트북 '최종 문서 표 만들기')
# ============================================================
DOC_COLS = [
    # --- 식별 정보 ---
    "doc_id",  # 이 스크립트에서 붙인 문서 ID  예) 'doc_000'
    "notice_key",  # '공고번호-차수' 또는 'NO_파일명'
    "has_notice_no",  # 공고 번호가 있는지 (True/False)
    COLS["notice_no"],  # 공고 번호 (원본)
    COLS["notice_seq"],  # 공고 차수 (원본)
    # --- 사업명 ---
    COLS["title"],  # 원본 사업명
    "title_clean",  # 태그를 뗀 사업명
    "title_tags",  # 떼어 낸 태그 리스트  예) ['긴급', '재공고']
    # --- 발주 기관 ---
    COLS["agency"],  # 원본 기관명
    "agency_clean",  # 정리한 기관명
    "agency_type",  # 기관 유형  예) '대학', '지자체'
    # --- 금액·날짜 ---
    "amount_num",  # 사업 금액 (원, 숫자)
    "pub_date_dt",  # 공개일 (날짜 형식)
    "start_dt",  # 입찰 시작일
    "deadline_dt",  # 입찰 마감일
    "prep_days",  # 준비 기간 (공개 -> 마감, 일)
    "date_suspect",  # 날짜 의심 표시 (True/False)
    # --- 내용 ---
    COLS["summary"],  # 사업 요약 (원본)
    COLS["filename"],  # 파일명 (원본)
    "ext",  # 파일 확장자
    "clean_text",  # 정제한 본문 텍스트
    "reqs",  # 분리한 요구사항 리스트
    "split_mode",  # 요구사항 분리 방식 ('label', 'line', None)
]
if SAVE_RAW_TEXT:
    DOC_COLS.append("raw_text")  # 정제 전 원문 (재청킹용)


# ################################################################
# 저장 (노트북 8. 저장)
# ################################################################

# ================================================================
# 반쪽짜리 서로게이트 문자란?
# ================================================================
# - HWP 글자는 2바이트(UTF-16) 단위로 저장됨
# - 2바이트로 표현할 수 없는 드문 글자(일부 특수 기호, 이모지 등)는
#   '앞 반쪽 + 뒤 반쪽' 두 칸(서로게이트 쌍)으로 나눠 저장함
# - 수정 전 HWP 추출 함수(_decode_para_text)는 한 칸씩 chr()로 바꾸기 때문에
#   이런 글자가 '반쪽짜리 글자 2개'로 쪼개졌음
# - 반쪽짜리 글자는 UTF-8로 저장할 수 없어서 parquet/JSON 저장 때 에러가 남
# -> extract.py의 _decode_para_text에서 쌍을 합치도록 함
#   이 함수는 다른 경로로 섞여 들어올 경우를 대비한 '안전장치'로 남겨 둠
# -> 저장 전에 서로게이트 전용 영역(0xD800~0xDFFF)의 글자를 모두 삭제
SURR = re.compile(r"[\ud800-\udfff]")


# 어떤 자료형이 와도 안쪽 글자까지 모두 청소하는 함수
# chunks는 '리스트 안에 딕셔너리, 딕셔너리 안에 또 딕셔너리(metadata)/리스트(tags)'처럼 겹겹이 들어 있음
# -> 함수가 안쪽 값에 대해 자기 자신을 다시 호출(재귀)해서 끝까지 파고들어 감
def strip_surrogates(x: Any) -> Any:
    # x     : 글자, 리스트, 딕셔너리, 숫자 등 어떤 값이든 (그래서 Any)
    # 반환값: 같은 구조에서 글자 안의 서로게이트 문자만 지운 값 (자료형은 들어온 것과 같음)
    # isinstance(값, 자료형): 값이 그 자료형인지 확인

    # 글자면 -> 서로게이트 문자 삭제
    if isinstance(x, str):
        return SURR.sub("", x)

    # 리스트면 -> 안의 값 하나하나에 이 함수를 다시 적용
    if isinstance(x, list):
        result = []
        for v in x:
            result.append(strip_surrogates(v))
        return result

    # 딕셔너리면 -> 값(value) 하나하나에 이 함수를 다시 적용 (키는 그대로)
    if isinstance(x, dict):
        result = {}
        for k, v in x.items():
            result[k] = strip_surrogates(v)
        return result

    # 숫자, None, 날짜 등 -> 청소할 글자가 없으므로 그대로
    return x


def save_outputs(
    doc_df: pd.DataFrame, chunks: list[dict[str, Any]], docs: pd.DataFrame
) -> None:
    # doc_df: 최종 문서 표 (DOC_COLS 열) → rfp_docs.parquet
    # chunks: build_chunks가 돌려준 청크 리스트 → rfp_chunks.jsonl
    # docs  : 추출 결과 표 (suspect 열 사용) → extraction_suspects.csv
    # 반환값 없음 (파일로 저장만 함)
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # 저장 폴더가 없으면 만듦

    # 청크 목록과 문서 표 청소
    chunks = strip_surrogates(chunks)  # 리스트 전체를 한 번에 (재귀로 안쪽까지)
    for col_name in doc_df.columns:
        if doc_df[col_name].dtype == object:  # 글자/리스트 등이 든 열만
            doc_df[col_name] = doc_df[col_name].map(strip_surrogates)

    # (1) 문서 표 -> parquet
    #   parquet: 열 단위로 압축 저장하는 형식. CSV보다 작고 빠르며, 날짜/리스트 같은 자료형도 그대로 보존
    #   index=False: pandas의 행 번호는 저장하지 않음
    doc_df.to_parquet(OUT_DIR / "rfp_docs.parquet", index=False)

    # (2) 청크 목록 -> JSONL (JSON Lines: 한 줄에 JSON 하나씩)
    #   벡터 DB나 임베딩 단계에서 한 줄씩 읽어 들이기 편한 형식
    #   ensure_ascii=False: 한글을 '사업'처럼 바꾸지 않고 그대로 저장 (사람이 읽을 수 있게)
    with open(
        OUT_DIR / "rfp_chunks.jsonl", "w", encoding="utf-8"
    ) as f:  # 'w' = 쓰기 모드
        for c in chunks:
            line = json.dumps(c, ensure_ascii=False)  # 딕셔너리 -> JSON 글자
            f.write(line + "\n")

    # (3) 추출 품질 의심 문서 목록 -> CSV (OCR 등 후속 처리 대상)
    #   utf-8-sig: 엑셀로 열어도 한글이 깨지지 않는 UTF-8 (앞에 BOM 표시를 붙임)
    suspect_cols = ["파일명_실제", "ext", "method", "n_chars", "hangul_ratio", "error"]
    docs.loc[docs["suspect"], suspect_cols].to_csv(
        OUT_DIR / "extraction_suspects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 저장된 파일 목록과 크기 확인
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name:28s} {p.stat().st_size / 1024 / 1024:6.1f} MB")


# ################################################################
# 전체 실행
# ################################################################
def main() -> None:
    # 파라미터 없음 (경로·설정은 config.py에서 가져옴) / 반환값 없음 (전체 전처리를 차례로 실행)

    print("=" * 60, "\n1. 메타데이터 정제\n" + "=" * 60)
    meta = load_and_prepare_metadata(META_PATH)

    print("=" * 60, "\n2. 파일 목록 + 텍스트 추출\n" + "=" * 60)
    file_df = list_files(FILES_DIR)

    # 메타데이터 ↔ 실제 파일 매칭 확인 (노트북 3-5)
    real_files = set(file_df["파일명_실제"])  # 폴더에 실제로 있는 파일
    listed_files = set(meta["파일명_norm"])  # CSV에 적혀 있는 파일
    print(
        "메타에는 있는데 파일 없음:",
        len(listed_files - real_files),
        "| 파일은 있는데 메타에 없음:",
        len(real_files - listed_files),
    )

    docs = extract_all(file_df)
    docs = add_quality(docs)

    print("=" * 60, "\n3. 요구사항 분리 (정제 전 원문 기준)\n" + "=" * 60)
    docs = split_all(docs)

    print("=" * 60, "\n4. 텍스트 정제\n" + "=" * 60)
    docs = clean_docs(docs)

    print("=" * 60, "\n5. 병합\n" + "=" * 60)
    merged = merge_meta_docs(meta, docs)
    # merged[열 목록]: 지정한 열만, 지정한 순서대로 꺼냄 / .copy(): 독립된 새 표로 만듦
    doc_df = merged.loc[merged["_merge"] == "both", DOC_COLS].copy()
    print("doc_df:", doc_df.shape)

    print("=" * 60, "\n6. 청킹\n" + "=" * 60)
    # 파일명 -> 정제 전 원문 (요구사항 분리는 표 구조가 살아 있는 원문에서 해야 하므로)
    raw_by_file = dict(zip(docs["파일명_실제"], docs["raw_text"], strict=False))
    chunks = build_chunks(doc_df, raw_by_file)
    summarize_chunks(chunks)

    print("=" * 60, "\n7. 저장 ->", OUT_DIR, "\n" + "=" * 60)
    save_outputs(doc_df, chunks, docs)
    print("완료")


if __name__ == "__main__":
    main()
