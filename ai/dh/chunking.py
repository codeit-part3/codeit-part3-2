"""
청킹(Chunking): 정제한 문서를 검색용 청크로 자르고 조각마다 메타데이터를 붙임

노트북: 7. 청킹 (그래프 제외)

RecursiveCharacterTextSplitter: 문단(\n\n) → 줄(\n) → 문장 → 공백 순으로
가능한 한 의미 단위를 유지하면서 지정 길이 이하로 자름
  - chunk_size   : 청크 최대 글자 수
  - chunk_overlap: 앞뒤 청크가 겹치는 글자 수 (문장이 경계에서 잘려 의미가 끊기는 것 방지)
값은 이후 RAG 실험에서 조정할 하이퍼파라미터 → config.py에서 변경
"""

from typing import Any  # Any: 어떤 자료형이든 올 수 있다는 뜻 (타입 힌트용)

import pandas as pd
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    REQ_LONG_FLAG,
    REQ_MAX,
    REQ_MIN,
    REQ_OVERLAP,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from metadata import COLS, normalize_name
from req_split import compact_requirement, split_requirements
from text_clean import clean_text, remove_toc

# ################################################################
# 1단계: 텍스트 분할기(splitter) 만들기
# ################################################################
# RecursiveCharacterTextSplitter: 정해진 크기를 넘지 않도록 텍스트를 자르는 LangChain 도구
# separators: 자를 위치의 우선순위
#   빈 줄('\n\n')에서 먼저 자르고 -> 안 되면 줄바꿈 -> 문장 끝('. ') -> 공백 -> 글자 단위
#   -> 가능한 한 문단/문장이 통째로 남도록 자름
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

splitter = RecursiveCharacterTextSplitter(  # 일반 텍스트용
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
)
req_splitter = RecursiveCharacterTextSplitter(  # 긴 요구사항을 다시 자르는 용도
    chunk_size=REQ_MAX,
    chunk_overlap=REQ_OVERLAP,
    separators=SEPARATORS,
)


# ################################################################
# 2단계: 요구사항 접두어 -> 분류명 대응표
# ################################################################
# 문서마다 접두어 표기가 달라도 같은 분류로 통일 (표에 없는 접두어는 나중에 '기타')
CAT_MAP = {
    "SFR": "기능",
    "FUN": "기능",
    "FUR": "기능",
    "BFR": "기능",
    "PER": "성능",
    "SER": "보안",
    "SIR": "인터페이스",
    "INR": "인터페이스",
    "IR": "인터페이스",
    "INT": "인터페이스",
    "DAR": "데이터",
    "TER": "테스트",
    "QUR": "품질",
    "COR": "제약사항",
    "CDR": "제약사항",
    "ECR": "시스템장비",
    "PMR": "프로젝트관리",
    "PSR": "프로젝트지원",
    "CSR": "컨설팅",
    "CNR": "컨설팅",
    "MAR": "운영·유지관리",
    "MPR": "운영·유지관리",
    "SOR": "운영·유지관리",
    "MHR": "운영·유지관리",
}


# 요구사항이 아닌 '일반' 청크의 메타데이터 기본값
GEN_META_BASE = {
    "chunk_type": "general",
    "req_id": None,
    "req_cat": None,
    "req_cat_name": None,
    "req_name": None,
}


# ################################################################
# 3단계: 문서마다 청크 만들기
# ################################################################
def build_chunks(
    doc_df: pd.DataFrame, raw_by_file: dict[str, str]
) -> list[dict[str, Any]]:
    # doc_df     : 최종 문서 표 (run_preprocess.py의 DOC_COLS 열을 가진 표)
    # raw_by_file: 파일명 -> 정제 전 원문 (요구사항 분리는 표 구조가 살아 있는 원문에서 해야 하므로)
    # 반환값      : 청크 딕셔너리 리스트  예) [{'chunk_id': ..., 'doc_id': ..., 'text': ..., 'metadata': {...}}, ...]
    chunks = []

    for _, row in doc_df.iterrows():
        # ------------------------------------------------------------
        # 3-1. 모든 청크 앞에 붙일 머리글 + 문서 공통 메타데이터
        # ------------------------------------------------------------
        # 머리글: 청크만 따로 떼어 봐도 '어느 사업, 어느 기관' 문서인지 알 수 있게
        #   예) '[AI 플랫폼 구축 | 한국OO진흥원]'
        header = f"[{row['title_clean']} | {row['agency_clean']}]"

        # 금액/마감일은 빈 값(NaN)이면 None으로 (저장/검색 도구들이 NaN을 잘 처리하지 못함)
        amount = None if pd.isna(row["amount_num"]) else float(row["amount_num"])

        if pd.isna(row["deadline_dt"]):
            deadline = None
        else:
            deadline = row["deadline_dt"].strftime(
                "%Y-%m-%d %H:%M"
            )  # 날짜 -> '2024-03-15 18:00' 글자

        base_meta = {
            "doc_id": row["doc_id"],
            "notice_key": row["notice_key"],
            "file_name": row[COLS["filename"]],
            "title": row["title_clean"],
            "agency": row["agency_clean"],
            "agency_type": row["agency_type"],
            "tags": list(row["title_tags"]),
            "amount": amount,
            "deadline": deadline,
            "date_suspect": bool(row["date_suspect"]),
        }

        doc_chunks = []  # 이 문서의 (청크 텍스트, 청크별 메타데이터) 쌍을 모음

        # ------------------------------------------------------------
        # 3-2. 요구사항 청크 (요구사항 표를 분리할 수 있었던 HWP 문서만)
        # ------------------------------------------------------------
        if pd.notna(row["split_mode"]):  # 'label' 또는 'line'이면 (None/NaN이 아니면)
            # 원문에서 다시 분리: 요구사항 리스트 + 요구사항이 아닌 나머지 텍스트
            # ← 수정: CSV 파일명을 normalize_name(metadata.py에서 정의)으로 정리한 뒤 찾음
            #   raw_by_file의 키는 NFC로 정리한 '실제 파일명'이라,
            #   CSV 파일명이 자모 분리('ㅎ+ㅏ+ㄴ') 형태로 저장돼 있으면 같은 이름인데도 못 찾고 KeyError가 남
            file_key = normalize_name(str(row[COLS["filename"]]))
            reqs, rest, _ = split_requirements(raw_by_file[file_key])

            # 나머지 텍스트도 목차 제거 + 정제 (remove_toc는 (텍스트, 개수)를 주므로 [0]만)
            rest_text = clean_text(remove_toc(rest)[0])

            for q in reqs:
                body = clean_text(
                    compact_requirement(q)
                )  # '[ID] 이름 / 정의: ...' 형식으로 정리

                # 너무 짧으면(목록표의 한 줄 등) 요구사항 청크로 만들지 않고 일반 텍스트에 합침
                if len(body) < REQ_MIN:
                    rest_text += "\n" + body
                    continue

                q_meta = {
                    "chunk_type": "requirement",
                    "req_id": q["req_id"],
                    "req_cat": q["req_cat"],
                    "req_cat_name": CAT_MAP.get(q["req_cat"], "기타"),
                    "req_name": q["req_name"],
                    "req_long": len(q["text"]) > REQ_LONG_FLAG,  # 원문이 너무 길면 True
                }

                # 최대 길이 이하면 통째로 1개, 넘으면 다시 잘라 여러 조각
                if len(body) <= REQ_MAX:
                    pieces = [body]
                else:
                    pieces = req_splitter.split_text(body)

                for k, piece in enumerate(pieces):  # k: 조각 순번 (0 = 첫 조각)
                    # (a) 경계가 의심되는 긴 요구사항의 '뒤쪽' 조각
                    #     -> 다른 내용이 섞였을 수 있으므로 요구사항이 아닌 일반 청크로 분류
                    if k > 0 and q_meta["req_long"]:
                        tail_meta = {**GEN_META_BASE, "req_long": True}
                        doc_chunks.append((f"{header}\n{piece}", tail_meta))
                        continue

                    # (b) 정상 요구사항의 뒤쪽 조각 -> 어느 요구사항의 이어지는 부분인지 앞에 표시
                    if k > 0:
                        req_name = q["req_name"] or ""
                        piece = f"[{q['req_id']}] {req_name} (계속)\n{piece}"

                    # (c) 요구사항 청크로 저장
                    doc_chunks.append((f"{header}\n{piece}", q_meta))

        else:
            # 요구사항 표가 없는 문서: 정제한 본문 전체가 일반 텍스트
            rest_text = row["clean_text"]

        # ------------------------------------------------------------
        # 3-3. 일반 텍스트 청크 (사업 개요·입찰 안내 등 + 요구사항 표 없는 문서)
        # ------------------------------------------------------------
        # {**A, '키': 값}: 딕셔너리 A를 복사하고 키 하나를 추가/덮어씀
        gen_meta = {**GEN_META_BASE, "req_long": False}

        for piece in splitter.split_text(rest_text):
            doc_chunks.append((f"{header}\n{piece}", gen_meta))

        # ------------------------------------------------------------
        # 3-4. 청크 ID를 붙여 전체 목록에 추가
        # ------------------------------------------------------------
        for j, (text, chunk_meta) in enumerate(doc_chunks):
            chunks.append(
                {
                    "chunk_id": f"{row['doc_id']}_{j:04d}",  # 예) 'doc_012_0003'
                    "doc_id": row["doc_id"],
                    "text": text,
                    # 문서 공통 정보 + 청크별 정보 + 문서 안에서의 순번을 하나로 합침
                    # (같은 키가 있으면 뒤에 쓴 쪽이 이김)
                    "metadata": {**base_meta, **chunk_meta, "chunk_index": j},
                }
            )

    return chunks


# ################################################################
# 4단계: 결과 확인
# ################################################################
def summarize_chunks(chunks: list[dict[str, Any]]) -> pd.DataFrame:
    # chunks: build_chunks가 돌려준 청크 딕셔너리 리스트
    # 반환값  : 청크 1개 = 1행인 표 (len, chunk_type 열 추가)
    chunk_df = pd.DataFrame(chunks)
    chunk_df["len"] = chunk_df["text"].str.len()

    # metadata 열(딕셔너리)에서 특정 키 값만 꺼내는 함수
    #   key: 꺼낼 메타데이터 키  예) 'chunk_type' / 반환값: 청크마다 그 키의 값이 담긴 Series
    def get_meta(key: str) -> pd.Series:
        values = []
        for m in chunk_df["metadata"]:
            values.append(m[key])
        return pd.Series(values, index=chunk_df.index)

    chunk_df["chunk_type"] = get_meta("chunk_type")

    avg_per_doc = chunk_df.groupby("doc_id").size().mean()  # size(): 그룹별 행 수
    print(f"청크 수: {len(chunk_df)} | 문서당 평균 {avg_per_doc:.1f}개")

    # 청크 종류(요구사항/일반)별 길이 통계
    print(chunk_df.groupby("chunk_type")["len"].describe().round(0).to_string())

    # 요구사항 분류별 청크 수 (일반 청크는 None이라 자동으로 빠짐)
    print(
        get_meta("req_cat_name")
        .value_counts()
        .to_frame("요구사항 분류별 청크 수")
        .to_string()
    )

    print("긴 요구사항(req_long) 청크:", int(get_meta("req_long").sum()))
    return chunk_df
