"""
메타데이터(data_list.csv) 읽기/정제

노트북 대응 셀:
  - 3. 메타데이터 EDA          : CSV 읽기, COLS 지정
  - 3-1. 공고번호/태그/기관      : notice_key, title_clean/title_tags, agency_clean
  - 3-2. 사업 금액              : amount_num
  - 3-3. 발주 기관              : agency_type
  - 3-4. 날짜                   : *_dt, bid_days, prep_days, date_suspect
  - 3-5. 파일 매칭              : 파일명_norm
"""

import re
import unicodedata
from pathlib import Path
from typing import Any  # 타입 힌트용

import numpy as np
import pandas as pd
from config import MIN_VALID_AMOUNT, SUSPECT_PREP_DAYS

# ################################################################
# 타입 힌트(type hint)
# ################################################################
# 함수가 받는 값과 돌려주는 값의 자료형을 적어 두는 표시
#   - 이름: 자료형          -> 파라미터의 자료형  예) name: str
#   - -> 자료형             -> 돌려주는 값의 자료형
#   - -> None              -> 돌려주는 값이 없음 (검사만 하고 문제가 있으면 에러를 일으키는 함수)
#   - tuple[A, B]          -> 값 2개를 묶은 튜플 (순서대로 A, B 자료형)
#   - list[str]            -> 글자들의 리스트
#   - Any                  -> 어떤 자료형이든 올 수 있음 (표의 칸 값은 글자/숫자/NaN 등 여러 가지라서)
#   - Union[str, Path]     -> 글자(str) 또는 Path 둘 중 하나
#
# 파일 경로는 글자('data/data_list.csv')로도, Path 객체로도 받을 수 있으므로 별칭(alias)을 하나 만들어 씀
PathLike = str | Path  # PathLike = Union[str, Path]

# ################################################################
# 1. CSV 읽기
# ################################################################

# ============================================================
# 배경: CSV 파일의 한글이 깨지는 이유
# ============================================================
# 같은 CSV라도 어떤 프로그램으로 저장했느냐에 따라 인코딩(글자 -> 바이트 규칙)이 다름
#   - 'utf-8-sig' : 국제 표준 UTF-8. 엑셀에서 'CSV UTF-8'로 저장하면 이 형식
#                   ('-sig'를 붙이면 파일 맨 앞의 보이지 않는 표시(BOM)를 자동으로 제거,
#                    BOM이 없는 일반 UTF-8 파일도 문제없이 읽음)
#   - 'cp949'     : 한국어 윈도우 방식. 엑셀에서 그냥 'CSV'로 저장하면 보통 이 형식
#   - 'euc-kr'    : cp949보다 오래된 한국어 방식
# -> 인코딩을 모르니, 하나씩 시도해 보고 에러 없이 읽히는 것을 사용


def read_csv_any(path: PathLike) -> tuple[pd.DataFrame, str]:
    # path  : CSV 파일 경로
    # 반환값: (읽은 표, 성공한 인코딩 이름)
    candidates = ["utf-8-sig", "cp949", "euc-kr"]  # 시도할 순서

    for enc in candidates:
        try:
            df = pd.read_csv(path, encoding=enc)
            return (
                df,
                enc,
            )  # 성공하면 (표, 사용한 인코딩) 두 값을 함께 돌려주고 함수 종료
        except UnicodeDecodeError:
            # 이 인코딩으로 읽다가 글자 해석 에러가 나면 -> 다음 후보로
            continue

    # 반복문이 끝까지 돌았다 = 모든 후보가 실패함
    raise ValueError("인코딩 판별 실패")


# ============================================================
# 영어 -> 실제 한글 열 이름 대응표
# ============================================================
# 딕셔너리(dict): '키(key): 값(value)' 쌍을 모아 둔 자료형
#   왼쪽(키)   : 코드에서 부를 짧은 영어 별명
#   오른쪽(값) : CSV 파일에 실제로 적힌 한글 열 이름
# 예) COLS['title'] -> '사업명'
COLS = {
    "notice_no": "공고 번호",
    "notice_seq": "공고 차수",
    "title": "사업명",
    "amount": "사업 금액",
    "agency": "발주 기관",
    "pub_date": "공개 일자",
    "start": "입찰 참여 시작일",
    "deadline": "입찰 참여 마감일",
    "summary": "사업 요약",
    "filetype": "파일형식",
    "filename": "파일명",
    "text": "텍스트",
}


def check_columns(meta: pd.DataFrame) -> None:
    # meta  : 메타데이터 표 / 반환값 없음 (빠진 열이 있으면 KeyError를 일으킴)
    # 위 한글 열 이름이 CSV에 실제로 모두 있는지 검사
    missing = []  # CSV에 없는 열 이름을 모을 리스트

    # COLS.values(): 딕셔너리의 '값'(한글 열 이름)만 하나씩 꺼냄
    for col_name in COLS.values():
        if col_name not in meta.columns:  # 실제 CSV 열 목록에 없으면
            missing.append(col_name)

    # 빠진 열이 하나라도 있으면 즉시 멈추고 어떤 열이 없는지 알려줌
    if missing:
        raise KeyError(f"없는 컬럼: {missing}")


# ################################################################
# 2. 문서 키 / 사업명 태그 / 발주기관 정리 (노트북 3-1)
# ################################################################

# 태그 대응표: '괄호 안에서 찾은 단어' -> '통일된 태그 이름'
# 괄호 안 단어가 '전부' 이 표에 있을 때만 태그로 떼어냄
#   예) '(재공고)' -> 태그로 분리 / '(ERP)' -> 표에 없으므로 사업명의 일부로 유지
TAG_MAP = {
    "긴급": "긴급",
    "재공고": "재공고",
    "사전공개": "사전공개",
    "입찰공고": "입찰공고",
    "국제": "국제",
    "지문": "지문",
    "변경공고": "변경공고",
    "장기계속계약": "장기계속계약",
    "협상": "협상",
    "협상형": "협상",  # 표기가 달라도
    "협상에의한계약": "협상",  # 같은 '협상' 태그로 통일
    "혁신-국고": "혁신-국고",
}

# 괄호 표현을 찾는 정규식
#   [\[\(]            -> 여는 괄호 '[' 또는 '('
#   (  ...  )         -> 괄호 '안쪽 내용'을 따로 꺼낼 수 있게 묶음 (그룹(group) 1번)
#   [^\[\]\(\)]{1,20} -> 괄호 기호가 아닌 글자 1~20개
#   [\]\)]            -> 닫는 괄호 ']' 또는 ')'
BRACKET_RE = r"[\[\(]([^\[\]\(\)]{1,20})[\]\)]"

# 규칙으로 정리되지 않는 이름은 여기에 직접 추가
# 예: {'BioIN': '한국보건산업진흥원'} <- 실제 기관명을 확인한 뒤 채우기
AGENCY_MAP = {}


def add_notice_key(meta: pd.DataFrame) -> pd.DataFrame:
    # meta  : 메타데이터 표 / 반환값: has_notice_no, notice_key 열을 추가한 표
    # ============================================================
    # 문서마다 고유한 키(notice_key) 만들기
    # ============================================================
    # 규칙: 공고 번호가 있으면 -> '공고번호-차수'   예) '20240101234-1'
    #       공고 번호가 없으면 -> 'NO_파일명'       예) 'NO_사업계획서.hwp'

    # --- 1-1. 공고 차수를 글자로 정리 ---
    # to_numeric(errors='coerce'): 숫자로 바꾸고, 바꿀 수 없는 값은 빈 칸(NaN)으로
    # astype('Int64'): 1.0 -> 1 처럼 소수점 없는 정수로 (대문자 I: 빈 칸을 허용하는 정수형)
    seq_num = pd.to_numeric(meta[COLS["notice_seq"]], errors="coerce").astype("Int64")
    seq_str = seq_num.astype(str)  # 키에 이어 붙이기 위해 글자로 변환

    # --- 1-2. 공고 번호가 있는지 표시 (새 열 추가) ---
    meta["has_notice_no"] = meta[COLS["notice_no"]].notna()  # 있으면 True

    # --- 1-3. 공고 번호를 글자로 정리 ---
    # 열에 빈 칸이 하나라도 있으면 pandas가 숫자를 소수(float)로 읽어서 '20240101234.0'이 됨
    # -> 끝의 '.0'을 정규식으로 제거  ($ = 글자의 맨 끝)
    num_str = meta[COLS["notice_no"]].astype(str).str.strip()  # strip(): 앞뒤 공백 제거
    num_str = num_str.str.replace(r"\.0$", "", regex=True)

    # --- 1-4. 두 가지 키 후보를 모두 만들어 둠 ---
    key_with_num = num_str + "-" + seq_str  # 공고 번호가 있을 때 쓸 키
    key_without_num = "NO_" + meta[COLS["filename"]].astype(
        str
    )  # 공고 번호가 없을 때 쓸 키

    # --- 1-5. 행마다 조건에 맞는 키 선택 ---
    # np.where(조건, A, B): 조건이 True인 행은 A, False인 행은 B
    meta["notice_key"] = np.where(meta["has_notice_no"], key_with_num, key_without_num)

    # --- 1-6. 키가 정말 겹치지 않는지 검사 ---
    if not meta["notice_key"].is_unique:
        raise ValueError("notice_key 중복 존재")

    return meta


# ============================================================
# 사업명에서 태그를 떼어내는 함수
# ============================================================
# 입력: '[긴급] AI 플랫폼 구축 (재공고, 협상)'
# 출력: ('AI 플랫폼 구축', ['긴급', '재공고', '협상']) <- (정리된 사업명, 태그 리스트)


def split_tags(title: Any) -> tuple[str, list[str]]:
    # title : 사업명 칸 값 (보통 글자지만 빈 칸이면 NaN이라 Any)
    # 반환값: (태그를 뗀 사업명, 태그 리스트)
    # 사업명이 비어 있으면 빈 값 반환
    if pd.isna(title):
        return "", []

    title = str(title)
    found_tags = []  # 이 사업명에서 찾은 태그를 모을 리스트

    # re.sub에 넘길 함수: 괄호 표현을 하나 찾을 때마다 호출되어 '무엇으로 바꿀지'를 돌려줌
    #   match.group(0) -> 찾은 전체  예) '(재공고, 협상)'
    #   match.group(1) -> 괄호 안쪽  예) '재공고, 협상'
    def replace_bracket(match: re.Match) -> str:
        # match : 정규식이 찾은 괄호 표현 하나 (re.Match = 찾은 위치·내용이 담긴 객체)
        # 반환값: 그 자리를 바꿀 글자 (태그면 공백, 아니면 원래 글자)
        inside = match.group(1)

        # 쉼표로 나누고 공백 제거 -> ['재공고', '협상']  (한 괄호에 태그가 여러 개인 경우 대응)
        parts = []
        for p in inside.split(","):
            parts.append(p.replace(" ", ""))

        # 괄호 안 단어가 '전부' TAG_MAP에 있는지 확인
        all_known = True
        for p in parts:
            if p not in TAG_MAP:
                all_known = False
                break

        if all_known:
            # 전부 태그 -> 통일된 이름으로 저장하고, 사업명에서는 공백으로 지움
            for p in parts:
                found_tags.append(TAG_MAP[p])
            return " "
        else:
            # 하나라도 모르는 단어 -> 태그가 아니므로 원래 글자 그대로 둠
            return match.group(0)

    # re.sub(패턴, 함수, 글자): 패턴에 맞는 부분마다 함수를 호출해 그 결과로 바꿈
    clean = re.sub(BRACKET_RE, replace_bracket, title)

    # 태그를 지운 자리에 생긴 연속 공백을 하나로 줄이고(\s+ -> ' '), 앞뒤 공백 제거
    clean = re.sub(r"\s+", " ", clean).strip()

    # set(): 중복 제거 (예: '협상형'과 '협상'이 둘 다 '협상'이 된 경우)
    # sorted(): 가나다순 정렬 -> 결과가 항상 같은 순서로 나옴
    unique_tags = sorted(set(found_tags))

    return clean, unique_tags


def add_title_tags(meta: pd.DataFrame) -> pd.DataFrame:
    # meta  : 메타데이터 표 / 반환값: title_clean, title_tags 열을 추가한 표
    # 모든 사업명에 함수 적용
    clean_titles = []
    tag_lists = []

    for title in meta[COLS["title"]]:
        clean, tags = split_tags(title)  # 함수가 돌려준 두 값을 나눠 받음
        clean_titles.append(clean)
        tag_lists.append(tags)

    meta["title_clean"] = clean_titles  # 태그를 뗀 사업명
    meta["title_tags"] = tag_lists  # 태그 리스트
    return meta


def add_agency_clean(meta: pd.DataFrame) -> pd.DataFrame:
    # meta  : 메타데이터 표 / 반환값: agency_clean 열을 추가한 표
    # ============================================================
    # 발주 기관명 정리
    # ============================================================
    agency = meta[COLS["agency"]].astype(str).str.strip()

    # 끝에 붙은 '(용역)', '(공사)', '(물품)' 제거
    #   \s*         -> 앞의 공백 0개 이상
    #   (A|B|C)     -> A 또는 B 또는 C
    agency = agency.str.replace(r"\s*\((용역|공사|물품)\)", "", regex=True)

    # 맨 끝에 붙은 '입찰공고', '전자조달' 제거  ($ = 맨 끝)
    agency = agency.str.replace(r"\s*(입찰공고|전자조달)$", "", regex=True)

    agency = agency.str.strip()

    # replace(대응표): 대응표에 있는 이름은 지정한 이름으로 바꿈
    agency = agency.replace(AGENCY_MAP)

    meta["agency_clean"] = agency
    return meta


# ################################################################
# 3. 사업 금액 (노트북 3-2)
# ################################################################

# 한글 금액 단위 -> 숫자 대응표
# 1e8 = 1 × 10^8 = 100,000,000 (지수 표기)
UNIT_MAP = {
    "조": 1_000_000_000_000,
    "억": 100_000_000,
    "천만": 10_000_000,
    "백만": 1_000_000,
    "만": 10_000,
    "천": 1_000,
}


# ============================================================
# 금액 글자 -> 원 단위 숫자로 바꾸는 함수
# ============================================================
def parse_krw(x: Any) -> float:
    # x     : 금액 칸 값 (숫자, '12억' 같은 글자, NaN 등 여러 가지라 Any)
    # 반환값: 원 단위 숫자 (읽을 수 없으면 NaN, NaN도 float 자료형)
    # 빈 칸이면 그대로 빈 값(NaN) 반환
    if pd.isna(x):
        return np.nan

    # 쉼표와 공백 제거: '3억 5천만원' -> '3억5천만원', '1,200원' -> '1200원'
    s = str(x).replace(",", "").replace(" ", "")

    # --- 경우 1: 숫자만 있거나 숫자+'원'인 경우 -> 바로 변환 ---
    # re.fullmatch: 글자 '전체'가 패턴과 정확히 일치하는지 검사
    #   [\d.]+ -> 숫자나 점(.) 1개 이상 / 원? -> '원'이 있어도 되고 없어도 됨
    if re.fullmatch(r"[\d.]+원?", s):
        return float(s.rstrip("원"))  # rstrip('원'): 오른쪽 끝의 '원' 제거

    # --- 경우 2: '억', '만' 같은 한글 단위가 섞인 경우 ---
    # re.findall: (숫자, 단위) 쌍을 모두 찾음
    #   예) '3억5천만원' -> [('3', '억'), ('5', '천만')]
    #   (조|억|천만|...)? -> 단위가 없을 수도 있음 (없으면 '' 빈 글자)
    #   '천만'을 '천'보다 앞에 써야 '5천만'을 '5천'+'만'으로 잘못 자르지 않음
    pairs = re.findall(r"([\d.]+)(조|억|천만|백만|만|천)?", s)

    if len(pairs) == 0:  # 숫자를 하나도 못 찾음 -> 빈 값
        return np.nan

    total = 0.0
    for num, unit in pairs:
        multiplier = UNIT_MAP.get(unit, 1)  # 표에 없는 단위(빈 글자 포함)는 1배
        total += float(num) * multiplier  # 예) 3 × 1억 + 5 × 1천만
    return total


def add_amount(meta: pd.DataFrame) -> pd.DataFrame:
    # meta  : 메타데이터 표 / 반환값: amount_num 열을 추가한 표
    # 모든 행에 함수를 적용해서 새 열 만들기
    meta["amount_num"] = meta[COLS["amount"]].map(parse_krw)

    # ============================================================
    # 의미 없는 금액 걸러내기
    # ============================================================
    # | : 둘 중 하나라도 True면 True (또는, OR)
    is_empty = meta["amount_num"].isna()
    is_too_small = meta["amount_num"] < MIN_VALID_AMOUNT
    invalid = is_empty | is_too_small

    # where(조건): 조건이 True인 칸은 값 유지, False인 칸은 빈 값(NaN)으로
    # ~invalid = '정상인 행' -> 정상인 값만 남기고 나머지는 지움
    meta["amount_num"] = meta["amount_num"].where(~invalid)
    print("금액 결측(정리 후):", meta["amount_num"].isna().sum(), "건")
    return meta


# ################################################################
# 4. 기관 유형 분류 (노트북 3-3)
# ################################################################

# (유형 이름, 정규식) 쌍의 리스트
# ★ 순서가 중요: 위에서부터 검사해서 '처음 걸리는' 유형으로 결정
#   예) '한국과학기술원' -> '대학'의 과학기술원$에 먼저 걸림 (연구기관의 기술원$보다 위에 있음)
#   예) 'OO교육청' -> '지자체'(교육청$)를 '중앙부처'(청$)보다 먼저 검사해야 지자체로 분류됨
#
# 정규식 기호:  $ = 이름의 맨 끝  /  ^ = 이름의 맨 앞  /  | = 또는
#              \b = 단어 경계 (뒤에 공백·괄호가 오거나 끝나는 지점)
AGENCY_TYPE = [
    ("대학", r"대학교|대학$|산학협력단|과학기술원$"),
    (
        "협회·단체",
        r"협회|협의회|체육회|중앙회|상공회의소|사무국|영화제|조직위원회|^\(사[)）]",
    ),
    ("기업", r"주식회사|\(주\)|㈜"),
    ("공기업·공단", r"공사$|공단$|공사\b|공단\b|은행$|KOICA"),
    ("연구기관", r"연구원$|연구소$|기술원$"),
    (
        "준정부·공공기관",
        r"진흥원|진흥회|평가원|정보원|보호원|개발원|서비스원|재단|센터|의료원|박물관|예술원|테크노파크|BioIN",
    ),
    ("지자체", r"교육청$"),  # '중앙부처'의 청$보다 먼저 검사
    (
        "중앙부처",
        r"부$|처$|청$|위원회$",
    ),  # '조직위원회'는 위의 협회·단체에서 이미 걸러짐
    (
        "지자체",
        r"(특별|광역)?시$|도$|군$|구$|^(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청|충북|충남|전라|전북|전남|경상|경북|경남|제주)",
    ),
]

# 규칙으로 잘못 분류되는 기관은 여기에 직접 지정 (규칙보다 우선 적용)
AGENCY_TYPE_OVERRIDE = {
    "국가과학기술지식정보서비스": "준정부·공공기관",
    # 공공기관 자회사를 '기업' 대신 공기업으로 보고 싶다면 아래 주석 해제
    # '인천공항운영서비스(주)': '공기업·공단',
    # '그랜드코리아레저(주)': '공기업·공단',
}


# ============================================================
# 기관명 하나를 유형으로 바꾸는 함수
# ============================================================
def classify(name: str) -> str:
    # name  : 정리한 기관명 / 반환값: 기관 유형  예) '대학', '지자체', '기타'
    # 직접 지정한 기관이면 규칙을 보지 않고 바로 반환
    if name in AGENCY_TYPE_OVERRIDE:
        return AGENCY_TYPE_OVERRIDE[name]

    # 규칙을 위에서부터 하나씩 검사
    # (유형, 정규식) 쌍을 꺼내서 type_name, pattern 두 변수에 나눠 담음
    for type_name, pattern in AGENCY_TYPE:
        # re.search: 이름 '어딘가에' 패턴이 있으면 찾음 (fullmatch와 달리 전체 일치가 아니어도 됨)
        if re.search(pattern, name):
            return type_name  # 처음 걸린 유형으로 결정하고 함수 종료

    # 어떤 규칙에도 안 걸림
    return "기타"


def add_agency_type(meta: pd.DataFrame) -> pd.DataFrame:  # ← 수정: 타입 힌트 추가
    # meta  : 메타데이터 표 / 반환값: agency_type 열을 추가한 표
    meta["agency_type"] = meta["agency_clean"].map(classify)

    # '기타'로 분류된 기관 목록 -> 보고 규칙이나 AGENCY_TYPE_OVERRIDE에 추가
    is_etc = meta["agency_type"] == "기타"
    if is_etc.any():
        print('기관 유형 "기타":', meta.loc[is_etc, "agency_clean"].tolist())
    return meta


# ################################################################
# 5. 날짜 (노트북 3-4)
# ################################################################
# 86400 = 하루의 초 수 (24시간 × 60분 × 60초)
SECONDS_PER_DAY = 86400


def add_dates(meta: pd.DataFrame) -> pd.DataFrame:
    # meta  : 메타데이터 표
    # 반환값: pub_date_dt, start_dt, deadline_dt, bid_days, prep_days, date_suspect 열을 추가한 표
    # ============================================================
    # 1단계: 날짜 글자 -> 날짜(datetime) 형식으로 변환
    # ============================================================
    # 글자 '2024-03-15'는 계산을 할 수 없지만,
    # 날짜 형식으로 바꾸면 '마감일 - 시작일' 같은 계산과 월별 묶기가 가능해짐
    for key in ["pub_date", "start", "deadline"]:
        col_name = COLS[key]  # 예) 'pub_date' -> '공개 일자'
        new_col = f"{key}_dt"  # 새로 만들 열 이름  예) 'pub_date_dt'

        # errors='coerce': 날짜로 읽을 수 없는 값은 에러 대신 NaT(날짜 없음)로
        meta[new_col] = pd.to_datetime(meta[col_name], errors="coerce")

    # ============================================================
    # 2단계: 입찰 기간(일) 계산
    # ============================================================
    # 날짜 - 날짜 = 기간(timedelta)  예) 3월 25일 - 3월 15일 = 10 days
    # .dt.days: 기간을 숫자(일 수)로 꺼냄
    meta["bid_days"] = (meta["deadline_dt"] - meta["start_dt"]).dt.days

    # ============================================================
    # 3단계: 준비 기간 계산 (공개일 -> 마감일)
    # ============================================================
    # 제안서를 쓸 수 있는 실제 시간 = 마감일 - 공개일
    period = meta["deadline_dt"] - meta["pub_date_dt"]  # 결과: 기간(timedelta)

    # .dt.days 대신 total_seconds()를 쓰는 이유: 시간 단위까지 소수로 반영하기 위해
    #   예) 1일 12시간 -> .dt.days = 1 / total_seconds() / 86400 = 1.5
    meta["prep_days"] = period.dt.total_seconds() / SECONDS_PER_DAY

    # ============================================================
    # 4단계: 의심 행에 표시(플래그)만 달기
    # ============================================================
    # 행을 지우지 않고 True/False 열로 '의심스럽다'는 표시만 남김
    #   -> 나중에 다시 포함하거나 원인을 확인할 수 있음
    short_prep = meta["prep_days"] < SUSPECT_PREP_DAYS  # 1일 미만이면 True
    meta["date_suspect"] = short_prep
    print("날짜 의심(준비기간 1일 미만):", short_prep.sum(), "건")
    return meta


# ################################################################
# 6. 파일명 정리 (노트북 3-5)
# ################################################################
# 실제 파일명(file_df['파일명_실제'])을 만들 때와 같은 방식으로 정리해야 정확히 비교됨
#   strip()         : 앞뒤 공백 제거
#   normalize('NFC'): 자모가 분리된 한글('ㅎ+ㅏ+ㄴ')을 하나로 합침('한')
def normalize_name(s: str) -> str:
    # s     : 파일명 / 반환값: 앞뒤 공백을 지우고 NFC로 정리한 파일명
    return unicodedata.normalize("NFC", s.strip())


# ################################################################
# 7. 전체 실행
# ################################################################
def load_and_prepare_metadata(meta_path: PathLike) -> pd.DataFrame:
    # meta_path: data_list.csv 경로
    # 반환값   : 정제가 끝난 메타데이터 표 (원본 열 + 새로 만든 열)
    # 함수가 두 값을 돌려주므로, 변수 두 개로 나눠 받음
    #   meta -> 읽은 표(DataFrame), enc -> 성공한 인코딩 이름
    meta, enc = read_csv_any(meta_path)
    print("인코딩:", enc, "| shape:", meta.shape)
    check_columns(meta)

    meta = add_notice_key(meta)
    meta = add_title_tags(meta)
    meta = add_agency_clean(meta)
    meta = add_amount(meta)
    meta = add_agency_type(meta)
    meta = add_dates(meta)
    meta["파일명_norm"] = meta[COLS["filename"]].astype(str).map(normalize_name)

    print("공고번호 없는 문서:", (~meta["has_notice_no"]).sum())
    return meta
