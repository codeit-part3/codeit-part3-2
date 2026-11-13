"""
텍스트 정제: (1) PDF 머리말/꼬리말 제거, (2) 깨진 문자/쪽 번호/구분선 정리, (3) 목차 제거

노트북 : 6-1. 텍스트 정제
"""

import re
import unicodedata
from collections import Counter  # 항목별 개수 세기

import pandas as pd  # 타입 힌트(pd.DataFrame)용

# ################################################################
# 타입 힌트(type hint)
# ################################################################
# 함수가 받는 값과 돌려주는 값의 자료형을 적어 두는 표시
#   - 이름: 자료형          -> 파라미터의 자료형  예) t: str
#   - 이름: 자료형 = 기본값  -> 기본값이 있는 파라미터  예) min_ratio: float = 0.5
#   - -> 자료형             -> 돌려주는 값의 자료형
#   - list[str]            -> 글자들의 리스트
#   - tuple[str, int]      -> 값 2개를 묶은 튜플 (순서대로 글자, 정수)
#   - set[str]             -> 글자들의 집합 (중복 없음)
#   - Counter[str]         -> 글자별 개수를 세는 Counter
#   - re.Match             -> 정규식이 찾은 결과 하나 (위치/내용이 담긴 객체)

# ################################################################
# (1) PDF 머리말/꼬리말 제거
# ################################################################
# PDF는 페이지마다 같은 머리말('OO사업 제안요청서')이나 꼬리말('대외비')이 반복됨
# -> '전체 페이지의 절반 이상에 나오는 짧은 줄'을 머리말/꼬리말로 보고 지움
#   min_ratio=0.5 : 페이지의 50% 이상에서 나오면 반복으로 판단
#   max_len=60    : 60글자 이하인 줄만 대상 (긴 본문 문장은 건드리지 않음)


def remove_repeated_lines(
    pages: list[str], min_ratio: float = 0.5, max_len: int = 60
) -> list[str]:
    # pages    : 페이지별 텍스트 리스트 (PDF)
    # min_ratio: 전체 페이지 중 이 비율 이상에서 나오면 반복 줄로 판단
    # max_len  : 이 글자 수 이하인 줄만 반복 줄 후보
    # 반환값   : 반복 줄을 지운 페이지별 텍스트 리스트

    # 페이지가 3장 미만이면 '반복'을 판단하기 어려우므로 그대로 반환
    if len(pages) < 3:
        return pages

    # (1) 줄별로 '몇 개의 페이지에 나왔는지' 세기
    freq = Counter()
    for page in pages:
        # 한 페이지 안의 줄을 set으로 모음 -> 같은 페이지에서 여러 번 나와도 1번으로 셈
        lines_in_page = set()
        for line in page.splitlines():  # splitlines(): 줄 단위로 나누기
            if line.strip():  # 빈 줄 제외
                lines_in_page.add(line.strip())

        for line in lines_in_page:
            freq[line] += 1

    # (2) 반복 줄 목록 만들기
    repeated = set()
    for line, count in freq.items():  # items(): (줄, 나온 페이지 수) 쌍
        appear_ratio = count / len(pages)
        if appear_ratio >= min_ratio and len(line) <= max_len:
            repeated.add(line)

    # (3) 페이지마다 반복 줄을 빼고 다시 합치기
    cleaned_pages = []
    for page in pages:
        kept_lines = []
        for line in page.splitlines():
            if line.strip() not in repeated:
                kept_lines.append(line)
        cleaned_pages.append("\n".join(kept_lines))

    return cleaned_pages


# ################################################################
# (2) 글자 정제
# ################################################################

# 이 데이터 원문에서 발견된 옛한글식 표기(조합용 자모로 쓰인 글자)를 바로잡는 표
OLD_HANGUL_FIX = {"ᄒᆞᆫ글": "한글", "ᄄᆞ라": "따라", "과ᅟᅥᆫ리": "관리", "지ᅟᅯᆫ": "지원"}

# NFKC 정규화: 모양만 다르고 뜻이 같은 글자를 표준 글자로 통일
#   예) 전각 'Ａ１' -> 'A1',  '①' -> '1',  '㎡' -> 'm2',  '…' -> '...'
#   단, 아래 두 종류는 정규화하면 오히려 문제가 생기므로 제외
#     \u3130-\u318F : 'ㄱ', 'ㅏ' 같은 한글 낱자 -> NFKC가 화면에서 깨져 보이는 '조합용 자모'로 바꿔 버림
#     \u3200-\u327F : '㉮', '㈎' 같은 원/괄호 한글 -> 목록 기호 역할이 사라짐
#   [^ ... ]+ : 위 두 범위가 '아닌' 글자들이 연속된 덩어리
#   원숫자(①~⑳ 등)도 NFKC에서 제외
NON_JAMO_RUN = r"[^\u3130-\u318F\u3200-\u327F\u2460-\u24FF]+"


def normalize_match(m: re.Match) -> str:
    # m     : 정규식이 찾은 덩어리 하나 / 반환값: NFKC로 정규화한 글자
    # re.sub에 넘길 함수: 찾은 덩어리를 NFKC로 정규화해서 돌려줌 (원래 코드의 lambda와 같음)
    return unicodedata.normalize("NFKC", m.group())


def clean_text(t: str) -> str:
    # t     : 정제할 텍스트 / 반환값: 정제한 텍스트

    # (1) 옛한글식 표기 교정
    for old, new in OLD_HANGUL_FIX.items():
        t = t.replace(old, new)

    # (2) 한글 낱자·원 문자 한글을 제외한 부분만 NFKC 정규화
    t = re.sub(NON_JAMO_RUN, normalize_match, t)

    # (3) 눈에 안 보이는 제어 문자 -> 공백
    #     \x00~\x1f 중 탭(\x09), 줄바꿈(\x0a), 캐리지 리턴(\x0d)은 남기고 나머지만
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)

    # (4) 보이지 않거나 의미 없는 특수 영역 문자 -> 삭제
    #     \ue000-\uf8ff         : 사용자 정의 영역(PUA, Private Use Area). 글꼴 전용 기호라 다른 곳에선 깨짐
    #     \U000f0000-\U0010ffff : 보조 평면의 사용자 정의 영역
    #                             (extract.py에서 서로게이트 쌍을 합치면 HWP 글꼴 전용 글머리 기호가
    #                              이 영역의 글자로 나옴. 대문자 \U + 8자리 = 4바이트 크기의 글자 번호)
    #     \u200b-\u200f         : 폭이 0인 공백 등 (zero-width)
    #     \ufeff                : BOM(파일 앞 표시 문자)
    t = re.sub(r"[\ue000-\uf8ff\U000f0000-\U0010ffff\u200b-\u200f\ufeff]", "", t)

    # (5) 목차의 점선/구분선 (같은 기호가 4개 이상 연속) -> 공백  예) '·······', '-----'
    t = re.sub(r"[·ㆍ・.\-_]{4,}", " ", t)

    # (6) 쪽 번호만 있는 줄 삭제
    #     flags=re.M: ^, $ 가 각 줄의 시작/끝을 뜻함
    t = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", t, flags=re.MULTILINE)  # 예) '- 3 -'
    t = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", t, flags=re.MULTILINE)  # 예) '3 / 40'

    # (7) 공백 정리
    t = re.sub(r"[ \t]+", " ", t)  # 연속 공백·탭 -> 공백 1개
    t = re.sub(r" *\n *", "\n", t)  # 줄 앞뒤 공백 제거
    t = re.sub(r"\n{3,}", "\n\n", t)  # 빈 줄 여러 개 -> 빈 줄 1개
    return t.strip()


# ################################################################
# (3) 목차 제거
# ################################################################
# 목차 줄 모양: '제목 + (탭 또는 점선) + 쪽 번호'
#   예) 'Ⅰ. 사업 개요 ······· 3'
#   예) '1. 추진 배경\t5'
#   예) '2. 요구사항 ------' 다음 줄에 '12' (번호가 다음 줄로 넘어간 경우)
TOC_LINE = re.compile(
    r"(?m)^"
    r"[^\n]{1,80}?"  # 제목: 줄바꿈 없는 글자 1~80개 (?: 최대한 짧게)
    r"(?:"
    r"\t[ \t]*"  #   (1) 탭으로 구분
    r"|[ \t]*[·・.…\-─]{4,}[ \t]*(?:\n[ \t]*)*"  #   (2) 점선·대시선 4개 이상 (뒤에 줄바꿈이 와도 됨)
    r")"
    r"\d{1,3}[ \t]*$"  # 쪽 번호 1~3자리로 줄이 끝남
    r"\n?"  # 줄바꿈까지 함께 지움
)

# '목차', '목   차', '목 차 -' 처럼 '목차'라는 제목만 있는 줄
#   : '<목차>', '[목차]', '【목 차】', '(목차)'처럼 괄호가 붙은 제목도 포함
#   [<\[【(]? : 여는 괄호가 있어도 되고 없어도 됨 (? = 0개 또는 1개)
#   [>\]】)]? : 닫는 괄호도 마찬가지
TOC_HEAD = re.compile(
    r"(?m)^[ \t]*[<\[【(]?[ \t]*목[ \t]*차[ \t]*[>\]】)]?[ \t\-]*$\n?"
)


def remove_toc(t: str) -> tuple[str, int]:
    # t     : 목차를 지울 텍스트 (정제 전 원문)
    # 반환값: (목차를 지운 텍스트, 지운 목차 줄 수)
    # subn: sub(바꾸기)와 같지만 (바뀐 텍스트, 바꾼 횟수) 두 값을 함께 돌려줌
    t, n_removed = TOC_LINE.subn("", t)
    t = TOC_HEAD.sub("", t)
    return t, n_removed


# ################################################################
# (4) 모든 문서에 적용: 머리말 제거(PDF만) -> 목차 제거 -> 글자 정제
# ################################################################
def clean_docs(docs: pd.DataFrame) -> pd.DataFrame:
    # docs  : 추출 결과 표 (pages, ext, raw_text, n_chars 열 필요)
    # 반환값: 같은 표에 clean_text, toc_removed, clean_chars 열을 추가한 표
    clean_list: list[str] = []
    toc_count_list: list[int] = []

    for pages, ext in zip(docs["pages"], docs["ext"], strict=False):
        if ext == "pdf":
            pages = remove_repeated_lines(
                pages
            )  # PDF만 페이지 구분이 있으므로 PDF에만 적용

        text = "\n".join(pages)  # 페이지들을 하나의 텍스트로
        text, n_removed = remove_toc(text)
        text = clean_text(text)

        clean_list.append(text)
        toc_count_list.append(n_removed)

    docs["clean_text"] = clean_list
    docs["toc_removed"] = toc_count_list

    # (5) 결과 확인
    docs["clean_chars"] = docs["clean_text"].str.len()
    before_avg = docs["n_chars"].mean()
    after_avg = docs["clean_chars"].mean()
    reduce_pct = (
        1 - docs["clean_chars"].sum() / max(docs["n_chars"].sum(), 1)
    ) * 100  # 전체 글자 감소율
    print(
        f"정제 전 평균 {before_avg:.0f}자 -> 정제 후 {after_avg:.0f}자 ({reduce_pct:.1f}% 감소)"
    )
    print("문서당 제거된 목차 줄 중앙값:", docs["toc_removed"].median())

    ct = docs["clean_text"]
    # \u1100-\u11FF: 조합용 자모 (화면에서 깨져 보이는 낱자) -> 정제가 잘 됐다면 0개
    print("정제 후 조합용 자모:", ct.str.count(r"[\u1100-\u11FF]").sum(), "(기대값 0)")
    # \u3200-\u327F: ㉮, ㈎ 같은 원/괄호 한글 -> 정규화에서 제외했으므로 원래 개수가 그대로 남아야 함
    print(
        "정제 후 원 문자 한글 기호:",
        ct.str.count(r"[\u3200-\u327F]").sum(),
        "| 원문:",
        docs["raw_text"].str.count(r"[\u3200-\u327F]").sum(),
    )
    return docs
