"""
요구사항 분리 + 요구사항 청크 형식 정리

노트북 대응 셀:
  - 4단계 마지막 : split_requirements (문서 텍스트를 요구사항 하나하나로 자름)
  - 7단계 앞부분 : compact_requirement (요구사항 1개를 짧은 형식으로 정리)

★ 요구사항 분리는 반드시 '정제 전' 원문(raw_text)에서 해야 함
   정제(NFKC)를 거치면 장 제목의 로마 숫자 'Ⅳ'가 영문 'IV'로 바뀌어 장의 경계를 못 찾음
"""

import re
from typing import Any  # 타입 힌트용

import pandas as pd  # 타입 힌트(pd.DataFrame)용

# ################################################################
# 타입 힌트(type hint)
# ################################################################
# 함수가 받는 값과 돌려주는 값의 자료형을 적어 두는 표시
#   - 이름: 자료형            -> 파라미터의 자료형  예) text: str
#   - -> 자료형               -> 돌려주는 값의 자료형
#   - bool                   -> True 또는 False
#   - re.Pattern             -> re.compile()로 미리 준비한 정규식
#   - re.Match               -> 정규식이 찾은 결과 하나 (위치/내용이 담긴 객체)
#   - Optional[str]          -> 글자(str) 또는 None
#   - tuple[A, B, C]         -> 값 3개를 묶은 튜플 (순서대로 A, B, C 자료형)
#   - dict[str, Any]         -> 키는 글자, 값은 아무 자료형인 딕셔너리
#                              (요구사항 1개: {'req_id': 'SFR-001', 'req_name': '...', 'text': '...', ...})

# 요구사항 1개를 담는 딕셔너리에 별칭(alias)을 붙여 아래 함수들에서 씀
Requirement = dict[str, Any]

# ################################################################
# 0단계: 정규식 패턴 정의
# ################################################################

# --- 요구사항 ID 모양 ---
# 예) SFR-001, STR-1, SFR-004-04, 'COR - 007'(하이픈 앞뒤 공백)
#   [A-Z]{2,4}           -> 영문 대문자 2~4개
#   \s*-\s*              -> 하이픈 (앞뒤 공백 허용)
#   \d{1,4}              -> 숫자 1~4개
#   (?:\s*-\s*\d{1,4})?  -> '-04' 같은 하위 번호가 있어도 되고 없어도 됨
#                          (?: ) = 묶기만 하고 따로 꺼내지는 않는 괄호
ID_PAT = r"[A-Z]{2,4}\s*-\s*\d{1,4}(?:\s*-\s*\d{1,4})?"

# --- ID 앞에 붙는 항목명 ---
# 예) '요구사항 고유번호', '요구사항 번호', '요구사항ID', '요구사항 아이디'
LABEL = r"요구사항\s*(?:고유\s*)?(?:번호|ID|아이디)"

# re.compile(): 정규식을 미리 준비해 두기 (여러 번 쓸 때 빠르고, 이름으로 관리하기 편함)
# rf'...'     : r(원시 문자열) + f(변수 끼워 넣기) -> {LABEL}, {ID_PAT} 자리에 위 패턴이 들어감

# --- 방식 1: '항목명 + ID'로 요구사항 시작점 찾기 ---
# 예) '요구사항 고유번호 : SFR-001'
#   [\s:：]*       -> 공백, 콜론(:), 전각 콜론(：)이 0개 이상
#   ({ID_PAT})     -> ID 부분만 따로 꺼냄 (그룹 1번)
#   (?![\dA-Za-z]) -> ID 바로 뒤에 숫자·영문자가 오면 안 됨
REQ_START = re.compile(rf"{LABEL}[\s:：]*({ID_PAT})(?![\dA-Za-z])")

# --- 방식 2: 항목명 없이 'ID만 있는 줄'로 시작점 찾기 ---
#   (?m)            -> 여러 줄 모드: ^ 와 $ 가 '각 줄'의 시작과 끝을 뜻함
#   ^[ \t]* … [ \t]*$ -> 그 줄에는 앞뒤 공백·탭과 ID만 있음
LINE_ID = re.compile(rf"(?m)^[ \t]*({ID_PAT})[ \t]*$")

# --- 시작점 바로 앞의 '요구사항 분류' 줄 ---
# 예) '요구사항 분류 : 기능' 다음 줄에 '요구사항 고유번호 : SFR-001'이 오는 표 구조
#   [^\n]{0,30} -> 같은 줄 안의 글자 0~30개
#   \s*$        -> 검사 범위의 맨 끝에 붙어 있어야 함 (= 시작점 '바로 앞' 줄이어야 함)
CAT_BEFORE = re.compile(r"요구사항\s*분류[\s:：]*[^\n]{0,30}\s*$")

# --- 요구사항 이름 ---
# 예) '요구사항 명칭 : 회원 로그인 기능' -> '회원 로그인 기능'
NAME_RE = re.compile(r"요구사항\s*명칭[\s:：]*([^\n]+)")

# --- 요구사항 구간이 끝나는 지점 ---
# 줄 맨 앞에 아래 중 하나가 오면 '다른 내용이 시작됨'으로 보고 거기서 자름
#   'Ⅳ. 제안서 작성 요령' 같은 장 제목 (Ⅰ~Ⅹ는 영문 I, V가 아닌 로마 숫자 전용 글자)
#   '【붙임', '[서식', '[붙임', '붙임 서식', '<서식'
# 파이썬은 나란히 쓴 문자열을 하나로 이어 붙이므로, 긴 정규식을 줄 단위로 나눠 씀
BOUNDARY = re.compile(
    r"\n[ \t]*"
    r"(?:"
    r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][ \t]*(?:\.|\n|[ \t])"
    r"|【[ \t]*붙임"
    r"|\[[ \t]*(?:서식|붙임)"
    r"|붙임[ \t]*서식"
    r"|<[ \t]*서식"
    r")"
)


# ################################################################
# 1단계: 요구사항 ID가 아닌 것 걸러내기
# ################################################################
# ID 모양이지만 요구사항 번호가 아닌 접두어 (IR은 정상 ID라 넣지 않음)
EXCLUDE_PREFIX = {"ID", "IP", "PC"}  # { }에 값만 나열하면 집합(set)


def valid_id(raw_id: str) -> bool:
    # raw_id: 정규식이 찾은 ID 글자  예) 'COR - 007'
    # 반환값: 요구사항 ID로 인정하면 True, 아니면 False
    no_space = re.sub(r"\s", "", raw_id)  # 공백 제거: 'COR - 007' -> 'COR-007'
    prefix = no_space.split("-")[0]  # 'COR'

    if prefix in EXCLUDE_PREFIX:
        return False

    # 'II-1', 'IV-2'처럼 영문 I, V, X로만 된 접두어는 목차 번호이므로 제외
    return not re.fullmatch(r"[IVX]+", prefix)


# 패턴으로 찾은 시작점 중 올바른 ID만 남기는 함수
def find_valid_starts(pattern: re.Pattern, text: str) -> list[re.Match]:
    # pattern: 시작점을 찾는 정규식 (REQ_START 또는 LINE_ID)
    # text   : 문서 원문
    # 반환값 : 올바른 ID로 시작하는 매치 객체 리스트 (문서 앞쪽부터 순서대로)
    starts = []
    # finditer(): 패턴에 맞는 곳을 앞에서부터 하나씩 꺼냄 (위치 정보가 담긴 '매치 객체')
    for m in pattern.finditer(text):
        if valid_id(m.group(1)):
            starts.append(m)
    return starts


# ################################################################
# 2단계: 텍스트를 요구사항 단위로 자르는 함수
# ################################################################
# 반환값 3개: (요구사항 리스트, 요구사항 외 나머지 텍스트, 사용한 방식)


def split_requirements(text: str) -> tuple[list[Requirement], str, str | None]:
    # text  : 정제 '전' 문서 원문
    # 반환값: (요구사항 리스트, 요구사항 외 나머지 텍스트, 분리 방식 'label'·'line' 또는 None)

    # --- (1) 시작점 찾기: 방식 1 -> 하나도 없으면 방식 2 ---
    starts = find_valid_starts(REQ_START, text)
    mode = "label"

    if len(starts) == 0:
        starts = find_valid_starts(LINE_ID, text)
        mode = "line"

    if len(starts) == 0:  # 둘 다 실패 -> 요구사항 없음, 전체를 나머지로
        return [], text, None

    # --- (2) 구간 시작 위치를 먼저 모두 계산  ← 수정 ---
    # 수정 전: 구간 끝을 '다음 요구사항의 고유번호 줄'로 잡음
    #   -> 다음 요구사항의 '요구사항 분류' 값 줄(예: '보안 요구사항')이
    #     앞 요구사항 끝에도 중복으로 붙는 문제가 있었음
    # 수정 후: '요구사항 분류' 줄까지 포함한 시작 위치를 미리 모두 계산해 두고,
    #          구간 끝을 '다음 요구사항의 (조정된) 시작 위치'로 잡음
    start_positions = []  # 요구사항별 시작 위치 (분류 줄 포함)

    for m in starts:
        start = m.start()

        if mode == "label":
            # 시작점 앞 60글자 안에 '요구사항 분류' 줄이 붙어 있으면 그 줄부터 포함
            window_start = max(0, start - 60)
            cat_match = CAT_BEFORE.search(text[window_start:start])
            if cat_match:
                # 잘라 낸 조각 안의 위치 -> 전체 텍스트 기준 위치로 환산
                start = window_start + cat_match.start()

        start_positions.append(start)

    # --- (3) 시작점마다 구간 잘라내기 ---
    best = {}  # ID -> 요구사항 정보 (같은 ID는 하나만 남김)
    first_start = len(text)  # 요구사항 구간들의 맨 앞 위치 (처음엔 일부러 최댓값)
    last_end = 0  # 요구사항 구간들의 맨 뒤 위치 (처음엔 일부러 최솟값)

    # enumerate(): (순번, 값)을 함께 꺼냄 -> 다음 시작점 start_positions[i + 1]을 찾는 데 순번이 필요
    for i, m in enumerate(starts):
        # (3-1) 구간 시작 위치: 위 (2)에서 미리 계산한 값
        start = start_positions[i]

        # (3-2) 구간 끝 위치: 다음 요구사항의 (조정된) 시작 위치 (마지막이면 텍스트 끝)  ← 수정
        end = start_positions[i + 1] if i + 1 < len(starts) else len(text)

        # 그 사이에 장 제목·붙임 서식이 나오면 거기서 끊음
        # 컴파일된 패턴.search(텍스트, 시작, 끝): 지정한 범위 안에서만 찾음
        boundary_match = BOUNDARY.search(text, m.end(), end)
        if boundary_match:
            end = boundary_match.start()

        # (3-3) 구간 텍스트와 ID
        body = text[start:end].strip()
        req_id = re.sub(r"\s", "", m.group(1))  # 공백 없는 형태로 통일

        # (3-4) 요구사항 이름 찾기
        if mode == "label":
            name_match = NAME_RE.search(body)
            name = name_match.group(1).strip() if name_match else None
        else:
            # 방식 2: ID 줄 다음에 오는 '처음으로 비어 있지 않은 줄'을 이름으로 사용
            lines = []
            for line in body.split("\n")[1:]:  # [1:]: 첫 줄(ID 줄)은 건너뜀
                if line.strip():  # 공백뿐인 줄 제외
                    lines.append(line.strip())
            name = lines[0] if lines else None

        # (3-5) 같은 ID가 이미 있으면 '내용이 더 긴 쪽'을 남김
        # 문서에 요구사항 목록표(짧음)와 상세표(김)가 둘 다 있는 경우가 많기 때문
        if req_id not in best or len(body) > len(best[req_id]["text"]):
            best[req_id] = {
                "req_id": req_id,
                "req_cat": req_id.split("-")[0],  # 접두어 = 분류  예) 'SFR'
                "req_name": name,
                "text": body,
                "mode": mode,
            }

        # (3-6) 요구사항 구간 전체의 맨 앞·맨 뒤 위치 갱신
        first_start = min(first_start, start)
        last_end = max(last_end, end)

    # --- (4) 요구사항 구간 앞뒤의 나머지 텍스트 (사업 개요, 붙임 서식 등) ---
    rest = text[:first_start] + "\n" + text[last_end:]

    # best.values(): 딕셔너리의 값(요구사항 정보)만 꺼내 리스트로
    return list(best.values()), rest, mode


def split_all(docs: pd.DataFrame) -> pd.DataFrame:
    # docs  : extract_all이 돌려준 추출 결과 표 (raw_text, ext 열 필요)
    # 반환값: 같은 표에 reqs, split_mode, n_req_split 열을 추가한 표

    # ============================================================
    # 문서마다 요구사항 분리 실행 (HWP만)
    # ============================================================
    reqs_list = []  # 문서별 요구사항 리스트
    mode_list = []  # 문서별 분리 방식 ('label', 'line', None)

    # zip(A, B): 두 목록에서 같은 순서의 값을 한 쌍씩 꺼냄
    #   예) ('1번 문서 텍스트', 'hwp'), ('2번 문서 텍스트', 'pdf'), ...
    for text, ext in zip(docs["raw_text"], docs["ext"], strict=False):
        if ext == "hwp":
            # 반환값 3개: (요구사항 리스트, 나머지 텍스트, 분리 방식)
            reqs, _rest, mode = split_requirements(text)
        else:
            # HWP가 아니면 분리하지 않음 -> 요구사항 없음, 전체가 나머지, 방식 없음
            reqs, _rest, mode = [], text, None

        reqs_list.append(reqs)
        mode_list.append(mode)

    docs["reqs"] = reqs_list  # 각 칸에 요구사항 딕셔너리들의 리스트
    docs["split_mode"] = mode_list
    docs["n_req_split"] = docs["reqs"].str.len()  # 문서별로 분리된 요구사항 개수

    # 'label': 항목명+ID 방식 / 'line': ID만 있는 줄 방식 / None: 요구사항을 못 찾음
    is_hwp = docs["ext"] == "hwp"
    print(
        "요구사항 분리 방식(HWP):",
        docs.loc[is_hwp, "split_mode"].value_counts(dropna=False).to_dict(),
    )
    print("분리된 요구사항 수:", int(docs["n_req_split"].sum()))
    return docs


# ################################################################
# 요구사항 1개를 짧은 형식으로 정리 (노트북 7단계 앞부분)
# ################################################################

# 본문 항목명 -> 통일된 표기
# 키는 '공백을 모두 뺀' 형태 -> '세부 내용', '세부내용' 모두 '세부내용:'으로 통일
FIELD = {
    "정의": "정의:",
    "내용": "세부내용:",
    "세부내용": "세부내용:",
    "산출정보": "산출정보:",
}

# 지울 '틀' 줄: 줄 전체가 아래 항목명 중 하나뿐인 경우
#   '요구사항 고유번호', '요구사항 명칭', '요구사항 분류', '상세 설명', '관련 요구사항' 등
#   '요구', '사항', '상세', '설명'처럼 표 칸이 좁아 항목명이 여러 줄로 쪼개진 조각도 포함
LABEL_LINES = re.compile(
    r"^\s*("
    r"요구사항\s*(고유\s*)?(번호|ID|아이디)"
    r"|요구사항\s*명칭"
    r"|요구사항\s*분류"
    r"|요구사항"
    r"|상세\s*설명"
    r"|세부"
    r"|관련\s*요구사항"
    r"|요구|사항|상세|설명"
    r")\s*$"
)


# [입력 예시] req['text']             [출력 예시]
#   요구사항 분류                        [SFR-001] 회원 로그인
#   기능                                 기능
#   요구사항 고유번호                    정의: 사용자 인증 기능
#   SFR-001                              세부내용: - 아이디/비밀번호 로그인
#   요구사항 명칭                        - 간편 로그인
#   회원 로그인
#   정의
#   사용자 인증 기능
#   세부 내용
#   - 아이디/비밀번호 로그인
#   - 간편 로그인


def compact_requirement(req: Requirement) -> str:
    # req   : split_requirements가 돌려준 요구사항 1개
    # 반환값: '[ID] 이름 / 정의: / 세부내용: / 산출정보:' 형식으로 정리한 글자

    out = []  # 남길 줄들
    in_body = (
        False  # '정의', '세부내용' 같은 본문 항목이 시작됐는지 (처음엔 아직 시작 전)
    )

    for raw_line in req["text"].split("\n"):
        line = raw_line.strip()

        # (1) 빈 줄은 건너뜀
        if not line:
            continue

        key = re.sub(r"\s", "", line)  # 공백을 모두 뺀 형태 (항목명 비교용)

        # (2) 본문 항목명 줄 -> 통일된 표기로 바꿔 넣고, 여기부터 '본문 시작'
        if key in FIELD:
            out.append(FIELD[key])
            in_body = True
            continue

        # (3) 틀(항목명) 줄 -> 삭제
        if LABEL_LINES.match(line):
            continue

        # (4) 본문이 시작되기 전에 나오는 ID 줄·이름 줄 -> 삭제
        #     (맨 첫 줄에 '[ID] 이름'으로 따로 붙일 것이므로 중복 방지)
        #     본문 안에서 같은 글자가 나오면 내용이므로 남김
        if not in_body and (key == req["req_id"] or line == req.get("req_name")):
            continue

        # (5) 나머지는 내용이므로 남김
        out.append(line)

    body = "\n".join(out)

    # (6) 항목명과 그 다음 줄을 한 줄로 합치기  예) '정의:\n사용자 인증' -> '정의: 사용자 인증'
    #   (정의:|세부내용:|산출정보:)  -> 항목명 (그룹 1번)
    #   \n                          -> 바로 뒤의 줄바꿈
    #   (?!(?:정의:|...))           -> 다음 줄이 또 다른 항목명이면 합치지 않음 (내용이 빈 항목)
    #   r'\1 '                      -> 그룹 1번(항목명) + 공백으로 바꿈 = 줄바꿈만 공백으로 바뀜
    body = re.sub(
        r"(정의:|세부내용:|산출정보:)\n(?!(?:정의:|세부내용:|산출정보:))",
        r"\1 ",
        body,
    )

    # req.get('req_name') or '': 이름이 None이면 빈 글자로 (줄 앞에 'None'이 찍히지 않도록)
    name = req.get("req_name") or ""
    return f"[{req['req_id']}] {name}\n{body}"
