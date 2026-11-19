"""작성 도우미: 제안서 작성 가이드 초안을 .docx 로 만드는 랭그래프 에이전트.

팀 화면 연결용 함수는 baseline 과 같은 모양입니다.
    answer(message, history=None) -> 답변 문자열 (파일 경로가 문장 안에 들어감)

입출력 계약 (baseline 과 비교)
    입력: message=이번 질문 문자열, history=[{"role": "user"|"assistant", "content": 문자열}, ...] 이전 대화.
    출력: 답변 문자열 하나. docx 파일 자체나 백엔드 응답 객체를 반환하지 않고, 파일이 만들어지면 경로를 답변에 적는다.
    baseline.answer 와 함수 인자·반환형은 같지만, history 사용은 다르다:
      baseline 은 이전 대화를 RAG 모델 메시지에 포함한다. writer 는 라우팅/이전 사업·서식 찾기에 쓰고,
      가이드 작성 모델에는 이전 대화를 전달하지 않는다. 일반 RAG 경로에만 history 를 baseline 으로 넘긴다.
    backend/routes/chat.py 도 엔진 함수에 (message, history) 를 넘기고 반환 문자열을 응답의 answer 로 감싼다.
    단, 현재 backend/services/engine_loader.py 는 테스트 stub 을 반환하므로 writer 와의 실제 연결은 아직 되어 있지 않다.

흐름:  START -> (작성 요청이면) find_doc -> extract -> [fallback] -> write -> build -> END
              (작성 요청이 아니면) rag -> END   (baseline.answer 그대로)

근거 수준 3단계
    확정/추정 : 공고문 평가표를 읽었고 배점 합계가 맞음(또는 90/80/70점대로 보임)
    부분      : 평가표는 읽었지만 합계가 안 맞음 -> 항목은 쓰되 배점은 확인 필요
    일반      : 평가표를 못 찾음 -> 일반 제안서 구조로 작성 (공고문 근거 아님)

원리 (왜 이렇게 만들었나)
    1) 평가표는 코드가 읽는다. 공고문(txt)의 표를 규칙 기반 파서가 읽어 평가항목·배점을 뽑는다.
       작은 로컬 모델은 숫자를 지어내기 쉬워서 표 읽기는 맡기지 않는다. 읽은 결과는 '배점 합계'로 검산하고,
       검산이 얼마나 됐는지를 근거 수준(확정/추정/부분/없음)으로 문서에 그대로 적는다.
    2) 모델은 문장만 쓴다. 항목마다 '쓰는 칸/꼭 넣을 내용/[예시]/팁/흔한 실수'를 로컬 모델(gemma4:e2b)이 JSON 으로 쓰고,
       형식을 어기거나 안내문을 베끼면 그 항목만 규칙 문장으로 바꾼다. 그래서 모델이 실패해도 초안은 항상 나온다.
    3) 양식 예시는 모델 없이 규칙으로 만든다. 공고문 별지 서식 표의 빈칸을 라벨(예: 대표자, 전화)로 채우고,
       값마다 [문서](공고문에서 가져옴) / [예시](가상 값) / [직접 작성](회사 정보 필요)을 붙인다.
       지어낸 값을 숨기지 않는 것이 합격 기준이다.
    4) 무엇을 할지는 랭그래프 route 가 고른다: 작성 가이드 / 서식 예시 / 그 외(baseline 검색 답변).

파일 구성: 1 평가표 추출 · 2 문서 찾기 · 3 모델로 안내 쓰기 · 4 docx 만들기 · 4-2 서식 예시
          5 랭그래프 · 6 @tool 과 양식 예시 루브릭 평가 · 맨 아래 CLI(질문 1회 / 대화 모드 / --rubric)

실행 확인:  uv run python -m ai.rahui.exp_filter_rerank.rahui_engine.writer "질문" [--no-model]
결과 파일은 outputs/guides/ 에 저장됩니다 (outputs 는 git 에 올라가지 않음).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import TypedDict

from ai.rahui.exp_filter_rerank.rahui_engine import config

ROOT = Path(__file__).resolve().parents[4]
CORPUS_DIR = ROOT / "data" / config.CORPUS
OUT_DIR = ROOT / "outputs" / "guides"

# 모델에게 줄 지시문 파일 이름 (rahui_engine/prompts/<이름>.txt). 바꾸면 항목별 안내문의 형식과 말투가 달라진다.
GUIDE_PROMPT = "guide_v3"
# 코드 안의 few-shot·필터 규칙이 바뀌면 올린다 (캐시 키에 들어가서 옛 캐시를 쓰지 않게 한다).
GUIDE_VERSION = "3.2"
USE_MODEL = True  # False 면 모델 없이 규칙만으로 만든 초안
PARALLEL = max(
    1, int(os.environ.get("WRITER_PARALLEL", "1"))
)  # 모델에 동시에 보낼 요청 수 (1=차례로)
_DEBUG_LOCK = __import__("threading").Lock()
MAX_MODEL_ITEMS = 25  # 항목이 이보다 많으면 앞쪽만 모델로 쓰고 나머지는 기본 문구

# 표에서 "합계 줄"로 보는 말. 공백을 없앤 칸이 이 중 하나와 똑같으면 합계 줄이다 (항목이 아니라 검산용으로만 쓴다).
TOT = ("계", "소계", "합계", "총계", "총점")
# "계량 합계", "정성평가 소계", "총 합계"처럼 앞말이 붙은 합계 줄 (앞말은 8글자까지). 항목이 아니라 합계다.
SUBTOTAL = re.compile(r"[가-힣A-Za-z0-9()·]{0,8}(?:합계|소계|총계|총점)\d?\)?")
MIN_FULL_ITEMS = 5  # 확정으로 보려면 항목이 이 개수 이상이어야 함 (표를 못 읽고 합계 100 한 줄만 읽은 경우를 확정으로 오판하지 않기 위해)
MIN_FULL_SUM = 70  # 평가표 전체로 보려면 항목 배점 합이 이 값 이상이어야 함 (정량 20점 등 일부 표를 확정으로 오판하지 않기 위해)
# 머리글에 이 말이 하나라도 있어야 평가표로 인정한다 ("배점" 열도 함께 있어야 한다).
KEY = ("항목", "평가요소", "평가기준", "평가부문", "평가내용", "평가지표", "구분")
# 배점 칸의 모양: 숫자 하나(소수 가능), 괄호나 "점" 허용. 예) 15 · (15) · 15점.
# 문장이나 "-" 는 점수가 아니므로 그 줄은 건너뛴다.
NUM = re.compile(r"^\(?(\d{1,3}(?:\.\d+)?)\)?점?$")
# 질문에 이 말이 있으면 "작성 가이드 요청"으로 본다 (route 에서 사용).
GUIDE_WORDS = (
    "작성 가이드",
    "작성가이드",
    "가이드",
    "초안",
    "제안서 작성",
    "어떻게 써",
    "어떻게 작성",
    "작성 방법",
    "작성방법",
    "작성 도우미",
    "작성도우미",
)
# 질문에 이 말이 있으면 "서식 예시 요청"으로 본다 (route 에서 사용).
FORM_WORDS = (
    "양식",
    "서식",
    "별지",
    "신청서",
    "표지",
    "위임장",
    "확약서",
    "서약서",
    "작성 예시",
    "작성예시",
    "예시 줘",
    "예시줘",
    "예시를",
    "예시 보여",
    "예시 좀",
    "참조표",
    "수용여부",
    "워드 파일",
    "워드파일",
    "워드로",
    "docx",
)
# 공고문 파일 이름을 찾을 때 버리는 흔한 말. 사업명이 아닌 단어가 점수를 올리지 않게 한다.
STOP = {
    "제안서",
    "작성",
    "가이드",
    "초안",
    "만들어",
    "만들어줘",
    "알려줘",
    "사업",
    "어떻게",
    "해줘",
    "주세요",
    "도와줘",
    "방법",
    "양식",
    "도우미",
    "대한",
    "관련",
    "위한",
    "서식",
    "별지",
    "신청서",
    "표지",
    "예시",
    "채워",
    "보여줘",
    "보여",
    "작성해줘",
    "있어",
    "있나요",
    "줘",
}

# 평가표를 못 찾았을 때 대신 쓰는 공통 항목 6개 (근거 수준 "없음", 배점 없음).  (항목 이름, 안내 문구)
GENERIC = [
    (
        "사업 이해도",
        "발주기관이 이 사업을 왜 하는지, 무엇을 이루려는지 이해하고 있음을 보여 주는 부분",
    ),
    ("추진 전략·수행 방안", "어떤 방법과 순서로 일을 진행할지 설명하는 부분"),
    ("투입 인력", "누가 참여하고 어떤 경험이 있는지 보여 주는 부분"),
    ("추진 일정", "단계별 기간과 결과물을 보여 주는 부분"),
    ("품질·보안 관리", "품질을 점검하고 정보를 지키는 방법을 설명하는 부분"),
    ("유지보수·교육", "사업이 끝난 뒤 운영 지원과 교육을 어떻게 할지 설명하는 부분"),
]


# 공백을 모두 없앤 문자열. 띄어쓰기가 제각각인 표 글자("합  계", "총 계")를 같은 것으로 비교하려고 쓴다.
def sq(s: str) -> str:
    return re.sub(r"\s+", "", s)


# =====================================================================
# 1. 도구: 평가표 추출
# =====================================================================
def table_blocks(txt: str) -> list[str]:
    """[표] 블록 + 줄 앞에 |가 붙은 마크다운 표 (빈 줄·구분선으로 잘린 조각은 이어 붙임)

    원리: 공고문 txt 에는 표가 두 가지 모양으로 들어 있다.
      (1) [표] ... [/표] 로 감싼 블록   (2) 줄이 | 로 시작하는 마크다운 표
    둘 다 "한 줄 = 칸을 | 로 나눈 문자열"인 블록 문자열로 맞춰서 돌려준다 (parse_table 이 같은 방식으로 읽도록).
    """
    # (1) [표] 블록은 정규식으로 그대로 가져온다.
    blocks = re.findall(r"\[표\]\n(.*?)\n\[/표\]", txt, flags=re.S)
    # (2) 마크다운 표: | 로 시작하는 연속된 줄을 run 에 쌓는다.
    run: list[str] = []

    # 쌓인 줄이 3줄 이상이면 하나의 표로 인정한다 (머리글 + 내용 최소 2줄). 짧은 조각은 표가 아님.
    def flush():
        if len(run) >= 3:
            blocks.append("\n".join(run))

    # 한 줄씩 보면서 표 줄이면 쌓고, 표가 아닌 줄을 만나면 쌓인 것을 마무리한다.
    # 구분선(|---|)은 내용이 없으므로 건너뛴다.
    # (표가 빈 줄로 끊긴 경우: 빈 줄은 마무리하지 않고 이어 붙여 병합 표가 두 조각이 되는 것을 막는다)
    for line in txt.split("\n"):
        if line.startswith("|"):
            s = line.strip().strip("|")
            if re.fullmatch(r"[\s|:\-]*", s):
                continue
            run.append(s)
        elif line.strip():
            flush()
            run.clear()
    flush()
    return blocks


# 머리글(head)에서 keys 중 하나가 들어 있는 열 번호를 돌려준다 (skip 에 있는 열은 제외, keys 앞쪽이 우선).
def _pick(head, keys, skip=()):
    for k in keys:
        for i, h in enumerate(head):
            if i not in skip and k in h:
                return i
    return None


# 병합 셀 표용 2차 읽기 스위치. 처음부터 켜면 잘 읽던 공고까지 달라지므로(98건 비교에서 확인),
# extract_items 가 "부분/없음"으로 끝났을 때만 켜서 다시 읽고, 더 나아졌을 때만 그 결과를 쓴다.
_LAST_CELL_FALLBACK = False


def parse_table(block: str):
    """한 표 -> (후보 목록, 표에 적힌 합계) 또는 None. 후보 = [(이름, 항목 리스트)]

    원리: 머리글 줄에서 열의 역할을 찾고(배점 열, 항목 이름 열, 대분류 열, 평가기준 열, 평가요소 열),
    본문을 한 줄씩 읽어 항목(이름·배점·기준)으로 만든다.
    공고문 표는 병합 셀 때문에 줄마다 칸 수가 달라서 다음 보정을 한다.
      - 줄이 머리글보다 짧으면(병합) 오른쪽 기준으로 열을 당겨 읽는다 (off 보정)
      - 대분류/항목 이름 칸이 비어 있으면 윗줄 값을 이어 쓴다
      - "합계/계/소계" 줄은 항목이 아니라 표의 합계(total)로 따로 저장한다
    소계가 섞인 표 때문에 합이 두 번 계산되는 경우를 대비해 후보를 3가지로 돌려준다
    (전체 / 소계 제외 / 칸이 다 찬 행만). extract_items 가 합계가 맞는 후보를 고른다.
    """
    lines = [ln for ln in block.split("\n") if ln.strip()]
    # 표 위에 '|' 없는 안내 문장(예: 고시 이름)이 붙은 표가 있어서, 앞쪽 3줄 안에서 '|'와 "배점"이 함께 있는 첫 줄부터 표로 본다.
    for k in range(min(3, len(lines))):
        if "|" in lines[k] and "배점" in lines[k]:
            lines = lines[k:]
            break
    if not lines or "|" not in lines[0]:
        return None
    # 첫 줄을 머리글로 보고, 공백을 없앤 칸 이름 목록을 만든다.
    head = [sq(c) for c in lines[0].split("|")]
    n = len(head)
    # 배점 열 = 머리글에 "배점"이 들어간 열 (여러 개면 맨 오른쪽 것을 점수로 쓴다).
    # 배점 열이 없거나 평가표 머리글 단어(KEY)가 하나도 없으면 평가표가 아니므로 버린다.
    cols = [i for i, h in enumerate(head) if "배점" in h]
    if not cols or not any(k in "".join(head) for k in KEY):
        return None
    sc = cols[-1]
    skip = set(cols)
    # 항목 이름 열 찾기:
    #   "세부평가항목"이 있으면 그것이 항목 이름이고, "평가항목/평가부문"은 대분류(group).
    #   없으면 "평가항목/항목/평가요소/구분"을 항목 이름으로 보고 대분류는 따로 찾는다.
    sub = _pick(head, ("세부평가항목", "세부항목"), skip)
    if sub is not None:
        name = sub
        group = _pick(head, ("평가부문", "평가항목", "평가분야", "구분"), skip | {name})
    else:
        name = _pick(head, ("평가항목", "항목", "평가요소", "구분"), skip)
        group = _pick(
            head,
            ("평가부문", "평가분야", "대분류", "구분"),
            skip | ({name} if name is not None else set()),
        )
        if name is None:
            name, group = group, None
    if name is None:
        return None
    # 이미 쓴 열(배점·이름·대분류)을 빼고 나머지에서 평가기준(공고문 설명 문장)과 평가요소 열을 찾는다.
    used = skip | {name} | ({group} if group is not None else set())
    crit = _pick(
        head, ("평가기준", "평가내용", "세부기준", "평가지표", "평가방법"), used
    )
    used2 = used | ({crit} if crit is not None else set())
    elem = _pick(head, ("평가요소",), used2)

    # 본문 읽기 준비: rows=항목 행 목록, total=표에 적힌 합계,
    # cur_group/cur_name=병합 셀로 비어 있는 칸에 이어 쓸 윗줄의 값.
    rows, total = [], None
    cur_group, cur_name = "", ""
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|")]
        # off = 머리글 칸 수 - 이 줄의 칸 수. 병합 셀이 있으면 줄이 머리글보다 짧아져 열 위치가 어긋나므로 off 로 보정한다.
        off = n - len(cells)

        # 열 번호 i 의 칸 값을 돌려준다.
        #   줄이 머리글보다 짧음(off>0): 오른쪽 기준으로 당겨서 읽는다 (왼쪽 칸이 병합돼 사라졌다고 보기 때문).
        #   같거나 긺(off<=0): 이름은 왼쪽 기준, 점수는 오른쪽 기준 (오른쪽 끝에 점수가 있는 표가 많다).
        def cell(i, right=False, off=off, cells=cells):
            if i is None:
                return None
            # 헤더보다 칸이 많거나 같음(off<=0): 이름은 왼쪽 기준, 점수는 오른쪽 기준.
            # 병합 셀로 칸이 줄어듦(off>0): 오른쪽 기준.
            j = i - off if (right or off > 0) else i
            return cells[j] if 0 <= j < len(cells) else None

        # 점수 칸이 없거나 숫자 모양이 아니면 항목 행이 아니므로 건너뛴다 (설명 줄, 빈 줄 등).
        s = cell(sc, right=True)
        m = NUM.match(sq(s)) if s is not None else None
        if not m and _LAST_CELL_FALLBACK and len(cells) >= 2:
            # (2차 시도에서만) 병합 셀 표에서는 "비고"가 비거나 없어서 점수가 맨 끝 칸에 오는 줄이 많다.
            # 원래 자리가 숫자가 아니면, 맨 끝 칸이 숫자일 때 그것을 점수로 읽는다 (예: "다) 제안 내용의 일치성 | 1").
            m = NUM.match(sq(cells[-1]))
        if not m:
            continue
        v = float(m.group(1))
        # 줄에 "합계/계/소계/총계/총점" 칸이 있으면 항목이 아니라 표의 합계다.
        # total 에 저장하고 항목에서는 뺀다. (벤처기업협회에서 합계를 항목으로 세어 200점이 된 실수의 재발 방지)
        if any(sq(c) in TOT for c in cells):
            total = v
            continue
        # 앞말이 붙은 합계 줄("계량 합계 | 10")도 항목이 아니다. "총"이 붙은 것은 전체 합계로 저장하고 나머지는 건너뛴다.
        sub_labels = [sq(c) for c in cells if SUBTOTAL.fullmatch(sq(c))]
        if sub_labels:
            if any("총" in x for x in sub_labels):
                total = v
            continue
        # 대분류(평가부문) 칸이 비어 있으면 윗줄 값을 계속 쓴다 (병합 셀은 첫 줄에만 값이 있다).
        g = cell(group)
        if g and g.strip():
            cur_group = g.strip()
        # 항목 이름이 비어 있으면 윗줄 항목에 이어지는 행이다 (cont=True).
        # 나중에 merge_rows 가 윗줄 항목에 배점을 더해 하나로 합친다.
        nm = (cell(name) or "").strip()
        cont = False
        if nm:
            cur_name = nm
        elif rows and cur_name:
            nm, cont = (
                cur_name,
                True,
            )  # 이름 칸이 병합되어 비어 있음 -> 윗줄 항목의 이어지는 행
        else:
            nm = cur_group or "(이름 없음)"
            cur_name = nm
        rows.append(
            {
                "group": cur_group,
                "name": nm,
                "score": v,
                "cont": cont,
                "criteria": (cell(crit) or "").strip(),
                "elements": (cell(elem) or "").strip(),
                "full": len(cells) == n,
            }
        )

    # "소계 제외" 후보 만들기: 앞선 항목 2개 이상의 합과 같은 배점이 나오면 그 줄은 소계로 보고 뺀다
    # (계층 표에서 소계가 항목처럼 한 번 더 더해지는 것을 막는다).
    nosub, run, cnt = [], 0.0, 0
    for r in rows:
        if cnt >= 2 and abs(r["score"] - run) < 0.01:
            run, cnt = 0.0, 0
            continue
        run += r["score"]
        cnt += 1
        nosub.append(r)
    # 후보 3가지를 돌려준다: 전체 / 소계 제외 / 칸이 다 찬 행만.
    cands = [
        ("전체", rows),
        ("소계 제외", nosub),
        ("전체 칸 행만", [r for r in rows if r["full"]]),
    ]
    return cands, total


# 이름 칸이 비어 윗줄에 이어진 행(cont)을 윗줄 항목에 합친다. 예) 한 항목이 세부 기준 3줄로 나뉜 표 -> 배점 3개를 더한 항목 1개.
def merge_rows(items: list[dict]) -> list[dict]:
    """이름이 비어 윗줄에 이어진 행(cont)을 윗줄 항목에 합침 (배점은 더함)"""
    out: list[dict] = []
    for r in items:
        if r.get("cont") and out:
            prev = out[-1]
            prev["score"] += r["score"]
            for k in ("criteria", "elements"):
                if r[k] and r[k] not in prev[k]:
                    prev[k] = (prev[k] + " " + r[k]).strip()
        else:
            out.append(dict(r))
    return out


# "전략 및 방법론 (20)"처럼 평가부문(대분류) 이름이 항목 이름 자리에 들어오고, 진짜 항목 이름은 평가기준 칸에
# 짧게("사업 이해도") 들어 있는 표가 있다 (머리글 칸 수가 줄 칸 수보다 적은 4단 표). 그런 경우 이름과 대분류를 바로잡는다.
_GROUP_NUM = re.compile(r"^(.*?)\s*\(\s*\d+(?:\.\d+)?\s*\)\s*$")
_BULLET = "․·ㆍ•ㅇ○-"


def _fix_group_name(it: dict) -> dict:
    m = _GROUP_NUM.match(it.get("name") or "")
    if not m:
        return it
    crit = (it.get("criteria") or "").lstrip(" \t" + _BULLET).strip()
    # 짧은 이름 꼴("사업 이해도")만 항목 이름으로 인정한다. 문장(끝이 ~다/함/음/가, 마침표 등)이면 평가기준 설명이므로 그대로 둔다.
    if not (2 <= len(crit) <= 14) or re.search(
        r"[.?!]|(?:다|함|음|임|요|가|까)$", crit
    ):
        return it
    if sq(crit) == sq(m.group(1)) or _GROUP_NUM.match(crit):
        return it
    return {
        **it,
        "group": it.get("group") or m.group(1).strip(),
        "name": crit,
        "criteria": it.get("elements") or "",
    }


# 오프라인에서 모델로 읽고 합계 검산을 통과한 평가표 (scripts/build_verified_tables.py 로 만든다).
# 공고 파일명 -> 항목 목록. 있으면 규칙 추출보다 먼저 쓴다. 파일이 없으면 기존 규칙 추출 그대로다.
VERIFIED_PATH = Path(__file__).with_name("verified_tables.json")
_VERIFIED: dict | None = None


def _verified_tables() -> dict:
    global _VERIFIED
    if _VERIFIED is None:
        try:
            _VERIFIED = json.loads(VERIFIED_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _VERIFIED = {}
    return _VERIFIED


def _verified_items(txt: str, name: str):
    """검증된 평가표가 있으면 (items, 확정, 설명), 없으면 None. 평가기준 설명은 규칙 추출 결과에서 이름이 같은 항목의 것을 가져온다."""
    v = _verified_tables().get(name)
    if not v:
        return None
    rule_items, _lv, _why = _extract_items_raw(txt)
    by_name = {re.sub(r"\s+", "", i.get("name") or ""): i for i in rule_items}
    items = []
    for it in v["items"]:
        ref = by_name.get(re.sub(r"\s+", "", it["name"]), {})
        items.append(
            {
                "group": ref.get("group") or it.get("group") or "",
                "name": it["name"],
                "score": it["score"],
                "criteria": ref.get("criteria") or it.get("criteria") or "",
                "elements": ref.get("elements") or it.get("elements") or "",
            }
        )
    total = sum(i["score"] or 0 for i in items)
    return (
        items,
        "확정",
        f"사전에 검증된 평가표입니다. 항목 배점 합이 {total:g}점입니다.",
    )


def extract_items(txt: str, name: str = ""):
    """평가표 읽기 (이름·대분류 바로잡기 포함). 본체는 _extract_items_raw. name 은 공고 파일명(검증된 평가표 조회용)."""
    if name and (hit := _verified_items(txt, name)):
        return hit
    items, level, why = _extract_items_raw(txt)
    items = [_fix_group_name(i) for i in items]
    for i in items:  # 배점이 100을 넘는 것은 표를 잘못 읽은 값(예: 줄바꿈된 '(101)')이므로 점수를 비운다
        if i.get("score") is not None and i["score"] > 100:
            i["score"] = None
    return _downgrade_thin_full(items, level, why)


def _downgrade_thin_full(items, level, why):
    """확정인데 항목이 너무 적거나 이름 없는 항목이 있으면 '부분'으로 낮추고 확인하라고 알린다.

    근거: 한국해양조사협회(원본 17항목)가 '(이름 없음) 100점' 1개만 읽고도 합계 100 이라 확정이 됐다.
    항목이 적은 정상 평가표를 부분으로 낮추는 비용(불필요한 확인 안내)보다 틀린 확정을 그대로 두는 비용이 더 크다.
    """
    if level != "확정":
        return items, level, why
    unnamed = any(
        not (i.get("name") or "").strip() or "이름 없음" in (i.get("name") or "")
        for i in items
    )
    if len(items) >= MIN_FULL_ITEMS and not unnamed:
        return items, level, why
    why += (
        f" 읽은 항목이 {len(items)}개뿐이라 평가표 일부만 읽었을 수 있어요(공고문 평가표를 꼭 확인하세요)."
        if len(items) < MIN_FULL_ITEMS
        else " 이름을 읽지 못한 항목이 있어 평가표를 정확히 읽지 못했을 수 있어요(공고문 평가표를 꼭 확인하세요)."
    )
    return items, "부분", why


def _extract_items_raw(txt: str):
    """평가표 읽기. 1차(기본 규칙)로 읽고, 결과가 "부분"이면 병합 셀용 2차 읽기를 해 본다.

    2차 결과는 '확정'이고 항목이 5개 이상이며 1차보다 많을 때만 채택한다 (잘 읽던 공고가 나빠지지 않게 하고,
    "정성 70 / 정량 30" 처럼 합계만 맞고 너무 뭉뚱그려진 결과를 확정으로 올리지 않기 위한 장치).
    """
    global _LAST_CELL_FALLBACK
    first = _extract_items(txt)
    if first[1] != "부분":
        return _note_low_sum(first)
    _LAST_CELL_FALLBACK = True
    try:
        second = _extract_items(txt)
    finally:
        _LAST_CELL_FALLBACK = False
    if second[1] == "확정" and len(second[0]) >= 5 and len(second[0]) > len(first[0]):
        return _note_low_sum(second)
    return first


# 확정/추정인데 항목 배점 합이 90 미만이면 "정성평가 항목만 읽은" 경우가 많다 (정량·가격 평가 항목이 빠짐).
# 근거: 추정 8건 중 4건, 확정 무작위 18건 중 3건(합이 90 미만인 것은 3건 모두)이 그랬다. 판정은 바꾸지 않고 설명문에 알린다.
def _note_low_sum(res):
    items, level, why = res
    s = sum(i["score"] or 0 for i in items)
    if level in ("확정", "추정") and 0 < s < 90 and "정성평가 항목만" not in why:
        why += " 정성평가 항목만 읽었을 수 있어요(정량·가격 평가 항목은 공고문에서 확인하세요)."
    return items, level, why


def _extract_items(txt: str):
    """-> (항목 리스트, 근거 수준 '확정'|'추정'|'부분'|'없음', 검증 설명)

    원리: 평가표를 읽은 결과가 맞는지를 배점 합계로 검산하고, 검산 결과를 근거 수준으로 돌려준다.
    판정 순서 (위에서부터 처음 맞는 것):
      1) 표에 적힌 합계 = 항목 배점 합, 그리고 합이 70 이상         -> 확정
         (70 미만이면 정량평가 20점 같은 일부 표가 우연히 맞은 것일 수 있어 확정으로 보지 않는다)
      2) 항목 합이 100                                            -> 확정
      3) 나뉜 표들을 합치면 100                                    -> 확정
      4) 가장 큰 표를 골라서:
         - 다른 표에 적힌 합계와 같고 합이 70 이상, 항목 5개 이상   -> 확정
         - 이 표에 합계가 있는데 다름 (항목이 빠졌을 수 있음)       -> 부분
         - 합계 줄이 없고 합이 90/80/70, 항목 5개 이상              -> 추정
         - 그 밖                                                  -> 부분
      표를 하나도 못 찾으면 '없음' (호출 쪽에서 GENERIC 공통 항목을 쓴다).
    """
    # 평가표 후보 모으기: 문서의 모든 표를 parse_table 로 읽어서, 항목이 2개 이상 나온 표만 남긴다.
    tabs = []
    for b in table_blocks(txt):
        t = parse_table(b)
        if t and any(len(c[1]) >= 2 for c in t[0]):
            tabs.append(t)
    if not tabs:
        return [], "없음", "문서에서 배점이 있는 평가표를 찾지 못했습니다."
    # 표마다, 후보(전체/소계 제외/칸이 다 찬 행만)마다 검산한다. 처음 맞는 것을 바로 돌려준다.
    for cands, total in tabs:
        for _label, items in cands:
            if len(items) < 2:
                continue
            s = sum(i["score"] for i in items)
            if (
                total is not None and abs(s - total) < 0.01 and s >= MIN_FULL_SUM
            ):  # 정량 소계(20점)처럼 일부만 맞는 경우는 확정으로 보지 않음
                return (
                    merge_rows(items),
                    "확정",
                    f"항목 배점 합 {s:g}점이 표에 적힌 합계 {total:g}점과 같습니다.",
                )
            if abs(s - 100) < 0.01:
                return merge_rows(items), "확정", f"항목 배점 합이 {s:g}점입니다."
    # 합계가 한 표에 없고 여러 표로 나뉜 경우(예: 기술/가격 표): 모든 표의 항목을 합쳐 100점이면 확정.
    allitems = [i for cands, _ in tabs for i in cands[0][1]]
    if abs(sum(i["score"] for i in allitems) - 100) < 0.01 and len(tabs) > 1:
        return (
            merge_rows(allitems),
            "확정",
            "나뉘어 있는 표를 합치면 배점이 100점입니다.",
        )
    # 여기까지 못 맞췄으면 항목이 가장 많은 표를 대표로 골라 느슨한 기준으로 판정한다.
    cands, total = max(tabs, key=lambda t: len(t[0][0][1]))
    items = cands[0][1]
    s = sum(i["score"] for i in items)
    # totals = 문서의 다른 표들에 적힌 합계 목록 (예: 별도의 "총계" 표).
    totals = [t for _, t in tabs if t is not None]
    if (
        len(items) >= 5 and s >= MIN_FULL_SUM and any(abs(s - t) < 0.01 for t in totals)
    ):  # 다른 표(총계 표)의 합계와 같음
        return (
            merge_rows(items),
            "확정",
            f"항목 배점 합 {s:g}점이 문서에 적힌 합계 {s:g}점과 같습니다.",
        )
    if (
        total is not None
    ):  # 표에 합계가 있는데 다르면(항목 누락 가능) 추정으로 올리지 않음
        return (
            merge_rows(items),
            "부분",
            f"항목 배점 합 {s:g}점이 표에 적힌 합계 {total:g}점과 다릅니다 (빠진 항목이 있을 수 있음).",
        )
    if len(items) >= 5 and any(abs(s - w) < 0.01 for w in (90, 80, 70)):
        return (
            merge_rows(items),
            "추정",
            f"항목 배점 합이 {s:g}점입니다 (표에 합계 줄이 없어 100점 여부는 확인 필요).",
        )
    return (
        merge_rows(items),
        "부분",
        f"항목 배점 합이 {s:g}점으로, 표의 합계와 맞지 않습니다.",
    )


# =====================================================================
# 2. 문서 찾기
# =====================================================================
# 검색·비교용 정규화: 전각/반각 통일(NFKC) + 소문자.
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()


# 질문에서 두 글자 이상 단어를 뽑는다 (STOP 단어는 제외).
def _tokens(msg: str) -> list[str]:
    return [t for t in re.findall(r"[0-9a-z가-힣]{2,}", _norm(msg)) if t not in STOP]


# 점수 = 질문 단어 중 파일 이름에 들어 있는 단어들의 글자 수 합. 긴 단어가 맞을수록 점수가 크다.
def _score_name(tokens: list[str], fname: str) -> int:
    f = _norm(fname)
    return sum(len(t) for t in tokens if t in f)


def find_doc(message: str, use_search: bool = True):
    """-> (경로 or None, 후보 이름 목록)

    원리: 사업명을 정확히 안 써도 질문으로 공고문 파일을 찾는다.
      1) 질문 단어와 파일 이름이 겹친 글자 수로 점수를 매기고, 4점 이상이면서 1등이 유일하면 그 파일.
      2) 아니면 검색(baseline.search)으로 1위 조각의 기관명·사업명을 받아 다시 파일 이름과 비교 (6점 이상, 1등 유일).
         use_search=False 이면 이 단계를 건너뛴다 (서식 요청은 엉뚱한 공고를 고르면 안 되므로 검색을 쓰지 않는다).
      3) 그래도 못 찾으면 점수가 있는 후보 이름 최대 3개를 돌려줘서 사용자에게 되묻게 한다.
    """
    files = sorted(CORPUS_DIR.glob("*.txt"))
    toks = _tokens(message)
    ranked = sorted(
        ((_score_name(toks, p.name), p) for p in files), key=lambda x: -x[0]
    )
    if (
        ranked
        and ranked[0][0] >= 4
        and (len(ranked) < 2 or ranked[0][0] > ranked[1][0])
    ):
        return ranked[0][1], []
    try:  # 이름이 정확하지 않으면 검색으로 한 번 더
        if not use_search:
            raise RuntimeError("검색 사용 안 함")
        from ai.rahui.exp_filter_rerank.rahui_engine import baseline

        hits = baseline.search(message, 1)
        if hits:
            md = hits[0][0].metadata
            toks2 = _tokens(f"{md.get('발주 기관', '')} {md.get('사업명', '')}")
            ranked2 = sorted(
                ((_score_name(toks2, p.name), p) for p in files), key=lambda x: -x[0]
            )
            if (
                ranked2
                and ranked2[0][0] >= 6
                and (len(ranked2) < 2 or ranked2[0][0] > ranked2[1][0])
            ):
                return ranked2[0][1], []
    except Exception:
        pass
    return None, [p.stem for s, p in ranked[:3] if s > 0]


# =====================================================================
# 3. 모델로 항목별 안내 쓰기
# =====================================================================
# 공고문 평가기준 문장을 "- ", "·", 문장 끝 등으로 쪼개 최대 3개, 각 90자까지 짧은 불릿으로 만든다 (모델 없이 쓰는 규칙 문장용).
def _split_criteria(text: str) -> list[str]:
    parts = [
        p.strip(" -·•*\t")
        for p in re.split(r"\s-\s|[·•]|(?<=[다음함])\.\s|\n", text)
        if p.strip(" -·•*\t")
    ]
    return [p[:90] for p in parts[:3]]


# 모델을 못 쓰거나 모델 출력이 형식에 안 맞을 때 쓰는 규칙 기반 안내문. 평가기준 문장을 쪼개 "꼭 넣을 내용"으로 쓴다.
# 예시·팁·실수는 모델이 쓸 때만 생기므로 비워 둔다.
def template_guide(item: dict) -> dict:
    must = _split_criteria(item.get("criteria", "")) or _split_criteria(
        item.get("elements", "")
    )
    return {
        "where": f"'{item['name']}'에 대해 발주기관이 알고 싶어 하는 내용을 보여 주는 부분입니다.",
        "must": must
        or ["공고문의 평가기준을 확인한 뒤 해당 내용을 빠짐없이 적으세요."],
        "example": "",
        "tip": "",
        "mistake": "",
    }


# 모델이 프롬프트의 형식 안내문("(가상의 예시 문장 한두 문장)" 등)을 그대로 베낄 때 지우기 위한 패턴.
_PLACEHOLDER = re.compile(
    r"\(\s*(?:가상의 예시 문장|이 항목에서|한 줄|한 문장|이 항목이)[^)]*\)\s*"
)


# 모델 출력 정리: 안내 라벨 줄을 버리고, 베낀 안내 괄호를 지우고, 문장 전체를 감싼 괄호를 벗긴다.
def _clean_text(t: str) -> str:
    """모델이 형식 안내문을 베끼거나 문장 전체를 괄호로 감싼 경우 정리"""
    parts = [
        p.strip()
        for p in t.strip().split("\n")
        if p.strip() and not _LABEL_HINT.match(p.strip())
    ]
    t = _PLACEHOLDER.sub("", " ".join(parts)).strip()
    if t.startswith("(") and t.endswith(")") and t.count("(") == 1:
        t = t[1:-1].strip()
    return t


# [예시]는 "우리 회사는"으로 시작하는 문장만 인정한다. 설명문이나 안내문이 들어오면 비워서 "예시 없음"으로 처리 (가짜 예시를 막는 장치).
def _clean_example(t: str) -> str:
    """[예시]는 '우리 회사는'으로 시작하는 문장만 인정 (안내문·설명문은 버림)"""
    t = _clean_text(t)
    return t if "우리 회사" in t[:12] else ""


# 프롬프트의 few-shot 예(다른 주제)를 모델이 그대로 복사했는지 알아보는 표식. 복사했으면 버린다.
_FEWSHOT_WHERE = "행사 운영 계획 항목은"
# 모델이 "첫째 내용:" 같은 형식 문구를 그대로 쓴 경우 앞부분을 지운다.
_LEAD = re.compile(r"^\s*(?:첫째|둘째|셋째)\s*내용\s*[:：]?\s*")
# 프롬프트의 라벨 안내 문구가 통째로 한 줄로 남은 경우를 걸러내는 패턴.
_LABEL_HINT = re.compile(r"^(?:점수를 더 받는 요령 한 문장|자주 하는 실수 한 문장)\s*$")


# "꼭 넣을 내용" 불릿 정리: 형식 문구를 지우고, 팁/흔한 실수/쓰는 칸 라벨이 섞인 줄은 버린다.
def _clean_must(lines):
    out = []
    for m in lines:
        m = _clean_text(_LEAD.sub("", m))
        if m and not re.match(r"(?:팁|흔한 실수|쓰는 칸)\s*[:：]", m):
            out.append(m)
    return out


# [예시] 후처리: 사업·시스템 이름이 예시 문장에 그대로 들어가면 "본 시스템" 같은 일반 말로 바꾼다.
# 이유: [예시]는 어느 회사나 따라 쓸 수 있는 가상 문장이어야 하는데, 모델이 사업명(예: 벤처확인종합관리시스템)을 베껴 넣는 경우가 있다.
# 원리: 사업명 안의 낱말 중 '시스템/플랫폼/사업'으로 끝나는 5글자 이상 낱말만 이름으로 보고 바꾼다 (짧은 일반어 '기존 시스템'은 건드리지 않는다).
_NAME_END = {"시스템": "본 시스템", "플랫폼": "본 플랫폼", "사업": "본 사업"}


def mask_names(text: str, project: str) -> str:
    """예시 문장 속 사업·시스템 고유 이름을 일반 말로 바꾼다 (이름이 없으면 그대로)."""
    if not text or not project:
        return text
    # 발주기관 이름: 파일 이름은 "기관명_사업명" 꼴이라 첫 "_" 앞이 기관이다. 4글자 이상 낱말을 "발주기관"으로 바꾼다
    # (예: 국립민속박물관, 한국철도공사). 모델이 기관명을 베끼는 경우가 99개 예시 중 2개 있었다.
    if "_" in project:
        for tok in re.findall(r"[가-힣A-Za-z]{4,}", project.split("_")[0]):
            text = text.replace(tok, "발주기관")
            for pre in (
                "한국",
                "국립",
                "대한",
                "서울",
            ):  # "한국발명진흥회" → "발명진흥회" 같은 줄임말도
                if tok.startswith(pre) and len(tok) - len(pre) >= 4:
                    text = text.replace(tok[len(pre) :], "발주기관")
        text = re.sub(
            r"\((?:사|재|주|유)\)\s*발주기관", "발주기관", text
        )  # "(사)벤처기업협회" -> "발주기관"
    for tok in re.findall(r"[가-힣A-Za-z0-9]+", project):
        for end, plain in _NAME_END.items():
            if len(tok) >= 5 and tok.endswith(end):
                text = text.replace(tok, plain)
                # 모델이 이름을 줄여 쓴 경우(벤처확인종합관리시스템 -> 벤처확인시스템): 앞 3글자가 같은 낱말도 이름으로 본다.
                text = re.sub(re.escape(tok[:3]) + r"[가-힣]*" + end, plain, text)
    return text


def parse_guide_json(text: str):
    """모델 출력(JSON)을 안내문 dict 로 바꾼다. 못 읽거나 필수(where, must)가 비면 None.

    원리: 작은 모델은 라벨 형식("[예시]:" 등)을 자주 빼먹어서, 키(where/must/example/tip/mistake)가 있는 JSON 으로 받는다.
    JSON 앞뒤에 잡설이 붙어도 가장 바깥 { } 만 잘라 읽는다. few-shot 문장을 베낀 출력은 거부한다.
    """
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    where = _clean_text(str(d.get("where", "")))
    must = d.get("must")
    if isinstance(must, str):
        must = [must]
    must = _clean_must([str(x) for x in (must or [])])
    if (
        not where
        or not must
        or where.startswith(_FEWSHOT_WHERE)
        or where in _FEWSHOT_VALUES
    ):
        return None
    if re.search(r"항목 이름을 넣어|무엇을 보여 주는 부분인지", where):
        return None
    ex = _clean_example(str(d.get("example", "")))
    if ex.startswith("우리 회사는 행사 준비부터") or ex in _FEWSHOT_VALUES:
        ex = ""
    return {
        "where": where,
        "must": must[:4],
        "example": ex,
        "tip": _clean_text(str(d.get("tip", ""))),
        "mistake": _clean_text(str(d.get("mistake", ""))),
    }


# JSON 으로 읽어 보고, 안 되면 라벨 방식으로 다시 읽는다 (둘 다 안 되면 None -> 규칙 문장으로 대체).
def parse_guide(text: str):
    g = parse_guide_json(text)
    if g:
        return g
    return parse_guide_labels(text)


# 옛 라벨 방식 출력("쓰는 칸:", "꼭 넣을 내용", "[예시]", "팁:", "흔한 실수:")을 읽는 대비용 파서.
def parse_guide_labels(text: str):
    where = re.search(r"쓰는 칸\s*[:：]\s*(.+)", text)
    ex = re.search(
        r"\[예시\]\s*[:：]?\s*(.+?)(?=\n\s*(?:팁|흔한 실수)\s*[:：]|\Z)",
        text,
        flags=re.S,
    )
    tip = re.search(
        r"팁\s*[:：]\s*(.+?)(?=\n\s*흔한 실수\s*[:：]|\Z)", text, flags=re.S
    )
    mis = re.search(r"흔한 실수\s*[:：]\s*(.+)", text, flags=re.S)
    block = re.search(
        r"꼭 넣을 내용\s*[:：]?\s*\n?(.*?)(?=\[예시\]|\n\s*[-•*]?\s*(?:팁|흔한 실수)\s*[:：]|\Z)",
        text,
        flags=re.S,
    )
    must = [
        ln.strip(" -·•*\t")
        for ln in (block.group(1).split("\n") if block else [])
        if ln.strip(" -·•*\t")
    ]
    must = _clean_must(must)
    if not (where and must):
        return None
    where_txt = _clean_text(where.group(1))
    if re.search(
        r"항목 이름을 넣어|무엇을 보여 주는 부분인지|한 문장으로?$", where_txt
    ):
        where_txt = ""  # 안내문을 베낀 경우 -> 호출 쪽에서 규칙 문장으로 대체
    if not where_txt:
        return None
    return {
        "where": where_txt,
        "must": [_clean_text(m) for m in must[:4]],
        "example": _clean_example(ex.group(1) if ex else ""),
        "tip": _clean_text(tip.group(1) if tip else ""),
        "mistake": _clean_text(mis.group(1) if mis else ""),
    }


# ---- 재무·신용·가격 항목: 고정 문장 ----
# 경영상태·신용평가등급·입찰가격 같은 항목은 글솜씨가 아니라 제출 서류나 산식으로 점수가 정해진다.
# 모델에게 맡기면 "신용평가 등급을 산정하여 제출하겠습니다"처럼 어색한 조언이 나와서 고정 문장을 쓴다.
_QUANT_FINANCE = re.compile(
    r"경영\s*상태|재무|신용\s*평가|신용\s*도|재정\s*(?:상태|건전)"
)
_QUANT_PRICE = re.compile(r"입찰\s*가격|가격\s*평가|가격\s*점수")


def _plain_name(name: str, kind: str = "finance") -> str:
    """항목 이름에서 글머리·번호·끝의 (배점)을 떼어 낸다. 너무 길면(문장이면) 일반 이름으로 바꾼다."""
    n = re.sub(r"^[\s\-–·ㆍ•∘◦○ㅇ․*]+", "", name or "")
    n = re.sub(r"^(?:\d+|[가-하A-Za-z])\s*[.)]\s*", "", n)
    n = re.sub(r"\s*\(\s*(?:\d+(?:\.\d+)?\s*점?|\*|계량)\s*\)\s*$", "", n).strip()
    if not n or len(n) > 18:
        return "입찰 가격" if kind == "price" else "재무·신용"
    return n


def quant_kind(item: dict):
    """'finance'(재무·신용), 'price'(가격) 또는 None. 항목 이름만 본다."""
    name = item.get("name") or ""
    if _QUANT_PRICE.search(name):
        return "price"
    if _QUANT_FINANCE.search(name):
        return "finance"
    return None


def quant_guide(item: dict):
    kind = quant_kind(item)
    if kind is None:
        return None
    name = _plain_name(item.get("name"), kind)
    if kind == "price":
        return {
            "where": f"{name} 항목은 입찰 가격에 따라 점수가 정해지는 부분입니다.",
            "must": [
                "공고문의 가격 평가 산식과 기준을 확인합니다.",
                "가격은 공고문이 정한 서식과 방법으로 제출합니다.",
                "제안서 본문에 가격을 써도 되는지 공고문에서 확인합니다.",
            ],
            "example": "",
            "tip": "가격 점수는 산식으로 정해지니, 산식과 제출 방법을 먼저 확인하세요.",
            "mistake": "산식이나 제출 방법을 확인하지 않은 채 가격서를 내는 것입니다.",
        }
    return {
        "where": f"{name} 항목은 회사의 재무 상태나 신용 등급처럼 정해진 기준으로 점수가 정해지는 부분입니다.",
        "must": [
            "공고문이 요구하는 재무·신용 관련 증빙 서류를 확인합니다.",
            "서류의 기준일과 발급 조건이 공고문과 맞는지 확인합니다.",
            "요구된 서식과 자료를 빠짐없이 제출합니다.",
        ],
        "example": "",
        "tip": "이 항목은 글솜씨보다 서류로 점수가 정해지니, 제출 기한과 서류 형식을 먼저 확인하세요.",
        "mistake": "증빙 서류의 기준일이나 발급 조건이 달라 인정받지 못하는 것입니다.",
    }


# ---- few-shot: 항목마다 돌아가며 다른 예를 보여 줘서 문장 틀이 한 가지로 굳지 않게 한다 ----
FEWSHOTS = [
    (
        "행사 운영 계획",
        {
            "where": "행사 운영 계획 항목은 행사를 어떻게 준비하고 진행할지 보여 주는 부분입니다.",
            "must": [
                "준비 일정과 담당 인력을 설명합니다.",
                "현장에서 문제가 생겼을 때의 대응 방법을 제시합니다.",
                "행사 후 결과를 점검하는 방법을 보여 줍니다.",
            ],
            "example": "우리 회사는 행사 준비부터 마무리까지 담당자를 정해 단계별로 점검하겠습니다.",
            "tip": "누가, 언제, 무엇을 하는지 나누어 쓰면 읽기 쉽습니다.",
            "mistake": "계획은 길게 쓰면서 문제가 생겼을 때의 대응은 빠뜨리는 것입니다.",
        },
    ),
    (
        "견학 프로그램 운영",
        {
            "where": "견학 프로그램 운영 항목은 방문객이 어떤 순서로 무엇을 체험하는지 보여 주는 부분입니다.",
            "must": [
                "방문 대상과 인원에 맞춘 코스를 설명합니다.",
                "안내 인력의 역할과 배치를 보여 줍니다.",
                "안전을 위한 준비 사항을 제시합니다.",
            ],
            "example": "우리 회사의 견학 코스는 대상별로 나누어 이동 시간과 설명 시간을 함께 계획하겠습니다.",
            "tip": "코스를 시간 순서로 보여 주면 평가하는 사람이 흐름을 쉽게 따라갑니다.",
            "mistake": "안내 인력이 몇 명인지만 적고 어디에서 무엇을 하는지는 밝히지 않는 경우가 많습니다.",
        },
    ),
    (
        "전시 해설 운영",
        {
            "where": "전시 해설 운영 항목은 해설을 누가 어떤 방식으로 제공할지 보여 주는 부분입니다.",
            "must": [
                "해설 방식과 대상별 구성을 설명합니다.",
                "해설 내용을 관리하고 고치는 절차를 제시합니다.",
                "이용자 의견을 받아 반영하는 방법을 보여 줍니다.",
            ],
            "example": "우리 회사는 해설 내용을 대상별로 준비하고 이용자 의견을 받아 정기적으로 고치겠습니다.",
            "tip": "해설을 만드는 과정과 고치는 과정을 나란히 쓰면 운영 방식이 분명해 보입니다.",
            "mistake": "해설 내용 자체만 길게 소개하고 운영과 관리 방법은 비워 두는 일이 흔합니다.",
        },
    ),
    (
        "고객 응대 체계",
        {
            "where": "고객 응대 체계 항목은 문의가 들어왔을 때 어떻게 받고 해결하는지 보여 주는 부분입니다.",
            "must": [
                "문의를 받는 창구와 시간을 설명합니다.",
                "문의 종류별 처리 순서를 제시합니다.",
                "처리 결과를 확인하고 기록하는 방법을 보여 줍니다.",
            ],
            "example": "우리 회사는 문의 접수부터 처리 결과 안내까지 담당자를 정해 기록으로 남기겠습니다.",
            "tip": "문의가 들어온 뒤 해결될 때까지의 순서를 한 줄씩 나열하면 이해하기 쉽습니다.",
            "mistake": "빠르게 응대하겠다는 다짐만 있고 실제 처리 순서가 보이지 않는 것이 문제가 됩니다.",
        },
    ),
]
_FEWSHOT_VALUES = {
    v for _t, g in FEWSHOTS for k, v in g.items() if isinstance(v, str)
} | {m for _t, g in FEWSHOTS for m in g["must"]}


def fewshot_text(k: int) -> str:
    topic, g = FEWSHOTS[k % len(FEWSHOTS)]
    return (
        f'\n\n아래는 형식을 보여 주는 다른 항목("{topic}")의 예입니다. '
        "내용과 문장 틀을 베끼지 말고, 주어진 평가 항목에 맞게 새로 쓰세요.\n\n"
        + json.dumps(g, ensure_ascii=False, indent=1)
    )


# ---- 결과를 약속하는 표현 거르기 ----
# "점수를 높일 수 있습니다", "좋은 점수를 받습니다" 같은 보장 투 문장은 컨설팅 문서에 맞지 않는다. 걸리면 그 칸을 비운다(팁·실수·예시)
# 또는 그 줄을 뺀다(꼭 넣을 내용). 프롬프트로 막고도 새어 나오는 것을 코드로 한 번 더 거른다.
PROMISE = re.compile(
    r"점수(?:를|가|도)?\s*(?:더\s*)?(?:높|올|받|얻|확보|잘)|높은\s*(?:점수|등급|평가)|좋은\s*(?:점수|평가|결과)|만점|고득점|"
    r"유리(?:하|합|해)|합격|선정(?:될|에)\s*(?:수|유리)|경쟁력을?\s*높"
)


# 회사가 쓰는 [예시] 문장이 과장되지 않도록, 단정·과장 낱말을 코드에서 뺀다 (예: "모든 위험에 대비", "철저히 점검").
ABSOLUTE = re.compile(
    r"(?:모든\s+|철저(?:히|하게)\s*|철저한\s+|완벽(?:히|하게)\s*|완벽한\s+|최고의\s+|최상의\s+|확실(?:히|하게)\s*|무조건\s+)"
)


def soften(text: str) -> str:
    return re.sub(r"\s{2,}", " ", ABSOLUTE.sub("", text or "")).strip()


def has_promise(text: str) -> bool:
    return bool(PROMISE.search(text or ""))


def drop_promises(g: dict) -> dict:
    for fld in ("example", "tip", "mistake"):
        if has_promise(g.get(fld, "")):
            g[fld] = ""
    g["must"] = [m for m in g.get("must", []) if not has_promise(m)]
    g["example"] = soften(g.get("example", ""))
    g["must"] = [soften(m) for m in g["must"]]
    # 작은 모델이 '단순히 …'로 시작하는 틀을 거의 매번 쓴다. 프롬프트로 안 막히니 코드에서 앞의 '단순히 '만 뗀다.
    g["mistake"] = re.sub(r"^\s*단순히\s+", "", g.get("mistake", ""))
    return g


# 모델에게 맡길 항목: 앞쪽 MAX_MODEL_ITEMS 개 중 고정 문장 항목(재무·신용·가격)을 뺀 것. (순번, 항목) 목록.
def _model_todo(items: list[dict]) -> list:
    return [
        (k, i) for k, i in enumerate(items[:MAX_MODEL_ITEMS]) if quant_kind(i) is None
    ]


# ---- 가이드 캐시 (로컬 전용: outputs/guides/_cache, 공고문 내용이 들어 있으니 깃에 올리지 않는다) ----
# 같은 공고문 + 같은 지시문 + 같은 모델이면 모델을 다시 돌리지 않고 저장해 둔 안내문을 쓴다. WRITER_NO_CACHE=1 이면 끈다.
CACHE_DIR = OUT_DIR / "_cache"


def _cache_path(project: str, items: list[dict]) -> Path:
    import hashlib

    sig = json.dumps(
        [
            [i.get("name"), i.get("score"), i.get("criteria"), i.get("elements")]
            for i in items
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    h = hashlib.sha1(
        f"{GUIDE_PROMPT}|{GUIDE_VERSION}|{config.CHAT_MODEL}|{sig}".encode()
    ).hexdigest()[:12]
    safe = re.sub(r'[\\/:*?"<>|]', "_", project)[:50]
    return CACHE_DIR / f"{safe}__{h}.json"


def write_guides(project: str, items: list[dict], use_model: bool):
    """캐시가 있으면 그것을, 없으면 모델로 만들어 캐시에 저장한다."""
    cp = (
        _cache_path(project, items)
        if (use_model and USE_MODEL and not os.environ.get("WRITER_NO_CACHE"))
        else None
    )
    if cp and cp.exists():
        try:
            d = json.loads(cp.read_text(encoding="utf-8"))
            return d["guides"], d["note"] + " (저장된 결과 사용)"
        except Exception:
            pass
    guides, note = _write_guides(project, items, use_model)
    if cp and note.startswith("로컬 모델"):
        m = re.search(r"(\d+)/", note)
        want = len(_model_todo(items))
        if m and int(m.group(1)) >= max(
            1, int(want * 0.8)
        ):  # 대부분 성공했을 때만 저장
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_text(
                json.dumps({"guides": guides, "note": note}, ensure_ascii=False),
                encoding="utf-8",
            )
    return guides, note


def _write_guides(project: str, items: list[dict], use_model: bool):
    """항목마다 안내문(dict)을 만든다 -> (안내문 목록, 작성 방식 설명 문자열).

    원리: 먼저 모든 항목을 규칙 문장(template_guide)으로 채워 두고, 모델이 쓴 결과가 형식에 맞는 항목만 덮어쓴다.
    모델 호출이 실패하거나 출력이 형식에 안 맞으면 그 항목은 규칙 문장이 남으므로 초안은 항상 완성된다.
    """
    # 1) 전부 규칙 문장으로 먼저 채운다 (안전망).
    guides = [template_guide(i) for i in items]
    for k, i in enumerate(
        items
    ):  # 재무·신용·가격 항목은 고정 문장 (모델을 쓰지 않는다)
        qg = quant_guide(i)
        if qg:
            guides[k] = qg
    n_fixed = sum(1 for i in items if quant_kind(i))
    if not (use_model and USE_MODEL):
        return guides, "모델 없이 규칙만으로 작성"
    try:
        from langchain.messages import HumanMessage, SystemMessage

        from ai.rahui.exp_filter_rerank.rahui_engine import baseline

        model = baseline.make_chat_model()
        system = baseline.load_prompt(GUIDE_PROMPT)
    except Exception as e:  # 모델을 못 불러도 초안은 만든다
        return guides, f"모델을 쓰지 못해 규칙만으로 작성 ({type(e).__name__})"
    # 2) 항목마다 모델에게 (사업명, 평가부문, 항목, 배점, 평가기준, 평가요소)를 주고 JSON 을 받는다.
    #    항목이 많으면 앞쪽 MAX_MODEL_ITEMS 개만 모델로 쓴다 (시간 제한).
    done = 0

    def one(k, item):
        """항목 하나를 모델로 써서 (k, 안내문 dict 또는 None) 을 돌려준다."""
        crit = (item.get("criteria") or "")[:500] or "(적혀 있지 않음)"
        elem = (item.get("elements") or "")[:300]
        score = f"{item['score']:g}점" if item.get("score") is not None else "(미표기)"
        msg = (
            f"사업: {project}\n평가부문: {item.get('group') or '-'}\n평가항목: {item['name']}\n"
            f"배점: {score}\n평가기준(공고문 원문): {crit}\n평가요소: {elem or '-'}"
        )
        try:
            r = model.invoke(
                [SystemMessage(system + fewshot_text(k)), HumanMessage(msg)]
            )
            raw = getattr(r, "text", None) or r.content
            g = parse_guide(raw)
            # WRITER_DEBUG=1 이면 모델 원본 출력을 파일에 남겨서 파싱 실패 원인을 볼 수 있게 한다.
            if os.environ.get("WRITER_DEBUG"):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                with (
                    _DEBUG_LOCK,
                    open(
                        OUT_DIR / "_raw_model_output.txt", "a", encoding="utf-8"
                    ) as fh,
                ):
                    fh.write(
                        f"=== [{k + 1}] {item['name']} (파싱 {'성공' if g else '실패'})\n{raw}\n\n"
                    )
        except Exception:
            g = None
        return k, g

    # 항목끼리는 서로 영향이 없으므로 WRITER_PARALLEL(기본 1) 개씩 동시에 요청할 수 있다.
    # 1이면 예전처럼 하나씩 차례로 쓴다. 2 이상은 Ollama 서버도 OLLAMA_NUM_PARALLEL 을 같은 값으로 켜야 실제로 빨라진다.
    todo = _model_todo(items)
    if PARALLEL > 1 and len(todo) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            results = list(
                ex.map(lambda kv: one(*kv), todo)
            )  # map 은 입력 순서대로 결과를 돌려준다
    else:
        results = [one(k, item) for k, item in todo]
    # 파싱에 성공한 항목만 규칙 문장을 모델 문장으로 바꾼다.
    for k, g in results:
        if g:
            for fld in (
                "where",
                "example",
                "tip",
                "mistake",
            ):  # 사업·시스템·기관 이름 일반화
                g[fld] = mask_names(g.get(fld, ""), project)
            g["must"] = [mask_names(b, project) for b in g.get("must", [])]
            g = drop_promises(g)
            if not g[
                "must"
            ]:  # 보장 문장을 빼고 나니 남은 줄이 없으면 규칙 문장을 그대로 둔다
                continue
            guides[k] = g
            done += 1
    return (
        guides,
        f"로컬 모델 {config.CHAT_MODEL}로 {done}/{len(items)}개 항목의 안내 작성"
        + (f" (재무·가격 항목 {n_fixed}개는 고정 문장)" if n_fixed else ""),
    )


# =====================================================================
# 4. .docx 만들기
# =====================================================================
# 근거 수준별 문서 표시: (제목 라벨, 설명). 문서 맨 위 노란 상자와 답변 문장에 쓰인다.
# 근거가 약할수록 사용자가 공고문을 다시 확인하도록 안내한다.
LEVEL_TEXT = {
    "확정": (
        "RFP 근거",
        "공고문의 평가표를 읽어 만든 초안입니다. 배점 합계도 확인했습니다.",
    ),
    "추정": (
        "RFP 근거 (일부 확인 필요)",
        "공고문의 평가표를 바탕으로 만든 초안입니다. 정정공고나 변경공고로 배점이 바뀌었을 수 있으니, 원본의 공고문도 함께 확인하세요.",
    ),
    "부분": (
        "부분 근거",
        "공고문의 평가표를 바탕으로 만든 초안입니다. 정정공고나 변경공고로 배점이 바뀌었을 수 있으니, 원본의 공고문도 함께 확인하세요.",
    ),
    "없음": (
        "일반 안내",
        "이 공고문에서 평가표를 찾지 못해 일반적인 제안서 구조로 작성했습니다. 평가 항목은 원본의 공고문과 제안서 작성 지침으로 확인하세요.",
    ),
}


# 근거 수준 라벨·설명. 평가표를 읽었는데(확정·추정·부분) 모든 항목의 평가기준 설명이 비어 있으면
# "RFP 근거"라고만 쓰면 오해하므로, 항목 이름·배점만 읽었다고 따로 알린다.
def level_text(level: str, items: list) -> tuple:
    label, desc = LEVEL_TEXT[level]
    if (
        level in ("확정", "추정", "부분")
        and items
        and all(not item_ref(it) for it in items)
    ):
        return (
            "평가표 항목만 읽음 (평가기준 설명 없음)",
            "공고문의 평가표에서 항목 이름과 배점은 읽었지만, 항목별 평가기준 설명은 찾지 못했습니다. 아래 내용은 일반적인 안내이니 공고문 원문의 평가표를 꼭 확인하세요.",
        )
    return label, desc


# 항목의 공고문 근거 문장. 머리글 낱말("평가기준" 등)만 들어 있거나 비어 있으면 근거 없음("").
def item_ref(item: dict) -> str:
    ref = (item.get("criteria") or item.get("elements") or "").strip()
    return (
        ""
        if sq(ref) in ("", "-", "평가기준", "평가요소", "평가항목", "평가내용", "배점")
        else ref
    )


def build_docx(project, level, note, items, guides, model_note, src_name, path: Path):
    """안내문을 .docx 로 저장한다.

    구성: 제목 -> 근거 수준 상자(노랑) -> 항목마다 [제목 + 항목 표(쓰는 칸/꼭 넣을 내용/[예시]/팁/흔한 실수/공고문 평가기준)]
          -> 끝에 "작성 가이드라인 안내"(내부 근거·모델 정보는 넣지 않음). 고객사에 그대로 전달할 수 있게 표 색으로 칸 종류를 구분한다.
    """
    try:
        from docx import Document
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt, RGBColor
    except ImportError as e:
        raise RuntimeError(
            "python-docx 가 필요합니다: uv pip install python-docx"
        ) from e

    # python-docx 도우미: fmt=글자 모양, shade=칸 배경색, para=문단 추가.
    FONT = "맑은 고딕"
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(1.8)

    def fmt(run, size=10.5, bold=False, color=None, italic=False):
        run.font.name = FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        if color:
            run.font.color.rgb = RGBColor.from_string(color)

    def shade(cell, fill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        tcPr.append(shd)

    def para(text="", size=10.5, bold=False, color=None, after=4, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(after)
        if text:
            fmt(p.add_run(text), size, bold, color, italic)
        return p

    para("제안서 작성 가이드 (초안)", 20, True, "1F2937", after=2)
    para(project, 11, False, "6B7280", after=8)

    label, desc = level_text(level, items)
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.rows[0].cells[0]
    shade(c, "FEF9C3")
    c.paragraphs[0].paragraph_format.space_after = Pt(2)
    fmt(c.paragraphs[0].add_run(f"근거 수준: {label}"), 10.5, True)
    p2 = c.add_paragraph()
    fmt(p2.add_run(desc), 10)
    p3 = c.add_paragraph()
    fmt(
        p3.add_run(
            "[예시] 표시는 이해를 돕기 위한 가상 내용입니다. 그대로 제출하지 말고 우리 회사 내용으로 바꿔 쓰세요."
        ),
        10,
        False,
        "92400E",
    )
    para("", after=4)

    # 항목마다: 제목(번호·이름·배점) -> 평가부문 -> 2열 표 (왼쪽 라벨, 오른쪽 내용).
    # 비어 있는 예시·팁·실수 행은 만들지 않는다.
    for k, (item, g) in enumerate(zip(items, guides, strict=False), 1):
        title = f"{k}. {item['name']}"
        if item.get("score") is not None:
            title += f"   (배점 {item['score']:g}점)"
        h = para(title, 13.5, True, "111827", after=1)
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.keep_with_next = True
        if item.get("group"):
            para(f"평가부문: {item['group']}", 9.5, False, "6B7280", after=3)
        rows = [
            ("쓰는 칸", g["where"]),
            ("꼭 넣을 내용", "\n".join(f"• {m}" for m in g["must"])),
        ]
        if g.get("example"):
            rows.append(("[예시]", g["example"]))
        if g.get("tip"):
            rows.append(("팁", g["tip"]))
        if g.get("mistake"):
            rows.append(("흔한 실수", g["mistake"]))
        ref = item_ref(item)
        if ref:
            rows.append(
                (
                    "공고문 평가기준",
                    ref if len(ref) <= 260 else ref[:260].rstrip() + " …",
                )
            )
        else:  # 근거가 없는 항목은 그 사실을 알린다 (일반 안내를 공고 근거로 착각하지 않게)
            rows.append(
                (
                    "공고문 평가기준",
                    "공고문에서 이 항목의 평가기준을 찾지 못했어요. 위 내용은 일반적인 안내이니, 공고문 원문의 평가표를 꼭 확인하세요.",
                )
            )
        tb = doc.add_table(rows=len(rows), cols=2)
        tb.style = "Table Grid"
        tb.autofit = False
        tb.columns[0].width, tb.columns[1].width = Cm(3.0), Cm(14.0)
        for r_i, (lab, body) in enumerate(rows):
            a, b = tb.rows[r_i].cells
            a.width, b.width = Cm(3.0), Cm(14.0)
            ex = lab == "[예시]"
            shade(
                a,
                "FFEDD5"
                if ex
                else "DCFCE7"
                if lab == "팁"
                else "FEE2E2"
                if lab == "흔한 실수"
                else "DBEAFE",
            )
            if ex:
                shade(b, "F8FAFC")
            lc = (
                "C2410C"
                if ex
                else "15803D"
                if lab == "팁"
                else "B91C1C"
                if lab == "흔한 실수"
                else "1D4ED8"
            )
            fmt(a.paragraphs[0].add_run(lab), 10, True, lc)
            fmt(
                b.paragraphs[0].add_run(body),
                10,
                False,
                ("B45309" if body.startswith("공고문에서 이 항목의") else "334155")
                if lab == "공고문 평가기준"
                else None,
                italic=ex,
            )

    para("", after=6)
    # 사용자에게는 근거 문서 이름·검증 문구·모델 이름 같은 내부 정보를 보이지 않는다(당황스러울 수 있음). 필요한 안내만 남긴다.
    para("작성 가이드라인 안내", 12, True, "111827", after=2)
    para(
        "• 이 문서는 작성 가이드라인입니다. 제출은 공고문 원본의 양식을 이용해 주세요.",
        10,
        after=1,
    )
    if level in ("부분", "없음", "추정"):
        para(
            "• 확인 필요: 위 근거 수준 안내를 참고해 공고문 원문에서 평가 항목과 배점을 확인하세요.",
            10,
            False,
            "B45309",
            after=1,
        )
    n_miss = sum(1 for it in items if not item_ref(it))
    if n_miss:
        para(
            f"• 확인 필요: 평가기준을 찾지 못한 항목 {n_miss}개는 공고문 근거 없이 일반적인 안내로 썼어요. 공고문 평가표를 꼭 확인하세요.",
            10,
            False,
            "B45309",
            after=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


# 저장 경로: outputs/guides/<사업명>_작성가이드_<월일_시분초>.docx (파일 이름에 못 쓰는 글자는 _ 로 바꿈).
def out_path(project: str) -> Path:
    safe = re.sub(r'[\\/:*?"<>|]', "_", project)[:60]
    return OUT_DIR / (safe + "_작성가이드_" + time.strftime("%m%d_%H%M%S") + ".docx")


def make_guide(doc_path: Path, use_model=True):
    """문서 하나로 가이드 .docx 를 만들고 요약 정보를 돌려준다 (그래프 없이도 호출 가능)

    랭그래프의 find_doc -> extract -> (fallback) -> write -> build 를 한 함수로 묶은 것. 평가표를 못 찾으면 GENERIC 항목을 쓴다.
    """
    txt = doc_path.read_text(encoding="utf-8", errors="replace")
    project = doc_path.stem
    items, level, note = extract_items(txt, doc_path.name)
    if not items:
        items = [
            {"group": "", "name": n, "score": None, "criteria": c, "elements": ""}
            for n, c in GENERIC
        ]
    guides, model_note = write_guides(project, items, use_model)
    out = out_path(project)
    build_docx(project, level, note, items, guides, model_note, doc_path.name, out)
    return {
        "file": out,
        "level": level,
        "note": note,
        "n_items": len(items),
        "project": project,
    }


# =====================================================================
# 4-2. 서식(양식) 예시: 서식 구간을 찾아 빈칸을 채우고, 값마다 출처를 표시
# =====================================================================
# 서식 제목 줄을 알아보는 패턴: "[별지 제3호 서식] 일반현황", "<별지서식 1>", "서식 2호" 등을 한 정규식으로 잡는다.
# 그룹1=호수(예: 3, 1-1), 그룹2=같은 줄에 붙은 제목.
FORM_TITLE = re.compile(
    r"^\s*[\[\(<【]?\s*(?:별\s*지(?:\s*서\s*식)?|서\s*식|양\s*식)\s*제?\s*(\d+(?:\s*-\s*\d+)?)\s*호?\s*(?:서\s*식)?\s*[\]\)>】]?\s*(.*)$"
)
# 위 규칙으로 서식을 하나도 못 찾은 공고에만 쓰는 느슨한 규칙: [붙임 1], 【첨부서식-1호】, [별지 서식 - 제1호], [서식 #1], [첨부1], [별첨 2] 등.
# 일부러 마지막에만 쓴다(붙임·첨부·별첨은 서식이 아닌 자료에도 붙어서, 잘 읽던 공고에 섞이면 안 되므로). 표가 없는 구간은 어차피 버려진다.
FORM_TITLE_LOOSE = re.compile(
    r"^\s*[\[\(<【]?\s*(?:별\s*지(?:\s*서\s*식)?|서\s*식|양\s*식|붙\s*임(?:\s*서\s*식)?|첨\s*부(?:\s*서\s*식)?|별\s*첨)\s*[-–#]?\s*제?\s*(\d+(?:\s*-\s*\d+)?(?:\s*의\s*\d+)?)\s*호?\s*(?:서\s*식)?\s*[\]\)>】]?\s*(.*)$"
)
# 한 서식의 최대 줄 수 (다음 서식 제목이 없을 때 문서 끝까지 먹지 않도록 자른다).
MAX_FORM_LINES = 120

# [문서] 값 규칙: 표의 빈칸 라벨에 이 말이 있으면 공고문(data_list.csv)의 사업 정보를 넣는다.  (라벨 키워드, meta 이름)
# 단, 격자 표(머리글이 있는 표)에서는 이 규칙을 끈다: 유사사업 실적표의 "사업명" 열은 이 공고가 아니기 때문.
DOC_RULES = [  # (라벨 키워드, 문서 값 이름)
    (("사업명", "과업명", "용역명", "사업의명칭"), "사업명"),
    (("발주기관", "발주처", "수요기관", "공고기관", "발주부서"), "발주기관"),
    (("공고번호", "입찰공고번호"), "공고번호"),
    (("사업금액", "사업예산", "추정가격", "배정예산"), "사업금액"),
]
# [예시] 값 규칙: 라벨에 이 말이 있으면 이해를 돕는 가상 값을 넣는다.  (라벨 키워드, 가상 값)
# 위에 있는 규칙이 먼저 적용된다(예: "휴대전화"가 "전화"보다 앞). 모든 값에 [예시]를 붙여 가상임을 숨기지 않는다.
FAKE_RULES = [  # (라벨 키워드, 가상 값)
    (("사업자등록번호", "사업자번호", "등록번호"), "000-00-00000"),
    (("법인등록번호",), "000000-0000000"),
    (("대표자", "성명", "담당자", "책임자", "작성자", "신청인"), "홍길동"),
    (
        (
            "회사명",
            "기관명",
            "업체명",
            "상호",
            "법인명",
            "기업명",
            "제안사",
            "제안업체",
            "참가업체",
        ),
        "(주)○○컨설팅",
    ),
    (("직위", "직급"), "팀장"),
    (("부서", "소속"), "사업개발팀"),
    (("휴대전화", "휴대폰", "핸드폰"), "010-0000-0000"),
    (("FAX", "팩스"), "02-000-0001"),
    (("전화", "TEL", "연락처"), "02-000-0000"),
    (("E-mail", "이메일", "메일"), "sample@example.com"),
    (("주소", "소재지"), "서울특별시 ○○구 ○○로 00"),
    (("설립", "창립"), "2010년 1월"),
    (("종업원", "근로자", "인원", "투입"), "00명"),
    (("도입년도", "도입연도", "수행년도", "기준년도", "사업년도"), "2020년"),
    (("자본금", "매출", "금액", "가격", "합계", "계약", "실적", "수주"), "000,000"),
    (("일자", "날짜", "기간", "연월"), "2024년 00월 00일"),
    (("구분",), "본사"),
]
# 서식 이름을 고를 때 버리는 말 ("보여줘", "양식" 등). 가이드용 STOP 과 따로 둔 이유: "표지"는 서식 이름이라 버리면 안 된다.
FORM_STOP = {
    "보여줘",
    "보여",
    "양식",
    "서식",
    "별지",
    "예시",
    "작성",
    "있어",
    "주세요",
    "알려줘",
    "해줘",
    "보여주세요",
    "도",
}
GRID_KEEP = 5  # 격자 표는 머리글 + 이 줄 수까지만 보여 주고 나머지는 생략 표시
GRID_ALL = 25  # 표 줄이 이 수 이하인 서식은 생략하지 않고 전부 보여 준다 (줄마다 내용이 달라 생략하면 정보가 빠짐)
FAKE_LABEL_MAX = 20  # 가상 값 규칙을 적용할 라벨의 최대 길이
SKIP_MARK = "줄 생략)"
# 단위만 있는 칸("명", "원", "(천원)" 등)을 알아보는 패턴. 이런 칸은 라벨이 아니다.
UNIT_ONLY = re.compile(r"^[\s년월일명원%건개점천만억,()~～\-]*$")


# 공고 정보(사업명, 발주기관, 공고번호, 사업금액)를 data_list.csv 에서 파일 이름으로 찾아 가져온다.
# 못 찾으면 파일 이름("기관_사업명")을 쪼개서 대신 쓴다. [문서] 값의 유일한 출처다.
def load_meta(doc_path: Path) -> dict:
    """data_list.csv 에서 이 문서의 사업 정보를 가져옴 (없으면 파일 이름으로 대신)"""
    meta = {}
    stem = unicodedata.normalize("NFC", doc_path.stem)
    org, _, title = stem.partition("_")
    meta["발주기관"], meta["사업명"] = org.strip(), (title or stem).strip()
    try:
        import csv

        with open(
            ROOT / "data" / "data_list.csv", encoding="utf-8-sig", newline=""
        ) as f:
            for row in csv.DictReader(f):
                if (
                    unicodedata.normalize("NFC", Path(row.get("파일명", "")).stem)
                    == stem
                ):
                    meta["사업명"] = row.get("사업명") or meta["사업명"]
                    meta["발주기관"] = row.get("발주 기관") or meta["발주기관"]
                    meta["공고번호"] = (row.get("공고 번호") or "").split(".")[0]
                    amt = row.get("사업 금액") or ""
                    if amt:
                        with contextlib.suppress(ValueError):
                            meta["사업금액"] = f"{int(float(amt)):,}원"
                    break
    except Exception:
        pass
    return {k: v for k, v in meta.items() if v}


# 서식 제목 정리: 마크다운(#)·HTML 태그·"[표]"·글자 사이 공백을 지우고, 제목이 아닌 줄(■ 지침 이름, 즉시, <신설…>)이면 빈 문자열.
def _clean_form_title(t: str) -> str:
    t = re.sub(r"<!--.*?-->|</?u>", "", t)
    t = re.sub(r"^[#\s]+", "", t).strip()
    t = re.sub(r"\s*\[/?표\]\s*", " ", t).strip()
    t = re.split(r"\s+[*※]|\s{3,}", t)[0].strip()
    if (
        t and all(len(x) == 1 for x in t.split()) and len(t.split()) >= 3
    ):  # "입 찰 참 가 신 청 서" -> "입찰참가신청서"
        t = "".join(t.split())
    if (
        re.match(
            r"^(?:즉\s*시$|[<■□◦▶●○❍※\[\-【]|의\s*\d+\]|입찰\s*건명|\[\s*\]|\d+\.\s)", t
        )
        or len(t) < 2
        or len(t) > 40
    ):
        return ""
    if re.search(r"별지\s*제?\s*\d+\s*호|지침|시행규칙", t):
        return ""
    return t


# 로마 숫자 번호 서식: "[Ⅰ-1] 입찰참가신청서" (한국건강가정진흥원). 앞의 규칙들로 하나도 못 찾았을 때만 쓴다.
FORM_TITLE_ROMAN = re.compile(
    r"^\s*[\[【]\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*-\s*\d+)\s*[\]】]\s*(.*)$"
)


def find_forms(lines: list[str]) -> list[dict]:
    """엄격한 제목 규칙으로 찾고, 하나도 없으면 느슨한 규칙(붙임·첨부·별첨)으로 한 번 더 찾는다."""
    strict = _find_forms(lines, FORM_TITLE)
    if strict:
        return strict
    # 느슨한 규칙은 서식이 아닌 자료(목차·목록·산정서·번호 칸 등)도 걸리므로 그런 제목은 뺀다.
    loose = [
        f
        for f in _find_forms(lines, FORM_TITLE_LOOSE)
        if not re.search(
            r"목차|목록|산정서|평가표|정보보호관리|^(?:사업명|순번|입찰공고번호|제안관련\s*서식)$|^\s*$|\(제목 없음\)",
            f["title"],
        )
    ]
    return loose or _find_forms(lines, FORM_TITLE_ROMAN)


def _find_forms(lines: list[str], title_re) -> list[dict]:
    """문서 줄 목록에서 '(별지 제N호 서식) 제목' 구간들을 찾음 (줄 번호는 1부터)

    원리: 서식 제목 줄(FORM_TITLE)을 찾고, 다음 제목 줄 직전까지(또는 최대 MAX_FORM_LINES 줄)를 그 서식의 구간으로 본다.
    표(|) 줄이 하나도 없는 구간은 서식이 아닌 것으로 버린다. 제목은 같은 줄 -> 다음 줄 -> 본문 첫 문장 앞부분 순으로 찾는다.
    같은 호수가 여러 번 나오면 표가 많은 쪽(본문)을 쓴다 (목차에 나온 줄을 버리기 위해).
    """
    heads = []
    for i, line in enumerate(lines):
        # 제목 줄은 짧다: 90자 넘는 줄은 서식 제목이 아니라 문장 속 언급이므로 건너뛴다.
        if len(line) > 90 or not line.strip():
            continue
        m = title_re.match(line)
        if not m:
            continue
        rest = m.group(2).strip()
        # "...를 작성하여야 한다." 같은 문장 끝이거나 너무 길면 제목이 아니라 문장이다.
        if re.search(r"(다|함|음)\.?$", rest) or len(rest) > 60:
            continue  # 문장 속 언급이지 서식 제목이 아님
        title = rest
        if not title:  # 제목이 다음 줄에 있는 경우
            for nxt in lines[i + 1 : i + 4]:
                t = nxt.strip()
                if t and not t.startswith("[") and "|" not in t and len(t) <= 50:
                    title = t
                    break
            if (
                not title
            ):  # 본문 첫 문장 앞부분이 제목인 경우 (예: "보안 서약서(...) 본인은 ...")
                for nxt in lines[i + 1 : i + 4]:
                    t = re.sub(r"^\[표\]\s*", "", nxt.strip())
                    if len(t) > 50 and "|" not in t[:40] and not t.startswith("["):
                        cut = re.search(
                            r"\s(?:본인|본\s*업체|귀\s|아래|다음|위\s|이\s)|\s{2,}",
                            t[4:],
                        )
                        title = (
                            t[: 4 + cut.start()]
                            if cut and cut.start() <= 40
                            else t[:25]
                        ).strip()
                        break
        # 제목 정리: 표 칸 구분자(|)가 섞였으면 마지막 칸만 쓰고, 앞의 "."와 뒤의 쪽번호 숫자를 지운다.
        if "|" in title:
            title = ([x.strip() for x in title.split("|") if x.strip()] or [""])[-1]
        title = re.sub(r"\s+\d{1,3}$", "", title.lstrip(". ").strip())
        title = _clean_form_title(title)
        if (
            not title
        ):  # 제목 후보가 이상하면 뒤쪽 몇 줄에서 제대로 된 제목을 다시 찾는다
            for nxt in lines[i + 1 : i + 7]:
                c = _clean_form_title(
                    nxt.strip().split("|")[0]
                    if "|" in nxt and nxt.strip().startswith("|") is False
                    else nxt.strip()
                )
                if c and "|" not in c:
                    title = c
                    break
        heads.append(
            {
                "no": re.sub(r"\s+", "", m.group(1)),
                "title": title or "(제목 없음)",
                "start": i,
            }
        )
    forms = []
    for k, h in enumerate(heads):
        end = heads[k + 1]["start"] if k + 1 < len(heads) else len(lines)
        end = min(end, h["start"] + MAX_FORM_LINES)
        body = lines[h["start"] : end]
        n_rows = sum(1 for ln in body if "|" in ln)
        forms.append({**h, "end": end, "n_rows": n_rows})
    # 표 줄이 없는 구간(목차 줄 등)은 제외.
    forms = [f for f in forms if f["n_rows"] >= 1]
    # 같은 호수가 여러 번 나오면 표가 많은 쪽만 남긴다.
    by_no: dict[str, dict] = {}
    for f in forms:  # 같은 호수가 여러 번 나오면 표가 많은 쪽을 씀
        if f["no"] not in by_no or f["n_rows"] > by_no[f["no"]]["n_rows"]:
            by_no[f["no"]] = f
    return sorted(by_no.values(), key=lambda f: f["start"])


# 서식을 못 골랐을 때 "그런 서식은 이 공고에 없다"고 말할 수 있는 경우를 찾는다 -> 없는 서식 이름 또는 None.
# 말할 수 있는 경우: ① "N번"이 목록 개수를 넘음 ② "제N호"가 없음 ③ 서식 종류 낱말(표지·위임장·확약서 등)이 어느 서식 제목에도 없음.
# 사업명만 말한 일반 요청("서식 보여줘")은 해당하지 않아 목록만 보여 준다 (없다고 잘못 말하지 않기 위해).
_FORM_KIND = (
    "표지",
    "위임장",
    "확약서",
    "서약서",
    "신청서",
    "각서",
    "동의서",
    "조직도",
    "이력서",
    "확인서",
    "청렴",
    "보안",
    "개인정보",
)


def missing_form_word(forms: list[dict], message: str):
    m = re.search(r"(\d+)\s*번", message)
    if m and int(m.group(1)) > len(forms):
        return f"{m.group(1)}번"
    m = re.search(r"제\s*(\d+(?:-\d+)?)\s*호", message)
    if m and all(f["no"] != m.group(1) for f in forms):
        return f"제{m.group(1)}호"
    for k in _FORM_KIND:
        if k in message and not any(k in sq(f["title"]) for f in forms):
            return k
    return None


def pick_form(forms: list[dict], message: str):
    """질문에서 어떤 서식인지 고름 -> (서식 or None)

    우선순위: "3번"(목록 순서) -> "제3호"(호수) -> 제목과 겹치는 단어(2글자 이상 겹쳐야 인정). 못 고르면 None -> 목록을 보여 줌.
    """
    m = re.search(r"(\d+(?:-\d+)?)\s*번", message)
    if m and 1 <= int(m.group(1).split("-")[0]) <= len(forms) and "-" not in m.group(1):
        return forms[int(m.group(1)) - 1]
    m = re.search(
        r"([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*-\s*\d+)", message
    )  # 로마 숫자 번호: "Ⅰ-1", "제Ⅰ-1호"
    if m:
        for f in forms:
            if sq(f["no"]) == sq(m.group(1)):
                return f
    m = re.search(r"제?\s*(\d+(?:-\d+)?)\s*호", message)
    if m:
        for f in forms:
            if f["no"] == m.group(1):
                return f
    toks = []
    for t in re.findall(r"[0-9a-z가-힣]{2,}", _norm(message)):
        if t not in FORM_STOP:
            toks.append(t)
            # 조사를 뗀 단어도 같이 비교한다 (예: "표지도" -> "표지").
            if (
                len(t) >= 3 and t[-1] in "도는은을를만이가"
            ):  # 조사 떼고도 비교 (예: 표지도 -> 표지)
                toks.append(t[:-1])
    best, best_s = None, 0
    for f in forms:
        title = sq(f["title"])
        sc = sum(len(t) for t in toks if t in title)
        if sc > best_s:
            best, best_s = f, sc
    return best if best_s >= 2 else None


# 라벨 정리: 앞의 번호("1)", "(2)")를 지우고 공백을 없앤다.
def _clean_label(c: str) -> str:
    c = re.sub(r"^\s*[\(\[]?\d+[\)\]\.]\s*", "", c)
    return sq(c)


def _lookup(label: str, meta: dict):
    """라벨 하나에 넣을 값을 찾는다 -> (값, "문서"|"예시") 또는 None.

    순서: DOC_RULES(공고문 값) -> FAKE_RULES(가상 값). 20자 넘는 긴 칸은 문장이지 라벨이 아니라서 가상 값을 넣지 않는다
    (체크리스트 문장에 "사업개발팀"이 들어가던 오류를 막는 장치).
    """
    for keys, name in DOC_RULES:
        if any(k in label for k in keys) and name in meta:
            return meta[name], "문서"
    if len(label) > FAKE_LABEL_MAX:  # 체크리스트 문장 같은 긴 칸은 라벨이 아님
        return None
    for keys, val in FAKE_RULES:
        if any(k.lower() in label.lower() for k in keys):
            return val, "예시"
    return None


def fill_form(lines: list[str], meta: dict):
    """서식 구간의 표 줄을 읽어 빈칸을 채운다 -> (출력 줄 목록, 원문 주석 줄 목록, 개수 집계).

    원리: 줄을 | 로 나눈 칸마다
      - 내용이 있으면 그대로 둔다 (원문 라벨/안내).
      - 비어 있으면 왼쪽에서 가장 가까운 라벨 -> 같은 열의 머리글 -> 더 왼쪽 라벨 순으로 규칙(_lookup)을 찾아
        "값 [문서]" 또는 "값 [예시]"를 넣고, 규칙이 없으면 "[직접 작성: 라벨]"로 남긴다.
    격자 표(머리글 줄이 있는 표)는 머리글 + 앞 GRID_KEEP 줄만 보여 주고 나머지는 "N줄 생략" 한 줄로 줄인다.
    """
    out, notes = [], []
    count = {"문서": 0, "예시": 0, "작성": 0}
    header: list[str] = []
    tbl_rows, skipped, skip_idx = 0, 0, None
    show_all = sum(1 for x in lines if "|" in x) <= GRID_ALL  # 작은 표는 생략하지 않음
    for raw in lines:
        line = raw.strip()
        if not line or line in ("[표]", "[/표]"):
            if line == "[/표]":
                header = []
                tbl_rows, skipped, skip_idx = 0, 0, None
            continue
        # 표가 아닌 줄: "※", "주)", "*" 등으로 시작하면 원문 주석으로 모아서 답변 끝에 보여 준다. 나머지는 버린다.
        if "|" not in line:
            # "(단위 : 천원)" 같은 단위 표기 줄은 금액 예시(000,000)를 읽는 데 필요하므로 본문에 그대로 남긴다.
            if len(line) <= 40 and re.match(r"^[(\[]?\s*단위\s*[:：]", line):
                out.append(line)
                continue
            if line.startswith(("※", "주)", "주 )", "*", "○ 주", "※")) or re.match(
                r"^\d+\.\s", line
            ):
                notes.append(line[:200])
            continue
        if line.startswith("|"):
            line = line.strip().strip("|")
            if re.fullmatch(r"[\s|:\-]*", line):
                continue
        cells = [c.strip() for c in line.split("|")]
        # 머리글 인식: 표의 첫 줄이 3칸 이상 차 있고 중간에 빈 칸이 없으면 격자 표의 머리글로 본다. [/표] 에서 초기화.
        if (
            not header
            and sum(1 for c in cells if c) >= 3
            and not any(not c for c in cells[1:])
        ):
            header = [_clean_label(c) for c in cells]
        # 격자 표 반복 줄 요약: 머리글 + GRID_KEEP 줄까지만 채우고, 나머지는 한 줄로 합친다 (같은 [직접 작성] 안내가 수백 번 반복되는 것을 막음).
        # 요약 줄의 숫자는 루브릭 채점(check_form_answer)이 구조 일치를 계산할 때 쓴다.
        if header:
            tbl_rows += 1
            if (
                tbl_rows > GRID_KEEP + 1 and not show_all
            ):  # 큰 표에서 같은 형식의 반복 줄은 생략
                skipped += 1
                if skip_idx is None:
                    out.append("")
                    skip_idx = len(out) - 1
                out[skip_idx] = f"• … (같은 형식이 반복되는 {skipped}{SKIP_MARK}"
                continue
        filled = []
        prev_fill = None
        for i, c in enumerate(cells):
            # "라벨 | 명" 처럼 단위만 있는 칸은 앞 칸의 라벨로 값을 만든다 (예: 상시근로자수 | 명 -> 00명 [예시]).
            if (
                c
                and sq(c) in ("년월", "명", "원", "천원", "건")
                and i > 0
                and cells[i - 1]
            ):
                hit = _lookup(
                    _clean_label(cells[i - 1]), meta
                )  # 예: 상시근로자수 | 명  ->  00명
                if hit and hit[1] == "예시":
                    filled.append(f"{hit[0]} [예시]")
                    count["예시"] += 1
                    continue
            if c:
                filled.append(c)
                continue
            if i == 0:
                filled.append("")
                continue
            # (아래 규칙 찾기) 왼쪽의 숫자/단위뿐인 칸은 라벨이 아니므로 제외한다.
            # 왼쪽에서 가장 가까운 라벨 -> 같은 칸의 머리글 -> 더 왼쪽 라벨 순으로 규칙을 찾음
            lefts = [
                _clean_label(x)
                for x in cells[:i]
                if x and not UNIT_ONLY.match(x) and not re.fullmatch(r"\d+", sq(x))
            ]
            cand = (
                ([lefts[-1]] if lefts else [])
                + ([header[i]] if i < len(header) and header[i] else [])
                + lefts[:-1][::-1]
            )
            # 바로 왼쪽 칸이 라벨이면(예: "업 종 |  ") 그 라벨이 기준이다. 더 먼 왼쪽 라벨(사업자번호)의 값을 대신 넣지 않는다.
            if (
                i > 0
                and cells[i - 1]
                and not UNIT_ONLY.match(cells[i - 1])
                and not re.fullmatch(r"\d+", sq(cells[i - 1]))
            ):
                cand = (
                    cand[:1] + cand[1:2]
                    if (i < len(header) and header[i])
                    else cand[:1]
                )
            hit = None
            for lab in cand:
                hit = _lookup(
                    lab, {} if header else meta
                )  # 격자 표(실적표 등)의 '사업명'은 이 공고가 아님
                if hit:
                    break
            if hit:
                # 같은 라벨의 같은 값이 연속 칸에 반복되면 두 번째부터 비운다 (예: 병합된 칸). 000 으로 시작하는 숫자 예시는 칸마다 채운다.
                same = (
                    bool(filled)
                    and prev_fill is not None
                    and prev_fill == (lab, hit[0])
                )
                if same and not hit[0].startswith(
                    "000"
                ):  # 같은 라벨의 값을 여러 칸에 반복해 채우지 않음
                    filled.append("")
                else:
                    filled.append(f"{hit[0]} [{hit[1]}]")
                    count[hit[1]] += 1
                    prev_fill = (lab, hit[0])
            elif lefts or (i < len(header) and header[i]):
                # 규칙이 없는 칸: 격자 표는 머리글 이름으로, 일반 표는 왼쪽 라벨로 "[직접 작성: 이름]"을 남긴다.
                ph = (
                    header[i] if (i < len(header) and header[i]) else lefts[-1]
                )  # 격자 표는 머리글 이름으로 안내
                if prev_fill == ("작성", ph) and not (i < len(header) and header[i]):
                    filled.append("")
                else:
                    filled.append(f"[직접 작성: {ph}]")
                    count["작성"] += 1
                    prev_fill = ("작성", ph)
            else:
                filled.append("")
        out.append(
            "• " + " | ".join(filled).rstrip(" |")
        )  # 끝의 빈 칸("| |")은 지움 (예: E-mail 줄)
    return out, notes, count


# ---------------------------------------------------------------------
# 서식 예시 .docx: "워드로", "docx", "파일", "다운로드" 같은 말이 있을 때만 만든다 (질문마다 파일이 쌓이지 않게).
# 구성: 제목 -> 안내 상자(가상 값 경고) -> 서식 본문(\" | \" 로 이어진 줄은 표로) -> 값 표시 개수·출처.
# 값 표시는 색으로 구분한다: [문서] 초록 / [예시] 주황 / [직접 작성: …] 파랑.
# ---------------------------------------------------------------------
FORM_FILE_WORDS = ("docx", "워드", "파일", "다운로드", "내려받", "문서로")
_MARK_COLOR = re.compile(r"(\[문서\]|\[예시\]|\[직접 작성[^\]]*\])")


def wants_form_file(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in FORM_FILE_WORDS)


def form_out_path(project: str, title: str) -> Path:
    safe = re.sub(r'[\\/:*?"<>|]', "_", f"{project}_{title}")[:70]
    return OUT_DIR / (safe + "_작성예시_" + time.strftime("%m%d_%H%M%S") + ".docx")


def build_form_docx(
    project: str, title: str, body: list[str], footer: list[str], path: Path
):
    """서식 예시를 .docx 로 저장한다. body=서식 본문 줄들, footer=값 표시 개수·출처·경고 줄들."""
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt, RGBColor
    except ImportError as e:
        raise RuntimeError(
            "python-docx 가 필요합니다: uv pip install python-docx"
        ) from e
    FONT = "맑은 고딕"
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(1.8)

    def fmt(run, size=10.5, bold=False, color=None):
        run.font.name = FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        run.font.size = Pt(size)
        run.bold = bold
        if color:
            run.font.color.rgb = RGBColor.from_string(color)

    def write_marked(par, text, size=10.5):
        """[문서]/[예시]/[직접 작성] 표시만 색을 입혀서 쓴다."""
        for piece in _MARK_COLOR.split(text):
            if not piece:
                continue
            color = (
                "047857"
                if piece == "[문서]"
                else "B45309"
                if piece == "[예시]"
                else "1D4ED8"
                if piece.startswith("[직접 작성")
                else None
            )
            fmt(par.add_run(piece), size, bold=bool(color), color=color)

    h = doc.add_paragraph()
    fmt(h.add_run(f"{title} 작성 예시 (참고용)"), 16, True)
    sub = doc.add_paragraph()
    fmt(sub.add_run(project), 10, False, "6B7280")
    note = doc.add_paragraph()
    fmt(
        note.add_run(
            "[예시] 값은 이해를 돕기 위한 가상 내용입니다. 제출 전에 우리 회사 정보로 바꿔 쓰세요. "
            "[문서]는 공고문에서 가져온 값, [직접 작성]은 회사 정보를 넣어야 하는 칸입니다."
        ),
        10,
        False,
        "92400E",
    )
    note2 = doc.add_paragraph()
    fmt(
        note2.add_run(
            "이 문서는 칸 구성과 쓰는 요령을 보여 주는 참고용 작성 예시이며, 공식 서식과 똑같지 않습니다. "
            "실제 제출은 공고문에 첨부된 원본 서식으로 하세요."
        ),
        10,
        True,
        "92400E",
    )

    grid = []  # 연속된 " | " 줄을 모아 표로 만든다.

    def flush():
        if not grid:
            return
        width = max(len(r) for r in grid)
        tb = doc.add_table(rows=0, cols=width)
        tb.style = "Table Grid"
        for r in grid:
            cells = tb.add_row().cells
            for j in range(width):
                write_marked(cells[j].paragraphs[0], r[j] if j < len(r) else "", 9.5)
        doc.add_paragraph()
        grid.clear()

    for ln in body:
        raw = ln.lstrip("•").strip()
        if " | " in raw:
            grid.append([c.strip() for c in raw.split(" | ")])
            continue
        flush()
        if raw:
            write_marked(doc.add_paragraph(), raw)
    flush()
    for ln in footer:
        if ln.strip():
            write_marked(doc.add_paragraph(), ln, 9.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


# 서식 번호 표기: 숫자면 "별지 제N호", 로마 숫자 등(Ⅰ-1)이면 원문 표기 그대로 "[Ⅰ-1]".
def form_label(no: str) -> str:
    return f"별지 제{no}호" if no[:1].isdigit() else no


def form_example(doc_path: Path, message: str) -> str:
    """서식 예시 답변 문자열을 만든다 (서식을 못 고르면 서식 목록을 보여 준다)

    답변 끝에는 값 표시 개수([문서]/[예시]/[직접 작성]), 출처 줄 번호, 가상 값 경고를 붙인다.
    서식 목록 답변에는 "문서: 파일명" 줄을 넣어서, 다음 턴(번호만 말해도)에 어느 공고인지 이어지게 한다.
    """
    txt = doc_path.read_text(encoding="utf-8", errors="replace")
    lines = txt.split("\n")
    meta = load_meta(doc_path)
    project = meta.get("사업명", doc_path.stem)
    forms = find_forms(lines)
    if not forms:
        return (
            f"{project} 문서에서 서식(별지) 구간을 찾지 못했어요. "
            f"제출 서식은 공고문 원문의 '제안서 작성 지침'이나 '제출서류' 부분을 확인해 주세요."
        )
    f = pick_form(forms, message)
    if f is None:
        # 특정 서식을 콕 집어 물었는데(예: "위임장", "7번", "제30호") 이 공고에 없으면, 없다고 먼저 알리고 있는 서식 목록을 보여 준다.
        miss = missing_form_word(forms, message)
        head = f"요청하신 '{miss}' 서식은 이 공고에 없어요.\n" if miss else ""
        listing = "\n".join(
            f"{k}. [{form_label(x['no'])}] {x['title']}"
            for k, x in enumerate(forms[:20], 1)
        )
        more = f"\n(외 {len(forms) - 20}개)" if len(forms) > 20 else ""
        return (
            f"{head}{project}에는 서식이 {len(forms)}개 있어요. 어떤 서식의 작성 예시를 볼까요?\n"
            f"{listing}{more}\n번호나 이름으로 알려 주세요. (예: 2번, 일반현황) Word 파일로 받으려면 '2번 워드로'처럼 말해 주세요.\n문서: {doc_path.name}"
        )
    body, notes, count = fill_form(lines[f["start"] : f["end"]], meta)
    parts = [
        f"{project}의 [{form_label(f['no'])}] {f['title']} 작성 예시예요.",
        "참고용 작성 예시예요. 공식 서식과 똑같지는 않으니, 제출은 공고문에 첨부된 원본 서식으로 해 주세요.",
        "",
    ]
    parts += body or ["(이 서식에서 채울 빈칸을 찾지 못했어요. 원문을 확인해 주세요.)"]
    if notes:
        parts += ["", "원문 주석"] + [f"- {n}" for n in notes[:6]]
    footer = [
        f"값 표시: [문서] 공고문에서 가져온 값 {count['문서']}개 · [예시] 이해를 돕는 가상 값 {count['예시']}개 · "
        f"[직접 작성] 회사 정보를 넣어야 하는 칸 {count['작성']}개",
        f"출처: {doc_path.name}, 원문 {f['start'] + 1}~{f['end']}줄",
        "※ [예시] 값은 가상이에요. 제출 전에 우리 회사 정보로 바꿔 쓰세요. 제출은 원본 서식을 이용하세요.",
    ]
    parts += [""] + footer
    # "워드로/파일로" 요청이면 .docx 도 만들고, 답변에 "파일: 경로" 줄을 붙인다 (화면이 이 줄을 보고 다운로드 카드를 보여 준다).
    if wants_form_file(message) and body:
        try:
            path = form_out_path(project, f["title"])
            # 문서 맨 위 안내문에 같은 설명이 있으니, 아래쪽 범례·출처·경고 줄은 넣지 않는다.
            docx_footer: list[str] = []
            build_form_docx(
                project,
                f"[{form_label(f['no'])}] {f['title']}",
                body,
                docx_footer,
                path,
            )
            parts += [
                "",
                f"파일: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}",
            ]
        except Exception as e:  # 파일 만들기에 실패해도 글 답변은 그대로 준다
            parts += ["", f"(docx 파일은 만들지 못했어요: {type(e).__name__})"]
    return "\n".join(parts)


def doc_from_history(history) -> Path | None:
    """앞 대화에서 마지막으로 다룬 사업 문서를 찾음

    1) 답변 속 "문서:/출처: 파일명" 줄을 최근 것부터 찾는다 (가장 정확).  2) 없으면 대화 내용에서 사업명 단어로 찾되 검색은 쓰지 않는다.
    """
    for turn in reversed(history or []):
        m = re.search(
            r"^(?:문서|출처):\s*(.+?\.txt)\s*(?:,.*)?$",
            turn.get("content", ""),
            flags=re.M,
        )
        if m and (CORPUS_DIR / m.group(1)).exists():
            return CORPUS_DIR / m.group(1)
    for turn in reversed(history or []):
        p, _ = find_doc(turn.get("content", ""), use_search=False)
        if p is not None:
            return p
    return None


# =====================================================================
# 5. 랭그래프
# =====================================================================
# 그래프 입출력: answer()가 question/history 를 넣고, 그래프는 answer 를 반환한다.
#   입력 question/history 는 외부 answer(message, history)의 인자에 대응한다.
#   내부 결과: doc_path/candidates=문서 찾기, items/level/note=평가표 추출,
#   guides/model_note=작성 안내, file=생성한 docx 경로, answer=호출자에게 돌려줄 답변 문자열.
class GuideState(TypedDict, total=False):
    question: str
    history: list
    doc_path: str
    candidates: list
    items: list
    level: str
    note: str
    guides: list
    model_note: str
    file: str
    answer: str


# 질문이 작성 가이드 요청처럼 보이는지 (GUIDE_WORDS).
def is_guide_request(message: str) -> bool:
    return any(w in message for w in GUIDE_WORDS)


# 질문이 서식 예시 요청처럼 보이는지 (FORM_WORDS).
def is_form_request(message: str) -> bool:
    return any(w in message for w in FORM_WORDS)


# 대화 기록에서 마지막 assistant 답변을 가져온다 (후속 질문 판단에 사용).
def _last_assistant(history) -> str:
    for turn in reversed(history or []):
        if turn.get("role") != "user":
            return turn.get("content", "")
    return ""


# 서식 목록 직후의 말이 '서식을 고르는 말'인지 본다: 번호·호·서식 단어가 있거나, 짧고 질문 투가 아니면 선택으로 본다.
def _looks_like_pick(q: str) -> bool:
    if re.search(
        r"\d+\s*번|\d+(?:-\d+)?\s*호|다른|전부|모두|목록", q
    ) or is_form_request(q):
        return True
    return len(q.strip()) <= 20 and not re.search(
        r"\?|알려|얼마|언제|어떻|무엇|뭐|왜|누구|몇|어디|인가|니까|까요", q
    )


# ---- 소요예산 질문에 사업명이 없을 때: 공고 목록을 보여 주고 어느 사업인지 고르게 한다 ----
# 사업명 없이 "소요예산 알려줘"만 물으면 검색이 엉뚱한 공고의 예산을 가져오기 때문이다.
ASK_MARK = "어느 사업의 소요예산"
_BUDGET_RE = re.compile(
    r"소요\s*예산|사업\s*예산|예산|사업\s*비|사업\s*금액|추정\s*(?:가격|금액)"
)
_BUDGET_STOP = re.compile(
    r"알려\s*줄래|알려\s*주세요|알려\s*줘|알려|궁금해요|궁금해|궁금합니다|얼마인가요|얼마예요|얼마야|얼마|"
    r"인가요|이에요|예요|입니까|뭐예요|뭐야|무엇|말해\s*줘|보여\s*줘|해\s*줘|이번|이거|이건|공고|입찰|사업|총|"
    r"좀|그럼|그러면|은|는|이|가|을|를|도|의|에|야|요|줘"
)


def _is_generic_budget(q: str) -> bool:
    """예산을 묻지만 어느 사업인지 말하지 않은 짧은 질문인가 ("소요 예산", "예산이 얼마야?"). 사업명 단어가 있으면 False."""
    if not _BUDGET_RE.search(q):
        return False
    rest = _BUDGET_STOP.sub("", _BUDGET_RE.sub("", q))
    return len(re.sub(r"[^0-9a-z가-힣]", "", _norm(rest))) <= 1


def _budget_files() -> list:
    return sorted(CORPUS_DIR.glob("*.txt"))


_BUDGET_NAMES = None


def _budget_names() -> dict:
    """공고 파일 이름(NFC, 확장자 뺀 것) -> (발주기관, 사업명). data_list.csv 의 전체 사업명을 쓴다 (파일 이름은 길이 제한으로 잘려 있음)."""
    global _BUDGET_NAMES
    if _BUDGET_NAMES is None:
        names = {}
        try:
            import csv

            with open(
                ROOT / "data" / "data_list.csv", encoding="utf-8-sig", newline=""
            ) as f:
                for row in csv.DictReader(f):
                    stem = unicodedata.normalize(
                        "NFC", Path(row.get("파일명", "")).stem
                    )
                    if stem and (row.get("사업명") or row.get("발주 기관")):
                        names[stem] = (
                            (row.get("발주 기관") or "").strip(),
                            (row.get("사업명") or "").strip(),
                        )
        except Exception:
            pass
        _BUDGET_NAMES = names
    return _BUDGET_NAMES


def _budget_parts(p: Path) -> tuple:
    stem = unicodedata.normalize("NFC", p.stem)
    org, title = _budget_names().get(stem, ("", ""))
    if not org or not title:
        o, _, t = stem.partition("_")
        org, title = org or o.strip(), title or (t or stem).strip()
    return org, title


def _budget_label(p: Path) -> str:
    org, title = _budget_parts(p)
    return f"{org} · {title}"


def _last_user(history) -> str:
    for turn in reversed(history or []):
        if turn.get("role") == "user":
            return turn.get("content", "")
    return ""


def _budget_doc(state, q: str):
    """공고 목록 직후의 대답에서 고른 공고 파일을 찾는다 (번호 / '이 공고' / 사업명). 못 찾으면 None."""
    files = _budget_files()
    m = re.fullmatch(
        r"\s*(\d{1,3})\s*(?:번째|번)?\s*(?:이요|이에요|요|으로|으로요|줘|주세요)?\s*", q
    )
    if m:
        n = int(m.group(1))
        return files[n - 1] if 1 <= n <= len(files) else None
    if re.search(r"이\s*공고|직전|방금|위\s*공고|이\s*사업", q):
        return doc_from_history((state.get("history") or [])[:-1])
    p, _ = find_doc(q, use_search=False)
    if p is not None:
        return p
    toks = _tokens(
        q
    )  # 목록을 보고 고르는 말이라 "평택시"처럼 짧은 이름도 받는다 (3점 이상, 1등이 유일할 때)
    ranked = sorted(
        ((_score_name(toks, f.name), f) for f in files), key=lambda x: -x[0]
    )
    if (
        ranked
        and ranked[0][0] >= 3
        and (len(ranked) < 2 or ranked[0][0] > ranked[1][0])
    ):
        return ranked[0][1]
    return None


def _budget_cands(q: str) -> list:
    """목록 직후에 쓴 이름과 가장 많이 겹치는 공고들 (3점 이상, 같은 점수끼리 최대 10개). 없으면 빈 목록."""
    toks = _tokens(q)
    ranked = sorted(
        ((_score_name(toks, f.name), f) for f in _budget_files()), key=lambda x: -x[0]
    )
    if not ranked or ranked[0][0] < 3:
        return []
    return [f for sc, f in ranked if sc == ranked[0][0]][:10]


def n_budget_ask(state: GuideState):
    """소요예산 질문에 사업명이 없을 때: 공고 목록을 보여 주고 고르게 한다.
    목록 직후에 쓴 이름이 여러 공고에 걸리면(예: 평택시 2곳) 그 공고들만 원래 번호로 다시 보여 준다."""
    files = _budget_files()
    if ASK_MARK in _last_assistant(state.get("history")):
        cands = _budget_cands(state.get("question", ""))
        if len(cands) >= 2:
            listing = "\n".join(
                f"{files.index(p) + 1}. {_budget_label(p)}" for p in cands
            )
            return {
                "answer": f"{ASK_MARK}인지 아직 하나로 정해지지 않았어요. 아래에서 번호로 골라 주세요.\n{listing}"
            }

    def cell(t: str, n: int) -> str:
        t = t.replace("|", "/").strip()
        return t if len(t) <= n else t[:n] + "…"

    rows = "\n".join(
        f"| {k} | {cell(_budget_parts(p)[0], 18)} | {cell(_budget_parts(p)[1], 42)} |"
        for k, p in enumerate(files, 1)
    )
    return {
        "answer": (
            f"{ASK_MARK}인지 알려 주세요. 기관명이나 사업명 일부(예: 평택시), 또는 번호(예: 3번)로 말씀해 주세요.\n\n"
            f"<details><summary>공고 {len(files)}개 목록 보기</summary>\n\n"
            f"| 번호 | 기관 | 사업명 |\n| --- | --- | --- |\n{rows}\n\n</details>"
        )
    }


def n_budget(state: GuideState):
    """고른 공고의 사업명을 질문 앞에 붙여 일반 질문(검색)으로 보낸다."""
    from ai.rahui.exp_filter_rerank.rahui_engine import baseline

    p = _budget_doc(state, state["question"])
    q = f"{_budget_label(p)} {_last_user(state.get('history'))}"
    ans = baseline.answer(q, None)
    if not (ans or "").strip():
        ans = "답변을 만들지 못했어요. 사업명을 넣어서 다시 물어봐 주세요."
    return {"answer": ans}


# 서식 목록 직후의 짧은 말("소요 예산" 등)이 실제로 목록의 서식과 맞는지 본다. 번호·호·서식 단어가 있으면 바로 서식 선택이고,
# 그런 단어 없이 짧기만 한 말은 서식 제목과 겹칠 때만 선택으로 본다 (안 맞으면 일반 질문으로 보낸다).
def _pick_ok(state, q: str) -> bool:
    if re.search(
        r"\d+\s*번|\d+(?:-\d+)?\s*호|다른|전부|모두|목록", q
    ) or is_form_request(q):
        return True
    try:
        doc = doc_from_history(state.get("history"))
        if doc is None:
            return True
        forms = find_forms(
            doc.read_text(encoding="utf-8", errors="replace").split("\n")
        )
        return pick_form(forms, q) is not None
    except Exception:
        return True


# ---- 인사·감사·도움말: 모델 없이 정해진 문장으로 바로 답한다 (공고 검색으로 흘러가지 않게) ----
HELP_TEXT = (
    "입찰 공고문(RFP)을 바탕으로 이런 일을 도와드려요.\n"
    '1. 공고문 질문: 예산, 일정, 자격 요건 등을 물어보세요. (예: "A 사업 예산이 얼마야?")\n'
    '2. 제안서 작성 가이드: 평가항목별로 쓸 내용을 정리해 드려요. (예: "A 사업 작성 가이드 만들어줘")\n'
    '3. 서식 작성 예시: 별지 서식의 빈칸 작성법을 보여 드려요. (예: "A 사업 서식 예시 보여줘")\n'
    "궁금한 공고 이름과 함께 말씀해 주세요."
)
_GREET = re.compile(
    r"^(?:안녕(?:하세요|하십니까)?|하이|헬로|hello|hi|hey|반가워(?:요)?|반갑습니다|좋은\s*(?:아침|하루)(?:이에요|입니다|예요)?)"
    r"(?:\s*(?:클로드|챗봇|봇))?[\s!.~?ㅎㅋ,]*$",
    re.I,
)
_THANKS = re.compile(
    r"^(?:고마워(?:요)?|고맙습니다|감사(?:해요|합니다|해)?|땡큐|수고(?:했어(?:요)?|하셨어요|하세요)?|잘\s*했어(?:요)?)[\s!.~?ㅎㅋ,]*$"
)
_HELP = re.compile(
    r"(?:도움말|사용법|사용\s*방법|^help$|뭘\s*(?:할\s*수|해\s*줄\s*수|도와)|뭐\s*(?:할\s*수|해\s*줄\s*수|도와)|"
    r"무엇을\s*(?:할\s*수|도와)|어떤\s*(?:기능|걸\s*할\s*수)|무슨\s*기능|(?:넌|너는|너)\s*누구|누구세요|누구야)"
)
# 입찰과 상관없는 가벼운 말. 입찰 낱말이 섞여 있으면 해당하지 않는다.
_OFFTOPIC = re.compile(
    r"점심|저녁\s*(?:뭐|메뉴)|메뉴\s*추천|날씨|심심|농담|노래\s*(?:추천|불러)|로또"
)
# 뜻 없는 추임새·막연한 말 ("음 글쎄", "몰라"): 모델에 보내면 "컨텍스트를 바탕으로 질문해 주세요" 같은 내부 용어가 나오므로 안내 문장으로 답한다.
_FILLER = re.compile(
    r"^(?:음+|흠+|어+|아+|으음+|글쎄(?:요)?|그냥|몰라(?:요)?|모르겠(?:어|어요|다)|잘\s*모르겠(?:어|어요|다)|뭐지|뭐더라|아무거나|ㅎㅎ+|ㅋㅋ+|[?.~]+)"
    r"(?:[\s,.!?~]*(?:음+|흠+|어+|아+|글쎄(?:요)?|그냥|몰라(?:요)?|모르겠(?:어|어요|다)|뭐지|ㅎㅎ+|ㅋㅋ+))*[\s,.!?~]*$"
)
WHAT_TO_ASK = (
    "어떤 내용이 궁금하신가요? 공고 이름과 함께 이렇게 물어보실 수 있어요.\n"
    '1. 공고 내용: "케빈랩 사업 일정이 어떻게 돼?"\n'
    '2. 제안서 작성 가이드: "케빈랩 작성 가이드 만들어줘"\n'
    '3. 서식 작성 예시: "벤처기업협회 서식 예시 보여줘"\n'
    '처음이시면 "도움말"이라고 입력해 보세요.'
)
_BID_WORDS = re.compile(
    r"공고|사업|입찰|제안|평가|서식|예산|가이드|초안|용역|기관|계약|과업|배점|일정|자격|사업자|rfp",
    re.I,
)


def small_talk(q: str):
    """인사·감사·도움말·입찰과 무관한 가벼운 말이면 정해진 답을, 아니면 None. 입찰 낱말이 섞이면 None."""
    q = (q or "").strip()
    if not q or len(q) > 30:
        return None
    if _HELP.search(q) and not re.search(r"공고|사업명|예산", q):
        return HELP_TEXT
    if _GREET.match(q):
        return (
            "안녕하세요! 입찰 공고문 질문, 제안서 작성 가이드, 서식 작성 예시를 도와드려요.\n"
            '궁금한 공고 이름과 함께 말씀해 주세요. 무엇을 할 수 있는지 궁금하시면 "도움말"이라고 입력해 보세요.'
        )
    if _FILLER.match(q):
        return WHAT_TO_ASK
    if _THANKS.match(q):
        return "도움이 되었다니 다행이에요. 다른 공고나 항목도 궁금하시면 편하게 말씀해 주세요."
    if _OFFTOPIC.search(q) and not _BID_WORDS.search(q):
        return "저는 입찰 공고문 관련 질문(공고 내용, 작성 가이드, 서식 예시)만 도와드릴 수 있어요. 궁금한 공고가 있으면 말씀해 주세요."
    return None


def route(state) -> str:
    """질문을 어느 길로 보낼지 정한다 -> "guide" | "form" | "rag".

    순서가 중요하다:
      1) 직전 답변이 서식 목록이었다면 "3번", "일반현황" 같은 짧은 대답은 서식 선택이다.
      2) 직전 답변이 서식 예시였고 "다른 서식도", "2번" 같은 말이면 서식 후속 질문이다.
      3) "가이드/초안"이 있으면 가이드 (서식 단어보다 우선: "제안서 초안"에 "서식"이 섞여도 가이드).
      4) 서식 단어가 있으면 서식.  5) 가이드 관련 단어가 있으면 가이드.  6) 그 외는 일반 질문(rag).
    """
    q = state["question"]
    if small_talk(q) is not None:  # 인사·도움말은 모델·검색 없이 바로 답한다
        return "chat"
    last = _last_assistant(state.get("history"))
    if (
        ASK_MARK in last
        and "가이드" not in q
        and "초안" not in q
        and not is_form_request(q)
        and _budget_doc(state, q) is not None
    ):  # 소요예산 공고 목록 직후에 고른 대답 (가이드·서식 요청이면 해당 흐름으로)
        return "budget"
    if (
        ASK_MARK in last
        and "가이드" not in q
        and "초안" not in q
        and not is_form_request(q)
        and len(q) <= 30
        and len(_budget_cands(q)) >= 2
    ):  # 목록 직후 이름이 여러 공고에 걸리면 그 공고들만 다시 보여 준다
        return "budget_ask"
    if (
        "작성 예시를 볼까요" in last
        and "가이드" not in q
        and "초안" not in q
        and _looks_like_pick(q)
        and _pick_ok(state, q)
    ):  # 서식 목록 직후의 대답 (예: "3번", "일반현황"). 딴 질문이면 일반 질문으로
        return "form"
    if (
        "작성 예시예요" in last
        and "가이드" not in q
        and "초안" not in q
        and re.search(r"다른|또|다음|\d+\s*번|\d+(?:-\d+)?\s*호", q)
    ):
        return "form"
    if "가이드" in q or "초안" in q:
        return "guide"
    if is_form_request(q):
        return "form"
    if is_guide_request(q):
        return "guide"
    return "budget_ask" if _is_generic_budget(q) else "rag"


# 노드: 질문으로 공고문을 찾는다. 못 찾으면 후보를 보여 주거나 사업명을 되묻는 답변을 만들고 그래프를 끝낸다.
def n_find_doc(state: GuideState):
    """입력: question. 출력: doc_path 또는 사용자에게 되물을 answer (+ candidates)."""
    q = state["question"]
    # 사업명이 없는 요청: 검색이 아무 공고나 고르지 않게, 앞 대화의 사업을 쓰고 없으면 되묻는다
    if not _tokens(q):
        p = doc_from_history(state.get("history"))
        if p is not None:
            return {"doc_path": str(p)}
        return {
            "answer": "어느 사업의 제안서 작성 가이드를 만들까요? 기관명이나 사업명을 알려 주세요."
        }
    p, cands = find_doc(q)
    if p is not None:
        return {"doc_path": str(p)}
    if cands:
        lines = "\n".join(f"- {c}" for c in cands)
        return {
            "answer": f"어느 사업의 제안서인지 정확히 모르겠어요. 아래 중에 있나요?\n{lines}\n사업명을 조금 더 정확히 알려 주세요."
        }
    return {
        "answer": "어느 사업의 제안서 작성 가이드를 만들까요? 기관명이나 사업명을 알려 주세요."
    }


# 노드: 공고문에서 평가표를 읽는다 (items, level, note).
def n_extract(state: GuideState):
    """입력: doc_path. 출력: 공고문에서 추출한 items, 근거 수준 level, 검산 설명 note."""
    txt = Path(state["doc_path"]).read_text(encoding="utf-8", errors="replace")
    items, level, note = extract_items(txt, Path(state["doc_path"]).name)
    return {"items": items, "level": level, "note": note}


# 노드: 평가표를 못 찾았을 때 GENERIC 공통 항목으로 대신한다 (근거 수준 "없음").
def n_fallback(state: GuideState):
    """평가표 항목이 없을 때 내부 items/level 을 기본값으로 채운다 (사용자 응답은 아직 만들지 않음)."""
    items = [
        {"group": "", "name": n, "score": None, "criteria": c, "elements": ""}
        for n, c in GENERIC
    ]
    return {"items": items, "level": "없음"}


# 노드: 항목마다 안내문을 쓴다 (모델 + 규칙 안전망).
def n_write(state: GuideState):
    """입력: doc_path/items. 출력: 항목별 guides 와 모델 또는 규칙 작성 방식 model_note."""
    project = Path(state["doc_path"]).stem
    guides, model_note = write_guides(project, state["items"], True)
    return {"guides": guides, "model_note": model_note}


# 노드: docx 를 저장하고, 화면에 보여 줄 답변 문장(근거 수준, 항목 수, 파일 경로, 주의 문구)을 만든다.
def n_build(state: GuideState):
    """입력: 문서·항목·안내문 상태. 출력: 저장된 docx 경로 file 과 최종 answer 문자열."""
    p = Path(state["doc_path"])
    project = p.stem
    out = out_path(project)
    build_docx(
        project,
        state["level"],
        state.get("note", ""),
        state["items"],
        state["guides"],
        state["model_note"],
        p.name,
        out,
    )
    label = level_text(state["level"], state["items"])[0]
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    answer = (
        f"{project} 제안서 작성 가이드 초안을 만들었어요.\n"
        f"- 근거 수준: {label}\n"
        f"- 항목 {len(state['items'])}개 (항목마다 쓰는 칸, 꼭 넣을 내용"
        + (", 예시)" if any(g.get("example") for g in state["guides"]) else ")")
        + "\n"
        f"- 파일: {rel}\n"
    )
    if any(g.get("example") for g in state["guides"]):
        answer += "※ [예시]는 이해를 돕는 가상 내용이에요. 그대로 제출하지 마세요."
    if state["level"] in ("부분", "없음", "추정"):
        answer += f"\n※ {level_text(state['level'], state['items'])[1]}"
    n_miss = sum(1 for it in state["items"] if not item_ref(it))
    if n_miss:
        answer += f"\n※ 평가기준을 찾지 못한 항목 {n_miss}개는 일반적인 안내예요. 공고문 평가표를 확인하세요."
    return {"file": str(out), "answer": answer}


# 노드: 서식 예시. 공고문은 질문 -> 대화 기록 순으로 찾고 검색은 쓰지 않는다 (엉뚱한 공고 방지).
# 못 찾으면 어느 사업인지 되묻는다.
def n_form(state: GuideState):
    """입력: question/history. 출력: 서식 예시 또는 선택 안내를 담은 answer 문자열."""
    q = state["question"]
    p, cands = find_doc(q, use_search=False)
    if p is None:
        p = doc_from_history(state.get("history"))
    if p is None:
        if cands:
            lines = "\n".join(f"- {c}" for c in cands)
            return {
                "answer": f"어느 사업의 서식인지 정확히 모르겠어요. 아래 중에 있나요?\n{lines}"
            }
        return {
            "answer": "어느 사업의 서식 예시를 볼까요? 기관명이나 사업명을 알려 주세요."
        }
    return {"doc_path": str(p), "answer": form_example(p, q)}


# 노드: 가이드·서식이 아닌 일반 질문은 팀 baseline 의 검색 답변을 그대로 쓴다.
# ---- 사업명 없이 특정 공고의 내용을 묻는 일반 질문: 검색이 아무 공고나 고르지 않게 되묻는다 ----
# 공고 하나의 세부 내용을 묻는 말 (일정, 자격, 평가 등)
_PROJECT_Q = re.compile(
    r"일정|마감|접수|기간|자격|요건|평가|배점|제출|서류|계약|과업|범위|요구|담당|문의|연락|보안|인력|하도급|예산|금액|가격|규모|발주|조건|기준|방식"
)
# 공고 전체를 묻는 말 (이런 질문은 사업명 없이도 정상이므로 되묻지 않는다)
_WHOLE_Q = re.compile(
    r"가장|전체|모든|모두|전부|각각|비교|평균|합계|몇\s*개|몇\s*곳|순위|목록|어떤\s*(?:공고|사업)|공고들|사업들"
)
ASK_PROJECT = "어느 사업에 대한 질문인지 알려 주세요. 기관명이나 사업명 일부(예: 평택시), 또는 번호(예: 3번)로 말씀해 주세요."
_PFX = ASK_PROJECT[:12]


def _project_table() -> str:
    """공고 목록 표 (번호로 고를 수 있게 되묻는 답변 끝에 붙인다)."""
    files = _budget_files()

    def cell(t: str, n: int) -> str:
        t = t.replace("|", "/").strip()
        return t if len(t) <= n else t[:n] + "…"

    rows = "\n".join(
        f"| {k} | {cell(_budget_parts(p)[0], 18)} | {cell(_budget_parts(p)[1], 42)} |"
        for k, p in enumerate(files, 1)
    )
    return (
        f"\n\n<details><summary>공고 {len(files)}개 목록 보기</summary>\n\n"
        f"| 번호 | 기관 | 사업명 |\n| --- | --- | --- |\n{rows}\n\n</details>"
    )


def _original_question(history) -> str:
    """사업을 되묻기 전에 했던 원래 질문 (되묻기가 여러 번 이어져도 처음 질문)."""
    h = list(history or [])
    i = len(h) - 1

    def is_ask(t) -> bool:
        return t.get("role") != "user" and t.get("content", "").startswith(_PFX)

    while i >= 1 and is_ask(h[i]):
        orig = h[i - 1].get("content", "")
        i -= 2
        if not (i >= 0 and is_ask(h[i])):
            return orig
    return ""


def _project_followup(state):
    """직전 답변이 사업을 되묻는 말이고 이번 말이 사업명·번호만 짧게 답한 것이면 처리한다.
    -> None(해당 없음) | {"question", "history"}(앞 질문에 사업명을 붙여 검색) | {"answer"}(다시 되묻기)"""
    q, history = state["question"], state.get("history")
    if not _last_assistant(history).startswith(_PFX):
        return None
    if len(q.strip()) > 20 or _PROJECT_Q.search(q) or _WHOLE_Q.search(q):
        return None  # 새 질문을 다시 한 것
    orig = _original_question(history)
    if not orig:
        return None
    p = _budget_doc({"history": history}, q)
    if p is not None:
        return {"question": f"{_budget_label(p)} {orig}", "history": None}
    cands = _budget_cands(q)
    if len(cands) >= 2:
        files = _budget_files()
        listing = "\n".join(f"{files.index(c) + 1}. {_budget_label(c)}" for c in cands)
        return {
            "answer": f"어느 사업에 대한 질문인지 아직 하나로 정해지지 않았어요. 아래에서 번호로 골라 주세요.\n{listing}"
        }
    return {"answer": ASK_PROJECT + _project_table()}


# ---- 공고 전체를 묻는 예산 순위 질문 ("예산이 가장 많은 사업"): 공고 정보의 사업 금액으로 모델 없이 계산한다 ----
_RANK_Q = re.compile(
    r"(?:예산|금액|사업비|사업\s*금액).*(?:가장|제일|최대|최소|많|큰|높|적|작|낮|비싼|상위|하위|순위)"
    r"|(?:가장|제일|최대|최소|상위|하위).*(?:예산|금액|사업비)"
)
_AMOUNTS = None


def _budget_amounts() -> list:
    """[(기관, 사업명, 금액(원))] — data_list.csv 의 '사업 금액' (없거나 100만 원 미만인 공고는 뺀다)."""
    global _AMOUNTS
    if _AMOUNTS is None:
        import csv

        out = []
        try:
            with open(
                ROOT / "data" / "data_list.csv", encoding="utf-8-sig", newline=""
            ) as f:
                for row in csv.DictReader(f):
                    try:
                        amt = int(float(row.get("사업 금액") or 0))
                    except ValueError:
                        continue
                    if amt >= 1_000_000:  # 1원 같은 입력 오류 값은 뺀다
                        out.append(
                            (
                                (row.get("발주 기관") or "").strip(),
                                (row.get("사업명") or "").strip(),
                                amt,
                            )
                        )
        except OSError:
            pass
        _AMOUNTS = out
    return _AMOUNTS


def _won(n: int) -> str:
    return f"{n:,}원" + (f" (약 {n / 1e8:.2f}억 원)" if n >= 1e8 else "")


def rank_answer(q: str):
    """예산이 가장 많은/적은 사업 질문이면 순위 표를 돌려준다 (아니면 None)."""
    if not _RANK_Q.search(q):
        return None
    data = _budget_amounts()
    if not data:
        return None
    asc = bool(re.search(r"적|작|낮|최소|하위|저렴", q)) and not re.search(
        r"많|큰|높|최대|상위|비싼", q
    )
    m = re.search(r"(\d+)\s*(?:개|곳|건|위)", q)
    n = min(int(m.group(1)), 20) if m else 3
    ranked = sorted(data, key=lambda x: x[2], reverse=not asc)[: max(n, 1)]
    rows = "\n".join(
        f"| {k} | {o} | {t[:46]}{'…' if len(t) > 46 else ''} | {_won(a)} |"
        for k, (o, t, a) in enumerate(ranked, 1)
    )
    head = ranked[0]
    return (
        f"사업 금액이 {'가장 적은' if asc else '가장 많은'} 사업은 {head[0]}의 「{head[1]}」으로 {_won(head[2])}이에요.\n\n"
        f"| 순위 | 기관 | 사업명 | 사업 금액 |\n| --- | --- | --- | --- |\n{rows}\n\n"
        f"공고 정보에 사업 금액이 적힌 {len(data)}개 공고 기준이에요 (금액이 비어 있거나 입력 오류인 공고는 뺐어요). 더 보려면 '상위 5개'처럼 말씀해 주세요."
    )


def needs_project(state) -> bool:
    """공고 하나의 내용을 묻는데 사업명도, 앞 대화의 사업도 없는가 (모델·검색 없이 판단)."""
    q = state["question"]
    if not _PROJECT_Q.search(q) or _WHOLE_Q.search(q):
        return False
    p, cands = find_doc(q, use_search=False)
    # 파일 이름과 겹치는 말이 있으면 사업명을 말한 것으로 본다
    if p is not None or cands:
        return False
    return doc_from_history(state.get("history")) is None


def n_rag(state: GuideState):
    """입력: question/history. baseline RAG의 답변 문자열을 그대로 answer 로 전달한다."""
    from ai.rahui.exp_filter_rerank.rahui_engine import baseline

    ranked = rank_answer(state["question"])
    if ranked:
        return {"answer": ranked}
    fu = _project_followup(state)
    if fu:
        if "answer" in fu:
            return fu
        state = {**state, **fu}
    elif needs_project(state):
        return {"answer": ASK_PROJECT + _project_table()}

    ans = baseline.answer(state["question"], state.get("history"))
    if not (
        ans or ""
    ).strip():  # 서식 대화 뒤 긴 기록 때문에 모델이 빈 답을 내는 경우: 기록 없이 한 번 더
        ans = baseline.answer(state["question"], None)
    if not (ans or "").strip():
        ans = "답변을 만들지 못했어요. 질문을 조금 바꿔서 다시 물어봐 주세요."
    elif (
        re.search(r"컨텍스트.{0,20}(?:질문해|질문이 없|질문을 주)|질문이 없습니다", ans)
        and len(ans) < 120
    ):
        ans = WHAT_TO_ASK  # 모델의 "컨텍스트를 바탕으로 질문해 주세요" 같은 안내는 내부 용어라 바꾼다
    return {"answer": ans}


def n_chat(state: GuideState):
    """인사·감사·도움말 고정 답변 (모델 호출 없음)"""
    q = state["question"]
    if _FILLER.match(q.strip()) and _last_assistant(state.get("history")).startswith(
        _PFX
    ):  # 사업을 되묻는 중에 추임새만 오면 다시 되묻는다
        return {"answer": ASK_PROJECT + _project_table()}
    return {"answer": small_talk(q) or HELP_TEXT}


def build_graph():
    """랭그래프를 만든다.

    START -route-> guide: find_doc -> extract -> (항목 있음: write / 없음: fallback -> write) -> build -> END
                   form : n_form -> END
                   rag  : n_rag  -> END
    find_doc 이 공고문을 못 찾으면 되묻는 답변과 함께 바로 END.
    """
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(GuideState)
    for name, fn in [
        ("find_doc", n_find_doc),
        ("extract", n_extract),
        ("fallback", n_fallback),
        ("write", n_write),
        ("build", n_build),
        ("rag", n_rag),
        ("chat", n_chat),
        ("form", n_form),
        ("budget_ask", n_budget_ask),
        ("budget", n_budget),
    ]:
        g.add_node(name, fn)
    g.add_conditional_edges(
        START,
        route,
        {
            "guide": "find_doc",
            "form": "form",
            "rag": "rag",
            "chat": "chat",
            "budget_ask": "budget_ask",
            "budget": "budget",
        },
    )
    g.add_edge("form", END)
    g.add_conditional_edges(
        "find_doc",
        lambda s: "extract" if s.get("doc_path") else "end",
        {"extract": "extract", "end": END},
    )
    g.add_conditional_edges(
        "extract",
        lambda s: "write" if s.get("items") else "fallback",
        {"write": "write", "fallback": "fallback"},
    )
    g.add_edge("fallback", "write")
    g.add_edge("write", "build")
    g.add_edge("build", END)
    g.add_edge("rag", END)
    g.add_edge("chat", END)
    g.add_edge("budget_ask", END)
    g.add_edge("budget", END)
    return g.compile()


# 그래프는 처음 쓸 때 한 번만 만들어 재사용한다 (매 호출마다 컴파일하지 않으려고).
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def answer(message, history=None):
    """팀 화면 연결용 함수 (baseline.answer 와 같은 모양)

    입력 위치: 백엔드는 engine_fn(request.message, history)로 호출한다.
      message=이번 질문 문자열, history=이전 턴 목록([{"role": "user"|"assistant", "content": ...}, ...]).
      백엔드는 현재 질문·답변을 이 호출 뒤에 저장하므로 history 에 현재 질문은 포함되지 않는다.
    출력 위치: 반환값은 최종 답변 문자열 하나이며, backend 는 이를 ChatResponse.answer 에 넣는다.
      가이드 docx 경로와 서식 파일 경로는 별도 반환값이 아니라 답변 안의 "파일:" 줄이다.
      가이드의 "파일:" 경로는 실제 저장 위치(OUT_DIR)를 가리킨다.

    baseline.answer 와 입력·출력 규칙은 같지만, 이전 대화 사용은 같지 않다.
    writer 는 history 를 요청 분류와 이전 문서/서식 확인에 활용한다. 가이드 문장 생성에는 전달하지 않고,
    일반 질문(rag)일 때만 history 를 baseline.answer 에 전달해 RAG 답변에 반영한다.
    """
    result = get_graph().invoke({"question": message, "history": history or []})
    return result["answer"]


# =====================================================================
# 6. 에이전트 도구 (@tool) 와 양식 예시 루브릭 평가
# =====================================================================
def make_tool():
    """팀 에이전트에 끼울 도구 (langchain). 도구에는 대화 기록이 안 오므로 질문에 사업명을 넣어야 한다.

    도구 설명(docstring)이 에이전트가 이 도구를 언제 쓸지 고르는 기준이므로 할 수 있는 일과 요청 방법을 적어 둔다.
    """
    from langchain_core.tools import tool

    @tool
    def proposal_helper(request: str) -> str:
        """입찰 공고문(RFP)을 바탕으로 두 가지를 해 준다.
        (1) 제안서 작성 가이드: 평가항목별로 쓰는 칸, 꼭 넣을 내용, 예시, 팁, 흔한 실수를 담은 docx를 만든다.
        (2) 양식 예시: 공고문의 별지 서식 빈칸을 채운 작성 예시를 보여 준다 (문서 값, 가상 값, 직접 작성 칸을 구분 표시).
        request 에 사업명(또는 기관명)과 요청('작성 가이드 만들어줘', '서식 예시 줘')을 함께 적는다."""
        return answer(request)

    return proposal_helper


# 루브릭 채점용: 서식 원문에서 칸 내용 집합, 줄별 첫 라벨, 표 줄 수를 모은다.
def _form_cells(lines: list[str]) -> tuple[set, list[str], int]:
    """서식 원문의 칸 집합, 줄별 첫 라벨, 표 줄 수"""
    cells, labels, n_rows = set(), [], 0
    for raw in lines:
        t = raw.strip()
        if "|" not in t or re.fullmatch(r"[\s|:\-]*", t):
            continue
        n_rows += 1
        cs = [c.strip() for c in t.strip("|").split("|")]
        for c in cs:
            if c:
                cells.add(sq(c))
        first = next((c for c in cs if c), "")
        if first:
            labels.append(sq(first))
    return cells, labels, n_rows


# 루브릭 채점용: 서식 원문 표를 줄별 칸 목록으로 읽는다 (위치 검사에 사용).
def _form_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for raw in lines:
        t = raw.strip()
        if "|" in t and not re.fullmatch(r"[\s|:\-]*", t):
            rows.append([c.strip() for c in t.strip("|").split("|")])
    return rows


# FAKE_RULES 를 "가상 값 -> 해당 키워드 모음"으로 뒤집은 표. 위치 검사에서 그 값이 맞는 칸에 들어갔는지 확인한다.
_FAKE_KEYS = {}
for _keys, _val in FAKE_RULES:
    _FAKE_KEYS.setdefault(_val, set()).update(k.lower() for k in _keys)


def check_form_answer(doc_path: Path, form: dict, text: str) -> dict:
    """양식 예시 답변 하나를 원문과 대조해 루브릭 점수를 계산

    원리: 모델 없이 규칙으로 만든 답변이므로, 같은 규칙이 아닌 방식(원문 대조)으로 검사한다.
      - 표시없는_값: 원문에 없는 값인데 [문서]/[예시]/[직접 작성] 표시가 없는 칸 수 (합격 기준 0)
      - 문서값_오류: [문서] 표시 값이 data_list.csv 사업 정보와 다른 칸 수
      - 구조일치: 원문 표 줄 수 = 출력 줄 수 ("N줄 생략" 요약 줄의 N 을 더해서 비교)
      - 라벨_누락: 출력된 줄의 원문 첫 칸 라벨이 출력에 없는 수
      - 위치오류: [예시] 값의 왼쪽 라벨이나 머리글에 그 값의 키워드가 없는 수 (엉뚱한 칸 의심)
      - 출처_표시: "출처:"와 가상 값 경고(※)가 있는지
    한계: 표시의 유무를 같은 규칙으로 검사하므로 순환적이다. 값이 맞는 칸에 들어갔는지 등은 사람이 읽어야 안다.
    """
    lines = doc_path.read_text(encoding="utf-8", errors="replace").split("\n")
    cells, labels, n_rows = _form_cells(lines[form["start"] : form["end"]])
    meta = load_meta(doc_path)
    meta_vals = {sq(v) for v in meta.values()}
    # 출력의 • 줄만 모은다. "N줄 생략" 요약 줄은 따로 세고(omitted), 나머지 줄이 원문 몇 번째 줄에 해당하는지(orig_idx)를 기록한다.
    all_bullets = [ln[1:].strip() for ln in text.split("\n") if ln.startswith("•")]
    bullets, omitted, orig_idx = [], 0, []
    pos = 0
    for b in all_bullets:
        if SKIP_MARK in b:
            k = int(re.search(r"(\d+)" + re.escape(SKIP_MARK), b).group(1))
            omitted += k
            pos += k
            continue
        bullets.append(b)
        orig_idx.append(pos)
        pos += 1
    orows = _form_rows(lines[form["start"] : form["end"]])
    # 칸마다 검사: [예시] 위치, [문서] 값 일치, 표시 없는 값.
    unmarked, bad_doc, n_cells, misplaced = [], [], 0, []
    for bi, b in enumerate(bullets):
        orow = orows[orig_idx[bi]] if orig_idx[bi] < len(orows) else []
        hdr = next(
            (r for r in orows[: orig_idx[bi]][::-1] if sum(1 for x in r if x) >= 3), []
        )
        for ci, c in enumerate([x.strip() for x in b.split("|")]):
            if not c:
                continue
            n_cells += 1
            if (
                c.endswith("[예시]") and sq(c[:-4]) not in cells
            ):  # 칸 위치 검사: 왼쪽 라벨이나 머리글에 해당 키워드가 있어야 함
                keys = _FAKE_KEYS.get(c[:-4].strip(), set())
                near = [x for x in orow[:ci] if x] + (
                    [hdr[ci]] if ci < len(hdr) else []
                )
                if keys and not any(
                    k in sq(x).lower()
                    for x in near
                    if len(sq(x)) <= FAKE_LABEL_MAX
                    for k in keys
                ):
                    misplaced.append(c)
            if c.endswith("[문서]"):
                if sq(c[:-4]) not in meta_vals:
                    bad_doc.append(c)
            elif c.endswith("[예시]") or c.startswith("[직접 작성"):
                continue
            elif sq(c) not in cells:
                unmarked.append(c)
    out_text = sq(text)
    shown = set(orig_idx)  # 생략 표시한 반복 줄의 라벨은 누락으로 세지 않음
    row_labels = [sq(next((c for c in r if c), "")) for r in orows]
    missing = [
        lb
        for i, lb in enumerate(row_labels)
        if i in shown and lb and lb not in out_text
    ]
    return {
        "원문_표줄": n_rows,
        "출력_줄": len(bullets) + omitted,
        "구조일치": len(bullets) + omitted == n_rows,
        "위치오류": len(misplaced),
        "라벨_누락": len(missing),
        "표시없는_값": len(unmarked),
        "문서값_오류": len(bad_doc),
        "문서값수": sum(b.count("[문서]") for b in bullets),
        "예시값수": sum(b.count("[예시]") for b in bullets),
        "직접작성수": sum(b.count("[직접 작성") for b in bullets),
        "출처_표시": "출처:" in text and "※" in text,
        "_unmarked": unmarked[:3],
        "_missing": missing[:3],
        "_misplaced": misplaced[:3],
    }


def run_guide_check(n: int = 8, use_model: bool = True):
    """서로 다른 공고 n곳으로 가이드를 만들어 품질을 자동 점검한다 (파일 docx 는 만들지 않음).

    고르는 법: 평가표를 읽은 공고(확정/추정/부분)에서 무작위(seed 1118) n-1곳 + "없음" 1곳.
    점검 항목은 5-1표와 같다: 형식 안내문 복사, 문장 전체 괄호, 지어낸 숫자, [예시] 시작 문장, 사업·시스템 이름 남음,
    불릿 길이, 모델이 쓴 항목 수, 걸린 시간. 자동 점검은 눈으로 읽는 확인을 대신하지 못한다.
    결과: 터미널 표 + outputs/eval_guides_check_<시각>.csv (로컬 전용).
    """
    import csv
    import datetime
    import random

    ok, none = [], []
    for p in sorted(CORPUS_DIR.glob("*.txt")):
        items, level, _ = extract_items(
            p.read_text(encoding="utf-8", errors="replace"), p.name
        )
        (ok if items else none).append((p, items, level))
    rnd = random.Random(1118)
    picked = rnd.sample(ok, min(n - 1, len(ok))) + rnd.sample(none, min(1, len(none)))
    bad_copy = (
        "가상의 예시",
        "첫째 내용",
        "점수를 더 받는 요령",
        "자주 하는 실수 한 문장",
    )
    num_pat = re.compile(r"\d+\s*(?:단계|시간|주|개월|일|명|%|배)")
    rows, dump = [], []
    for p, items, level in picked:
        project = p.stem
        if not items:
            items = [
                {"group": "", "name": nm, "score": None, "criteria": c, "elements": ""}
                for nm, c in GENERIC
            ]
        t0 = time.time()
        guides, model_note = write_guides(project, items, use_model)
        sec = time.time() - t0
        m = re.search(r"(\d+)/(\d+)개", model_note)
        n_model = int(m.group(1)) if m else 0
        dump.append(
            f"################ {project}  (근거 {level}, 항목 {len(items)}, 모델 {n_model}, {sec:.0f}초)"
        )
        for it, g in zip(items, guides, strict=False):
            dump.append(
                f"--- {it['name']}"
                + (f" ({it['score']:g}점)" if it.get("score") is not None else "")
            )
            dump.append(f"쓰는 칸: {g.get('where', '')}")
            dump += [f"  · {b}" for b in g.get("must", [])]
            dump.append(f"[예시] {g.get('example', '')}")
            dump.append(f"팁: {g.get('tip', '')}")
            dump.append(f"흔한 실수: {g.get('mistake', '')}")
        dump.append("")
        fields = [
            (k, g.get(k, ""))
            for g in guides
            for k in ("where", "example", "tip", "mistake")
        ]
        must = [b for g in guides for b in g.get("must", [])]
        texts = [v for _, v in fields] + must
        name_toks = [
            t
            for t in re.findall(r"[가-힣A-Za-z0-9]+", project)
            if len(t) >= 5 and t.endswith(("시스템", "플랫폼", "사업"))
        ]
        if "_" in project:  # 발주기관 이름(첫 "_" 앞의 4글자 이상 낱말)도 남았는지 본다
            name_toks += re.findall(r"[가-힣A-Za-z]{4,}", project.split("_")[0])
        leak = sum(
            1
            for tx in texts
            if any(
                t in tx or (t[:3] in tx and t[-3:] in tx and len(t) > 6)
                for t in name_toks
            )
        )
        ex = [g.get("example", "") for g in guides if g.get("example")]
        rows.append(
            {
                "공고": project[:40],
                "근거수준": level,
                "항목수": len(items),
                "모델항목": n_model,
                "초": round(sec, 1),
                "예시있음": len(ex),
                "예시_우리회사시작": sum(1 for e in ex if e.startswith("우리 회사")),
                "안내문복사": sum(1 for tx in texts if any(b in tx for b in bad_copy)),
                "괄호감쌈": sum(
                    1 for _, v in fields if v.startswith("(") and v.endswith(")")
                ),
                "지어낸숫자": sum(1 for e in ex if num_pat.search(e)),
                "이름남음": leak,
                "불릿평균자": round(sum(len(b) for b in must) / max(1, len(must)), 1),
                "불릿최대자": max((len(b) for b in must), default=0),
            }
        )
        print(
            f"[{len(rows)}/{len(picked)}] {rows[-1]['공고']} {level} 항목 {len(items)} 모델 {n_model} {sec:.0f}초",
            flush=True,
        )
    keys = [
        "예시있음",
        "예시_우리회사시작",
        "안내문복사",
        "괄호감쌈",
        "지어낸숫자",
        "이름남음",
    ]
    print("\n=== 가이드 품질 점검 요약 ===")
    for r in rows:
        print(
            " | ".join(
                f"{k}={r[k]}"
                for k in ["공고", "근거수준", "항목수", "모델항목", "초"] + keys
            )
        )
    tot = {k: sum(r[k] for r in rows) for k in keys + ["항목수", "모델항목"]}
    print(
        f"합계: 항목 {tot['항목수']} · 모델이 쓴 항목 {tot['모델항목']} · 예시 {tot['예시있음']} (우리 회사로 시작 {tot['예시_우리회사시작']}) · "
        f"안내문복사 {tot['안내문복사']} · 괄호감쌈 {tot['괄호감쌈']} · 지어낸숫자 {tot['지어낸숫자']} · 이름남음 {tot['이름남음']}"
    )
    print(
        f"공고당 평균 {sum(r['초'] for r in rows) / len(rows):.0f}초, 최대 {max(r['초'] for r in rows):.0f}초"
    )
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    path = (
        OUT_DIR.parent
        / f"eval_guides_check_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"저장: {path}  (로컬 전용, 외부 반출 금지)")
    txt_path = path.with_name(path.stem + "_내용.txt")
    txt_path.write_text("\n".join(dump), encoding="utf-8")
    print(
        f"저장: {txt_path}  (가이드 문장 전체, 사람이 읽어 보는 용도, 로컬 전용·외부 반출 금지)"
    )


def run_rubric(n: int = 20):
    """서로 다른 공고 n곳에서 서식을 하나씩 뽑아 양식 예시를 만들고 루브릭 점수를 낸다 (모델 안 씀)

    서식이 있는 공고를 일정 간격으로 고르고, 공고마다 서식을 번갈아 첫 서식 / 표가 가장 큰 서식으로 고른다.
    결과: 터미널 표 + outputs/eval_forms_rubric_<시각>.csv + 사람이 읽는 _답변.txt (둘 다 로컬 전용, 외부 반출 금지).
    """
    import csv
    import datetime

    docs = []
    for p in sorted(CORPUS_DIR.glob("*.txt")):
        forms = find_forms(p.read_text(encoding="utf-8", errors="replace").split("\n"))
        if forms:
            docs.append((p, forms))
    step = max(1, len(docs) // n)
    picked = docs[::step][:n]
    rows, dump = [], []
    for k, (p, forms) in enumerate(picked):
        f = (
            forms[0] if k % 2 == 0 else max(forms, key=lambda x: x["n_rows"])
        )  # 번갈아 첫 서식 / 표가 가장 큰 서식
        try:
            text = form_example(p, f"제{f['no']}호 서식 예시 줘")
            r = check_form_answer(p, f, text)
            err = ""
        except Exception as e:
            text, r, err = "", {}, f"{type(e).__name__}: {e}"
        row = {
            "번호": k + 1,
            "문서": p.stem[:28],
            "서식": f"제{f['no']}호 {f['title'][:20]}",
            "오류": err,
            **{x: y for x, y in r.items() if not x.startswith("_")},
        }
        rows.append(row)
        dump.append(
            f"=== [{k + 1}] {p.name} 제{f['no']}호 {f['title']}\n{text}\n위치 의심 예: {r.get('_misplaced')} / 표시없는 값 예: {r.get('_unmarked')} / 누락 라벨 예: {r.get('_missing')}\n"
        )
    ok = [r for r in rows if not r["오류"]]
    tot = len(rows)
    print(
        f"{'#':>2} {'문서':<30} {'서식':<26} 줄(원/출) 구조 누락 미표시 문서오류  [문서]/[예시]/[직접]"
    )
    for r in rows:
        if r["오류"]:
            print(f"{r['번호']:>2} {r['문서']:<30} 오류: {r['오류'][:50]}")
            continue
        print(
            f"{r['번호']:>2} {r['문서']:<30} {r['서식']:<26} {r['원문_표줄']:>3}/{r['출력_줄']:<3} {'O' if r['구조일치'] else 'X':^4}{r['라벨_누락']:^5}{r['표시없는_값']:^7}{r['문서값_오류']:^7} {r['문서값수']}/{r['예시값수']}/{r['직접작성수']}"
        )

    def pct(c):
        return f"{c}/{tot} ({100 * c / tot:.0f}%)" if tot else "-"

    print("\n=== 루브릭 요약 ===")
    print(
        "지어낸 값 표시(합격 기준 100%):",
        pct(sum(1 for r in ok if r["표시없는_값"] == 0)),
        "  ← 표시 없이 들어간 값 0건인 서식",
    )
    print(
        "근거 값 정확성([문서] 값이 사업 정보와 일치):",
        pct(sum(1 for r in ok if r["문서값_오류"] == 0)),
    )
    print(
        "구조 일치(원문 표 줄 수 = 출력 줄 수):",
        pct(sum(1 for r in ok if r["구조일치"])),
    )
    print("필수 라벨 누락 없음:", pct(sum(1 for r in ok if r["라벨_누락"] == 0)))
    print(
        "칸 위치 의심([예시]가 엉뚱한 칸):",
        f"{sum(r['위치오류'] for r in ok)}건 / 서식 {sum(1 for r in ok if r['위치오류'])}곳",
    )
    print("출처·가상 안내 표시:", pct(sum(1 for r in ok if r["출처_표시"])))
    print("실행 오류:", sum(1 for r in rows if r["오류"]), "건")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR.parent / f"eval_forms_rubric_{ts}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    txt_path = OUT_DIR.parent / f"eval_forms_rubric_{ts}_답변.txt"
    txt_path.write_text("\n".join(dump), encoding="utf-8")
    print(f"\n저장: {csv_path}\n저장: {txt_path}  (사람이 직접 확인용, 외부 반출 금지)")


# 터미널 대화 모드: 이전 질문·답변을 history 에 쌓아서 "3번", "다른 서식도" 같은 후속 질문이 이어지게 한다.
def chat():
    """터미널 대화 모드: 이전 대화를 기억하면서 이어서 물어볼 수 있음 (종료: 빈 줄 또는 q)"""
    history = []
    print("대화 모드예요. 끝내려면 q 또는 빈 줄을 입력하세요.")
    while True:
        try:
            msg = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() in ("q", "quit", "exit"):
            break
        reply = answer(msg, history)
        print("\n" + reply)
        history += [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": reply},
        ]


def precompute(limit: int | None = None, only: list[str] | None = None):
    """공고문 전체의 가이드를 미리 만들어 캐시에 저장한다 (이미 있으면 건너뜀). 터미널: --precompute [N] [--only 낱말 ...]"""
    files = sorted(CORPUS_DIR.glob("*.txt"))
    if only:  # 공고 이름에 낱말이 하나라도 들어 있는 것만 (작은 시험용)
        files = [f for f in files if any(o in f.stem for o in only)]
    files = files[: limit or None]
    t0 = time.time()
    for n, f in enumerate(files, 1):
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
            items, _lv, _nt = extract_items(txt, f.name)
            if not items:
                items = [
                    {
                        "group": "",
                        "name": a,
                        "score": None,
                        "criteria": b,
                        "elements": "",
                    }
                    for a, b in GENERIC
                ]
            cp = _cache_path(f.stem, items)
            if cp.exists():
                print(f"[{n}/{len(files)}] 건너뜀(이미 있음) {f.stem[:40]}", flush=True)
                continue
            t = time.time()
            _g, note = write_guides(f.stem, items, True)
            print(
                f"[{n}/{len(files)}] {time.time() - t:.0f}초 {note[:40]} | {f.stem[:40]}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[{n}/{len(files)}] 실패 {type(e).__name__}: {f.stem[:40]}", flush=True
            )
    print(
        f"끝: {(time.time() - t0) / 60:.1f}분, 캐시 {len(list(CACHE_DIR.glob('*.json')))}개",
        flush=True,
    )


# 터미널 실행:  질문을 붙이면 1회 답변 / 질문 없이 실행하면 대화 모드 / --rubric N 이면 양식 예시 루브릭 평가 / --no-model 이면 모델 없이 규칙만.
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=None, help="없으면 대화 모드")
    ap.add_argument(
        "--no-model", action="store_true", help="모델 없이 규칙만으로 만들기"
    )
    ap.add_argument(
        "--rubric",
        type=int,
        nargs="?",
        const=20,
        default=None,
        help="양식 예시 루브릭 평가 (공고 N곳, 기본 20)",
    )
    ap.add_argument(
        "--guides",
        type=int,
        nargs="?",
        const=8,
        default=None,
        help="여러 공고로 가이드 품질 점검 (공고 N곳, 기본 8, 모델 사용)",
    )
    ap.add_argument(
        "--precompute",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="공고문 전체(또는 앞 N개) 가이드를 미리 만들어 캐시에 저장",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="--precompute 와 함께: 이름에 이 낱말이 들어간 공고만 (시험용)",
    )
    a = ap.parse_args()
    if a.precompute is not None:
        precompute(a.precompute or None, a.only)
        raise SystemExit(0)
    if a.guides is not None:
        run_guide_check(a.guides, use_model=not a.no_model)
        raise SystemExit(0)
    if a.rubric is not None:
        run_rubric(a.rubric)
        raise SystemExit(0)
    if a.no_model:
        USE_MODEL = False
    if a.question:
        print(answer(a.question))
    else:
        chat()
