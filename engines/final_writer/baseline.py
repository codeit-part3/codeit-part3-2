"""writer(작성 도우미)를 백엔드에 연결하는 어댑터.

백엔드(engine_loader)는 engines.<engine_id>.baseline 에서 warmup() 과 get_graph() 를 찾고,
chat 라우트는 그래프의 ainvoke({"question","history"}) 결과에서 "answer" 와 "retrieved" 를 읽는다.
writer 의 그래프는 "retrieved" 를 돌려주지 않아 그대로 연결하면 500 이 나므로 빈 목록을 채워 준다.

주의! 이 파일을 불러오면 final.baseline 의 format_context 가 메모리에서 바뀐다.
  (모델에게 주는 컨텍스트의 표만 다듬음. 문서, 인덱스, baseline.py 파일은 그대로)
  같은 서버에서 final 엔진을 함께 띄우면 그쪽 답변에도 표 정리가 적용된다.
  평가(evaluate.py)는 이 어댑터를 쓰지 않으므로 평가 수치와는 무관하다.
  표 정리를 끄려면 환경변수 TABLE_CLEAN=0.
"""

import os
import re
import threading

from engines.final import baseline as _base
from engines.final import writer as _writer

# ── 표 정리 (모델에게 보여 주는 컨텍스트 글만 다듬음. 문서/인덱스/baseline.py 는 그대로) ──────────────
# 원문 OCR 표의 문제: ① 깨진 글머리 기호 ② "M1 | M2 | ... | M12" 월 헤더 줄과 12개 빈 칸
# (색칠로 표시한 일정이라 글자가 없음) → 모델이 칸을 한 칸씩 밀어 단계 칸에 M1, M2 를 적고 있었음.
_PUA_CH = "[\ue000-\uf8ff\U000f0000-\U000ffffd]"
_PUA = re.compile(r"[ \t]*" + _PUA_CH + r"[ \t]*")
_TABLE = re.compile(r"\[표\]\r?\n(.*?)\r?\n?\[/표\]", re.S)
_MONTH = re.compile(r"^M\s?\d{1,2}$")
_SPACED = re.compile(r"^[가-힣](?: [가-힣]){1,3}$")


def _fix_table(m):
    """월 칸(M1~M12)이 있는 일정표만 `단계 | 과업 | 세부 업무 | M1 ... M12` 의 제대로 된 표로 다시 만들어 줍니다.
    (원문은 '단 계 | 과 업 | …' 줄과 'M1 | M2 | …' 줄이 따로 있고 행마다 빈 칸 수가 달라, 모델이 칸을 한 칸씩 밀어 썼음)
    월 칸이 없는 표는 그대로 둡니다."""
    rows = [
        [c.strip() for c in ln.rstrip("\r").split("|")]
        for ln in m.group(1).split("\n")
        if ln.strip()
    ]
    month_idx = [
        i for i, r in enumerate(rows) if sum(bool(_MONTH.match(c)) for c in r) >= 3
    ]
    if not month_idx or month_idx[0] == 0:
        return m.group(0)
    mi = month_idx[0]
    months = [c for c in rows[mi] if _MONTH.match(c)]
    nm = len(months)
    head = [c for c in rows[0] if c][
        :3
    ]  # 단계 / 과업 / 세부 업무 ('추진 일정' 은 월 칸이 대신함)
    if len(head) < 3:
        return m.group(0)
    head = [c.replace(" ", "") if _SPACED.match(c) else c for c in head]
    out = ["| " + " | ".join(head + months) + " |", "|" + " --- |" * (3 + nm)]
    for i, r in enumerate(rows):
        if i <= mi:
            continue
        text = [c.replace(" ", "") if _SPACED.match(c) else c for c in r[:3]]
        text += [""] * (3 - len(text))
        cells = (r[3:] + [""] * nm)[:nm]
        out.append("| " + " | ".join(text + cells) + " |")
    return "\n".join(out)


def clean_context_tables(text: str) -> str:
    text = _TABLE.sub(_fix_table, text)
    text = _PUA.sub(" • ", text)
    return re.sub(r"(?m)^[ \t]+•", "•", text)


_orig_format_context = _base.format_context


def _format_context(retrieved):
    text = _orig_format_context(retrieved)
    if os.getenv("TABLE_CLEAN") == "0":  # 비교용으로 끄기: $env:TABLE_CLEAN="0"
        return text
    cleaned = clean_context_tables(text)
    if (
        os.getenv("SHOW_CTX") == "1"
    ):  # 확인용: $env:SHOW_CTX="1" → 모델에게 간 글을 백엔드 터미널에 출력
        print(
            "\n=====[모델에게 간 컨텍스트]=====\n"
            + cleaned[:6000]
            + "\n=====[끝]=====",
            flush=True,
        )
    return cleaned


_base.format_context = _format_context  # baseline 의 retrieve/generate 가 이 이름을 불러 씁니다 (임시, 메모리에서만)


# [백엔드 참고] final / final_writer 는 같은 모델·DB·리랭커(final.baseline._resources)를 공유한다.
# 웜업(시험 호출)이 두 번 돌지 않게, final 의 warmup 을 "한 번만" 돌도록 감싼다.
# 하나씩 load_engine 하면 순서와 상관없이 시험 호출은 한 번만 돌고, 시작 시간이 늘지 않는다.
_orig_warmup = _base.warmup
_warm_lock = threading.Lock()
_warmed = False


def _warmup_once():
    global _warmed
    with _warm_lock:
        if _warmed:
            return
        if not _base._resources:  # 이미 올라와 있으면(공유 보관함이 차 있으면) 건너뜀
            _orig_warmup()  # 채팅 모델, 빠른 모델, 인덱스, 임베딩, 리랭커
        _warmed = True


_base.warmup = _warmup_once  # engine_loader 가 final.baseline.warmup 을 불러도 한 번만 돈다 (메모리에서만)


def warmup():
    _warmup_once()
    _writer.get_graph()  # 작성 도우미 그래프를 미리 만들어 둠 (생성은 수십 ms)


# "제출서류 체크리스트를 만들어 줘" 는 writer 에 처리하는 길이 없어 일반 검색에서 "정보 없음"이 나왔다.
# 가이드·서식 말이 없을 때만 "목록을 표로 정리해 줘" 로 바꿔 검색에 넣는다 (질문 문장만 바꿈, 그 밖은 그대로).
_CHECKLIST = re.compile(
    r"체크\s?리스트(?:를|을)?(?:\s*(?:만들어|작성해|짜))?\s*(?:줘|주세요|줄래)?[.?!]?"
)
_KEEP_WORDS = ("가이드", "초안", "서식", "양식", "별지")


def _checklist_to_list(q):
    if not isinstance(q, str) or "체크" not in q or any(w in q for w in _KEEP_WORDS):
        return q
    return _CHECKLIST.sub("목록을 표로 정리해 줘.", q)


class _WriterGraph:
    async def ainvoke(self, inputs):
        inputs = dict(inputs)
        inputs["question"] = _checklist_to_list(inputs.get("question"))
        out = await _writer.get_graph().ainvoke(inputs)
        out = dict(out)
        out.setdefault("retrieved", [])
        return out


_graph = _WriterGraph()


def get_graph():
    return _graph
