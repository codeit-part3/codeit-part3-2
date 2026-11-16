"""전처리 개선 실험: 검색용 색인 글(index)만 바꾸고, 점수 판정용 본문(body)은 그대로 둔다.

플래그(대문자 조합, 예: 'TBC')
  T  표 줄을 '머리칸: 값' 문장으로 풀어 색인에 넣음
  B  청크 앞 메타 헤더에 상위 절 제목 경로(예: 3 > 3.1 > 평가기준)를 붙임
  C  문서 안에서 5번 이상 되풀이되는 짧은 줄(머리글·바닥글)을 본문에서 뺌
"""

import re
from collections import Counter

from ai.data_basic.experiments.chunk_compare import heading_text

_DEPTH = [
    (1, re.compile(r"^\s*(제\s*\d+\s*[장편부]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|[IVX]+\s*[.)])")),
    (3, re.compile(r"^\s*(\d+\s*[.-]\s*\d+|[가-하]\s*[.)])")),
    (4, re.compile(r"^\s*(\(\s*\d+\s*\)|\d+\s*\)|[①-⑩])")),
    (2, re.compile(r"^\s*(\d+\s*[.]|[□■◆◇])")),
    (5, re.compile(r"^\s*[○●◦ㅇ·-]")),
]


def depth(title: str) -> int:
    for d, rx in _DEPTH:
        if rx.search(title):
            return d
    return 6


def drop_repeated(text: str, min_count: int = 5) -> str:
    """머리글·바닥글처럼 되풀이되는 짧은 줄을 지움. 제목 줄·표 줄·숫자만 있는 줄은 남김."""

    def norm(line):
        return re.sub(r"\s+", " ", line).strip()

    lines = text.split("\n")
    cnt = Counter(norm(ln) for ln in lines if 6 <= len(norm(ln)) <= 80)
    out = []
    for ln in lines:
        n = norm(ln)
        if cnt.get(n, 0) >= min_count and "|" not in ln and not heading_text(ln):
            continue
        out.append(ln)
    return "\n".join(out)


def verbalize(body: str) -> str:
    """연속된 표 줄 묶음의 첫 줄을 머리칸으로 보고, 칸 수가 같은 줄을 '머리: 값; ...'으로 바꿈."""
    lines, out, i = body.split("\n"), [], 0
    while i < len(lines):
        if "|" not in lines[i]:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and "|" in lines[j]:
            j += 1
        block = [[c.strip() for c in ln.split("|") if c.strip()] for ln in lines[i:j]]
        head = block[0]
        head_ok = (
            len(head) >= 2
            and all(len(h) <= 20 and not re.fullmatch(r"[\d.,%\s-]+", h) for h in head)
            and len(block) >= 2
        )
        out.append(" ; ".join(head) if head_ok else " ; ".join(head))
        for cells in block[1:]:
            if head_ok and len(cells) == len(head):
                out.append(
                    "; ".join(f"{h}: {v}" for h, v in zip(head, cells, strict=False))
                )
            else:
                out.append(" ; ".join(cells))
        i = j
    return "\n".join(out)


def chunk_x(text: str, target: int, flags: str) -> list[tuple[str, str, str]]:
    """chunk_struct와 같은 규칙으로 자르되 (절 제목 경로, 본문, 색인용 본문)을 돌려줌."""
    if "C" in flags:
        text = drop_repeated(text)
    max_sz = int(target * 1.5)
    out, cur, cur_len, cur_bc, stack, prev_tbl = [], [], 0, "", [], False

    def bc_now() -> str:
        if "B" in flags:
            s = " > ".join(t for _, t in stack)
        else:
            s = stack[-1][1] if stack else ""
        return s[:120]

    def flush():
        nonlocal cur, cur_len
        body = "\n".join(cur).strip()
        if body:
            out.append((cur_bc, body, verbalize(body) if "T" in flags else body))
        cur, cur_len = [], 0

    for line in text.split("\n"):
        h = heading_text(line)
        is_tbl = "|" in line
        if h:
            if cur_len >= target * 0.4:
                flush()
            d = depth(h)
            while stack and stack[-1][0] >= d:
                stack.pop()
            stack.append((d, h))
        if (
            cur_len > 0
            and cur_len + len(line) + 1 > target
            and not (is_tbl and prev_tbl and cur_len + len(line) + 1 <= max_sz)
        ):
            flush()
        if not cur:
            cur_bc = bc_now()
        cur.append(line)
        cur_len += len(line) + 1
        prev_tbl = is_tbl
    flush()
    return out
