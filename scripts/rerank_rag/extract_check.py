"""평가표 추출 점검 (읽기 전용, 파일 생성 없음)
실행: uv run python scripts/rerank_rag/extract_check.py
"""

import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "clean_mixed_ocr_v2"
TOT = ("계", "소계", "합계", "총계", "총점")
KEY = ("항목", "평가요소", "평가기준", "평가부문", "평가내용", "평가지표", "구분")
NUM = re.compile(r"^\(?(\d{1,3}(?:\.\d+)?)\)?점?$")


def sq(s):
    return re.sub(r"\s+", "", s)


def table_blocks(txt):
    """[표] 블록 + 줄 앞에 |가 붙은 마크다운 표 (빈 줄·구분선으로 잘린 조각은 이어 붙임)"""
    blocks = re.findall(r"\[표\]\n(.*?)\n\[/표\]", txt, flags=re.S)
    run = []

    def flush():
        if len(run) >= 3:
            blocks.append("\n".join(run))

    for ln in txt.split("\n"):
        if ln.startswith("|"):
            s = ln.strip().strip("|")
            if re.fullmatch(r"[\s|:\-]*", s):  # --- 구분선은 버림
                continue
            run.append(s)
        elif ln.strip():  # 다른 글이 나오면 표 끝
            flush()
            run.clear()
    flush()
    return blocks


def parse(path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for b in table_blocks(txt):
        lines = [ln for ln in b.split("\n") if ln.strip()]
        if not lines or "|" not in lines[0]:
            continue
        head = [sq(c) for c in lines[0].split("|")]
        cols = [i for i, h in enumerate(head) if "배점" in h]
        if not cols or not any(k in "".join(head) for k in KEY):
            continue
        j = cols[-1] - len(head)
        rows, total = [], None  # (배점열 값, 세부점수, 전체칸 여부)
        for ln in lines[1:]:
            cells = [sq(c) for c in ln.split("|")]
            if len(cells) < -j:
                continue
            m = NUM.match(cells[j])
            if not m:
                continue
            v = float(m.group(1))
            if any(c in TOT for c in cells):
                total = v
                continue
            full = len(cells) == len(head)
            sub = v
            if len(cols) >= 2:
                m2 = NUM.match(cells[cols[-2]] if full else cells[-1])
                sub = float(m2.group(1)) if m2 else v
            rows.append((v, sub, full))
        c1 = sum(r[0] for r in rows)  # 그대로 합
        c2 = sum(r[0] for r in rows if r[2])  # 전체 칸 행만 (계층 구조)
        c3 = sum(r[1] for r in rows)  # 세부점수 합
        run, cnt, c4 = 0.0, 0, 0.0  # 소계 행 제거
        for v, _, _ in rows:
            if cnt >= 2 and abs(v - run) < 0.01:
                run, cnt = 0.0, 0
                continue
            run += v
            cnt += 1
            c4 += v
        cs = [c1] + [x for x in dict.fromkeys((c2, c3, c4)) if x != c1]
        out.append((len(rows), cs, total))
    return out


def judge(tabs):
    tabs = [t for t in tabs if t[0] >= 2]
    if not tabs:
        return "FAIL_empty"
    if any(t[2] is not None and any(abs(c - t[2]) < 0.01 for c in t[1]) for t in tabs):
        return "PASS_total"
    if any(any(abs(c - 100) < 0.01 for c in t[1]) for t in tabs):
        return "PASS_100"
    if abs(sum(t[1][0] for t in tabs) - 100) < 0.01:
        return "PASS_split"
    if any(
        t[0] >= 5 and any(abs(c - w) < 0.01 for c in t[1] for w in (90, 80, 70))
        for t in tabs
    ):
        return "WEAK_90"  # 합계 줄은 없지만 90/80/70점대: 사람이 확인
    return "FAIL_sum"
    tabs = [t for t in tabs if t[0] >= 2]
    if not tabs:
        return "FAIL_empty"
    if any(t[2] is not None and abs(t[1] - t[2]) < 0.01 for t in tabs):
        return "PASS_total"  # 항목 합 = 표에 적힌 합계
    if any(abs(t[1] - 100) < 0.01 for t in tabs):
        return "PASS_100"  # 합계 줄 없지만 항목 합 100
    if abs(sum(t[1] for t in tabs) - 100) < 0.01:
        return "PASS_split"  # 여러 표를 합치면 100
    return "FAIL_sum"


def main():
    files = sorted(CORPUS.glob("*.txt"))
    res = {f.name: parse(f) for f in files}
    verdict = {n: ("NOHEADER" if not t else judge(t)) for n, t in res.items()}
    cnt = collections.Counter(verdict.values())
    print(f"전체 {len(files)}개")
    for k in (
        "PASS_total",
        "PASS_100",
        "PASS_split",
        "WEAK_90",
        "FAIL_sum",
        "FAIL_empty",
        "NOHEADER",
    ):
        print(f"  {k:11s} {cnt[k]:3d}")
    ok = cnt["PASS_total"] + cnt["PASS_100"] + cnt["PASS_split"]
    print(f"  => 합계 검증 통과 {ok}/{len(files)} ({ok / len(files) * 100:.0f}%)")
    for k in ("WEAK_90", "FAIL_sum", "FAIL_empty"):
        print(f"\n[{k}] (이름 앞 28자 | 표별 항목수/후보합들/표합계)")
        for n, v in verdict.items():
            if v == k:
                d = " ; ".join(
                    f"{a}/{'|'.join(format(x, 'g') for x in b)}/{c if c is None else format(c, 'g')}"
                    for a, b, c in res[n][:4]
                )
                print(f"  {n[:28]} | {d}")
    print("\n[NOHEADER]")
    for n, v in verdict.items():
        if v == "NOHEADER":
            print(" ", n[:40])


if __name__ == "__main__":
    main()
