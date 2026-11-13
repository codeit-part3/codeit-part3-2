"""문서 제목 추출 규칙을 표본으로 검증합니다.

실행:
  uv run python -m ai.rahui.eda_validate
  uv run python -m ai.rahui.eda_validate --score

입력: data/processed/*.txt, 채점 시트(--score 실행 시)
출력: ai/rahui/eda_results/의 검증 폴더
"""

import argparse
import csv
import random
import re
import statistics as st
from datetime import datetime
from pathlib import Path

from ai.rahui.eda_stats import (
    PAGE_NO,
    SECTIONS,
    heading_cell,
    load_docs,
    strict_heading,
)

OUT_ROOT = Path(__file__).resolve().parent / "eda_results"
FORCE = [65, 93]  # 앞쪽 가짜 일치가 알려진 문서는 항상 포함


def loose_hit(text: str, pat: str):
    rx = re.compile(pat)
    for i, line in enumerate(text.split("\n")):
        b = line.strip()
        if b and len(b) <= 60 and rx.search(b) and not PAGE_NO.search(b):
            return i, b
    return None


def strict_hit(text: str, pat: str):
    h = strict_heading(
        text, pat
    )  # 엄격 판정 본체는 eda_stats.py 에 있음 (실제 EDA와 같은 코드)
    return (h[0], h[1]) if h else None


def rel_pos(text: str, line_no: int) -> float:
    lines = text.split("\n")
    return 100 * sum(len(x) + 1 for x in lines[:line_no]) / max(len(text), 1)


ANY = re.compile(r".")


def outline(text: str) -> str:
    """번호·기호로 시작하는 짧은 줄만 모아 문서 뼈대를 보여 줌 (사람이 채점할 때 참고)."""
    out = []
    for i, line in enumerate(text.split("\n")):
        b = line.strip()
        if b and len(b) <= 120 and heading_cell(b, ANY, i):
            out.append(f"{i:>5}: {b}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument(
        "--n", type=int, default=12, help="표본 문서 수(강제 포함 문서 포함)"
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--dir", help="--score 때 시트가 있는 폴더(생략하면 가장 최근 검증_ 폴더)"
    )
    a = ap.parse_args()

    if a.score:
        score(Path(a.dir) if a.dir else sorted(OUT_ROOT.glob("*_검증"))[-1])
        return

    docs = [d for d in load_docs() if d["text"]]
    by_no = {d["no"]: d for d in docs}
    rng = random.Random(a.seed)
    pool = [n for n in by_no if n not in FORCE]
    pick = [n for n in FORCE if n in by_no] + rng.sample(pool, max(0, a.n - len(FORCE)))
    run = OUT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_검증"
    (run / "outline").mkdir(parents=True)

    # --- 1) 전체 100건: 느슨 vs 엄격 비교
    print("전체 문서 기준: 느슨 vs 엄격 (제목 줄 판정)")
    print(
        f"{'항목':<8}{'느슨 문서수':>10}{'엄격 문서수':>10}{'위치 차이 20%p↑ 문서':>22}{'느슨 중앙값':>12}{'엄격 중앙값':>12}"
    )
    cmp_rows = []
    for key, pat in SECTIONS.items():
        ln = sn = 0
        lp, sp, big = [], [], []
        for d in docs:
            lo, s = loose_hit(d["text"], pat), strict_hit(d["text"], pat)
            ln += bool(lo)
            sn += bool(s)
            lpos = rel_pos(d["text"], lo[0]) if lo else None
            spos = rel_pos(d["text"], s[0]) if s else None
            if lpos is not None:
                lp.append(lpos)
            if spos is not None:
                sp.append(spos)
            if lpos is not None and spos is not None and spos - lpos >= 20:
                big.append(d["no"])
            cmp_rows.append(
                [
                    d["no"],
                    key,
                    lo[0] if lo else "",
                    lo[1] if lo else "",
                    f"{lpos:.1f}" if lo else "",
                    s[0] if s else "",
                    s[1] if s else "",
                    f"{spos:.1f}" if s else "",
                ]
            )
        ml = f"{st.median(lp):.0f}%" if lp else "-"
        ms = f"{st.median(sp):.0f}%" if sp else "-"
        print(f"{key:<8}{ln:>10}{sn:>10}{len(big):>22}{ml:>12}{ms:>12}   {big[:8]}")
    with (run / "비교_전체.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "문서",
                "항목",
                "느슨_줄",
                "느슨_내용",
                "느슨_위치%",
                "엄격_줄",
                "엄격_내용",
                "엄격_위치%",
            ]
        )
        w.writerows(cmp_rows)

    # --- 2) 채점용 시트
    with (run / "채점시트.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "문서",
                "항목",
                "느슨_줄:내용",
                "엄격_줄:내용",
                "느슨_맞음(O/X)",
                "엄격_맞음(O/X)",
                "실제_있음(O/X)",
                "메모",
            ]
        )
        for n in pick:
            t = by_no[n]["text"]
            (run / "outline" / f"doc{n}.txt").write_text(outline(t), encoding="utf-8")
            for key, pat in SECTIONS.items():
                lo, s = loose_hit(t, pat), strict_hit(t, pat)
                w.writerow(
                    [
                        n,
                        key,
                        f"{lo[0]}:{lo[1]}" if lo else "(없음)",
                        f"{s[0]}:{s[1]}" if s else "(없음)",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
    print(f"\n표본 문서 {len(pick)}건: {pick}")
    print(f"폴더: {run}")
    print(
        "채점시트.csv 의 '느슨_맞음/엄격_맞음/실제_있음' 칸을 O/X 로 채우세요 (outline/ 폴더의 뼈대 참고)."
    )
    print(
        "  - 느슨_맞음: 그 줄이 진짜 그 항목의 제목 줄이면 O, 아니면 X, 줄이 (없음)이면 비워 둠"
    )
    print(
        "  - 실제_있음: 문서에 그 항목 절이 실제로 있으면 O, 없으면 X (판정 줄이 (없음)이어도 반드시 기입)"
    )
    print("다 채운 뒤: uv run python -m ai.rahui.eda_validate --score")


def score(run: Path) -> None:
    rows = list(csv.DictReader((run / "채점시트.csv").open(encoding="utf-8-sig")))
    rows = [r for r in rows if r["실제_있음(O/X)"].strip().upper() in ("O", "X")]
    if not rows:
        print("채워진 '실제_있음' 칸이 없습니다.")
        return
    lines = [f"채점 행 {len(rows)}개 (문서 {len({r['문서'] for r in rows})}건)", ""]
    for name, col, hitcol in (
        ("느슨", "느슨_맞음(O/X)", "느슨_줄:내용"),
        ("엄격", "엄격_맞음(O/X)", "엄격_줄:내용"),
    ):
        tp = sum(
            1 for r in rows if r[hitcol] != "(없음)" and r[col].strip().upper() == "O"
        )
        fp = sum(
            1 for r in rows if r[hitcol] != "(없음)" and r[col].strip().upper() == "X"
        )
        fn = sum(
            1
            for r in rows
            if r[hitcol] == "(없음)" and r["실제_있음(O/X)"].strip().upper() == "O"
        )
        # 문서 단위 '있다/없다' 판정: 첫 줄이 틀려도 실제로 있으면 있음 판정은 맞는 것으로 봄
        pres_tp = sum(
            1
            for r in rows
            if r[hitcol] != "(없음)" and r["실제_있음(O/X)"].strip().upper() == "O"
        )
        pres_fp = sum(
            1
            for r in rows
            if r[hitcol] != "(없음)" and r["실제_있음(O/X)"].strip().upper() == "X"
        )
        p = tp / (tp + fp) if tp + fp else float("nan")
        pp = pres_tp / (pres_tp + pres_fp) if pres_tp + pres_fp else float("nan")
        rc = pres_tp / (pres_tp + fn) if pres_tp + fn else float("nan")
        lines.append(
            f"[{name}] 첫 줄 정확도 {p:.0%} (맞음 {tp} / 틀림 {fp}) | 있음 판정 정밀도 {pp:.0%} | 재현율 {rc:.0%} (놓침 {fn})"
        )
    bad = [
        f"  문서{r['문서']} {r['항목']}: 느슨 {r['느슨_줄:내용']} / 엄격 {r['엄격_줄:내용']}"
        for r in rows
        if "X"
        in (r["느슨_맞음(O/X)"].strip().upper(), r["엄격_맞음(O/X)"].strip().upper())
    ]
    lines += ["", "틀린 사례:"] + bad
    txt = "\n".join(lines)
    (run / "채점결과.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n저장: {run / '채점결과.txt'}")


if __name__ == "__main__":
    main()
