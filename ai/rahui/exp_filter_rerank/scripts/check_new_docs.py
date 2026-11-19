"""처음 보는 공고로 평가표 추출을 점검한다 (모델 없이 규칙만 사용, 몇 초면 끝난다).

    uv run python ai/rahui/exp_filter_rerank/scripts/check_new_docs.py <새 공고 텍스트(.txt) 폴더>

- 폴더 안의 *.txt 마다 평가표 추출(writer.extract_items)을 돌려 판정(확정/추정/부분/없음)·항목 수·배점 합을 낸다.
- 결과: outputs/new_docs_check_<시각>.csv (원문 대조용 빈 칸 포함) 와 _항목.txt (추출한 항목 전체).
  공고문 내용이 들어 있으니 로컬 전용이며 깃에 올리지 않는다.
- 기존 98곳(data/clean_mixed_ocr_v2)이 섞이지 않게, 그 폴더를 주면 중단한다.
- 사람이 할 일: _항목.txt 를 원문 평가표와 한 줄씩 비교하고 CSV 의 '대조'(정확/일부 누락/오류)와 '비고'를 채운다.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ai.rahui.exp_filter_rerank.rahui_engine import writer as w  # noqa: E402

OUT_DIR = Path(os.environ.get("CHECK_OUT_DIR") or ROOT / "outputs")

# 독립 점검: 추출기와 별개로 원문의 '배점' 표에서 마지막 칸이 점수인 줄(소계·합계 제외)을 세어 둔다.
# 추출한 항목 수가 이 줄 수보다 많이 적으면 의심으로 표시한다 (정답이 아니라 사람이 볼 곳을 알려 주는 용도).
_NUM = re.compile(r"^\(?(\d{1,3}(?:\.\d+)?)\)?점?$")
_SUBTOTAL = re.compile(r"소\s*계|합\s*계|총\s*점|총\s*계|^\s*계\s*$|^계\b")


def _score_blocks(txt: str):
    cur = None
    for line in txt.split("\n"):
        if line.startswith("[표]"):
            cur = []
        elif line.startswith("[/표]"):
            if cur is not None:
                yield cur
            cur = None
        elif cur is not None:
            cur.append(line)


def source_score_rows(txt: str) -> tuple[int, float]:
    """원문에서 점수가 적힌 줄 수와 그 합 (가장 줄이 많은 '배점' 표 기준)."""
    best = (0, 0.0)
    for block in _score_blocks(txt):
        if not block or "배점" not in block[0]:
            continue
        n, total = 0, 0.0
        for line in block[1:]:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 2:
                continue
            if _SUBTOTAL.search(" ".join(cells[:-1])) or _SUBTOTAL.search(cells[-1]):
                continue
            m = _NUM.match(cells[-1])
            if m:
                n += 1
                total += float(m.group(1))
        if n > best[0]:
            best = (n, total)
    return best


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1]).expanduser().resolve()
    files = sorted(folder.glob("*.txt")) if folder.is_dir() else []
    if not files:
        print(f"*.txt 가 없습니다: {folder}")
        return 2
    if folder == Path(w.CORPUS_DIR).resolve():
        print("기존 98곳 폴더입니다. 처음 보는 공고만 따로 모은 폴더를 주세요.")
        return 2
    known = {p.name for p in Path(w.CORPUS_DIR).glob("*.txt")}
    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"new_docs_check_{stamp}.csv"
    txt_path = OUT_DIR / f"new_docs_check_{stamp}_항목.txt"

    rows, counts, lines = [], Counter(), []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            items, level, why = w.extract_items(text)
        except Exception as e:  # 한 파일이 깨져도 나머지는 계속
            items, level, why = [], "오류", f"{type(e).__name__}: {e}"
        total = sum(i.get("score") or 0 for i in items)
        src_n, src_sum = source_score_rows(text)
        # 확정인데 추출 항목이 원문 점수 줄의 60% 미만이거나 5개 미만이면 의심
        suspect = (
            "의심"
            if level == "확정" and (len(items) < 0.6 * src_n or len(items) < 5)
            else ""
        )
        counts[level] += 1
        dup = "  (기존 98곳과 파일 이름이 같음)" if p.name in known else ""
        rows.append(
            {
                "파일": p.name + dup,
                "판정": level,
                "항목수": len(items),
                "배점합": total,
                "원문점수줄": src_n,
                "원문점수합": src_sum,
                "의심": suspect,
                "설명": why,
                "대조": "",  # 정확 / 일부 누락 / 오류 (사람이 채움)
                "비고": "",
            }
        )
        lines.append(f"=== {p.name}  [{level}] 항목 {len(items)}개 · 합 {total}")
        lines.append(f"    {why}")
        for k, i in enumerate(items, 1):
            lines.append(f"    {k:>2}. {i.get('name', '')}  ({i.get('score')})")
        lines.append("")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"{'파일':<40}{'판정':<6}{'항목':>4}{'합':>6}{'원문줄':>6}  의심")
    for r in rows:
        print(
            f"{r['파일'][:38]:<40}{r['판정']:<6}{r['항목수']:>4}{r['배점합']:>6}{r['원문점수줄']:>6}  {r['의심']}"
        )
    print(
        "\n요약:",
        " · ".join(f"{k} {v}" for k, v in counts.items()),
        f"(총 {len(files)}곳)",
    )
    print(f"CSV : {csv_path}\n항목: {txt_path}")
    print(
        "다음: 항목 파일을 원문 평가표와 비교해 CSV 의 '대조' 칸을 채우세요 (정확/일부 누락/오류)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
