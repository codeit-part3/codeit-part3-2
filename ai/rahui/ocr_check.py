"""OCR 전후 텍스트에서 이미지 의존 문항의 근거를 확인합니다.

실행:
  uv run python -m ai.rahui.ocr_check processed_kordoc processed_kordoc_ocr
  uv run python -m ai.rahui.ocr_check processed_kordoc --extra kordoc_work/ocr

입력: data/<폴더>/*.txt, ai/rahui/rahui_golden_set.json
출력: 터미널 결과
"""

import argparse
import json
import re
from pathlib import Path

from ai.rahui.convert import DATA, read_rows

KEEP = re.compile(r"[^0-9A-Za-z가-힣]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--ids", default="q29,q30")
    ap.add_argument(
        "--extra",
        default="",
        help="data 아래 폴더. 그 안의 *.md(그림을 OCR 한 결과)를 근거 문서 본문 뒤에 붙여서 검사",
    )
    a = ap.parse_args()
    gold = {
        g["id"]: g
        for g in json.loads(
            (Path(__file__).parent / "rahui_golden_set.json").read_text(
                encoding="utf-8"
            )
        )
    }
    rows = read_rows()
    extra = ""
    if a.extra:
        extra = "\n".join(
            f.read_text(encoding="utf-8") for f in sorted((DATA / a.extra).glob("*.md"))
        )
        print(f"추가 OCR 텍스트: {a.extra} (글자 {len(KEEP.sub('', extra)):,})")
    for qid in a.ids.split(","):
        g = gold[qid]
        keys = [KEEP.sub("", x)[:30] for x in g["evidence_quote"].split(" / ")]
        print(f"\n{qid} (근거 문서 {g['evidence_docs']}, 인용 조각 {len(keys)}개)")
        for fol in a.folders:
            found, size = [], 0
            for d in g["evidence_docs"]:
                p = DATA / fol / f"{Path(rows[d - 1]['파일명']).stem}.txt"
                if not p.exists():
                    print(f"  {fol}: 문서 {d} 파일 없음")
                    continue
                t = p.read_text(encoding="utf-8") + extra
                size = len(KEEP.sub("", t))
                found += [k in KEEP.sub("", t) for k in keys]
            print(
                f"  {fol}{' (+OCR)' if extra else ''}: 근거 조각 {sum(found)}/{len(found)} 찾음 (본문 글자 {size:,})"
            )


if __name__ == "__main__":
    main()
