"""kordoc으로 HWP/PDF를 Markdown 텍스트로 변환합니다.

실행:
  uv run python -m ai.rahui.convert_kordoc
  uv run python -m ai.rahui.convert_kordoc --limit 3
  uv run python -m ai.rahui.convert_kordoc --ocr

입력: data/files/
출력: data/kordoc_work/, data/processed_kordoc_raw/, data/processed_kordoc/
후처리:
  uv run python -m ai.rahui.preprocess --src processed_kordoc --out clean_kordoc
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ai.rahui.convert import DATA, FILES, read_rows

KORDOC = "kordoc@4.24.66"  # 결과를 다시 만들 수 있도록 버전을 고정 (2026-11-12 에 시험한 버전)


def clean(text: str) -> str:
    """이미지 링크와 칸 안 줄바꿈 태그만 지움 (검색에 잡음이 되는 것들)."""
    text = re.sub(r"^[ \t]*!\[[^\]]*\]\([^)]*\)[ \t]*$\n?", "", text, flags=re.M)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    return re.sub(r"<br\s*/?>", " ", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (시험용)")
    ap.add_argument("--only", default="", help="문서 번호만 쉼표로 (예: 8,13,40,51)")
    ap.add_argument(
        "--ocr", action="store_true", help="kordoc --ocr (이미지·스캔 글자 읽기)"
    )
    ap.add_argument("--work", default="kordoc_work")
    ap.add_argument("--raw", default="processed_kordoc_raw")
    ap.add_argument("--out", default="processed_kordoc")
    a = ap.parse_args()

    if not shutil.which("node"):
        sys.exit("node 가 없습니다. Node.js 20 이상을 설치하세요.")
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("npx 가 없습니다. Node.js 를 설치하면 함께 설치됩니다.")

    work, raw, out = DATA / a.work, DATA / a.raw, DATA / a.out
    for d in (work, raw, out):
        d.mkdir(parents=True, exist_ok=True)

    rows = read_rows()
    nos = list(range(1, len(rows) + 1))
    if a.only:
        nos = [int(x) for x in a.only.split(",")]
    if a.limit:
        nos = nos[: a.limit]

    env = {
        **os.environ,
        "KORDOC_OFFLINE": "1",
    }  # 외부 통신 차단 (문서가 밖으로 나가지 않음)
    log = []
    for no in nos:
        name = rows[no - 1]["파일명"]
        src, stem = FILES / name, Path(name).stem
        md = work / f"{no}.md"
        cmd = [npx, "--yes", KORDOC, str(src), "--table-format", "gfm", "-o", str(md)]
        if a.ocr:
            cmd.append("--ocr")
        t0 = time.time()
        p = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        sec = time.time() - t0
        ok = p.returncode == 0 and md.exists() and md.stat().st_size > 0
        chars = 0
        if ok:
            text = md.read_text(encoding="utf-8")
            (raw / f"{stem}.txt").write_text(text, encoding="utf-8", newline="\n")
            cleaned = clean(text)
            (out / f"{stem}.txt").write_text(cleaned, encoding="utf-8", newline="\n")
            chars = len(re.sub(r"[\W_]+", "", cleaned))
        log.append(
            {
                "문서번호": no,
                "성공": int(ok),
                "초": round(sec, 1),
                "공백기호제외글자": chars,
                "파일명": name,
                "오류": "" if ok else (p.stderr or p.stdout)[-200:].replace("\n", " "),
            }
        )
        print(
            f"{no:>3} {'OK ' if ok else 'FAIL'} {sec:5.1f}초 {chars:>8}자  {name[:40]}",
            flush=True,
        )

    with open(raw / "_convert_log.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    n_ok = sum(r["성공"] for r in log)
    print(
        f"\n변환 {len(log)}건 중 성공 {n_ok}건, 실패 {len(log) - n_ok}건, 총 {sum(r['초'] for r in log):.0f}초"
    )
    print(f"원본 {raw} / 정리본 {out} / 이미지 {work}")


if __name__ == "__main__":
    main()
