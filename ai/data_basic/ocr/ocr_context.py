"""OCR 블록의 각 그림 아래에, 그 그림이 원래 있던 자리의 앞 문맥(절 제목 등)을 한 줄 붙입니다.

배경: ocr_images.py 는 그림 OCR 글자를 문서 맨 끝 [이미지 OCR] 구역에 모아 붙여서,
검색 청크에 "어느 절의 그림인지" 정보가 없습니다. kordoc 원문(이미지 링크가 남아 있는 processed_kordoc_raw)
에서 그림 바로 앞의 글줄을 찾아 '(앞 문맥: 1. 일반 현황 > 가. 교육이념 및 교육목표)' 한 줄로 붙입니다.

실행:
  uv run --no-sync python -m ai.data_basic.ocr.ocr_context --src processed_mixed_ocr_clean --out processed_mixed_ocr_ctx
후처리:
  uv run --no-sync python -m ai.data_basic.preprocessing.preprocess --src processed_mixed_ocr_ctx --out clean_mixed_ocr_ctx

입력: data/<src>/ (ocr_quality 결과), data/<raw>/ (kordoc 원문, 기본 processed_kordoc_raw)
출력: data/<out>/*.txt (그림 블록 제목 아래 한 줄만 추가, 나머지는 그대로)
"""

import argparse
import re
from pathlib import Path

from ai.data_basic.conversion.convert import DATA, read_rows
from ai.data_basic.ocr.ocr_quality import HEAD, split_doc

IMG = re.compile(r"!\[[^\]]*\]\(images/\d+/(image_\d+\.\w+)\)")
NWS = re.compile(r"[\W_]+")


def contexts(raw_text: str, n_lines: int = 2, max_len: int = 50) -> dict[str, str]:
    """그림 이름 -> 바로 앞의 글줄 n_lines 개 ('상위 > 하위' 순서)."""
    out, recent = {}, []
    for line in raw_text.splitlines():
        ms = list(IMG.finditer(line))
        if ms:  # 그림 링크가 한 줄에 단독이든 글·표 안에 섞였든, 링크 앞까지의 글을 문맥에 포함
            before = IMG.sub("", line[: ms[0].start()]).strip(" |\t")
            ctx = recent[-n_lines:] + (
                [re.sub(r"^#+\s*", "", before)[:max_len]]
                if len(NWS.sub("", before)) >= 4
                else []
            )
            for m in ms:
                out.setdefault(m.group(1), " > ".join(ctx[-n_lines:]))
            continue
        s = line.strip()
        if len(NWS.sub("", s)) < 4 or s.startswith("|") or s.startswith("---"):
            continue
        s = re.sub(r"^#+\s*", "", s)
        recent.append(s[:max_len])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True, help="OCR 블록이 있는 텍스트 폴더 (data 아래)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="processed_kordoc_raw")
    a = ap.parse_args()

    rows = read_rows()
    out = DATA / a.out
    out.mkdir(parents=True, exist_ok=True)
    n_docs = n_blocks = n_ctx = 0
    for r in rows:
        stem = Path(r["파일명"]).stem
        src = DATA / a.src / f"{stem}.txt"
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        body, blocks = split_doc(text)
        if blocks:
            rp = DATA / a.raw / f"{stem}.txt"
            ctx = contexts(rp.read_text(encoding="utf-8")) if rp.exists() else {}
            parts = []
            for name, ocr in blocks:
                n_blocks += 1
                c = ctx.get(name, "")
                n_ctx += bool(c)
                head = f"### {name}" + (f"\n(앞 문맥: {c})" if c else "")
                parts.append(f"{head}\n{ocr}")
            text = body.rstrip("\n") + "\n\n" + HEAD + "\n" + "\n\n".join(parts) + "\n"
            n_docs += 1
        (out / f"{stem}.txt").write_text(text, encoding="utf-8", newline="\n")
    print(
        f"저장 -> {out}: OCR 블록 문서 {n_docs}건, 그림 {n_blocks}개 중 문맥 붙임 {n_ctx}개"
    )


if __name__ == "__main__":
    main()
