"""pdfplumber로 PDF의 표와 본문을 텍스트로 변환합니다.

실행: python ai/rahui/convert_pdf_plumber.py <입력.pdf> <출력.txt>
입력: PDF 파일
출력: 지정한 TXT 파일
"""

import os
import re
import sys
import time

import pdfplumber
import pypdf


def cell_txt(c):
    return re.sub(r"\s+", " ", (c or "")).strip()


def table_block(rows):
    lines = []
    for r in rows:
        cells = [cell_txt(c) for c in r]
        if any(cells):
            lines.append(" | ".join(cells))
    return "[표]\n" + "\n".join(lines) + "\n[/표]"


def nws(s):
    return len(re.sub(r"[\W_]+", "", s or ""))


def page_text(page, ptext):
    W, H = page.width, page.height
    plain = page.extract_text() or ""
    tabs = []
    for t in page.find_tables():
        rows = t.extract()
        ncol = max((len(r) for r in rows), default=0)
        if len(rows) >= 2 and ncol >= 2:
            cellc = sum(nws(c) for r in rows for c in r)
            inside = nws(page.crop(t.bbox).extract_text() or "")
            if inside and cellc / inside >= 0.9:
                tabs.append((t.bbox, rows))
    if not tabs:
        return ptext, 0, "plain"
    tabs.sort(key=lambda x: x[0][1])
    out, cur = [], 0.0

    def crop_txt(y0, y1):
        y0, y1 = max(0, y0), min(H, y1)
        if y1 - y0 < 1:
            return ""
        return page.crop((0, y0, W, y1)).extract_text() or ""

    for bbox, rows in tabs:
        top, bot = bbox[1], bbox[3]
        if top > cur:
            s = crop_txt(cur, top)
            if s.strip():
                out.append(s)
        out.append(table_block(rows))
        cur = max(cur, bot)
    s = crop_txt(cur, H)
    if s.strip():
        out.append(s)
    res = "\n".join(out)
    if nws(res) < 0.95 * nws(plain):
        return ptext, 0, "fallback"
    return res, len(tabs), "table"


def convert(path, outpath):
    t0 = time.time()
    pages = []
    ntab = 0
    modes = {"plain": 0, "fallback": 0, "table": 0}
    rd = pypdf.PdfReader(path)
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            ptext = rd.pages[i].extract_text() or ""
            try:
                s, n, m = page_text(p, ptext)
                modes[m] += 1
            except Exception as e:
                s, n = ptext, 0
                modes["plain"] += 1
                print(f"  쪽 {i + 1} 오류 -> 일반 추출: {e}", flush=True)
            pages.append(s)
            ntab += n
            p.flush_cache()
    text = "\n".join(pages)
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(
        f"{os.path.basename(path)[:20]} 쪽 {len(pages)} 표 {ntab} 글자 {len(text)} {time.time() - t0:.0f}초 쪽유형 {modes}",
        flush=True,
    )


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
