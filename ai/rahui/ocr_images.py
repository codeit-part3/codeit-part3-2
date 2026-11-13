"""kordoc이 추출한 그림을 OCR하고 결과를 문서 텍스트에 추가합니다.

실행:
  uv run python -m ai.rahui.ocr_images --base processed_mixed --out processed_mixed_ocr --count
  uv run python -m ai.rahui.ocr_images --base processed_mixed --out processed_mixed_ocr --workers 2

입력: data/kordoc_work/images/, data/<base>/
출력: data/<out>/*.txt, data/kordoc_work/ocr/_ocr_log.csv
후처리:
  uv run python -m ai.rahui.preprocess --src processed_mixed_ocr --out clean_mixed_ocr
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai.rahui.convert import DATA, read_rows
from ai.rahui.convert_kordoc import KORDOC, clean

EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}  # bmp 는 kordoc 이 못 읽어서 png 로 바꿔 OCR (Pillow 필요: uv add pillow)
NWS = re.compile(r"[\W_]+")


def png_size(p: Path):
    with open(p, "rb") as f:
        h = f.read(24)
    return struct.unpack(">II", h[16:24]) if h[:8] == b"\x89PNG\r\n\x1a\n" else None


def collect(
    images_dir: Path, nos: list[int], min_bytes: int, min_side: int
) -> list[dict]:
    items = []
    for no in nos:
        d = images_dir / str(no)
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.name == "manifest.json":
                continue
            size, ext = f.stat().st_size, f.suffix.lower()
            dims = png_size(f) if ext == ".png" else None
            if ext == ".bmp" and ext in EXTS:
                try:
                    from PIL import Image

                    with Image.open(f) as im:
                        dims = im.size
                except Exception:
                    dims = None
            why = ""
            if ext not in EXTS:
                why = "형식 미지원"
            elif (
                ext == ".bmp"
            ):  # bmp 는 압축이 없어 용량 기준이 무의미 -> 픽셀 크기로만 거름
                if dims is None:
                    why = "bmp 읽기 실패"
                elif min(dims) < min_side:
                    why = f"작음({dims[0]}x{dims[1]})"
            elif size < min_bytes:
                why = f"작음({size}B)"
            elif dims and min(dims) < min_side:
                why = f"작음({dims[0]}x{dims[1]})"
            items.append(
                {
                    "no": no,
                    "path": f,
                    "name": f.name,
                    "bytes": size,
                    "w": dims[0] if dims else "",
                    "h": dims[1] if dims else "",
                    "skip": why,
                    "md5": hashlib.md5(f.read_bytes()).hexdigest(),
                }
            )
    return items


def ocr_one(npx: str, img: Path, out: Path, timeout: int, offline: bool):
    env = {**os.environ, **({"KORDOC_OFFLINE": "1"} if offline else {})}
    t0 = time.time()
    p = subprocess.run(
        [npx, "--yes", KORDOC, str(img), "-o", str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    ok = out.exists() and (
        p.returncode == 0 or "OK" in (p.stdout + p.stderr)
    )  # 글자가 없는 그림은 빈 파일이 나옴 (정상)
    return (
        ok,
        time.time() - t0,
        "" if ok else (p.stderr or p.stdout)[-200:].replace("\n", " "),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=True,
        help="기존 텍스트 폴더 (data 아래, 예: processed_mixed)",
    )
    ap.add_argument(
        "--out", required=True, help="결과 폴더 (data 아래, 예: processed_mixed_ocr)"
    )
    ap.add_argument("--images", default="kordoc_work/images")
    ap.add_argument("--work", default="kordoc_work/ocr")
    ap.add_argument("--only", default="", help="문서 번호만 쉼표로")
    ap.add_argument(
        "--min-kb", type=float, default=30, help="이보다 작은 파일은 건너뜀 (기본 30KB)"
    )
    ap.add_argument(
        "--min-side",
        type=int,
        default=200,
        help="PNG 가로·세로 중 작은 쪽이 이보다 작으면 건너뜀 (기본 200px)",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="OCR 결과 글자(공백·기호 제외)가 이보다 적으면 붙이지 않음",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--count", action="store_true", help="작업량만 보고 끝냄")
    a = ap.parse_args()

    rows = read_rows()
    nos = (
        [int(x) for x in a.only.split(",")] if a.only else list(range(1, len(rows) + 1))
    )
    items = collect(DATA / a.images, nos, int(a.min_kb * 1024), a.min_side)
    todo = [i for i in items if not i["skip"]]
    uniq: dict[str, dict] = {}
    for i in todo:
        uniq.setdefault(i["md5"], i)
    work = DATA / a.work
    cache = work / "_by_md5"
    pending = [i for m, i in uniq.items() if not (cache / f"{m}.md").exists()]
    skipped: dict[str, int] = {}
    for i in items:
        if i["skip"]:
            k = i["skip"].split("(")[0]
            skipped[k] = skipped.get(k, 0) + 1
    print(
        f"문서 {len({i['no'] for i in items})}건의 그림 {len(items)}개 -> OCR 대상 {len(todo)}개 (내용이 같은 것 제외 {len(uniq)}개), 이미 한 것 제외 남은 것 {len(pending)}개"
    )
    print(
        f"건너뜀: {skipped or '없음'}, 남은 것 용량 {sum(i['bytes'] for i in pending) / 1e6:,.0f}MB"
    )
    if a.count:
        return
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("npx 가 없습니다. Node.js 20 이상을 설치하세요.")
    cache.mkdir(parents=True, exist_ok=True)

    fail: dict[str, str] = {}
    secs: dict[str, float] = {}
    t_all = time.time()

    def job(i):
        src = i["path"]
        if src.suffix.lower() == ".bmp":
            from PIL import Image

            png = work / "_png" / f"{i['md5']}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            if not png.exists():
                with Image.open(src) as im:
                    im.convert("RGB").save(png)
            src = png
        ok, sec, err = ocr_one(npx, src, cache / f"{i['md5']}.md", a.timeout, a.offline)
        secs[i["md5"]] = sec
        if not ok:
            fail[i["md5"]] = err
        return i, ok, sec

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for n, (i, ok, sec) in enumerate(ex.map(job, pending), 1):
            print(
                f"  {n}/{len(pending)} 문서{i['no']} {i['name']} {'OK ' if ok else 'FAIL'} {sec:.1f}초",
                flush=True,
            )
    if pending:
        print(f"OCR {len(pending)}개 {time.time() - t_all:.0f}초, 실패 {len(fail)}개")

    out = DATA / a.out
    out.mkdir(parents=True, exist_ok=True)
    log, n_docs, n_added = [], 0, 0
    by_doc: dict[int, list[dict]] = {}
    for i in items:
        by_doc.setdefault(i["no"], []).append(i)
    for no in nos:
        stem = Path(rows[no - 1]["파일명"]).stem
        src = DATA / a.base / f"{stem}.txt"
        if not src.exists():
            print(f"  경고: {src.name} 없음 (문서 {no})")
            continue
        text, pieces, seen = src.read_text(encoding="utf-8"), [], set()
        for i in by_doc.get(no, []):
            f = cache / f"{i['md5']}.md"
            state, nch = i["skip"] and f"건너뜀:{i['skip']}", 0
            if not i["skip"]:
                if i["md5"] in fail:
                    state = "실패"
                elif i["md5"] in seen:
                    state = "같은 그림(문서 안 중복)"
                elif f.exists():
                    ocr = clean(f.read_text(encoding="utf-8")).strip()
                    nch = len(NWS.sub("", ocr))
                    seen.add(i["md5"])
                    if nch == 0:
                        state = "글자 없음"
                    elif nch >= a.min_chars:
                        pieces.append(f"### {i['name']}\n{ocr}")
                        state = "붙임"
                    else:
                        state = f"글자 적음({nch})"
                else:
                    state = "결과 없음"
            log.append(
                {
                    "문서번호": no,
                    "그림": i["name"],
                    "바이트": i["bytes"],
                    "가로": i["w"],
                    "세로": i["h"],
                    "상태": state,
                    "OCR글자": nch,
                    "초": round(secs.get(i["md5"], 0), 1),
                    "오류": fail.get(i["md5"], ""),
                }
            )
        if pieces:
            text = text.rstrip("\n") + "\n\n[이미지 OCR]\n" + "\n\n".join(pieces) + "\n"
            n_added += 1
        (out / f"{stem}.txt").write_text(text, encoding="utf-8", newline="\n")
        n_docs += 1
    work.mkdir(parents=True, exist_ok=True)
    if log:
        with open(work / "_ocr_log.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(log)
    print(
        f"\n저장 {n_docs}건 -> {out} (OCR 텍스트를 붙인 문서 {n_added}건). 기록: {work / '_ocr_log.csv'}"
    )


if __name__ == "__main__":
    main()
