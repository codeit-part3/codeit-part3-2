"""OCR 블록 품질을 확인하고 필요하면 낮은 품질 블록을 제외합니다.

실행:
  uv run --no-sync python -m ai.rahui.ocr_quality
  uv run --no-sync python -m ai.rahui.ocr_quality --apply --out processed_mixed_ocr_clean

입력: data/<src>/, data/kordoc_work/ocr/_ocr_log.csv
출력: data/kordoc_work/ocr/_quality.csv, _quality_preview.txt, (apply) data/<out>/
"""

import argparse
import csv
import html
import re
from collections import Counter
from pathlib import Path

from ai.rahui.convert import DATA

HEAD = "[이미지 OCR]"
BLOCK = re.compile(r"^### (image_\d+\.\w+)[ \t]*$", re.M)
READ = re.compile(r"[가-힣A-Za-z0-9]")


def split_doc(text: str):
    """(본문, [(그림이름, OCR글)]) — OCR 구역이 없으면 빈 목록."""
    i = text.find("\n" + HEAD + "\n")
    if i < 0:
        return text, []
    body, tail = text[:i], text[i + len(HEAD) + 2 :]
    ms = list(BLOCK.finditer(tail))
    blocks = []
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(tail)
        blocks.append((m.group(1), tail[m.end() : end].strip("\n")))
    return body, blocks


TAG = re.compile(r"<[^>]+>")
HG = re.compile(r"[가-힣]+")


def plain(s: str) -> str:
    """HTML 표 태그·마크다운 표 기호를 벗겨 글자만 남김 (지표가 태그 양에 끌려가지 않게)."""
    s = html.unescape(TAG.sub(" ", s))
    s = re.sub(r"^\s*\|?\s*-{3,}[\s|:-]*$", " ", s, flags=re.M)
    return s.replace("|", " ")


def bigrams(s: str):
    for run in HG.findall(s):
        for i in range(len(run) - 1):
            yield run[i : i + 2]


def build_vocab(src: Path, min_count: int) -> set:
    c = Counter()
    for f in src.glob("*.txt"):
        t = f.read_text(encoding="utf-8")
        i = t.find("\n" + HEAD + "\n")
        c.update(bigrams(t if i < 0 else t[:i]))  # OCR 아닌 본문만으로 사전을 만듦
    return {k for k, v in c.items() if v >= min_count}


VOCAB: set = set()


def metrics(raw: str) -> dict:
    s = plain(raw)
    nows = re.sub(r"\s+", "", s)
    n = len(nows)
    if n == 0:
        return {
            "글자": 0,
            "읽기비율": 0,
            "한글자토큰": 1,
            "짧은줄": 1,
            "반복": 1,
            "한글이음새": "",
        }
    toks = s.split()
    lines = [ln for ln in s.splitlines() if ln.strip()]
    top = Counter(nows).most_common(1)[0][1]
    bg = list(bigrams(s))
    return {
        "한글이음새": round(sum(b in VOCAB for b in bg) / len(bg), 3)
        if len(bg) >= 10
        else "",
        "글자": n,
        "읽기비율": round(len(READ.findall(nows)) / n, 3),
        "한글자토큰": round(sum(len(t) == 1 for t in toks) / max(1, len(toks)), 3),
        "짧은줄": round(
            sum(len(re.sub(r"\s+", "", ln)) <= 2 for ln in lines) / max(1, len(lines)),
            3,
        ),
        "반복": round(top / n, 3),
    }


def bad(m: dict, a) -> str:
    why = []
    if m["글자"] < a.min_chars:
        why.append("글자적음")
    if m["읽기비율"] < a.min_read:
        why.append("읽기비율낮음")
    if m["한글이음새"] != "" and m["한글이음새"] < a.min_bigram:
        why.append("한글이음새낮음")
    if m["한글자토큰"] > a.max_single:
        why.append("한글자토큰많음")
    if m["짧은줄"] > a.max_short:
        why.append("짧은줄많음")
    if m["반복"] > a.max_repeat:
        why.append("반복")
    return "+".join(why)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="processed_mixed_ocr")
    ap.add_argument("--out", default="processed_mixed_ocr_clean")
    ap.add_argument("--work", default="kordoc_work/ocr")
    ap.add_argument(
        "--apply", action="store_true", help="기준 미만 블록을 뺀 폴더를 만듦"
    )
    ap.add_argument(
        "--preview", type=int, default=40, help="미리보기에 넣을 낮은 블록 수"
    )
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--min-read", type=float, default=0.6)
    ap.add_argument(
        "--min-bigram",
        type=float,
        default=0.75,
        help="블록의 한글 두 글자 조합 중 본문 사전에 있는 비율의 하한",
    )
    ap.add_argument(
        "--vocab-min",
        type=int,
        default=3,
        help="본문에서 이 횟수 이상 나온 두 글자 조합만 사전에 넣음",
    )
    ap.add_argument("--max-single", type=float, default=0.5)
    ap.add_argument("--max-short", type=float, default=0.5)
    ap.add_argument("--max-repeat", type=float, default=0.3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src, work = DATA / a.src, DATA / a.work
    work.mkdir(parents=True, exist_ok=True)
    VOCAB.update(build_vocab(src, a.vocab_min))
    print(f"본문 사전: 두 글자 조합 {len(VOCAB):,}개")
    rows, store = [], {}
    for f in sorted(src.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        body, blocks = split_doc(text)
        store[f.name] = (body, blocks, [])
        for name, ocr in blocks:
            m = metrics(ocr)
            why = bad(m, a)
            rows.append(
                {"파일": f.name[:30], "그림": name, **m, "기준미만": why, "_ocr": ocr}
            )
            store[f.name][2].append(why)
    if not rows:
        print("OCR 블록이 없습니다. --src 를 확인하세요.")
        return

    with open(work / "_quality.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=[k for k in rows[0] if k != "_ocr"], extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)
    n_bad = sum(bool(r["기준미만"]) for r in rows)
    tot = sum(r["글자"] for r in rows)
    bad_chars = sum(r["글자"] for r in rows if r["기준미만"])
    print(
        f"OCR 블록 {len(rows)}개, 글자 {tot:,}자. 기준 미만 {n_bad}개 ({bad_chars:,}자, {bad_chars / max(1, tot):.1%})"
    )
    from collections import Counter as C

    print(
        "이유별:",
        dict(C(x for r in rows if r["기준미만"] for x in r["기준미만"].split("+"))),
    )

    # 미리보기: 읽기비율 낮은 순 (기준 미만이 아니어도 경계선을 볼 수 있게 섞음)

    def key(r):
        return (r["한글이음새"] if r["한글이음새"] != "" else 1, r["읽기비율"])

    low = sorted(rows, key=key)[: a.preview]
    near = sorted([r for r in rows if not r["기준미만"]], key=key)[:15]
    with open(work / "_quality_preview.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "※ 본문이 들어 있으므로 공유 문서·채팅에 붙이지 마세요.\n\n== 기준 미만 후보 (읽기비율 낮은 순) ==\n"
        )
        for r in low:
            fh.write(
                f"\n--- {r['파일']} {r['그림']} 글자{r['글자']} 읽기{r['읽기비율']} 이음새{r['한글이음새']} 한글자{r['한글자토큰']} 짧은줄{r['짧은줄']} 반복{r['반복']} [{r['기준미만'] or '통과'}]\n{r['_ocr'][:400]}\n"
            )
        fh.write("\n\n== 통과한 것 중 경계선 (읽기비율 낮은 순) ==\n")
        for r in near:
            fh.write(
                f"\n--- {r['파일']} {r['그림']} 글자{r['글자']} 읽기{r['읽기비율']} 이음새{r['한글이음새']} 한글자{r['한글자토큰']} 짧은줄{r['짧은줄']} 반복{r['반복']}\n{r['_ocr'][:400]}\n"
            )
    print(
        f"지표: {work / '_quality.csv'}\n미리보기(로컬 전용): {work / '_quality_preview.txt'}"
    )

    if not a.apply:
        print(
            "\n미리보기를 확인한 뒤, 기준을 조정하고 --apply 로 적용하세요. (지금은 아무 폴더도 만들지 않음)"
        )
        return
    out = DATA / a.out
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(
            f"{out} 에 이미 결과가 있습니다. 다른 --out 을 쓰거나 --force 를 붙이세요."
        )
    out.mkdir(parents=True, exist_ok=True)
    kept_blocks = 0
    for name, (body, blocks, whys) in store.items():
        keep = [(n, o) for (n, o), w in zip(blocks, whys, strict=False) if not w]
        kept_blocks += len(keep)
        text = body
        if keep:
            text = (
                body
                + "\n"
                + HEAD
                + "\n"
                + "\n\n".join(f"### {n}\n{o}" for n, o in keep)
                + "\n"
            )
        elif not blocks:
            text = body if body.endswith("\n") else body + "\n"
        else:
            text = body.rstrip("\n") + "\n"
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    print(
        f"저장 {len(store)}건 -> {out} (OCR 블록 {len(rows)}개 중 {kept_blocks}개 유지, {len(rows) - kept_blocks}개 제거)"
    )


if __name__ == "__main__":
    main()
