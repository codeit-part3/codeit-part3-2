"""kordoc 변환본과 기존 변환본을 문서별로 비교합니다.

실행:
  uv run python -m ai.rahui.kordoc_compare
  uv run python -m ai.rahui.kordoc_compare --limit 5
  uv run python -m ai.rahui.kordoc_compare --make-mixed

입력: data/processed/, data/processed_kordoc_raw/, data/processed_kordoc/, data/clean_hybrid/, data/clean_kordoc/
출력: ai/rahui/eda_results/<실행 폴더>/ (results_by_doc.csv, summary.txt)
"""

import argparse
import csv
import re
import statistics as st
from datetime import datetime
from pathlib import Path

from ai.rahui import chunk_compare as cc
from ai.rahui.convert import DATA, read_rows

OLD, RAW, NEW = "processed", "processed_kordoc_raw", "processed_kordoc"
SH = 12  # 내용 일치를 볼 조각 길이(글자)
LOW_RATIO, HIGH_RATIO, LOW_COVER = (
    0.95,
    1.10,
    0.95,
)  # 검토 대상 기준: 글자 비율 / 기존->kordoc 일치율

NWS = re.compile(r"[\W_]+")
KEEP = re.compile(r"[^0-9A-Za-z가-힣]")
IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
OLD_TBL = re.compile(r"\[표\](.*?)\[/표\]", re.S)
HTML_TBL = re.compile(r"<table.*?</table>", re.S | re.I)


def nws(s: str) -> int:
    return len(NWS.sub("", s))


def shingles(s: str) -> set[int]:
    t = KEEP.sub("", s)
    return {hash(t[i : i + SH]) for i in range(len(t) - SH + 1)}


def kordoc_tables(text: str) -> tuple[int, int]:
    """(표 개수, 표 안 글자 수): HTML <table> 블록 + 마크다운 | 줄 묶음."""
    html = HTML_TBL.findall(text)
    rest = HTML_TBL.sub("", text)
    n, chars, inb = len(html), sum(nws(h) for h in html), False
    for ln in rest.split("\n"):
        b = ln.lstrip().startswith("|")
        if b:
            chars += nws(ln)
            if not inb:
                n += 1
        inb = b
    return n, chars


def old_tables(text: str) -> tuple[int, int]:
    ts = OLD_TBL.findall(text)
    return len(ts), sum(nws(t) for t in ts)


def long_lines(text: str) -> int:
    return sum(len(x) > 3000 for x in text.split("\n"))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def golden_check(clean_name: str) -> tuple[int, list[str], set[str]]:
    """chunk_compare 와 같은 방식으로, 그 말뭉치에서 평가에 쓰이는 문항 수와 제외된 문항."""
    cc.CLEAN = DATA / clean_name
    docs, rep = cc.load_corpus()
    qs, skipped = cc.load_questions(docs, rep)
    return len(qs), skipped, {q["id"] for q in qs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--make-mixed", action="store_true")
    a = ap.parse_args()
    rows = read_rows()
    nos = list(range(1, len(rows) + 1))
    if a.limit:
        nos = nos[: a.limit]
    log = {}
    lp = DATA / RAW / "_convert_log.csv"
    if lp.exists():
        with open(lp, encoding="utf-8-sig", newline="") as fh:
            log = {int(r["문서번호"]): r for r in csv.DictReader(fh)}

    res = []
    for no in nos:
        name = rows[no - 1]["파일명"]
        stem, fmt = Path(name).stem, Path(name).suffix.lower().lstrip(".")
        po, pr, pn = (DATA / d / f"{stem}.txt" for d in (OLD, RAW, NEW))
        if not (po.exists() and pn.exists()):
            res.append({"문서번호": no, "형식": fmt, "비고": "파일 없음"})
            continue
        old, raw, new = (p.read_text(encoding="utf-8") for p in (po, pr, pn))
        so, sn = shingles(old), shingles(new)
        ot, otc = old_tables(old)
        kt, ktc = kordoc_tables(new)
        no_, nn = nws(old), nws(new)
        res.append(
            {
                "문서번호": no,
                "형식": fmt,
                "변환성공": log.get(no, {}).get("성공", ""),
                "변환초": log.get(no, {}).get("초", ""),
                "기존글자": no_,
                "kordoc글자": nn,
                "글자비율": round(nn / no_, 3) if no_ else "",
                "기존줄": old.count("\n") + 1,
                "kordoc줄": new.count("\n") + 1,
                "기존긴줄": long_lines(old),
                "kordoc긴줄": long_lines(new),
                "기존표": ot,
                "kordoc표": kt,
                "기존표글자비율": round(otc / no_, 3) if no_ else "",
                "kordoc표글자비율": round(ktc / nn, 3) if nn else "",
                "kordoc이미지링크": len(IMG.findall(raw)),
                "기존_kordoc일치": round(len(so & sn) / len(so), 3) if so else "",
                "kordoc_기존일치": round(len(so & sn) / len(sn), 3) if sn else "",
                "비고": "",
            }
        )
    ok = [r for r in res if r.get("글자비율") != "" and "기존글자" in r]

    L = [f"kordoc 비교 ({len(ok)}건 비교, 비교 대상 폴더: {OLD} vs {NEW})", ""]
    n_fail = sum(1 for r in ok if str(r.get("변환성공")) == "0")
    L.append(
        f"변환 실패 {n_fail}건, 변환 시간 합계 {sum(float(r['변환초'] or 0) for r in ok):.0f}초"
        + ("" if log else " (변환 로그 없음)")
    )
    L += [
        "",
        f"{'구분':<8}{'건수':>5}{'기존글자':>12}{'kordoc글자':>12}{'비율':>8}{'비율중앙':>9}{'기존표':>8}{'kordoc표':>9}{'이미지링크':>10}{'기존>k일치(중앙)':>16}{'k>기존일치(중앙)':>16}",
    ]
    for lab, sel in (
        ("전체", ok),
        ("HWP", [r for r in ok if r["형식"] == "hwp"]),
        ("PDF", [r for r in ok if r["형식"] == "pdf"]),
    ):
        if not sel:
            continue
        so_, sn_ = sum(r["기존글자"] for r in sel), sum(r["kordoc글자"] for r in sel)

        def med(k, sel=sel):
            return st.median(r[k] for r in sel if r[k] != "")

        L.append(
            f"{lab:<8}{len(sel):>5}{so_:>12,}{sn_:>12,}{sn_ / so_:>8.3f}{med('글자비율'):>9.3f}{sum(r['기존표'] for r in sel):>8,}{sum(r['kordoc표'] for r in sel):>9,}{sum(r['kordoc이미지링크'] for r in sel):>10,}{med('기존_kordoc일치'):>16.3f}{med('kordoc_기존일치'):>16.3f}"
        )
    L += ["", "PDF 문서별:"]
    for r in ok:
        if r["형식"] == "pdf":
            L.append(
                f"  {r['문서번호']:>3}번 글자 {r['기존글자']:,} -> {r['kordoc글자']:,} ({r['글자비율']}), 표 {r['기존표']} -> {r['kordoc표']}, 이미지링크 {r['kordoc이미지링크']}, 일치 기존>k {r['기존_kordoc일치']} / k>기존 {r['kordoc_기존일치']}"
            )
    low = [r for r in ok if r["글자비율"] < LOW_RATIO]
    high = [r for r in ok if r["글자비율"] > HIGH_RATIO]
    lowc = [
        r for r in ok if r["기존_kordoc일치"] != "" and r["기존_kordoc일치"] < LOW_COVER
    ]
    L += [
        "",
        f"검토 대상 (기준: 글자비율 < {LOW_RATIO} 또는 > {HIGH_RATIO}, 기존->kordoc 일치율 < {LOW_COVER})",
        f"  kordoc 글자가 적음: {[(r['문서번호'], r['글자비율']) for r in low]}",
        f"  kordoc 글자가 많음: {[(r['문서번호'], r['글자비율']) for r in high]}",
        f"  기존 내용이 kordoc에 덜 있음: {[(r['문서번호'], r['기존_kordoc일치']) for r in lowc]}",
        "  (원인은 이 표만으로 알 수 없음: 해당 문서 몇 건을 원본과 열어서 눈으로 확인해야 함)",
    ]
    L += ["", "골든셋 (두 말뭉치에서 chunk_compare 가 평가에 쓰는 문항):"]
    got = {}
    for nm in ("clean_hybrid", "clean_kordoc"):
        if (DATA / nm / "_meta.csv").exists():
            n, sk, ids = golden_check(nm)
            got[nm] = (n, sk, ids)
            L.append(
                f"  {nm}: 평가 문항 {n}개, 제외 {len(sk)}개: {[s.split()[0] for s in sk]}"
            )
        else:
            L.append(f"  {nm}: 폴더 없음 (preprocess 를 먼저 실행)")
    if len(got) == 2:
        h, k = got["clean_hybrid"][2], got["clean_kordoc"][2]
        L.append(
            f"  기존에만 평가 가능: {sorted(h - k)} / kordoc 에만 평가 가능: {sorted(k - h)}"
        )
        L.append(
            "  평가 문항 집합이 다르면 두 말뭉치의 점수를 그대로 비교하면 안 됨 (같은 문항으로 다시 맞춰야 함)"
        )

    run = cc.RESULTS / f"{datetime.now():%Y%m%d_%H%M%S}_kordoc비교"
    run.mkdir(parents=True, exist_ok=True)
    cols = list(ok[0]) if ok else ["문서번호"]
    with open(run / "results_by_doc.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ok)
    L += ["", f"저장 폴더: {run}"]
    (run / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    if a.make_mixed:
        mixed = DATA / "processed_mixed"
        mixed.mkdir(exist_ok=True)
        for no in range(1, len(rows) + 1):
            name = rows[no - 1]["파일명"]
            src = (
                DATA
                / (NEW if name.lower().endswith(".pdf") else OLD)
                / f"{Path(name).stem}.txt"
            )
            (mixed / src.name).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
            )
        print(
            f"\n혼합 말뭉치 저장: {mixed} (HWP=기존, PDF=kordoc). 이어서: python -m ai.rahui.preprocess --src processed_mixed --out clean_mixed"
        )


if __name__ == "__main__":
    main()
