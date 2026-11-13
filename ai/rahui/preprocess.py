"""변환된 TXT를 정리하고 문서 메타데이터를 만듭니다.

실행:
  uv run python -m ai.rahui.preprocess
  uv run python -m ai.rahui.preprocess --src processed_kordoc --out clean_kordoc

입력: data/<src>/*.txt (기본 processed)
출력: data/<out>/*.txt, data/<out>/_meta.csv
"""

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

from ai.rahui.convert import DATA, read_rows

LONG_LINE = 3000  # 이 길이를 넘는 줄만 나눔
PIECE = 1000  # 나눈 조각의 최대 길이
TABLE = re.compile(r"\[표\](.*?)\[/표\]", re.DOTALL)

# 문서번호: (정제 금액, 비고). None 이면 금액 없음으로 처리.
AMOUNT_FIX: dict[int, tuple[int | None, str]] = {
    13: (242_900_000, "본문에서 확인한 값으로 보완"),
    14: (
        400_000_000,
        "본문 총액(VAT 포함). 1·2차년도 각 2억 이내, 2차년도는 1차년도 평가 후 결정",
    ),
    17: (470_000_000, "본문 470백만원(VAT 포함)"),
    35: (843_000_000, "개발비 359백만원 + H/W 484백만원(VAT 포함)"),
    42: (None, "비공개 입찰 (사업예산 비공개)"),
    72: (
        None,
        "메타 0원. 본문에 정확한 금액 없음, '사업예산 20억 미만'(상한)만 있어 보정하지 않음",
    ),
    84: (None, "메타 1원은 입력 오류, 본문은 사업예산 미포함"),
}


def norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


def drop_empty_tables(text: str) -> tuple[str, int]:
    removed = 0

    def repl(m: re.Match) -> str:
        nonlocal removed
        if re.sub(r"[\s|]", "", m.group(1)):
            return m.group(0)
        removed += 1
        return ""

    return TABLE.sub(repl, text), removed


def split_long(line: str) -> list[str]:
    """3000자 넘는 한 줄을 문장 -> 표 칸 -> 글자 수 순으로 잘라 PIECE 이하 조각들로."""
    parts = re.split(r"(?<=[.!?])\s+", line)
    if len(parts) == 1:
        parts = re.split(r"(?<= \|) ", line)
    out, buf = [], ""
    for p in parts:
        while len(p) > PIECE:  # 그래도 길면 글자 수로 자름
            if buf:
                out.append(buf)
                buf = ""
            out.append(p[:PIECE])
            p = p[PIECE:]
        if buf and len(buf) + len(p) + 1 > PIECE:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip() if buf else p
    if buf:
        out.append(buf)
    return out


def clean_text(text: str) -> tuple[str, int, int]:
    text, n_empty = drop_empty_tables(text)
    lines, n_split = [], 0
    for ln in text.split("\n"):
        ln = ln.rstrip()
        if len(ln) > LONG_LINE:
            lines.extend(split_long(ln))
            n_split += 1
        else:
            lines.append(ln)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"
    return out, n_empty, n_split


def main() -> None:
    ap = argparse.ArgumentParser(
        description="전처리: data/<src>/*.txt -> data/<out>/*.txt + _meta.csv"
    )
    ap.add_argument(
        "--src", default="processed", help="data/ 아래 입력 폴더 이름 (기본 processed)"
    )
    ap.add_argument(
        "--out", default="clean", help="data/ 아래 출력 폴더 이름 (기본 clean)"
    )
    ap.add_argument(
        "--force", action="store_true", help="출력 폴더에 이미 결과가 있어도 덮어씀"
    )
    a = ap.parse_args()
    SRC, CLEAN = DATA / a.src, DATA / a.out
    if not SRC.is_dir():
        sys.exit(f"입력 폴더가 없습니다: {SRC}")
    if SRC.resolve() == CLEAN.resolve():
        sys.exit("입력과 출력 폴더가 같습니다.")
    if CLEAN.is_dir() and any(CLEAN.glob("*.txt")) and not a.force:
        sys.exit(
            f"{CLEAN} 에 이미 결과가 있어 멈춥니다. 다른 이름을 --out 으로 주거나, 덮어쓰려면 --force 를 붙이세요."
        )

    rows = read_rows()
    CLEAN.mkdir(parents=True, exist_ok=True)

    def val(n: int, col: str) -> str:
        """문서번호 n 의 data_list.csv 값 (앞뒤 공백 제거)."""
        return (rows[n - 1].get(col) or "").strip()

    texts: dict[int, str] = {}
    for no, row in enumerate(rows, 1):
        p = SRC / f"{Path(row['파일명']).stem}.txt"
        if p.exists():
            texts[no] = p.read_text(encoding="utf-8")

    # 본문이 같은 문서 묶기: 번호가 작은 쪽이 대표
    groups: dict[str, list[int]] = {}
    for no, t in texts.items():
        key = hashlib.sha1(norm(t).encode(), usedforsecurity=False).hexdigest()
        groups.setdefault(key, []).append(no)
    rep_of: dict[int, int] = {}
    for g in groups.values():
        for no in g:
            rep_of[no] = min(g)

    meta, kept = [], 0
    for no, row in enumerate(rows, 1):
        if no not in texts:
            continue
        rep = rep_of[no]
        same = sorted(n for n, r in rep_of.items() if r == rep)
        multi = len(same) > 1  # 본문이 같은 문서가 둘 이상인 묶음
        orgs = sorted({val(n, "발주 기관") for n in same})
        names = sorted({val(n, "사업명") for n in same})
        # 문서마다 다를 수 있는 값은 '문서번호:값' 으로 모두 남김 (값이 있는 문서만)
        ids = [f"{n}:{val(n, '공고 번호')}" for n in same if val(n, "공고 번호")]
        pubs = [f"{n}:{val(n, '공개 일자')[:10]}" for n in same if val(n, "공개 일자")]
        terms = [
            f"{n}:{val(n, '입찰 참여 시작일')[:10]}~{val(n, '입찰 참여 마감일')[:10]}"
            for n in same
            if val(n, "입찰 참여 시작일") or val(n, "입찰 참여 마감일")
        ]
        raw_amt = row["사업 금액"].strip()
        fix = AMOUNT_FIX.get(no)
        amt, note = (fix[0], fix[1]) if fix else (raw_amt or None, "")
        n_empty = n_split = 0
        if rep == no:
            cleaned, n_empty, n_split = clean_text(texts[no])
            (CLEAN / f"{Path(row['파일명']).stem}.txt").write_text(
                cleaned, encoding="utf-8"
            )
            kept += 1
        meta.append(
            {
                "문서번호": no,
                "파일명": row["파일명"],
                "발주기관": row["발주 기관"],
                "사업명": row["사업명"],
                "원본_사업금액": raw_amt,
                "정제_사업금액": "" if amt is None else amt,
                "금액비고": note,
                "중복대표": "" if rep == no else rep,
                "대표문서의_기관들": " / ".join(orgs)
                if multi and len(orgs) > 1
                else "",
                "대표문서의_사업명들": " / ".join(names)
                if multi and len(names) > 1
                else "",
                "대표문서의_공고번호들": " / ".join(ids) if multi else "",
                "대표문서의_공개일들": " / ".join(pubs) if multi else "",
                "대표문서의_입찰기간들": " / ".join(terms) if multi else "",
                "빈표제거수": n_empty,
                "긴줄분할수": n_split,
            }
        )

    with open(CLEAN / "_meta.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)

    dups = [(m["문서번호"], m["중복대표"]) for m in meta if m["중복대표"]]
    split_docs = [m["문서번호"] for m in meta if m["긴줄분할수"]]
    print(f"문서 {len(meta)}건 중 저장 {kept}건: {SRC} -> {CLEAN}")
    print(f"중복으로 제외(번호, 대표): {dups}")
    print(f"제거한 빈 표: {sum(m['빈표제거수'] for m in meta)}개")
    print(f"나눈 긴 줄: {sum(m['긴줄분할수'] for m in meta)}개 (문서 {split_docs})")


if __name__ == "__main__":
    main()
