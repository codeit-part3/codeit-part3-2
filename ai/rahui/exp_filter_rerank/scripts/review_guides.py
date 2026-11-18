"""가이드 캐시 품질 점검 (모델 없이 캐시 파일만 읽는다).

    uv run python ai/rahui/exp_filter_rerank/scripts/review_guides.py                 # 전체 점검 + 표본 5곳 출력
    uv run python ai/rahui/exp_filter_rerank/scripts/review_guides.py --docs 벤처 평택  # 이름에 들어간 낱말로 공고 고르기
    uv run python ai/rahui/exp_filter_rerank/scripts/review_guides.py --pack guide_cache.zip   # 점검 통과 시 팀 공유용 zip 만들기

무엇을 보나: 지난번 검토에서 나온 문제를 숫자로 바로 확인한다. 하나라도 FAIL 이면 종료 코드 1.
  1) 현재 설정(프롬프트·모델·평가표 읽기)에 맞는 캐시가 공고 수만큼 있는가 (옛 캐시는 세지 않는다)
  2) 점수·합격을 약속하는 표현이 0건인가   3) 프롬프트의 few-shot 문장을 베낀 것이 0건인가
  4) 항목 이름이 쓰는 칸(where) 문장에 들어 있는가   5) 합계 줄이 항목으로 들어간 것이 0건인가
  6) 흔한 실수 문장이 "단순히"로 시작하는 비율, 예시가 같은 말로 시작하는 비율이 너무 높지 않은가
  7) 같은 문장이 여러 항목에 반복되지 않는가   8) 팁·흔한 실수가 비어 있는 비율
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ai.rahui.exp_filter_rerank.rahui_engine import writer as w  # noqa: E402

LIMITS = {
    "단순히_시작": 0.30,
    "예시_같은_시작": 0.50,
    "팁_빈칸": 0.15,
    "이름_포함_최소": 0.90,
    "반복_문장_최대": 8,
}


def items_for(path: Path):
    items, level, _ = w.extract_items(
        path.read_text(encoding="utf-8", errors="replace")
    )
    if not items:
        items = [
            {"group": "", "name": a, "score": None, "criteria": b, "elements": ""}
            for a, b in w.GENERIC
        ]
    return items, level


def collect(only: list[str] | None):
    files = sorted(w.CORPUS_DIR.glob("*.txt"))
    if only:
        files = [
            f
            for f in files
            if all(o in f.stem for o in only) or any(o in f.stem for o in only)
        ]
    docs, missing = [], []
    for f in files:
        items, level = items_for(f)
        cp = w._cache_path(f.stem, items)
        if cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            docs.append((f, items, level, d["guides"], cp))
        else:
            missing.append(f.stem)
    return files, docs, missing


def review(docs, missing, n_files, cache_dir_files: int, current: int):
    fails, notes = [], []
    n = 0
    promise, copied, no_name, summed = [], [], [], []
    absolute = []
    mistakes, examples, tips_empty, mist_empty, model_items = [], [], 0, 0, 0
    if not docs:
        return [
            f"FAIL  현재 설정에 맞는 캐시가 하나도 없습니다 (공고 {n_files}곳 모두 없음). 먼저 --precompute 로 만드세요."
        ], notes
    sent = collections.Counter()
    for f, items, _lv, guides, _cp in docs:
        for it, g in zip(items, guides, strict=False):
            n += 1
            if w.quant_kind(it):
                continue  # 고정 문장 항목은 따로 (모델 문장 점검에서 뺀다)
            if re.search(r"합\s*계|소\s*계", it["name"]):
                summed.append((f.stem[:20], it["name"]))
            is_model = bool(g.get("example") or g.get("tip") or g.get("mistake"))
            if not is_model:
                continue  # 규칙 문장으로 대체된 항목
            model_items += 1
            allt = " ".join(
                [g.get(k, "") for k in ("example", "tip", "mistake")]
                + list(g.get("must", []))
            )
            if w.has_promise(allt):
                promise.append((f.stem[:20], it["name"], allt[:60]))
            if (
                g.get("where") in w._FEWSHOT_VALUES
                or g.get("example") in w._FEWSHOT_VALUES
            ):
                copied.append((f.stem[:20], it["name"]))
            stem = w._plain_name(it["name"])
            if stem and stem.replace(" ", "") not in g.get("where", "").replace(
                " ", ""
            ):
                no_name.append((f.stem[:20], it["name"], g.get("where", "")[:40]))
            if w.ABSOLUTE.search(g.get("example", "") + " ".join(g.get("must", []))):
                absolute.append((f.stem[:20], it["name"]))
            mistakes.append(g.get("mistake", ""))
            examples.append(g.get("example", ""))
            tips_empty += not g.get("tip")
            mist_empty += not g.get("mistake")
            for k in (
                "tip",
                "mistake",
                "example",
            ):  # where 는 같은 이름의 항목이면 같은 문장이 자연스럽다
                if g.get(k):
                    sent[g[k]] += 1

    def check(name, ok, detail):
        (notes if ok else fails).append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    check(
        "현재 설정에 맞는 캐시",
        not missing,
        f"{len(docs)}/{n_files}곳 있음"
        + (
            f" (없는 공고 {len(missing)}곳: {', '.join(m[:14] for m in missing[:3])}…)"
            if missing
            else ""
        ),
    )
    stale = cache_dir_files - current
    notes.append(
        f"INFO  캐시 폴더의 파일 {cache_dir_files}개 중 현재 설정과 맞지 않는(옛) 파일 {max(stale, 0)}개 — 공유 zip 에는 넣지 않는다"
    )
    check(
        "점수·합격 약속 표현",
        not promise,
        f"{len(promise)}건" + (f" 예) {promise[0]}" if promise else ""),
    )
    check("few-shot 문장 베낌", not copied, f"{len(copied)}건")
    check("단정·과장 낱말(모든·철저히 등)", not absolute, f"{len(absolute)}건")
    mm = 1 - len(no_name) / max(model_items, 1)
    check(
        "where 문장에 항목 이름 포함",
        mm >= LIMITS["이름_포함_최소"],
        f"{mm:.0%} ({len(no_name)}건 빠짐)" + (f" 예) {no_name[0]}" if no_name else ""),
    )
    check(
        "합계 줄이 항목으로 들어감",
        not summed,
        f"{len(summed)}건" + (f" 예) {summed[0]}" if summed else ""),
    )
    if model_items:
        r1 = sum(m.startswith("단순히") for m in mistakes) / model_items
        check(
            "'흔한 실수'가 '단순히'로 시작",
            r1 <= LIMITS["단순히_시작"],
            f"{r1:.0%} (한도 {LIMITS['단순히_시작']:.0%})",
        )
        top = collections.Counter(e[:8] for e in examples if e).most_common(1)
        r2 = top[0][1] / max(sum(1 for e in examples if e), 1) if top else 0
        check(
            "'예시'가 같은 말로 시작",
            r2 <= LIMITS["예시_같은_시작"],
            f"{r2:.0%} ('{top[0][0] if top else ''}')",
        )
        r3 = tips_empty / model_items
        check(
            "팁이 비어 있음",
            r3 <= LIMITS["팁_빈칸"],
            f"{r3:.0%} (보장 표현을 지운 칸 포함)",
        )
    rep = [(s, c) for s, c in sent.items() if c > LIMITS["반복_문장_최대"]]
    check(
        "같은 문장 반복",
        not rep,
        f"{len(rep)}종" + (f" 예) {rep[0][1]}번: {rep[0][0][:40]}" if rep else ""),
    )
    notes.append(
        f"INFO  항목 {n}개 중 모델이 쓴 항목 {model_items}개, 고정 문장(재무·가격) {sum(1 for _f, its, *_ in docs for i in its if w.quant_kind(i))}개"
    )
    return fails, notes


def dump_sample(docs, k: int, seed: int):
    rnd = random.Random(seed)
    for f, items, level, guides, _cp in rnd.sample(docs, min(k, len(docs))):
        print("=" * 90)
        print(f.stem[:60], "| 근거:", level, "| 항목", len(items))
        idx = sorted({0, len(items) // 2, len(items) - 1})
        for i in idx:
            it, g = items[i], guides[i]
            sc = (
                f"({it['score']:g}점)" if it.get("score") is not None else "(배점 없음)"
            )
            ref = (w.item_ref(it) or "(없음)")[:100]
            print(f"\n[{i + 1}] {it['name']} {sc}  공고 평가기준: {ref}")
            for key, lab in (
                ("where", "쓰는 칸"),
                ("must", "꼭 넣을 내용"),
                ("example", "예시"),
                ("tip", "팁"),
                ("mistake", "흔한 실수"),
            ):
                v = g.get(key)
                v = " / ".join(v) if isinstance(v, list) else v
                if v:
                    print(f"  {lab}: {v[:170]}")


def pack(docs, out: Path):
    """점검을 통과한 캐시만 zip 으로 묶는다. 프로젝트 루트에 풀면 outputs/guides/_cache/ 에 들어간다 (공고문 내용이 들어 있으니 팀 밖 공유 금지)."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for *_x, cp in docs:
            z.write(cp, f"outputs/guides/_cache/{cp.name}")
    print(f"zip 만듦: {out} ({len(docs)}개)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--docs", nargs="*", help="공고 이름에 들어 있는 낱말로 고르기 (없으면 전체)"
    )
    ap.add_argument(
        "--sample", type=int, default=5, help="사람이 읽어 볼 표본 공고 수 (기본 5)"
    )
    ap.add_argument("--seed", type=int, default=1118)
    ap.add_argument("--pack", type=Path, help="점검 통과 시 zip 으로 묶기")
    a = ap.parse_args()
    files, docs, missing = collect(a.docs)
    cache_files = len(list(w.CACHE_DIR.glob("*.json"))) if w.CACHE_DIR.exists() else 0
    fails, notes = review(docs, missing, len(files), cache_files, len(docs))
    print("\n".join(notes + fails))
    if a.sample and docs:
        print(
            "\n"
            + "#" * 90
            + f"\n# 표본 {a.sample}곳 (사람이 직접 읽어 보세요: 어색한 문장, 항목과 안 맞는 내용, 지어낸 숫자)\n"
            + "#" * 90
        )
        dump_sample(docs, a.sample, a.seed)
    if a.pack:
        if fails:
            print("\n점검에서 FAIL 이 있어 zip 을 만들지 않았습니다.")
        else:
            pack(docs, a.pack)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
