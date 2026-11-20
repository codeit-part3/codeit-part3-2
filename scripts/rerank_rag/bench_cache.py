"""가이드 캐시 전/후 속도 비교 (같은 공고, 같은 PC, 모델 사용).

각 공고마다 한 번은 캐시를 끄고(WRITER_NO_CACHE=1, 모델이 항목마다 새로 씀), 한 번은 캐시를 켜고(저장된 안내문 사용) 시간을 잰다.
시간은 질문을 받아 Word 파일을 만들 때까지다. 모델을 처음 읽는 시간은 빼려고 가장 작은 공고로 한 번 미리 돌린다(워밍업).
실행(프로젝트 루트에서): uv run python scripts/rerank_rag/bench_cache.py
주의: 매 실행마다 outputs/guides 에 docx 가 쌓인다(git 제외 폴더). 캐시 파일은 바꾸지 않는다(캐시를 끈 실행은 저장도 안 함).
"""

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engines.rerank_rag import writer  # noqa: E402

QUESTIONS = [
    ("한국해양조사협회", "한국해양조사협회 항해용 간행물 제안서 작성 가이드 만들어줘"),
    (
        "한국철도공사 예약발매",
        "한국철도공사 예약발매시스템 제안서 작성 가이드 만들어줘",
    ),
    ("평택시 BIS", "평택시 버스정보시스템 제안서 작성 가이드 만들어줘"),
    ("벤처기업협회", "벤처기업협회 벤처확인종합관리시스템 제안서 작성 가이드 만들어줘"),
    ("한영대학", "한영대학 학사정보시스템 제안서 작성 가이드 만들어줘"),
]


def run(q: str, cache: bool) -> tuple[float, str]:
    if cache:
        os.environ.pop("WRITER_NO_CACHE", None)
    else:
        os.environ["WRITER_NO_CACHE"] = "1"
    t = time.time()
    ans = writer.answer(q)
    return time.time() - t, ans


def main():
    print("워밍업(모델 읽기)…", flush=True)
    run(QUESTIONS[0][1], cache=False)
    rows = []
    for name, q in QUESTIONS:
        t_no, ans = run(q, cache=False)
        m = re.search(r"항목\s*(\d+)개", ans)
        n = int(m.group(1)) if m else 0
        t_ca, _ = run(q, cache=True)
        rows.append((name, n, t_no, t_ca))
        print(
            f"{name}: 항목 {n}개, 캐시 없이 {t_no:.1f}초, 캐시 사용 {t_ca:.1f}초",
            flush=True,
        )
    print(
        "\n| 공고 | 항목 수 | 캐시 없이(초) | 캐시 사용(초) | 배수 |\n| --- | --- | --- | --- | --- |"
    )
    for name, n, a, b in rows:
        print(f"| {name} | {n} | {a:.1f} | {b:.1f} | {a / max(b, 0.1):.0f}배 |")
    ta, tb = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(
        f"| 합계 | {sum(r[1] for r in rows)} | {ta:.1f} | {tb:.1f} | {ta / max(tb, 0.1):.0f}배 |"
    )


if __name__ == "__main__":
    main()
