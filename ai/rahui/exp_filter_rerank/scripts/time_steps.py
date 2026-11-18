"""실험 2: 단계별 시간 측정 (질문 임베딩 / 거리 계산·정렬 / 답변 생성)

baseline.py 를 고치지 않고 안의 부품을 그대로 불러 단계별로 시간만 잰다.
실행 (프로젝트 루트에서):
    uv run python ai/rahui/exp_filter_rerank/scripts/time_steps.py --n 30
결과: outputs/time_steps_<시각>.csv (git 제외), 화면에는 단계별 평균/p50/p95만 출력.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # 프로젝트 루트
sys.path.insert(0, str(ROOT))

from langchain.messages import HumanMessage, SystemMessage  # noqa: E402

from engines.local_model import baseline as b  # noqa: E402

GOLDEN = ROOT / "data" / "golden" / "rahui_golden_set_v2_2.json"


def pct(values, q):
    return float(np.percentile(values, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="clean_mixed_ocr_v2")
    ap.add_argument("--n", type=int, default=30, help="측정할 문항 수 (앞에서부터)")
    ap.add_argument("--no-generate", action="store_true", help="생성 단계 생략")
    args = ap.parse_args()

    b.configure(corpus=args.corpus)
    items = json.loads(GOLDEN.read_text(encoding="utf-8"))[: args.n]

    print("준비 중 (모델/인덱스 로딩, 측정에서 제외)...")
    t0 = time.perf_counter()
    b.warmup()
    print(f"준비 완료: {time.perf_counter() - t0:.1f}초")

    model, vector_store = b.get_resources()
    matrix, docs = b.get_exact_index()
    rows = []
    for it in items:
        q = it["question"]

        t = time.perf_counter()
        vec = np.asarray(vector_store.embeddings.embed_query(q), dtype=np.float32)
        t_embed = time.perf_counter() - t

        t = time.perf_counter()
        dist = ((matrix - vec) ** 2).sum(axis=1)
        top = np.argsort(dist, kind="stable")[: b.TOP_K]
        retrieved = [(docs[i], float(dist[i])) for i in top]
        t_search = time.perf_counter() - t

        t_gen = 0.0
        if not args.no_generate:
            messages = [
                SystemMessage(b.SYSTEM_PROMPT),
                *b.history_to_messages(it.get("history")),
                HumanMessage(f"컨텍스트:\n{b.format_context(retrieved)}\n\n질문: {q}"),
            ]
            t = time.perf_counter()
            model.invoke(messages)
            t_gen = time.perf_counter() - t

        rows.append(
            {
                "id": it["id"],
                "embed": t_embed,
                "search": t_search,
                "generate": t_gen,
                "total": t_embed + t_search + t_gen,
            }
        )

    print(f"\n문항 {len(rows)}개, corpus={args.corpus}, top_k={b.TOP_K}")
    print(f"{'단계':<10}{'평균':>8}{'p50':>8}{'p95':>8}  (초)")
    for key in ("embed", "search", "generate", "total"):
        v = [r[key] for r in rows]
        print(f"{key:<10}{np.mean(v):>8.3f}{pct(v, 50):>8.3f}{pct(v, 95):>8.3f}")
    tot = np.mean([r["total"] for r in rows])
    if tot > 0:
        for key in ("embed", "search", "generate"):
            share = np.mean([r[key] for r in rows]) / tot * 100
            print(f"비중 {key:<9}{share:>5.1f}%")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"time_steps_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
