"""리랭커 검색 성능 비교 (베이스라인 조건, 검색 지표만, 답변 모델·채점 호출 없음)

하는 일
  같은 코드 경로로 (1) 베이스라인 검색 (2) 임베딩 후보 N개 -> 리랭커로 재정렬 을 비교한다.
  지표 계산은 retrieval.py / evaluate.py 와 같은 함수(hit_rate, mrr, context_hit)를 그대로 쓴다.
  baseline.py 는 고치지 않는다.

실행 (프로젝트 루트에서)
    uv run python scripts/rerank_rag/rerank_baseline.py --limit 10     # 빠른 확인
    uv run python scripts/rerank_rag/rerank_baseline.py --pool 20 30   # 전체

결과: outputs/rerank_baseline_<시각>.csv (요약), ..._detail.csv (문항별). git 제외.
주의: 첫 실행 때 리랭커 모델을 내려받습니다(용량이 큽니다). 시트 기록은 하지 않습니다.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engines.local_model import baseline  # noqa: E402

import evaluate  # noqa: E402

GOLDEN = ROOT / "data" / "golden" / "rahui_golden_set_v2_2.json"
RERANKER = "BAAI/bge-reranker-v2-m3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="clean_mixed_ocr_v2")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument(
        "--pool", type=int, nargs="+", default=[20], help="재정렬할 후보 수"
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512, help="리랭커 입력 최대 토큰")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise SystemExit(
            f"필요한 패키지가 없습니다: {e}\n"
            "개인 환경에 `uv pip install sentence-transformers` 로 설치하세요 (공통 파일은 건드리지 않음)."
        ) from e

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8-sig"))
    if args.limit:
        golden = golden[: args.limit]
    file_to_notice = evaluate.load_file_to_notice()
    notices = [evaluate.evidence_notice_ids(it, file_to_notice) for it in golden]

    baseline.configure(corpus=args.corpus)
    baseline.get_resources()
    baseline.get_exact_index()
    ks = sorted(set(args.top_k))
    pools = sorted(set(args.pool))
    max_pool = max(pools + [max(ks)])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"리랭커 불러오는 중: {RERANKER} (장치: {dev})", flush=True)
    ce = CrossEncoder(RERANKER, device=dev, max_length=args.max_length)
    ce.predict([("워밍업", "워밍업")], batch_size=1)  # 첫 호출 지연을 측정에서 제외

    details = []
    n = len(golden)
    for i, (item, wanted) in enumerate(zip(golden, notices, strict=True), 1):
        q = item["question"]
        cand = baseline.search(q, k=max_pool)  # 임베딩 순 후보 [(조각, 거리), ...]
        variants = {"baseline": (cand, 0.0)}
        for p in pools:
            pool = cand[:p]
            t = time.perf_counter()
            scores = ce.predict(
                [(q, d.page_content) for d, _ in pool], batch_size=args.batch
            )
            sec = time.perf_counter() - t
            order = np.argsort(-np.asarray(scores), kind="stable")
            variants[f"rerank_pool{p}"] = (
                [(pool[j][0], float(-scores[j])) for j in order],
                sec,
            )
        for name, (ranked, sec) in variants.items():
            for k in ks:
                top = ranked[:k]
                doc_hit, rr = evaluate.retrieval_scores(item, top, wanted)
                details.append(
                    {
                        "variant": name,
                        "top_k": k,
                        "id": item["id"],
                        "level": item["level"],
                        "doc_hit": doc_hit,
                        "rr": rr,
                        "context_hit": evaluate.context_hit(item, top),
                        "rerank_sec": round(sec, 4),
                    }
                )
        if i % 10 == 0 or i == n:
            print(f"  {i}/{n}", flush=True)

    df = pd.DataFrame(details)
    rows = []
    for (name, k), part in df.groupby(["variant", "top_k"], sort=False):
        rows.append(
            {
                "variant": name,
                "top_k": k,
                "questions": len(part),
                "hit_rate": evaluate.mean_or_none(part["doc_hit"]),
                "mrr": evaluate.mean_or_none(part["rr"]),
                "context_hit": evaluate.mean_or_none(part["context_hit"]),
                "rerank_sec_mean": round(float(part["rerank_sec"].mean()), 3),
                "rerank_sec_p95": round(
                    float(np.percentile(part["rerank_sec"], 95)), 3
                ),
            }
        )
    table = pd.DataFrame(rows)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    baseline.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(
        baseline.OUTPUT_DIR / f"rerank_baseline_{stamp}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df.to_csv(
        baseline.OUTPUT_DIR / f"rerank_baseline_{stamp}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n골든셋 {n}문항, corpus={args.corpus}, 장치={dev}")
    print(table.to_string(index=False))
    print(f"\n저장: outputs/rerank_baseline_{stamp}.csv (+ _detail.csv)")


if __name__ == "__main__":
    main()
