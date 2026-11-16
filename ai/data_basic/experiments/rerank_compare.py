"""재순위화(리랭커) 실험: 임베딩으로 후보 N개를 뽑고, 질문-청크 쌍을 교차 인코더로 다시 채점해 순서를 바꾼다.

임베딩은 embed_compare 가 저장한 캐시를 그대로 쓴다(없으면 새로 만든다).
사용: python -m ai.data_basic.experiments.rerank_compare --configs S1500M --models gemma --rerankers bge-reranker-v2-m3 --pool 20,30
"""

import argparse
import csv
import hashlib
import time
from datetime import datetime

import numpy as np

from ai.data_basic.experiments import chunk_compare as cc
from ai.data_basic.experiments import embed_compare as ec

RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "ko-reranker": "Dongjin-kr/ko-reranker",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="S1500M")
    ap.add_argument("--models", default="gemma")
    ap.add_argument("--rerankers", default="bge-reranker-v2-m3")
    ap.add_argument("--pool", default="20,30", help="재순위화할 후보 개수(쉼표 구분)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rerank-batch", type=int, default=8)
    ap.add_argument(
        "--rerank-len", type=int, default=1024, help="리랭커 입력 최대 토큰"
    )
    a = ap.parse_args()
    cfgs = [c for c in a.configs.split(",") if c in cc.CONFIGS]
    model = a.models.split(",")[0]
    pools = [int(x) for x in a.pool.split(",")]
    docs, rep = cc.load_corpus()
    qs, skipped = cc.load_questions(docs, rep)
    print(
        f"문서 {len(docs)}건, 평가 문항 {len(qs)}개 (제외 {len(skipped)}개)", flush=True
    )
    queries = [q["query"] for q in qs]
    built = {c: cc.build_chunks(docs, *cc.CONFIGS[c]) for c in cfgs}
    for c in cfgs:
        print(f"{c}: 청크 {len(built[c])}개", flush=True)

    emb = ec.Embedder(model, a.max_tokens, a.batch)
    qe = emb.encode(queries, "q", "queries")
    dense = {}
    for c in cfgs:
        de = emb.encode([x["index"] for x in built[c]], "d", c)
        dense[c] = [np.argsort(-s)[: max(pools)] for s in (qe @ de.T)]
    del emb

    import torch
    from sentence_transformers import CrossEncoder

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, perq, t0 = [], [], time.time()
    for c in cfgs:
        chunks = built[c]
        r, p = ec.evaluate(
            f"{c}|{model}", chunks, qs, lambda qi, _q, c=c: dense[c][qi][:10].tolist()
        )
        rows.append(r)
        perq += p
    for rr_name in [x for x in a.rerankers.split(",") if x in RERANKERS]:
        print(f"  리랭커 불러오는 중: {RERANKERS[rr_name]} (장치: {dev})", flush=True)
        ce = CrossEncoder(RERANKERS[rr_name], device=dev, max_length=a.rerank_len)
        for c in cfgs:
            chunks = built[c]
            pool_max = max(pools)
            pairs, spans = [], []
            for qi, q in enumerate(qs):
                cand = dense[c][qi][:pool_max]
                spans.append((len(pairs), len(cand)))
                pairs += [(q["query"], chunks[int(i)]["index"]) for i in cand]
            print(f"  {rr_name} | {c}: {len(pairs)}쌍 채점", flush=True)
            sc = np.asarray(
                ce.predict(pairs, batch_size=a.rerank_batch, show_progress_bar=True)
            )
            for pool in pools:
                order = {}
                for qi, (s0, _n) in enumerate(spans):
                    cand = dense[c][qi][:pool]
                    s = sc[s0 : s0 + pool]
                    order[qi] = [int(cand[j]) for j in np.argsort(-s)[:10]]
                label = f"{c}|{model}+{rr_name}@{pool}"
                r, p = ec.evaluate(label, chunks, qs, lambda qi, _q, o=order: o[qi])
                rows.append(r)
                perq += p
        del ce
    names = [r["설정"] for r in rows]
    base = [n for n in names if "+" not in n][0]
    cc.PAIRS = [(n, base) for n in names if n != base]
    lines = [
        f"문서 {len(docs)}건, 평가 문항 {len(qs)}개 (제외 {len(skipped)}개)",
        f"임베딩 {model}, 설정 {cfgs}, 후보 수 {pools}, 리랭커 {a.rerankers}",
        "",
    ]
    lines.append(
        f"{'설정':<44}"
        + "".join(f"{'근거@' + str(k):>8}" for k in cc.KS)
        + f"{'MRR':>7}"
    )
    for r in sorted(rows, key=lambda r: -r["근거@5"]):
        lines.append(
            f"{r['설정']:<44}"
            + "".join(f"{r[f'근거@{k}']:>8.2f}" for k in cc.KS)
            + f"{r['MRR@10']:>7.2f}"
        )
    lines += [""] + cc.strict_block(rows, perq, qs, names)
    lines += [""] + cc.bootstrap(perq, qs, names)
    run = cc.RESULTS / f"{datetime.now():%Y%m%d_%H%M%S}_재순위화비교_{cc.CLEAN.name}"
    run.mkdir(parents=True, exist_ok=True)
    cols = (
        ["설정"]
        + [f"근거@{k}" for k in cc.KS]
        + [f"완전근거@{k}" for k in cc.KS]
        + [f"조각커버@{k}" for k in cc.KS]
        + ["MRR@10"]
    )
    with open(run / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(run / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["config", "id", "level", "근거_순위", "문서_순위", "조각순위"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(perq)
    (run / "summary.txt").write_text(
        "\n".join(lines) + "\n\n저장 폴더: " + str(run) + "\n", encoding="utf-8"
    )

    def md5(p):
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    info = [
        f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S} (걸린 시간 {time.time() - t0:.0f}초)",
        "실행 명령: python -m ai.data_basic.experiments.rerank_compare "
        + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in vars(a).items()),
        f"CLEAN 폴더: {cc.CLEAN}",
        f"골든셋: {cc.GOLDEN} md5 {md5(cc.GOLDEN)}",
        f"rerank_compare.py md5: {md5(__file__)}",
        f"장치: {dev}",
    ]
    (run / "run_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print("\n".join(lines[: 4 + len(rows) + 1]))
    print("\n저장 폴더:", run)


if __name__ == "__main__":
    main()
