"""Qdrant를 사용해 임베딩 검색 성능을 평가합니다.

실행:
  uv run --no-sync python -m ai.data_basic.experiments.qdrant_eval

입력: data/<CLEAN_DIR>/, data/golden/rahui_golden_set_v2_2.json
출력: data/qdrant_db/, data/results/<실행 폴더>/
OpenAI 모델은 --allow-external 지정 시 문서 본문을 API로 전송합니다.
"""

import argparse
import csv
import time
from datetime import datetime

import numpy as np

from ai.data_basic.experiments import chunk_compare as cc
from ai.data_basic.experiments import embed_compare as ec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=list(ec.MODELS))
    ap.add_argument("--configs", default="S1000M,S1500M")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--url", default="", help="Qdrant 서버 주소. 비우면 로컬 모드")
    ap.add_argument(
        "--path", default="qdrant_db", help="로컬 모드 저장 폴더 (data/ 아래)"
    )
    ap.add_argument("--allow-external", action="store_true")
    a = ap.parse_args()
    ec.ALLOW_EXTERNAL = a.allow_external

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = (
        QdrantClient(url=a.url) if a.url else QdrantClient(path=str(ec.DATA / a.path))
    )
    docs, rep = cc.load_corpus()
    qs, skipped = cc.load_questions(docs, rep)
    queries = [q["query"] for q in qs]
    print(
        f"문서 {len(docs)}건, 평가 문항 {len(qs)}개, CLEAN={cc.CLEAN.name}, Qdrant={'서버 ' + a.url if a.url else '로컬 모드'}",
        flush=True,
    )

    emb = ec.Embedder(a.model, a.max_tokens, a.batch)
    qe = emb.encode(queries, "q", "queries")
    rows, same_all = [], True
    for c in [x for x in a.configs.split(",") if x in cc.CONFIGS]:
        kind, size, meta = cc.CONFIGS[c]
        chunks = cc.build_chunks(docs, kind, size, meta)
        de = emb.encode([x["index"] for x in chunks], "d", c)
        name = f"{a.model}_{c}_{cc.CLEAN.name}"
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=de.shape[1], distance=Distance.COSINE),
        )
        t0 = time.time()
        client.upload_points(
            collection_name=name,
            points=[
                PointStruct(
                    id=i, vector=de[i].tolist(), payload={"doc": chunks[i]["doc"]}
                )
                for i in range(len(chunks))
            ],
            batch_size=256,
        )
        t_up = time.time() - t0
        t0 = time.time()
        q_top = [
            [
                p.id
                for p in client.query_points(
                    collection_name=name, query=qe[qi].tolist(), limit=10
                ).points
            ]
            for qi in range(len(queries))
        ]
        t_q = time.time() - t0
        np_top = [np.argsort(-s)[:10].tolist() for s in (qe @ de.T)]
        n_same = sum(
            set(x) == set(y) for x, y in zip(q_top, np_top, strict=False)
        )  # 상위 10개 집합이 같은 문항 수
        same_all &= n_same == len(qs)
        r, _ = ec.evaluate(
            f"{c}|{a.model}|Qdrant", chunks, qs, lambda qi, _q, q_top=q_top: q_top[qi]
        )
        r0, _ = ec.evaluate(
            f"{c}|{a.model}|numpy", chunks, qs, lambda qi, _q, np_top=np_top: np_top[qi]
        )
        rows += [r, r0]
        print(
            f"{c}: 청크 {len(chunks)}개 업로드 {t_up:.1f}초, 질문 {len(qs)}개 검색 {t_q:.1f}초, numpy와 상위10 일치 {n_same}/{len(qs)}",
            flush=True,
        )
    print(
        "\n"
        + f"{'설정|모델|검색':<30}"
        + "".join(f"{'근거@' + str(k):>8}" for k in cc.KS)
        + f"{'MRR':>7}"
    )
    for r in rows:
        print(
            f"{r['설정']:<30}"
            + "".join(f"{r[f'근거@{k}']:>8.2f}" for k in cc.KS)
            + f"{r['MRR@10']:>7.2f}"
        )
    print(
        "\nQdrant 결과가 numpy 직접 계산과 모든 문항에서 같음"
        if same_all
        else "\n주의: 일부 문항에서 Qdrant 와 numpy 의 상위 10개가 다름 (동점 처리 등) — 위 일치 수 확인"
    )

    run = (
        cc.RESULTS
        / f"{datetime.now():%Y%m%d_%H%M%S}_Qdrant평가_{a.model}_{cc.CLEAN.name}"
    )
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장 폴더: {run}")


if __name__ == "__main__":
    main()
