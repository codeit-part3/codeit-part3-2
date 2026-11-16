"""Qdrant에 문서 청크를 색인하고 검색합니다.

실행:
  uv run --no-sync python -m ai.data_basic.experiments.qdrant_index index --model gemma --config S1000M
  uv run --no-sync python -m ai.data_basic.experiments.qdrant_index search --model gemma --config S1000M --q "질문"

입력: data/<CLEAN_DIR>/
출력: data/qdrant_db/ (로컬 모드)
OpenAI 모델은 --allow-external 지정 시 문서 본문을 API로 전송합니다.
"""

import argparse

from ai.data_basic.experiments import chunk_compare as cc
from ai.data_basic.experiments import embed_compare as ec


def collection_name(a) -> str:
    return a.collection or f"rfp_{a.model}_{a.config}_{cc.CLEAN.name}"


def connect(a):
    from qdrant_client import QdrantClient

    return (
        QdrantClient(url=a.url) if a.url else QdrantClient(path=str(ec.DATA / a.path))
    )


def cmd_index(a) -> None:
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )

    docs, _rep = cc.load_corpus()
    kind, size, meta = cc.CONFIGS[a.config]
    chunks = cc.build_chunks(docs, kind, size, meta)
    print(
        f"문서 {len(docs)}건 -> 청크 {len(chunks)}개 ({a.config}, CLEAN={cc.CLEAN.name})",
        flush=True,
    )
    emb = ec.Embedder(a.model, a.max_tokens, a.batch)
    de = emb.encode([x["index"] for x in chunks], "d", a.config)
    client, name = connect(a), collection_name(a)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=de.shape[1], distance=Distance.COSINE),
    )
    seen: dict[int, int] = {}
    points = []
    for i, ch in enumerate(chunks):
        d = docs[ch["doc"]]
        seen[ch["doc"]] = seen.get(ch["doc"], 0) + 1
        head = ch["index"].split("\n", 1)[0] if ch["index"].startswith("[") else ""
        points.append(
            PointStruct(
                id=i,
                vector=de[i].tolist(),
                payload={
                    "doc": ch["doc"],
                    "chunk_no": seen[ch["doc"]],
                    "org": d["org"],
                    "title": d["title"],
                    "head": head,
                    "text": ch["body"],
                },
            )
        )
    client.upload_points(collection_name=name, points=points, batch_size=256)
    client.create_payload_index(
        collection_name=name, field_name="doc", field_schema=PayloadSchemaType.INTEGER
    )
    client.create_payload_index(
        collection_name=name, field_name="org", field_schema=PayloadSchemaType.KEYWORD
    )
    print(
        f"컬렉션 '{name}' 에 {client.count(name).count}개 저장 (필터용 색인: doc, org)"
    )


def cmd_search(a) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    emb = ec.Embedder(a.model, a.max_tokens, a.batch)
    qv = emb.encode([a.q], "q", "adhoc")[0]
    must = []
    if a.org:
        must.append(FieldCondition(key="org", match=MatchValue(value=a.org)))
    if a.doc:
        must.append(FieldCondition(key="doc", match=MatchValue(value=a.doc)))
    client, name = connect(a), collection_name(a)
    res = client.query_points(
        collection_name=name,
        query=qv.tolist(),
        limit=a.k,
        query_filter=Filter(must=must) if must else None,
    ).points
    if not res:
        print(
            "결과 없음 (필터 값이 정확한지 확인: 기관 이름은 data_list.csv 의 '발주 기관' 과 같아야 함)"
        )
    for rank, p in enumerate(res, 1):
        pl = p.payload
        text = pl["text"].replace("\n", " ")
        print(
            f"[{rank}] 점수 {p.score:.3f} | 문서 {pl['doc']} 청크 {pl['chunk_no']} | {pl['org']} | {pl['title'][:40]}\n    {text[:200]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("index", "search"):
        p = sub.add_parser(n)
        p.add_argument("--model", default="gemma", choices=list(ec.MODELS))
        p.add_argument("--config", default="S1000M", choices=list(cc.CONFIGS))
        p.add_argument("--max-tokens", type=int, default=1024)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--url", default="")
        p.add_argument("--path", default="qdrant_db")
        p.add_argument("--collection", default="")
        p.add_argument("--allow-external", action="store_true")
        if n == "search":
            p.add_argument("--q", required=True)
            p.add_argument("--k", type=int, default=5)
            p.add_argument("--org", default="")
            p.add_argument("--doc", type=int, default=0)
    a = ap.parse_args()
    ec.ALLOW_EXTERNAL = a.allow_external
    {"index": cmd_index, "search": cmd_search}[a.cmd](a)


if __name__ == "__main__":
    main()
