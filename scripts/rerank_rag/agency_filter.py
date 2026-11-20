"""실험 1: 기관명 필터 (검색 지표만, 답변 모델·채점 호출 없음)

질문에서 발주 기관명을 찾으면 그 기관의 조각만 후보로 남긴 뒤 top-k 를 뽑는다.
기관명을 못 찾으면 필터 없이 베이스라인과 똑같이 검색한다 (잘못된 필터 방지).
비교: baseline / filter / filter+rerank / rerank (리랭커는 --pool 후보 수)

실행 (프로젝트 루트에서)
    uv run python scripts/rerank_rag/agency_filter.py --limit 10    # 빠른 확인
    uv run python scripts/rerank_rag/agency_filter.py --pool 10     # 전체
    uv run python scripts/rerank_rag/agency_filter.py --no-rerank   # 리랭커 없이
baseline.py 는 고치지 않는다. 결과: outputs/agency_filter_<시각>.csv (+ _detail.csv), git 제외.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
from engines.local_model import baseline  # noqa: E402

GOLDEN = ROOT / "data" / "golden" / "rahui_golden_set_v2_2.json"
RERANKER = "BAAI/bge-reranker-v2-m3"
MIN_LEN = 3  # 이보다 짧은 기관명은 오탐이 많아 매칭에서 제외


def norm_agency(s):
    """매칭용 이름: 법인 유형 표기 제거 + 공백·괄호·기호 제거 (원본은 따로 보존)"""
    s = unicodedata.normalize("NFKC", str(s))  # 전각 괄호(（）) 도 통일
    s = re.sub(
        r"\((사|재|주|유|합|학|의)\)|㈜|주식회사|유한회사|사단법인|재단법인", "", s
    )
    return re.sub(r"[\s()\[\]·ㆍ_\-,.]", "", s)


def build_agency_index(docs):
    """정규화한 기관명 -> 그 기관 조각의 위치 목록"""
    idx = {}
    for i, d in enumerate(docs):
        key = norm_agency(d.metadata.get("발주 기관", ""))
        if len(key) >= MIN_LEN:
            idx.setdefault(key, []).append(i)
    return idx


def find_agencies(question, agency_idx):
    """질문에 들어 있는 기관명들 (다른 매칭 이름의 일부인 짧은 것은 버림)"""
    q = norm_agency(question)
    hits = [k for k in agency_idx if k in q]
    return [k for k in hits if not any(k != o and k in o for o in hits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="clean_mixed_ocr_v2")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--pool", type=int, default=10, help="리랭커 후보 수")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8-sig"))
    if args.limit:
        golden = golden[: args.limit]
    f2n = evaluate.load_file_to_notice()
    notices = [evaluate.evidence_notice_ids(it, f2n) for it in golden]

    baseline.configure(corpus=args.corpus)
    _, store = baseline.get_resources()
    matrix, docs = baseline.get_exact_index()
    agency_idx = build_agency_index(docs)
    names = np.array([d.metadata.get("사업명", "") for d in docs])
    ks = sorted(set(args.top_k))
    max_k = max(ks + [args.pool])
    print(f"조각 {len(docs)}개, 기관 {len(agency_idx)}곳")

    ce = None
    if not args.no_rerank:
        import torch
        from sentence_transformers import CrossEncoder

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"리랭커 불러오는 중 (장치: {dev})", flush=True)
        ce = CrossEncoder(RERANKER, device=dev, max_length=args.max_length)
        ce.predict([("워밍업", "워밍업")], batch_size=1)

    def topn(dist, idxs, n):
        order = idxs[np.argsort(dist[idxs], kind="stable")[:n]]
        return [(docs[i], float(dist[i])) for i in order]

    def rerank(q, cand):
        sc = ce.predict([(q, d.page_content) for d, _ in cand], batch_size=args.batch)
        o = np.argsort(-np.asarray(sc), kind="stable")
        return [(cand[j][0], float(-sc[j])) for j in o]

    all_idx = np.arange(len(docs))
    details = []
    for n_done, (item, wanted) in enumerate(zip(golden, notices, strict=True), 1):
        q = item["question"]
        t = time.perf_counter()
        qv = np.asarray(store.embeddings.embed_query(q), dtype=np.float32)
        dist = ((matrix - qv) ** 2).sum(axis=1)
        t = time.perf_counter()
        found = find_agencies(q, agency_idx)
        sel = np.concatenate([agency_idx[k] for k in found]) if found else all_idx
        filt_sec = time.perf_counter() - t
        filtered = bool(found)
        # 필터 오작동: 필터를 걸었는데 정답 문서(사업명)가 후보 범위에 없음
        misfilter = bool(
            filtered and wanted and not set(wanted) <= set(names[sel].tolist())
        )

        variants = {
            "baseline": topn(dist, all_idx, max_k),
            "filter": topn(dist, sel, max_k),
        }
        sec = {"baseline": 0.0, "filter": filt_sec}
        if ce is not None:
            t = time.perf_counter()
            variants["rerank"] = rerank(q, topn(dist, all_idx, args.pool))
            sec["rerank"] = time.perf_counter() - t
            t = time.perf_counter()
            variants["filter+rerank"] = rerank(q, topn(dist, sel, args.pool))
            sec["filter+rerank"] = time.perf_counter() - t + filt_sec
        for name, ranked in variants.items():
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
                        "filtered": filtered,
                        "misfilter": misfilter,
                        "extra_sec": round(sec[name], 4),
                    }
                )
        if n_done % 10 == 0 or n_done == len(golden):
            print(f"  {n_done}/{len(golden)}", flush=True)

    df = pd.DataFrame(details)
    rows = []
    for (name, k), part in df.groupby(["variant", "top_k"], sort=False):
        rows.append(
            {
                "variant": name,
                "top_k": k,
                "hit_rate": evaluate.mean_or_none(part["doc_hit"]),
                "mrr": evaluate.mean_or_none(part["rr"]),
                "context_hit": evaluate.mean_or_none(part["context_hit"]),
                "extra_sec_mean": round(float(part["extra_sec"].mean()), 3),
            }
        )
    table = pd.DataFrame(rows)
    one = df[(df.variant == "baseline") & (df.top_k == ks[0])]
    valid = one[one.doc_hit.notna()]
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    baseline.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(
        baseline.OUTPUT_DIR / f"agency_filter_{stamp}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df.to_csv(
        baseline.OUTPUT_DIR / f"agency_filter_{stamp}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n골든셋 {len(golden)}문항 (검색 대상 {len(valid)}문항)")
    print(
        f"기관명을 찾아 필터를 건 문항: {int(valid.filtered.sum())}  / 필터 오작동: {int(valid.misfilter.sum())}"
    )
    print(table.to_string(index=False))
    print(f"\n저장: outputs/agency_filter_{stamp}.csv (+ _detail.csv)")


if __name__ == "__main__":
    main()
