"""리랭커 엔진(engines/rerank_rag) 검색 강건성 측정 (검색만, 답변 모델은 부르지 않음, 토큰 0)

retrieval.py 는 베이스라인 엔진(engines/local_model)만 쓰기 때문에, 리랭커 엔진용으로 따로 만든 실험 파일이다.
--typo-fix 를 주면 실행 중에만 엔진 설정을 바꾸고 (config.py 파일은 그대로), 안 주면 config.py 의 TYPO_FIX 를 그대로 쓴다.
query_fix 교정은 엔진에 들어 있는 코드(engines/rerank_rag/query_fix.py, baseline.get_query_fixer)를 그대로 쓴다.

엔진의 retrieve() 와 똑같은 순서로 검색한다.
    질문 -> fix_typos (팀 오타교정, --typo-fix on 일 때만) -> rewrite_followup (후속질문) -> multi_entity_search
그래서 답변 모델이 실제로 받는 조각과 같은 조각으로 채점된다.

실행 (프로젝트 루트):
    uv run python scripts/typo/typo_retrieval_rerank.py                                   # 원본, 리랭커 켬, 팀 교정 끔
    uv run python scripts/typo/typo_retrieval_rerank.py --golden data/golden/typo/rahui_golden_set_v2_2_typo_medium.json
    uv run python scripts/typo/typo_retrieval_rerank.py --golden ..._typo_medium.json --typo-fix on          # 팀 typo_fix
    uv run python scripts/typo/typo_retrieval_rerank.py --golden ..._typo_medium_fixed.json                 # 내 사전 교정 (미리 고친 파일)
    uv run python scripts/typo/typo_retrieval_rerank.py --golden ..._typo_medium.json --typo-fix query_fix  # 내 사전 교정 (엔진 안에서, 엔진 파일 수정 없음)
    uv run python scripts/typo/typo_retrieval_rerank.py --rerank off ...                                    # 리랭커 끈 비교
    uv run python scripts/typo/typo_retrieval_rerank.py --limit 5                                           # 빠른 시험

결과: outputs/typo/rerank_typo_<시각>.csv (요약), _detail.csv (문항별)
      typo_report.py 에 그대로 넣을 수 있다 (같은 열 이름: chunk_size, chunk_overlap, top_k, hit_rate, mrr, context_hit)

추가 지표 (리랭커 엔진에만 있는 위험을 보기 위해)
    cutoff_refuse : 답이 있는 문항인데 리랭커 1등 점수가 컷오프(0.5)보다 낮아
                    LLM 을 부르지 않고 "문서에 없다"고 답하게 되는 비율 (낮을수록 좋음).
                    오타 때문에 리랭커 점수가 떨어지면 이 값이 오른다.
    noans_refuse  : 답이 없는 문항(no_answer)에서 컷오프로 거절되는 비율 (참고용)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = (
    Path(__file__).resolve().parents[2]
)  # 프로젝트 루트 (scripts/typo/ 에서 두 단계 위)
sys.path.insert(
    0, str(ROOT)
)  # 루트의 evaluate.py, engines/ 를 불러오기 위해 (팀 scripts/rerank_rag/ 와 같은 방식)

from engines.local_model import baseline as local_baseline  # noqa: E402  # evaluate.load_golden 이 읽는 골든셋 경로
from engines.rerank_rag import baseline as rr  # noqa: E402  # 리랭커 엔진

import evaluate  # noqa: E402  # 팀 공용 채점 함수 (retrieval_scores, context_hit 등) - 고치지 않고 불러만 씀


def use_query_fix(engine):
    """--typo-fix query_fix 일 때 엔진이 내 사전 교정을 쓰게 한다.

    엔진 baseline.py 에 교정 코드(get_query_fixer)가 이미 들어 있으면 그 코드를 그대로 쓴다 (엔진 통합 검증).
    없으면 실행 중에만 엔진의 fix_typos 를 scripts/typo/query_fix.py 로 바꿔 끼운다 (파일은 그대로).
    """
    if hasattr(engine, "get_query_fixer"):
        print("[오타 교정] 엔진 baseline.py 에 넣은 교정 코드를 사용합니다")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 query_fix.py
    from query_fix import QueryFixer

    engine.fix_typos = QueryFixer(
        data_list=engine.DATA_PATH, corpus_dir=engine.TXT_DIR
    ).fix
    print(
        "[오타 교정] 엔진에 교정 코드가 없어 실행 중에만 scripts/typo/query_fix.py 를 끼웁니다"
    )


def parse_args():
    p = argparse.ArgumentParser(description="리랭커 엔진 검색 강건성 측정 (검색만)")
    p.add_argument(
        "--golden", default=None, help="골든셋 경로 (기본: 엔진 config 의 GOLDEN_FILE)"
    )
    p.add_argument(
        "--rerank",
        choices=["on", "off"],
        default="on",
        help="리랭커 켜기/끄기 (기본 on)",
    )
    p.add_argument(
        "--typo-fix",
        choices=["off", "on", "query_fix"],
        default=None,
        help="검색 직전 오타 교정 (안 주면 config.py 의 TYPO_FIX 그대로 / off: 끔 / on: 팀 교정 / query_fix: 내 사전 교정)",
    )
    p.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (빠른 시험)")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- 1. 골든셋: evaluate.load_golden 은 베이스라인 엔진의 GOLDEN_PATH 를 읽으므로 그쪽에도 같은 경로를 넣는다
    golden_rel = args.golden or rr.config.GOLDEN_FILE
    local_baseline.configure(golden=golden_rel)
    rr.configure(golden=golden_rel)
    golden = evaluate.load_golden(args.limit)
    file_to_notice = evaluate.load_file_to_notice()
    notices = [evaluate.evidence_notice_ids(it, file_to_notice) for it in golden]

    # ---- 2. 엔진 설정을 실행 중에만 바꿈 (config.py 파일은 그대로)
    rr.set_search(rerank=(args.rerank == "on"))
    if (
        args.typo_fix is not None
    ):  # 안 주면 config.py 의 TYPO_FIX 를 그대로 씀 (엔진 기본 동작 확인용)
        rr.TYPO_FIX = {"off": False, "on": True, "query_fix": "query_fix"}[
            args.typo_fix
        ]
    if rr.TYPO_FIX == "query_fix":
        use_query_fix(rr)

    print(
        f"골든셋: {Path(golden_rel).name} ({len(golden)}문항) | 리랭커 {args.rerank} | 오타 교정 {rr.TYPO_FIX!r}"
    )
    rr.get_resources()  # 벡터 DB 준비 (없으면 만듦)
    rr.get_exact_index()
    if rr.RERANK:
        rr.get_reranker()  # 처음이면 모델을 내려받아서 오래 걸릴 수 있음 (첫 문항 시간에 섞이지 않게 미리)

    # ---- 3. 문항마다 엔진 retrieve() 와 같은 순서로 검색
    cutoff = rr.RERANK_CUTOFF if rr.RERANK else None
    details = []
    for item, wanted_ids in zip(golden, notices, strict=True):
        start = time.time()
        query = rr.fix_typos(item["question"])
        query = rr.rewrite_followup(query, item.get("history") or [])
        try:
            retrieved = rr.multi_entity_search(query)
            error = ""
        except Exception as e:  # 한 문항이 실패해도 계속
            retrieved, error = [], str(e)
        sec = round(time.time() - start, 3)

        doc_hit, rr_score = evaluate.retrieval_scores(item, retrieved, wanted_ids)
        # 리랭커를 쓰면 점수는 '-리랭커점수'로 저장돼 있다 (엔진 generate() 와 같은 방식으로 복원)
        top1 = -retrieved[0][1] if (retrieved and rr.RERANK) else None
        refused = bool(cutoff is not None and top1 is not None and top1 < cutoff)
        details.append(
            {
                "id": item["id"],
                "level": item["level"],
                "question": item["question"],
                "query_used": query,
                "doc_hit": doc_hit,
                "rr": rr_score,
                "context_hit": evaluate.context_hit(item, retrieved),
                "top1_score": None if top1 is None else round(top1, 4),
                "cutoff_refuse": refused,
                "n_chunks": len(retrieved),
                "search_sec": sec,
                "error": error,
                "sources": " || ".join(
                    f"{d.metadata.get('공고 번호')} {d.metadata.get('사업명')}"
                    for d, _ in retrieved
                ),
            }
        )

    # ---- 4. 요약 (retrieval.py 와 같은 열 이름 -> typo_report.py 로 바로 비교 가능)
    det = pd.DataFrame(details)
    det.insert(0, "top_k", rr.TOP_K)
    det.insert(0, "chunk_overlap", rr.CHUNK_OVERLAP)
    det.insert(0, "chunk_size", rr.CHUNK_SIZE)
    answerable = det[
        det["doc_hit"].notna()
    ]  # 답이 있는 문항 (no_answer 는 doc_hit 이 비어 있음)
    noanswer = det[det["doc_hit"].isna()]
    summary = {
        "engine": "rerank_rag",
        "golden": Path(golden_rel).stem,
        "rerank": args.rerank,
        "typo_fix": rr.TYPO_FIX,
        "chunk_size": rr.CHUNK_SIZE,
        "chunk_overlap": rr.CHUNK_OVERLAP,
        "top_k": rr.TOP_K,
        "questions": len(det),
        "hit_rate": evaluate.mean_or_none(det["doc_hit"]),
        "mrr": evaluate.mean_or_none(det["rr"]),
        "context_hit": evaluate.mean_or_none(det["context_hit"]),
        "cutoff_refuse": round(float(answerable["cutoff_refuse"].mean()), 4)
        if len(answerable)
        else None,
        "noans_refuse": round(float(noanswer["cutoff_refuse"].mean()), 4)
        if len(noanswer)
        else None,
        "search_sec": evaluate.mean_or_none(det["search_sec"]),
        "error_count": int((det["error"] != "").sum()),
    }

    out_dir = (
        ROOT / "outputs" / "typo"
    )  # 오타 실험 결과는 outputs/typo/ 에 따로 모음 (팀 결과와 섞이지 않게)
    out_dir.mkdir(parents=True, exist_ok=True)  # outputs/typo 폴더가 없으면 만듦
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    summary_path = out_dir / f"rerank_typo_{stamp}.csv"
    detail_path = out_dir / f"rerank_typo_{stamp}_detail.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    det.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("\n=== 검색 지표 (0~1) ===")
    print(pd.DataFrame([summary]).drop(columns=["engine"]).to_string(index=False))
    print(f"\n비교표 저장: {summary_path}\n문항별 상세 저장: {detail_path}")


if __name__ == "__main__":
    main()
