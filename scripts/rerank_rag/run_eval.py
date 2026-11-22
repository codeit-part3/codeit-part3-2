"""기관명 필터 + 리랭커를 넣어 evaluate.py 와 똑같이 전체 평가하는 실행 파일

evaluate.py / baseline.py 는 한 줄도 고치지 않는다.
  - evaluate.main() 이 쓰는 make_answer_fn 만 이 파일에서 바꿔 끼운다.
  - 검색만 (기관명 필터 -> 후보 -> 리랭커) 로 바꾸고, 답변 생성은 baseline.generate 를 그대로 쓴다.
  - 채점, LLM 채점(judge), 시트 기록은 evaluate.py 가 하던 그대로다.

실행 (프로젝트 루트에서)
    # 빠른 확인 (5문항, LLM 채점·시트 기록 없음)
    uv run python scripts/rerank_rag/run_eval.py --corpus clean_mixed_ocr_v2 --limit 5 --no-judge
    # 전체 평가 + 시트 기록
    uv run python scripts/rerank_rag/run_eval.py --corpus clean_mixed_ocr_v2 \
        --golden data/golden/rahui_golden_set_v2_2.json --pool 10

이 파일이 추가로 받는 옵션: --variant (filter_rerank | filter | rerank), --pool, --max-length, --rerank-batch
  --pkg      : 이 파일의 검색기 대신 engines/rerank_rag/ 의 엔진(구조 청킹 1500 기본)으로 실행
  --chunker  : --pkg 일 때 청킹 방식 fixed | struct (기본: engines/rerank_rag/config.py)
나머지 옵션(--corpus, --golden, --limit, --no-sheet, --no-judge, --memo ...)은 evaluate.py 와 같다.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # 프로젝트 루트
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from agency_filter import RERANKER, build_agency_index, find_agencies  # noqa: E402

import evaluate  # noqa: E402
from engines.local_model import baseline  # noqa: E402

# 추가: 채점 규칙 보강: evaluate.py 는 그대로 두고, 이 파일로 실행할 때만 적용
# "사업예산은 비공개" 처럼 금액이 공개되지 않았다고 답한 경우를 거절로 인정 (q43 골든셋 정답 기준)
if "비공개" not in evaluate.DONT_KNOW:
    evaluate.DONT_KNOW.append("비공개")


# 추가: LLM 채점 보정: 답 없는 문항에서 규칙상 거절(DONT_KNOW 표현 + 금액 없음)이면 refused=True 로 인정
# (LLM 채점이 고정 거절 문구도 '거절 아님'으로 판정하는 사례가 있어서. judge.py 는 그대로 둠)
def _rule_refused(answer):
    text = str(answer)
    squashed = evaluate.squash(text)
    said = any(evaluate.squash(p) in squashed for p in evaluate.DONT_KNOW)
    return said and re.search(evaluate.MONEY, text) is None


_orig_summarize_judge = evaluate.judge.summarize_judge


def _summarize_judge_with_rule(df):
    df = df.copy()
    fix = (df["level"] == "no_answer") & df["answer"].fillna("").map(_rule_refused)
    df.loc[fix, "judge_refused"] = True
    return _orig_summarize_judge(df)


evaluate.judge.summarize_judge = _summarize_judge_with_rule


class FilterRerankSearcher:
    """baseline.search 와 같은 모양 [(조각, 점수), ...] 을 돌려주는 검색기"""

    def __init__(self, variant, pool, max_length, batch):
        self.use_filter = "filter" in variant
        self.use_rerank = "rerank" in variant
        self.pool, self.batch = pool, batch
        _, self.store = baseline.get_resources()
        self.matrix, self.docs = baseline.get_exact_index()
        self.all_idx = np.arange(len(self.docs))
        self.agency_idx = build_agency_index(self.docs)
        self.ce = None
        if self.use_rerank:
            import torch
            from sentence_transformers import CrossEncoder

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"리랭커 불러오는 중: {RERANKER} (장치: {dev})", flush=True)
            self.ce = CrossEncoder(RERANKER, device=dev, max_length=max_length)
            self.ce.predict([("워밍업", "워밍업")], batch_size=1)

    def _top(self, dist, idxs, n):
        order = idxs[np.argsort(dist[idxs], kind="stable")[:n]]
        return [(self.docs[i], float(dist[i])) for i in order]

    def search(self, question, k):
        qv = np.asarray(self.store.embeddings.embed_query(question), dtype=np.float32)
        dist = ((self.matrix - qv) ** 2).sum(axis=1)
        sel = self.all_idx
        if self.use_filter:
            found = find_agencies(question, self.agency_idx)
            if found:  # 기관명을 못 찾으면 필터 없이 (베이스라인과 같게)
                sel = np.concatenate([self.agency_idx[a] for a in found])
        if not self.use_rerank:
            return self._top(dist, sel, k)
        cand = self._top(dist, sel, max(self.pool, k))
        scores = self.ce.predict(
            [(question, d.page_content) for d, _ in cand], batch_size=self.batch
        )
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [(cand[j][0], float(-scores[j])) for j in order][:k]


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--variant",
        default="filter_rerank",
        choices=["filter_rerank", "filter", "rerank"],
    )
    ap.add_argument("--pool", type=int, default=10, help="리랭커 후보 수")
    ap.add_argument(
        "--max-length", type=int, default=None, help="없으면 config.RERANK_MAX_LENGTH"
    )
    ap.add_argument(
        "--fp16", action="store_true", help="리랭커 fp16 (GPU, 속도 실험용)"
    )
    ap.add_argument(
        "--rerank-batch", type=int, default=None, help="없으면 config.RERANK_BATCH"
    )
    ap.add_argument(
        "--pkg", action="store_true", help="engines/rerank_rag 의 엔진으로 실행"
    )
    ap.add_argument("--chunker", default=None, choices=["fixed", "struct"])
    ap.add_argument(
        "--rewrite", action="store_true", help="후속질문 사업명/기관명 이어 붙이기"
    )
    ap.add_argument(
        "--cutoff", type=float, default=None, help="리랭커 1등 점수 컷오프 (예: 0.5)"
    )
    ap.add_argument(
        "--multi-entity",
        action="store_true",
        help="사업/기관 2개 이상이면 사업별 검색 병합",
    )
    mine, rest = ap.parse_known_args()
    if mine.max_length is None:
        from engines.rerank_rag import config as _cfg

        mine.max_length = _cfg.RERANK_MAX_LENGTH
    if mine.rerank_batch is None:
        from engines.rerank_rag import config as _cfg2

        mine.rerank_batch = _cfg2.RERANK_BATCH

    # if "--engine" in rest and rest[rest.index("--engine") + 1] == "agent":
    #     raise SystemExit(
    #         "이 파일은 rag 방식만 지원합니다 (--engine agent 는 쓰지 않음)."
    #     )
    # evaluate.py 쪽 기본값 채우기: 결과 이름과 시트 메모에 어떤 조합인지 남김
    tag = (
        ("pkg_" if mine.pkg else "")
        + mine.variant
        + (str(mine.pool) if "rerank" in mine.variant else "")
        + (f"_{mine.chunker}" if mine.pkg and mine.chunker else "")
        + ("_rw" if mine.pkg and mine.rewrite else "")
        + (
            f"_c{str(mine.cutoff).replace('.', '')}"
            if mine.pkg and mine.cutoff is not None
            else ""
        )
        + ("_me" if mine.pkg and mine.multi_entity else "")
    )
    if "--version" not in rest:
        rest += ["--version", tag]
    if "--memo" not in rest:
        desc = {
            "filter_rerank": f"기관명 필터 + bge-reranker 후보{mine.pool}",
            "filter": "기관명 필터",
            "rerank": f"bge-reranker 후보{mine.pool}",
        }[mine.variant]
        if mine.pkg:
            desc += " [rahui_engine]"
        rest += ["--memo", desc]
    sys.argv = [sys.argv[0], *rest]

    if (
        mine.pkg
    ):  # engines/rerank_rag/baseline.py 를 evaluate.py 의 baseline 자리에 통째로 끼움
        from engines.rerank_rag import baseline as eng

        eng.set_search(
            agency_filter="filter" in mine.variant,
            rerank="rerank" in mine.variant,
            pool=mine.pool,
            max_length=mine.max_length,
            batch=mine.rerank_batch,
            chunker=mine.chunker,
            fp16=mine.fp16 or None,
            followup_rewrite=mine.rewrite or None,
            cutoff=mine.cutoff,
            multi_entity=mine.multi_entity or None,
        )
        evaluate.baseline = eng
        evaluate.main()
        return

    def make_answer_fn(_engine):
        searcher = FilterRerankSearcher(
            mine.variant, mine.pool, mine.max_length, mine.rerank_batch
        )

        def answer_fn(question, history):
            retrieved = searcher.search(question, baseline.TOP_K)
            out = baseline.generate(
                {"question": question, "history": history or [], "retrieved": retrieved}
            )
            return {
                "answer": out["answer"],
                "retrieved": retrieved,
                "prompt_tokens": out.get("prompt_tokens"),
                "n_tool_calls": None,
            }

        return answer_fn

    evaluate.make_answer_fn = make_answer_fn  # evaluate.main() 이 이 함수를 쓰게 함
    evaluate.main()


if __name__ == "__main__":
    main()
