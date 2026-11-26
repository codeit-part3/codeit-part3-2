"""검색 전용 평가 스크립트 (답변 모델은 부르지 않고 검색 지표만 비교)

실행 예
    uv run python retrieval.py --top-k 1 3 5 --chunk-size 300 500 800
    uv run python retrieval.py --top-k 3 5 --chunk-overlap 50 100
    uv run python retrieval.py --top-k 3 5 --limit 5                 # 빠른 확인 (시트 기록 안 함)
    uv run python retrieval.py --chunk-size 500 800 --name cs --sheet
    uv run python retrieval.py --corpus processed --golden data/goldenset.json   # 예전 데이터

하는 일
1. chunk_size, chunk_overlap, top_k 를 여러 개 주면 모든 조합의 검색 지표를 계산
   (hit_rate, mrr, context_hit: evaluate.py 와 같은 계산 함수를 그대로 사용)
2. 조합별 비교표와 문항별 상세를 outputs/ 에 저장
3. --sheet 를 주면 구글 시트(기본: evaluate.py 와 같은 탭)에 조합마다 한 줄 기록
"""

import argparse
import time
from datetime import datetime

import pandas as pd

import evaluate
from engines.local_final import baseline

EMBED_METHOD = "ollama"  # 임베딩 방식 (선택 옵션은 다음 작업에서 추가)
SHEET_METRICS = ("hit_rate", "mrr", "context_hit", "search_sec")


def chunk_count():
    """방금 쓴 벡터 DB 의 조각 수 (build_done.txt 에서 읽음)"""
    try:
        text = baseline.DONE_FILE.read_text(encoding="utf-8")
        return int(text.split("=")[1].strip())
    except (OSError, ValueError, IndexError):
        return None


def run_combo(golden, notices, chunk_size, chunk_overlap, top_ks):
    """한 가지 조각 설정에서 top_k 별 검색 지표를 계산 -> (요약 행 목록, 상세 행 목록)

    질문마다 가장 큰 top_k 로 한 번만 검색하고, 작은 k 는 앞에서 잘라 쓴다.
    (search_sec 는 가장 큰 top_k 로 검색한 시간)
    """
    max_k = max(top_ks)
    baseline.configure(chunk_size, chunk_overlap, max_k)
    baseline.get_resources()  # 벡터 DB 준비 (없으면 새로 만듦)
    if baseline.EXACT_SEARCH:
        baseline.get_exact_index()  # 불러오는 시간이 첫 문항 검색 시간에 섞이지 않게
    n_chunks = chunk_count()
    search_mode = baseline.current_settings()["search"]

    details = []
    for item, wanted_ids in zip(golden, notices, strict=True):
        start = time.time()
        try:
            retrieved = baseline.search(item["question"], k=max_k)
            error = ""
        except Exception as e:  # 한 문항이 실패해도 나머지는 계속 실행
            retrieved = []
            error = str(e)
        search_sec = round(time.time() - start, 3)

        for k in top_ks:
            top = retrieved[:k]
            doc_hit, rr = evaluate.retrieval_scores(item, top, wanted_ids)
            details.append(
                {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "top_k": k,
                    "id": item["id"],
                    "level": item["level"],
                    "question": item["question"],
                    "doc_hit": doc_hit,
                    "rr": rr,
                    "context_hit": evaluate.context_hit(item, top),
                    "search_sec": search_sec,
                    "error": error,
                    "sources": " || ".join(
                        f"{doc.metadata['공고 번호']} {doc.metadata['사업명']}"
                        f" (거리 {score:.3f})"
                        for doc, score in top
                    ),
                }
            )

    detail_df = pd.DataFrame(details)
    summary = []
    for k in top_ks:
        part = detail_df[detail_df["top_k"] == k]
        summary.append(
            {
                "corpus": baseline.CORPUS,
                "golden": baseline.GOLDEN_PATH.stem,
                "embed_method": EMBED_METHOD,
                "embed_model": baseline.EMBED_MODEL,
                "search": search_mode,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": k,
                "chunks": n_chunks,
                "questions": len(part),
                "hit_rate": evaluate.mean_or_none(part["doc_hit"]),
                "mrr": evaluate.mean_or_none(part["rr"]),
                "context_hit": evaluate.mean_or_none(part["context_hit"]),
                "search_sec": evaluate.mean_or_none(part["search_sec"]),
                "error_count": int((part["error"] != "").sum()),
            }
        )
    return summary, details


def log_rows_to_sheet(rows, name, memo, tab):
    from common.sheet_log import log_experiment

    for row in rows:
        metrics = {k: row[k] for k in SHEET_METRICS if row[k] is not None}
        params = {  # 시트 experiment 칸에 글자로 들어감 (열이 없어도 기록이 남음)
            "engine": "retrieval",
            "corpus": row["corpus"],
            "golden": row["golden"],
            "embed_method": row["embed_method"],
            "embed_model": row["embed_model"],
            "search": row["search"],
            "chunk_size": row["chunk_size"],
            "chunk_overlap": row["chunk_overlap"],
            "top_k": row["top_k"],
        }
        log_experiment(
            name=f"{name}_cs{row['chunk_size']}_ov{row['chunk_overlap']}_k{row['top_k']}",
            memo=memo,
            metrics=metrics,
            params=params,
            tab=tab,
            extra={
                "embed_method": row["embed_method"],
                "embed_model": row["embed_model"],
                "chunk_size": row["chunk_size"],
                "chunk_overlap": row["chunk_overlap"],
                "top_k": row["top_k"],
                "questions": row["questions"],
            },
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="검색 지표(hit_rate, mrr, context_hit)만 여러 설정으로 비교합니다."
    )
    parser.add_argument(
        "--golden",
        default=None,
        help="골든셋 파일 경로 (기본: config.py 의 GOLDEN_FILE)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="앞에서 N문항만 실행 (시트 기록 안 함)"
    )

    exp = parser.add_argument_group("비교할 설정 (여러 값을 주면 모든 조합 실행)")
    exp.add_argument("--top-k", type=int, nargs="+", default=[baseline.TOP_K])
    exp.add_argument("--chunk-size", type=int, nargs="+", default=[baseline.CHUNK_SIZE])
    exp.add_argument(
        "--chunk-overlap", type=int, nargs="+", default=[baseline.CHUNK_OVERLAP]
    )
    exp.add_argument(
        "--corpus",
        default=None,
        help="말뭉치 폴더 이름 (data/ 아래, 기본: config.py 의 CORPUS)",
    )

    rec = parser.add_argument_group("기록")
    rec.add_argument("--name", default="retrieval", help="시트 실험 이름의 앞부분")
    rec.add_argument("--memo", default="", help="시트에 남길 한 줄 설명")
    rec.add_argument(
        "--sheet", action="store_true", help="구글 시트에 기록 (기본 꺼짐)"
    )
    rec.add_argument(
        "--tab",
        default=None,
        help="기록할 시트 탭 이름 (안 쓰면 evaluate.py 와 같은 기본 탭)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    top_ks = sorted(set(args.top_k))

    # 말뭉치와 골든셋은 모든 조합에 공통
    baseline.configure(corpus=args.corpus, golden=args.golden)
    print(f"말뭉치: data/{baseline.CORPUS} | 골든셋: {baseline.GOLDEN_PATH.name}")
    golden = evaluate.load_golden(args.limit)

    file_to_notice = evaluate.load_file_to_notice()
    notices = [evaluate.evidence_notice_ids(it, file_to_notice) for it in golden]

    combos = []
    for size in args.chunk_size:
        for overlap in args.chunk_overlap:
            if 0 <= overlap < size:
                combos.append((size, overlap))
            else:
                print(
                    f"건너뜀: chunk_size={size}, chunk_overlap={overlap} (겹침이 너무 큼)"
                )
    if not combos:
        raise SystemExit("실행할 조합이 없습니다.")
    print(
        f"골든셋 {len(golden)}문항, 조각 설정 {len(combos)}개 x top_k {top_ks} 를 평가합니다."
    )

    baseline.OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = baseline.OUTPUT_DIR / f"retrieval_{stamp}.csv"
    detail_path = baseline.OUTPUT_DIR / f"retrieval_{stamp}_detail.csv"
    use_sheet = args.sheet and not args.limit
    if args.sheet and args.limit:
        print("[시트 기록 안 함] --limit 로 일부 문항만 실행함")

    all_summary, all_details = [], []
    for size, overlap in combos:
        print(f"\n--- chunk_size={size}, chunk_overlap={overlap} ---")
        summary, details = run_combo(golden, notices, size, overlap, top_ks)
        all_summary += summary
        all_details += details
        # 조합마다 저장 (중간에 멈춰도 여기까지 결과는 남음)
        pd.DataFrame(all_summary).to_csv(
            summary_path, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(all_details).to_csv(detail_path, index=False, encoding="utf-8-sig")
        if use_sheet:
            log_rows_to_sheet(summary, args.name, args.memo, args.tab)

    table = pd.DataFrame(all_summary)
    print("\n=== 검색 지표 비교 (0~1, 높을수록 좋음) ===")
    print(table.drop(columns=["embed_method", "embed_model"]).to_string(index=False))
    print(f"\n비교표 저장: {summary_path}\n문항별 상세 저장: {detail_path}")
    if not use_sheet and not args.limit:
        print("[시트 기록 안 함] 기록하려면 --sheet 를 붙이세요")


if __name__ == "__main__":
    main()
