"""오타 강건성 판정표 (오타 강건성 실험 마지막 단계)

retrieval.py 또는 typo_retrieval_rerank.py 가 만든 결과 csv 세 개(A 원본 / B 오타 / C 오타+보정)를 받아 회복률을 계산하고,
문항별로 무엇이 바뀌었는지(오타로 놓친 문항, 보정으로 되찾은 문항)도 보여준다.

    회복률 = (C - B) / (A - B)    ->  0.5 이상이면 통과 (하락의 절반 이상 회복)

실행 (프로젝트 루트):
    uv run python scripts/typo/typo_report.py --a outputs/typo/rerank_typo_A.csv --b outputs/typo/rerank_typo_B.csv --c outputs/typo/rerank_typo_C.csv
    # 결과 csv 는 이름 끝이 _detail 이 아닌 '비교표' 파일을 넣는다.
    # 같은 이름의 _detail.csv 가 옆에 있으면 문항별 변화도 자동으로 출력한다.
    # --out outputs/typo/typo_report_medium.csv 를 붙이면 판정표를 파일로도 저장한다.

답변 평가(evaluate.py) 결과로 판정할 때는 볼 지표를 직접 고른다:
    uv run python scripts/typo/typo_report.py --a A.csv --b B.csv --c C.csv --metrics accuracy keyword_recall hallucination
"""

import argparse
from pathlib import Path

import pandas as pd

METRICS = ("hit_rate", "mrr", "context_hit")  # 기본으로 판정할 지표 (retrieval.py 비교표의 열 이름)
LOWER_IS_BETTER = {"hallucination", "latency_sec", "search_sec", "cutoff_refuse"}  # 낮을수록 좋은 지표 (하락/회복 방향을 뒤집음)
KEYS = ["chunk_size", "chunk_overlap", "top_k"]  # A, B, C 에서 같은 조건끼리 짝지을 때 쓰는 열
MIN_DROP = 0.02  # 하락이 이보다 작으면 회복률을 계산하지 않음 (61문항 기준 약 1문항 차이 = 우연 수준)
PASS_RATIO = 0.5  # 회복률 통과 기준


# =====================================================================
# 1. 판정표 (조건별 x 지표별)
# =====================================================================
def judge(a, b, c, lower_is_better=False):
    """A, B, C 점수 세 개 -> (회복률, 판정 글자)"""
    if lower_is_better:  # 환각률처럼 낮을수록 좋은 지표는 부호를 뒤집어 '높을수록 좋음'으로 맞춤
        a, b, c = -a, -b, -c
    drop = a - b  # 오타 때문에 나빠진 양
    if drop < MIN_DROP:  # 거의 안 떨어졌으면 '회복'을 말할 수 없음
        return None, "하락 작음(판정 불가)"
    ratio = (c - b) / drop  # 떨어진 양 중 되찾은 비율
    return ratio, ("통과" if ratio >= PASS_RATIO else "미달")


def pick_same_condition(df, row):
    """df 에서 row 와 조건(조각 크기, 겹침, top_k)이 같은 행 하나. 없으면 None"""
    match = df
    for key in [k for k in KEYS if k in df.columns and k in row.index]:  # 결과 파일에 있는 조건 열만 사용
        match = match[match[key] == row[key]]
    return match.iloc[0] if len(match) else None


def build_table(df_a, df_b, df_c, metrics=METRICS):
    rows = []
    keys = [k for k in KEYS if k in df_a.columns]
    for _, a in df_a.iterrows():  # A 의 조건 하나마다
        b, c = pick_same_condition(df_b, a), pick_same_condition(df_c, a)
        if b is None or c is None:  # B 나 C 에 같은 조건 결과가 없으면 비교 불가
            continue
        for metric in metrics:
            if metric not in df_a.columns:  # 파일에 없는 지표는 건너뜀
                continue
            lower = metric in LOWER_IS_BETTER
            sign = -1 if lower else 1
            ratio, verdict = judge(a[metric], b[metric], c[metric], lower)
            rows.append({
                **{k: a[k] for k in keys},
                "metric": metric,
                "A_원본": round(a[metric], 4),
                "B_오타": round(b[metric], 4),
                "C_보정": round(c[metric], 4),
                # 낮을수록 좋은 지표는 부호를 뒤집어서 '나빠진 양'과 '되찾은 양'이 항상 양수로 보이게
                "하락": round(sign * (a[metric] - b[metric]), 4),
                "회복": round(sign * (c[metric] - b[metric]), 4),
                "회복률": None if ratio is None else round(ratio, 2),
                "판정": verdict,
            })
    return pd.DataFrame(rows)


# =====================================================================
# 2. 문항별 변화 (detail csv 가 있을 때)
# =====================================================================
def detail_path(summary_path):
    """outputs/retrieval_X.csv -> outputs/retrieval_X_detail.csv"""
    p = Path(summary_path)
    return p.with_name(p.stem + "_detail.csv")


def question_changes(path_a, path_b, path_c):
    """문항별 doc_hit(정답 문서를 찾았으면 1)을 A, B, C 나란히 놓고 바뀐 문항만 골라 보여준다."""
    paths = {"A": detail_path(path_a), "B": detail_path(path_b), "C": detail_path(path_c)}
    if not all(p.exists() for p in paths.values()):
        print("\n(문항별 상세 _detail.csv 를 찾지 못해 문항별 변화는 건너뜁니다)")
        return

    # 각 파일에서 필요한 열만 꺼내 이름 뒤에 _A, _B, _C 를 붙인 뒤 문항 id 로 옆으로 이어 붙인다
    merged = None
    for name, path in paths.items():
        d = pd.read_csv(path)
        if "doc_hit" not in d.columns:  # 검색 결과 상세가 아니면(답변 평가 등) 건너뜀
            print(f"\n({path.name} 에 doc_hit 열이 없어 문항별 변화는 건너뜁니다)")
            return
        if "top_k" not in d.columns:
            d["top_k"] = "-"
        cols = ["id", "top_k", "doc_hit"] + (["query_used"] if "query_used" in d.columns else [])
        d = d[cols].rename(columns={"doc_hit": f"hit_{name}", "query_used": f"query_{name}"})
        merged = d if merged is None else merged.merge(d, on=["id", "top_k"], how="inner")

    merged = merged.dropna(subset=["hit_A", "hit_B", "hit_C"])  # 정답 문서가 없는 문항(no_answer)은 빈칸이라 제외
    for k, part in merged.groupby("top_k"):
        lost = part[(part.hit_A == 1) & (part.hit_B == 0)]  # 원본은 맞췄는데 오타 때문에 놓침
        saved = lost[lost.hit_C == 1]  # 그중 보정으로 되찾음
        still = lost[lost.hit_C == 0]  # 보정으로도 못 되찾음
        hurt = part[(part.hit_B == 1) & (part.hit_C == 0)]  # 오타로는 맞췄는데 보정 때문에 놓침 (부작용)

        print(f"\n[top_k={k}] 문항별 변화 (정답 문서를 찾았는지 기준)")
        print(f"  오타 때문에 놓친 문항: {len(lost)}개 {list(lost.id)}")
        print(f"    -> 보정으로 되찾음: {len(saved)}개 {list(saved.id)}")
        print(f"    -> 못 되찾음:       {len(still)}개 {list(still.id)}")
        print(f"  보정 때문에 오히려 놓친 문항: {len(hurt)}개 {list(hurt.id)}")
        if "query_B" in part.columns and "query_C" in part.columns:
            for _, r in still.iterrows():  # 못 되찾은 문항은 실제 검색어를 보여줘서 원인을 볼 수 있게
                print(f"    {r.id} 오타: {str(r.query_B)[:50]}")
                print(f"    {' ' * len(str(r.id))} 보정: {str(r.query_C)[:50]}")


# =====================================================================
# 실행
# =====================================================================
def main():
    p = argparse.ArgumentParser(description="오타 강건성 회복률 계산")
    p.add_argument("--a", required=True, help="원본 골든셋 결과 csv (A)")
    p.add_argument("--b", required=True, help="오타 골든셋 결과 csv, 보정 없음 (B)")
    p.add_argument("--c", required=True, help="오타 골든셋 + 보정 결과 csv (C)")
    p.add_argument("--out", default=None, help="판정표 저장 경로 (선택)")
    p.add_argument("--metrics", nargs="+", default=list(METRICS), help="판정할 지표 열 이름 (기본: 검색 지표 3개)")
    args = p.parse_args()

    table = build_table(pd.read_csv(args.a), pd.read_csv(args.b), pd.read_csv(args.c), args.metrics)
    if table.empty:
        raise SystemExit("A, B, C 에 같은 조건 결과나 고른 지표가 없습니다. 파일과 --metrics 를 확인하세요.")
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n판정표 저장: {args.out}")

    question_changes(args.a, args.b, args.c)


if __name__ == "__main__":
    main()
