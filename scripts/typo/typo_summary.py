"""오타 강건성 전체 요약: outputs/ 와 outputs/typo/ 의 검색 결과를 모두 모아 seed 여러 개의 결과를 한 표로 정리

파일 경로를 하나하나 적을 필요 없이, retrieval.py 가 만든 비교표 csv 를 자동으로 찾는다.
골든셋 이름(rahui_golden_set_v2_2_typo_medium_s7)에서 오타 세기와 seed 를 읽어 A·B·C 를 짝짓는다.

실행 (프로젝트 루트):
    uv run python scripts/typo/typo_summary.py
    uv run python scripts/typo/typo_summary.py --metric mrr          # 다른 지표로
    uv run python scripts/typo/typo_summary.py --out outputs/typo/typo_summary.csv

판정 방식
    seed 마다 회복률을 따로 내면 하락이 1~2문항일 때 값이 크게 출렁인다 (0.5 -> 1.0).
    그래서 세기별로 모든 seed 의 하락과 회복을 '더한 뒤' 나눈다 (합산 회복률).
        합산 회복률 = (seed 별 회복의 합) / (seed 별 하락의 합)
"""

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # 프로젝트 루트 (scripts/typo/ 에서 두 단계 위)
OUTPUT_DIR = BASE_DIR / "outputs"
ORIGINAL_GOLDEN = "rahui_golden_set_v2_2"  # 원본 골든셋 이름 (A)
DEFAULT_SEED = 42  # typo_golden.py 의 기본 seed (파일 이름에 _s 가 없으면 42)
SCORED_QUESTIONS = 61  # 검색 지표를 계산하는 문항 수 (답이 있는 문항). 점수 -> 문항 수 환산용
MIN_DROP = 0.02  # typo_report.py 와 같은 기준
PASS_RATIO = 0.5

# 골든셋 이름에서 세기와 seed 를 읽는 정규식. 예: ..._typo_medium_s7 -> ('medium', '7')
TYPO_NAME = re.compile(rf"^{ORIGINAL_GOLDEN}_typo_(light|medium|heavy)(?:_s(\d+))?$")


def load_runs():
    """outputs/ 와 outputs/typo/ 의 retrieval_*.csv (비교표) 를 모두 읽어 한 표로. 같은 조건을 여러 번 돌렸으면 가장 최근 것만"""
    frames = []
    # retrieval.py(팀 파일)는 outputs/ 에 저장하므로, 나중에 outputs/typo/ 로 옮긴 파일도 함께 읽는다
    found = [*OUTPUT_DIR.glob("retrieval_*.csv"), *(OUTPUT_DIR / "typo").glob("retrieval_*.csv")]
    for path in sorted(found, key=lambda p: p.name):  # 이름에 시각이 있어서 이름순 = 시간순 (폴더가 달라도)
        if path.stem.endswith("_detail"):  # 문항별 상세 파일은 제외
            continue
        df = pd.read_csv(path)
        if "query_fix" not in df.columns:  # 보정 옵션이 생기기 전의 옛 결과는 제외
            continue
        df["file"] = path.name
        frames.append(df)
    if not frames:
        raise SystemExit("outputs/ 와 outputs/typo/ 에 retrieval 결과가 없습니다.")
    runs = pd.concat(frames, ignore_index=True)
    runs = runs[runs["questions"] >= 70]  # --limit 로 일부만 돌린 시험 실행은 제외 (74문항 전체만)
    keys = ["golden", "query_fix", "chunk_size", "chunk_overlap", "top_k"]
    return runs.drop_duplicates(subset=keys, keep="last")  # 같은 조건이면 마지막(최신) 실행만 남김


def main():
    p = argparse.ArgumentParser(description="오타 강건성 결과를 seed 별로 모아 요약합니다.")
    p.add_argument("--metric", default="hit_rate", help="판정 지표 (기본 hit_rate)")
    p.add_argument("--out", default=None, help="seed 별 표 저장 경로 (선택)")
    args = p.parse_args()
    m = args.metric

    runs = load_runs()
    base = runs[(runs.golden == ORIGINAL_GOLDEN) & (runs.query_fix == "none")]
    if base.empty:
        raise SystemExit("A(원본, 보정 없음) 결과가 없습니다. 먼저 retrieval.py 를 원본으로 실행하세요.")
    a = base.iloc[-1][m]  # 가장 최근 A 점수

    # ---- seed 별 표: 세기 x seed 마다 B(none) 와 C(dict) 를 짝지음
    rows = []
    for golden, part in runs.groupby("golden"):
        match = TYPO_NAME.match(golden)
        if not match:  # 오타 골든셋이 아니면(원본, _fixed 등) 건너뜀
            continue
        level, seed = match.group(1), int(match.group(2) or DEFAULT_SEED)
        b = part[part.query_fix == "none"]
        c = part[part.query_fix == "dict"]
        if b.empty or c.empty:  # B 나 C 중 하나를 아직 안 돌렸으면 표에 못 넣음
            print(f"(건너뜀: {golden} 은 B 또는 C 결과가 없음)")
            continue
        vb, vc = b.iloc[-1][m], c.iloc[-1][m]
        rows.append({
            "level": level, "seed": seed, "A": a, "B": vb, "C": vc,
            "하락": a - vb, "회복": vc - vb,
            "하락_문항": round((a - vb) * SCORED_QUESTIONS), "회복_문항": round((vc - vb) * SCORED_QUESTIONS),
        })
    if not rows:
        raise SystemExit("짝이 맞는 B, C 결과가 없습니다.")

    order = {"light": 0, "medium": 1, "heavy": 2}  # 가나다순이 아니라 약한 오타 -> 센 오타 순서로 보기
    table = pd.DataFrame(rows)
    table = table.sort_values(["level", "seed"], key=lambda col: col.map(order) if col.name == "level" else col)
    print(f"=== seed 별 결과 ({m}) ===")
    print(table.round(4).to_string(index=False))

    # ---- 세기별 합산 판정
    print(f"\n=== 세기별 합산 판정 ({m}, seed 를 모두 더해서 계산) ===")
    summary = []
    for level, part in sorted(table.groupby("level"), key=lambda kv: order[kv[0]]):
        drop, rec = part["하락"].sum(), part["회복"].sum()
        mean_drop = drop / len(part)  # seed 하나당 평균 하락
        if mean_drop < MIN_DROP:
            ratio, verdict = None, "하락 작음(판정 불가)"
        else:
            ratio = rec / drop
            verdict = "통과" if ratio >= PASS_RATIO else "미달"
        summary.append({
            "level": level, "seeds": len(part),
            "평균 B": round(part["B"].mean(), 4), "평균 C": round(part["C"].mean(), 4),
            "하락 합(문항)": int(part["하락_문항"].sum()), "회복 합(문항)": int(part["회복_문항"].sum()),
            "합산 회복률": None if ratio is None else round(ratio, 2),
            "seed별 회복률 범위": _ratio_range(part),
            "판정": verdict,
        })
    print(f"A(원본) = {a:.4f}")
    print(pd.DataFrame(summary).to_string(index=False))

    if args.out:
        table.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nseed 별 표 저장: {args.out}")


def _ratio_range(part):
    """seed 마다 따로 낸 회복률의 최솟값~최댓값 (결과가 얼마나 출렁이는지 보기용)"""
    ratios = [r["회복"] / r["하락"] for _, r in part.iterrows() if r["하락"] >= MIN_DROP]
    if not ratios:
        return "-"
    return f"{min(ratios):.2f} ~ {max(ratios):.2f}"


if __name__ == "__main__":
    main()
