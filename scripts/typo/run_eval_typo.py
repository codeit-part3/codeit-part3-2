"""리랭커 엔진 답변 평가를 오타 교정 조건만 바꿔서 돌리는 실험용 실행 파일

엔진 기본값은 config.py 의 TYPO_FIX = "query_fix" 라서 팀 run_eval.py 를 그냥 돌려도 교정이 켜진다.
이 파일은 A/B/C 비교처럼 config.py 를 고치지 않고 조건을 바꿔 돌릴 때만 쓴다 (실행 중에만 바뀜).
  off       : TYPO_FIX = False (교정 끔, B 조건)
  team      : TYPO_FIX = True (기존 팀 교정)
  query_fix : TYPO_FIX = "query_fix" (기관명·사업명 사전 교정, 엔진 코드 그대로)

실행 (프로젝트 루트):
    uv run python scripts/typo/run_eval_typo.py --typo-fix query_fix --pkg --variant rerank --pool 30 --chunker struct --rewrite --cutoff 0.5 --golden data/golden/typo/rahui_golden_set_v2_2_typo_medium.json --limit 5 --no-judge --no-sheet

--typo-fix : off(끔) / team(팀 교정 fix_typos) / query_fix(기관명·사업명 사전 교정)
나머지 옵션은 run_eval.py, evaluate.py 와 같다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 프로젝트 루트
sys.path.insert(0, str(ROOT))  # engines/, evaluate.py
sys.path.insert(0, str(ROOT / "scripts" / "rerank_rag"))  # run_eval.py

MODES = {"off": False, "team": True, "query_fix": "query_fix"}


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--typo-fix", required=True, choices=list(MODES))
    mine, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # 나머지 옵션은 run_eval.py 가 읽게 넘김

    from engines.rerank_rag import baseline as eng

    eng.TYPO_FIX = MODES[mine.typo_fix]  # 팀 fix_typos 스위치 + 결과·시트 기록용
    if mine.typo_fix == "query_fix" and not hasattr(eng, "get_query_fixer"):
        # 엔진에 교정 코드가 아직 없을 때만 실행 중에 바꿔 끼움 (있으면 엔진 코드를 그대로 씀)
        sys.path.insert(0, str(ROOT / "scripts" / "typo"))  # scripts/typo/query_fix.py
        from query_fix import QueryFixer

        eng.fix_typos = QueryFixer(data_list=eng.DATA_PATH, corpus_dir=eng.TXT_DIR).fix
    where = "엔진 baseline.py 코드" if hasattr(eng, "get_query_fixer") else f"실행 중 끼움 ({eng.fix_typos.__qualname__})"
    print(f"[run_eval_typo] TYPO_FIX = {eng.TYPO_FIX!r} | 교정 = {where}")

    import run_eval  # noqa: E402

    run_eval.main()


if __name__ == "__main__":
    main()
