"""final 엔진 평가를 짧은 명령으로 실행 (#43)

scripts/rerank_rag/run_eval.py 를 그대로 불러 쓰고, 팀 표준 조건을 기본값으로 채운다.
시트 이름(name)과 메모(memo)는 답변 모델과 켠 스위치를 보고 자동으로 만든다. (공용 파일은 수정하지 않음)

실행 (프로젝트 루트에서)
    uv run python scripts/final/eval_final.py                      # 기준선 (스위치 전부 꺼짐)
    uv run python scripts/final/eval_final.py --on CORR_WON        # 스위치 하나 켜기
    uv run python scripts/final/eval_final.py --on CORR_REFUSE CORR_WON --rep   # 여러 개 + 반복 실행
    uv run python scripts/final/eval_final.py --local --rule dh    # 로컬 모델(gemma4:e2b) + 동현님 빠른 모델 규칙
    uv run python scripts/final/eval_final.py --limit 5 --no-judge # 빠른 확인 (시트 기록 안 함)

기본 조건 (팀 표준 rerank_rag 조건)
    엔진 final, 기관명 필터 꺼짐, 리랭커 후보 30개, 말뭉치 clean_mixed_ocr_v2,
    골든셋은 engines/final/config.py 의 GOLDEN_FILE, 답변 gpt-6-luna (노력 low 고정)

시트 이름 규칙: final_<답변모델>_<켠 스위치>[_rep]
    예) final_luna_base, final_luna_CORR_WON, final_luna_CORR_REFUSE+CORR_WON_rep,
        final_gemma4e2b_ADAPTIVE_dh

이 파일이 받는 옵션
    --on      켤 스위치 (CORR_REFUSE, CORR_DOC_FILTER, CORR_WON, DOCS_REFUSE, REAL_WARMUP)
    --local   로컬 모델(Ollama gemma4:e2b)로 답변
    --rule    로컬 전용 빠른 모델 규칙 off / dh / corr (기본 off)
    --rep     반복 실행 (이름 끝에 _rep)
나머지 옵션(--golden, --limit, --no-judge, --no-sheet, --name, --memo ...)은 run_eval.py 로 그대로 넘긴다.
--name, --memo 를 직접 주면 자동 값 대신 그 값을 쓴다.
"""

import argparse
import os
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 프로젝트 루트
RUN_EVAL = ROOT / "scripts" / "rerank_rag" / "run_eval.py"

CLOUD_MODEL, CLOUD_SHORT = (
    "gpt-6-luna",
    "luna",
)  # 클라우드 답변 모델 (이번 실험은 luna 고정)
CLOUD_EFFORT = "low"  # 노력 정도 고정 (이름에는 넣지 않음)
LOCAL_MODEL, LOCAL_SHORT = "gemma4:e2b", "gemma4e2b"  # --local 일 때 답변 모델
SWITCHES = ["CORR_REFUSE", "CORR_DOC_FILTER", "CORR_WON", "DOCS_REFUSE", "REAL_WARMUP"]
BASE_ARGS = [
    "--pkg",
    "--pkg-engine",
    "final",
    "--variant",
    "rerank",
    "--pool",
    "30",
    "--corpus",
    "clean_mixed_ocr_v2",
]


def golden_tag(rest):
    """골든셋 파일 이름에서 버전 표시를 뽑음 (rahui_golden_set_v2_4 -> v24)"""
    if "--golden" in rest:
        path = rest[rest.index("--golden") + 1]
    else:
        text = (ROOT / "engines" / "final" / "config.py").read_text(encoding="utf-8")
        m = re.search(r'GOLDEN_FILE\s*=\s*"([^"]+)"', text)
        path = m.group(1) if m else ""
    m = re.search(r"v(\d+)_(\d+)", Path(path).stem)
    return f"v{m.group(1)}{m.group(2)}" if m else Path(path).stem


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--on", nargs="+", default=[], choices=SWITCHES)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--rule", default="off", choices=["off", "dh", "corr"])
    ap.add_argument("--rep", action="store_true")
    mine, rest = ap.parse_known_args()
    if mine.rule != "off" and not mine.local:
        raise SystemExit("--rule 은 로컬 모델 전용입니다 (--local 과 함께 쓰세요).")

    # 환경변수: 켠 스위치는 1, 나머지는 0 으로 확실히 덮어씀 (이전에 설정한 값이 남지 않게)
    os.environ["FINAL_PROVIDER"] = "ollama" if mine.local else "openai"
    os.environ["FINAL_LOCAL_MODEL"] = LOCAL_MODEL
    os.environ["OPENAI_RAG_MODEL"] = CLOUD_MODEL
    os.environ["OPENAI_RAG_EFFORT"] = CLOUD_EFFORT
    os.environ["FINAL_ADAPTIVE_RULE"] = mine.rule
    for sw in SWITCHES:
        os.environ[f"FINAL_{sw}"] = "1" if sw in mine.on else "0"

    parts = [sw for sw in SWITCHES if sw in mine.on]
    if mine.rule != "off":
        parts.append(f"ADAPTIVE_{mine.rule}")
    on = "+".join(parts) or "base"
    short = LOCAL_SHORT if mine.local else CLOUD_SHORT
    answer = f"{LOCAL_MODEL}(로컬)" if mine.local else f"{CLOUD_MODEL}({CLOUD_EFFORT})"
    tag = golden_tag(rest)
    name = f"final_{short}_{on}" + ("_rep" if mine.rep else "")
    memo = (
        f"final 스위치: {on}{' 반복' if mine.rep else ''}, 골든셋 {tag}"
        f" / 답변: {answer} / 채점: gpt-5-nano"
    )
    if "--name" not in rest:
        rest += ["--name", name]
    if "--memo" not in rest:
        rest += ["--memo", memo]

    print(f"[eval_final] 스위치: {on} / 답변: {answer} / 골든셋: {tag}")
    print(f"[eval_final] 시트 이름: {rest[rest.index('--name') + 1]}")
    sys.argv = [str(RUN_EVAL), *BASE_ARGS, *rest]
    runpy.run_path(str(RUN_EVAL), run_name="__main__")


if __name__ == "__main__":
    main()
