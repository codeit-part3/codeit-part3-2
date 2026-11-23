"""final 스위치 실험을 순서대로 연속 실행 (#43, 돌려 두고 자리를 비워도 되게)

eval_final.py 를 실험마다 한 번씩 실행한다. 한 실험이 실패해도 다음 실험으로 넘어간다.
끝난 실험은 outputs/final_batch_done.txt 에 기록해 두고, 다시 실행하면 건너뛴다(이어서 하기).

실행 (프로젝트 루트에서)
    uv run python scripts/final/run_batch.py --dry-run    # 실행할 목록만 보기 (실행 안 함)
    uv run python scripts/final/run_batch.py              # 전체 연속 실행 (중간에 멈췄으면 이어서)
    uv run python scripts/final/run_batch.py --only 3 4   # 목록 번호 3, 4번만 실행

실행 목록 (기준선, 스위치 하나씩, 서류 거절 2개 조합 = 7개 x 2회 = 14회)
    1회차 7개를 먼저 돌리고, 반복(2회차) 7개를 이어서 돌린다.
화면 출력은 outputs/final_batch_<시각>.log 에도 그대로 저장된다.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_FINAL = ROOT / "scripts" / "final" / "eval_final.py"
DONE_FILE = ROOT / "outputs" / "final_batch_done.txt"

CONFIGS = [
    [],
    ["CORR_REFUSE"],
    ["CORR_DOC_FILTER"],
    ["CORR_WON"],
    ["DOCS_REFUSE"],
    ["REAL_WARMUP"],
    [
        "CORR_REFUSE",
        "DOCS_REFUSE",
    ],  # 둘 다 서류 질문 거절을 다뤄서 함께 켠 효과를 따로 확인
]
ORDER = ["CORR_REFUSE", "CORR_DOC_FILTER", "CORR_WON", "DOCS_REFUSE", "REAL_WARMUP"]


def plan():
    """[(시트 이름, eval_final.py 인자)] 1회차 전부 -> 반복 전부"""
    runs = []
    for rep in (False, True):
        for on in CONFIGS:
            name = "final_luna_" + ("+".join(s for s in ORDER if s in on) or "base")
            name += "_rep" if rep else ""
            args = (["--on", *on] if on else []) + (["--rep"] if rep else [])
            runs.append((name, args))
    return runs


def load_done():
    if not DONE_FILE.is_file():
        return set()
    return set(DONE_FILE.read_text(encoding="utf-8").split())


def run_one(args, log):
    """eval_final.py 실행. 화면과 로그 파일에 같이 출력하고 종료 코드를 돌려줌"""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, str(EVAL_FINAL), *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return proc.wait()


def main():
    ap = argparse.ArgumentParser(description="final 스위치 실험 연속 실행")
    ap.add_argument("--dry-run", action="store_true", help="목록만 보기")
    ap.add_argument("--only", nargs="+", type=int, help="실행할 목록 번호")
    ap.add_argument(
        "--redo", action="store_true", help="끝난 기록을 무시하고 다시 실행"
    )
    opt = ap.parse_args()

    runs = plan()
    done = set() if opt.redo else load_done()
    todo = []
    print("=== 실행 목록 ===")
    for i, (name, args) in enumerate(runs, 1):
        pick = (not opt.only or i in opt.only) and name not in done
        mark = "실행" if pick else ("완료(건너뜀)" if name in done else "-")
        print(f"{i:2d}. {name:45s} {mark}")
        if pick:
            todo.append((i, name, args))
    print(f"실행할 실험 {len(todo)}개 (1개 약 10분)")
    if opt.dry_run or not todo:
        return

    DONE_FILE.parent.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "outputs" / f"final_batch_{stamp}.log"
    results = []
    with log_path.open("w", encoding="utf-8") as log:
        for k, (i, name, args) in enumerate(todo, 1):
            head = f"\n##### [{k}/{len(todo)}] {i}. {name} 시작 {datetime.now():%H:%M:%S}\n"
            print(head, flush=True)
            log.write(head)
            code = run_one(args, log)
            ok = code == 0
            if ok:
                with DONE_FILE.open("a", encoding="utf-8") as f:
                    f.write(name + "\n")
            results.append((name, ok))
            tail = f"##### {name} {'성공' if ok else f'실패(코드 {code})'} {datetime.now():%H:%M:%S}\n"
            print(tail, flush=True)
            log.write(tail)

        summary = "\n=== 연속 실행 결과 ===\n" + "".join(
            f"{'O' if ok else 'X'} {name}\n" for name, ok in results
        )
        print(summary)
        log.write(summary)
    print(f"로그: {log_path}")
    if not all(ok for _, ok in results):
        print("실패한 실험은 같은 명령을 다시 실행하면 그것만 이어서 실행합니다.")


if __name__ == "__main__":
    main()
