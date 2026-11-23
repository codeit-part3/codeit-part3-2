"""오타 골든셋 만들기 (오타 강건성 실험 1단계)

원본 골든셋의 question 에만 오타를 넣고, 정답·근거 등 나머지 필드는 그대로 둔다.
그래서 retrieval.py 에 --golden 으로 넣으면 원본과 똑같은 방식으로 채점된다.

실행 (프로젝트 루트):
    uv run python scripts/typo/typo_golden.py                 # light, medium, heavy 세 개를 모두 만듦
    uv run python scripts/typo/typo_golden.py --levels light  # 하나만
    uv run python scripts/typo/typo_golden.py --seed 7        # 다른 오타 세트 (파일 이름 끝에 _s7 이 붙음)

출력 (data/golden/typo/ 아래)
    rahui_golden_set_v2_2_typo_light.json    질문당 오타 1개
    rahui_golden_set_v2_2_typo_medium.json   질문당 오타 2개
    rahui_golden_set_v2_2_typo_heavy.json    기관명·사업명 글자의 20% (최소 3개)
    typo_log.csv                             문항별 원본 질문 / 오타 질문 / 넣은 오타 종류

흐름
    원본 json 읽기 -> 문항마다 [오타 넣을 위치 찾기 -> 오타 n개 넣기] -> 세기별 json 저장 -> 기록 csv 저장
"""

import argparse  # 명령어 옵션(--levels, --seed)을 읽기 위해
import json  # 골든셋 json 을 읽고 쓰기 위해
import re  # 질문에서 따옴표 안 사업명을 찾기 위해 (정규식)
from pathlib import Path  # 운영체제(맥/윈도우)에 상관없이 경로를 다루기 위해

import pandas as pd  # data_list.csv 를 읽고 기록 csv 를 쓰기 위해

from hangul_typo import add_typos, is_syllable, make_rng  # 오타를 실제로 만드는 부품

# =====================================================================
# 설정
# =====================================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # 프로젝트 루트 (scripts/typo/ 에서 두 단계 위)
DEFAULT_GOLDEN = "data/golden/rahui_golden_set_v2_2.json"
DEFAULT_SEED = 42  # 기본 seed. 이 seed 로 만든 파일은 이름에 seed 를 붙이지 않는다 (처음 만든 파일과 이름 유지)  # 원본 골든셋 (config.py 의 GOLDEN_FILE 과 같음)
DATA_LIST = BASE_DIR / "data" / "data_list.csv"  # 문서 100건 목록 (발주 기관 이름을 가져옴)
OUT_DIR = BASE_DIR / "data" / "golden" / "typo"  # 결과를 저장할 폴더 (원본 골든셋과 섞이지 않게 따로)

# 오타를 넣지 않는 문항 유형과 그 이유
SKIP_LEVELS = {
    "no_answer",  # 정답 문서가 없어서 검색 점수 계산에서 원래 빠지는 문항
    "followup",  # "그 사업의 사업비는?"처럼 질문에 기관명·사업명이 없는 문항
    "fuzzy_filter",  # 이미 기관명 오타·약칭이 들어 있는 문항 (오타를 겹쳐 넣으면 원인이 섞임)
}

# 세기별 오타 개수 규칙. n_syll = 오타를 넣을 수 있는 글자 수
TYPO_COUNT = {
    "light": lambda n_syll: 1,  # 실제 사용자가 흔히 하는 실수 1개
    "medium": lambda n_syll: 2,
    "heavy": lambda n_syll: max(3, round(n_syll * 0.2)),  # 일부러 세게: 20%, 적어도 3개
}

MIN_QUOTED_LEN = 6  # 따옴표 안이 6글자 이상일 때만 사업명으로 봄 ('5. 종합의견' 같은 짧은 인용 제외)


# =====================================================================
# 오타를 넣을 위치 찾기
# =====================================================================
def load_org_names():
    """data_list.csv 의 발주 기관 이름 목록. 긴 이름부터 정렬해서 돌려준다.

    긴 것부터 찾는 이유: '한국철도공사 (용역)'과 '한국철도공사'가 둘 다 있을 때 긴 쪽을 먼저 잡아야
    기관명 전체에 오타가 들어갈 수 있다.
    """
    orgs = set(pd.read_csv(DATA_LIST)["발주 기관"].dropna().astype(str))  # 빈칸 제거 후 중복 없이
    # '한국철도공사 (용역)'처럼 끝에 괄호가 붙은 이름은 괄호를 뗀 이름도 추가 (질문에는 괄호 없이 나오기도 함)
    orgs |= {re.sub(r"\s*\(.*?\)\s*$", "", name) for name in orgs}
    # key=(-길이, 이름): 길이가 긴 것부터, 길이가 같으면 가나다순 -> 실행할 때마다 순서가 같다
    return sorted((name for name in orgs if len(name) >= 2), key=lambda name: (-len(name), name))


def typo_positions(question, orgs):
    """오타를 넣을 글자 위치 목록: 기관명 + 따옴표 안 사업명. 둘 다 없으면 질문 앞쪽 절반."""
    positions = set()

    # 1) 기관명: 질문 안에서 처음 나오는 위치를 찾아 그 글자들을 후보로
    for org in orgs:
        start = question.find(org)  # 없으면 -1
        if start < 0:
            continue
        span = set(range(start, start + len(org)))  # 기관명이 차지하는 글자 위치들
        if not (positions & span):  # 이미 더 긴 기관명으로 잡은 자리와 겹치면 건너뜀
            positions |= span

    # 2) 사업명: 작은따옴표 '...' 안의 글자들
    for m in re.finditer(rf"'([^']{{{MIN_QUOTED_LEN},}})'", question):
        positions |= set(range(m.start(1), m.end(1)))  # group(1) = 따옴표 안쪽만

    # 3) 둘 다 없으면 질문 앞쪽 절반 (보통 기관명·사업명은 질문 앞에 온다)
    if not positions:
        positions = set(range(len(question) // 2))

    # 한글 글자 위치만 남김 -> 숫자(2024), 영문(ERP), 요구사항 ID(PER-004)는 자동으로 오타 대상에서 빠진다
    return sorted(p for p in positions if is_syllable(question[p]))


# =====================================================================
# 골든셋 한 벌 만들기
# =====================================================================
def make_typo_set(items, level, orgs, seed):
    """원본 문항 목록 -> (오타 문항 목록, 기록 행 목록)"""
    rng = make_rng(f"{seed}-{level}")  # 세기마다 다른 seed -> light 와 heavy 의 오타가 서로 독립
    out, log = [], []
    for item in items:
        question = item.get("question") or ""  # 질문이 비어 있는 문항도 오류 없이 지나가게
        if item.get("level") in SKIP_LEVELS or not question:
            new_question, ops = question, []  # 오타 없이 그대로
        else:
            positions = typo_positions(question, orgs)
            n_typos = TYPO_COUNT[level](len(positions))
            new_question, ops = add_typos(question, positions, n_typos, rng)

        # 원본 문항을 복사한 뒤 질문만 바꾸고, 나중에 비교할 수 있게 원문과 오타 종류도 함께 저장
        new_item = dict(item)  # dict(item) = 얕은 복사. 원본 items 는 바뀌지 않는다
        new_item["question"] = new_question
        new_item["question_original"] = question
        new_item["typo_level"] = level
        new_item["typo_ops"] = ops
        out.append(new_item)

        log.append({
            "typo_level": level,
            "id": item["id"],
            "level": item.get("level"),
            "n_typos": len(ops),
            "ops": ",".join(ops),
            "question_original": question,
            "question_typo": new_question,
        })
    return out, log


def main():
    # ---- 명령어 옵션 읽기
    parser = argparse.ArgumentParser(description="질문에 오타를 넣은 골든셋을 만듭니다.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help="원본 골든셋 경로 (프로젝트 루트 기준)")
    parser.add_argument("--levels", nargs="+", default=list(TYPO_COUNT), choices=list(TYPO_COUNT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="같은 seed 면 같은 오타가 나옴")
    args = parser.parse_args()

    # ---- 준비
    golden_path = BASE_DIR / args.golden
    items = json.loads(golden_path.read_text(encoding="utf-8"))
    orgs = load_org_names()
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 만들고, 있으면 그대로

    # ---- 세기별로 만들어 저장
    all_logs = []
    for level in args.levels:
        typo_items, log = make_typo_set(items, level, orgs, args.seed)
        all_logs += log
        # stem = 확장자를 뺀 파일 이름. 기본 seed 가 아니면 _s7 처럼 seed 를 붙여 덮어쓰지 않게 한다
        seed_tag = "" if args.seed == DEFAULT_SEED else f"_s{args.seed}"
        out_path = OUT_DIR / f"{golden_path.stem}_typo_{level}{seed_tag}.json"
        # ensure_ascii=False: 한글을 \uXXXX 로 바꾸지 않고 그대로 저장 (사람이 열어 읽을 수 있게)
        out_path.write_text(json.dumps(typo_items, ensure_ascii=False, indent=1), encoding="utf-8")
        changed = sum(1 for it in typo_items if it["typo_ops"])
        print(f"[{level}] 오타 넣은 문항 {changed}/{len(typo_items)} -> {out_path.relative_to(BASE_DIR)}")

    # ---- 기록 csv (utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않게)
    log_path = OUT_DIR / ("typo_log.csv" if args.seed == DEFAULT_SEED else f"typo_log_s{args.seed}.csv")
    pd.DataFrame(all_logs).to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"문항별 기록: {log_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 main() 을 부름 (다른 파일이 import 할 땐 안 부름)
    main()
