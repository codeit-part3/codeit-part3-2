"""골든셋 평가 스크립트

실행 예
    uv run python evaluate.py                                   # v2, 기본 설정, 채점과 시트 기록 포함
    uv run python evaluate.py --top-k 5 --name topk5 --memo "검색 조각 5개"
    uv run python evaluate.py --chunk-size 800 --chunk-overlap 100 --name chunk800
    uv run python evaluate.py --engine agent --no-judge
    uv run python evaluate.py --limit 2                          # 빠른 확인 (시트 기록 안 함)

하는 일
1. 골든셋 질문마다 답변 생성 (v1 / v2 / agent 중 선택)
2. 규칙 지표 계산 (LLM 없이): accuracy, keyword_recall, hit_rate, mrr, latency_sec, n_tool_calls
3. 문항별 결과를 outputs/ 에 한 문항씩 저장
4. LLM 채점 (judge.py, 기본 켜짐): faithfulness, correctness
5. 구글 시트에 한 줄 기록 (기본 켜짐)
"""

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain.messages import HumanMessage

import baseline
import judge

# 골든셋의 level 을 4개 유형으로 묶습니다. (팀 집계 방식 초안)
GROUPS = {
    "lookup": "단일문서 추출",
    "interpret": "단일문서 추출",
    "calc": "단일문서 추출",
    "fuzzy_filter": "단일문서 추출",
    "composite": "여러문서 종합",
    "followup": "후속질문",
    "no_answer": "모른다고 답변",
}
GROUP_ORDER = ["단일문서 추출", "여러문서 종합", "후속질문", "모른다고 답변"]

# 검색 지표(hit_rate, mrr) 계산에서 뺄 level (팀 결정 전 임시 기준, 여기만 바꾸면 됨)
RETRIEVAL_EXCLUDE_LEVELS = {"no_answer"}

# LLM 채점 결과 중 시트로 보낼 지표 (judge.py 의 summarize_judge 와 같은 이름)
JUDGE_METRICS = (
    "faithfulness",
    "correctness",
    "completeness",
    "hallucination",
    "refusal",
)

# 답 없음 문항: 모른다고 답했는지 확인할 표현 (띄어쓰기는 무시하고 비교)
DONT_KNOW = [
    "잘 모르겠",
    "알 수 없",
    "찾을 수 없",
    "제공되지 않",
    "공고문 확인",
    "확인할 수 없",
    "내용이 없",
    "정보가 없",
    "포함되어 있지 않",
    "공개되어 있지 않",
    "공개되지 않",
]
# 금액을 지어냈는지 확인할 패턴 (쉼표가 있는 큰 숫자, 또는 숫자+원/만원/억)
MONEY = r"\d{1,3}(?:,\d{3})+|\d+\s*(?:원|만원|천원|백만원|천만원|억)"


# =====================================================================
# 1. 평가할 엔진 : 엔진 변환 시 이곳에서 진행
# =====================================================================
def make_answer_fn(engine):
    """answer_fn(question, history) -> dict(answer, retrieved, prompt_tokens, n_tool_calls)"""
    if engine == "v1":
        model, vector_store = baseline.get_resources()

        def answer_fn(question, history):
            result, retrieved = baseline.answer_question(
                model, vector_store, question, history
            )
            return {
                "answer": result.text,
                "retrieved": retrieved,
                "prompt_tokens": result.response_metadata.get("prompt_eval_count"),
                "n_tool_calls": None,
            }

    elif engine == "v2":
        graph = baseline.get_graph()

        def answer_fn(question, history):
            state = graph.invoke({"question": question, "history": history or []})
            return {
                "answer": state["answer"],
                "retrieved": state["retrieved"],
                "prompt_tokens": state.get("prompt_tokens"),
                "n_tool_calls": None,
            }

    else:  # agent
        agent = baseline.make_agent()

        def answer_fn(question, history):
            baseline.TOOL_LOG.clear()
            baseline.TOOL_CALLS.clear()
            messages = [*baseline.history_to_messages(history), HumanMessage(question)]
            result = agent.invoke({"messages": messages})
            return {
                "answer": result["messages"][-1].text,
                "retrieved": list(baseline.TOOL_LOG),
                "prompt_tokens": None,
                "n_tool_calls": len(baseline.TOOL_CALLS),
            }

    return answer_fn


# =====================================================================
# 2. 답변 채점 (규칙 기반)
# =====================================================================
def squash(text):
    """쉼표와 띄어쓰기 제거"""
    return re.sub(r"[,\s]", "", str(text))


def contains(squashed_text, keyword):
    key = squash(keyword)
    if key.isdigit():
        # 숫자는 앞뒤에 다른 숫자가 붙지 않은 경우만 인정 (6 이 165 에 걸리지 않게)
        return re.search(rf"(?<!\d){key}(?!\d)", squashed_text) is not None
    return key in squashed_text


def score(item, answer):
    """자동 채점 -> (정답 여부, 빠진 키워드, keyword_recall, 모름 판정 표현)"""
    squashed = squash(answer)

    if item["level"] == "no_answer":
        matched = [p for p in DONT_KNOW if squash(p) in squashed]
        made_up_money = re.search(MONEY, answer) is not None
        return bool(matched) and not made_up_money, [], None, ", ".join(matched)

    wanted = list(item["keywords"]) + [str(v) for v in item["answer_values"]]
    missing = [w for w in wanted if not contains(squashed, w)]
    recall = (len(wanted) - len(missing)) / len(wanted) if wanted else None
    return len(missing) == 0, missing, recall, ""


# =====================================================================
# 3. 검색 지표 (정답 문서 기준)
# =====================================================================
def load_file_to_notice():
    """data_list.csv 로 '파일명 -> 사업명' 표를 생성"""
    df = pd.read_csv(baseline.DATA_PATH)
    return {
        baseline.normalize(str(row["파일명"])): str(row["사업명"])
        for _, row in df.iterrows()
    }


def evidence_notice_ids(item, file_to_notice):
    ids = set()
    for name in item["evidence_files"]:
        notice = file_to_notice.get(baseline.normalize(name))
        if notice is None:
            print(
                f"  주의: {item['id']} 정답 파일을 data_list.csv 에서 못 찾음: {name}"
            )
        else:
            ids.add(notice)
    return ids


def retrieval_scores(item, retrieved, wanted_ids):
    """(doc_hit, rr). 계산 대상이 아니면 (None, None)

    doc_hit : 정답 문서가 '모두' 검색 결과 안에 있으면 1, 아니면 0
    rr      : 정답 문서가 모두 나온 순위의 역수 (1위=1, 2위=0.5 ...), 하나라도 없으면 0
              순위는 조각이 아니라 문서 기준 (같은 문서의 조각은 처음 나온 순위만 씀)
              ※ 팀 확정 전 기준안
    """
    if item["level"] in RETRIEVAL_EXCLUDE_LEVELS or not wanted_ids:
        return None, None
    ranked = []
    for doc, _ in retrieved:
        name = doc.metadata["사업명"]
        if name not in ranked:
            ranked.append(name)
    if not wanted_ids <= set(ranked):
        return 0, 0.0
    last_rank = max(ranked.index(n) + 1 for n in wanted_ids)
    return 1, round(1 / last_rank, 4)


def context_hit(item, retrieved):
    """정답 키워드가 '검색된 조각 안에' 모두 있으면 1, 아니면 0. 대상이 아니면 None

    hit_rate 는 문서 단위, context_hit 는 조각 단위라서
    '문서는 찾았는데 정답이 든 조각이 아님' 을 구분할 수 있다.
    """
    if item["level"] in RETRIEVAL_EXCLUDE_LEVELS or not item["keywords"]:
        return None
    context = squash(
        "".join(
            f"{doc.metadata['발주 기관']}{doc.metadata['사업명']}{doc.page_content}"
            for doc, _ in retrieved
        )
    )
    return int(all(contains(context, k) for k in item["keywords"]))


# =====================================================================
# 4. 틀린 원인 추정
# =====================================================================
def guess_cause(item, retrieved, wanted_ids):
    if not item["evidence_files"]:
        return "모델 문제(추정): 문서에 없는 내용인데 모른다고 답하지 않음"
    if not wanted_ids:
        return "확인 불가: 정답 문서를 data_list.csv 에서 못 찾음"

    found_ids = {doc.metadata["사업명"] for doc, _ in retrieved}
    if not wanted_ids <= found_ids:
        return (
            f"검색 문제(추정): 정답 문서 {len(wanted_ids)}개 중 "
            f"{len(wanted_ids & found_ids)}개만 검색됨"
        )

    context = squash(
        "".join(
            f"{doc.metadata['발주 기관']}{doc.metadata['사업명']}{doc.page_content}"
            for doc, _ in retrieved
        )
    )
    if not all(contains(context, k) for k in item["keywords"]):
        return "검색 문제(추정): 정답 문서는 찾았지만 정답이 든 조각이 아님"
    return "모델 문제(추정): 정답이 든 조각을 받고도 틀림"


# =====================================================================
# 5. 실행과 집계
# =====================================================================
def run_eval(golden, answer_fn, file_to_notice, version, result_path):
    rows = []
    settings = baseline.current_settings()
    for item in golden:
        start = time.time()
        try:
            out = answer_fn(item["question"], item.get("history"))
            error = None
        except Exception as e:  # 한 문항이 실패해도 나머지는 계속 실행
            out = {
                "answer": "",
                "retrieved": [],
                "prompt_tokens": None,
                "n_tool_calls": None,
            }
            error = str(e)
        seconds = round(time.time() - start, 2)
        answer, retrieved = out["answer"], out["retrieved"]

        wanted_ids = evidence_notice_ids(item, file_to_notice)
        doc_hit, rr = retrieval_scores(item, retrieved, wanted_ids)
        ctx_hit = context_hit(item, retrieved)
        if error:
            correct, missing, recall, unknown, cause = (
                False,
                [],
                None,
                "",
                f"실행 오류: {error}",
            )
        else:
            correct, missing, recall, unknown = score(item, answer)
            cause = "" if correct else guess_cause(item, retrieved, wanted_ids)

        print(
            f"[{item['id']}] {'O' if correct else 'X'} ({item['level']}, {seconds}초)"
        )

        rows.append(
            {
                "id": item["id"],
                "level": item["level"],
                "group": GROUPS.get(item["level"], item["level"]),
                "category": item["category"],
                "question": item["question"],
                "answer": answer,
                "gold_answer": item["gold_answer"],
                "correct": correct,
                "keyword_recall": recall,
                "doc_hit": doc_hit,
                "rr": rr,
                "context_hit": ctx_hit,
                "모름판정표현": unknown,
                "수동확인": "수동확인 필요" if item["level"] == "no_answer" else "",
                "빠진_키워드": ", ".join(missing),
                "원인(자동추정)": cause,
                "원인(직접기입)": "",  # 변환 / 검색 / 모델 중 직접 적기
                "seconds": seconds,
                "error": error or "",
                "prompt_tokens": out["prompt_tokens"],
                "n_tool_calls": out["n_tool_calls"],
                "sources": " || ".join(
                    f"{doc.metadata['공고 번호']} {doc.metadata['사업명']} (거리 {score_:.3f})"
                    for doc, score_ in retrieved
                ),
                "contexts": baseline.format_context(
                    retrieved
                ),  # LLM 채점용 검색 조각 본문
                "version": version,
                **settings,
            }
        )
        # 문항마다 저장 (중간에 멈춰도 여기까지 결과는 남음)
        pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def mean_or_none(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(values.mean()), 4) if len(values) else None


def overall_metrics(df):
    """시트에 보낼 전체 지표 (0~1 비율, 응답시간은 초)"""
    return {
        "accuracy": mean_or_none(df["correct"].astype(float)),
        "keyword_recall": mean_or_none(df["keyword_recall"]),
        "hit_rate": mean_or_none(df["doc_hit"]),
        "mrr": mean_or_none(df["rr"]),
        "context_hit": mean_or_none(df["context_hit"]),
        "latency_sec": mean_or_none(df["seconds"]),
        "n_tool_calls": mean_or_none(df["n_tool_calls"]),
        "error_count": int((df["error"].fillna("") != "").sum()),
    }


# 점수표에 보여 줄 지표: (이름, 한글 설명, 표시 방식, 좋은 방향)
# 표시 방식 pct: 0~1 을 % 로, sec: 초, count: 개수, avg: 평균 횟수
SCOREBOARD = [
    ("답변", None, None, None),
    ("accuracy", "정답률 (규칙 채점)", "pct", "높을수록"),
    ("keyword_recall", "정답 키워드 포함률", "pct", "높을수록"),
    ("correctness", "정확성 (LLM 채점)", "pct", "높을수록"),
    ("completeness", "완결성 (LLM 채점)", "pct", "높을수록"),
    ("faithfulness", "근거성 (LLM 채점)", "pct", "높을수록"),
    ("hallucination", "지어낸 답 비율 (LLM 채점)", "pct", "낮을수록"),
    ("refusal", "모름 문항 거절 성공률 (LLM 채점)", "pct", "높을수록"),
    ("검색", None, None, None),
    ("hit_rate", "정답 문서 검색 성공률", "pct", "높을수록"),
    ("mrr", "정답 문서 순위 점수", "pct", "높을수록"),
    ("context_hit", "정답 조각 검색 성공률", "pct", "높을수록"),
    ("운영", None, None, None),
    ("latency_sec", "평균 응답시간", "sec", "낮을수록"),
    ("n_tool_calls", "평균 도구 호출 수 (agent)", "avg", "-"),
    ("error_count", "실행 오류 문항 수", "count", "낮을수록"),
]


def pad(text, width):
    """한글은 화면에서 2칸을 차지하므로 그만큼 계산해 오른쪽을 공백으로 채움"""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def format_value(value, kind):
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value * 100:5.1f} %"
    if kind == "sec":
        return f"{value:5.1f} 초"
    if kind == "count":
        return f"{int(value):5d} 개"
    return f"{value:5.2f} 회"


def print_scoreboard(values):
    """지표마다 단위가 달라 보기 어려우므로, 비율은 모두 % 로 바꿔 한 표로 보여 줌"""
    print("\n" + "=" * 60)
    print(" 점수표  (비율 지표는 모두 0~100 %)")
    print("=" * 60)
    for key, label, kind, better in SCOREBOARD:
        if label is None:
            print(f"[{key}]")
            continue
        shown = format_value(values.get(key), kind)
        print(f"  {pad(label, 34)}{shown:>9s}   {pad(better, 8)} {key}")
    print("=" * 60)


def summarize(df):
    """유형별 문항 수, 정답 수, 정답률, 평균 응답시간 등을 표로 생성"""
    table = df.groupby("group").agg(
        문항수=("id", "count"),
        정답수=("correct", "sum"),
        평균응답시간=("seconds", "mean"),
    )
    table = table.reindex([g for g in GROUP_ORDER if g in table.index])
    table.loc["전체"] = [len(df), df["correct"].sum(), df["seconds"].mean()]
    table[["문항수", "정답수"]] = table[["문항수", "정답수"]].astype(int)
    table["정답률(%)"] = (table["정답수"] / table["문항수"] * 100).round(1)
    table["평균응답시간"] = table["평균응답시간"].round(1)
    return table


def log_to_sheet(args, metrics, judge_summary, golden_size):
    from common.sheet_log import log_experiment

    settings = baseline.current_settings()
    params = {"engine": args.engine, **settings, "questions": golden_size}
    memo = args.memo
    judge_scores = {}
    pending = judge_summary.get("judge_pending", 0)
    if judge_summary.get("faithfulness") is not None:
        params["judge"] = f"{args.judge_provider}:{args.judge_model}"
    if pending:
        # 일부만 채점된 평균은 오해를 주므로 시트에 올리지 않음
        memo = f"{memo} (LLM 채점 {pending}문항 대기, judge.py 로 이어서 채점)".strip()
    else:
        judge_scores = {k: judge_summary.get(k) for k in JUDGE_METRICS}
    all_metrics = {
        k: v for k, v in {**metrics, **judge_scores}.items() if v is not None
    }
    log_experiment(name=args.name, memo=memo, metrics=all_metrics, params=params)


def parse_args():
    parser = argparse.ArgumentParser(description="골든셋으로 베이스라인을 평가합니다.")
    parser.add_argument(
        "--golden", default=str(baseline.BASE_DIR / "data" / "goldenset.json")
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="앞에서 N문항만 실행 (시트 기록 안 함)"
    )
    parser.add_argument("--engine", default="v2", choices=["v1", "v2", "agent"])
    parser.add_argument(
        "--version", default=None, help="결과 파일 이름 (기본: 엔진 이름)"
    )

    exp = parser.add_argument_group("실험 설정 (안 쓰면 baseline.py 의 값)")
    exp.add_argument(
        "--top-k", type=int, default=None, help="검색해서 모델에 넘길 조각 개수"
    )
    exp.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="조각 하나의 최대 글자 수 (바꾸면 새 DB 생성)",
    )
    exp.add_argument(
        "--chunk-overlap", type=int, default=None, help="이웃 조각끼리 겹치는 글자 수"
    )
    exp.add_argument("--no-warmup", action="store_true", help="모델 미리 불러오기 생략")

    rec = parser.add_argument_group("기록")
    rec.add_argument(
        "--name", default=None, help="시트에 남길 실험 이름 (기본: 버전_시각)"
    )
    rec.add_argument("--memo", default="", help="시트에 남길 한 줄 설명")
    rec.add_argument(
        "--no-sheet", action="store_true", help="구글 시트에 기록하지 않음"
    )

    jd = parser.add_argument_group("LLM 채점 (기본 켜짐)")
    jd.add_argument("--no-judge", action="store_true", help="LLM 채점을 하지 않음")
    jd.add_argument(
        "--judge-provider",
        default=judge.DEFAULT_PROVIDER,
        choices=["groq", "ollama", "openai"],
    )
    jd.add_argument("--judge-model", default=judge.DEFAULT_MODEL)
    jd.add_argument(
        "--wait-minutes", type=float, default=10, help="한도 초과 시 기다릴 분"
    )
    jd.add_argument("--max-waits", type=int, default=3, help="오래 기다리기 최대 횟수")
    return parser.parse_args()


def main():
    args = parse_args()
    args.version = args.version or args.engine
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    args.name = args.name or f"{args.version}_{stamp}"

    baseline.configure(args.chunk_size, args.chunk_overlap, args.top_k)
    print(f"설정: {baseline.current_settings()}")

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8-sig"))
    if args.limit:
        golden = golden[: args.limit]
    print(f"골든셋 {len(golden)}문항을 평가합니다.")

    baseline.OUTPUT_DIR.mkdir(exist_ok=True)
    result_path = baseline.OUTPUT_DIR / f"eval_{args.version}_{stamp}.csv"
    summary_path = baseline.OUTPUT_DIR / f"eval_{args.version}_{stamp}_summary.csv"

    answer_fn = make_answer_fn(args.engine)
    if not args.no_warmup:
        print("모델 미리 불러오는 중 (응답시간에서 제외)")
        baseline.warmup()
    df = run_eval(golden, answer_fn, load_file_to_notice(), args.version, result_path)

    table = summarize(df)
    metrics = overall_metrics(df)
    table.to_csv(summary_path, encoding="utf-8-sig")
    print("\n=== 유형별 결과 ===")
    print(table.to_string())
    manual = df.loc[df["수동확인"] != "", "id"].tolist()
    if manual:
        print(f"\n수동확인 필요 문항: {', '.join(manual)}")
    print(f"\n결과 저장: {result_path}\n요약 저장: {summary_path}")

    judge_summary = {}
    if not args.no_judge:
        judge_summary = judge.judge_file(
            result_path,
            args.judge_provider,
            args.judge_model,
            args.wait_minutes,
            args.max_waits,
        )

    print_scoreboard({**metrics, **judge_summary})
    if judge_summary.get("judge_pending"):
        print(
            f"  ※ LLM 채점 {judge_summary['judge_pending']}문항 대기 중이라 LLM 지표는 일부 문항 기준"
        )

    if args.no_sheet:
        print("\n[시트 기록 안 함] --no-sheet")
    elif args.limit:
        print("\n[시트 기록 안 함] --limit 로 일부 문항만 실행함")
    else:
        log_to_sheet(args, metrics, judge_summary, len(golden))


if __name__ == "__main__":
    main()
