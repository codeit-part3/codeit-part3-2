"""골든셋 평가 스크립트

변경사항
- 골든셋 경로 : data/goldenset.json

"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain.messages import HumanMessage

import baseline

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
    """answer_fn(question, history) -> (답변 글자, [(조각, 거리), ...])"""
    if engine == "v1":
        model, vector_store = baseline.get_resources()

        def answer_fn(question, history):
            result, retrieved = baseline.answer_question(
                model, vector_store, question, history
            )
            return result.text, retrieved

    elif engine == "v2":
        graph = baseline.get_graph()

        def answer_fn(question, history):
            state = graph.invoke({"question": question, "history": history or []})
            return state["answer"], state["retrieved"]

    else:  # agent
        agent = baseline.make_agent()

        def answer_fn(question, history):
            baseline.TOOL_LOG.clear()
            messages = [*baseline.history_to_messages(history), HumanMessage(question)]
            result = agent.invoke({"messages": messages})
            return result["messages"][-1].text, list(baseline.TOOL_LOG)

    return answer_fn


# =====================================================================
# 2. 채점
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
    """자동 채점"""
    squashed = squash(answer)

    if item["level"] == "no_answer":
        said_unknown = any(squash(p) in squashed for p in DONT_KNOW)
        made_up_money = re.search(MONEY, answer) is not None
        return said_unknown and not made_up_money, []

    wanted = list(item["keywords"]) + [str(v) for v in item["answer_values"]]
    missing = [w for w in wanted if not contains(squashed, w)]
    return len(missing) == 0, missing


# =====================================================================
# 3. 틀린 원인 추정
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
# 4. 실행과 집계
# =====================================================================
def run_eval(golden, answer_fn, file_to_notice, version):
    rows = []
    for item in golden:
        start = time.time()
        try:
            answer, retrieved = answer_fn(item["question"], item.get("history"))
            error = None
        except Exception as e:  # 한 문항이 실패해도 나머지는 계속 실행
            answer, retrieved, error = "", [], str(e)
        seconds = round(time.time() - start, 2)

        if error:
            correct, missing, cause = False, [], f"실행 오류: {error}"
        else:
            correct, missing = score(item, answer)
            cause = ""
            if not correct:
                wanted_ids = evidence_notice_ids(item, file_to_notice)
                cause = guess_cause(item, retrieved, wanted_ids)

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
                "수동확인": "수동확인 필요" if item["level"] == "no_answer" else "",
                "빠진_키워드": ", ".join(missing),
                "원인(자동추정)": cause,
                "원인(직접기입)": "",  # 변환 / 검색 / 모델 중 직접 적기
                "seconds": seconds,
                "sources": " || ".join(
                    f"{doc.metadata['공고 번호']} {doc.metadata['사업명']} (거리 {score_:.3f})"
                    for doc, score_ in retrieved
                ),
                "version": version,
                "chat_model": baseline.CHAT_MODEL,
                "embed_model": baseline.EMBED_MODEL,
                "chunk_size": baseline.CHUNK_SIZE,
                "chunk_overlap": baseline.CHUNK_OVERLAP,
                "top_k": baseline.TOP_K,
                "num_ctx": baseline.NUM_CTX,
            }
        )
    return pd.DataFrame(rows)


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


def main():
    parser = argparse.ArgumentParser(description="골든셋으로 베이스라인을 평가합니다.")
    parser.add_argument(
        "--golden", default=str(baseline.BASE_DIR / "data" / "goldenset.json")
    )
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 실행")
    parser.add_argument("--engine", default="v2", choices=["v1", "v2", "agent"])
    parser.add_argument(
        "--version", default=None, help="결과 파일 이름 (기본: 엔진 이름)"
    )
    args = parser.parse_args()
    args.version = args.version or args.engine

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8-sig"))
    if args.limit:
        golden = golden[: args.limit]
    print(f"골든셋 {len(golden)}문항을 평가합니다.")

    answer_fn = make_answer_fn(args.engine)
    df = run_eval(golden, answer_fn, load_file_to_notice(), args.version)
    table = summarize(df)

    print("\n=== 유형별 결과 ===")
    print(table.to_string())
    manual = df.loc[df["수동확인"] != "", "id"].tolist()
    if manual:
        print(f"\n수동확인 필요 문항: {', '.join(manual)}")

    baseline.OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    result_path = baseline.OUTPUT_DIR / f"eval_{args.version}_{stamp}.csv"
    summary_path = baseline.OUTPUT_DIR / f"eval_{args.version}_{stamp}_summary.csv"
    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    table.to_csv(summary_path, encoding="utf-8-sig")
    print(f"\n결과 저장: {result_path}\n요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
