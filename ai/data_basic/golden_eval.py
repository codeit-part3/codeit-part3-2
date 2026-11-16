"""골든셋을 점검하고 답변 결과를 채점합니다. (팀 공통판)

실행 (프로젝트 루트): uv run python -m ai.data_basic.golden_eval
입력: data/golden/rahui_golden_set_v2_2.json (경로는 ai/data_basic/paths.py 에서 바꿉니다)
출력: 터미널 요약

다른 코드에서 불러 쓸 때:
    from ai.data_basic.golden_eval import load_golden, score_item
원본: ai/rahui/evaluation/golden_eval.py (경로 설정만 팀 공통 경로로 바꿨습니다)
"""

import json  # 골든셋 JSON 파일을 읽기 위해 가져옵니다.
import re  # 답변에서 공백·쉼표를 정리하고 숫자를 찾기 위해 가져옵니다.
from collections import Counter  # level별 문항 수와 중복 ID를 세기 위해 가져옵니다.
from pathlib import Path  # 파일 경로를 운영체제에 맞게 다루기 위해 가져옵니다.
from typing import NamedTuple  # 골든셋 한 문항의 필드 구조를 정의하기 위해 가져옵니다.

from ai.data_basic.paths import (  # 공통 골든셋 파일과 원본 문서 폴더 경로를 가져옵니다.
    GOLDEN_PATH,
    RAW_FILES_DIR,
)

REFUSAL = (  # 답변이 알 수 없음을 적절히 밝혔는지 판별할 때 찾을 표현들입니다.
    "잘 모르",  # 잘 모르겠다는 표현을 찾습니다.
    "알 수 없",  # 알 수 없다는 표현을 찾습니다.
    "찾을 수 없",  # 찾을 수 없다는 표현을 찾습니다.
    "확인할 수 없",  # 확인할 수 없다는 표현을 찾습니다.
    "문서에 없",  # 문서에 정보가 없다는 표현을 찾습니다.
    "비공개",  # 공개되지 않은 정보라는 표현을 찾습니다.
    "공개하지 않",  # 공개하지 않았다는 표현을 찾습니다.
    "포함되어 있지 않",  # 자료에 포함되지 않았다는 표현을 찾습니다.
)
LEVELS = (  # 골든셋에서 허용하는 문항 유형 목록입니다.
    "lookup",  # 문서에서 사실을 그대로 찾는 유형입니다.
    "interpret",  # 문서 내용을 해석해야 하는 유형입니다.
    "calc",  # 문서의 값으로 계산해야 하는 유형입니다.
    "composite",  # 여러 정보나 근거를 종합해야 하는 유형입니다.
    "no_answer",  # 문서에 답이 없어 모른다고 해야 하는 유형입니다.
    "followup",  # 앞선 대화의 맥락을 이어 답하는 유형입니다.
    "fuzzy_filter",  # 조건에 맞는 항목을 유연하게 찾는 유형입니다.
)
MAX_QUOTE = 80  # 공개 저장소에 긴 원문이 들어가지 않도록 허용 인용 길이를 제한합니다.


class GoldenItem(NamedTuple):  # 골든셋의 문항 하나를 담는 읽기 전용 데이터 구조입니다.
    id: str  # 문항을 구분하는 고유 ID입니다.
    level: str  # 이 문항의 유형(lookup, interpret, calc 등)입니다.
    category: str  # 사업 개요·기간·예산 등 세부 분류입니다.
    question: str | None  # 질문 내용이며, 아직 작성되지 않았으면 None입니다.
    gold_answer: str | None  # 기준 정답이며, 아직 작성되지 않았으면 None입니다.
    keywords: tuple[str, ...]  # 정답에 포함되어야 할 핵심 키워드들입니다.
    evidence_docs: tuple[
        int, ...
    ]  # 근거 문서 번호이며 data_list.csv 기준 1부터 시작합니다.
    evidence_files: tuple[str, ...]  # 근거가 있는 원본 파일 이름들입니다.
    needs_calc: bool  # 정답을 구하는 데 계산이 필요한지 나타냅니다.
    answer_values: tuple[float, ...]  # 답변에 포함되어야 하는 숫자 정답들입니다.
    needs_image: bool  # 답을 찾을 때 문서의 이미지 확인이 필요한지 나타냅니다.
    history: tuple[
        dict, ...
    ] = ()  # 후속 질문의 앞선 대화(user와 assistant 한 쌍)입니다.
    evidence_quote: str = ""  # 근거의 짧은 인용문이며 기본값은 빈 문자열입니다.

    @property
    def ready(self) -> bool:
        """질문과 정답이 채워진 문항인지."""
        return bool(
            self.question and self.gold_answer
        )  # 질문과 정답이 모두 참값일 때 작성 완료로 봅니다.


def load_golden(
    path: Path = GOLDEN_PATH,
) -> list[GoldenItem]:  # JSON 골든셋 파일을 문항 객체 목록으로 읽습니다.
    raw = json.loads(
        path.read_text(encoding="utf-8")
    )  # UTF-8 JSON 파일 전체를 읽어 파이썬 자료형으로 바꿉니다.
    return [  # 원본 JSON의 각 항목을 GoldenItem으로 변환한 목록을 돌려줍니다.
        GoldenItem(  # 딕셔너리 하나를 필드 순서에 맞춰 문항 객체로 만듭니다.
            d["id"],  # 필수 ID를 읽습니다.
            d["level"],  # 필수 문항 유형을 읽습니다.
            d.get("category", ""),  # 분류가 없으면 빈 문자열을 사용합니다.
            d["question"],  # 질문 내용을 읽습니다.
            d["gold_answer"],  # 기준 정답을 읽습니다.
            tuple(d["keywords"]),  # 키워드 목록을 변경 불가능한 튜플로 바꿉니다.
            tuple(d["evidence_docs"]),  # 근거 문서 번호 목록을 튜플로 바꿉니다.
            tuple(d["evidence_files"]),  # 근거 파일 이름 목록을 튜플로 바꿉니다.
            d["needs_calc"],  # 계산 필요 여부를 읽습니다.
            tuple(d["answer_values"]),  # 숫자 정답 목록을 튜플로 바꿉니다.
            d.get(
                "needs_image", False
            ),  # 이미지 필요 여부가 없으면 False로 간주합니다.
            tuple(
                d.get("history", [])
            ),  # 대화 이력이 없으면 빈 목록을 튜플로 바꿉니다.
            d.get("evidence_quote", ""),  # 근거 인용문이 없으면 빈 문자열을 사용합니다.
        )  # 현재 JSON 항목을 GoldenItem으로 만드는 호출을 끝냅니다.
        for d in raw  # JSON 배열의 각 문항 딕셔너리를 순회합니다.
    ]  # 변환된 모든 문항을 담은 목록을 반환합니다.


# ------------------------------------------------------------------ 채점
def _squash(text: str) -> str:  # 문자열을 비교하기 쉬운 형태로 정규화합니다.
    return re.sub(
        r"[\s,]", "", text
    ).lower()  # 공백과 쉼표를 없애고 소문자로 바꿔 문자열 비교를 쉽게 합니다.


def numbers_in(
    text: str,
) -> list[float]:  # 입력 문자열에 포함된 숫자를 실수 목록으로 추출합니다.
    nums = re.findall(
        r"\d[\d,]*\.?\d*", text
    )  # 정수·소수·천 단위 쉼표가 있는 숫자 표현을 모두 찾습니다.
    out = []  # 변환에 성공한 숫자를 모아 둘 목록입니다.
    for n in nums:  # 찾은 숫자 문자열을 하나씩 처리합니다.
        try:  # 쉼표 등을 정리해 실수로 변환을 시도합니다.
            out.append(
                float(n.replace(",", "").rstrip("."))
            )  # 천 단위 쉼표와 끝의 점을 제거한 뒤 숫자로 저장합니다.
        except ValueError:  # 숫자 형식이 예상과 달라 변환할 수 없는 경우입니다.
            continue  # 해당 표현만 건너뛰고 다음 숫자를 계속 처리합니다.
    return out  # 변환에 성공한 숫자 목록을 반환합니다.


def keyword_score(
    answer: str, item: GoldenItem
) -> float | None:  # 기준 키워드가 답변에 포함된 비율을 계산합니다.
    """들어 있어야 할 키워드 중 답변에 있는 비율. 키워드가 없으면 None."""
    if not item.keywords:  # 비교할 키워드가 없으면 점수를 계산할 수 없습니다.
        return None  # 키워드가 없는 상태를 숫자 점수와 구분해 반환합니다.
    a = _squash(answer)  # 답변의 공백·쉼표를 정리하고 소문자로 변환합니다.
    hit = sum(
        1 for k in item.keywords if _squash(k) in a
    )  # 답변에 들어 있는 키워드의 개수를 셉니다.
    return hit / len(item.keywords)  # 전체 키워드 중 포함된 비율을 반환합니다.


def numbers_ok(
    answer: str, item: GoldenItem
) -> bool | None:  # 필수 숫자 정답이 답변에 포함됐는지 검사합니다.
    """answer_values 의 값이 모두 답변 속 숫자에 있는지. 값이 없으면 None."""
    if (
        not item.answer_values
    ):  # 확인할 숫자 정답이 없으면 숫자 점검은 적용하지 않습니다.
        return None  # 숫자 점검을 하지 않았음을 None으로 표시합니다.
    found = numbers_in(answer)  # 답변에서 숫자들을 추출합니다.
    return all(
        any(abs(f - v) < 1e-6 for f in found) for v in item.answer_values
    )  # 각 정답값이 오차 범위 내 숫자로 답변에 있는지 확인합니다.


def refused(answer: str) -> bool:  # 답변이 알 수 없음을 적절히 밝혔는지 검사합니다.
    return any(
        r in answer for r in REFUSAL
    )  # 답변에 거절·정보 부재 표현 중 하나라도 있는지 확인합니다.


def retrieval_scores(
    retrieved: list[int], item: GoldenItem, k: int = 5
) -> dict:  # 검색된 문서 목록의 recall과 hit@k를 계산합니다.
    """근거 문서를 얼마나 찾았는지. 근거 문서가 없는 문항(답 없음)은 빈 값."""
    if (
        not item.evidence_docs
    ):  # 근거 문서가 지정되지 않은 문항은 검색 성능을 평가하지 않습니다.
        return {}  # 평가할 값이 없음을 빈 딕셔너리로 나타냅니다.
    need = set(
        item.evidence_docs
    )  # 필요한 문서 번호를 집합으로 바꿔 교집합 계산을 준비합니다.
    top = retrieved[:k]  # 검색 결과 중 상위 k개 문서만 추립니다.
    return {  # 검색 성능 지표 두 가지를 딕셔너리로 반환합니다.
        "recall": len(need & set(retrieved))
        / len(need),  # 검색된 전체 결과에서 필요한 문서가 차지하는 비율입니다.
        "hit_at_k": int(
            bool(need & set(top))
        ),  # 상위 k개에 필요한 문서가 하나라도 있으면 1, 아니면 0입니다.
    }


def score_item(
    item: GoldenItem, result: dict, k: int = 5
) -> dict:  # 문항 하나의 답변과 검색 결과를 채점합니다.
    answer = result.get("answer", "")  # 결과에 답변이 없으면 빈 문자열로 채점합니다.
    row: dict = {
        "id": item.id,
        "level": item.level,
        "category": item.category,
    }  # 결과 행에 문항 ID·유형·분류를 넣습니다.
    if item.level == "no_answer":  # 답이 없어야 하는 문항인지 확인합니다.
        row["pass"] = refused(
            answer
        )  # 답변에 정보 부재 표현이 있으면 통과로 기록합니다.
    else:  # 답변 내용이 있는 문항은 키워드와 숫자 기준으로 채점합니다.
        kw = keyword_score(answer, item)  # 핵심 키워드 포함 비율을 계산합니다.
        nm = numbers_ok(answer, item)  # 필수 숫자들이 답변에 있는지 확인합니다.
        row["keyword"] = kw  # 키워드 점수를 결과 행에 추가합니다.
        row["numbers"] = nm  # 숫자 검사 결과를 결과 행에 추가합니다.
        row["pass"] = (kw == 1.0) and (
            nm is not False
        )  # 키워드가 모두 있고 숫자 검사에서 실패하지 않으면 통과입니다.
    if "retrieved_docs" in result:  # 검색된 문서 목록도 결과에 제공됐는지 확인합니다.
        row.update(
            retrieval_scores(result["retrieved_docs"], item, k)
        )  # 검색 문서가 있으면 recall과 hit@k를 함께 추가합니다.
    return row  # 이 문항의 채점 결과를 반환합니다.


def evaluate(
    items: list[GoldenItem],
    answers: dict[str, dict],
    k: int = 5,  # 채점할 문항·답변·검색 상한을 입력받습니다.
) -> list[dict]:  # 답변이 있는 작성 완료 문항의 채점 결과를 반환합니다.
    """answers 에 있는 문항만 채점합니다. 질문이 비어 있는 문항은 건너뜁니다."""
    out = []  # 채점된 결과 행을 모아 둘 목록입니다.
    for item in items:  # 골든셋의 각 문항을 차례로 확인합니다.
        if (
            not item.ready or item.id not in answers
        ):  # 미완성 문항이거나 답변 결과가 없으면 채점 대상에서 제외합니다.
            continue  # 다음 문항을 확인합니다.
        out.append(
            score_item(item, answers[item.id], k)
        )  # 해당 문항의 답변을 채점해 결과 목록에 추가합니다.
    return out  # 채점이 끝난 결과 행 목록을 반환합니다.


def print_report(
    results: list[dict],
) -> None:  # 채점 결과를 유형별·전체 요약으로 터미널에 출력합니다.
    if not results:  # 결과가 비어 있는지 먼저 확인합니다.
        print("채점할 결과가 없습니다.")  # 채점할 항목이 없음을 출력합니다.
        return  # 보고서 생성을 마치고 함수를 끝냅니다.
    print(f"채점 {len(results)}문항")  # 채점 결과에 포함된 전체 문항 수를 출력합니다.
    by_level: dict[str, list[dict]] = {}  # 문항 유형별 결과를 모을 딕셔너리입니다.
    for r in results:  # 채점 결과 한 행씩 유형별로 분류합니다.
        by_level.setdefault(r["level"], []).append(
            r
        )  # 유형별 목록을 만들거나 찾아 현재 결과를 추가합니다.
    for level, rows in by_level.items():  # 각 유형에 대한 결과 목록을 순회합니다.
        ok = sum(1 for r in rows if r["pass"])  # 해당 유형에서 통과한 문항 수를 셉니다.
        rec = [
            r["recall"] for r in rows if "recall" in r
        ]  # recall이 계산된 문항만 모읍니다.
        extra = (
            f" | 근거 문서 recall {sum(rec) / len(rec):.2f}" if rec else ""
        )  # recall이 있으면 평균을 표시할 문자열을 만듭니다.
        print(
            f"  {level:<10} {ok}/{len(rows)} 통과{extra}"
        )  # 유형별 통과 수와 평균 recall을 출력합니다.
    total = sum(1 for r in results if r["pass"])  # 전체 통과 문항 수를 셉니다.
    print(
        f"전체 {total}/{len(results)} 통과 ({total / len(results) * 100:.0f}%)"
    )  # 전체 통과 개수와 비율을 출력합니다.


# ------------------------------------------------------------------ 점검
def check(
    items: list[GoldenItem],
) -> list[str]:  # 골든셋 필드와 문항 간 일관성을 검사합니다.
    """골든셋 자체의 문제를 찾습니다. 문제가 없으면 빈 목록."""
    problems = []  # 발견한 문제 설명을 모아 둘 목록입니다.
    ids = [i.id for i in items]  # 모든 문항의 ID를 뽑아 중복 검사를 준비합니다.
    problems += [
        f"id 중복: {k}" for k, c in Counter(ids).items() if c > 1
    ]  # 두 번 이상 나온 ID를 문제 목록에 추가합니다.
    for i in items:  # 골든셋의 각 문항을 순서대로 검사합니다.
        if not i.ready:  # 질문이나 정답이 빠져 있는지 확인합니다.
            problems.append(
                f"{i.id}: 질문/정답이 비어 있음 (작성 필요)"
            )  # 미완성 문항의 ID와 문제를 기록합니다.
            continue  # 미완성 문항은 나머지 필드 검사를 건너뜁니다.
        if i.level not in LEVELS:  # level 값이 허용 목록에 포함되는지 확인합니다.
            problems.append(
                f"{i.id}: 알 수 없는 level '{i.level}'"
            )  # 허용되지 않은 level을 문제로 기록합니다.
        if (
            len(i.evidence_quote) > MAX_QUOTE
        ):  # 인용문이 설정된 최대 길이보다 긴지 확인합니다.
            problems.append(  # 긴 인용문 문제를 여러 줄로 작성해 목록에 추가합니다.
                f"{i.id}: evidence_quote 가 {MAX_QUOTE}자보다 긺 (원문 노출 주의)"  # 인용 길이 제한을 초과했다고 기록합니다.
            )  # 문제 설명을 problems 목록에 추가하는 호출을 끝냅니다.
        if (
            i.level == "followup"
        ):  # 후속 질문 유형이면 앞선 대화의 역할 순서를 검사합니다.
            roles = [
                h.get("role") for h in i.history
            ]  # history 각 항목에서 역할 이름을 추출합니다.
            if roles != [
                "user",
                "assistant",
            ]:  # 사용자 질문 다음 assistant 답변 순서인지 확인합니다.
                problems.append(
                    f"{i.id}: history 는 user 1턴 + assistant 1턴이어야 함"
                )  # 대화 구조가 다르면 문제로 기록합니다.
        elif i.history:  # 후속 질문이 아닌데 대화 이력이 들어 있는지 확인합니다.
            problems.append(
                f"{i.id}: followup 이 아닌데 history 가 있음"
            )  # 불필요한 history가 있으면 문제로 기록합니다.
        if i.level == "no_answer":  # 답이 없는 유형은 근거나 키워드 규칙이 다릅니다.
            continue  # 아래의 일반 답변 문항 검사들을 건너뜁니다.
        if not i.keywords:  # 답변 채점에 필요한 키워드가 있는지 확인합니다.
            problems.append(
                f"{i.id}: keywords 가 없음"
            )  # 키워드가 없으면 문제로 기록합니다.
        if not i.evidence_docs:  # 답을 뒷받침할 문서 번호가 지정됐는지 확인합니다.
            problems.append(
                f"{i.id}: 근거 문서 번호가 없음"
            )  # 근거 번호가 없으면 문제로 기록합니다.
        if len(i.evidence_docs) != len(
            i.evidence_files
        ):  # 문서 번호와 파일 이름의 개수가 서로 맞는지 확인합니다.
            problems.append(
                f"{i.id}: 근거 문서 번호와 파일명 개수가 다름"
            )  # 개수가 다르면 문제로 기록합니다.
        if (
            i.needs_calc and not i.answer_values
        ):  # 계산 문항인데 비교할 숫자 정답이 없는지 확인합니다.
            problems.append(
                f"{i.id}: 계산형인데 answer_values 가 없음"
            )  # 계산 정답값이 없으면 문제로 기록합니다.
    return problems  # 발견한 문제 설명을 모두 반환합니다.


def main() -> None:  # 골든셋 현황 요약과 형식 점검을 실행합니다.
    items = load_golden()  # 기본 경로의 골든셋 전체를 읽습니다.
    ready = [i for i in items if i.ready]  # 질문과 정답이 모두 작성된 문항만 모읍니다.
    print(
        f"문항 {len(items)}개 (작성 완료 {len(ready)}개)"
    )  # 전체 문항 수와 작성 완료 문항 수를 출력합니다.
    print(
        "level 별:", dict(Counter(i.level for i in items))
    )  # level 유형별 문항 수를 출력합니다.
    print(
        "유형 별:", dict(Counter(i.category for i in items))
    )  # 세부 category별 문항 수를 출력합니다.
    print(
        f"그림 의존 문항: {sum(1 for i in items if i.needs_image)}개"
    )  # 이미지 확인이 필요한 문항 수를 출력합니다.
    docs = sorted(
        {d for i in items for d in i.evidence_docs}
    )  # 근거에 등장하는 문서 번호를 중복 없이 정렬합니다.
    print(
        f"근거로 쓰인 문서: {len(docs)}건 {docs}"
    )  # 근거 문서의 개수와 번호 목록을 출력합니다.
    problems = check(items)  # 골든셋의 형식과 필수 필드를 검사합니다.
    print(
        "점검 결과:", "이상 없음" if not problems else ""
    )  # 문제가 없으면 이상 없음을 출력합니다.
    for p in problems:  # 발견된 문제 설명을 하나씩 출력합니다.
        print("  -", p)  # 문제 내용을 읽기 쉬운 목록 형식으로 출력합니다.

    if (
        RAW_FILES_DIR.exists()
    ):  # 원본 파일 폴더가 있을 때만 실제 근거 파일 존재 여부를 확인합니다.
        missing = [  # 디스크에서 찾지 못한 근거 파일을 문항 ID와 함께 모읍니다.
            f"{i.id}: {f}"  # 누락 파일에 해당하는 문항 ID와 파일명을 표시합니다.
            for i in items  # 모든 골든셋 문항을 살펴봅니다.
            for f in i.evidence_files  # 각 문항의 근거 파일 이름을 살펴봅니다.
            if not (
                RAW_FILES_DIR / f
            ).exists()  # 원본 폴더 아래에 해당 파일이 없을 때만 선택합니다.
        ]
        print(
            "근거 파일이 data/files 에 없는 문항:", missing if missing else "없음"
        )  # 누락 문항 목록 또는 없음을 출력합니다.


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 점검 작업을 시작합니다.
    main()  # 골든셋 요약과 형식 검사를 실행합니다.
