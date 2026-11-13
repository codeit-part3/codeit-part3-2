"""골든셋을 점검하고 답변 결과를 채점합니다.

실행: uv run python -m ai.rahui.golden_eval
입력: ai/rahui/rahui_golden_set.json
출력: 터미널 요약
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

GOLDEN_PATH = Path(__file__).resolve().parent / "rahui_golden_set.json"
REFUSAL = (
    "잘 모르",
    "알 수 없",
    "찾을 수 없",
    "확인할 수 없",
    "문서에 없",
    "비공개",
    "공개하지 않",
    "포함되어 있지 않",
)
LEVELS = (
    "lookup",
    "interpret",
    "calc",
    "composite",
    "no_answer",
    "followup",
    "fuzzy_filter",
)
MAX_QUOTE = 80  # 공개 저장소이므로 원문 인용은 짧게


class GoldenItem(NamedTuple):
    id: str
    level: str  # lookup / interpret / calc / composite / no_answer / followup / fuzzy_filter
    category: str  # 사업 개요·기간·예산, 요구사항 조회 등
    question: str | None
    gold_answer: str | None
    keywords: tuple[str, ...]
    evidence_docs: tuple[int, ...]  # data_list.csv 행 순서 (1부터)
    evidence_files: tuple[str, ...]
    needs_calc: bool
    answer_values: tuple[float, ...]
    needs_image: bool
    history: tuple[dict, ...] = ()  # 후속 질문의 앞선 대화 (user, assistant 한 쌍)
    evidence_quote: str = ""

    @property
    def ready(self) -> bool:
        """질문과 정답이 채워진 문항인지."""
        return bool(self.question and self.gold_answer)


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenItem(
            d["id"],
            d["level"],
            d.get("category", ""),
            d["question"],
            d["gold_answer"],
            tuple(d["keywords"]),
            tuple(d["evidence_docs"]),
            tuple(d["evidence_files"]),
            d["needs_calc"],
            tuple(d["answer_values"]),
            d.get("needs_image", False),
            tuple(d.get("history", [])),
            d.get("evidence_quote", ""),
        )
        for d in raw
    ]


# ------------------------------------------------------------------ 채점
def _squash(text: str) -> str:
    return re.sub(r"[\s,]", "", text).lower()


def numbers_in(text: str) -> list[float]:
    nums = re.findall(r"\d[\d,]*\.?\d*", text)
    out = []
    for n in nums:
        try:
            out.append(float(n.replace(",", "").rstrip(".")))
        except ValueError:
            continue
    return out


def keyword_score(answer: str, item: GoldenItem) -> float | None:
    """들어 있어야 할 키워드 중 답변에 있는 비율. 키워드가 없으면 None."""
    if not item.keywords:
        return None
    a = _squash(answer)
    hit = sum(1 for k in item.keywords if _squash(k) in a)
    return hit / len(item.keywords)


def numbers_ok(answer: str, item: GoldenItem) -> bool | None:
    """answer_values 의 값이 모두 답변 속 숫자에 있는지. 값이 없으면 None."""
    if not item.answer_values:
        return None
    found = numbers_in(answer)
    return all(any(abs(f - v) < 1e-6 for f in found) for v in item.answer_values)


def refused(answer: str) -> bool:
    return any(r in answer for r in REFUSAL)


def retrieval_scores(retrieved: list[int], item: GoldenItem, k: int = 5) -> dict:
    """근거 문서를 얼마나 찾았는지. 근거 문서가 없는 문항(답 없음)은 빈 값."""
    if not item.evidence_docs:
        return {}
    need = set(item.evidence_docs)
    top = retrieved[:k]
    return {
        "recall": len(need & set(retrieved)) / len(need),
        "hit_at_k": int(bool(need & set(top))),
    }


def score_item(item: GoldenItem, result: dict, k: int = 5) -> dict:
    answer = result.get("answer", "")
    row: dict = {"id": item.id, "level": item.level, "category": item.category}
    if item.level == "no_answer":
        row["pass"] = refused(answer)
    else:
        kw = keyword_score(answer, item)
        nm = numbers_ok(answer, item)
        row["keyword"] = kw
        row["numbers"] = nm
        row["pass"] = (kw == 1.0) and (nm is not False)
    if "retrieved_docs" in result:
        row.update(retrieval_scores(result["retrieved_docs"], item, k))
    return row


def evaluate(
    items: list[GoldenItem], answers: dict[str, dict], k: int = 5
) -> list[dict]:
    """answers 에 있는 문항만 채점합니다. 질문이 비어 있는 문항은 건너뜁니다."""
    out = []
    for item in items:
        if not item.ready or item.id not in answers:
            continue
        out.append(score_item(item, answers[item.id], k))
    return out


def print_report(results: list[dict]) -> None:
    if not results:
        print("채점할 결과가 없습니다.")
        return
    print(f"채점 {len(results)}문항")
    by_level: dict[str, list[dict]] = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r)
    for level, rows in by_level.items():
        ok = sum(1 for r in rows if r["pass"])
        rec = [r["recall"] for r in rows if "recall" in r]
        extra = f" | 근거 문서 recall {sum(rec) / len(rec):.2f}" if rec else ""
        print(f"  {level:<10} {ok}/{len(rows)} 통과{extra}")
    total = sum(1 for r in results if r["pass"])
    print(f"전체 {total}/{len(results)} 통과 ({total / len(results) * 100:.0f}%)")


# ------------------------------------------------------------------ 점검
def check(items: list[GoldenItem]) -> list[str]:
    """골든셋 자체의 문제를 찾습니다. 문제가 없으면 빈 목록."""
    problems = []
    ids = [i.id for i in items]
    problems += [f"id 중복: {k}" for k, c in Counter(ids).items() if c > 1]
    for i in items:
        if not i.ready:
            problems.append(f"{i.id}: 질문/정답이 비어 있음 (작성 필요)")
            continue
        if i.level not in LEVELS:
            problems.append(f"{i.id}: 알 수 없는 level '{i.level}'")
        if len(i.evidence_quote) > MAX_QUOTE:
            problems.append(
                f"{i.id}: evidence_quote 가 {MAX_QUOTE}자보다 긺 (원문 노출 주의)"
            )
        if i.level == "followup":
            roles = [h.get("role") for h in i.history]
            if roles != ["user", "assistant"]:
                problems.append(f"{i.id}: history 는 user 1턴 + assistant 1턴이어야 함")
        elif i.history:
            problems.append(f"{i.id}: followup 이 아닌데 history 가 있음")
        if i.level == "no_answer":
            continue
        if not i.keywords:
            problems.append(f"{i.id}: keywords 가 없음")
        if not i.evidence_docs:
            problems.append(f"{i.id}: 근거 문서 번호가 없음")
        if len(i.evidence_docs) != len(i.evidence_files):
            problems.append(f"{i.id}: 근거 문서 번호와 파일명 개수가 다름")
        if i.needs_calc and not i.answer_values:
            problems.append(f"{i.id}: 계산형인데 answer_values 가 없음")
    return problems


def main() -> None:
    items = load_golden()
    ready = [i for i in items if i.ready]
    print(f"문항 {len(items)}개 (작성 완료 {len(ready)}개)")
    print("level 별:", dict(Counter(i.level for i in items)))
    print("유형 별:", dict(Counter(i.category for i in items)))
    print(f"그림 의존 문항: {sum(1 for i in items if i.needs_image)}개")
    docs = sorted({d for i in items for d in i.evidence_docs})
    print(f"근거로 쓰인 문서: {len(docs)}건 {docs}")
    problems = check(items)
    print("점검 결과:", "이상 없음" if not problems else "")
    for p in problems:
        print("  -", p)

    try:  # 원본 파일이 있으면 파일명이 실제로 있는지도 확인
        from ai.rahui.convert import FILES
    except ImportError:
        return
    missing = [
        f"{i.id}: {f}"
        for i in items
        for f in i.evidence_files
        if not (FILES / f).exists()
    ]
    print("근거 파일이 data/files 에 없는 문항:", missing if missing else "없음")


if __name__ == "__main__":
    main()
