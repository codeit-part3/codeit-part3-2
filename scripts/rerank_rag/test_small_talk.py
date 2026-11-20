"""인사·도움말 처리 점검 (모델·검색 없이 라우팅과 고정 답변만 확인한다).

    uv run python scripts/rerank_rag/test_small_talk.py

표의 '기대'가 '고정'이면 정해진 문장으로 바로 답해야 하고, '원래 경로'이면 인사로 걸러지지 않고
가이드/서식/일반 질문(rag)으로 가야 한다. 하나라도 어긋나면 종료 코드 1.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engines.rerank_rag import writer as w  # noqa: E402

# (입력, 기대, 설명)
CASES = [
    ("안녕", "고정", "인사"),
    ("안녕하세요", "고정", "인사"),
    ("안녕하세요!", "고정", "인사"),
    ("하이", "고정", "인사"),
    ("hello", "고정", "영어 인사"),
    ("고마워", "고정", "감사"),
    ("감사합니다", "고정", "감사"),
    ("도움말", "고정", "도움말"),
    ("뭘 할 수 있어?", "고정", "도움말"),
    ("너 누구야", "고정", "도움말"),
    ("오늘 점심 뭐 먹을까", "고정", "입찰과 무관 -> 안내"),
    ("음 글쎄", "고정", "막연한 말 -> 안내"),
    ("음...", "고정", "막연한 말 -> 안내"),
    ("글쎄요", "고정", "막연한 말 -> 안내"),
    ("몰라", "고정", "막연한 말 -> 안내"),
    ("음 케빈랩 일정", "원래 경로", "추임새가 섞인 공고 질문"),
    ("안녕, 케빈랩 평가항목 알려줘", "원래 경로", "인사가 섞인 공고 질문"),
    ("케빈랩 평택시 사업 작성 가이드 만들어줘", "원래 경로", "가이드 요청"),
    ("벤처기업협회 서식 예시 보여줘", "원래 경로", "서식 요청"),
    ("이 사업 예산이 얼마야?", "원래 경로", "일반 공고 질문"),
    ("3번", "원래 경로", "서식 목록 고르기"),
]


def main() -> int:
    bad = 0
    print(f"{'입력':<28}{'기대':<10}{'결과':<10}{'시간(초)':<10}설명")
    for q, want, why in CASES:
        t = time.perf_counter()
        got_text = w.small_talk(q)
        dt = time.perf_counter() - t
        got = "고정" if got_text is not None else "원래 경로"
        ok = got == want
        bad += not ok
        print(
            f"{q:<28}{want:<10}{got:<10}{dt:<10.4f}{why}{'' if ok else '  <-- 어긋남'}"
        )
    # 사업명 없는 공고 질문은 되묻고, 사업명·앞 대화·전체 질문은 통과하는지 (모델·검색 없음)
    hist = [
        {
            "role": "assistant",
            "content": "2024년 벤처확인종합관리시스템 기능 고도화 용역사업 입찰공고에는 서식이 9개 있어요.",
        }
    ]
    ask_cases = [
        ("일정이 어떻게 돼?", None, True, "사업명·앞 대화 없음 -> 되묻기"),
        ("평가항목 알려줘", None, True, "사업명·앞 대화 없음 -> 되묻기"),
        ("그 사업 일정은?", None, True, "앞 대화 없음 -> 되묻기"),
        ("그 사업 일정은?", hist, False, "앞 대화에 사업 있음 -> 통과"),
        ("케빈랩 평가항목 알려줘", None, False, "사업명 있음 -> 통과"),
        ("예산이 가장 큰 사업은?", None, False, "공고 전체 질문 -> 통과"),
        ("입찰이 뭐야?", None, False, "세부 내용 질문 아님 -> 통과"),
    ]
    print()
    for q, h, want, why in ask_cases:
        got = w.needs_project({"question": q, "history": h})
        ok = got == want
        bad += not ok
        print(f"{q:<24}되묻기={got!s:<6}{why}{'' if ok else '  <-- 어긋남'}")
    # 되물은 뒤 사업명·번호로 답하면: 하나로 정해지면 앞 질문과 합치고, 여러 공고면 번호로 고르게 하고, 새 질문이면 그대로 둔다
    ask = [
        {"role": "user", "content": "일정이 어떻게 돼?"},
        {"role": "assistant", "content": w.ASK_PROJECT},
    ]
    fu_cases = [
        ("케빈랩", "question", "한 공고로 정해짐 -> 앞 질문과 합침"),
        ("평택시", "answer", "2개 공고에 걸림 -> 번호로 고르게"),
        ("음 글쎄", "answer", "사업명 아님 -> 다시 되묻기"),
        ("케빈랩 평가항목 알려줘", None, "새 질문 -> 건드리지 않음"),
    ]
    print()
    for reply, want, why in fu_cases:
        r = w._project_followup({"question": reply, "history": ask})
        got = None if r is None else ("answer" if "answer" in r else "question")
        ok = got == want
        bad += not ok
        print(f"{reply:<24}{got!s:<10}{why}{'' if ok else '  <-- 어긋남'}")
    r = w._project_followup({"question": "케빈랩", "history": ask})
    r2 = w.n_chat({"question": "음 글쎄", "history": ask})["answer"]
    ok = r2.startswith(w.ASK_PROJECT)
    bad += not ok
    print(f"{'되묻는 중 추임새':<22}{'다시 되묻기 PASS' if ok else 'FAIL'}")
    print("  합친 질문:", r["question"])
    # 예산 순위 질문은 모델 없이 계산해서 답한다
    for q, must in [
        ("예산이 가장 많은 사업", "가장 많은"),
        ("사업 금액이 가장 적은 사업 알려줘", "가장 적은"),
        ("예산 상위 5개", "| 5 |"),
    ]:
        a = w.rank_answer(q) or ""
        ok = must in a
        bad += not ok
        print(f"{q:<24}{'PASS' if ok else 'FAIL'}")
    print(w.rank_answer("예산이 가장 많은 사업").split("\n")[0])
    print(w.rank_answer("예산이 가장 적은 사업").split("\n")[0])
    # 그래프 전체를 거쳐도 인사는 모델 없이 즉시 돌려주는지
    t = time.perf_counter()
    ans = w.answer("안녕하세요", [])
    dt = time.perf_counter() - t
    ok = ans.startswith("안녕하세요!") and dt < 2
    bad += not ok
    print(f"\n그래프 전체(answer) 인사 응답 {dt:.3f}초: {'PASS' if ok else 'FAIL'}")
    print("PASS" if not bad else f"FAIL  {bad}건 어긋남")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
