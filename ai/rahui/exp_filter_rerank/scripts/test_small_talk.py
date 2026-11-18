"""인사·도움말 처리 점검 (모델·검색 없이 라우팅과 고정 답변만 확인한다).

    uv run python ai/rahui/exp_filter_rerank/scripts/test_small_talk.py

표의 '기대'가 '고정'이면 정해진 문장으로 바로 답해야 하고, '원래 경로'이면 인사로 걸러지지 않고
가이드/서식/일반 질문(rag)으로 가야 한다. 하나라도 어긋나면 종료 코드 1.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ai.rahui.exp_filter_rerank.rahui_engine import writer as w  # noqa: E402

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
