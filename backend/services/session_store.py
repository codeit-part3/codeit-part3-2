# services/session_store.py
# 세션 ID별 대화 기록을 메모리에 저장하고 다시 꺼내는 모듈

from typing import Literal

Role = Literal["user", "assistant"]

ROLE_USER: Role = "user"
ROLE_ASSISTANT: Role = "assistant"

_history: dict[str, list[dict[str, str]]] = {}


def load_history(session_id: str) -> list[dict]:
    """
    특정 세션의 대화 기록을 조회하는 함수
    이전 대화기록이 없으면 빈 리스트를 반환합니다.
    """
    return list(_history.get(session_id, []))


def add_turn(session_id: str, message: str, answer: str) -> None:
    """질문과 답변을 한 라운드로 묶어 저장하는 함수"""
    round = [
        {"role": ROLE_USER, "content": message},
        {"role": ROLE_ASSISTANT, "content": answer},
    ]

    _history.setdefault(session_id, []).extend(round)
