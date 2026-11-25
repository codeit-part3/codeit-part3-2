# services/session_store.py
# 세션 ID별 대화 기록을 메모리에 저장하고 다시 꺼내는 모듈
# 저장된 대화기록은 30분 후 삭제됩니다(마지막 대화 기준).

import threading
from typing import Literal

from cachetools import TTLCache

Role = Literal["user", "assistant"]

ROLE_USER: Role = "user"
ROLE_ASSISTANT: Role = "assistant"

_chat_cache = TTLCache(maxsize=500, ttl=1800)
lock = threading.Lock()


def load_history(session_id: str) -> list[dict]:
    """특정 세션의 대화 기록을 조회하는 함수"""
    with lock:
        return list(_chat_cache.get(session_id, []))


def add_turn(session_id: str, message: str, answer: str) -> None:
    """
    질문과 답변을 한 턴으로 저장합니다.
    마지막 저장 후 30분이 지나면 삭제합니다.
    """
    turn = [
        {"role": ROLE_USER, "content": message},
        {"role": ROLE_ASSISTANT, "content": answer},
    ]
    with lock:
        history = list(_chat_cache.get(session_id, []))
        history.extend(turn)

        _chat_cache[session_id] = history
