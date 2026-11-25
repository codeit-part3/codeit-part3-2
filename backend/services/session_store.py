# services/session_store.py
# 세션 ID별 대화 기록을 메모리에 저장하고 다시 꺼내는 모듈

from typing import Literal

from acachetools import TTLCache

Role = Literal["user", "assistant"]

ROLE_USER: Role = "user"
ROLE_ASSISTANT: Role = "assistant"

_chat_cache = TTLCache(maxsize=500, ttl=1800)


async def load_history(session_id: str) -> list[dict]:
    """
    특정 세션의 대화 기록을 조회하는 함수
    """
    return list(_chat_cache.get(session_id, []))


async def add_turn(session_id: str, message: str, answer: str) -> None:
    """
    질문과 답변을 한 턴으로 저장합니다.
    마지막 저장 후 30분이 지나면 삭제합니다.
    """
    turn = [
        {"role": ROLE_USER, "content": message},
        {"role": ROLE_ASSISTANT, "content": answer},
    ]
    history = await load_history(session_id)
    history.extend(turn)

    _chat_cache[session_id] = history
