# backend/routes/chat.py
from fastapi import APIRouter
from loguru import logger

from backend.schemas import ChatRequest, ChatResponse
from backend.services.engine_loader import get_engine
from backend.services.session_store import add_turn, load_history

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """서버와 통신"""

    # 저장된 대화 기록 및 엔진 불러오기
    history = load_history(request.session_id)
    engine_fn = get_engine(request.engine_id)
    try:
        answer = engine_fn(request.message, history)
        # 현재 대화를 history에 저장
        add_turn(request.session_id, request.message, answer)

        return ChatResponse(
            engine_id=request.engine_id,
            session_id=request.session_id,
            answer=answer,
            history_length=len(history),
        )

    except Exception:
        logger.exception("응답 생성에 실패했습니다.")
