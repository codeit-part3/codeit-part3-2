# backend/routes/chat.py
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from backend.schemas import ChatRequest, ChatResponse
from backend.services.auth import verify_key
from backend.services.engine_loader import load_engine
from backend.services.session_store import add_turn, load_history

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(verify_key)])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """서버와 통신"""

    # 엔진 로딩
    engine_fn = await load_engine(request.engine_id)

    # 대화 기록 로딩
    history = load_history(request.session_id)

    # 응답 생성
    try:
        model_resp = await engine_fn.ainvoke(
            {"question": request.message, "history": history}
        )

        # 현재 대화를 history에 저장
        add_turn(request.session_id, request.message, model_resp["answer"])

        return ChatResponse(
            engine_id=request.engine_id,
            session_id=request.session_id,
            answer=model_resp["answer"],
            retrieved=model_resp["retrieved"],
            history_length=len(history),
        )
    except Exception as e:
        logger.exception("응답 생성에 실패했습니다.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="응답 생성 실패"
        ) from e
