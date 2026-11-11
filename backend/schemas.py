# backend/schemas.py
# /chat 요청과 응답에 쓰이는 Pydantic 모델을 정의합니다.

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# 검증 전에 문자열의 앞뒤 공백을 제거합니다.
# 공백 문자열(' ')은 제거되고, min_length 검사에서 걸립니다.
SHARED_CONFIG = ConfigDict(str_strip_whitespace=True)


class ChatRequest(BaseModel):
    """FrontEnd에서 /chat으로 보내는 요청 본문"""

    model_config = SHARED_CONFIG

    engine_id: Annotated[
        str, Field(description="요청할 엔진의 ID(configs.ENGINE_IDS 중 하나)")
    ]
    session_id: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            description="프론트가 만든 uuid. 같은 값이면 같은 대화로 인식",
        ),
    ]
    message: Annotated[str, Field(..., min_length=1, description="사용자 질문")]


class ChatResponse(BaseModel):
    """서버가 /chat에서 프론트엔드로 돌려주는 응답"""

    model_config = SHARED_CONFIG

    engine_id: Annotated[str, Field(description="응답을 만든 엔진의 ID")]
    session_id: Annotated[
        str, Field(..., min_length=1, description="요청과 같은 session_id")
    ]
    answer: Annotated[str, Field(description="엔진이 만든 답변")]
    history_length: Annotated[
        int, Field(description="이번 요청 이전까지 저장된 대화 턴 수")
    ]
