import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger

load_dotenv()
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def verify_key(api_key: Annotated[str, Security(api_key_header)]):
    """
    request의 header를 통해 API KEY 일치 여부를 확인합니다.
    미포함 시 422, 불일치 시 401 에러를 발생시킵니다.
    """
    server_key = os.getenv("X_API_KEY")
    # 인증 키 불일치
    if api_key != server_key:
        logger.warning("API KEY 불일치")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API KEY가 일치하지 않습니다.",
        )
