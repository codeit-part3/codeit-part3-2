from typing import Annotated

from dotenv import get_key
from fastapi import Header, HTTPException, status
from loguru import logger

X_API_KEY = get_key(dotenv_path=".env", key_to_get="X_API_KEY")


def compare_key(x_api_key: Annotated[str, Header()]):
    """
    request의 header를 통해 API KEY 일치 여부를 확인합니다.
    미포함 시 422, 불일치 시 401 에러를 발생시킵니다.
    """
    # 인증 키 불일치
    print(f"x_api_key: {type(x_api_key)}")
    print(f"x_api_key: {x_api_key}")
    if x_api_key != X_API_KEY:
        logger.warning("API KEY 불일치")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API KEY가 일치하지 않습니다.",
        )
