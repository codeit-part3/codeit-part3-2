import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from backend.routes import chat
from backend.services.engine_loader import load_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 graph warmup
    logger.info("[엔진 로딩 시작]")
    await asyncio.gather(
        load_engine("local_model"),  # local_final로 교체 예정
        load_engine("final"),
    )
    logger.info("[엔진 로딩 완료]")
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)


# 서버 연결 확인
@app.get("/health")
def health():
    return {"server_status": "OK"}
