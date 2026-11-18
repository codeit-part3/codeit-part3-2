from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routes import chat
from backend.services.engine_loader import load_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 graph warmup
    load_engine("local_model")
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)


# 서버 연결 확인
@app.get("/health")
def health():
    return {"server_status": "OK"}
