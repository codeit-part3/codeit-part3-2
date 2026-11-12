from fastapi import FastAPI

from backend.routes import chat

app = FastAPI()

app.include_router(chat.router)

# 서버 연결 확인
app.get("/health")


def health():
    return {"server_status": "OK"}
