from fastapi import FastAPI

from backend.routes import chat

app = FastAPI()

app.include_router(chat.router)
