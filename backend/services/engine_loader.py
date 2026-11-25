# backend/services/engine_loader.py
import asyncio
import importlib

from fastapi import HTTPException, status
from loguru import logger

from backend.config import is_valid_engine

_engine_tasks: dict[str, asyncio.Task] = {}


async def get_engine(engine_id: str):
    module = importlib.import_module(f"engines.{engine_id}.baseline")
    await asyncio.to_thread(module.warmup)

    return module.get_graph()


async def load_engine(engine_id: str):
    """
    엔진 로딩 Task를 캐시합니다. 동시에 들어온 요청도 같은 Task를 기다립니다.
    로딩이 실패하면 캐시에서 제거하여 다음 요청에서 다시 시도하게 합니다.
    """
    if not is_valid_engine(engine_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="유효하지 않은 engine_id입니다.",
        )

    if engine_id not in _engine_tasks:
        _engine_tasks[engine_id] = asyncio.create_task(get_engine(engine_id))

    task = _engine_tasks[engine_id]

    try:
        return await task

    except Exception as e:
        # 실패한 Task는 제외하여 다음 요청에서 다시 로딩하게 합니다.
        if _engine_tasks.get(engine_id) is task:
            del _engine_tasks[engine_id]
        logger.exception(f"{engine_id} 엔진을 로딩하는 데 실패했습니다")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="엔진 로딩 실패"
        ) from e
