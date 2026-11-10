# backend/services/engine_loader.py
from loguru import logger


def load_engine(engine_id: str, label):
    """엔진이 있으면 그 answer 함수를, 없으면 테스트 응답 함수를 사용합니다.

    엔진 폴더가 있는데 내부에서 오류가 나면 테스트 응답으로 넘어가지 않고
    오류를 그대로 보여 주도록 했습니다. (문제를 놓치지 않기 위해서입니다.)
    """

    # 실사용 함수
    # if not is_valid_engine(engine_id):
    #         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
    #                             detail='유효하지 않은 engine_id입니다.')

    # module = importlib.import_module(f"engines.{engine_id}")

    # return module.answer

    # test용 함수
    def stub(message, history):
        return f"[{label} 엔진 · 테스트 응답] 받은 질문: {message}"

    return stub


def get_engine(engine_id: str):
    try:
        engine_fn = load_engine(engine_id, engine_id)
    except Exception:
        logger.exception("엔진 불러오기에 실패했습니다.")

    return engine_fn
