# backend/services/engine_loader.py
def load_engine(engine_id: str):
    """
    ***현재는 테스트 함수를 반환합니다.***

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
        return f"[{engine_id} 엔진 · 테스트 응답] 받은 질문: {message}"

    return stub
