"""백엔드 설정

ENGINES_IDS는 app.py의 ENGINES 목록 id와 반드시 같아야 합니다.
엔진을 추가하거나 이름을 바꾸면 두 곳을 함께 수정해주세요.
"""

import importlib

ENGINE_IDS: tuple[str, ...] = ("temp1", "temp2", "temp3", "temp4", "temp5")


def is_valid_engine(engine_id: str) -> bool:
    if engine_id in ENGINE_IDS:
        try:
            return importlib.util.find_spec(f"engines.{engine_id}") is not None
        except ModuleNotFoundError:  # engines 폴더 자체가 아직 없는 경우
            return False
