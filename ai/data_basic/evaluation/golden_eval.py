"""호환용 연결 파일: 실제 구현은 ai/data_basic/golden_eval.py 한 곳에만 둡니다."""

from ai.data_basic.golden_eval import *  # noqa: F401,F403
from ai.data_basic.golden_eval import main  # noqa: F401

if __name__ == "__main__":
    main()
