"""팀 공통 경로. 말뭉치와 골든셋의 위치와 이름을 이 파일에서만 정합니다.

원본 RFP가 들어 있는 data/ 는 .gitignore 로 막혀 있어 git 에 올라가지 않습니다.
팀 비밀방에서 받은 파일을 아래 경로에 두면 됩니다. (README.md 참고)
"""

from pathlib import (
    Path,  # 운영체제별 경로 구분자를 안전하게 처리하는 도구를 가져옵니다.
)

SHARED_DIR = (
    Path(__file__).resolve().parent
)  # 이 파일의 절대 경로에서 현재 폴더(ai/data_basic)를 구합니다.
PROJECT_ROOT = SHARED_DIR.parents[
    1
]  # 현재 폴더의 두 단계 위인 프로젝트 루트를 구합니다.
DATA_DIR = PROJECT_ROOT / "data"  # 프로젝트 루트 아래의 data 폴더를 가리킵니다.

DATA_LIST = (
    DATA_DIR / "data_list.csv"
)  # 운영진이 제공하는 원본 문서 메타데이터 파일 경로입니다.
RAW_FILES_DIR = DATA_DIR / "files"  # 원본 HWP/PDF 파일 폴더 경로입니다(있을 때만 사용).

CORPUS_NAME = "clean_mixed_ocr_v2"  # 팀에서 기준으로 삼는 말뭉치 폴더의 이름입니다(문서 98건과 메타 CSV).
CORPUS_DIR = DATA_DIR / CORPUS_NAME  # 기준 말뭉치 폴더의 전체 경로를 만듭니다.

GOLDEN_NAME = "rahui_golden_set_v2_2.json"  # 최종 기준 골든셋 파일 이름입니다(74문항).
GOLDEN_DIR = DATA_DIR / "golden"  # 골든셋 파일이 들어 있는 폴더의 경로입니다.
GOLDEN_PATH = GOLDEN_DIR / GOLDEN_NAME  # 골든셋 JSON 파일의 전체 경로를 만듭니다.

MANIFEST_PATH = (
    DATA_DIR / "manifest.json"
)  # 전달받은 데이터가 팀 기준과 같은지 비교하는 기준값 파일 경로입니다.

RESULTS_DIR = (
    DATA_DIR / "results"
)  # 실험·EDA 결과가 쌓이는 폴더입니다(깃에 올라가지 않는 data/ 아래).
GOLDEN_SET_PATH = GOLDEN_PATH  # 이식한 코드가 쓰는 이름(GOLDEN_PATH와 같은 파일)입니다.
