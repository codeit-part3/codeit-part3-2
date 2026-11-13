from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAHUI_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = RAHUI_DIR / "eda_results"
GOLDEN_SET_PATH = RAHUI_DIR / "rahui_golden_set.json"
