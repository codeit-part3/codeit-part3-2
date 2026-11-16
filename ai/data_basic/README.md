# ai/data_basic: 팀 공통 도구

말뭉치와 골든셋의 위치를 하나로 맞추고, 받은 파일이 같은지 확인하고, 골든셋을 점검·채점하는 코드입니다.
라희의 변환·OCR·전처리·EDA·평가·실험 코드를 **출발점(복사본)** 으로 함께 두었습니다.
팀원이 고치고 보태면서 이 폴더의 코드를 팀 공통 기준으로 키워 갑니다. 각자 실험은 `ai/<이름>/`에 둬도 됩니다.

## 폴더 구성

| 파일 | 하는 일 |
| --- | --- |
| `paths.py` | 말뭉치·골든셋 경로와 이름 (바꿀 때는 여기만) |
| `golden_eval.py` | 골든셋 점검(`python -m ai.data_basic.golden_eval`)과 답변 채점 함수 |
| `check_data.py` | 받은 파일이 팀 기준과 같은지 확인 |
| `GOLDEN_FORMAT.md` | 골든셋 필드 설명 |
| `conversion/` | HWP/PDF → 텍스트 변환 (`convert.py`, `hwp_olefile.py`, `convert_kordoc.py`) |
| `ocr/` | 그림 속 글자 OCR과 품질 정리 |
| `preprocessing/` | 말뭉치 전처리(`preprocess.py`)와 청킹용 색인 코드 |
| `analysis/` | 변환 커버리지·EDA 통계와 검증 |
| `experiments/` | 청킹·임베딩·재순위·Qdrant 비교 실험 |
| `evaluation/` | 답변 채점(`answer_eval.py`), LLM 판정(`judge_groq.py`) |

실행은 항상 프로젝트 루트에서 `uv run python -m ai.data_basic.<폴더>.<파일>` 형식입니다.
결과는 `data/results/`에 쌓이며 깃에 올라가지 않습니다.

## 데이터 받기 (NDA)

말뭉치와 골든셋에는 RFP 본문이 들어 있어서 **GitHub에 올리지 않습니다.** `data/`는 `.gitignore`로 막혀 있습니다.
**팀 비밀방**에서 받아 아래 위치에 두세요. 파일을 비밀방 밖, GitHub, 외부 AI 서비스에 올리지 않습니다.

```
data/
├─ clean_mixed_ocr_v2/             말뭉치 (문서 98개 + _meta.csv)
├─ golden/
│  └─ rahui_golden_set_v2_2.json   골든셋 (74문항, 최종 버전)
└─ manifest.json                   받은 파일이 같은지 비교하는 기준값 (비밀방에서 같이 받음)
```

> 이 폴더(`ai/data_basic/`)에는 **코드와 문서만** 둡니다. 데이터 파일은 넣지 마세요.
> `.gitignore`의 `data/` 규칙은 이름이 `data`인 폴더만 막아서, 이 폴더에 둔 파일은 커밋될 수 있습니다.

## 환경과 확인

```bash
uv sync
cp .env.example .env        # API 키는 .env 에만 적고 커밋하지 않습니다

uv run python -m ai.data_basic.check_data    # [일치] 가 나오면 같은 파일입니다
uv run python -m ai.data_basic.golden_eval   # 골든셋 형식 점검
```

`check_data`가 [불일치]를 알려 주면 말뭉치 폴더 이름과 압축을 푼 위치를 먼저 확인하세요.
출력은 개수와 지문값뿐이라 채팅에 붙여도 됩니다.

## 필요한 패키지

| 하는 일 | 추가 설치 | 해당 코드 |
| --- | --- | --- |
| 데이터 확인, 골든셋 점검, 청킹 비교(BM25) | 없음 (기본 패키지) | `check_data`, `golden_eval`, `experiments/chunk_compare` |
| 임베딩·재순위·답변 평가 | `torch`, `sentence-transformers` | `experiments/embed_compare`, `rerank_compare`, `evaluation/answer_eval` |
| Qdrant 실험 | `qdrant-client` (+ 위 두 개) | `experiments/qdrant_*` |
| 원본 변환·OCR (원본 RFP 파일이 있을 때만) | `pyhwp`, `six`, `pypdf`, `pillow` | `conversion/`, `ocr/` |

- 팀원은 변환된 말뭉치를 받아 쓰므로 변환·OCR 패키지는 필요 없습니다. 원본 RFP 파일은 NDA 때문에 공유하지 않습니다.
- `torch`는 PC의 GPU와 드라이버에 맞는 빌드를 직접 고르세요(CUDA 인덱스 설정은 공통 `pyproject.toml`에 넣지 않았습니다).
- 의존성 변경은 이 폴더와 별도 PR로 올립니다.

## 기준 데이터 요약

- 말뭉치 `clean_mixed_ocr_v2`: 변환 100건 중 본문이 같은 2쌍을 합친 98건.
  HWP는 자체 변환기, PDF 4건은 kordoc, 그림 속 글자는 OCR로 본문 뒤에 붙임.
- 골든셋 `rahui_golden_set_v2_2`: 74문항 (답 있음 61 + 답 없음 13). 구성은 `GOLDEN_FORMAT.md`.

## 다른 코드와의 관계

- 이 폴더의 코드는 `ai/rahui/`의 코드를 복사해 `ai.rahui` → `ai.data_basic`으로 import와 경로만 바꾼 것입니다
  (기준 커밋 `543c7d4`). 복사 뒤에는 이 폴더가 팀 공통 기준이고, `ai/rahui/`는 라희 개인 작업본으로 따로 갑니다.
- `evaluation/golden_eval.py`는 `golden_eval.py`를 다시 내보내는 호환용 파일입니다. 실제 구현은 `golden_eval.py` 한 곳입니다.
- `preprocessing/req_split.py`는 `ai/dh/req_split.py`의 `split_requirements` 부분을 출처를 밝혀 가져온 사본입니다(`ai/dh` 없이 동작).
- 팀 평가 스크립트는 `common/`과 루트의 `evaluate.py`를 따릅니다.
