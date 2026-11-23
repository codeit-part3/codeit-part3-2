"""final 엔진 설정값 (openai_rag 기반 + 라희님 CORR, 동현님 수정을 스위치로 이식)

engines/local_model/config.py 와 같은 형식이고, 아래 '검색 방식' 항목이 추가되었다.
비밀 값(API 키 등)은 .env 에 둔다.
"""

import os


def _flag(name, default=False):
    """환경변수 FINAL_<name> 이 있으면 그 값(1/true/on), 없으면 기본값"""
    v = os.getenv(f"FINAL_{name}")
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "on", "yes")


# =====================================================================
# final 엔진 스위치 (기본값은 전부 꺼짐 = main 의 openai_rag 와 같은 동작 = 기준선)
# 파일을 고치지 않고 환경변수로도 바꿀 수 있다. 예) $env:FINAL_CORR_WON="1"
# =====================================================================
PROVIDER = os.getenv("FINAL_PROVIDER", "openai")  # "openai": 클라우드 / "ollama": 로컬
CORR_REFUSE = _flag(
    "CORR_REFUSE"
)  # 라희님 CORR: 입찰공고문 참조 거절, 비공개 거절, 성과품 오답 거절
CORR_DOC_FILTER = _flag(
    "CORR_DOC_FILTER"
)  # 라희님 CORR: 단일 사업 질문에서 top1 고신뢰면 타 사업 조각 제거
CORR_DOC_FILTER_TOP1 = 0.995  # 위 기능이 작동하는 리랭커 1등 점수 기준
CORR_WON = _flag("CORR_WON")  # 라희님 CORR: "천원" 금액 옆에 원 환산값 병기
DOCS_REFUSE = _flag(
    "DOCS_REFUSE"
)  # 동현님: 서류 질문인데 조각에 서류 항목이 없으면 LLM 없이 거절 (CORR_REFUSE 다음 순서)
REAL_WARMUP = _flag("REAL_WARMUP")  # 동현님: 실제 질문 하나로 전체 흐름 워밍업
ADAPTIVE_RULE = os.getenv(
    "FINAL_ADAPTIVE_RULE", "off"
)  # 로컬 전용 빠른 모델 규칙: "off" / "dh"(개수·합계 외 전부) / "corr"(top1 0.995 이상 + 단순 질문)
ADAPTIVE_THINK_TOP1 = 0.995  # ADAPTIVE_RULE="corr" 의 점수 기준

# =====================================================================
# 모델
# =====================================================================
LOCAL_CHAT_MODEL = os.getenv(
    "FINAL_LOCAL_MODEL", "gemma4:e2b"
)  # PROVIDER="ollama" 일 때 답변 모델
CHAT_MODEL = (
    os.getenv("OPENAI_RAG_MODEL", "gpt-5-nano")
    if PROVIDER == "openai"
    else LOCAL_CHAT_MODEL
)
REASONING_EFFORT = os.getenv(
    "OPENAI_RAG_EFFORT", "low"
)  # OpenAI 노력 정도 (low / high)
EMBED_MODEL = "embeddinggemma"
NUM_CTX = 8192
TEMPERATURE = 0
SEED = 42
THINK = True  # True: 생각 모드 켜기 (정확도·무환각 우선) / False: 끄기 (속도 우선)

# =====================================================================
# 조각과 검색
# =====================================================================
CHUNK_SIZE = 1500  # fixed: 조각 최대 글자 수 / struct: 목표 글자 수 (바꾸면 새 DB 생성)
CHUNK_OVERLAP = 100  # struct 는 쓰지 않지만 DB 폴더 이름에 들어간다
TOP_K = 5
EMBED_BATCH = 50

REBUILD_DB = False
EXACT_SEARCH = True  # 이 엔진은 정확한 검색만 지원한다 (False 로 두면 안 됨)

# =====================================================================
# 검색 방식 (rahui 엔진에서 추가된 설정)
# =====================================================================
CHUNKER = "struct"  # "fixed": 글자 수로 자르기 / "struct": 제목 단위, 표 보존, 기관|사업명|절 헤더
DB_TAG = "v2s"  # struct + clean_mixed_ocr_v2 로 이미 만든 DB(indexes/chroma_v2s_...) 를 재사용

AGENCY_FILTER = (
    False  # 질문에 발주 기관명이 있으면 그 기관의 조각만 후보로 남김 (없으면 필터 없이)
)
AGENCY_MIN_LEN = 3  # 이보다 짧은 기관명은 오탐이 많아 매칭에서 제외

RERANK = True  # 후보를 리랭커로 다시 순위 매김
RERANK_POOL = 30  # 리랭커에 넣을 후보 수
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 1024  # 리랭커가 질문+조각을 읽는 최대 토큰 수 (11/17 측정: 512->1024 정답률 +11.1%p)
RERANK_FP16 = True  # 리랭커를 fp16으로 올림 (GPU에서만 적용). 11/18 측정: 정답률 그대로, 응답시간 16초대 -> 3초 안팎 (8GB GPU 메모리 부족 해소)
RERANK_BATCH = 4  # 한 번에 리랭크하는 쌍 수 (작을수록 GPU 메모리를 덜 씀)
FOLLOWUP_REWRITE = True  # True면 후속질문에 이전 대화의 사업명/ 기관명 이어붙여 검색, 실험 전/후비교위함
RERANK_CUTOFF = (
    0.5  # 예) 0.5 -> 리랭커 1등 점수가 이 값보다 낮으면 LLM호출 없이 문서에 없다고 답변
)
ADAPTIVE_TOPK = True  # True면 1~3등 유지, 4~5등은 리랭커 점수 0.5 이상일 때만 포함
TYPO_FIX = (
    False  # True면 검색어에서 기관명·공공입찰 용어 오타를 사전 기반으로 자동 교정
)
ADAPTIVE_THINK = (
    PROVIDER == "ollama" and ADAPTIVE_RULE != "off"
)  # final: ADAPTIVE_RULE 로 정함 (클라우드는 항상 꺼짐). True면 확실한 단일문서 단순 조회만 think=False(초고속), 복잡/종합 질문은 think=True
MULTI_ENTITY_SEARCH = (
    True  # True면 질문에 사업/기관이 2개 이상일 때 각 사업별로 분리 검색해 합침
)
# =====================================================================
# 데이터 (팀 기준과 동일)
# =====================================================================
CORPUS = "clean_mixed_ocr_v2"
GOLDEN_FILE = "data/golden/rahui_golden_set_v2_2.json"
GOLDEN_EXCLUDE = {}

# =====================================================================
# 프롬프트 (prompts/ 폴더의 파일 이름, 확장자 제외)
# =====================================================================
RAG_PROMPT = "rag_v3"
AGENT_PROMPT = "agent_v1"
