"""final 엔진: openai_rag 를 복사해 라희님 CORR 와 동현님 수정을 config 스위치로 이식 (#43)

스위치(config.py, 기본 전부 꺼짐 = openai_rag 와 같은 동작):
  PROVIDER        "openai"(클라우드) / "ollama"(로컬)
  CORR_REFUSE     라희님: 입찰공고문 참조 거절, 비공개 거절, 성과품 오답 거절
  CORR_DOC_FILTER 라희님: 단일 사업 질문에서 top1 고신뢰면 타 사업 조각 제거
  CORR_WON        라희님: 천원 금액에 원 환산값 병기
  DOCS_REFUSE     동현님: 서류 질문인데 조각에 서류 항목이 없으면 LLM 없이 거절
                  (CORR_REFUSE 와 함께 켜면 CORR 를 먼저 적용: 입찰공고문 안내가 더 구체적이라서)
  REAL_WARMUP     동현님: 실제 질문 하나로 전체 흐름 워밍업
  ADAPTIVE_RULE   로컬 전용 빠른 모델 규칙 "off" / "dh" / "corr"

원본 설명 (rahui 엔진: 구조 청킹 + 기관명 필터 + 리랭커 (rag + agent))

engines/local_model/baseline.py 와 같은 이름의 함수·변수를 가져서 evaluate.py 가 그대로 쓸 수 있다.
다른 점은 세 가지다.
  1) 청킹  : config.CHUNKER  "fixed"(글자 수로 자르기) / "struct"(제목 단위, 표 보존, 기관|사업명|절 헤더)
  2) 앞단  : config.AGENCY_FILTER 가 True 면 질문에 든 발주 기관의 조각만 후보로 남김
  3) 뒷단  : config.RERANK 가 True 면 후보 RERANK_POOL 개를 리랭커로 다시 순위 매김
검색 결과 [(조각, 점수)] 의 점수는 '작을수록 좋음'이다. (리랭커를 쓰면 관련도에 -1 을 곱한 값)
agent 방식(make_agent)도 이 엔진의 search() 를 도구로 쓰고, 컷오프(고정 거절 문구), 동적 think,
후속질문 재작성, 오타교정을 rag 와 같이 적용한다. 최종 채택은 rag 그래프(answer)이고, agent 는 비교 실험용이다.

설정값은 config.py, 프롬프트는 prompts/ 에서 바꾼다.
engines/<id>/ 로 옮길 때 바꿀 곳: 아래 config import 한 줄, struct 청킹 import(chunk_struct) 한 줄.
"""

import atexit
import os
import re
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from engines.local_final import config

# 직접 실행(python -m ...baseline)할 때 시험하는 질문 (평가는 evaluate.py / run_eval.py 가 골든셋으로 함)
QUESTIONS = [
    "한영대학교 트랙운영 학사정보시스템 고도화 사업의 사업 예산은 얼마인가요?",
    "한영대학교 트랙운영 학사정보시스템 고도화 사업의 사업 기간은 어떻게 되나요?",
    "한영대학교 트랙운영 학사정보시스템 고도화 사업의 입찰 방법은 무엇인가요?",
]

# =====================================================================
# 설정값 (값은 config.py 에서 바꿈. 아래는 실행 중 configure()/set_search() 로 바뀔 수 있는 사본)
# =====================================================================
CHAT_MODEL = config.CHAT_MODEL
EMBED_MODEL = config.EMBED_MODEL
NUM_CTX = config.NUM_CTX
TEMPERATURE = config.TEMPERATURE
SEED = config.SEED
THINK = getattr(config, "THINK", False)
REASONING_EFFORT = getattr(config, "REASONING_EFFORT", "medium")
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
TOP_K = config.TOP_K
EMBED_BATCH = config.EMBED_BATCH

REBUILD_DB = config.REBUILD_DB
EXACT_SEARCH = config.EXACT_SEARCH

CHUNKER = config.CHUNKER
AGENCY_FILTER = config.AGENCY_FILTER
AGENCY_MIN_LEN = config.AGENCY_MIN_LEN
RERANK = config.RERANK
RERANK_POOL = config.RERANK_POOL
RERANK_MODEL = config.RERANK_MODEL
RERANK_MAX_LENGTH = config.RERANK_MAX_LENGTH
RERANK_FP16 = getattr(config, "RERANK_FP16", False)
RERANK_BATCH = config.RERANK_BATCH
FOLLOWUP_REWRITE = getattr(config, "FOLLOWUP_REWRITE", False)
RERANK_CUTOFF = getattr(config, "RERANK_CUTOFF", None)
ADAPTIVE_TOPK = getattr(config, "ADAPTIVE_TOPK", False)
TYPO_FIX = getattr(config, "TYPO_FIX", False)
RAG_PROMPT = config.RAG_PROMPT
AGENT_PROMPT = getattr(config, "AGENT_PROMPT", "agent_v1")
ADAPTIVE_THINK = getattr(config, "ADAPTIVE_THINK", False)
MULTI_ENTITY_SEARCH = getattr(config, "MULTI_ENTITY_SEARCH", False)

# final 스위치
PROVIDER = getattr(config, "PROVIDER", "openai")
CORR_REFUSE = getattr(config, "CORR_REFUSE", False)
CORR_DOC_FILTER = getattr(config, "CORR_DOC_FILTER", False)
CORR_DOC_FILTER_TOP1 = getattr(config, "CORR_DOC_FILTER_TOP1", 0.995)
CORR_WON = getattr(config, "CORR_WON", False)
DOCS_REFUSE = getattr(config, "DOCS_REFUSE", False)
REAL_WARMUP = getattr(config, "REAL_WARMUP", False)
ADAPTIVE_RULE = getattr(config, "ADAPTIVE_RULE", "off")
ADAPTIVE_THINK_TOP1 = getattr(config, "ADAPTIVE_THINK_TOP1", 0.995)


# =====================================================================
# 경로
# =====================================================================
def _find_root():
    """pyproject.toml 이 있는 폴더 = 프로젝트 폴더 (이 파일을 어디로 옮겨도 동작)"""
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("프로젝트 폴더(pyproject.toml)를 찾지 못했습니다.")


BASE_DIR = _find_root()
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
load_dotenv(BASE_DIR / ".env")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"

DATA_PATH = BASE_DIR / "data" / "data_list.csv"
CORPUS = config.CORPUS
TXT_DIR = BASE_DIR / "data" / CORPUS
GOLDEN_PATH = BASE_DIR / config.GOLDEN_FILE
GOLDEN_EXCLUDE = config.GOLDEN_EXCLUDE
OUTPUT_DIR = BASE_DIR / "outputs"


def safe_name(text):
    """폴더 이름에 쓸 수 없는 글자를 - 로 바꿈 (글자, 숫자, _, - 는 그대로)"""
    return "".join(c if c.isalnum() or c in "_-" else "-" for c in text)


def db_dir_for(corpus, embed_model, chunk_size, chunk_overlap):
    """설정마다 벡터 DB 폴더를 따로 둠. struct 청킹의 기존 DB(v2s)는 같은 말뭉치일 때만 재사용"""
    if CHUNKER == "struct":
        tag = (
            config.DB_TAG if corpus == config.CORPUS else f"{safe_name(corpus)}-struct"
        )
    else:
        tag = safe_name(corpus)
    name = f"chroma_{tag}_{safe_name(embed_model)}_{chunk_size}_{chunk_overlap}"
    return BASE_DIR / "indexes" / name


DB_DIR = db_dir_for(CORPUS, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)
DONE_FILE = DB_DIR / "build_done.txt"


def load_prompt(name):
    """prompts/<name>.txt 내용을 읽음"""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.is_file():
        names = ", ".join(sorted(p.stem for p in PROMPT_DIR.glob("*.txt")))
        raise FileNotFoundError(
            f"프롬프트 파일이 없습니다: {path.name} (사용 가능: {names})"
        )
    return path.read_text(encoding="utf-8").strip()


def configure(
    chunk_size=None,
    chunk_overlap=None,
    top_k=None,
    corpus=None,
    golden=None,
    prompt=None,
):
    """실험 설정 바꾸기 (팀 baseline.configure 와 같은 인자). 모델, DB 를 처음 쓰기 전에 호출"""
    global CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, CORPUS, TXT_DIR, DB_DIR, DONE_FILE, _graph
    global GOLDEN_PATH, RAG_PROMPT, AGENT_PROMPT
    if prompt is not None:
        load_prompt(prompt)  # 파일이 있는지 바로 확인
        if prompt.startswith("rag_"):
            RAG_PROMPT = prompt
        elif prompt.startswith("agent_"):
            AGENT_PROMPT = prompt
        else:
            raise ValueError(f"rag_ 또는 agent_ 프롬프트만 씁니다: {prompt}")
    if golden is not None:
        GOLDEN_PATH = BASE_DIR / golden
    if corpus is not None:
        if not (BASE_DIR / "data" / corpus).is_dir():
            raise FileNotFoundError(f"말뭉치 폴더가 없습니다: data/{corpus}")
        CORPUS = corpus
        TXT_DIR = BASE_DIR / "data" / CORPUS
    if chunk_size is not None:
        CHUNK_SIZE = chunk_size
    if chunk_overlap is not None:
        CHUNK_OVERLAP = chunk_overlap
    if top_k is not None:
        TOP_K = top_k
    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(f"조각 설정 오류: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    DB_DIR = db_dir_for(CORPUS, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)
    DONE_FILE = DB_DIR / "build_done.txt"
    _resources.clear()
    _graph = None


def set_search(
    agency_filter=None,
    rerank=None,
    pool=None,
    max_length=None,
    batch=None,
    chunker=None,
    fp16=None,
    followup_rewrite=None,
    cutoff=None,
    multi_entity=None,
):
    """검색 방식 바꾸기 (rahui 엔진 전용). configure() 보다 먼저 또는 그 뒤에 호출해도 된다."""
    global \
        AGENCY_FILTER, \
        RERANK, \
        RERANK_POOL, \
        RERANK_MAX_LENGTH, \
        RERANK_BATCH, \
        RERANK_FP16, \
        FOLLOWUP_REWRITE
    global CHUNKER, DB_DIR, DONE_FILE, _graph, RERANK_CUTOFF, MULTI_ENTITY_SEARCH
    if chunker not in (None, "fixed", "struct"):
        raise ValueError(f"chunker 는 fixed 또는 struct: {chunker}")
    if agency_filter is not None:
        AGENCY_FILTER = agency_filter
    if rerank is not None:
        RERANK = rerank
    if pool is not None:
        RERANK_POOL = pool
    if max_length is not None:
        RERANK_MAX_LENGTH = max_length
    if batch is not None:
        RERANK_BATCH = batch
    if fp16 is not None:
        RERANK_FP16 = fp16
    if chunker is not None:
        CHUNKER = chunker
    if followup_rewrite is not None:
        FOLLOWUP_REWRITE = followup_rewrite
    if cutoff is not None:
        RERANK_CUTOFF = cutoff
    if multi_entity is not None:
        MULTI_ENTITY_SEARCH = multi_entity
    DB_DIR = db_dir_for(CORPUS, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)
    DONE_FILE = DB_DIR / "build_done.txt"
    _resources.clear()
    _graph = None


def final_switches():
    """켜진 final 스위치를 한 줄로 (결과 파일, 시트 기록용). 다 꺼져 있으면 "base" """
    on = [
        name
        for name, val in [
            ("CORR_REFUSE", CORR_REFUSE),
            ("CORR_DOC_FILTER", CORR_DOC_FILTER),
            ("CORR_WON", CORR_WON),
            ("DOCS_REFUSE", DOCS_REFUSE),
            ("REAL_WARMUP", REAL_WARMUP),
            ("TYPO_FIX", TYPO_FIX == "query_fix"),
        ]
        if val
    ]
    if PROVIDER == "ollama" and ADAPTIVE_RULE != "off":
        on.append(f"ADAPTIVE_{ADAPTIVE_RULE}")
    return "+".join(on) or "base"


def current_settings(engine=None):
    """지금 쓰는 설정값 (결과 파일, 시트 기록용)"""
    settings = {
        "chat_model": CHAT_MODEL,
        "corpus": CORPUS,
        "golden": GOLDEN_PATH.stem,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "num_ctx": NUM_CTX if PROVIDER == "ollama" else "n/a",
        "provider": PROVIDER,
        "reasoning_effort": REASONING_EFFORT if PROVIDER == "openai" else "n/a",
        "search": "exact",
        "chunker": CHUNKER,
        "agency_filter": AGENCY_FILTER,
        "rerank": f"{RERANK_MODEL.split('/')[-1]}@{RERANK_POOL}" if RERANK else False,
        "followup_rewrite": FOLLOWUP_REWRITE,
        "rerank_cutoff": RERANK_CUTOFF,
        "think": THINK,
        "adaptive_topk": ADAPTIVE_TOPK,
        "typo_fix": TYPO_FIX,
        "adaptive_think": ADAPTIVE_THINK,
        "multi_entity_search": MULTI_ENTITY_SEARCH,
        "final_switches": final_switches(),
    }
    if engine == "rag":
        settings["prompt"] = RAG_PROMPT
    elif engine == "agent":
        settings["prompt"] = AGENT_PROMPT
    return settings


def normalize(name):
    """윈도우와 파일 이름의 한글 표기 차이 설정"""
    return unicodedata.normalize("NFC", name)


# =====================================================================
# 1. 문서 불러오기 (CSV 한 행 + 같은 이름의 txt 파일)
# =====================================================================
def load_documents():
    df = pd.read_csv(DATA_PATH)
    txt_files = {normalize(p.name): p for p in TXT_DIR.glob("*.txt")}

    documents = []
    missing = []
    for _, row in df.iterrows():
        txt_name = normalize(Path(str(row["파일명"])).stem + ".txt")
        txt_path = txt_files.get(txt_name)
        if txt_path is None:
            missing.append(str(row["파일명"]))
            continue

        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            missing.append(str(row["파일명"]))
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "공고 번호": str(row["공고 번호"]),
                    "사업명": str(row["사업명"]),
                    "발주 기관": str(row["발주 기관"]),
                },
            )
        )

    if missing:
        print(f"주의: txt 를 못 찾았거나 비어 있는 문서 {len(missing)}개 (제외됨)")
        for name in missing[:5]:
            print("  -", name)
    return documents


# =====================================================================
# 2. 벡터 DB 만들기 (자르기 -> 임베딩 -> Chroma 저장)
# =====================================================================
def make_store(embedding_model):
    return Chroma(
        collection_name="baseline_v1",
        embedding_function=embedding_model,
        persist_directory=str(DB_DIR),
    )


def split_documents(documents):
    """CHUNKER 에 따라 자르기. 메타데이터는 조각에도 유지된다."""
    if CHUNKER == "struct":
        # 제목 단위로 자르고 표는 쪼개지 않음. 조각 맨 앞에 [기관 | 사업명 | 절] 헤더를 붙임
        from ai.data_basic.experiments.chunk_compare import chunk_struct

        chunks = []
        for doc in documents:
            org = doc.metadata["발주 기관"]
            title = doc.metadata["사업명"]
            for sec, body in chunk_struct(doc.page_content, CHUNK_SIZE):
                head = f"[{org} | {title}" + (f" | {sec}" if sec else "") + "]\n"
                chunks.append(
                    Document(page_content=head + body, metadata=dict(doc.metadata))
                )
        return chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    return splitter.split_documents(documents)


def build_vector_store():
    embedding_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    vector_store = make_store(embedding_model)

    if DONE_FILE.exists() and not REBUILD_DB:
        print(f"저장된 벡터 DB를 사용합니다: {DB_DIR.name}")
        return vector_store

    if not TXT_DIR.is_dir():
        raise FileNotFoundError(f"말뭉치 폴더가 없습니다: data/{CORPUS}")

    vector_store.delete_collection()
    DONE_FILE.unlink(missing_ok=True)
    vector_store = make_store(embedding_model)

    documents = load_documents()
    chunks = split_documents(documents)
    print(
        f"문서 {len(documents)}개 -> 조각 {len(chunks)}개를 임베딩합니다. (몇 분 걸림)"
    )

    start = time.time()
    for i in range(0, len(chunks), EMBED_BATCH):
        vector_store.add_documents(documents=chunks[i : i + EMBED_BATCH])
        done = min(i + EMBED_BATCH, len(chunks))
        if done % (EMBED_BATCH * 10) == 0 or done == len(chunks):
            print(f"  {done} / {len(chunks)} ({time.time() - start:.0f}초 경과)")

    DONE_FILE.write_text(f"chunks={len(chunks)}\n", encoding="utf-8")
    print("벡터 DB 저장 완료")
    return vector_store


# =====================================================================
# 3. 공용 도구
# =====================================================================
def make_chat_model(reasoning=None):
    if PROVIDER == "ollama":
        # 로컬 답변 모델 (rerank_rag 원본과 같은 설정)
        return init_chat_model(
            CHAT_MODEL,
            model_provider="ollama",
            base_url=OLLAMA_URL,
            temperature=TEMPERATURE,
            seed=SEED,
            num_ctx=NUM_CTX,
            reasoning=THINK if reasoning is None else reasoning,
        )
    # OpenAI 답변 모델. reasoning 인자는 원본과 호환용으로 받기만 하고,
    # 노력 정도는 모든 호출에 REASONING_EFFORT 하나로 같게 적용한다.
    return init_chat_model(
        CHAT_MODEL,
        model_provider="openai",
        reasoning_effort=REASONING_EFFORT,
    )


USAGE = {"calls": 0, "input": 0, "output": 0, "reasoning": 0}  # 답변 모델 토큰 누적


def _report_usage():
    """실행이 끝나면 답변 모델 토큰과 추정 비용을 출력하고 token_usage.csv에 기록"""
    if not USAGE["calls"] or PROVIDER != "openai":
        return
    avg = USAGE["reasoning"] / USAGE["calls"]
    print(f"답변 모델 생각 토큰: {USAGE['reasoning']:,} (호출당 평균 {avg:.0f})")
    try:
        from judge import report_usage

        report_usage(
            "openai", CHAT_MODEL, Path(f"answer_{CHAT_MODEL}_{REASONING_EFFORT}"), USAGE
        )
    except ImportError:
        print(f"답변 토큰 사용: {USAGE}")


atexit.register(_report_usage)


_resources = {}  # 모델, 벡터 DB, 검색 색인, 리랭커를 한 번만 만들어 재사용하기 위한 보관함


def get_resources():
    if not _resources:
        _resources["model"] = make_chat_model()
        _resources["vector_store"] = build_vector_store()
    return _resources["model"], _resources["vector_store"]


def get_fast_model():
    if "model_fast" not in _resources:
        _resources["model_fast"] = make_chat_model(reasoning=False)
    return _resources["model_fast"]


def get_reranker():
    """리랭커(CrossEncoder)를 한 번만 불러와 재사용"""
    if "reranker" not in _resources:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"리랭커 불러오는 중: {RERANK_MODEL} (장치: {device})", flush=True)
        ce = CrossEncoder(RERANK_MODEL, device=device, max_length=RERANK_MAX_LENGTH)
        if RERANK_FP16 and device == "cuda":
            ce.model.half()
            print("리랭커 fp16 적용", flush=True)
        _resources["reranker"] = ce
    return _resources["reranker"]


def warmup():
    """모델, DB, 리랭커를 미리 불러오고 한 번 호출 (첫 문항 응답시간에 로딩 시간이 섞이지 않게)"""
    model, vector_store = get_resources()
    get_exact_index()
    vector_store.embeddings.embed_query("워밍업")
    if RERANK:
        get_reranker().predict([("워밍업", "워밍업")], batch_size=1)
    model.invoke("안녕")
    if ADAPTIVE_THINK and THINK:  # 로컬 빠른 모델도 미리 불러옴 (CORR 와 같음)
        get_fast_model().invoke("안녕")
    if REAL_WARMUP:
        # 동현님: 실제 질문 하나로 전체 흐름 실행 (사업명 사전, 임베딩, 리랭크, 긴 입력까지 미리 거침)
        saved = dict(USAGE)
        get_graph().invoke({"question": QUESTIONS[0], "history": []})
        SEARCH_TIMES.clear()  # 워밍업 검색 시간은 기록에서 제외
        USAGE.update(saved)  # 워밍업 토큰은 비용 집계에서 제외


def get_exact_index():
    """저장된 조각 임베딩과 내용을 한 번만 메모리에 올려 재사용"""
    if "exact" not in _resources:
        _, vector_store = get_resources()
        data = vector_store._collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        matrix = np.asarray(data["embeddings"], dtype=np.float32)
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"], strict=True)
        ]
        _resources["exact"] = (matrix, docs)
    return _resources["exact"]


# --- 기관명 필터 ---------------------------------------------------------
def norm_agency(s):
    """매칭용 이름: 법인 유형 표기 제거 + 공백·괄호·기호 제거 (원본은 따로 보존)"""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(
        r"\((사|재|주|유|합|학|의)\)|㈜|주식회사|유한회사|사단법인|재단법인", "", s
    )
    return re.sub(r"[\s()\[\]·ㆍ_\-,.]", "", s)


def build_agency_index(docs):
    """정규화한 기관명 -> 그 기관 조각의 위치 목록"""
    idx = {}
    for i, d in enumerate(docs):
        key = norm_agency(d.metadata.get("발주 기관", ""))
        if len(key) >= AGENCY_MIN_LEN:
            idx.setdefault(key, []).append(i)
    return idx


def find_agencies(question, agency_idx):
    """질문에 들어 있는 기관명들 (다른 매칭 이름의 일부인 짧은 것은 버림)"""
    q = norm_agency(question)
    hits = [k for k in agency_idx if k in q]
    return [k for k in hits if not any(k != o and k in o for o in hits)]


def get_agency_index():
    if "agency" not in _resources:
        _, docs = get_exact_index()
        _resources["agency"] = build_agency_index(docs)
    return _resources["agency"]


# --- 검색 ----------------------------------------------------------------
SEARCH_TIMES = []


def search(query, k=None):
    """질문과 관련 있는 조각 k 개를 [(조각, 점수), ...] 로 반환 (점수가 작을수록 좋음)

    1) 모든 조각과 임베딩 거리를 계산  2) AGENCY_FILTER: 질문에 든 기관의 조각만 후보로
    3) 거리순 상위 후보 추리기  4) RERANK: 후보를 리랭커로 다시 순위 매김
    """
    start = time.time()
    k = k or TOP_K
    _, vector_store = get_resources()
    matrix, docs = get_exact_index()
    q = np.asarray(vector_store.embeddings.embed_query(query), dtype=np.float32)
    dist = ((matrix - q) ** 2).sum(axis=1)

    sel = np.arange(len(docs))
    if AGENCY_FILTER:
        agency_idx = get_agency_index()
        found = find_agencies(query, agency_idx)
        if found:  # 기관명을 못 찾으면 필터 없이 (잘못된 필터 방지)
            sel = np.concatenate([agency_idx[a] for a in found])

    n = max(RERANK_POOL, k) if RERANK else k
    order = sel[np.argsort(dist[sel], kind="stable")[:n]]
    cand = [(docs[i], float(dist[i])) for i in order]
    if not RERANK:
        SEARCH_TIMES.append(time.time() - start)
        return cand[:k]

    scores = get_reranker().predict(
        [(query, d.page_content) for d, _ in cand], batch_size=RERANK_BATCH
    )
    ranked = np.argsort(-np.asarray(scores), kind="stable")
    out = [(cand[j][0], float(-scores[j])) for j in ranked][:k]
    if ADAPTIVE_TOPK and len(out) > 3:
        out = out[:3] + [x for x in out[3:] if -x[1] >= 0.5]
    SEARCH_TIMES.append(time.time() - start)
    return out


def history_to_messages(history):
    """앞 대화 기록(role/content 목록)을 LangChain 메시지로 변경"""
    messages = []
    for turn in history or []:
        if turn["role"] == "user":
            messages.append(HumanMessage(turn["content"]))
        else:
            messages.append(AIMessage(turn["content"]))
    return messages


# =====================================================================
# 4. rag: LangGraph 구조
# =====================================================================
class RAGState(TypedDict, total=False):
    question: str
    history: list
    retrieved: list
    answer: str
    prompt_tokens: int


def _extract_entities(text):
    """문장에서 발주 기관명·사업명을 찾아 (기관 목록, (기관, 사업명) 목록)으로 반환"""
    _, docs = get_exact_index()
    q = norm_agency(text)
    if not q:
        return [], []

    if "entities" not in _resources:
        seen = set()
        orgs, titles = {}, {}
        for d in docs:
            org = str(d.metadata.get("발주 기관", "")).strip()
            title = str(d.metadata.get("사업명", "")).strip()
            if (org, title) in seen:
                continue
            seen.add((org, title))
            ok = norm_agency(org)
            tk = norm_agency(title)
            if len(ok) >= AGENCY_MIN_LEN:
                orgs.setdefault(ok, org)
            if len(tk) >= 4:
                titles.setdefault(tk, (org, title))
        _resources["entities"] = (orgs, titles)

    orgs, titles = _resources["entities"]
    hit_titles = [titles[k] for k in sorted(titles, key=len, reverse=True) if k in q]
    hit_orgs = [orgs[k] for k in sorted(orgs, key=len, reverse=True) if k in q]
    return hit_orgs, hit_titles


_RFP_TERMS = (
    "사업예산",
    "배정예산",
    "추정가격",
    "사업기간",
    "계약기간",
    "입찰방법",
    "제안요청서",
    "제안서",
    "평가기준",
    "요구사항",
    "제출서류",
    "구비서류",
    "발주기관",
    "유지보수",
    "하자보수",
    "공동수급",
    "하도급",
    "성과품",
)
_JOSA = re.compile(
    r"(에서|으로|까지|부터|은|는|이|가|을|를|의|에|로|과|와|도|만|[?!.,]+)+$"
)


_query_fixer = None


def get_query_fixer():
    """[#38 이식] engines/local_final/query_fix.py 의 교정기를 한 번만 만들어 재사용 (rerank_rag 와 같은 코드)"""
    global _query_fixer
    if _query_fixer is None:
        from engines.local_final.query_fix import QueryFixer

        _query_fixer = QueryFixer(data_list=DATA_PATH, corpus_dir=TXT_DIR)
    return _query_fixer


def fix_typos(question):
    """검색어에서 발주 기관명·사업 핵심어·공공입찰 용어의 한글 자모 오타를 교정"""
    if not TYPO_FIX:
        return question
    if TYPO_FIX == "query_fix":
        return get_query_fixer().fix(question)
    if "typo_vocab" not in _resources:
        _, docs = get_exact_index()
        vocab = set(_RFP_TERMS)
        for d in docs:
            org = norm_agency(d.metadata.get("발주 기관", ""))
            if len(org) >= AGENCY_MIN_LEN:
                vocab.add(org)
            for w in re.findall(
                r"[가-힣A-Za-z0-9]{3,}", str(d.metadata.get("사업명", ""))
            ):
                vocab.add(w)
        _resources["typo_vocab"] = {w: unicodedata.normalize("NFD", w) for w in vocab}

    vocab = _resources["typo_vocab"]
    fixed = []
    for tok in question.split():
        m = _JOSA.search(tok)
        stem, tail = (tok[: m.start()], m.group(0)) if m else (tok, "")
        if len(stem) < 3 or stem in vocab:
            fixed.append(tok)
            continue
        sn = unicodedata.normalize("NFD", stem)
        best_w, best_r = None, 0.85  # 자모 유사도 0.85 이상일 때만 보수적 교정
        for w, wn in vocab.items():
            if abs(len(w) - len(stem)) <= 1:
                r = SequenceMatcher(None, sn, wn).ratio()
                if r > best_r:
                    best_w, best_r = w, r
        fixed.append((best_w + tail) if best_w else tok)
    return " ".join(fixed)


def rewrite_followup(question, history):
    """후속질문이면 이전 대화에서 가장 최근 사업명/기관명을 찾아 검색어 앞에 붙임"""
    if not FOLLOWUP_REWRITE or not history:
        return question

    # 현재 질문에 이미 기관명이나 사업명이 있으면 '주제 전환'으로 보고 원문 유지
    cur_orgs, cur_titles = _extract_entities(question)
    if cur_orgs or cur_titles:
        return question

    # 이전 대화를 최근 순서대로 보며 가장 마지막에 언급된 사업명/기관명 찾기
    for turn in reversed(history):
        orgs, titles = _extract_entities(turn.get("content", ""))
        if titles:
            org, title = titles[0]
            return f"{org} {title} {question}"
        # #59: 답변(assistant)에만 나온 기관명은 질문 주제와 다를 수 있어(예: 총괄 부처) 건너뜀
        if orgs and turn.get("role") == "user":
            return f"{orgs[0]} {question}"

    # #59: 사업명을 정확히 못 찾으면(일부만 쓴 경우 등) 앞 사용자 질문을 검색어 앞에 붙임.
    # "금액은?"처럼 짧은 후속 질문은 건너뛰고, 주제가 담긴 질문(공백 빼고 12자 이상)을 거슬러 올라가 찾음.
    # 그런 질문이 없으면 바로 앞 사용자 질문을 붙임.
    user_qs = [t.get("content", "") for t in history if t.get("role") == "user"]
    anchor = next((q for q in reversed(user_qs) if len(re.sub(r"\s", "", q)) >= 12), "")
    prev_q = anchor or (user_qs[-1] if user_qs else "")
    if prev_q:
        return f"{prev_q} {question}"
    return question


def multi_entity_search(query):
    """질문에 서로 다른 사업/기관이 2개 이상 있으면 각 사업 조각 안에서만 따로 검색해 합친다."""
    if not MULTI_ENTITY_SEARCH:
        return search(query)

    orgs, titles = _extract_entities(query)
    targets, seen_orgs = [], set()
    for org, title in titles:
        targets.append((org, title))
        seen_orgs.add(norm_agency(org))
    for org in orgs:
        if norm_agency(org) not in seen_orgs:
            targets.append((org, ""))
            seen_orgs.add(norm_agency(org))

    # 사업/기관이 1개 이하면 기존 단일 검색 경로 그대로 (단일문서·후속질문 회귀 0%)
    if len(targets) < 2:
        out = search(query)
        # 라희님 CORR: 명시된 사업/기관 1개 + top1 고신뢰면, 그 사업 조각이 2개 이상일 때 타 사업 조각 제거
        if (
            CORR_DOC_FILTER
            and len(targets) == 1
            and out
            and -out[0][1] >= CORR_DOC_FILTER_TOP1
        ):
            org, title = targets[0]
            ok = norm_agency(org)
            matched = [
                x
                for x in out
                if (title and str(x[0].metadata.get("사업명", "")).strip() == title)
                or (not title and norm_agency(x[0].metadata.get("발주 기관", "")) == ok)
            ]
            if len(matched) >= 2:
                return matched
        return out

    start = time.time()
    _, vector_store = get_resources()
    matrix, docs = get_exact_index()
    q = np.asarray(vector_store.embeddings.embed_query(query), dtype=np.float32)
    dist = ((matrix - q) ** 2).sum(axis=1)

    per_k = 3 if len(targets) == 2 else 2
    merged = {}
    for org, title in targets[:3]:
        if title:
            idxs = [
                i
                for i, d in enumerate(docs)
                if str(d.metadata.get("사업명", "")).strip() == title
            ]
        else:
            ok = norm_agency(org)
            idxs = [
                i
                for i, d in enumerate(docs)
                if norm_agency(d.metadata.get("발주 기관", "")) == ok
            ]
        if not idxs:
            continue
        sel = np.asarray(idxs, dtype=np.int64)
        n = min(
            len(sel), 10 if RERANK else per_k
        )  # 사업당 최대 10개만 리랭크 (속도 3배 단축)
        order = sel[np.argsort(dist[sel], kind="stable")[:n]]
        cand = [(docs[i], float(dist[i])) for i in order]
        if RERANK:
            scores = get_reranker().predict(
                [(query, d.page_content) for d, _ in cand], batch_size=RERANK_BATCH
            )
            ranked = np.argsort(-np.asarray(scores), kind="stable")
            picked = [(cand[j][0], float(-scores[j])) for j in ranked][:per_k]
        else:
            picked = cand[:per_k]
        for doc, score in picked:
            key = (doc.metadata.get("공고 번호", ""), doc.page_content)
            if key not in merged or score < merged[key][1]:
                merged[key] = (doc, score)

    if not merged:
        return search(query)
    SEARCH_TIMES.append(time.time() - start)
    return sorted(merged.values(), key=lambda x: x[1])[:6]


def retrieve(state: RAGState):
    query = fix_typos(state["question"])
    query = rewrite_followup(query, state.get("history"))
    return {"retrieved": multi_entity_search(query)}


_COMPLEX_Q = re.compile(
    r"비교|차이|모두|전부|모든|몇\s*개|합계|총|가장|제일|제출서류|구비서류|제출문서|목록"
)
_REFUSE_MSG = "제공된 공고 문서에 해당 정보가 없어 답변할 수 없습니다."
_REFUSE_HEAD = re.compile(
    r"(확인할\s*수\s*없|찾을\s*수\s*없|정보가\s*없|포함되지\s*않|포함되어\s*있지\s*않|입찰공고문[”\"'\s]*에\s*(따릅|의함))"
)
# 라희님 CORR (CORR_REFUSE)
_REFUSE_NOTICE_MSG = "제공된 공고 문서에는 구체적인 목록 정보가 없어 답변할 수 없으며, 입찰공고문을 확인해야 합니다."
_REFUSE_HEAD_CORR = re.compile(
    r"(확인할\s*수\s*없|찾을\s*수\s*없|정보가\s*없|포함되지\s*않|포함되어\s*있지\s*않|비공개|공개하지\s*않)"
)
_REFUSE_NOTICE_HEAD = re.compile(r"입찰공고문[”\"'\s]*에\s*(따름|따릅|의함)")
_SUBMISSION_Q = re.compile(r"제출서류|구비서류|등록서류")
_BID_DOC_A = re.compile(r"제안서|입찰참가신청서|사업자등록증|실적증명")
# 동현님 (DOCS_REFUSE, ADAPTIVE_RULE="dh")
_THINK_Q = re.compile(r"몇\s*개|합계|총")  # 생각 모드가 필요한 질문 (개수·합계 계산)
_DOCS_Q = re.compile(r"(제출|구비|등록)\s*서류|제출\s*문서")  # 서류 목록을 묻는 질문
_DOCS_CTX = re.compile(
    r"(제출|구비|등록)\s*서류|제출\s*문서"
)  # 조각에 서류 항목이 있는지


# #59 스몰토크: 관련 문서 점수가 낮아 거절할 질문 중, 공고 관련 단어도 사업명/기관명도 없으면 잡담으로 보고 짧게 응대
# (골든셋 v2_4 146문항(일반, 오타)은 모두 아래 단어를 포함해 이 경로를 타지 않음)
_DOMAIN_Q = re.compile(
    r"사업|공고|입찰|예산|금액|비용|가격|기관|요구|제안|평가|계약|용역|기간|서류|발주|과업|구축|시스템|참가|자격|낙찰|일정|문서|요청서|원|RFP"
)
_SMALLTALK_PROMPT = (
    "너는 공공 입찰 공고문(제안요청서) 상담 도우미다. 지금 사용자의 말은 공고문과 관계없는 잡담이다.\n"
    "1) 먼저 사용자의 말에 직접, 자연스럽게 반응한다. 인사에는 인사로, 감사에는 감사로, 질문에는 그 질문에 맞는 답을 한다.\n"
    "2) 네가 알 수 없거나 볼 수 없는 것(사용자의 외모, 날씨, 뉴스, 개인 정보 등)은 솔직하게 모른다고 하되, 딱딱하지 않게 가볍고 따뜻하게 말한다.\n"
    "3) 마지막에 한 문장으로만 입찰 공고문에 대해 궁금한 점을 물어보면 도와줄 수 있다고 안내한다. 안내 항목을 길게 나열하지 않는다.\n"
    "4) 공고문 내용을 묻는 말이면 내용을 추측하지 말고, 공고 문서에서 찾지 못했으니 사업명이나 발주기관을 넣어 다시 질문해 달라고 안내한다.\n"
    "존댓말로 2~3문장. 앞 대화가 있으면 이어지는 말로 이해한다.\n"
    "예) 사용자: 나 잘생겼어? -> 제가 얼굴을 볼 수는 없지만, 그렇게 자신 있게 물어보시는 걸 보니 멋진 분일 것 같아요. 입찰 공고문에 대해 궁금한 점이 있으면 편하게 물어봐 주세요."
)


def _is_smalltalk(question):
    if _DOMAIN_Q.search(question):
        return False
    orgs, titles = _extract_entities(question)
    return not (orgs or titles)


def _smalltalk_reply(question, history=None):
    model, _ = get_resources()
    result = model.invoke(
        [
            SystemMessage(_SMALLTALK_PROMPT),
            *history_to_messages((history or [])[-4:]),  # 최근 2번 주고받은 대화만
            HumanMessage(question),
        ]
    )
    usage = getattr(result, "usage_metadata", None) or {}
    USAGE["calls"] += 1
    USAGE["input"] += usage.get("input_tokens", 0)
    USAGE["output"] += usage.get("output_tokens", 0)
    USAGE["reasoning"] += (usage.get("output_token_details") or {}).get("reasoning", 0)
    return {
        "answer": result.text.strip(),
        "prompt_tokens": usage.get("input_tokens", 0),
    }


def generate(state: RAGState):
    if RERANK and RERANK_CUTOFF is not None and state["retrieved"]:
        top1_score = -state["retrieved"][0][
            1
        ]  # 저장된 값(-점수)을 원래 리랭커 점수로 복원
        if top1_score < RERANK_CUTOFF:
            if _is_smalltalk(
                state["question"]
            ):  # #59: 공고와 관계없는 잡담은 짧게 응대
                return _smalltalk_reply(state["question"], state.get("history"))
            return {"answer": _REFUSE_MSG, "prompt_tokens": 0}
    model, _ = get_resources()
    question = state["question"]
    ctx_text = format_context(state["retrieved"])
    # 라희님 CORR: 서류 질문인데 조각이 "입찰공고문에 따름"이면 LLM 없이 입찰공고문 안내
    if (
        CORR_REFUSE
        and _SUBMISSION_Q.search(question)
        and _REFUSE_NOTICE_HEAD.search(ctx_text)
    ):
        return {"answer": _REFUSE_NOTICE_MSG, "prompt_tokens": 0}
    # 동현님: 서류 질문인데 조각에 서류 항목 자체가 없으면 LLM 없이 거절 (목록 지어내기 방지)
    if DOCS_REFUSE and _DOCS_Q.search(question) and not _DOCS_CTX.search(ctx_text):
        return {"answer": _REFUSE_MSG, "prompt_tokens": 0}
    # 로컬 전용 빠른 모델 규칙
    if ADAPTIVE_THINK and THINK and state["retrieved"]:
        top1_score = -state["retrieved"][0][1]
        # "dh"(동현님): 개수·합계 질문이 아니면 빠른 모델
        # "corr"(라희님): top1 고신뢰 + 단순 질문만 빠른 모델
        use_fast = (ADAPTIVE_RULE == "dh" and not _THINK_Q.search(question)) or (
            ADAPTIVE_RULE == "corr"
            and top1_score >= ADAPTIVE_THINK_TOP1
            and not _COMPLEX_Q.search(question)
        )
        if use_fast:
            model = get_fast_model()
    messages = [
        SystemMessage(load_prompt(RAG_PROMPT)),
        *history_to_messages(state.get("history")),
        HumanMessage(f"컨텍스트:\n{ctx_text}\n\n질문: {question}"),
    ]
    result = model.invoke(messages)
    usage = getattr(result, "usage_metadata", None) or {}
    USAGE["calls"] += 1
    USAGE["input"] += usage.get("input_tokens", 0)
    USAGE["output"] += usage.get("output_tokens", 0)
    USAGE["reasoning"] += (usage.get("output_token_details") or {}).get("reasoning", 0)
    ans = result.text
    first_line = ans.strip().split("\n")[0]
    if CORR_REFUSE:
        # 라희님 CORR 거절 후처리
        if _REFUSE_NOTICE_HEAD.search(ans):
            ans = _REFUSE_NOTICE_MSG
        elif _REFUSE_HEAD_CORR.search(first_line) or (
            _SUBMISSION_Q.search(question)
            and "성과품" not in question
            and "성과품" in ans
            and not _BID_DOC_A.search(ans)
        ):
            ans = _REFUSE_MSG
    elif _REFUSE_HEAD.search(first_line):
        ans = _REFUSE_MSG
    prompt_tokens = usage.get("input_tokens")
    if prompt_tokens is None:  # ollama 는 응답 정보에 들어 있음
        prompt_tokens = (getattr(result, "response_metadata", None) or {}).get(
            "prompt_eval_count"
        )
    return {
        "answer": ans,
        "prompt_tokens": prompt_tokens,
    }


_THOUSAND_WON = re.compile(r"(\d{1,3}(?:,\d{3})*)\s*천원")


def format_context(retrieved):
    """검색된 조각들을 통합 (CORR_WON 이면 천원 단위 금액에 원 환산값 병기)"""
    text = "\n\n".join(
        f"[출처: {doc.metadata['발주 기관']} / {doc.metadata['사업명']}]\n{doc.page_content}"
        for doc, _ in retrieved
    )
    if CORR_WON:
        text = _THOUSAND_WON.sub(lambda m: f"{m.group(0)}(={m.group(1)},000원)", text)
    return text


def build_graph():
    """START -> retrieve -> generate -> END"""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def answer(message, history=None):
    """팀 화면 연결용 함수 (engines/<id>/ 에서 불러 쓰면 됨)"""
    result = get_graph().invoke({"question": message, "history": history or []})
    return result["answer"]


# =====================================================================
# 5. 검색 도구와 에이전트 (evaluate.py --engine agent 용)
#    agent 방식(make_agent)도 이 엔진의 search() 를 도구로 쓰고, 컷오프(고정 거절 문구), 동적 think,
# 후속질문 재작성, 오타교정을 rag 와 같이 적용한다. (사전 검색 1회가 더해져 검색 시간이 늘어난다)
# 최종 채택은 rag 그래프(answer)이고, agent 는 비교 실험용이다. (11-20 측정: rag 78.4%/2.84초, agent 71.6%/4.83초)
# =====================================================================
TOOL_LOG = []  # 에이전트가 검색 도구로 가져온 조각 기록
TOOL_CALLS = []  # 에이전트가 검색 도구를 부른 검색어 기록 (호출 횟수 확인용)
REFUSE_TEXT = "제공된 공고 문서에서 해당 내용을 찾을 수 없습니다."
_CUR_HISTORY = []  # 지금 질문의 앞 대화 (도구가 후속질문 재작성에 씀)
_PRESEARCH = {
    "used": True,
    "query": None,
    "retrieved": [],
}  # 사전 검색 결과 (첫 도구 호출에 재사용)


@tool
def search_documents(query: str) -> str:
    """공공 입찰 제안요청서(RFP) 문서에서 질문과 관련된 내용을 검색.
    사업 예산, 사업 기간, 입찰 방법, 평가 기준, 요구사항 등을 찾을 때 사용"""
    if not _PRESEARCH["used"]:
        # 첫 호출: 에이전트가 만든 검색어 대신, 질문 전체로 한 사전 검색 결과를 씀 (rag 와 같은 검색)
        _PRESEARCH["used"] = True
        query, retrieved = _PRESEARCH["query"], _PRESEARCH["retrieved"]
    else:
        query = rewrite_followup(fix_typos(query), _CUR_HISTORY)
        retrieved = search(query)
    TOOL_LOG.extend(retrieved)
    TOOL_CALLS.append(query)
    if (
        RERANK
        and RERANK_CUTOFF is not None
        and retrieved
        and -retrieved[0][1] < RERANK_CUTOFF
    ):
        return (
            f"관련 문서를 찾을 수 없습니다. 사용자에게 '{REFUSE_TEXT}'라고만 답하세요."
        )
    return format_context(retrieved)


class _AgentRunner:
    """rag 와 같은 최종 세팅을 지키는 에이전트 (evaluate.py 는 .invoke 만 부름)
    1) 오타교정 + 후속질문 재작성  2) 컷오프(고정 거절 문구)  3) 동적 think(빠른 모델 / 일반 모델)
    을 코드가 먼저 처리하고, 통과한 질문만 도구를 쓰는 에이전트가 답한다."""

    def __init__(self):
        self._agents = {}

    def _agent(self, fast):
        key = "fast" if fast else "normal"
        if key not in self._agents:
            model = get_fast_model() if fast else get_resources()[0]
            self._agents[key] = create_agent(
                model, tools=[search_documents], system_prompt=load_prompt(AGENT_PROMPT)
            )
        return self._agents[key]

    def invoke(self, inputs):
        global _CUR_HISTORY
        messages = inputs["messages"]
        question = messages[-1].content
        history = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages[:-1]
        ]
        _CUR_HISTORY = history

        query = rewrite_followup(fix_typos(question), history)
        retrieved = search(query)  # 사전 검색: 컷오프와 동적 think 판단용
        top1 = -retrieved[0][1] if retrieved else None

        if (
            RERANK
            and RERANK_CUTOFF is not None
            and top1 is not None
            and top1 < RERANK_CUTOFF
        ):
            TOOL_LOG.extend(retrieved)
            TOOL_CALLS.append(query)
            return {"messages": [AIMessage(REFUSE_TEXT)]}

        fast = bool(
            ADAPTIVE_THINK
            and THINK
            and top1 is not None
            and top1 >= 0.995
            and not _COMPLEX_Q.search(question)
        )
        _PRESEARCH.update(used=False, query=query, retrieved=retrieved)
        return self._agent(fast).invoke(inputs)


def make_agent():
    """rag 와 같은 세팅(컷오프, 동적 think, 후속질문 재작성, 오타교정)을 지키는 에이전트 생성"""
    return _AgentRunner()


# =====================================================================
# 6. 실행 (샘플 질문으로 빠르게 시험: python -m engines.local_final.baseline)
# =====================================================================
def main():
    graph = get_graph()

    rows = []
    for i, question in enumerate(QUESTIONS, 1):
        start = time.time()
        result = graph.invoke({"question": question, "history": []})
        seconds = round(time.time() - start, 2)

        print(f"\n[{i}] 질문: {question}")
        print(f"답변: {result['answer']}")
        for doc, score in result["retrieved"]:
            # 점수는 작을수록 좋음 (리랭커를 쓰면 -리랭커점수)
            print(f"  - 점수 {score:.3f} | {doc.metadata['사업명']}")
        print(f"  (입력 토큰 {result.get('prompt_tokens')} / {NUM_CTX}, {seconds}초)")

        rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": " || ".join(
                    f"{doc.metadata['공고 번호']} {doc.metadata['사업명']} (점수 {score:.3f})"
                    for doc, score in result["retrieved"]
                ),
                "prompt_tokens": result.get("prompt_tokens"),
                "seconds": seconds,
                "version": "rag",
                "prompt": RAG_PROMPT,
                "chat_model": CHAT_MODEL,
                "embed_model": EMBED_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": TOP_K,
                "num_ctx": NUM_CTX,
                "chunker": CHUNKER,
                "rerank": f"{RERANK_MODEL.split('/')[-1]}@{RERANK_POOL}"
                if RERANK
                else False,
                "rerank_cutoff": RERANK_CUTOFF,
            }
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"baseline_rag_{datetime.now():%Y%m%d_%H%M%S}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
