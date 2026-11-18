"""rahui 엔진: 구조 청킹 + 기관명 필터 + 리랭커 (rag 전용)

engines/local_model/baseline.py 와 같은 이름의 함수·변수를 가져서 evaluate.py 가 그대로 쓸 수 있다.
다른 점은 세 가지다.
  1) 청킹  : config.CHUNKER  "fixed"(글자 수로 자르기) / "struct"(제목 단위, 표 보존, 기관|사업명|절 헤더)
  2) 앞단  : config.AGENCY_FILTER 가 True 면 질문에 든 발주 기관의 조각만 후보로 남김
  3) 뒷단  : config.RERANK 가 True 면 후보 RERANK_POOL 개를 리랭커로 다시 순위 매김
검색 결과 [(조각, 점수)] 의 점수는 '작을수록 좋음'이다. (리랭커를 쓰면 관련도에 -1 을 곱한 값)

설정값은 config.py, 프롬프트는 prompts/ 에서 바꾼다.
engines/<id>/ 로 옮길 때 바꿀 곳: 아래 config import 한 줄, struct 청킹 import(chunk_struct) 한 줄.
"""

import os
import re
import time
import unicodedata
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from ai.rahui.exp_filter_rerank.rahui_engine import config

# =====================================================================
# 설정값 (값은 config.py 에서 바꿈. 아래는 실행 중 configure()/set_search() 로 바뀔 수 있는 사본)
# =====================================================================
CHAT_MODEL = config.CHAT_MODEL
EMBED_MODEL = config.EMBED_MODEL
NUM_CTX = config.NUM_CTX
TEMPERATURE = config.TEMPERATURE
SEED = config.SEED

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

RAG_PROMPT = config.RAG_PROMPT


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
    global GOLDEN_PATH, RAG_PROMPT
    if prompt is not None:
        if not prompt.startswith("rag_"):
            raise ValueError(f"rahui 엔진은 rag_ 프롬프트만 씁니다: {prompt}")
        load_prompt(prompt)
        RAG_PROMPT = prompt
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
):
    """검색 방식 바꾸기 (rahui 엔진 전용). configure() 보다 먼저 또는 그 뒤에 호출해도 된다."""
    global \
        AGENCY_FILTER, \
        RERANK, \
        RERANK_POOL, \
        RERANK_MAX_LENGTH, \
        RERANK_BATCH, \
        RERANK_FP16
    global CHUNKER, DB_DIR, DONE_FILE, _graph
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
    DB_DIR = db_dir_for(CORPUS, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)
    DONE_FILE = DB_DIR / "build_done.txt"
    _resources.clear()
    _graph = None


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
        "num_ctx": NUM_CTX,
        "search": "exact",
        "chunker": CHUNKER,
        "agency_filter": AGENCY_FILTER,
        "rerank": f"{RERANK_MODEL.split('/')[-1]}@{RERANK_POOL}" if RERANK else False,
    }
    if engine == "rag":
        settings["prompt"] = RAG_PROMPT
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
        from ai.rahui.experiments.chunk_compare import chunk_struct

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
def make_chat_model():
    return init_chat_model(
        CHAT_MODEL,
        model_provider="ollama",
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        seed=SEED,
        num_ctx=NUM_CTX,
    )


_resources = {}  # 모델, 벡터 DB, 검색 색인, 리랭커를 한 번만 만들어 재사용하기 위한 보관함


def get_resources():
    if not _resources:
        _resources["model"] = make_chat_model()
        _resources["vector_store"] = build_vector_store()
    return _resources["model"], _resources["vector_store"]


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
    model, _ = get_resources()
    get_exact_index()
    if RERANK:
        get_reranker().predict([("워밍업", "워밍업")], batch_size=1)
    model.invoke("안녕")


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
def search(query, k=None):
    """질문과 관련 있는 조각 k 개를 [(조각, 점수), ...] 로 반환 (점수가 작을수록 좋음)

    1) 모든 조각과 임베딩 거리를 계산  2) AGENCY_FILTER: 질문에 든 기관의 조각만 후보로
    3) 거리순 상위 후보 추리기  4) RERANK: 후보를 리랭커로 다시 순위 매김
    """
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
        return cand[:k]

    scores = get_reranker().predict(
        [(query, d.page_content) for d, _ in cand], batch_size=RERANK_BATCH
    )
    ranked = np.argsort(-np.asarray(scores), kind="stable")
    return [(cand[j][0], float(-scores[j])) for j in ranked][:k]


def history_to_messages(history):
    """앞 대화 기록(role/content 목록)을 LangChain 메시지로 변경"""
    messages = []
    for turn in history or []:
        if turn["role"] == "user":
            messages.append(HumanMessage(turn["content"]))
        else:
            messages.append(AIMessage(turn["content"]))
    return messages


def format_context(retrieved):
    """검색된 조각들을 통합"""
    return "\n\n".join(
        f"[출처: {doc.metadata['발주 기관']} / {doc.metadata['사업명']}]\n{doc.page_content}"
        for doc, _ in retrieved
    )


# =====================================================================
# 4. rag: LangGraph 구조
# =====================================================================
class RAGState(TypedDict, total=False):
    question: str
    history: list
    retrieved: list
    answer: str
    prompt_tokens: int


def retrieve(state: RAGState):
    return {"retrieved": search(state["question"])}


def generate(state: RAGState):
    model, _ = get_resources()
    messages = [
        SystemMessage(load_prompt(RAG_PROMPT)),
        *history_to_messages(state.get("history")),
        HumanMessage(
            f"컨텍스트:\n{format_context(state['retrieved'])}\n\n질문: {state['question']}"
        ),
    ]
    result = model.invoke(messages)
    return {
        "answer": result.text,
        "prompt_tokens": result.response_metadata.get("prompt_eval_count"),
    }


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
