"""RAG 베이스라인 (rag: LangGraph 검색+답변 / agent: 검색 도구 에이전트)

설정값은 config.py, 프롬프트는 prompts/ 폴더에서 바꿉니다.
직접 실행 (프로젝트 폴더에서): uv run python -m engines.local_model.baseline
"""

import os
import time
import unicodedata
from datetime import datetime
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

from engines.local_model import config

# =====================================================================
# 설정값 (값은 config.py 에서 바꿈. 아래는 실행 중 configure() 로 바뀔 수 있는 사본)
# =====================================================================
CHAT_MODEL = config.CHAT_MODEL
EMBED_MODEL = config.EMBED_MODEL
NUM_CTX = config.NUM_CTX
TEMPERATURE = config.TEMPERATURE
SEED = config.SEED
THINK = config.THINK

CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
TOP_K = config.TOP_K
EMBED_BATCH = config.EMBED_BATCH

REBUILD_DB = config.REBUILD_DB
EXACT_SEARCH = config.EXACT_SEARCH

RAG_PROMPT = config.RAG_PROMPT  # 사용할 프롬프트 파일 이름 (prompts/<이름>.txt)
AGENT_PROMPT = config.AGENT_PROMPT

# 정답: 예산 130,000,000원(VAT 포함) / 기간 계약일로부터 3개월 / 입찰 방법 제한경쟁입찰
QUESTIONS = [
    "한영대학교 트랙운영 학사정보시스템 고도화 사업의 사업 예산은 얼마인가요?",
    "한영대학교 트랙운영 학사정보시스템 고도화 사업의 사업 기간은 어떻게 되나요?",
    "한영대학교 트랙운영 학사정보시스템 고도화 사업의 입찰 방법은 무엇인가요?",
]

# =====================================================================
# 경로
# =====================================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # 프로젝트 폴더
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
load_dotenv(BASE_DIR / ".env")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"

DATA_PATH = BASE_DIR / "data" / "data_list.csv"
CORPUS = config.CORPUS  # 말뭉치 폴더 이름 (data/ 아래)
TXT_DIR = BASE_DIR / "data" / CORPUS
GOLDEN_PATH = BASE_DIR / config.GOLDEN_FILE
GOLDEN_EXCLUDE = config.GOLDEN_EXCLUDE
OUTPUT_DIR = BASE_DIR / "outputs"


def safe_name(text):
    """폴더 이름에 쓸 수 없는 글자를 - 로 바꿈 (글자, 숫자, _, - 는 그대로)"""
    return "".join(c if c.isalnum() or c in "_-" else "-" for c in text)


def db_dir_for(corpus, embed_model, chunk_size, chunk_overlap):
    """설정마다 벡터 DB 폴더를 따로 둠 (말뭉치나 조각 크기를 바꾸면 새 DB를 만든다)"""
    name = f"chroma_{safe_name(corpus)}_{safe_name(embed_model)}_{chunk_size}_{chunk_overlap}"
    return BASE_DIR / "indexes" / name


DB_DIR = db_dir_for(CORPUS, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)
DONE_FILE = DB_DIR / "build_done.txt"  # DB 만들기가 끝까지 됐는지 표시하는 파일


def prompt_engine(name):
    """프롬프트 이름 앞부분으로 어느 엔진용인지 판단 (rag_v2 -> rag)"""
    for engine in ("rag", "agent"):
        if name.startswith(engine + "_"):
            return engine
    raise ValueError(f"프롬프트 이름은 rag_ 또는 agent_ 로 시작해야 합니다: {name}")


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
    """실험 설정 바꾸기 (모델, DB 를 처음 쓰기 전에 호출해야 함)

    예: configure(chunk_size=800, chunk_overlap=100, top_k=5, corpus="clean_mixed_ocr_v2")
    값을 넣지 않은 항목은 위 설정값을 그대로 쓴다.
    """
    global CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, CORPUS, TXT_DIR, DB_DIR, DONE_FILE, _graph
    global GOLDEN_PATH, RAG_PROMPT, AGENT_PROMPT
    if prompt is not None:
        load_prompt(prompt)  # 파일이 있는지 바로 확인
        if prompt_engine(prompt) == "rag":
            RAG_PROMPT = prompt
        else:
            AGENT_PROMPT = prompt
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
    _resources.clear()  # 이미 만든 모델, DB 가 있으면 새 설정으로 다시 만들게 비움
    _graph = None


def current_settings(engine=None):
    """지금 쓰는 설정값 (결과 파일, 시트 기록용). engine 을 주면 그 엔진의 프롬프트 이름도 포함"""
    settings = {
        "chat_model": CHAT_MODEL,
        "corpus": CORPUS,
        "golden": GOLDEN_PATH.stem,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "num_ctx": NUM_CTX,
        "think": THINK,
        "search": "exact" if EXACT_SEARCH else "hnsw",
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
        # CSV 의 파일명(.hwp 또는 .pdf)에서 확장자만 .txt 로 바꾼 이름을 검색
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


def build_vector_store():
    embedding_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    vector_store = make_store(embedding_model)

    # 이전에 끝까지 만들어 둔 DB 가 있으면 그대로 사용
    if DONE_FILE.exists() and not REBUILD_DB:
        print(f"저장된 벡터 DB를 사용합니다: {DB_DIR.name}")
        return vector_store

    if not TXT_DIR.is_dir():
        raise FileNotFoundError(
            f"말뭉치 폴더가 없습니다: data/{CORPUS} "
            "(비밀방의 data_bundle.zip 을 data/ 에 풀었는지 확인)"
        )

    # 중간에 멈춘 DB 이거나 다시 만들기를 원하면 지우고 처음부터 재생성
    vector_store.delete_collection()
    DONE_FILE.unlink(missing_ok=True)
    vector_store = make_store(embedding_model)

    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)  # 메타데이터가 조각에도 유지됨
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
    """답변 모델 생성"""
    return init_chat_model(
        CHAT_MODEL,
        model_provider="ollama",
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        seed=SEED,
        reasoning=THINK,  # 생각 모드 (config.py 의 THINK)
        num_ctx=NUM_CTX,
    )


GET_BATCH = 5000  # 정확 검색용으로 DB 에서 한 번에 불러올 조각 수 (Chroma 내부 한도 32,766 보다 작게)
_resources = {}  # 모델과 벡터 DB 를 한 번만 만들어 재사용하기 위한 보관함


def get_resources():
    """모델, 벡터 DB 생성(최초 호출 시) 및 재호출(재사용 시)"""
    if not _resources:
        _resources["model"] = make_chat_model()
        _resources["vector_store"] = build_vector_store()
    return _resources["model"], _resources["vector_store"]


def warmup():
    """모델과 DB 를 미리 불러오고 한 번 호출 (첫 문항 응답시간에 로딩 시간이 섞이지 않게)"""
    model, _ = get_resources()
    if EXACT_SEARCH:
        get_exact_index()
    model.invoke("안녕")


def get_exact_index():
    """정확한 검색용: 저장된 조각 임베딩과 내용을 한 번만 메모리에 올려 재사용"""
    if "exact" not in _resources:
        _, vector_store = get_resources()
        # 한 번에 모두 불러오면 조각이 32,766개를 넘을 때 Chroma 내부 한도 오류
        # ("too many SQL variables")가 나서 GET_BATCH 개씩 나눠 불러옴 (순서는 그대로)
        collection = vector_store._collection
        parts, docs = [], []
        for offset in range(0, collection.count(), GET_BATCH):
            data = collection.get(
                include=["embeddings", "documents", "metadatas"],
                limit=GET_BATCH,
                offset=offset,
            )
            parts.append(np.asarray(data["embeddings"], dtype=np.float32))
            docs.extend(
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(data["documents"], data["metadatas"], strict=True)
            )
        matrix = np.vstack(parts)
        _resources["exact"] = (matrix, docs)
    return _resources["exact"]


SEARCH_TIMES = []  # search() 를 부를 때마다 걸린 초 (evaluate.py 가 문항마다 비우고 더해서 search_sec 로 기록)


def search(query, k=None):
    """질문과 가까운 조각 k 개를 [(조각, 거리), ...] 로 반환 (거리가 작을수록 가까움)

    rag 검색 노드, agent 검색 도구, retrieval.py 가 모두 이 함수를 쓴다.
    """
    start = time.time()
    k = k or TOP_K
    _, vector_store = get_resources()
    if not EXACT_SEARCH:
        result = vector_store.similarity_search_with_score(query, k=k)
    else:
        matrix, docs = get_exact_index()
        q = np.asarray(vector_store.embeddings.embed_query(query), dtype=np.float32)
        dist = ((matrix - q) ** 2).sum(axis=1)  # Chroma 의 l2 와 같은 거리 (제곱 거리)
        top = np.argsort(dist, kind="stable")[:k]  # 거리가 같으면 저장 순서대로
        result = [(docs[i], float(dist[i])) for i in top]
    SEARCH_TIMES.append(time.time() - start)
    return result


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
    """노드끼리 주고받는 공용 바구니 (필요 시 추가 가능)."""

    question: str  # 사용자 질문
    history: list  # 앞 대화 [{"role": "user"/"assistant", "content": ...}, ...]
    retrieved: list  # 검색 결과 [(조각, 거리), ...]
    answer: str  # 최종 답변
    prompt_tokens: int  # 모델이 읽은 입력 토큰 수


def retrieve(state: RAGState):
    """검색 노드: 질문과 비슷한 조각을 TOP_K 개 가져오기"""
    retrieved = search(state["question"])
    return {"retrieved": retrieved}


def generate(state: RAGState):
    """답변 노드: 검색 결과 + 이전 대화 참고"""
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
    """START -> retrieve -> generate -> END 흐름을 생성"""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # [노드 추가 예시] 후속 질문을 검색하기 좋게 다시 쓰는 노드를 검색 앞에 끼우려면
    #   1) 노드 함수 만들기: def rewrite(state): ... return {"question": 새로_쓴_질문}
    #   2) graph.add_node("rewrite", rewrite)
    #   3) 위의 graph.add_edge(START, "retrieve") 를 지우고 아래 두 줄로 연결
    #      graph.add_edge(START, "rewrite")
    #      graph.add_edge("rewrite", "retrieve")
    return graph.compile()


_graph = None


def get_graph():
    """완성된 그래프를 한 번만 만들어 재사용"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def answer(message, history=None):
    """팀 화면 연결용 함수 (engines/<id>/ 에서 불러 쓰면 됨)"""
    result = get_graph().invoke({"question": message, "history": history or []})
    return result["answer"]


# =====================================================================
# 5. 검색 도구와 에이전트
# =====================================================================
TOOL_LOG = []  # 에이전트가 검색 도구로 가져온 조각 기록
TOOL_CALLS = []  # 에이전트가 검색 도구를 부른 검색어 기록 (호출 횟수 확인용)


@tool
def search_documents(query: str) -> str:
    """공공 입찰 제안요청서(RFP) 문서에서 질문과 관련된 내용을 검색.
    사업 예산, 사업 기간, 입찰 방법, 평가 기준, 요구사항 등을 찾을 때 사용"""
    retrieved = search(query)
    TOOL_LOG.extend(retrieved)
    TOOL_CALLS.append(query)
    return format_context(retrieved)


def make_agent():
    """검색 도구를 가진 에이전트 생성 (create_agent 방식)"""
    model, _ = get_resources()
    return create_agent(
        model, tools=[search_documents], system_prompt=load_prompt(AGENT_PROMPT)
    )


# =====================================================================
# 6. 실행
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
            print(f"  - 거리 {score:.3f} | {doc.metadata['사업명']}")
        print(f"  (입력 토큰 {result.get('prompt_tokens')} / {NUM_CTX}, {seconds}초)")

        rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": " || ".join(
                    f"{doc.metadata['공고 번호']} {doc.metadata['사업명']} (거리 {score:.3f})"
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
            }
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"baseline_rag_{datetime.now():%Y%m%d_%H%M%S}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
