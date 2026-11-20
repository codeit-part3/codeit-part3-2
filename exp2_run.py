import argparse
import re
import runpy
import sys
import time

import numpy as np
from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from engines.local_model import baseline as b

parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--chunk-mode", choices=["fixed", "structure"], default="fixed")
parser.add_argument("--search-mode", choices=["dense", "hybrid"], default="dense")
parser.add_argument("--no-prefix", action="store_true")
parser.add_argument("--preview", action="store_true")
args, rest = parser.parse_known_args()

size_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
size_parser.add_argument("--chunk-size", type=int, nargs="+", default=[b.CHUNK_SIZE])
size_parser.add_argument(
    "--chunk-overlap", type=int, nargs="+", default=[b.CHUNK_OVERLAP]
)
sizes, _ = size_parser.parse_known_args(rest)

CHUNK_MODE = args.chunk_mode
SEARCH_MODE = args.search_mode
NO_PREFIX = args.no_prefix
RRF_K = 60

HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|\d{1,2})\s*\|\s*\|\s*[^|]+\|?\s*$"
    r"|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*\.\s*\S"
    r"|\d{1,2}\.\s+[가-힣A-Za-z]"
    r"|제\s?\d+\s?[장절조]"
    r")"
)
ROMAN_RE = re.compile(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]")

_kiwi = Kiwi()


def tokenize(text):
    return [
        t.form.lower()
        for t in _kiwi.tokenize(text)
        if t.tag.startswith(("N", "V", "XR", "SL", "SN"))
    ]


def split_sections(text):
    sections, parent, title, buf = [], "", "", []
    for line in text.splitlines():
        s = line.strip()
        if s and len(s) <= 50 and HEADING_RE.match(line):
            if "".join(buf).strip():
                sections.append((title, "\n".join(buf).strip()))
            clean = re.sub(r"\s*\|\s*", " ", s).strip()
            if ROMAN_RE.match(clean):
                parent, title = clean, clean
            else:
                title = f"{parent} > {clean}" if parent else clean
            buf = []
            continue
        buf.append(line)
    if "".join(buf).strip():
        sections.append((title, "\n".join(buf).strip()))
    merged = []
    for t, x in sections:
        if merged and len(x) < 150:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + x)
        else:
            merged.append((t, x))
    return merged


def split_documents(documents):
    if CHUNK_MODE != "structure":
        splitter = b.RecursiveCharacterTextSplitter(
            chunk_size=b.CHUNK_SIZE,
            chunk_overlap=b.CHUNK_OVERLAP,
            length_function=len,
        )
        return splitter.split_documents(documents)
    chunks = []
    for doc in documents:
        name = doc.metadata.get("사업명", "")
        for title, body in split_sections(doc.page_content):
            if NO_PREFIX:
                head = ""
            else:
                head = f"[{name}] {title}\n" if title else f"[{name}]\n"
            size = max(b.CHUNK_SIZE - len(head), b.CHUNK_OVERLAP * 2)
            splitter = b.RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=b.CHUNK_OVERLAP,
                length_function=len,
            )
            for piece in splitter.split_text(body):
                chunks.append(
                    Document(page_content=head + piece, metadata=dict(doc.metadata))
                )
    return chunks


def db_dir_for(corpus, embed_model, chunk_size, chunk_overlap):
    suffix = ""
    if CHUNK_MODE != "fixed":
        suffix = f"_{CHUNK_MODE}" + ("_np" if NO_PREFIX else "")
    name = (
        f"chroma_{b.safe_name(corpus)}_{b.safe_name(embed_model)}"
        f"_{chunk_size}_{chunk_overlap}{suffix}"
    )
    return b.BASE_DIR / "indexes" / name


def build_vector_store():
    embedding_model = b.OllamaEmbeddings(model=b.EMBED_MODEL, base_url=b.OLLAMA_URL)
    vector_store = b.make_store(embedding_model)

    if b.DONE_FILE.exists() and not b.REBUILD_DB:
        print(f"저장된 벡터 DB를 사용합니다: {b.DB_DIR.name}")
        return vector_store

    if not b.TXT_DIR.is_dir():
        raise FileNotFoundError(f"말뭉치 폴더가 없습니다: data/{b.CORPUS}")

    vector_store.delete_collection()
    b.DONE_FILE.unlink(missing_ok=True)
    vector_store = b.make_store(embedding_model)

    documents = b.load_documents()
    chunks = split_documents(documents)
    print(
        f"[{CHUNK_MODE}] 문서 {len(documents)}개 -> 조각 {len(chunks)}개를 임베딩합니다. (몇 분 걸림)"
    )

    start = time.time()
    for i in range(0, len(chunks), b.EMBED_BATCH):
        vector_store.add_documents(documents=chunks[i : i + b.EMBED_BATCH])
        done = min(i + b.EMBED_BATCH, len(chunks))
        if done % (b.EMBED_BATCH * 10) == 0 or done == len(chunks):
            print(f"  {done} / {len(chunks)} ({time.time() - start:.0f}초 경과)")

    b.DONE_FILE.write_text(f"chunks={len(chunks)}\n", encoding="utf-8")
    print("벡터 DB 저장 완료")
    return vector_store


def rank_of(values):
    ranks = np.empty(len(values), dtype=int)
    ranks[np.argsort(values, kind="stable")] = np.arange(1, len(values) + 1)
    return ranks


def get_bm25():
    if "bm25" not in b._resources:
        _, docs = b.get_exact_index()
        print("BM25 인덱스를 만듭니다. (1~3분 걸릴 수 있음)")
        b._resources["bm25"] = BM25Okapi([tokenize(d.page_content) for d in docs])
    return b._resources["bm25"]


def rrf_scores(dist, query):
    dense = 1 / (RRF_K + rank_of(dist))
    tokens = tokenize(query)
    if not tokens:
        return dense
    bm25 = np.asarray(get_bm25().get_scores(tokens))
    return dense + 1 / (RRF_K + rank_of(-bm25))


_orig_get_exact_index = b.get_exact_index


def get_exact_index():
    first = "exact" not in b._resources
    result = _orig_get_exact_index()
    if first and SEARCH_MODE == "hybrid" and b.EXACT_SEARCH:
        get_bm25()
    return result


def search(query, k=None):
    start = time.time()
    k = k or b.TOP_K
    _, vector_store = b.get_resources()
    if not b.EXACT_SEARCH:
        result = vector_store.similarity_search_with_score(query, k=k)
    else:
        matrix, docs = b.get_exact_index()
        q = np.asarray(vector_store.embeddings.embed_query(query), dtype=np.float32)
        dist = ((matrix - q) ** 2).sum(axis=1)
        if SEARCH_MODE == "hybrid":
            order = np.argsort(-rrf_scores(dist, query), kind="stable")
        else:
            order = np.argsort(dist, kind="stable")
        result = [(docs[i], float(dist[i])) for i in order[:k]]
    if hasattr(b, "SEARCH_TIMES"):
        b.SEARCH_TIMES.append(time.time() - start)
    return result


b.db_dir_for = db_dir_for
b.build_vector_store = build_vector_store
b.get_exact_index = get_exact_index
b.search = search

if args.preview:
    b.configure(chunk_size=sizes.chunk_size[0], chunk_overlap=sizes.chunk_overlap[0])
    docs = b.load_documents()[:3]
    chunks = split_documents(docs)
    if chunks:
        avg = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"조각 {len(chunks)}개, 평균 길이 {avg:.0f}자")
        mid = len(chunks) // 2
        for c in chunks[mid : mid + 6]:
            print(repr(c.page_content[:150]))
    print([t for t, _ in split_sections(docs[0].page_content)][:25])
    sys.exit()

print(
    f"실험 설정: chunk_mode={CHUNK_MODE}, search_mode={SEARCH_MODE}, no_prefix={NO_PREFIX}"
)
sys.argv = ["retrieval.py"] + rest
runpy.run_path(str(b.BASE_DIR / "retrieval.py"), run_name="__main__")
