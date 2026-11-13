"""청킹 설정별 검색 성능을 비교합니다.

실행:
  uv run python -m ai.rahui.experiments.chunk_compare
  uv run python -m ai.rahui.experiments.chunk_compare --only F1000,S1000,S1000M

입력: data/<CLEAN_DIR>/, ai/rahui/rahui_golden_set.json
출력: ai/rahui/eda_results/<실행 폴더>/ (results.csv, per_question.csv, summary.txt)
"""

import argparse
import csv
import heapq
import json
import math
import os
import random
import re
import statistics as st
import time
from array import array
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from ai.rahui.analysis.eda_stats import _MARK, heading_cell
from ai.rahui.conversion.convert import DATA
from ai.rahui.paths import GOLDEN_SET_PATH, RESULTS_DIR

CLEAN = DATA / os.environ.get("CLEAN_DIR", "clean")
GOLDEN = Path(os.environ.get("GOLDEN_PATH") or GOLDEN_SET_PATH)  # 기본은 v1
RESULTS = RESULTS_DIR
KS = (1, 3, 5, 10)
ANY = re.compile(r".")
NORM = re.compile(r"[^0-9A-Za-z가-힣]")
KEY_LEN = 30  # 근거 인용의 앞 30자(기호·공백 제외)가 청크에 들어 있으면 적중

# 설정 이름: 방식(F=글자 수 고정, S=제목·표 구조) + 목표 크기 + (M=문서·섹션 메타를 청크 앞에 붙임)
CONFIGS = {
    "F500": ("fixed", 500, False),
    "F1000": ("fixed", 1000, False),
    "F1500": ("fixed", 1500, False),
    "F1000M": ("fixed", 1000, True),
    "S600": ("struct", 600, False),
    "S1000": ("struct", 1000, False),
    "S1500": ("struct", 1500, False),
    "S1000M": ("struct", 1000, True),
    "S1500M": ("struct", 1500, True),  # 기준선 (embed_compare와 같은 이름)
    # dh 방식 요구사항 단위 청킹 (req_chunk.py). R: 나머지 본문도 dh 일반 청킹, RS: 나머지 본문은 우리 구조 청킹
    "R1000M": ("req", 1000, True),
    "RS1500M": ("reqstruct", 1500, True),
    # 전처리 개선 (prep_index.py): T 표 문장화, B 절 제목 경로, C 반복 줄 제거
    "S1500MT": ("sx:T", 1500, True),
    "S1500MB": ("sx:B", 1500, True),
    "S1500MC": ("sx:C", 1500, True),
    "S1500MTB": ("sx:TB", 1500, True),
    "S1500MTBC": ("sx:TBC", 1500, True),
}


def norm(s: str) -> str:
    return NORM.sub("", s)


# ---------------------------------------------------------------- 문서 불러오기
def load_corpus() -> tuple[dict[int, dict], dict[int, int]]:
    with open(CLEAN / "_meta.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    docs, rep = {}, {}
    for r in rows:
        no = int(r["문서번호"])
        rep[no] = (
            int(r["중복대표"]) if r["중복대표"].strip() else no
        )  # 중복으로 빠진 문서는 대표 문서로
        p = CLEAN / f"{Path(r['파일명']).stem}.txt"
        if p.exists():
            docs[no] = {
                "no": no,
                "org": r["발주기관"].strip(),
                "title": r["사업명"].strip(),
                "text": p.read_text(encoding="utf-8"),
            }
    return docs, rep


# ---------------------------------------------------------------- 청킹
def chunk_fixed(
    text: str, size: int, overlap_ratio: float = 0.1
) -> list[tuple[str, str]]:
    """글자 수 기준. 창 끝 20% 안에 줄바꿈이 있으면 거기서 자름. 반환: [(섹션, 본문)]"""
    out, n, start = [], len(text), 0
    overlap = int(size * overlap_ratio)
    while start < n:
        end = min(n, start + size)
        if end < n:
            cut = text.rfind("\n", start + int(size * 0.8), end)
            if cut > start:
                end = cut
        body = text[start:end].strip()
        if body:
            out.append(("", body))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def heading_text(line: str) -> str | None:
    """번호·기호가 붙은 제목 줄이면 제목 글자를, 아니면 None."""
    b = line.strip()
    if not b or len(b) > 120:
        return None
    c = heading_cell(b, ANY)
    if not c or not re.search(r"[가-힣]{2}", c):
        return None
    cells = [x for x in b.split("|") if x.strip()]
    if len(cells) > 1 or _MARK.search(c):
        return c
    return None


def chunk_struct(text: str, target: int) -> list[tuple[str, str]]:
    """제목에서 끊고(청크가 목표의 40% 이상일 때), 표는 가능하면 쪼개지 않음(목표의 1.5배까지)."""
    max_sz = int(target * 1.5)
    out, cur, cur_len, cur_sec, sec = [], [], 0, "", ""
    prev_tbl = False

    def flush():
        nonlocal cur, cur_len
        body = "\n".join(cur).strip()
        if body:
            out.append((cur_sec, body))
        cur, cur_len = [], 0

    for line in text.split("\n"):
        h = heading_text(line)
        is_tbl = "|" in line
        if h:
            if cur_len >= target * 0.4:
                flush()
            sec = h
        if (
            cur_len > 0
            and cur_len + len(line) + 1 > target
            and not (is_tbl and prev_tbl and cur_len + len(line) + 1 <= max_sz)
        ):
            flush()
        if not cur:
            cur_sec = sec
        cur.append(line)
        cur_len += len(line) + 1
        prev_tbl = is_tbl
    flush()
    return out


def build_chunks(docs: dict[int, dict], kind: str, size: int, meta: bool) -> list[dict]:
    chunks = []
    for no, d in docs.items():
        if kind.startswith("sx:"):
            from ai.rahui.preprocessing.prep_index import chunk_x

            for sec, body, ibody in chunk_x(d["text"], size, kind[3:]):
                head = (
                    f"[{d['org']} | {d['title']}" + (f" | {sec}" if sec else "") + "]\n"
                    if meta
                    else ""
                )
                chunks.append(
                    {
                        "doc": no,
                        "body": body,
                        "nbody": norm(body),
                        "index": head + ibody,
                    }
                )
            continue
        if kind in ("req", "reqstruct"):
            from ai.rahui.preprocessing.req_chunk import chunk_req

            parts = chunk_req(
                d["text"], "struct" if kind == "reqstruct" else "general", size
            )
        else:
            parts = (
                chunk_fixed(d["text"], size)
                if kind == "fixed"
                else chunk_struct(d["text"], size)
            )
        for sec, body in parts:
            head = (
                f"[{d['org']} | {d['title']}" + (f" | {sec}" if sec else "") + "]\n"
                if meta
                else ""
            )
            chunks.append(
                {"doc": no, "body": body, "nbody": norm(body), "index": head + body}
            )
    return chunks


# ---------------------------------------------------------------- BM25 (글자 2-gram)
def tokens(s: str) -> list[str]:
    out = []
    for w in re.findall(r"[0-9A-Za-z가-힣]+", s):
        out.extend([w] if len(w) == 1 else (w[i : i + 2] for i in range(len(w) - 1)))
    return out


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(texts)
        self.post: dict[str, tuple[array, array]] = {}
        self.dl = array("I")
        for cid, t in enumerate(texts):
            c = Counter(tokens(t))
            self.dl.append(sum(c.values()))
            for tok, tf in c.items():
                p = self.post.get(tok)
                if p is None:
                    p = self.post[tok] = (array("I"), array("H"))
                p[0].append(cid)
                p[1].append(min(tf, 65535))
        self.avg = sum(self.dl) / max(1, self.n)

    def top(self, query: str, k: int) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for tok in set(tokens(query)):
            p = self.post.get(tok)
            if not p:
                continue
            idf = math.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5))
            for cid, tf in zip(*p, strict=False):
                norm_len = 1 - self.b + self.b * self.dl[cid] / self.avg
                scores[cid] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm_len)
        return [cid for cid, _ in heapq.nlargest(k, scores.items(), key=lambda x: x[1])]


# ---------------------------------------------------------------- 질문
def load_questions(
    docs: dict[int, dict], rep: dict[int, int]
) -> tuple[list[dict], list[str]]:
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    nd = {no: norm(d["text"]) for no, d in docs.items()}
    qs, skipped = [], []
    for g in gold:
        quote = (g.get("evidence_quote") or "").strip()
        if g["level"] == "no_answer" or not quote or not g.get("evidence_docs"):
            continue  # 정답 근거가 없는 문항은 검색 적중 평가에서 제외
        evid = {rep.get(d, d) for d in g["evidence_docs"]}
        # 인용이 " / "로 여러 조각을 이은 경우: 조각마다 앞 30자를 키로 쓰고, 전처리 본문에 실제로 있는 키만 남김 (하나라도 든 청크가 있으면 적중)
        segs = [norm(x)[:KEY_LEN] for x in quote.split(" / ")]
        key = tuple(
            k for k in segs if len(k) >= 8 and any(k in nd.get(d, "") for d in evid)
        )
        if not key:
            skipped.append(
                f"{g['id']} (근거 인용을 전처리 본문에서 못 찾음: 이미지 속 내용·메타데이터 등)"
            )
            continue
        query = g["question"]
        hist = [h for h in g.get("history", []) if h.get("role") == "user"]
        if hist:  # 후속 질문은 앞 질문을 붙여서 검색 (대화 맥락 없이는 검색 불가)
            query = (
                (hist[-1].get("content") or hist[-1].get("text") or "") + " " + query
            )
        qs.append(
            {
                "id": g["id"],
                "level": g["level"],
                "query": query,
                "key": key,
                "docs": evid,
                "cluster": min(evid),
            }
        )
    return qs, skipped


# ---------------------------------------------------------------- 부트스트랩 (우연한 차이인지 확인)
PAIRS = [
    ("F1000M", "F1000"),
    ("S1000M", "S1000"),
    ("S1000", "F1000"),
    ("S1500", "F1500"),
    ("S1000M", "F1000M"),
    ("F1000", "F500"),
    ("F1500", "F1000"),
]


def bootstrap(
    perq: list[dict], qs: list[dict], names: list[str], B: int = 5000, seed: int = 0
) -> list[str]:
    """문항을 다시 뽑아(B회) 지표의 95% 구간과 설정 간 차이의 구간을 계산. 같은 문항으로 짝지어 비교(paired).
    같은 문서에서 나온 문항은 서로 비슷하게 맞고 틀리므로 문서 단위로 묶어서 뽑는 방식(cluster)을 씀."""
    ids = [q["id"] for q in qs]
    rank = {(p["config"], p["id"]): p["근거_순위"] for p in perq}
    h5 = {
        n: [1.0 if rank[(n, i)] != "" and rank[(n, i)] <= 5 else 0.0 for i in ids]
        for n in names
    }
    rr = {
        n: [1 / rank[(n, i)] if rank[(n, i)] != "" else 0.0 for i in ids] for n in names
    }
    cl = defaultdict(list)
    for ix, q in enumerate(qs):
        cl[q["cluster"]].append(ix)
    groups = list(cl.values())
    rng = random.Random(seed)
    reps = []
    for _ in range(B):
        idx = [ix for g in (rng.choice(groups) for _ in groups) for ix in g]
        reps.append(idx)

    def ci(vals):
        v = sorted(vals)
        return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]

    def metric(m, n, idx):
        return sum(m[n][i] for i in idx) / len(idx)

    out = [
        f"부트스트랩 {B}회 (문서 단위로 묶어 다시 뽑기, 문서 묶음 {len(groups)}개 / 문항 {len(ids)}개, 시드 {seed})",
        "",
    ]
    out.append(f"{'설정':<8}{'근거@5':>8}{'95% 구간':>16}{'MRR':>8}{'95% 구간':>16}")
    for n in names:
        a = [metric(h5, n, ix) for ix in reps]
        b = [metric(rr, n, ix) for ix in reps]
        al, ah = ci(a)
        bl, bh = ci(b)
        out.append(
            f"{n:<8}{sum(h5[n]) / len(ids):>8.2f}{f'[{al:.2f}, {ah:.2f}]':>16}{sum(rr[n]) / len(ids):>8.2f}{f'[{bl:.2f}, {bh:.2f}]':>16}"
        )
    out += [
        "",
        "설정 간 차이 (앞 - 뒤). 구간이 0을 포함하지 않아야 '우연이 아니다'라고 볼 수 있음. 승/패 = 근거@5에서 앞 설정만 맞춘 문항 수/뒤 설정만 맞춘 문항 수, p = 승패 부호검정(양측)",
    ]
    out.append(
        f"{'비교':<16}{'근거@5 차이':>12}{'95% 구간':>16}{'승/패':>8}{'p':>8}{'MRR 차이':>10}{'95% 구간':>16}"
    )
    for x, y in PAIRS:
        if x not in names or y not in names:
            continue
        d5 = [metric(h5, x, ix) - metric(h5, y, ix) for ix in reps]
        dr = [metric(rr, x, ix) - metric(rr, y, ix) for ix in reps]
        w = sum(1 for i in range(len(ids)) if h5[x][i] > h5[y][i])
        lo = sum(1 for i in range(len(ids)) if h5[x][i] < h5[y][i])
        k, m = min(w, lo), w + lo
        p = (
            min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2**m)
            if m
            else 1.0
        )
        a, b = ci(d5)
        c, d = ci(dr)
        mean5 = sum(h5[x]) / len(ids) - sum(h5[y]) / len(ids)
        meanr = sum(rr[x]) / len(ids) - sum(rr[y]) / len(ids)
        out.append(
            f"{x + ' - ' + y:<16}{mean5:>+12.2f}{f'[{a:+.2f}, {b:+.2f}]':>16}{f'{w}/{lo}':>8}{p:>8.3f}{meanr:>+10.2f}{f'[{c:+.2f}, {d:+.2f}]':>16}"
        )
    return out


# ---------------------------------------------------------------- 실험


def piece_ranks(top: list[int], chunks: list[dict], q: dict) -> list:
    """근거 인용 조각마다, 근거 문서의 청크 중 그 조각이 든 청크가 처음 나온 순위 (없으면 None)."""
    return [
        next(
            (
                i + 1
                for i, cid in enumerate(top)
                if chunks[cid]["doc"] in q["docs"] and k in chunks[cid]["nbody"]
            ),
            None,
        )
        for k in q["key"]
    ]


def strict_fields(pr: list) -> dict:
    """per_question.csv 에 남길 조각별 순위 (나중에 기준을 바꿔도 다시 돌리지 않게)."""
    return {"조각순위": ";".join(str(r) if r else "" for r in pr)}


def _flags(p: dict, k: int) -> tuple[float, float]:
    """(완전근거@k, 조각커버@k): 모든 조각이 상위 k 안에 있으면 1 / 찾은 조각 비율."""
    rs = [int(x) if x != "" else None for x in str(p["조각순위"]).split(";")]
    ok = [1 if r and r <= k else 0 for r in rs]
    return float(all(ok)), sum(ok) / len(ok)


def strict_block(
    rows: list[dict], perq: list[dict], qs: list[dict], names: list[str]
) -> list[str]:
    """엄격 지표 요약 + 완전근거@5 의 95% 구간(문서 묶음 부트스트랩 5000회, 시드 0)."""
    ids = [q["id"] for q in qs]
    by = {(p["config"], p["id"]): p for p in perq}
    cl = defaultdict(list)
    for ix, q in enumerate(qs):
        cl[q["cluster"]].append(ix)
    groups = list(cl.values())
    rng = random.Random(0)
    reps = [
        [ix for g in (rng.choice(groups) for _ in groups) for ix in g]
        for _ in range(5000)
    ]
    out = [
        "엄격 지표 (같은 청크·같은 검색 결과를 더 까다롭게 채점)",
        "  완전근거@k = 근거 인용 조각을 하나도 빠짐없이 상위 k개 안에서 찾은 문항 비율 (조각이 여러 개인 문항은 전부 필요)",
        "  조각커버@k = 문항별로 찾은 조각의 비율을 평균낸 값 (부분 점수)",
        "  근거@k(기존) = 조각 하나만 찾아도 적중. 값이 같으면 모든 문항이 조각 하나로 판정된다는 뜻",
        "",
        f"{'설정':<26}{'근거@5':>8}{'완전@5':>8}{'95% 구간':>16}{'완전@10':>9}{'조각커버@5':>11}{'조각커버@10':>12}",
    ]
    base = {r["설정"]: r for r in rows}
    for nm in names:
        f5 = [_flags(by[(nm, i)], 5)[0] for i in ids]
        f10 = [_flags(by[(nm, i)], 10)[0] for i in ids]
        c5 = [_flags(by[(nm, i)], 5)[1] for i in ids]
        c10 = [_flags(by[(nm, i)], 10)[1] for i in ids]
        v = sorted(sum(f5[i] for i in ix) / len(ix) for ix in reps)
        lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        n = len(ids)
        out.append(
            f"{nm:<26}{base[nm]['근거@5']:>8.2f}{sum(f5) / n:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}{sum(f10) / n:>9.2f}{sum(c5) / n:>11.2f}{sum(c10) / n:>12.2f}"
        )
    multi = [q["id"] for q in qs if len(q["key"]) > 1]
    out += ["", f"조각이 2개 이상인 문항 {len(multi)}개: " + ", ".join(multi)]
    return out


def run_config(
    name: str, docs: dict[int, dict], qs: list[dict]
) -> tuple[dict, list[dict]]:
    kind, size, meta = CONFIGS[name]
    t0 = time.time()
    chunks = build_chunks(docs, kind, size, meta)
    bm = BM25([c["index"] for c in chunks])
    build_s = time.time() - t0
    lens = [len(c["body"]) for c in chunks]
    per_q, hit = [], {k: 0 for k in KS}
    dhit = {k: 0 for k in KS}
    full = {k: 0 for k in KS}
    cov = {k: 0.0 for k in KS}
    rr = 0.0
    for q in qs:
        top = bm.top(q["query"], max(KS))
        ev_rank = next(
            (
                i + 1
                for i, cid in enumerate(top)
                if chunks[cid]["doc"] in q["docs"]
                and any(k in chunks[cid]["nbody"] for k in q["key"])
            ),
            None,
        )
        dc_rank = next(
            (i + 1 for i, cid in enumerate(top) if chunks[cid]["doc"] in q["docs"]),
            None,
        )
        pr = piece_ranks(top, chunks, q)
        for k in KS:
            hit[k] += bool(ev_rank and ev_rank <= k)
            dhit[k] += bool(dc_rank and dc_rank <= k)
            full[k] += all(r and r <= k for r in pr)
            cov[k] += sum(1 for r in pr if r and r <= k) / len(pr)
        rr += 1 / ev_rank if ev_rank else 0
        per_q.append(
            {
                "config": name,
                "id": q["id"],
                "level": q["level"],
                "근거_순위": ev_rank or "",
                "문서_순위": dc_rank or "",
                **strict_fields(pr),
            }
        )
    n = len(qs)
    row = {
        "설정": name,
        "방식": {
            "fixed": "글자 수",
            "struct": "제목·표 구조",
            "req": "요구사항(dh)",
            "reqstruct": "요구사항+구조",
        }.get(kind, "구조+전처리" if kind.startswith("sx:") else kind),
        "크기": size,
        "메타": "O" if meta else "",
        "청크수": len(chunks),
        "청크길이_중앙값": int(st.median(lens)),
        "청크길이_최대": max(lens),
        **{f"근거@{k}": round(hit[k] / n, 3) for k in KS},
        **{f"문서@{k}": round(dhit[k] / n, 3) for k in KS},
        **{f"완전근거@{k}": round(full[k] / n, 3) for k in KS},
        **{f"조각커버@{k}": round(cov[k] / n, 3) for k in KS},
        "MRR@10": round(rr / n, 3),
        "소요(초)": round(build_s, 1),
    }
    return row, per_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="쉼표로 구분한 설정 이름 (예: F1000,S1000)")
    a = ap.parse_args()
    names = [x for x in (a.only.split(",") if a.only else CONFIGS) if x in CONFIGS]
    docs, rep = load_corpus()
    qs, skipped = load_questions(docs, rep)
    lines = [
        f"문서 {len(docs)}건 (data/{CLEAN.name}), 평가 문항 {len(qs)}개 (정답 근거 있는 문항, 후속 질문은 앞 질문을 붙여 검색)",
        f"제외 {len(skipped)}개: " + (", ".join(skipped) if skipped else "없음"),
        "검색기: 글자 2-gram BM25 (기준선). 근거@k = 상위 k개 청크 중 근거 인용(앞 30자)이 든 청크가 있는 비율, 문서@k = 근거 문서의 청크가 있는 비율",
        "",
    ]
    rows, perq = [], []
    for nm in names:
        print(f"{nm} 실행 중...", flush=True)
        r, p = run_config(nm, docs, qs)
        rows.append(r)
        perq += p
    cols = (
        ["설정", "방식", "크기", "메타", "청크수", "청크길이_중앙값", "청크길이_최대"]
        + [f"근거@{k}" for k in KS]
        + [f"문서@{k}" for k in KS]
        + [f"완전근거@{k}" for k in KS]
        + [f"조각커버@{k}" for k in KS]
        + ["MRR@10", "소요(초)"]
    )
    head = (
        f"{'설정':<8}{'방식':<10}{'크기':>5}{'메타':>3}{'청크수':>7}{'길이중앙':>8}"
        + "".join(f"{'근거@' + str(k):>8}" for k in KS)
        + "".join(f"{'문서@' + str(k):>8}" for k in KS)
        + f"{'MRR':>7}{'초':>6}"
    )
    lines.append(head)
    for r in rows:
        lines.append(
            f"{r['설정']:<8}{r['방식']:<9}{r['크기']:>5}{r['메타']:>3}{r['청크수']:>7}{r['청크길이_중앙값']:>8}"
            + "".join(f"{r[f'근거@{k}']:>8.2f}" for k in KS)
            + "".join(f"{r[f'문서@{k}']:>8.2f}" for k in KS)
            + f"{r['MRR@10']:>7.2f}{r['소요(초)']:>6.0f}"
        )
    # 유형별 근거@5
    levels = sorted({q["level"] for q in qs})
    lines += ["", "유형별 근거@5 (문항 수)"]
    lines.append(
        f"{'설정':<8}"
        + "".join(
            f"{lv + '(' + str(sum(q['level'] == lv for q in qs)) + ')':>16}"
            for lv in levels
        )
    )
    for nm in names:
        cells = []
        for lv in levels:
            rs = [p for p in perq if p["config"] == nm and p["level"] == lv]
            cells.append(
                f"{sum(1 for p in rs if p['근거_순위'] != '' and p['근거_순위'] <= 5) / max(1, len(rs)):>16.2f}"
            )
        lines.append(f"{nm:<8}" + "".join(cells))
    lines += [""] + strict_block(rows, perq, qs, names)
    lines += [""] + bootstrap(perq, qs, names)
    tag = "" if CLEAN.name == "clean" else f"_{CLEAN.name}"
    run = RESULTS / f"{datetime.now():%Y%m%d_%H%M%S}_청킹비교{tag}"
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(run / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["config", "id", "level", "근거_순위", "문서_순위", "조각순위"],
        )
        w.writeheader()
        w.writerows(perq)
    lines += ["", f"저장 폴더: {run}"]
    (run / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
