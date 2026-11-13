"""청킹 설정별 임베딩 검색 성능을 비교합니다.

실행:
  uv run --no-sync python -m ai.rahui.embed_compare --models gemma

입력: data/<CLEAN_DIR>/, ai/rahui/rahui_golden_set.json
출력: data/embed_cache/, ai/rahui/eda_results/<실행 폴더>/
OpenAI 모델은 --allow-external 지정 시 문서 본문을 API로 전송합니다.
"""

import argparse
import csv
import hashlib
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

from ai.rahui import chunk_compare as cc
from ai.rahui.convert import DATA

CACHE = DATA / "embed_cache"
MODELS = {
    "bge-m3": {"hf": "BAAI/bge-m3", "q": "", "d": ""},
    "kure-v1": {"hf": "nlpai-lab/KURE-v1", "q": "", "d": ""},
    "gemma": {
        "hf": "google/embeddinggemma-300m",
        "q": "task: search result | query: ",
        "d": "title: none | text: ",
    },
    "hash": {"hf": None, "q": "", "d": ""},
    # OpenAI API (유료, 문서 본문이 OpenAI 서버로 전송됨). usd = 100만 토큰당 달러 (OpenAI 모델 안내 페이지, 2026-11-12 확인)
    "openai-small": {"api": "text-embedding-3-small", "usd": 0.02, "q": "", "d": ""},
    "openai-large": {"api": "text-embedding-3-large", "usd": 0.13, "q": "", "d": ""},
}
ALLOW_EXTERNAL = False  # --allow-external 을 줘야 API 로 보냄
# chunk_compare 의 설정에 길이를 달리한 메타 헤더 설정을 더함
cc.CONFIGS.update(
    {
        "F1500M": ("fixed", 1500, True),
        "S1500M": ("struct", 1500, True),
        "S600M": ("struct", 600, True),
    }
)


def hash_embed(texts: list[str], dim: int = 512) -> np.ndarray:
    """모델 없이 파이프라인을 점검하기 위한 가짜 임베딩 (글자 2-gram 해시)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in cc.tokens(t):
            out[i, int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % dim] += 1
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


class Embedder:
    def __init__(self, name: str, max_tokens: int, batch: int):
        self.name, self.spec, self.batch = name, MODELS[name], batch
        self.model = None
        self.device = "없음(hash)"
        if self.spec.get("api"):
            self.device = "OpenAI API"
        elif self.spec["hf"]:
            import torch
            from sentence_transformers import SentenceTransformer

            dev = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                    else "cpu"
                )
            )
            print(
                f"  모델 불러오는 중: {self.spec['hf']} (장치: {dev}, 최대 {max_tokens}토큰)",
                flush=True,
            )
            self.device = dev
            self.model = SentenceTransformer(self.spec["hf"], device=dev)
            self.model.max_seq_length = max_tokens

    def api_embed(self, texts: list[str]) -> np.ndarray:
        """OpenAI 임베딩 API 호출. 문서 본문이 외부로 나가므로 --allow-external 이 있어야만 실행."""
        if not ALLOW_EXTERNAL:
            sys.exit(
                "OpenAI 임베딩은 문서 본문을 OpenAI 서버로 보냅니다. 사용해도 되는지 확인한 뒤 --allow-external 을 붙여 다시 실행하세요."
            )
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit(
                "환경변수 OPENAI_API_KEY 가 없습니다. 키를 코드나 명령어에 적지 말고 환경변수로만 설정하세요."
            )
        from openai import OpenAI

        client, out, step = OpenAI(api_key=key), [], 64
        for i in range(0, len(texts), step):
            for attempt in range(6):
                try:
                    r = client.embeddings.create(
                        model=self.spec["api"],
                        input=texts[i : i + step],
                        encoding_format="float",
                    )
                    break
                except Exception as e:  # 속도 제한 등은 기다렸다가 다시
                    if attempt == 5:
                        raise
                    print(
                        f"  API 오류({type(e).__name__}) {2**attempt}초 뒤 재시도",
                        flush=True,
                    )
                    time.sleep(2**attempt)
            out += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
            if (i // step) % 20 == 0:
                print(
                    f"  {self.name}: {min(i + step, len(texts))}/{len(texts)}",
                    flush=True,
                )
        emb = np.array(out, dtype=np.float32)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    def encode(self, texts: list[str], kind: str, tag: str) -> np.ndarray:
        """kind: 'q' 질문 / 'd' 문서(청크). 같은 입력은 캐시에서 재사용."""
        pre = self.spec[kind]
        h = hashlib.md5(
            (
                "\n".join(texts) + pre + str(getattr(self.model, "max_seq_length", ""))
            ).encode()
        ).hexdigest()[:12]
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{self.name}_{tag}_{h}.npy"
        if f.exists():
            return np.load(f)
        t0 = time.time()
        if self.spec.get("api"):
            emb = self.api_embed([pre + t for t in texts])
        elif self.model is None:
            emb = hash_embed([pre + t for t in texts])
        else:
            order = sorted(
                range(len(texts)), key=lambda i: len(texts[i])
            )  # 길이순으로 묶어 패딩 낭비를 줄임
            res = self.model.encode(
                [pre + texts[i] for i in order],
                batch_size=self.batch,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            emb = np.zeros_like(res)
            emb[order] = res
        emb = emb.astype(np.float32)
        np.save(f, emb)
        print(
            f"  임베딩 {len(texts)}개 {time.time() - t0:.0f}초 -> {f.name}", flush=True
        )
        return emb


def evaluate(
    label: str, chunks: list[dict], qs: list[dict], ranker
) -> tuple[dict, list[dict]]:
    """ranker(q_index, query) -> 상위 10개 청크 번호. chunk_compare.run_config 와 같은 지표."""
    hit = {k: 0 for k in cc.KS}
    rr, per_q = 0.0, []
    for qi, q in enumerate(qs):
        top = ranker(qi, q["query"])
        ev = next(
            (
                i + 1
                for i, cid in enumerate(top)
                if chunks[cid]["doc"] in q["docs"]
                and any(k in chunks[cid]["nbody"] for k in q["key"])
            ),
            None,
        )
        for k in cc.KS:
            hit[k] += bool(ev and ev <= k)
        rr += 1 / ev if ev else 0
        per_q.append(
            {
                "config": label,
                "id": q["id"],
                "level": q["level"],
                "근거_순위": ev or "",
                "문서_순위": "",
            }
        )
    n = len(qs)
    return {
        "설정": label,
        **{f"근거@{k}": round(hit[k] / n, 3) for k in cc.KS},
        "MRR@10": round(rr / n, 3),
    }, per_q


def _md5(path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return "읽지 못함"


def run_info(a, models, cfgs, devices, n_docs, n_qs, skipped, seconds) -> list[str]:
    """이 실행을 다시 재현하는 데 필요한 조건을 적는다 (결과 폴더의 run_info.txt)."""
    from pathlib import Path

    here = Path(__file__).resolve().parent
    out = [
        f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S} (걸린 시간 {seconds:.0f}초)",
        "실행 명령: python -m ai.rahui.embed_compare " + " ".join(sys.argv[1:]),
        "모델: "
        + ", ".join(
            m + " (" + (MODELS[m].get("api") or MODELS[m]["hf"] or "hash") + ")"
            for m in models
        ),
        f"청킹 설정: {', '.join(cfgs)}",
        f"최대 토큰: {a.max_tokens}",
        f"배치 크기: {a.batch}",
        f"하이브리드(RRF) 계산: {'안 함' if a.no_hybrid else '함'}",
        "장치: " + ", ".join(m + "=" + d for m, d in devices.items()),
        f"문서 {n_docs}건, 평가 문항 {n_qs}개, 제외 {len(skipped)}개 ({', '.join(s.split()[0] for s in skipped) or '없음'})",
        f"CLEAN 폴더: {cc.CLEAN}",
        f"골든셋 md5: {_md5(cc.GOLDEN)} ({cc.GOLDEN.name})",
        f"embed_compare.py md5: {_md5(here / 'embed_compare.py')}",
        f"chunk_compare.py md5: {_md5(here / 'chunk_compare.py')}",
        f"python {platform.python_version()} / {platform.platform()}",
    ]
    for pkg in ("torch", "sentence_transformers", "transformers", "numpy"):
        try:
            out.append(f"{pkg} {__import__(pkg).__version__}")
        except Exception:  # 설치 안 된 패키지는 건너뜀 (hash 모델 점검 때)
            out.append(f"{pkg} 없음")
    try:
        import torch

        if torch.cuda.is_available():
            out.append(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", default="kure-v1", help="쉼표 구분: bge-m3,kure-v1,gemma,hash"
    )
    ap.add_argument(
        "--configs",
        default="F1000M,S1000M,F1500M,S1500M",
        help="chunk_compare 의 설정 이름",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="청크 하나당 최대 토큰 (클수록 느림)",
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument(
        "--allow-external",
        action="store_true",
        help="OpenAI 같은 외부 API 로 문서 본문을 보내는 것을 허용",
    )
    ap.add_argument(
        "--estimate",
        action="store_true",
        help="API 모델의 토큰 수와 예상 비용만 계산하고 끝냄 (키·전송 없음, tiktoken 필요)",
    )
    a = ap.parse_args()
    global ALLOW_EXTERNAL
    ALLOW_EXTERNAL = a.allow_external
    models = [m for m in a.models.split(",") if m in MODELS]
    cfgs = [c for c in a.configs.split(",") if c in cc.CONFIGS]
    docs, rep = cc.load_corpus()
    qs, skipped = cc.load_questions(docs, rep)
    print(
        f"문서 {len(docs)}건, 평가 문항 {len(qs)}개 (제외 {len(skipped)}개)", flush=True
    )
    queries = [q["query"] for q in qs]
    rows, perq = [], []
    devices, t_start = {}, time.time()
    built = {}
    for c in cfgs:
        kind, size, meta = cc.CONFIGS[c]
        built[c] = cc.build_chunks(docs, kind, size, meta)
        print(f"{c}: 청크 {len(built[c])}개", flush=True)
    if a.estimate:
        import tiktoken

        enc = tiktoken.get_encoding(
            "cl100k_base"
        )  # text-embedding-3 계열이 쓰는 토크나이저

        def ntok(ts):
            return sum(len(enc.encode(x)) for x in ts)

        qt = ntok(queries)
        per = {c: ntok([x["index"] for x in built[c]]) for c in cfgs}
        for c in cfgs:
            print(f"{c}: 청크 {len(built[c])}개, 약 {per[c]:,} 토큰")
        tot = qt + sum(per.values())
        print(
            f"질문 {len(queries)}개 {qt:,} 토큰, 합계 {tot:,} 토큰 (캐시가 없을 때 기준)"
        )
        for m in models:
            if MODELS[m].get("api"):
                print(
                    f"  {m}: 약 ${tot / 1e6 * MODELS[m]['usd']:.2f}  (${MODELS[m]['usd']}/100만 토큰)"
                )
        return
    for m in models:
        emb = Embedder(m, a.max_tokens, a.batch)
        devices[m] = emb.device
        qe = emb.encode(queries, "q", "queries")
        for c in cfgs:
            chunks = built[c]
            de = emb.encode([x["index"] for x in chunks], "d", c)
            sims = qe @ de.T  # (문항, 청크) 코사인 (정규화된 벡터)
            dense_top = [np.argsort(-s)[:100] for s in sims]
            label = f"{c}|{m}"
            r, p = evaluate(
                label,
                chunks,
                qs,
                lambda qi, _q, dt=dense_top: dt[qi][:10].tolist(),
            )
            rows.append(r)
            perq += p
            if not a.no_hybrid:
                bm = cc.BM25([x["index"] for x in chunks])
                bm_top = [bm.top(q, 100) for q in queries]

                def rrf(qi, _q, dense_top=dense_top, bm_top=bm_top):
                    sc: dict[int, float] = {}
                    for lst in (dense_top[qi].tolist(), bm_top[qi]):
                        for rank, cid in enumerate(lst):
                            sc[cid] = sc.get(cid, 0) + 1 / (60 + rank + 1)
                    return [
                        cid for cid, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:10]
                    ]

                r, p = evaluate(f"{label}+BM25", chunks, qs, rrf)
                rows.append(r)
                perq += p
    # BM25 기준선 (같은 설정, 비교용)
    for c in cfgs:
        chunks = built[c]
        bm = cc.BM25([x["index"] for x in chunks])
        r, p = evaluate(f"{c}|bm25", chunks, qs, lambda qi, q, bm=bm: bm.top(q, 10))
        rows.append(r)
        perq += p
    names = [r["설정"] for r in rows]
    lines = [
        f"문서 {len(docs)}건, 평가 문항 {len(qs)}개 (제외 {len(skipped)}개: {', '.join(s.split()[0] for s in skipped) or '없음'})",
        f"모델 {models}, 설정 {cfgs}, 최대 {a.max_tokens}토큰",
        "",
    ]
    lines.append(
        f"{'설정|검색기':<26}"
        + "".join(f"{'근거@' + str(k):>8}" for k in cc.KS)
        + f"{'MRR':>7}"
    )
    for r in sorted(rows, key=lambda r: -r["근거@5"]):
        lines.append(
            f"{r['설정']:<26}"
            + "".join(f"{r[f'근거@{k}']:>8.2f}" for k in cc.KS)
            + f"{r['MRR@10']:>7.2f}"
        )
    # 부트스트랩: 모델별 최고 설정 vs BM25, 그리고 모델 간
    best = sorted(rows, key=lambda r: -r["근거@5"])
    top1 = best[0]["설정"]
    base = max(
        (r for r in rows if r["설정"].endswith("|bm25")), key=lambda r: r["근거@5"]
    )["설정"]
    cc.PAIRS = [(top1, base)] + [(top1, r["설정"]) for r in best[1:4]]
    cc.PAIRS += [(f"{c}|{models[0]}", f"{c}|bm25") for c in cfgs] + [
        (f"{c}|{models[0]}+BM25", f"{c}|{models[0]}") for c in cfgs if not a.no_hybrid
    ]
    seen, uniq = set(), []
    for pr in cc.PAIRS:
        if pr not in seen and pr[::-1] not in seen and pr[0] != pr[1]:
            seen.add(pr)
            uniq.append(pr)
    cc.PAIRS = uniq
    lines += [""] + cc.bootstrap(perq, qs, names)
    lines += [
        "",
        "참고: 근거@k = 상위 k개 청크 중 정답 근거가 든 청크가 있는 비율. 'X+BM25' = 임베딩과 BM25를 RRF로 섞은 하이브리드. 'bm25' = 키워드 기준선.",
    ]
    tag = "" if cc.CLEAN.name == "clean" else f"_{cc.CLEAN.name}"
    run = cc.RESULTS / f"{datetime.now():%Y%m%d_%H%M%S}_임베딩비교{tag}"
    run.mkdir(parents=True, exist_ok=True)
    cols = ["설정"] + [f"근거@{k}" for k in cc.KS] + ["MRR@10"]
    with open(run / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(run / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["config", "id", "level", "근거_순위", "문서_순위"]
        )
        w.writeheader()
        w.writerows(perq)
    lines += ["", f"저장 폴더: {run}"]
    (run / "run_info.txt").write_text(
        "\n".join(
            run_info(
                a,
                models,
                cfgs,
                devices,
                len(docs),
                len(qs),
                skipped,
                time.time() - t_start,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
