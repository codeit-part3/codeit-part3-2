"""검색 결과를 LLM(Ollama)에 넘겨 답변을 만들고, 골든셋 정답과 대조해 채점합니다.

실행 (프로젝트 루트, PowerShell):
  $env:GOLDEN_PATH = "...\\rahui_golden_set_v2.json"; $env:CLEAN_DIR = "clean_mixed_ocr_clean"
  uv run --no-sync python -m ai.rahui.evaluation.answer_eval --llm <ollama 모델 이름>

방식: gemma 임베딩 + S1500M 청킹으로 상위 k개 청크를 찾아 LLM에 주고 답을 받는다 (모드 'retrieved').
      같은 질문에 정답 근거 청크(인용 청크 + 같은 문서의 키워드 청크)를 직접 주는 모드('oracle')도 돌려, 검색 탓인지 답변 생성 탓인지 나눈다.
채점: golden_eval.score_item (정답 키워드 전부 + 숫자 값 일치. 답 없음 문항은 거절 표현 여부).
      규칙 채점의 한계가 있는 문항(날짜·금액 표기 차이 등)은 표시만 하고 수동 확인이 필요하다.
주의: 문서 본문이 --ollama-url 이 가리키는 서버로 전송된다. 기본값은 내 컴퓨터(localhost).
출력: ai/rahui/eda_results/<실행 폴더>/ (summary.txt, per_question.csv, answers.jsonl, run_info.txt)
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ai.rahui.evaluation.golden_eval import load_golden, refused, score_item
from ai.rahui.experiments import chunk_compare as cc
from ai.rahui.experiments import embed_compare as ec

SYSTEM = (
    "당신은 공공 입찰 공고문(RFP)에 대한 질문에 답하는 도우미입니다. "
    "아래 [문서 조각]에 있는 내용만 근거로 한국어로 간결하게 답하세요. "
    "문서 조각에서 답을 찾을 수 없으면 추측하지 말고 '문서에서 확인할 수 없습니다'라고만 답하세요. "
    "숫자와 이름은 문서 조각에 적힌 그대로 쓰세요."
)
# 규칙 채점이 믿기 어려운 문항 (정답이 맞아도 표기가 달라 틀린 것으로 나올 수 있음)
FRAGILE = {"q36", "q50"}


def ask_ollama(url: str, model: str, messages: list[dict], num_ctx: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # qwen3 같은 생각(think) 모델의 생각 과정을 끔 (지원 안 하는 버전은 무시됨)
            "options": {"temperature": 0, "num_ctx": num_ctx},
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())["message"]["content"]
    # 생각 과정이 본문에 섞여 나오면 채점(키워드 대조)이 오염되므로 제거
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    if "</think>" in out:  # 여는 태그 없이 닫는 태그만 남는 경우: 그 앞은 생각 과정
        out = out.split("</think>")[-1]
    return out.strip()


JUDGE = (
    "당신은 엄격한 채점자입니다. [질문]에 대한 [정답]과 [답변]을 비교합니다.\n"
    "1) [정답]의 핵심 값(이름·숫자·개수·날짜)을 모두 나열합니다.\n"
    "2) 각 핵심 값이 [답변]에 같은 의미로 들어 있는지 하나씩 확인합니다. 값이 다르거나 하나라도 빠지면 틀린 것입니다.\n"
    "3) '확인할 수 없다'고 답했거나 질문과 다른 것을 답했으면 틀린 것입니다.\n"
    "4) 표기만 다른 것(날짜 형식, 단위, 쉼표, 띄어쓰기)은 같은 것으로 봅니다. 답변이 더 길거나 설명이 덧붙은 것은 상관없습니다.\n"
    "마지막 줄에 '결과: O' 또는 '결과: X' 만 적습니다.\n\n"
    "[예시 1]\n[정답] 데이터 요구사항 11개, 전체 41개\n[답변] 데이터 요구사항은 4개이고 전체는 11개입니다.\n"
    "핵심 값: 11개, 41개. 답변에는 4개와 11개가 있어 값이 다릅니다.\n결과: X\n\n"
    "[예시 2]\n[정답] 사업기간 90일, 예산 50백만원\n[답변] 기간은 계약일부터 90일이며 예산은 5천만 원입니다.\n"
    "핵심 값: 90일, 50백만원. 둘 다 같은 값입니다(표기만 다름).\n결과: O"
)


def judge_answer(
    url: str, model: str, item, answer: str, num_ctx: int, numbers_ok=None
) -> tuple[str, str]:
    """(판정 O/X, 근거). 규칙으로 확실히 틀린 경우는 모델을 부르지 않고 X (모델이 후하게 O를 주는 것 방지)."""
    if refused(answer):
        return "X", "규칙: '확인할 수 없다'고 답함"
    if numbers_ok is False:
        return "X", "규칙: 정답 숫자가 답변에 없음"
    msgs = [
        {"role": "system", "content": JUDGE},
        {
            "role": "user",
            "content": f"[질문]\n{item.question}\n\n[정답]\n{item.gold_answer}\n\n[답변]\n{answer}",
        },
    ]
    out = ask_ollama(url, model, msgs, num_ctx).strip()
    m = re.findall(r"결과\s*[:：]\s*([OX])", out.upper())
    return (m[-1] if m else "X"), out[-300:]


def build_messages(item, chunks: list[dict]) -> list[dict]:
    ctx = "\n\n".join(
        f"[문서 조각 {i + 1}]\n{c['index']}" for i, c in enumerate(chunks)
    )
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in item.history:  # 후속 질문: 앞선 대화
        txt = h.get("content") or h.get("text") or ""
        if h.get("role") in ("user", "assistant") and txt:
            msgs.append({"role": h["role"], "content": txt})
    msgs.append({"role": "user", "content": f"{ctx}\n\n질문: {item.question}"})
    return msgs


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llm", required=True, help="Ollama 모델 이름 (ollama list 로 확인)"
    )
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--config", default="S1500M")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", default="retrieved,oracle")
    ap.add_argument(
        "--ids", default="", help="쉼표로 구분한 문항 id만 실행 (예: q36,q50)"
    )
    ap.add_argument("--embedder", default="gemma", help="gemma (기본). hash 는 점검용")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument(
        "--judge",
        default="",
        help="채점용 Ollama 모델. 규칙 채점에서 틀린 문항과 수동확인 문항만 정답과 같은 뜻인지 다시 판정",
    )
    ap.add_argument("--limit", type=int, default=0, help="앞쪽 N문항만 (시험용)")
    ap.add_argument(
        "--rerank",
        default="",
        help="재순위화 모델 (예: bge-reranker-v2-m3). 비우면 임베딩 순위 그대로",
    )
    ap.add_argument("--pool", type=int, default=20, help="재순위화할 후보 개수")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM을 부르지 않고 첫 청크를 답으로 씀 (점검용)",
    )
    a = ap.parse_args()
    modes = [m for m in a.modes.split(",") if m in ("retrieved", "oracle")]

    docs, rep = cc.load_corpus()
    qs, _ = cc.load_questions(
        docs, rep
    )  # 검색 평가가 가능한 문항 (근거 인용이 코퍼스에 있음)
    qmap = {q["id"]: q for q in qs}
    items = [
        it
        for it in load_golden(cc.GOLDEN)
        if it.ready and (it.id in qmap or it.level == "no_answer")
    ]
    if a.ids:
        want = {x.strip() for x in a.ids.split(",") if x.strip()}
        items = [i for i in items if i.id in want]
    if a.limit:
        items = items[: a.limit]
    print(
        f"문서 {len(docs)}건, 답변 확인 문항 {len(items)}개 (답 없음 {sum(i.level == 'no_answer' for i in items)}개 포함)",
        flush=True,
    )

    kind, size, meta = ec.cc.CONFIGS.get(a.config) or ec.cc.CONFIGS[a.config]
    chunks = cc.build_chunks(docs, kind, size, meta)
    queries = []
    for it in items:
        q = it.question
        users = [h for h in it.history if h.get("role") == "user"]
        if users:  # 후속 질문: 검색 때는 앞 질문을 붙임 (chunk_compare 와 같은 규칙)
            q = (users[-1].get("content") or users[-1].get("text") or "") + " " + q
        queries.append(q)
    emb = ec.Embedder(a.embedder, a.max_tokens, a.batch)
    qe = emb.encode(queries, "q", "answer_queries")
    de = emb.encode([c["index"] for c in chunks], "d", a.config)
    if a.rerank:
        import torch
        from sentence_transformers import CrossEncoder

        from ai.rahui.experiments.rerank_compare import RERANKERS

        cand = [np.argsort(-s)[: a.pool].tolist() for s in qe @ de.T]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  리랭커 불러오는 중: {RERANKERS[a.rerank]} (장치: {dev})", flush=True)
        ce = CrossEncoder(RERANKERS[a.rerank], device=dev, max_length=1024)
        pairs = [
            (queries[qi], chunks[i]["index"]) for qi, cs in enumerate(cand) for i in cs
        ]
        sc = np.asarray(ce.predict(pairs, batch_size=8, show_progress_bar=True))
        top, pos = [], 0
        for cs in cand:
            s_ = sc[pos : pos + len(cs)]
            pos += len(cs)
            top.append([cs[j] for j in np.argsort(-s_)[: a.k]])
        del ce
        if dev == "cuda":
            torch.cuda.empty_cache()
    else:
        top = [np.argsort(-s)[: a.k].tolist() for s in qe @ de.T]

    rows, answers = [], []
    t0 = time.time()
    for mode in modes:
        for n, (it, tp) in enumerate(zip(items, top, strict=True)):
            q = qmap.get(it.id)
            if mode == "oracle":
                if q is None:
                    continue  # 답 없음 문항은 정답 근거 청크가 없음
                # 인용 조각이 든 청크를 먼저, 이어서 증거 문서 안에서 정답 키워드·값이 든 청크를 채움
                # (인용 청크만 주면 PER-004 같은 키워드가 다른 청크에 있을 때 답할 수 없음)
                ev = [c for c in chunks if c["doc"] in q["docs"]]
                kws = [
                    cc.norm(str(k))
                    for k in (*it.keywords, *it.answer_values)
                    if str(k).strip()
                ]
                first = [c for c in ev if any(k in c["nbody"] for k in q["key"])]
                rest = [
                    c
                    for c in ev
                    if c not in first and any(k in c["nbody"] for k in kws)
                ]
                sel = (first + rest)[: a.k]
                if not sel:
                    continue
            else:
                sel = [chunks[i] for i in tp]
            if a.dry_run:
                ans = sel[0]["body"][:200]
            else:
                ans = ask_ollama(
                    a.ollama_url, a.llm, build_messages(it, sel), a.num_ctx
                )
            res = {"answer": ans, "retrieved_docs": [c["doc"] for c in sel]}
            row = score_item(it, res, a.k)
            row["mode"] = mode
            if q is not None and mode == "retrieved":
                row["근거적중"] = int(
                    any(
                        c["doc"] in q["docs"] and any(k in c["nbody"] for k in q["key"])
                        for c in sel
                    )
                )
            row["수동확인"] = int(it.id in FRAGILE)
            row["최종"] = int(row["pass"])
            if (
                a.judge
                and not a.dry_run
                and it.level != "no_answer"
                and (not row["pass"] or it.id in FRAGILE)
            ):
                row["모델판정"], row["판정근거"] = judge_answer(
                    a.ollama_url, a.judge, it, ans, a.num_ctx, row.get("numbers")
                )
                row["최종"] = int(row["pass"] or row["모델판정"] == "O")
            rows.append(row)
            answers.append({"mode": mode, "id": it.id, "answer": ans})
            if (n + 1) % 10 == 0:
                print(
                    f"  {mode}: {n + 1}/{len(items)} ({time.time() - t0:.0f}초)",
                    flush=True,
                )

    # 요약
    lines = [
        f"LLM {a.llm} / 검색 {a.embedder}+{a.config}"
        + (f"+{a.rerank}@{a.pool}" if a.rerank else "")
        + f" 상위 {a.k} / 문항 {len(items)}개",
        "",
    ]
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        ans_rs = [r for r in rs if r["level"] != "no_answer"]
        no_rs = [r for r in rs if r["level"] == "no_answer"]
        lines.append(
            f"[{mode}] 답이 있는 문항 {len(ans_rs)}개: 정답 {sum(r['pass'] for r in ans_rs)}개 ({sum(r['pass'] for r in ans_rs) / max(1, len(ans_rs)):.0%})"
        )
        if a.judge:
            judged = [r for r in ans_rs if "모델판정" in r]
            lines.append(
                f"    모델 판정 반영: 규칙에서 틀린 문항 등 {len(judged)}개를 다시 판정 -> 최종 정답 {sum(r['최종'] for r in ans_rs)}개 ({sum(r['최종'] for r in ans_rs) / max(1, len(ans_rs)):.0%})"
            )
        if no_rs:
            lines.append(
                f"[{mode}] 답이 없는 문항 {len(no_rs)}개: '확인할 수 없다'고 답한 것 {sum(r['pass'] for r in no_rs)}개"
            )
        if mode == "retrieved":
            hit = [r for r in ans_rs if r.get("근거적중") == 1]
            miss = [r for r in ans_rs if r.get("근거적중") == 0]
            lines.append(
                f"    근거 청크가 상위 {a.k}에 있던 문항 {len(hit)}개 중 정답 {sum(r['pass'] for r in hit)}개, 없던 문항 {len(miss)}개 중 정답 {sum(r['pass'] for r in miss)}개"
            )
        by = defaultdict(list)
        for r in ans_rs:
            by[r["level"]].append(r["pass"])
        lines.append(
            "    유형별: "
            + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by.items()))
        )
        lines.append("")
    lines.append(
        "주의: 규칙 채점(키워드·숫자 일치)이라 표기가 다른 정답은 틀린 것으로 나올 수 있음. per_question.csv 의 수동확인=1 문항(q36, q50)과 오답 문항은 answers.jsonl 로 직접 확인할 것."
    )
    print("\n".join(lines))

    tag = "" if cc.CLEAN.name == "clean" else f"_{cc.CLEAN.name}"
    run = cc.RESULTS / f"{datetime.now():%Y%m%d_%H%M%S}_답변확인{tag}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    cols = [
        "mode",
        "id",
        "level",
        "category",
        "pass",
        "keyword",
        "numbers",
        "근거적중",
        "수동확인",
        "모델판정",
        "판정근거",
        "최종",
        "recall",
        "hit_at_k",
    ]
    with open(run / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(run / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in answers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = [
        f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S} (걸린 시간 {time.time() - t0:.0f}초)",
        "실행 명령: python -m ai.rahui.evaluation.answer_eval "
        + " ".join(sys.argv[1:]),
        f"LLM: {a.llm} (Ollama {a.ollama_url}), temperature 0, num_ctx {a.num_ctx}"
        + (" [dry-run]" if a.dry_run else ""),
        f"검색: {a.embedder} + {a.config}"
        + (f" + 재순위화 {a.rerank}(후보 {a.pool})" if a.rerank else "")
        + f", 상위 {a.k}, 최대 {a.max_tokens}토큰",
        f"모드: {', '.join(modes)}",
        "채점: 규칙" + (f" + 모델 판정({a.judge})" if a.judge else ""),
        f"문서 {len(docs)}건, 답변 확인 문항 {len(items)}개",
        f"CLEAN 폴더: {cc.CLEAN}",
        f"골든셋 md5: {md5(cc.GOLDEN)} ({cc.GOLDEN.name})",
        f"answer_eval.py md5: {md5(Path(__file__))}",
        f"embed_compare.py md5: {md5(Path(ec.__file__))}",
        f"chunk_compare.py md5: {md5(Path(cc.__file__))}",
        f"python {platform.python_version()} / {platform.platform()}",
    ]
    (run / "run_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print(f"저장 폴더: {run}")


if __name__ == "__main__":
    main()
