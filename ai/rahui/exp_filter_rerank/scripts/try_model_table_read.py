"""평가표 모델 읽기 시험 (실험용, writer.py 는 건드리지 않는다).

    uv run python ai/rahui/exp_filter_rerank/scripts/try_model_table_read.py [--levels 부분] [--limit 5] [--dry]

규칙 추출기가 확정으로 못 읽은 공고(기본: 부분)의 배점 표 덩어리만 로컬 모델(config.CHAT_MODEL)에게 주고
항목 이름·점수를 JSON 으로 읽게 한다. 모델 답은 코드로 검산해서 통과한 것만 '채택 가능'으로 센다.
  검산: 항목 5개 이상 · 이름과 점수가 모두 표 원문에 실제로 있음 · 점수 합이 100 이거나 표의 합계와 같고 70 이상
통과 못 하면 지금처럼 부분으로 남는다 (엉뚱한 값이 확정이 되지 않게 하는 장치).
결과: outputs/model_read_<시각>.csv 와 _항목.txt (공고문 내용이 들어 있으니 로컬 전용). --dry 는 모델 없이 프롬프트 크기만 본다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ai.rahui.exp_filter_rerank.rahui_engine import writer as w  # noqa: E402

MAX_CHARS = 6000  # 모델에 주는 표 글자 수 상한 (작은 모델의 문맥 제한)
SYSTEM = (
    "너는 입찰 공고문의 '제안서 평가 배점표'를 읽는 도구다. 아래 표 텍스트에서 평가 항목 이름과 배점을 찾아 "
    'JSON 만 출력하라. 형식: {"total": 표에 적힌 총점 또는 null, "items": [{"name": "항목 이름", "score": 숫자}]}\n'
    "규칙: ① 가장 아래 단계의 세부 평가 줄(점수가 직접 붙은 줄)만 쓴다. 그 위의 대분류·중분류 제목 줄과 그 괄호 안 합계 점수는 쓰지 않는다. "
    "② 이름은 표에 적힌 글자를 그대로 복사하고 요약하거나 고쳐 쓰지 않는다(가), 나) 같은 기호만 빼도 된다). "
    "③ 표에 없는 값을 지어내지 않는다. ④ 소계·합계·총점 줄은 항목이 아니다. 가격평가 항목은 포함한다. "
    "⑤ 점수 구간표(신용평가등급 등급별 점수, 실적 비율별 점수)는 항목이 아니다. 점수가 없으면 null."
)


def _norm(s: str) -> str:
    return re.sub(r"[\s·ㆍ∘○●▪■\-:()（）\[\]]", "", s or "")


def pick_blocks(txt: str) -> str:
    """배점 머리글이 있는 표 중 긴 것부터 모아 MAX_CHARS 안으로 자른다."""
    blocks = []
    for b in w.table_blocks(txt):
        text = "\n".join(b) if isinstance(b, list) else str(b)
        if "배점" in text.split("\n", 1)[0] or "배점" in text[:200]:
            blocks.append(text)
    blocks.sort(key=len, reverse=True)
    out, n = [], 0
    for b in blocks[:3]:
        if n + len(b) > MAX_CHARS and out:
            break
        out.append(b[: MAX_CHARS - n])
        n += len(b)
    return "\n\n".join(out)


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def tech_ratio_stated(n: float, doc: str) -> bool:
    """문서에 기술평가 비율이 n(%)이고 가격이 100-n(%)라고 적혀 있는가 (예: 기술 90%, 가격 10%)."""
    a, b = f"{n:g}", f"{100 - n:g}"
    pats = [
        rf"기술[^\n]{{0,25}}{a}\s*%[^\n]{{0,25}}가격[^\n]{{0,25}}{b}\s*%",
        rf"기술[^\n]{{0,20}}[\(（]{a}[\)）][^\n]{{0,30}}가격[^\n]{{0,20}}[\(（]{b}[\)）]",
        rf"기술[^\n]{{0,12}}비중을\s*100분의\s*{a}",
    ]
    return any(re.search(p, doc) for p in pats)


def verify(data, src: str, doc: str = "") -> tuple[bool, str]:
    """모델이 읽은 결과를 코드로 검산한다 -> (통과 여부, 사유)."""
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return False, "JSON 형식 아님"
    items = data["items"]
    if len(items) < w.MIN_FULL_ITEMS:
        return False, f"항목 {len(items)}개(5개 미만)"
    flat = _norm(src)
    nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", src)}
    total = 0.0
    for i in items:
        name, sc = (i.get("name") or "").strip(), i.get("score")
        if not name or not isinstance(sc, (int, float)):
            return False, "이름 또는 점수 누락"
        if _norm(name)[:8] not in flat:
            return False, f"표에 없는 이름: {name[:15]}"
        if float(sc) not in nums:
            return False, f"표에 없는 점수: {sc}"
        total += float(sc)
    if (
        total > 100.01
    ):  # 평가 점수 합이 100을 넘을 수 없다 (표 여러 개를 합쳐 읽은 경우)
        return False, f"합 {total:g}이 100 초과"
    # 합이 100이 아니면: 문서에 '기술 N% / 가격 (100-N)%' 처럼 기술 비율이 N 이라고 적혀 있을 때만 인정.
    # (표의 소계 칸 숫자와 우연히 같은 경우를 통과시키지 않기 위해 합계 칸 일치는 쓰지 않는다)
    ok = abs(total - 100) < 0.01 or (
        total >= w.MIN_FULL_SUM and tech_ratio_stated(total, doc)
    )
    if ok:
        return True, f"합 {total:g}"
    return False, f"합 {total:g}이 100도 아니고 문서의 기술 비율과도 안 맞음"


_EMPTY_INFO: list[str] = []
_CALL_LOCK = threading.Lock()
_LAST_CALL = 0.0


def call_exaone(src: str) -> str:
    """EXAONE API (OpenAI 호환 chat completions). 키/주소는 환경변수에서만 읽는다."""
    import os
    import urllib.request

    key, base = (
        os.environ.get("EXAONE_API_KEY", ""),
        os.environ.get("EXAONE_BASE_URL", "https://api.exaone.ai/v1").rstrip("/"),
    )
    if not key or not base:
        raise RuntimeError("EXAONE_API_KEY 환경변수가 필요합니다")
    body = {
        "model": os.environ.get("EXAONE_MODEL", "k-exaone_v2.5"),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": src},
        ],
        "temperature": 0,
        "max_tokens": int(
            os.environ.get("EXAONE_MAX_TOKENS", "30000")
        ),  # 추론 토큰 포함 상한
        "reasoning_effort": os.environ.get("EXAONE_EFFORT", "low"),
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    import urllib.error

    global _LAST_CALL
    out = ""
    for attempt in range(4):  # 시간 초과·429 는 기다렸다가 다시 시도 (최대 4번)
        with _CALL_LOCK:  # 요청을 보내는 시각을 1.5초 이상 벌린다 (초당 1회 한도)
            wait = 1.5 - (time.time() - _LAST_CALL)
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.load(r)
            ch = resp["choices"][0]
            out = ch["message"].get("content") or ""
            if not out.strip():  # 본문이 비면 추론 내용에서 JSON 을 찾고, 이유를 남긴다
                rc = ch["message"].get("reasoning_content") or ""
                rt = (resp.get("usage") or {}).get("reasoning_tokens")
                _EMPTY_INFO.append(
                    f"본문 비어 있음 finish={ch.get('finish_reason')} 추론토큰={rt}"
                )
                out = rc
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:160].replace("\n", " ")
            if e.code == 429 and attempt < 3:
                try:
                    sec = float(e.headers.get("Retry-After", "") or 0)
                except ValueError:
                    sec = 0
                time.sleep(min(max(sec, 20), 120))  # 분당 한도면 기다리면 풀린다
                continue
            raise RuntimeError(f"HTTP {e.code} {detail}") from e
        except TimeoutError:
            if attempt >= 1:
                raise
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="부분", help="대상 판정 (쉼표로 여러 개)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N곳만")
    ap.add_argument(
        "--model", default="", help="Ollama 모델 이름 (비우면 config.CHAT_MODEL)"
    )
    ap.add_argument("--api", default="", help="exaone 이면 EXAONE API 사용")
    ap.add_argument("--skip", type=int, default=0, help="앞에서 N곳은 건너뛰고 이어서")
    ap.add_argument(
        "--only", default="", help="번호만 다시 (예: 4,5,7). 번호는 실행 때 [N] 표시"
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="동시에 보낼 수 (EXAONE 만, 기본 1)"
    )
    ap.add_argument("--dry", action="store_true", help="모델 없이 프롬프트 크기만 확인")
    a = ap.parse_args()
    levels = set(a.levels.split(","))

    model = None
    if not a.dry and not a.api:
        from langchain.messages import HumanMessage, SystemMessage

        from ai.rahui.exp_filter_rerank.rahui_engine import baseline

        if a.model:
            from langchain.chat_models import init_chat_model

            model = init_chat_model(
                a.model,
                model_provider="ollama",
                base_url=baseline.OLLAMA_URL,
                temperature=0,
                seed=baseline.SEED,
                num_ctx=baseline.NUM_CTX,
            )
        else:
            model = baseline.make_chat_model()
        nothink = "qwen3" in (a.model or "")

    only = {int(x) for x in a.only.split(",") if x.strip()}
    jobs, done, seen = [], 0, 0
    for p in sorted(Path(w.CORPUS_DIR).glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="replace")
        items, level, _why = w.extract_items(txt)
        if level not in levels:
            continue
        seen += 1
        if seen <= a.skip or (only and seen not in only):
            continue
        if a.limit and done >= a.limit:
            break
        done += 1
        jobs.append((seen, p, txt, items, level))

    def process(job):
        seen, p, txt, items, level = job
        src = pick_blocks(txt)
        row = {
            "파일": p.name,
            "현재판정": level,
            "현재항목수": len(items),
            "표글자수": len(src),
        }
        if a.dry or not src:
            row.update(
                {"결과": "표 없음" if not src else "dry", "사유": "", "모델항목수": ""}
            )
            return row, [], ""
        t0 = time.time()
        raw = ""
        try:
            if a.api == "exaone":
                raw = call_exaone(src)
            else:
                msgs = [
                    SystemMessage(SYSTEM),
                    HumanMessage(src + ("\n/no_think" if nothink else "")),
                ]
                try:
                    r = model.invoke(msgs, format="json")  # JSON 출력 강제
                except TypeError:
                    r = model.invoke(msgs)
                raw = getattr(r, "text", None) or r.content or ""
            data = parse_json(raw)
            err = ""
        except Exception as e:  # 모델 오류는 그 공고만 건너뜀
            data, err = None, f"{type(e).__name__} {str(e)[:60]}"
        ok, why = (
            verify(data, src, txt)
            if data
            else (False, err or (_EMPTY_INFO[-1] if _EMPTY_INFO else "JSON 못 읽음"))
        )
        n = len(data.get("items", [])) if isinstance(data, dict) else 0
        row.update(
            {
                "결과": "채택가능" if ok else "실패",
                "사유": why,
                "모델항목수": n,
                "초": round(time.time() - t0, 1),
            }
        )
        out_lines = [f"=== {p.name} [{row['결과']}] {why}"]
        if not isinstance(data, dict):
            out_lines.append("  [모델 원문] " + raw[:600].replace("\n", " "))
        else:
            out_lines += [
                f"  {i.get('name')} ({i.get('score')})" for i in data.get("items", [])
            ]
        msg = f"[{seen}] {row['결과']:<5} 항목 {n:>2} {row.get('초', ''):>5}초 {why[:28]:<28} {p.name[:30]}"
        print(msg, flush=True)
        return row, out_lines, msg

    workers = max(1, a.workers) if a.api == "exaone" else 1
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(process, jobs))
    else:
        results = [process(j) for j in jobs]
    rows = [r[0] for r in results]
    lines = [x for r in results for x in r[1]]

    ok_n = sum(r["결과"] == "채택가능" for r in rows)
    print(f"\n대상 {len(rows)}곳 중 검산 통과(채택 가능) {ok_n}곳")
    if not a.dry and rows:
        out = ROOT / "outputs"
        out.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        with open(
            out
            / f"model_read_{(a.api or a.model or 'default').replace(':', '-')}_{stamp}.csv",
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            wr = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            wr.writeheader()
            wr.writerows(rows)
        (
            out
            / f"model_read_{(a.api or a.model or 'default').replace(':', '-')}_{stamp}_항목.txt"
        ).write_text("\n".join(lines), encoding="utf-8")
        print(
            f"저장: outputs/model_read_{stamp}.csv, _항목.txt  (원문 대조 후 '채택가능'이 맞는지 확인)"
        )
    elif a.dry:
        print("표 글자 수:", [r["표글자수"] for r in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
