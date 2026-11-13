"""이미 만든 답변(answers.jsonl)을 Groq 모델로 다시 판정한다 (답변은 새로 만들지 않음).

사용:  uv run --no-sync python -m ai.rahui.evaluation.judge_groq --run <답변확인 결과 폴더 이름>
- 대상: 규칙 채점에서 틀린 문항 + 수동확인 문항(q36, q50) (answer_eval.py 와 같은 대상)
- 규칙 관문(확인할 수 없다고 답함 / 정답 숫자 없음)은 answer_eval.py 와 똑같이 먼저 적용
- 키는 환경변수 GROQ_API_KEY (없으면 프로젝트 폴더의 .env). 키는 출력하지 않는다.
- 주의: 질문·정답·답변 텍스트가 Groq 서버로 전송된다 (검색 조각은 보내지 않음).
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from ai.rahui.evaluation.answer_eval import FRAGILE, JUDGE
from ai.rahui.evaluation.golden_eval import load_golden, refused
from ai.rahui.experiments import chunk_compare as cc
from ai.rahui.paths import PROJECT_ROOT

URL = "https://api.groq.com/openai/v1/chat/completions"


def get_key() -> str:
    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        for p in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("GROQ_API_KEY="):
                        k = line.split("=", 1)[1].strip().strip("\"'")
    if not k:
        raise SystemExit("GROQ_API_KEY 가 없습니다. 환경변수나 .env 에 넣어 주세요.")
    return k


def ask_groq(key: str, model: str, messages: list[dict]) -> str:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0}).encode()
    for attempt in range(8):
        req = urllib.request.Request(
            URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "rahui-judge/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = float(e.headers.get("retry-after") or 20)
                print(f"  한도 도달, {wait:.0f}초 기다림 ({attempt + 1}/8)", flush=True)
                time.sleep(min(wait, 300) + 1)
                continue
            raise SystemExit(f"Groq 오류 {e.code}: {e.read()[:300]!r}") from e
    raise SystemExit(
        "한도 때문에 계속할 수 없습니다. 잠시 뒤 같은 명령을 다시 실행하세요."
    )


def b(v):
    s = str(v).strip().lower()
    return True if s in ("1", "true") else False if s in ("0", "false") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", required=True, help="eda_results 안의 답변확인 결과 폴더 이름"
    )
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    a = ap.parse_args()
    run = cc.RESULTS / a.run
    key = get_key()
    gold = {it.id: it for it in load_golden(cc.GOLDEN)}
    ans = {}
    for line in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ans[(r["mode"], r["id"])] = r["answer"]
    with open(run / "per_question.csv", encoding="utf-8-sig") as f_pq:
        rows = list(csv.DictReader(f_pq))
    out, t0 = [], time.time()
    targets = [
        r
        for r in rows
        if r["level"] != "no_answer"
        and (b(r["pass"]) is not True or r["id"] in FRAGILE)
    ]
    print(f"다시 판정할 문항 {len(targets)}개 (모델 {a.model})", flush=True)
    for n, r in enumerate(targets, 1):
        it, answer = gold[r["id"]], ans[(r["mode"], r["id"])]
        if refused(answer):
            v, why = "X", "규칙: '확인할 수 없다'고 답함"
        elif b(r.get("numbers")) is False:
            v, why = "X", "규칙: 정답 숫자가 답변에 없음"
        else:
            msgs = [
                {"role": "system", "content": JUDGE},
                {
                    "role": "user",
                    "content": f"[질문]\n{it.question}\n\n[정답]\n{it.gold_answer}\n\n[답변]\n{answer}",
                },
            ]
            txt = ask_groq(key, a.model, msgs)
            m = re.findall(r"결과\s*[:：]\s*([OX])", txt.upper())
            v, why = (m[-1] if m else "X"), txt[-300:]
        out.append(
            {
                "mode": r["mode"],
                "id": r["id"],
                "규칙통과": int(b(r["pass"]) is True),
                "이전판정(로컬)": r.get("모델판정", ""),
                "groq판정": v,
                "groq근거": why,
            }
        )
        if n % 10 == 0:
            print(f"  {n}/{len(targets)} ({time.time() - t0:.0f}초)", flush=True)

    lines = [f"Groq 재판정 {a.model} / 원본 {a.run} / 대상 {len(out)}개", ""]
    for mode in ("retrieved", "oracle"):
        rs = [r for r in rows if r["mode"] == mode and r["level"] != "no_answer"]
        o = {x["id"]: x for x in out if x["mode"] == mode}
        base = sum(b(r["pass"]) is True for r in rs)
        fin_local = sum(int(r.get("최종") or 0) for r in rs)
        fin_groq = sum(
            1
            for r in rs
            if b(r["pass"]) is True or o.get(r["id"], {}).get("groq판정") == "O"
        )
        both = [x for x in o.values() if x["이전판정(로컬)"] in ("O", "X")]
        same = sum(x["이전판정(로컬)"] == x["groq판정"] for x in both)
        lines.append(
            f"[{mode}] 답이 있는 {len(rs)}개: 규칙 {base}개 / 로컬 판정 반영 {fin_local}개 / Groq 판정 반영 {fin_groq}개"
        )
        lines.append(
            f"    로컬과 Groq 판정이 같은 문항 {same}/{len(both)}개; 다른 문항: "
            + (
                ", ".join(x["id"] for x in both if x["이전판정(로컬)"] != x["groq판정"])
                or "없음"
            )
        )
    lines.append(
        "\n참고: 판정이 갈린 문항은 answers.jsonl 로 사람이 직접 읽어 확인할 것."
    )
    new = cc.RESULTS / f"{datetime.now():%Y%m%d_%H%M%S}_판정재확인_groq"
    new.mkdir(parents=True, exist_ok=True)
    (new / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(new / "judge_compare.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    (new / "run_info.txt").write_text(
        f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}\n원본 폴더: {a.run}\n판정 모델: {a.model} (Groq)\n"
        f"골든셋: {cc.GOLDEN.name}\n전송 내용: 질문·정답·답변 텍스트 (검색 조각은 전송 안 함)\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"저장 폴더: {new}")


if __name__ == "__main__":
    main()
