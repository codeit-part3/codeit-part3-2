"""모델이 읽은 평가표 중 코드 검산을 통과한 것을 모아 verified_tables.json 초안을 만든다 (오프라인 1회 작업).

    uv run python scripts/rerank_rag/build_verified_tables.py

입력: outputs/model_read_*_항목.txt (try_model_table_read.py 결과, 로컬 전용)
검산: try_model_table_read.verify (이름·점수가 표 원문에 있음, 합 100 또는 문서에 적힌 기술 비율, 100 초과 불가, 5개 이상)
출력: engines/rerank_rag/verified_tables.json  (공고 파일명 -> 항목 목록. writer 가 규칙 추출보다 먼저 본다)
원문(HWP)과 직접 대조한 공고는 "원문대조": true 로 표시한다.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engines.rerank_rag import writer as w  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tmr", Path(__file__).with_name("try_model_table_read.py")
)
tmr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmr)

# 이름이 평가기준 문장처럼 길게 읽힌 공고는 원문의 짧은 항목 이름으로 바꾼다 (점수는 그대로).
NAME_FIX: dict[str, dict[int, str]] = {
    "강릉어선": {
        0: "경영상태",
        1: "유사용역분야 수행 실적",
        2: "기술인력 보유상태",
        3: "사업의 이해",
        4: "시스템 구축방안",
        5: "인테리어 구현방안",
        6: "기술 및 기능의 우수성",
        7: "성능시험 및 품질관리",
        8: "사업관리",
        9: "사업지원",
        10: "기타(추가제안 등)",
        11: "가격평가",
    },
    "세종테크노파크": {
        0: "신용평가등급·자본금·매출액",
        1: "유사사업 실적 건수",
        2: "유사사업 실적 금액",
        3: "보유인력(기술등급)",
    },
    "도시계획위원회": {10: "가격평가"},
    "광주문화예술통합플랫폼": {8: "가격평가"},
    "부산국제영화제": {3: "사업이해도"},
}
# 원문(HWP) 평가표와 직접 대조해 맞는 것을 확인한 공고 (파일명 일부)
CHECKED = (
    "해양조사협회",
    "인천광역시_도시계획",
    "수산물사이버직매장",
    "강릉어선",
    "광주과학기술원_실시간",
)

LINE = re.compile(r"^  (.*) \((None|[\d.]+)\)$")


def parse(path: Path):
    """_항목.txt -> [(파일명, 결과, 항목[(name, score)])]"""
    out, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("=== "):
            m = re.match(r"=== (.+?\.txt) \[(.+?)\]", line)
            cur = (m.group(1), m.group(2), [])
            out.append(cur)
        elif cur and (m := LINE.match(line)):
            cur[2].append(
                (
                    m.group(1).strip(),
                    None if m.group(2) == "None" else float(m.group(2)),
                )
            )
    return out


def main() -> int:
    outdir = ROOT / "outputs"
    files = sorted(outdir.glob("model_read_exaone_*_항목.txt"), reverse=True) + sorted(
        outdir.glob("model_read_2*_항목.txt"), reverse=True
    )
    result: dict[str, dict] = {}
    for f in files:
        for name, _res, items in parse(f):
            if name in result or not items:
                continue
            p = Path(w.CORPUS_DIR) / name
            if not p.exists():
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            data = {"items": [{"name": n, "score": s} for n, s in items]}
            ok, why = tmr.verify(data, tmr.pick_blocks(txt), txt)
            if not ok:
                continue
            fix = next((v for k, v in NAME_FIX.items() if k in name), {})
            rows = []
            for k, (n, s) in enumerate(items):
                n = fix.get(k, n)
                n = re.sub(
                    r"^\d+\.\d+\s+", "", n
                )  # '1.1 수행경험' 같은 번호 접두어 제거
                rows.append(
                    {"group": "", "name": n, "score": s, "criteria": "", "elements": ""}
                )
            result[name] = {
                "원문대조": any(c in name for c in CHECKED),
                "검산": why,
                "읽은곳": f.name,
                "items": rows,
            }
    dst = ROOT / "engines/rerank_rag/verified_tables.json"
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"검증된 평가표 {len(result)}곳 -> {dst.relative_to(ROOT)}")
    for k, v in result.items():
        mark = "원문대조" if v["원문대조"] else "코드검산"
        print(f"  [{mark}] {v['검산']:<8} 항목 {len(v['items']):>2}  {k[:48]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
