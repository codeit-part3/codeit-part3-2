"""변환된 RFP 문서의 통계와 그래프를 만듭니다.

실행:
  uv run python -m ai.rahui.eda_stats
  uv run python -m ai.rahui.eda_stats --deep --name 청킹_표

입력: data/data_list.csv, data/processed/*.txt
출력: ai/rahui/eda_results/<실행 폴더>/ (report.html, summary.txt, doc_stats.csv)
"""

import argparse
import csv
import hashlib
import html
import math
import re
import statistics as st
from collections import Counter
from datetime import datetime
from pathlib import Path

from ai.rahui.convert import DATA, OUT, nonspace_len, read_rows

RESULTS = Path(__file__).resolve().parent / "eda_results"

# 공통으로 나올 만한 항목. 문서마다 표현이 달라 정규식으로 느슨하게 찾습니다.
SECTIONS = {
    "사업개요": r"사업\s*개요|과업\s*개요|용역\s*개요|^[Ⅰ-Ⅹ\d.\s]*개요\s*$",
    "제안요청": r"제안\s*요청",
    "요구사항": r"요구\s*사항",
    "평가기준": r"평가\s*(기준|항목|방법)|배점",
    "참가자격": r"참가\s*자격|입찰\s*참가",
    "제출서류": r"제출\s*서류|제출\s*목록|구비\s*서류|입찰\s*(?:등록\s*)?서류|제안서\s*(?:등의\s*|작성\s*및\s*)?제출",
    "사업기간": r"사업\s*기간|과업\s*(?:수행\s*)?기간|계약\s*기간|수행\s*기간|용역\s*기간",
    "사업예산": r"사업\s*비|예산|추정\s*가격|사업\s*금액|용역\s*금액",
    "보안": r"보안",
    "유지보수": r"유지\s*보수|하자\s*보수",
    "계약방법": r"협상에\s*의한|계약\s*방법|낙찰\s*자",
}

ORG_RULES = [  # 위에서부터 먼저 맞는 것으로 분류 (키워드 기반 추정)
    ("대학·교육", r"대학|학교|교육청|교육과정|교육"),
    (
        "지자체",
        r"광역시|특별시|특별자치|도청|시청|군청|구청|^경기도|^경상|^전라|^충청|^강원|^제주|^[가-힣]+시$|^[가-힣]+군$",
    ),
    ("공사·공단", r"공사|공단"),
    (
        "연구·진흥·재단",
        r"연구원|연구소|연구재단|진흥|재단|센터|협회|협의회|위원회|조합|회의소",
    ),
]

PAGE_NO = re.compile(r"[\s.·…]\d{1,3}\s*$")  # 목차 줄은 끝에 쪽 번호가 붙음


def heading_pos(text: str, pat: str) -> int | None:
    """항목 이름이 들어 있는 '제목 줄'의 첫 위치. 목차 줄(끝에 쪽 번호)과 긴 줄은 건너뜀."""
    rx = re.compile(pat)
    pos = 0
    for line in text.split("\n"):
        body = line.strip()
        if body and len(body) <= 60 and rx.search(body) and not PAGE_NO.search(body):
            return pos
        pos += len(line) + 1
    return None


# ---- 엄격 제목 줄 판정 (eda_validate 12+10건 채점: 첫 줄 정확도 느슨 약 62% -> 엄격 약 85~89%)
_MARK = re.compile(
    r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.\s]|제\s*\d+\s*[장절조]|\d{1,2}(?:\.\d{1,2}){0,3}[.)\s]"
    r"|[가-하][.)]|[①-⑩]|\(\s*[\d가-하A-Za-z]{1,2}\s*\)|[□■○●◆◇▣▶▷◎]\s)"
)
_SENT_END = re.compile(
    r"(?:다|함|음|됨|임)\.?\s*$|[을를이가은는에의와과로]\s*$"
)  # '개요'의 요는 문장 끝이 아님
_SENT_MID = re.compile(
    r"하며|하여|에\s*의하여|에\s*따라|합니다|한다|할\s*수|되어|으며|이며|에서는|으로\s*적용|체결시|,\s|,[가-힣]|[“”\"]"
)
_NUMCELL = re.compile(
    r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]+|\d{1,2}|[가-하A-Za-z])[.)]?$"
)  # 'Ⅰ', '1', '가.' 같은 번호 칸
_KEYCELL = re.compile(
    r"^(?:요구사항\s*명(?:칭)?|(?:사업자\s*)?선정\s*방식|계약\s*방법|추진\s*방법|사업\s*기간|사업\s*예산|사업명)$"
)
_KV = re.compile(
    r"^[^:：]{1,15}[:：]\s*\S"
)  # '사업예산 : 금…' 처럼 짧은 이름 뒤에 값이 오는 줄


def heading_cell(line: str, rx: re.Pattern, lineno: int = 99) -> str | None:
    """줄(표 줄이면 칸 단위)에서 '제목처럼 생긴' 칸을 찾아 돌려줌. 없으면 None."""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if (
        not cells or len(cells) > 3 or line.lstrip()[:1] in "-‑·ㆍ*"
    ):  # 하이픈·점 불릿은 본문 항목
        return None
    for k, c in enumerate(cells):
        if len(c) > 50 or PAGE_NO.search(c) or not rx.search(c):
            continue
        if lineno < 3 and re.search(r"제안요청서|입찰공고", c):  # 문서 제목 줄
            continue
        body = re.sub(
            r"\([^)]*\)", "", c
        )  # 괄호 안(금액의 쉼표 등)은 문장 판정에서 제외
        if _SENT_END.search(c) or _SENT_MID.search(body):
            continue
        if k < len(cells) - 1 and re.fullmatch(r"\d{1,3}", cells[-1]):
            continue  # '제목 | 16' 처럼 뒤에 쪽 번호가 붙은 목차 줄
        if not all(
            _NUMCELL.match(o) or _KEYCELL.match(o)
            for j, o in enumerate(cells)
            if j != k
        ):
            continue  # 나머지 칸은 'Ⅰ', '1', '가' 같은 번호만 허용
        key = re.split(r"[:：]", c)[0]
        if _MARK.search(c) or len(c) <= 25 or (_KV.match(c) and rx.search(key)):
            return c
    return None


def strict_heading(text: str, pat: str) -> tuple[int, str, int] | None:
    """(줄 번호, 줄 내용, 글자 위치). 번호·기호가 붙었거나 짧은 '제목처럼 생긴' 줄만 인정."""
    rx = re.compile(pat)
    pos = 0
    for i, line in enumerate(text.split("\n")):
        b = line.strip()
        if b and len(b) <= 120 and heading_cell(b, rx, i):
            return i, b, pos
        pos += len(line) + 1
    return None


def heading_pos_strict(text: str, pat: str) -> int | None:
    h = strict_heading(text, pat)
    return h[2] if h else None


BAR = "█"
_lines: list[str] = []
_charts: list[tuple[str, list[str], list[float], str]] = []
_tables: list[tuple[str, list[str], list[list[object]]]] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def dist_line(vals: list[float], unit: str = "") -> str:
    return (
        f"최소 {min(vals):,.0f}{unit} | 25% {pct(vals, 0.25):,.0f}{unit} | "
        f"중앙값 {st.median(vals):,.0f}{unit} | 75% {pct(vals, 0.75):,.0f}{unit} | "
        f"최대 {max(vals):,.0f}{unit}"
    )


def add_chart(title: str, labels: list[str], values: list[float], unit: str) -> None:
    _charts.append((title, labels, values, unit))


def hist(
    title: str,
    vals: list[float],
    edges: list[float],
    labels: list[str],
    unit: str = "건",
) -> None:
    counts = [0] * len(labels)
    for v in vals:
        for i in range(len(labels)):
            if v < edges[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    top = max(counts) or 1
    for lab, c in zip(labels, counts, strict=True):
        bar = BAR * (max(1, round(40 * c / top)) if c else 0)
        say(f"  {lab:>14} | {bar} {c}")
    add_chart(title, labels, [float(c) for c in counts], unit)


def norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


def parse_dt(s: str):
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
    except ValueError:
        return None


def org_type(org: str) -> str:
    for label, pat in ORG_RULES:
        if re.search(pat, org):
            return label
    return "기타"


def load_docs() -> list[dict]:
    docs = []
    for no, row in enumerate(read_rows(), 1):
        name = row["파일명"]
        txt = OUT / f"{Path(name).stem}.txt"
        text = txt.read_text(encoding="utf-8") if txt.exists() else None
        try:
            amount = float(row["사업 금액"]) if row["사업 금액"].strip() else None
        except ValueError:
            amount = None
        docs.append(
            {
                "no": no,
                "row": row,
                "ext": row["파일형식"],
                "org": row["발주 기관"],
                "amount": amount,
                "text": text,
            }
        )
    return docs


# ------------------------------------------------------------------ HTML 보고서
def _fmt(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def svg_bars(labels: list[str], values: list[float], unit: str) -> str:
    width, row, left, right = 760, 26, 150, 100
    top = max(values) if values and max(values) > 0 else 1
    height = row * len(labels) + 8
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for i, (lab, v) in enumerate(zip(labels, values, strict=True)):
        y = i * row + 4
        bw = (width - left - right) * v / top
        parts.append(
            f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" font-size="13">{html.escape(lab)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y + 3}" width="{bw:.1f}" height="17" rx="2" fill="#3b7dd8"/>'
        )
        parts.append(
            f'<text x="{left + bw + 6:.1f}" y="{y + 16}" font-size="13">{_fmt(v)}{html.escape(unit)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def write_report(path: Path, title: str, info: list[str]) -> None:
    css = (
        "body{font-family:'Malgun Gothic',sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#222;background:#fff}"
        "h1{font-size:22px}h2{font-size:17px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}"
        "table{border-collapse:collapse;font-size:13px;margin:8px 0}"
        "td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}th{background:#f3f3f3}"
        "pre{background:#f6f6f6;padding:12px;font-size:12px;overflow-x:auto}.meta{color:#666;font-size:13px}"
    )
    body = [f"<h1>{html.escape(title)}</h1>"]
    body += [f'<p class="meta">{html.escape(x)}</p>' for x in info]
    for t_title, header, rows in _tables:
        body.append(f"<h2>{html.escape(t_title)}</h2><table><tr>")
        body += [f"<th>{html.escape(h)}</th>" for h in header]
        body.append("</tr>")
        for r in rows:
            body.append(
                "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
            )
        body.append("</table>")
    for c_title, labels, values, unit in _charts:
        body.append(f"<h2>{html.escape(c_title)}</h2>")
        body.append(svg_bars(labels, values, unit))
    body.append("<h2>요약 (summary.txt)</h2>")
    body.append(f"<pre>{html.escape(chr(10).join(_lines))}</pre>")
    path.write_text(
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
        f"<style>{css}</style></head><body>{''.join(body)}</body></html>",
        encoding="utf-8",
    )


def make_run_dir(name: str) -> Path:
    key = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", name).strip("_") or "eda"
    base = f"{datetime.now():%Y%m%d_%H%M%S}_{key}"  # noqa: DTZ005
    run = RESULTS / base
    n = 2
    while run.exists():
        run = RESULTS / f"{base}_{n}"
        n += 1
    run.mkdir(parents=True)
    return run


PUA = re.compile(r"[-\U000f0000-\U0010ffff]")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
H1 = re.compile(
    r"^\s*(?:제\s*\d+\s*장|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*[.．]|[IVX]+\s*\.)\s*\S[^\n]{0,40}$", re.M
)
H2 = re.compile(r"^\s*\d{1,2}\s*\.\s+\S[^\n]{0,40}$", re.M)
H3 = re.compile(r"^\s*[가-하]\s*\.\s+\S[^\n]{0,40}$", re.M)
TABLE = re.compile(r"\[표\](.*?)\[/표\]", re.DOTALL)

REQ_PREFIXES = [
    "ECR",
    "SFR",
    "PER",
    "SIR",
    "DAR",
    "TER",
    "SER",
    "QUR",
    "COR",
    "PMR",
    "PSR",
]
_P = "|".join(REQ_PREFIXES)
REQ_ID = re.compile(rf"(?<![A-Za-z0-9])({_P})-\s?(\d{{1,3}})(?!\d)")
# 요구사항 상세: '요구사항 고유번호 | SFR-001', '요구사항번호 PER-001', '요구사항 ID | SFR-02' 등
REQ_DETAIL = re.compile(
    rf"(?:고유\s*번호|번호|ID)\s*[|:]?\s*({_P})-\s?(\d{{1,3}})(?!\d)"
)


def req_blocks(text: str) -> int:
    """라벨과 무관하게 '요구사항 명칭/명' 과 '세부 내용'이 같이 든 표의 개수."""
    return sum(
        1
        for b in TABLE.findall(text)
        if re.search(r"요구\s*사항\s*(?:명칭|명)\b", b) and re.search(r"세부\s*내용", b)
    )


CONTRACT_WORDS = [
    "협상에 의한 계약",
    "제한경쟁",
    "일반경쟁",
    "지명경쟁",
    "수의계약",
    "최저가",
    "협상적격자",
]
KINDS = {
    "구축": r"구축",
    "고도화": r"고도화",
    "개선": r"개선",
    "개발": r"개발",
    "운영·유지관리": r"운영|유지\s*관리|유지\s*보수",
    "ISP·컨설팅": r"ISP|정보화\s*전략|컨설팅|기본\s*계획",
    "이전·전환": r"이전|전환|마이그레이션",
    "통합·연계": r"통합|연계",
}


def add_table(title: str, header: list[str], rows: list[list[object]]) -> None:
    _tables.append((title, header, rows))


def headings(text: str) -> tuple[int, int, int]:
    def count(rx: re.Pattern) -> int:
        return sum(1 for m in rx.finditer(text) if not PAGE_NO.search(m.group(0)))

    return count(H1), count(H2), count(H3)


def rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def tech_price(text: str) -> tuple[int, int] | None:
    for line in text.split("\n"):
        if len(line) > 200 or "기술" not in line or "가격" not in line:
            continue
        pats = [
            r"(\d{2,3})\s*[:：]\s*(\d{1,3})",
            r"기술\D{0,20}?(\d{2,3})\s*(?:점|%)\D{0,40}?가격\D{0,20}?(\d{1,3})\s*(?:점|%)",
        ]
        for p in pats:
            for m in re.finditer(p, line):
                a, b = int(m.group(1)), int(m.group(2))
                if a + b == 100 and a >= b:
                    return a, b
    return None


def duration_days(text: str) -> tuple[int, str] | None:
    """'사업기간/과업기간/용역기간/수행기간/계약기간' 라벨 뒤의 N일·N개월 (날짜 속 숫자 제외)."""
    m = re.search(
        r"(?:사업|과업|수행|용역|계약)\s*기간[^\n]{0,40}?(?<![\d.])(\d{1,3})\s*(일|개월)",
        text,
    )
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return (n * 30 if unit != "일" else n), unit


def joint_label(text: str) -> str:
    pos = neg = False
    for m in re.finditer(r"공동\s*(?:수급|계약|도급|이행)[^\n]{0,40}", text):
        s = m.group(0)
        if re.search(r"불허|허용하지|불가|금지|할\s*수\s*없|않", s):
            neg = True
        elif re.search(r"허용|가능|할\s*수\s*있|인정", s):
            pos = True
    if pos and neg:
        return "허용·불허 표현 둘 다"
    if neg:
        return "불허 표현만"
    if pos:
        return "허용 표현만"
    return "관련 표현 없음"


# 라벨 글자 사이 공백('사 업 비')도 허용. 예전 버전은 `|` 가 정규식 전체를 가르는 버그가 있어
# 라벨 없이 'N백만원'만 나와도 잡혔고, 그 분기가 우연히 17번 같은 문서를 잡아 주고 있었음.
_AMOUNT_LABELS = [
    "사업비",
    "사업예산",
    "예산금액",
    "추정가격",
    "추정금액",
    "사업금액",
    "소요예산",
    "배정예산",
]
BODY_AMOUNT = re.compile(
    "(?:" + "|".join(r"\s*".join(w) for w in _AMOUNT_LABELS) + ")"
    r"[^\n]{0,60}?(?:\d[\d,]{4,}\s*원|\d[\d,.]*\s*백만\s*원)"
)

# 가설 H4: 제출 마감 안내 (날짜까지 있는 표현)
DEADLINE = re.compile(
    r"(?:제출\s*기한|제출\s*마감|입찰\s*마감|접수\s*마감|마감\s*일시|제안서\s*제출)"
    r"[^\n]{0,80}?20\d{2}\s*[.\-년]\s*\d{1,2}\s*[.\-월]\s*\d{1,2}"
)
JOINT = re.compile(r"공동\s*(?:수급|계약|도급|이행)[^\n]{0,40}")
NEG_JOINT = re.compile(r"불허|허용하지|불가|금지|할\s*수\s*없|않")
POS_JOINT = re.compile(r"허용|가능|할\s*수\s*있|인정")


def iqr_fence(vals: list[float]) -> tuple[float, float]:
    """튜키 울타리: 1사분위 - 1.5*IQR, 3사분위 + 1.5*IQR 바깥을 이상치로 봄."""
    s = sorted(vals)
    q1, q3 = s[len(s) // 4], s[(3 * len(s)) // 4]
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def deep_analysis(docs: list[dict]) -> tuple[list[str], dict[int, list]]:
    """심화 분석(--deep). 화면·보고서에 쓰고, doc_stats.csv 에 붙일 열을 돌려줌."""
    # ------------------------------------------------------------ 1. 줄·제목
    say()
    say("=" * 60)
    say("[심화] 1. 줄 길이와 제목 구조 (청킹 기준 잡기)")
    say("=" * 60)
    for d in docs:
        t = d["text"]
        lines = [x for x in t.split("\n") if x.strip()]
        lens = [len(x) for x in lines]
        d["n_lines"] = len(lines)
        d["line_med"] = st.median(lens)
        d["line_max"] = max(lens)
        d["long_share"] = sum(x for x in lens if x > 500) / max(1, sum(lens))
        d["h1"], d["h2"], d["h3"] = headings(t)
    say(f"문서당 줄 수: {dist_line([d['n_lines'] for d in docs], '줄')}")
    say(f"줄 길이 중앙값(문서별): {dist_line([d['line_med'] for d in docs], '자')}")
    say(f"가장 긴 한 줄(문서별): {dist_line([d['line_max'] for d in docs], '자')}")
    say(
        f"500자 넘는 줄이 차지하는 글자 비율(문서별 %): "
        f"{dist_line([d['long_share'] * 100 for d in docs], '%')}"
    )
    over = [d["no"] for d in docs if d["line_max"] > 3000]
    say(f"한 줄이 3000자 넘는 문서: {len(over)}건 {over[:15]}")
    for lab, key in (
        ("대제목(Ⅰ./제1장)", "h1"),
        ("1. 형태 제목", "h2"),
        ("가. 형태 제목", "h3"),
    ):
        v = [d[key] for d in docs]
        say(
            f"{lab}: {dist_line(v, '개')} | 하나도 없는 문서 {sum(1 for x in v if x == 0)}건"
        )
    add_table(
        "제목 구조 (문서당 개수 중앙값)",
        ["종류", "중앙값", "없는 문서"],
        [
            [
                lab,
                f"{st.median([d[k] for d in docs]):.0f}",
                sum(1 for d in docs if d[k] == 0),
            ]
            for lab, k in (("대제목", "h1"), ("1. 제목", "h2"), ("가. 제목", "h3"))
        ],
    )

    # ------------------------------------------------------------ 2. 표
    say()
    say("=" * 60)
    say("[심화] 2. 표 크기 (청크 크기와 비교)")
    say("=" * 60)
    sizes: list[int] = []
    for d in docs:
        t = d["text"]
        blocks = TABLE.findall(t)
        d["t_empty"] = d["t_big"] = 0
        for b in blocks:
            body = re.sub(r"[\s|]", "", b)
            sizes.append(len(b))
            if not body:
                d["t_empty"] += 1
            if len(b) > 1000:
                d["t_big"] += 1
    total_chars = sum(sizes)
    big = [s for s in sizes if s > 1000]
    huge = [s for s in sizes if s > 3000]
    say(f"표 {len(sizes):,}개, 표 글자 합계 {total_chars:,}자")
    say(f"표 한 개 크기(글자): {dist_line(sizes, '자')}")
    hist(
        "표 한 개 크기 분포 (표 개수)",
        sizes,
        [0, 1, 50, 200, 500, 1000, 3000, 1e12],
        ["~0", "1~49", "50~199", "200~499", "500~999", "1000~2999", "3000~"],
        "개",
    )
    empty_n = sum(d["t_empty"] for d in docs)
    say(f"내용 없는 빈 표: {empty_n:,}개 ({empty_n / max(1, len(sizes)) * 100:.1f}%)")
    say(
        f"1000자 초과 표: {len(big):,}개 (표 글자의 {sum(big) / max(1, total_chars) * 100:.0f}%), "
        f"3000자 초과: {len(huge):,}개 (표 글자의 {sum(huge) / max(1, total_chars) * 100:.0f}%)"
    )
    say(f"1000자 초과 표가 있는 문서: {sum(1 for d in docs if d['t_big'])}건")
    say("청크 크기별 예상 청크 수 (표를 무시하고 글자 수로만 자른 단순 계산):")
    chunk_rows = []
    for size in (500, 1000, 1500):
        per = [math.ceil(len(d["text"]) / size) for d in docs]
        for d, c in zip(docs, per, strict=True):
            d[f"chunk{size}"] = c
        say(
            f"  {size}자: 전체 {sum(per):,}개 | 문서당 중앙값 {st.median(per):.0f}개 | 최대 {max(per)}개"
        )
        chunk_rows.append(
            [f"{size}자", f"{sum(per):,}", f"{st.median(per):.0f}", max(per)]
        )
    add_table(
        "청크 크기별 예상 청크 수",
        ["크기", "전체", "문서당 중앙값", "문서당 최대"],
        chunk_rows,
    )

    # ------------------------------------------------------------ 3. 품질
    say()
    say("=" * 60)
    say("[심화] 3. 텍스트 품질")
    say("=" * 60)
    for d in docs:
        t = d["text"]
        d["pua"] = len(PUA.findall(t))
        d["ctrl"] = len(CTRL.findall(t))
        ns = re.sub(r"\s", "", t)
        d["hangul"] = len(re.findall(r"[가-힣]", ns)) / max(1, len(ns))
        ls = [norm(x) for x in t.split("\n")]
        ls = [x for x in ls if len(x) >= 10]
        c = Counter(ls)
        rep = sum(len(k) * (v - 1) for k, v in c.items() if v >= 3)
        d["rep_share"] = rep / max(1, sum(len(x) for x in ls))
    pua_docs = [d for d in docs if d["pua"]]
    say(
        f"특수 기호(사용자 정의 영역 문자) 포함 문서: {len(pua_docs)}건, 합계 {sum(d['pua'] for d in docs):,}자 "
        f"(전체 글자의 {sum(d['pua'] for d in docs) / max(1, sum(len(d['text']) for d in docs)) * 100:.2f}%)"
    )
    top_pua = sorted(pua_docs, key=lambda x: -x["pua"])[:5]
    say("  많은 문서 5건: " + ", ".join(f"{d['no']}:{d['pua']}" for d in top_pua))
    ctrl_docs = [d for d in docs if d["ctrl"]]
    say(
        f"제어문자·깨진 글자(�) 포함 문서: {len(ctrl_docs)}건, 합계 {sum(d['ctrl'] for d in docs):,}자"
    )
    hs = [d["hangul"] * 100 for d in docs]
    say(f"한글 비율(공백 제외, %): {dist_line(hs, '%')}")
    low = sorted(docs, key=lambda x: x["hangul"])[:5]
    say(
        "  한글 비율이 낮은 문서 5건: "
        + ", ".join(f"{d['no']}:{d['hangul'] * 100:.0f}%" for d in low)
    )
    rs = [d["rep_share"] * 100 for d in docs]
    say(f"한 문서 안에서 3번 이상 반복되는 줄의 글자 비율(%): {dist_line(rs, '%')}")

    # ------------------------------------------------------------ 4. 문서 간 반복
    say()
    say("=" * 60)
    say("[심화] 4. 문서 간 반복 문장 (검색 때 같은 문장이 여러 문서에서 나오는 정도)")
    say("=" * 60)
    df: Counter = Counter()
    sample: dict[str, str] = {}
    per_doc_lines: list[list[str]] = []
    for d in docs:
        keys = []
        for ln in d["text"].split("\n"):
            k = norm(ln)
            if len(k) >= 20:
                keys.append(k)
                sample.setdefault(k, ln.strip())
        per_doc_lines.append(keys)
        df.update(set(keys))
    for th in (5, 10, 30):
        n = sum(1 for v in df.values() if v >= th)
        say(f"{th}개 이상의 문서에 똑같이 나오는 줄(20자 이상): 서로 다른 줄 {n:,}개")
    for d, keys in zip(docs, per_doc_lines, strict=True):
        tot = sum(len(k) for k in keys)
        d["boiler"] = sum(len(k) for k in keys if df[k] >= 10) / max(1, tot)
    bs = [d["boiler"] * 100 for d in docs]
    say(
        f"문서별로 10개 이상 문서에 반복되는 줄이 차지하는 글자 비율(%): {dist_line(bs, '%')}"
    )
    top_b = sorted(docs, key=lambda x: -x["boiler"])[:5]
    say(
        "  반복 비율이 높은 문서 5건: "
        + ", ".join(f"{d['no']}:{d['boiler'] * 100:.0f}%" for d in top_b)
    )
    print("  [화면에만 표시, 저장 안 됨] 가장 많은 문서에 반복되는 줄:", flush=True)
    for k, v in df.most_common(6):
        print(f"    {v}건 | {sample[k][:30]}...", flush=True)

    # ------------------------------------------------------------ 5. 핵심 사실
    say()
    say("=" * 60)
    say("[심화] 5. 핵심 사실이 어떤 모양으로 나오는지 (추정, 정규식)")
    say("=" * 60)
    tp = {d["no"]: tech_price(d["text"]) for d in docs}
    found = [v for v in tp.values() if v]
    say(f"기술:가격 배점을 찾은 문서: {len(found)}/{len(docs)}건")
    say(
        "  배점 조합: "
        + ", ".join(f"{a}:{b}({c})" for (a, b), c in Counter(found).most_common())
    )
    add_table(
        "기술:가격 배점 조합 (추정)",
        ["기술:가격", "문서 수"],
        [[f"{a}:{b}", c] for (a, b), c in Counter(found).most_common()],
    )
    dur = {d["no"]: duration_days(d["text"]) for d in docs}
    dfound = [v for v in dur.values() if v]
    say(f"과업 기간(기간 라벨 뒤 N일/개월)을 찾은 문서: {len(dfound)}/{len(docs)}건")
    if dfound:
        days = [v[0] for v in dfound]
        say(f"  기간(일 환산, 1개월=30일): {dist_line(days, '일')}")
        say(f"  단위: {dict(Counter(v[1] for v in dfound))}")
        hist(
            "과업 기간 분포 (일 환산, 문서 수·추정)",
            days,
            [0, 60, 90, 120, 180, 270, 365, 1e9],
            ["~59일", "60~89", "90~119", "120~179", "180~269", "270~364", "365일~"],
        )
    jl = {d["no"]: joint_label(d["text"]) for d in docs}
    jc = Counter(jl.values())
    say(f"공동수급(공동계약) 표현 (추정): {dict(jc)}")
    add_table(
        "공동수급 표현 (추정)", ["분류", "문서 수"], [[k, v] for k, v in jc.items()]
    )
    cw = []
    for w in CONTRACT_WORDS:
        n = sum(1 for d in docs if w in d["text"])
        cw.append([w, n])
        say(f"  '{w}' 나오는 문서: {n}건")
    add_table("계약·입찰 방식 용어가 나오는 문서", ["용어", "문서 수"], cw)
    kc = Counter()
    for d in docs:
        d["kinds"] = [k for k, p in KINDS.items() if re.search(p, d["row"]["사업명"])]
        kc.update(d["kinds"])
    say(f"사업명 키워드로 본 사업 종류(중복 포함): {dict(kc.most_common())}")
    add_chart(
        "사업명 키워드별 문서 수 (중복 포함)",
        list(kc),
        [float(v) for v in kc.values()],
        "건",
    )

    # ------------------------------------------------------------ 6. 상관
    say()
    say("=" * 60)
    say("[심화] 6. 숫자들 사이의 관계 (스피어만 순위상관, -1~1)")
    say("=" * 60)
    pairs = [d for d in docs if d["amount"]]
    r1 = spearman(
        [d["chars"] for d in pairs],
        [d["amount"] for d in pairs],
    )
    r2 = spearman(
        [float(len(d["text"])) for d in docs],
        [float(d["text"].count("[표]")) for d in docs],
    )
    say(f"글자 수 vs 사업 금액 ({len(pairs)}건): {r1:.2f}")
    say(f"글자 수 vs 표 개수 ({len(docs)}건): {r2:.2f}")
    say("  (0에 가까우면 관계가 약함. 0.3 이하는 거의 관계없다고 봐도 됨)")

    # ------------------------------------------------------------ 7. 단어
    say()
    say("=" * 60)
    say("[심화] 7. 자주 나오는 단어 (한글 2자 이상, 불용어 제거 전)")
    say("=" * 60)
    tot: Counter = Counter()
    dfw: Counter = Counter()
    for d in docs:
        toks = re.findall(r"[가-힣]{2,}", d["text"])
        tot.update(toks)
        dfw.update(set(toks))
    words = tot.most_common(25)
    say("  " + ", ".join(f"{w}({c:,}회/{dfw[w]}건)" for w, c in words))
    add_table(
        "자주 나오는 단어 상위 25",
        ["단어", "전체 횟수", "나오는 문서"],
        [[w, f"{c:,}", dfw[w]] for w, c in words],
    )

    # ------------------------------------------------------------ 8. 요구사항 ID
    say()
    say("=" * 60)
    say("[심화] 8. 요구사항 ID 구조 (ECR·SFR·SER 등, 상세 표 기준)")
    say("=" * 60)
    pref_docs: Counter = Counter()
    pref_reqs: Counter = Counter()
    pref_per_doc: dict[str, list[int]] = {p: [] for p in REQ_PREFIXES}
    for d in docs:
        t = d["text"]
        all_ids = {f"{p}-{int(n)}" for p, n in REQ_ID.findall(t)}
        det = {f"{p}-{int(n)}" for p, n in REQ_DETAIL.findall(t)}
        d["req_ids"] = len(all_ids)
        d["req_detail"] = len(det)
        d["req_blocks"] = req_blocks(t)
        d["req_by_prefix"] = Counter(i.split("-")[0] for i in det)
        for p, c in d["req_by_prefix"].items():
            pref_docs[p] += 1
            pref_reqs[p] += c
            pref_per_doc[p].append(c)
    has_det = [d for d in docs if d["req_detail"] or d["req_blocks"]]
    no_det = [d["no"] for d in docs if not (d["req_detail"] or d["req_blocks"])]
    by_id = sum(1 for d in docs if d["req_detail"])
    by_blk = sum(1 for d in docs if d["req_blocks"])
    say(
        f"  (ID 줄 기준 {by_id}건, 표 구조 기준 {by_blk}건, 둘 중 하나라도 {len(has_det)}건)"
    )
    say(f"요구사항 상세가 있는 문서: {len(has_det)}/{len(docs)}건")
    say(f"  상세 표가 없는 문서: {no_det[:30]}")
    if has_det:
        best = [max(d["req_detail"], d["req_blocks"]) for d in has_det]
        say(f"  문서당 요구사항 상세 수(두 기준 중 큰 값): {dist_line(best, '개')}")
        say(f"  전체 요구사항 상세 {sum(best):,}개")
        hist(
            "문서당 요구사항 상세 수 (문서 수)",
            best,
            [0, 20, 40, 60, 80, 100, 1e9],
            ["~19", "20~39", "40~59", "60~79", "80~99", "100~"],
        )
    say(
        f"본문 어디서든 요구사항 ID가 나오는 문서: {sum(1 for d in docs if d['req_ids'])}건"
    )
    prow = []
    for p in REQ_PREFIXES:
        meds = pref_per_doc[p]
        med = f"{st.median(meds):.0f}" if meds else "-"
        say(
            f"  {p}: 상세가 있는 문서 {pref_docs[p]:>3}건 | 상세 합계 {pref_reqs[p]:>5,}개 | 문서당 중앙값 {med}"
        )
        prow.append([p, pref_docs[p], f"{pref_reqs[p]:,}", med])
    add_table(
        "요구사항 접두어별 현황 (상세 표 기준)",
        ["접두어", "문서 수", "상세 합계", "문서당 중앙값"],
        prow,
    )
    add_chart(
        "접두어별 요구사항 상세 합계",
        REQ_PREFIXES,
        [float(pref_reqs[p]) for p in REQ_PREFIXES],
        "개",
    )

    cols = [
        "줄수", "줄길이중앙값", "최장줄", "긴줄글자비율", "대제목수", "제목1수", "제목가수",
        "빈표수", "큰표수", "청크500", "청크1000", "청크1500", "특수기호수", "깨진글자수",
        "한글비율", "문서내반복비율", "문서간반복비율", "기술배점", "가격배점",
        "과업기간_일", "공동수급표현", "사업종류",
    ]  # fmt: skip
    rows: dict[int, list] = {}
    for d in docs:
        t_p = tp[d["no"]]
        du = dur[d["no"]]
        rows[d["no"]] = [
            d["n_lines"], d["line_med"], d["line_max"], round(d["long_share"], 3),
            d["h1"], d["h2"], d["h3"], d["t_empty"], d["t_big"],
            d["chunk500"], d["chunk1000"], d["chunk1500"], d["pua"], d["ctrl"],
            round(d["hangul"], 3), round(d["rep_share"], 3), round(d["boiler"], 3),
            t_p[0] if t_p else "", t_p[1] if t_p else "", du[0] if du else "",
            jl[d["no"]], "/".join(d["kinds"]),
        ]  # fmt: skip
    cols += ["요구사항ID수", "요구사항상세수", "요구사항상세표수"] + [
        f"{p}상세수" for p in REQ_PREFIXES
    ]
    for d in docs:
        rows[d["no"]] += [d["req_ids"], d["req_detail"], d["req_blocks"]] + [
            d["req_by_prefix"].get(p, 0) for p in REQ_PREFIXES
        ]
    return cols, rows


def table_share_after(text: str, pat: str, window: int = 5000) -> float | None:
    """항목 제목 줄부터 window 글자 구간에서 표 안 글자가 차지하는 비율 (근사)."""
    pos = heading_pos_strict(text, pat)
    if pos is None:
        pos = heading_pos(text, pat)
    if pos is None:
        return None
    end = min(len(text), pos + window)
    inside = 0
    for m in TABLE.finditer(text):
        a, b = max(m.start(), pos), min(m.end(), end)
        if b > a:
            inside += b - a
    return inside / max(1, end - pos)


def joint_samples(docs: list[dict], n: int = 15) -> tuple[int, int, str]:
    """허용·불허 표현이 둘 다 있는 문서에서 일정 간격으로 n건을 뽑아 해당 문장을 보여 줌."""
    both = [d for d in docs if joint_label(d["text"]) == "허용·불허 표현 둘 다"]
    picked = both[:: max(1, len(both) // n)][:n]
    lines = []
    for d in picked:
        neg, pos = [], []
        for m in JOINT.finditer(d["text"]):
            sent = m.group(0)
            if NEG_JOINT.search(sent):
                neg.append(sent)
            elif POS_JOINT.search(sent):
                pos.append(sent)
        lines.append(f"[문서 {d['no']}]")
        lines += [f"  불허 계열: {x[:70]}" for x in neg[:2]]
        lines += [f"  허용 계열: {x[:70]}" for x in pos[:2]]
    return len(both), len(picked), "\n".join(lines)


def hypothesis_section(docs: list[dict], docs_ok: list[dict]) -> None:
    """입찰 대상자에게 근거를 대고 준비물을 알려 주는 챗봇 기준의 가설 7개."""
    say()
    say("=" * 60)
    say("8. 가설 vs 결과 (근거를 대고 준비물을 알려 주는 챗봇 기준)")
    say("=" * 60)

    def med_pos(key: str) -> float | None:
        v = [d[f"pos_{key}"] for d in docs_ok if d[f"pos_{key}"] != ""]
        return st.median(v) if v else None

    mp = {
        k: med_pos(k)
        for k in ("사업개요", "사업예산", "참가자격", "제출서류", "평가기준")
    }
    h1_ok = all(v is not None for v in mp.values()) and (
        all(mp[k] <= 3 for k in ("사업개요", "사업예산"))
        and all(40 <= mp[k] <= 65 for k in ("참가자격", "제출서류", "평가기준"))
    )
    h1_val = ", ".join(f"{k} {v:.0f}%" for k, v in mp.items() if v is not None)

    n_sub = sum(d["has_제출서류"] for d in docs_ok)

    tshare = {}
    for key in ("참가자격", "제출서류"):
        v = [
            x
            for x in (table_share_after(d["text"], SECTIONS[key]) for d in docs_ok)
            if x is not None
        ]
        tshare[key] = st.median(v) * 100 if v else 0.0
    h3_val = ", ".join(f"{k} 절 표 비율 중앙값 {v:.0f}%" for k, v in tshare.items())

    miss = [d for d in docs_ok if not d["row"]["입찰 참여 마감일"].strip()]
    found = [d["no"] for d in miss if DEADLINE.search(d["text"])]
    lost = sorted({d["no"] for d in miss} - set(found))

    total, picked, sample = joint_samples(docs_ok)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "H5_표본.txt").write_text(sample + "\n", encoding="utf-8")

    name_hit = sum(d["name_in_text"] for d in docs_ok)
    amt_n = sum(1 for d in docs_ok if d["amt_in_text"] != "")
    amt_hit = sum(1 for d in docs_ok if d["amt_in_text"] == 1)

    items = [
        (
            "H1",
            "참가자격·제출서류·평가기준은 본문 중후반(40~65%), 사업개요·예산은 앞 3% 안에 있다",
            h1_val,
            h1_ok,
            "청크에 섹션명 메타를 붙이고, 준비물 질문은 뒤쪽 절에서 찾는다",
        ),
        (
            "H2",
            "제출서류 안내는 95% 이상의 문서에 제목 줄로 있다",
            f"제목 줄 {n_sub}/{len(docs_ok)}건",
            n_sub / len(docs_ok) >= 0.95,
            "제목 줄이 없는 문서는 준비물 답변이 어려우니 따로 점검",
        ),
        (
            "H3",
            "참가자격·제출서류 절의 글자는 절반 이상이 표 안에 있다 (제목 줄부터 5,000자 구간, 근사)",
            h3_val,
            all(v >= 50 for v in tshare.values()),
            "준비물 목록은 표 구조를 살려 청킹",
        ),
        (
            "H4",
            "메타에 마감일이 없는 문서도 본문에 제출 마감 날짜 안내가 있다 (80% 이상)",
            f"마감일 결측 {len(miss)}건 중 본문에서 찾음 {len(found)}건, 못 찾음 {lost[:12]}",
            bool(miss) and len(found) / len(miss) >= 0.8,
            "기한은 본문에서 보완하거나 '확인 불가'로 답한다",
        ),
        (
            "H5",
            "허용·불허 표현이 같이 있는 문서의 대부분(80% 이상)은 서로 다른 대상 때문이지 모순이 아니다",
            f"공존 {total}건 중 표본 {picked}건을 data/H5_표본.txt 에 저장 (사람이 읽고 비율 확인)",
            None,
            "키워드 검색 대신 해당 조항 문장 단위로 근거를 인용",
        ),
        (
            "H6",
            "사업명이 본문에 그대로 있는 문서가 90% 이상이다",
            f"{name_hit}/{len(docs_ok)}건",
            name_hit / len(docs_ok) >= 0.9,
            "사업명·기관명을 메타로 청크에 붙여 검색",
        ),
        (
            "H7",
            "메타 사업 금액이 본문 금액과 90% 이상 일치한다 (쉼표 표기·부가세 ±10% 기준)",
            f"{amt_hit}/{amt_n}건",
            amt_n > 0 and amt_hit / amt_n >= 0.9,
            "금액 답변은 메타가 아니라 본문 근거를 인용",
        ),
    ]
    rows = []
    for no, claim, actual, ok, impact in items:
        verdict = {True: "예상과 같음", False: "예상과 다름", None: "사람 확인 필요"}[
            ok
        ]
        say(f"  {no} {claim}")
        say(f"      실제: {actual} -> {verdict}")
        say(f"      영향: {impact}")
        rows.append([no, claim, actual, verdict, impact])
    diff = sum(1 for r in rows if r[3] == "예상과 다름")
    pend = sum(1 for r in rows if r[3] == "사람 확인 필요")
    say(f"예상과 다른 항목 {diff}/{len(rows)}개 (사람 확인 필요 {pend}개)")
    _tables.insert(1, ("가설 vs 결과", ["번호", "가설", "실제", "판정", "영향"], rows))


def write_doc_index(docs_ok: list[dict]) -> None:
    """문서번호 표(md). 공개 저장소용은 사업명·금액을 빼고, 사업명 포함본은 data/ 에만 둠."""
    pub = [
        "# 문서 번호표",
        "",
        "문서번호는 `data/data_list.csv` 의 행 순서(1부터)입니다.",
        "사업명·금액은 공개 저장소라 싣지 않았고, 점검표시는 eda_stats 7번 섹션 기준입니다.",
        "",
        "| 번호 | 형식 | 발주 기관 | 글자수 | 표 | 점검표시 |",
        "|---:|:---:|:---|---:|---:|:---|",
    ]
    loc = [
        "# 문서 번호표 (로컬 전용, 사업명 포함)",
        "",
        "| 번호 | 형식 | 발주 기관 | 사업명 | 파일명 | 점검표시 |",
        "|---:|:---:|:---|:---|:---|:---|",
    ]
    for d in docs_ok:
        flags = ", ".join(d["flags"])
        pub.append(
            f"| {d['no']} | {d['ext']} | {d['org']} | {d['chars']:,} | {d['tables']} | {flags} |"
        )
        loc.append(
            f"| {d['no']} | {d['ext']} | {d['org']} | {d['row']['사업명']} | {d['row']['파일명']} | {flags} |"
        )
    (Path(__file__).resolve().parent / "문서번호표.md").write_text(
        "\n".join(pub) + "\n", encoding="utf-8"
    )
    (DATA / "문서번호표_전체.md").write_text("\n".join(loc) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RFP 문서 단위 EDA (1단계)")
    parser.add_argument(
        "--name", default="doc_stats", help="결과 폴더 이름에 붙일 실험 키워드"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="심화 분석(청킹·표·품질·반복·핵심 사실) 추가",
    )
    args = parser.parse_args()

    docs = load_docs()
    missing = [d["no"] for d in docs if d["text"] is None]
    if missing:
        say(f"! 변환된 txt 가 없는 문서: {missing} (convert --all 을 먼저 돌리세요)")
    docs_ok = [d for d in docs if d["text"] is not None]

    # ---------------------------------------------------------------- 1. 기본
    say("=" * 60)
    say("1. 기본 현황")
    say("=" * 60)
    say(f"문서 {len(docs)}건 (변환 txt 있음 {len(docs_ok)}건)")
    say(f"형식: {dict(Counter(d['ext'] for d in docs))}")
    say("메타데이터 결측 (data_list.csv):")
    miss_rows = []
    for col in [c for c in docs[0]["row"] if c != "텍스트"]:
        n = sum(1 for d in docs if not d["row"][col].strip())
        if n:
            say(f"  {col}: {n}건")
            miss_rows.append([col, n])
    if miss_rows:
        _tables.append(("메타데이터 결측", ["열", "결측 건수"], miss_rows))

    # ---------------------------------------------------------------- 2. 길이
    say()
    say("=" * 60)
    say("2. 문서 길이와 표")
    say("=" * 60)
    for d in docs_ok:
        t = d["text"]
        d["chars"] = nonspace_len(t)
        d["tables"] = t.count("[표]")
        in_tbl = sum(
            nonspace_len(m) for m in re.findall(r"\[표\].*?\[/표\]", t, flags=re.DOTALL)
        )
        d["tbl_share"] = in_tbl / d["chars"] if d["chars"] else 0.0
    chars = [d["chars"] for d in docs_ok]
    tbls = [d["tables"] for d in docs_ok]
    shares = [d["tbl_share"] * 100 for d in docs_ok]
    say(f"글자 수(공백 제외): {dist_line(chars, '자')}")
    say(f"  합계 {sum(chars):,}자")
    hist(
        "글자 수 분포 (공백 제외, 문서 수)",
        chars,
        [0, 40000, 50000, 60000, 80000, 100000, 1e9],
        ["~4만", "4~5만", "5~6만", "6~8만", "8~10만", "10만~"],
    )
    say(f"표 개수: {dist_line(tbls, '개')}")
    hist(
        "표 개수 분포 (문서 수)",
        tbls,
        [0, 50, 75, 100, 125, 150, 200, 1e9],
        ["~50", "50~75", "75~100", "100~125", "125~150", "150~200", "200~"],
    )
    say(f"표 안 글자 비율(%): {dist_line(shares, '%')}")
    ext_rows = []
    for e in sorted({d["ext"] for d in docs_ok}):
        v = [d["chars"] for d in docs_ok if d["ext"] == e]
        say(f"  {e}: {len(v)}건, 글자 수 중앙값 {st.median(v):,.0f}자")
        ext_rows.append([e, len(v), f"{st.median(v):,.0f}"])
    _tables.append(("형식별 글자 수", ["형식", "건수", "글자 수 중앙값"], ext_rows))
    top_long = sorted(docs_ok, key=lambda x: -x["chars"])[:5]
    top_short = sorted(docs_ok, key=lambda x: x["chars"])[:5]
    top_tbl = sorted(docs_ok, key=lambda x: -x["tables"])[:5]
    say(
        "가장 긴 문서 5건 (문서번호:글자수): "
        + ", ".join(f"{d['no']}:{d['chars']:,}" for d in top_long)
    )
    say(
        "가장 짧은 문서 5건: "
        + ", ".join(f"{d['no']}:{d['chars']:,}" for d in top_short)
    )
    say(
        "표가 가장 많은 문서 5건 (문서번호:표 수): "
        + ", ".join(f"{d['no']}:{d['tables']}" for d in top_tbl)
    )
    _tables.append(
        (
            "길이·표 상위/하위 5건 (문서번호)",
            ["구분", "문서번호:값"],
            [
                [
                    "가장 긴 문서(글자 수)",
                    ", ".join(f"{d['no']}:{d['chars']:,}" for d in top_long),
                ],
                [
                    "가장 짧은 문서(글자 수)",
                    ", ".join(f"{d['no']}:{d['chars']:,}" for d in top_short),
                ],
                [
                    "표가 가장 많은 문서(표 수)",
                    ", ".join(f"{d['no']}:{d['tables']}" for d in top_tbl),
                ],
            ],
        )
    )

    # ---------------------------------------------------------------- 3. 메타
    say()
    say("=" * 60)
    say("3. 메타데이터 분포 (data_list.csv)")
    say("=" * 60)
    amts = [d["amount"] for d in docs if d["amount"] is not None]
    say(f"사업 금액 (결측 {len(docs) - len(amts)}건): {dist_line(amts, '원')}")
    hist(
        "사업 금액 구간 (문서 수)",
        amts,
        [0, 5e7, 1e8, 3e8, 5e8, 1e9, 1e30],
        ["~5천만", "5천만~1억", "1억~3억", "3억~5억", "5억~10억", "10억~"],
    )
    org_cnt = Counter(d["org"] for d in docs)
    multi = [(o, c) for o, c in org_cnt.most_common() if c > 1]
    say(
        f"발주 기관 {len(org_cnt)}곳. 2건 이상: "
        + ", ".join(f"{o}({c})" for o, c in multi)
    )
    types = Counter(org_type(d["org"]) for d in docs)
    say(f"기관 유형(키워드 기반 추정): {dict(types)}")
    add_chart(
        "기관 유형 (키워드 기반 추정, 문서 수)",
        list(types),
        [float(v) for v in types.values()],
        "건",
    )
    _tables.append(
        ("2건 이상 나온 발주 기관", ["기관", "문서 수"], [[o, c] for o, c in multi])
    )
    pub = [p for p in (parse_dt(d["row"]["공개 일자"]) for d in docs) if p]
    if pub:
        say(f"공개 일자: {min(pub):%Y-%m-%d} ~ {max(pub):%Y-%m-%d}")
        months = sorted(Counter(f"{p:%Y-%m}" for p in pub).items())
        say("  월별: " + ", ".join(f"{m}:{c}" for m, c in months))
        add_chart(
            "공개 일자 월별 (문서 수)",
            [m for m, _ in months],
            [float(c) for _, c in months],
            "건",
        )
    gaps, bad = [], 0
    for d in docs:
        s = parse_dt(d["row"]["입찰 참여 시작일"])
        e = parse_dt(d["row"]["입찰 참여 마감일"])
        if s and e:
            gaps.append((e - s).days)
        p = parse_dt(d["row"]["공개 일자"])
        if p and e and e < p:
            bad += 1
    if gaps:
        say(
            f"입찰 기간(마감-시작, 일): {dist_line(gaps, '일')} (둘 다 있는 {len(gaps)}건)"
        )
    say(f"마감일이 공개일보다 앞선 건: {bad}건")
    say(f"공고 차수 값: {dict(Counter(d['row']['공고 차수'] or '결측' for d in docs))}")

    # ---------------------------------------------------------------- 4. 중복
    say()
    say("=" * 60)
    say("4. 중복·유사 문서 (문서번호로만 표시)")
    say("=" * 60)

    def groups(key_fn) -> list[list[int]]:
        g: dict[str, list[int]] = {}
        for d in docs_ok:
            k = key_fn(d)
            if k:
                g.setdefault(k, []).append(d["no"])
        return [v for v in g.values() if len(v) > 1]

    dup_rows = [
        ["사업명이 같은 문서", groups(lambda d: d["row"]["사업명"].strip())],
        ["공고 번호가 같은 문서", groups(lambda d: d["row"]["공고 번호"].strip())],
        [
            "본문 전체가 같은 문서",
            groups(
                lambda d: hashlib.sha1(
                    norm(d["text"]).encode(), usedforsecurity=False
                ).hexdigest()
            ),
        ],
        [
            "본문 앞 3000자가 같은 문서",
            groups(
                lambda d: hashlib.sha1(
                    norm(d["text"])[:3000].encode(), usedforsecurity=False
                ).hexdigest()
            ),
        ],
    ]
    for label, g in dup_rows:
        say(f"{label}: {g}")
    _tables.append(("중복·유사 문서 (문서번호)", ["기준", "묶음"], dup_rows))

    # ---------------------------------------------------------------- 5. 구조
    say()
    say("=" * 60)
    say("5. 공통 항목이 나오는지, 본문 어디쯤 나오는지 (목차 제외)")
    say("=" * 60)
    say(
        "유무 = 60자 이하이고 끝에 쪽 번호가 없는 줄에 키워드가 있는지 (목차 줄을 빼기 위함, 느슨)"
    )
    say(
        "위치 = 번호·기호가 붙은 '제목처럼 생긴' 줄만 인정한 엄격 판정의 첫 위치 (문장·표 항목·목차 줄 제외)"
    )
    say(
        "       본문 앞에서부터의 %, 0이 처음 / 100이 끝. 엄격 판정에 안 잡힌 문서는 위치에서 뺌"
    )
    sec_rows, present, med_pos = [], [], []
    for key, pat in SECTIONS.items():
        poss, loose_poss, n_has = [], [], 0
        anywhere = 0
        for d in docs_ok:
            n = len(d["text"])
            if re.search(pat, d["text"]):
                anywhere += 1
            hp = heading_pos(d["text"], pat)  # 느슨: 유무 판정용
            sp = heading_pos_strict(d["text"], pat)  # 엄격: 위치용
            d[f"has_{key}"] = 1 if hp is not None else 0
            d[f"pos_{key}"] = round(sp / n * 100, 1) if sp is not None else ""
            d[f"posL_{key}"] = round(hp / n * 100, 1) if hp is not None else ""
            n_has += hp is not None
            if hp is not None:
                loose_poss.append(hp / n * 100)
            if sp is not None:
                poss.append(sp / n * 100)
        present.append(float(n_has))
        lm = f"{st.median(loose_poss):.0f}%" if loose_poss else "-"
        head = f"  {key:<6} 제목 줄 {n_has:>3}/{len(docs_ok)}건 (키워드만 {anywhere}건, 엄격 {len(poss)}건, 느슨 중앙값 {lm})"
        if poss:
            say(
                f"{head} | 엄격 첫 위치 중앙값 {st.median(poss):.0f}% "
                f"(25% {pct(poss, 0.25):.0f}% ~ 75% {pct(poss, 0.75):.0f}%)"
            )
            med_pos.append(round(st.median(poss), 1))
            sec_rows.append(
                [
                    key,
                    f"{n_has}/{len(docs_ok)}",
                    anywhere,
                    f"{st.median(poss):.0f}% (느슨 {lm})",
                ]
            )
        else:
            say(head)
            med_pos.append(0.0)
            sec_rows.append(
                [key, f"{n_has}/{len(docs_ok)}", anywhere, f"- (느슨 {lm})"]
            )
    add_chart("제목 줄로 나오는 문서 수 (목차 제외)", list(SECTIONS), present, "건")
    add_chart(
        "엄격 제목 줄 첫 위치 중앙값 (본문 앞에서부터 %, 목차 제외)",
        list(SECTIONS),
        med_pos,
        "%",
    )
    _tables.append(
        (
            "공통 항목 등장 현황",
            ["항목", "제목 줄로 나오는 문서", "키워드만 나오는 문서", "첫 위치 중앙값"],
            sec_rows,
        )
    )

    # ---------------------------------------------------------------- 6. 메타와 본문 일치
    say()
    say("=" * 60)
    say("6. 메타데이터 값이 본문에도 있는가 (골든셋 정답 근거 확인용)")
    say("=" * 60)
    name_hit = amt_hit = amt_n = 0
    for d in docs_ok:
        tn = norm(d["text"])
        d["name_in_text"] = int(norm(d["row"]["사업명"]) in tn)
        name_hit += d["name_in_text"]
        d["amt_in_text"] = ""
        if d["amount"]:
            amt_n += 1
            a = int(d["amount"])
            forms = {f"{a:,}", str(a), f"{round(a * 1.1):,}", f"{round(a / 1.1):,}"}
            hit = int(any(f in d["text"] for f in forms))
            d["amt_in_text"] = hit
            amt_hit += hit
    say(f"사업명이 본문에 그대로 있는 문서: {name_hit}/{len(docs_ok)}건")
    say(
        f"사업 금액이 본문에 있는 문서(원 단위 쉼표 표기, 부가세 ±10% 포함): {amt_hit}/{amt_n}건"
    )
    say("  (표기가 다르면 못 찾으므로 '없다'가 아니라 '이 방식으로는 못 찾음'입니다)")
    _tables.append(
        (
            "메타데이터 값이 본문에 있는가",
            ["항목", "있는 문서"],
            [
                ["사업명", f"{name_hit}/{len(docs_ok)}"],
                ["사업 금액(쉼표 표기)", f"{amt_hit}/{amt_n}"],
            ],
        )
    )

    # ---------------------------------------------------------------- 7. 중복·결측·이상치 종합
    say()
    say("=" * 60)
    say("7. 중복·결측·이상치 종합 (문서번호만 표시, 문서별 금액 값은 싣지 않음)")
    say("=" * 60)
    by_no = {d["no"]: d for d in docs_ok}
    for d in docs_ok:
        d["flags"] = []
    findings: list[list[object]] = []

    def add_finding(
        kind: str, item: str, nos: list[int], action: str, flag: str = ""
    ) -> None:
        nos = sorted(set(nos))
        shown = ", ".join(map(str, nos[:12])) + (" …" if len(nos) > 12 else "")
        findings.append([kind, item, len(nos), shown or "-", action])
        say(f"  [{kind}] {item}: {len(nos)}건 {nos[:12]}")
        if flag:
            for n in nos:
                if n in by_no:
                    by_no[n]["flags"].append(flag)

    say("[중복]")
    add_finding("중복", "본문 전체가 같은 문서", [n for g in dup_rows[2][1] for n in g],
                "한 쪽만 남기고 인덱싱 제외", "중복")  # fmt: skip
    add_finding("중복", "사업명이 같은 문서", [n for g in dup_rows[0][1] for n in g],
                "위 항목과 겹치는지 확인")  # fmt: skip
    add_finding("중복", "본문 앞 3000자가 같은 문서", [n for g in dup_rows[3][1] for n in g],
                "개정판·재공고인지 확인")  # fmt: skip

    say("[결측]")
    for col in (
        "공고 번호",
        "공고 차수",
        "입찰 참여 시작일",
        "입찰 참여 마감일",
        "사업 금액",
    ):
        nos = [d["no"] for d in docs if not d["row"][col].strip()]
        if nos:
            add_finding(
                "결측", f"{col} 비어 있음", nos, "필요하면 본문에서 보완", f"{col}결측"
            )
    zero = [d["no"] for d in docs if d["amount"] == 0]
    z_body = [n for n in zero if n in by_no and BODY_AMOUNT.search(by_no[n]["text"])]
    add_finding(
        "결측", "사업 금액 0원 (사실상 결측)", zero, "금액으로 쓰지 말 것", "금액0"
    )
    add_finding(
        "결측", "  └ 본문에서 금액 표현을 찾은 문서", z_body, "본문 값으로 보완 가능"
    )
    add_finding("결측", "  └ 본문에서도 못 찾은 문서", [n for n in zero if n not in z_body],
                "원문 확인")  # fmt: skip

    say("[이상치] 튜키 울타리(1사분위-1.5*IQR ~ 3사분위+1.5*IQR) 바깥")
    gaps_by_no: dict[int, float] = {}
    for d in docs:
        s = parse_dt(d["row"]["입찰 참여 시작일"])
        e = parse_dt(d["row"]["입찰 참여 마감일"])
        if s and e:
            gaps_by_no[d["no"]] = float((e - s).days)
    groups_iqr = [
        ("글자 수", {d["no"]: float(d["chars"]) for d in docs_ok}, "자"),
        ("표 개수", {d["no"]: float(d["tables"]) for d in docs_ok}, "개"),
        (
            "사업 금액(0원 제외)",
            {d["no"]: d["amount"] for d in docs if d["amount"]},
            "원",
        ),
        ("입찰 기간", gaps_by_no, "일"),
    ]
    for label, pairs, unit in groups_iqr:
        lo, hi = iqr_fence(list(pairs.values()))
        add_finding("이상치", f"{label} 상한 {hi:,.0f}{unit} 초과",
                    [n for n, v in pairs.items() if v > hi], "삭제하지 말고 따로 확인", f"{label}큼")  # fmt: skip
        low = [n for n, v in pairs.items() if v < lo]
        if low:
            add_finding(
                "이상치",
                f"{label} 하한 {lo:,.0f}{unit} 미만",
                low,
                "원문 확인",
                f"{label}작음",
            )
    add_finding("이상치", "사업 금액이 1,000원 미만 (입력 오류 의심)",
                [d["no"] for d in docs if d["amount"] and d["amount"] < 1000],
                "원문 확인", "금액극소")  # fmt: skip
    add_finding("이상치", "입찰 시작·마감이 같은 날 (0일)",
                [n for n, v in gaps_by_no.items() if v == 0], "설명회·현장 접수 등인지 확인")  # fmt: skip
    pub_by_no = {d["no"]: parse_dt(d["row"]["공개 일자"]) for d in docs}
    month_cnt = Counter(f"{p:%Y-%m}" for p in pub_by_no.values() if p)
    add_finding("이상치", "공개 일자 달에 문서가 1건뿐 (다른 문서와 동떨어짐)",
                [n for n, p in pub_by_no.items() if p and month_cnt[f"{p:%Y-%m}"] <= 1],
                "오래된 공고인지 확인", "공개일동떨어짐")  # fmt: skip
    end_by_no = {d["no"]: parse_dt(d["row"]["입찰 참여 마감일"]) for d in docs}
    add_finding("이상치", "마감일이 공개일보다 앞섬",
                [n for n, p in pub_by_no.items() if p and end_by_no[n] and end_by_no[n] < p],
                "날짜 오류")  # fmt: skip
    _tables.insert(
        0,
        (
            "중복·결측·이상치 종합 (문서번호는 data_list.csv 행 순서)",
            ["구분", "항목", "건수", "문서번호(최대 12개)", "조치 제안"],
            findings,
        ),
    )
    flagged = sum(1 for d in docs_ok if d["flags"])
    say(
        f"점검 표시가 하나라도 붙은 문서: {flagged}/{len(docs_ok)}건 (doc_stats.csv 의 '점검표시' 열)"
    )

    hypothesis_section(docs, docs_ok)
    write_doc_index(docs_ok)

    deep_cols: list[str] = []
    deep_rows: dict[int, list] = {}
    if args.deep:
        deep_cols, deep_rows = deep_analysis(docs_ok)

    # ---------------------------------------------------------------- 저장
    run = make_run_dir(args.name)
    cols = [
        "문서번호",
        "형식",
        "글자수",
        "표개수",
        "표글자비율",
        "사업명_본문",
        "금액_본문",
        "점검표시",
    ]
    cols += [f"{k}_유무" for k in SECTIONS] + [f"{k}_위치%" for k in SECTIONS]
    cols += [f"{k}_위치%(느슨)" for k in SECTIONS]
    cols += deep_cols
    with open(run / "doc_stats.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for d in docs_ok:
            w.writerow(
                [
                    d["no"],
                    d["ext"],
                    d["chars"],
                    d["tables"],
                    round(d["tbl_share"], 3),
                    d["name_in_text"],
                    d["amt_in_text"],
                    ";".join(d["flags"]),
                ]
                + [d[f"has_{k}"] for k in SECTIONS]
                + [d[f"pos_{k}"] for k in SECTIONS]
                + [d[f"posL_{k}"] for k in SECTIONS]
                + deep_rows.get(d["no"], [])
            )
    info = [
        f"실행: {datetime.now():%Y-%m-%d %H:%M:%S}",  # noqa: DTZ005
        f"키워드: {args.name}" + (" (심화 포함)" if args.deep else ""),
        f"문서 {len(docs)}건 (변환 txt 있음 {len(docs_ok)}건)",
    ]
    (run / "run_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    write_report(run / "report.html", f"EDA - {args.name}", info)
    say()
    say(f"저장 폴더: {run}")
    (run / "summary.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print("  report.html / summary.txt / doc_stats.csv / run_info.txt", flush=True)


if __name__ == "__main__":
    main()
