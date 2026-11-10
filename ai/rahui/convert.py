"""원본 RFP(hwp, pdf)를 텍스트로 변환하고 CSV 텍스트와 길이를 비교합니다.

변환 결과는 data/processed/ 에만 저장합니다. data/ 는 gitignore 대상이라
원문이 커밋되지 않습니다. (저장소가 Public 이므로 원문을 data/ 밖에 두지 마세요.)

준비 (프로젝트 루트에서, 한 번만):
    uv add pypdf pyhwp six olefile

실행 (프로젝트 루트에서):
    uv run python -m ai.rahui.convert              # 샘플 9건 (짧은/중간/긴 각 3건)
    uv run python -m ai.rahui.convert --per-group 2  # 그룹별 2건
    uv run python -m ai.rahui.convert --all        # 전체 100건

출력:
    data/processed/<파일명>.txt        변환된 텍스트
    data/processed/_compare.csv        CSV 텍스트와 길이 비교표 (hwp 는 누락 점검 '커버리지' 포함)
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ai.rahui.hwp_olefile import convert_hwp_olefile, coverage

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FILES = DATA / "files"
OUT = DATA / "processed"
CSV_PATH = DATA / "data_list.csv"

HWP_TIMEOUT_SEC = 180
COVERAGE_MIN = 0.999  # hwp 원본 문단이 변환 결과에 들어 있어야 하는 최소 비율


def read_rows() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nonspace_len(text: str) -> int:
    """공백을 뺀 글자 수. 줄바꿈과 공백이 많은 CSV 텍스트와 공정하게 비교하기 위함."""
    return len("".join(text.split()))


def pick_samples(rows: list[dict], per_group: int) -> list[dict]:
    """CSV 텍스트 길이 기준으로 짧은 / 중간 / 긴 문서를 고릅니다."""
    ordered = sorted(rows, key=lambda r: nonspace_len(r["텍스트"]))
    n = len(ordered)
    mid = n // 2
    half = per_group // 2
    groups = [
        ordered[:per_group],
        ordered[mid - half : mid - half + per_group],
        ordered[-per_group:],
    ]
    picked, seen = [], set()
    for g in groups:
        for r in g:
            if r["파일명"] not in seen:
                seen.add(r["파일명"])
                picked.append(r)
    return picked


def find_tool(tool: str) -> list[str] | None:
    """pyhwp 가 설치하는 실행 파일(hwp5proc, hwp5txt)을 찾습니다."""
    found = shutil.which(tool)
    if found:
        return [found]
    scripts = Path(sys.executable).parent
    for name in (f"{tool}.exe", tool):
        cand = scripts / name
        if cand.exists():
            return [str(cand)]
    return None


def _para_text(p) -> str:
    """문단 하나의 글자. 문단 안에 표가 있으면 표 텍스트를 이어 붙입니다."""
    parts = []
    for line in p.findall("LineSeg"):
        for ch in line:
            if ch.tag == "Text":
                parts.append(ch.text or "")
            elif ch.tag == "TableControl":
                parts.append("\n" + _table_text(ch) + "\n")
    return "".join(parts)


def _table_text(tbl) -> str:
    """표를 [표] ... [/표] 로 감싸고, 행마다 셀을 ' | ' 로 이어 붙입니다.

    병합된 셀은 한 번만 나오므로 행마다 칸 수가 다를 수 있습니다.
    """
    lines = []
    for row in tbl.iter("TableRow"):
        cells = []
        for cell in row.findall("TableCell"):
            texts = (_para_text(p).strip() for p in cell.findall("Paragraph"))
            cells.append(" ".join(t for t in texts if t))
        if any(cells):
            lines.append(" | ".join(cells))
    return "[표]\n" + "\n".join(lines) + "\n[/표]"


def hwp_xml_to_text(xml_bytes: bytes) -> str:
    """hwp5proc xml 결과에서 본문과 표를 텍스트로 만듭니다."""
    from lxml import etree

    root = etree.fromstring(xml_bytes, etree.XMLParser(huge_tree=True))
    out = []
    for colset in root.find("BodyText").iter("ColumnSet"):
        for p in colset.findall("Paragraph"):
            out.append(_para_text(p))
    return "\n".join(out)


def convert_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _run(cmd: list[str], path: Path) -> bytes:
    res = subprocess.run(
        [*cmd, str(path)],
        capture_output=True,
        timeout=HWP_TIMEOUT_SEC,
        check=False,
    )
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError(f"코드 {res.returncode}: {err[-1][:150] if err else ''}")
    return res.stdout


def convert_hwp_pyhwp(path: Path) -> str:
    """표까지 살리는 hwp5proc xml → hwp5txt → olefile 직접 읽기 순으로 시도합니다."""
    proc = find_tool("hwp5proc")
    if proc is not None:
        try:
            text = hwp_xml_to_text(_run([*proc, "xml"], path))
            if text.strip():
                return text
        except Exception as e:  # noqa: BLE001
            print(f"        (표 포함 변환 실패 -> hwp5txt 로 대신: {e})", flush=True)
    txt = find_tool("hwp5txt")
    if txt is not None:
        try:
            text = _run(txt, path).decode("utf-8", errors="replace")
            if text.strip():
                return text
        except Exception as e:  # noqa: BLE001
            print(f"        (hwp5txt 실패 -> olefile 로 대신: {e})", flush=True)
    from ai.rahui.hwp_olefile import convert_hwp_olefile

    return convert_hwp_olefile(path)


def convert_hwp(path: Path) -> str:
    """원본을 olefile 로 직접 읽습니다(문단 누락 점검 가능). 읽지 못하면 pyhwp 방식으로 대신합니다."""
    try:
        text = convert_hwp_olefile(path)
        if text.strip():
            return text
    except Exception as e:  # noqa: BLE001  (실패하면 아래 pyhwp 방식으로 대신)
        print(f"        (olefile 변환 실패 -> pyhwp 방식으로 대신: {e})", flush=True)
    return convert_hwp_pyhwp(path)


def convert(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return convert_pdf(path)
    if ext == ".hwp":
        return convert_hwp(path)
    raise ValueError(f"지원하지 않는 형식: {ext}")


def _result_line(item: dict, csv_len: int, ext: str) -> str:
    """한 건의 결과를 파일 유형별 구간(csv, hwp 또는 pdf)으로 나눠 보여 줍니다."""
    n = item["변환_글자수"]
    got = f"{n}자" if n != "" else "변환 안 됨"
    if ext == "hwp" and n != "":
        body = f"{got} | 표 {item['표_개수']}개 | 커버리지 {item['커버리지']}%"
    else:
        body = got
    return (
        "        ===== csv 시작 (data_list.csv 의 텍스트 열) =====\n"
        f"        글자 수: {csv_len}자\n"
        "        ===== csv 끝 =====\n"
        f"        ===== {ext} 시작 (원본 {ext} 파일을 직접 읽은 결과) =====\n"
        f"        글자 수: {body}\n"
        f"        ===== {ext} 끝 =====\n"
        f"        상태: {item['상태']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RFP 원본을 텍스트로 변환하고 비교합니다."
    )
    parser.add_argument("--all", action="store_true", help="전체 문서를 변환")
    parser.add_argument("--per-group", type=int, default=3, help="샘플 그룹별 문서 수")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"CSV 가 없습니다: {CSV_PATH}")

    rows = read_rows()
    targets = rows if args.all else pick_samples(rows, args.per_group)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"변환 대상 {len(targets)}건 -> {OUT}")

    report = []
    started = time.time()
    for i, row in enumerate(targets, 1):
        name = row["파일명"]
        src = FILES / name
        csv_len = nonspace_len(row["텍스트"])
        src_ext = src.suffix.lower().lstrip(".")
        elapsed = int(time.time() - started)
        print(
            f"[{i}/{len(targets)}] 변환 중... (경과 {elapsed}초) [{src_ext}] {name}",
            flush=True,
        )
        item = {
            "파일명": name,
            "형식": row["파일형식"],
            "원본_MB": "",
            "CSV_글자수": csv_len,
            "변환_글자수": "",
            "변환/CSV": "",
            "표_개수": "",
            "커버리지": "",
            "상태": "",
        }
        if not src.exists():
            item["상태"] = "원본 없음"
        else:
            item["원본_MB"] = round(src.stat().st_size / 1e6, 2)
            try:
                text = convert(src)
                n = nonspace_len(text)
                if n == 0:
                    raise RuntimeError("변환 결과가 비었습니다")
                if src.suffix.lower() == ".hwp":  # 원본 문단이 빠짐없이 들어갔는지 점검
                    cov, missing = coverage(src, text)
                    item["커버리지"] = round(cov * 100, 3)
                    if cov < COVERAGE_MIN:
                        raise RuntimeError(
                            f"누락 의심 {len(missing)}문단 (커버리지 {cov * 100:.3f}%)"
                        )
                (OUT / f"{src.stem}.txt").write_text(text, encoding="utf-8")
                item["변환_글자수"] = n
                item["변환/CSV"] = round(n / csv_len, 2) if csv_len else ""
                if src.suffix.lower() == ".hwp":  # pdf 는 표를 따로 세지 못함
                    item["표_개수"] = text.count("[표]")
                item["상태"] = "ok"
            except Exception as e:  # 한 건 실패로 전체가 멈추지 않게 함
                item["상태"] = f"실패: {type(e).__name__}: {e}"[:120]
        report.append(item)
        print(_result_line(item, csv_len, src_ext), flush=True)

    cmp_path = OUT / "_compare.csv"
    with open(cmp_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    ok = [r for r in report if r["상태"] == "ok"]
    print(f"\n성공 {len(ok)} / 전체 {len(report)}")
    ratios = sorted(r["변환/CSV"] for r in ok if r["변환/CSV"] != "")
    if ratios:
        print(
            f"변환/CSV 글자수 비율: 최소 {ratios[0]}, 중앙값 {ratios[len(ratios) // 2]}, 최대 {ratios[-1]}"
        )
        print(
            "  1보다 크면 원본 파일에서 읽은 글자가 CSV 텍스트 열보다 많다는 뜻입니다."
        )
    print(f"비교표: {cmp_path}")


if __name__ == "__main__":
    main()
