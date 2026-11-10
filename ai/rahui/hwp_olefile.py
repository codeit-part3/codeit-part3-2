"""pyhwp 로 읽지 못하는 hwp 를 olefile 로 직접 읽는 대체 변환 (표는 [표]...[/표], 셀은 ' | ').

필요 패키지: uv add olefile
"""

import re
import struct
import zlib
from pathlib import Path

_PARA_HEADER = 0x42
_PARA_TEXT = 0x43
_CTRL_HEADER = 0x47
_LIST_HEADER = 0x48
_WIDE_CONTROLS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
}
_SKIP_CTRLS = {b"daeh", b"toof"}  # 머리말/꼬리말: 쪽마다 반복되어 잡음이 됨


def _decode_text(data: bytes) -> str:
    units: list[
        int
    ] = []  # UTF-16 코드 단위. 이모지·희귀 한자는 2개가 한 쌍이라 마지막에 한꺼번에 해석
    i, n = 0, len(data)
    while i + 1 < n:
        c = struct.unpack_from("<H", data, i)[0]
        if c >= 32:
            if not 0xE000 <= c <= 0xF8FF:  # 한글 전용 기호(사용자 정의 영역)는 버림
                units.append(c)
            i += 2
        elif c in _WIDE_CONTROLS:
            if c == 9:
                units.append(32)
            i += 16
        elif c == 10:
            units.append(10)
            i += 2
        elif c in (24, 25, 30, 31):
            units.append(32)
            i += 2
        else:
            i += 2
    return struct.pack(f"<{len(units)}H", *units).decode("utf-16-le", errors="replace")


def _records(path: Path) -> list[tuple[int, int, bytes]]:
    import olefile

    ole = olefile.OleFileIO(str(path))
    try:
        header = ole.openstream("FileHeader").read()
        compressed = bool(struct.unpack_from("<I", header, 36)[0] & 1)
        sections = sorted(
            (s for s in ole.listdir() if len(s) == 2 and s[0] == "BodyText"),
            key=lambda s: int(re.sub(r"\D", "", s[1]) or 0),
        )
        out = []
        for sec in sections:
            raw = ole.openstream(sec).read()
            if compressed:
                raw = zlib.decompress(raw, -15)
            pos = 0
            while pos + 4 <= len(raw):
                head = struct.unpack_from("<I", raw, pos)[0]
                pos += 4
                tag, level, size = head & 0x3FF, (head >> 10) & 0x3FF, head >> 20
                if size == 0xFFF:
                    size = struct.unpack_from("<I", raw, pos)[0]
                    pos += 4
                out.append((tag, level, raw[pos : pos + size]))
                pos += size
        return out
    finally:
        ole.close()


def _tree(records):
    """레벨(들여쓰기) 정보로 레코드를 트리로 만듭니다. 노드 = [tag, data, children]."""
    root = [None, b"", []]
    stack = [(-1, root)]
    for tag, level, data in records:
        node = [tag, data, []]
        while stack[-1][0] >= level:
            stack.pop()
        stack[-1][1][2].append(node)
        stack.append((level, node))
    return root


def _paragraph(node) -> str:
    """문단 하나의 글자. 문단에 딸린 표/글상자 등의 글도 이어 붙입니다."""
    text, extra = "", []
    for tag, data, children in node[2]:
        if tag == _PARA_TEXT:
            text = _decode_text(data)
        elif tag == _PARA_HEADER:  # 문단 안에 문단이 겹쳐 들어간 문서도 있음
            inner = _paragraph([tag, data, children])
            if inner:
                extra.append(inner)
        elif tag == _CTRL_HEADER:
            ctrl = data[:4]
            if ctrl in _SKIP_CTRLS:
                continue
            if ctrl == b" lbt":
                extra.append(_table([None, data, children]))
            else:
                inner = _paragraphs(children)
                if inner:
                    extra.append(inner)
    return "\n".join(p for p in [text.strip(), *extra] if p)


def _paragraphs(nodes) -> str:
    lines = []
    for n in nodes:
        if n[0] == _PARA_HEADER:
            lines.append(_paragraph(n))
        else:  # 도형·글상자 등은 문단이 한 단계 더 안쪽에 있을 수 있음
            lines.append(_paragraphs(n[2]))
    return "\n".join(x for x in lines if x)


def _table(ctrl) -> str:
    """표 컨트롤: LIST_HEADER 가 셀의 시작이고, 그 뒤 문단들이 셀 내용입니다."""
    rows: dict[int, list[str]] = {}
    row, cell_paras = None, []

    def flush():
        if row is not None:
            txt = " ".join(p for p in cell_paras if p)
            rows.setdefault(row, []).append(txt)

    for tag, data, children in ctrl[2]:
        if tag == _LIST_HEADER:
            flush()
            cell_paras = []
            row = struct.unpack_from("<H", data, 10)[0] if len(data) >= 12 else 0
        elif tag == _PARA_HEADER and row is not None:
            cell_paras.append(_paragraph([tag, data, children]))
    flush()
    lines = [" | ".join(rows[r]) for r in sorted(rows) if any(rows[r])]
    return "[표]\n" + "\n".join(lines) + "\n[/표]"


def convert_hwp_olefile(path: Path) -> str:
    root = _tree(_records(path))
    lines = (_paragraph(n) for n in root[2] if n[0] == _PARA_HEADER)
    return "\n".join(line for line in lines if line)


def flat_paragraphs(path: Path) -> list[str]:
    """누락 점검의 기준: 본문의 모든 문단 글자를 구조와 상관없이 순서대로 뽑습니다.

    머리말/꼬리말은 변환에서도 뺐으므로 여기서도 뺍니다.
    """
    out: list[str] = []
    skip_below = None
    for tag, level, data in _records(path):
        if skip_below is not None:
            if level > skip_below:
                continue
            skip_below = None
        if tag == _CTRL_HEADER and data[:4] in _SKIP_CTRLS:
            skip_below = level
        elif tag == _PARA_TEXT:
            out += [ln.strip() for ln in _decode_text(data).split("\n") if ln.strip()]
    return out


def _key(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


def coverage(path: Path, converted: str) -> tuple[float, list[str]]:
    """원본 문단 중 변환 결과에 들어 있는 글자의 비율(0~1)과 빠진 문단 목록."""
    haystack = _key(converted)
    total = missing_chars = 0
    missing: list[str] = []
    for para in flat_paragraphs(path):
        key = _key(para)
        if len(key) < 4:
            continue
        total += len(key)
        if key not in haystack:
            missing_chars += len(key)
            missing.append(para)
    return (1 - missing_chars / total if total else 1.0), missing
