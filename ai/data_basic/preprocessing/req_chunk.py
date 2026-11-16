"""dh 방식(요구사항 단위) 청킹을 우리 코퍼스에 맞춰 시험하기 위한 모듈.

- dh 의 split_requirements 를 가져온 사본(preprocessing/req_split.py)을 쓴다. 우리 변환 텍스트는 표 칸을
  '|' 로 구분하므로(예: '요구사항 고유번호 | SFR-001') 사본의 시작점 정규식에서 '|' 를 허용한다.
  (ai/dh 원본은 수정하지 않았고, 이 폴더는 ai/dh 없이 동작한다)
- 요구사항은 한 청크(최대 REQ_MAX, 넘으면 REQ_OVERLAP 겹쳐 자름), 나머지 본문은 일반 청크로 만든다.
- dh 의 compact_requirement(틀 줄 삭제 등 본문 재작성)는 쓰지 않는다. 우리 골든셋 인용은 우리 변환 텍스트
  그대로를 기준으로 하므로 본문을 바꾸면 청킹 효과와 섞이기 때문. 앞에 '[ID] 이름' 한 줄만 붙인다.
- 일반 본문·긴 요구사항 분할은 dh 와 같은 langchain RecursiveCharacterTextSplitter (없으면 간이 구현, SPLITTER 변수에 기록).
- 한계: dh 의 split_requirements 는 (1) 같은 ID 가 여럿이면 가장 긴 구간만 남기고 (2) 구간 사이·장 제목 이후
  텍스트는 '나머지'에 넣지 않는다. 그래서 원문 전체가 청크에 들어가지 않을 수 있다 (coverage 로 확인).
"""

from ai.data_basic.preprocessing import req_split as _rs

REQ_MAX, REQ_OVERLAP, REQ_MIN = 1500, 150, 50
GEN_SIZE, GEN_OVERLAP = 1000, 200


def _req_split():
    return _rs  # dh 코드의 사본(preprocessing/req_split.py). '|' 허용 수정은 사본에 들어 있음


SPLITTER = "미사용"  # 어떤 분할기를 썼는지 (결과 기록용): 'langchain' 또는 '간이 구현'


def rsplit(text: str, size: int, overlap: int) -> list[str]:
    """dh 와 같은 langchain RecursiveCharacterTextSplitter 를 쓰고, 설치돼 있지 않으면 같은 순서의 간이 구현을 쓴다."""
    global SPLITTER
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        SPLITTER = "langchain"
        sp = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [x for x in sp.split_text(text) if x.strip()]
    except ImportError:
        SPLITTER = "간이 구현"
    return _rsplit_simple(text, size, overlap)


def _rsplit_simple(text: str, size: int, overlap: int) -> list[str]:
    """langchain 이 없을 때 쓰는 간이 구현 (빈 줄 -> 줄 -> '. ' -> 공백 -> 글자 순)."""
    seps = ["\n\n", "\n", ". ", " ", ""]

    def split(t: str, level: int) -> list[str]:
        if len(t) <= size:
            return [t]
        sep = seps[level]
        parts = t.split(sep) if sep else list(t)
        if len(parts) == 1 and level + 1 < len(seps):
            return split(t, level + 1)
        out, cur = [], ""
        for p in parts:
            piece = p
            if len(piece) > size and level + 1 < len(seps):
                if cur:
                    out.append(cur)
                    cur = ""
                out.extend(split(piece, level + 1))
                continue
            cand = f"{cur}{sep}{piece}" if cur else piece
            if len(cand) > size and cur:
                out.append(cur)
                tail = cur[-overlap:] if overlap else ""
                cur = f"{tail}{sep}{piece}" if tail else piece
                if len(cur) > size:
                    cur = piece
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out

    return [x.strip() for x in split(text, 0) if x.strip()]


def chunk_req(
    text: str, rest_kind: str = "general", size: int = 1000
) -> list[tuple[str, str]]:
    """(섹션, 본문) 목록. rest_kind: 'general'(dh 일반 청킹) 또는 'struct'(우리 구조 청킹으로 나머지 본문 분할)."""
    rs = _req_split()
    reqs, rest, mode = rs.split_requirements(text)
    out: list[tuple[str, str]] = []
    if mode is None:  # 요구사항 표를 못 찾은 문서: 전체가 일반 본문
        rest = text
        reqs = []
    for q in reqs:
        body = f"[{q['req_id']}] {q.get('req_name') or ''}\n{q['text']}".strip()
        if len(body) < REQ_MIN:
            rest += "\n" + body
            continue
        pieces = [body] if len(body) <= REQ_MAX else rsplit(body, REQ_MAX, REQ_OVERLAP)
        for k, p in enumerate(pieces):
            out.append(("", p if k == 0 else f"[{q['req_id']}] (계속)\n{p}"))
    if rest_kind == "struct":
        from ai.data_basic.experiments.chunk_compare import chunk_struct

        out.extend(chunk_struct(rest, size))
    else:
        out.extend(("", p) for p in rsplit(rest, GEN_SIZE, GEN_OVERLAP))
    return out
