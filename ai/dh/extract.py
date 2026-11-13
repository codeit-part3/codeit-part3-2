"""
문서 파일 목록 만들기 + 본문 텍스트 추출 (PDF / HWP / HWPX)

노트북 대응 셀:
  - 2. 파일 구조 EDA     : file_df (그래프 제외)
  - 4. 문서 텍스트 추출   : pdf_to_pages, hwp_to_text, hwpx_to_text, extract
  - 5. 텍스트 EDA         : hangul_ratio, 추출 품질 의심 문서(suspect)

[참고 사항]
  - list_files: 숨김 파일·대상이 아닌 확장자 제외 (로컬 폴더에는 .DS_Store 등이 섞일 수 있음, 로컬 전용)
"""

import html  # XML 특수문자(&amp; 등)를 원래 글자(&)로 되돌리기
import re
import struct  # 바이너리(0과 1로 된) 데이터를 숫자로 해석하기
import unicodedata  # 한글 자모 분리 문제 정규화
import zipfile  # zip 파일 열기 (HWPX, DOCX 등은 내부가 zip 구조)
import zlib  # 압축된 데이터 풀기 (HWP 파일 내부가 압축되어 있음)
from pathlib import Path

import olefile  # HWP 같은 OLE 형식 파일 열기
import pandas as pd
import pymupdf as fitz  # PDF 읽기 (예전 이름이 fitz라서 관례상 이 별명을 씀)
from config import ALLOWED_EXTS, MIN_CHARS, MIN_HANGUL_RATIO
from tqdm.auto import tqdm  # 반복문 진행률 막대(progress bar)

# ################################################################
# 타입 힌트(type hint)
# ################################################################
# 함수가 받는 값과 돌려주는 값의 자료형을 적어 두는 표시
#   - 이름: 자료형      -> 파라미터의 자료형  예) text: str
#   - -> 자료형         -> 돌려주는 값의 자료형
#   - list[str]         -> 글자들의 리스트
#   - tuple[A, B, C]    -> 값 3개를 묶은 튜플 (순서대로 A, B, C 자료형)
#   - Optional[str]     -> 글자(str) 또는 None
#   - Union[str, Path]  -> 글자(str) 또는 Path 둘 중 하나
#
# 파일 경로는 글자('data/a.hwp')로도, Path 객체로도 받을 수 있으므로 별칭(alias)을 하나 만들어 씀
PathLike = str | Path  # PathLike = Union[str, Path]

# PDF 파일 내부 구조의 사소한 오류 경고(MuPDF error: ...)를 화면에 출력하지 않음
# (추출 결과에는 영향 없음)
fitz.TOOLS.mupdf_display_errors(False)


# ################################################################
# 0. 폴더 안의 모든 파일 정보를 표(file_df)로 만들기 (노트북 2단계)
# ################################################################
def list_files(files_dir: PathLike) -> pd.DataFrame:
    # files_dir: 원본 문서 폴더 (예: data/files)
    # 반환값    : 파일 1개 = 1행인 표 (열: 파일명_실제, ext, size_kb, path)
    rows = []  # 파일 하나당 정보 1줄씩 여기에 모음

    # rglob('*'): files_dir 아래를 하위 폴더까지 전부 훑음
    for p in Path(files_dir).rglob("*"):
        # 폴더는 건너뛰고 '파일'만 처리
        if not p.is_file():
            continue

        # 숨김 파일(.DS_Store 등)과 오피스 임시 파일(~$로 시작) 제외
        if p.name.startswith(".") or p.name.startswith("~$"):
            continue

        # 확장자: '.PDF' -> 소문자 '.pdf' -> 앞의 점 제거 'pdf'
        ext = p.suffix.lower().lstrip(".")

        # 처리 대상 확장자만
        if ext not in ALLOWED_EXTS:
            continue

        # 한글 파일명 정규화: macOS에서 'ㅎ+ㅏ+ㄴ'처럼 자모가 분리된 이름을 '한'으로 합침
        file_name = unicodedata.normalize("NFC", p.name)

        # 파일 크기: 바이트(byte) 단위를 KB로 변환
        size_kb = p.stat().st_size / 1024

        # 이 파일의 정보를 딕셔너리 1개로 묶어 리스트에 추가
        rows.append(
            {
                "파일명_실제": file_name,
                "ext": ext,
                "size_kb": size_kb,
                "path": str(p),
            }
        )

    # 딕셔너리 리스트 -> 표(DataFrame)로 변환
    file_df = pd.DataFrame(rows)
    print("총 파일 수:", len(file_df))
    if len(file_df):
        print(file_df["ext"].value_counts().to_dict())
    return file_df


# ################################################################
# 1. PDF -> 페이지별 텍스트 리스트
# ################################################################
def pdf_to_pages(path: PathLike) -> list[str]:
    # path  : PDF 파일 경로
    # 반환값: 페이지별 텍스트 리스트
    pages = []
    with fitz.open(path) as doc:  # with: 블록이 끝나면 파일 자동 닫기
        for page in doc:  # 페이지를 한 장씩 꺼냄
            pages.append(page.get_text("text"))  # 'text': 서식 없이 글자만
    return pages  # 예) ['1쪽 내용', '2쪽 내용', ...]


# ################################################################
# 2. HWP (한글 5.0 바이너리 형식)
# ################################################################
# [HWP 파일 구조 요약]
# - HWP 파일은 'OLE'라는 형식: 파일 하나 안에 작은 폴더/파일 구조가 들어 있음
#     FileHeader          ← 문서 설정 (압축 여부, 배포용 여부 등)
#     BodyText/Section0   ← 본문 1구역
#     BodyText/Section1   ← 본문 2구역 ...
#     PrvText             ← 미리보기용 텍스트 (앞부분 일부)
# - 각 Section 안은 '레코드(record)'가 줄줄이 이어진 구조
#     [헤더 4바이트: 종류(tag)·크기(size)] [내용 size바이트] [헤더] [내용] ...
#   -> 택배 상자처럼 '라벨(헤더)'을 읽고, 라벨에 적힌 크기만큼 내용을 꺼내는 걸 반복
# - 그중 종류 번호가 67(문단 텍스트)인 레코드에만 글자가 들어 있음

HWPTAG_PARA_TEXT = 67  # '문단 텍스트' 레코드의 종류 번호 (HWP 공식 규격)

# 문단 텍스트 안에는 글자 외에 '제어 문자'(표, 그림, 탭 위치 등)가 섞여 있음
# 제어 문자는 종류에 따라 차지하는 칸 수가 다름: 1칸짜리 / 8칸짜리
LINE_BREAK_CODES = (10, 13)  # 줄바꿈, 문단 끝 (1칸)
TAB_CODE = 9  # 탭 (8칸)
ONE_CHAR_CONTROLS = (0, 24, 25, 26, 27, 28, 29, 30, 31)  # 그 밖의 1칸짜리 제어 문자
CONTROL_WIDTH = 8  # 8칸짜리 제어 문자의 길이

# 서로게이트(surrogate) 범위
# - 2바이트(1칸)로 표현할 수 없는 드문 글자(일부 특수 기호, 이모지 등)는
#   '앞 반쪽(0xD800~0xDBFF) + 뒤 반쪽(0xDC00~0xDFFF)' 두 칸에 나눠 저장됨
# - 두 칸을 따로 chr()로 바꾸면 '반쪽짜리 글자 2개'가 되어 UTF-8 저장 시 에러가 남
#   -> 두 칸을 계산식으로 합쳐 원래 글자 1개로 만듦
HIGH_SURROGATE = (0xD800, 0xDBFF)
LOW_SURROGATE = (0xDC00, 0xDFFF)


# ---------------------------------------------------------------
# 2-1. 문단 텍스트 레코드(바이트) -> 글자
# ---------------------------------------------------------------
def _decode_para_text(rec: bytes) -> str:
    # rec   : 문단 텍스트 레코드의 내용 (바이트)
    # 반환값: 문단 글자 (제어 문자는 줄바꿈·탭만 남기고 제거)
    # HWP 글자는 한 글자당 2바이트 (UTF-16)
    n = len(rec) // 2  # 글자(칸) 수

    # struct.unpack_from: 바이트를 숫자로 해석
    #   '<'  : 리틀 엔디언(little-endian, 작은 자리 바이트가 먼저 오는 저장 순서)
    #   'nH' : 2바이트 양수(unsigned short)를 n개
    # 결과: 각 칸의 글자 코드 번호 목록  예) (54620, 44544, ...) = '한', '글', ...
    chars = struct.unpack_from(f"<{n}H", rec)

    out = []
    i = 0
    while i < n:
        c = chars[i]

        # --- 서로게이트 쌍 -> 글자 1개 ---
        if HIGH_SURROGATE[0] <= c <= HIGH_SURROGATE[1]:  # 앞 반쪽이 나오면
            # 다음 칸이 뒤 반쪽이면 두 칸을 합쳐 한 글자로
            #   글자 번호 = 0x10000 + (앞 반쪽 - 0xD800) × 1024 + (뒤 반쪽 - 0xDC00)
            #   << 10 : 왼쪽으로 10비트 밀기 = 1024를 곱하는 것과 같음
            if i + 1 < n and LOW_SURROGATE[0] <= chars[i + 1] <= LOW_SURROGATE[1]:
                code = 0x10000 + ((c - 0xD800) << 10) + (chars[i + 1] - 0xDC00)
                out.append(chr(code))
                i += 2  # 두 칸을 한 번에 사용했으므로 2칸 이동
            else:
                i += 1  # 짝이 없는 반쪽 글자는 버림
            continue  # continue: 아래 코드를 건너뛰고 다음 반복으로
        if LOW_SURROGATE[0] <= c <= LOW_SURROGATE[1]:  # 앞 반쪽 없이 나온 뒤 반쪽
            i += 1  # 버림
            continue

        if c >= 32:  # 32 이상 = 일반 글자
            out.append(chr(c))  # chr(): 코드 번호 -> 글자
            i += 1
        elif c in LINE_BREAK_CODES:  # 줄바꿈
            out.append("\n")
            i += 1
        elif c == TAB_CODE:  # 탭: '\t'로 바꾸고 8칸 건너뜀
            out.append("\t")
            i += CONTROL_WIDTH
        elif c in ONE_CHAR_CONTROLS:  # 1칸짜리 제어 문자: 버리고 1칸 이동
            i += 1
        else:  # 표/그림 등 8칸짜리 제어 문자: 버리고 8칸 건너뜀
            i += CONTROL_WIDTH

    return "".join(out)  # 글자 리스트를 하나의 문자열로 합침


# ---------------------------------------------------------------
# 2-2. HWP 파일 -> 본문 전체 텍스트
# ---------------------------------------------------------------
def section_number(name: str) -> int:
    # 'Section12' -> 12 (정렬할 때 숫자 크기 순서로 비교하기 위해)
    return int(name.replace("Section", ""))


def hwp_to_text(path: PathLike) -> str:
    # path  : HWP(5.0 바이너리) 파일 경로
    # 반환값: 본문 전체 텍스트 (문단을 줄바꿈으로 이어 붙임)
    # 배포용(암호화) 문서면 ValueError를 일으킴 -> extract()에서 미리보기 텍스트로 대신함
    with olefile.OleFileIO(path) as ole:
        # --- (1) 문서 설정 읽기 ---
        header = ole.openstream("FileHeader").read()

        # FileHeader의 36번째 바이트부터 4바이트가 '속성' 숫자 ('<I' = 4바이트 양수)
        # [0]: unpack_from은 결과를 튜플로 주므로 첫 번째 값만 꺼냄
        flags = struct.unpack_from("<I", header, 36)[0]

        # 속성 숫자의 각 비트(bit)가 켜져 있는지 & 로 확인
        compressed = bool(flags & 1)  # 1번째 비트: 본문 압축 여부
        distribution = bool(flags & 4)  # 3번째 비트: 배포용 문서 여부

        if distribution:  # 배포용 문서는 본문이 암호화되어 있어 읽을 수 없음
            raise ValueError("배포용(암호화) HWP")

        # --- (2) 본문 구역(Section) 목록 찾기 ---
        # listdir(): OLE 안의 모든 경로를 리스트로  예) [['FileHeader'], ['BodyText', 'Section0'], ...]
        section_names = []
        for entry in ole.listdir():
            if entry[0] == "BodyText" and entry[1].startswith("Section"):
                section_names.append(entry[1])

        # 숫자 순서로 정렬 (글자 순서로 정렬하면 Section10이 Section2보다 앞에 옴)
        section_names.sort(key=section_number)

        # --- (3) 구역마다 레코드를 하나씩 읽기 ---
        texts = []
        for sec in section_names:
            data = ole.openstream(f"BodyText/{sec}").read()

            if compressed:
                # -15: 앞에 압축 정보(헤더)가 없는 'raw' 압축 형식이라는 뜻
                data = zlib.decompress(data, -15)

            pos = 0  # 지금 읽고 있는 위치 (바이트 번호)
            while pos + 4 <= len(data):  # 헤더 4바이트를 읽을 공간이 남아 있는 동안
                # 레코드 헤더 4바이트를 숫자 하나로 읽기
                header_value = struct.unpack_from("<I", data, pos)[0]
                pos += 4

                # 32비트 숫자 안에 정보가 나뉘어 들어 있음
                #   0~9번 비트   : 종류 번호(tag)  -> & 0x3FF 로 아래 10비트만 꺼냄
                #   20~31번 비트 : 내용 크기(size) -> >> 20 으로 오른쪽으로 밀고 & 0xFFF 로 12비트만
                tag_id = header_value & 0x3FF
                size = (header_value >> 20) & 0xFFF

                # 크기가 최댓값(0xFFF = 4095)이면, 진짜 크기는 다음 4바이트에 따로 적혀 있음
                if size == 0xFFF:
                    size = struct.unpack_from("<I", data, pos)[0]
                    pos += 4

                # 문단 텍스트 레코드면 글자로 변환해서 저장
                if tag_id == HWPTAG_PARA_TEXT:
                    record = data[pos : pos + size]
                    texts.append(_decode_para_text(record))

                pos += size  # 내용만큼 건너뛰어 다음 레코드 헤더로 이동

        return "\n".join(texts)  # 문단들을 줄바꿈으로 이어 붙임


# ---------------------------------------------------------------
# 2-3. 본문 읽기에 실패했을 때 대신 쓸 미리보기 텍스트
# ---------------------------------------------------------------
def hwp_preview_text(path: PathLike) -> str:
    # path  : HWP 파일 경로 / 반환값: 앞부분 미리보기 텍스트
    # PrvText: 한글 프로그램이 저장해 두는 앞부분 미리보기 (본문 전체가 아님)
    with olefile.OleFileIO(path) as ole:
        raw = ole.openstream("PrvText").read()
        # utf-16-le: 2바이트 글자, 리틀 엔디언 / errors='ignore': 해석 안 되는 바이트는 버림
        return raw.decode("utf-16-le", errors="ignore")


# ################################################################
# 3. HWPX (한글의 새 형식)
# ################################################################
# [HWPX 파일 구조 요약]
# - 실제로는 zip 파일이고, 안에 XML(태그로 된 글자 파일)이 들어 있음
#     Contents/section0.xml, Contents/section1.xml, ...
# - XML 안에서  <hp:p> ... </hp:p> = 문단 하나
#              <hp:t>글자</hp:t>   = 실제 글자 조각

SECTION_RE = r"Contents/section(\d+)\.xml"  # (\d+): 구역 번호 부분을 따로 꺼냄


def hwpx_to_text(path: PathLike) -> str:
    # path  : HWPX 파일 경로 (확장자가 .hwp여도 속이 zip이면 여기로 옴)
    # 반환값: 본문 전체 텍스트
    with zipfile.ZipFile(path) as zf:
        # --- (1) 구역 파일 목록을 (번호, 파일명) 쌍으로 모으기 ---
        section_files = []
        for name in zf.namelist():  # zip 안의 모든 파일 이름
            m = re.match(SECTION_RE, name)
            if m:
                number = int(m.group(1))  # 'section12.xml' -> 12
                section_files.append((number, name))

        section_files.sort()  # 튜플은 첫 값(번호) 기준으로 정렬됨 -> 숫자 순서

        # --- (2) 구역마다 문단 -> 글자 조각 꺼내기 ---
        texts = []
        for _, name in section_files:
            xml = zf.read(name).decode("utf-8", errors="ignore")

            # re.S: 점(.)이 줄바꿈 문자까지 포함하도록 (문단이 여러 줄에 걸쳐 있어도 찾음)
            # .*? : 가능한 한 '짧게' 일치 (문단 하나씩 끊어서 찾기 위해)
            paragraphs = re.findall(r"<hp:p\b.*?</hp:p>", xml, flags=re.DOTALL)

            for p in paragraphs:
                pieces = re.findall(
                    r"<hp:t>(.*?)</hp:t>", p, flags=re.DOTALL
                )  # 글자 조각들
                para_text = "".join(pieces)

                # XML은 &, <, > 를 &amp; &lt; &gt; 로 바꿔 저장함 -> 원래 글자로 복원
                #   예) 'R&amp;D' -> 'R&D'
                texts.append(html.unescape(para_text))

        return "\n".join(texts)


# ################################################################
# 4. 확장자를 보고 맞는 추출 함수를 골라 실행
# ################################################################
# 반환값 3개 (튜플):
#   pages  : 페이지별 텍스트 리스트  예) ['1쪽 내용', '2쪽 내용']
#   method : 실제로 사용한 추출 방법  예) 'pdf', 'hwp', 'hwp_preview'
#   error  : 에러 메시지 (문제가 없으면 None)


def extract(path: PathLike) -> tuple[list[str], str, str | None]:
    # path  : 문서 파일 경로
    # 반환값: (페이지별 텍스트 리스트, 사용한 추출 방법, 에러 메시지 또는 None)
    path = Path(path)
    ext = path.suffix.lower()  # 확장자를 소문자로  예) '.HWP' -> '.hwp'

    # 바깥 try: 어떤 에러가 나도 멈추지 않고 'fail'로 기록하고 넘어가도록
    try:
        # --- PDF ---
        if ext == ".pdf":
            return pdf_to_pages(path), "pdf", None

        # --- HWP / HWPX ---
        if ext in (".hwp", ".hwpx"):
            # 확장자보다 실제 내용으로 판단: zip 구조면 HWPX
            # (확장자는 .hwp인데 속은 HWPX인 파일도 있어서 이렇게 확인)
            if zipfile.is_zipfile(path):
                text = hwpx_to_text(path)
                return (
                    [text],
                    "hwpx",
                    None,
                )  # 페이지 구분이 없으므로 리스트에 1개만 담음

            # zip이 아니면 HWP 바이너리 -> 본문 추출 시도
            # 안쪽 try: 본문 추출에 실패하면 미리보기 텍스트로 대신함
            try:
                text = hwp_to_text(path)
                return [text], "hwp", None
            except Exception as e:  # noqa: BLE001
                text = hwp_preview_text(path)
                return [text], "hwp_preview", f"본문 실패->미리보기: {e}"

        # --- 그 외 확장자 ---
        return [], "unsupported", f"지원하지 않는 확장자 {ext}"

    except Exception as e:  # noqa: BLE001
        # repr(e): 에러 종류와 메시지를 함께 글자로  예) "ValueError('배포용(암호화) HWP')"
        return [], "fail", repr(e)


def extract_all(file_df: pd.DataFrame) -> pd.DataFrame:
    # file_df: list_files가 돌려준 파일 목록 표
    # 반환값 : 파일 1개 = 1행인 추출 결과 표 (열: 파일명_실제, ext, pages, n_pages, method, error, raw_text)
    records = []  # 파일 하나당 결과 딕셔너리 1개씩 모음

    # iterrows(): 표를 한 행씩 꺼냄 -> (행 번호, 행 내용) 한 쌍
    #   행 번호는 쓰지 않으므로 '_'로 받음 (관례: 쓰지 않는 변수 이름)
    # tqdm(..., total=전체 개수): 반복하는 동안 진행률 막대를 표시
    for _, row in tqdm(file_df.iterrows(), total=len(file_df), desc="텍스트 추출"):
        pages, method, err = extract(row["path"])  # 돌려받은 3개 값을 나눠 담기

        records.append(
            {
                "파일명_실제": row["파일명_실제"],
                "ext": row["ext"],
                "pages": pages,  # 페이지별 텍스트 리스트
                "n_pages": len(pages),  # 페이지 수
                "method": method,  # 사용한 추출 방법
                "error": err,  # 에러 메시지 (없으면 None)
                "raw_text": "\n".join(
                    pages
                ),  # 모든 페이지를 줄바꿈으로 이은 전체 텍스트
            }
        )

    docs = pd.DataFrame(records)

    # 추출 방법별 파일 수
    print("추출 방법:", docs["method"].value_counts().to_dict())

    # 에러가 기록된 파일만 보기
    has_error = docs["error"].notna()
    if has_error.any():
        print(docs.loc[has_error, ["파일명_실제", "method", "error"]].to_string())
    return docs


# ################################################################
# 5. 추출 품질 점검 (노트북 5. 텍스트 EDA)
# ################################################################
# 추출이 잘못되면 글자가 깨져서 '¿½¿½' 같은 이상한 문자나 기호만 잔뜩 나옴
# -> 공백을 뺀 전체 글자 중 '완성형 한글'이 몇 %인지로 품질을 가늠


def hangul_ratio(text: str) -> float:
    # text  : 문서 텍스트 / 반환값: 공백을 뺀 글자 중 완성형 한글의 비율 (0~1)
    chars = re.sub(r"\s", "", text)  # 공백/줄바꿈/탭 모두 제거

    # [가-힣]: '가'부터 '힣'까지 = 완성형 한글 글자 전체 (ㄱ, ㅏ 같은 낱자는 포함 안 됨)
    hangul_count = len(re.findall(r"[가-힣]", chars))

    # max(글자 수, 1): 빈 문서면 0으로 나누는 에러가 나므로 최소 1로 나눔 -> 결과 0
    total_count = max(len(chars), 1)

    return hangul_count / total_count  # 0~1 사이 값  예) 0.65 = 65%


def add_quality(docs: pd.DataFrame) -> pd.DataFrame:
    # docs  : extract_all이 돌려준 추출 결과 표
    # 반환값: 같은 표에 n_chars, hangul_ratio, suspect 열을 추가한 표
    docs["n_chars"] = docs["raw_text"].str.len()
    docs["hangul_ratio"] = docs["raw_text"].map(hangul_ratio)

    is_short = docs["n_chars"] < MIN_CHARS
    is_low_hangul = docs["hangul_ratio"] < MIN_HANGUL_RATIO
    docs["suspect"] = is_short | is_low_hangul

    print("추출 품질 의심 문서:", int(docs["suspect"].sum()))
    return docs
