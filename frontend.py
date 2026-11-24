"""RAG 프로젝트 Gradio 인터페이스 (백엔드 연결 버전).

채팅 기록은 백엔드가 session_id 기준으로 관리합니다.
이 파일은 화면을 그리고, 탭마다 session_id를 만들어 백엔드 /chat 으로 질문을 보냅니다.
"""

# =============================================================================
# [수정 내역] 화면 기능 추가
#   - 작성 도우미가 만든 .docx 다운로드 링크, 답변 화면용 정리, 기다리는 동안 보이는 점 3개
#   - 백엔드와 관련된 코드(call_backend, make_chat_fn, 안의 chat)는 원본 그대로이고 한 글자도 바꾸지 않았습니다.
#   - 새 기능은 전부 아래 "[수정 시작]" ~ "[수정 끝]" 사이의 별도 함수(with_ui_extras)에 있고,
#     원본 chat 이 돌려준 답변 글을 받아 화면에 보일 모양으로만 가공합니다.
#   - 원본에서 바뀐 곳: ① import 3줄  ② PROGRESS_CSS_PATH 1줄  ③ 새 함수 묶음 추가
#                       ④ build_app 의 make_chat_fn(...) 을 with_ui_extras(make_chat_fn(...)) 로 감쌈 1곳
#                       ⑤ launch 의 css_paths 에 PROGRESS_CSS_PATH 추가, allowed_paths 추가
#                       ⑥ 표·깨진 글머리 기호 화면용 정리(tidy_tables) 추가, _decorate_answer 1줄 수정 (표 모양 CSS 는 style.css 맨 아래)
# =============================================================================
import os

# [수정 시작 ①] 답변 글에서 파일 경로를 찾고(re), 링크 주소를 만들고(quote), 기다리는 동안 점을 보여 주기(threading) 위한 import
import re
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import httpx
from dotenv import load_dotenv

# [수정 끝 ①]

load_dotenv()
# 서버 접속용 API KEY입니다. Request객체의 header에 포함합니다.
X_API_KEY = os.getenv("X_API_KEY")

APP_TITLE = "임시 제목"

# Test용
# 로컬 백엔드 서버 주소
# BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
# # VM 서버 주소
# BACKEND_URL = os.getenv("BACKEND_URL", "http://136.111.105.21:8000")

# 실사용(compose.yaml파일에서 관리)
BACKEND_URL = os.getenv("BACKEND_URL")

# 백엔드 응답을 기다리는 최대 시간(초)입니다. httpx 기본값(5초)은 LLM 답변에 너무 짧아 늘렸습니다.
# 서버에 아예 연결이 안 될 때는 5초 안에 포기하고, 연결된 뒤 답변을 기다리는 시간만 길게 잡습니다.
# 환경 변수 BACKEND_TIMEOUT 으로 바꿀 수 있습니다.
BACKEND_TIMEOUT = httpx.Timeout(float(os.getenv("BACKEND_TIMEOUT", "60")), connect=5.0)

# 화면 디자인(CSS)은 같은 폴더의 style.css 에 모아 두었습니다.
CSS_PATH = Path(__file__).parent / "style.css"
PROGRESS_CSS_PATH = (
    Path(__file__).parent / "progress.css"
)  # [수정 ②] 기다리는 동안 보이는 통통 튀는 점

# 항상 밝은(화이트) 화면으로 보이도록, 브라우저가 다크 모드여도 라이트 모드로 한 번 전환합니다.
FORCE_LIGHT_JS = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url);
    }
}
"""

# (engine_id, 탭 이름, 소개). engine_id 는 백엔드 config.ENGINE_IDS 와 글자까지 같아야 합니다.
ENGINES = [
    ("local_model", "local_model", "gemma4:3b 기반 로컬 모델"),  # 로컬파이널로교체예정
    ("final", "final", "최종 엔진 (검색 답변)"),  # 파이널
    ("final_writer", "작성 도우미", "가이드·서식·검색 답변"),  # 파이널Writer
]


def call_backend(engine_id: str, session_id: str, message: str) -> str:
    """백엔드 /chat 을 호출하고 응답의 answer 문자열을 돌려줍니다."""
    url = f"{BACKEND_URL}/chat/"
    headers = {
        "X-API-KEY": X_API_KEY
    }  # HTTP 헤더는 _를 -로 변환해서 읽    기 때문에, 연결문자로 -를 사용했습니다.
    request_body = {
        "engine_id": engine_id,
        "session_id": session_id,
        "message": message,
    }

    response = (
        httpx.post(url=url, headers=headers, json=request_body, timeout=BACKEND_TIMEOUT)
        .raise_for_status()
        .json()
    )

    return response["answer"]


def _message_text(message) -> str:
    """입력창 값에서 텍스트만 꺼냅니다.

    multimodal=True 이면 입력이 {"text": ..., "files": [...]} 형태로 들어오므로 text만 씁니다.
    """
    if isinstance(message, dict):
        return message.get("text", "")
    return str(message)


def make_chat_fn(engine_id: str):
    """탭 하나에 연결할 채팅 함수를 만듭니다.

    engine_id 를 바깥 함수의 인자로 받아 탭마다 값을 고정합니다.
    반복문 안에서 람다로 바로 넘기면 모든 탭이 마지막 engine_id 를 쓰게 되므로 이 방식을 씁니다.
    """

    def chat(message, history, session_id: str) -> str:
        # history 인자는 백엔드가 session_id로 기록을 관리하므로 쓰지 않습니다.
        text = _message_text(message).strip()
        if not text:
            return "질문을 입력해 주세요."
        try:
            return call_backend(engine_id, session_id, text)
        except httpx.HTTPStatusError as e:
            # 백엔드에서 에러가 발생한 경우입니다(엔진 로딩, API Key 인증 등).
            resp = e.response.json()
            detail = resp["detail"]
            print(f"[backend error] {e.response.status_code}")
            return detail
        except httpx.ReadTimeout:
            # 서버에는 연결됐지만 답변이 늦어지는 경우입니다.
            return "답변이 오래 걸리고 있습니다. 잠시 후 다시 시도해 주세요."
        except Exception as e:
            # 개발 중에는 터미널에서 원인을 확인할 수 있도록 출력합니다.
            print(f"[backend error] {e}")
            return "서버 연결에 실패했습니다. 잠시 후 다시 시도해 주세요."

    return chat


# =============================================================================
# [수정 시작 ③] 화면 기능 (백엔드 코드와 분리된 별도 함수들)
#
#   find_download_files  : 답변 글의 "파일: ....docx" 줄을 찾아, outputs/guides 안의 실제 파일이면 경로를 돌려줍니다.
#   prettify_answer      : 작성 도우미 답변을 화면에 보일 때만 정리합니다 (서식 줄은 표로, 내부용 줄은 숨김).
#                          백엔드가 저장하는 대화 기록은 원본 그대로라서 "표지도" 같은 후속 질문은 그대로 동작합니다.
#   _decorate_answer     : 위 둘과 다운로드 링크 붙이기를 합쳐, 원본 chat 의 답변을 화면용 글로 바꿉니다.
#   with_ui_extras       : 원본 chat 함수를 감싸서 (1) 기다리는 동안 점 3개를 보여 주고 (2) 답이 오면 화면용 글로 바꿉니다.
#
#   동작 조건: 이 화면 프로그램(frontend.py)과 백엔드가 같은 컴퓨터에서 같은 프로젝트 폴더로 돌 때 다운로드 링크가 붙습니다.
#             파일이 없으면 링크만 안 붙고 답변 글은 그대로 보입니다 (오류 없음). outputs/guides 가 다른 위치면 환경 변수 GUIDE_DIR 로 지정합니다.
#   참고: gr.File 부품을 말풍선에 넣는 방식은 쓰지 않습니다. Gradio 6.31 에서 답변이 끝난 뒤 대화 내용을 다시 읽을 때
#         FileData 오류가 나서 입력창이 멈추기 때문입니다. 그래서 링크 문장 한 줄을 덧붙입니다.
# =============================================================================
# 이 파일(frontend.py)이 있는 폴더 = 프로젝트 루트. 작성 도우미는 outputs/guides 에 .docx 를 저장합니다.
PROJECT_ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = Path(
    os.getenv("GUIDE_DIR", PROJECT_ROOT / "outputs" / "guides")
).resolve()

# 답변 속 "파일: outputs\guides\이름.docx" 줄을 찾는 패턴 (줄 끝까지, .docx 로 끝나는 것만)
DOWNLOAD_LINE = re.compile(r"파일:\s*(.+?\.docx)\s*$", re.MULTILINE)

# 기다리는 동안 보이는 점 3개 (모양과 움직임은 progress.css 의 .bounce-dots)
BOUNCE_DOTS_HTML = '<div class="bounce-dots"><span class="d1"></span><span class="d2"></span><span class="d3"></span></div>'


def find_download_files(answer: str) -> list[str]:
    """답변 글에서 내려받을 수 있는 .docx 파일 경로를 찾아 돌려줍니다 (없으면 빈 목록).

    보안: outputs/guides 밖의 파일은 절대 내보내지 않습니다. 답변 글에 엉뚱한 경로가 적혀 있어도
    DOWNLOAD_DIR 안에 있는 실제 파일일 때만 인정합니다 (경로에 ..이 섞인 경우도 resolve 로 걸러짐).
    """
    found = []
    for raw in DOWNLOAD_LINE.findall(answer):
        path = Path(raw.strip().replace("\\", "/"))  # 윈도우 경로(\)도 읽도록 / 로 통일
        if not path.is_absolute():
            path = PROJECT_ROOT / path  # 답변의 경로는 프로젝트 루트 기준 상대 경로
        path = path.resolve()
        if path.is_file() and path.is_relative_to(DOWNLOAD_DIR):
            found.append(str(path))
    return found


# 작성 도우미 답변 화면용 정리 (화면에 보이는 글만 바꿈. 백엔드의 대화 기록은 원본 그대로)
# - 서식 본문의 "• 칸 | 칸 | 칸" 줄은 표로 바꿔서 보여 줍니다.
# - 사용자에게 필요 없는 내부용 줄(값 표시 개수, 파일 이름·줄 번호 출처, 파일 경로, 문서: 줄)은 숨기거나 쉬운 말로 바꿉니다.
_RE_COUNT_LINE = re.compile(r"^값 표시:.*$", re.MULTILINE)
_RE_SRC_LINE = re.compile(r"^출처:\s*(.+?)\.txt\s*,\s*원문.*$", re.MULTILINE)
_RE_DOC_LINE = re.compile(r"^문서:\s*.+?\.txt\s*$", re.MULTILINE)
_RE_FILE_LINE = re.compile(r"^파일:\s*.+?\.docx\s*$", re.MULTILINE)


def _pretty_source(name: str) -> str:
    """'기관명_사업명 ' 형태의 파일 이름을 '기관명 · 사업명'으로 바꿉니다."""
    name = name.strip()
    if "_" in name:
        org, biz = name.split("_", 1)
        return f"{org.strip()} · {biz.strip()}"
    return name


def _bullets_to_tables(text: str) -> str:
    """연속된 '• 칸 | 칸 | 칸' 줄을 하나의 표(마크다운)로 바꿉니다. 칸 수가 다른 줄은 가장 긴 줄에 맞춰 빈칸을 채웁니다."""
    out, grid = [], []

    def flush():
        if not grid:
            return
        width = max(len(r) for r in grid)
        out.append("")
        out.append("| " + " | ".join([" "] * width) + " |")
        out.append("|" + "---|" * width)
        for r in grid:
            r = r + [""] * (width - len(r))
            out.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
        out.append("")
        grid.clear()

    for ln in text.split("\n"):
        if ln.startswith("•") and " | " in ln:
            raw = re.sub(
                r"(?:\s*\|)+\s*$", "", ln.lstrip("•").strip()
            )  # 끝에 남은 빈 칸 기호 제거
            grid.append([c.strip() for c in raw.split(" | ")])
        else:
            flush()
            out.append(ln)
    flush()
    return "\n".join(out)


def prettify_answer(answer: str, has_file_link: bool) -> str:
    """작성 도우미 답변을 화면에 보일 글로 다듬습니다 (다른 엔진의 답변은 해당 줄이 없어서 그대로 나옵니다)."""
    text = answer
    if (
        has_file_link and "작성 예시예요" in text
    ):  # 파일을 요청한 경우: 본문 표 대신 제목 + 안내 + 링크만 보여 줌 (미리보기는 Word 파일에서)
        keep = [
            ln
            for ln in text.split("\n")
            if ln.endswith("작성 예시예요.")
            or ln.startswith("참고용 작성 예시예요")
            or ln.startswith("※ [예시]")
        ]
        keep = [("> " + ln) if ln.startswith("참고용") else ln for ln in keep]
        return "\n\n".join(keep).strip()
    if "작성 예시예요" in text:  # 서식 작성 예시 답변: 본문을 표로
        text = _bullets_to_tables(text)
        text = re.sub(r"^(참고용 작성 예시예요.*)$", r"> \1", text, flags=re.MULTILINE)
    text = _RE_COUNT_LINE.sub(
        "범례: [예시] 가상 값 · [직접 작성] 회사 정보를 직접 입력 · [문서] 공고문에서 가져온 값",
        text,
    )
    text = _RE_SRC_LINE.sub(lambda m: f"출처: {_pretty_source(m.group(1))}", text)
    text = _RE_DOC_LINE.sub(
        "", text
    )  # 서식 목록 답변 끝의 "문서: 파일명" 줄은 화면에서 숨김 (기록에는 남음)
    if has_file_link:
        text = _RE_FILE_LINE.sub("", text)  # 파일 경로 줄은 아래 다운로드 링크로 대신함
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decorate_answer(answer: str) -> str:
    """원본 chat 이 돌려준 답변 글을 화면에 보일 글로 바꿉니다 (정리 + 다운로드 링크 붙이기)."""
    files = find_download_files(answer)  # 파일 찾기는 원본 answer 로
    shown = tidy_tables(prettify_answer(answer, bool(files)))  # [수정 ⑥] 표 정리 추가
    shown = shown.replace(
        "**", ""
    )  # 굵게 표시(**)가 '…)**입니다'처럼 한글 앞에서 그대로 보이는 것을 막습니다.
    if files:
        shown = re.sub(
            r"(?m)^\s*[-*]\s*파일:.*\n?", "", shown
        )  # 파일 위치 줄은 내부 정보라 숨깁니다 (아래에 내려받기 링크가 있음)
        # 링크는 Gradio 의 파일 주소(/gradio_api/file=절대경로)를 씁니다. launch(allowed_paths) 로 허용한 폴더 안 파일만 열립니다.
        # 한글·공백·괄호가 든 파일 이름도 쓸 수 있게 quote 로 주소를 인코딩합니다.
        links = "\n".join(
            f"[📄 {Path(f).name} 내려받기](/gradio_api/file={quote(Path(f).as_posix(), safe='/:')})"
            for f in files
        )
        return shown + "\n\n" + links
    return shown


def with_ui_extras(chat_fn):
    """원본 chat 함수를 감싸 화면 기능만 얹습니다. 백엔드 호출은 원본 chat_fn 이 그대로 합니다.

    (1) 답을 기다리는 동안 점 3개를 보여 주고, (2) 답이 오면 화면용 글(정리 + 다운로드 링크)로 바꿉니다.
    채팅 함수의 인자(message, history, session_id)는 원본과 같습니다.
    """

    def chat(message, history, session_id: str):
        if not _message_text(message).strip():
            yield chat_fn(
                message, history, session_id
            )  # 빈 질문 안내는 원본 chat 이 합니다
            return

        result: dict = {}

        def work():
            result["answer"] = chat_fn(message, history, session_id)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        yield BOUNCE_DOTS_HTML
        th.join()

        answer = result.get("answer")
        if answer is None:
            yield "서버 연결에 실패했습니다. 잠시 후 다시 시도해 주세요."
        elif isinstance(answer, str):
            yield _decorate_answer(answer)
        else:
            yield answer

    return chat


# [수정 끝 ③]


# [수정 시작 ⑥] 표·글머리 기호 화면용 정리 (화면에 보일 때만 적용, 백엔드의 대화 기록과 답변 원본은 그대로)
#   1) 원문 OCR 의 깨진 글머리 기호(사설 영역 문자)를 • 로 바꿉니다.
#   2) 표 바로 앞의 "[표] 단 계 | 과 업 | ..." 글자 줄을 지웁니다.
#   (표 칸 모양 CSS 는 style.css 맨 아래에 있습니다)
_PUA_CH = "[\ue000-\uf8ff\U000f0000-\U000ffffd]"
_PUA = re.compile(r"[ \t]*" + _PUA_CH + r"[ \t]*")
_PUA_ONLY_CELL = re.compile(r"\|[ \t]*" + _PUA_CH + r"+[ \t]*(?=\|)")
_TABLE_TAG = re.compile(r"(?m)^[ \t]*\[표\][^\n]*\n+(?=[ \t]*\|)")


def tidy_tables(text: str) -> str:
    for _ in range(
        3
    ):  # 칸에 깨진 기호만 있으면 빈 칸으로 (연속된 칸도 처리하려고 여러 번)
        text = _PUA_ONLY_CELL.sub("| ", text)
    text = _TABLE_TAG.sub(
        "", text
    )  # 표 바로 앞의 "[표] 단 계 | 과 업 | ..." 글자 줄 삭제
    text = _PUA.sub(" • ", text)  # 남은 깨진 글머리 기호 → •
    return re.sub(r"\n{3,}", "\n\n", text)


# [수정 끝 ⑥]


def build_app():
    """엔진마다 탭 하나씩, 채팅 화면을 만듭니다."""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n궁금한 분야의 탭을 골라 질문해 보세요.",
            elem_id="app-header",
        )
        for engine_id, label, intro in ENGINES:
            with gr.Tab(label):
                gr.Markdown(intro)

                # 탭마다 session_id 를 따로 둡니다.
                # 값은 화면을 여는 순간 uuid 로 자동 생성되고, 이후 요청마다 같은 값이 쓰입니다.
                # 화면의 '지우기' 버튼은 화면만 비우고 서버 기록은 지우지 않습니다.
                session_state = gr.State(value=lambda: str(uuid.uuid4()))

                gr.ChatInterface(
                    with_ui_extras(
                        make_chat_fn(engine_id)
                    ),  # [수정 ④] 원본 chat 을 화면 기능으로 감쌈
                    # 채팅 함수의 세 번째 인자(session_id)로 전달됩니다.
                    additional_inputs=[session_state],
                    # dvh: 모바일 주소창 높이를 뺀 실제 화면 높이를 기준으로 하여,
                    # 제목·탭·입력창을 빼고 남는 만큼을 대화 영역으로 씁니다.
                    chatbot=gr.Chatbot(
                        show_label=False,
                        height="calc(100dvh - 340px)",
                        min_height=260,
                        max_height=720,
                    ),
                    # sources=[] : 파일 첨부 버튼은 쓰지 않으므로 숨깁니다.
                    textbox=gr.MultimodalTextbox(
                        placeholder="메시지를 입력하세요",
                        show_label=False,
                        container=False,
                        sources=[],
                    ),
                    multimodal=True,
                    autofocus=False,  # 모바일에서 화면을 열자마자 키보드가 올라오지 않도록 합니다.
                    # examples=["테스트 질문입니다"],
                    api_name=f"chat_{engine_id}",
                )
    return demo


if __name__ == "__main__":
    # 기본값은 내 컴퓨터에서만 접속되는 127.0.0.1 입니다.
    # VM에서는 GRADIO_SERVER_NAME=0.0.0.0 으로 실행하면 외부에서 접속할 수 있습니다.
    build_app().launch(
        theme=gr.themes.Base(),
        css_paths=[CSS_PATH, PROGRESS_CSS_PATH],  # [수정 ⑤]
        js=FORCE_LIGHT_JS,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        # [수정 ⑤] outputs/guides 안의 파일을 화면에서 내려받을 수 있게 허용합니다 (그 밖의 폴더는 허용하지 않음).
        allowed_paths=[str(DOWNLOAD_DIR)],
    )
