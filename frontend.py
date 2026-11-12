"""RAG 프로젝트 Gradio 인터페이스 (백엔드 연결 버전).

채팅 기록은 백엔드가 session_id 기준으로 관리합니다.
이 파일은 화면을 그리고, 탭마다 session_id를 만들어 백엔드 /chat 으로 질문을 보냅니다.
"""

import os
import uuid
from pathlib import Path

import gradio as gr
import httpx
from dotenv import get_key

# 서버 접속용 API KEY입니다. Request객체의 header에 포함합니다.
X_API_KEY = get_key(dotenv_path=".env", key_to_get="X_API_KEY")

APP_TITLE = "임시 제목"

# 백엔드 서버 주소입니다. 로컬에서는 기본값을 쓰고, VM에서는 환경 변수 BACKEND_URL 로 바꿉니다.
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# 백엔드 응답을 기다리는 최대 시간(초)입니다. httpx 기본값(5초)은 LLM 답변에 너무 짧아 늘렸습니다.
# 서버에 아예 연결이 안 될 때는 5초 안에 포기하고, 연결된 뒤 답변을 기다리는 시간만 길게 잡습니다.
# 환경 변수 BACKEND_TIMEOUT 으로 바꿀 수 있습니다.
BACKEND_TIMEOUT = httpx.Timeout(float(os.getenv("BACKEND_TIMEOUT", "60")), connect=5.0)

# 화면 디자인(CSS)은 같은 폴더의 style.css 에 모아 두었습니다.
CSS_PATH = Path(__file__).parent / "style.css"

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
    ("temp1", "temp1", "임시 엔진 1"),
    ("temp2", "temp2", "임시 엔진 2"),
    ("temp3", "temp3", "임시 엔진 3"),
    ("temp4", "temp4", "임시 엔진 4"),
    ("temp5", "temp5", "임시 엔진 5"),
]


def call_backend(engine_id: str, session_id: str, message: str) -> str:
    """백엔드 /chat 을 호출하고 응답의 answer 문자열을 돌려줍니다."""
    url = f"{BACKEND_URL}/chat/"
    headers = {
        "X-API-KEY": X_API_KEY
    }  # HTTP 헤더는 _를 -로 변환해서 읽기 때문에, 연결문자로 -를 사용했습니다.
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
                    make_chat_fn(engine_id),
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
        css_paths=[CSS_PATH],
        js=FORCE_LIGHT_JS,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
