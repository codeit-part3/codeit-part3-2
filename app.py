"""RAG 프로젝트 Gradio 인터페이스 (주제 확정 전 임시 버전).

엔진이 아직 없어도 화면을 확인할 수 있도록, 엔진 모듈이 없으면 테스트 응답을 보여 줍니다.
(이 응답은 LLM을 호출하지 않아 토큰을 사용하지 않습니다.)

엔진은 `engines/<id>/__init__.py` 에 `answer(message, history)` 함수를 만들어 두면
이 파일을 수정하지 않아도 해당 탭에 연결되도록 구성해 보았습니다.
"""

import importlib
import importlib.util
import inspect
import os
from pathlib import Path

import gradio as gr

APP_TITLE = "임시 제목"

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

# (엔진 id, 탭 이름, 소개). 주제가 정해지면 이 목록만 수정하면 됩니다.
ENGINES = [
    ("temp1", "temp1", "임시 엔진 1"),
    ("temp2", "temp2", "임시 엔진 2"),
    ("temp3", "temp3", "임시 엔진 3"),
    ("temp4", "temp4", "임시 엔진 4"),
    ("temp5", "temp5", "임시 엔진 5"),
    ("temp6", "temp6", "임시 엔진 6"),
]


def _engine_exists(engine_id):
    """engines/<id> 모듈이 있는지만 확인합니다 (실제로 불러오지는 않습니다)."""
    try:
        return importlib.util.find_spec(f"engines.{engine_id}") is not None
    except ModuleNotFoundError:  # engines 폴더 자체가 아직 없는 경우
        return False


def _plain_text(content):
    """Gradio가 주는 메시지 형태(문자열, {"text": ...}, 조각 목록)를 문자열로 바꿉니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        return "".join(_plain_text(part) for part in content)
    return ""


def _with_plain_text(answer):
    """엔진 함수가 항상 문자열 질문과 문자열 기록만 받도록 감쌉니다.

    보내기 버튼이 달린 입력창은 질문을 {"text": ..., "files": []} 형태로 넘겨주기 때문에,
    엔진 담당자는 이 부분을 신경 쓰지 않고 answer(message, history) 만 만들면 됩니다.
    history 는 [{"role": "user" 또는 "assistant", "content": 문자열}, ...] 입니다.
    """

    def plain_args(message, history):
        plain_history = [
            {"role": turn["role"], "content": _plain_text(turn["content"])}
            for turn in history
        ]
        return _plain_text(message), plain_history

    if inspect.isgeneratorfunction(
        answer
    ):  # 답변을 조금씩 보여 주는(yield) 엔진도 지원합니다.

        def wrapped(message, history):
            yield from answer(*plain_args(message, history))

    else:

        def wrapped(message, history):
            return answer(*plain_args(message, history))

    return wrapped


def load_engine(engine_id, label):
    """엔진이 있으면 그 answer 함수를, 없으면 테스트 응답 함수를 사용합니다.

    엔진 폴더가 있는데 내부에서 오류가 나면 테스트 응답으로 넘어가지 않고
    오류를 그대로 보여 주도록 했습니다. (문제를 놓치지 않기 위해서입니다.)
    """
    if _engine_exists(engine_id):
        module = importlib.import_module(f"engines.{engine_id}")
        if not hasattr(module, "answer"):
            raise AttributeError(
                f"engines/{engine_id} 에 answer(message, history) 함수가 필요합니다."
            )
        return _with_plain_text(module.answer)

    def stub(message, history):
        return f"[{label} 엔진 · 테스트 응답] 받은 질문: {message}"

    return _with_plain_text(stub)


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
                gr.ChatInterface(
                    load_engine(engine_id, label),
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
                    examples=["테스트 질문입니다"],
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
