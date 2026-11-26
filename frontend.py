"""RAG 프로젝트 Gradio 인터페이스 (백엔드 연결 버전).

채팅 기록은 백엔드가 session_id 기준으로 관리합니다.
이 파일은 화면을 그리고, 탭마다 session_id를 만들어 백엔드 /chat 으로 질문을 보냅니다.
"""

# =============================================================================
# [수정 내역] 화면 기능 추가 & 꾹꾹이 랩스 UI 적용
#   - 백엔드와 관련된 코드(call_backend, make_chat_fn, find_download_files,
#     prettify_answer, _decorate_answer, with_ui_extras, tidy_tables)는 원본 동작을 그대로 유지합니다.
#   - 랜딩 페이지 + 시작하기 전환 → gguk_ui 2단 레이아웃(로컬 꾹꾹이 / 고성능 꾹꾹이 / 작성 꾹꾹이)을 적용했습니다.
#   - 대화창에는 목업 메시지를 넣지 않고 실제 백엔드 연결 채팅만 표시합니다.
# =============================================================================
import base64
import os

# [수정 시작 ①] 답변 글에서 파일 경로를 찾고(re), 링크 주소를 만들고(quote), 기다리는 동안 로딩을 보여 주기(threading) 위한 import
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

APP_TITLE = "꾹꾹이 랩스 | 공고 작성 챗봇 서비스"

# 실사용(compose.yaml파일에서 관리)
BACKEND_URL = os.getenv("BACKEND_URL")

# 백엔드 응답을 기다리는 최대 시간(초)입니다. httpx 기본값(5초)은 LLM 답변에 너무 짧아 늘렸습니다.
# 서버에 아예 연결이 안 될 때는 5초 안에 포기하고, 연결된 뒤 답변을 기다리는 시간만 길게 잡습니다.
# 환경 변수 BACKEND_TIMEOUT 으로 바꿀 수 있습니다.
BACKEND_TIMEOUT = httpx.Timeout(float(os.getenv("BACKEND_TIMEOUT", "300")), connect=5.0)

# 화면 디자인(CSS)은 같은 폴더의 style.css 에 모아 두었습니다.
CSS_PATH = Path(__file__).parent / "style.css"
PROGRESS_CSS_PATH = (
    Path(__file__).parent / "progress.css"
)  # [수정 ②] 기다리는 동안 보이는 고양이 발바닥 도도동 로딩

# (engine_id, 탭 이름, 소개). engine_id 는 백엔드 config.ENGINE_IDS 와 글자까지 같아야 합니다.
ENGINES = [
    ("local_final", "로컬 꾹꾹이", "gemma4:3b 기반 로컬 모델"),
    ("final", "고성능 꾹꾹이", "최종 엔진 (검색 답변)"),  # 파이널
    ("final_writer", "작성 꾹꾹이", "가이드·서식·검색 답변"),  # 파이널Writer
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


# =============================================================================
# [수정 시작 ③] 화면 기능 (백엔드 코드와 분리된 별도 함수들)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DOWNLOAD_DIR = Path(
    os.getenv("GUIDE_DIR", PROJECT_ROOT / "outputs" / "guides")
).resolve()

# 답변 속 "파일: outputs\guides\이름.docx" 줄을 찾는 패턴 (줄 끝까지, .docx 로 끝나는 것만)
DOWNLOAD_LINE = re.compile(r"파일:\s*(.+?\.docx)\s*$", re.MULTILINE)

# 기다리는 동안 보이는 고양이 발바닥 3개 도도동 로딩 (모양과 움직임은 progress.css 의 .bounce-dots)
BOUNCE_DOTS_HTML = (
    '<div class="bounce-dots">'
    '<span class="paw-loader">'
    '<span class="paw-icon d1"></span>'
    '<span class="paw-icon d2"></span>'
    '<span class="paw-icon d3"></span>'
    "</span>"
    '<span class="paw-loader-text">공고문을 확인하고 있어요</span>'
    "</div>"
)


def find_download_files(answer: str) -> list[str]:
    """답변 글에서 내려받을 수 있는 .docx 파일 경로를 찾아 돌려줍니다 (없으면 빈 목록)."""
    found = []
    for raw in DOWNLOAD_LINE.findall(answer):
        path = Path(raw.strip().replace("\\", "/"))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if path.is_file() and path.is_relative_to(DOWNLOAD_DIR):
            found.append(str(path))
    return found


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
    """연속된 '• 칸 | 칸 | 칸' 줄을 하나의 표(마크다운)로 바꿉니다."""
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
            raw = re.sub(r"(?:\s*\|)+\s*$", "", ln.lstrip("•").strip())
            grid.append([c.strip() for c in raw.split(" | ")])
        else:
            flush()
            out.append(ln)
    flush()
    return "\n".join(out)


def prettify_answer(answer: str, has_file_link: bool) -> str:
    """작성 도우미 답변을 화면에 보일 글로 다듬습니다."""
    text = answer
    if has_file_link and "작성 예시예요" in text:
        keep = [
            ln
            for ln in text.split("\n")
            if ln.endswith("작성 예시예요.")
            or ln.startswith("참고용 작성 예시예요")
            or ln.startswith("※ [예시]")
        ]
        keep = [("> " + ln) if ln.startswith("참고용") else ln for ln in keep]
        return "\n\n".join(keep).strip()
    if "작성 예시예요" in text:
        text = _bullets_to_tables(text)
        text = re.sub(r"^(참고용 작성 예시예요.*)$", r"> \1", text, flags=re.MULTILINE)
    text = _RE_COUNT_LINE.sub(
        "범례: [예시] 가상 값 · [직접 작성] 회사 정보를 직접 입력 · [문서] 공고문에서 가져온 값",
        text,
    )
    text = _RE_SRC_LINE.sub(lambda m: f"출처: {_pretty_source(m.group(1))}", text)
    text = _RE_DOC_LINE.sub("", text)
    if has_file_link:
        text = _RE_FILE_LINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decorate_answer(answer: str) -> str:
    """원본 chat 이 돌려준 답변 글을 화면에 보일 글로 바꿉니다 (정리 + 다운로드 링크 붙이기)."""
    files = find_download_files(answer)
    shown = tidy_tables(prettify_answer(answer, bool(files)))
    shown = shown.replace("**", "")
    if files:
        shown = re.sub(r"(?m)^\s*[-*]\s*파일:.*\n?", "", shown)
        links = "\n".join(
            f"[📄 {Path(f).name} 내려받기](/gradio_api/file={quote(Path(f).as_posix(), safe='/:')})"
            for f in files
        )
        return shown + "\n\n" + links
    return shown


def with_ui_extras(chat_fn):
    """원본 chat 함수를 감싸 화면 기능만 얹습니다. 백엔드 호출은 원본 chat_fn 이 그대로 합니다."""

    def chat(message, history, session_id: str):
        if not _message_text(message).strip():
            yield chat_fn(message, history, session_id)
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


# [수정 시작 ⑥] 표·글머리 기호 화면용 정리
_PUA_CH = "[\ue000-\uf8ff\U000f0000-\U000ffffd]"
_PUA = re.compile(r"[ \t]*" + _PUA_CH + r"[ \t]*")
_PUA_ONLY_CELL = re.compile(r"\|[ \t]*" + _PUA_CH + r"+[ \t]*(?=\|)")
_TABLE_TAG = re.compile(r"(?m)^[ \t]*\[표\][^\n]*\n+(?=[ \t]*\|)")


def tidy_tables(text: str) -> str:
    for _ in range(3):
        text = _PUA_ONLY_CELL.sub("| ", text)
    text = _TABLE_TAG.sub("", text)
    text = _PUA.sub(" • ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


# [수정 끝 ⑥]


# =============================================================================
# 꾹꾹이 랩스 랜딩 + gguk_ui 레이아웃 템플릿 & 클라이언트 인터랙션 JS
# =============================================================================
def _img_data_uri(filename: str) -> str:
    """assets 폴더의 이미지를 base64 data URI로 변환합니다 (프록시·도커·모바일 어디서나 안전)."""
    path = ASSETS_DIR / filename
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


APP_HEAD = """
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin />
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
/>
"""

# 라이트 모드 고정 + 랜딩/챗봇 화면 전환 + 히어로 탭 전환 + 고양이 말풍선 클릭 효과
APP_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url);
    return;
  }

  document.body.dataset.view = 'landing';

  const CAT_QUOTES = [
    '공고문 핵심 내용, 편하게 물어보세요!',
    '별지 서식 작성 예시도 바로 보여드려요!',
    '확인되지 않는 내용은 추측하지 않아요 🐾',
    '시작하기를 누르면 바로 질문해 볼 수 있어요!'
  ];
  let catQuoteIdx = -1;
  let bubbleHideTimer = null;

  const FORM_TABLE_HTML = `
    <div class="form-example-box" style="margin-top: 0;">
      <div class="form-example-header">
        <strong>[별지 제2호] 제안요청서 수용여부 참조표 작성 예시 (참고용)</strong>
        <span>2024년 벤처확인종합관리시스템 기능 고도화 용역사업 입찰공고</span>
      </div>
      <div class="form-table-scroll">
        <table class="form-example-table">
          <thead>
            <tr>
              <th scope="col">제안요청내용</th>
              <th scope="col">수용 여부</th>
              <th scope="col">제안내용</th>
              <th scope="col">제안서 쪽</th>
              <th scope="col">비 고</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td data-label="제안요청내용">SFR-001 사용자 인증 기능 <em class="tag-ex">[예시]</em></td>
              <td data-label="수용 여부">수용 <em class="tag-ex">[예시]</em></td>
              <td data-label="제안내용">공동인증서 기반 로그인 구현 <em class="tag-ex">[예시]</em></td>
              <td data-label="제안서 쪽">12쪽 <em class="tag-ex">[예시]</em></td>
              <td data-label="비 고">해당 없음 <em class="tag-ex">[예시]</em></td>
            </tr>
            <tr>
              <td data-label="제안요청내용">추가제안 <em class="tag-ex">[예시]</em></td>
              <td data-label="수용 여부">수용 <em class="tag-ex">[예시]</em></td>
              <td data-label="제안내용">실시간 모니터링 화면 추가 제공 <em class="tag-ex">[예시]</em></td>
              <td data-label="제안서 쪽">15쪽 <em class="tag-ex">[예시]</em></td>
              <td data-label="비 고">원문 주석 2항 참고 <em class="tag-ex">[예시]</em></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="form-example-note">
        ※ [예시] 값은 이해를 돕기 위한 가상 내용이며, [문서]는 공고문 값, [직접 작성]은 회사 정보를 넣어야 하는 칸입니다. 실제 제출은 공고문에 첨부된 원본 서식으로 하세요.
      </p>
    </div>
  `;

  const HERO_DEMOS = {
    formExample: FORM_TABLE_HTML,
    eligibility: `
      <div class="showcase-answer-label">꾹꾹이 AI · 공고문 기반 답변</div>
      <p>공고문에 명시된 <strong>지원 자격 요건 및 제외 대상</strong>을 정리해 드릴게요.</p>
      <ul class="demo-bullet-list">
        <li><strong>신청 대상:</strong> 공고일 기준 국내에 사업장을 둔 중소·벤처기업 및 중견기업 (법인·개인사업자 포함)</li>
        <li><strong>참여 조건:</strong> 공고 내 지정된 수행 요건 및 컨소시엄 구성 기준을 충족하는 기업</li>
        <li><strong>지원 제외:</strong> 접수 마감일 기준 국세·지방세 체납 중이거나 휴·폐업 상태인 기업</li>
        <li><strong>제한 사항:</strong> 최근 2년 이내 국가연구개발사업 참여제한 처분을 받은 기업</li>
      </ul>
    `,
    deadline: `
      <div class="showcase-answer-label">꾹꾹이 AI · 공고문 기반 답변</div>
      <p>공고문에 안내된 <strong>접수 마감 일정 및 필수 제출 서류</strong>입니다.</p>
      <ul class="demo-bullet-list">
        <li><strong>접수 마감:</strong> 2026년 12월 18일(금) 16:00까지 (전산 마감 이후 추가 제출 불가)</li>
        <li><strong>필수 서류 ①:</strong> 사업계획서 (공고문 첨부 지정 서식 활용)</li>
        <li><strong>필수 서류 ②:</strong> 사업자등록증 사본 및 최근 2개년 표준재무제표증명원</li>
        <li><strong>유의 사항:</strong> 서류 누락 시 평가 대상에서 제외될 수 있으니 제출 전 확인이 필요합니다.</li>
      </ul>
    `
  };

  function triggerHiddenBtn(id) {
    const btn = document.querySelector('#' + id + ' button, #' + id);
    if (btn) btn.click();
  }

  document.addEventListener('click', (e) => {
    const scrollBtn = e.target.closest('[data-scroll-target]');
    if (scrollBtn) {
      e.preventDefault();
      e.stopPropagation();
      const targetId = scrollBtn.dataset.scrollTarget;
      const targetEl = targetId ? document.getElementById(targetId) : null;
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      return;
    }

    const startBtn = e.target.closest('#hero-start-btn, #bottom-start-btn');
    if (startBtn) {
      document.body.dataset.view = 'chatbot';
      triggerHiddenBtn('btn-go-chat');
      window.scrollTo({ top: 0, behavior: 'instant' });
      return;
    }

    const homeBtn = e.target.closest('#brand-home-btn');
    if (homeBtn) {
      document.body.dataset.view = 'landing';
      triggerHiddenBtn('btn-go-landing');
      window.scrollTo({ top: 0, behavior: 'instant' });
      return;
    }

    const sampleChip = e.target.closest('.sample-q-chip');
    if (sampleChip) {
      const key = sampleChip.dataset.demo;
      const box = document.getElementById('hero-demo-answer');
      if (box && HERO_DEMOS[key]) {
        document.querySelectorAll('.sample-q-chip').forEach((c) => {
          const active = c === sampleChip;
          c.classList.toggle('is-active', active);
          c.setAttribute('aria-selected', String(active));
        });
        box.innerHTML = HERO_DEMOS[key];
      }
      return;
    }

    const catBtn = e.target.closest('#peeking-cat-btn');
    if (catBtn) {
      const bubble = document.getElementById('cat-speech-bubble');
      if (!bubble) return;
      catQuoteIdx = (catQuoteIdx + 1) % CAT_QUOTES.length;
      bubble.textContent = CAT_QUOTES[catQuoteIdx];
      bubble.classList.remove('is-visible');
      void bubble.offsetWidth;
      bubble.classList.add('is-visible');
      if (bubbleHideTimer) window.clearTimeout(bubbleHideTimer);
      bubbleHideTimer = window.setTimeout(() => {
        bubble.classList.remove('is-visible');
      }, 3200);
      return;
    }

    const stepCard = e.target.closest('.step-card');
    if (stepCard) {
      document.querySelectorAll('.step-card').forEach((c) => c.classList.remove('is-active'));
      stepCard.classList.add('is-active');
    }
  });
}
"""


def _build_header_html(logo_uri: str) -> str:
    return f"""
<div class="xd-top-bars" aria-hidden="true">
  <span class="bar-black"></span>
  <span class="bar-dark"></span>
  <span class="bar-light"></span>
  <span class="bar-blue"></span>
</div>
<header class="site-header" id="site-header">
  <div class="gguk-container header-inner">
    <button type="button" class="brand-lockup" id="brand-home-btn" aria-label="꾹꾹이 랩스 홈으로 이동">
      <img src="{logo_uri}" alt="꾹꾹이 랩스 로고" class="brand-logo-img" />
      <span class="brand-text">
        <strong class="brand-title">꾹꾹이 랩스</strong>
        <span class="brand-sub">Gguk-Gguk-i Labs</span>
      </span>
    </button>
    <div class="header-service-title" id="header-service-title">
      공고 작성 챗봇 서비스
    </div>
    <nav class="header-nav" id="landing-nav" aria-label="주요 섹션 이동">
      <button type="button" class="nav-link" data-scroll-target="features-section">선택 이유</button>
      <button type="button" class="nav-link" data-scroll-target="problems-section">해결하는 문제</button>
      <button type="button" class="nav-link" data-scroll-target="how-it-works-section">이용 방법</button>
    </nav>
  </div>
</header>
"""


def _build_landing_html(cat_peek_uri: str) -> str:
    form_table = """
<div class="form-example-box" style="margin-top: 0;">
  <div class="form-example-header">
    <strong>[별지 제2호] 제안요청서 수용여부 참조표 작성 예시 (참고용)</strong>
    <span>2024년 벤처확인종합관리시스템 기능 고도화 용역사업 입찰공고</span>
  </div>
  <div class="form-table-scroll">
    <table class="form-example-table">
      <thead>
        <tr>
          <th scope="col">제안요청내용</th>
          <th scope="col">수용 여부</th>
          <th scope="col">제안내용</th>
          <th scope="col">제안서 쪽</th>
          <th scope="col">비 고</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td data-label="제안요청내용">SFR-001 사용자 인증 기능 <em class="tag-ex">[예시]</em></td>
          <td data-label="수용 여부">수용 <em class="tag-ex">[예시]</em></td>
          <td data-label="제안내용">공동인증서 기반 로그인 구현 <em class="tag-ex">[예시]</em></td>
          <td data-label="제안서 쪽">12쪽 <em class="tag-ex">[예시]</em></td>
          <td data-label="비 고">해당 없음 <em class="tag-ex">[예시]</em></td>
        </tr>
        <tr>
          <td data-label="제안요청내용">추가제안 <em class="tag-ex">[예시]</em></td>
          <td data-label="수용 여부">수용 <em class="tag-ex">[예시]</em></td>
          <td data-label="제안내용">실시간 모니터링 화면 추가 제공 <em class="tag-ex">[예시]</em></td>
          <td data-label="제안서 쪽">15쪽 <em class="tag-ex">[예시]</em></td>
          <td data-label="비 고">원문 주석 2항 참고 <em class="tag-ex">[예시]</em></td>
        </tr>
      </tbody>
    </table>
  </div>
  <p class="form-example-note">
    ※ [예시] 값은 이해를 돕기 위한 가상 내용이며, [문서]는 공고문 값, [직접 작성]은 회사 정보를 넣어야 하는 칸입니다. 실제 제출은 공고문에 첨부된 원본 서식으로 하세요.
  </p>
</div>
"""

    return f"""
<main class="landing-main" aria-label="꾹꾹이 랩스 랜딩 페이지">
  <section class="hero-section">
    <div class="gguk-container hero-inner">
      <div class="hero-badge">정부지원사업 · 공공입찰 공고문 전용 AI</div>
      <h1 class="hero-title" id="landing-heading" tabindex="-1">
        공고문은 물어보고,<br />
        제안서는 근거로 쓰세요
      </h1>
      <p class="hero-subtitle">
        공고문을 검색해 물어보면 실제 공고문 내용을 바탕으로 핵심만 정리해 답해드려요.<br />
        복잡한 조건 확인부터 별지 서식 작성 예시까지 한 번에 살펴볼 수 있어요.
      </p>
      <div class="hero-cta-wrap">
        <button type="button" class="btn-hero-primary" id="hero-start-btn">시작하기</button>
        <button type="button" class="btn-hero-ghost" data-scroll-target="features-section">선택 이유 먼저 보기</button>
      </div>

      <div class="hero-showcase-wrapper">
        <button type="button" class="peeking-cat-trigger" id="peeking-cat-btn" aria-label="꾹꾹이 마스코트와 상호작용" title="꾹꾹이를 눌러보세요!">
          <span class="cat-speech-bubble" id="cat-speech-bubble" role="status" aria-live="polite"></span>
          <img src="{cat_peek_uri}" alt="카드 위로 빼꼼 고개를 내민 검은 고양이 꾹꾹이" class="peeking-cat-img" />
        </button>

        <div class="hero-showcase-card">
          <div class="showcase-body showcase-body-single">
            <div class="showcase-chat-col">
              <div class="showcase-prompts" role="tablist" aria-label="예시 질문 선택">
                <button type="button" class="sample-q-chip is-active" data-demo="formExample" role="tab" aria-selected="true">[별지 제2호] 작성 예시 보여줘</button>
                <button type="button" class="sample-q-chip" data-demo="eligibility" role="tab" aria-selected="false">지원 자격 요건이 어떻게 되나요?</button>
                <button type="button" class="sample-q-chip" data-demo="deadline" role="tab" aria-selected="false">제출 마감일과 필수 서류는?</button>
              </div>
              <div class="showcase-answer-box" id="hero-demo-answer" aria-live="polite">
                {form_table}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="features-section" id="features-section" aria-labelledby="features-heading">
    <div class="gguk-container">
      <div class="features-header">
        <span class="section-eyebrow">FEATURES</span>
        <h2 class="features-title" id="features-heading">
          꾹꾹이를 <span class="text-accent-blue">선택해야 하는 이유</span>
        </h2>
        <p class="features-subtitle">
          일반 AI 챗봇과 다른 점은 하나예요. 모든 답이 실제 공고문 원문에서 나와요.
        </p>
      </div>

      <div class="bento-grid">
        <article class="bento-card bento-blue bento-span-2">
          <div class="bento-top">
            <span class="bento-label">근거 답변</span>
            <h3 class="bento-title">답변에 대한 신뢰성이 높아져요</h3>
            <p class="bento-desc">
              실제 공고문 내용을 바탕으로 핵심만 정리해 답해요.<br />
              근거를 찾지 못하면 추측하지 않고 찾지 못했다고 답해요.
            </p>
          </div>
          <div class="bento-interactive bento-citation-demo">
            <div class="mini-qa-line">
              <span>Q. “최대 지원금 한도는 얼마인가요?”</span>
            </div>
            <div class="mini-source-quote is-highlighted">
              <span class="quote-tag">꾹꾹이 AI 답변 예시</span>
              <p>과제당 최대 <strong>2억 원 이내</strong>로 지원되며, 총사업비의 20% 이상 민간부담금 매칭이 필요해요.</p>
              <div class="mini-source-footer">참고 문서: 2026 지원사업 공고문</div>
            </div>
          </div>
        </article>

        <article class="bento-card bento-white bento-span-2">
          <div class="bento-top">
            <div class="bento-label-row">
              <span class="bento-label bento-label-dark">서식 작성 예시</span>
              <div class="legend-pills" aria-label="작성 태그 안내">
                <span class="legend-pill pill-example">[예시] 가상 내용</span>
                <span class="legend-pill pill-doc">[문서] 공고문 추출</span>
                <span class="legend-pill pill-user">[직접 작성] 회사 정보 입력</span>
              </div>
            </div>
            <h3 class="bento-title bento-title-dark">별지 서식 작성 예시를 표로 바로 보여드려요</h3>
            <p class="bento-desc bento-desc-dark">
              제안요청서 수용여부 참조표 등 서식의 칸 구성과 작성 요령을 참고용 예시 표로 정리해 드려요.
            </p>
          </div>
          {form_table}
        </article>
      </div>
    </div>
  </section>

  <section class="problems-section" id="problems-section" aria-labelledby="problems-heading">
    <div class="docs-strip">
      <div class="gguk-container docs-strip-inner">
        <strong class="docs-strip-label">이런 문서를 읽어요</strong>
        <ul class="docs-strip-list" aria-label="지원 문서 유형">
          <li>정부지원사업 공고</li>
          <li>공공입찰 제안요청서(RFP)</li>
          <li>R&amp;D 과제 공고</li>
          <li>운영·평가 지침</li>
        </ul>
        <div class="docs-format-badges" aria-label="지원 파일 형식">
          <span class="format-badge">HWP</span>
          <span class="format-badge">PDF</span>
        </div>
      </div>
    </div>

    <div class="gguk-container problems-inner">
      <h2 class="problems-title" id="problems-heading">
        제안서 작성보다 <span class="text-brand-blue">공고문 읽기</span>가<br />
        더 오래 걸리지 않나요?
      </h2>

      <div class="staggered-cards">
        <article class="problem-card align-left">
          <div class="problem-icon icon-blue" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="8" y1="13" x2="16" y2="13" />
              <line x1="8" y1="17" x2="13" y2="17" />
            </svg>
          </div>
          <div class="problem-content">
            <h3 class="problem-heading heading-blue"># 끝이 안 보이는 공고문</h3>
            <p class="problem-copy">
              공고문, 운영지침, 평가표까지 합치면 수십 쪽이에요.<br />
              자격 요건 하나 찾으려고 처음부터 다시 읽어요.
            </p>
          </div>
        </article>

        <article class="problem-card align-right">
          <div class="problem-icon icon-amber" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
              <line x1="3" y1="20" x2="21" y2="20" />
            </svg>
          </div>
          <div class="problem-content">
            <h3 class="problem-heading heading-amber"># 어디에 점수가 걸려 있는지 몰라요</h3>
            <p class="problem-copy">
              평가 항목과 배점은 공고문 뒤쪽 표에 숨어 있어요.<br />
              무엇을 길게 써야 하는지 감으로 정하게 돼요.
            </p>
          </div>
        </article>

        <article class="problem-card align-left">
          <div class="problem-icon icon-red" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div class="problem-content">
            <h3 class="problem-heading heading-red"># AI 답변, 믿어도 될까요?</h3>
            <p class="problem-copy">
              일반 AI 챗봇은 그럴듯하게 답하지만 공고문 어디에 있는 내용인지 알려주지 않아요.<br />
              결국 다시 확인해야 해요.
            </p>
          </div>
        </article>
      </div>
    </div>
  </section>

  <section class="how-section" id="how-it-works-section" aria-labelledby="how-heading">
    <div class="gguk-container">
      <div class="how-header">
        <span class="section-eyebrow">HOW IT WORKS</span>
        <h2 class="how-title" id="how-heading">
          고르고, 묻고, 쓰면 <span class="text-brand-blue">끝</span>
        </h2>
      </div>

      <div class="steps-grid">
        <article class="step-card is-active" data-step="1">
          <span class="step-num">01</span>
          <h3 class="step-title">공고문 선택하기</h3>
          <p class="step-desc">검색창이나 키워드로 원하는 공고를 고르면 바로 탐색 준비가 끝나요.</p>
        </article>
        <article class="step-card" data-step="2">
          <span class="step-num">02</span>
          <h3 class="step-title">궁금한 걸 물어보기</h3>
          <p class="step-desc">편하게 물어보면 공고문 내용을 바탕으로 핵심만 정리해 답해요.</p>
        </article>
        <article class="step-card" data-step="3">
          <span class="step-num">03</span>
          <h3 class="step-title">핵심 내용 정리하기</h3>
          <p class="step-desc">확인한 요건과 별지 서식 작성 예시를 보고서 작성에 바로 활용해요.</p>
        </article>
      </div>

      <div class="bottom-cta-banner">
        <div class="bottom-cta-text">
          <h3>지금 바로 공고문에 대해 물어보세요</h3>
          <p>키워드 검색부터 공고문 기반 답변, 별지 작성 예시까지 한 번에 확인할 수 있어요.</p>
        </div>
        <button type="button" class="btn-bottom-cta" id="bottom-start-btn">시작하기</button>
      </div>
    </div>
  </section>

  <footer class="site-footer">
    <div class="gguk-container footer-inner">
      <span>© 2026 꾹꾹이 랩스 (Gguk-Gguk-i Labs)</span>
      <span>정부지원사업 · 공공입찰 공고문 전용 RAG AI</span>
    </div>
  </footer>
</main>
"""


def _build_sidebar_html(cat_box_uri: str) -> str:
    return f"""
<aside class="gguk-sidebar-inner">
  <div class="gguk-sidebar-intro">
    <h1 class="gguk-sidebar-title">어떤 공고를<br />준비하고 계세요?</h1>
    <p class="gguk-sidebar-desc">
      업무에 도움을 드리기 위해 등록된 공고문을 근거로 답합니다.
    </p>
  </div>
  <div class="gguk-ask-group">
    <span class="gguk-ask-label">이렇게 물어보세요</span>
    <div class="gguk-ask-item">사업 예산은 얼마인가요?</div>
    <div class="gguk-ask-item">입찰 제출 서류를 알려 주세요</div>
    <div class="gguk-ask-item">기능 요구사항은 몇 개인가요?</div>
  </div>
  <div class="gguk-cat-box-wrap">
    <img src="{cat_box_uri}" alt="" aria-hidden="true" class="gguk-cat-box-img" />
  </div>
</aside>
"""


def build_app():
    """랜딩 화면 + 실제 백엔드 연결 챗봇 화면을 만듭니다."""
    logo_uri = _img_data_uri("logo.png")
    cat_peek_uri = _img_data_uri("cat-peek.png")
    cat_box_uri = _img_data_uri("cat_box.png")

    with gr.Blocks(title=APP_TITLE) as demo:
        # 전역 상단 헤더 (랜딩 ↔ 대화 화면 전환 시 로고 클릭으로 홈 복귀)
        gr.HTML(_build_header_html(logo_uri), elem_id="gguk-global-header")

        # 숨김 화면 전환 트리거 (JS에서 #hero-start-btn / #brand-home-btn 클릭 시 호출)
        btn_go_chat = gr.Button("go_chat", elem_id="btn-go-chat", visible=True)
        btn_go_landing = gr.Button("go_landing", elem_id="btn-go-landing", visible=True)

        # VIEW 1: 랜딩 페이지
        with gr.Column(visible=True, elem_id="landing-view") as landing_col:
            gr.HTML(_build_landing_html(cat_peek_uri), elem_id="landing-html")

        # VIEW 2: 실제 백엔드 연결 챗봇 화면 (gguk_ui 2단 레이아웃, 목업 대화 없음)
        with (
            gr.Column(visible=False, elem_id="chatbot-view") as chat_col,
            gr.Row(elem_id="gguk-chat-layout"),
        ):
            with gr.Column(scale=3, min_width=280, elem_id="gguk-sidebar-col"):
                gr.HTML(_build_sidebar_html(cat_box_uri), elem_id="gguk-sidebar-html")

            with gr.Column(scale=7, min_width=320, elem_id="gguk-chat-panel-col"):
                gr.HTML(
                    f'<img src="{cat_peek_uri}" alt="" aria-hidden="true" class="gguk-chat-peek-img" />',
                    elem_id="gguk-chat-peek-wrap",
                )
                with gr.Tabs(elem_id="gguk-tabs"):
                    for engine_id, label, _intro in ENGINES:
                        with gr.Tab(label):
                            # 탭마다 session_id 를 따로 둡니다 (원본 동작 100% 유지)
                            session_state = gr.State(value=lambda: str(uuid.uuid4()))

                            gr.ChatInterface(
                                with_ui_extras(
                                    make_chat_fn(engine_id)
                                ),  # [수정 ④] 원본 chat 을 화면 기능으로 감쌈
                                additional_inputs=[session_state],
                                chatbot=gr.Chatbot(
                                    show_label=False,
                                    height="calc(100dvh - 300px)",
                                    min_height=340,
                                    max_height=680,
                                ),
                                textbox=gr.MultimodalTextbox(
                                    placeholder="공고에 대해 물어보세요",
                                    show_label=False,
                                    container=False,
                                    sources=[],
                                ),
                                multimodal=True,
                                autofocus=False,
                                api_name=f"chat_{engine_id}",
                            )

        btn_go_chat.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[landing_col, chat_col],
            show_progress="hidden",
        )
        btn_go_landing.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[landing_col, chat_col],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    build_app().launch(
        theme=gr.themes.Base(),
        css_paths=[CSS_PATH, PROGRESS_CSS_PATH],  # [수정 ⑤]
        head=APP_HEAD,
        js=APP_JS,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "").lower() == "true",
        # [수정 ⑤] outputs/guides 안의 파일을 화면에서 내려받을 수 있게 허용합니다.
        allowed_paths=[str(DOWNLOAD_DIR)],
    )
