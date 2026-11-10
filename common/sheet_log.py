"""실험 결과를 구글 시트에 한 줄 자동 추가합니다.

사용 예 (실험 코드 맨 끝에서):

    from sheet_log import log_experiment

    log_experiment(
        name="chunk512_ov50",
        memo="청킹 512 / 겹침 50",
        metrics={"hit_rate": 0.81, "mrr": 0.64},
        params={"chunk_size": 512, "chunk_overlap": 50, "top_k": 5},
    )

필요한 환경 변수 (.env 에 넣고, .env 는 커밋하지 않습니다):
    SHEET_WEBHOOK_URL   Apps Script 웹 앱 주소 (.../exec)
    SHEET_SECRET        Apps Script 스크립트 속성 SECRET 과 같은 값
    EXPERIMENT_AUTHOR   내 이름 (선택)

시트 첫 줄(헤더) 이름과 같은 키만 해당 칸에 들어갑니다.
시트에 없는 키는 무시되고, 코드에서 안 보낸 칸은 비워 둡니다.
(예: 결론 / 다음 실험 칸은 나중에 손으로 적습니다.)

기록에 실패해도 예외를 던지지 않습니다. 실험 결과는 이미 나왔는데
시트 때문에 코드가 멈추는 일이 없도록 경고만 출력합니다.
"""

import os
from datetime import datetime

import httpx

try:  # python-dotenv 가 설치돼 있으면 .env 를 읽습니다. 없으면 환경 변수만 씁니다.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

TIMEOUT = httpx.Timeout(30.0, connect=5.0)


def log_experiment(
    name: str,
    memo: str = "",
    metrics: dict | None = None,
    params: dict | None = None,
    author: str | None = None,
    tab: str | None = None,
    extra: dict | None = None,
) -> bool:
    """시트에 한 줄을 추가합니다. 성공하면 True, 아니면 False."""
    url = os.getenv("SHEET_WEBHOOK_URL", "").strip()
    secret = os.getenv("SHEET_SECRET", "").strip()

    row = {
        "time": datetime.now().strftime("%m-%d %H:%M"),
        "author": author or os.getenv("EXPERIMENT_AUTHOR", ""),
        "name": name,
        "memo": memo,
        **(metrics or {}),
        "experiment": params or {},  # 이번에 바꾼 파라미터 (시트에는 문자열로 들어감)
        **(extra or {}),
    }

    if not url:
        print(f"[시트 기록 건너뜀] SHEET_WEBHOOK_URL 이 비어 있습니다. 내용: {row}")
        return False

    body = {"secret": secret, "row": row}
    if tab:
        body["tab"] = tab

    try:
        # Apps Script 웹 앱은 302 리다이렉트로 응답하므로 follow_redirects 가 꼭 필요합니다.
        res = httpx.post(url, json=body, timeout=TIMEOUT, follow_redirects=True)
        res.raise_for_status()
        result = res.json()
    except Exception as e:
        print(f"[시트 기록 실패] {type(e).__name__}: {e}")
        return False

    if not result.get("ok"):
        print(f"[시트 기록 실패] 서버 응답: {result}")
        return False

    print(
        f"[시트 기록 완료] {name} ({result.get('tab', '')} {result.get('row', '')}행)"
    )
    return True
