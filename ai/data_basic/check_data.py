"""받은 말뭉치·골든셋이 팀 기준과 같은지 확인합니다. (본문은 출력하지 않습니다)

실행 (프로젝트 루트):
  uv run python -m ai.data_basic.check_data                 # 확인
  uv run python -m ai.data_basic.check_data --write-manifest  # 기준값 만들기 (파일 배포하는 사람만)

확인하는 것
  1) 말뭉치: 파일 수, 전체 글자 수, 내용 지문(SHA256), 본문이 같은 중복 문서
  2) 골든셋: 문항 수, 내용 지문(SHA256), 형식 점검(golden_eval.check)
  3) manifest.json 의 기준값과 비교 -> 일치/불일치
출력은 개수와 지문뿐이라 NDA 걱정 없이 채팅에 붙여도 됩니다.
"""

import argparse  # 명령줄 옵션(--write-manifest)을 읽는 기능을 가져옵니다.
import hashlib  # 파일 내용의 SHA256/SHA1 지문을 계산하는 기능을 가져옵니다.
import json  # 골든셋과 manifest JSON 파일을 읽고 쓰는 기능을 가져옵니다.
import re  # 텍스트에서 비교에 필요 없는 문자를 정규식으로 제거합니다.
from collections import Counter  # 골든셋 level별 문항 수를 셀 도구를 가져옵니다.

from ai.data_basic.golden_eval import (  # 골든셋 구조 점검과 JSON 읽기 함수를 가져옵니다.
    check,
    load_golden,
)
from ai.data_basic.paths import (  # 공통 데이터 경로와 이름을 가져옵니다.
    CORPUS_DIR,
    CORPUS_NAME,
    GOLDEN_NAME,
    GOLDEN_PATH,
    MANIFEST_PATH,
)


def norm(text: str) -> str:  # 중복 본문 비교를 위해 한글·영문·숫자만 남깁니다.
    """한글·영문·숫자만 남김 (중복 판정용, ai/rahui 전처리와 같은 규칙)."""
    return re.sub(
        r"[^0-9A-Za-z가-힣]", "", text
    )  # 한글·영문·숫자가 아닌 문자를 모두 없애 정규화합니다.


def corpus_summary() -> dict:  # 말뭉치의 개수·크기·지문·중복 요약을 계산합니다.
    files = sorted(
        CORPUS_DIR.glob("*.txt")
    )  # 말뭉치 폴더의 텍스트 파일을 이름순으로 가져옵니다.
    total = (
        hashlib.sha256()
    )  # 파일 목록과 각 파일의 지문을 합칠 전체 지문 계산기를 만듭니다.
    chars = 0  # 말뭉치 전체의 문자 수를 누적할 변수를 준비합니다.
    body_groups: dict[
        str, list[int]
    ] = {}  # 정규화한 본문 지문별로 파일 순번을 모아 둡니다.
    for n, p in enumerate(
        files, 1
    ):  # 각 파일을 1부터 시작하는 순번과 함께 차례로 처리합니다.
        raw = p.read_bytes()  # 파일 원본 바이트를 읽어 지문을 계산할 때 사용합니다.
        total.update(
            p.name.encode("utf-8") + hashlib.sha256(raw).digest()
        )  # 파일명과 내용 지문을 합쳐 전체 지문에 반영합니다.
        text = raw.decode(
            "utf-8", errors="replace"
        )  # UTF-8로 텍스트를 읽고 잘못된 바이트는 대체 문자로 바꿉니다.
        chars += len(text)  # 현재 파일의 문자 수를 전체 개수에 더합니다.
        h = hashlib.sha1(
            norm(text).encode(), usedforsecurity=False
        ).hexdigest()  # 문장부호 등을 뺀 본문 지문을 계산합니다.
        body_groups.setdefault(h, []).append(
            n
        )  # 같은 본문 지문을 가진 파일의 순번을 같은 목록에 넣습니다.
    dups = [
        g for g in body_groups.values() if len(g) > 1
    ]  # 같은 본문 지문이 두 개 이상인 중복 그룹만 고릅니다.
    return {  # 팀 기준 비교에 필요한 말뭉치 요약값을 돌려줍니다.
        "corpus_name": CORPUS_NAME,  # 말뭉치 폴더 이름을 기록합니다.
        "corpus_files": len(files),  # 발견한 텍스트 파일의 개수를 기록합니다.
        "corpus_chars": chars,  # 말뭉치의 전체 문자 수를 기록합니다.
        "corpus_sha256": total.hexdigest(),  # 파일명과 내용으로 만든 말뭉치 전체 지문을 기록합니다.
        "has_meta_csv": (
            CORPUS_DIR / "_meta.csv"
        ).exists(),  # 메타데이터 CSV 파일이 있는지 기록합니다.
        "duplicate_groups": len(dups),  # 본문이 같은 파일들의 그룹 수를 기록합니다.
    }


def golden_summary() -> dict:  # 골든셋의 문항 수·유형·지문·형식 문제 수를 계산합니다.
    items = load_golden(
        GOLDEN_PATH
    )  # 기준 경로의 골든셋 JSON을 문항 객체 목록으로 읽습니다.
    return {  # 팀 기준 비교에 필요한 골든셋 요약값을 돌려줍니다.
        "golden_name": GOLDEN_NAME,  # 골든셋 파일 이름을 기록합니다.
        "golden_items": len(items),  # 골든셋 문항 수를 기록합니다.
        "golden_levels": dict(
            Counter(i.level for i in items)
        ),  # level별 문항 수를 세어 일반 딕셔너리로 기록합니다.
        "golden_sha256": hashlib.sha256(
            GOLDEN_PATH.read_bytes()
        ).hexdigest(),  # JSON 파일 전체의 SHA256 지문을 기록합니다.
        "golden_problems": len(
            check(items)
        ),  # 골든셋 형식 점검에서 발견한 문제 수를 기록합니다.
    }


def main() -> None:  # 현재 데이터 요약을 출력하고 필요하면 기준 manifest와 비교합니다.
    ap = argparse.ArgumentParser(
        description="말뭉치·골든셋 확인"
    )  # 명령줄 프로그램의 설명과 옵션 파서를 만듭니다.
    ap.add_argument(
        "--write-manifest", action="store_true", help="기준값(manifest.json) 만들기"
    )  # 기준값 저장 옵션을 등록합니다.
    args = ap.parse_args()  # 사용자가 입력한 명령줄 옵션을 읽습니다.

    if not CORPUS_DIR.is_dir():  # 말뭉치 폴더가 실제로 있는지 확인합니다.
        raise SystemExit(
            f"말뭉치 폴더가 없습니다: {CORPUS_DIR}\nREADME.md 의 '데이터 받기'를 보세요."
        )  # 없으면 안내와 함께 종료합니다.
    if not GOLDEN_PATH.is_file():  # 골든셋 JSON 파일이 실제로 있는지 확인합니다.
        raise SystemExit(
            f"골든셋 파일이 없습니다: {GOLDEN_PATH}\nREADME.md 의 '데이터 받기'를 보세요."
        )  # 없으면 안내와 함께 종료합니다.

    now = {
        **corpus_summary(),
        **golden_summary(),
    }  # 말뭉치와 골든셋의 현재 요약 정보를 하나로 합칩니다.
    for k, v in now.items():  # 계산된 각 항목의 이름과 값을 순서대로 확인합니다.
        print(f"  {k}: {v}")  # 현재 데이터의 항목별 요약값을 터미널에 출력합니다.

    if args.write_manifest:  # 기준값 생성 옵션이 켜져 있는지 확인합니다.
        MANIFEST_PATH.write_text(
            json.dumps(now, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )  # 요약값을 읽기 쉬운 JSON으로 저장합니다.
        print(
            f"기준값을 저장했습니다: {MANIFEST_PATH}"
        )  # 저장된 파일 경로를 안내합니다.
        return  # 기준값 생성이 끝났으므로 비교 작업은 생략하고 종료합니다.

    if (
        not MANIFEST_PATH.exists()
    ):  # 비교에 사용할 기준 manifest 파일이 있는지 확인합니다.
        print(
            "manifest.json 이 없어 기준값 비교는 건너뜁니다."
        )  # 기준값이 없으면 비교할 수 없음을 알립니다.
        return  # 비교 대상이 없으므로 정상 종료합니다.
    base = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )  # 기준 manifest JSON을 읽습니다.
    bad = [
        k for k, v in base.items() if now.get(k) != v
    ]  # 기준과 현재값이 다른 항목의 이름을 모읍니다.
    if bad:  # 서로 다른 항목이 하나 이상 있는지 확인합니다.
        print(
            "\n[불일치]",
            ", ".join(bad),
            "-> 팀 비밀방에서 받은 파일이 맞는지 확인하세요.",
        )  # 불일치 항목과 확인 방법을 안내합니다.
    else:  # 모든 비교 항목이 기준값과 같을 때 실행합니다.
        print(
            "\n[일치] 팀 기준 말뭉치·골든셋과 같습니다."
        )  # 팀 기준 데이터와 같음을 알립니다.


if (
    __name__ == "__main__"
):  # 이 파일을 모듈로 가져온 것이 아니라 직접 실행했을 때만 아래 코드를 수행합니다.
    main()  # 명령줄 점검 절차를 시작합니다.
