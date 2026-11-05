## 시작하기

```bash
git clone https://github.com/temp-team-test/temp-team-test.git
cd temp-team-test
uv sync                  # uv.lock 기준으로 팀원 전원이 같은 환경이 됩니다
uvx prek install         # 커밋 전 자동 검사 설치 (프로젝트마다 한 번)
cp .env.example .env     # 값을 채웁니다. .env는 절대 커밋하지 않습니다
```

- `cp`는 Git Bash·MSYS2에서 쓰는 명령입니다. Windows PowerShell에서는 `Copy-Item .env.example .env`를 쓰세요.
- 패키지는 `uv add <패키지>`로 추가하고 `uv.lock`을 **반드시 함께 커밋**합니다. (`pip` 사용 금지)
- `.venv`와 `uv.lock`은 직접 수정하지 않습니다. `.venv`는 올리지 않고, `uv.lock`은 올립니다.
- 처음 한 번 `git lg`를 등록해 두면 커밋 기록을 보기 편합니다.

```bash
git config --global alias.lg "log --color --graph --pretty=format:'%C(bold blue)%h%Creset -%C(bold green)%ad%Creset%C(red)%d%Creset %s %C(bold yellow)[%an]%Creset' --abbrev-commit --date=short --all"
```

## 협업 방식

1. **이슈를 만든다.** 템플릿의 "작업할 내용"을 체크리스트로 적습니다.
2. **브랜치를 만든다.** `main`에서 새로 따고, `main`에는 직접 푸시하지 않습니다.
3. **작업하고 커밋한다.** 커밋하면 pre-commit이 자동으로 파일을 정리합니다. 커밋이 멈추면 정리된 파일을 다시 `git add` 하고 `git commit` 합니다.
4. **PR을 올린다.** 템플릿 체크리스트를 채우고, `pre-commit` 검사가 통과하면 머지합니다. 중요한 변경은 팀원 1명에게 확인을 요청한 뒤 머지합니다.

- 작업 시작 전 `git pull`, 완료 즉시 `git push`.
- 같은 파일을 수정해야 하면 팀 채팅에 먼저 알립니다.

## 컨벤션

타입은 `feat` / `fix` / `chore` / `docs` / `refactor` / `eval` 6개입니다.

| 대상 | 형식 | 예시 |
|---|---|---|
| Branch | `타입/#이슈번호-작업내용` | `feat/#12-bm25-search`, `chore/#3-env-setup` |
| Commit | `타입: 작업 내용` | `feat: BM25 검색 추가` |
| Issue / PR | `[타입] 작업 내용` | `[Feat] BM25 검색 추가` |

## 코드 스타일 (자동)

`.pre-commit-config.yaml`에 정리 규칙이 있고, 커밋할 때와 PR을 올릴 때 같은 검사가 돌아갑니다.
직접 돌려 보려면 `uvx prek run --all-files`.
