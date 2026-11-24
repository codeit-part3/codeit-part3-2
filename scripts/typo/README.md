# 오타 강건성 실험 (scripts/typo)

모든 명령은 **프로젝트 루트**에서 실행합니다. 오타 골든셋·교정본은 `data/golden/typo/`, 이 폴더 스크립트의 결과는 `outputs/typo/`에 저장됩니다.
팀 파일(`retrieval.py`, `run_eval.py`)은 `outputs/`에 저장하므로, 오타 실험분은 실행 후 `outputs/typo/`로 옮겨 둡니다 (`typo_summary.py`는 두 폴더를 모두 읽습니다).

| 파일 | 하는 일 |
|---|---|
| hangul_typo.py | 한글 자모 분해·조합, 오타 만들기 부품 (다른 파일이 불러 씀) |
| typo_golden.py | 원본 골든셋 → 오타 골든셋(light/medium/heavy) 만들기 |
| query_fix.py | 기관명·사업명 사전으로 질문 오타 교정 (`--save` 로 `_fixed.json` 저장) |
| typo_retrieval_rerank.py | 리랭커 엔진(engines/rerank_rag) 검색 강건성 측정 (토큰 0). `--typo-fix on`(팀 교정) / `query_fix`(내 교정) |
| run_eval_typo.py | 팀 `scripts/rerank_rag/run_eval.py` 를 오타 교정을 켠 채로 실행 (답변 평가) |
| typo_report.py | A(원본)/B(오타)/C(교정) 결과로 회복률 판정 |
| typo_summary.py | outputs/, outputs/typo/ 의 베이스라인 검색 결과를 모아 한 번에 요약 |

베이스라인 엔진 검색 강건성은 루트의 `retrieval.py --query-fix dict` 로 잽니다 (query_fix.py 를 이 폴더에서 불러옴).

**엔진 연동 (이슈 #38).** 교정 코드는 리랭커 엔진에 들어 있고 기본으로 켜져 있습니다.
- `engines/rerank_rag/query_fix.py` : 엔진용 교정기 (다른 파일 의존 없음, 서비스 이미지에 포함)
- `engines/rerank_rag/config.py` 의 `TYPO_FIX` : `"query_fix"`(기본, 사전 교정) / `True`(기존 팀 교정) / `False`(끔)
- 그래서 팀 `run_eval.py` 나 챗봇을 평소처럼 돌리면 검색 직전에 교정이 같이 돕니다.

이 폴더의 `query_fix.py` 는 실험용(`--save` 로 `_fixed.json` 만들기)이고, 교정 결과는 엔진용과 같습니다.
`--typo-fix` 옵션은 A/B/C 비교처럼 `config.py` 를 고치지 않고 조건을 바꿔 돌릴 때만 씁니다 (안 주면 config 값 그대로).

```powershell
uv run python scripts/typo/query_fix.py data/golden/typo/rahui_golden_set_v2_2_typo_medium.json --save
uv run python scripts/typo/typo_retrieval_rerank.py --limit 5
uv run python scripts/typo/typo_retrieval_rerank.py --golden data/golden/typo/rahui_golden_set_v2_2_typo_medium.json --typo-fix query_fix
uv run python scripts/typo/run_eval_typo.py --typo-fix query_fix --pkg --variant rerank --pool 30 --chunker struct --rewrite --cutoff 0.5 --golden data/golden/typo/rahui_golden_set_v2_2_typo_medium.json --limit 5 --no-judge --no-sheet
uv run python scripts/typo/typo_report.py --a outputs/typo/A.csv --b outputs/typo/B.csv --c outputs/typo/C.csv --metrics hit_rate mrr context_hit cutoff_refuse
```

주의: 베이스라인과 비교할 때는 이미 만든 오타 골든셋을 그대로 쓰고, typo_golden.py 를 다시 돌리지 않습니다 (같은 문제지로 비교).
