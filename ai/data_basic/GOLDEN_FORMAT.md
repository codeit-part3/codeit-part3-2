# 골든셋 형식 설명

기준 파일: `data/golden/rahui_golden_set_v2_2.json` (74문항, JSON 목록).
이 문서에는 문항 내용을 적지 않았습니다. 구조와 규칙만 설명합니다.

## 문항 구성

| level | 문항 수 | 뜻 |
| --- | --- | --- |
| lookup | 34 | 한 문서에서 값을 찾는 질문 |
| composite | 11 | 여러 문서를 종합하는 질문 |
| no_answer | 13 | 문서에 답이 없어서 "모른다"고 해야 정답인 질문 |
| interpret | 6 | 해석이 필요한 질문 |
| followup | 6 | 앞선 대화(history)가 있는 후속 질문 |
| fuzzy_filter | 3 | 기관명·사업명을 부정확하게 입력한 질문 |
| calc | 1 | 계산이 필요한 질문 |

- 평가 대상은 답이 있는 문항 61개 중 q27, q29(그림·메타데이터에만 답이 있어 본문 근거를 못 찾음)를 제외한 59개와,
  답 없음 13개입니다. 근거 검색 지표(hit_rate, context_hit)는 `no_answer`를 뺍니다.
- 그림 의존 8문항, PDF 표 7문항이 포함되어 있어서 변환이 약한 영역도 점수에 반영됩니다.

## 필드

| 필드 | 설명 |
| --- | --- |
| id | `q01`~`q74` |
| level, category | 위 level, 그리고 세부 유형(요구사항 조회, 평가기준, 제출서류·준비물 등) |
| question, gold_answer | 질문과 정답 |
| keywords | 답변에 들어 있어야 할 핵심어 (규칙 채점에 사용) |
| answer_values | 답변에 있어야 할 숫자 (금액, 개수 등) |
| evidence_docs | 근거 문서 번호. `data_list.csv`의 행 순서(**1부터**) |
| evidence_files | 근거 문서의 원본 파일명 |
| evidence_quote | 근거 문장 일부. 말뭉치 본문에서 앞 30자로 찾음 (RFP 본문 조각이라 NDA 대상) |
| evidence_location | 근거가 있는 위치 설명 |
| needs_calc, needs_image | 계산 필요 / 그림 속 내용 필요 |
| difficulty, verified, note | 난이도와 검증 방법·메모 |
| history | followup 문항만. user 1턴 + assistant 1턴 |

## 지킬 점

- `evidence_quote`는 80자 이하로 유지합니다. `python -m ai.data_basic.golden_eval`이 80자를 넘는 문항을 알려 줍니다.
  현재 q71이 한 건 걸립니다.
- 근거 문서 번호는 **병합 전 문서 번호** 기준입니다. 본문이 같은 문서 두 쌍(16·54번, 22·47번)은
  말뭉치에서 16, 22번만 남기고 `_meta.csv`의 `중복대표` 열에 대표 번호를 적어 두었습니다.
  이 네 문서를 근거로 하는 문항은 현재 없습니다.

## 다른 형식에서 옮길 때

팀원이 만든 골든셋(`.jsonl`)은 다음 규칙으로 변환합니다.

| jsonl | 이 형식 |
| --- | --- |
| `doc_NNN` (0부터) | `evidence_docs` 의 `NNN + 1` |
| `evidence_text` (`doc_NNN::문장`) | `evidence_quote` (문장 부분) |
| `evidence_chunk_ids` | 사용하지 않음 (청킹마다 달라짐) |
| `q_type` 가 금액·마감일·발주기관·집계·비교 | 메타데이터로 푸는 문항이라 `evidence_quote`가 비어 있음 |

변환한 뒤에는 `evidence_quote`의 앞 30자가 말뭉치 본문에서 찾아지는지 먼저 확인하고,
못 찾는 문항은 점수에 넣지 않습니다.
