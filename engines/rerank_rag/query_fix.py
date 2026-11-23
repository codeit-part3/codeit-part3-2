"""질문 오타 보정: 기관명·사업명 사전 교정 (rerank_rag 엔진용, 이슈 #38 오타 강건성)

baseline.py 의 fix_typos() 가 TYPO_FIX = "query_fix" 일 때 검색 직전에 부른다.
Docker 이미지에는 engines/ 만 들어가므로 엔진이 쓰는 교정 코드는 엔진 폴더 안에 둔다.
교정 규칙은 실험에 쓴 scripts/typo/query_fix.py 의 QueryFixer 와 같다 (같은 질문 -> 같은 결과).

두 개의 단어장을 만든다
    names (정답 사전) : data_list.csv 의 발주 기관·사업명에 나오는 한글 단어. 오타를 '이 단어로' 고친다.
    known (아는 단어) : 말뭉치 본문에 2번 이상 나온 한글 단어 + names. 여기 있는 단어는 '맞는 단어'로 보고 건드리지 않는다.

질문 하나를 고치는 순서 (fix)
    1. 띄어쓰기 오타 되돌리기 : '차세 대' 처럼 붙이면 정답 사전 단어가 되는 두 조각을 붙임
    2. 글자 오타 고치기       : known 에 없는 단어 -> 정답 사전에서 자모가 가장 비슷한 단어로 바꿈
    오타가 없는 질문은 모든 단어가 known 에 있으므로 바뀌지 않는다.
"""

import re  # 문장에서 한글 단어만 골라내기 위해 (정규식)
import unicodedata  # 단어를 자모로 풀어 쓰기 위해 (NFD)
from collections import Counter  # 단어가 몇 번 나왔는지 세기 위해
from difflib import SequenceMatcher  # 두 문자열이 얼마나 비슷한지 0~1 점수로 재는 파이썬 기본 도구
from pathlib import Path

import pandas as pd

# =====================================================================
# 설정
# =====================================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # 프로젝트 루트 (engines/rerank_rag/ 에서 두 단계 위)
DATA_LIST = BASE_DIR / "data" / "data_list.csv"  # 문서 목록 (기관명, 사업명, 본문 앞부분)
DEFAULT_CORPUS = BASE_DIR / "data" / "clean_mixed_ocr_v2"  # 말뭉치 txt 폴더 (config.py 의 CORPUS)

HANGUL_WORD = re.compile(r"[가-힣]+")  # 한글이 연속된 덩어리 = '단어'로 본다 (숫자·영문·괄호에서 끊김)

# 단어 끝에 붙는 조사. 긴 것부터 확인해야 '에서는'을 '는'으로 잘못 떼지 않는다.
JOSA = ("에서는", "으로", "에서", "의", "은", "는", "이", "가", "을", "를", "과", "와", "로", "에", "도")

SIMILARITY = 0.8  # 자모 유사도가 이 값 이상이어야 고친다 (1.0 = 똑같음)
SHORT_WORD_SIMILARITY = 0.9  # 두 글자 이하 단어는 더 엄격하게 ('국민' -> '구미' 같은 잘못된 교정 방지)
MAX_LEN_DIFF = 2  # 글자 수가 2개 넘게 차이 나는 단어는 비교하지 않음 (속도 + 엉뚱한 교정 방지)
MIN_KNOWN_FREQ = 2  # 말뭉치에 2번 이상 나온 단어만 '아는 단어' (1번 = OCR 잡음이나 오타일 수 있음)


def to_jamo(text):
    """'협회' -> 'ㅎㅕㅂㅎㅚ' 처럼 자모 단위로 풀어 쓴 문자열 (scripts/typo/hangul_typo.py 의 to_jamo 와 같음)"""
    return unicodedata.normalize("NFD", text)


def strip_josa(word):
    """단어 끝 조사를 뗀 후보들. '협회의' -> [('협회', '의')]. 조사가 없으면 빈 목록"""
    return [
        (word[: -len(j)], j)  # (조사를 뗀 앞부분, 뗀 조사)
        for j in JOSA
        if word.endswith(j) and len(word) - len(j) >= 2  # 떼고 남은 부분이 두 글자 이상일 때만
    ]


class QueryFixer:
    """질문 오타 보정기. 단어장은 만들 때 한 번만 만들고, fix() 는 여러 번 부른다."""

    def __init__(self, data_list=DATA_LIST, corpus_dir=DEFAULT_CORPUS):
        df = pd.read_csv(data_list)
        self.names = self._build_names(df)  # 정답 사전
        self.known = self._build_known(df, corpus_dir) | self.names  # 아는 단어 (정답 사전도 포함)
        # 정답 사전 단어의 자모 풀이를 미리 계산 (매번 풀면 느림).
        # sorted: 파이썬 set 은 실행할 때마다 순서가 달라질 수 있다. 점수가 같은 후보가 여럿일 때
        #         먼저 본 쪽이 뽑히므로, 순서를 고정해야 실행마다 같은 보정 결과가 나온다.
        self._jamo = {w: to_jamo(w) for w in sorted(self.names)}
        self._cache = {}  # 같은 단어를 다시 고칠 때 계산을 건너뛰기 위한 저장소

    # ------------------------------------------------------------ 단어장 만들기
    @staticmethod
    def _build_names(df):
        """발주 기관 + 사업명에 나오는 두 글자 이상 한글 단어 (정답 사전)"""
        text = " ".join(df["발주 기관"].dropna().astype(str).tolist() + df["사업명"].dropna().astype(str).tolist())
        return {w for w in HANGUL_WORD.findall(text) if len(w) >= 2}  # 한 글자 단어는 너무 흔해서 제외

    def _build_known(self, df, corpus_dir):
        """말뭉치에 MIN_KNOWN_FREQ 번 이상 나온 한글 단어 (아는 단어)

        data_list.csv 의 텍스트 열은 본문 앞부분(평균 3,800자)만 있어서 단어가 부족하다.
        그래서 말뭉치 txt 폴더가 있으면 그것도 읽는다. 없으면 csv 만으로 만든다.
        """
        freq = Counter()  # 단어 -> 나온 횟수
        for col in ("텍스트", "사업 요약", "사업명", "발주 기관"):
            if col in df.columns:
                for text in df[col].dropna().astype(str):
                    freq.update(HANGUL_WORD.findall(text))

        corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.used_corpus = bool(corpus_dir and corpus_dir.is_dir())  # 말뭉치를 읽었는지 기록 (확인용)
        if self.used_corpus:
            for path in corpus_dir.glob("*.txt"):
                freq.update(HANGUL_WORD.findall(path.read_text(encoding="utf-8", errors="replace")))
        else:
            print(f"[query_fix] 말뭉치 폴더가 없어 data_list.csv 만으로 사전을 만듭니다: {corpus_dir}")

        return {w for w, count in freq.items() if count >= MIN_KNOWN_FREQ}

    # ------------------------------------------------------------ 단어 하나 판단
    def _is_known(self, word):
        """맞는 단어인지. 조사가 붙어 있으면 떼고도 확인. 한 글자 조각은 '모름'으로 본다.

        한 글자를 모름으로 보는 이유: '차세 대'의 '대'처럼 띄어쓰기 오타로 떨어져 나온 조각일 가능성이 커서,
        붙여 보는 대상에 넣어야 하기 때문이다.
        """
        if len(word) < 2:
            return False
        return word in self.known or any(stem in self.known for stem, _ in strip_josa(word) if len(stem) >= 2)

    def _closest_name(self, word):
        """word 와 자모가 가장 비슷한 정답 사전 단어 -> (단어, 유사도). 기준 미달이면 (None, 유사도)"""
        if word in self._cache:  # 이미 계산한 단어면 바로 돌려줌
            return self._cache[word]

        word_jamo = to_jamo(word)
        best, best_score = None, 0.0
        for cand, cand_jamo in self._jamo.items():
            if abs(len(cand) - len(word)) > MAX_LEN_DIFF:  # 길이가 너무 다르면 건너뜀
                continue
            sm = SequenceMatcher(None, word_jamo, cand_jamo, autojunk=False)
            # real_quick_ratio, quick_ratio = 정확한 ratio 의 '상한'을 아주 빨리 계산한다.
            # 상한조차 기준 미만이면 진짜 점수도 기준 미만이므로 느린 ratio() 를 부르지 않고 넘어간다.
            if sm.real_quick_ratio() < SIMILARITY or sm.quick_ratio() < SIMILARITY:
                continue
            score = sm.ratio()  # 진짜 유사도 (0~1)
            # 점수가 더 높으면 교체. 점수가 같으면 더 긴 단어를 고름 (더 구체적인 이름일 가능성이 큼)
            if score > best_score or (score == best_score and best and len(cand) > len(best)):
                best, best_score = cand, score

        need = SHORT_WORD_SIMILARITY if len(word) <= 2 else SIMILARITY
        result = (best, best_score) if best_score >= need else (None, best_score)
        self._cache[word] = result
        return result

    def _fix_word(self, word):
        """단어 하나: 맞는 단어면 그대로, 오타로 보이면 고친 단어를 돌려준다."""
        if len(word) < 2 or word in self.known:  # 한 글자이거나 아는 단어면 손대지 않음
            return word
        stems = strip_josa(word)
        if any(stem in self.known for stem, _ in stems):  # 조사만 붙은 맞는 단어 ('사업비는' = '사업비' + '는')
            return word

        # 조사를 안 뗀 경우 + 뗀 경우를 모두 사전과 비교해서 가장 비슷한 것을 고른다
        best_word, best_score, best_josa = None, 0.0, ""
        for stem, josa in [(word, "")] + stems:
            cand, score = self._closest_name(stem)
            if cand and score > best_score:
                best_word, best_score, best_josa = cand, score, josa
        return best_word + best_josa if best_word else word  # 고칠 후보가 없으면 원래 단어 그대로

    # ------------------------------------------------------------ 문장 단위
    def _join_split_words(self, text):
        """띄어쓰기 오타 되돌리기: '차세 대' -> '차세대'

        두 조각을 붙였을 때 정답 사전 단어(조사 포함 가능)가 되고,
        두 조각 중 하나라도 '모르는 단어'일 때만 붙인다.
        (둘 다 아는 단어면 원래 띄어 쓰는 게 맞는 경우라서 건드리지 않음. 예: '통합운영관리 시스템')
        """
        def maybe_join(m):  # 정규식이 찾은 '한글 + 공백 + 한글' 하나에 대해 붙일지 정함
            left, right = m.group(1), m.group(2)
            joined = left + right
            is_name = joined in self.names or any(stem in self.names for stem, _ in strip_josa(joined))
            if is_name and not (self._is_known(left) and self._is_known(right)):
                return joined  # 붙인 결과로 바꿈
            return m.group(0)  # 원래 문자열 그대로

        previous = None
        while previous != text:  # 더 이상 바뀌지 않을 때까지 반복 ('정 보 화'처럼 여러 조각으로 쪼개진 경우)
            previous = text
            text = re.sub(r"([가-힣]+) ([가-힣]+)", maybe_join, text)
        return text

    def fix(self, question):
        """질문 하나를 고친다: 1) 띄어쓰기 붙이기 -> 2) 한글 단어마다 글자 오타 고치기"""
        text = self._join_split_words(question)
        return HANGUL_WORD.sub(lambda m: self._fix_word(m.group(0)), text)  # 한글 단어만 골라 _fix_word 적용