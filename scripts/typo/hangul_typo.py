"""한글 오타 공용 도구 (직접 실행하지 않는 '부품' 파일)

typo_golden.py(오타 골든셋 만들기)와 query_fix.py(오타 보정)가 이 파일을 불러서 쓴다.

핵심 아이디어
    컴퓨터 안에서 한글 한 글자('한')는 "초성 ㅎ + 중성 ㅏ + 종성 ㄴ" 세 조각을 숫자 하나로 합친 것이다.
    그래서 글자를 세 조각으로 나눈 뒤 한 조각만 바꾸고 다시 합치면 사람이 치는 오타와 비슷한 글자가 나온다.
        '협' -> (ㅎ, ㅕ, ㅂ) -> 초성 ㅎ을 옆 키 ㅊ 으로 -> (ㅊ, ㅕ, ㅂ) -> '쳡'
"""

import random  # 오타를 넣을 위치와 종류를 무작위로 고르기 위해
import unicodedata  # 글자를 자모로 풀어 쓰는 표준 기능(NFD)을 쓰기 위해

# =====================================================================
# 1. 자모 목록 (유니코드 한글 규칙의 순서 그대로여야 함)
# =====================================================================
# 유니코드는 한글 11,172자를 "초성 19개 x 중성 21개 x 종성 28개" 순서로 번호를 매겨 두었다.
# 아래 세 목록의 순서가 그 번호 순서와 같아야 글자를 나누고 합칠 수 있다.
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")  # 초성 19개
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")  # 중성 21개
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")  # 종성 28개 (0번 = 받침 없음)

HANGUL_START = 0xAC00  # '가'의 유니코드 번호. 모든 한글 글자 번호는 여기서부터 센다
PER_CHO = 21 * 28  # 초성이 하나 바뀔 때 건너뛰는 글자 수 (중성 21 x 종성 28 = 588)
PER_JUNG = 28  # 중성이 하나 바뀔 때 건너뛰는 글자 수 (종성 28)

# =====================================================================
# 2. 오타 재료: 자판에서 붙어 있는 키, 소리가 비슷한 모음
# =====================================================================
# 두벌식 자판에서 손가락이 미끄러지기 쉬운 '붙어 있는 키' 쌍 (좌우 + 위아래)
_ADJACENT_PAIRS = [
    "ㅂㅈ", "ㅈㄷ", "ㄷㄱ", "ㄱㅅ", "ㅁㄴ", "ㄴㅇ", "ㅇㄹ", "ㄹㅎ", "ㅋㅌ", "ㅌㅊ", "ㅊㅍ",  # 자음 좌우
    "ㅂㅁ", "ㅈㄴ", "ㄷㅇ", "ㄱㄹ", "ㅅㅎ", "ㅁㅋ", "ㄴㅌ", "ㅇㅊ", "ㄹㅍ",  # 자음 위아래
    "ㅛㅕ", "ㅕㅑ", "ㅑㅐ", "ㅐㅔ", "ㅗㅓ", "ㅓㅏ", "ㅏㅣ", "ㅠㅜ", "ㅜㅡ",  # 모음 좌우
    "ㅛㅗ", "ㅕㅓ", "ㅑㅏ", "ㅐㅣ", "ㅗㅠ", "ㅓㅜ", "ㅏㅡ",  # 모음 위아래
]

# 위 쌍을 "키 -> 옆에 있는 키 목록" 사전으로 바꾼다. 예: NEIGHBORS["ㄱ"] = ["ㄷ", "ㅅ", "ㄹ"]
NEIGHBORS = {}
for a, b in _ADJACENT_PAIRS:
    NEIGHBORS.setdefault(a, []).append(b)  # a 옆에 b가 있고
    NEIGHBORS.setdefault(b, []).append(a)  # b 옆에도 a가 있다 (양방향)

# 소리가 비슷해서 사람이 자주 헷갈리는 모음 (예: '의료원'을 '의로원'으로 쓰는 것과 같은 부류)
SOUND_ALIKE = {
    "ㅐ": ["ㅔ"], "ㅔ": ["ㅐ"],
    "ㅚ": ["ㅙ", "ㅞ"], "ㅙ": ["ㅚ", "ㅞ"], "ㅞ": ["ㅚ", "ㅙ"],
    "ㅕ": ["ㅓ"], "ㅛ": ["ㅗ"], "ㅠ": ["ㅜ"], "ㅢ": ["ㅣ"],
}


# =====================================================================
# 3. 글자 나누기 / 합치기
# =====================================================================
def is_syllable(ch):
    """완성된 한글 한 글자('가'~'힣')인지. 숫자, 영어, 공백, 'ㄱ' 같은 낱자는 False"""
    # 문자열 크기 비교: 유니코드 번호가 '가'와 '힣' 사이면 완성형 한글이다.
    # 참고: 띄어쓰기 오타로 한 칸이 '가 '(두 글자)가 되면 이 검사는 True 가 된다.
    #   그 칸은 이미 오타를 넣은 위치라 다시 고치지 않으므로 오류는 나지 않고,
    #   옆 글자와 순서가 바뀌는 정도의 차이만 생긴다. 이미 만든 오타 골든셋과 결과를 똑같이 유지하려고
    #   len(ch) == 1 검사를 일부러 넣지 않았다 (넣으면 medium, heavy 오타 일부가 달라진다).
    return "가" <= ch <= "힣"


def split_syllable(ch):
    """'한' -> (초성 번호, 중성 번호, 종성 번호) = (18, 0, 4)"""
    code = ord(ch) - HANGUL_START  # '가'에서 몇 번째 글자인지
    return code // PER_CHO, (code % PER_CHO) // PER_JUNG, code % PER_JUNG  # 나눗셈으로 세 조각 번호를 꺼냄


def join_syllable(cho, jung, jong):
    """(18, 0, 4) -> '한'. split_syllable 의 반대"""
    return chr(HANGUL_START + cho * PER_CHO + jung * PER_JUNG + jong)


def to_jamo(text):
    """'한국' -> 'ㅎㅏㄴㄱㅜㄱ' 처럼 자모 단위로 풀어 쓴 문자열 (두 단어가 얼마나 비슷한지 비교할 때 사용)

    '협회'와 '첩회'는 글자 단위로는 반이 다르지만, 자모 단위로는 6개 중 1개만 다르다.
    오타는 보통 자모 하나가 틀린 것이므로 자모 단위로 비교해야 '거의 같은 단어'를 잘 찾는다.
    """
    return unicodedata.normalize("NFD", text)  # NFD = 유니코드 표준 '풀어 쓰기'. 한글이 아닌 글자는 그대로


# =====================================================================
# 4. 오타 연산 (글자 위치 i 하나에 오타 하나)
# =====================================================================
# 모든 연산의 모양을 똑같이 맞춘다: (글자 목록, 위치, 난수 생성기) -> 성공했으면 True
# 모양이 같아야 add_typos 에서 이름만 골라 똑같은 방식으로 부를 수 있다.
# 오타를 넣을 수 없는 글자면(예: 받침이 없는데 받침 빼기) False 를 돌려주고 아무것도 바꾸지 않는다.

def op_key_slip(chars, i, rng):
    """옆 키 누르기: 초성이나 중성 하나를 자판에서 붙어 있는 키로 바꿈 ('협회' -> '첩회')"""
    cho, jung, jong = split_syllable(chars[i])
    options = []  # 만들 수 있는 오타 후보 (초성, 중성, 종성) 묶음들
    for key in NEIGHBORS.get(CHO[cho], []):  # 초성의 옆 키들
        if key in CHO:  # 옆 키가 초성으로 쓸 수 있는 자음일 때만 (모음 키는 제외)
            options.append((CHO.index(key), jung, jong))
    for key in NEIGHBORS.get(JUNG[jung], []):  # 중성의 옆 키들
        if key in JUNG:
            options.append((cho, JUNG.index(key), jong))
    if not options:  # ㅘ 같은 겹모음 + 옆 키가 없는 초성이면 오타를 못 만듦
        return False
    chars[i] = join_syllable(*rng.choice(options))  # 후보 중 하나를 골라 글자를 바꿈
    return True


def op_drop_final(chars, i, rng):
    """받침 빠뜨리기: '사업' -> '사어'"""
    cho, jung, jong = split_syllable(chars[i])
    if jong == 0:  # 받침이 없는 글자는 뺄 받침이 없음
        return False
    chars[i] = join_syllable(cho, jung, 0)  # 종성 번호 0 = 받침 없음
    return True


def op_sound_alike(chars, i, rng):
    """소리가 비슷한 모음으로 바꾸기: '개' -> '게'"""
    cho, jung, jong = split_syllable(chars[i])
    options = SOUND_ALIKE.get(JUNG[jung])  # 이 모음과 헷갈리는 모음 목록
    if not options:
        return False
    chars[i] = join_syllable(cho, JUNG.index(rng.choice(options)), jong)
    return True


def op_swap(chars, i, rng):
    """옆 글자와 순서 바꾸기: '공단' -> '단공'"""
    if i + 1 >= len(chars):  # 마지막 글자면 바꿀 옆 글자가 없음
        return False
    if not is_syllable(chars[i + 1]) or chars[i] == chars[i + 1]:  # 옆이 한글이 아니거나 같은 글자면 의미 없음
        return False
    chars[i], chars[i + 1] = chars[i + 1], chars[i]  # 파이썬에서 두 값을 한 줄로 맞바꾸는 방법
    return True


def op_space_drop(chars, i, rng):
    """띄어쓰기 빼먹기: '지원센터 홈페이지' -> '지원센터홈페이지'"""
    if i + 1 < len(chars) and chars[i + 1] == " ":  # 이 글자 바로 뒤가 공백일 때만
        chars[i + 1] = ""  # 공백 칸을 빈 문자열로 (칸을 지우지 않는 이유: 다른 위치 번호가 밀리지 않게)
        return True
    return False


def op_space_add(chars, i, rng):
    """띄어쓰기 잘못 넣기: '차세대' -> '차세 대'"""
    if i + 1 < len(chars) and is_syllable(chars[i + 1]):  # 단어 중간(뒤에 한글이 이어질 때)만
        chars[i] = chars[i] + " "  # 글자 뒤에 공백을 붙임 (역시 칸 수는 그대로 유지)
        return True
    return False


# 이름 -> 함수. add_typos 가 이 중 하나를 무작위로 고른다.
# 순서를 바꾸면 같은 seed 라도 다른 오타가 나오므로 순서는 고정한다.
SYLLABLE_OPS = {
    "key_slip": op_key_slip,
    "drop_final": op_drop_final,
    "sound_alike": op_sound_alike,
    "swap": op_swap,
}
SPACE_OPS = {
    "space_drop": op_space_drop,
    "space_add": op_space_add,
}


# =====================================================================
# 5. 문장에 오타 n개 넣기
# =====================================================================
def add_typos(text, positions, n_typos, rng, use_space_ops=True):
    """text 의 positions(오타를 넣어도 되는 글자 위치) 안에서 오타 n_typos 개를 만든다.

    돌려주는 값: (오타가 들어간 문장, 실제로 넣은 오타 종류 목록)
    예: ("국가철도단공 '철도인프라 ...'", ["swap"])
    """
    chars = list(text)  # 문자열은 바꿀 수 없어서, 글자 하나하나를 담은 목록으로 바꿔서 고친다
    pool = [p for p in positions if is_syllable(chars[p])]  # 후보 위치 중 한글 글자만
    ops = {**SYLLABLE_OPS, **(SPACE_OPS if use_space_ops else {})}  # 쓸 오타 종류 모음
    names = list(ops)  # 무작위로 고를 이름 목록

    applied = []  # 실제로 넣은 오타 종류 (기록용)
    used = set()  # 이미 오타를 넣은 위치 (한 글자에 오타가 두 번 겹치지 않게)
    max_tries = n_typos * 30  # 실패가 계속될 때 무한 반복하지 않도록 시도 횟수 상한

    for _ in range(max_tries):
        if len(applied) >= n_typos or not pool:  # 목표 개수를 채웠거나 넣을 곳이 없으면 끝
            break
        i = rng.choice(pool)  # 오타 넣을 위치를 하나 고르고
        if i in used:  # 이미 고친 위치면 다시 고름
            continue
        name = rng.choice(names)  # 오타 종류를 하나 고른다
        if ops[name](chars, i, rng):  # 그 오타를 시도해서 성공하면
            used.add(i)
            if name == "swap":  # 순서 바꾸기는 옆 글자(i+1)도 건드렸으니 함께 잠금
                used.add(i + 1)
            applied.append(name)
        # 실패하면(예: 받침 없는 글자에 받침 빼기) 아무것도 안 바뀌고 다음 시도로 넘어간다

    return "".join(chars), applied  # 글자 목록을 다시 문자열로 합쳐서 돌려줌


def make_rng(seed):
    """seed 가 같으면 항상 같은 순서로 '무작위' 값을 내는 생성기 (팀원 누가 돌려도 같은 오타가 나오게)"""
    return random.Random(seed)
