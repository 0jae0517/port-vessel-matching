"""선박 매칭: 후보 검색 + 쌍 피처 + LightGBM 랭커.

모든 처리는 행 단위로 독립적이다. 쿼리 한 행의 후보·피처는 그 행의 입력값과
공식 데이터(train.csv에서 만든 사전, vessel_catalog.csv)만으로 계산한다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode

HANGUL_RE = re.compile(r"[가-힣]")
SPLIT_RE = re.compile(r"[0-9]+|[^0-9]+")
STOP_TOKENS = {"NO"}
ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10"}
KOREAN_NUMBERS = {"원": "1", "투": "2", "쓰리": "3"}
# 한글로 적은 영문 알파벳 이름 (에이치씨 → HC, 케이티에스 → KTS)
KOREAN_LETTERS = {
    "에이": "A", "비": "B", "씨": "C", "디": "D", "이": "E", "에프": "F", "지": "G", "에이치": "H", "에치": "H",
    "아이": "I", "제이": "J", "케이": "K", "엘": "L", "엠": "M", "엔": "N", "오": "O", "피": "P", "큐": "Q",
    "알": "R", "에스": "S", "티": "T", "유": "U", "브이": "V", "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z",
    "시": "C", "앤": "N",  # 시앤시 → CNC (C&C)
}

TOP_COMPACT = 30
TOP_SKELETON = 30
TOP_COMBINED = 30
TOP_TONNAGE = 10
TONNAGE_BONUS = 25.0
CHUNK = 200


# ---------------------------------------------------------------- 문자열 처리

def is_hangul(value: object) -> bool:
    return isinstance(value, str) and bool(HANGUL_RE.search(value))


def tokenize(value: object) -> list[str]:
    """NFKC·대문자 → 영숫자·한글 이외는 공백 → 숫자/문자 경계 분리 → 'NO' 제거.

    로마 숫자 단독 토큰은 아라비아 숫자로, 한글 선박명 끝의 '호'는 뗀다 (그린오션호 → 그린오션).
    """
    if not isinstance(value, str):
        return []
    text = unicodedata.normalize("NFKC", value).upper()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    out: list[str] = []
    for word in text.split():
        for part in SPLIT_RE.findall(word):
            if part.isdigit():
                part = part.lstrip("0") or "0"
            elif part in ROMAN:
                part = ROMAN[part]
            elif HANGUL_RE.search(part) and len(part) >= 3 and part.endswith("호"):
                part = part[:-1]
            if part not in STOP_TOKENS:
                out.append(part)
    return out


def korean_letters(token: str) -> str | None:
    """토큰 전체가 알파벳 이름(2글자 이상)으로만 이루어지면 영문 약어로 바꾼다."""
    names = sorted(KOREAN_LETTERS, key=len, reverse=True)

    def parse(rest: str) -> list[str] | None:
        if not rest:
            return []
        for name in names:
            if rest.startswith(name):
                tail = parse(rest[len(name):])
                if tail is not None:
                    return [KOREAN_LETTERS[name]] + tail
        return None

    letters = parse(token)
    return "".join(letters) if letters and len(letters) >= 2 else None


# ---------------------------------------------------------------- 자모 기반 로마자 변형
# 같은 한글이라도 영문 원어 표기는 제각각이다 (커리지 → COURAGE, 포천 → FORTUNE, 베마 → VEMA).
# 표준 로마자(RR) 외에 영문 원어에 가까운 변형 2개를 만든다: [RR, 영어식, 외래어 역변환]
_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
INITIAL = {
    "ㄱ": ("G", "G", "G"), "ㄲ": ("KK", "K", "K"), "ㄴ": ("N", "N", "N"), "ㄷ": ("D", "D", "D"),
    "ㄸ": ("TT", "T", "T"), "ㄹ": ("R", "R", "R"), "ㅁ": ("M", "M", "M"), "ㅂ": ("B", "B", "V"),
    "ㅃ": ("PP", "P", "P"), "ㅅ": ("S", "S", "S"), "ㅆ": ("SS", "S", "S"), "ㅇ": ("", "", ""),
    "ㅈ": ("J", "J", "J"), "ㅉ": ("JJ", "J", "J"), "ㅊ": ("CH", "CH", "T"), "ㅋ": ("K", "K", "C"),
    "ㅌ": ("T", "T", "T"), "ㅍ": ("P", "P", "F"), "ㅎ": ("H", "H", "H"),
}
VOWEL = {
    "ㅏ": ("A", "A", "A"), "ㅐ": ("AE", "E", "A"), "ㅑ": ("YA", "YA", "IA"), "ㅒ": ("YAE", "YE", "YA"),
    "ㅓ": ("EO", "U", "E"), "ㅔ": ("E", "E", "E"), "ㅕ": ("YEO", "YO", "YE"), "ㅖ": ("YE", "YE", "YE"),
    "ㅗ": ("O", "O", "O"), "ㅘ": ("WA", "WA", "UA"), "ㅙ": ("WAE", "WE", "WE"), "ㅚ": ("OE", "OE", "WE"),
    "ㅛ": ("YO", "YO", "IO"), "ㅜ": ("U", "OO", "U"), "ㅝ": ("WO", "WO", "UO"), "ㅞ": ("WE", "WE", "WE"),
    "ㅟ": ("WI", "WI", "UI"), "ㅠ": ("YU", "YU", "U"), "ㅡ": ("EU", "", ""), "ㅢ": ("UI", "UI", "I"),
    "ㅣ": ("I", "I", "I"),
}
FINAL = {
    " ": ("", "", ""), "ㄱ": ("K", "K", "C"), "ㄲ": ("K", "K", "C"), "ㄳ": ("K", "K", "C"),
    "ㄴ": ("N", "N", "N"), "ㄵ": ("N", "N", "N"), "ㄶ": ("N", "N", "N"), "ㄷ": ("T", "T", "D"),
    "ㄹ": ("L", "L", "L"), "ㄺ": ("K", "K", "K"), "ㄻ": ("M", "M", "M"), "ㄼ": ("L", "L", "L"),
    "ㄽ": ("L", "L", "L"), "ㄾ": ("L", "L", "L"), "ㄿ": ("P", "P", "P"), "ㅀ": ("L", "L", "L"),
    "ㅁ": ("M", "M", "M"), "ㅂ": ("P", "P", "B"), "ㅄ": ("P", "P", "P"), "ㅅ": ("T", "T", "S"),
    "ㅆ": ("T", "S", "S"), "ㅇ": ("NG", "NG", "NG"), "ㅈ": ("T", "J", "Z"), "ㅊ": ("T", "CH", "T"),
    "ㅋ": ("K", "K", "C"), "ㅌ": ("T", "T", "T"), "ㅍ": ("P", "P", "F"), "ㅎ": ("T", "", ""),
}
Y_VOWELS = {"ㅑ": "A", "ㅕ": "O", "ㅛ": "O", "ㅠ": "U", "ㅒ": "E", "ㅖ": "E"}
N_VARIANTS = 3


def romanize_variants(token: str) -> list[str]:
    """한글 토큰 → [RR, 영어식, 외래어 역변환] 로마자. 한글이 아닌 글자는 그대로 둔다."""
    outs = [[] for _ in range(N_VARIANTS)]
    prev_final = None
    skip_next_i = False
    for pos, ch in enumerate(token):
        code = ord(ch) - 0xAC00
        if skip_next_i:
            # '에이' → A (레이크 → LAKE, 그레이스 → GRACE): 영어식 변형에서는 '이'를 건너뛴다
            skip_next_i = False
            outs[0].append("I")
            prev_final = " "
            continue
        if not 0 <= code < 11172:
            for out in outs:
                out.append(ch)
            prev_final = None
            continue
        lead, vowel, tail = _L[code // 588], _V[(code % 588) // 28], _T[code % 28]
        for k, out in enumerate(outs):
            ini, vow = INITIAL[lead][k], VOWEL[vowel][k]
            if k > 0:
                if lead == "ㄹ" and prev_final == "ㄹ":
                    ini = ""  # 얼라이언스 → ALLIANCE (ㄹㄹ → L)
                if lead in "ㅅㅈㅊ" and vowel in Y_VOWELS:
                    ini, vow = {"ㅅ": "SH", "ㅈ": "J", "ㅊ": "CH"}[lead], Y_VOWELS[vowel]  # 슈 → SHU
                if vowel == "ㅔ" and tail == " " and token[pos + 1 : pos + 2] == "이":
                    vow = "A"
                if vowel == "ㅡ" and tail != " ":
                    vow = "U"  # 받침 있는 ㅡ는 실제 모음 (흥 → HUNG)
            out.append(ini + vow + FINAL[tail][k])
        if vowel == "ㅔ" and tail == " " and token[pos + 1 : pos + 2] == "이":
            skip_next_i = True
        prev_final = tail
    return ["".join(o) for o in outs]


def romanize_token(token: str) -> str:
    if not HANGUL_RE.search(token):
        return token
    return "".join(ch for ch in unidecode(token).upper() if ch.isalnum())


_SKELETON_MAP = {}
for letters, code in (("KGCQ", "K"), ("BPFV", "P"), ("DT", "T"), ("SZJ", "S"), ("LR", "L"), ("M", "M"), ("N", "N")):
    for ch in letters:
        _SKELETON_MAP[ch] = code
VOWELS = set("AEIOUYWH")


def skeleton(latin: str, english: bool) -> str:
    """자음 골격. 한글 로마자와 영문 표기를 같은 축에서 비교하기 위한 키.

    영문(english=True)은 한글 표기 발음에 맞춘다.
      - 모음 앞이 아닌 R 제거 (EVER → 에버)
      - TH → S (CATHERINE → 캐서린), TI+모음 → S (REFLECTION → 리플렉션)
      - C/G + E/I/Y → S/J (COURAGE → 커리지, GRACE → 그레이스)
    """
    s = latin.upper()
    if english:
        s = re.sub(r"TI(?=[AEIOU])", "SI", s.replace("TH", "S"))
        s = re.sub(r"C(?=[EIY])", "S", s)
        s = re.sub(r"G(?=[EIY])", "J", s)
    s = s.replace("PH", "F").replace("CK", "K").replace("X", "KS")
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch.isdigit():
            out.append(ch)
            continue
        if english and ch == "R" and (i + 1 >= len(s) or s[i + 1] not in "AEIOUY"):
            continue
        code = _SKELETON_MAP.get(ch)
        if code and (not out or out[-1] != code):
            out.append(code)
    return "".join(out)


def numbers(tokens: list[str]) -> frozenset[str]:
    return frozenset(t for t in tokens if t.isdigit())


def positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def relative_gap(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 2.0
    return min(abs(left - right) / max(left, right, 1.0), 2.0)


# ---------------------------------------------------------------- train 기반 사전

def build_resources(train: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    """train 정답 쌍에서 한글 토큰 사전, 국적·선종 동시출현 확률, 호출부호 표를 만든다."""
    cat = catalog.set_index("vessel_id")
    token_votes: dict[str, Counter] = defaultdict(Counter)
    whole_votes: dict[str, Counter] = defaultdict(Counter)
    flag_pair: dict[str, Counter] = defaultdict(Counter)
    type_pair: dict[str, Counter] = defaultdict(Counter)
    callsign: dict[str, str] = {}
    name_votes: dict[str, Counter] = defaultdict(Counter)

    for row in train.itertuples(index=False):
        target = cat.loc[row.target_vessel_id]
        callsign[row.call_sign] = row.target_vessel_id
        key = "".join(tokenize(row.vessel_name))
        if key:
            name_votes[key][row.target_vessel_id] += 1
        flag_pair[str(row.flag_name)][str(target.flag_name)] += 1
        type_pair[str(row.vessel_type)][str(target.vessel_type)] += 1
        if not is_hangul(row.vessel_name):
            continue
        q_words = [t for t in tokenize(row.vessel_name) if not t.isdigit()]
        c_words = [t for t in tokenize(target.vessel_name) if not t.isdigit()]
        if not q_words or not c_words:
            continue
        whole_votes["".join(q_words)][" ".join(c_words)] += 1
        if len(q_words) == len(c_words):
            for q, c in zip(q_words, c_words):
                if HANGUL_RE.search(q):
                    token_votes[q][c] += 1
        elif len(q_words) == 1:
            token_votes[q_words[0]][" ".join(c_words)] += 1

    def best(votes: dict[str, Counter]) -> dict[str, str]:
        return {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, v in votes.items()}

    def probs(pairs: dict[str, Counter]) -> dict[str, dict[str, float]]:
        out = {}
        for q, counter in pairs.items():
            total = sum(counter.values())
            out[q] = {c: n / total for c, n in counter.items()}
        return out

    return {
        "token_dict": best(token_votes),
        "whole_dict": best(whole_votes),
        "flag_prob": probs(flag_pair),
        "type_prob": probs(type_pair),
        "callsign": callsign,
        "name_lookup": {k: dict(v) for k, v in name_votes.items()},
        "syl_map": learn_syllable_map(token_votes),
    }


# ---------------------------------------------------------------- 음절 표기 학습
# train 정답 쌍(한글 토큰 ↔ 영문)을 음절 단위로 정렬해 '음절 → 영문 조각' 빈도를 배운다.
# 예: 스피릿 ↔ SPIRIT → 스:S, 피:PI, 릿:RIT. 처음 보는 단어도 음절 조합으로 영문 표기를 만들 수 있다.
SYL_MAX_PIECE = 6
SYL_TOP = 4
LEARN_BEAM = 12
LEARN_VARIANTS = 6


def _is_hangul_word(token: str) -> bool:
    return bool(token) and all(0xAC00 <= ord(ch) <= 0xD7A3 for ch in token)


_syl_rule_cache: dict[str, tuple[str, ...]] = {}


def syllable_rule_variants(syl: str) -> tuple[str, ...]:
    if syl not in _syl_rule_cache:
        _syl_rule_cache[syl] = tuple(dict.fromkeys(romanize_variants(syl)))
    return _syl_rule_cache[syl]


def align_syllables(word: str, english: str) -> list[str] | None:
    """한글 음절 각각에 영문 연속 조각을 대응시키는 DP. 규칙 로마자와 가장 비슷한 분할을 고른다."""
    n, m = len(word), len(english)
    if n == 0 or m == 0 or m > n * SYL_MAX_PIECE:
        return None
    neg = -1e9
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i, syl in enumerate(word):
        rules = syllable_rule_variants(syl)
        for j in range(m + 1):
            if dp[i][j] <= neg:
                continue
            for length in range(0, min(SYL_MAX_PIECE, m - j) + 1):
                piece = english[j : j + length]
                score = -0.5 if not piece else max(fuzz.ratio(r, piece) for r in rules) / 100.0
                if dp[i][j] + score > dp[i + 1][j + length]:
                    dp[i + 1][j + length] = dp[i][j] + score
                    back[i + 1][j + length] = length
    if dp[n][m] <= neg or dp[n][m] / n < 0.35:
        return None
    pieces, j = [], m
    for i in range(n, 0, -1):
        length = back[i][j]
        pieces.append(english[j - length : j])
        j -= length
    return pieces[::-1]


def learn_syllable_map(token_votes: dict[str, Counter]) -> dict[str, list[tuple[str, float]]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for word, votes in token_votes.items():
        if not _is_hangul_word(word):
            continue
        for english, n in votes.items():
            english = "".join(ch for ch in english if ch.isalpha())
            pieces = align_syllables(word, english)
            if pieces is None:
                continue
            for syl, piece in zip(word, pieces):
                counts[syl][piece] += n
    out = {}
    for syl, counter in counts.items():
        total = sum(counter.values())
        top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:SYL_TOP]
        out[syl] = [(piece, c / total) for piece, c in top]
    return out


def learned_word_variants(word: str, syl_map: dict) -> list[tuple[str, float]]:
    """음절별 (학습 표기 + 규칙 표기) 후보를 beam search로 조합. (표기, log확률) 목록."""
    beams = [("", 0.0)]
    for syl in word:
        rules = syllable_rule_variants(syl)
        learned = syl_map.get(syl, [])
        options: dict[str, float] = {}
        if learned:
            for piece, p in learned:
                options[piece] = max(options.get(piece, -1e9), np.log(0.85 * p))
            for r in rules:
                options[r] = max(options.get(r, -1e9), np.log(0.15 / len(rules)))
        else:
            for r in rules:
                options[r] = np.log(1.0 / len(rules))
        beams = sorted(((b + o, s + lp) for b, s in beams for o, lp in options.items()),
                       key=lambda x: (-x[1], x[0]))[:LEARN_BEAM]
    return beams


def learned_variants(tokens: list[str], res: dict) -> list[str]:
    """선박명 토큰 전체에 대한 학습 기반 영문 표기 후보(공백 없이) 상위 LEARN_VARIANTS개."""
    syl_map = res.get("syl_map", {})
    beams = [("", 0.0)]
    for t in tokens:
        mapped = res["token_dict"].get(t)
        letters = korean_letters(t) if HANGUL_RE.search(t) else None
        if mapped is not None:
            options = [(mapped.replace(" ", ""), 0.0)]
        elif letters:
            options = [(letters, 0.0)]
        elif _is_hangul_word(t):
            options = learned_word_variants(t, syl_map)
        else:
            options = [(t, 0.0)]
        beams = sorted(((b + o, s + lp) for b, s in beams for o, lp in options),
                       key=lambda x: (-x[1], x[0]))[:LEARN_BEAM]
    return list(dict.fromkeys(b for b, _ in beams))[:LEARN_VARIANTS]


def lookup_by_name(row: dict, res: dict, index: CatalogIndex) -> str | None:
    """train에 같은 표기의 선박명이 있으면 그 정답 선박. 여러 척이면 빈도 → 톤수 차이 순."""
    votes = res.get("name_lookup", {}).get("".join(tokenize(row.get("vessel_name"))))
    if not votes:
        return None
    gross = positive_float(row.get("gross_tonnage")) or positive_float(row.get("international_gross_tonnage"))

    def order(item):
        vid, count = item
        gap = relative_gap(gross, positive_float(index.gt[index.id_to_index[vid]])) if vid in index.id_to_index else 9.0
        return (-count, gap, vid)

    best = sorted(votes.items(), key=order)[0][0]
    return best if best in index.id_to_index else None


# ---------------------------------------------------------------- 카탈로그 색인

class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame) -> None:
        self.catalog = catalog.reset_index(drop=True)
        self.ids = self.catalog["vessel_id"].astype(str).to_numpy()
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        toks = [tokenize(v) for v in self.catalog["vessel_name"]]
        self.clean = [" ".join(t) for t in toks]
        self.compact = ["".join(t) for t in toks]
        self.skel = [skeleton(c, english=True) for c in self.compact]
        self.first = [t[0] if t else "" for t in toks]
        # 단어 희귀도(IDF): HAFNIA·GLOVIS 같은 선단 접두어는 흔하고, WREN 같은 고유 이름은 드물다
        self.words = [[w for w in t if not w.isdigit()] for t in toks]
        df = Counter(w for ws in self.words for w in set(ws))
        n_docs = len(toks)
        self.idf = {w: float(np.log(n_docs / (1 + c))) for w, c in df.items()}
        self.max_idf = float(np.log(n_docs))
        self.word_idf = [np.array([self.idf[w] for w in ws], dtype=np.float32) for ws in self.words]
        # 같은 총톤수를 가진 카탈로그 선박 수 (톤수 정확 일치가 얼마나 결정적인지)
        gt_all = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.gt_count = Counter(float(v) for v in gt_all if np.isfinite(v) and v > 0)
        self.nums = [numbers(t) for t in toks]
        self.gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.flag = self.catalog["flag_name"].astype(str).to_numpy()
        self.vtype = self.catalog["vessel_type"].astype(str).to_numpy()
        valid = np.flatnonzero(np.isfinite(self.gt) & (self.gt > 0))
        order = np.argsort(self.gt[valid], kind="mergesort")
        self.sorted_gt_idx = valid[order]
        self.sorted_gt = self.gt[self.sorted_gt_idx]

    def nearest_gt(self, value: float | None, count: int) -> list[int]:
        if value is None:
            return []
        pos = int(np.searchsorted(self.sorted_gt, value))
        lo, hi = max(pos - count, 0), min(pos + count, len(self.sorted_gt))
        window = self.sorted_gt_idx[lo:hi]
        gaps = np.abs(self.gt[window] - value)
        return [int(i) for i in window[np.lexsort((window, gaps))][:count]]


# ---------------------------------------------------------------- 쿼리 준비

def idf_words(tokens: list[str], res: dict, hangul: bool) -> list[str]:
    """단어 단위 비교용 영문 단어 목록. 한글 단어는 음절 학습 1순위 표기로 바꾼다."""
    out = []
    for t in tokens:
        if t.isdigit():
            continue
        if hangul and HANGUL_RE.search(t):
            letters = korean_letters(t)
            if letters:
                out.append(letters)
            elif _is_hangul_word(t):
                out.append(learned_word_variants(t, res.get("syl_map", {}))[0][0])
            else:
                out.append(romanize_token(t))
        else:
            out.append(t)
    return [w for w in out if w]


def trans_compact_of(tokens: list[str]) -> str:
    return "".join(tokens)


def prepare_query(row: dict, res: dict) -> dict:
    name = row.get("vessel_name")
    toks = tokenize(name)
    hangul = is_hangul(name)
    words = [t for t in toks if not t.isdigit()]
    covered = 0
    hangul_words = [t for t in words if HANGUL_RE.search(t)]
    whole = res["whole_dict"].get("".join(words)) if hangul else None
    if whole is not None:
        digits = [t for t in toks if t.isdigit()]
        # 숫자 위치 유지 (109효동케미 → 109 HYODONG CHEMI)
        translated = digits + whole.split() if toks and toks[0].isdigit() else whole.split() + digits
        covered = len(hangul_words)
    else:
        translated = []
        for t in toks:
            if HANGUL_RE.search(t):
                mapped = res["token_dict"].get(t)
                letters = korean_letters(t)
                if mapped is not None:
                    covered += 1
                    translated.extend(mapped.split())
                elif t in KOREAN_NUMBERS:
                    translated.append(KOREAN_NUMBERS[t])
                elif letters is not None:
                    translated.append(letters)
                else:
                    translated.append(romanize_token(t))
            else:
                translated.append(t)
    toks = [KOREAN_NUMBERS.get(t, t) for t in toks]
    var_compacts: list[str] = []
    if hangul and whole is not None:
        var_compacts = [trans_compact_of(translated)] * N_VARIANTS
    elif hangul:
        parts = [[] for _ in range(N_VARIANTS)]
        for t in toks:
            if HANGUL_RE.search(t):
                mapped = res["token_dict"].get(t)
                letters = korean_letters(t)
                if mapped is not None:  # 사전에 있는 토큰은 사전 번역을 그대로 쓴다 (글로비스 → GLOVIS)
                    vs = [mapped.replace(" ", "")] * N_VARIANTS
                elif letters:
                    vs = [letters] * N_VARIANTS
                else:
                    vs = romanize_variants(t)
            else:
                vs = [t] * N_VARIANTS
            for k in range(N_VARIANTS):
                parts[k].append(vs[k])
        var_compacts = ["".join(p) for p in parts]
    romanized = [romanize_token(t) for t in toks]
    trans_compact = "".join(translated)
    rom_compact = "".join(romanized)
    return {
        "hangul": hangul,
        "whole_hit": whole is not None,
        "dict_cov": covered / len(hangul_words) if hangul_words else 1.0,
        "clean": " ".join(translated),
        "compact": trans_compact,
        "rom_compact": rom_compact,
        "variants": var_compacts,
        "learned": ([trans_compact] if whole is not None else learned_variants(toks, res)) if hangul else [],
        "skel_rom": skeleton(rom_compact, english=not hangul),
        "skel_trans": skeleton(trans_compact, english=not hangul),
        "first": translated[0] if translated else "",
        "words": idf_words(toks, res, hangul),
        "nums": numbers(toks),
        "gross": positive_float(row.get("gross_tonnage")),
        "intl": positive_float(row.get("international_gross_tonnage")),
        "flag": str(row.get("flag_name")),
        "vtype": str(row.get("vessel_type")),
    }


# ---------------------------------------------------------------- 후보 + 피처

FEATURE_NAMES = (
    "is_hangul", "dict_cov", "whole_hit",
    "s_compact", "s_skel_rom", "s_skel_trans", "s_token_set", "s_token_sort", "s_partial", "s_jw",
    "s_rom_compact", "exact", "first_eq", "len_ratio",
    "num_equal", "num_conflict", "q_has_num", "c_has_num",
    "gap_gross", "gap_intl", "gap_best", "gt_exact", "q_gt_missing", "c_gt_missing",
    "flag_prob", "flag_seen", "type_prob", "type_seen",
    "rel_compact", "rel_skel", "rel_token_set", "rel_gap", "rank_compact", "n_cand",
    "s_var_ratio", "s_var_jw", "s_var_partial", "rel_var",
    "s_learn_ratio", "s_learn_top1", "s_learn_jw", "rel_learn",
    "idf_cover_q", "idf_cover_c", "last_word", "rarest_word", "rel_idf_cover",
    "gt_exact_count", "gt_exact_unique", "flag_type",
)


def _topk(matrix: np.ndarray, k: int) -> np.ndarray:
    # uint8 행렬이라 부호를 뒤집지 않고 큰 쪽 k개를 고른다
    k = min(k, matrix.shape[1])
    n = matrix.shape[1]
    return np.argpartition(matrix, n - k, axis=1)[:, n - k :]


def candidates_and_features(
    queries: list[dict], index: CatalogIndex, res: dict, force: list[int] | None = None,
    hits: list[bool] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """쿼리마다 후보 카탈로그 인덱스 배열과, 모든 (쿼리, 후보) 쌍의 피처 행렬을 반환.

    force: 학습용. 해당 인덱스(정답)를 후보에 넣는다. hits에는 넣기 전 포함 여부를 기록한다.
    """
    cand_lists: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    for start in range(0, len(queries), CHUNK):
        part = queries[start : start + CHUNK]
        m_compact = process.cdist([q["compact"] for q in part], index.compact, scorer=fuzz.ratio,
                                  dtype=np.uint8, workers=-1)
        m_skel = process.cdist([q["skel_rom"] for q in part], index.skel, scorer=fuzz.ratio,
                               dtype=np.uint8, workers=-1)
        hangul_rows = [i for i, q in enumerate(part) if q["hangul"]]
        m_skel_t = None
        if hangul_rows:
            m_skel_t = process.cdist([part[i]["skel_trans"] for i in hangul_rows], index.skel,
                                     scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        top_c = _topk(m_compact, TOP_COMPACT)
        top_s = _topk(m_skel, TOP_SKELETON)
        top_t = _topk(m_skel_t, TOP_SKELETON) if m_skel_t is not None else None
        hangul_pos = {r: j for j, r in enumerate(hangul_rows)}

        # 이름 점수 + 톤수 보너스: 골격이 짧아 동점이 많은 경우(우찬 → WOO CHAN) 톤수로 가른다
        bonus = _tonnage_bonus(part, index)
        top_cb = _topk(m_compact + bonus, TOP_COMBINED)
        top_sb = _topk(m_skel + bonus, TOP_COMBINED)
        top_tb = _topk(m_skel_t + bonus[hangul_rows], TOP_COMBINED) if m_skel_t is not None else None
        # 로마자 변형(영어식·외래어 역변환)별 후보: 이름만 / 이름+톤수
        top_var: list[np.ndarray] = []
        if hangul_rows:
            for k in range(1, N_VARIANTS):
                m_var = process.cdist([part[i]["variants"][k] for i in hangul_rows], index.compact,
                                      scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
                top_var.append(_topk(m_var, TOP_COMPACT))
                top_var.append(_topk(m_var + bonus[hangul_rows], TOP_COMBINED))
            # 음절 학습 표기 상위 3개별 후보
            for k in range(3):
                queries_k = [(part[i]["learned"] + [""] * 3)[k] or part[i]["compact"] for i in hangul_rows]
                m_learn = process.cdist(queries_k, index.compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
                top_var.append(_topk(m_learn, TOP_COMPACT))
                top_var.append(_topk(m_learn + bonus[hangul_rows], TOP_COMBINED))

        for i, q in enumerate(part):
            pool = list(top_c[i]) + list(top_s[i]) + list(top_cb[i]) + list(top_sb[i])
            if i in hangul_pos:
                pool += list(top_t[hangul_pos[i]]) + list(top_tb[hangul_pos[i]])
                for tv in top_var:
                    pool += list(tv[hangul_pos[i]])
            pool += index.nearest_gt(q["gross"], TOP_TONNAGE)
            if q["intl"] != q["gross"]:
                pool += index.nearest_gt(q["intl"], TOP_TONNAGE)
            if force is not None:
                if hits is not None:
                    hits.append(force[start + i] in pool)
                pool.append(force[start + i])
            cand = np.array(sorted(set(int(c) for c in pool)), dtype=np.int64)
            cand_lists.append(cand)
            skel_t_row = m_skel_t[hangul_pos[i], cand] if i in hangul_pos else None
            blocks.append(_pair_features(q, cand, index, res, m_compact[i, cand], m_skel[i, cand], skel_t_row))
    return cand_lists, np.vstack(blocks).astype(np.float32)


def _tonnage_bonus(part: list[dict], index: CatalogIndex) -> np.ndarray:
    """(쿼리, 카탈로그) 톤수 차이 10% 이내면 최대 TONNAGE_BONUS 점. 톤수가 없으면 0."""
    gt = index.gt[None, :]
    best = np.full((len(part), len(index.gt)), np.inf, dtype=np.float32)
    for key in ("gross", "intl"):
        q = np.array([np.nan if p[key] is None else p[key] for p in part], dtype=np.float64)[:, None]
        with np.errstate(invalid="ignore"):
            gap = np.abs(q - gt) / np.maximum(np.maximum(q, gt), 1.0)
        best = np.fmin(best, np.nan_to_num(gap, nan=np.inf).astype(np.float32))
    return TONNAGE_BONUS * np.clip(1.0 - best / 0.1, 0.0, 1.0)


def _pair_features(q, cand, index, res, s_compact, s_skel, s_skel_t) -> np.ndarray:
    n = len(cand)
    c_clean = [index.clean[c] for c in cand]
    c_compact = [index.compact[c] for c in cand]
    one = lambda scorer, a, b: process.cdist([a], b, scorer=scorer, dtype=np.float32, workers=1)[0]
    s_token_set = one(fuzz.token_set_ratio, q["clean"], c_clean)
    s_token_sort = one(fuzz.token_sort_ratio, q["clean"], c_clean)
    s_partial = one(fuzz.partial_ratio, q["compact"], c_compact)
    s_jw = one(JaroWinkler.normalized_similarity, q["compact"], c_compact)
    s_rom = one(fuzz.ratio, q["rom_compact"], c_compact)
    if s_skel_t is None:
        s_skel_t = one(fuzz.ratio, q["skel_trans"], [index.skel[c] for c in cand])
    if q["variants"]:
        s_var = np.max([one(fuzz.ratio, v, c_compact) for v in q["variants"]], axis=0)
        s_var_jw = np.max([one(JaroWinkler.normalized_similarity, v, c_compact) for v in q["variants"]], axis=0)
        s_var_partial = np.max([one(fuzz.partial_ratio, v, c_compact) for v in q["variants"]], axis=0)
    else:
        s_var, s_var_jw, s_var_partial = s_compact.astype(np.float32), s_jw, s_partial
    if q["learned"]:
        learn_ratio = np.array([one(fuzz.ratio, v, c_compact) for v in q["learned"]])
        s_learn = learn_ratio.max(axis=0)
        s_learn_top1 = learn_ratio[0]
        s_learn_jw = np.max([one(JaroWinkler.normalized_similarity, v, c_compact) for v in q["learned"]], axis=0)
    else:
        s_learn, s_learn_top1, s_learn_jw = s_compact.astype(np.float32), s_compact.astype(np.float32), s_jw

    qlen = max(len(q["compact"]), 1)
    gt = index.gt[cand]
    gaps_g = np.array([relative_gap(q["gross"], positive_float(v)) for v in gt])
    gaps_i = np.array([relative_gap(q["intl"], positive_float(v)) for v in gt])
    gap_best = np.minimum(gaps_g, gaps_i)
    flag_p = res["flag_prob"].get(q["flag"], {})
    type_p = res["type_prob"].get(q["vtype"], {})
    c_nums = [index.nums[c] for c in cand]
    s_compact = s_compact.astype(np.float32)
    order = np.argsort(-s_compact, kind="mergesort")
    rank = np.empty(n, dtype=np.float32)
    rank[order] = np.arange(n)

    cols = [
        np.full(n, float(q["hangul"])), np.full(n, q["dict_cov"]), np.full(n, float(q["whole_hit"])),
        s_compact, s_skel.astype(np.float32), s_skel_t.astype(np.float32), s_token_set, s_token_sort,
        s_partial, s_jw, s_rom,
        np.array([float(bool(q["compact"]) and q["compact"] == c) for c in c_compact]),
        np.array([float(bool(q["first"]) and q["first"] == index.first[c]) for c in cand]),
        np.array([min(qlen, len(c)) / max(qlen, len(c), 1) for c in c_compact]),
        np.array([float(bool(q["nums"]) and q["nums"] == cn) for cn in c_nums]),
        np.array([float(bool(q["nums"]) and bool(cn) and q["nums"] != cn) for cn in c_nums]),
        np.full(n, float(bool(q["nums"]))),
        np.array([float(bool(cn)) for cn in c_nums]),
        gaps_g, gaps_i, gap_best,
        np.array([float(v in (q["gross"], q["intl"])) for v in gt]),
        np.full(n, float(q["gross"] is None and q["intl"] is None)),
        np.array([float(positive_float(v) is None) for v in gt]),
        np.array([flag_p.get(index.flag[c], 0.0) for c in cand]),
        np.full(n, float(bool(flag_p))),
        np.array([type_p.get(index.vtype[c], 0.0) for c in cand]),
        np.full(n, float(bool(type_p))),
        s_compact - s_compact.max(),
        s_skel.astype(np.float32) - s_skel.max(),
        s_token_set - s_token_set.max(),
        gap_best - gap_best.min(),
        rank,
        np.full(n, float(n)),
        s_var, s_var_jw, s_var_partial, s_var - s_var.max(),
        s_learn, s_learn_top1, s_learn_jw, s_learn - s_learn.max(),
    ]
    cols += _word_features(q, cand, index)
    # 톤수 정확 일치의 희소성: 같은 총톤수 선박이 1척뿐이면 거의 확정
    counts = [index.gt_count.get(float(v), 0) for v in (q["gross"], q["intl"]) if v is not None]
    q_count = min(counts) if counts else 0
    exact = np.array([float(v in (q["gross"], q["intl"])) for v in gt])
    flag_vec = np.array([flag_p.get(index.flag[c], 0.0) for c in cand])
    type_vec = np.array([type_p.get(index.vtype[c], 0.0) for c in cand])
    cols += [
        np.full(n, float(np.log1p(q_count))),
        exact * float(q_count == 1),
        flag_vec * type_vec,
    ]
    return np.column_stack(cols)


def _soft(sim: np.ndarray) -> np.ndarray:
    """단어 유사도 0~1 → 0.6 이하는 0, 1에 가까울수록 1."""
    return np.clip((sim - 0.6) / 0.4, 0.0, 1.0)


def _word_features(q: dict, cand: np.ndarray, index: CatalogIndex) -> list[np.ndarray]:
    """희귀도 가중 단어 일치: 쿼리·후보 단어 커버리지, 마지막 단어, 가장 드문 단어."""
    n = len(cand)
    # 한글 쿼리는 단어 표기가 불안정해(루비 에이스 → Garnet Ace 오답) 비한글 쿼리에만 적용한다
    qw = [] if q["hangul"] else q["words"]
    cws = [index.words[c] for c in cand]
    lens = np.array([len(w) for w in cws])
    if not qw or lens.sum() == 0:
        zeros = np.zeros(n, dtype=np.float32)
        return [zeros, zeros, zeros, zeros, zeros]
    flat = [w for ws in cws for w in ws]
    sim = _soft(process.cdist(qw, flat, scorer=fuzz.ratio, dtype=np.float32, workers=1) / 100.0)
    q_idf = np.array([index.idf.get(w, index.max_idf) for w in qw], dtype=np.float32)
    rarest = int(np.argmax(q_idf))
    ends = np.cumsum(lens)
    starts = ends - lens
    cover_q = np.zeros(n, dtype=np.float32)
    cover_c = np.zeros(n, dtype=np.float32)
    last = np.zeros(n, dtype=np.float32)
    rare = np.zeros(n, dtype=np.float32)
    for k in range(n):
        if lens[k] == 0:
            continue
        block = sim[:, starts[k] : ends[k]]
        best_q = block.max(axis=1)
        best_c = block.max(axis=0)
        c_idf = index.word_idf[cand[k]]
        cover_q[k] = float((q_idf * best_q).sum() / max(q_idf.sum(), 1e-6))
        cover_c[k] = float((c_idf * best_c).sum() / max(c_idf.sum(), 1e-6))
        last[k] = block[-1, -1]
        rare[k] = best_q[rarest]
    return [cover_q, cover_c, last, rare, cover_q - cover_q.max()]


# ---------------------------------------------------------------- 예측

def predict(
    model, frame: pd.DataFrame, index: CatalogIndex, res: dict, use_callsign: bool = False,
    candidates_out: list | None = None, use_name_lookup: bool = False,
) -> list[str]:
    """모델 예측. use_name_lookup이면 train에 같은 선박명이 있을 때 그 정답으로,
    use_callsign이면 train에 같은 호출부호가 있을 때 그 정답으로 덮어쓴다(호출부호 우선)."""
    rows = frame.to_dict("records")
    queries = [prepare_query(r, res) for r in rows]
    cand_lists, features = candidates_and_features(queries, index, res)
    if candidates_out is not None:
        candidates_out.extend(cand_lists)
    scores = model.predict(features)
    out: list[str] = []
    offset = 0
    for row, cand in zip(rows, cand_lists):
        local = scores[offset : offset + len(cand)]
        offset += len(cand)
        best = index.ids[cand[int(np.argmax(local))]]
        if use_name_lookup:
            best = lookup_by_name(row, res, index) or best
        if use_callsign:
            best = res["callsign"].get(row.get("call_sign"), best)
        out.append(str(best))
    return out
