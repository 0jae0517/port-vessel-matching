"""중국어 병음(pinyin) 선박명 판별 + 외래어 표기법(중국어) 규칙으로 한글 변환.

  is_pinyin_name("Xin Wen Zhou") → True
  to_hangul("Xin Wen Zhou")      → "신 원 저우"
"""
from __future__ import annotations

import re
from functools import lru_cache

CHO = {"": 11, "b": 7, "p": 17, "m": 6, "f": 17, "d": 3, "t": 16, "n": 2, "l": 5, "g": 0, "k": 15, "h": 18,
       "j": 12, "q": 14, "x": 9, "zh": 12, "ch": 14, "sh": 9, "r": 5, "z": 13, "c": 14, "s": 10}
# 운모 → 음절별 (중성, 종성). 첫 음절에 성모를 붙이고 나머지는 ㅇ 초성
FIN = {
    "a": [(0, 0)], "o": [(8, 0)], "e": [(4, 0)], "ai": [(0, 0), (20, 0)], "ei": [(5, 0), (20, 0)],
    "ao": [(0, 0), (8, 0)], "ou": [(4, 0), (13, 0)], "an": [(0, 4)], "en": [(4, 4)], "ang": [(0, 21)],
    "eng": [(4, 21)], "ong": [(13, 21)], "er": [(4, 8)],
    "i": [(20, 0)], "ia": [(2, 0)], "ie": [(7, 0)], "iao": [(2, 0), (8, 0)], "iu": [(17, 0)], "ian": [(7, 4)],
    "in": [(20, 4)], "iang": [(2, 21)], "ing": [(20, 21)], "iong": [(17, 21)],
    "u": [(13, 0)], "ua": [(9, 0)], "uo": [(14, 0)], "uai": [(9, 0), (20, 0)], "ui": [(13, 0), (20, 0)],
    "un": [(13, 4)], "uan": [(9, 4)], "uang": [(9, 21)],
    "v": [(16, 0)], "ve": [(15, 0)], "van": [(16, 0), (0, 4)], "vn": [(16, 4)],
}
Y = {"yi": "i", "ya": "ia", "ye": "ie", "yao": "iao", "you": "iu", "yan": "ian", "yin": "in", "yang": "iang",
     "ying": "ing", "yong": "iong", "yu": "v", "yue": "ve", "yuan": "van", "yun": "vn"}
W = {"wu": [(13, 0)], "wa": [(9, 0)], "wo": [(14, 0)], "wai": [(9, 0), (20, 0)], "wei": [(15, 0), (20, 0)],
     "wan": [(9, 4)], "wen": [(14, 4)], "wang": [(9, 21)], "weng": [(14, 21)]}
BUZZ = {"zhi", "chi", "shi", "ri", "zi", "ci", "si"}  # 'ㅡ' 모음
SOFT = {12, 13, 14}  # ㅈ ㅉ ㅊ 다음 ㅑㅖㅛㅠ → ㅏㅔㅗㅜ
SOFTEN = {2: 0, 7: 5, 12: 8, 17: 13}
INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]


def _syllables() -> dict[str, list[tuple[int, int, int]]]:
    out: dict[str, list[tuple[int, int, int]]] = {}
    for ini in INITIALS:
        for fin, sy in FIN.items():
            if fin.startswith("v"):
                continue
            key = ini + fin
            if ini in ("j", "q", "x") and fin in ("u", "ue", "uan", "un"):
                sy = FIN[{"u": "v", "ue": "ve", "uan": "van", "un": "vn"}[fin]]
            if ini in ("j", "q", "x") and fin.startswith(("a", "o", "e")) and fin != "e":
                continue  # ja, qo … 는 병음에 없음
            out[key] = [(CHO[ini], *sy[0]), *[(11, *s) for s in sy[1:]]]
        for fin in ("ue", "uan", "un"):
            if ini in ("j", "q", "x"):
                sy = FIN[{"ue": "ve", "uan": "van", "un": "vn"}[fin]]
                out[ini + fin] = [(CHO[ini], *sy[0]), *[(11, *s) for s in sy[1:]]]
    for b in BUZZ:
        ini = b[:-1]
        out[b] = [(CHO[ini], 18, 0)]
    for fin in ("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er"):
        out[fin] = [(11, *s) for s in FIN[fin]]
    for k, fin in Y.items():
        out[k] = [(11, *s) for s in FIN[fin]]
    for k, sy in W.items():
        out[k] = [(11, *s) for s in sy]
    return out


SYL = _syllables()
MAXLEN = max(map(len, SYL))
DISTINCT = re.compile(r"^(x|q|zh)|iu|ui|uo|iao|iang|iong|ian|ong|eng|ou")


@lru_cache(maxsize=None)
def split(word: str) -> tuple[str, ...] | None:
    """병음 음절로 나누기 (음절 수 최소). 안 되면 None."""
    word = word.lower()
    best: list[tuple[str, ...] | None] = [None] * (len(word) + 1)
    best[0] = ()
    for i in range(len(word)):
        if best[i] is None:
            continue
        for j in range(i + 1, min(len(word), i + MAXLEN) + 1):
            if word[i:j] in SYL and (best[j] is None or len(best[j]) > len(best[i]) + 1):
                best[j] = best[i] + (word[i:j],)
    return best[-1]


def alpha_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z]+", name) if t.upper() not in {"NO", "I", "II", "III", "IV", "V", "VI",
                                                                           "VII", "VIII", "IX", "X", "XI", "XII"}]


def is_pinyin_name(name: str) -> bool:
    toks = alpha_tokens(str(name))
    if not toks or any(split(t) is None for t in toks):
        return False
    n_syl = sum(len(split(t)) for t in toks)
    distinct = sum(bool(DISTINCT.search(s)) for t in toks for s in split(t))
    return n_syl >= 2 and distinct >= max(1, n_syl // 3)


def _compose(cho: int, jung: int, jong: int) -> str:
    if cho in SOFT:
        jung = SOFTEN.get(jung, jung)
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def to_hangul(name: str) -> str:
    """병음 토큰은 규칙 변환, 숫자는 그대로, 나머지 토큰은 버린다."""
    out = []
    for tok in re.findall(r"[A-Za-z]+|\d+", str(name)):
        if tok.isdigit():
            out.append(tok)
            continue
        parts = split(tok)
        if parts:
            out.append("".join(_compose(*j) for s in parts for j in SYL[s]))
    return " ".join(out)


if __name__ == "__main__":
    for n in ["Xin Wen Zhou", "Feng Xiu Hai", "Hui Zhi Hai", "Zhong Gu Shan Dong", "Yuan Zhen Hai", "Xiamen",
              "Qingdao", "Tianjin", "Jiangsu", "Zhou Shan", "Sun Long", "Golden Star", "Hongxing 8"]:
        print(n, is_pinyin_name(n), to_hangul(n))
