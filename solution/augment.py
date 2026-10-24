"""train 쿼리 노이즈 함수 (학습 데이터 증강 · 스트레스 검증 공용).

EDA(analysis/eda_noise.py)에서 관찰한 규칙을 재현한다.
  - 선박명: 옆 글자 중복 삽입 · 삭제 · 인접 자리바꿈, 첫/마지막 단어 변형
  - 톤수: 결측, 곱셈 오차
  - 국적: 결측
행마다 sample_id로 시드를 고정해 재현 가능하다.
"""
from __future__ import annotations

import random
import zlib

import numpy as np
import pandas as pd


def row_rng(sample_id: object, salt: str = "") -> random.Random:
    return random.Random(zlib.crc32(f"{sample_id}{salt}".encode()))


def _edit_once(s: str, rng: random.Random) -> str:
    """옆 글자 중복 삽입 · 삭제 · 인접 자리바꿈 (공백은 건드리지 않음)."""
    pos = [i for i, ch in enumerate(s) if not ch.isspace()]
    if len(pos) < 2:
        return s
    op = rng.choice(["dup", "delete", "swap"])
    i = rng.choice(pos)
    if op == "dup":
        return s[: i + 1] + s[i] + s[i + 1 :]
    if op == "delete":
        return s[:i] + s[i + 1 :]
    j = i + 1 if i + 1 < len(s) and not s[i + 1].isspace() else i - 1
    if j < 0 or s[j].isspace():
        return s
    a, b = sorted((i, j))
    return s[:a] + s[b] + s[a] + s[b + 1 :]


def name_edits(k: int):
    def f(row: dict, rng: random.Random) -> dict:
        s = str(row["vessel_name"])
        for _ in range(k):
            s = _edit_once(s, rng)
        return {**row, "vessel_name": s}
    return f


def drop_word(row: dict, rng: random.Random) -> dict:
    words = str(row["vessel_name"]).split()
    if len(words) >= 2:
        words = words[1:] if rng.random() < 0.5 else words[:-1]
    return {**row, "vessel_name": " ".join(words)}


def ton_missing(row: dict, rng: random.Random) -> dict:
    return {**row, "gross_tonnage": np.nan, "international_gross_tonnage": np.nan}


def ton_noise(lo: float, hi: float):
    def f(row: dict, rng: random.Random) -> dict:
        out = dict(row)
        mult = 1 + rng.choice([-1, 1]) * rng.uniform(lo, hi)
        for key in ("gross_tonnage", "international_gross_tonnage"):
            if pd.notna(row[key]):
                out[key] = round(float(row[key]) * mult)
        return out
    return f


def flag_missing(row: dict, rng: random.Random) -> dict:
    return {**row, "flag_name": np.nan}


def combo(*fns):
    def f(row: dict, rng: random.Random) -> dict:
        for fn in fns:
            row = fn(row, rng)
        return row
    return f


def apply(frame: pd.DataFrame, fn, salt: str = "") -> pd.DataFrame:
    if fn is None:
        return frame
    rows = [fn(r, row_rng(r["sample_id"], salt)) for r in frame.to_dict("records")]
    return pd.DataFrame(rows, columns=frame.columns, index=frame.index)


def substitute_char(row: dict, rng: random.Random) -> dict:
    """글자 하나를 다른 글자로 (train에서는 1.4%로 드묾). 영문은 알파벳, 한글은 원래 글자 유지."""
    s = str(row["vessel_name"])
    pos = [i for i, ch in enumerate(s) if ch.isascii() and ch.isalpha()]
    if not pos:
        return row
    i = rng.choice(pos)
    ch = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return {**row, "vessel_name": s[:i] + ch + s[i + 1 :]}


def respace(row: dict, rng: random.Random) -> dict:
    """띄어쓰기 변화: 두 단어 붙이기 또는 한 단어 쪼개기."""
    words = str(row["vessel_name"]).split()
    if len(words) >= 2 and rng.random() < 0.5:
        i = rng.randrange(len(words) - 1)
        words[i : i + 2] = [words[i] + words[i + 1]]
    else:
        long = [i for i, w in enumerate(words) if len(w) >= 4]
        if long:
            i = rng.choice(long)
            cut = rng.randrange(2, len(words[i]) - 1)
            words[i : i + 1] = [words[i][:cut], words[i][cut:]]
    return {**row, "vessel_name": " ".join(words)}


def swap_words(row: dict, rng: random.Random) -> dict:
    words = str(row["vessel_name"]).split()
    if len(words) >= 2:
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return {**row, "vessel_name": " ".join(words)}


EXTRA_NOISE = False  # True: 치환 · 띄어쓰기 · 단어 순서 노이즈도 섞는다 (v19)
ABBR_NOISE = False  # True: 영문 이름에 약칭형 변형(뒤쪽 잘림, 앞 단어 이니셜)을 섞는다 (v30)


def _latin_name(row: dict) -> bool:
    s = row.get("vessel_name")
    return isinstance(s, str) and bool(s.strip()) and not any("가" <= ch <= "힣" for ch in s)


def truncate_name(row: dict, rng: random.Random) -> dict:
    """약칭: 이름 뒤쪽을 잘라낸다 (TORM GANGA → TORM G, PAGANELLA → PAGAN)."""
    s = str(row["vessel_name"])
    keep = max(3, int(round(len(s) * rng.uniform(0.55, 0.8))))
    return {**row, "vessel_name": s[:keep].rstrip()}


def initial_name(row: dict, rng: random.Random) -> dict:
    """약칭: 마지막이 아닌 단어 하나를 이니셜로 (GARNET EXPRESS → G. EXPRESS)."""
    words = str(row["vessel_name"]).split()
    if len(words) >= 2:
        k = rng.randrange(len(words) - 1)
        words[k] = words[k][0] + "."
    return {**row, "vessel_name": " ".join(words)}


def random_noise(row: dict, rng: random.Random) -> dict:
    """학습 증강용: 행마다 노이즈 조합을 무작위로 고른다 (train보다 거친 쿼리)."""
    row = name_edits(rng.choice([0, 1, 1, 2, 2, 3]))(row, rng)
    if EXTRA_NOISE:
        if rng.random() < 0.15:
            row = substitute_char(row, rng)
        if rng.random() < 0.10:
            row = respace(row, rng)
        if rng.random() < 0.05:
            row = swap_words(row, rng)
    if rng.random() < 0.15:
        row = drop_word(row, rng)
    r = rng.random()
    if r < 0.25:
        row = ton_missing(row, rng)
    elif r < 0.40:
        row = ton_noise(0.05, 0.20)(row, rng)
    if rng.random() < 0.15:
        row = flag_missing(row, rng)
    if ABBR_NOISE and _latin_name(row):  # 기존 노이즈 뒤에 따로 뽑아서, 꺼져 있을 때 결과가 바뀌지 않게
        r = rng.random()
        if r < 0.10:
            row = truncate_name(row, rng)
        elif r < 0.20:
            row = initial_name(row, rng)
    return row


def augment(frame: pd.DataFrame, copies: int) -> pd.DataFrame:
    """노이즈를 건 복사본 copies개를 만든다. 정답(target_vessel_id)은 그대로."""
    parts = []
    for c in range(copies):
        rows = [random_noise(r, row_rng(r["sample_id"], f"aug{c}")) for r in frame.to_dict("records")]
        aug = pd.DataFrame(rows, columns=frame.columns)
        aug["sample_id"] = aug["sample_id"].astype(str) + f"_aug{c}"
        parts.append(aug)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0]
