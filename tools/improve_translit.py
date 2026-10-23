"""카탈로그 한글 표기표의 '뜻 번역' 고치기 (GPU 불필요).

LLM(kanana)이 흔한 영어 단어를 뜻으로 번역하는 경우가 남아 있다 (Queen → 여왕, General → 장군).
같은 단어가 다른 선박명에서는 소리대로 적힌 경우가 많으므로,
  1) 표기표와 train 한글 기록에서 '영어 단어 → 한글 소리 표기' 사전을 만들고 (단어 수가 맞는 쌍만 정렬)
  2) 소리가 안 맞는 단어(번역으로 의심)를 사전 표기로 바꾼 변형을 표기 목록 맨 앞에 추가한다
원래 표기는 지우지 않는다 (비교 때 변형 중 최고점을 쓰므로 손해 없음).

  실행: .venv\\Scripts\\python.exe tools\\improve_translit.py
  결과: outputs/translit/kanana2b_fixed.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import match_lib as ml  # noqa: E402

GOOD, BAD = 60, 45  # 단어 소리 유사도: 사전에 넣을 기준 / 번역으로 의심할 기준


def word_sound(en: str, ko: str) -> float:
    """영어 단어 하나와 한글 단어 하나의 소리 유사도 (한글 → 로마자 변형 중 최고)."""
    if not ml.HANGUL_RE.search(ko):
        return 100.0 if en == ko else 0.0
    return max(fuzz.ratio(en, v) for v in ml.romanize_variants(ko))


def as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def main() -> None:
    src = ROOT / "outputs" / "translit" / "kanana2b.json"
    table: dict[str, str | list[str]] = json.loads(src.read_text(encoding="utf-8"))
    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")

    # 1) 단어 사전: (영어 단어, 한글 단어) 정렬 쌍 중 소리가 맞는 것만 센다. train 실제 기록은 가중치 5
    counts: dict[str, Counter] = defaultdict(Counter)

    def align(en_name: str, ko_name: str, weight: int) -> None:
        en_t, ko_t = ml.tokenize(en_name), ml.tokenize(ko_name)
        if len(en_t) != len(ko_t):
            return
        for e, k in zip(en_t, ko_t):
            if e.isalpha() and len(e) >= 2 and ml.HANGUL_RE.search(k) and word_sound(e, k) >= GOOD:
                counts[e][k] += weight

    for en, v in table.items():
        for ko in as_list(v):
            align(en, ko, 1)
    name_of = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str)))
    h = train[train["vessel_name"].map(ml.is_hangul)].drop_duplicates(["vessel_name", "target_vessel_id"])
    for q, t in zip(h["vessel_name"], h["target_vessel_id"]):
        align(name_of[t], str(q), 5)
    vocab = {e: c.most_common(1)[0][0] for e, c in counts.items() if sum(c.values()) >= 2}
    print(f"단어 소리 사전 {len(vocab)}개 (예: QUEEN={vocab.get('QUEEN')}, GENERAL={vocab.get('GENERAL')}, "
          f"LUCKY={vocab.get('LUCKY')}, WISDOM={vocab.get('WISDOM')}, SUCCESS={vocab.get('SUCCESS')})")

    # 2) 표기마다 번역 의심 단어를 사전 표기로 바꾼 변형 추가
    fixed, n_fix, examples = {}, 0, []
    for en, v in table.items():
        variants = as_list(v)
        en_t = ml.tokenize(en)
        new = []
        for ko in variants:
            ko_t = ml.tokenize(ko)
            if len(ko_t) == len(en_t):
                out, changed = [], False
                for e, k in zip(en_t, ko_t):
                    if e.isalpha() and e in vocab and word_sound(e, k) < BAD:
                        out.append(vocab[e])
                        changed = True
                    else:
                        out.append(k)
                if changed:
                    new.append(" ".join(out))
        # 단어 수가 안 맞으면(띄어쓰기 없이 붙인 '써니여왕호' 등) 모든 영어 단어가 사전에 있을 때 사전 표기로 조립
        best = max((sum(word_sound(e, k) for e, k in zip(en_t, ml.tokenize(ko))) / max(len(en_t), 1)
                    if len(ml.tokenize(ko)) == len(en_t) else 0.0) for ko in variants) if variants else 0.0
        if not new and best < GOOD and en_t and all(t.isdigit() or t in vocab for t in en_t):
            new.append(" ".join(t if t.isdigit() else vocab[t] for t in en_t))
        new = [x for x in dict.fromkeys(new) if x not in variants]
        if new:
            n_fix += 1
            if len(examples) < 25:
                examples.append((en, variants[:2], new[0]))
        fixed[en] = (new + variants)[:4] if new else v
    out = ROOT / "outputs" / "translit" / "kanana2b_fixed.json"
    out.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
    print(f"고친 이름 {n_fix}개 / {len(table)} → {out}")
    for en, old, new in examples:
        print(f"  {en:28s} {str(old):40s} → {new}")
    for en in ["Sunny Queen", "Southern Queen", "General", "New Wisdom", "Winter Sun", "Good Treasure", "Lucky Joy"]:
        print(f"  [확인] {en:20s} {fixed.get(en)}")


if __name__ == "__main__":
    main()
