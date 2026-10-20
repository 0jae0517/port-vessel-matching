"""스트레스 검증: train에서 관찰한 노이즈 규칙을 더 세게 걸어 모델 약점을 찾는다.

EDA(analysis/eda_noise.py) 결과
  - 선박명: 편집 0~2회, 연산은 거의 삽입(옆 글자 중복)·삭제·자리바꿈, 첫/마지막 단어 하나만 변형
  - 톤수: 52.6% 정확 일치, 나머지 대부분 ±3~5% 곱셈 오차, 6.5% 결측
  - 국적: 16.5% 결측, 대표 영문 국적과 77% 일치

vessel 단위 fold 0 하나로 학습(4/5 선박)하고, 검증 선박(1/5)의 쿼리에 시나리오별 추가 노이즈를 건다.
설정: v11(임베딩 없음) · v14(임베딩 e5-small+LaBSE, 영문 포함) · v15(임베딩 한글 전용)
  실행: .venv\\Scripts\\python.exe analysis\\stress_test.py [v11 v14 v15]
"""
from __future__ import annotations

import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

HANGUL = re.compile(r"[가-힣]")


from augment import apply, combo, drop_word, flag_missing, name_edits, ton_missing, ton_noise  # noqa: E402


SCENARIOS = {
    "원본": None,
    "이름 편집 +1": name_edits(1),
    "이름 편집 +2": name_edits(2),
    "이름 편집 +4": name_edits(4),
    "단어 하나 삭제": drop_word,
    "톤수 결측": ton_missing,
    "톤수 오차 5~10%": ton_noise(0.05, 0.10),
    "톤수 오차 10~20%": ton_noise(0.10, 0.20),
    "국적 결측": flag_missing,
    "편집+2 & 톤수 결측": combo(name_edits(2), ton_missing),
    "편집+2 & 톤수·국적 결측": combo(name_edits(2), ton_missing, flag_missing),
}


# ---------------------------------------------------------------- 실행

def main() -> None:
    configs = sys.argv[1:] or ["v11", "v14", "v15"]
    T.NO_DICT = True
    train, catalog = T.load()
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=T.SEED)
    tr_idx, va_idx = next(splitter.split(train, hangul, train["target_vessel_id"]))
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    print(f"학습 {len(tr)}행 · 검증 {len(va)}행 (한글 {va['vessel_name'].map(ml.is_hangul).sum()})", flush=True)

    noisy = {name: apply(va, fn) for name, fn in SCENARIOS.items()}
    ex = va[va["vessel_name"].map(ml.is_hangul)].head(1).index.tolist() + va.head(2).index.tolist()
    for name in ("이름 편집 +2", "단어 하나 삭제"):
        print(f"  예시 [{name}]:", [f"{va.loc[i, 'vessel_name']} → {noisy[name].loc[i, 'vessel_name']}" for i in ex])

    plain = ml.CatalogIndex(catalog)
    emb_indexes: dict[tuple, ml.CatalogIndex] = {}
    results = []
    for cfg in configs:
        t0 = time.time()
        if cfg == "v11":
            index = plain
        else:
            # v20: 파인튜닝 e5(fold 0 학습 선박으로만 학습한 e5-small-ft0) + LaBSE
            names = ("e5-small-ft0", "labse") if cfg in ("v20", "v21") else ("e5-small", "labse")
            if names not in emb_indexes:
                emb_indexes[names] = ml.CatalogIndex(catalog)
                for name in names:
                    T.attach_embedding(emb_indexes[names], name)
            index = emb_indexes[names]
        # v14: 임베딩(영문 포함) / v15: 임베딩 한글 전용 / v16: v14 + 노이즈 증강 + 단어 단위 검색
        # v16a: v14 + 노이즈 증강만 / v16w: v14 + 단어 단위 검색만 / v20: v16 + 파인튜닝 e5
        T.EMB_ALL = cfg in ("v14", "v16", "v16a", "v16w", "v20", "v21")
        T.AUGMENT = 1 if cfg in ("v16", "v16a", "v20", "v21") else 0
        T.WORD_INDEX = cfg in ("v16", "v16w", "v20", "v21")
        # v21: v20 + 교차 학습 (LightGBM 학습 피처는 그 선박을 학습하지 않은 절반 모델로 계산)
        T.CROSS_INDEXES = T.attach_cross({0: "e5-small-ft0h0", 1: "e5-small-ft0h1"}, ["labse"]) if cfg == "v21" else None
        model, _ = T.fit(tr, catalog, index)
        T.CROSS_INDEXES = None
        res = T.resources(tr, catalog)
        for scen, frame in noisy.items():
            pred = ml.predict(model, frame, index, res)
            s = T.score(va, pred)
            results.append({"설정": cfg, "시나리오": scen, **{k: round(float(v), 4) for k, v in s.items()}})
            print(f"  [{cfg}] {scen:22s} weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} "
                  f"non_hangul={s['non_hangul']:.4f}", flush=True)
        print(f"[{cfg}] 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    df = pd.DataFrame(results)
    out = ROOT / "outputs" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stress_test.csv", index=False, encoding="utf-8-sig")
    for metric in ("weighted", "hangul", "non_hangul"):
        print(f"\n===== {metric} =====")
        print(df.pivot(index="시나리오", columns="설정", values=metric).reindex(list(SCENARIOS)).to_string())


if __name__ == "__main__":
    main()
