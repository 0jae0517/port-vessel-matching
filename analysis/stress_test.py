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

    # v22~: LLM 영문 복원 결과를 임베딩 모델을 올리기 전에 미리 계산 (GPU 메모리 8GB)
    # v22·v23·v24: Qwen 3B LoRA / v25·v26: kanana LoRA (fold 0 학습 선박으로만 학습)
    LLMS = {"qwen": ("qwen3b-lora0-light", {0: "qwen3b-lora0h0", 1: "qwen3b-lora0h1"}),
            "kanana": ("kanana-lora0-light", {0: "kanana-lora0h0", 1: "kanana-lora0h1"})}
    llm_of = {"v22": "qwen", "v23": "qwen", "v24": "qwen", "v25": "kanana", "v26": "kanana"}
    for kind in dict.fromkeys(llm_of[c] for c in configs if c in llm_of):
        full, halves = LLMS[kind]
        T.AUGMENT = 1
        T.warm_cross_llm(tr, halves)
        T.LLMCache(full).warm([str(n) for f in noisy.values() for n in f["vessel_name"] if ml.is_hangul(n)])
    details = []

    plain = ml.CatalogIndex(catalog)
    emb_indexes: dict[tuple, ml.CatalogIndex] = {}
    results = []
    for cfg in configs:
        t0 = time.time()
        if cfg == "v11":
            index = plain
        else:
            # v20: 파인튜닝 e5(fold 0 학습 선박으로만 학습한 e5-small-ft0) + LaBSE
            # v23: e5 파인튜닝에 단어 삭제 노이즈 강화 (--drop-word 0.3) → e5-small-ft0w
            names = (("e5-small-ft0w", "labse") if cfg == "v23" else
                     ("e5-small-ft0", "labse") if cfg in ("v20", "v21", "v22", "v24", "v25", "v26")
                     else ("e5-small", "labse"))
            if names not in emb_indexes:
                emb_indexes[names] = ml.CatalogIndex(catalog)
                for name in names:
                    T.attach_embedding(emb_indexes[names], name)
            index = emb_indexes[names]
        # v14: 임베딩(영문 포함) / v15: 임베딩 한글 전용 / v16: v14 + 노이즈 증강 + 단어 단위 검색
        # v16a: v14 + 노이즈 증강만 / v16w: v14 + 단어 단위 검색만 / v20: v16 + 파인튜닝 e5
        late = ("v20", "v21", "v22", "v23", "v24", "v25", "v26")
        T.EMB_ALL = cfg in ("v14", "v16", "v16a", "v16w", *late)
        T.AUGMENT = 1 if cfg in ("v16", "v16a", *late) else 0
        T.WORD_INDEX = cfg in ("v16", "v16w", *late)
        # v21: v20 + 교차 학습 (LightGBM 학습 피처는 그 선박을 학습하지 않은 절반 모델로 계산)
        # v22: v21 + LoRA LLM 영문 복원 후보·피처 (LLM도 교차 학습) / v24·v25: + 병음 역변환
        use_llm = cfg in llm_of
        llm_full, llm_halves = LLMS[llm_of[cfg]] if use_llm else (None, None)
        T.USE_LLM = use_llm
        T.USE_PINYIN = cfg in ("v24", "v25")
        T.CROSS_INDEXES = None
        if cfg in ("v21", *llm_of):
            halves = {0: "e5-small-ft0wh0", 1: "e5-small-ft0wh1"} if cfg == "v23" else {0: "e5-small-ft0h0", 1: "e5-small-ft0h1"}
            T.CROSS_INDEXES = T.attach_cross(halves, ["labse"], llm_halves if use_llm else None)
        index.llm = T.LLMCache(llm_full) if use_llm else None
        model, _ = T.fit(tr, catalog, index)
        T.CROSS_INDEXES = None
        res = T.resources(tr, catalog)
        for scen, frame in noisy.items():
            pred, rows = diagnose(model, frame, va, index, res)
            s = T.score(va, pred)
            d = pd.DataFrame(rows)
            wrong = d[~d["correct"]]
            miss = (~wrong["in_cand"]).sum()
            results.append({"설정": cfg, "시나리오": scen, **{k: round(float(v), 4) for k, v in s.items()},
                            "오답": len(wrong), "후보누락": int(miss), "순위오류": int(len(wrong) - miss),
                            "한글 후보누락": int((~wrong["in_cand"] & wrong["hangul"]).sum()),
                            "한글 순위오류": int((wrong["in_cand"] & wrong["hangul"]).sum()),
                            "정답 2위": int((wrong["rank"] == 1).sum()), "정답 3~5위": int(wrong["rank"].between(2, 4).sum())})
            details += [{"설정": cfg, "시나리오": scen, **r} for r in rows]
            print(f"  [{cfg}] {scen:22s} weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} "
                  f"non_hangul={s['non_hangul']:.4f} | 오답 {len(wrong)} = 후보누락 {miss} + 순위오류 {len(wrong) - miss}",
                  flush=True)
        print(f"[{cfg}] 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    df = pd.DataFrame(results)
    out = ROOT / "outputs" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"stress_test_{'_'.join(configs)}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(out / f"stress_diag_{'_'.join(configs)}.csv", index=False, encoding="utf-8-sig")
    for metric in ("weighted", "hangul", "non_hangul"):
        print(f"\n===== {metric} =====")
        print(df.pivot(index="시나리오", columns="설정", values=metric).reindex(list(SCENARIOS)).to_string())
    print("\n===== 오답 분해 (후보누락 = 정답이 후보에 없음 / 순위오류 = 후보에 있지만 1위가 아님) =====")
    cols = ["설정", "시나리오", "오답", "후보누락", "순위오류", "한글 후보누락", "한글 순위오류", "정답 2위", "정답 3~5위"]
    print(df[cols].to_string(index=False))


def diagnose(model, frame: pd.DataFrame, truth: pd.DataFrame, index: ml.CatalogIndex, res: dict):
    """ml.predict 와 같은 예측 + 행마다 정답이 후보에 있었는지, 랭커 점수로 몇 위였는지."""
    rows_in = frame.to_dict("records")
    queries = [ml.prepare_query(r, res) for r in rows_in]
    cands, feats = ml.candidates_and_features(queries, index, res)
    scores = model.predict(feats)
    names = index.catalog["vessel_name"].astype(str).to_numpy()
    pred, rows, offset = [], [], 0
    for r, q, cand, tid in zip(rows_in, queries, cands, truth["target_vessel_id"]):
        local = scores[offset : offset + len(cand)]
        offset += len(cand)
        order = np.argsort(-local, kind="mergesort")
        t = index.id_to_index[tid]
        pos = np.flatnonzero(cand == t)
        rank = int((local > local[pos[0]]).sum()) if len(pos) else -1
        top = cand[order[0]]
        pred.append(index.ids[top])
        rows.append({
            "sample_id": r["sample_id"], "hangul": bool(q["hangul"]), "query": r["vessel_name"],
            "truth": names[t], "pred": names[top], "correct": bool(top == t), "in_cand": bool(len(pos)),
            "rank": rank, "n_cand": len(cand), "margin": float(local[order[0]] - local[order[1]]) if len(cand) > 1 else 0.0,
            "top5": " | ".join(names[cand[order[:5]]]),
            # LLM 재정렬 프로토타입용: 상위 10개 후보 id·랭커 점수
            "top10_ids": " ".join(index.ids[cand[order[:10]]]),
            "top10_scores": " ".join(f"{v:.4f}" for v in local[order[:10]]),
            "truth_id": tid,
        })
    return pred, rows


if __name__ == "__main__":
    main()
