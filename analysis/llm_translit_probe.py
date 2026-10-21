"""소형 LLM으로 한글 선박명 → 영문 철자 복원 가능성 확인 (학습 없음, few-shot).

fold 0 검증 선박의 한글 이름(이름·정답 중복 제거)을 LLM으로 영문으로 바꾸고,
그 영문만으로 카탈로그를 글자 유사도(fuzz.ratio) 검색해 정답 순위를 잰다.
비교: 파인튜닝 e5(e5-small-ft0) 임베딩 단독, 둘의 상호 보완(합집합) 정도.

  실행: .venv\\Scripts\\python.exe analysis\\llm_translit_probe.py [qwen1.5b|qwen3b] [--beams 4]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import augment as A  # noqa: E402
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

from llm_lib import Transliterator  # noqa: E402


def compact(name: str) -> str:
    return "".join(ml.tokenize(name))


def few_shot(train: pd.DataFrame, catalog: pd.DataFrame, n_shot: int) -> list[tuple[str, str]]:
    """fold 0 학습 선박에서만 예시를 뽑는다 (검증 선박 정답 누수 없음)."""
    name_of = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str)))
    h = train[train["vessel_name"].map(ml.is_hangul)].drop_duplicates("target_vessel_id")
    pairs = [(str(q), name_of[t]) for q, t in zip(h["vessel_name"], h["target_vessel_id"])
             if not ml.is_hangul(name_of[t]) and len(name_of[t]) <= 30]
    return random.Random(0).sample(pairs, n_shot) if n_shot else []


def ranks_from_scores(S: np.ndarray, truth: np.ndarray) -> np.ndarray:
    true_s = S[np.arange(len(truth)), truth][:, None]
    return (S > true_s).sum(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="qwen3b")
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--shots", type=int, default=12, help="few-shot 예시 수 (LoRA 미세조정 모델은 0)")
    args = p.parse_args()

    train, catalog = T.load()
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=T.SEED)
    tr_idx, va_idx = next(splitter.split(train, hangul, train["target_vessel_id"]))
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    hv = va[va["vessel_name"].map(ml.is_hangul)].drop_duplicates(["vessel_name", "target_vessel_id"])
    cat_index = {v: i for i, v in enumerate(catalog["vessel_id"])}
    names = hv["vessel_name"].astype(str).tolist()
    truth = np.array([cat_index[v] for v in hv["target_vessel_id"]])
    cat_compact = [compact(n) for n in catalog["vessel_name"].astype(str)]
    print(f"한글 검증 이름 {len(names)}개", flush=True)

    llm = Transliterator(ROOT / "models" / args.model, beams=args.beams, shots=few_shot(tr, catalog, args.shots))

    index = ml.CatalogIndex(catalog)
    T.attach_embedding(index, "e5-small-ft0")

    scenarios = {"원본": None, "편집+2": A.name_edits(2), "단어 삭제": A.drop_word}
    report, samples = {}, []
    for label, fn in scenarios.items():
        qs = [q if fn is None else fn({"vessel_name": q}, A.row_rng(i, label))["vessel_name"] for i, q in enumerate(names)]
        t0 = time.time()
        outs = [llm(q) for q in qs]
        secs = (time.time() - t0) / len(qs)
        # 빔 결과마다 카탈로그 글자 유사도 → 선박별 최댓값
        S_llm = np.zeros((len(qs), len(catalog)), dtype=np.float32)
        S_top = np.zeros_like(S_llm)
        for b in range(args.beams):
            sc = process.cdist([compact(o[b]) for o in outs], cat_compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
            S_llm = np.maximum(S_llm, sc)
            if b == 0:
                S_top = sc.astype(np.float32)
        r_llm, r_top = ranks_from_scores(S_llm, truth), ranks_from_scores(S_top, truth)
        r_emb = ranks_from_scores(index.query_emb(qs) @ index.cat_emb.T, truth)
        report[label] = {
            "LLM 1빔 top1": round(float((r_top == 0).mean()), 3),
            "LLM 빔합 top1": round(float((r_llm == 0).mean()), 3),
            "LLM 빔합 top10": round(float((r_llm < 10).mean()), 3),
            "e5-ft top1": round(float((r_emb == 0).mean()), 3),
            "e5-ft top10": round(float((r_emb < 10).mean()), 3),
            "LLM만 맞힘(top1)": int(((r_llm == 0) & (r_emb != 0)).sum()),
            "e5만 맞힘(top1)": int(((r_llm != 0) & (r_emb == 0)).sum()),
            "합집합 top10": round(float(((r_llm < 10) | (r_emb < 10)).mean()), 3),
            "행당 초": round(secs, 2),
        }
        print(label, json.dumps(report[label], ensure_ascii=False), flush=True)
        name_of = catalog["vessel_name"].astype(str).tolist()
        for q, o, t, r in zip(qs, outs, truth, r_llm):
            samples.append({"시나리오": label, "쿼리": q, "LLM": " | ".join(o), "정답": name_of[t], "LLM순위": int(r)})

    out = ROOT / "outputs" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_csv(out / f"llm_probe_{args.model}.csv", index=False, encoding="utf-8-sig")
    print(f"\n예시 저장: outputs/cv/llm_probe_{args.model}.csv")


if __name__ == "__main__":
    main()
