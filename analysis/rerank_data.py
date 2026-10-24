"""LLM 재정렬 학습 데이터: train 전체 행에 v28 제출본 그대로 상위 10개 후보를 뽑는다 (실제 train 기록만 사용).

v28 제출 폴더(outputs/submissions/v28/src)의 match_lib·LightGBM·resources 를 그대로 써서
추론 때와 같은 후보·순위를 만든다. train 행은 v28 이 학습에 쓴 행이라 순위는 실제보다 쉬우므로,
재정렬 학습 때는 후보 순서를 섞고 LightGBM 점수는 보여 주지 않는다. 평가는 fold 0 진단(stress_diag, 진짜 OOF)으로 한다.

  실행: .venv\\Scripts\\python.exe analysis\\rerank_data.py
  결과: outputs/rerank/train_cands.csv (sample_id, top10_ids, top10_scores, truth_id, in_top5)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "submissions" / "v28" / "src"
sys.path.insert(0, str(SRC / "model"))
import match_lib as ml  # noqa: E402  (v28 제출본의 match_lib: 피처 정의가 모델과 같다)

sys.path.insert(0, str(ROOT / "solution"))
import train as T  # noqa: E402


def main() -> None:
    train, catalog = T.load()
    model = lgb.Booster(model_str=(SRC / "model" / "lgb_ranker.txt").read_text(encoding="utf-8"))
    with open(SRC / "model" / "resources.pkl", "rb") as f:
        res = pickle.load(f)
    index = ml.CatalogIndex(catalog)
    # LLM 영문 복원: 전체 train 으로 학습한 kanana-lora (추론과 같은 모델), 결과는 캐시
    llm = T.LLMCache("kanana-lora")
    llm.warm([str(n) for n in train["vessel_name"] if ml.is_hangul(n)])
    index.attach_llm(llm)
    if res.get("use_khan"):
        index.attach_catalog_hangul(json.loads((SRC / "model" / "catalog_hangul.json").read_text(encoding="utf-8")))
    for name in ("e5-small-ft", "labse"):
        T.attach_embedding(index, name)

    rows = train.to_dict("records")
    out, t0 = [], time.time()
    for s in range(0, len(rows), 500):
        part = rows[s : s + 500]
        queries = [ml.prepare_query(r, res) for r in part]
        cands, feats = ml.candidates_and_features(queries, index, res)
        scores = model.predict(feats)
        off = 0
        for r, cand in zip(part, cands):
            local = scores[off : off + len(cand)]
            off += len(cand)
            order = np.argsort(-local, kind="mergesort")[:10]
            ids = index.ids[cand[order]]
            out.append({"sample_id": r["sample_id"], "hangul": ml.is_hangul(r["vessel_name"]),
                        "top10_ids": " ".join(ids), "top10_scores": " ".join(f"{v:.4f}" for v in local[order]),
                        "truth_id": r["target_vessel_id"], "in_top5": r["target_vessel_id"] in set(ids[:5]),
                        "top1_ok": ids[0] == r["target_vessel_id"]})
        print(f"  {s + len(part)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    dst = ROOT / "outputs" / "rerank" / "train_cands.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"저장: {dst} ({len(df)}행) top1 {df.top1_ok.mean():.4f}, 정답 상위5 안 {df.in_top5.mean():.4f}")


if __name__ == "__main__":
    main()
