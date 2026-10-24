"""LLM 재정렬 평가 데이터: fold 0 검증 선박의 v28 진짜 OOF 후보 (stress_diag_v25_v28.csv) 중
현실적인 시나리오(원본, 톤수 결측·오차, 국적 결측)만 골라 노이즈가 걸린 기록과 함께 저장.

  실행: .venv\\Scripts\\python.exe analysis\\rerank_eval_data.py
  결과: outputs/rerank/fold0_diag.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "analysis"))
import stress_test as S  # noqa: E402
from augment import apply  # noqa: E402

KEEP = ["원본", "톤수 결측", "톤수 오차 5~10%", "톤수 오차 10~20%", "국적 결측", "이름 편집 +1", "이름 편집 +2"]
# 평가 전용 (학습에는 쓰지 않음): 원본류는 오답이 적어 차이를 보기 어려워 오타 시나리오도 함께 본다


def main() -> None:
    d = pd.read_csv(ROOT / "outputs" / "cv" / "stress_diag_v25_v28.csv", encoding="utf-8-sig")
    d = d[(d["설정"] == "v28") & d["시나리오"].isin(KEEP)].copy()
    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    va = train[train["sample_id"].isin(set(d["sample_id"]))]
    frames = {s: apply(va, S.SCENARIOS[s]).set_index("sample_id") for s in KEEP}
    clean = lambda v: None if isinstance(v, float) and np.isnan(v) else v
    d["rec"] = [json.dumps({k: clean(v) for k, v in frames[s].loc[i].to_dict().items()}, ensure_ascii=False, default=str)
                for s, i in zip(d["시나리오"], d["sample_id"])]
    d["key"] = d["시나리오"] + "|" + d["sample_id"]
    out = ROOT / "outputs" / "rerank" / "fold0_diag.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d[["key", "시나리오", "hangul", "top10_ids", "top10_scores", "truth_id", "rec"]].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(d)}행)")
    print(d.groupby("시나리오")["correct"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
