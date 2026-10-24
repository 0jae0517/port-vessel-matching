"""LLM 재정렬 재조정 (v34): 저장된 평가 점수(outputs/rerank/eval_lp_k*.npz)로 후보 수·적용 범위(gate)·가중치(w)를 고른다.

GPU 없이 몇 초. tools/rerank_train.py --eval --k K 가 만든 npz 를 읽는다.
  - 후보 수 k: K 모델의 로그확률을 앞 k 개(LightGBM 순위)만 남겨 다시 정규화해 비교 (k ≤ K)
  - gate: LightGBM 1·2위 점수 차가 gate 미만인 행만 재정렬 (100 = 사실상 전체 행)
  - w: LightGBM 점수 + w × LLM 로그확률 (100 = 사실상 LLM 단독)
  - 선택은 선박 절반씩 교차 (한쪽 절반에서 고르고 다른 절반에서 채점) → 과적합 없는 추정치
최종 점수 = 0.7 × 한글 + 0.3 × 비한글 (대회 가중치)

  .venv\\Scripts\\python.exe tools\\rerank_tune.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
WS = (0.1, 0.25, 0.5, 1, 2, 4, 100)
GATES = (0.5, 1, 2, 3, 5, 100)


def load(path: Path, k: int) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    lp = d["lp"][:, :k]
    d["lp"] = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)  # 앞 k 개로 다시 정규화
    d["lgb"] = d["lgb"][:, :k]
    d["labels"] = np.where(d["labels"] < k, d["labels"], -1)
    return d


def decide(d: dict, mask: np.ndarray, w: float, gate: float) -> np.ndarray:
    lgb, lp = d["lgb"][mask], d["lp"][mask]
    pick = np.zeros(len(lgb), dtype=int)
    rr = (lgb[:, 0] - lgb[:, 1]) < gate
    pick[rr] = np.argmax(lgb[rr] + w * lp[rr], axis=1)
    return pick


def weighted(d: dict, ok: np.ndarray) -> float:
    h = d["hangul"].astype(bool)
    return 0.7 * ok[h].mean() + 0.3 * ok[~h].mean()


def cross(d: dict) -> tuple[np.ndarray, list]:
    """절반 교차로 (w, gate) 선택 → 행별 정답 여부, 고른 값."""
    ok, chosen = np.zeros(len(d["labels"]), dtype=bool), []
    for h in (0, 1):
        tune, test = d["halves"] != h, d["halves"] == h
        wh = d["hangul"].astype(bool)[tune]

        def obj(wg):
            c = decide(d, tune, *wg) == d["labels"][tune]
            return 0.7 * c[wh].mean() + 0.3 * c[~wh].mean()

        best = max(((w, g) for w in WS for g in GATES), key=obj)
        ok[test] = decide(d, test, *best) == d["labels"][test]
        chosen.append(best)
    return ok, chosen


def main() -> None:
    rows, per = [], {}
    for path in sorted((ROOT / "outputs" / "rerank").glob("eval_lp_k*.npz")):
        big = int(path.stem.split("k")[-1])
        for k in sorted({5, big} if big >= 5 else {big}):
            d = load(path, k)
            base = d["labels"] == 0
            ok, chosen = cross(d)
            fixed = decide(d, np.ones(len(base), bool), 0.5, 2) == d["labels"]  # v33 설정
            name = f"모델K{big}·후보{k}"
            rows.append({"설정": name, "LightGBM": weighted(d, base), "v33설정(w0.5,gate2)": weighted(d, fixed),
                         "교차선택": weighted(d, ok), "고른값(half0/1)": chosen})
            per[name] = pd.DataFrame({"시나리오": d["scen"], "hangul": d["hangul"], "lgb": base, "rr": ok})
    if not rows:
        print("outputs/rerank/eval_lp_k*.npz 가 없습니다 (RunPod 결과를 먼저 풀어 주세요)")
        return
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for name, df in per.items():
        print(f"\n[{name}] 시나리오별 (교차선택)")
        print(df.groupby(["시나리오", "hangul"])[["lgb", "rr"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
