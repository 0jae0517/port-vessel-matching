"""vessel 단위 CV 오답이 '그 fold의 train에 있던 다른 선박'을 고른 경우인지 확인.

test 선박이 train과 겹치지 않는다면, train 정답 선박은 test의 정답이 될 수 없다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import match_lib as ml  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "v11_nodict"  # outputs/cv/<tag>_oof.csv
oof = pd.read_csv(ROOT / "outputs" / "cv" / f"{tag}_oof.csv", encoding="utf-8-sig")
hangul = oof["vessel_name"].map(ml.is_hangul).to_numpy(bool)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)  # train.py와 동일
known = np.zeros(len(oof), dtype=bool)
for tr_idx, va_idx in splitter.split(oof, hangul, oof["target_vessel_id"]):
    train_ids = set(oof["target_vessel_id"].iloc[tr_idx])
    known[va_idx] = oof["pred_vessel_id"].iloc[va_idx].isin(train_ids).to_numpy()

wrong = ~oof["correct"].to_numpy()
print(f"오답 {wrong.sum()}건 중 fold train 선박을 고른 오답 {(wrong & known).sum()}건 "
      f"(한글 {(wrong & known & hangul).sum()}, 비한글 {(wrong & known & ~hangul).sum()})")
print(f"정답 중 train 선박 예측(있으면 안 됨): {(~wrong & known).sum()}건")
h = (~wrong | known)[hangul].mean()
n = (~wrong | known)[~hangul].mean()
print(f"train 선박 오답을 모두 고칠 수 있다면 상한: weighted={0.7*h+0.3*n:.4f} (hangul {h:.4f}, non {n:.4f})")
