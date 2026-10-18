"""베이스라인 RandomForest를 vessel 단위 GroupKFold로 검증한다.

같은 target_vessel_id의 행은 한 fold에만 들어가도록 나누고, fold마다 RF를 새로 학습한다.
(rf.pkl은 train 전체로 학습되었을 수 있어 검증에 그대로 쓰지 않는다.)

실행: .venv\\Scripts\\python.exe analysis\\validate_baseline.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline_submit" / "model"))
from baseline_lib import (  # noqa: E402
    CatalogIndex,
    is_hangul_name,
    local_weighted_accuracy,
    make_training_pairs,
    predict_ids,
)

N_SPLITS = 5
SEED = 42
# baseline_submit/model/rf.pkl 에 저장된 하이퍼파라미터와 동일
RF_PARAMS = dict(
    n_estimators=60,
    max_depth=12,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=4,
    random_state=42,
)


def candidate_recall(frame: pd.DataFrame, index: CatalogIndex) -> np.ndarray:
    hits = []
    for row in frame.to_dict("records"):
        hits.append(index.id_to_index[row["target_vessel_id"]] in index.candidates(row))
    return np.asarray(hits, dtype=bool)


def main() -> None:
    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    index = CatalogIndex(catalog)

    hangul = train["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
    groups = train["target_vessel_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.empty(len(train), dtype=object)
    recall = np.zeros(len(train), dtype=bool)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, hangul, groups)):
        start = time.time()
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        assert not set(tr["target_vessel_id"]) & set(va["target_vessel_id"])

        x, y, _ = make_training_pairs(tr, index)
        model = RandomForestClassifier(**RF_PARAMS).fit(x, y)
        pred = predict_ids(model, va, index)
        oof[va_idx] = pred
        recall[va_idx] = candidate_recall(va, index)

        score = local_weighted_accuracy(va, pred)
        fold_rows.append({"fold": fold, "n_valid": len(va), "hangul_n": int(hangul[va_idx].sum()), **score})
        print(
            f"fold {fold}: weighted={score['weighted_accuracy']:.4f} "
            f"hangul={score['hangul_accuracy']:.4f} non_hangul={score['non_hangul_accuracy']:.4f} "
            f"({time.time() - start:.0f}s)"
        )

    folds = pd.DataFrame(fold_rows)
    total = local_weighted_accuracy(train, list(oof))
    correct = oof == train["target_vessel_id"].to_numpy()

    print("\n===== OOF (vessel GroupKFold, 5 folds) =====")
    print(f"weighted accuracy   : {total['weighted_accuracy']:.4f}")
    print(f"hangul accuracy     : {total['hangul_accuracy']:.4f}  (n={hangul.sum()})")
    print(f"non-hangul accuracy : {total['non_hangul_accuracy']:.4f}  (n={(~hangul).sum()})")
    print(f"fold weighted mean±std: {folds['weighted_accuracy'].mean():.4f} ± {folds['weighted_accuracy'].std():.4f}")
    print("\n후보 검색 recall (정답이 40개 후보 안에 있는 비율)")
    print(f"  hangul     : {recall[hangul].mean():.4f}")
    print(f"  non-hangul : {recall[~hangul].mean():.4f}")
    print("후보 안에 정답이 있을 때 정확도 (재순위 성능)")
    print(f"  hangul     : {correct[hangul & recall].mean():.4f}")
    print(f"  non-hangul : {correct[~hangul & recall].mean():.4f}")

    out_dir = ROOT / "outputs" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / "baseline_folds.csv", index=False, encoding="utf-8-sig")
    train.assign(pred_vessel_id=oof, correct=correct, candidate_hit=recall, is_hangul=hangul).to_csv(
        out_dir / "baseline_oof.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
