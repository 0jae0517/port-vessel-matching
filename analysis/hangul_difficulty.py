"""train 한글 이름마다 '정답 영문명을 우리 표기 후보로 얼마나 재현하는지'를 OOF로 측정.

fold의 학습 부분으로만 음절표를 만들고(사전 없음), 검증 부분 한글 이름의 표기 후보
(정규화·RR·규칙 변형·음절 학습 표기) 중 정답 영문명과 가장 비슷한 점수를 기록한다.
점수가 낮은 이름 = 지금 방식이 약한 유형.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

T.NO_DICT = True
tr, cat = T.load()
cat_by_id = cat.set_index("vessel_id")
hangul = tr["vessel_name"].map(ml.is_hangul).to_numpy(bool)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for tr_idx, va_idx in splitter.split(tr, hangul, tr["target_vessel_id"]):
    res = T.resources(tr.iloc[tr_idx], cat)
    va = tr.iloc[va_idx]
    for r in va[va["vessel_name"].map(ml.is_hangul)].to_dict("records"):
        q = ml.prepare_query(r, res)
        true_name = cat_by_id.loc[r["target_vessel_id"], "vessel_name"]
        true_compact = "".join(ml.tokenize(true_name))
        forms = {"compact": q["compact"], "rr": q["rom_compact"]}
        forms.update({f"var{k}": v for k, v in enumerate(q["variants"])})
        forms.update({f"learn{k}": v for k, v in enumerate(q["learned"])})
        scores = {k: fuzz.ratio(v, true_compact) for k, v in forms.items()}
        best = max(scores, key=scores.get)
        rows.append({
            "vessel_name": r["vessel_name"], "true_name": true_name, "best_form": best,
            "best_text": forms[best], "best_score": scores[best],
            "learn0": q["learned"][0] if q["learned"] else "", "target": r["target_vessel_id"],
        })

df = pd.DataFrame(rows).drop_duplicates(["vessel_name", "target"])
oof = pd.read_csv(ROOT / "outputs" / "cv" / "v11_nodict_oof.csv", encoding="utf-8-sig")
wrong_targets = set(oof.loc[~oof["correct"] & oof["is_hangul"], "target_vessel_id"])
df["v5_wrong"] = df["target"].isin(wrong_targets)
out = ROOT / "outputs" / "cv" / "hangul_difficulty.csv"
df.sort_values("best_score").to_csv(out, index=False, encoding="utf-8-sig")

print(f"한글 고유 이름 {len(df)}개")
print("best_score 분포:", np.percentile(df["best_score"], [5, 10, 25, 50]).round(1))
print("가장 잘 맞는 표기 종류:", df["best_form"].str.replace(r"\d", "", regex=True).value_counts().to_dict())
print(f"\n표기 재현이 어려운 이름 하위 60개 (저장: {out.name})")
pd.set_option("display.width", 250)
print(df.sort_values("best_score").head(60)[
    ["vessel_name", "true_name", "best_text", "best_score", "learn0", "v5_wrong"]].to_string(index=False))
