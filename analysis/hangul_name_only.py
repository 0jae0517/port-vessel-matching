"""어려운 지표: 톤수·국적 없이 '이름 표기만'으로 한글 쿼리의 정답이 카탈로그 1위인 비율 (OOF).

자매선(같은 선단·비슷한 톤수)은 결국 이름 표기로만 갈린다. 이 지표가 오르면
test에 남은 한글 오답에도 효과가 있을 가능성이 크다. 모델 학습이 없어 몇 분이면 끝난다.
  실행: .venv\\Scripts\\python.exe analysis\\hangul_name_only.py [tag]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "latest"
T.NO_DICT = True
tr, cat = T.load()
index = ml.CatalogIndex(cat)
hangul = tr["vessel_name"].map(ml.is_hangul).to_numpy(bool)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
records = []
for tr_idx, va_idx in splitter.split(tr, hangul, tr["target_vessel_id"]):
    res = T.resources(tr.iloc[tr_idx], cat)
    va = tr.iloc[va_idx]
    va = va[va["vessel_name"].map(ml.is_hangul)].drop_duplicates(["vessel_name", "target_vessel_id"])
    for r in va.to_dict("records"):
        q = ml.prepare_query(r, res)
        forms = list(dict.fromkeys([q["compact"], q["rom_compact"], *q["variants"], *q["learned"]]))
        scores = process.cdist(forms, index.compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).max(axis=0)
        true_i = index.id_to_index[r["target_vessel_id"]]
        rank = int((scores > scores[true_i]).sum())  # 동점은 유리하게
        vrank = -1
        if q.get("vocab_forms"):
            vs = process.cdist(q["vocab_forms"], index.compact, scorer=fuzz.ratio, dtype=np.uint8,
                               workers=-1).max(axis=0)
            vrank = int((vs > vs[true_i]).sum())
        records.append({"vessel_name": r["vessel_name"], "true": index.catalog["vessel_name"][true_i],
                        "top1": index.catalog["vessel_name"][int(np.argmax(scores))], "rank": rank,
                        "vocab_rank": vrank, "vocab_form": (q.get("vocab_forms") or [""])[0],
                        "true_score": int(scores[true_i]), "top_score": int(scores.max())})

df = pd.DataFrame(records)
print(f"[{tag}] 한글 고유 이름 {len(df)}개 | 이름만 top-1 {(df['rank'] == 0).mean():.4f} "
      f"| top-5 {(df['rank'] < 5).mean():.4f} | top-30 {(df['rank'] < 30).mean():.4f}")
if (df["vocab_rank"] >= 0).any():
    print(f"   단어장 표기만 top-1 {(df['vocab_rank'] == 0).mean():.4f} | "
          f"둘 중 하나라도 top-1 {((df['rank'] == 0) | (df['vocab_rank'] == 0)).mean():.4f}")
df.sort_values("rank", ascending=False).to_csv(
    ROOT / "outputs" / "cv" / f"name_only_{tag}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(df[df["rank"] > 0].sort_values("rank", ascending=False).head(40).to_string(index=False))
