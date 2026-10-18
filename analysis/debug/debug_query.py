"""특정 한글 쿼리의 표기 후보·후보 피처 확인 (사전 없음, 해당 선박을 뺀 train으로 자원 생성)."""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

T.NO_DICT = True
tr, cat = T.load()
idx = ml.CatalogIndex(cat)
print("USE_VOCAB =", ml.USE_VOCAB)
for name in sys.argv[1:] or ["레지슬레이터", "레이디가스"]:
    row = tr[tr["vessel_name"] == name].iloc[0].to_dict()
    res = T.resources(tr[tr["target_vessel_id"] != row["target_vessel_id"]], cat)
    q = ml.prepare_query(row, res)
    print(f"\n== {name} (gt {row['gross_tonnage']}) true={cat.set_index('vessel_id').loc[row['target_vessel_id'], 'vessel_name']}")
    for k in ("compact", "rom_compact", "variants", "learned", "vocab_forms", "skel_rom", "skel_trans"):
        print(f"  {k}: {q[k]}")
    cands, feats = ml.candidates_and_features([q], idx, res)
    c = cands[0]
    names = [idx.catalog["vessel_name"][i] for i in c]
    for target in [row["target_vessel_id"], "Resolute", "Radiance", "Legislator", "Lady Gas"]:
        hits = [j for j, i in enumerate(c) if idx.ids[i] == target or names[j] == target]
        for j in hits:
            f = dict(zip(ml.FEATURE_NAMES, feats[j].round(2)))
            print(f"  [{names[j]}] gt={idx.gt[c[j]]} " + " ".join(f"{k}={f[k]}" for k in (
                "s_compact", "s_skel_rom", "s_var_ratio", "s_learn_ratio", "s_vocab_ratio", "gap_best", "flag_prob", "type_prob")))
