"""앙상블 학습 속도·pickle 왕복 확인 (train 2,000행)."""
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

T.NO_DICT = True
tr, cat = T.load()
idx = ml.CatalogIndex(cat)
sub = tr.sample(2000, random_state=0)
t = time.time()
members, _ = T.fit_members(sub, cat, idx, list(T.MEMBERS))
print("fit", round(time.time() - t), "s")
ens = ml.RankEnsemble(list(members.values()))
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "ensemble.pkl"
    p.write_bytes(pickle.dumps(ens))
    ens2 = pickle.loads(p.read_bytes())
res = T.resources(sub, cat)
va = tr.drop(sub.index).sample(300, random_state=1)
a = ml.predict(ens, va, idx, res)
b = ml.predict(ens2, va, idx, res)
print("pickle roundtrip equal:", a == b)
print("acc:", T.score(va, a))
