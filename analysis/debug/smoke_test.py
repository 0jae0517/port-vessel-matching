"""match_lib 동작·속도 빠른 확인용."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

tr, cat = T.load()
t = time.time()
idx = ml.CatalogIndex(cat)
print("index", round(time.time() - t, 1), "s")
res = ml.build_resources(tr, cat)
print("token_dict", len(res["token_dict"]), list(res["token_dict"].items())[:10])
print("whole_dict", len(res["whole_dict"]))
sub = tr.sample(600, random_state=0)
t = time.time()
qs = [ml.prepare_query(r, res) for r in sub.to_dict("records")]
hits: list = []
truth = [idx.id_to_index[v] for v in sub["target_vessel_id"]]
c, f = ml.candidates_and_features(qs, idx, res, force=truth, hits=hits)
print("600 rows", round(time.time() - t, 1), "s", f.shape, "avg cand", round(f.shape[0] / 600, 1),
      "recall(in-sample dict)", round(sum(hits) / len(hits), 3))
