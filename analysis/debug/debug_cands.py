"""한글 쿼리의 후보 수·학습 표기 확인."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

tr, cat = T.load()
idx = ml.CatalogIndex(cat)
res = ml.build_resources(tr, cat)
sub = tr[tr["vessel_name"].map(ml.is_hangul)].head(5)
qs = [ml.prepare_query(r, res) for r in sub.to_dict("records")]
for q in qs:
    print(q["hangul"], q["compact"], q["variants"], q["learned"])
c, f = ml.candidates_and_features(qs, idx, res)
print([len(x) for x in c])
lat = tr[~tr["vessel_name"].map(ml.is_hangul)].head(5)
c2, _ = ml.candidates_and_features([ml.prepare_query(r, res) for r in lat.to_dict("records")], idx, res)
print([len(x) for x in c2])
