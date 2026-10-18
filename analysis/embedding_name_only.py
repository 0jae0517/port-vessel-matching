"""다국어 임베딩이 한글 선박명 매칭에 도움이 되는지 '이름만 top-1'로 비교 (OOF, 사전 없음).

각 한글 이름에 대해 카탈로그 78,595척 중 정답 순위를 본다.
  base  : 현재 v11의 표기 후보들(정규화·RR·변형·음절 학습) fuzzy ratio 최고점
  emb   : 한글 원문 ↔ 영문 카탈로그명 임베딩 코사인
  fused : base/100 + w * emb
  실행: .venv\\Scripts\\python.exe analysis\\embedding_name_only.py labse e5-small minilm
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import embed_lib  # noqa: E402  (오프라인 환경변수 먼저 설정)
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

PREFIX = {"e5-small": "query: ", "e5-base": "query: "}
names = sys.argv[1:] or ["labse", "e5-small", "minilm"]
T.NO_DICT = True
tr, cat = T.load()
index = ml.CatalogIndex(cat)
hangul = tr["vessel_name"].map(ml.is_hangul).to_numpy(bool)

# base 점수 (OOF 자원)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
rows, base = [], []
for tr_idx, va_idx in splitter.split(tr, hangul, tr["target_vessel_id"]):
    res = T.resources(tr.iloc[tr_idx], cat)
    va = tr.iloc[va_idx]
    va = va[va["vessel_name"].map(ml.is_hangul)].drop_duplicates(["vessel_name", "target_vessel_id"])
    for r in va.to_dict("records"):
        q = ml.prepare_query(r, res)
        forms = list(dict.fromkeys([q["compact"], q["rom_compact"], *q["variants"], *q["learned"]]))
        base.append(process.cdist(forms, index.compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).max(axis=0))
        rows.append(r)
base = np.vstack(base).astype(np.float32) / 100.0
truth = np.array([index.id_to_index[r["target_vessel_id"]] for r in rows])
qnames = [r["vessel_name"] for r in rows]


def ranks(scores: np.ndarray) -> np.ndarray:
    true_s = scores[np.arange(len(truth)), truth][:, None]
    return (scores > true_s).sum(axis=1)


def report(label: str, r: np.ndarray) -> None:
    print(f"  {label:22s} top-1 {(r == 0).mean():.4f} | top-5 {(r < 5).mean():.4f} | top-30 {(r < 30).mean():.4f}")


rb = ranks(base)
print(f"한글 고유 이름 {len(rows)}개")
report("base (v11 표기)", rb)
hard = rb > 0
print(f"  base로 1위 못 한 이름 {hard.sum()}개\n")

cat_names = cat["vessel_name"].astype(str).tolist()
out = ROOT / "outputs" / "emb"
out.mkdir(parents=True, exist_ok=True)
detail = pd.DataFrame({"vessel_name": qnames, "true": [cat_names[i] for i in truth], "base_rank": rb})
sims: dict[str, np.ndarray] = {}
for name in names:
    t0 = time.time()
    emb = embed_lib.NameEmbedder(ROOT / "models" / name, prefix=PREFIX.get(name, ""))
    cache = out / f"{name}_catalog.npy"
    if cache.exists():
        C = np.load(cache)
    else:
        C = emb.encode(cat_names)
        np.save(cache, C)
    Q = emb.encode(qnames)
    E = Q @ C.T
    re_ = ranks(E)
    print(f"[{name}] ({time.time() - t0:.0f}s)")
    report("emb 단독", re_)
    report("base 또는 emb 1위", np.where((rb == 0) | (re_ == 0), 0, 1))
    for w in (0.5, 1.0, 1.5, 2.0, 3.0):
        rf = ranks(base + w * E)
        report(f"fused w={w}", rf)
        print(f"      └ base 오답 {hard.sum()}개 중 해결 {((rf == 0) & hard).sum()} · base 정답 중 새로 틀림 {((rf > 0) & ~hard).sum()}")
    detail[f"{name}_rank"] = re_
    detail[f"{name}_top1"] = [cat_names[i] for i in E.argmax(axis=1)]
    sims[name] = E
    del emb
    print()

# 두 모델 함께: base + 1.5 * E1 + w * E2
model_names = list(sims)
for a in range(len(model_names)):
    for b in range(a + 1, len(model_names)):
        n1, n2 = model_names[a], model_names[b]
        print(f"[{n1} + {n2}]")
        for w in (0.5, 1.0, 1.5):
            rf = ranks(base + 1.5 * sims[n1] + w * sims[n2])
            report(f"1.5*{n1} + {w}*{n2}", rf)
            print(f"      └ base 오답 {hard.sum()}개 중 해결 {((rf == 0) & hard).sum()} · base 정답 중 새로 틀림 {((rf > 0) & ~hard).sum()}")
        print()

detail.sort_values("base_rank", ascending=False).to_csv(out / "embedding_name_only.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
cols = ["vessel_name", "true", "base_rank"] + [c for n in names for c in (f"{n}_rank", f"{n}_top1")]
print(detail[hard].sort_values("base_rank", ascending=False).head(30)[cols].to_string(index=False))
