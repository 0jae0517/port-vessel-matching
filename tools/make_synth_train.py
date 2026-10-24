"""카탈로그 전체로 한글 학습 데이터 만들기 (v31).

train의 한글 선박은 382척뿐이라 LightGBM이 '한글 기록 → 영문 선박' 판단을 좁게 배운다.
카탈로그 한글 표기표(kanana, 공식 카탈로그로만 생성)에서 train에 없는 선박을 국적별로 고르게 뽑아
한글 기록 행을 만든다. 노이즈는 새로 지어내지 않고 train에서 관찰된 수준만 쓴다.
  - 선박명: 표기표의 첫 번째(가장 좋은) 표기. 학습 때 정답 선박의 이 표기는 한글 비교 피처에서 빼서
            '쿼리와 똑같은 표기가 표에 있는' 너무 쉬운 문제가 되지 않게 한다 (match_lib: q["synthetic"])
  - 톤수: 절반은 정확, 나머지 ±3~5% (EDA), 국적·선종: train 입력 표기(한글)로, 국적 16% 결측
  - train 정답은 쓰지 않는다 (train 선박 제외) → 누수 없음

  실행: .venv\\Scripts\\python.exe tools\\make_synth_train.py --n 2000
  결과: outputs/synth/synth_train.csv (sample_id = SYNTR_xxxxx)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import match_lib as ml  # noqa: E402


def majority_map(train: pd.DataFrame, catalog: pd.DataFrame, col: str) -> dict:
    m = train.merge(catalog[["vessel_id", col]].rename(columns={col: "c"}), left_on="target_vessel_id", right_on="vessel_id")
    return m.groupby("c")[col].agg(lambda s: s.dropna().mode().iat[0] if s.notna().any() else np.nan).to_dict()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--table", default="outputs/translit/kanana2b.json")
    args = p.parse_args()

    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    table = json.loads((ROOT / args.table).read_text(encoding="utf-8"))
    seen = set(train["target_vessel_id"])
    pool = catalog[~catalog["vessel_id"].isin(seen) & catalog["vessel_name"].astype(str).isin(table)
                   & ~catalog["vessel_name"].astype(str).map(ml.is_hangul)].copy()
    # 국적 비율은 train 입항 선박(정답 국적)의 분포를 따른다 → 실제 항만에 들어오는 선박 구성과 비슷하게
    #   (카탈로그 그대로 뽑으면 중국·인도네시아 내항선이 과다, 국적별 균등이면 희귀 국적이 과다)
    port_flags = train[["target_vessel_id"]].merge(catalog[["vessel_id", "flag_name"]], left_on="target_vessel_id",
                                                   right_on="vessel_id")
    w_flag = port_flags["flag_name"].value_counts(normalize=True)
    w = pool["flag_name"].map(w_flag).fillna(0) / pool["flag_name"].map(pool["flag_name"].value_counts())
    pick = pool.sample(n=min(args.n, int((w > 0).sum())), weights=w, random_state=0)
    per_flag = "train 분포"

    flag_map, type_map = majority_map(train, catalog, "flag_name"), majority_map(train, catalog, "vessel_type")
    rng = np.random.default_rng(0)
    rows = []
    for k, r in enumerate(pick.itertuples(index=False)):
        v = table[str(r.vessel_name)]
        variants = [v] if isinstance(v, str) else list(v)
        name = variants[0]  # 가장 좋은 표기. 학습 때 이 표기(첫 슬롯)는 정답 비교에서 뺀다 (match_lib: synthetic)
        if not name or not ml.is_hangul(name):
            continue
        gt = r.gross_tonnage
        if pd.notna(gt) and rng.random() > 0.5:
            gt = round(gt * (1 + rng.choice([-1, 1]) * rng.uniform(0.03, 0.05)))
        rows.append({
            "sample_id": f"SYNTR_{k:05d}", "vessel_name": name, "call_sign": np.nan,
            "gross_tonnage": gt, "international_gross_tonnage": gt,
            "flag_name": flag_map.get(r.flag_name, np.nan) if rng.random() > 0.16 else np.nan,
            "vessel_type": type_map.get(r.vessel_type, np.nan),
            "arrival_year": int(rng.integers(2015, 2027)), "target_vessel_id": r.vessel_id,
        })
    out = ROOT / "outputs" / "synth" / "synth_train.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    top = pick["flag_name"].value_counts().head(12).to_dict()
    print(f"저장: {out} ({len(df)}행, 국적 {pick['flag_name'].nunique()}개, 국적당 최대 {per_flag})\n상위 국적: {top}")
    print(df[["vessel_name", "target_vessel_id", "flag_name"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
