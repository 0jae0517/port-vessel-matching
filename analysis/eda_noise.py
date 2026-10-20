"""EDA: train 쿼리가 정답 카탈로그 행에서 어떻게 '망가져' 있는지 (노이즈 생성 방식 추정).

  1. 선박명 노이즈 (영문 쿼리): 편집 거리 · 연산 종류(삽입/삭제/치환/자리바꿈/글자 중복) · 단어 단위 변화
  2. 톤수 노이즈: 정답 대비 오차 분포 · 결측 · 총톤수/국제총톤수 차이
  3. 국적 · 선종: 결측 · 한글→영문 대응의 일관성
  4. arrival_year 별 경향
  실행: .venv\\Scripts\\python.exe analysis\\eda_noise.py
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, OSA

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

tr = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
cat = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig").set_index("vessel_id")
df = tr.join(cat, on="target_vessel_id", rsuffix="_cat")
df["hangul"] = df["vessel_name"].astype(str).map(lambda s: bool(re.search(r"[가-힣]", s)))


def norm(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).upper()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


print(f"train {len(df)}행 · 한글 {df['hangul'].mean():.3f} · 선박 {df['target_vessel_id'].nunique()}척")

# ---------------------------------------------------------------- 1. 선박명 노이즈
section("1. 선박명 노이즈 (영문 쿼리)")
lat = df[~df["hangul"]].copy()
lat["q"] = lat["vessel_name"].map(norm)
lat["c"] = lat["vessel_name_cat"].map(norm)
lat["qc"] = lat["q"].str.replace(" ", "")
lat["cc"] = lat["c"].str.replace(" ", "")
lat["exact"] = lat["q"] == lat["c"]
lat["exact_nospace"] = lat["qc"] == lat["cc"]
lat["lev"] = [Levenshtein.distance(a, b) for a, b in zip(lat["qc"], lat["cc"])]
lat["osa"] = [OSA.distance(a, b) for a, b in zip(lat["qc"], lat["cc"])]
lat["len_c"] = lat["cc"].str.len()

print(f"영문 쿼리 {len(lat)}행")
print(f"  완전 일치(대소문자·기호 무시)      {lat['exact'].mean():.3f}")
print(f"  공백만 다름                          {(lat['exact_nospace'] & ~lat['exact']).mean():.3f}")
print("  편집 거리(공백 제거) 분포:")
dist = lat["lev"].clip(upper=6).value_counts(normalize=True).sort_index()
print("   " + " · ".join(f"{int(k)}{'+' if k == 6 else ''}: {v:.3f}" for k, v in dist.items()))
print(f"  자리바꿈 포함(OSA < Levenshtein)    {(lat['osa'] < lat['lev']).mean():.3f}")

# 편집 연산 종류
ops = Counter()
dup_ins = 0
for a, b in zip(lat["cc"], lat["qc"]):  # 정답 → 쿼리 로 가는 연산
    if a == b:
        continue
    for op in Levenshtein.editops(a, b):
        ops[op.tag] += 1
        if op.tag == "insert":
            ch = b[op.dest_pos]
            neighbors = {b[op.dest_pos - 1] if op.dest_pos > 0 else "", b[op.dest_pos + 1] if op.dest_pos + 1 < len(b) else ""}
            dup_ins += ch in neighbors
total_ops = sum(ops.values())
print("  편집 연산 비율 (정답 → 쿼리):", {k: round(v / total_ops, 3) for k, v in ops.items()}, f"(총 {total_ops})")
print(f"  삽입 중 '옆 글자 중복'(VICCTORY 형)  {dup_ins / max(ops['insert'], 1):.3f}")

# 단어 단위 변화
def word_change(q: str, c: str) -> str:
    qw, cw = q.split(), c.split()
    if q == c:
        return "같음"
    if qw == cw[1:] or qw == cw[:-1]:
        return "단어 하나 빠짐(앞/뒤)"
    if len(qw) == len(cw) and sum(x != y for x, y in zip(qw, cw)) == 1:
        diff = [i for i, (x, y) in enumerate(zip(qw, cw)) if x != y][0]
        return f"단어 1개 변형({'첫' if diff == 0 else '마지막' if diff == len(qw) - 1 else '중간'})"
    if len(qw) != len(cw):
        return f"단어 수 다름({len(cw)}→{len(qw)})"
    return "여러 단어 변형"

lat["word_change"] = [word_change(q, c) for q, c in zip(lat["q"], lat["c"])]
print("  단어 단위 변화:")
print(lat["word_change"].value_counts(normalize=True).round(3).to_string())

print("\n  이름 길이별 편집 거리 평균:")
lat["len_bin"] = pd.cut(lat["len_c"], [0, 6, 10, 15, 20, 100])
print(lat.groupby("len_bin", observed=True)["lev"].agg(["mean", "count"]).round(2).to_string())

print("\n  편집 거리 큰 예시 (lev ≥ 4):")
print(lat[lat["lev"] >= 4].sample(min(15, (lat["lev"] >= 4).sum()), random_state=0)[
    ["vessel_name", "vessel_name_cat", "lev", "arrival_year"]].to_string(index=False))

# ---------------------------------------------------------------- 2. 톤수 노이즈
section("2. 톤수 노이즈")
g, ig, cg = df["gross_tonnage"], df["international_gross_tonnage"], df["gross_tonnage_cat"]
print(f"  쿼리 총톤수 결측 {g.isna().mean():.3f} · 국제총톤수 결측 {ig.isna().mean():.3f} · 둘 다 결측 {(g.isna() & ig.isna()).mean():.3f}")
print(f"  총톤수 ≠ 국제총톤수 (둘 다 있음)   {((g != ig) & g.notna() & ig.notna()).mean():.3f}")
print(f"  카탈로그 총톤수 결측               {cg.isna().mean():.3f}")
rel = ((g - cg) / cg).where(g.notna() & cg.notna())
print(f"  정확 일치                          {(rel == 0).mean():.3f}")
bins = [-np.inf, -0.5, -0.2, -0.1, -0.05, -0.01, -1e-9, 1e-9, 0.01, 0.05, 0.1, 0.2, 0.5, np.inf]
labels = ["<-50%", "-50~-20%", "-20~-10%", "-10~-5%", "-5~-1%", "-1~0%", "0", "0~1%", "1~5%", "5~10%", "10~20%", "20~50%", ">50%"]
print("  (쿼리 − 정답)/정답 분포:")
print(pd.cut(rel.dropna(), bins, labels=labels).value_counts(normalize=True).reindex(labels).round(3).to_string())
nz = rel[(rel != 0) & rel.notna()]
print(f"  0이 아닌 오차의 |상대오차| 분위수 (25/50/75/90/99%): {np.percentile(nz.abs(), [25, 50, 75, 90, 99]).round(4)}")
# 국제총톤수 오차
rel_i = ((ig - cg) / cg).where(ig.notna() & cg.notna())
print(f"  국제총톤수 정확 일치               {(rel_i == 0).mean():.3f}")
print("  국제총톤수가 총톤수와 다른 행에서 어느 쪽이 정답과 가까운가:",
      pd.Series(np.where(((g - cg).abs() < (ig - cg).abs()), "총톤수", np.where((g - cg).abs() > (ig - cg).abs(), "국제총톤수", "같음")))
      [((g != ig) & g.notna() & ig.notna() & cg.notna()).to_numpy()].value_counts().to_dict())

# ---------------------------------------------------------------- 3. 국적 · 선종
section("3. 국적 · 선종")
print(f"  국적 결측 {df['flag_name'].isna().mean():.3f} · 선종 결측 {df['vessel_type'].isna().mean():.3f}")
flag_pairs = df.dropna(subset=["flag_name"]).groupby("flag_name")["flag_name_cat"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
flag_top = df.dropna(subset=["flag_name"]).groupby("flag_name")["flag_name_cat"].agg(lambda s: s.value_counts().index[0])
d = df.dropna(subset=["flag_name"])
consistent = d["flag_name_cat"] == d["flag_name"].map(flag_top)
print(f"  쿼리 국적의 대표 영문 국적과 정답 국적이 같은 비율: {consistent.mean():.3f}")
print("  행이 많은 국적별 대표 대응·일치율:")
top_flags = d["flag_name"].value_counts().head(10).index
print(pd.DataFrame({"대표 영문": flag_top[top_flags], "일치율": flag_pairs[top_flags].round(3),
                    "행 수": d["flag_name"].value_counts()[top_flags]}).to_string())
print("  국적 불일치 예시:")
print(d[~consistent].sample(8, random_state=0)[["vessel_name", "flag_name", "flag_name_cat"]].to_string(index=False))

type_top = df.groupby("vessel_type")["vessel_type_cat"].agg(lambda s: s.value_counts().index[0])
type_rate = df.groupby("vessel_type")["vessel_type_cat"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
print("\n  선종별 대표 카탈로그 선종·일치율 (상위 8):")
tt = df["vessel_type"].value_counts().head(8).index
print(pd.DataFrame({"대표": type_top[tt], "일치율": type_rate[tt].round(3), "행 수": df["vessel_type"].value_counts()[tt]}).to_string())

# ---------------------------------------------------------------- 4. arrival_year
section("4. arrival_year 별 경향")
lat_year = lat.groupby("arrival_year").agg(행=("lev", "size"), 이름완전일치=("exact", "mean"), 평균편집거리=("lev", "mean"))
df["ton_exact"] = (rel == 0)
df["flag_ok"] = df["flag_name_cat"] == df["flag_name"].map(flag_top)
yr = df.groupby("arrival_year").agg(전체행=("sample_id", "size"), 한글비율=("hangul", "mean"),
                                    톤수정확=("ton_exact", "mean"), 톤수결측=("gross_tonnage", lambda s: s.isna().mean()),
                                    국적일치=("flag_ok", "mean"))
print(yr.join(lat_year).round(3).to_string())

# ---------------------------------------------------------------- 5. 선박당 행
section("5. 같은 선박의 여러 행은 서로 다른가")
multi = df.groupby("target_vessel_id").filter(lambda g: len(g) > 1)
per = multi.groupby("target_vessel_id").agg(이름종류=("vessel_name", "nunique"), 톤수종류=("gross_tonnage", "nunique"),
                                           국적종류=("flag_name", "nunique"), 연도종류=("arrival_year", "nunique"), 행=("sample_id", "size"))
print(per.describe().round(2).to_string())
