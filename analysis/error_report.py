"""CV OOF 오답을 쿼리 / 예측 / 정답 카탈로그 행으로 나란히 출력.

  실행: .venv\\Scripts\\python.exe analysis\\error_report.py v11_nodict
        (outputs/cv/<tag>_oof.csv 를 읽는다)
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
tag = sys.argv[1] if len(sys.argv) > 1 else "v11_nodict"
oof = pd.read_csv(ROOT / "outputs" / "cv" / f"{tag}_oof.csv", encoding="utf-8-sig")
cat = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig").set_index("vessel_id")
if "candidate_hit" not in oof.columns:
    oof["candidate_hit"] = True

wrong = oof[~oof["correct"]]
print(f"[{tag}] 오답 {len(wrong)}건 (한글 {wrong['is_hangul'].sum()}, 비한글 {(~wrong['is_hangul']).sum()}), "
      f"후보 누락 {(~wrong['candidate_hit']).sum()}건")
for _, r in wrong.sort_values(["is_hangul", "candidate_hit"], ascending=[False, True]).iterrows():
    p, t = cat.loc[r["pred_vessel_id"]], cat.loc[r["target_vessel_id"]]
    lang = "H" if r["is_hangul"] else "L"
    hit = "hit " if r["candidate_hit"] else "MISS"
    print(f"[{lang} {hit}] Q {r['vessel_name']!s:22} gt={r['gross_tonnage']!s:>8} {r['flag_name']!s:8} {r['vessel_type']}")
    print(f"          P {p['vessel_name']!s:22} gt={p['gross_tonnage']!s:>8} {p['flag_name']!s:12} {p['vessel_type']}")
    print(f"          T {t['vessel_name']!s:22} gt={t['gross_tonnage']!s:>8} {t['flag_name']!s:12} {t['vessel_type']}")
