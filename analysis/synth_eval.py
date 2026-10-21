"""합성 한글 쿼리(outputs/synth/synth_test.csv)로 제출본들을 그대로 실행해 국적별 정확도 비교.

합성 선박은 train에 없으므로 전체 train으로 학습한 제출본을 써도 누수가 없다.
  실행: .venv\\Scripts\\python.exe analysis\\synth_eval.py v21 v22 v23
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TEST_COLS = ["sample_id", "vessel_name", "call_sign", "gross_tonnage", "international_gross_tonnage", "flag_name",
             "vessel_type", "arrival_year"]


def run_submission(version: str, test: pd.DataFrame) -> pd.Series:
    src = ROOT / "outputs" / "submissions" / version / "src"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data").mkdir()
        test[[c for c in TEST_COLS if c in test]].to_csv(tmp / "data" / "test.csv", index=False, encoding="utf-8-sig")
        shutil.copy(ROOT / "open" / "vessel_catalog.csv", tmp / "data" / "vessel_catalog.csv")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
        subprocess.run([sys.executable, str(src / "script.py")], cwd=tmp, env=env, check=True)
        return pd.read_csv(tmp / "output" / "submission.csv")["target_vessel_id"]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("versions", nargs="+")
    p.add_argument("--file", default="synth_test.csv")
    p.add_argument("--by", default="catalog_flag", help="정확도를 나눌 열 (group 등)")
    args = p.parse_args()
    test = pd.read_csv(ROOT / "outputs" / "synth" / args.file, encoding="utf-8-sig")
    keep = [c for c in ["vessel_name", "catalog_name", "catalog_flag", "group", "target_vessel_id"] if c in test]
    out = test[keep].copy()
    for v in args.versions:
        pred = run_submission(v, test).to_numpy()
        out[f"{v}_pred"] = pred
        out[v] = pred == test["target_vessel_id"].to_numpy()
        print(f"[{v}] 정확도 {out[v].mean():.4f}", flush=True)
    print(f"\n===== {args.by}별 정확도 =====")
    table = out.groupby(args.by)[args.versions].mean().round(3)
    table["n"] = out.groupby(args.by).size()
    print(table.sort_values(args.versions[-1]).to_string())
    out.to_csv(ROOT / "outputs" / "synth" / args.file.replace(".csv", "_eval.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
