"""제출 폴더를 대회 규칙대로 검사하고 submit.zip으로 묶는다.

검사 항목 (docs/02_evaluation.md, 03_rules.md 기준)
  1. 구조: 최상위에 model/, script.py, requirements.txt (추가 상위 폴더 없음)
  2. requirements.txt: 서버 기본 패키지를 다른 버전으로 고정하지 않음
  3. 외부 다운로드 흔적: http(s) URL, from_pretrained("hub-id") 등
  4. 서버 흉내 실행: 임시 폴더에 zip을 풀고 data/를 넣은 뒤 script.py 실행
     - output/submission.csv 형식, 행 수, sample_id 순서, 카탈로그 ID 여부
     - 실행 시간 (제한 20분)
  5. 평가 행 독립: 전체 / 순서 뒤섞기 / 부분집합 / 단독 행으로 실행해도 같은 행의 예측이 같음
  6. 용량: zip 10GB, 압축 해제 32GB 이하

test.csv는 비공개이므로 train.csv에서 정답 열을 뺀 3,000행으로 대신한다.

실행 예:
  .venv\\Scripts\\python.exe tools\\make_submit.py outputs\\submissions\\v11\\src outputs\\submissions\\v11
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPEN_DIR = ROOT / "open"

REQUIRED = ("model", "script.py", "requirements.txt")
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}
SERVER_PACKAGES = {
    "torch": "2.7.1+cu128", "pandas": "2.0.3", "numpy": "1.26.4", "scipy": "1.15.3",
    "scikit-learn": "1.5.2", "joblib": "1.5.2", "threadpoolctl": "3.7.0", "lightgbm": "4.6.0",
    "catboost": "1.2.8", "rapidfuzz": "3.14.3", "unidecode": "1.4.0", "transformers": "4.57.6",
    "accelerate": "1.9.0", "safetensors": "0.6.2", "sentencepiece": "0.2.0", "tqdm": "4.66.5",
}
ZIP_LIMIT = 10 * 1024**3
UNZIP_LIMIT = 32 * 1024**3
TIME_LIMIT = 20 * 60
N_TEST = 3000

errors: list[str] = []
warnings: list[str] = []


def check(ok: bool, message: str, warn_only: bool = False) -> None:
    mark = "OK  " if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{mark}] {message}")
    if not ok:
        (warnings if warn_only else errors).append(message)


def files_to_pack(src: Path) -> list[Path]:
    return sorted(
        p for p in src.rglob("*")
        if p.is_file() and not SKIP_PARTS & set(p.relative_to(src).parts) and p.suffix != ".pyc"
    )


def check_structure(src: Path) -> None:
    print("\n[1] 제출 구조")
    for name in REQUIRED:
        check((src / name).exists(), f"{name} 존재")
    check((src / "model").is_dir() and any((src / "model").iterdir()), "model/ 이 비어 있지 않음")
    extra = sorted(p.name for p in src.iterdir() if p.name not in REQUIRED and p.name not in SKIP_PARTS)
    check(not extra, f"최상위에 다른 파일·폴더 없음 {extra if extra else ''}", warn_only=True)
    for reserved in ("data", "output"):
        check(not (src / reserved).exists(), f"{reserved}/ 를 직접 넣지 않음 (서버가 자동 생성)")


def check_requirements(src: Path) -> None:
    print("\n[2] requirements.txt")
    lines = [l.split("#")[0].strip() for l in (src / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    pkgs = [l for l in lines if l]
    print(f"  추가 설치 패키지: {pkgs if pkgs else '없음 (서버 기본 패키지만 사용)'}")
    for line in pkgs:
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=|>|<)?\s*([^\s;]*)", line)
        if not m:
            continue
        name = m.group(1).lower().replace("_", "-")
        if name in SERVER_PACKAGES:
            same = m.group(2) == "==" and m.group(3) == SERVER_PACKAGES[name]
            check(same, f"{name}: 서버 기본 패키지({SERVER_PACKAGES[name]}) → requirements에서 빼는 것을 권장", warn_only=True)


def check_offline(src: Path) -> None:
    print("\n[3] 인터넷 사용 흔적 (추론 단계 인터넷 차단)")
    patterns = [r"https?://", r"from_pretrained\(\s*['\"][\w\-]+/[\w\-.]+['\"]", r"hf_hub_download", r"requests\.", r"urllib"]
    hits = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(src)}:{line} '{m.group(0)}'")
    check(not hits, "외부 다운로드 코드 없음" + ("" if not hits else f" → {hits}"), warn_only=True)


def make_fake_test() -> pd.DataFrame:
    train = pd.read_csv(OPEN_DIR / "train.csv", encoding="utf-8-sig")
    fake = train.sample(n=N_TEST, random_state=0).drop(columns=["target_vessel_id"]).reset_index(drop=True)
    fake["sample_id"] = [f"TEST_{i:04d}" for i in range(len(fake))]
    return fake


def run_script(workdir: Path, test: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    data = workdir / "data"
    out = workdir / "output"
    shutil.rmtree(data, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    data.mkdir()
    out.mkdir()
    test.to_csv(data / "test.csv", index=False, encoding="utf-8")
    shutil.copy(OPEN_DIR / "vessel_catalog.csv", data / "vessel_catalog.csv")
    start = time.time()
    # 평가 서버는 추론 단계에서 인터넷 차단 → HuggingFace 다운로드 시도가 있으면 여기서 실패하게 한다
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    proc = subprocess.run(
        [sys.executable, "script.py"], cwd=workdir, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"script.py 실패 (exit {proc.returncode})\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    sub_path = out / "submission.csv"
    if not sub_path.exists():
        raise RuntimeError("output/submission.csv 가 생성되지 않음")
    return pd.read_csv(sub_path, encoding="utf-8-sig"), elapsed


def check_run(zip_path: Path) -> None:
    print("\n[4] 서버 흉내 실행 (zip 압축 해제 → data/ 투입 → python script.py)")
    catalog_ids = set(pd.read_csv(OPEN_DIR / "vessel_catalog.csv", usecols=["vessel_id"], encoding="utf-8-sig")["vessel_id"])
    test = make_fake_test()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(work)
        try:
            sub, elapsed = run_script(work, test)
        except RuntimeError as exc:
            check(False, str(exc))
            return
        check(elapsed < TIME_LIMIT, f"추론 시간 {elapsed:.1f}s (제한 {TIME_LIMIT}s, 로컬 기준)")
        check(list(sub.columns) == ["sample_id", "target_vessel_id"], f"컬럼 {list(sub.columns)}")
        check(len(sub) == len(test), f"행 수 {len(sub)} / {len(test)}")
        check(sub["sample_id"].tolist() == test["sample_id"].tolist(), "sample_id 순서가 test.csv와 같음")
        check(sub["target_vessel_id"].isin(catalog_ids).all(), "모든 예측이 카탈로그 vessel_id")
        check(sub["target_vessel_id"].notna().all(), "빈 예측 없음")

        print("\n[5] 평가 행 독립 (같은 행은 어떤 조합으로 실행해도 같은 예측)")
        base = dict(zip(sub["sample_id"], sub["target_vessel_id"]))
        variants = {
            "순서 뒤섞기": test.sample(frac=1.0, random_state=1),
            "부분집합 500행": test.sample(n=500, random_state=2),
        }
        for name, frame in variants.items():
            got, _ = run_script(work, frame.reset_index(drop=True))
            same = all(base[s] == t for s, t in zip(got["sample_id"], got["target_vessel_id"]))
            check(same, f"{name}: 예측 일치")
        singles = test.sample(n=5, random_state=3)
        same = True
        for _, row in singles.iterrows():
            got, _ = run_script(work, row.to_frame().T)
            same &= base[row["sample_id"]] == got["target_vessel_id"].iloc[0]
        check(same, "단독 행 5개: 예측 일치")


def build_zip(src: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = files_to_pack(src)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(src).as_posix())
    print("\n[zip 내용]")
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:40s} {info.file_size:>12,d} B")
        top = {n.split("/")[0] for n in zf.namelist()}
        unzipped = sum(i.file_size for i in zf.infolist())
    print("\n[6] 구조·용량")
    check(top == set(REQUIRED), f"zip 최상위 = {sorted(top)}")
    size = zip_path.stat().st_size
    check(size <= ZIP_LIMIT, f"zip {size / 1024**2:.1f} MB (제한 10GB)")
    check(unzipped <= UNZIP_LIMIT, f"압축 해제 {unzipped / 1024**2:.1f} MB (제한 32GB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="model/, script.py, requirements.txt 가 있는 폴더")
    parser.add_argument("out_dir", type=Path, help="submit.zip 을 저장할 폴더")
    parser.add_argument("--skip-run", action="store_true", help="서버 흉내 실행 생략")
    args = parser.parse_args()
    src, zip_path = args.src.resolve(), args.out_dir.resolve() / "submit.zip"

    check_structure(src)
    check_requirements(src)
    check_offline(src)
    if errors:
        sys.exit(f"\n구조 오류로 중단: {errors}")
    build_zip(src, zip_path)
    if not args.skip_run:
        check_run(zip_path)

    print("\n" + "=" * 60)
    print(f"submit.zip: {zip_path}")
    print(f"FAIL {len(errors)}건, WARN {len(warnings)}건")
    for msg in errors:
        print(f"  FAIL: {msg}")
    for msg in warnings:
        print(f"  WARN: {msg}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
