"""카탈로그 영문 선박명 → 한글 표기 합성 (Qwen2.5 few-shot). 새 test의 '어려운 한글' 가설 검증·학습 데이터용.

가설: train 한글 선박 382척 중 180척이 한국 국적(한국어 이름)이고, 중국·인도네시아·일본·베트남 등
카탈로그에 많은 국적의 선박은 거의 없다. 새 test의 한글 행에 이런 외국 선박이 많다면 지금 모델은 약하다.

  check: fold 0 검증 한글 선박의 카탈로그명을 한글로 바꿔 실제 한글 표기와 비교 (생성 품질)
  make : train에 없는 선박을 국적별로 뽑아 한글 쿼리 합성 → outputs/synth/synth_test.csv (정답 포함)

  실행: .venv\\Scripts\\python.exe analysis\\synth_hangul.py check
        .venv\\Scripts\\python.exe analysis\\synth_hangul.py make --per-flag 60
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "analysis"))
import augment as A  # noqa: E402
import match_lib as ml  # noqa: E402

SYSTEM = ("A Korean port authority records foreign and domestic vessel names in Hangul, written by sound "
          "(how a Korean would pronounce the name). Convert the given vessel name to that Hangul form. "
          "Output only the Hangul name.")
FLAGS = ["Peoples' Republic of China", "Indonesia", "Japan", "Vietnam", "Russia", "Panama", "Liberia",
         "Marshall Islands", "Singapore", "Hong Kong", "Malaysia", "Philippines", "Turkey", "India", "Greece"]


def shots(train: pd.DataFrame, catalog: pd.DataFrame, n: int = 16) -> list[tuple[str, str]]:
    name_of = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str)))
    h = train[train["vessel_name"].map(ml.is_hangul)].drop_duplicates("target_vessel_id")
    pairs = [(name_of[t], str(q)) for q, t in zip(h["vessel_name"], h["target_vessel_id"])
             if not ml.is_hangul(name_of[t]) and len(name_of[t]) <= 30]
    return random.Random(1).sample(pairs, n)


class HangulOnly:
    """생성 토큰을 한글 음절·숫자·공백(·종료)으로 제한하는 logits processor (영문·한자 섞임 방지)."""

    def __init__(self, tok, eos_ids: list[int]) -> None:
        ok = []
        for i in range(len(tok)):
            s = tok.decode([i])
            if s and all(("가" <= ch <= "힣") or ch.isdigit() or ch == " " for ch in s) and s.strip(" ") != "" or s == " ":
                ok.append(i)
        ok += list(eos_ids)
        self.mask = None
        self.ok = torch.tensor(sorted(set(i for i in ok if i is not None and i >= 0)))

    def __call__(self, input_ids, scores):
        if self.mask is None or self.mask.shape[-1] != scores.shape[-1]:
            self.mask = torch.full((scores.shape[-1],), float("-inf"), device=scores.device)
            self.mask[self.ok.to(scores.device)] = 0
        return scores + self.mask


class Generator:
    def __init__(self, model: str, examples: list[tuple[str, str]]) -> None:
        from transformers import LogitsProcessorList

        path = ROOT / "models" / model
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).to("cuda").eval()
        self.examples = examples
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        self.procs = LogitsProcessorList([HangulOnly(self.tok, [*eos, self.tok.eos_token_id])])

    @torch.no_grad()
    def __call__(self, name: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}]
        for en, ko in self.examples:
            msgs += [{"role": "user", "content": en}, {"role": "assistant", "content": ko}]
        msgs.append({"role": "user", "content": name})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        gen = self.model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=self.tok.eos_token_id,
                                  logits_processor=self.procs, num_beams=3)
        return " ".join(self.tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).split())


def majority_map(train: pd.DataFrame, catalog: pd.DataFrame, col: str) -> dict[str, str]:
    """카탈로그 값(영문) → train 입력에서 가장 흔한 값(한글)."""
    m = train.merge(catalog[["vessel_id", col]].rename(columns={col: "c"}), left_on="target_vessel_id", right_on="vessel_id")
    return m.groupby("c")[col].agg(lambda s: s.dropna().mode().iat[0] if s.notna().any() else np.nan).to_dict()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "make", "make2"])
    p.add_argument("--model", default="kanana2b")
    p.add_argument("--per-flag", type=int, default=60)
    p.add_argument("--per-group", type=int, default=150)
    args = p.parse_args()

    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    tr_idx, va_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                          .split(train, hangul, train["target_vessel_id"]))
    t0 = time.time()

    if args.mode == "check":
        gen = Generator(args.model, shots(train.iloc[tr_idx], catalog))
        name_of = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str)))
        va = train.iloc[va_idx]
        hv = va[va["vessel_name"].map(ml.is_hangul)].drop_duplicates("target_vessel_id")
        rows = []
        for q, t in zip(hv["vessel_name"], hv["target_vessel_id"]):
            g = gen(name_of[t])
            a, b = str(q).replace(" ", ""), g.replace(" ", "")
            rows.append({"catalog": name_of[t], "real": q, "synth": g, "exact": a == b, "ratio": fuzz.ratio(a, b)})
        df = pd.DataFrame(rows)
        print(f"{len(df)}척: 완전 일치 {df.exact.mean():.3f}, 평균 글자 유사도 {df.ratio.mean():.1f} ({time.time() - t0:.0f}s)")
        print(df.head(30).to_string())
        return

    gen = Generator(args.model, shots(train, catalog))
    seen = set(train["target_vessel_id"])
    unseen = catalog[~catalog["vessel_id"].isin(seen) & ~catalog["vessel_name"].astype(str).map(ml.is_hangul)]
    flag_map, type_map = majority_map(train, catalog, "flag_name"), majority_map(train, catalog, "vessel_type")
    rng = np.random.default_rng(0)
    rows = []

    def add(pick: pd.DataFrame, group: str, convert) -> None:
        for r in pick.itertuples(index=False):
            gt = r.gross_tonnage
            if pd.notna(gt) and rng.random() > 0.5:  # EDA: 톤수 절반은 정확, 나머지는 ±3~5% 오차
                gt = round(gt * (1 + rng.choice([-1, 1]) * rng.uniform(0.03, 0.05)))
            rows.append({
                "sample_id": f"SYN_{len(rows):05d}", "vessel_name": convert(str(r.vessel_name)), "call_sign": np.nan,
                "gross_tonnage": gt, "international_gross_tonnage": gt,
                "flag_name": flag_map.get(r.flag_name, np.nan), "vessel_type": type_map.get(r.vessel_type, np.nan),
                "arrival_year": int(rng.integers(2015, 2027)), "target_vessel_id": r.vessel_id,
                "catalog_name": r.vessel_name, "catalog_flag": r.flag_name, "group": group,
            })
            if len(rows) % 100 == 0:
                print(f"  {len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    if args.mode == "make":
        pool = unseen[unseen["flag_name"].isin(FLAGS)]
        add(pool.groupby("flag_name", group_keys=False).apply(lambda g: g.sample(min(len(g), args.per_flag), random_state=0)),
            "foreign", gen)
        name = "synth_test.csv"
    else:
        # make2: 한국 국적(한국어 이름) · 중국식 병음 이름 (규칙 표기 / LLM 표기)
        from pinyin_hangul import is_pinyin_name, to_hangul

        kr = unseen[unseen["flag_name"].eq("South Korea")].sample(args.per_group, random_state=0)
        add(kr, "KR", gen)
        py = unseen[unseen["vessel_name"].astype(str).map(is_pinyin_name)].sample(args.per_group, random_state=0)
        print(f"병음 이름 후보 예: {py['vessel_name'].head(10).tolist()}")
        add(py, "pinyin_rule", to_hangul)
        add(py, "pinyin_llm", gen)
        name = "synth_test2.csv"
    out = ROOT / "outputs" / "synth"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / name, index=False, encoding="utf-8-sig")
    print(f"저장: {out / name} ({len(df)}행, {time.time() - t0:.0f}s)")
    print(df.groupby("group").head(8)[["group", "vessel_name", "catalog_name", "catalog_flag"]].to_string())


if __name__ == "__main__":
    main()
