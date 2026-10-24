"""LLM 재정렬(listwise) 학습·평가 — 실제 train 기록만 사용 (RunPod 등 GPU에서 실행).

입력
  outputs/rerank/train_cands.csv : train 전체 행의 v28 상위 10개 후보 (analysis/rerank_data.py)
  outputs/rerank/fold0_diag.csv  : fold 0 검증 선박의 v28 진짜 OOF 후보 (stress_diag 에서 원본·톤수·국적 시나리오만)
방법
  - 프롬프트: 기록(선박명·톤수·국적·선종) + 후보 K개(카탈로그명·톤수·국적·선종) → 같은 선박 번호
  - 학습: fold 0 선박을 뺀 train 행 중 정답이 상위 K 안에 있는 행. 후보 순서는 섞고 LightGBM 점수는 주지 않는다
          (train 행은 v28 이 학습에 쓴 행이라 순위가 실제보다 쉬움 → 순서에 기대지 않게)
  - 평가: fold 0 진단 행에서 LightGBM 단독 vs LightGBM 점수 + w × LLM 로그확률 (w·gate 는 선박 절반씩 교차로 고름)
  - --final: fold 0 포함 전체로 학습해 제출용 LoRA 를 병합 저장 (models/<out>)

  평가: python tools/rerank_train.py --base kanana2b --eval
  제출용: python tools/rerank_train.py --base kanana2b --final --out kanana-rerank
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
K = 5  # --k: 후보 수 (v34: 10까지)
LABELS = [str(i) for i in range(1, 10)]  # --letters: A..J (후보 10개면 '10'이 두 토큰이라 글자 라벨)
SYSTEM = ("You match a Korean port entry record to a standard vessel catalog. The record's vessel name may be "
          "written in Hangul or have small spelling differences, and tonnage may be slightly off or missing. "
          "Pick the catalog candidate that is the same vessel. Answer with the candidate number only.")
SYSTEM_LETTER = SYSTEM.replace("candidate number only", "candidate letter only")


def fmt_num(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) or v <= 0 else f"{int(round(v))}"


def build_prompt(tok, rec: dict, cands: list[dict]) -> str:
    lines = [f"Record: name={rec['vessel_name']} | GT {fmt_num(rec.get('gross_tonnage'))} | "
             f"intl GT {fmt_num(rec.get('international_gross_tonnage'))} | flag {rec.get('flag_name') or '-'} | "
             f"type {rec.get('vessel_type') or '-'}", "Candidates:"]
    for k, c in enumerate(cands):
        lines.append(f"{LABELS[k]}. {c['vessel_name']} | GT {fmt_num(c['gross_tonnage'])} | flag {c['flag_name']} | type {c['vessel_type']}")
    system = SYSTEM_LETTER if LABELS[0] == "A" else SYSTEM
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def vessel_half(vid: str) -> int:
    return zlib.crc32(str(vid).encode()) % 2


def clean(v):
    return None if isinstance(v, float) and np.isnan(v) else v


def examples(df: pd.DataFrame, recs: dict, cat: pd.DataFrame, tok, shuffle: bool, seed: int):
    """(프롬프트, 정답 위치, LightGBM 상위 K 점수, 원래 순서의 후보 id) — 정답이 상위 K 밖이면 정답 위치 -1."""
    rng = random.Random(seed)
    out = []
    for r in df.itertuples(index=False):
        ids = str(r.top10_ids).split()[:K]
        scores = [float(x) for x in str(r.top10_scores).split()[:K]]
        order = list(range(len(ids)))
        if shuffle:
            rng.shuffle(order)
        shown = [ids[o] for o in order]
        info = [{"vessel_name": cat.at[i, "vessel_name"], "gross_tonnage": cat.at[i, "gross_tonnage"],
                 "flag_name": cat.at[i, "flag_name"], "vessel_type": cat.at[i, "vessel_type"]} for i in shown]
        label = shown.index(r.truth_id) if r.truth_id in shown else -1
        out.append((build_prompt(tok, recs[r.sample_id], info), label, order, scores, ids))
    return out


def train_lora(base: Path, tok, data: list[tuple[str, int]], epochs: int, lr: float, batch: int, rank: int):
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr)
    steps = epochs * ((len(data) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, 1 - s / steps))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rng, t0, step = random.Random(0), time.time(), 0
    model.train()
    for _ in range(epochs):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            enc = []
            for i in idx[s : s + batch]:
                prompt, label = data[i]
                p = tok(prompt, add_special_tokens=False)["input_ids"]
                a = tok(LABELS[label] + tok.eos_token, add_special_tokens=False)["input_ids"]
                enc.append((p + a, [-100] * len(p) + a))
            w = max(len(x) for x, _ in enc)
            ids = torch.tensor([x + [pad] * (w - len(x)) for x, _ in enc], device="cuda")
            lab = torch.tensor([y + [-100] * (w - len(y)) for _, y in enc], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (w - len(x)) for x, _ in enc], device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"    step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def score(model, tok, prompts: list[str]) -> np.ndarray:
    digit_ids = [tok(lb, add_special_tokens=False)["input_ids"][0] for lb in LABELS[:K]]
    out = np.zeros((len(prompts), K), dtype=np.float32)
    for i, p in enumerate(prompts):  # 행마다 단독 (평가 행 독립과 같은 조건)
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits[0, -1].float()
        out[i] = torch.log_softmax(logits[digit_ids], dim=-1).cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="kanana2b")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--k", type=int, default=5, help="후보 수 (최대 10)")
    p.add_argument("--letters", action="store_true", help="후보 라벨을 A..J 로 (후보 10개일 때 필수)")
    p.add_argument("--merge", action="store_true", help="models/<out>-adapter 를 기반 모델에 병합해 models/<out> 으로 저장")
    p.add_argument("--out", default="kanana-rerank")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rank", type=int, default=16)
    args = p.parse_args()
    global K, LABELS
    K = args.k
    LABELS = list("ABCDEFGHIJ") if args.letters else [str(i) for i in range(1, 10)]
    assert K <= len(LABELS)
    in_k = lambda df: [t in str(ids).split()[:K] for t, ids in zip(df["truth_id"], df["top10_ids"])]

    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    cat = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig").set_index("vessel_id")
    recs = {r["sample_id"]: {k: clean(v) for k, v in r.items()} for r in train.to_dict("records")}
    cands = pd.read_csv(ROOT / "outputs" / "rerank" / "train_cands.csv", encoding="utf-8-sig")
    base = ROOT / "models" / args.base
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)

    if args.eval:
        diag = pd.read_csv(ROOT / "outputs" / "rerank" / "fold0_diag.csv", encoding="utf-8-sig")
        fold0_vessels = set(diag["truth_id"])
        tr = cands[~cands["truth_id"].isin(fold0_vessels) & pd.Series(in_k(cands), index=cands.index)]
        data = [(pr, lb) for pr, lb, *_ in examples(tr, recs, cat, tok, shuffle=True, seed=0) if lb >= 0]
        print(f"학습 {len(data)}개 (fold 0 선박 제외), 평가 {len(diag)}행", flush=True)
        model = train_lora(base, tok, data, args.epochs, args.lr, args.batch, args.rank)
        # 평가 행: 진단 쿼리(노이즈 걸린 기록)를 그대로 써야 하므로 recs 를 진단 기록으로 교체
        drecs = {r["key"]: json.loads(r["rec"]) for r in diag.to_dict("records")}
        dd = diag.rename(columns={"key": "sample_id"})
        ev = examples(dd, drecs, cat, tok, shuffle=False, seed=0)
        t0 = time.time()
        lp = score(model, tok, [e[0] for e in ev])
        print(f"평가 LLM 점수 {len(ev)}행 ({time.time() - t0:.0f}s)", flush=True)
        labels = np.array([e[1] for e in ev])
        lgb = np.array([e[3] + [-1e9] * (K - len(e[3])) for e in ev])
        margin = lgb[:, 0] - lgb[:, 1]
        halves = dd["truth_id"].map(vessel_half).to_numpy()

        def decide(mask, w, gate):
            pick = np.zeros(mask.sum(), dtype=int)
            sub_l, sub_p, sub_m = lgb[mask], lp[mask], margin[mask]
            for j in range(len(pick)):
                if sub_m[j] < gate:
                    pick[j] = int(np.argmax(sub_l[j] + w * sub_p[j]))
            return pick

        grid = [(w, g) for w in (0.25, 0.5, 1, 2, 4, 100) for g in (0.5, 1, 2, 3, 100)]
        final = np.zeros(len(ev), dtype=int)
        for h in (0, 1):
            tune, test = halves != h, halves == h
            best = max(grid, key=lambda wg: (decide(tune, *wg) == labels[tune]).sum())
            final[test] = decide(test, *best)
            print(f"[half {h}] w={best[0]}, gate={best[1]}")
        dd["lgb_ok"] = labels == 0
        dd["rr_ok"] = final == labels
        t = dd.groupby(["시나리오", "hangul"])[["lgb_ok", "rr_ok"]].mean().round(4)
        print(t.to_string())
        print("전체 LightGBM", dd.lgb_ok.mean().round(4), "→ 재정렬", dd.rr_ok.mean().round(4))
        dd.drop(columns=["rec"]).to_csv(ROOT / "outputs" / "rerank" / f"eval_result_k{K}.csv", index=False, encoding="utf-8-sig")
        # 오프라인 재조정용 (K·gate·w): LLM 로그확률과 LightGBM 점수를 저장 (tools/rerank_tune.py)
        np.savez(ROOT / "outputs" / "rerank" / f"eval_lp_k{K}.npz", lp=lp, lgb=lgb, labels=labels,
                 scen=dd["시나리오"].to_numpy(), hangul=dd["hangul"].to_numpy(), halves=halves)

    if args.final:
        tr = cands[pd.Series(in_k(cands), index=cands.index)]
        data = [(pr, lb) for pr, lb, *_ in examples(tr, recs, cat, tok, shuffle=True, seed=1) if lb >= 0]
        print(f"제출용 학습 {len(data)}개", flush=True)
        model = train_lora(base, tok, data, args.epochs, args.lr, args.batch, args.rank)
        # 다운로드 용량을 줄이려고 LoRA 어댑터만 저장 (수십 MB). 로컬에서 기반 모델과 병합: --merge
        out = ROOT / "models" / f"{args.out}-adapter"
        model.save_pretrained(out, safe_serialization=True)
        print(f"저장: {out}")

    if args.merge:
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, ROOT / "models" / f"{args.out}-adapter").merge_and_unload()
        out = ROOT / "models" / args.out
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        print(f"병합 저장: {out}")


if __name__ == "__main__":
    main()
