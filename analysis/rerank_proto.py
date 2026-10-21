"""LLM 재정렬(listwise) 프로토타입: LightGBM 상위 K개 후보 중 같은 선박을 LLM이 고른다.

입력: outputs/cv/stress_diag_<cfg>.csv (stress_test.py 진단, fold 0 검증 선박 × 노이즈 시나리오, 상위 10개 후보 id·점수)
방법: 검증 선박을 vessel_half로 둘로 나눠, 한쪽으로 kanana LoRA 학습 → 다른 쪽에서 평가 (2-fold, 선박 겹침 없음)
  프롬프트: 기록(선박명·톤수·국적·선종) + 후보 K개(카탈로그명·톤수·국적·선종) → 정답 번호
  점수: 답 위치에서 숫자 토큰 '1'..'K' 의 로그확률 → LightGBM 점수와 결합
평가: LightGBM 단독 vs 재정렬 (전체 / 한글 / 시나리오별), 결합 가중치는 학습 쪽에서 고른다

  실행: .venv\\Scripts\\python.exe analysis\\rerank_proto.py --cfg v25
"""
from __future__ import annotations

import argparse
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

K = 5
SYSTEM = ("You match a Korean port entry record to a standard vessel catalog. The record's vessel name may be "
          "written in Hangul or have typos, and tonnage may be slightly off or missing. Pick the catalog candidate "
          "that is the same vessel. Answer with the candidate number only.")


def fmt_num(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) or v <= 0 else f"{int(round(v))}"


def build_prompt(tok, rec: dict, cands: list[dict]) -> str:
    lines = [f"Record: name={rec['vessel_name']} | GT {fmt_num(rec['gross_tonnage'])} | "
             f"intl GT {fmt_num(rec['international_gross_tonnage'])} | flag {rec.get('flag_name') or '-'} | "
             f"type {rec.get('vessel_type') or '-'}", "Candidates:"]
    for k, c in enumerate(cands, 1):
        lines.append(f"{k}. {c['vessel_name']} | GT {fmt_num(c['gross_tonnage'])} | flag {c['flag_name']} | type {c['vessel_type']}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def vessel_half(vid: str) -> int:
    return zlib.crc32(str(vid).encode()) % 2


def load_examples(cfg: str) -> pd.DataFrame:
    """진단 행 + 노이즈가 걸린 원래 기록(톤수·국적·선종) + 후보 정보."""
    sys.path.insert(0, str(ROOT / "solution"))
    sys.path.insert(0, str(ROOT / "analysis"))
    import stress_test as S
    import train as T
    from augment import apply

    d = pd.read_csv(ROOT / "outputs" / "cv" / f"stress_diag_{cfg}.csv", encoding="utf-8-sig")
    train, catalog = T.load()
    from sklearn.model_selection import StratifiedGroupKFold
    import match_lib as ml

    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    _, va_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=T.SEED)
                     .split(train, hangul, train["target_vessel_id"]))
    va = train.iloc[va_idx]
    frames = {name: apply(va, fn).set_index("sample_id") for name, fn in S.SCENARIOS.items()}
    recs = []
    for r in d.itertuples(index=False):
        recs.append(frames[r.시나리오].loc[r.sample_id].to_dict())
    d["rec"] = recs
    d["cands"] = d["top10_ids"].str.split().map(lambda ids: ids[:K])
    d["lgb"] = d["top10_scores"].astype(str).str.split().map(lambda s: [float(x) for x in s[:K]])
    d["label"] = [c.index(t) if t in c else -1 for c, t in zip(d["cands"], d["truth_id"])]
    d["half"] = d["truth_id"].map(vessel_half)
    return d, catalog.set_index("vessel_id")


def encode_train(tok, prompt: str, label: int) -> tuple[list[int], list[int]]:
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(str(label + 1) + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p + a, [-100] * len(p) + a


def train_lora(base: Path, tok, examples: list[tuple[str, int]], epochs: int, lr: float, batch: int):
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.float16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    steps = epochs * ((len(examples) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rng, t0, step = random.Random(0), time.time(), 0
    model.train()
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            exs = [encode_train(tok, *examples[i]) for i in order[s : s + batch]]
            w = max(len(x) for x, _ in exs)
            ids = torch.tensor([x + [pad] * (w - len(x)) for x, _ in exs], device="cuda")
            lab = torch.tensor([y + [-100] * (w - len(y)) for _, y in exs], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (w - len(x)) for x, _ in exs], device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"    step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def score(model, tok, prompts: list[str]) -> np.ndarray:
    """답 첫 토큰 위치에서 '1'..'K' 로그확률 (행마다 단독 실행)."""
    digit_ids = [tok(str(k), add_special_tokens=False)["input_ids"][0] for k in range(1, K + 1)]
    out = np.zeros((len(prompts), K), dtype=np.float32)
    for i, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**enc).logits[0, -1].float()
        out[i] = torch.log_softmax(logits[digit_ids], dim=-1).cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="v25")
    p.add_argument("--base", default="kanana2b")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--easy-keep", type=float, default=0.15, help="LightGBM이 확신하는(마진 큰) 정답 행 중 학습에 남길 비율")
    p.add_argument("--gate", type=float, default=None, help="LightGBM 마진이 이보다 작은 행만 재정렬 (None: 학습 쪽에서 선택)")
    args = p.parse_args()

    d, cat = load_examples(args.cfg)
    cand_info = lambda ids: [{"vessel_name": cat.at[i, "vessel_name"], "gross_tonnage": cat.at[i, "gross_tonnage"],
                              "flag_name": cat.at[i, "flag_name"], "vessel_type": cat.at[i, "vessel_type"]} for i in ids]
    base = ROOT / "models" / args.base
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    d["prompt"] = [build_prompt(tok, r, cand_info(c)) for r, c in zip(d["rec"], d["cands"])]
    print(f"진단 행 {len(d)}개, 정답이 상위 {K}개 안: {(d.label >= 0).mean():.4f}", flush=True)

    rng = np.random.default_rng(0)
    llm_lp: dict[int, np.ndarray] = {}
    for h in (0, 1):
        tr, te = d[d.half != h], d[d.half == h]
        # 학습: 정답이 상위 K 안에 있는 행. 쉬운(마진 큰) 정답 행은 일부만
        hard = (tr.label > 0) | (tr.margin < 1.0) | tr.hangul
        keep = tr[(tr.label >= 0) & (hard | (rng.random(len(tr)) < args.easy_keep))]
        print(f"[half {h}] 학습 {len(keep)}개 (정답 1위 아님 {int((keep.label > 0).sum())}), 평가 {len(te)}개", flush=True)
        model = train_lora(base, tok, list(zip(keep.prompt, keep.label)), args.epochs, args.lr, args.batch)
        # 평가는 LightGBM이 애매한 행만 (마진 < 3) → 나머지는 LightGBM 그대로
        cand_rows = te[te.margin < 3.0]
        t0 = time.time()
        lp = score(model, tok, cand_rows.prompt.tolist())
        print(f"  평가 {len(cand_rows)}행 LLM 점수 ({time.time() - t0:.0f}s, 행당 {(time.time() - t0) / max(len(cand_rows), 1):.2f}s)", flush=True)
        llm_lp.update(zip(cand_rows.index, lp))
        del model
        torch.cuda.empty_cache()
    d["llm_lp"] = [llm_lp.get(i) for i in d.index]

    # 결합: LightGBM 상위 K 점수 + w × LLM 로그확률, 마진 gate 미만 행만. w·gate는 반대쪽 절반에서 고른다
    def decide(rows: pd.DataFrame, w: float, gate: float) -> np.ndarray:
        out = np.zeros(len(rows), dtype=int)
        for j, r in enumerate(rows.itertuples(index=False)):
            if r.llm_lp is None or r.margin >= gate:
                continue
            out[j] = int(np.argmax(np.asarray(r.lgb) + w * np.asarray(r.llm_lp)))
        return out

    grid = [(w, g) for w in (0.25, 0.5, 1.0, 2.0, 4.0, 100.0) for g in (0.5, 1.0, 2.0, 3.0)]
    final = np.zeros(len(d), dtype=int)
    for h in (0, 1):
        tune, te = d[d.half != h], d[d.half == h]
        best = max(grid, key=lambda wg: (decide(tune, *wg) == tune.label.to_numpy()).sum())
        print(f"[half {h}] 선택한 결합 w={best[0]}, gate={best[1]}")
        final[(d.half == h).to_numpy()] = decide(te, *best)
    d["rr_correct"] = final == d.label.to_numpy()
    d["lgb_correct"] = d.label.to_numpy() == 0

    def acc(g: pd.DataFrame) -> pd.Series:
        hg = g[g.hangul]
        ng = g[~g.hangul]
        return pd.Series({
            "LGB 한글": hg.lgb_correct.mean(), "재정렬 한글": hg.rr_correct.mean(),
            "LGB 비한글": ng.lgb_correct.mean(), "재정렬 비한글": ng.rr_correct.mean(),
            "LGB 가중": 0.7 * hg.lgb_correct.mean() + 0.3 * ng.lgb_correct.mean(),
            "재정렬 가중": 0.7 * hg.rr_correct.mean() + 0.3 * ng.rr_correct.mean(),
        })

    table = d.groupby("시나리오").apply(acc).round(4)
    print("\n===== 시나리오별: LightGBM 단독 vs LLM 재정렬 =====")
    print(table.to_string())
    print("\n전체:", acc(d).round(4).to_dict())
    d.drop(columns=["rec", "prompt"]).to_csv(ROOT / "outputs" / "cv" / f"rerank_proto_{args.cfg}.csv",
                                            index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
