"""Qwen2.5 LoRA 미세조정: 한글 선박명(노이즈 포함) → 카탈로그 영문명.

  - 학습 쌍: train 한글 행 (쿼리 선박명 → 정답 카탈로그 영문명), 매 epoch 쿼리에 이름 편집 노이즈를 새로 건다
  - 프롬프트는 llm_lib.build_prompt (few-shot 없음), 손실은 답 토큰에만
  - 저장: LoRA를 원래 가중치에 합쳐(merge) models/<out>/ 에 fp16으로 → 추론에 peft 불필요

  fold 0 학습: .venv\\Scripts\\python.exe solution\\finetune_llm.py --base qwen3b --fold 0 --out qwen3b-lora0
  평가:        .venv\\Scripts\\python.exe analysis\\llm_translit_probe.py qwen3b-lora0 --shots 0
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import augment as A  # noqa: E402
import match_lib as ml  # noqa: E402
from llm_lib import build_prompt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def vessel_half(vessel_id: str) -> int:
    return zlib.crc32(str(vessel_id).encode()) % 2


def noisy(name: str, rng: random.Random) -> str:
    """EDA 규칙 기반 이름 편집 (옆 글자 중복·삭제·자리바꿈) 0~2회."""
    return A.name_edits(rng.choice([0, 0, 0, 1, 1, 2]))({"vessel_name": name}, rng)["vessel_name"]


def encode_example(tok, query: str, target: str) -> tuple[list[int], list[int]]:
    prompt = tok(build_prompt(tok, query), add_special_tokens=False)["input_ids"]
    answer = tok(target.upper() + tok.eos_token, add_special_tokens=False)["input_ids"]
    return prompt + answer, [-100] * len(prompt) + answer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="qwen3b")
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--half", type=int, choices=[0, 1])
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(42)
    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    if args.fold >= 0:
        hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
        splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(train, hangul, train["target_vessel_id"]))
        train = train.iloc[splits[args.fold][0]]
    if args.half is not None:
        train = train[train["target_vessel_id"].map(vessel_half) == args.half]
    name_of = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str)))
    h = train[train["vessel_name"].map(ml.is_hangul)]
    pairs = [(str(q), name_of[t]) for q, t in zip(h["vessel_name"], h["target_vessel_id"]) if not ml.is_hangul(name_of[t])]
    print(f"학습 쌍 {len(pairs)}개 (고유 이름 {len(set(pairs))})", flush=True)

    base = ROOT / "models" / args.base
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.float16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for prm in model.parameters():  # LoRA 가중치만 fp32 (fp16 기본 가중치는 고정)
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()

    params = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * ((len(pairs) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(7)
    t0, step = time.time(), 0
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        total = 0.0
        for s in range(0, len(order), args.batch):
            exs = [encode_example(tok, noisy(pairs[i][0], rng), pairs[i][1]) for i in order[s : s + args.batch]]
            width = max(len(x) for x, _ in exs)
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in exs], device="cuda")
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in exs], device="cuda")
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in exs], device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            total += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        print(f"epoch {epoch} 평균 loss {total / ((len(order) + args.batch - 1) // args.batch):.4f}", flush=True)

    model = model.merge_and_unload()
    for prm in model.parameters():
        prm.data = prm.data.half()
    out = ROOT / "models" / args.out
    if out.exists():
        shutil.rmtree(out)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    if (base / "generation_config.json").exists():
        shutil.copy(base / "generation_config.json", out / "generation_config.json")
    print(f"저장: {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
