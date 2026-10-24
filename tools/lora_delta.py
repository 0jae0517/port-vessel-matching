"""LoRA 차이분(delta) 만들기: 제출물에 kanana 기반 모델은 한 번만 넣고, 용도별 LoRA 차이분을 메모리에서 바꿔 낀다 (v33).

  translit : 병합 저장된 영문 복원 모델(models/kanana-lora)과 기반 모델(models/kanana2b)의 차이를
             모듈별 SVD로 저랭크(rank 8, LoRA 랭크와 같음) 분해 → 원래 LoRA와 같은 변화 (병합 때 생긴 fp16 반올림만 차이)
  rerank   : peft 어댑터(lora_A, lora_B, alpha/r)를 같은 형식으로 변환
결과 형식: safetensors {"<모듈 이름>.A": [r, in], "<모듈 이름>.B": [out, r]} (B 에 스케일 포함), W' = W + B @ A

  .venv\\Scripts\\python.exe tools\\lora_delta.py translit
  .venv\\Scripts\\python.exe tools\\lora_delta.py rerank
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
OUT = ROOT / "models" / "kanana-deltas"


def translit(rank: int = 8) -> None:
    base = safe_open(ROOT / "models" / "kanana2b" / "model.safetensors", "pt")
    merged = safe_open(ROOT / "models" / "kanana-lora" / "model.safetensors", "pt")
    keys = [k for k in merged.keys() if k.endswith(".weight") and any(f".{t}." in k for t in TARGETS)]
    other = [k for k in merged.keys() if k not in keys]
    t0, out, errs = time.time(), {}, []
    for k in keys:
        d = merged.get_tensor(k).float() - base.get_tensor(k).float()
        u, s, vh = torch.linalg.svd(d, full_matrices=False)
        b = (u[:, :rank] * s[:rank]).contiguous().half()
        a = vh[:rank].contiguous().half()
        name = k[: -len(".weight")]
        out[f"{name}.A"], out[f"{name}.B"] = a, b
        errs.append(float((d - b.float() @ a.float()).norm() / max(d.norm(), 1e-12)))
    # 대상 모듈 밖의 가중치(임베딩·정규화)는 LoRA가 안 건드렸으므로 기반 모델과 같아야 한다 (fp16 반올림 차이만)
    other_diff = max(float((merged.get_tensor(k).float() - base.get_tensor(k).float()).abs().max()) for k in other
                     if k in base.keys())
    OUT.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT / "translit.safetensors"))
    print(f"translit 차이분: 모듈 {len(keys)}개, 상대 오차 최대 {max(errs):.2e} 평균 {sum(errs) / len(errs):.2e}, "
          f"대상 밖 가중치 최대 차이 {other_diff:.2e} ({time.time() - t0:.0f}s)")


def rerank(adapter: Path) -> None:
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(adapter / "adapter_model.safetensors", "pt")
    out = {}
    for k in f.keys():
        if ".lora_A." not in k:
            continue
        name = k.replace("base_model.model.", "", 1).replace(".lora_A.weight", "")
        out[f"{name}.A"] = f.get_tensor(k).half()
        out[f"{name}.B"] = (f.get_tensor(k.replace("lora_A", "lora_B")).float() * scale).half()
    OUT.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT / "rerank.safetensors"))
    print(f"rerank 차이분: 모듈 {len(out) // 2}개, scale {scale}")


if __name__ == "__main__":
    if sys.argv[1] == "translit":
        translit()
    else:
        rerank(Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "rerank" / "result" / "models" / "kanana-rerank-adapter")
