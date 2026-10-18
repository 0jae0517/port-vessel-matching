"""다국어 사전학습 임베딩 (오프라인 · 로컬 경로 전용).

평가 서버는 인터넷이 막혀 있으므로 모델은 반드시 로컬 폴더 경로로만 로드한다.
sentence-transformers는 서버에 없으므로 풀링(CLS/mean) · Dense · 정규화를 직접 구현한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


class NameEmbedder:
    def __init__(self, model_dir: str | Path, prefix: str = "", device: str | None = None) -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"모델 폴더가 없음 (오프라인 로드만 지원): {model_dir}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prefix = prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        pooling = model_dir / "1_Pooling" / "config.json"
        mode = json.loads(pooling.read_text()) if pooling.exists() else {}
        self.cls = bool(mode.get("pooling_mode_cls_token", False))
        self.dense = None
        dense_dir = model_dir / "2_Dense"
        if dense_dir.exists():  # LaBSE: CLS → Dense(tanh) → 정규화
            from safetensors.torch import load_file

            weights = load_file(str(next(dense_dir.glob("*.safetensors"))))
            layer = torch.nn.Linear(weights["linear.weight"].shape[1], weights["linear.weight"].shape[0])
            layer.load_state_dict({"weight": weights["linear.weight"], "bias": weights["linear.bias"]})
            self.dense = layer.to(self.device).eval()
            if self.device == "cuda":
                self.dense.half()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = [self.prefix + str(t) for t in texts[start : start + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=32, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            if self.cls:
                vec = hidden[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            if self.dense is not None:
                vec = torch.tanh(self.dense(vec))
            vec = torch.nn.functional.normalize(vec.float(), dim=-1)
            out.append(vec.cpu().numpy())
        return np.vstack(out).astype(np.float32)
