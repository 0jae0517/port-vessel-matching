"""이름 매칭용 임베딩 파인튜닝 (대조 학습).

목표: '노이즈가 섞인 입출항 선박명(한글·영문)' ↔ '카탈로그 영문명' 을 같은 선박이면 가깝게.
  - 양성 쌍: train 정답 쌍 (쿼리 선박명에 augment 노이즈를 매번 새로 건다)
            + 카탈로그 영문명에 노이즈를 건 합성 쌍 (공식 데이터만 사용)
  - 한글 쌍은 수가 적어(train 1,191행) 여러 번 반복해 비중을 높인다
  - 손실: in-batch negatives + hard negative(이름이 비슷한 다른 선박) InfoNCE
  - 기반 모델: models/e5-small (사전학습 가중치), 결과: models/<out>/

  fold 0 학습·평가: .venv\\Scripts\\python.exe solution\\finetune_emb.py --fold 0 --out e5-small-ft0
  전체 학습:        .venv\\Scripts\\python.exe solution\\finetune_emb.py --out e5-small-ft
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz, process
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import augment as A  # noqa: E402
import match_lib as ml  # noqa: E402

PREFIX = "query: "
DROP_WORD = 0.10  # --drop-word: 학습 쿼리에서 단어 하나를 지울 확률 (v23: 0.3)


def vessel_half(vessel_id: str) -> int:
    """교차 학습용 선박 절반 나누기 (train.py 와 같은 규칙)."""
    return zlib.crc32(str(vessel_id).encode()) % 2


def name_noise(name: str, rng: random.Random) -> str:
    """EDA 규칙 기반 이름 노이즈 (train보다 약간 거칠게): 편집 0~3회, 단어 하나 삭제 DROP_WORD."""
    row = {"vessel_name": name}
    row = A.name_edits(rng.choice([0, 0, 1, 1, 2, 3]))(row, rng)
    if rng.random() < DROP_WORD:
        row = A.drop_word(row, rng)
    return row["vessel_name"]


def hard_negatives(names: list[str], catalog_names: list[str], k: int = 5) -> list[list[int]]:
    """정답 선박명과 철자가 비슷한 다른 카탈로그 선박 (자매선 · 비슷한 이름)."""
    compact = ["".join(ml.tokenize(n)) for n in catalog_names]
    queries = ["".join(ml.tokenize(n)) for n in names]
    unique = list(dict.fromkeys(queries))  # 같은 선박이 여러 쌍에 나오므로 이름마다 한 번만 계산
    cache: dict[str, list[int]] = {}
    for s in range(0, len(unique), 500):
        part = unique[s : s + 500]
        scores = process.cdist(part, compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        top = np.argpartition(scores, scores.shape[1] - (k + 1), axis=1)[:, -(k + 1):]
        for q, row, idx in zip(part, scores, top):
            idx = idx[np.argsort(-row[idx])]
            cache[q] = [int(i) for i in idx if compact[i] != q][:k]
    return [cache[q] for q in queries]


class Encoder(torch.nn.Module):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(path, local_files_only=True)

    def forward(self, enc: dict) -> torch.Tensor:
        hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(vec, dim=-1)


def build_pairs(train: pd.DataFrame, catalog: pd.DataFrame, hangul_repeat: int, n_synth: int, seed: int):
    """(쿼리 원문, 정답 카탈로그 행 번호) 목록. 노이즈는 학습 단계마다 새로 건다."""
    cat_index = {v: i for i, v in enumerate(catalog["vessel_id"])}
    pairs = []
    for row in train.itertuples(index=False):
        reps = hangul_repeat if ml.is_hangul(row.vessel_name) else 1
        pairs += [(str(row.vessel_name), cat_index[row.target_vessel_id])] * reps
    rng = random.Random(seed)
    used = set(cat_index[v] for v in train["target_vessel_id"])
    pool = [i for i in range(len(catalog)) if i not in used]
    for i in rng.sample(pool, min(n_synth, len(pool))):
        pairs.append((str(catalog["vessel_name"].iloc[i]), i))
    return pairs


def evaluate(enc: Encoder, tok, catalog_names: list[str], queries: list[str], truth: np.ndarray, device: str) -> dict:
    """한글 쿼리의 이름만 top-1 (임베딩 단독): 원본 · 편집+2 · 단어 삭제."""
    enc.eval()
    with torch.no_grad():
        C = encode(enc, tok, catalog_names, device)
        out = {}
        for label, fn in {"원본": None, "편집+2": A.name_edits(2), "단어 삭제": A.drop_word}.items():
            qs = [q if fn is None else fn({"vessel_name": q}, A.row_rng(i, label))["vessel_name"] for i, q in enumerate(queries)]
            Q = encode(enc, tok, qs, device)
            S = Q @ C.T
            true_s = S[np.arange(len(truth)), truth][:, None]
            rank = (S > true_s).sum(1)
            out[label] = {"top1": float((rank == 0).mean()), "top10": float((rank < 10).mean())}
    enc.train()
    return out


def encode(enc: Encoder, tok, texts: list[str], device: str, bs: int = 512) -> np.ndarray:
    vecs = []
    for s in range(0, len(texts), bs):
        batch = tok([PREFIX + t for t in texts[s : s + bs]], padding=True, truncation=True, max_length=32,
                    return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            vecs.append(enc(batch).float().cpu().numpy())
    return np.vstack(vecs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=-1, help="0~4: 해당 fold 검증 선박을 빼고 학습·평가, -1: 전체 학습")
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="e5-small")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--hangul-repeat", type=int, default=6)
    p.add_argument("--synth", type=int, default=15000)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--half", type=int, choices=[0, 1], help="교차 학습용: vessel_half()가 이 값인 선박만으로 학습")
    p.add_argument("--drop-word", type=float, default=0.10, help="학습 쿼리 단어 하나 삭제 확률")
    args = p.parse_args()
    global DROP_WORD
    DROP_WORD = args.drop_word

    torch.manual_seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    cat_names = catalog["vessel_name"].astype(str).tolist()

    valid = None
    if args.fold >= 0:
        hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
        splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(train, hangul, train["target_vessel_id"]))
        tr_idx, va_idx = splits[args.fold]
        train, valid = train.iloc[tr_idx], train.iloc[va_idx]
    if args.half is not None:
        # 교차 학습: 이 모델은 나머지 절반 선박의 LightGBM 학습 피처를 계산하는 데 쓰인다
        train = train[train["target_vessel_id"].map(vessel_half) == args.half]

    pairs = build_pairs(train, catalog, args.hangul_repeat, args.synth, seed=42)
    print(f"학습 쌍 {len(pairs)}개 (train {len(train)}행, 한글 반복 {args.hangul_repeat}, 합성 {args.synth})", flush=True)
    t0 = time.time()
    negs = hard_negatives([cat_names[c] for _, c in pairs], cat_names)
    print(f"hard negative 준비 {time.time() - t0:.0f}s", flush=True)

    base = ROOT / "models" / args.base
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    enc = Encoder(base).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(pairs) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    hangul_valid = None
    if valid is not None:
        hv = valid[valid["vessel_name"].map(ml.is_hangul)].drop_duplicates(["vessel_name", "target_vessel_id"])
        cat_index = {v: i for i, v in enumerate(catalog["vessel_id"])}
        hangul_valid = (hv["vessel_name"].astype(str).tolist(), np.array([cat_index[v] for v in hv["target_vessel_id"]]))
        print("기본 모델:", json.dumps(evaluate(enc, tok, cat_names, *hangul_valid, device), ensure_ascii=False), flush=True)

    rng = random.Random(7)
    step = 0
    enc.train()
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for s in range(0, len(order) - args.batch + 1, args.batch):
            idx = order[s : s + args.batch]
            # 같은 배치에 같은 카탈로그 선박이 두 번 들어가면 in-batch negative가 틀리므로 하나만 남긴다
            seen, batch = set(), []
            for i in idx:
                if pairs[i][1] not in seen:
                    seen.add(pairs[i][1])
                    batch.append(i)
            queries = [name_noise(pairs[i][0], rng) for i in batch]
            positives = [cat_names[pairs[i][1]] for i in batch]
            negatives = [cat_names[rng.choice(negs[i])] if negs[i] else cat_names[rng.randrange(len(cat_names))] for i in batch]
            docs = positives + negatives
            q = tok([PREFIX + t for t in queries], padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
            d = tok([PREFIX + t for t in docs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                qv, dv = enc(q), enc(d)
                logits = (qv @ dv.T).float() / args.temp
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if hangul_valid is not None:
            print(f"epoch {epoch} 끝:", json.dumps(evaluate(enc, tok, cat_names, *hangul_valid, device), ensure_ascii=False), flush=True)

    out = ROOT / "models" / args.out
    if out.exists():
        shutil.rmtree(out)
    enc.model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    for extra in ("modules.json", "1_Pooling"):
        src = base / extra
        if src.is_dir():
            shutil.copytree(src, out / extra)
        elif src.exists():
            shutil.copy(src, out / extra)
    stale = ROOT / "outputs" / "emb" / f"{args.out}_catalog.npy"
    if stale.exists():  # 같은 이름으로 다시 학습하면 예전 카탈로그 임베딩 캐시는 무효
        stale.unlink()
    print(f"저장: {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
