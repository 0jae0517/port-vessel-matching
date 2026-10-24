"""소형 LLM으로 한글 선박명 → 영문 철자 복원 (학습·평가·추론 공용 프롬프트).

평가 행 독립: 이름마다 단독(배치 1)·결정적(beam search, 샘플링 없음)으로 생성하고 이름 단위로 캐시한다.
"""
from __future__ import annotations

import torch

SYSTEM = ("You restore ship names. A Korean port record wrote a vessel's name in Hangul, usually by sound "
          "(transliteration of an English or foreign name). Write the vessel's original name in Latin letters, "
          "in upper case. Output only the name.")


def build_prompt(tok, name: str, shots: list[tuple[str, str]] = ()) -> str:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ko, en in shots:
        msgs += [{"role": "user", "content": ko}, {"role": "assistant", "content": en.upper()}]
    msgs.append({"role": "user", "content": name})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def apply_delta(model, path, sign: int = 1) -> None:
    """LoRA 차이분(tools/lora_delta.py 형식: <모듈>.A, <모듈>.B)을 가중치에 더하거나(+1) 뺀다(-1).

    제출물에는 kanana 기반 모델을 한 번만 넣고, 영문 복원·재정렬 차이분을 메모리에서 바꿔 낀다 (v33).
    """
    from safetensors import safe_open

    f = safe_open(str(path), "pt")
    for key in f.keys():
        if not key.endswith(".A"):
            continue
        name = key[:-2]
        weight = model.get_submodule(name).weight
        delta = f.get_tensor(name + ".B").float().to(weight.device) @ f.get_tensor(key).float().to(weight.device)
        weight.data = (weight.data.float() + sign * delta).to(weight.dtype)


class Transliterator:
    def __init__(self, model_dir, beams: int = 4, shots: list[tuple[str, str]] = (), device: str | None = None,
                 delta=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=dtype).to(self.device).eval()
        if delta is not None:  # 기반 모델 + 영문 복원 LoRA 차이분
            apply_delta(self.model, delta, +1)
        self.beams, self.shots = beams, list(shots)
        self.cache: dict[str, list[str]] = {}

    @torch.no_grad()
    def __call__(self, name: str) -> list[str]:
        if name not in self.cache:
            # 채팅 템플릿이 BOS를 이미 넣으므로 다시 붙이지 않는다 (Llama 계열 kanana; Qwen은 변화 없음)
            enc = self.tok(build_prompt(self.tok, name, self.shots), return_tensors="pt",
                           add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=24, do_sample=False, num_beams=self.beams,
                                      num_return_sequences=self.beams, pad_token_id=self.tok.eos_token_id)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            self.cache[name] = [t.strip().split("\n")[0] for t in texts]
        return self.cache[name]


# ---------------------------------------------------------------- LLM 재정렬 (v33)
# LightGBM 상위 K개 후보 중 같은 선박을 LLM이 고른다 (tools/rerank_train.py 와 같은 프롬프트).
# 학습은 실제 train 기록만 사용. LightGBM 1·2위 점수 차가 작은 행만 다시 판단한다.
RERANK_SYSTEM = ("You match a Korean port entry record to a standard vessel catalog. The record's vessel name may be "
                 "written in Hangul or have small spelling differences, and tonnage may be slightly off or missing. "
                 "Pick the catalog candidate that is the same vessel. Answer with the candidate number only.")
# 후보 10개(v34)는 '10'이 두 토큰이라 A..J 라벨 (tools/rerank_train.py --letters 와 같음)
RERANK_SYSTEM_LETTER = RERANK_SYSTEM.replace("candidate number only", "candidate letter only")
LETTERS = "ABCDEFGHIJ"


def _fmt_num(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if v != v or v <= 0 or v == float("inf") else f"{int(round(v))}"


def _txt(v) -> str:
    return "-" if v is None or (isinstance(v, float) and v != v) or v == "" else str(v)


def rerank_prompt(tok, rec: dict, cands: list[dict], letters: bool = False) -> str:
    lines = [f"Record: name={rec.get('vessel_name')} | GT {_fmt_num(rec.get('gross_tonnage'))} | "
             f"intl GT {_fmt_num(rec.get('international_gross_tonnage'))} | flag {_txt(rec.get('flag_name'))} | "
             f"type {_txt(rec.get('vessel_type'))}", "Candidates:"]
    for k, c in enumerate(cands, 1):
        lines.append(f"{LETTERS[k - 1] if letters else k}. {c['vessel_name']} | GT {_fmt_num(c['gross_tonnage'])} | flag {c['flag_name']} | "
                     f"type {c['vessel_type']}")
    system = RERANK_SYSTEM_LETTER if letters else RERANK_SYSTEM
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Reranker:
    """행 하나의 기록 + 후보 K개 → 후보별 로그확률 (답 첫 토큰 '1'..'K' 또는 'A'..). 행마다 단독 실행."""

    def __init__(self, model, tok, k: int = 5, letters: bool = False):
        self.model, self.tok, self.k, self.letters = model, tok, k, letters
        labels = LETTERS[:k] if letters else [str(i) for i in range(1, k + 1)]
        self.digits = [tok(lb, add_special_tokens=False)["input_ids"][0] for lb in labels]

    @torch.no_grad()
    def __call__(self, rec: dict, cands: list[dict]):
        enc = self.tok(rerank_prompt(self.tok, rec, cands, self.letters), return_tensors="pt", add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits[self.digits[: len(cands)]], dim=-1)
        return lp.cpu().numpy()
