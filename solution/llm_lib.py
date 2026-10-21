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


class Transliterator:
    def __init__(self, model_dir, beams: int = 4, shots: list[tuple[str, str]] = (), device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=dtype).to(self.device).eval()
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
