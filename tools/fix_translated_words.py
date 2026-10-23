"""카탈로그 한글 표기표에서 LLM이 '뜻으로 번역'하는 단어만 단어 단위 외래어 표기로 다시 만들어 변형으로 추가.

kanana는 문장(선박명) 단위로 쓰면 익숙한 단어를 번역한다 (QUEEN → 여왕 116회, WISDOM → 지혜, WINNER → 승리).
  1) 표기표에서 영어 단어별 대표 한글 표기를 모으고, 대표 표기가 소리와 안 맞는 단어를 '번역 습관 단어'로 고른다
  2) 그 단어만 '영어 단어를 한국어 외래어 표기로' 프롬프트로 단어 하나씩 생성 (QUEEN → 퀸)
  3) 그 단어가 든 이름마다 번역 단어를 외래어 표기로 바꾼 변형을 목록 맨 앞에 추가 (원래 표기 유지)

  실행: .venv\\Scripts\\python.exe tools\\fix_translated_words.py [--model kanana2b]
  결과: outputs/translit/kanana2b_fixed.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from rapidfuzz import fuzz
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "analysis"))
import match_lib as ml  # noqa: E402
from synth_hangul import HangulOnly  # noqa: E402

SYSTEM = ("Write the given English word as a Korean loanword in Hangul, the way it is commonly written in Korean "
          "(외래어 표기). Do NOT translate the meaning. Output only the Hangul.")
# 외래어 표기 관례 예시 (사람이 쓴 표기 예, 데이터가 아님)
SHOTS = [("STAR", "스타"), ("OCEAN", "오션"), ("GLORY", "글로리"), ("LUCKY", "럭키"), ("KING", "킹"),
         ("PRINCESS", "프린세스"), ("HOPE", "호프"), ("GOLDEN", "골든"), ("BRIDGE", "브리지"), ("SPIRIT", "스피릿")]


def word_sound(en: str, ko: str) -> float:
    if not ml.HANGUL_RE.search(ko):
        return 0.0
    return max(fuzz.ratio(en, v) for v in ml.romanize_variants(ko))


def as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="kanana2b")
    p.add_argument("--min-count", type=int, default=3, help="이 횟수 이상 나온 단어만 검사")
    p.add_argument("--max-sound", type=float, default=35, help="고친 변형을 넣을 때: 원래 단어 표기의 소리 유사도가 이보다 낮을 때만 교체")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--beams", type=int, default=1, help="단어 외래어 표기 빔 수 (첫 후보만 쓰므로 1이면 충분, 늘리면 느려짐)")
    args = p.parse_args()

    src = ROOT / "outputs" / "translit" / "kanana2b.json"
    table = json.loads(src.read_text(encoding="utf-8"))

    # 1) 단어별 대표 표기 (첫 번째 표기에서 단어 수가 맞는 경우만 정렬)
    #    중국·홍콩 병음 이름(규칙 표기)과 대문자 약어(KMTC, CNC)는 제외
    import re

    import pandas as pd

    sys.path.insert(0, str(ROOT / "tools"))
    from translit_catalog import PINYIN_FLAGS, is_pinyin_for_cn

    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    skip_names = set(n for n, f in zip(catalog["vessel_name"].astype(str), catalog["flag_name"])
                     if f in PINYIN_FLAGS and is_pinyin_for_cn(n))
    acronyms = {t.upper() for n in catalog["vessel_name"].astype(str) for t in re.findall(r"[A-Za-z]+", n)
                if t.isupper() and len(t) <= 5}
    counts: dict[str, Counter] = defaultdict(Counter)
    for en, v in table.items():
        if en in skip_names:
            continue
        ko = as_list(v)[0]
        e, k = ml.tokenize(en), ml.tokenize(ko)
        if len(e) == len(k):
            for a, b in zip(e, k):
                if a.isalpha() and len(a) >= 3 and a not in acronyms:
                    counts[a][b] += 1
    words = sorted((w for w, c in counts.items() if sum(c.values()) >= args.min_count), key=lambda w: -sum(counts[w].values()))
    print(f"검사할 영어 단어 {len(words)}개 (등장 {args.min_count}회 이상, 병음·약어 제외)", flush=True)

    # 2) 단어 단위 외래어 표기 생성
    path = ROOT / "models" / args.model
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    procs = LogitsProcessorList([HangulOnly(tok, [*eos, tok.eos_token_id])])

    def prompt(word: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}]
        for e, k in SHOTS:
            msgs += [{"role": "user", "content": e}, {"role": "assistant", "content": k}]
        msgs.append({"role": "user", "content": word})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # 단어별 외래어 표기 캐시 (끊겨도 이어서). GPU 메모리가 넘치면 GTX 1070은 수십 배 느려지므로 배치를 작게
    cache = ROOT / "outputs" / "translit" / "loanwords.json"
    loan_all: dict[str, str] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [w for w in words if w not in loan_all]
    print(f"단어 외래어 표기: 캐시 {len(loan_all)}개, 새로 {len(todo)}개 (배치 {args.batch})", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(todo), args.batch):
            part = todo[s : s + args.batch]
            enc = tok([prompt(w) for w in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=12, do_sample=False, num_beams=args.beams, num_return_sequences=args.beams,
                                 logits_processor=procs, pad_token_id=tok.pad_token_id)
            texts = ["".join(tok.decode(g, skip_special_tokens=True).split()) for g in gen[:, enc["input_ids"].shape[1]:]]
            for j, w in enumerate(part):
                cands = [x for x in texts[j * args.beams : (j + 1) * args.beams] if x]
                if cands:
                    loan_all[w] = cands[0]
            done = s + len(part)
            if (s // args.batch) % 10 == 9 or done == len(todo):
                cache.write_text(json.dumps(loan_all, ensure_ascii=False), encoding="utf-8")
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)} ({time.time() - t0:.0f}s, 남은 약 {(len(todo) - done) / rate / 60:.0f}분)", flush=True)
    # 번역 판정: 대표 표기가 외래어 표기와 크게 다르고(자모 유사도 < 50), 외래어 쪽이 영어 소리에 더 가깝다
    loan: dict[str, str] = {}
    report = []
    for w, l in loan_all.items():
        top, n = counts[w].most_common(1)[0]
        if fuzz.ratio(ml.jamo_compact(top), ml.jamo_compact(l)) < 50 and word_sound(w, l) >= word_sound(w, top) + 10:
            loan[w] = l
            report.append((w, top, l, f"{n}/{sum(counts[w].values())}"))
    report.sort(key=lambda r: -int(r[3].split("/")[1]))
    print(f"번역으로 판정된 단어 {len(loan)}개: {report[:40]}", flush=True)

    # 3) 이름마다 번역 단어를 외래어 표기로 바꾼 변형 추가
    fixed, n_fix, examples = {}, 0, []
    for en, v in table.items():
        variants = as_list(v)
        e_t = ml.tokenize(en)
        if not any(t in loan for t in e_t):
            fixed[en] = v
            continue
        new = []
        for ko in variants:
            k_t = ml.tokenize(ko)
            if len(k_t) == len(e_t):
                out = [loan[a] if a in loan and word_sound(a, b) < args.max_sound else b for a, b in zip(e_t, k_t)]
                if out != k_t:
                    new.append(" ".join(out))
        if not new:  # 단어 수가 안 맞으면(붙여 쓴 표기) 첫 표기에서 번역 단어 문자열을 직접 치환
            ko = variants[0]
            for a in e_t:
                bad = counts[a].most_common(1)[0][0] if a in loan else None
                if bad and bad in ko:
                    ko = ko.replace(bad, " " + loan[a] + " ")
            if ko != variants[0]:
                new.append(" ".join(ko.split()))
        new = [x for x in dict.fromkeys(new) if x not in variants]
        if new:
            n_fix += 1
            if len(examples) < 20:
                examples.append((en, variants[:2], new[0]))
        fixed[en] = (new + variants)[:4] if new else v
    out = ROOT / "outputs" / "translit" / "kanana2b_fixed.json"
    out.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
    print(f"고친 이름 {n_fix}개 → {out}")
    for en, old, new in examples:
        print(f"  {en:28s} {str(old):42s} → {new}")
    for en in ["Sunny Queen", "Southern Queen", "General", "New Wisdom", "Winner 6", "Good Treasure"]:
        print(f"  [확인] {en:18s} {fixed.get(en)}")


if __name__ == "__main__":
    main()
