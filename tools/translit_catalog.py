"""카탈로그 영문 선박명 → 한국 항만 기록식 한글 표기 (kanana few-shot, 배치 생성) 사전 계산.

추론 때 LLM을 돌리지 않고, 한글 쿼리와 '카탈로그명의 한글 표기'를 한글끼리 비교하기 위한 표.
  - 카탈로그는 test와 무관한 공식 데이터라 배치로 미리 계산해도 평가 행 독립과 상관없다
  - 출력은 한글·숫자·공백만 (HangulOnly), 결과는 outputs/translit/<모델>.json 에 이어서 저장

  전체:  .venv\\Scripts\\python.exe tools\\translit_catalog.py
  일부:  .venv\\Scripts\\python.exe tools\\translit_catalog.py --names-from outputs\\cv\\stress_diag_v25.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "analysis"))
from synth_hangul import SYSTEM, HangulOnly, shots  # noqa: E402

import re  # noqa: E402

import match_lib as ml  # noqa: E402
from pinyin_hangul import alpha_tokens, split  # noqa: E402


def is_pinyin_for_cn(name: str) -> bool:
    """중국·홍콩 국적 선박용 병음 판별: 영문 토큰이 모두 병음 음절로 나뉘고 2음절 이상 (An Ding Hai, Han Ren).
    국적으로 이미 걸렀으므로 analysis 의 is_pinyin_name 보다 느슨하게 (특징 음절 조건 없음)."""
    toks = alpha_tokens(name)
    return bool(toks) and all(split(t) for t in toks) and sum(len(split(t)) for t in toks) >= 2

# 중국·홍콩 국적의 병음 이름은 LLM이 뜻으로 번역하는 경우가 많아(Feng Xiu Hai → 풍수해) 규칙 표기를 쓴다
PINYIN_FLAGS = {"Peoples' Republic of China", "Hong Kong", "Taiwan"}


def pinyin_rule_variants(name: str, limit: int = 6) -> list[str]:
    """병음 이름 → 한글 표기 여러 개 (공식 표기 + 항만 기록식 영어식 읽기, 음절별 순위 합이 작은 조합 우선)."""
    combos = [(0, "")]
    for tok in re.findall(r"[A-Za-z]+|\d+", name):
        if tok.isdigit():
            opts = [tok]
        else:
            parts = split(tok)
            if not parts:
                return []
            ranked = [(0, "")]
            for syl in parts:
                if syl not in ml.PINYIN_FORWARD:
                    return []  # 규칙에 없는 음절이면 규칙 표기를 쓰지 않는다 (LLM 표기로)
                ranked = sorted({(c + r, s + h) for c, s in ranked for r, h in enumerate(ml.PINYIN_FORWARD[syl])})[: limit * 2]
            opts = [s for _, s in ranked]
        sep = " " if combos[0][1] else ""
        combos = sorted({(c + r, s + sep + o) for c, s in combos for r, o in enumerate(opts)})[: limit * 2]
    return list(dict.fromkeys(s for _, s in combos))[:limit]


SYSTEM_SOUND = (
    "A Korean port authority records vessel names in Hangul by SOUND: each English (or foreign) word is written "
    "the way a Korean would pronounce it, like a loanword. NEVER translate the meaning of a word. "
    "For example Queen is 퀸 (not 여왕), Lucky is 럭키 (not 행운), Eastern is 이스턴 (not 동쪽), Success is 석세스 "
    "(not 성공), Star is 스타 (not 별). Single letters and abbreviations are read letter by letter "
    "(KMTC → 케이엠티씨, MS → 엠에스). Keep numbers as digits. Output only the Hangul name.")
# 뜻으로 번역하기 쉬운 영어 단어가 들어간 train 쌍을 few-shot 예시로 우선 고른다
COMMON_WORDS = {"QUEEN", "SUN", "SUNNY", "STAR", "OCEAN", "GOLDEN", "LUCKY", "GLORY", "SUCCESS", "VICTORY", "PIONEER",
                "BRIGHT", "DREAM", "SPIRIT", "COURAGE", "GRACE", "EXPRESS", "BRIDGE", "HARMONY", "HOPE", "PACIFIC",
                "EASTERN", "ROYAL", "MORNING", "BLUE", "GREEN", "GRAND", "GLOBAL", "NEW", "SEA", "SKY", "WISDOM",
                "TREASURE", "RIVER", "SUPERIOR", "HARVEST", "ACE", "CHEMI", "GAS", "PRINCESS", "LADY", "SMILE"}


def sound_shots(train: pd.DataFrame, catalog: pd.DataFrame, n: int = 20) -> list[tuple[str, str]]:
    """train 한글 기록 ↔ 카탈로그 영문명 중, 흔한 영어 단어가 든 쌍 위주로 few-shot 예시 (공식 데이터)."""
    import random

    name_of = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str)))
    h = train[train["vessel_name"].map(ml.is_hangul)].drop_duplicates("target_vessel_id")
    pairs = [(name_of[t], str(q)) for q, t in zip(h["vessel_name"], h["target_vessel_id"])
             if not ml.is_hangul(name_of[t]) and len(name_of[t]) <= 30]
    common = [p for p in pairs if set(ml.tokenize(p[0])) & COMMON_WORDS]
    rest = [p for p in pairs if p not in common]
    rng = random.Random(2)
    pick = rng.sample(common, min(len(common), n * 3 // 4))
    return pick + rng.sample(rest, n - len(pick))


def sound_score(en: str, ko: str) -> float:
    """한글 표기를 로마자로 되돌려 영문명과 글자 유사도 비교 (뜻 번역 '여왕'은 낮고, 소리 표기 '퀸'은 높다)."""
    from rapidfuzz import fuzz

    target = "".join(ml.tokenize(en))
    toks = ml.tokenize(ko)
    if not toks:
        return 0.0
    variants = ["".join(p) for p in zip(*[ml.romanize_variants(t) if ml.HANGUL_RE.search(t) else [t] * ml.N_VARIANTS
                                           for t in toks])]
    return max(fuzz.ratio(target, v) for v in variants)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="kanana2b")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--names-from", help="stress_diag CSV: 그 행들의 상위 후보 이름만 계산 (빠른 확인용)")
    p.add_argument("--names", nargs="+", help="이 카탈로그 영문명만 계산")
    p.add_argument("--rule-only", action="store_true", help="병음 규칙 표기만 채우고 LLM 생성은 하지 않음 (GPU 불필요)")
    p.add_argument("--redo-llm", action="store_true", help="기존 LLM 표기(문자열 값)를 지우고 다시 계산 (프롬프트를 바꿨을 때)")
    p.add_argument("--beams", type=int, default=4, help="빔 후보 수 (소리 유사도로 고름)")
    p.add_argument("--shots", type=int, default=20, help="few-shot 예시 수 (줄이면 빨라짐)")
    p.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력 (--names 와 함께 프롬프트 확인용)")
    args = p.parse_args()

    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    if args.names:
        names = pd.Series(args.names)
    elif args.names_from:
        d = pd.read_csv(args.names_from, encoding="utf-8-sig")
        d = d[d["hangul"]]
        ids = set(i for s in d["top10_ids"] for i in str(s).split()) | set(d["truth_id"])
        names = catalog[catalog["vessel_id"].isin(ids)]["vessel_name"].astype(str)
    else:
        names = catalog["vessel_name"].astype(str)
    names = list(dict.fromkeys(names))

    out = ROOT / "outputs" / "translit" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, str | list[str]] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if args.redo_llm:
        done = {k: v for k, v in done.items() if isinstance(v, list)}  # 병음 규칙 표기(목록)만 남김
    if args.dry_run:
        done = {}
    # 중국·홍콩 병음 이름: 규칙 표기 목록으로 (LLM 계산 생략)
    py_names = set(catalog.loc[catalog["flag_name"].isin(PINYIN_FLAGS), "vessel_name"].astype(str))
    n_rule = 0
    for n in names:
        if n in py_names and is_pinyin_for_cn(n):
            v = pinyin_rule_variants(n)
            if v:
                done[n] = v
                n_rule += 1
    if not args.dry_run:
        out.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    todo = [n for n in names if n not in done]
    print(f"이름 {len(names)}개: 병음 규칙 표기 {n_rule}개, LLM으로 새로 계산 {len(todo)}개", flush=True)
    if not todo or args.rule_only:
        return

    path = ROOT / "models" / args.model
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    procs = LogitsProcessorList([HangulOnly(tok, [*eos, tok.eos_token_id])])
    # few-shot 예시는 train 한글 선박 (공식 데이터). 같은 프롬프트 앞부분을 모든 이름이 공유
    examples = sound_shots(train, catalog, args.shots)

    def prompt(name: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_SOUND}]
        for en, ko in examples:
            msgs += [{"role": "user", "content": en}, {"role": "assistant", "content": ko}]
        msgs.append({"role": "user", "content": name})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(todo), args.batch):
            part = todo[s : s + args.batch]
            enc = tok([prompt(n) for n in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False, num_beams=args.beams,
                                 num_return_sequences=args.beams, logits_processor=procs, pad_token_id=tok.pad_token_id)
            texts = [" ".join(tok.decode(g, skip_special_tokens=True).split()) for g in gen[:, enc["input_ids"].shape[1]:]]
            for j, n in enumerate(part):
                # 빔 후보 중 영문과 소리가 가장 비슷한 표기 (뜻 번역 걸러내기). 비슷한 것들은 함께 저장 (최대 3개)
                cands = list(dict.fromkeys(x for x in texts[j * args.beams : (j + 1) * args.beams] if x))
                scored = sorted(((sound_score(n, c), c) for c in cands), reverse=True)
                keep = [c for s, c in scored if s >= scored[0][0] - 10][:3] if scored else [""]
                done[n] = keep[0] if len(keep) == 1 else keep
            if not args.dry_run and ((s // args.batch) % 10 == 9 or s + args.batch >= len(todo)):
                out.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
                rate = (s + len(part)) / (time.time() - t0)
                print(f"  {s + len(part)}/{len(todo)} ({time.time() - t0:.0f}s, 남은 약 {(len(todo) - s - len(part)) / rate / 60:.0f}분)",
                      flush=True)
    print("예:", list(done.items())[:10] if not args.dry_run else "")
    if args.dry_run:
        for k, v in done.items():
            print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
