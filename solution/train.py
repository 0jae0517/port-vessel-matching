"""LightGBM 랭커 학습 · vessel 단위 CV · 제출 폴더 생성.

  CV       : .venv\\Scripts\\python.exe solution\\train.py --cv
  제출 폴더 : .venv\\Scripts\\python.exe solution\\train.py --out outputs\\submissions\\v11\\src --no-dict --callsign --name-lookup

학습 쿼리의 사전(토큰 사전, 국적·선종 확률)은 같은 선박이 빠진 inner fold로 만든다(OOF).
그래야 학습 피처가 '처음 보는 선박'에서의 피처 분포와 같아진다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import shutil
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import match_lib as ml  # noqa: E402

SEED = 42
RANKER_PARAMS = dict(
    objective="lambdarank",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=8,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def load():
    train = pd.read_csv(ROOT / "open" / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
    return train, catalog


NO_DICT = False  # --no-dict: 한글 토큰 사전·전체 이름 사전을 비워 사전 의존도 확인
OOF_TAG = "latest"


USE_VOCAB = False  # --vocab: 카탈로그 단어장 표기 피처 사용
EMB_NAMES: list[str] = []  # --emb: models/<이름> 다국어 임베딩 (쉼표로 여러 개, 예: e5-small,labse)
EMB_ALL = False  # --emb-all: 임베딩을 영문 쿼리에도 적용 (v13·v14 방식). 기본은 한글 쿼리만 (v15)
AUGMENT = 0  # --augment N: 학습 쿼리마다 노이즈 복사본 N개 추가 (v16)
AUG_EXTRA = False  # --aug-extra: 증강에 치환·띄어쓰기·단어 순서 노이즈 추가 (v19)
WORD_INDEX = False  # --word: 단어 단위 후보 검색·피처 (v16)
CROSS_INDEXES: dict | None = None  # 교차 학습 (attach_cross), v21
class _E5Prefix(dict):
    """e5 계열(파인튜닝 모델 e5-small-ft* 포함)은 'query: ' 접두어."""

    def get(self, name, default=""):
        return "query: " if str(name).startswith("e5") else default


EMB_PREFIX = _E5Prefix()


def emb_dir_name(k: int) -> str:
    return "emb" if k == 0 else f"emb{k + 1}"


def attach_embedding(index: ml.CatalogIndex, name: str) -> None:
    """models/<name> 임베딩을 카탈로그 색인에 붙인다. 카탈로그 임베딩은 outputs/emb 캐시를 재사용."""
    import embed_lib

    embedder = embed_lib.NameEmbedder(ROOT / "models" / name, prefix=EMB_PREFIX.get(name, ""))
    cache = ROOT / "outputs" / "emb" / f"{name}_catalog.npy"
    cat_emb = np.load(cache) if cache.exists() else None
    index.attach_embedder(embedder, cat_emb)
    if cat_emb is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, index.cat_emb)


def resources(train: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    res = ml.build_resources(train, catalog)
    if NO_DICT:
        res["token_dict"], res["whole_dict"] = {}, {}
    res["use_vocab"] = USE_VOCAB
    res["emb_hangul_only"] = not EMB_ALL
    res["word_index"] = WORD_INDEX
    return res


def training_matrix(train: pd.DataFrame, catalog: pd.DataFrame, index: ml.CatalogIndex, inner: int = 5):
    xs, ys, groups = [], [], []
    hits = 0
    splitter = GroupKFold(n_splits=inner)
    for fit_idx, prep_idx in splitter.split(train, groups=train["target_vessel_id"]):
        res = resources(train.iloc[fit_idx], catalog)
        part = train.iloc[prep_idx]
        if AUGMENT:
            # 노이즈를 건 복사본을 함께 학습: 거친 쿼리에서 무엇을 믿을지 모델이 배우게 한다
            import augment as aug_mod

            aug_mod.EXTRA_NOISE = AUG_EXTRA
            part = pd.concat([part, aug_mod.augment(part, AUGMENT)], ignore_index=True)
        # 교차 학습: 파인튜닝 임베딩은 그 선박을 학습하지 않은 절반 모델로 피처를 계산 (누수 방지)
        if CROSS_INDEXES:
            halves = part["target_vessel_id"].map(vessel_half).to_numpy()
            chunks = [(part[halves == h], CROSS_INDEXES[h]) for h in (0, 1)]
        else:
            chunks = [(part, index)]
        for chunk, idx in chunks:
            if chunk.empty:
                continue
            rows = chunk.to_dict("records")
            queries = [ml.prepare_query(r, res) for r in rows]
            truth = [idx.id_to_index[r["target_vessel_id"]] for r in rows]
            # 정답이 후보에 없으면 학습용으로만 추가
            hit_flags: list[bool] = []
            cands, feats = ml.candidates_and_features(queries, idx, res, force=truth, hits=hit_flags)
            hits += sum(hit_flags)
            for cand, t in zip(cands, truth):
                ys.append((cand == t).astype(np.int8))
                groups.append(len(cand))
            xs.append(feats)
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups), hits / len(groups)  # 증강 행 포함


def vessel_half(vessel_id: str) -> int:
    """교차 학습용 선박 절반 나누기 (finetune_emb.py 와 같은 규칙)."""
    return zlib.crc32(str(vessel_id).encode()) % 2


def attach_cross(names_by_half: dict[int, str], other: list[str]) -> dict[int, ml.CatalogIndex]:
    """CROSS_INDEXES 만들기: 절반 h 선박의 피처는 '나머지 절반(1-h)으로 학습한 모델'로 계산한다.

    names_by_half: {0: 절반 0으로 학습한 모델 이름, 1: 절반 1로 학습한 모델 이름}, other: 함께 쓸 임베딩(LaBSE 등)
    """
    _, catalog = load()
    out = {}
    for h in (0, 1):
        idx = ml.CatalogIndex(catalog)
        for name in [names_by_half[1 - h], *other]:
            attach_embedding(idx, name)
        out[h] = idx
    return out


# 단조 제약: 유사도·국적·선종 확률은 높을수록(+), 톤수 차이·번호 충돌은 클수록(-) 점수가 낮아지도록 강제
MONOTONE_UP = {
    "s_compact", "s_skel_rom", "s_skel_trans", "s_token_set", "s_token_sort", "s_partial", "s_jw", "s_rom_compact",
    "exact", "first_eq", "num_equal", "gt_exact", "flag_prob", "type_prob",
    "s_var_ratio", "s_var_jw", "s_var_partial", "s_learn_ratio", "s_learn_top1", "s_learn_jw",
    "s_vocab_ratio", "s_vocab_jw", "s_alpha_ratio", "s_emb", "s_emb2", "word_hit",
}
MONOTONE_DOWN = {"num_conflict", "gap_gross", "gap_intl", "gap_best", "num_q_only", "num_c_only"}
USE_MONOTONE = True


def ranker_params(seed: int = SEED) -> dict:
    params = dict(RANKER_PARAMS, random_state=seed)
    if USE_MONOTONE:
        params["monotone_constraints"] = [
            1 if f in MONOTONE_UP else -1 if f in MONOTONE_DOWN else 0 for f in ml.FEATURE_NAMES
        ]
    return params


def fit(train, catalog, index):
    x, y, groups, recall = training_matrix(train, catalog, index)
    model = lgb.LGBMRanker(**ranker_params())
    model.fit(x, y, group=groups, feature_name=list(ml.FEATURE_NAMES))
    return model, recall


# ---------------------------------------------------------------- 앙상블
CAT_TASK_TYPE = "GPU"
MEMBERS = {"lgb42": ("lgb", 42), "lgb7": ("lgb", 7), "lgb2024": ("lgb", 2024), "cat": ("cat", 42)}
COMBOS = {
    "lgb42": ["lgb42"],
    "lgb3": ["lgb42", "lgb7", "lgb2024"],
    "cat": ["cat"],
    "lgb42+cat": ["lgb42", "cat"],
    "lgb3+cat": ["lgb42", "lgb7", "lgb2024", "cat"],
}


def train_member(kind: str, seed: int, x, y, groups) -> tuple[str, object]:
    if kind == "lgb":
        model = lgb.LGBMRanker(**ranker_params(seed)).fit(x, y, group=groups, feature_name=list(ml.FEATURE_NAMES))
        return "lgb", model.booster_.model_to_string()
    from catboost import CatBoostRanker

    group_id = np.repeat(np.arange(len(groups)), groups)
    # 학습만 로컬 GPU로 (추론은 평가 서버 CPU에서 그대로 동작)
    model = CatBoostRanker(
        loss_function="YetiRank", iterations=600, learning_rate=0.08, depth=6,
        random_seed=seed, thread_count=8, verbose=0, allow_writing_files=False,
        task_type=CAT_TASK_TYPE,
    )
    model.fit(x, y, group_id=group_id)
    return "cat", model


def fit_members(train, catalog, index, names: list[str]) -> tuple[dict, float]:
    x, y, groups, recall = training_matrix(train, catalog, index)
    members = {}
    for name in names:
        t0 = time.time()
        members[name] = train_member(*MEMBERS[name], x, y, groups)
        print(f"    {name} trained ({time.time()-t0:.0f}s)", flush=True)
    return members, recall


def run_cv_ensemble(train, catalog, index) -> None:
    """같은 fold·같은 피처로 단일 모델과 앙상블 조합을 함께 비교."""
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {name: np.empty(len(train), dtype=object) for name in COMBOS}
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, hangul, train["target_vessel_id"])):
        t0 = time.time()
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        members, _ = fit_members(tr, catalog, index, list(MEMBERS))
        res = resources(tr, catalog)
        queries = [ml.prepare_query(r, res) for r in va.to_dict("records")]
        cands, feats = ml.candidates_and_features(queries, index, res)
        sizes = [len(c) for c in cands]
        z = {name: ml.RankEnsemble([m]).predict_grouped(feats, sizes) for name, m in members.items()}
        line = []
        for combo, names in COMBOS.items():
            total = sum(z[n] for n in names)
            preds, offset = [], 0
            for cand in cands:
                preds.append(index.ids[cand[int(np.argmax(total[offset : offset + len(cand)]))]])
                offset += len(cand)
            oof[combo][va_idx] = preds
            line.append(f"{combo}={score(va, preds)['weighted']:.4f}")
        print(f"fold {fold}: " + " ".join(line) + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n===== OOF (vessel GroupKFold, 5 folds) =====")
    for combo in COMBOS:
        s = score(train, list(oof[combo]))
        print(f"{combo:10s} weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} non_hangul={s['non_hangul']:.4f}")
    out = ROOT / "outputs" / "cv"
    for combo in COMBOS:
        correct = oof[combo] == train["target_vessel_id"].to_numpy()
        train.assign(pred_vessel_id=oof[combo], correct=correct, is_hangul=hangul).to_csv(
            out / f"{OOF_TAG}_{combo.replace('+', '_')}_oof.csv", index=False, encoding="utf-8-sig")


def score(truth: pd.DataFrame, pred: list[str]) -> dict:
    hangul = truth["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    correct = truth["target_vessel_id"].to_numpy() == np.asarray(pred)
    h, nh = correct[hangul].mean(), correct[~hangul].mean()
    return {"weighted": 0.7 * h + 0.3 * nh, "hangul": h, "non_hangul": nh}


def run_cv(train, catalog, index) -> None:
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.empty(len(train), dtype=object)
    recall_hits = np.zeros(len(train), dtype=bool)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, hangul, train["target_vessel_id"])):
        t0 = time.time()
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        model, train_recall = fit(tr, catalog, index)
        res = resources(tr, catalog)
        cands: list = []
        pred = ml.predict(model, va, index, res, candidates_out=cands)
        oof[va_idx] = pred
        for j, (cand, tid) in enumerate(zip(cands, va["target_vessel_id"])):
            recall_hits[va_idx[j]] = index.id_to_index[tid] in set(cand.tolist())
        s = score(va, pred)
        print(f"fold {fold}: weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} "
              f"non_hangul={s['non_hangul']:.4f} (train recall {train_recall:.3f}, {time.time()-t0:.0f}s)", flush=True)
    s = score(train, list(oof))
    print("\n===== OOF (vessel GroupKFold, 5 folds) =====")
    print(f"weighted   : {s['weighted']:.4f}")
    print(f"hangul     : {s['hangul']:.4f}")
    print(f"non_hangul : {s['non_hangul']:.4f}")
    print(f"candidate recall  hangul={recall_hits[hangul].mean():.4f}  non_hangul={recall_hits[~hangul].mean():.4f}")
    out = ROOT / "outputs" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    correct = oof == train["target_vessel_id"].to_numpy()
    train.assign(pred_vessel_id=oof, correct=correct, candidate_hit=recall_hits, is_hangul=hangul).to_csv(
        out / f"{OOF_TAG}_oof.csv", index=False, encoding="utf-8-sig")


def run_rowcheck(train, catalog, index) -> None:
    """행 단위 랜덤 분할 1 fold: test에 train과 같은 선박이 섞인 상황에서 조회 규칙 효과 확인."""
    from sklearn.model_selection import KFold

    tr_idx, va_idx = next(KFold(n_splits=5, shuffle=True, random_state=SEED).split(train))
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    model, _ = fit(tr, catalog, index)
    res = resources(tr, catalog)
    seen = va["target_vessel_id"].isin(set(tr["target_vessel_id"])).to_numpy()
    print(f"valid {len(va)}행 중 train에 있는 선박 {seen.mean():.3f}")
    for name, kw in {
        "model only": {},
        "+ name lookup": {"use_name_lookup": True},
        "+ callsign": {"use_callsign": True},
        "+ both": {"use_name_lookup": True, "use_callsign": True},
    }.items():
        pred = np.asarray(ml.predict(model, va, index, res, **kw))
        s = score(va, list(pred))
        correct = pred == va["target_vessel_id"].to_numpy()
        print(f"{name:15s} weighted={s['weighted']:.4f} hangul={s['hangul']:.4f} non_hangul={s['non_hangul']:.4f}"
              f" | seen acc={correct[seen].mean():.4f} unseen acc={correct[~seen].mean():.4f}")


def build_submission(train, catalog, index, out_dir: Path, use_callsign: bool, use_name_lookup: bool,
                     ensemble: str | None = None) -> None:
    if ensemble:
        members, recall = fit_members(train, catalog, index, COMBOS[ensemble])
        model = None
    else:
        model, recall = fit(train, catalog, index)
    res = resources(train, catalog)
    res["use_callsign"] = use_callsign
    res["use_name_lookup"] = use_name_lookup
    # 제출 폴더의 model/emb, model/emb2 … 와 각 접두어 (script.py가 이 순서대로 로드)
    res["emb_dirs"] = [(emb_dir_name(k), EMB_PREFIX.get(n, "")) for k, n in enumerate(EMB_NAMES)]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)
    if ensemble:
        with open(out_dir / "model" / "ensemble.pkl", "wb") as f:
            pickle.dump(ml.RankEnsemble([members[n] for n in COMBOS[ensemble]]), f)
    else:
        # LightGBM C++ 저장은 한글 경로를 못 열어서 문자열로 받아 Python으로 쓴다
        (out_dir / "model" / "lgb_ranker.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
    with open(out_dir / "model" / "resources.pkl", "wb") as f:
        pickle.dump(res, f)
    shutil.copy(ROOT / "solution" / "match_lib.py", out_dir / "model" / "match_lib.py")
    if EMB_NAMES:
        # 평가 서버는 오프라인 → 가중치를 model/emb* 에 동봉하고 로컬 경로로만 로드
        shutil.copy(ROOT / "solution" / "embed_lib.py", out_dir / "model" / "embed_lib.py")
    for k, name in enumerate(EMB_NAMES):
        shutil.copytree(ROOT / "models" / name, out_dir / "model" / emb_dir_name(k),
                        ignore=shutil.ignore_patterns(".cache", "*.lock", "*.metadata", ".gitignore"))
    shutil.copy(ROOT / "solution" / "script.py", out_dir / "script.py")
    (out_dir / "requirements.txt").write_text(
        "# 평가 서버 기본 설치 패키지(pandas, numpy, lightgbm, rapidfuzz, Unidecode)만 사용\n", encoding="utf-8")
    print(f"saved {out_dir} (train candidate recall {recall:.3f}, "
          f"use_callsign={use_callsign}, use_name_lookup={use_name_lookup})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--rowcheck", action="store_true", help="행 단위 랜덤 분할로 조회 규칙 효과 확인")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--callsign", action="store_true", help="train call_sign 조회로 예측 덮어쓰기")
    parser.add_argument("--name-lookup", action="store_true", help="train 선박명 조회로 예측 덮어쓰기")
    parser.add_argument("--no-dict", action="store_true", help="한글 토큰 사전 없이 실행")
    parser.add_argument("--tag", default="latest", help="CV 결과 파일 이름 (outputs/cv/<tag>_oof.csv)")
    parser.add_argument("--ensemble-cv", action="store_true", help="단일·앙상블 조합 CV 비교")
    parser.add_argument("--ensemble", choices=list(COMBOS), help="제출 폴더를 이 앙상블 조합으로 생성")
    parser.add_argument("--vocab", action="store_true", help="카탈로그 단어장 표기 피처 사용 (에이스 → ACE)")
    parser.add_argument("--emb", help="다국어 임베딩 모델 이름 (models/<이름>, 쉼표로 여러 개, 예: e5-small,labse)")
    parser.add_argument("--emb-all", action="store_true", help="임베딩을 영문 쿼리에도 적용 (v13·v14 재현용)")
    parser.add_argument("--augment", type=int, default=0, help="학습 쿼리마다 노이즈 복사본 N개 추가")
    parser.add_argument("--word", action="store_true", help="단어 단위 후보 검색·피처 사용")
    parser.add_argument("--aug-extra", action="store_true", help="증강에 치환·띄어쓰기·단어 순서 노이즈 추가")
    parser.add_argument("--cross", help="교차 학습 절반 모델 '절반0 모델,절반1 모델' (--emb 첫 모델 자리를 대신해 학습 피처 계산, v21)")
    args = parser.parse_args()
    global NO_DICT, OOF_TAG, USE_VOCAB, EMB_NAMES, EMB_ALL, AUGMENT, WORD_INDEX, AUG_EXTRA, CROSS_INDEXES
    NO_DICT, OOF_TAG, USE_VOCAB, EMB_ALL = args.no_dict, args.tag, args.vocab, args.emb_all
    AUGMENT, WORD_INDEX, AUG_EXTRA = args.augment, args.word, args.aug_extra
    EMB_NAMES = [n.strip() for n in (args.emb or "").split(",") if n.strip()]
    train, catalog = load()
    index = ml.CatalogIndex(catalog)
    for name in EMB_NAMES:
        attach_embedding(index, name)
    if args.cross:
        h0, h1 = [n.strip() for n in args.cross.split(",")]
        CROSS_INDEXES = attach_cross({0: h0, 1: h1}, EMB_NAMES[1:])
    if args.cv:
        run_cv(train, catalog, index)
    if args.rowcheck:
        run_rowcheck(train, catalog, index)
    if args.ensemble_cv:
        run_cv_ensemble(train, catalog, index)
    if args.out:
        build_submission(train, catalog, index, args.out.resolve(), args.callsign, args.name_lookup, args.ensemble)


if __name__ == "__main__":
    main()
