# 항만 입출항 기록 ↔ 선박 제원 카탈로그 매칭

- **대회**: 항만 문제해결을 위한 AI 기술 공모전 (DACON)
- **주제**: AI를 활용한 선박 정보 표기 차이 해소
- **과제**: 입출항 기록 1행(선박명·호출부호·톤수·국적·선종) → 선박 제원 카탈로그 78,595척 중 같은 선박의 `vessel_id` 예측
- **평가**: `0.7 × 한글 기록 정확도 + 0.3 × 비한글 기록 정확도`
- **대회 규칙·평가·데이터 설명**: [`docs/`](docs/)

- **2026-10-19 평가 방식 변경**: 평가 데이터 추가 + Private 도입, 리더보드 초기화 ([`docs/05_leaderboard_update.md`](docs/05_leaderboard_update.md))

| | 새 Public (10/19~) | 서버 추론 | 이전 Public (3,000건) |
|---|---|---|---|
| 베이스라인 (RandomForest) | – | – | 0.48066 |
| v11 | 0.88829 | – | 0.992444 |
| v14 | 0.90165 | 3분 52초 | 0.996778 |
| v16 | 0.91379 | 4분 12초 | – |
| v21 (파인튜닝 e5 + 교차 학습) | 0.91740 | 4분 13초 | – |
| v22 (v21 + LLM 영문 복원) | 0.91799 | 10분 22초 | – |
| **현재 최고: v25** (kanana LoRA 영문 복원 + 병음 역변환) | **0.92079** | 9분 26초 | – |

---

## 빠른 시작

### 1. 환경 (평가 서버와 같은 버전)

- 평가 서버: Ubuntu 22.04 · Python 3.11.17 · CUDA 12.8 · L4 GPU
- 로컬도 Python 3.11.17 64bit로 통일

```bash
python -m pip install uv
python -m uv python install 3.11.17
python -m uv venv .venv --python 3.11.17
python -m uv pip install --python .venv/Scripts/python.exe -r requirements-local.txt
python -m uv pip install --python .venv/Scripts/python.exe -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

- `requirements-torch.txt`: 다국어 임베딩(v13~)에 필요 (torch·transformers, 서버 기본 설치와 같은 버전)
- Windows에서 `uv venv`가 "minor version link" 오류를 낼 경우 → 받은 파이썬으로 직접 생성
  - `%APPDATA%\uv\python\cpython-3.11.17-windows-x86_64-none\python.exe -m venv .venv`
- 일반 터미널에서 `No Python at ...` 오류가 날 경우 → `.venv\pyvenv.cfg`의 `home` 경로 확인
  - Claude 데스크톱 앱에서 설치한 파이썬은 앱 격리 폴더(`AppData\Local\Packages\Claude_…`)에 저장되어 일반 터미널에서 보이지 않음
  - 이 PC는 격리 밖 `D:\dev\python\cpython-3.11.17`로 복사해 `home`을 바꿔 둠

### 2. 데이터 · 모델 배치 (저장소 미포함)

- 대회 데이터: 대회 규칙상 재배포 금지 → DACON에서 각자 내려받아 배치

```
open/train.csv
open/vessel_catalog.csv
open/sample_submission.csv
baseline_submit/            # baseline_submit.zip 압축 해제 (베이스라인 비교용, 선택)
```

- 다국어 임베딩 가중치: 공개 범용 사전학습 모델 (대회 규칙상 허용)

```bash
.venv\Scripts\python.exe tools\download_models.py e5-small labse
```

- 큰 가중치 파일이 0바이트에서 멈추면 `curl`로 직접 받기 (HuggingFace Xet 전송 문제)
  - `curl.exe -L --retry 10 -C - -o models\labse\model.safetensors https://huggingface.co/sentence-transformers/LaBSE/resolve/main/model.safetensors`

### 3. 검증 · 제출 파일 생성

```bash
# vessel 단위 5-fold CV
.venv\Scripts\python.exe solution\train.py --cv --no-dict --emb e5-small,labse --emb-all --tag v14_nodict

# 전체 train으로 학습 → 제출 폴더 생성 (현재 최고 v14 설정)
.venv\Scripts\python.exe solution\train.py --out outputs\submissions\v14\src --no-dict --emb e5-small,labse --emb-all --callsign --name-lookup

# 대회 규칙 검사 + 오프라인 서버 흉내 실행 + submit_v14.zip 생성 (zip 이름 = 폴더 이름)
.venv\Scripts\python.exe tools\make_submit.py outputs\submissions\v14\src outputs\submissions\v14
```

- `make_submit.py` 결과 FAIL 0건일 때만 제출
- 소요 시간: 학습 약 6분 · CV 약 30분 · 추론 로컬(GTX 1070) 약 270초 / 서버(L4) 약 2분 예상
- `submit_v14.zip` 약 1.9GB (임베딩 가중치 동봉)
- zip 이름은 `submit_<버전>.zip` (버전 = 출력 폴더 이름, `--name`으로 지정 가능). DACON은 파일 이름이 아니라 zip 내부 구조만 검사

### `train.py` 옵션

| 옵션 | 설명 |
|---|---|
| `--cv` | vessel 단위 StratifiedGroupKFold 5-fold CV, 결과 `outputs/cv/<tag>_oof.csv` |
| `--tag` | CV 결과 파일 이름 |
| `--no-dict` | 한글 토큰 사전·전체 이름 사전 끔 (**v5부터 기본 사용**) |
| `--out DIR` | 전체 train으로 학습 → 제출 폴더(`script.py`, `requirements.txt`, `model/`) 생성 |
| `--emb NAME[,NAME2]` | `models/<NAME>` 다국어 임베딩 사용 (최대 2개), 제출 시 가중치를 `model/emb/`, `model/emb2/`에 동봉 |
| `--emb-all` | 임베딩을 영문 쿼리에도 적용 (v13·v14). 빼면 한글 쿼리에만 적용 |
| `--callsign` / `--name-lookup` | train에 같은 호출부호·선박명이 있으면 그 정답으로 덮어쓰기 (test 효과 없음, 손해도 없음) |
| `--rowcheck` | 행 단위 랜덤 분할로 조회 규칙 효과 확인 |
| `--ensemble-cv` / `--ensemble NAME` | LightGBM 시드·CatBoost 앙상블 비교 / 앙상블 제출 생성 (미채택) |
| `--vocab` | 카탈로그 단어장 표기 피처 사용 (v12, Public 하락으로 미채택) |
| `--augment N` | 학습 쿼리마다 노이즈 복사본 N개 추가 (`solution/augment.py`, v16) |
| `--word` | 단어 단위 후보 검색·피처 (단어 하나가 빠진 쿼리 대응, v16) |
| `--aug-extra` | 증강에 글자 치환·띄어쓰기·단어 순서 노이즈 추가 (v19) |
| `--cross H0,H1` | 교차 학습: LightGBM 학습 피처를 절반 모델로 계산 (`--emb` 첫 모델 자리, 제출용. v21) |

### 파인튜닝 임베딩 (v21)

```bash
# 전체 train으로 e5-small 파인튜닝 + 교차 학습용 절반 모델 2개 (각 약 10분, GPU)
.venv\Scripts\python.exe solution\finetune_emb.py --out e5-small-ft --epochs 3
.venv\Scripts\python.exe solution\finetune_emb.py --half 0 --out e5-small-fth0 --epochs 3
.venv\Scripts\python.exe solution\finetune_emb.py --half 1 --out e5-small-fth1 --epochs 3

# 제출 폴더: 추론은 e5-small-ft, 학습 피처는 절반 모델로 (v16 설정 + 파인튜닝)
.venv\Scripts\python.exe solution\train.py --out outputs\submissions\v21\src --no-dict --emb e5-small-ft,labse --emb-all --callsign --name-lookup --augment 1 --word --cross e5-small-fth0,e5-small-fth1
```

- `--fold 0`: fold 0 검증 선박을 빼고 학습 + 임베딩 단독 한글 top-1 평가 (스트레스 테스트용 `e5-small-ft0`, `e5-small-ft0h0/h1`)
- 저장 시 토크나이저 정규식 경고가 뜨지만 기본 e5-small과 토큰화 결과 동일 (무시 가능)

### LLM 영문 복원 (v25·v26, 코드 검증용 전체 재현 경로)

- DACON 답변: LLM LoRA 미세조정 + 가중치 동봉 오프라인 추론 허용, 코드 검증 때 학습·전처리·추론 전체 코드와 가중치 요청 가능 ([docs/06_qna.md](docs/06_qna.md))
- 기반 모델: kakaocorp/kanana-1.5-2.1b-instruct-2505 (Apache 2.0). 학습 데이터는 `train.csv` 한글 행 → 정답 카탈로그 영문명뿐

```bash
# 0. 사전학습 가중치 (인터넷 필요, 로컬 1회)
.venv\Scripts\python.exe tools\download_models.py e5-small labse kanana2b

# 1. e5-small 파인튜닝 (전체 + 교차 학습용 절반 2개)
.venv\Scripts\python.exe solution\finetune_emb.py --out e5-small-ft --epochs 3
.venv\Scripts\python.exe solution\finetune_emb.py --half 0 --out e5-small-fth0 --epochs 3
.venv\Scripts\python.exe solution\finetune_emb.py --half 1 --out e5-small-fth1 --epochs 3

# 2. kanana 가벼운 LoRA (전체 + 절반 2개, LoRA는 원래 가중치에 병합해 저장)
.venv\Scripts\python.exe solution\finetune_llm.py --base kanana2b --out kanana-lora --epochs 1 --lr 1e-4 --rank 8
.venv\Scripts\python.exe solution\finetune_llm.py --base kanana2b --half 0 --out kanana-lorah0 --epochs 1 --lr 1e-4 --rank 8
.venv\Scripts\python.exe solution\finetune_llm.py --base kanana2b --half 1 --out kanana-lorah1 --epochs 1 --lr 1e-4 --rank 8

# 3. LightGBM 랭커 학습 + 제출 폴더 (v26; v25는 끝에 --pinyin 추가)
.venv\Scripts\python.exe solution\train.py --out outputs\submissions\v26\src --no-dict --emb e5-small-ft,labse --emb-all --callsign --name-lookup --augment 1 --word --cross e5-small-fth0,e5-small-fth1 --llm kanana-lora --llm-cross kanana-lorah0,kanana-lorah1

# 4. 규칙 검사 + 서버 흉내 실행 + zip
.venv\Scripts\python.exe tools\make_submit.py outputs\submissions\v26\src outputs\submissions\v26
```

- 학습 피처용 LLM 결과는 `outputs/llm/<모델>.json`에 캐시 (지워도 다시 계산됨, 약 25분)
- 최종 선택한 버전의 `models/` 가중치(e5-small-ft*, labse, kanana2b, kanana-lora*)는 대회 종료 후 검증이 끝날 때까지 **지우지 말 것**

---

## 폴더 구조

```
port-vessel-matching/
├── docs/                    # 대회 안내: 01_overview · 02_evaluation · 03_rules · 04_data · 05_leaderboard_update
├── solution/
│   ├── match_lib.py         # 전처리 · 로마자 변환 · 후보 검색 · 피처 · 예측 (제출 시 model/에 복사됨)
│   ├── embed_lib.py         # 다국어 임베딩 (오프라인 · 로컬 경로 전용, 제출 시 model/에 복사됨)
│   ├── train.py             # 학습 · CV · 제출 폴더 생성
│   └── script.py            # 평가 서버 추론 진입점 (data/ → output/submission.csv)
├── tools/
│   ├── make_submit.py       # 대회 규칙 자동 검사 + 오프라인 서버 흉내 실행 + submit_<버전>.zip 생성
│   └── download_models.py   # 다국어 임베딩 가중치 받기 → models/
├── analysis/
│   ├── hangul_name_only.py      # 어려운 지표: 이름 표기만으로 한글 정답 top-1 비율
│   ├── embedding_name_only.py   # 임베딩 모델별 이름만 top-1 비교 (단독 · 결합 · 2개 조합)
│   ├── hangul_difficulty.py     # 한글 이름별 표기 재현 난이도 (OOF)
│   ├── error_report.py          # CV 오답을 쿼리 / 예측 / 정답으로 나란히 출력 (인자: CV 태그)
│   ├── known_vessel_check.py    # 오답이 train에 있는 다른 선박을 고른 경우인지 확인
│   ├── validate_baseline.py     # 베이스라인 vessel 단위 검증
│   └── debug/                   # 일회성 디버그·스모크 (smoke_test.py, debug_query.py …)
├── experiments/             # 미채택 시도의 코드 기록 (README 참고)
├── models/                  # ⛔ gitignore — 임베딩 가중치 (e5-small, labse …)
├── outputs/                 # ⛔ gitignore
│   ├── cv/                  # CV OOF 결과 <tag>_oof.csv, 카탈로그 임베딩 캐시(emb/)
│   └── submissions/         # 제출 폴더 vNN/src/ + vNN/submit_vNN.zip
├── open/ · baseline_submit/ # ⛔ gitignore — 대회 데이터
├── requirements-local.txt   # 서버 기본 패키지와 같은 버전 (pandas, lightgbm, rapidfuzz …)
└── requirements-torch.txt   # torch cu128 · transformers (임베딩용)
```

- 분석 스크립트 실행 예시

```bash
.venv\Scripts\python.exe analysis\hangul_name_only.py v14
.venv\Scripts\python.exe analysis\embedding_name_only.py e5-small labse
.venv\Scripts\python.exe analysis\error_report.py v13_e5s_nodict
.venv\Scripts\python.exe analysis\debug\smoke_test.py
```

---

## 솔루션 구조

- **후보 검색 → 재순위(Retrieve & Rerank)** 2단계 엔티티 매칭

```
입출항 기록 1행
 → ① 선박명 정규화 · 한글 → 영문 표기 후보 생성 · 다국어 임베딩
 → ② 카탈로그 78,595척에서 후보 약 100척 검색 (rapidfuzz cdist + 임베딩 코사인)
 → ③ (쿼리, 후보) 쌍 피처 55개
 → ④ LightGBM LambdaRank (단조 제약) 로 1위 선택
 → ⑤ train에 같은 호출부호·선박명이 있으면 그 정답으로 덮어쓰기
```

- 모든 처리는 **행 단위 독립** (평가 행 독립 규칙 준수)
- 사용 데이터: `train.csv`, `vessel_catalog.csv` + 공개 범용 사전학습 가중치
- 사용 패키지: 평가 서버 기본 설치 패키지만 사용 (pandas, numpy, rapidfuzz, Unidecode, lightgbm, torch, transformers) → `requirements.txt` 추가 설치 없음

### ① 전처리 · 한글 표기 (점수의 70%가 한글)

- 카탈로그 선박명은 100% 영문
- 한글 선박명 → 영문 후보 여러 개로 변환 후 비교

| 기법 | 내용 | 예 |
|---|---|---|
| 정규화 | NFKC·대문자, `NO.` 제거, 로마 숫자 → 숫자, 끝의 '호' 제거, '제N호' → N | 제2 오션파이오니어 → 2 OCEAN… |
| 한글 알파벳 이름 | 토큰 전체 또는 접두어가 알파벳 이름이면 약어로 변환 | 케이엠티씨기륭 → KMTC + 기륭 |
| 표준 로마자 | Unidecode | 거영썬 → GEOYEONGSSEON |
| 자음 골격 | 모음 제거, 소리 그룹(K/G/C, B/P/F/V, L/R…) + 영어 발음 규칙(TH→S, C/G+E/I, 묵음 R) | 커리지 ↔ COURAGE → KLS |
| 자모 로마자 변형 4종 | 표준 · 영어식 · 외래어 역변환 · 묵음 e | 에이스 → ACE, 베마그레이스 → VEMAGRACE |
| **음절 대응 학습** ⭐ | train 한글-영문 정답 쌍 DP 음절 정렬 → 음절별 영문 표기 확률 → beam search 상위 6개 표기 | 스피릿 → S/PI/RIT, 코랄젬 → CORALGEM |
| **다국어 임베딩** ⭐ | 한글 원문 ↔ 영문 카탈로그명 의미 유사도 (e5-small + LaBSE) | 리플렉션 → Reflection, 스피드웨이 → Speedway |

- 토큰 사전(train 한글 단어 → 영문 단어)은 **미사용** (`--no-dict`)
  - 다른 배의 번역으로 잘못 연결되는 경우 발생
  - 음절 표는 계속 사용

### ② 후보 검색

- `rapidfuzz.process.cdist`로 쿼리 × 카탈로그 전체 유사도 계산
- 아래 후보를 합쳐 행당 약 100척 확보 (정답 포함률: 한글 1.000 / 비한글 0.9999)
  - 정규화 이름 · 자음 골격 · 로마자 변형 · 음절 학습 표기별 유사도 상위 30
  - 위 점수 + **톤수 보너스**(차이 10% 이내 최대 25점) 상위 30 → 이름 골격이 짧아 동점이 많은 경우 톤수로 구분
  - 톤수가 가장 가까운 10척
  - 임베딩 모델별 코사인 상위 30 + (이름 + 톤수 + 1.5×e5-small + 0.5×LaBSE) 결합 상위 30

### ③ 피처 (쌍마다 55개)

| 분류 | 피처 |
|---|---|
| 이름 유사도 | ratio · token_set · token_sort · partial · Jaro-Winkler · 완전 일치 · 첫 단어 · 길이 비 · 숫자 뺀 이름 유사도 |
| 발음 유사도 | 자음 골격 · 로마자 변형 최고 · 음절 학습 표기 최고/1순위 |
| 임베딩 | 모델별(e5-small, LaBSE) 코사인 · 행 내 최고점과의 차이 · 순위 |
| 번호 | 일치 · 충돌 · 한쪽에만 있음 |
| 톤수 | 총톤수·국제총톤수 상대 차이 · 정확 일치 · 결측 |
| 국적·선종 | train 기반 한글↔영문 동시출현 확률 (파나마 → Panama, 산물선 → Bulk) |
| 행 내 상대값 | 같은 행 후보 중 최고점과의 차이 · 순위 · 후보 수 |

### ④ 모델

- `lightgbm.LGBMRanker(objective="lambdarank")`, 400 trees
- **단조 제약**
  - 유사도·임베딩·국적·선종 확률: 높을수록 점수 상승(+)
  - 톤수 차이·번호 충돌: 클수록 점수 하락(−)
  - 톤수·국적이 전혀 안 맞는데 이름만 조금 높은 후보를 고르는 비상식적 판단 방지
- **누수 방지**
  - 학습 피처용 표(국적·선종 확률, 음절 표)는 같은 선박을 뺀 inner fold로 생성(OOF)
  - 학습 시점과 추론 시점의 피처 분포 일치

### ⑤ 다국어 임베딩 (v13~)

- 평가 서버는 인터넷 차단 → 가중치를 제출 zip의 `model/emb/`, `model/emb2/`에 동봉하고 **로컬 경로로만 로드**
  - `embed_lib.py`가 `HF_HUB_OFFLINE=1`, `local_files_only=True` 강제
  - `make_submit.py` 서버 흉내 실행도 오프라인 환경변수로 실행
- `sentence-transformers`는 서버에 없어 미사용 → 풀링(CLS·평균)·Dense·정규화 직접 구현
- 평가 행 독립: 쿼리 이름은 이름마다 단독(batch 1)으로 임베딩·캐시, 카탈로그는 test와 무관하게 한 번만 계산
- 모델 비교 (`analysis/embedding_name_only.py`, 이름만 top-1, 한글 고유 이름 382개)

| 모델 | 단독 | v11 표기와 결합 |
|---|---|---|
| (v11 표기만) | – | 0.812 |
| paraphrase-multilingual-MiniLM | 0.018 | 효과 없음 |
| multilingual-e5-base | 0.343 | 0.864 |
| multilingual-e5-small | 0.450 | 0.880 (w=1.5) |
| LaBSE | 0.458 | 0.885 (w=0.5) |
| **e5-small 1.5 + LaBSE 0.5** | – | **0.908** |

---

## 검증 방법

| 방법 | 용도 |
|---|---|
| **vessel 단위 StratifiedGroupKFold (5-fold)** | 같은 선박은 한 fold에만 배치, 한글 비율 층화 → 처음 보는 선박에 대한 일반화 측정 |
| 사전 없는 CV (`--no-dict`) | 처음 보는 표기에 대한 성능 측정 |
| **이름만 top-1** (`analysis/hangul_name_only.py`, `embedding_name_only.py`) | 톤수·국적 없이 이름만으로 한글 정답이 1위인 비율, **CV 포화 이후 Public 개선을 가장 잘 예측한 지표** |
| 행 단위 랜덤 분할 (`--rowcheck`) | train 선박이 test에 섞인 경우의 조회 규칙 효과 확인 |
| `make_submit.py` | 구조·용량·requirements·인터넷 흔적·오프라인 서버 흉내 실행·평가 행 독립(전체/뒤섞기/부분집합/단독 행) 검사 |

### Public 점수 역산

- 가정: Public = **한글 300행 + 비한글 2,700행**
  - 지금까지의 모든 Public 점수와 정확히 일치
- 오답 1행당 감점
  - 한글: −0.00233
  - 비한글: −0.00011
  - 한글 1행 ≈ 비한글 21행

| 버전 | Public | 한글 오답 | 비한글 오답 |
|---|---|---|---|
| 베이스라인 | 0.48066 | 220 | 54 |
| v1 | 0.97855 | 9 | 4 |
| v2 | 0.985444 | 6 | 5 |
| v4 | 0.99 | 4 | 6 |
| v5 | 0.990222 | 4 | 4 |
| v11 | 0.992444 | 3 | 5 |
| v12 | 0.990111 | 4 | 5 |
| v13 | 0.994778 | 2 | 5 |
| **v14** | **0.996778** | **1** | **8** |

- 1.0까지 남은 오답: **한글 1행 + 비한글 8행** (v14 기준)
  - 한글 1행: −0.00233
  - 비한글 8행: −0.00089
- v14 비한글 3행 증가: 임베딩이 영문 쿼리에서 철자보다 '뜻'이 비슷한 다른 배를 끌어올린 영향으로 추정

---

## 실험 기록

| 버전 | 변경 | CV | Public | 판단 |
|---|---|---|---|---|
| baseline | RF, 이름 n-gram + 톤수 | 0.479 | 0.48066 | |
| v1 | 로마자 변환 · 자음 골격 · LightGBM 랭커 | 0.9716 | 0.97855 | ✅ |
| v1_callsign | + 호출부호 조회 | – | 0.97855 | test 선박은 train과 안 겹침 |
| v2 | '호'·알파벳·숫자 처리, 영어 발음 규칙, 톤수 결합 검색 | 0.9939 | 0.98544 | ✅ |
| v3 | + 선박명·호출부호 조회 | – | 0.98544 | 효과 없음(손해도 없음), 유지 |
| v4 | + 자모 로마자 변형 | 0.9933 | 0.99 | ✅ CV 그대로인데 Public 상승 |
| v5 | + 음절 대응 학습, 토큰 사전 끔 | 0.9986 | 0.990222 | ✅ |
| v6 | + 단어 희귀도(IDF) · 톤수 희소성 | 0.9950~0.9963 | – | ❌ 한글 판단 흔들림 |
| v7 | LightGBM 시드 · CatBoost 앙상블 | 0.9902~0.9986 | – | ❌ 시드는 동일, CatBoost는 손해 |
| v8 | + 알파벳 접두어 · 묵음 e · 카탈로그 단어장 피처 | 0.9951 | – | 단어장은 끔 (`USE_VOCAB=False`) |
| v9 | v8 − 단어장, + '제N호' 처리 | 0.9950 | – | |
| v10 | + 번호 비대칭 피처 · **단조 제약** | 0.9963 | – | 비상식 오답 제거 |
| v11 | + 숫자 뺀 이름 유사도 | 0.9951 | 0.992444 | ✅ |
| v12 | v11 + 카탈로그 단어장 피처 (`--vocab`) | – | 0.990111 | ❌ 한글 1행 추가 오답, 폐기 |
| v13 | v11 + 다국어 임베딩 e5-small | 0.9951 | 0.994778 | ✅ 한글 −1행 (이름만 top-1 0.812 → 0.880) |
| **v14** | v13 + LaBSE (2개 임베딩) | – | **0.996778** | ✅ **현재 최고**. 한글 −1행, 비한글 +3행 (이름만 top-1 0.880 → 0.908) |

### 새 평가 방식 이후 (10/19~)

- 추가된 평가 데이터가 훨씬 거침 → 모든 버전이 약 0.1 하락 (v11 0.992 → 0.888)
- EDA(`analysis/eda_noise.py`)로 train 노이즈 규칙 파악
  - 선박명: 64.7% 완전 일치, 나머지 편집 1~2회 (옆 글자 중복 삽입 · 삭제 · 인접 자리바꿈, 치환 1.4%), 첫/마지막 단어 하나만 변형
  - 톤수: 52.6% 정확 일치, 나머지 대부분 ±3~5% 곱셈 오차, 결측 6.5%
  - 국적: 결측 16.5%, 대표 영문 국적과 77% 일치 (오래된 입항 연도일수록 불일치 ↑)
- 스트레스 검증(`analysis/stress_test.py`): 노이즈를 더 세게 건 검증 세트로 약점 확인 → 이름 편집 증가, 단어 삭제, 한글 톤수 결측에 취약

| 버전 | 변경 | 새 Public | 서버 추론 | 판단 |
|---|---|---|---|---|
| v5 | (규칙 기반 초기, 로마자 변형) | 0.86346 | 1분 14초 | |
| v11 | (이전 최고 규칙 기반) | 0.88829 | – | |
| v13 | v11 + 임베딩 e5-small | 0.89384 | 2분 45초 | |
| v15 | v14, 임베딩 한글 전용 | 0.89823 | 2분 7초 | 이전 Public과 반대로 v14보다 낮음 |
| v14 | 임베딩 e5-small + LaBSE (영문 포함) | 0.90165 | 3분 52초 | |
| **v16** | v14 + 노이즈 증강 1배 (`--augment 1`) + 단어 단위 검색 (`--word`) | **0.91379** | 4분 12초 | ✅ |
| v17 | v16, 증강 2배 (`--augment 2`) | 0.91272 | 4분 22초 | ➖ v16과 비슷 (증강 1배 근처가 적정) |
| v19 | v16 + 치환·띄어쓰기·단어 순서 노이즈 (`--aug-extra`) | 0.90407 | 4분 17초 | ❌ −0.0097. 새 데이터에 없는 노이즈를 학습하면 손해 |
| **v21** | v16 + 파인튜닝 e5-small (`e5-small-ft`) + 교차 학습 (`--cross`) | **0.91740** | 4분 13초 | ✅ +0.0036. 단어 삭제 약점은 남음 |
| **v22** | v21 + Qwen2.5-3B 가벼운 LoRA 영문 복원 후보·피처 (`--llm qwen3b-lora --llm-cross qwen3b-lorah0,qwen3b-lorah1`) | **0.91799** | 10분 22초 | ✅ +0.0006 (스트레스 테스트보다 작음). zip 7.4GB |
| v24 | v22 + 한글 → 중국어 병음 역변환 후보·피처 (`--pinyin`) | 0.91706 | 10분 17초 | ➖ −0.0009 (1행 안팎). 합성 test에서는 외국 선박 0.9756 → 0.9789 (중국 0.967 → 1.000), 한국·병음 0.9689 → 0.9733 |
| **v25** | v24, LLM을 kanana-1.5-2.1B LoRA로 교체 (Apache 2.0) | **0.92079** | 9분 26초 | ✅ 현재 최고 (v22 대비 +0.0028). zip 5.6GB. 합성 test: 외국 0.9789 → 0.9822, 한국·병음 0.9733 → 0.9756 |
| v27 | v25 + LightGBM 시드 3개 앙상블 (`--ensemble lgb3`) | 0.91964 | 9분 27초 | ➖ −0.0012 (1~2행). zip 5.6GB. 합성 test: 한국·병음 0.9756 → 0.9778, 외국 0.9822 → 0.9800 (±2행, 차이 없음) |
| v26 | v22, LLM을 kanana LoRA로 교체 (병음 없음, Apache 2.0) | 0.91706 | 8분 54초 | zip 5.6GB. 합성 test: 외국 0.9822 (v25와 같음), 한국·병음 0.9711 (v25 0.9756) |
| v23 | v22 + e5 파인튜닝 단어 삭제 노이즈 10% → 30% (`finetune_emb.py --drop-word 0.3`) | 0.91141 | 10분 10초 | ❌ −0.0066. 실제 데이터에 드문 노이즈를 강하게 학습하면 손해 (v19와 같은 패턴) |

- **진단 제출** (`v16_diag_hangul`: v16에서 비한글 예측만 고정 ID로 바꿈 → 점수 = 0.7 × 한글 정확도)
  - 진단 점수 0.6263158 → **한글 정확도 0.8947** (= 17/19, 한글 행 수는 19의 배수)
  - 비한글 정확도 (0.9137878 − 0.6263158) / 0.3 = **0.9582**
  - 손실: 한글 0.0737 · 비한글 0.0125 → **손실의 85%가 한글**
  - 기존 3,000건의 한글은 99.7% → 새로 추가된 한글 행이 훨씬 어려움
- 스트레스 검증 (가중 정확도, 선박 1/5 검증): v16이 노이즈 시나리오에서 v14보다 크게 우세, 원본(노이즈 없음)만 0.9997 → 0.9890 손해

| 시나리오 | v14 | v16 |
|---|---|---|
| 원본 | 0.9997 | 0.9890 |
| 이름 편집 +2 | 0.9086 | 0.9491 |
| 이름 편집 +4 | 0.7996 | 0.8965 |
| 단어 하나 삭제 | 0.5544 | 0.8046 |
| 편집+2 & 톤수·국적 결측 | 0.6484 | 0.7705 |

- **e5-small 파인튜닝** (`solution/finetune_emb.py`, 대조 학습: 노이즈 쿼리 ↔ 카탈로그 영문명, hard negative)
  - 임베딩 단독 한글 top-1 (fold 0 검증 선박): 원본 0.425 → 0.763, 편집+2 0.138 → 0.538, 단어 삭제는 0.15 → 0.16 (개선 없음)
  - v20 (v16 + 파인튜닝 e5): 개선 없음 → 원인은 **스태킹 누수** (파인튜닝 모델이 LightGBM 학습 선박을 이미 봐서 학습 피처가 과하게 정확)
  - v21 (v20 + **교차 학습**: 선박을 절반으로 나눠 각 절반 피처는 반대쪽 절반으로 학습한 모델로 계산, `--half`)

| 시나리오 (한글 정확도, 258행) | v16 | v21 |
|---|---|---|
| 원본 | 0.9845 | 0.9845 |
| 이름 편집 +1 | 0.9651 | 0.9690 |
| 이름 편집 +2 | 0.9341 | 0.9302 |
| 이름 편집 +4 | 0.8721 | 0.8760 |
| 단어 하나 삭제 | **0.8256** | 0.7442 |
| 톤수 결측 | 0.9612 | 0.9690 |
| 톤수 오차 5~10% | 0.9729 | 0.9806 |
| 편집+2 & 톤수 결측 | 0.7248 | **0.7674** |
| 편집+2 & 톤수·국적 결측 | 0.6977 | **0.7287** |

- v21: 톤수가 흔들리거나 빠진 경우 한글 +1~4%, 단어 삭제 −8% (파인튜닝 임베딩이 단어 삭제를 못 배움). 비한글은 거의 같음

- **LLM 영문 복원** (`analysis/llm_translit_probe.py`, Qwen2.5 Instruct, 한글 선박명 → 영문 표기 → 카탈로그 글자 유사도 검색)
  - LLM 단독 한글 top1 (fold 0 검증 이름 80개, 빔 4): 1.5B few-shot 0.738 · 3B few-shot 0.838 · 3B LoRA 4 epoch 0.700 (학습 선박 이름을 외워 과적합) · **3B 가벼운 LoRA (1 epoch, lr 1e-4, r 8) 0.838, top10 0.938, 행당 1.2초** (few-shot의 1/3)
  - 파인튜닝 e5와 서로 다른 이름을 맞힘 → 두 방법 합집합 top10 0.988
  - **kanana-1.5-2.1B 가벼운 LoRA: top1 0.863, top10 0.963, 편집+2 0.512, 행당 1.0초** → Qwen 3B보다 모든 항목에서 우수
  - ⚠️ Qwen2.5-3B는 Qwen Research License(비상업). kanana(Apache 2.0)로 교체 (v25~)
  - `solution/finetune_llm.py` (LoRA → 원래 가중치에 병합, 추론에 peft 불필요), `solution/llm_lib.py` (프롬프트·변환기 공용)
- **v22** (v21 + LLM 후보 40개·피처 5개 `s_llm, s_llm1, rel_llm, rank_llm, llm_exact`, LLM도 교차 학습)

| 시나리오 (한글 정확도, 258행) | v21 | v22 |
|---|---|---|
| 원본 | 0.9845 | 0.9845 |
| 이름 편집 +1 | **0.9690** | 0.9574 |
| 이름 편집 +2 | 0.9302 | **0.9380** |
| 이름 편집 +4 | 0.8760 | **0.8876** |
| 단어 하나 삭제 | 0.7442 | **0.7791** |
| 톤수 결측 | 0.9690 | 0.9690 |
| 편집+2 & 톤수 결측 | 0.7674 | **0.7907** |
| 편집+2 & 톤수·국적 결측 | 0.7287 | **0.7519** |

### 배운 것

- **CV 0.995 이후로는 CV로 개선 판단 불가**
  - CV 오답이 선박 3~4척(대부분 자매선) 수준
  - 한 척이 오갈 때마다 ±0.0023씩 변동
  - v4·v11·v13은 CV가 그대로거나 낮았는데 Public 상승
- **'이름만 top-1' 지표가 Public과 같은 방향으로 움직임**
  - v5 0.780 → v11 0.812 → v13 0.880 → v14 0.908, Public 한글 오답 4 → 3 → 2 → 1
- **규칙 기반 표기의 한계 → 사전학습 다국어 임베딩이 돌파구**
  - 규칙을 늘리면 하나를 고치고 하나가 깨지는 단계(v8~v12)
  - 임베딩은 외래어(Reflection, Speedway, Galaxy)를 뜻으로 연결
  - 단, 영문 쿼리에는 부작용 (v14 비한글 +3행)
- test 선박은 train과 거의 겹치지 않음 (호출부호·선박명 조회 효과 0)
- 이름으로 풀 수 없는 유형이 남음
  - 이름이 바뀐 배: HAFNIA WREN → BW Wren
  - 톤수까지 같은 자매선: 마스그라치 (Maasgracht / Marsgracht)

---

## 남은 과제

1. **임베딩을 한글 쿼리에만 적용 (v15, 진행 중)**
   - 코드 반영 완료: `--emb-all`을 빼면 한글 쿼리에만 적용
   - 기대: 비한글 8행 → 5행 (+0.00033)
2. **남은 한글 1행**
   - 후보: 임베딩 결합 비중 조정, 약한 유형(중국어·일본어 표기, 긴 영어 단어, 복합어) 추가 대응
3. **본선 발표 자료 준비**
   - PDF 제출: 2026-11-09(월)
   - 온라인 발표: 2026-11-10(화)

---

## 공유 시 주의

- **`open/`, `baseline_submit/`, `outputs/`, `models/` 커밋 금지** ([`.gitignore`](.gitignore) 반영됨)
  - 대회 데이터 재배포 금지 규칙
  - 제출 폴더 `resources.pkl`: train 정답 기반 조회표 포함
  - OOF CSV: train 행 그대로 포함
  - 임베딩 가중치: 용량이 큼 (각자 `tools/download_models.py`로 받기)
- 저장소는 **private** 권장
- LightGBM 모델 저장 방식: 문자열 변환 후 Python으로 저장·로드 (`model_to_string` / `Booster(model_str=...)`)
  - LightGBM C++ 저장 함수가 Windows 한글 경로를 열지 못하기 때문
