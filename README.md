# 항만 입출항 기록 ↔ 선박 제원 카탈로그 매칭

- **대회**: 항만 문제해결을 위한 AI 기술 공모전 (DACON)
- **주제**: AI를 활용한 선박 정보 표기 차이 해소
- **과제**: 입출항 기록 1행(선박명·호출부호·톤수·국적·선종) → 선박 제원 카탈로그 78,595척 중 같은 선박의 `vessel_id` 예측
- **평가**: `0.7 × 한글 기록 정확도 + 0.3 × 비한글 기록 정확도`
- **대회 규칙·평가·데이터 설명**: [`docs/`](docs/)

| | Public | vessel 단위 CV (사전 없음) |
|---|---|---|
| 베이스라인 (RandomForest) | 0.48066 | 0.479 |
| **현재 최고: v11** | **0.992444** | 0.9951 |

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
```

- Windows에서 `uv venv`가 "minor version link" 오류를 낼 경우 → 받은 파이썬으로 직접 생성
  - `%APPDATA%\uv\python\cpython-3.11.17-windows-x86_64-none\python.exe -m venv .venv`
- torch·transformers까지 서버와 맞출 경우 (현재 솔루션은 미사용)

```bash
python -m uv pip install --python .venv/Scripts/python.exe -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

### 2. 데이터 배치 (저장소 미포함)

- 대회 규칙상 대회 데이터 재배포 금지
- DACON에서 각자 내려받아 아래 위치에 배치

```
open/train.csv
open/vessel_catalog.csv
open/sample_submission.csv
baseline_submit/            # baseline_submit.zip 압축 해제 (베이스라인 비교용, 선택)
```

### 3. 검증 · 제출 파일 생성

```bash
# vessel 단위 5-fold CV (사전 없음 = 현재 기본 설정)
.venv\Scripts\python.exe solution\train.py --cv --no-dict --tag v11_nodict

# 전체 train으로 학습 → 제출 폴더 생성 (현재 최고 v11 설정)
.venv\Scripts\python.exe solution\train.py --out outputs\submissions\v11\src --no-dict --callsign --name-lookup

# 대회 규칙 검사 + 서버 흉내 실행 + submit.zip 생성
.venv\Scripts\python.exe tools\make_submit.py outputs\submissions\v11\src outputs\submissions\v11
```

- `make_submit.py` 결과 FAIL 0건일 때만 제출
- 소요 시간(8코어 기준): 학습 약 5분 · CV 약 25분 · 서버 추론 약 25초

### `train.py` 옵션

| 옵션 | 설명 |
|---|---|
| `--cv` | vessel 단위 StratifiedGroupKFold 5-fold CV, 결과 `outputs/cv/<tag>_oof.csv` |
| `--no-dict` | 한글 토큰 사전·전체 이름 사전 끔 (**v5부터 기본 사용**) |
| `--out DIR` | 전체 train으로 학습 → 제출 폴더(`script.py`, `requirements.txt`, `model/`) 생성 |
| `--callsign` / `--name-lookup` | train에 같은 호출부호·선박명이 있으면 그 정답으로 덮어쓰기 (test 효과 없음, 손해도 없음) |
| `--rowcheck` | 행 단위 랜덤 분할로 조회 규칙 효과 확인 |
| `--ensemble-cv` / `--ensemble NAME` | LightGBM 시드·CatBoost 앙상블 비교 / 앙상블 제출 생성 (미채택) |
| `--tag` | CV 결과 파일 이름 |

---

## 폴더 구조

```
port-vessel-matching/
├── docs/                    # 대회 안내: 01_overview · 02_evaluation · 03_rules · 04_data
├── solution/
│   ├── match_lib.py         # 전처리 · 로마자 변환 · 후보 검색 · 피처 · 예측 (제출 시 model/에 복사됨)
│   ├── train.py             # 학습 · CV · 제출 폴더 생성
│   └── script.py            # 평가 서버 추론 진입점 (data/ → output/submission.csv)
├── tools/
│   └── make_submit.py       # 대회 규칙 자동 검사 + submit.zip 생성
├── analysis/
│   ├── hangul_name_only.py  # 어려운 지표: 이름 표기만으로 한글 정답 top-1 비율
│   ├── hangul_difficulty.py # 한글 이름별 표기 재현 난이도 (OOF)
│   ├── error_report.py      # CV 오답을 쿼리 / 예측 / 정답으로 나란히 출력 (인자: CV 태그)
│   ├── known_vessel_check.py# 오답이 train에 있는 다른 선박을 고른 경우인지 확인
│   ├── validate_baseline.py # 베이스라인 vessel 단위 검증
│   └── debug/               # 일회성 디버그·스모크 (smoke_test.py, debug_query.py …)
├── experiments/             # 미채택 시도의 코드 기록 (README 참고)
├── outputs/                 # ⛔ gitignore
│   ├── cv/                  # CV OOF 결과 <tag>_oof.csv
│   └── submissions/         # 제출 폴더 (v5, v11)
├── open/ · baseline_submit/ # ⛔ gitignore — 대회 데이터
├── requirements-local.txt   # 서버 기본 패키지와 같은 버전
└── requirements-torch.txt   # torch cu128 스택 (선택)
```

- 분석 스크립트 실행 예시

```bash
.venv\Scripts\python.exe analysis\hangul_name_only.py v11
.venv\Scripts\python.exe analysis\error_report.py v11_nodict
.venv\Scripts\python.exe analysis\debug\smoke_test.py
```

---

## 솔루션 구조

- **후보 검색 → 재순위(Retrieve & Rerank)** 2단계 엔티티 매칭

```
입출항 기록 1행
 → ① 선박명 정규화 · 한글 → 영문 표기 후보 생성
 → ② 카탈로그 78,595척에서 후보 약 100척 검색 (rapidfuzz cdist)
 → ③ (쿼리, 후보) 쌍 피처 49개
 → ④ LightGBM LambdaRank (단조 제약) 로 1위 선택
 → ⑤ train에 같은 호출부호·선박명이 있으면 그 정답으로 덮어쓰기
```

- 모든 처리는 **행 단위 독립** (평가 행 독립 규칙 준수)
- 사용 데이터: `train.csv`, `vessel_catalog.csv`만 사용
- 사용 패키지: 평가 서버 기본 설치 패키지만 사용 (pandas, numpy, rapidfuzz, Unidecode, lightgbm) → `requirements.txt` 추가 설치 없음

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

- 토큰 사전(train 한글 단어 → 영문 단어)은 **미사용** (`--no-dict`)
  - 다른 배의 번역으로 잘못 연결되는 경우 발생
  - 음절 표는 계속 사용

### ② 후보 검색

- `rapidfuzz.process.cdist`로 쿼리 × 카탈로그 전체 유사도 계산
- 아래 후보를 합쳐 행당 약 100척 확보 (정답 포함률: 한글 1.000 / 비한글 0.9999)
  - 정규화 이름 · 자음 골격 · 로마자 변형 · 음절 학습 표기별 유사도 상위 30
  - 위 점수 + **톤수 보너스**(차이 10% 이내 최대 25점) 상위 30 → 이름 골격이 짧아 동점이 많은 경우 톤수로 구분
  - 톤수가 가장 가까운 10척

### ③ 피처 (쌍마다 49개)

| 분류 | 피처 |
|---|---|
| 이름 유사도 | ratio · token_set · token_sort · partial · Jaro-Winkler · 완전 일치 · 첫 단어 · 길이 비 · 숫자 뺀 이름 유사도 |
| 발음 유사도 | 자음 골격 · 로마자 변형 최고 · 음절 학습 표기 최고/1순위 |
| 번호 | 일치 · 충돌 · 한쪽에만 있음 |
| 톤수 | 총톤수·국제총톤수 상대 차이 · 정확 일치 · 결측 |
| 국적·선종 | train 기반 한글↔영문 동시출현 확률 (파나마 → Panama, 산물선 → Bulk) |
| 행 내 상대값 | 같은 행 후보 중 최고점과의 차이 · 순위 · 후보 수 |

### ④ 모델

- `lightgbm.LGBMRanker(objective="lambdarank")`, 400 trees
- **단조 제약**
  - 유사도·국적·선종 확률: 높을수록 점수 상승(+)
  - 톤수 차이·번호 충돌: 클수록 점수 하락(−)
  - 톤수·국적이 전혀 안 맞는데 이름만 조금 높은 후보를 고르는 비상식적 판단 방지
- **누수 방지**
  - 학습 피처용 표(국적·선종 확률, 음절 표)는 같은 선박을 뺀 inner fold로 생성(OOF)
  - 학습 시점과 추론 시점의 피처 분포 일치

---

## 검증 방법

| 방법 | 용도 |
|---|---|
| **vessel 단위 StratifiedGroupKFold (5-fold)** | 같은 선박은 한 fold에만 배치, 한글 비율 층화 → 처음 보는 선박에 대한 일반화 측정 |
| 사전 없는 CV (`--no-dict`) | 처음 보는 표기에 대한 성능 측정, Public과 경향 일치도 높음 |
| **이름만 top-1** (`analysis/hangul_name_only.py`) | 톤수·국적 없이 이름 표기만으로 한글 정답이 1위인 비율, **CV 포화 이후 Public 개선을 가장 잘 예측한 지표** |
| 행 단위 랜덤 분할 (`--rowcheck`) | train 선박이 test에 섞인 경우의 조회 규칙 효과 확인 |
| `make_submit.py` | 구조·용량·requirements·인터넷 흔적·서버 흉내 실행·평가 행 독립(전체/뒤섞기/부분집합/단독 행) 검사 |

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
| **v11** | **0.992444** | **3** | **5** |

- 1.0까지 남은 오답: **한글 3행 + 비한글 5행**
- 점수 기준으로는 사실상 한글 3행이 전부

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
| **v11** | + 숫자 뺀 이름 유사도 | 0.9951 | **0.992444** | ✅ 현재 최고 |

### 배운 것

- **CV 0.995 이후로는 CV로 개선 판단 불가**
  - CV 오답이 선박 3~4척(대부분 자매선) 수준
  - 한 척이 오갈 때마다 ±0.0023씩 변동
  - v4·v11은 CV가 그대로거나 낮았는데 Public 상승
- **'이름만 top-1' 지표가 Public과 같은 방향으로 움직임** (v5 0.780 → v11 0.809)
- test 선박은 train과 거의 겹치지 않음 (호출부호·선박명 조회 효과 0)
- 이름으로 풀 수 없는 유형이 남음
  - 이름이 바뀐 배: HAFNIA WREN → BW Wren
  - 톤수까지 같은 자매선: 마스그라치 (Maasgracht / Marsgracht)

---

## 남은 과제

1. **카탈로그 단어장 피처 재시험** (`match_lib.USE_VOCAB = True`)
   - CV는 손해였지만 이름만 top-1 보완 효과 큼 (둘 중 하나라도 1위 0.853)
   - Public으로 확인 필요
2. **약한 한글 유형 대응** (`analysis/hangul_difficulty.py` 결과)
   - 중국어·일본어·지명 표기: 지하이 셩 → ZY Hi Sheng, 기륭 → Keelung, 쮸루기 → Tsurugi
   - 긴 영어 단어: 스피드웨이 → Speedway, 레지슬레이터 → Legislator
   - 붙여 쓴 복합어: 레이디가스 → Lady Gas
3. **비한글 1행 회귀 원인 확인** (v5 4행 → v11 5행)
   - 영향 0.0001로 우선순위 낮음
4. **본선 발표 자료 준비**
   - PDF 제출: 2026-11-09(월)
   - 온라인 발표: 2026-11-10(화)

---

## 공유 시 주의

- **`open/`, `baseline_submit/`, `outputs/` 커밋 금지** ([`.gitignore`](.gitignore) 반영됨)
  - 대회 데이터 재배포 금지 규칙
  - 제출 폴더 `resources.pkl`: train 정답 기반 조회표 포함
  - OOF CSV: train 행 그대로 포함
