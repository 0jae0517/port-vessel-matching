# HANDOFF — 노트북(RTX 4060 Laptop 8GB)으로 작업 이전 (2026-10-21)

> 노트북의 Claude Code에서 **이 파일을 먼저 읽고** 이어서 작업한다. 상세 실험 기록은 [README.md](README.md), 대회 규칙은 [docs/](docs/).

---

## 1. 현재 상태 요약

- 대회: DACON 항만 선박 매칭 (입출항 기록 한 행 → 카탈로그 `vessel_id`), 점수 = 0.7 × 한글 정확도 + 0.3 × 비한글 정확도
- 마감: **2026-11-02(월) 10:00** — 최종 제출물 1개 선택 고정. 최종 점수 = 선택 제출물의 **Private** 점수
- 하루 제출 5회, 추론 30분, zip 10GB. 다시 제출하면 최종 선택이 Public 최고 점수로 자동 변경됨
- 1위 Public **0.97434** (2026-10-21)

| 버전 | 구성 | 새 Public | 서버 추론 |
|---|---|---|---|
| **v25 (현재 최고, 최종 후보)** | v21 + kanana-1.5-2.1B LoRA 영문 복원 + 한글→병음 역변환 | **0.92079** | 9분 26초 |
| v27 | v25 + LightGBM 시드 3개 앙상블 | 0.91964 | 9분 27초 |
| v26 | v25에서 병음 제외 | 0.91706 | 8분 54초 |
| v22 / v24 | Qwen2.5-3B LoRA (비상업 라이선스 → 최종 후보 제외) | 0.91799 / 0.91706 | 약 10분 |
| v21 | e5 파인튜닝 + 교차 학습 (LLM 없음) | 0.91740 | 4분 13초 |

- 언어별 추정: v25 한글 약 0.90, 비한글 약 0.96 (v16 진단 제출 0.8947/0.9582, 팀원 v11 비한글 진단 0.9084)
- 버전 간 차이 0.001~0.002 는 LightGBM 흔들림 수준 (v24·v26 이 소수점 10자리까지 같은 점수)

## 2. 지금까지 알아낸 것 (다음 방향의 근거)

1. **오답 대부분은 '순위 오류'** (stress_test 진단, v25, fold 0)
   - 정답이 후보에 있는데 1위가 아님. 특히 정답 2위가 가장 많음. 정답이 LightGBM 상위 5개 안에 있는 비율 98.8%
2. **쌍둥이 배** (톤수·국적·선종까지 같고 이름 한 단어만 다름)가 실제 한글 오답의 전부
   - 에버 얼라이언스(Alliance↔Reliance), 펭슈하이(Xiu↔Shou), 마스그라치(Mars↔Maas), 제2 오션파이오니어(번호), 글로비스 슈피리어(Superior↔Supreme), 스타크리퍼(Clipper↔Skipper)
   - 목록: `outputs/analysis/hangul_name_pairs_errors.csv`
3. **우리 노이즈(stress_test·augment)는 비현실적**
   - 한글: train 382척 모두 같은 배 = 같은 한글 표기 (오타 없음). 음절 중복·삭제 노이즈는 근거 없음 → 실제 변동은 '어떤 외래어 표기를 골랐나'
   - 영문: 오타 횟수가 길이에 비례 (6자 이하 89% 무편집, 11자 이상 약 50% 무편집·27% 1회·17% 2회, 4회 이상 거의 없음)
   - 그래서 stress test가 Public을 잘 예측하지 못했고 v19·v23(비현실 노이즈 강화)은 Public 하락
4. **카탈로그명 → 한글 표기(kanana) 후 한글끼리 비교**가 쌍둥이 배를 가름
   - Superior 94 vs Supreme 84, Clipper 95 vs Skipper 70, Alliance 87 vs Reliance 75 (자모 단위 유사도)
   - 진단 한글 2,836행 top10 단순 결합: 0.9016 → 0.9048
5. 합성 test (kanana로 카탈로그명 한글화, train에 없는 선박): 외국 900척·한국/병음 450행에서 v25 0.98 안팎 → '표기' 자체는 대부분 해결됨
6. **중국계 이름의 실제 기록은 공식 외래어 표기법이 아니라 영어식 병음 읽기** (Feng 펭, Heng 헹, Zhen 젠, Shen 센, Shandong 샨동, Yuan 유안)
   - match_lib 병음 역변환(`PINYIN_MAP`)과 정방향(`PINYIN_FORWARD`)에 영어식 읽기 추가 → train 중국계 기록 13개 모두 정답 병음 읽기 생성 (이전 일부 실패)
   - 이 변경으로 v25 구성을 다시 빌드해도 병음 피처 값이 조금 달라짐 (v28에 포함)
   - 카탈로그 한글 표기표의 중국·홍콩 병음 이름: Feng Xiu Hai 유사도 53(풍수해) → 100, 정답 순위 12 → 1

## 3. 노트북에서 할 일 (우선순위 순)

1. **환경 설정 + v25 재현 확인** (아래 4·5절)
2. ✅ **카탈로그 전체 한글 표기표 완성 (2026-10-21, RunPod RTX 4090 약 2.5시간, 약 $2)** — `outputs/translit/kanana2b.json`, 78,594개 (빔 4·예시 20, 이름마다 표기 최대 3개 목록)
   - 쌍둥이 배 확인: Superior 94 vs Supreme 88, Clipper 95 vs Skipper 76, Alliance 96 vs Reliance 82, Xiu 100 vs Shou 82 (Marsgracht/Maasgracht 는 80 동점, Queen → 여왕 은 남음)
   - 다음: v28 빌드·평가 (아래 3번). 아래 2번 하위 항목은 생성 과정 기록
   - (기록) 생성 방법: `tools/runpod_translit.sh` + `tools/make_runpod_bundle.ps1`
   - 완료: kanana 표기 약 4,600개 + **중국·홍콩 국적 병음 이름 9,101개는 규칙 표기 목록** (공식 + 항만 기록식 영어식 읽기, LLM 생략)
   - 프롬프트 개선 완료 (2026-10-21): 소리 표기만 지시 + 흔한 영어 단어가 든 train 쌍 위주 few-shot + **빔 4개 중 영문과 소리가 가장 비슷한 표기 선택**(`sound_score`, 비슷한 표기는 최대 3개 목록 저장)
     - 개선 예: Chemist 화학 → 케미스트, River 강변 → 리버, Winner 승자 → 위너, Superior 슈퍼이아이 → 슈퍼리어, Supreme → 슈프림
     - 남은 문제: Queen → 여왕, General → 장군 같은 일부 단어 (2.1B 모델 한계)
   - 현재 표: 병음 규칙 표기 9,101개 + 새 프롬프트 LLM 표기 60개 (예전 프롬프트 표기는 `--redo-llm` 으로 지움). **남은 약 69,400개는 노트북에서** `--redo-llm` 없이 이어서 실행 (붙이면 진행분이 지워짐)
     - 데스크탑(GTX 1070)은 빔 4·예시 20개로 초당 0.2개 (약 94시간, VRAM 초과) → 중단
     - 2026-10-21 저녁부터 데스크탑에서 `--batch 12 --beams 2 --shots 8` 로 이어서 실행 중 (초당 약 1개, 약 18시간). 노트북으로 옮길 때 이 실행을 멈추고 json 을 복사한 뒤 같은 옵션으로 이어서 실행 (옵션이 다르면 표기 품질이 섞임)
     - 노트북에서 먼저 속도 측정: `.venv\Scripts\python.exe tools\translit_catalog.py --batch 12 --beams 4 --shots 20` 를 몇 분 돌려 '남은 약 N분' 확인
     - 너무 느리면 `--beams 2 --shots 10` 처럼 줄이기 (빔·예시가 줄면 뜻 번역 걸러내는 힘은 약해짐). VRAM 8GB 를 넘기면 급격히 느려지므로 `nvidia-smi` 로 확인하며 `--batch` 조정
     - 프롬프트 확인용: `--dry-run --names "Sunny Queen" "Lucky River"` (저장 안 함)
   - 데스크탑(GTX 1070)은 초당 약 2개. 노트북은 훨씬 빠를 것
   - **한 번만 만들면 됨**: 카탈로그는 바뀌지 않으므로 이후 모든 버전이 재사용. 제출 zip에 `model/catalog_hangul.json`으로 동봉돼 추론 때 LLM 생성 없음. 프롬프트·생성 모델을 바꿀 때만 다시 (해당 이름만 지우고 재실행하면 됨)
   - kanana는 병음·영어 단어를 뜻으로 번역하는 경우가 있음 (Feng Xiu Hai → 풍수해, Queen → 여왕) → 병음은 규칙으로 대체, 영어 단어 번역 문제는 남아 있음 (필요하면 프롬프트·few-shot 보강)
3. **v28 = v25 + 한글 표기 비교 피처** (`--catalog-hangul outputs\translit\kanana2b.json`, 코드는 완료·미검증)
   - 후보 20+20개, 피처 `s_khan, rel_khan, rank_khan` (match_lib `jamo_compact`, `attach_catalog_hangul`)
   - 빌드 명령은 5절. 스트레스 진단(v25 대비)으로 확인 후 제출
4. **현실적인 노이즈로 재설계** (augment.py, stress_test.py)
   - 영문: 길이별 실제 편집 분포에서 횟수 추출, 연산 비율 (옆 글자 중복 53% · 삭제 · 자리바꿈 11%)
   - 한글: 오타 제거 → 같은 영문명의 다른 한글 표기 (kanana 샘플링, ㅓ↔ㅗ·ㅔ↔ㅐ·장음·띄어쓰기 규칙)
   - 단어 삭제 시나리오·증강 제거
   - 새 스트레스 세트로 검증 → 현실적 증강으로 LightGBM·e5·LoRA 재학습
   - **A/B 두 버전으로 Public 비교** (제출 2회): A = train 실제 분포 그대로 / B = 같은 패턴, 오타 빈도만 1.5~2배
     - 근거: 비현실 노이즈를 늘린 v19(−0.0097)·v23(−0.0066)은 하락 → 줄이면 오를 여지. 반대로 v16 증강(+0.012)이 거친 새 test에 맞았을 가능성도 있어 B로 대비
     - 현실적 노이즈는 번호·이름 일치를 더 믿게 해서 '번호만 다른 배' 오답과 쌍둥이 배 구분도 덤으로 좋아질 수 있음
6. **'제N ○○' ↔ '○○ No. N' 번호 위치** — 실제 한글 오답 중 숫자 관련은 '제2 오션파이오니어' 1척(4행)뿐 (숫자 있는 이름은 오히려 더 잘 맞힘)
   - 한글 표기 비교(jamo_compact)에서 숫자를 떼어 따로 비교하도록 보완 ('2오션파이오니어' vs '오션파이오니어2' 순서 차이로 유사도 하락)
5. **LLM 재정렬** (`analysis/rerank_proto.py`, listwise: 기록 + LightGBM 상위 5개 → 번호) — 4번 노이즈로 다시 학습·평가
   - 노트북이면 7~8B 4bit QLoRA 가능 (kanana-1.5-8B-Instruct, Qwen2.5-7B-Instruct: 둘 다 Apache 2.0). 서버는 `bitsandbytes`를 requirements로 설치해야 함 → 설치 확인 제출 필요 (설치 오류는 제출 횟수 차감 없음)
7. (선택) CatBoost·앙상블 재비교: 예전엔 만점 근처 CV로 버렸음. 현실적 스트레스 세트에서 다시 확인

## 4. 노트북 환경 설정

```bash
# 1) 코드
git clone https://github.com/0jae0517/port-vessel-matching.git
# 2) 외장 드라이브의 open/, models/, outputs/ 를 저장소 폴더에 그대로 복사 (tools/pack_for_laptop.ps1 로 만든 것)
# 3) Python 3.11.17 가상환경 (uv)
uv python install 3.11.17
uv venv --python 3.11.17 .venv
uv pip install --python .venv -r requirements-dev.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
# 4) 확인
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
```

- 전원 연결 + 고성능 모드로 학습 (배터리 모드는 GPU 성능 크게 떨어짐)
- Claude 메모리: 외장 드라이브의 `claude-memory/` 내용을 노트북의 `%USERPROFILE%\.claude\projects\<프로젝트 경로 폴더명>\memory\` 에 복사 (폴더명은 프로젝트 경로의 `:`·`\` 를 `-` 로 바꾼 것, 예: `D--dev-dacon-port-vessel-matching`)

## 5. 주요 명령

```bash
# v25 재현 (models/ 와 outputs/llm 캐시가 있으면 LightGBM 학습만, 약 30분)
.venv\Scripts\python.exe solution\train.py --out outputs\submissions\v25\src --no-dict --emb e5-small-ft,labse --emb-all --callsign --name-lookup --augment 1 --word --cross e5-small-fth0,e5-small-fth1 --llm kanana-lora --llm-cross kanana-lorah0,kanana-lorah1 --pinyin
.venv\Scripts\python.exe tools\make_submit.py outputs\submissions\v25\src outputs\submissions\v25

# v28 (한글 표기표 완성 후)
.venv\Scripts\python.exe solution\train.py --out outputs\submissions\v28\src --no-dict --emb e5-small-ft,labse --emb-all --callsign --name-lookup --augment 1 --word --cross e5-small-fth0,e5-small-fth1 --llm kanana-lora --llm-cross kanana-lorah0,kanana-lorah1 --pinyin --catalog-hangul outputs\translit\kanana2b.json

# 스트레스 진단 (fold 0, 오답 = 후보 누락 / 순위 오류 분해, outputs/cv/stress_diag_<cfg>.csv)
.venv\Scripts\python.exe analysis\stress_test.py v25

# 합성 test 평가 (제출본 그대로 실행)
.venv\Scripts\python.exe analysis\synth_eval.py v25 v28 --file synth_test2.csv --by group
```

- 처음부터 모든 모델을 다시 만드는 방법은 README '코드 검증용 전체 재현 경로' 참고

## 6. 노트북 Claude 작업 지침

**소통**
- 사용자와 한국어로 대화. README는 개조식으로 갱신 (버전 표에 점수·서버 추론 시간 기록)
- 커밋·푸시는 사용자가 직접 함. Claude는 커밋 메시지만 작성
- 제출도 사용자가 함. Claude는 `make_submit.py`가 FAIL 0건인 `submit_vNN.zip` 까지만 준비
- 브라우저로 DACON '나의 제출' 페이지 점수 확인은 읽기만 (제출·제출선택 버튼 금지)

**대회 규칙 (위반 시 실격)**
- 학습·검증 데이터는 `train.csv`, `vessel_catalog.csv` 만. 외부 데이터·외부 API 금지 (발음 사전 등도 외부 데이터로 볼 수 있음 → 규칙을 코드로 구현하거나 사전학습 모델 지식 사용)
- 평가 행 독립: 행마다 단독 예측. LLM은 행(이름)마다 배치 1·결정적 생성. 카탈로그 전처리는 배치 가능
- 추론은 오프라인: 가중치를 `model/` 에 동봉하고 로컬 경로로만 로드
- 사전학습 가중치는 공개 범용 모델만, 라이선스는 Apache 2.0·MIT 등 상업 허용 (Qwen2.5-3B 는 비상업 → 사용 안 함)
- DACON 답변(docs/06_qna.md): LLM LoRA·가중치 동봉 허용, 코드 검증 때 학습·전처리·추론 전체 코드와 가중치 요청 가능 → 최종 후보의 가중치·코드는 재현 가능하게 유지, 삭제 금지

**기술 원칙**
- train 정답으로 학습한 모델(e5 파인튜닝, LoRA)을 LightGBM 피처로 쓸 때는 **교차 학습** 필수 (절반 모델로 학습 피처 계산, `--cross`, `--llm-cross`). 안 하면 스태킹 누수 (v20 실패 원인)
- Public 차이 0.001~0.002 는 잡음. 판단은 진단·합성 test·현실적 스트레스 세트를 함께 보고
- train CV는 거의 만점이라 변별력 없음 (새 평가 데이터가 훨씬 어려움)
- 큰 파일 삭제 전 사용자 확인. 디스크 여유 확인 (제출 폴더 하나 약 11GB = src + zip)
- GPU 8GB: LLM과 임베딩을 동시에 올리지 않기 (LLM 결과를 먼저 계산·캐시 후 내리고 임베딩 로드)
