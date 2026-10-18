# experiments

채택하지 않은 시도의 코드를 기록용으로 남긴다. 실행 경로에는 쓰지 않는다.

| 파일 | 시도 | 결과 |
|---|---|---|
| `match_lib_v6_idf_features.py` | v5 `match_lib.py` + 단어 희귀도(IDF) 커버리지 · 마지막/가장 드문 단어 일치 · 톤수 희소성 · 국적×선종 피처 | vessel CV 0.9963 / 0.9950 (v5 0.9986) → 채택 안 함. `HAFNIA MERLIN`은 맞혔지만 `루비 에이스`가 Garnet Ace로 틀림 |

다른 기각 실험(앙상블, 카탈로그 단어장)은 현재 코드에 옵션으로 남아 있다.

- 앙상블: `solution/train.py --ensemble-cv`, `--ensemble NAME`
- 카탈로그 단어장 표기 피처: `solution/match_lib.py`의 `USE_VOCAB = True`
