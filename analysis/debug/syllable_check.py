"""음절 정렬·학습 결과 눈으로 확인."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solution"))
import match_lib as ml  # noqa: E402
import train as T  # noqa: E402

for w, e in [("스피릿", "SPIRIT"), ("커리지", "COURAGE"), ("캐서린", "CATHERINE"), ("베마그레이스", "VEMAGRACE"),
             ("리플렉션", "REFLECTION"), ("얼라이언스", "ALLIANCE"), ("포천", "FORTUNE")]:
    print(w, e, ml.align_syllables(w, e))

tr, cat = T.load()
res = ml.build_resources(tr, cat)
sm = res["syl_map"]
print("\n학습된 음절 수:", len(sm))
for s in ["스", "피", "서", "리", "레", "커", "천", "젬", "샤", "먼", "그", "이"]:
    print(s, sm.get(s))
res["token_dict"], res["whole_dict"] = {}, {}
print()
for name in ["글로비스 스피릿", "포천 카렌", "흥아 샤먼", "모닝 캐서린", "코랄젬", "베마그레이스", "리플렉션", "마스그라치"]:
    print(name, ml.learned_variants(ml.tokenize(name), res))
