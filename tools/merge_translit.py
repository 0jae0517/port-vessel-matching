"""2.1B 표기표에 8B 보강 표기를 합친다: 8B로 다시 만든 이름은 8B 표기를 앞에, 2.1B 표기를 뒤에 (최대 5개).

비교 때 표기 목록 중 최고 유사도를 쓰므로 2.1B 표기를 지우지 않아도 손해가 없다.
  실행: .venv\\Scripts\\python.exe tools\\merge_translit.py
  결과: outputs/translit/kanana_merged.json  (v29: --catalog-hangul outputs\\translit\\kanana_merged.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
d = ROOT / "outputs" / "translit"
base = json.loads((d / "kanana2b.json").read_text(encoding="utf-8"))
big = json.loads((d / "kanana8b.json").read_text(encoding="utf-8"))
as_list = lambda v: [v] if isinstance(v, str) else list(v)
merged, n = {}, 0
for name, v in base.items():
    if name in big:
        merged[name] = list(dict.fromkeys(x for x in as_list(big[name]) + as_list(v) if x))[:5]
        n += 1
    else:
        merged[name] = v
(d / "kanana_merged.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
print(f"전체 {len(merged)}개 중 8B 표기 추가 {n}개 → {d / 'kanana_merged.json'}")
for name in ["Sunny Queen", "Southern Queen", "General", "New Wisdom", "Korea Victory"]:
    print(f"  {name:18s} {merged.get(name)}")
