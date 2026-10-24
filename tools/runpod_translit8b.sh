#!/usr/bin/env bash
# 임대 GPU(RunPod RTX 4090 24GB)에서 kanana-1.5-8B로 '의심 이름'(outputs/translit/suspects.txt, 약 4,600개)만 다시 표기.
# 2.1B는 흔한 영어 단어를 뜻으로 번역하는 경우가 있어(Queen → 여왕) 큰 모델로 보강한다. 8B는 전처리용이라 제출물에 넣지 않는다.
#   1) 데스크탑에서 tools\make_runpod_bundle.ps1 로 만든 runpod_bundle.zip 을 Pod(/workspace)에 업로드
#   2) Pod 터미널:  cd /workspace && unzip -o runpod_bundle.zip -d port-vessel && bash port-vessel/tools/runpod_translit8b.sh
#   3) 끝나면 /workspace/port-vessel/translit8b_result.tgz 를 내려받고 Pod 를 Terminate
# 끊기면 같은 명령을 다시 실행하면 진행분부터 이어서 한다. 메모리 부족(OOM)이면 BATCH=4
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0))"
pip install -q "transformers==4.57.6" accelerate safetensors sentencepiece "rapidfuzz==3.14.3" pandas scikit-learn Unidecode huggingface_hub hf_xet hf_transfer

if [ ! -f models/kanana8b/model.safetensors.index.json ]; then
  python tools/download_models.py kanana8b
fi

export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# 2.1B 표기표와 같은 설정 (소리 표기 프롬프트, few-shot 20개, 빔 4개 중 소리가 가장 비슷한 표기)
python tools/translit_catalog.py --model kanana8b --names-file outputs/translit/suspects.txt \
  --batch "${BATCH:-8}" --beams 4 --shots 20

tar czf translit8b_result.tgz outputs/translit/kanana8b.json
python - <<'EOF'
import json
d = json.load(open("outputs/translit/kanana8b.json", encoding="utf-8"))
for n in ["Sunny Queen", "Southern Queen", "General", "New Wisdom", "Korea Victory", "Monjasa Supplier"]:
    print(f"  {n:20s} {d.get(n)}")
print(f"완료: {len(d)}개. translit8b_result.tgz 를 내려받고 Pod 를 Terminate 하세요")
EOF
