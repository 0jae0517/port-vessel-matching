#!/usr/bin/env bash
# RunPod(RTX 4090) 등 임대 GPU에서 카탈로그 한글 표기표를 새로 만든다.
#   1) 데스크탑에서 tools\make_runpod_bundle.ps1 로 만든 runpod_bundle.zip 을 Pod(/workspace)에 업로드
#   2) Pod 터미널:  cd /workspace && unzip -o runpod_bundle.zip -d port-vessel && bash port-vessel/tools/runpod_translit.sh
#   3) 끝나면 /workspace/port-vessel/translit_result.tgz 를 내려받고 Pod 를 Terminate
# 중간에 끊기면 RESUME=1 bash port-vessel/tools/runpod_translit.sh  (진행분부터 이어서)
# 메모리가 남으면 BATCH=12 처럼 늘리고, 넘치면(OOM) 줄인다. nvidia-smi 로 확인
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0))"
# 표기 결과에 영향을 주는 transformers 만 대회 서버와 같은 버전으로 고정
# hf_transfer: RunPod 이미지가 HF_HUB_ENABLE_HF_TRANSFER=1 로 켜 두므로 없으면 다운로드가 실패한다
pip install -q "transformers==4.57.6" accelerate safetensors sentencepiece "rapidfuzz==3.14.3" pandas scikit-learn Unidecode huggingface_hub hf_xet hf_transfer

if [ ! -f models/kanana2b/model.safetensors ]; then
  python tools/download_models.py kanana2b
fi

export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
REDO="--redo-llm"
if [ "${RESUME:-0}" = "1" ]; then REDO=""; fi
# 품질 설정: 빔 4개 중 소리가 가장 비슷한 표기, few-shot 예시 20개 (데스크탑 비교에서 뜻 번역이 가장 적었던 설정)
python tools/translit_catalog.py $REDO --batch "${BATCH:-8}" --beams 4 --shots 20

tar czf translit_result.tgz outputs/translit/kanana2b.json
python - <<'EOF'
import json
d = json.load(open("outputs/translit/kanana2b.json", encoding="utf-8"))
print(f"완료: {len(d)}개 (병음 규칙 표기 {sum(isinstance(v, list) and len(v) > 0 and ' ' in v[0] for v in d.values())}개 포함)")
print("결과: translit_result.tgz 를 내려받고 Pod 를 Terminate 하세요")
EOF
