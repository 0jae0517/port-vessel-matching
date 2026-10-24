#!/usr/bin/env bash
# 임대 GPU(RunPod)에서 LLM 재정렬(kanana 2.1B LoRA)을 학습·평가하고, 제출용 LoRA 어댑터를 만든다.
#   v34: 후보 10개(A..J 라벨) 평가 + 제출용, 후보 5개(v33 과 같은 설정) 평가 — 평가 점수는 npz 로 저장해
#        로컬에서 tools/rerank_tune.py 로 후보 수·gate·w 를 다시 고른다 (약 1시간 30분)
#   1) 데스크탑에서 tools\make_rerank_bundle.ps1 로 만든 rerank_bundle.zip 을 Pod(/workspace)에 업로드
#   2) Pod 터미널:  cd /workspace && unzip -o rerank_bundle.zip -d port-vessel && bash port-vessel/tools/runpod_rerank.sh
#   3) 끝나면 /workspace/port-vessel/rerank_result.tgz (평가 결과 + 어댑터) 를 내려받고 Pod 를 Terminate
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0))"
pip install -q "transformers==4.57.6" "peft==0.17.1" accelerate safetensors sentencepiece pandas huggingface_hub hf_xet hf_transfer

if [ ! -f models/kanana2b/model.safetensors ]; then
  python tools/download_models.py kanana2b
fi
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

echo "=== [1/3] 평가 (후보 10개): fold 0 선박 제외 학습 → fold 0 진단에서 LightGBM vs 재정렬 ==="
python tools/rerank_train.py --base kanana2b --eval --k 10 --letters 2>&1 | tee rerank_eval_k10.log
echo "=== [2/3] 제출용 (후보 10개): train 전체로 학습 (어댑터만 저장) ==="
python tools/rerank_train.py --base kanana2b --final --k 10 --letters --out kanana-rerank-k10 2>&1 | tee rerank_final_k10.log
tar czf rerank_result.tgz rerank_eval_k10.log rerank_final_k10.log outputs/rerank/eval_result_k10.csv \
  outputs/rerank/eval_lp_k10.npz models/kanana-rerank-k10-adapter
echo "=== [3/3] 평가 (후보 5개, v33 설정): 점수 저장용 ==="
python tools/rerank_train.py --base kanana2b --eval --k 5 2>&1 | tee rerank_eval_k5.log
tar czf rerank_result.tgz rerank_eval_k10.log rerank_final_k10.log outputs/rerank/eval_result_k10.csv \
  outputs/rerank/eval_lp_k10.npz models/kanana-rerank-k10-adapter \
  rerank_eval_k5.log outputs/rerank/eval_result_k5.csv outputs/rerank/eval_lp_k5.npz
ls -la rerank_result.tgz
echo "완료: rerank_result.tgz 를 내려받고 Pod 를 Terminate 하세요"
