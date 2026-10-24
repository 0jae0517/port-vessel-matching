"""공개 다국어 사전학습 임베딩 모델을 models/<이름>/ 으로 받는다 (평가 서버는 오프라인이라 제출 시 model/에 동봉).

  실행: .venv\\Scripts\\python.exe tools\\download_models.py [이름 ...]
"""
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "e5-small": "intfloat/multilingual-e5-small",
    "e5-base": "intfloat/multilingual-e5-base",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # 한글 선박명 → 영문 철자 복원 실험용 소형 LLM (Apache 2.0)
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    # 카탈로그 영문명 → 한글 표기 합성용 (한국어 특화, Apache 2.0). 로컬 데이터 생성에만 사용
    "kanana2b": "kakaocorp/kanana-1.5-2.1b-instruct-2505",
    # 카탈로그 한글 표기표 보강용 (로컬·임대 GPU 전처리에만 사용, 제출물에는 들어가지 않음)
    "kanana8b": "kakaocorp/kanana-1.5-8b-instruct-2505",
}
PATTERNS = ["config.json", "*.safetensors", "*.safetensors.index.json", "tokenizer*", "vocab.txt", "vocab.json",
            "merges.txt", "generation_config.json", "sentencepiece.bpe.model",
            "special_tokens_map.json", "modules.json", "1_Pooling/*", "2_Dense/*"]

for name in sys.argv[1:] or list(MODELS):
    out = ROOT / "models" / name
    path = snapshot_download(MODELS[name], local_dir=out, allow_patterns=PATTERNS)
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    print(f"{name:10s} {MODELS[name]:60s} {size / 2**20:8.1f} MB -> {out}")
