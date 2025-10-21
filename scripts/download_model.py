from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-large"
OUT_DIR = Path("models") / "multilingual-e5-large"

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    file_patterns = [
        "README.md",
        "*.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "sentencepiece.bpe.model",
        "1_Pooling/config.json",
    ]
    
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(OUT_DIR),
        local_dir_use_symlinks=False,
        revision="main",
        allow_patterns=file_patterns
    )
    print(f"Downloaded to: {OUT_DIR.resolve()}")
