import json
import time
from pathlib import Path

import numpy as np

from . import publish
from .bank import FeatureBank, sha256_file

# encoder -> (repository, commit, width, input prefix)
ENCODERS = {
    "electra": ("google/electra-base-discriminator", "1ae76a97c7e84a4e640876a07453fccd636f0667",
                768, ""),
    "e5": ("intfloat/e5-base-v2", "f52bf8ec8c7124536f0efb74aca902b2995e5bcd", 768, "query: "),
}
MAX_TOKENS = 512
BATCH = 32
ALLOW = ["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"]


def read_texts(task_dir: Path) -> list[dict]:
    with (task_dir / "texts.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records or len({r["row_id"] for r in records}) != len(records):
        raise ValueError(f"Invalid texts in {task_dir}")
    return records


def mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    # sum_t m_t h_t / sum_t m_t over the attention mask m, (B, width)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)


# mean of the last hidden states over at most 512 tokens, for every row text
def extract(encoder: str, task_dir: Path, output: Path, device: str = "cuda",
            **_) -> FeatureBank:
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    repo, revision, width, prefix = ENCODERS[encoder]
    records = read_texts(task_dir)
    local = Path(snapshot_download(repo, revision=revision, allow_patterns=ALLOW))
    weights = {p.name: sha256_file(p) for p in sorted(local.iterdir())
               if p.suffix in (".safetensors", ".bin")}
    if local.name != revision or not weights:
        raise ValueError(f"Unexpected snapshot of {repo}")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32, TF32 off
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(local)
    model = AutoModel.from_pretrained(local, dtype=torch.float32).eval().to(device)
    features = np.zeros((len(records), width), dtype=np.float32)
    truncated = 0
    started = time.perf_counter()
    for start in range(0, len(records), BATCH):
        batch = [prefix + r["text"] for r in records[start:start + BATCH]]  # e5: 'query: '
        truncated += sum(len(ids) > MAX_TOKENS
                         for ids in tokenizer(batch, truncation=False)["input_ids"])
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_TOKENS,
                           return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        values = mean_pool(hidden, inputs["attention_mask"]).float().cpu().numpy()
        if values.shape != (len(batch), width) or not np.isfinite(values).all():
            raise ValueError("Unexpected or nonfinite encoder output")
        features[start:start + len(batch)] = values
        if start // BATCH % 50 == 0:
            print(json.dumps({"encoder": encoder, "completed": start + len(batch),
                              "total": len(records),
                              "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    provenance = {
        "encoder": encoder, "repo": repo, "revision": revision, "weight_sha256": weights,
        "pooling": "mean over attention mask", "prefix": prefix, "max_tokens": MAX_TOKENS,
        "texts_truncated": truncated, "texts_truncated_fraction": truncated / len(records),
        "precision": "float32_tf32_disabled", "batch_size": BATCH,
        "texts_sha256": sha256_file(task_dir / "texts.jsonl"),
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device}
    return publish(features, [r["row_id"] for r in records], provenance, output)
