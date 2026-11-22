import json
import time
from pathlib import Path

import numpy as np

from . import publish
from .bank import FeatureBank, sha256_file
from .images import checked_rgb, image_root, read_images

REPO = "google/siglip-so400m-patch14-384"
REVISION: str | None = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
WIDTH = 1152  # so400m pooled output
BATCH = 32
ALLOW = ["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"]


# so400m vision tower in float32, and its slow image processor
def load_encoder(revision: str | None = REVISION, device: str = "cuda"):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, SiglipVisionModel

    local = Path(snapshot_download(REPO, revision=revision, allow_patterns=ALLOW))
    weights = {p.name: sha256_file(p) for p in sorted(local.iterdir())
               if p.suffix in (".safetensors", ".bin")}
    if not weights or (revision is not None and local.name != revision):
        raise ValueError(f"Unexpected snapshot of {REPO}")
    model = SiglipVisionModel.from_pretrained(local, dtype=torch.float32).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(local, use_fast=False)
    return model, processor, {"repo": REPO, "revision": local.name, "weight_sha256": weights}


# pooled output (1152) of every image of a task, in inventory order
def extract(task_dir: Path, output: Path, root: Path | None = None, revision: str | None = REVISION,
            device: str = "cuda", **_) -> FeatureBank:
    import torch
    import transformers

    records = read_images(task_dir)
    root = image_root(task_dir, root)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32, TF32 off
    torch.backends.cudnn.allow_tf32 = False
    model, processor, info = load_encoder(revision, device)
    features = np.zeros((len(records), WIDTH), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(records), BATCH):
        chunk = records[start:start + BATCH]
        inputs = processor(images=[checked_rgb(r, root) for r in chunk],
                           return_tensors="pt").to(device)
        with torch.inference_mode():
            values = model(pixel_values=inputs["pixel_values"]).pooler_output
        values = values.float().cpu().numpy()
        if values.shape != (len(chunk), WIDTH) or not np.isfinite(values).all():
            raise ValueError("Unexpected or nonfinite encoder output")
        features[start:start + len(chunk)] = values
        if start // BATCH % 50 == 0:
            print(json.dumps({"completed": start + len(chunk), "total": len(records),
                              "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    provenance = {
        "encoder": "siglip", **info, "representation": "pooler_output",
        "processor": type(processor).__name__,
        "precision": "float32_no_autocast_tf32_disabled", "batch_size": BATCH,
        "images_sha256": sha256_file(task_dir / "images.jsonl"),
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device}
    return publish(features, [r["row_id"] for r in records], provenance, output)
