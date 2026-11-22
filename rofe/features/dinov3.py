import json
import time
from pathlib import Path

import numpy as np

from ..data.sources import DINOV3_REVISION, import_from, verify_clone
from . import publish
from .bank import FeatureBank, sha256_file
from .images import checked_rgb, image_root, read_images

WEIGHT_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
WEIGHT_FILE = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
BATCH = 16
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet


# official README transform: resize to 256 x 256, scale to [0, 1], ImageNet normalization
def make_transform(size: int = 256):
    import torch
    from torchvision.transforms import v2

    return v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True),
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=MEAN, std=STD)])


# official ViT-B/16 (LVD-1689M) from the pinned DINOv3 clone, strict native state dict
def load_encoder(source: Path, checkpoint: Path):
    import importlib

    import torch

    if sha256_file(checkpoint) != WEIGHT_SHA256:
        raise ValueError(f"Expected the official DINOv3 ViT-B/16 weights, got {checkpoint}")
    source = verify_clone(source, DINOV3_REVISION)
    import_from(source)
    module = importlib.import_module("dinov3.hub.backbones")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError("Imported another DINOv3 installation")
    model = module.dinov3_vitb16(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True),
                          strict=True)
    return model.eval().requires_grad_(False)


# class token after the final norm (768) of every image of a task, in inventory order
def extract(task_dir: Path, output: Path, source: Path, checkpoint: Path,
            root: Path | None = None, device: str = "cuda") -> FeatureBank:
    import torch
    import torchvision

    records = read_images(task_dir)
    root = image_root(task_dir, root)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32, TF32 off
    torch.backends.cudnn.allow_tf32 = False
    model = load_encoder(source, checkpoint).to(device)
    transform = make_transform()
    features = np.zeros((len(records), 768), dtype=np.float32)  # (N, 768)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(records), BATCH):
            chunk = records[start:start + BATCH]
            batch = torch.stack([transform(checked_rgb(r, root)) for r in chunk]).to(device)
            values = model.forward_features(batch)["x_norm_clstoken"].float().cpu().numpy()
            if values.shape != (len(chunk), 768) or not np.isfinite(values).all():
                raise ValueError("Unexpected or nonfinite encoder output")
            features[start:start + len(chunk)] = values
            if start // BATCH % 100 == 0:
                print(json.dumps({"completed": start + len(chunk), "total": len(records),
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    provenance = {
        "encoder": "dinov3", "source_revision": DINOV3_REVISION, "checkpoint_sha256": WEIGHT_SHA256,
        "pixel_pipeline": "ToImage_Resize256_antialias_ToDtype_Normalize",
        "mean": list(MEAN), "std": list(STD), "representation": "x_norm_clstoken",
        "precision": "float32_no_autocast_tf32_disabled", "batch_size": BATCH,
        "images_sha256": sha256_file(task_dir / "images.jsonl"),
        "torch_version": str(torch.__version__), "torchvision_version": torchvision.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device}
    return publish(features, [r["row_id"] for r in records], provenance, output)

