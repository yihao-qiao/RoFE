import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .bank import FeatureBank

# encoder -> module with extract(task_dir, output, ...)
EXTRACTORS = {
    "dinov3": "rofe.features.dinov3",
    "siglip": "rofe.features.siglip",
    "electra": "rofe.features.text",
    "e5": "rofe.features.text",
}


# imported lazily: each extractor runs in its own environment
def get_extractor(encoder: str) -> Callable:
    if encoder not in EXTRACTORS:
        raise ValueError(f"Unknown encoder {encoder}; known: {sorted(EXTRACTORS)}")
    extract = importlib.import_module(EXTRACTORS[encoder]).extract
    if EXTRACTORS[encoder].endswith(".text"):
        return lambda *args, **kwargs: extract(encoder, *args, **kwargs)
    return extract


def publish(features: np.ndarray, row_ids: list[str], provenance: dict,
            output: Path) -> FeatureBank:
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    partial.parent.mkdir(parents=True, exist_ok=True)
    FeatureBank(features, tuple(row_ids), provenance).save(partial)
    FeatureBank.load(partial)  # checked before the rename publishes it
    partial.rename(output)
    return FeatureBank.load(output)


__all__ = ["EXTRACTORS", "FeatureBank", "get_extractor", "publish"]
