from pathlib import Path

import numpy as np
import pandas as pd

from . import register_pfn

CHECKPOINT = ("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt")  # (repository, file)
CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
ESTIMATORS = 8  # TabICLv2's default ensemble size


@register_pfn("tabicl")
class TabICL:
    in_process = True  # fitted and predicted in this process

    def __init__(self, checkpoint: Path, n_estimators: int = ESTIMATORS, device: str = "cuda",
                 **options):
        self.checkpoint = Path(checkpoint)
        self.n_estimators = n_estimators
        self.device = device
        self.options = options

    # one fresh TabICLv2 classifier per labeled set
    def model(self, seed: int, **options):
        from tabicl import TabICLClassifier

        return TabICLClassifier(model_path=self.checkpoint, allow_auto_download=False,
                                device=self.device, random_state=seed,
                                n_estimators=self.n_estimators, **{**self.options, **options})

    def fit_predict(self, X: pd.DataFrame, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                    seed: int, **options) -> tuple[np.ndarray, np.ndarray]:
        model = self.model(seed, **options)
        model.fit(X.iloc[train], y[train])  # the rows train are the context
        # classes (c_fit,), probabilities (len(test), c_fit)
        return model.classes_, np.asarray(model.predict_proba(X.iloc[test]), np.float64)


def download_checkpoint(directory: Path) -> Path:
    from huggingface_hub import hf_hub_download

    from ..features.bank import sha256_file

    path = Path(hf_hub_download(repo_id=CHECKPOINT[0], filename=CHECKPOINT[1],
                                local_dir=directory))
    if sha256_file(path) != CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected checkpoint {path}")
    return path
