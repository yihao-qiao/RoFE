from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..draws import Draw, draws
from ..method import pfn_frame
from ..metrics import padded, scores
from ..tasks import ENCODERS, Fold

SETTINGS = ("embedding", "joint")  # embedding alone, embedding + table columns


def settings(fold: Fold, cfg: dict) -> list[str]:
    wanted = cfg.get("settings", SETTINGS)
    return [s for s in wanted if s == "embedding" or fold.table.shape[1]]


def labeled_sets(fold: Fold, cfg: dict) -> Iterator[Draw]:
    return draws(fold.y, fold.pool, fold.fold, cfg["levels"], cfg.get("draws", 3))


def stream(encoder: str) -> int:
    return ENCODERS[encoder][1]  # enters the rotation seeds


def base_record(fold: Fold, draw: Draw, setting: str, encoder: str) -> dict:
    return {"task": fold.task, "fold": fold.fold, "shots": draw.shots, "repeat": draw.repeat,
            "n_labeled": int(len(draw.rows)), "setting": setting, "encoder": encoder}


# proba: (n_q, c) over fold.labels
def scored(base: dict, method: str, proba: np.ndarray, fold: Fold, **extra) -> dict:
    return {**base, "method": method, **extra,
            **scores(fold.labels, proba, fold.y[fold.query], fold.labels)}


# predict(block, seed=fold, **options) -> (n_q, c) query probabilities over fold.labels
def tabicl_predictor(fold: Fold, draw: Draw, pfn, setting: str):
    def predict(block: np.ndarray, seed: int | None = None, **options) -> np.ndarray:
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite block")
        X = pfn_frame(block, fold.table, setting == "joint")  # table columns first for 'joint'
        classes, proba = pfn.fit_predict(X, fold.y, draw.rows, fold.query,
                                         fold.fold if seed is None else seed, **options)
        return padded(classes, proba, fold.labels)
    return predict


def score_exported(task: str, directory: Path, cfg: dict, ctx) -> Iterator[dict]:
    from ..pfn.tabpfn import score_exports

    if not (directory / "predictions" / "COMPLETE").is_file():
        raise FileNotFoundError(f"Predictions of {task} are incomplete")
    methods = cfg.get("methods")
    for record in score_exports(directory / "exports", directory / "predictions"):
        if record["setting"] in cfg.get("settings", SETTINGS) and (
                methods is None or record["method"] in methods):
            yield {"task": task, **record}
