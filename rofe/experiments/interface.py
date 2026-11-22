from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..method import ROTATIONS, labeled_scores
from ..pfn.tabpfn import table_codes, write_export
from ..tasks import Fold
from . import Context, register_experiment
from .common import (
    base_record,
    labeled_sets,
    score_exported,
    scored,
    settings,
    stream,
    tabicl_predictor,
)

METHODS = ("comet", "rofe")


# CoMET and RoFE blocks of every labeled set, for a PFN run in its own environment
def export(fold: Fold, cfg: dict, ctx: Context, directory: Path) -> None:
    codes, categorical = table_codes(fold.table, fold.categorical, fold.pool)
    for draw in labeled_sets(fold, cfg):
        used = np.concatenate([draw.rows, fold.query])  # labeled rows, then query rows
        for encoder in cfg["encoders"]:
            x = fold.embedding(encoder)
            ls = labeled_scores(x, draw.rows, fold.fold, draw.shots, draw.repeat,
                                stream(encoder))
            blocks = {"table": codes[used].astype(np.float32), "categorical": categorical,
                      "support": ls.scores[used]}
            for r, block in enumerate(ls.rotated()):  # Z R_1, ..., Z R_K
                blocks[f"rot{r}"] = block[used]
            jobs = [(name, name, fold.fold) for name in blocks if name not in
                    ("table", "categorical")]
            meta = {"labels": fold.labels, "rows": draw.rows, "query": fold.query,
                    "y_rows": fold.y[draw.rows], "y_query": fold.y[fold.query],
                    "rank": ls.gd_rank, "width": ls.width, "fold": fold.fold,
                    "shots": draw.shots, "repeat": draw.repeat, "encoder": encoder}
            # method -> jobs whose probabilities are averaged
            methods = {"comet": ["support"], "rofe": [f"rot{r}" for r in range(ROTATIONS)],
                       "rot1": ["rot0"]}
            write_export(directory, f"f{fold.fold}_{encoder}_s{draw.shots}_r{draw.repeat}",
                         meta, blocks, jobs, methods)


# CoMET: PFN(Z); RoFE: mean over k of PFN(Z R_k), K = 4
@register_experiment("interface", export=export, score=score_exported)
def run(fold: Fold, cfg: dict, ctx: Context) -> Iterator[dict]:
    methods = cfg.get("methods", METHODS)
    for draw in labeled_sets(fold, cfg):
        for encoder in cfg["encoders"]:
            ls = labeled_scores(fold.embedding(encoder), draw.rows, fold.fold, draw.shots,
                                draw.repeat, stream(encoder))
            rotated = ls.rotated() if "rofe" in methods else []
            for setting in settings(fold, cfg):
                predict = tabicl_predictor(fold, draw, ctx.pfn, setting)
                base = {**base_record(fold, draw, setting, encoder), "width": ls.width,
                        "gd_rank": ls.gd_rank}
                if "comet" in methods:
                    yield scored(base, "comet", predict(ls.scores), fold)
                if "rofe" in methods:
                    proba = np.mean([predict(b) for b in rotated], axis=0)
                    yield scored(base, "rofe", proba, fold, rotations=len(rotated))
