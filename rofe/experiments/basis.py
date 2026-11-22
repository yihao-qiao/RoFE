from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..mechanism import givens_basis, hadamard_basis, partial_rotation
from ..method import labeled_scores
from ..pfn.tabpfn import table_codes, write_export
from ..rotation import extra_seeds
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

METHODS = ("comet", "comet_x4", "rot1", "rofe", "rot_strong", "rot_weak", "hadamard", "givens")
SEED_STEP = 1000  # PFN seeds fold + 1000 d of comet_x4


def bases(fold: Fold, draw, encoder: str):
    ls = labeled_scores(fold.embedding(encoder), draw.rows, fold.fold, draw.shots, draw.repeat,
                        stream(encoder))
    z, k, w = ls.scores, ls.gd_rank, ls.width  # Z (N, w), Gavish-Donoho rank k
    extra = extra_seeds(fold.fold, draw.shots, draw.repeat, stream(encoder))  # 3, 4: strong, weak
    rotated = ls.rotated()  # [Z R_1, ..., Z R_K]
    # rot_strong: R = diag(R_k, I); rot_weak: R = diag(I_k, R_{w-k});
    # hadamard: (N, 2^ceil(log2 w)); givens: equal column variances on the labeled rows
    blocks = {"comet": z, "rot1": rotated[0],
              "rot_strong": partial_rotation(z, np.arange(k), int(extra[3])),
              "rot_weak": partial_rotation(z, np.arange(k, w), int(extra[4])),
              "hadamard": hadamard_basis(z), "givens": givens_basis(z, draw.rows)}
    return ls, blocks, rotated


# every change of basis of every labeled set, for a PFN run in its own environment
def export(fold: Fold, cfg: dict, ctx: Context, directory: Path) -> None:
    methods = cfg.get("methods", METHODS)
    codes, categorical = table_codes(fold.table, fold.categorical, fold.pool)
    for draw in labeled_sets(fold, cfg):
        used = np.concatenate([draw.rows, fold.query])  # labeled rows, then query rows
        for encoder in cfg["encoders"]:
            ls, made, rotated = bases(fold, draw, encoder)
            blocks = {"table": codes[used].astype(np.float32), "categorical": categorical,
                      **{name: b[used] for name, b in made.items()},
                      **{f"turn{r}": b[used] for r, b in enumerate(rotated[1:], start=1)}}
            jobs = [(name, name, fold.fold) for name in made]  # (job, block, PFN seed)
            jobs += [(f"turn{r}", f"turn{r}", fold.fold) for r in range(1, len(rotated))]
            jobs += [(f"comet_d{d}", "comet", fold.fold + SEED_STEP * d) for d in range(1, 4)]
            groups = {name: [name] for name in made}  # method -> jobs averaged
            groups["rofe"] = ["rot1", *(f"turn{r}" for r in range(1, len(rotated)))]
            groups["comet_x4"] = ["comet", *(f"comet_d{d}" for d in range(1, 4))]
            meta = {"labels": fold.labels, "rows": draw.rows, "query": fold.query,
                    "y_rows": fold.y[draw.rows], "y_query": fold.y[fold.query],
                    "rank": ls.gd_rank, "width": ls.width, "fold": fold.fold,
                    "shots": draw.shots, "repeat": draw.repeat, "encoder": encoder}
            write_export(directory, f"f{fold.fold}_{encoder}_s{draw.shots}_r{draw.repeat}",
                         meta, blocks, jobs, {m: g for m, g in groups.items() if m in methods})


# changes of basis of the same principal scores Z
@register_experiment("basis", export=export, score=score_exported)
def run(fold: Fold, cfg: dict, ctx: Context) -> Iterator[dict]:
    methods = cfg.get("methods", METHODS)
    for draw in labeled_sets(fold, cfg):
        for encoder in cfg["encoders"]:
            ls, blocks, rotated = bases(fold, draw, encoder)
            z, k, w = ls.scores, ls.gd_rank, ls.width
            for setting in settings(fold, cfg):
                predict = tabicl_predictor(fold, draw, ctx.pfn, setting)
                base = {**base_record(fold, draw, setting, encoder), "width": w, "gd_rank": k}
                done = {}
                for method in methods:
                    if method in blocks:
                        done[method] = predict(blocks[method])
                        yield scored(base, method, done[method], fold,
                                     columns=int(blocks[method].shape[1]))
                if "rofe" in methods:  # mean over the K rotations
                    first = done["rot1"] if "rot1" in done else predict(rotated[0])
                    proba = np.mean([first, *(predict(b) for b in rotated[1:])], axis=0)
                    yield scored(base, "rofe", proba, fold, columns=w)
                if "comet_x4" in methods:  # same Z, four PFN seeds
                    first = done["comet"] if "comet" in done else predict(z)
                    runs = [first, *(predict(z, fold.fold + SEED_STEP * d) for d in range(1, 4))]
                    yield scored(base, "comet_x4", np.mean(runs, axis=0), fold, columns=w)
