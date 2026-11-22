from collections.abc import Iterator

import numpy as np

from ..method import labeled_scores, pfn_frame
from ..metrics import padded
from ..scaling import NORM_METHODS, bypass, check_pipelines, global_scaled
from ..tasks import Fold
from . import Context, register_experiment
from .common import base_record, labeled_sets, scored, stream

# method -> (block, bypass the per-column standardization, bypass the outlier clip)
CALLS = {
    "pc_std": ("principal", False, False),
    "pc_glob": ("principal_global", True, False),
    "pc_raw": ("principal", True, False),
    "rot1_std": ("rotated", False, False),
    "rot1_glob": ("rotated_global", True, False),
    "pc_std_nc": ("principal", False, True),
    "pc_glob_nc": ("principal_global", True, True),
    "rot1_std_nc": ("rotated", False, True),
    "rot1_glob_nc": ("rotated_global", True, True),
}
METHODS = ("pc_std", "pc_glob", "rot1_std", "rot1_glob")


# TabICL's per-column standardization against one global scale, principal or rotated axes
@register_experiment("scaling")
def run(fold: Fold, cfg: dict, ctx: Context) -> Iterator[dict]:
    methods = cfg.get("methods", METHODS)

    def predict(block: np.ndarray, rows: np.ndarray, scaler: bool, clip: bool) -> np.ndarray:
        X = pfn_frame(block, fold.table, joint=False)
        model = ctx.pfn.model(fold.fold, norm_methods=NORM_METHODS)  # no power transform
        with bypass(scaler, clip):  # identity in place of the skipped steps
            model.fit(X.iloc[rows], fold.y[rows])
            proba = model.predict_proba(X.iloc[fold.query])
        check_pipelines(model, scaler, clip)
        return padded(model.classes_, np.asarray(proba, np.float64), fold.labels)

    for draw in labeled_sets(fold, cfg):
        for encoder in cfg["encoders"]:
            ls = labeled_scores(fold.embedding(encoder), draw.rows, fold.fold, draw.shots,
                                draw.repeat, stream(encoder))
            rotated = ls.rotated()
            # *_global: block / s, s^2 = mean column variance on the labeled rows
            blocks = {"principal": ls.scores, "rotated": rotated[0],
                      "principal_global": global_scaled(ls.scores, draw.rows),
                      "rotated_global": global_scaled(rotated[0], draw.rows)}
            base = {**base_record(fold, draw, "embedding", encoder), "width": ls.width,
                    "gd_rank": ls.gd_rank}
            done = {}
            for method in methods:
                if method in CALLS:
                    block, scaler, clip = CALLS[method]
                    done[method] = predict(blocks[block], draw.rows, scaler, clip)
                    yield scored(base, method, done[method], fold)
            if "rot4_std" in methods:  # rot1_std and rotations 2-4, averaged
                first = done.get("rot1_std")
                if first is None:
                    first = predict(rotated[0], draw.rows, False, False)
                runs = [first, *(predict(b, draw.rows, False, False) for b in rotated[1:])]
                yield scored(base, "rot4_std", np.mean(runs, axis=0), fold)
