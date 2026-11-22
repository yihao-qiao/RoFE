from collections.abc import Iterator
from pathlib import Path

from ..experiments import Context, register_experiment
from ..experiments.common import labeled_sets
from ..tasks import Fold
from . import mmpfn, zhang
from .common import export_rows, score_predictions


# baselines in their own environments: export the labeled rows, run outside, score the saved
# predictions
def _external(name: str):
    def run(fold: Fold, cfg: dict, ctx: Context) -> Iterator[dict]:
        raise ValueError(f"{name} runs in its own environment: use the export, external and "
                         "score stages")
    return run


def _export_rows(fold: Fold, cfg: dict, ctx: Context, directory: Path) -> None:
    export_rows(fold, labeled_sets(fold, cfg), directory)


def _score_saved(task: str, directory: Path, cfg: dict, ctx: Context) -> Iterator[dict]:
    wanted = cfg.get("settings", ("embedding", "joint"))
    methods = cfg.get("methods")
    for record in score_predictions(task, Path(directory) / "predictions"):
        if record["setting"] in wanted and (methods is None or record["method"] in methods):
            yield record


register_experiment("catboost", export=_export_rows, score=_score_saved)(_external("CatBoost"))
register_experiment("mmpfn", export=mmpfn.export, score=mmpfn.score)(_external("MultiModalPFN"))
register_experiment("zhang_sweep", export=zhang.export, score=zhang.score)(
    _external("The TabPFN sweep"))
register_experiment("zhang_heads")(zhang.run_heads)
