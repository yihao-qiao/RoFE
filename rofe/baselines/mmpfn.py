from collections.abc import Iterator
from pathlib import Path

from ..experiments.common import labeled_sets
from ..tasks import Fold, modality
from .common import export_rows, score_predictions


# read by mmpfn_run in the MultiModalPFN environment
def export(fold: Fold, cfg: dict, ctx, directory: Path) -> None:
    if modality(fold.task) != "image":
        raise ValueError("MultiModalPFN runs on the image tasks only")
    export_rows(fold, labeled_sets(fold, cfg), directory)


# saved predictions, every candidate of a sweep included
def score(task: str, directory: Path, cfg: dict, ctx) -> Iterator[dict]:
    wanted = cfg.get("settings", ("embedding", "joint"))
    methods = cfg.get("methods")
    for record in score_predictions(task, Path(directory) / "predictions"):
        if record["setting"] in wanted and (methods is None or record["method"] in methods):
            yield record
