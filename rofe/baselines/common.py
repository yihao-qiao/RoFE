import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

ROWS_FILE = "fold{fold}.json"  # labeled row ids of every draw of a fold


# read by the baselines that run in another environment
def export_rows(fold, draws, directory: Path) -> Path:
    entries = [{"shots": int(d.shots), "repeat": int(d.repeat),
                "row_ids": [fold.ids[i] for i in d.rows]} for d in draws]
    record = {"task": fold.task, "fold": int(fold.fold),
              "labels": [int(v) for v in fold.labels],
              "query_row_ids": [fold.ids[i] for i in fold.query], "entries": entries}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ROWS_FILE.format(fold=fold.fold)
    path.write_text(json.dumps(record))
    return path


def read_rows(directory: Path, fold: int) -> dict:
    return json.loads((Path(directory) / ROWS_FILE.format(fold=fold)).read_text())


def save_prediction(path: Path, classes: np.ndarray, proba: np.ndarray, y_query: np.ndarray,
                    labels: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, classes=np.asarray(classes), proba=np.asarray(proba, np.float64),
             y_query=np.asarray(y_query), labels=np.asarray(labels),
             meta=np.array(json.dumps(meta, default=_plain)))


def score_predictions(task: str, directory: Path) -> Iterator[dict]:
    from ..metrics import scores

    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["meta"]))
            values = scores(saved["classes"], saved["proba"], saved["y_query"], saved["labels"])
        if meta.get("task", task) != task:
            raise ValueError(f"{path} holds another task")
        yield {"task": task, **meta, **values}


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
