import math
import zlib

import numpy as np
import pandas as pd

MAX_CLASSES = 10  # a task with more classes keeps ten drawn per fold
CLASS_STREAM = 11000


def scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Infinite CSV value")
    if not isinstance(value, str | int | float | bool):
        raise TypeError(f"Unsupported CSV scalar: {type(value)}")
    return value


def class_draw(task: str, fold: int, classes) -> list:
    classes = sorted(set(classes))
    if len(classes) <= MAX_CLASSES:
        return classes
    # ten classes drawn uniformly, seeded per task and fold
    rng = np.random.default_rng([CLASS_STREAM, zlib.crc32(task.encode()), int(fold)])
    keep = rng.choice(len(classes), size=MAX_CLASSES, replace=False)
    return sorted(classes[int(i)] for i in keep)


def restrict_classes(task: str, fold: int, xs: pd.DataFrame, ys: pd.Series, xq: pd.DataFrame,
                     yq: pd.Series) -> tuple:
    kept = class_draw(task, fold, ys.unique())
    if len(kept) == ys.nunique():
        return xs, ys, xq, yq, None
    if not set(yq.unique()) <= set(ys.unique()):
        raise ValueError("Query classes absent from the labeled pool")
    keep_s, keep_q = ys.isin(kept), yq.isin(kept)
    if set(yq[keep_q].unique()) != set(kept):
        raise ValueError("A kept class has no query rows")
    drawn = {"rule": "R1", "support_classes": int(ys.nunique()),
             "kept_classes": [scalar(v) for v in kept],
             "dropped_classes": [scalar(v) for v in sorted(set(ys.unique()) - set(kept))],
             "support_rows_dropped": int((~keep_s).sum()),
             "query_rows_dropped": int((~keep_q).sum())}
    return xs[keep_s], ys[keep_s], xq[keep_q], yq[keep_q], drawn
