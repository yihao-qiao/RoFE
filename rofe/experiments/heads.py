from collections.abc import Iterator

import numpy as np

from ..heads import HEADS, head_inputs, make_head, svm_status
from ..metrics import scores
from ..table import encoded
from ..tasks import Fold
from . import Context, register_experiment
from .common import base_record, labeled_sets, settings


# linear probe, kNN, linear SVM and MLP on the raw embedding; 'joint' puts the table first
@register_experiment("heads")
def run(fold: Fold, cfg: dict, ctx: Context) -> Iterator[dict]:
    heads = cfg.get("methods", HEADS)
    embeddings = {e: fold.embedding(e).astype(np.float64) for e in cfg["encoders"]}
    for draw in labeled_sets(fold, cfg):
        # one-hot table, encoding fitted on the labeled rows
        dense = (encoded(fold.table, fold.categorical, draw.rows, one_hot=True)
                 if fold.table.shape[1] else None)
        for encoder, x in embeddings.items():
            for setting in settings(fold, cfg):
                X = head_inputs(x, dense if setting == "joint" else None)
                for head in heads:
                    model = make_head(head, len(draw.rows), fold.fold)
                    model.fit(X[draw.rows], fold.y[draw.rows])
                    proba = np.asarray(model.predict_proba(X[fold.query]), np.float64)
                    extra = svm_status(model[-1]) if head == "linear_svm" else {}
                    yield {**base_record(fold, draw, setting, encoder), "method": head, **extra,
                           **scores(model.classes_, proba, fold.y[fold.query], fold.labels)}
