from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import padded
from .pca import gavish_donoho_rank, pca_width, support_pca
from .rotation import rotate, rotation_seeds

ROTATIONS = 4  # K


@dataclass(frozen=True)
class LabeledScores:
    scores: np.ndarray  # (N, w) float32, every row of the fold
    width: int  # w = min(256, n - 1, p)
    gd_rank: int  # Gavish-Donoho rank k
    seeds: tuple[int, ...]

    def rotated(self, count: int | None = None) -> list[np.ndarray]:
        return [rotate(self.scores, s) for s in self.seeds[:count]]  # [Z R_k], each (N, w)


def labeled_scores(x: np.ndarray, rows: np.ndarray, fold: int, shots: int, repeat: int,
                   stream: int, count: int = ROTATIONS) -> LabeledScores:
    """x: (N, p), PCA on x[rows] -> scores (N, w), Gavish-Donoho rank and K rotation seeds."""
    scores, _ = support_pca(x, rows)
    width = scores.shape[1]
    k = gavish_donoho_rank(x, rows, width)  # k strong columns, w - k weak
    seeds = rotation_seeds(fold, shots, repeat, stream, x.shape[1], width - k, count)
    return LabeledScores(scores, width, k, tuple(seeds))


# PFN input: the table columns as they are, then visual_0, ..., visual_{w-1}
def pfn_frame(block: np.ndarray, table: pd.DataFrame, joint: bool) -> pd.DataFrame:
    columns = pd.DataFrame(block, index=table.index,
                           columns=[f"visual_{i}" for i in range(block.shape[1])])
    if set(columns) & set(table):
        raise ValueError("Embedding column names collide with table fields")
    return pd.concat([table, columns], axis=1) if joint else columns


class RoFE:
    """Rotated frozen embeddings in front of a PFN; rotate=False gives the CoMET interface."""

    def __init__(self, pfn, n_rotations: int = ROTATIONS, rotate: bool = True,
                 random_state: int = 0, seeds: Sequence[Sequence[int]] | None = None):
        self.pfn = pfn
        self.n_rotations = n_rotations if rotate else 1  # CoMET: principal scores, one fit
        self.rotate = rotate
        self.random_state = random_state
        self.seeds = seeds

    def fit(self, embeddings: np.ndarray | Sequence[np.ndarray], y: np.ndarray,
            table: pd.DataFrame | None = None) -> "RoFE":
        """embeddings: (n, p) or list of (n, p_b), y: (n,); stores the rows, no training."""
        self.embeddings_ = _blocks(embeddings)
        self.y_ = np.asarray(y)
        self.table_ = pd.DataFrame(index=range(len(self.y_))) if table is None else table
        self.classes_ = np.unique(self.y_)  # (c,), sorted
        return self

    def _seeds(self, block: int) -> list[int]:
        if self.seeds is not None:
            return list(self.seeds[block])
        rng = np.random.default_rng([self.random_state, block])
        return [int(s) for s in rng.integers(2 ** 31, size=self.n_rotations)]

    def predict_proba(self, embeddings: np.ndarray | Sequence[np.ndarray],
                      table: pd.DataFrame | None = None) -> np.ndarray:
        """embeddings: (n_q, p) or list of (n_q, p_b) -> probabilities (n_q, c), mean over K."""
        query = _blocks(embeddings)
        if (table is None) != (self.table_.shape[1] == 0):
            raise ValueError("Pass table columns both to fit and to predict_proba, or to neither")
        if not getattr(self.pfn, "in_process", True):
            raise ValueError("This PFN runs through the export and score stages of scripts/run.py")
        n, m = len(self.y_), len(query[0])  # n labeled rows, m = n_q query rows
        rows, test = np.arange(n), np.arange(n, n + m)
        q_table = pd.DataFrame(index=range(m)) if table is None else table
        full = pd.concat([self.table_, q_table], ignore_index=True)
        scores = []
        for b, (a, c) in enumerate(zip(self.embeddings_, query, strict=True)):
            x = np.vstack([a, c]).astype(np.float32)  # (n + n_q, p_b)
            z, _ = support_pca(x, rows, pca_width(n, x.shape[1]))  # (n + n_q, w_b)
            # K rotated copies Z R_k, or Z alone for CoMET
            scores.append([rotate(z, s) for s in self._seeds(b)] if self.rotate else [z])
        y = np.concatenate([self.y_, np.full(m, self.y_[0])])  # query labels: unused fillers
        out = []
        for k in range(self.n_rotations):
            block = np.hstack([s[k] for s in scores])  # (n + n_q, sum_b w_b)
            X = pfn_frame(block, full, joint=full.shape[1] > 0)
            classes, proba = self.pfn.fit_predict(X, y, rows, test, self.random_state)
            out.append(padded(classes, proba, self.classes_))  # (n_q, c)
        return np.mean(out, axis=0)  # mean of the K rotations' probabilities

    def predict(self, embeddings, table=None) -> np.ndarray:
        return self.classes_[self.predict_proba(embeddings, table).argmax(1)]


def _blocks(embeddings) -> list[np.ndarray]:
    if isinstance(embeddings, np.ndarray):
        return [embeddings]
    return [np.asarray(e) for e in embeddings]
