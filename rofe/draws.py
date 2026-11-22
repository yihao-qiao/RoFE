from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

FEW = (5, 10, 25, 50, 100)  # labeled rows per class
MANY = (250, 500, 0)  # 0 = the whole labeled pool
DRAWS = 3  # draws per level


@dataclass(frozen=True)
class Draw:
    shots: int  # labeled rows per class, 0 for the whole pool
    repeat: int
    rows: np.ndarray  # sorted row indices of the labeled set


def subsample(y: np.ndarray, pool: np.ndarray, shots: int, seed: tuple) -> np.ndarray:
    rng = np.random.default_rng(seed)  # seed = (fold, shots, repeat)
    chosen = []
    for label in np.unique(y[pool]):
        rows = pool[y[pool] == label]
        # min(shots, class size) rows of each class, without replacement
        chosen.append(rng.choice(rows, size=min(shots, len(rows)), replace=False))
    return np.sort(np.concatenate(chosen))


def labeled_rows(y: np.ndarray, pool: np.ndarray, shots: int, fold: int,
                 repeat: int = 0) -> np.ndarray:
    if shots == 0:
        if repeat:
            raise ValueError("The whole pool has a single draw")
        return np.sort(pool)
    return subsample(y, pool, shots, (fold, shots, repeat))


def many_draws(y: np.ndarray, pool: np.ndarray, fold: int,
               repeats: Sequence[int] = range(DRAWS)) -> list[Draw]:
    # whole pool first, then decreasing levels; a level equal to an earlier one is left out,
    # and so is a draw r > 0 equal to draw 0 of its level
    kept: list[tuple[int, np.ndarray]] = []
    for shots in sorted(MANY, key=lambda s: (s != 0, -s)):
        rows = labeled_rows(y, pool, shots, fold)
        if not any(np.array_equal(rows, r) for _, r in kept):
            kept.append((shots, rows))
    out = [Draw(s, 0, r) for s, r in kept] if 0 in repeats else []
    for repeat in (r for r in sorted(set(repeats)) if r):
        for shots, first in kept:
            if shots == 0:
                continue
            rows = labeled_rows(y, pool, shots, fold, repeat)
            if not np.array_equal(rows, first):
                out.append(Draw(shots, repeat, rows))
    return out


def draws(y: np.ndarray, pool: np.ndarray, fold: int, levels: Sequence[int],
          repeats: int = DRAWS) -> Iterator[Draw]:
    unknown = set(levels) - set(FEW) - set(MANY)
    if unknown:
        raise ValueError(f"Unknown levels {sorted(unknown)}")
    for shots in (s for s in levels if s in FEW):
        for repeat in range(repeats):
            yield Draw(shots, repeat, subsample(y, pool, shots, (fold, shots, repeat)))
    if set(levels) & set(MANY):
        for draw in many_draws(y, pool, fold, range(repeats)):
            if draw.shots in levels:
                yield draw
