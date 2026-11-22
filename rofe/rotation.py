from collections.abc import Sequence

import numpy as np

FIRST_STREAM = 9100  # first rotation
EXTRA_STREAM = 9200
EXTRA_SIZE = 9  # entries 0-2: rotations 2-4; 3, 4: strong, weak; 5: 32 columns; 6-8 unused
MORE_STREAM = 10600  # rotations 5-16
MORE_SIZE = 12
ROTATIONS = 4  # K
MAX_ROTATIONS = ROTATIONS + MORE_SIZE


def haar_rotation(width: int, seed: int) -> np.ndarray:
    gaussian = np.random.default_rng([seed, width]).standard_normal((width, width))
    q, b = np.linalg.qr(gaussian)  # G = Q B, G ~ N(0, 1)
    return q * np.sign(np.diag(b))  # R = Q diag(sign(diag(B))), Haar on O(w), (w, w)


def rotate(block: np.ndarray, seed: int, columns: Sequence[int] | None = None) -> np.ndarray:
    out = np.asarray(block, np.float64).copy()
    columns = np.arange(out.shape[1]) if columns is None else np.asarray(columns, int)
    if len(columns) > 1:  # a single column is left as it is
        out[:, columns] = out[:, columns] @ haar_rotation(len(columns), seed)  # Z_A R, (N, |A|)
    return out.astype(np.float32)


def first_seed(fold: int, shots: int, repeat: int, stream: int, dimension: int,
               weak: int) -> int:
    rng = np.random.default_rng([FIRST_STREAM, fold, shots, repeat, stream])
    if weak > 0:
        rng.standard_normal((dimension, weak))  # stream position of the seed
    return int(rng.integers(2 ** 31))


def extra_seeds(fold: int, shots: int, repeat: int, stream: int) -> np.ndarray:
    rng = np.random.default_rng([EXTRA_STREAM, fold, shots, repeat, stream])
    return rng.integers(2 ** 31, size=EXTRA_SIZE)


def rotation_seeds(fold: int, shots: int, repeat: int, stream: int, dimension: int,
                   weak: int, count: int = ROTATIONS) -> list[int]:
    # K seeds of one labeled set, shared by every query row
    # shots: 0 for the whole pool; stream: the encoder; dimension: embedding width;
    # weak: principal columns beyond the Gavish-Donoho rank
    if not 1 <= count <= MAX_ROTATIONS:
        raise ValueError(f"At most {MAX_ROTATIONS} rotations have fixed seeds")
    first = first_seed(fold, shots, repeat, stream, dimension, weak)
    rest = extra_seeds(fold, shots, repeat, stream)[:min(count, ROTATIONS) - 1]
    seeds = [first, *(int(s) for s in rest)]
    if count > ROTATIONS:
        more = np.random.default_rng([MORE_STREAM, fold, shots, repeat, stream])
        seeds += [int(s) for s in more.integers(2 ** 31, size=MORE_SIZE)[:count - ROTATIONS]]
    return seeds
