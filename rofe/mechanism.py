import numpy as np
from scipy.linalg import hadamard

from .rotation import rotate


# orthogonal Q with diag(Q^T diag(λ) Q) = μ, μ = mean(λ), from Givens rotations
def givens_equal_variance(variances: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    w = len(variances)
    C, Q = np.diag(np.asarray(variances, np.float64)), np.eye(w)
    mu = C.trace() / w
    free = np.ones(w, dtype=bool)
    while True:
        open_ = np.flatnonzero(free & (np.abs(np.diag(C) - mu) > tolerance * max(mu, 1e-300)))
        if len(open_) < 2:
            break
        diag = np.diag(C)
        i, j = open_[np.argmax(diag[open_])], open_[np.argmin(diag[open_])]  # a > μ > d
        a, d, b = C[i, i], C[j, j], C[i, j]
        # t = tan θ solves (d - μ) t^2 - 2 b t + (a - μ) = 0, so that the new C_ii = μ
        root = np.sqrt(b * b - (d - mu) * (a - mu))
        t = (b + root) / (d - mu) if b >= 0 else (a - mu) / (b - root)  # stable root
        c = 1 / np.sqrt(1 + t * t)
        s = t * c
        R = np.array([[c, s], [-s, c]])
        Q[:, [i, j]] = Q[:, [i, j]] @ R  # Q <- Q R on columns i, j
        C[:, [i, j]] = C[:, [i, j]] @ R  # C <- R^T C R
        C[[i, j], :] = R.T @ C[[i, j], :]
        free[i] = False  # column i at variance μ, kept fixed
    if not np.allclose(np.diag(Q.T @ np.diag(variances) @ Q), mu, rtol=1e-8, atol=0):
        raise ValueError("Equal-variance rotation did not converge")
    return Q


def givens_basis(scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
    variances = scores[rows].astype(np.float64).var(0)  # λ_j on the labeled rows
    # Z Q: (N, w), equal column variances on the labeled rows
    return (scores.astype(np.float64) @ givens_equal_variance(variances)).astype(np.float32)


def hadamard_basis(scores: np.ndarray) -> np.ndarray:
    w = scores.shape[1]
    m = 1 << (w - 1).bit_length()  # m = 2^ceil(log2 w)
    padded = np.zeros((len(scores), m))
    padded[:, :w] = scores
    return (padded @ (hadamard(m) / np.sqrt(m))).astype(np.float32)  # [Z, 0] H_m / sqrt(m), (N, m)


def partial_rotation(scores: np.ndarray, columns: np.ndarray, seed: int) -> np.ndarray:
    return rotate(scores, seed, columns)  # Haar R_A on the columns A, identity elsewhere
