import numpy as np
from sklearn.decomposition import PCA

MAX_WIDTH = 256  # CoMET's default PCA width


def pca_width(n: int, p: int, cap: int = MAX_WIDTH) -> int:
    return min(cap, n - 1, p)


def support_pca(x: np.ndarray, rows: np.ndarray,
                width: int | None = None) -> tuple[np.ndarray, PCA]:
    mask = np.zeros(len(x), dtype=bool)
    mask[rows] = True
    width = pca_width(len(rows), x.shape[1]) if width is None else width
    if not 1 <= width <= min(int(mask.sum()) - 1, x.shape[1]):
        raise ValueError(f"PCA width {width} exceeds the labeled rows")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite embeddings")
    pca = PCA(n_components=width, whiten=False, svd_solver="full")
    pca.fit(x[mask])  # mean and V_w from the labeled rows only
    return pca.transform(x), pca  # Z = (X - mean) V_w, (N, w), in the dtype of x


def gavish_donoho_rank(x: np.ndarray, rows: np.ndarray, cap: int) -> int:
    z = np.asarray(x, np.float64)[rows]
    m = min(len(rows) - 1, z.shape[1])
    singular = np.linalg.svd(z - z.mean(0), compute_uv=False)[:m]  # σ of the centered rows
    beta = m / max(len(rows) - 1, z.shape[1])
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43  # ω(β), unknown noise
    # k = #{j: σ_j > ω(β) median(σ)}, clipped to [1, cap]
    return int(np.clip((singular > omega * np.median(singular)).sum(), 1, cap))
