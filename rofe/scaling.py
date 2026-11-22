from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

NORM_METHODS = "none"  # TabICL without its power-transform members


# stands in for TabICL's per-column standardization (CustomStandardScaler)
class IdentityScaler:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# stands in for TabICL's outlier clip (OutlierRemover)
class IdentityClip(IdentityScaler):
    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold


# TabICL models fitted inside the block skip the chosen preprocessing steps
@contextmanager
def bypass(scaler: bool, clip: bool = False) -> Iterator[None]:
    import tabicl._sklearn.preprocessing as tp

    saved = tp.CustomStandardScaler, tp.OutlierRemover
    if scaler:
        tp.CustomStandardScaler = IdentityScaler
    if clip:
        tp.OutlierRemover = IdentityClip
    try:
        yield
    finally:
        tp.CustomStandardScaler, tp.OutlierRemover = saved


def global_scaled(block: np.ndarray, rows: np.ndarray) -> np.ndarray:
    spread = float(np.asarray(block, np.float64)[rows].var(0).mean())  # s^2 = mean_j Var(z_j)
    if not spread > 0:
        raise ValueError("Constant block")
    # Z / s: one scalar for all columns, mean labeled column variance 1, column ratios kept
    return (np.asarray(block, np.float64) / np.sqrt(spread)).astype(np.float32)


# every member runs the single 'none' pipeline with exactly the intended scaler and clip
def check_pipelines(model, scaler: bool, clip: bool = False) -> None:
    import tabicl._sklearn.preprocessing as tp

    pipes = model.ensemble_generator_.preprocessors_
    expected = (IdentityScaler if scaler else tp.CustomStandardScaler,
                IdentityClip if clip else tp.OutlierRemover)
    if set(pipes) != {"none"} or any(
            p.normalizer_ is not None or type(p.standard_scaler_) is not expected[0]
            or type(p.outlier_remover_) is not expected[1] for p in pipes.values()):
        raise ValueError("TabICL preprocessing not as intended")
