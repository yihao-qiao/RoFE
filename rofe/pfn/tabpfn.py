from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..metrics import padded, scores
from . import register_pfn
from .tabpfn_predict import CHECKPOINTS, SETTINGS


# TabPFN runs in its own environment: labeled sets are exported, predicted, then scored
class TabPFN:
    in_process = False
    version = ""

    def __init__(self, **_):
        pass


for _version in CHECKPOINTS:  # tabpfn_v2, tabpfn_v2.5, tabpfn_v3.5
    register_pfn(f"tabpfn_{_version}")(type(f"TabPFN_{_version}", (TabPFN,), {"version": _version}))


# blocks: labeled rows first, then the query rows; jobs: (name, block, seed) of each fit;
# methods: method -> the jobs whose probabilities it averages
def write_export(directory: Path, stem: str, meta: dict, blocks: dict[str, np.ndarray],
                 jobs: list[tuple[str, str, int]], methods: dict[str, list[str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, block in blocks.items():
        if name not in ("table", "categorical") and not np.isfinite(block).all():
            raise ValueError(f"Nonfinite {name} block")
    meta = {**meta, "job_names": np.array([j[0] for j in jobs]),
            "job_blocks": np.array([j[1] for j in jobs]),
            "job_seeds": np.array([j[2] for j in jobs]),
            "method_names": np.array(list(methods)),
            "method_jobs": np.array([",".join(v) for v in methods.values()])}
    np.savez(directory / f"{stem}_meta.npz", **meta)
    np.savez(directory / f"{stem}_blocks.npz", **blocks)


def table_codes(table, categorical: list[str],
                fit_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from ..table import categorical_first, encoded

    codes = encoded(table, categorical, fit_rows, one_hot=False)  # fit_rows: the fold's pool
    return categorical_first(codes, len([c for c in table if c not in categorical]))


def score_exports(exports: Path, predictions: Path) -> Iterator[dict]:
    for path in sorted(exports.glob("*_meta.npz")):
        stem = path.name[:-len("_meta.npz")]
        with np.load(path) as m, np.load(predictions / f"{stem}.npz") as p:
            meta, saved = dict(m), dict(p)
        labels = meta["labels"]
        for setting in SETTINGS:
            def proba(job: str, setting: str = setting) -> np.ndarray:  # (n_q, c)
                return padded(saved[f"{setting}_{job}_classes"],
                              saved[f"{setting}_{job}_proba"].astype(np.float64), labels)

            for method, jobs in zip(meta["method_names"], meta["method_jobs"]):
                p = np.mean([proba(j) for j in str(jobs).split(",")], axis=0)  # mean over fits
                yield {"fold": int(meta["fold"]), "shots": int(meta["shots"]),
                       "repeat": int(meta["repeat"]), "n_labeled": int(len(meta["rows"])),
                       "setting": setting, "encoder": str(meta["encoder"]),
                       "method": str(method), "width": int(meta["width"]),
                       "gd_rank": int(meta["rank"]),
                       **scores(labels, p, meta["y_query"], labels)}
