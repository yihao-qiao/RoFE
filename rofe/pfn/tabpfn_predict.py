from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SETTINGS = ("embedding", "joint")  # embedding alone, table + embedding
V2_ESTIMATORS = 4  # TabPFN v2 ensemble members
# version -> (repository, file, SHA-256); the v2 file ships with MultiModalPFN's TabPFN v2
CHECKPOINTS = {
    "v2": ("Prior-Labs/TabPFN-v2-clf@f851f2a3c941544733b712d8c0f96dfae9b28862",
           "tabpfn-v2-classifier.ckpt",
           "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f"),
    "v2.5": ("Prior-Labs/tabpfn_2_5@6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
             "tabpfn-v2.5-classifier-v2.5_default.ckpt",
             "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12"),
    "v3.5": ("Prior-Labs/tabpfn_3_5@06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
             "tabpfn-v3.5-20260909.safetensors",
             "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"),
}


def verify_checkpoint(path: Path, version: str) -> Path:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINTS[version][2]:
        raise ValueError(f"{path} is not the pinned TabPFN {version} checkpoint")
    return Path(path)


def design(meta: dict, blocks: dict, setting: str, block: str):
    # rows: the labeled rows, then the query rows; table columns empty on the labeled rows
    # are left out of the fit
    n = len(meta["rows"])
    x = blocks[block]  # (n + n_q, ·)
    if setting == "embedding":
        X, categorical = x, None
    else:
        table = blocks["table"]  # ordinal codes, categorical columns first
        seen = ~np.isnan(table[:n]).all(0)
        kept = np.flatnonzero(seen)
        X = np.hstack([table[:, kept], x])
        # categorical indices remapped to the kept columns
        categorical = [int(np.searchsorted(kept, int(i))) for i in blocks["categorical"]
                       if seen[int(i)]]
    return X[:n], meta["y_rows"], X[n:], categorical  # X_train, y_train, X_test, categorical


def classifier(version: str, checkpoint: Path, seed: int, categorical):
    if version == "v2":
        from mmpfn.models.tabpfn_v2 import TabPFNClassifier

        return TabPFNClassifier(n_estimators=V2_ESTIMATORS, random_state=seed, device="cuda",
                                model_path=checkpoint, ignore_pretraining_limits=True,
                                categorical_features_indices=categorical)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    # v2.5 and v3.5: the released defaults of the version
    return TabPFNClassifier.create_default_for_version(
        ModelVersion(version), model_path=str(checkpoint), device="cuda", random_state=seed,
        ignore_pretraining_limits=True, categorical_features_indices=categorical)


def predict_file(path: Path, version: str, checkpoint: Path) -> dict:
    stem = path.name[:-len("_meta.npz")]
    with np.load(path) as m, np.load(path.with_name(f"{stem}_blocks.npz")) as b:
        meta, blocks = dict(m), dict(b)
    saved = {}
    for setting in SETTINGS:
        for job, block, seed in zip(meta["job_names"], meta["job_blocks"], meta["job_seeds"]):
            X_train, y_train, X_test, categorical = design(meta, blocks, setting, str(block))
            model = classifier(version, checkpoint, int(seed), categorical)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)  # (n_q, c_fit)
            if not np.isfinite(proba).all():
                raise ValueError(f"Nonfinite probabilities for {stem} {job}")
            saved[f"{setting}_{job}_proba"] = proba
            saved[f"{setting}_{job}_classes"] = model.classes_
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabPFN predictions for exported labeled sets (TabPFN environment)")
    parser.add_argument("--version", choices=("v2", "v2.5", "v3.5"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mmpfn-source", type=Path)
    args = parser.parse_args()
    if args.version == "v2":
        if args.mmpfn_source is None:
            parser.error("TabPFN v2 needs --mmpfn-source")
        sys.path.insert(0, str(args.mmpfn_source.resolve()))
    checkpoint = verify_checkpoint(args.checkpoint, args.version)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.exports.glob("*_meta.npz")):
        stem = path.name[:-len("_meta.npz")]
        target = args.output / f"{stem}.npz"
        if target.exists():  # finished files are skipped
            continue
        start = time.time()
        saved = predict_file(path, args.version, checkpoint)
        np.savez(target, **saved)
        print(json.dumps({"file": stem, "seconds": round(time.time() - start, 1)}), flush=True)
    (args.output / "COMPLETE").write_text("predictions saved\n")


if __name__ == "__main__":
    main()
