from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..pfn.tabpfn_predict import SETTINGS, design, verify_checkpoint


def classifier(version: str, checkpoint: Path, seed: int, categorical, unlimited: bool = False):
    # the version's released defaults; unlimited lifts the pretraining limits (anchor fit only,
    # as in the CoMET-interface runs it pairs with)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    extra = {"ignore_pretraining_limits": True} if unlimited else {}
    return TabPFNClassifier.create_default_for_version(
        ModelVersion(version), model_path=str(checkpoint), device="cuda", random_state=seed,
        categorical_features_indices=categorical, **extra)


def predict_file(path: Path, version: str, checkpoint: Path) -> dict:
    stem = path.name[:-len("_meta.npz")]
    with np.load(path) as m, np.load(path.with_name(f"{stem}_blocks.npz")) as b:
        meta, blocks = dict(m), dict(b)
    unlimited = {str(j) for j in meta.get("unlimited_jobs", ())}
    saved = {}
    for setting in SETTINGS:
        for job, block, seed in zip(meta["job_names"], meta["job_blocks"], meta["job_seeds"]):
            X_train, y_train, X_test, categorical = design(meta, blocks, setting, str(block))
            model = classifier(version, checkpoint, int(seed), categorical,
                               str(job) in unlimited)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            if not np.isfinite(proba).all():
                raise ValueError(f"Nonfinite probabilities for {stem} {job}")
            saved[f"{setting}_{job}_proba"] = proba
            saved[f"{setting}_{job}_classes"] = model.classes_
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabPFN predictions for the exported PCA sweep (TabPFN environment)")
    parser.add_argument("--version", choices=("v2.5", "v3.5"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = verify_checkpoint(args.checkpoint, args.version)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.exports.glob("*_meta.npz")):
        stem = path.name[:-len("_meta.npz")]
        target = args.output / f"{stem}.npz"
        if target.exists():
            continue
        start = time.time()
        np.savez(target, **predict_file(path, args.version, checkpoint))
        print(json.dumps({"file": stem, "seconds": round(time.time() - start, 1)}), flush=True)
    (args.output / "COMPLETE").write_text("predictions saved\n")


if __name__ == "__main__":
    main()
