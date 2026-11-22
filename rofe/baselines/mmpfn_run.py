# MultiModalPFN with the authors' code on exported labeled rows (its own Python 3.10 environment)
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import numpy as np

OWN_TASKS = ("petfinder", "pad_ufes_20", "cbis_mass", "cbis_calc")  # MultiModalPFN's own tasks
# author configs whose head grids must agree for a MulTaBench task
MULTABENCH_CONFIGS = ("pad_ufes_20", "petfinder-adoption-prediction", "cbis_ddsm")
GRID_SIZE = 39  # (mgm, cap) head pairs with mgm >= cap
FINETUNING = {"learning_rate": 1e-5, "batch_size": 1, "max_steps": 100}  # the authors' settings
TIME_LIMIT = 60
AUTHOR_CROSS_VAL_SPLITS = 10
INPUTS = {"image_table": "joint", "image_only": "embedding"}  # --inputs -> setting
GPU_TYPES = ("L40S", "A100")
COMMIT = "e7a8f93205a9901f199dead7d39b9afc78d1471d"  # pinned MultiModalPFN commit
# summary line of the authors' fine-tuning log
REPORT = re.compile(r"Initial Validation Loss: (\S+) Best Validation Loss: (\S+) "
                    r"Total Steps: (\d+) Best Step: (\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_name(task: str) -> str:
    if task.startswith("cbis_"):
        return "cbis_ddsm"
    return "petfinder-adoption-prediction" if task == "petfinder" else "pad_ufes_20"


def author_configuration(source: Path, task: str, candidate: int) -> dict:
    import yaml

    names = MULTABENCH_CONFIGS if task not in OWN_TASKS else (config_name(task),)
    configs = [yaml.safe_load((source / f"mmpfn/configs/{n}.yaml").read_text()) for n in names]
    keys = ("mgm_heads_list", "cap_heads_list", "features_per_group", "mixer_type")
    if any({k: c.get(k) for k in keys} != {k: configs[0].get(k) for k in keys} for c in configs):
        raise ValueError("Author search configurations differ")
    config = configs[0]
    grid = [(m, c) for m, c in product(config["mgm_heads_list"], config["cap_heads_list"])
            if m >= c]
    if len(grid) != GRID_SIZE or not 0 <= candidate < GRID_SIZE:
        raise ValueError("Unexpected author grid or candidate")
    mgm, cap = grid[candidate]  # c-th pair with mgm >= cap, product order
    return {"mixer_type": config.get("mixer_type", "MGM+CAP"),
            "features_per_group": config["features_per_group"], "mgm_heads": mgm,
            "cap_heads": cap}


def load_bank(directory: Path):
    meta = json.loads((directory / "manifest.json").read_text())
    for name in ("features.npy", "row_ids.json"):
        if sha256_file(directory / name) != meta["sha256"][name]:
            raise ValueError(f"Checksum mismatch: {directory / name}")
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    ids = json.loads((directory / "row_ids.json").read_text())
    return features, {key: i for i, key in enumerate(ids)}


# the authors' fill: missing -> column minimum - 1
def source_fill_missing(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    for column in result.T:
        column[np.isnan(column)] = np.nanmin(column) - 1
    return result


# the authors' fill per split; an all-missing query column has no minimum and takes
# the support's minimum - 1 instead
def fill_support_query(support: np.ndarray, query: np.ndarray):
    query = query.copy()
    observed = ~np.isnan(support).all(axis=0)
    fallback = np.flatnonzero(np.isnan(query).all(axis=0) & observed).tolist()
    for index in fallback:
        query[:, index] = np.nanmin(support[:, index]) - 1
    return source_fill_missing(support), source_fill_missing(query), fallback


def multabench_arrays(data: Path, task: str, fold: int) -> dict:
    import pandas as pd
    import torch
    from sklearn.preprocessing import OrdinalEncoder

    manifest = json.loads((data / "tasks" / task / f"seed{fold}.json").read_text())
    rows, cats, columns = (manifest["rows"], manifest["categorical_columns"],
                           manifest["table_columns"])
    numeric = [c for c in columns if c not in cats]
    order = [*cats, *numeric]
    frame = pd.DataFrame([r["table"] for r in rows], columns=order)
    frame = frame.where(frame.notna(), np.nan)
    encoded = (OrdinalEncoder().fit_transform(frame[cats]) if cats
               else np.empty((len(frame), 0), dtype=np.float64))
    # (N, columns): ordinal codes, then numbers, NaN left unfilled
    values = np.concatenate([encoded, frame[numeric].to_numpy(dtype=np.float64)], axis=1)
    support = np.array([r["split"] == "support" for r in rows])
    y = np.array([r["target"] for r in rows], dtype=np.int64)
    classes = np.arange(len(manifest["classes"]))
    if (not 2 <= len(classes) <= 10 or not np.array_equal(np.unique(y[support]), classes)
            or any(len(r["image_ids"]) != 1 for r in rows)):
        raise ValueError("Unexpected rows, classes or images for MultiModalPFN")
    features, lookup = load_bank(data / "features" / "dinov3" / task)
    images = torch.from_numpy(np.stack([features[lookup[r["image_ids"][0]]]
                                        for r in rows])[:, None, :])  # (N, 1, 768)
    return {"family": "multabench", "values": values, "row_ids": [r["row_id"] for r in rows],
            "y": y, "support": support, "images": images, "n_categorical": len(cats)}


# the authors' table encoding of their own tasks, fitted at the scope of their loaders
def own_table(audit: dict, source_root: Path | None) -> np.ndarray:
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder

    rows = audit["source_rows"]
    categorical, numeric = audit["categorical_columns"], audit["numeric_columns"]
    if audit["task"].startswith("cbis_"):  # one encoder per source CSV, before exclusions
        root = Path(audit["root"]) if source_root is None else source_root
        encoded = np.empty((len(rows), len(categorical) + len(numeric)))
        for relative in dict.fromkeys(r["source_csv"] for r in rows):
            path = root / relative
            if sha256_file(path) != audit["csv_sha256"][relative]:
                raise ValueError(f"Source CSV differs: {path}")
            frame = pd.read_csv(path)
            cats = OrdinalEncoder().fit_transform(frame[categorical])
            full = np.concatenate([cats, frame[numeric].to_numpy()], axis=1)
            positions = [i for i, r in enumerate(rows) if r["source_csv"] == relative]
            encoded[positions] = full[[rows[i]["source_row"] for i in positions]]
        return encoded
    retained = pd.DataFrame([r["table"] for r in rows], columns=categorical + numeric)
    retained = retained.where(retained.notna(), np.nan)
    cats = OrdinalEncoder().fit_transform(retained[categorical])
    return np.concatenate([cats, retained[numeric].to_numpy()], axis=1).astype(float)


def own_arrays(data: Path, task: str, fold: int, source_root: Path | None) -> dict:
    import torch

    directory = data / "tasks" / task
    audit = json.loads((directory / "audit.json").read_text())
    path = directory / f"seed{fold}.json"
    pins = audit.get("manifest_sha256", {})
    if path.name in pins and sha256_file(path) != pins[path.name]:
        raise ValueError("Manifest differs from its preparation audit")
    manifest = json.loads(path.read_text())
    x = own_table(audit, source_root)
    lookup_rows = {r["row_id"]: i for i, r in enumerate(audit["source_rows"])}
    rows = manifest["rows"]
    if set(lookup_rows) != {r["row_id"] for r in rows}:
        raise ValueError("Manifest rows differ from the prepared source rows")
    ordered = np.asarray([lookup_rows[r["row_id"]] for r in rows])
    features, lookup = load_bank(data / "features" / "dinov3" / task)
    images = torch.from_numpy(np.stack(
        [np.stack([features[lookup[k]] for k in r["image_ids"]]) for r in rows]))  # (N, v, 768)
    return {"family": "own", "values": x[ordered], "row_ids": [r["row_id"] for r in rows],
            "y": np.asarray([r["target"] for r in rows]),
            "support": np.asarray([r["split"] == "support" for r in rows]), "images": images,
            "n_categorical": len(audit["categorical_columns"])}


def labeled_arrays(arrays: dict, labeled_ids: list, query_ids: list) -> dict:
    position = {key: i for i, key in enumerate(arrays["row_ids"])}
    rows = np.array([position[key] for key in labeled_ids])
    support = np.asarray(arrays["support"])
    query = np.flatnonzero(~support)
    if not support[rows].all() or [arrays["row_ids"][i] for i in query] != list(query_ids):
        raise ValueError("Exported rows do not match the fold")
    values = np.asarray(arrays["values"], dtype=np.float64)
    in_pool = ~np.isnan(values[support]).all(axis=0)
    # columns observed in the pool but not on the labeled rows are left out
    keep = ~np.isnan(values[rows]).all(axis=0) | ~in_pool
    train, test = values[rows][:, keep], values[query][:, keep]
    if arrays["family"] == "multabench":
        xs, xq, _ = fill_support_query(train, test)
    else:
        xs, xq = source_fill_missing(train), source_fill_missing(test)
    y = np.asarray(arrays["y"])
    return {"X_train": xs, "X_test": xq, "y_train": y[rows], "y_test": y[query],
            "image_train": arrays["images"][rows], "image_test": arrays["images"][query],
            "categorical_indices": list(range(int(keep[:arrays["n_categorical"]].sum()))),
            "dropped_columns": int((~keep).sum())}


# min(10, largest class count): the authors' 10 splits unless every class has fewer rows
def cross_val_splits_for(y_train) -> int:
    _, counts = np.unique(np.asarray(y_train).ravel(), return_counts=True)
    if int(counts.max()) < 2:
        raise ValueError("Every class has a single training row")
    return min(AUTHOR_CROSS_VAL_SPLITS, int(counts.max()))


# the authors' loader, with fewer cross-validation splits only at few labels
@contextmanager
def scoped_cross_val_splits(main_module, data_module, record: dict):
    original_loader = main_module.get_data_loader
    original_dataset = data_module.ImageTabularDataset

    def loader(**kwargs):
        n = cross_val_splits_for(kwargs["y_train"])
        record.update(cross_val_splits=n)
        if n == AUTHOR_CROSS_VAL_SPLITS:
            return original_loader(**kwargs)
        data_module.ImageTabularDataset = functools.partial(original_dataset, cross_val_splits=n)
        try:
            return original_loader(**kwargs)
        finally:
            data_module.ImageTabularDataset = original_dataset

    main_module.get_data_loader = loader
    try:
        yield record
    finally:
        main_module.get_data_loader = original_loader
        data_module.ImageTabularDataset = original_dataset


class _Messages(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def check_clone(source: Path) -> None:
    import subprocess

    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise ValueError(f"MultiModalPFN clone is at {head}, expected {COMMIT}")
    if not (source / "mmpfn/parameters/tabpfn-v2-classifier.ckpt").is_file():
        raise FileNotFoundError("Place tabpfn-v2-classifier.ckpt in mmpfn/parameters")


def import_author(source: Path):
    sys.path.insert(0, str(source))
    from mmpfn.models.mmpfn import MMPFNClassifier
    from mmpfn.models.mmpfn.constants import ModelInterfaceConfig
    from mmpfn.models.mmpfn.preprocessing import PreprocessorConfig
    from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import fine_tune_mmpfn

    main_module = importlib.import_module(fine_tune_mmpfn.__module__)
    data_module = importlib.import_module("mmpfn.scripts_finetune_mm.training_utils.data_utils")
    for obj in (MMPFNClassifier, fine_tune_mmpfn, data_module):
        if not Path(inspect.getfile(obj)).resolve().is_relative_to(source):
            raise ValueError("Imported another MultiModalPFN installation")
    return {"classifier": MMPFNClassifier, "interface": ModelInterfaceConfig,
            "preprocessor": PreprocessorConfig, "fine_tune": fine_tune_mmpfn,
            "main_module": main_module, "data_module": data_module}


def fit_predict(author: dict, data: dict, kwargs: dict, inputs: str, fold: int,
                checkpoint: Path):
    import pandas as pd
    import torch

    table = inputs == "image_table"
    cats = data["categorical_indices"] if table else None
    details = {}
    torch.manual_seed(fold)  # fine-tuning seed = fold
    started = time.perf_counter()
    handler = _Messages()
    logger = getattr(author["main_module"], "logger", None)
    if logger is not None:
        logger.addHandler(handler)
    try:
        with scoped_cross_val_splits(author["main_module"], author["data_module"], details):
            author["fine_tune"](
                save_path_to_fine_tuned_model=checkpoint, time_limit=TIME_LIMIT,
                finetuning_config=dict(FINETUNING), validation_metric="log_loss",
                X_train=pd.DataFrame(data["X_train"]) if table else None,
                image_train=data["image_train"], y_train=pd.Series(data["y_train"]),
                categorical_features_index=cats, device="cuda", task_type="multiclass",
                show_training_curve=False, logger_level=0, freeze_input=True, **kwargs)
    finally:
        if logger is not None:
            logger.removeHandler(handler)
    found = [m for m in (REPORT.search(line) for line in handler.messages) if m]
    if len(found) == 1:
        details["initial_validation_loss"] = float(found[0][1])
        details["best_validation_loss"] = float(found[0][2])
        details["total_steps"] = int(found[0][3])
        details["best_step"] = int(found[0][4])
    model = author["classifier"](
        model_path=checkpoint, ignore_pretraining_limits=True,
        inference_config=author["interface"](
            FINGERPRINT_FEATURE=False,
            PREPROCESS_TRANSFORMS=[author["preprocessor"](name="none")]),
        categorical_features_indices=cats, **kwargs)
    if model.n_estimators != 4 or model.random_state != 0:
        raise ValueError("Author inference defaults changed")
    model.fit(data["X_train"] if table else None, data["image_train"], data["y_train"])
    proba = model.predict_proba(data["X_test"] if table else None, data["image_test"])
    details["seconds"] = time.perf_counter() - started
    classes = np.asarray(model.classes_)
    del model
    torch.cuda.empty_cache()
    return classes, np.asarray(proba, dtype=np.float64), details


def gpu_type(device_name: str) -> str | None:
    found = [t for t in GPU_TYPES if t in device_name]
    return found[0] if len(found) == 1 else None


# (candidate, kwargs, file suffix, method): the selected candidate, or all 39
def candidate_runs(source: Path, task: str, choice: dict | None, every: bool) -> list[tuple]:
    if not every:
        kwargs = author_configuration(source, task, int(choice["candidate"]))
        if any(k in choice and choice[k] != kwargs[k] for k in ("mgm_heads", "cap_heads")):
            raise ValueError("Selected heads differ from the authors' grid")
        return [(int(choice["candidate"]), kwargs, "mmpfn", "mmpfn")]
    runs = [(c, author_configuration(source, task, c), f"c{c:02d}", f"mmpfn_c{c:02d}")
            for c in range(GRID_SIZE)]
    if len({(k["mgm_heads"], k["cap_heads"]) for _, k, _, _ in runs}) != GRID_SIZE:
        raise ValueError("The authors' candidates do not have distinct heads")
    return runs


def rows_digest(row_ids: list) -> str:
    return hashlib.sha256("\n".join(row_ids).encode()).hexdigest()


def run_fold(args, task: str, fold: int, choice: dict | None, runs: list[tuple],
             author: dict) -> None:
    import torch

    record = json.loads((args.exports / f"fold{fold}.json").read_text())
    arrays = (own_arrays(args.data, task, fold, args.source_root) if task in OWN_TASKS
              else multabench_arrays(args.data, task, fold))
    labels = np.array(record["labels"])
    device = torch.cuda.get_device_name()
    expected = (choice or {}).get("reference_gpu", [None] * 5)[fold]
    if expected and gpu_type(device) != expected:
        message = f"{task} fold {fold}: running on {device}, the selected cell ran on {expected}"
        if args.match_gpu:
            raise ValueError(message)
        print(json.dumps({"warning": message}), flush=True)
    for entry in record["entries"]:  # every candidate of a draw before the next draw
        stem = f"f{fold}_s{entry['shots']}_r{entry['repeat']}_dinov3_{INPUTS[args.inputs]}"
        for candidate, kwargs, suffix, method in runs:
            target = args.output / f"{stem}_{suffix}.npz"
            if target.exists():
                continue
            work = args.output / "work" / f"{stem}_{suffix}"
            (work / "logs").mkdir(parents=True, exist_ok=True)
            os.chdir(work)  # the fine-tuner writes ./logs
            data = labeled_arrays(arrays, entry["row_ids"], record["query_row_ids"])
            if not np.array_equal(np.unique(data["y_train"]), labels):
                raise ValueError("Labeled rows miss a class of the pool")
            checkpoint = work / "finetuned.ckpt"
            classes, proba, details = fit_predict(author, data, kwargs, args.inputs, fold,
                                                  checkpoint)
            checkpoint.unlink(missing_ok=True)
            meta = {"task": task, "fold": fold, "shots": entry["shots"],
                    "repeat": entry["repeat"], "n_labeled": len(entry["row_ids"]),
                    "row_ids_sha256": rows_digest(entry["row_ids"]),
                    "setting": INPUTS[args.inputs], "encoder": "dinov3", "method": method,
                    "candidate": candidate, **{k: kwargs[k] for k in ("mgm_heads", "cap_heads")},
                    "dropped_columns": (data["dropped_columns"] if args.inputs == "image_table"
                                        else 0),
                    "gpu": device, **details}
            if method != "mmpfn" and choice is not None:
                meta["grid_selected_candidate"] = int(choice["candidate"])
            args.output.mkdir(parents=True, exist_ok=True)
            np.savez(target, classes=classes, proba=proba, y_query=np.asarray(data["y_test"]),
                     labels=labels, meta=np.array(json.dumps(meta)))
            print(json.dumps({k: meta[k] for k in ("fold", "shots", "repeat", "setting",
                                                   "candidate")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MultiModalPFN on exported labeled rows (MultiModalPFN environment)")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mmpfn-source", type=Path, required=True,
                        help="MultiModalPFN clone with the TabPFN v2 weight in mmpfn/parameters")
    parser.add_argument("--selection", type=Path,
                        help="per-task configuration of the authors' grid (and its GPU types)")
    parser.add_argument("--candidates", choices=("selected", "all"), default="selected",
                        help="the selected configuration, or all 39 of the authors' grid")
    parser.add_argument("--inputs", choices=sorted(INPUTS), default="image_table")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--source-root", type=Path,
                        help="raw CBIS-DDSM directory, if moved since preparation")
    parser.add_argument("--match-gpu", action="store_true",
                        help="stop unless the GPU type equals the selected cell's")
    args = parser.parse_args()
    if args.selection is None and args.candidates == "selected":
        parser.error("--selection is needed unless --candidates all")
    args.data, args.exports = args.data.resolve(), args.exports.resolve()
    args.output = args.output.resolve()
    source = args.mmpfn_source.resolve()
    check_clone(source)
    choice = (None if args.selection is None
              else json.loads(args.selection.read_text())["tasks"][args.task])
    runs = candidate_runs(source, args.task, choice, args.candidates == "all")
    start = Path.cwd()
    (args.output / "work" / "logs").mkdir(parents=True, exist_ok=True)
    os.chdir(args.output / "work")  # the fine-tuner opens ./logs at import
    try:
        author = import_author(source)
        for fold in args.folds:
            run_fold(args, args.task, fold, choice, runs, author)
    finally:
        os.chdir(start)
    (args.output / f"COMPLETE_{args.inputs}").write_text(
        f"MultiModalPFN predictions saved, {len(runs)} configurations\n")


if __name__ == "__main__":
    main()
