import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.bank import FeatureBank
from ..tasks import ENCODERS, bank_path, manifest_path, modality
from .common import read_rows, save_prediction

# MulTaBench's CatBoost defaults
SEED = 42
ITERATIONS = 2000
EARLY_STOPPING_ROUNDS = 50
OD_PVAL = 0.001
VAL_RATIO = 0.1  # validation rows: min(0.1 n, MAX_VAL_SIZE)
MAX_VAL_SIZE = 1000
VAL_FOLD = -1  # validation split seed = SEED + VAL_FOLD
PCA_COMPONENTS = 30  # per image view or text embedding
MISSING_VALUE = "Unknown Value"  # fill of a missing categorical value
RESOLVED = ("od_type", "od_pval", "od_wait", "random_seed", "iterations", "use_best_model",
            "loss_function")  # recorded with each fit
REFIT_COPIED = ("learning_rate", "boosting_type")  # from the early-stopped fit
REFIT_MUST_EQUAL = ("loss_function", "depth", "l2_leaf_reg", "random_seed", "border_count",
                    "learning_rate", "boosting_type")  # a mismatch is recorded as a violation


@dataclass(frozen=True)
class FoldInputs:
    ids: list[str]
    y: np.ndarray
    support: np.ndarray  # (N,) bool, the labeled pool
    columns: list[str]
    numeric_columns: list[str]
    numeric: np.ndarray  # (N, numeric) float64, NaN where missing
    categorical_columns: list[str]
    categorical: np.ndarray  # (N, categorical) str
    embeddings: dict[str, np.ndarray]  # encoder -> (N, views * dim) float32
    views: int


def load_inputs(data: Path, task: str, fold: int, encoders: list[str]) -> FoldInputs:
    manifest = json.loads(manifest_path(data, task, fold).read_text())
    rows = manifest["rows"]
    ids = [r["row_id"] for r in rows]
    y = np.array([r["target"] for r in rows])
    support = np.array([r["split"] == "support" for r in rows])
    columns = list(rows[0]["table"])
    categorical = list(manifest.get("categorical_columns", []))
    cats = [c for c in columns if c in categorical]
    nums = [c for c in columns if c not in categorical]

    def number(value) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, str):
            raise ValueError("String value in a numeric table column")
        return float(value)

    def text(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return MISSING_VALUE
        return str(value)

    numeric = np.array([[number(r["table"][c]) for c in nums] for r in rows],
                       dtype=np.float64).reshape(len(rows), len(nums))
    strings = np.array([[text(r["table"][c]) for c in cats] for r in rows],
                       dtype=object).reshape(len(rows), len(cats))
    if modality(task) == "image":
        keys = [list(r["image_ids"]) for r in rows]
    else:
        keys = [[i] for i in ids]
    embeddings = {e: FeatureBank.load(bank_path(data, e, task)).rows(keys).astype(np.float32)
                  for e in encoders}
    return FoldInputs(ids, y, support, columns, nums, numeric, cats, strings, embeddings,
                      len(keys[0]))


def merge_splits(x1: pd.Series, x2: pd.Series, y1: pd.Series, y2: pd.Series, seed: int):
    x, y = pd.concat([x1, x2]), pd.concat([y1, y2])
    shuffled = x.sample(frac=1, random_state=seed).index
    return x.loc[shuffled], y.loc[shuffled]


# TabSTAR's validation split as MulTaBench runs it; at few labels the size is raised to the
# class count and the second binary split skipped, so that the split stays valid
def validation_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    from sklearn.model_selection import train_test_split

    index = pd.RangeIndex(len(y))
    x, ys = pd.Series(np.arange(len(y)), index=index), pd.Series(np.asarray(y), index=index)
    size = min(int(len(ys) * VAL_RATIO), MAX_VAL_SIZE)
    seed = SEED + VAL_FOLD
    classes = ys.nunique()
    notes = {"multabench_val_size": size, "val_size_raised": False,
             "binary_second_split_skipped": False}
    if classes == 2:
        if ys.value_counts().min() <= 2:
            raise ValueError("Binary validation split needs at least 3 labeled rows per class")
        rng = np.random.RandomState(seed)
        picks = [rng.choice(ys[ys == v].index.tolist()) for v in sorted(set(ys))]  # 1 per class
        x_single, y_single = x.loc[picks], ys.loc[picks]
        x_rest, y_rest = x.drop(index=picks), ys.drop(index=picks)
        if size < 2:  # keep the two per-class picks as the validation part
            notes["binary_second_split_skipped"] = True
            x_train, x_val = x_rest, x_single.sample(frac=1, random_state=seed)
        else:
            x_train, x_test, _, y_test = train_test_split(
                x_rest, y_rest, test_size=size, random_state=seed, stratify=y_rest)
            x_val, _ = merge_splits(x_test, x_single, y_test, y_single, seed)
    elif ys.value_counts().min() <= 1:  # classes with a single row stay in training
        counts = ys.value_counts()
        single = ys.isin(counts[counts == 1].index)
        x_rest, y_rest = x[~single], ys[~single]
        test = max(size, len(set(y_rest)))
        notes["val_size_raised"] = test != size
        x_train, x_val, y_train, _ = train_test_split(
            x_rest, y_rest, test_size=test, random_state=seed, stratify=y_rest)
        x_train, _ = merge_splits(x_train, x[single], y_train, ys[single], seed)
    else:
        test = max(size, classes)
        notes["val_size_raised"] = test != size
        x_train, x_val, _, _ = train_test_split(x, ys, test_size=test, random_state=seed,
                                                stratify=ys)
    return x_train.to_numpy(), x_val.to_numpy(), notes


# StandardScaler (images only), then PCA, both fitted on the training rows
def pca_columns(values: np.ndarray, train: np.ndarray, scaled: bool) -> tuple[np.ndarray, dict]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fitted = values[train]
    scaler = StandardScaler().fit(fitted) if scaled else None
    transform = scaler.transform if scaled else (lambda a: a)
    components = min(PCA_COMPONENTS, len(train) - 1, values.shape[1])  # d = min(30, n_train - 1, p)
    if components < 1:
        raise ValueError("PCA needs at least two training rows")
    pca = PCA(n_components=components, random_state=SEED).fit(transform(fitted))
    out = pca.transform(transform(values)).astype(np.float32)  # (len(values), d)
    return out, {"components": components,
                 "explained_variance": float(pca.explained_variance_ratio_.sum())}


def embedding_frame(inputs: FoldInputs, encoder: str, order: np.ndarray, train: np.ndarray,
                    scaled: bool) -> tuple[pd.DataFrame, list[dict]]:
    x = inputs.embeddings[encoder]
    dim = x.shape[1] // inputs.views
    blocks, details = [], []
    for view in range(inputs.views):  # one PCA per view
        block, detail = pca_columns(x[order, view * dim:(view + 1) * dim], train, scaled)
        names = [f"__{encoder}_view{view}_pca_{i}" for i in range(block.shape[1])]
        blocks.append(pd.DataFrame(block, columns=names))
        details.append(detail)
    return pd.concat(blocks, axis=1), details


def table_frame(inputs: FoldInputs, order: np.ndarray) -> tuple[pd.DataFrame, list[int]]:
    data = {}
    for column in inputs.columns:
        if column in inputs.numeric_columns:
            data[column] = inputs.numeric[order, inputs.numeric_columns.index(column)]
        else:
            data[column] = pd.Series(
                inputs.categorical[order, inputs.categorical_columns.index(column)],
                dtype=object)
    frame = pd.DataFrame(data, columns=list(inputs.columns))
    return frame, [i for i, c in enumerate(inputs.columns) if c in inputs.categorical_columns]


def fit_predict(x_train, y_train, x_val, y_val, x_query, cat_features, threads: int,
                full_params: dict | None = None):
    import catboost

    model = catboost.CatBoostClassifier(
        iterations=ITERATIONS, early_stopping_rounds=EARLY_STOPPING_ROUNDS, od_pval=OD_PVAL,
        random_state=SEED, thread_count=threads, allow_writing_files=False)
    started = time.perf_counter()
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True,
              cat_features=cat_features, logging_level="Silent")
    proba = np.asarray(model.predict_proba(x_query), dtype=np.float64)
    resolved = model.get_all_params()
    if full_params is not None:
        full_params.update(resolved)
    return np.asarray(model.classes_), proba, {
        "fit_seconds": time.perf_counter() - started, "tree_count": int(model.tree_count_),
        "best_iteration": model.get_best_iteration(), "thread_count": threads,
        "resolved_params": {k: resolved.get(k) for k in RESOLVED}}


# all labeled rows, the early-stopped tree count and learning rate, no evaluation set
def refit_predict(x_train, y_train, x_query, cat_features, threads: int, iterations: int,
                  first: dict):
    import catboost

    if iterations < 1:
        raise ValueError("The early-stopped fit kept no tree")
    copied = {k: first[k] for k in REFIT_COPIED if k in first}
    if "learning_rate" not in copied:
        raise ValueError("The early-stopped fit has no resolved learning rate")
    model = catboost.CatBoostClassifier(iterations=iterations, random_state=SEED,
                                        thread_count=threads, allow_writing_files=False,
                                        **copied)
    started = time.perf_counter()
    model.fit(x_train, y_train, cat_features=cat_features, logging_level="Silent")
    proba = np.asarray(model.predict_proba(x_query), dtype=np.float64)
    resolved = model.get_all_params()
    if int(model.tree_count_) != iterations:
        raise ValueError("The refit kept another number of trees")
    violations = [k for k in REFIT_MUST_EQUAL
                  if k not in first or k not in resolved or first[k] != resolved[k]]
    if resolved.get("iterations") != iterations:
        violations.append("iterations")
    return np.asarray(model.classes_), proba, {
        "fit_seconds": time.perf_counter() - started, "tree_count": int(model.tree_count_),
        "thread_count": threads, "copied_params": copied,
        "refit_must_equal_violations": violations}


def run_draw(inputs: FoldInputs, entry: dict, labels: np.ndarray, base: dict, threads: int,
             refit: bool, output: Path) -> int:
    position = {key: i for i, key in enumerate(inputs.ids)}
    rows = np.array([position[key] for key in entry["row_ids"]])
    query = np.flatnonzero(~inputs.support)
    y = inputs.y
    train, val, notes = validation_split(y[rows])
    order = np.concatenate([rows[train], rows[val], query])  # train | val | query
    n_train, n_labeled = len(train), len(train) + len(val)
    table, cats = table_frame(inputs, order)
    scaled = base["modality"] == "image"  # text embeddings: PCA without scaling
    fits = 0
    for encoder in inputs.embeddings:
        pcs, details = embedding_frame(inputs, encoder, order, np.arange(n_train), scaled)
        if refit:
            pcs_all, details_all = embedding_frame(inputs, encoder, order, np.arange(n_labeled),
                                                   scaled)
        for setting in ("embedding", "joint") if table.shape[1] else ("embedding",):
            frame = pcs if setting == "embedding" else pd.concat([table, pcs], axis=1)
            features = cats if setting == "joint" else []  # native categorical columns
            first = {} if refit else None
            classes, proba, fitted = fit_predict(
                frame.iloc[:n_train], y[order[:n_train]], frame.iloc[n_train:n_labeled],
                y[order[n_train:n_labeled]], frame.iloc[n_labeled:], features, threads, first)
            meta = {**base, "setting": setting, "encoder": encoder, "method": "catboost",
                    "train_rows": n_train, "val_rows": n_labeled - n_train, **notes,
                    "pca": details, **fitted}
            stem = f"f{base['fold']}_s{base['shots']}_r{base['repeat']}_{encoder}_{setting}"
            save_prediction(output / f"{stem}_catboost.npz", classes, proba, y[query], labels,
                            meta)
            fits += 1
            if not refit:
                continue
            frame_all = pcs_all if setting == "embedding" else pd.concat([table, pcs_all], axis=1)
            classes, proba, refitted = refit_predict(
                frame_all.iloc[:n_labeled], y[order[:n_labeled]], frame_all.iloc[n_labeled:],
                features, threads, fitted["tree_count"], first)
            meta = {**base, "setting": setting, "encoder": encoder, "method": "catboost_refit",
                    "train_rows": n_labeled, "val_rows": 0, "pca": details_all, **refitted}
            save_prediction(output / f"{stem}_catboost_refit.npz", classes, proba, y[query],
                            labels, meta)
            fits += 1
    return fits


def run_fold(data: Path, task: str, fold: int, exports: Path, output: Path, threads: int,
             refit: bool, encoders: list[str]) -> int:
    record = read_rows(exports, fold)
    inputs = load_inputs(data, task, fold, encoders)
    query_ids = [inputs.ids[i] for i in np.flatnonzero(~inputs.support)]
    if record["task"] != task or record["query_row_ids"] != query_ids:
        raise ValueError("Exported rows do not match the fold")
    labels = np.array(record["labels"])
    fits = 0
    for entry in record["entries"]:
        base = {"task": task, "fold": fold, "shots": entry["shots"], "repeat": entry["repeat"],
                "n_labeled": len(entry["row_ids"]), "modality": modality(task)}
        fits += run_draw(inputs, entry, labels, base, threads, refit, output)
        print(json.dumps({"fold": fold, "shots": entry["shots"], "repeat": entry["repeat"]}),
              flush=True)
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost with MulTaBench's defaults")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--encoders", nargs="+")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--refit", action="store_true",
                        help="also refit on all labeled rows with the early-stopped tree count")
    args = parser.parse_args()
    encoders = args.encoders or [e for e, v in ENCODERS.items() if v[0] == modality(args.task)]
    for fold in args.folds:
        run_fold(args.data, args.task, fold, args.exports, args.output, args.threads,
                 args.refit, encoders)
    (args.output / "COMPLETE").write_text("CatBoost predictions saved\n")


if __name__ == "__main__":
    main()
