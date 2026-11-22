import ast
import hashlib
import importlib.metadata
import inspect
import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..features.bank import sha256_file
from ..features.images import image_record, write_images
from ..tasks import FOLDS, MULTABENCH_IMAGE, MULTABENCH_TEXT, RELEASED_IMAGE, RELEASED_TEXT
from .classes import MAX_CLASSES, restrict_classes, scalar
from .sources import MULTABENCH_REVISION, import_from, verify_clone

TRAIN_EXAMPLES = 10000  # at most this many training rows per fold
# package versions of the MulTaBench lock
VERSIONS = {"tabstar": "1.1.15", "numpy": "2.3.5", "pandas": "2.3.3", "scikit-learn": "1.6.1",
            "kagglehub": "1.0.0"}
SPLIT_RULE = "official_subsample_dataset_then_tabstar_split_to_test"
TEXT_STATE = "text_in_subdirectory"
SEPARATOR = "; "  # between the "name: value" fields of a row's text
ALL_CLASSES = "all_classes"
# more than ten classes; ten are drawn per fold unless all_classes
MANY_CLASS_TASKS = ("MUL_IMAGE_HEARTHSTONE_CLASS", "MUL_TEXT_SPOTIFY_GENRES",
                    "MUL_TEXT_WINE_REVIEW")
RECIPES_WITHOUT_TARGET = ("MUL_TEXT_BOX_OFFICE",)  # text recipes that name no target


def official_runtime(source: Path) -> dict:
    source = verify_clone(source, MULTABENCH_REVISION)
    import_from(source)
    from multabench.benchmark.load import load_multabench_dataset
    from multabench.datasets.all_multabench_datasets import (
        MULTABENCH_CORE_IMAGE,
        MULTABENCH_CORE_TEXT,
        MULTABENCH_FULL_IMAGE_EXTRA,
        MULTABENCH_FULL_TEXT_EXTRA,
    )
    from tabstar.preprocessing import splits

    def classification(listed) -> tuple:
        return tuple(d.name for d in listed if not d.name.startswith("REG_"))

    if (classification(MULTABENCH_CORE_IMAGE) != MULTABENCH_IMAGE
            or not set(RELEASED_IMAGE) <= {d.name for d in MULTABENCH_FULL_IMAGE_EXTRA}
            or classification(MULTABENCH_CORE_TEXT) != MULTABENCH_TEXT
            or classification(MULTABENCH_FULL_TEXT_EXTRA) != RELEASED_TEXT):
        raise ValueError("MulTaBench task inventory differs")
    if not Path(inspect.getfile(load_multabench_dataset)).resolve().is_relative_to(source):
        raise ValueError("Imported another MulTaBench installation")
    versions = {p: importlib.metadata.version(p) for p in VERSIONS}
    if versions != VERSIONS:
        raise ValueError(f"Preparation versions differ from the MulTaBench lock: {versions}")
    return {"versions": versions, "tabstar_splits_sha256": sha256_file(inspect.getfile(splits)),
            "loader_sha256": sha256_file(inspect.getfile(load_multabench_dataset))}


def load(task: str, text: bool):
    from multabench.benchmark.load import load_multabench_dataset
    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    from multabench.datasets.multimodal import MultimodalState

    state = MultimodalState.ALL if text else MultimodalState.NO_TEXT  # image tasks: no text
    return load_multabench_dataset(MulTaBenchDatasetID[task], state)


def recipe_target(source: Path, task: str) -> str | None:
    recipe = Path(source) / "multabench/benchmark/datasets" / f"{task}.py"
    tree = ast.parse(recipe.read_text(encoding="utf-8"))
    if task.startswith(("BIN_IMAGE", "MUL_IMAGE")):
        values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == "TARGET_COL"]
        return values[0] if len(values) == 1 else None
    # text recipes: "Target Variable: <name> (" in the recipe's docstring
    names = re.findall(r"^\s*Target Variable: (.+?) \(", ast.get_docstring(tree) or "", re.M)
    if task in RECIPES_WITHOUT_TARGET:
        if names:
            raise ValueError(f"{task} recipe now names a target")
        return None
    if len(names) != 1:
        raise ValueError(f"{task} recipe names no single target")
    return names[0]


def folds(task: str, x: pd.DataFrame, y: pd.Series, all_classes: bool = False):
    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from tabstar.preprocessing.splits import split_to_test

    if not x.index.is_unique or not x.index.equals(y.index) or y.isna().any():
        raise ValueError("Invalid source indices or targets")
    # official subsample, then TabSTAR's test split, first five folds
    for fold in FOLDS:
        sx, sy = subsample_dataset(x=x, y=y, is_cls=True, train_examples=TRAIN_EXAMPLES,
                                   fold=fold)
        xs, xq, ys, yq = split_to_test(x=sx, y=sy, is_cls=True, fold=fold,
                                       train_examples=TRAIN_EXAMPLES)
        if not len(xs) or not len(xq) or set(xs.index) & set(xq.index):
            raise ValueError("Invalid official partition")
        if all_classes:
            drawn = None
        else:
            xs, ys, xq, yq, drawn = restrict_classes(task, fold, xs, ys, xq, yq)  # <= 10 classes
        yield fold, xs, xq, ys, yq, drawn, len(sx)


def encoded_labels(ys: pd.Series, yq: pd.Series, all_classes: bool = False) -> LabelEncoder:
    encoder = LabelEncoder().fit(ys)
    if len(encoder.classes_) < 2 or (not all_classes and len(encoder.classes_) > MAX_CLASSES):
        raise ValueError("Unexpected class count")
    encoder.transform(yq)  # query classes must occur in the labeled pool
    return encoder


def check_all_classes(task: str, all_classes: bool) -> None:
    if all_classes and task not in MANY_CLASS_TASKS:
        raise ValueError(f"Every class is kept only for {MANY_CLASS_TASKS}")


def class_fields(splits: dict, manifest: dict, fold: int, encoder: LabelEncoder,
                 yq: pd.Series) -> None:
    splits[str(fold)]["classes"] = int(len(encoder.classes_))
    splits[str(fold)]["support_classes_missing_from_query"] = int(
        len(encoder.classes_) - yq.nunique())
    manifest["class_rule"] = ALL_CLASSES


def prepare_image(task: str, source: Path, data: Path, all_classes: bool = False) -> dict:
    check_all_classes(task, all_classes)
    provenance = official_runtime(source)
    dataset = load(task, text=False)
    if not dataset.is_cls:
        raise ValueError(f"{task} is not a classification task")
    output = Path(data) / "tasks" / task
    if output.exists():
        raise FileExistsError(output)
    x, y = dataset.x, dataset.y
    root = Path(dataset.image_folder).resolve(strict=True)
    meta = json.loads((root / "metadata.json").read_text())
    image_col, target = meta["image_col"], meta["target"]
    stated = recipe_target(source, task)
    if target in x or y.name != target or image_col not in x or y.name != stated:
        raise ValueError("Loaded columns or target differ from the official metadata")
    table_cols = [c for c in x if c != image_col]
    if not table_cols:
        raise ValueError("No table columns")
    cats = [c for c in table_cols if not pd.api.types.is_numeric_dtype(x[c])]
    images, manifests, splits = {}, {}, {}
    for fold, xs, xq, ys, yq, drawn, sampled in folds(task, x, y, all_classes):
        encoder = encoded_labels(ys, yq, all_classes)
        rows = []
        for split, frame, labels in (("support", xs, ys), ("query", xq, yq)):
            if not frame.index.equals(labels.index):
                raise ValueError("Misaligned official split")
            for (index, row), label in zip(frame.iterrows(), encoder.transform(labels),
                                           strict=True):
                relative = Path(str(row[image_col]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Image path must lie inside the dataset root")
                path = (root / relative).resolve(strict=True)
                image_id = hashlib.sha256(f"{task}/{relative.as_posix()}".encode()).hexdigest()
                if image_id not in images:
                    images[image_id] = image_record(image_id, path, root)
                source_row = scalar(index)
                rows.append({"row_id": f"{task}/csv/{source_row}", "source_row": source_row,
                             "split": split, "target": int(label),
                             "raw_target": scalar(labels.loc[index]),
                             "table": {c: scalar(row[c]) for c in table_cols},
                             "image_ids": [image_id]})
        if len({r["row_id"] for r in rows}) != len(rows):
            raise ValueError("Row identity collision")
        support_ids = {r["image_ids"][0] for r in rows if r["split"] == "support"}
        query_ids = {r["image_ids"][0] for r in rows if r["split"] == "query"}
        splits[str(fold)] = {
            "support_rows": len(xs), "query_rows": len(xq), "subsampled_rows": sampled,
            "shared_image_ids": sorted(support_ids & query_ids),
            "shared_image_hashes": sorted({images[k]["sha256"] for k in support_ids}
                                          & {images[k]["sha256"] for k in query_ids})}
        manifest = {
            "task": task, "seed": fold, "target_name": target,
            "classes": [scalar(v) for v in encoder.classes_], "categorical_columns": cats,
            "table_columns": table_cols, "multimodal_state": "no_text",
            "source_revision": MULTABENCH_REVISION, "train_examples": TRAIN_EXAMPLES,
            "split_rule": SPLIT_RULE, "image_overlap_policy": "preserve_source_split",
            "image_selection_policy": "official_curated_single_image_column", "rows": rows}
        if drawn is not None:
            splits[str(fold)]["class_draw"] = drawn
            manifest["class_draw"] = drawn
        if all_classes:
            class_fields(splits, manifest, fold, encoder, yq)
        manifests[f"seed{fold}.json"] = manifest
    output.mkdir(parents=True)
    for name, value in manifests.items():
        (output / name).write_text(json.dumps(value, allow_nan=False))
    write_images(output, list(images.values()))
    audit = {"task": task, "source_revision": MULTABENCH_REVISION, "folds": list(FOLDS),
             "image_root": str(root), "source_rows": len(x), "target_name": target,
             "table_columns": table_cols, "categorical_columns": cats,
             "source_sha256": {n: sha256_file(root / n) for n in ("data.csv", "metadata.json")},
             "kaggle_version": root.name, "provenance": provenance, "splits": splits,
             "files_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir())}}
    if all_classes:
        audit["class_rule"] = ALL_CLASSES
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    return audit


def valid_text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def field_order(columns: dict[str, list]) -> tuple[list[str], dict[str, float]]:
    medians = {}
    for name, values in columns.items():
        lengths = [len(t) for t in map(valid_text, values) if t is not None]
        if not lengths:
            raise ValueError(f"Text column {name!r} has no values")
        medians[name] = float(statistics.median(lengths))
    return sorted(medians, key=lambda n: (medians[n], n)), medians  # shortest median first


# "name: value" for every nonmissing field, joined by "; "
def joined(row: dict, order: list[str]) -> str:
    return SEPARATOR.join(f"{n}: {t}" for n in order if (t := valid_text(row[n])) is not None)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_text(task: str, source: Path, data: Path, all_classes: bool = False) -> dict:
    check_all_classes(task, all_classes)
    provenance = official_runtime(source)
    from multabench.preprocessing.feat_types import detect_text_features

    dataset = load(task, text=True)
    if not dataset.is_cls:
        raise ValueError(f"{task} is not a classification task")
    output = Path(data) / "tasks" / task
    if output.exists():
        raise FileExistsError(output)
    x, y = dataset.x, dataset.y
    root = Path(dataset.image_folder).resolve(strict=True)
    meta = json.loads((root / "metadata.json").read_text())
    target = meta["target"]
    stated = recipe_target(source, task)
    if (target in x or y.name != target or meta.get("image_col") is not None
            or (stated is not None and y.name != stated)):
        raise ValueError("Loaded columns or target differ from the official metadata")
    text_cols = detect_text_features(x)  # official detection on the whole table
    table_cols = [c for c in x if c not in text_cols]
    if not text_cols or not table_cols:
        raise ValueError("Need both text and table columns")
    cats = [c for c in table_cols if not pd.api.types.is_numeric_dtype(x[c])]
    order, medians = field_order({c: [scalar(v) for v in x[c]] for c in text_cols})
    texts, manifests, splits = {}, {}, {}
    for fold, xs, xq, ys, yq, drawn, sampled in folds(task, x, y, all_classes):
        encoder = encoded_labels(ys, yq, all_classes)
        rows, row_text = [], {}
        for split, frame, labels in (("support", xs, ys), ("query", xq, yq)):
            if not frame.index.equals(labels.index):
                raise ValueError("Misaligned official split")
            for (index, row), label in zip(frame.iterrows(), encoder.transform(labels),
                                           strict=True):
                source_row = scalar(index)
                row_id = f"{task}/csv/{source_row}"
                text = joined({c: scalar(row[c]) for c in text_cols}, order)  # one text per row
                if texts.setdefault(row_id, text) != text:
                    raise ValueError("One source row with two texts")
                row_text[row_id] = text
                rows.append({"row_id": row_id, "source_row": source_row, "split": split,
                             "target": int(label), "raw_target": scalar(labels.loc[index]),
                             "table": {c: scalar(row[c]) for c in table_cols},
                             "image_ids": []})
        if len(row_text) != len(rows):
            raise ValueError("Row identity collision")
        support_texts = {text_sha256(row_text[r["row_id"]]) for r in rows
                         if r["split"] == "support" and row_text[r["row_id"]]}
        splits[str(fold)] = {
            "support_rows": len(xs), "query_rows": len(xq), "subsampled_rows": sampled,
            "query_texts_identical_to_a_support_text": sum(
                bool(row_text[r["row_id"]]) and text_sha256(row_text[r["row_id"]]) in support_texts
                for r in rows if r["split"] == "query"),
            "empty_texts": sum(not row_text[r["row_id"]] for r in rows)}
        manifest = {
            "task": task, "seed": fold, "target_name": target,
            "classes": [scalar(v) for v in encoder.classes_], "categorical_columns": cats,
            "table_columns": table_cols, "text_columns": order, "multimodal_state": TEXT_STATE,
            "source_revision": MULTABENCH_REVISION, "train_examples": TRAIN_EXAMPLES,
            "split_rule": SPLIT_RULE, "text_overlap_policy": "preserve_source_split",
            "rows": rows}
        if drawn is not None:
            splits[str(fold)]["class_draw"] = drawn
            manifest["class_draw"] = drawn
        if all_classes:
            class_fields(splits, manifest, fold, encoder, yq)
        manifests[f"seed{fold}.json"] = manifest
    output.mkdir(parents=True)
    for name, value in manifests.items():
        (output / name).write_text(json.dumps(value, allow_nan=False))
    (output / "texts.jsonl").write_text(
        "".join(json.dumps({"row_id": k, "text": t}) + "\n" for k, t in texts.items()),
        encoding="utf-8")
    audit = {"task": task, "source_revision": MULTABENCH_REVISION, "folds": list(FOLDS),
             "source_rows": len(x), "target_name": target, "text_columns": order,
             "median_characters_per_field": medians, "table_columns": table_cols,
             "categorical_columns": cats, "empty_texts": sum(not t for t in texts.values()),
             "source_sha256": {n: sha256_file(root / n) for n in ("data.csv", "metadata.json")},
             "kaggle_version": root.name, "provenance": provenance, "splits": splits,
             "files_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir())}}
    if all_classes:
        audit["class_rule"] = ALL_CLASSES
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    return audit
