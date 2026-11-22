import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from itertools import permutations
from pathlib import Path

import pandas as pd

from ..features.bank import sha256_file
from ..features.images import image_record, write_images
from ..tasks import MMPFN_IMAGE
from .sources import MMPFN_REVISION, verify_clone

# version-pinned Kaggle copies of the sources
KAGGLE = {
    "pad": ("dataset", "mahdavi1202/skin-cancer/versions/1"),
    "cbis": ("dataset", "awsaf49/cbis-ddsm-breast-cancer-image-dataset/versions/1"),
    "petfinder": ("competition", "petfinder-adoption-prediction"),
}
# PAD-UFES-20 metadata of the official Mendeley release
PAD_METADATA_URL = ("https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/"
                    "fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded")
PAD_METADATA_SHA256 = "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"
PAD_METADATA = "metadata.mendeley_v1.csv"

PAD_CAT = ["smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
           "has_piped_water", "has_sewage_system", "itch", "grew", "hurt", "bleed", "elevation",
           "biopsed", "changed", "background_father", "background_mother", "gender", "region"]
PAD_NUM = ["age", "diameter_1", "diameter_2"]
PET_CAT = ["Breed1", "Breed2", "Color1", "Color2", "Color3", "Dewormed", "FurLength", "Gender",
           "Health", "MaturitySize", "State", "Sterilized", "Type", "Vaccinated"]
PET_NUM = ["Age", "VideoAmt", "Quantity", "PhotoAmt", "Fee"]
CBIS_VIEWS = ["image file path", "cropped image file path", "ROI mask file path"]
# view -> DICOM SeriesDescription
CBIS_ROLES = {"image file path": "full mammogram images",
              "cropped image file path": "cropped images",
              "ROI mask file path": "ROI mask images"}
CBIS_POLICIES = ("dicom_metadata", "author_first")  # by DICOM role, or the authors' first file
# columns whose spelling may differ between the Mendeley and Kaggle metadata
PAD_BOOLEAN = {"smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
               "has_piped_water", "has_sewage_system", "itch", "grew", "hurt", "changed", "bleed",
               "elevation", "biopsed"}
PAD_DECIMAL = {"fitspatrick", "diameter_1", "diameter_2"}


def download(dataset: str, output: Path) -> Path:
    import kagglehub

    kind, handle = KAGGLE[dataset]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    fetch = kagglehub.dataset_download if kind == "dataset" else kagglehub.competition_download
    resolved = Path(fetch(handle, output_dir=str(output))).resolve()
    if resolved != output:
        raise ValueError(f"Downloader returned {resolved}")
    return output


# equal up to boolean case ("True" / "TRUE") and decimal notation, else an error
def compare_pad_metadata(reference: Path, mirror: Path) -> dict:
    with reference.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns, original = reader.fieldnames, list(reader)
    with mirror.open(newline="") as handle:
        reader = csv.DictReader(handle)
        mirror_columns, downloaded = reader.fieldnames, list(reader)
    if not original or columns != mirror_columns or len(original) != len(downloaded):
        raise ValueError("PAD metadata schema or row count differs")
    differences = Counter()
    for index, (left, right) in enumerate(zip(original, downloaded, strict=True)):
        for column in columns:
            a, b = left[column], right[column]
            if a == b:
                continue
            equal = False
            if column in PAD_BOOLEAN and a in {"True", "False"}:
                equal = b == a.upper()
            elif column in PAD_DECIMAL and a and b:
                try:
                    equal = Decimal(a).is_finite() and Decimal(a) == Decimal(b)
                except InvalidOperation:
                    pass
            if not equal:
                raise ValueError(f"PAD metadata differs at row {index}, {column}")
            differences[column] += 1
    return {"rows": len(original), "reference_sha256": sha256_file(reference),
            "mirror_sha256": sha256_file(mirror), "differences": dict(differences)}


def fetch_pad_metadata(root: Path) -> Path:
    import urllib.request

    target = Path(root) / PAD_METADATA
    if not target.exists():
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(PAD_METADATA_URL, timeout=120) as response:
            partial.write_bytes(response.read())
        partial.rename(target)
    if sha256_file(target) != PAD_METADATA_SHA256:
        raise ValueError(f"Unexpected PAD metadata {target}")
    compare_pad_metadata(target, Path(root) / "metadata.csv")  # against the Kaggle mirror
    return target


def _table(row: pd.Series, columns: list[str]) -> dict:
    out = {}
    for column in columns:
        value = row[column]
        if pd.isna(value):
            value = None
        elif hasattr(value, "item"):
            value = value.item()
        out[column] = value
    return out


def _read(root: Path, relative: str, required: list[str]) -> tuple[Path, pd.DataFrame]:
    path = root / relative
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return path, frame


def dicom_index(root: Path) -> dict:
    index = defaultdict(list)
    with (root / "csv/dicom_info.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"SeriesInstanceUID", "SeriesDescription", "image_path"} <= set(
                reader.fieldnames or []):
            raise ValueError("dicom_info.csv lacks the mapping fields")
        for row in reader:
            index[row["SeriesInstanceUID"]].append(row)  # series -> rows of dicom_info.csv
    return dict(index)


# files of the three views from the DICOM roles: the single consistent assignment, else an error
def dicom_views(root: Path, series_by_view: dict, index: dict) -> tuple[list[Path], list[dict]]:
    grouped = defaultdict(list)
    for view, series in series_by_view.items():
        grouped[series].append(view)
    resolved, evidence = {}, []
    for series, views in grouped.items():
        records = index.get(series, [])
        files = []
        for record in records:
            relative = Path(record["image_path"])
            if (len(relative.parts) != 4 or relative.parts[:2] != ("CBIS-DDSM", "jpeg")
                    or relative.parts[2] != series):
                raise ValueError(f"Unexpected CBIS image_path: {relative}")
            path = (root / Path(*relative.parts[1:])).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Invalid CBIS image path")
            files.append(path)
        actual = {p.resolve() for p in (root / "jpeg" / series).iterdir()}
        if not files or len(set(files)) != len(files) or set(files) != actual:
            raise ValueError(f"DICOM mapping does not cover the series exactly: {series}")
        if len(files) == 1:
            for view in views:
                resolved[view] = files[0]
            assignment, rule = [0] * len(views), "single_file_series_from_case_csv"
        else:
            if len(files) != len(views):
                raise ValueError(f"Ambiguous series/view cardinality: {series}")
            # file-to-view bijections consistent with the roles; a blank role fits any view
            candidates = [a for a in permutations(range(len(files)))
                          if all(records[i]["SeriesDescription"] in {"", CBIS_ROLES[v]}
                                 for v, i in zip(views, a, strict=True))]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous DICOM role assignment: {series}")
            assignment = candidates[0]
            rule = "unique_dicom_role_bijection_blank_role_by_elimination"
            resolved.update({v: files[i] for v, i in zip(views, assignment, strict=True)})
        for view, i in zip(views, assignment, strict=True):
            evidence.append({"view": view, "series": series,
                             "image_path": str(files[i].relative_to(root)),
                             "dicom_role": records[i]["SeriesDescription"], "rule": rule,
                             "role_conflict": records[i]["SeriesDescription"]
                             not in {"", CBIS_ROLES[view]}})
    return [resolved[v] for v in series_by_view], evidence


def _pad_rows(root: Path) -> tuple[list[dict], dict]:
    csv_path, frame = _read(root, PAD_METADATA, PAD_CAT + PAD_NUM + ["diagnostic", "img_id"])
    if frame["img_id"].isna().any() or frame["img_id"].duplicated().any():
        raise ValueError("PAD needs unique image identities")
    rows = []
    for index, row in frame.iterrows():
        name = str(row["img_id"])
        candidates = [root / "imgs" / name] + [
            root / f"imgs_part_{p}" / f"imgs_part_{p}" / name for p in (1, 2, 3)]
        matches = [p for p in candidates if p.is_file()]
        if Path(name).name != name or len(matches) != 1:
            raise ValueError(f"PAD image needs one exact match: {name}")
        rows.append({"row_id": f"pad/{name}", "source_csv": csv_path.name,
                     "source_row": int(index), "table": _table(row, PAD_CAT + PAD_NUM),
                     "target": row["diagnostic"], "image_paths": matches,
                     "groups": _table(row, [c for c in ("patient_id", "lesion_id") if c in frame])})
    return rows, {"csvs": [csv_path], "categorical_columns": PAD_CAT, "numeric_columns": PAD_NUM,
                  "target": "diagnostic", "excluded": [], "image_mapping": [],
                  "image_selection_policy": "exact_filename"}


def _petfinder_rows(root: Path) -> tuple[list[dict], dict]:
    csv_path, frame = _read(root, "train/train.csv", PET_CAT + PET_NUM + ["AdoptionSpeed", "PetID"])
    if frame["PetID"].isna().any() or frame["PetID"].duplicated().any():
        raise ValueError("PetFinder needs unique PetID")
    rows, excluded = [], []
    for index, row in frame.iterrows():
        identity = str(row["PetID"])
        path = root / "train_images" / f"{identity}-1.jpg"
        if Path(identity).name != identity:
            raise ValueError("Invalid PetID")
        if not path.is_file():  # rows without a first photo, as in the authors' loader
            excluded.append({"source_row": int(index), "row_id": identity,
                             "reason": "missing_first_image_author_filter"})
            continue
        rows.append({"row_id": f"petfinder/{identity}", "source_csv": "train/train.csv",
                     "source_row": int(index), "table": _table(row, PET_CAT + PET_NUM),
                     "target": int(row["AdoptionSpeed"]), "image_paths": [path], "groups": {}})
    return rows, {"csvs": [csv_path], "categorical_columns": PET_CAT, "numeric_columns": PET_NUM,
                  "target": "AdoptionSpeed", "excluded": excluded, "image_mapping": [],
                  "image_selection_policy": "exact_filename"}


def _cbis_rows(task: str, root: Path, policy: str) -> tuple[list[dict], dict]:
    kind = task.removeprefix("cbis_")
    index = dicom_index(root) if policy == "dicom_metadata" else None
    categorical = ["left or right breast", "image view", "abnormality id"] + (
        ["mass shape", "mass margins"] if kind == "mass" else ["calc type", "calc distribution"])
    numeric = ["breast_density" if kind == "mass" else "breast density", "assessment", "subtlety"]
    rows, excluded, mapping = [], [], []
    csvs = [root / "csv/dicom_info.csv"] if index is not None else []
    for split, suffix in (("support", "train"), ("query", "test")):
        csv_path, frame = _read(root, f"csv/{kind}_case_description_{suffix}_set.csv",
                                categorical + numeric + CBIS_VIEWS + ["pathology", "patient_id"])
        csvs.append(csv_path)
        for i, row in frame.iterrows():
            paths, missing, series_by_view, selections = [], [], {}, []
            for view in CBIS_VIEWS:
                value = row[view]
                if pd.isna(value) or len(str(value).strip().split("/")) < 2:
                    raise ValueError(f"Invalid CBIS path at {csv_path.name}:{i}:{view}")
                series = str(value).split("/")[-2]
                if series in {"", ".", ".."}:
                    raise ValueError("Invalid CBIS series directory")
                directory = root / "jpeg" / series
                series_by_view[view] = series
                if not directory.is_dir():
                    missing.append(view)
                    continue
                if index is None:  # the authors' loader: first unsorted directory entry
                    entries = os.listdir(directory)
                    paths.append(directory / entries[0])
                    selections.append({"view": view, "series": series,
                                       "directory_entries": entries, "selected_file": entries[0],
                                       "rule": "author_os_listdir_first_unsorted"})
            if missing:  # rows with a missing view directory, as in the authors' loader
                excluded.append({"source_csv": csv_path.name, "source_row": int(i),
                                 "reason": "missing_view_directory_author_filter",
                                 "views": missing})
                continue
            if index is not None:
                paths, selections = dicom_views(root, series_by_view, index)
            mapping.append({"source_csv": csv_path.name, "source_row": int(i),
                            "views": selections})
            label = row["pathology"]
            label = "BENIGN" if label == "BENIGN_WITHOUT_CALLBACK" else label  # binary target
            if label not in {"BENIGN", "MALIGNANT"}:
                raise ValueError(f"Unexpected CBIS pathology: {label!r}")
            rows.append({"row_id": f"{task}/{suffix}/{i}", "source_csv": f"csv/{csv_path.name}",
                         "source_row": int(i), "split": split,
                         "table": _table(row, categorical + numeric), "target": label,
                         "image_paths": paths, "groups": _table(row, ["patient_id"])})
    return rows, {"csvs": csvs, "categorical_columns": categorical, "numeric_columns": numeric,
                  "target": "pathology", "excluded": excluded, "image_mapping": mapping,
                  "image_selection_policy": policy}


# raw rows in source order, with the authors' row filters
def load_rows(task: str, root: Path,
              cbis_policy: str = "dicom_metadata") -> tuple[list[dict], dict]:
    root = Path(root).resolve(strict=True)
    if task == "pad_ufes_20":
        rows, evidence = _pad_rows(root)
    elif task == "petfinder":
        rows, evidence = _petfinder_rows(root)
    elif task in ("cbis_mass", "cbis_calc"):
        if cbis_policy not in CBIS_POLICIES:
            raise ValueError(f"Unknown CBIS image policy {cbis_policy}")
        rows, evidence = _cbis_rows(task, root, cbis_policy)
    else:
        raise ValueError(f"Unknown task {task}")
    if not rows or any(pd.isna(r["target"]) for r in rows):
        raise ValueError("Empty task or missing targets")
    return rows, evidence


def prepare(task: str, root: Path, data: Path, cbis_policy: str = "dicom_metadata",
            source: Path | None = None) -> dict:
    import torch
    from torch.utils.data import random_split

    if task not in MMPFN_IMAGE:
        raise ValueError(f"Unknown task {task}")
    if source is not None:
        verify_clone(source, MMPFN_REVISION)
    output = Path(data) / "tasks" / task
    if output.exists():
        raise FileExistsError(output)
    root = Path(root).resolve(strict=True)
    rows, evidence = load_rows(task, root, cbis_policy)
    labels = sorted({r["target"] for r in rows})
    label_ids = {label: i for i, label in enumerate(labels)}
    images, source_rows = {}, []
    family = "cbis" if task.startswith("cbis_") else task
    for row in rows:
        image_ids = []
        for path in row["image_paths"]:
            path = path.resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(f"Image outside the dataset root: {path}")
            relative = path.relative_to(root).as_posix()
            key = hashlib.sha256(f"{family}/{relative}".encode()).hexdigest()  # image id
            if key not in images:
                images[key] = image_record(key, path, root)
            image_ids.append(key)
        source_rows.append({k: v for k, v in row.items() if k != "image_paths"}
                           | {"target": label_ids[row["target"]], "image_ids": image_ids})
    if len({r["row_id"] for r in source_rows}) != len(source_rows):
        raise ValueError("Duplicate row identities")
    manifests, splits = {}, {}
    for seed in range(5):
        if task.startswith("cbis_"):
            ordered, rule = source_rows, "original_train_test_csv"  # same split in every fold
        else:
            n = len(source_rows)
            # 80/20 split, torch generator seeded by the fold
            support, query = random_split(range(n), [int(n * 0.8), n - int(n * 0.8)],
                                          generator=torch.Generator().manual_seed(seed))
            ordered = [source_rows[i] | {"split": split}
                       for split, part in (("support", support.indices), ("query", query.indices))
                       for i in part]
            rule = "torch_random_split_80_20_floor_keep_randperm_order"
        support_rows = [r for r in ordered if r["split"] == "support"]
        query_rows = [r for r in ordered if r["split"] == "query"]
        if not support_rows or not query_rows or not (
                {r["target"] for r in query_rows} <= {r["target"] for r in support_rows}):
            raise ValueError("Invalid source split")
        s_img = {k for r in support_rows for k in r["image_ids"]}
        q_img = {k for r in query_rows for k in r["image_ids"]}
        splits[str(seed)] = {
            "support_rows": len(support_rows), "query_rows": len(query_rows),
            "shared_image_ids": sorted(s_img & q_img),
            "shared_image_hashes": sorted({images[k]["sha256"] for k in s_img}
                                          & {images[k]["sha256"] for k in q_img})}
        manifests[f"seed{seed}.json"] = {
            "task": task, "seed": seed, "target_name": evidence["target"], "classes": labels,
            "categorical_columns": evidence["categorical_columns"], "split_rule": rule,
            "source_revision": MMPFN_REVISION, "image_overlap_policy": "preserve_source_split",
            "image_selection_policy": evidence["image_selection_policy"],
            "rows": [{k: r[k] for k in ("row_id", "split", "table", "target", "image_ids")}
                     for r in ordered]}
    output.mkdir(parents=True)
    for name, value in manifests.items():
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False))
    write_images(output, list(images.values()))
    audit = {"task": task, "root": str(root), "image_root": str(root),
             "source_revision": MMPFN_REVISION, "source_rows": source_rows,
             "manifest_sha256": {name: sha256_file(output / name) for name in manifests},
             "target": evidence["target"], "categorical_columns": evidence["categorical_columns"],
             "numeric_columns": evidence["numeric_columns"],
             "image_selection_policy": evidence["image_selection_policy"],
             "csv_sha256": {str(p.relative_to(root)): sha256_file(p) for p in evidence["csvs"]},
             "excluded": evidence["excluded"], "image_mapping": evidence["image_mapping"],
             "torch_version": str(torch.__version__), "splits": splits,
             "files_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir())}}
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    return audit
