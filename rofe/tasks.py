import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .features.bank import FeatureBank

FOLDS = tuple(range(5))

### Task sets
# MulTaBench core image classification
MULTABENCH_IMAGE = (
    "BIN_IMAGE_CELEB_ATTRACTIVENESS", "BIN_IMAGE_HATEFUL_MEME", "BIN_IMAGE_MAMMOGRAPHY_CMMD",
    "MUL_IMAGE_CHEXPERT", "MUL_IMAGE_CBIS_DDSM", "MUL_IMAGE_GLAUCOMA_SMDG",
    "MUL_IMAGE_CSGO_SKIN_PRICE", "MUL_IMAGE_FLOWER_BOUQUETS", "MUL_IMAGE_HUBMAP_HPA",
    "MUL_IMAGE_JUSTIN_INSTAGRAM", "MUL_IMAGE_PETFINDER", "MUL_IMAGE_ZOOSCAN_ZOOPLANKTON",
)
# released with MulTaBench, outside its core set
RELEASED_IMAGE = (
    "MUL_IMAGE_MINECRAFT", "BIN_IMAGE_OASIS_ALZHEIMERS", "MUL_IMAGE_REDDIT_MEMES",
    "MUL_IMAGE_POKEMON_HEIGHT", "MUL_IMAGE_HAM10000_LESION", "BIN_IMAGE_PINTEREST_POPULAR",
    "MUL_IMAGE_HEARTHSTONE_CLASS", "MUL_IMAGE_PAD_UFES_LESION",
)
MMPFN_IMAGE = ("petfinder", "pad_ufes_20", "cbis_mass", "cbis_calc")  # MultiModalPFN's tasks
IMAGE_TASKS = (*MULTABENCH_IMAGE, *RELEASED_IMAGE, *MMPFN_IMAGE)

MULTABENCH_TEXT = (
    "BIN_TEXT_FAKE_JOB_POSTING", "BIN_TEXT_JIGSAW_TOXICITY", "BIN_TEXT_KICKSTARTER_FUNDING",
    "MUL_TEXT_DATA_SCIENTIST_SALARY", "MUL_TEXT_MICHELIN_RESTAURANTS",
    "MUL_TEXT_PRODUCT_SENTIMENT", "MUL_TEXT_SPOTIFY_GENRES", "MUL_TEXT_US_ACCIDENTS",
    "MUL_TEXT_WINE_REVIEW", "MUL_TEXT_WOMEN_CLOTHING_REVIEW",
)
RELEASED_TEXT = (
    "MUL_TEXT_KOREAN_DRAMA", "MUL_TEXT_CONSUMER_COMPLAINT", "MUL_TEXT_BOOKS_GOODREADS",
    "BIN_TEXT_OSHA_INJURY", "BIN_TEXT_CALIFORNIA_PRICES", "MUL_TEXT_AMERICAN_EAGLE_PRICES",
    "MUL_TEXT_NEWS_CHANNEL", "BIN_TEXT_IMDB_GENRE", "MUL_TEXT_MELBOURNE_AIRBNB",
    "MUL_TEXT_BOX_OFFICE",
)
TEXT_TASKS = (*MULTABENCH_TEXT, *RELEASED_TEXT)

# CoMET's own benchmarks
COMET_IMAGENET = ("MUL_IMAGE_IMAGENET_TERRIER", "MUL_IMAGE_IMAGENET_SNAKES",
                  "MUL_IMAGE_IMAGENET_BEETLE", "MUL_IMAGE_IMAGENET_FELINE")
COMET_TEXT = ("TEXT_IMDB", "TEXT_AG_NEWS", "TEXT_20_NEWSGROUPS", "TEXT_YELP_FULL")
# multimodal task -> its embedding blocks (table columns are not a block)
COMET_MM_BLOCKS = {
    "CMM_JIGSAW": ("text",), "CMM_WINE": ("text",), "CMM_AIRBNB": ("text",),
    "CMM_PETFINDER": ("image", "text"), "CMM_WIKIART": ("image",),
    "CMM_MMIMDB": ("image", "text"), "CMM_SALARY": ("text",), "CMM_NEWS_CHANNEL": ("text",),
}
COMET_MM = tuple(COMET_MM_BLOCKS)
TASK_SETS = {"image": IMAGE_TASKS, "text": TEXT_TASKS, "comet_imagenet": COMET_IMAGENET,
             "comet_text": COMET_TEXT, "comet_mm": COMET_MM}

# encoder -> (modality, seed stream, dimension); the stream enters the rotation seeds
ENCODERS = {
    "dinov3": ("image", 0, 768),
    "siglip": ("image", 1, 1152),
    "electra": ("text", 0, 768),
    "e5": ("text", 1, 768),
}


def modality(task: str) -> str:
    if task in IMAGE_TASKS or task in COMET_IMAGENET:
        return "image"
    if task in TEXT_TASKS or task in COMET_TEXT:
        return "text"
    if task in COMET_MM_BLOCKS:
        return "multimodal"
    raise ValueError(f"Unknown task {task}")


def blocks(task: str) -> tuple[str, ...]:
    kind = modality(task)
    return COMET_MM_BLOCKS[task] if kind == "multimodal" else (kind,)


def resolve_tasks(spec: str | list[str]) -> list[str]:
    if isinstance(spec, str):
        return list(TASK_SETS[spec])
    for task in spec:
        modality(task)
    return list(spec)


### Prepared folds
def manifest_path(data: Path, task: str, fold: int) -> Path:
    return Path(data) / "tasks" / task / f"seed{fold}.json"


def bank_path(data: Path, encoder: str, task: str) -> Path:
    return Path(data) / "features" / encoder / task


@dataclass(frozen=True)
class Fold:
    task: str
    fold: int
    table: pd.DataFrame  # raw table fields, (N, table columns)
    y: np.ndarray
    support: np.ndarray  # True for the labeled pool, False for the query rows
    ids: list[str]
    categorical: list[str]
    keys: list[list[str]]  # feature-bank identities of each row (views in order)
    data: Path
    cache: dict = field(default_factory=dict, compare=False, repr=False)  # encoder -> (N, p)

    @cached_property
    def pool(self) -> np.ndarray:
        return np.flatnonzero(self.support)

    @cached_property
    def query(self) -> np.ndarray:
        return np.flatnonzero(~self.support)

    @cached_property
    def labels(self) -> np.ndarray:
        return np.unique(self.y[self.pool])

    def embedding(self, encoder: str) -> np.ndarray:
        if encoder not in self.cache:
            kind = ENCODERS[encoder][0]
            if kind not in blocks(self.task):
                raise ValueError(f"{encoder} does not embed {self.task}")
            keys = self.keys if kind == "image" else [[i] for i in self.ids]
            bank = FeatureBank.load(bank_path(self.data, encoder, self.task))
            # (N, p) float32, the v views of a row concatenated before any PCA
            self.cache[encoder] = bank.rows(keys).astype(np.float32)
        return self.cache[encoder]


def load_fold(data: Path, task: str, fold: int) -> Fold:
    meta = json.loads(manifest_path(data, task, fold).read_text())
    rows = meta["rows"]
    ids = [row["row_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Need nonempty rows with unique row_id")
    split = np.array([row["split"] for row in rows])
    if set(split) != {"support", "query"}:
        raise ValueError("Each fold needs an explicit support/query split")
    y = np.array([row["target"] for row in rows])
    support = split == "support"
    if not set(y[~support]) <= set(y[support]) or len(set(y[support])) < 2:
        raise ValueError("Query labels must belong to support classes")
    columns = list(rows[0]["table"])
    if any(set(row["table"]) != set(columns) for row in rows):
        raise ValueError("Every row needs the same table fields")
    frame = pd.DataFrame([row["table"] for row in rows], columns=columns)
    frame = frame.where(frame.notna(), np.nan)
    categorical = meta.get("categorical_columns", [])
    for column in categorical:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    table = pd.concat([frame], axis=1)
    if "image" in blocks(task):
        keys = [list(row["image_ids"]) for row in rows]
        if not keys[0] or len({len(k) for k in keys}) != 1:
            raise ValueError("Every row needs the same nonzero number of images")
        shared = ({k for i in np.flatnonzero(support) for k in keys[i]}
                  & {k for i in np.flatnonzero(~support) for k in keys[i]})
        if shared and meta.get("image_overlap_policy") != "preserve_source_split":
            raise ValueError("The same image occurs in the labeled pool and the query")
    else:
        keys = [[i] for i in ids]
    return Fold(task, fold, table, y, support, ids, categorical, keys, Path(data))
