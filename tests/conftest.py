import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rofe.features.bank import FeatureBank  # noqa: E402


# deterministic stand-in for a PFN: nearest class mean in the given columns
class FakePFN:
    in_process = True

    def __init__(self):
        self.calls = []

    def fit_predict(self, X, y, train, test, seed, **options):
        x = X.select_dtypes("number").to_numpy(np.float64)
        classes = np.unique(y[train])
        means = np.stack([x[train][y[train] == c].mean(0) for c in classes])
        d = ((x[test][:, None, :] - means[None]) ** 2).sum(-1)
        logits = -d / (d.std() + 1e-9)
        p = np.exp(logits - logits.max(1, keepdims=True))
        self.calls.append({"columns": list(X.columns), "train": np.array(train), "seed": seed,
                           "x": x.copy()})
        return classes, p / p.sum(1, keepdims=True)


# synthetic task in the prepared layout: five folds and one image feature bank
def write_task(data: Path, task: str, n: int = 120, p: int = 24, views: int = 1,
               classes: int = 3, seed: int = 0, table: bool = True) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    centres = rng.normal(size=(classes, p))
    images, rows = {}, []
    for i in range(n):
        keys = []
        for v in range(views):
            key = f"img{i}_{v}"
            images[key] = centres[y[i]] + rng.normal(size=p)
            keys.append(key)
        fields = {"age": float(rng.normal()), "kind": ["a", "b", None][i % 3]} if table else {}
        rows.append({"row_id": f"r{i}", "target": int(y[i]), "table": fields, "image_ids": keys})
    directory = data / "tasks" / task
    directory.mkdir(parents=True)
    for fold in range(5):
        order = np.random.default_rng(fold).permutation(n)
        split = {int(i): "query" if j < n // 4 else "support" for j, i in enumerate(order)}
        manifest = {"categorical_columns": ["kind"] if table else [],
                    "rows": [{**r, "split": split[i]} for i, r in enumerate(rows)]}
        (directory / f"seed{fold}.json").write_text(json.dumps(manifest))
    keys = sorted(images)
    bank = FeatureBank(np.stack([images[k] for k in keys]).astype(np.float32), tuple(keys),
                       {"kind": "synthetic"})
    bank.save(data / "features" / "dinov3" / task)


@pytest.fixture
def data(tmp_path: Path) -> Path:
    write_task(tmp_path, "petfinder")
    write_task(tmp_path, "cbis_mass", views=3, seed=1)
    return tmp_path


@pytest.fixture
def fake_pfn() -> FakePFN:
    return FakePFN()
