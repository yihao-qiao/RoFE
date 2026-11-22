import numpy as np
import pytest

from rofe.experiments import Context, get_experiment
from rofe.method import labeled_scores
from rofe.pfn.tabpfn import score_exports
from rofe.pfn.tabpfn_predict import design
from rofe.scaling import global_scaled
from rofe.tasks import load_fold

CFG = {"encoders": ["dinov3"], "levels": [5, 0], "draws": 2, "settings": ["embedding", "joint"]}


def test_fold_loads_views_concatenated(data):
    fold = load_fold(data, "cbis_mass", 0)
    assert fold.embedding("dinov3").shape == (120, 72)  # three views of width 24
    assert fold.table.shape[1] == 2 and fold.categorical == ["kind"]
    assert set(fold.y[fold.query]) <= set(fold.labels)


def test_interface_records(data, fake_pfn):
    fold = load_fold(data, "petfinder", 0)
    records = list(get_experiment("interface").run(fold, CFG, Context(data, data, fake_pfn)))
    methods = {(r["shots"], r["repeat"], r["setting"], r["method"]) for r in records}
    assert (5, 1, "joint", "rofe") in methods and (0, 0, "embedding", "comet") in methods
    assert all(0 <= r["accuracy"] <= 1 for r in records)
    first = [r for r in records if r["method"] == "comet" and r["shots"] == 5][0]
    assert first["width"] == first["n_labeled"] - 1


def test_comet_records_match_direct_scores(data, fake_pfn):
    fold = load_fold(data, "petfinder", 1)
    cfg = {**CFG, "levels": [5], "draws": 1, "settings": ["embedding"], "methods": ["comet"]}
    record = next(iter(get_experiment("interface").run(fold, cfg, Context(data, data, fake_pfn))))
    draw_rows = np.flatnonzero(fold.support)
    assert record["n_labeled"] <= len(draw_rows)
    call = fake_pfn.calls[0]
    ls = labeled_scores(fold.embedding("dinov3"), call["train"], 1, 5, 0, 0)
    np.testing.assert_array_equal(call["x"], ls.scores.astype(np.float64))


def test_basis_records(data, fake_pfn):
    fold = load_fold(data, "petfinder", 0)
    cfg = {**CFG, "levels": [10], "draws": 1, "settings": ["embedding"]}
    records = list(get_experiment("basis").run(fold, cfg, Context(data, data, fake_pfn)))
    assert {r["method"] for r in records} == {"comet", "comet_x4", "rot1", "rofe", "rot_strong",
                                             "rot_weak", "hadamard", "givens"}


def test_heads_records(data):
    fold = load_fold(data, "petfinder", 0)
    cfg = {**CFG, "levels": [5], "draws": 1, "methods": ["logreg_cv", "knn"]}
    records = list(get_experiment("heads").run(fold, cfg, Context(data, data)))
    assert {(r["setting"], r["method"]) for r in records} == {
        ("embedding", "logreg_cv"), ("embedding", "knn"), ("joint", "logreg_cv"), ("joint", "knn")}


def test_tabpfn_export_and_score(data, tmp_path):
    fold = load_fold(data, "petfinder", 0)
    cfg = {**CFG, "levels": [5], "draws": 1}
    exports, predictions = tmp_path / "exports", tmp_path / "predictions"
    get_experiment("interface").export(fold, cfg, Context(data, data), exports)
    predictions.mkdir()
    for path in sorted(exports.glob("*_meta.npz")):
        stem = path.name[:-len("_meta.npz")]
        with np.load(path) as m, np.load(exports / f"{stem}_blocks.npz") as b:
            meta, blocks = dict(m), dict(b)
        saved = {}
        for setting in ("embedding", "joint"):
            for job, block in zip(meta["job_names"], meta["job_blocks"]):
                X_train, y_train, X_test, _ = design(meta, blocks, setting, str(block))
                assert X_train.shape[0] == len(meta["rows"]) == len(y_train)
                labels = meta["labels"]
                proba = np.full((len(X_test), len(labels)), 1 / len(labels))  # (n_q, c)
                saved[f"{setting}_{job}_proba"] = proba
                saved[f"{setting}_{job}_classes"] = labels
        np.savez(predictions / f"{stem}.npz", **saved)
    records = list(score_exports(exports, predictions))
    assert {r["method"] for r in records} == {"comet", "rofe", "rot1"}


def test_global_scale_sets_the_mean_variance():
    rng = np.random.default_rng(0)
    block = rng.normal(size=(20, 5)).astype(np.float32) * np.arange(1, 6)
    rows = np.arange(10)
    out = global_scaled(block, rows)
    np.testing.assert_allclose(out[rows].var(0).mean(), 1, rtol=1e-5)
    np.testing.assert_allclose(out / block, out[0, 0] / block[0, 0], rtol=1e-5)


def test_unknown_experiment():
    with pytest.raises(ValueError):
        get_experiment("nothing")
