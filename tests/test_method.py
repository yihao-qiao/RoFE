import numpy as np
import pandas as pd

from rofe.mechanism import givens_basis, hadamard_basis
from rofe.method import RoFE, labeled_scores, pfn_frame
from rofe.metrics import padded, scores


def _data(seed: int = 0, n: int = 30, m: int = 12, p: int = 16):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n // 3)
    centres = rng.normal(size=(3, p)) * 3
    x = (centres[y] + rng.normal(size=(n, p))).astype(np.float32)
    yq = rng.integers(0, 3, size=m)
    xq = (centres[yq] + rng.normal(size=(m, p))).astype(np.float32)
    return x, y, xq, yq


def test_comet_path_is_rofe_without_rotation(fake_pfn):
    x, y, xq, _ = _data(p=40)
    comet = RoFE(fake_pfn, rotate=False).fit(x, y).predict_proba(xq)
    assert len(fake_pfn.calls) == 1
    call = fake_pfn.calls[0]
    assert call["x"].shape[1] == len(y) - 1  # w = min(n - 1, 256) principal columns
    np.testing.assert_allclose(call["x"][:len(y)].mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(comet.sum(1), 1)


def test_rofe_averages_k_rotations(fake_pfn):
    x, y, xq, _ = _data()
    proba = RoFE(fake_pfn, n_rotations=4, seeds=[[1, 2, 3, 4]]).fit(x, y).predict_proba(xq)
    assert len(fake_pfn.calls) == 4
    first = fake_pfn.calls[0]["x"]
    for call in fake_pfn.calls[1:]:
        d0 = ((first[:, None] - first[None]) ** 2).sum(-1)
        d1 = ((call["x"][:, None] - call["x"][None]) ** 2).sum(-1)
        np.testing.assert_allclose(d0, d1, rtol=1e-3, atol=1e-3)  # same geometry
        assert not np.allclose(call["x"], first)
    np.testing.assert_allclose(proba.sum(1), 1)


def test_labels_are_only_context(fake_pfn):
    x, y, xq, _ = _data()
    RoFE(fake_pfn, rotate=False).fit(x, y).predict_proba(xq)
    a = fake_pfn.calls[-1]["x"]
    RoFE(fake_pfn, rotate=False).fit(x, np.roll(y, 1)).predict_proba(xq)
    np.testing.assert_array_equal(a, fake_pfn.calls[-1]["x"])


def test_table_columns_pass_unchanged(fake_pfn):
    x, y, xq, _ = _data()
    table = pd.DataFrame({"age": np.arange(len(y), dtype=float)})
    query = pd.DataFrame({"age": -np.arange(len(xq), dtype=float)})
    RoFE(fake_pfn, seeds=[[5, 6, 7, 8]]).fit(x, y, table).predict_proba(xq, query)
    call = fake_pfn.calls[0]
    assert call["columns"][0] == "age" and call["columns"][1] == "visual_0"
    np.testing.assert_array_equal(call["x"][:, 0], np.r_[table.age, query.age])


def test_labeled_scores_seeds_follow_the_draw():
    x, _, _, _ = _data(p=40)
    rows = np.arange(30)
    a = labeled_scores(x, rows, 0, 10, 0, 0)
    b = labeled_scores(x, rows, 0, 10, 0, 0)
    c = labeled_scores(x, rows, 0, 10, 1, 0)
    assert a.seeds == b.seeds and a.seeds != c.seeds and len(a.seeds) == 4
    assert a.width == 29 and 1 <= a.gd_rank <= 29


def test_equal_variance_bases():
    x, _, _, _ = _data(p=40)
    rows = np.arange(30)
    z = labeled_scores(x, rows, 0, 10, 0, 0).scores
    g = givens_basis(z, rows)
    np.testing.assert_allclose(g[rows].var(0), z[rows].var(0).mean(), rtol=1e-3)
    h = hadamard_basis(z)
    assert h.shape[1] == 32
    np.testing.assert_allclose(h[rows].var(0), z[rows].var(0).sum() / 32, rtol=1e-3)


def test_padded_and_scores():
    labels = np.array([0, 1, 2])
    p = padded(np.array([0, 2]), np.array([[0.7, 0.3], [0.2, 0.8]]), labels)
    assert p[:, 1].tolist() == [0.0, 0.0]
    s = scores(labels, p, np.array([0, 2]), labels)
    assert s["accuracy"] == 1.0 and np.isnan(s["auroc"])


def test_pfn_frame():
    table = pd.DataFrame({"a": [1.0, 2.0]})
    block = np.ones((2, 3), np.float32)
    assert list(pfn_frame(block, table, joint=True).columns) == ["a", "visual_0", "visual_1",
                                                                  "visual_2"]
    assert list(pfn_frame(block, table, joint=False).columns) == ["visual_0", "visual_1",
                                                                   "visual_2"]
