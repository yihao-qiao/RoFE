import numpy as np
import pytest

from rofe.pca import gavish_donoho_rank, pca_width, support_pca
from rofe.rotation import haar_rotation, rotate, rotation_seeds


@pytest.mark.parametrize("width", [1, 2, 7, 49, 256])
def test_haar_rotation_is_orthogonal(width):
    r = haar_rotation(width, 3)
    assert r.shape == (width, width)
    np.testing.assert_allclose(r.T @ r, np.eye(width), atol=1e-10)


def test_haar_rotation_is_reproducible_and_seed_dependent():
    np.testing.assert_array_equal(haar_rotation(16, 5), haar_rotation(16, 5))
    assert not np.allclose(haar_rotation(16, 5), haar_rotation(16, 6))


def test_rotation_keeps_distances_and_mean_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 30)).astype(np.float32)
    rows = np.arange(20)
    z, _ = support_pca(x, rows)
    zr = rotate(z, 11)
    dz = ((z[:, None] - z[None]) ** 2).sum(-1)
    dr = ((zr[:, None] - zr[None]) ** 2).sum(-1)
    np.testing.assert_allclose(dr, dz, rtol=1e-4, atol=1e-4)
    # column variances of Z R average to the mean eigenvalue
    np.testing.assert_allclose(zr[rows].var(0).mean(), z[rows].var(0).mean(), rtol=1e-4)
    np.testing.assert_allclose(zr[rows].mean(0), 0, atol=1e-5)


def test_partial_rotation_leaves_other_columns():
    z = np.random.default_rng(1).normal(size=(10, 6)).astype(np.float32)
    out = rotate(z, 2, np.arange(3, 6))
    np.testing.assert_array_equal(out[:, :3], z[:, :3])
    np.testing.assert_array_equal(rotate(z, 2, [4]), z)


def test_rotation_seeds():
    seeds = rotation_seeds(1, 10, 2, 0, 768, 30)
    assert len(seeds) == 4 and len(set(seeds)) == 4
    assert seeds == rotation_seeds(1, 10, 2, 0, 768, 30)
    assert seeds[1:] == rotation_seeds(1, 10, 2, 0, 768, 0)[1:]
    assert seeds[0] != rotation_seeds(1, 10, 2, 1, 768, 30)[0]
    assert rotation_seeds(1, 10, 2, 0, 768, 30, count=1) == seeds[:1]
    assert rotation_seeds(1, 10, 2, 0, 768, 30, count=16)[:4] == seeds
    with pytest.raises(ValueError):
        rotation_seeds(1, 10, 2, 0, 768, 30, count=17)


@pytest.mark.parametrize("n,p,expected", [(10, 768, 9), (50, 768, 49), (600, 768, 256),
                                          (600, 64, 64)])
def test_pca_width(n, p, expected):
    assert pca_width(n, p) == expected
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n + 5, p)).astype(np.float32)
    z, _ = support_pca(x, np.arange(n))
    assert z.shape == (n + 5, expected) and z.dtype == np.float32


def test_pca_is_fitted_on_the_labeled_rows_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 12)).astype(np.float32)
    rows = np.arange(0, 30, 2)
    z, _ = support_pca(x, rows)
    y = x.copy()
    y[1::2] += 100.0
    z2, _ = support_pca(y, rows)
    np.testing.assert_allclose(z[rows], z2[rows], atol=1e-5)
    np.testing.assert_allclose(z[rows].mean(0), 0, atol=1e-5)


def test_gavish_donoho_rank_is_bounded():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 20))
    x[:, 0] *= 50
    k = gavish_donoho_rank(x, np.arange(40), 20)
    assert 1 <= k <= 20
