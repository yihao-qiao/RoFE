import numpy as np

from rofe.draws import draws, labeled_rows, many_draws, subsample


def test_subsample_is_reproducible_and_capped():
    y = np.repeat([0, 1, 2], [30, 5, 40])
    pool = np.arange(len(y))
    rows = subsample(y, pool, 10, (0, 10, 0))
    np.testing.assert_array_equal(rows, subsample(y, pool, 10, (0, 10, 0)))
    assert np.all(np.diff(rows) > 0)
    counts = np.bincount(y[rows])
    assert counts.tolist() == [10, 5, 10]  # a class with fewer rows enters with all of them
    assert not np.array_equal(rows, subsample(y, pool, 10, (0, 10, 1)))


def test_whole_pool_and_duplicate_levels():
    y = np.repeat([0, 1], [300, 200])
    pool = np.arange(len(y))
    np.testing.assert_array_equal(labeled_rows(y, pool, 0, 3), pool)
    found = [(d.shots, d.repeat) for d in many_draws(y, pool, 3)]
    assert (0, 0) in found and (250, 0) in found
    assert all(s != 500 for s, _ in found)  # 500 per class equals the whole pool
    assert sum(1 for s, _ in found if s == 0) == 1
    assert (250, 1) in found and (250, 2) in found


def test_draws_cover_the_requested_levels():
    y = np.repeat([0, 1, 2], [400, 600, 700])
    pool = np.arange(len(y))
    levels = [10, 25, 250, 0]
    found = [(d.shots, d.repeat) for d in draws(y, pool, 1, levels)]
    assert {s for s, _ in found} == set(levels)
    assert sum(1 for s, _ in found if s == 10) == 3
    again = [d.rows for d in draws(y, pool, 1, levels)]
    assert all(np.array_equal(a.rows, b) for a, b in zip(draws(y, pool, 1, levels), again))
