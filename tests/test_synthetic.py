import numpy as np

from waferlens.data.synthetic import generate, save_npz


def test_generate_shapes(synth):
    X, Y, classes = synth
    assert X.shape == (240, 52, 52)
    assert Y.shape == (240, 8)
    assert set(np.unique(X)).issubset({0, 1, 2})
    assert len(classes) == 8


def test_generate_has_all_states():
    X, Y = generate(n=200, size=52, seed=3)
    # blank, good, and defective dies should all appear
    assert set(np.unique(X)) == {0, 1, 2}
    # labels are multi-hot (some rows have >1 active class)
    assert (Y.sum(axis=1) > 1).any()
    # some rows are 'none' (all zero)
    assert (Y.sum(axis=1) == 0).any()


def test_save_npz_roundtrip(tmp_path):
    p = tmp_path / "s.npz"
    save_npz(p, n=50, size=52, seed=1)
    from waferlens.data.mixedwm38 import load_npz
    d = load_npz(p)
    assert len(d.X) == 50
    assert d.Y.shape[1] == 8
