import numpy as np

from waferlens.data.transforms import augment_batch, pad_or_crop, to_onehot_chw


def test_onehot_channels():
    m = np.array([[[0, 1, 2], [2, 1, 0], [1, 1, 1]]], dtype=np.uint8)
    x = to_onehot_chw(m)
    assert x.shape == (1, 3, 3, 3)
    # one-hot: channels sum to 1 everywhere
    assert np.allclose(x.sum(axis=1), 1.0)


def test_pad_or_crop_to_size():
    m = np.ones((1, 40, 40), dtype=np.uint8)
    out = pad_or_crop(m, 52)
    assert out.shape == (1, 52, 52)
    big = np.ones((1, 60, 60), dtype=np.uint8)
    out2 = pad_or_crop(big, 52)
    assert out2.shape == (1, 52, 52)


def test_augment_preserves_shape_and_values():
    rng = np.random.default_rng(0)
    x = to_onehot_chw(np.random.randint(0, 3, size=(4, 52, 52)).astype(np.uint8))
    out = augment_batch(x, rng, rotate90=True, flip=True)
    assert out.shape == x.shape
    # still one-hot after augmentation
    assert np.allclose(out.sum(axis=1), 1.0)
