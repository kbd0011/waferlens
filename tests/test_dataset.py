from waferlens.data.dataset import make_splits


def test_make_splits_partitions(synth):
    X, Y, classes = synth
    s = make_splits(X, Y, classes, val_fraction=0.2, test_fraction=0.2, seed=0,
                    augment=True, rotate90=True, flip=True)
    total = len(s.train) + len(s.val) + len(s.test)
    assert total == len(X)
    # multi-label -> pos_weight present, one per class
    assert s.pos_weight is not None
    assert len(s.pos_weight) == 8


def test_dataset_item_tensor_shapes(synth):
    X, Y, classes = synth
    s = make_splits(X, Y, classes, 0.2, 0.2, 0, True, True, True)
    xb, yb = s.train[0]
    assert tuple(xb.shape) == (3, 52, 52)
    assert tuple(yb.shape) == (8,)
