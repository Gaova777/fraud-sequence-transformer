from fraud_sequence_transformer.domain.windows import label, windows


def test_windows_never_cross_customers():
    users = [1] * 25 + [2] * 9 + [3] * 10
    assert windows(users) == [0, 10, 34]  # user 2 is too short; user 1's last 5 rows don't fill a window
    assert windows([1] * 12, size=10, stride=1) == [0, 1, 2]


def test_label():
    assert label([0, 0, 1]) == 1 and label([0, 0]) == 0
