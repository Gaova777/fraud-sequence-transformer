"""Sequences of a customer's consecutive transactions, as in the TabFormer paper: windows of 10, stride 10."""


def windows(user_ids: list[int], size: int = 10, stride: int = 10) -> list[int]:
    """Start row of every full window that stays inside one customer's history (rows sorted by user, then time)."""
    starts, begin = [], 0
    for i in range(1, len(user_ids) + 1):
        if i == len(user_ids) or user_ids[i] != user_ids[begin]:
            starts += range(begin, i - size + 1, stride)
            begin = i
    return starts


def label(fraud: list[int]) -> int:
    """A window is fraudulent if any of its transactions is."""
    return int(any(fraud))
