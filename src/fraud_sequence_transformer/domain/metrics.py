"""Fraud metrics in plain Python: F1 at a threshold, and the money a threshold saves or loses."""


def f1_at(labels: list[int], scores: list[float], threshold: float) -> dict:
    tp = fp = fn = 0
    for y, s in zip(labels, scores):
        flagged = s >= threshold
        tp += flagged and y == 1
        fp += flagged and y == 0
        fn += (not flagged) and y == 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def best_f1_threshold(labels, scores, candidates) -> float:
    return max(candidates, key=lambda t: f1_at(labels, scores, t)["f1"])


def cost(labels, scores, amounts, threshold: float, review_cost: float) -> float:
    """What a threshold costs: every missed fraud loses its amount, every alert costs a review (paid on true alerts too,
    since someone has to look). Lower is better."""
    total = 0.0
    for y, s, a in zip(labels, scores, amounts):
        if s >= threshold:
            total += review_cost
        elif y == 1:
            total += abs(a)
    return total


def best_cost_threshold(labels, scores, amounts, candidates, review_cost: float) -> float:
    return min(candidates, key=lambda t: cost(labels, scores, amounts, t, review_cost))
