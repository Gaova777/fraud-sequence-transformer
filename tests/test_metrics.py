from fraud_sequence_transformer.domain.metrics import best_cost_threshold, best_f1_threshold, cost, f1_at


def test_f1_and_its_threshold():
    y, s = [1, 1, 0, 0], [0.9, 0.6, 0.7, 0.1]
    assert f1_at(y, s, 0.65)["f1"] == 0.5
    assert best_f1_threshold(y, s, [0.5, 0.65, 0.8]) == 0.5


def test_cost_counts_missed_amounts_and_reviews():
    y, s, a = [1, 0, 1], [0.9, 0.8, 0.2], [500.0, 20.0, 1000.0]
    assert cost(y, s, a, 0.5, review_cost=10) == 10 + 10 + 1000
    # a big missed fraud makes the low threshold worth its reviews
    assert best_cost_threshold(y, s, a, [0.1, 0.5], review_cost=10) == 0.1
