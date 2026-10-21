from datetime import datetime

import polars as pl

from fraud_sequence_transformer.application.features import history


def test_history_uses_only_earlier_transactions():
    df = pl.DataFrame({
        "user": [1, 1, 1, 2],
        "ts": [datetime(2020, 1, 1, 10), datetime(2020, 1, 1, 20), datetime(2020, 1, 3, 9), datetime(2020, 1, 1, 10)],
        "amount": [10.0, 30.0, 40.0, 5.0],
        "merchant": ["a", "a", "b", "a"], "state": ["CA", "CA", "NY", "CA"],
        "errors": ["", "", "Bad PIN", ""], "chip": ["Swipe Transaction"] * 3 + ["Online Transaction"],
        "mcc": ["5411"] * 4,
    })
    out = history(df)
    assert out["amount_vs_mean"].to_list() == [1.0, 3.0, 2.0, 1.0]
    assert out["new_merchant"].to_list() == [1, 0, 1, 1]
    assert out["new_state"].to_list() == [1, 0, 1, 1]
    assert out["tx_last_24h"].to_list() == [0, 1, 0, 0]
    assert out["minutes_since_prev"].to_list()[1] == 600
    assert out["has_error"].to_list() == [0, 0, 1, 0] and out["online"].to_list() == [0, 0, 0, 1]
