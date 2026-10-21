"""Per-transaction history features, computed only from the customer's *earlier* transactions (no peeking ahead).

These are what a fraud analyst would look at, and what the per-transaction baseline gets."""
import polars as pl

FEATURES = ["amount", "log_amount", "hour", "weekday", "amount_vs_mean", "minutes_since_prev", "new_merchant",
            "new_state", "tx_last_24h", "has_error", "online", "mcc_code", "card"]


def history(df: pl.DataFrame) -> pl.DataFrame:
    """df sorted by user and time."""
    seen = pl.col("amount").shift(1).over("user")
    prev_mean = seen.cum_sum().over("user") / seen.cum_count().over("user")
    df = df.with_columns(
        hour=pl.col("ts").dt.hour(),
        weekday=pl.col("ts").dt.weekday(),
        log_amount=pl.col("amount").abs().log1p() * pl.col("amount").sign(),
        amount_vs_mean=(pl.col("amount") / prev_mean).fill_null(1.0).fill_nan(1.0).clip(-50, 50),
        minutes_since_prev=(pl.col("ts") - pl.col("ts").shift(1).over("user")).dt.total_minutes()
        .fill_null(1_000_000).clip(0, 1_000_000),
        # the first time this customer buys at this merchant, or in this state
        new_merchant=(pl.int_range(pl.len()).over(["user", "merchant"]) == 0).cast(pl.Int8),
        new_state=(pl.int_range(pl.len()).over(["user", "state"]) == 0).cast(pl.Int8),
        has_error=(pl.col("errors") != "").cast(pl.Int8),
        online=(pl.col("chip") == "Online Transaction").cast(pl.Int8),
        mcc_code=pl.col("mcc").cast(pl.Int32),
    )
    # transactions by the same customer in the 24 hours before (the current one excluded)
    counts = df.rolling(index_column="ts", period="24h", group_by="user").agg(pl.len().alias("n"))
    return df.with_columns(tx_last_24h=(counts["n"] - 1).cast(pl.Int32))
