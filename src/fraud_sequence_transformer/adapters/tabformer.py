"""IBM's TabFormer synthetic card transactions (Apache-2.0): one CSV, 24.4M rows, 20,000 users."""
from pathlib import Path

import polars as pl


def load(path: Path = Path("data/raw/card_transaction.v1.csv")) -> pl.LazyFrame:
    """Typed columns, one timestamp, sorted by user and time."""
    df = pl.scan_csv(path, schema_overrides={"Merchant Name": pl.Utf8, "Zip": pl.Utf8, "MCC": pl.Utf8})
    hh_mm = pl.col("Time").str.split(":")
    return (
        df.with_columns(
            amount=pl.col("Amount").str.replace(r"\$", "").cast(pl.Float64),
            ts=pl.datetime(pl.col("Year"), pl.col("Month"), pl.col("Day"),
                           hh_mm.list.get(0).cast(pl.Int32), hh_mm.list.get(1).cast(pl.Int32)),
            fraud=(pl.col("Is Fraud?") == "Yes").cast(pl.Int8),
            errors=pl.col("Errors?").fill_null(""),
        )
        .rename({"User": "user", "Card": "card", "Use Chip": "chip", "Merchant Name": "merchant",
                 "Merchant City": "city", "Merchant State": "state", "Zip": "zip", "MCC": "mcc"})
        .select("user", "card", "ts", "amount", "chip", "merchant", "city", "state", "zip", "mcc", "errors", "fraud")
        .sort("user", "ts")
    )
