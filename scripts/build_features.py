"""Customer profiles as of a cutoff date, written to Feast's offline source and materialized to the online store.

usage: uv run --group mlops scripts/build_features.py --cutoff 2018-01-01
"""
import argparse
from datetime import datetime
from pathlib import Path

import polars as pl
from feast import FeatureStore

REPO = Path("feature_repo")


def main(args):
    cutoff = datetime.fromisoformat(args.cutoff)
    df = pl.read_parquet("data/processed/transactions.parquet", columns=["user", "ts", "amount", "state"])
    df = df.filter(pl.col("ts") < cutoff)
    profile = df.group_by("user").agg(
        mean_amount=pl.col("amount").mean(),
        last_ts=pl.col("ts").max().dt.epoch("s"),
        home_state=pl.col("state").fill_null("ONLINE").mode().first(),
        tx_count=pl.len().cast(pl.Int64),
    ).with_columns(event_timestamp=pl.lit(cutoff))
    (REPO / "data").mkdir(parents=True, exist_ok=True)
    profile.write_parquet(REPO / "data" / "customer_profile.parquet")

    import sys
    sys.path.insert(0, str(REPO))
    from definitions import customer, customer_profile

    store = FeatureStore(repo_path=str(REPO))
    store.apply([customer, customer_profile])
    store.materialize(start_date=datetime(1990, 1, 1), end_date=cutoff)
    print(f"{profile.height} customer profiles materialized as of {cutoff:%Y-%m-%d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2018-01-01")
    main(parser.parse_args())
