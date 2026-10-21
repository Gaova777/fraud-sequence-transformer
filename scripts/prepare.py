"""Clean TabFormer, add history features, encode fields as tokens, cut windows and split them.

Writes data/processed/transactions.parquet (one row per transaction) and data/processed/windows.npz (window start
rows, labels, end times, and the random and chronological splits).

usage: uv run scripts/prepare.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from fraud_sequence_transformer.adapters.tabformer import load
from fraud_sequence_transformer.application.features import history
from fraud_sequence_transformer.domain.windows import windows

# hashed vocabularies for the high-cardinality fields (merchant has ~100k values)
BUCKETS = {"merchant": 50_000, "city": 10_000, "zip": 10_000}
VOCAB = ["chip", "state", "mcc", "errors"]


def encode(df: pl.DataFrame) -> pl.DataFrame:
    cols = {f"tok_{c}": (pl.col(c).hash(seed=0) % n + 1).cast(pl.Int32) for c, n in BUCKETS.items()}
    for c in VOCAB:
        cols[f"tok_{c}"] = pl.col(c).fill_null("").rank("dense").cast(pl.Int32)
    # amount in 64 log-spaced bins, keeping refunds (negative amounts) apart
    cols["tok_amount"] = (pl.col("log_amount").clip(-10, 10) * 3.1 + 32).cast(pl.Int32).clip(1, 64)
    cols["tok_hour"] = pl.col("hour").cast(pl.Int32) + 1
    cols["tok_weekday"] = pl.col("weekday").cast(pl.Int32)
    cols["tok_card"] = pl.col("card").cast(pl.Int32) + 1
    return df.with_columns(**cols)


def main(args):
    df = load(args.csv).collect()
    df = encode(history(df))
    args.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out / "transactions.parquet")

    starts = np.array(windows(df["user"].to_list(), args.size, args.size), dtype=np.int64)
    fraud = df["fraud"].to_numpy()
    labels = np.array([fraud[s:s + args.size].max() for s in starts], dtype=np.int8)
    end = df["ts"].to_numpy()[starts + args.size - 1]
    rng = np.random.default_rng(args.seed)
    r = rng.random(len(starts))
    year = end.astype("datetime64[Y]").astype(int) + 1970
    split = {"random": np.where(r < 0.6, 0, np.where(r < 0.8, 1, 2)),  # the paper's 60/20/20
             "chrono": np.where(year <= 2015, 0, np.where(year <= 2017, 1, 2))}  # train on the past
    np.savez_compressed(args.out / "windows.npz", starts=starts, labels=labels, end=end.astype("int64"),
                        split_random=split["random"], split_chrono=split["chrono"])
    summary = {"transactions": df.height, "users": df["user"].n_unique(), "fraud_transactions": int(fraud.sum()),
               "windows": len(starts), "fraud_windows": int(labels.sum())}
    for name, s in split.items():
        summary[name] = {p: {"windows": int((s == i).sum()), "fraud": int(labels[s == i].sum())}
                         for i, p in enumerate(("train", "val", "test"))}
    Path("results").mkdir(exist_ok=True)
    Path("results/dataset.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/raw/card_transaction.v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
