"""The industry baseline: XGBoost scores each transaction from its own fields and the customer's history features.
A window's score is the highest of its ten transactions, so it is compared with the sequence models on the same task.

usage: uv run scripts/train_xgboost.py --split random
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from fraud_sequence_transformer.application.features import FEATURES
from fraud_sequence_transformer.domain.metrics import best_f1_threshold, f1_at

SIZE = 10


def rows_of(starts):
    return (starts[:, None] + np.arange(SIZE)).ravel()


def main(args):
    df = pl.read_parquet("data/processed/transactions.parquet", columns=FEATURES + ["fraud"])
    w = np.load("data/processed/windows.npz")
    split = w[f"split_{args.split}"]
    x, y = df.select(FEATURES).to_numpy().astype(np.float32), df["fraud"].to_numpy()
    tr, va = rows_of(w["starts"][split == 0]), rows_of(w["starts"][split == 1])
    model = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", device=args.device, scale_pos_weight=(y[tr] == 0).sum() / max(1, y[tr].sum()),
                              eval_metric="aucpr", early_stopping_rounds=40 if args.trees is None else None,
                              n_jobs=16)
    if args.trees:
        model.set_params(n_estimators=args.trees)
    model.fit(x[tr], y[tr], eval_set=[(x[va], y[va])], verbose=100)

    scores = np.zeros(len(df), dtype=np.float32)
    keep = rows_of(w["starts"][split > 0])
    scores[keep] = model.predict_proba(x[keep])[:, 1]
    window_scores = scores[w["starts"][:, None] + np.arange(SIZE)].max(1)
    labels = w["labels"]
    val, test = split == 1, split == 2
    grid = np.unique(np.quantile(window_scores[val], np.linspace(0.9, 0.99999, 300)))
    t = best_f1_threshold(labels[val].tolist(), window_scores[val].tolist(), grid.tolist())
    result = {"model": "xgboost", "split": args.split, "threshold": float(t),
              "val": f1_at(labels[val].tolist(), window_scores[val].tolist(), t),
              "test": f1_at(labels[test].tolist(), window_scores[test].tolist(), t),
              "trees": args.trees or int(model.best_iteration) + 1}
    Path("results/detection").mkdir(parents=True, exist_ok=True)
    tag = f"xgboost_{args.split}" + (f"_{args.trees}trees" if args.trees else "")
    Path(f"results/detection/{tag}.json").write_text(json.dumps(result, indent=1) + "\n")
    Path("data/models").mkdir(parents=True, exist_ok=True)
    model.save_model(f"data/models/{tag}.json")
    np.save(f"data/models/scores_{tag}.npy", window_scores)
    print(json.dumps({k: result[k] for k in ("threshold", "test")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "chrono"], default="random")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trees", type=int, default=None, help="a fixed number of trees instead of early stopping")
    main(parser.parse_args())
