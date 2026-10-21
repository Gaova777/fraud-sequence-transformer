"""Pick the alert threshold by money instead of F1, after calibrating the scores.

- Calibration: isotonic regression fitted on validation windows, so a score of 0.8 means about 80% of such windows
  are fraud. Reported as expected calibration error (ECE) before and after.
- Cost: a missed fraud window loses the amount of its fraudulent transactions; every alert costs a fixed review fee
  (an analyst's time plus the customer's friction). The fee is an assumption, so it is swept.

usage: uv run scripts/cost.py lstm --split random
"""
import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from fraud_sequence_transformer.domain.metrics import best_cost_threshold, best_f1_threshold, cost, f1_at

SIZE = 10
FEES = [5, 25, 100]


def ece(labels, probs, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in pairwise(edges):
        m = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo)
        if m.any():
            total += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(total)


def main(args):
    w = np.load("data/processed/windows.npz")
    split, labels, starts = w[f"split_{args.split}"], w["labels"], w["starts"]
    scores = np.load(f"data/models/scores_{args.model}_{args.split}.npy")
    tx = pl.read_parquet("data/processed/transactions.parquet", columns=["amount", "fraud"])
    rows = starts[:, None] + np.arange(SIZE)
    lost = (np.abs(tx["amount"].to_numpy())[rows] * tx["fraud"].to_numpy()[rows]).sum(1)  # what a miss costs
    val, test = split == 1, split == 2
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores[val], labels[val])
    calibrated = iso.predict(scores)

    grid = np.unique(np.quantile(calibrated[val], np.linspace(0.5, 0.99999, 400))).tolist()
    f1_t = best_f1_threshold(labels[val].tolist(), calibrated[val].tolist(), grid)
    out = {"model": args.model, "split": args.split,
           "ece_test_raw": ece(labels[test], scores[test]), "ece_test_calibrated": ece(labels[test], calibrated[test]),
           "fraud_amount_test": float(lost[test].sum()), "by_review_fee": {}}
    y_t, s_t, a_t = labels[test].tolist(), calibrated[test].tolist(), lost[test].tolist()
    for fee in FEES:
        cost_t = best_cost_threshold(labels[val].tolist(), calibrated[val].tolist(), lost[val].tolist(), grid, fee)
        out["by_review_fee"][fee] = {
            "f1_threshold": {"threshold": f1_t, "cost": cost(y_t, s_t, a_t, f1_t, fee), **f1_at(y_t, s_t, f1_t)},
            "cost_threshold": {"threshold": cost_t, "cost": cost(y_t, s_t, a_t, cost_t, fee), **f1_at(y_t, s_t, cost_t)},
            "no_model_cost": float(lost[test].sum()),
        }
        r = out["by_review_fee"][fee]
        print(f"  fee ${fee:>3}: F1 threshold costs ${r['f1_threshold']['cost']:,.0f} "
              f"(recall {r['f1_threshold']['recall']:.2f}); cost threshold ${r['cost_threshold']['cost']:,.0f} "
              f"(recall {r['cost_threshold']['recall']:.2f}); doing nothing ${r['no_model_cost']:,.0f}")
    print(f"  ECE on test: {out['ece_test_raw']:.4f} raw -> {out['ece_test_calibrated']:.4f} calibrated")
    Path("results/cost").mkdir(parents=True, exist_ok=True)
    Path(f"results/cost/{args.model}_{args.split}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--split", choices=["random", "chrono"], default="random")
    main(parser.parse_args())
