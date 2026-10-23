"""Train a window model (MLP, LSTM or transformer) on the token fields of ten consecutive transactions.

Like the paper, fraud windows are upsampled to roughly balance each training epoch. The threshold is the one with the
best F1 on validation; window scores are saved for the cost analysis.

usage: uv run scripts/train_sequence.py lstm --split random
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from fraud_sequence_transformer.application.sequence_models import FIELDS, MODELS
from fraud_sequence_transformer.domain.metrics import best_f1_threshold, f1_at

SIZE = 10


def gather(table, starts, device):
    """The ten token rows of each window, gathered in RAM and sent to the GPU one batch at a time: the 24M-row table
    stays out of GPU memory, which another training may be using."""
    return torch.as_tensor(table[starts[:, None] + np.arange(SIZE)], device=device)


@torch.no_grad()
def score(model, table, starts, device, batch=8192):
    model.eval()
    out = [torch.sigmoid(model(gather(table, starts[i:i + batch], device))).float().cpu()
           for i in range(0, len(starts), batch)]
    return torch.cat(out).numpy()


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = pl.read_parquet("data/processed/transactions.parquet", columns=FIELDS).to_numpy().astype(np.int64)
    vocab = [int(tokens[:, i].max()) for i in range(tokens.shape[1])]
    table = tokens
    w = np.load("data/processed/windows.npz")
    split, labels, starts = w[f"split_{args.split}"], w["labels"], w["starts"]
    train = np.flatnonzero(split == 0)
    pos, neg = train[labels[train] == 1], train[labels[train] == 0]
    model = MODELS[args.model](vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        # every epoch: all normal windows plus fraud windows drawn with replacement up to a 1:4 ratio
        idx = np.concatenate([neg, rng.choice(pos, len(neg) // 4, replace=True)])
        rng.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            y = torch.as_tensor(labels[b], device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(model(gather(table, starts[b], device)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  epoch {epoch}: loss {loss.item():.4f}  {time.time() - start_time:.0f}s", flush=True)

    val, test = np.flatnonzero(split == 1), np.flatnonzero(split == 2)
    scores = np.zeros(len(starts), dtype=np.float32)
    scores[val] = score(model, table, starts[val], device)
    scores[test] = score(model, table, starts[test], device)
    grid = np.unique(np.quantile(scores[val], np.linspace(0.9, 0.99999, 300)))
    t = best_f1_threshold(labels[val].tolist(), scores[val].tolist(), grid.tolist())
    result = {"model": args.model, "split": args.split, "threshold": float(t), "epochs": args.epochs,
              "parameters": sum(p.numel() for p in model.parameters()),
              "train_seconds": round(time.time() - start_time),
              "val": f1_at(labels[val].tolist(), scores[val].tolist(), t),
              "test": f1_at(labels[test].tolist(), scores[test].tolist(), t)}
    Path("results/detection").mkdir(parents=True, exist_ok=True)
    Path(f"results/detection/{args.model}_{args.split}.json").write_text(json.dumps(result, indent=1) + "\n")
    Path("data/models").mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "vocab": vocab, "threshold": t}, f"data/models/{args.model}_{args.split}.pt")
    np.save(f"data/models/scores_{args.model}_{args.split}.npy", scores)
    print(json.dumps({k: result[k] for k in ("threshold", "test")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODELS))
    parser.add_argument("--split", choices=["random", "chrono"], default="random")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
