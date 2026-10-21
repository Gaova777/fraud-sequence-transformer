"""HTTP API for the fraud operations console.

GET  /stream     test-set windows with each model's calibrated risk, for the console's replay
GET  /cost       for a review fee: cost, recall and alerts at every threshold, and the cheapest one
POST /score      a new transaction: the customer's profile from Feast, history features, XGBoost's risk
GET  /health
"""
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA = Path(os.environ.get("FST_DATA", "data"))
SPLIT = os.environ.get("FST_SPLIT", "random")
MODELS = ["transformer", "lstm", "mlp", "xgboost"]
SIZE = 10

app = FastAPI(title="fraud-sequence-transformer")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5175"], allow_methods=["*"], allow_headers=["*"])


@lru_cache
def test_set() -> dict:
    """Test windows: label, amount lost if missed, the last transaction's details, and calibrated scores per model."""
    import polars as pl
    from sklearn.isotonic import IsotonicRegression

    w = np.load(DATA / "processed" / "windows.npz")
    split, labels, starts = w[f"split_{SPLIT}"], w["labels"], w["starts"]
    val, test = np.flatnonzero(split == 1), np.flatnonzero(split == 2)
    tx = pl.read_parquet(DATA / "processed" / "transactions.parquet",
                         columns=["user", "ts", "amount", "merchant", "city", "state", "mcc", "chip", "fraud"])
    rows = starts[test][:, None] + np.arange(SIZE)
    amounts, fraud = np.abs(tx["amount"].to_numpy()), tx["fraud"].to_numpy()
    out = {"labels": labels[test], "lost": (amounts[rows] * fraud[rows]).sum(1), "last": starts[test] + SIZE - 1,
           "tx": tx, "scores": {}}
    for m in MODELS:
        path = DATA / "models" / f"scores_{m}_{SPLIT}.npy"
        if path.exists():
            s = np.load(path)
            iso = IsotonicRegression(out_of_bounds="clip").fit(s[val], labels[val])
            out["scores"][m] = iso.predict(s[test])
    return out


@lru_cache
def feature_store():
    from feast import FeatureStore

    return FeatureStore(repo_path="feature_repo")


@lru_cache
def xgboost_model():
    import xgboost as xgb

    m = xgb.XGBClassifier()
    m.load_model(DATA / "models" / f"xgboost_{SPLIT}.json")
    return m


@app.get("/health")
def health():
    return {"split": SPLIT, "models": list(test_set()["scores"])}


@app.get("/stream")
def stream(model: str = "transformer", n: int = 300, seed: int = 0):
    t = test_set()
    if model not in t["scores"]:
        raise HTTPException(404, f"no scores for {model}")
    rng = np.random.default_rng(seed)
    fraud, normal = np.flatnonzero(t["labels"] == 1), np.flatnonzero(t["labels"] == 0)
    # a replay with more fraud than the real 0.5%, so there is something to look at
    pick = np.concatenate([rng.choice(fraud, n // 10, replace=False), rng.choice(normal, n - n // 10, replace=False)])
    rng.shuffle(pick)
    tx = t["tx"]
    out = []
    for i in pick:
        row = tx.row(int(t["last"][i]), named=True)
        out.append({"user": row["user"], "ts": row["ts"].isoformat(), "amount": row["amount"], "city": row["city"],
                    "state": row["state"], "mcc": row["mcc"], "chip": row["chip"],
                    "risk": round(float(t["scores"][model][i]), 4), "fraud": int(t["labels"][i]),
                    "at_stake": round(float(t["lost"][i]), 2)})
    return out


@app.get("/cost")
def cost_curve(model: str = "transformer", fee: float = 25.0):
    t = test_set()
    if model not in t["scores"]:
        raise HTTPException(404, f"no scores for {model}")
    s, y, lost = t["scores"][model], t["labels"], t["lost"]
    points = []
    for th in np.unique(np.quantile(s, np.linspace(0.9, 0.99999, 120))):
        flagged = s >= th
        points.append({"threshold": float(th), "alerts": int(flagged.sum()), "recall": float(flagged[y == 1].mean()),
                       "cost": float(fee * flagged.sum() + lost[(~flagged) & (y == 1)].sum())})
    best = min(points, key=lambda p: p["cost"])
    return {"model": model, "fee": fee, "no_model_cost": float(lost.sum()), "best": best, "curve": points}


class Transaction(BaseModel):
    user: int
    amount: float
    ts: datetime
    state: str = ""
    mcc: int = 5411
    chip: str = "Chip Transaction"
    errors: str = ""
    card: int = 0


@app.post("/score")
def score(tx: Transaction):
    from fraud_sequence_transformer.application.features import FEATURES

    profile = feature_store().get_online_features(
        features=["customer_profile:mean_amount", "customer_profile:last_ts", "customer_profile:home_state"],
        entity_rows=[{"user": tx.user}]).to_dict()
    mean, last, home = profile["mean_amount"][0], profile["last_ts"][0], profile["home_state"][0]
    if mean is None:
        raise HTTPException(404, f"no profile for customer {tx.user}")
    # the same history features as in training, from the profile instead of the full history
    values = {
        "amount": tx.amount, "log_amount": float(np.sign(tx.amount) * np.log1p(abs(tx.amount))),
        "hour": tx.ts.hour, "weekday": tx.ts.isoweekday(), "amount_vs_mean": float(np.clip(tx.amount / mean, -50, 50)),
        "minutes_since_prev": float(np.clip((tx.ts.timestamp() - last) / 60, 0, 1_000_000)),
        "new_merchant": 0, "new_state": int(bool(tx.state) and tx.state != home), "tx_last_24h": 0,
        "has_error": int(bool(tx.errors)), "online": int(tx.chip == "Online Transaction"), "mcc_code": tx.mcc,
        "card": tx.card,
    }
    x = np.array([[values[f] for f in FEATURES]], dtype=np.float32)
    risk = float(xgboost_model().predict_proba(x)[0, 1])
    return {"risk": risk, "features": values, "profile": {"mean_amount": mean, "home_state": home}}
