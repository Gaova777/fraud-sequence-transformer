import numpy as np
from fastapi.testclient import TestClient

from fraud_sequence_transformer.entrypoints import api


def test_cost_curve_finds_the_cheapest_threshold(monkeypatch):
    fake = {"labels": np.array([1, 0, 0, 1, 0]), "lost": np.array([500.0, 0, 0, 30.0, 0]),
            "scores": {"lstm": np.array([0.9, 0.1, 0.2, 0.3, 0.05])}}
    monkeypatch.setattr(api, "test_set", lambda: fake)
    c = TestClient(api.app)
    r = c.get("/cost", params={"model": "lstm", "fee": 10}).json()
    assert r["no_model_cost"] == 530.0
    assert r["best"]["cost"] <= 530.0 and r["best"]["recall"] >= 0.5
    assert c.get("/cost", params={"model": "nope"}).status_code == 404
