import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("quality_gate", Path(__file__).parents[1] / "scripts" / "quality_gate.py")


def test_gate():
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    limits = {"min_test_f1": {"lstm_random": 0.7}, "sequence_beats_xgboost": True}
    assert gate.check({"lstm_random": 0.8, "xgboost_random": 0.6}, limits) == []
    assert len(gate.check({"lstm_random": 0.5, "xgboost_random": 0.6}, limits)) == 2
