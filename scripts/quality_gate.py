"""Fail the build if the sequence models fall below their F1 floor or stop beating the per-transaction baseline.

usage: uv run scripts/quality_gate.py
"""
import json
import sys
from pathlib import Path


def check(results: dict, limits: dict) -> list[str]:
    failures = [f"{name}: test F1 {results[name]:.3f} below {floor}"
                for name, floor in limits["min_test_f1"].items() if results.get(name, 0.0) < floor]
    if limits.get("sequence_beats_xgboost"):
        best_seq = max(v for k, v in results.items() if not k.startswith("xgboost"))
        if best_seq <= results.get("xgboost_random", 0.0):
            failures.append("no sequence model beats xgboost on the random split")
    return failures


def main():
    limits = json.loads(Path("quality_gate.json").read_text())
    results = {p.stem: json.loads(p.read_text())["test"]["f1"] for p in Path("results/detection").glob("*.json")}
    failures = check(results, limits)
    for name, f1 in sorted(results.items()):
        print(f"  {name:28} F1 {f1:.3f}")
    if failures:
        print("quality gate FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("quality gate passed")


if __name__ == "__main__":
    main()
