"""The whole pipeline as a Prefect flow: each step is a task with retries, logs and timings in the Prefect UI.

usage: uv run --group mlops flows/pipeline.py --split random
"""
import argparse
import subprocess

from prefect import flow, task


@task(retries=1, log_prints=True)
def run(*cmd: str):
    print(" ".join(cmd))
    subprocess.run(["uv", "run", *cmd], check=True)


@flow(name="fraud-pipeline", log_prints=True)
def pipeline(split: str = "random", models: tuple[str, ...] = ("mlp", "lstm", "transformer")):
    run("scripts/prepare.py")
    run("--group", "mlops", "scripts/build_features.py")
    run("scripts/train_xgboost.py", "--split", split)
    for m in models:
        run("scripts/train_sequence.py", m, "--split", split)
    for m in (*models, "xgboost"):
        run("scripts/cost.py", m, "--split", split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "chrono"], default="random")
    pipeline(parser.parse_args().split)
