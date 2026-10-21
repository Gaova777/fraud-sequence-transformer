#!/usr/bin/env bash
# Shortcuts for this project (make isn't installed on the machine it was built on).
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-help}" in
  test)      uv run --group mlops pytest -q && uv run ruff check src scripts tests flows ;;
  pipeline)  uv run --group mlops flows/pipeline.py --split "${2:-random}" ;;
  prefect)   uv run --group mlops prefect server start ;;
  features)  uv run --group mlops scripts/build_features.py ;;
  gate)      uv run scripts/quality_gate.py ;;
  api)       uv run --group mlops uvicorn fraud_sequence_transformer.entrypoints.api:app --port 8000 ;;
  web)       cd web && PATH=$HOME/.local/opt/node/bin:$PATH npm run dev ;;
  *)         echo "usage: ./run.sh test|pipeline [random|chrono]|prefect|features|gate|api|web" ;;
esac
