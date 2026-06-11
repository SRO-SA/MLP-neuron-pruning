#!/usr/bin/env bash
# Run the full pruning experiment with the default config.
# Usage: bash scripts/run_qwen05b.sh [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
python run_experiment.py --config configs/default.yaml "$@"
