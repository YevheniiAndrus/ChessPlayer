#!/usr/bin/env bash
#
# run_pipeline.sh
#
# Runs the two-stage pipeline end to end: search for good hyperparameters
# with tune.py, then train the real, full-length model on the winning
# hyperparameters with train.py. Both stages can take many hours, so this
# is meant to be left running unattended -- see the note at the bottom
# about nohup/screen/tmux if you're running it over SSH or don't want it
# tied to your terminal window staying open.
#
# Usage:
#   ./run_pipeline.sh --train-args "--epochs 40"
#   ./run_pipeline.sh --tune-args "--tuner bayesian --max-trials 40 --steps-per-epoch 500 --validation-steps 100" --train-args "--epochs 40 --lr-schedule plateau"
#   ./run_pipeline.sh --skip-tune --train-args "--epochs 40 --resume-from checkpoints/best.weights.h5"
#
# Anything you'd normally pass to tune.py / train.py directly can be
# passed through via --tune-args / --train-args (each as ONE quoted
# string). Omit either and that stage falls back to its own
# config.yaml-driven defaults -- except train.py's --epochs, which it
# requires explicitly, so --train-args needs at least that.
#
# --skip-tune skips stage 1 entirely and goes straight to train.py,
# reusing whatever <project_name>_best_hyperparameters.json already
# exists from an earlier tune.py run -- useful if you've already tuned
# and just want to (re)run training, e.g. after a --resume-from restart.
#
# If you override --project-dir/--project-name in --tune-args, pass the
# matching --best-hyperparameters (or --project-dir/--project-name) in
# --train-args too, so train.py finds the right file -- by default both
# scripts derive the same path from config.yaml, so this only matters if
# you deviate from the defaults.

set -euo pipefail

# Resolve paths relative to this script's own location, so it works
# regardless of the directory you invoke it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TUNE_ARGS=""
TRAIN_ARGS=""
SKIP_TUNE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tune-args)
      TUNE_ARGS="$2"
      shift 2
      ;;
    --train-args)
      TRAIN_ARGS="$2"
      shift 2
      ;;
    --skip-tune)
      SKIP_TUNE=1
      shift
      ;;
    -h|--help)
      sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1 (use --tune-args, --train-args, --skip-tune, or --help)" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d .venv ]]; then
  echo "No .venv found in $SCRIPT_DIR -- expected the project's virtualenv here." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ "$SKIP_TUNE" -eq 0 ]]; then
  echo "=================================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1/2: tune.py ${TUNE_ARGS}"
  echo "=================================================================="
  # shellcheck disable=SC2086
  python3 scripts/tune.py ${TUNE_ARGS}
else
  echo "Skipping tune.py (--skip-tune) -- reusing whatever best_hyperparameters.json already exists."
fi

echo ""
echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2/2: train.py ${TRAIN_ARGS}"
echo "=================================================================="
# shellcheck disable=SC2086
python3 scripts/train.py ${TRAIN_ARGS}

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete."
