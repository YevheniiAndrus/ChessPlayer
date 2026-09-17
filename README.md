# ChessPlayer

A decoder-only transformer (Keras/TensorFlow) that predicts the best next
chess move from a sequence of previous moves, trained on Lichess games
between strong players (rated above a configurable ELO threshold).

Moves are tokenized as UCI strings (e.g. `e2e4`, `g1f3`, `e7e8q`), which
keeps the vocabulary small (~4,210 tokens) and context-free -- no board
state is fed to the model; it has to infer everything about the position
purely from the move history via attention.

## Pipeline overview

```
raw Lichess PGN dump
  -> filter_pgn_by_elo.py   (keep only games where both players > min ELO)
  -> build_vocab.py         (enumerate the fixed UCI move vocabulary)
  -> build_dataset.py       (PGN -> TFRecord shards of training windows)
  -> tune.py                (Keras Tuner search for good hyperparameters)
  -> train.py                (the real, full-length training run)
```

`run_pipeline.sh` runs the last two stages (tune -> train) back to back;
the three data-prep stages before that are one-off and only need
rerunning if you change the source data, the ELO threshold, or `seq_len`.

## 1. Requirements

- **macOS on Apple Silicon** (M1/M2/M3), for GPU acceleration via
  `tensorflow-metal`. The pipeline still runs on other platforms, just on
  CPU only -- `requirements.txt` is written so the Apple-Silicon-only
  packages are skipped automatically elsewhere (see below).
- **Python 3.9** in a virtualenv at the project root (`.venv/`). Newer
  Python versions may work but haven't been tested against this exact
  `tensorflow`/`tensorflow-metal` pairing.
- A Lichess games database dump (see step 3).

## 2. Set up the environment

From the project root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs `tensorflow` capped below 2.20 and
`tensorflow-metal==1.2.0` *only* on macOS arm64 (via PEP 508 environment
markers) -- this exact pairing matters: `tensorflow-metal`'s last release
(1.2.0, Jan 2025) does not load on `tensorflow>=2.20`. On any other
platform you just get plain CPU `tensorflow`.

Verify TensorFlow can actually see and use the GPU (not just that it
imports without errors):

```bash
python scripts/check_gpu.py
```

This prints the detected devices, a CPU-vs-GPU matmul benchmark, and a
CPU-vs-GPU benchmark of one real training step of the model. Look for
Metal's own init log line (something like `Metal device set to: Apple
M1 Pro`) and a GPU speedup clearly greater than 1x before trusting a long
training run to actually be using the GPU.

## 3. Get the data

Download a games database dump from [database.lichess.org](https://database.lichess.org/)
(the standard monthly PGN dumps work well; each page also publishes the
exact game count for that month, useful for `--total-games` below) and
place it anywhere on disk -- it doesn't need to live inside this project.

## 4. Build the dataset (one-off, unless the source data / ELO threshold / seq_len changes)

All three steps below read their defaults from `config.yaml`, so you only
need to pass flags that override those defaults (e.g. `--min-elo`).

```bash
# 1) Keep only games where BOTH players are rated above the threshold
#    (config.yaml: data.min_elo, default 2000).
python scripts/filter_pgn_by_elo.py --input /path/to/lichess_db_standard_rated_YYYY-MM.pgn \
    --total-games <count from the Lichess download page>

# 2) Enumerate the fixed UCI move vocabulary (independent of the dataset --
#    only needs to run once, ever, unless you change the tokenization scheme).
python scripts/build_vocab.py

# 3) Convert the filtered PGN into TFRecord shards of (input_ids, labels)
#    training windows (config.yaml: data.seq_len, data.games_per_shard).
python scripts/build_dataset.py --total-games <kept-game count printed by step 1>
```

By default this reads/writes:

| File | Default path |
|---|---|
| Filtered PGN | `data/games_filtered.pgn` |
| Vocabulary | `data/vocab.json` |
| TFRecord shards | `data/tfrecords/shard-NNNNN.tfrecord` |

`config.yaml`'s `paths.train_tfrecord_pattern` / `paths.val_tfrecord_pattern`
hold out the *last* shard as validation (whole-shard, so no game's moves
leak across the train/val split) -- if you change `games_per_shard`
enough to end up with very few shards, adjust these glob patterns so the
split still makes sense.

## 5. Tune + train

### Quick start: the combined script

```bash
./run_pipeline.sh \
  --tune-args "--tuner bayesian --max-trials 40 --epochs-per-trial 5 --steps-per-epoch 500 --validation-steps 100" \
  --train-args "--epochs 40"
```

`run_pipeline.sh` activates `.venv`, runs `tune.py` with `--tune-args`,
then runs `train.py` with `--train-args`, stopping immediately if either
stage fails (so training never runs against a missing/stale
hyperparameters file). Run `./run_pipeline.sh --help` for the full usage
note, including `--skip-tune` (reuse an existing tuned-hyperparameters
file and just (re)run training).

**This takes a long time** -- each tuner trial and each training epoch is
a real pass over the model, easily tens of minutes to a few hours per
epoch depending on model size and hardware. Run it somewhere that
survives your terminal closing:

```bash
nohup ./run_pipeline.sh --train-args "--epochs 40" > pipeline.log 2>&1 &
```

(or `screen`/`tmux`). Tail `pipeline.log` to watch progress.

### Stage 1: `tune.py` -- hyperparameter search

Runs a Keras Tuner search (bayesian / hyperband / random, see the
script's own docstring for the tradeoffs) over `config.yaml`'s
`tuner_search_space`, and writes the winning combination to
`tuner_runs/<project_name>_best_hyperparameters.json`.

```bash
python scripts/tune.py --tuner bayesian --max-trials 40 --epochs-per-trial 5 \
    --steps-per-epoch 500 --validation-steps 100
```

`--steps-per-epoch`/`--validation-steps` cap each trial to a fast partial
pass over the data -- strongly recommended for the search itself, since
you only need a fast, *relative* signal between configurations, not a
full epoch every trial. Omit them for a full-pass search once you've
budgeted the time for it.

The search is resumable: rerunning the same command (same
`--project-dir`/`--project-name`) picks up where it left off rather than
starting over, since `tune.py` never overwrites a previous run
(`overwrite=False`). Each trial's own checkpoint is deleted automatically
right after that trial finishes (this project never reloads a trial's
trained weights, only the winning hyperparameter *values*), so a long
search no longer fills up disk the way it originally did.

### Stage 2: `train.py` -- the real training run

Builds a fresh model from the winning hyperparameters (or
`config.yaml`'s `model_defaults`/`optimizer_defaults` if you skip tuning
via `--no-tuned`) and trains it over the full training shards.

```bash
python scripts/train.py --epochs 40
```

Useful flags:

| Flag | What it does |
|---|---|
| `--lr-schedule plateau` (default) | Warmup then hold at peak LR, only backing off via `ReduceLROnPlateau` once `val_loss` genuinely stalls -- lets you not have to guess the right epoch count up front. |
| `--lr-schedule cosine` | Warmup then cosine-decay to a fixed floor by the end of `--epochs` -- use this if you want the LR fully decayed by a specific epoch. |
| `--resume-from checkpoints/best.weights.h5` | Continue from a previous checkpoint's weights instead of a random init (e.g. after an interrupted run). Only weights are restored, not optimizer state, so warmup restarts briefly. |
| `--learning-rate <value>` | Override just the peak learning rate from the tuned/config hyperparameters, keeping everything else. Useful if the tuner (short trials) picked an LR that's unstable over a full-length run. |
| `--no-tuned` | Ignore any `best_hyperparameters.json` and use `config.yaml`'s `model_defaults`/`optimizer_defaults` as-is. |
| `--early-stopping-patience N` (default 5) | Epochs with no `val_loss` improvement before stopping early and restoring the best weights. |

Outputs, all under `checkpoints/`:

| File | What it is |
|---|---|
| `best.weights.h5` | Weights from the best `val_loss` epoch so far (updated every time it improves). |
| `final.weights.h5` | Weights after training ends (== best.weights.h5, since `EarlyStopping(restore_best_weights=True)`). |
| `used_hyperparameters.json` | The exact hyperparameters this run used -- needed to reconstruct `ChessTransformerDecoder` with matching arguments before `load_weights()` will work. |
| `training_log.csv` | Per-epoch loss/perplexity/accuracy history. |

## Monitoring while it runs

Metrics to watch, per epoch, in `training_log.csv` (or the live progress
bar): `val_loss` should decrease (or at least not get worse) every few
epochs; `val_top1_acc`/`val_top5_acc` should climb. `perplexity`/
`val_perplexity` are currently miscalibrated relative to `loss`/`val_loss`
(a known padding-averaging quirk in the custom loss) -- trust `loss` and
the accuracy metrics over `perplexity` for now.

## Disk space

Training checkpoints (`checkpoints/*.weights.h5`) and, historically, per
-trial tuner checkpoints are the main disk consumers here (tens to
hundreds of MB each, scaling with model size). Keep an eye on free disk
space during a long run, especially if `config.yaml`'s `tuner_search_space`
samples large architectures (`d_model` up to 512, `num_layers` up to 8).

## Configuration

Every hyperparameter and path lives in `config.yaml` at the project root
(loaded by `scripts/config.py`, with all paths resolved relative to
`config.yaml`'s own location, so scripts behave the same regardless of
which directory you run them from). CLI flags on any script override the
config for a one-off run without editing the file. See the comments in
`config.yaml` itself for what each section controls.

## Project layout

```
config.yaml               central configuration (paths, hyperparameters, tuner search space)
requirements.txt          Python dependencies (platform-conditional for tensorflow-metal)
run_pipeline.sh           runs tune.py then train.py in one command
scripts/
  config.py                loads config.yaml
  filter_pgn_by_elo.py     PGN -> ELO-filtered PGN
  build_vocab.py           enumerates the UCI move vocabulary -> vocab.json
  build_dataset.py         filtered PGN -> TFRecord shards; also exposes create_dataset()
                            (the tf.data pipeline used by tune.py/train.py)
  metrics.py               Perplexity metric, build_metrics()
  model.py                 ChessTransformerDecoder, WarmupCosineDecay, LinearWarmup, MoveLoss
  hypermodel.py             ChessHyperModel (keras_tuner.HyperModel wrapper)
  tune.py                   Keras Tuner search CLI
  train.py                  full training run CLI
  check_gpu.py              verifies tensorflow-metal is actually accelerating on the GPU
data/                      filtered PGN, vocab.json, TFRecord shards (gitignored -- regenerate, don't commit)
checkpoints/               training outputs (weights, logs, used_hyperparameters.json)
tuner_runs/                Keras Tuner's own search state and best_hyperparameters.json
```

## What's not built yet

An inference script that loads a trained checkpoint and actually
generates/ranks moves for a given position isn't part of this pipeline
yet -- `train.py` produces `checkpoints/best.weights.h5` and
`checkpoints/used_hyperparameters.json` (everything needed to reconstruct
the exact model and load those weights), but turning that into "give me
the best move for this position" is the next step once a training run
you're happy with has finished.
