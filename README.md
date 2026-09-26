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
  -> find_lr.py             (optional: fast LR range test, to sanity-check
                              tuner_search_space's learning_rate ceiling)
  -> tune.py                (Keras Tuner search for good hyperparameters)
  -> plot_tuner_trials.py   (optional: inspect real per-epoch trajectories
                              before trusting the tuner's raw scores)
  -> train.py                (the real, full-length training run)
  -> predict.py              (load a trained checkpoint, predict the next
                              move -- one-shot or --interactive)
```

`run_pipeline.sh` runs the tune -> train stages back to back; the three
data-prep stages before that are one-off and only need rerunning if you
change the source data, the ELO threshold, or `seq_len`. `find_lr.py` and
`plot_tuner_trials.py` are optional diagnostic tools -- see "Before tuning"
and "Picking a trial" under step 5 below -- neither feeds a value into the
next stage automatically; you look at their output and update
`config.yaml`, or pick a trial's hyperparameters, yourself. `predict.py` is
the standalone inference step you run afterwards, as many times (or as one
persistent `--interactive` session) as you like, against whatever
checkpoint `train.py` last produced.

## 1. Requirements

`requirements.txt` uses PEP 508 environment markers to pick the right
TensorFlow/GPU setup for whatever machine you run it on -- no separate
requirements file needed per platform:

- **macOS on Apple Silicon** (M1/M2/M3): GPU acceleration via
  `tensorflow-metal`, with TensorFlow capped below 2.20 (see the
  comments in `requirements.txt` for why) and **Python 3.9** in a
  virtualenv at the project root (`.venv/`) -- this specific Python
  version is a constraint of this pairing (`tensorflow-metal`'s last
  release, plus `keras-hub`'s last Python-3.9-compatible release), not
  a project-wide requirement.
- **Linux with an NVIDIA GPU**: GPU acceleration via the
  `tensorflow[and-cuda]` extra (pulls in matching CUDA/cuDNN runtime
  libraries as pip packages -- you still need the NVIDIA driver itself
  installed at the OS level; `nvidia-smi` should show your GPU before
  you trust the rest). No Python 3.9 pin here -- use whatever recent
  Python 3.x your environment provides, and `keras-hub` resolves to its
  latest release rather than the older pin macOS needs.
- **Anything else** (Windows native, Intel Mac, no GPU): plain CPU
  TensorFlow. For GPU on Windows, run under WSL2 instead (it presents
  as Linux and gets the CUDA branch above) -- native Windows GPU support
  was dropped after TensorFlow 2.10.
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

### Before tuning: `find_lr.py` -- a fast learning-rate sanity check

`tune.py` searches `learning_rate` like any other hyperparameter, but it
has a specific blind spot worth knowing about before you trust its pick:
Keras Tuner scores a trial by the *best* value its objective (`val_loss`)
reaches at any point during the trial, not its final or sustained value.
An unstable learning rate that dips low early and then diverges scores
exactly as well as one that reaches the same low point and stays there --
so the tuner has no way to tell "genuinely good" apart from "transiently
good." That's exactly what happened on this project, twice: two separate
searches each picked a `learning_rate` within ~4% of the search space's
ceiling (0.0065, then 0.0096), and both looked great for a couple of
epochs before destabilizing over a longer run.

`find_lr.py` sidesteps this by not using the tuner's trial-scoring at all.
It runs a classic "LR range test" (Smith, 2015): starting from a tiny
learning rate and ramping exponentially upward, batch by batch, over a
single short run on one freshly-initialized model (minutes, not hours; no
validation, no checkpointing, weights discarded when it finishes). The
output is a loss-vs-learning-rate curve; the suggested LR it prints is
where the *smoothed* loss is falling fastest, restricted to the region
before the loss bottoms out -- a conservative pick that leaves headroom
before the point where training visibly starts destabilizing.

```bash
python scripts/find_lr.py
# or, sweeping with a specific architecture instead of config.yaml's defaults:
python scripts/find_lr.py --hyperparameters tuner_runs/chess_transformer_best_hyperparameters.json
```

**What this found on this project:** the range test's steepest-descent
region spans roughly `2e-3` to `2e-1`, with a single suggested peak at
`9.22e-3` -- almost exactly the `0.0096` that had already broken sustained
training. That's not a contradiction, it's the same blind spot from a
different angle: the range test only measures whether one gradient step at
a given LR is still productive *in the moment* (real divergence there
doesn't start until ~`0.3`-`0.4`); it says nothing about whether that LR
stays stable over thousands of sustained steps with warmup/decay and
accumulated optimizer state, which is the failure mode that actually bit
this project both times. So the result was used to *corroborate* -- not
raise -- the existing decision to lower `tuner_search_space.learning_rate`'s
ceiling to `3.0e-3` in `config.yaml` (comfortably inside the range test's
productive region, well clear of the value with a real history of blowing
up). See the comment above `tuner_search_space.learning_rate` in
`config.yaml` for the full reasoning.

### Stage 1: `tune.py` -- hyperparameter search

Runs a Keras Tuner search (bayesian / hyperband / random, see the
script's own docstring for the tradeoffs) over `config.yaml`'s
`tuner_search_space`, and writes the winning combination to
`tuner_runs/<project_name>_best_hyperparameters.json` -- **but only if the
search actually finishes**; see the worked example below for what happens
otherwise.

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

**Trial scores alone aren't enough to pick a winner from.** Because of the
same best-value-not-sustained-value scoring described above, `tune.py`
also logs each trial's full per-epoch history (train + val loss,
perplexity, accuracy, etc.) to `<trial_dir>/epoch_history.csv` as it
trains, on top of the single collapsed score Keras Tuner keeps in
`trial.json` by default. Use `plot_tuner_trials.py` (below) to look at the
actual trajectories before trusting the top raw score -- especially with
short, few-epoch trials like the `--epochs-per-trial 5` above, which
aren't long enough to reveal instability that only shows up over a much
longer full training run.

**A long search can still get killed by the OS running out of RAM**
(shows up as `zsh: killed`, not a Python error) partway through -- e.g.
"killed after 17 trials." This isn't a bug in any particular trial:
TensorFlow (and `tensorflow-metal`) never returns memory to the OS within
a single process, so a many-trial search's memory footprint is a
high-water mark that only grows, trial over trial (`hypermodel.py`'s
`clear_session()`/`gc.collect()` between trials only releases Keras' own
bookkeeping, not the allocator's already-claimed memory), until whichever
trial's incremental need finally exceeds available RAM. Resuming (as
above) is the actual fix, since a fresh process starts with a clean
memory footprint -- `run_pipeline.sh` does this automatically, restarting
`tune.py` up to 20 times if it dies. Running `tune.py` directly, wrap it
the same way:

```bash
until python scripts/tune.py --tuner bayesian --max-trials 40 \
    --epochs-per-trial 5 --steps-per-epoch 500 --validation-steps 100; do
  echo "tune.py died -- resuming from where it left off in 5s..."
  sleep 5
done
```

### Picking a trial: `plot_tuner_trials.py`

`trial.json` only ever records ONE observation per metric per trial --
Keras Tuner's own `Tuner.on_epoch_end()` is a no-op by default ("Intermediate
results are not passed to the Oracle"), so out of the box there's no way to
tell a trial whose `val_loss` fell smoothly from one whose `val_loss`
dipped early and then flattened or got worse -- exactly the ambiguity that
let earlier searches pick a `learning_rate` that only looked good for a
couple of epochs. `tune.py` now overrides that hook (the
`_LogEpochHistory` mixin) so every trial writes its real per-epoch history
to `epoch_history.csv` as it trains -- only trials run since that change
have one; older trials fall back to reporting just their single final
score.

```bash
python scripts/plot_tuner_trials.py --top-n 8
```

This ranks the completed trials by score, prints a plain-language verdict
per trial ("monotonic improvement across all 5 recorded epochs" vs.
"improved for 2 epoch(s), then got worse at epoch 3"), and writes
`top_trials_<metric>.png` (train loss dashed, val loss solid, if
matplotlib is installed) plus `top_trials_summary.csv` under
`tuner_runs/<project_name>_trial_plots/`.

**Worked example -- how this project actually picked its hyperparameters.**
A bayesian search that appeared to finish with "27 tries" turned out, on
inspection of `oracle.json`, to still have trial 26 `RUNNING` -- the
process had been interrupted mid-trial, so
`chess_transformer_best_hyperparameters.json` was never written. Ranking
the 26 trials that had genuinely completed by raw score put trial 03
marginally ahead of trial 22 -- but `plot_tuner_trials.py`'s per-epoch
trajectories told a different story: trial 03's `val_loss` dipped early
and then flattened, the same "looks great for a couple of epochs" pattern
that had already burned this project's learning-rate search twice, while
trial 22 improved monotonically across every recorded epoch. **Trial 22
was picked over trial 03 despite its very slightly worse raw score**,
because its trajectory was independently verifiable as stable rather than
just momentarily good.

Its hyperparameters were copied out of
`tuner_runs/chess_transformer/trial_22/trial.json` into a new file,
`tuner_runs/chess_transformer_trial22_hyperparameters.json`, with two
manual adjustments made on top before training -- the tuner's own 5-epoch
trials are too short to reward choices that only pay off over a much
longer run:

| Hyperparameter | Tuner's value | Used value | Why |
|---|---|---|---|
| `attention_dropout_rate` | `0.0` | `0.1` | A short trial won't show the benefit of regularization; a full multi-epoch run can overfit without it. |
| `warmup_steps` | `400` | `2000` | Same reasoning -- a short trial doesn't need much warmup to look good, but a full run has room to benefit from ramping up over more steps. |

This hand-edited file -- not `tuner_runs/chess_transformer_best_hyperparameters.json`,
which was never written -- is what the real training run in this project
actually uses, via `train.py --best-hyperparameters
tuner_runs/chess_transformer_trial22_hyperparameters.json`. The general
lesson generalizes past this one run: treat `tune.py`'s output as a
*starting point* for judgement, not an answer to apply blindly, especially
when trial budgets are too short to fully trust a single score.

### Stage 2: `train.py` -- the real training run

Builds a fresh model from the winning hyperparameters (or
`config.yaml`'s `model_defaults`/`optimizer_defaults` if you skip tuning
via `--no-tuned`) and trains it over the full training shards.

```bash
python scripts/train.py --epochs 40
# or, training from a specific hand-picked trial's hyperparameters
# (see "Picking a trial" above) instead of an auto-discovered best-hyperparameters file:
python scripts/train.py --epochs 40 \
    --best-hyperparameters tuner_runs/chess_transformer_trial22_hyperparameters.json
```

Useful flags:

| Flag | What it does |
|---|---|
| `--best-hyperparameters PATH` | Hyperparameters JSON to train with. Defaults to `tuner_runs/<project_name>_best_hyperparameters.json`, which only exists if a `tune.py` search actually finished -- point this at a specific trial's file (e.g. one you hand-picked per "Picking a trial" above) otherwise. |
| `--lr-schedule plateau` (default) | Warmup then hold at peak LR, only backing off via `ReduceLROnPlateau` once `val_loss` genuinely stalls -- lets you not have to guess the right epoch count up front. |
| `--lr-schedule cosine` | Warmup then cosine-decay to a fixed floor by the end of `--epochs` -- use this if you want the LR fully decayed by a specific epoch. |
| `--intra-epoch-safety` | `plateau` mode only. A second, faster safety net on top of `ReduceLROnPlateau`, which only ever looks at `val_loss` at epoch end -- on a run where one epoch takes hours, a too-aggressive learning rate can burn most of an epoch (or several, during `ReduceLROnPlateau`'s own patience window) before anything reacts. This instead watches smoothed *training* loss every batch and halves `learning_rate` immediately on a sustained rise. Tunable via `--intra-epoch-smoothing`/`--intra-epoch-threshold`/`--intra-epoch-patience-checks`/`--intra-epoch-cooldown-steps`/`--intra-epoch-factor` (see `train.py --help` for defaults). |
| `--resume-from checkpoints/best.weights.h5` | Continue from a previous checkpoint's weights instead of a random init (e.g. after an interrupted run). Only weights are restored, not optimizer state, so warmup restarts briefly. |
| `--learning-rate <value>` | Override just the peak learning rate from the tuned/config hyperparameters, keeping everything else. Useful if the tuner (short trials) picked an LR that's unstable over a full-length run. |
| `--no-tuned` | Ignore any `best_hyperparameters.json`/`--best-hyperparameters` file and use `config.yaml`'s `model_defaults`/`optimizer_defaults` as-is. |
| `--early-stopping-patience N` (default 5) | Epochs with no `val_loss` improvement before stopping early and restoring the best weights. |

Outputs, all under `checkpoints/`:

| File | What it is |
|---|---|
| `best.weights.h5` | Weights from the best `val_loss` epoch so far (updated every time it improves). |
| `final.weights.h5` | Weights after training ends (== best.weights.h5, since `EarlyStopping(restore_best_weights=True)`). |
| `used_hyperparameters.json` | The exact hyperparameters this run used -- needed to reconstruct `ChessTransformerDecoder` with matching arguments before `load_weights()` will work. |
| `training_log.csv` | Per-epoch loss/perplexity/accuracy history. |

## 6. Run inference (`predict.py`)

Once you have `checkpoints/best.weights.h5` and
`checkpoints/used_hyperparameters.json` from a training run you're happy
with, `predict.py` loads them and predicts the next move for a given
sequence of previous moves -- either once per invocation, or in a
persistent `--interactive` session.

```bash
# Moves as plain text -- SAN ("Nf3", "O-O", "exd5", ...) and/or UCI
# ("g1f3", "e1g1", "e7d5", ...) are both accepted, move-by-move,
# auto-detected; PGN move numbers ("1.", "1...") are stripped automatically.
python scripts/predict.py --moves "e4 e5 Nf3 Nc6 Bb5 a6"
python scripts/predict.py --moves "1. e4 e5 2. Nf3 Nc6 3. Bb5"

# Or read the mainline moves out of a PGN file instead:
python scripts/predict.py --pgn game_in_progress.pgn

# Show more/fewer candidate moves, or see the model's raw preference
# without restricting it to legal moves:
python scripts/predict.py --moves "e4 e5 Nf3 Nc6" --top-k 10
python scripts/predict.py --moves "e4 e5 Nf3 Nc6" --allow-illegal

# Interactive mode: one process stays running -- it predicts, waits for
# you to type the move actually played, appends it to the sequence, and
# predicts again, instead of reloading the model from scratch per move.
# --moves/--pgn become optional starting context instead of the whole game.
python scripts/predict.py --interactive
python scripts/predict.py --interactive --moves "e4 e5 Nf3 Nc6"
```

The input moves are replayed on a real `chess.Board()` -- the same
handling `build_dataset.py` uses while preparing training data -- which
both resolves SAN/UCI/castling/promotions correctly and gives us the
current position, used to mask the model's output down to only
currently-legal moves before picking a candidate (since move-sequence
-only training doesn't hard-constrain the model to legal play). Pass
`--allow-illegal` to inspect the raw, unmasked distribution instead --
useful for sanity-checking the model itself, not for actually choosing a
move.

Since training windows were always exactly `seq_len` moves (`config.yaml`:
`data.seq_len`, default 40) starting from position 0, the model has no
notion of context beyond that many moves back. Past `seq_len` moves into
a game, `predict.py` feeds it only the most recent `seq_len` moves --
still the best available input given how it was trained, but a bit more
of an extrapolation than early-game predictions, since the model never
specifically saw a window positioned that way during training.

**Why `--interactive` doesn't call `model.predict()`.** Inference goes
through `make_predict_fn()`, a `tf.function`-wrapped direct call to the
model, rather than `model.predict()`. `build_input_window()` always hands
back a *fixed* `(1, seq_len)` shape -- padded or truncated, never grown --
so, unlike naive autoregressive text generation (where the sequence grows
every step and would force a recompile on every call), there's no
retracing risk here at all: the first call, a warm-up at startup, pays a
one-time graph-compilation cost, and every call after that -- including
every move of an `--interactive` session -- reuses the same compiled
graph. That gets the same speedup `.predict()` would from its own
compilation, without `.predict()`'s per-call Dataset-wrapping and
callback overhead, which exists to support batched iteration and buys
nothing for what's always a single `(1, seq_len)` example here.

Useful flags:

| Flag | What it does |
|---|---|
| `--interactive` | Run as a persistent loop: predict, read the move actually played, append it, predict again -- instead of one prediction per invocation. `--moves`/`--pgn` become optional starting context; type `quit` to exit. |
| `--top-k N` (default 5) | Number of candidate next moves to show. |
| `--allow-illegal` | Skip the legal-move mask; show the model's raw top-k over the whole vocabulary. |
| `--checkpoint PATH` | Weights file to load. Default: `checkpoints/best.weights.h5`. |
| `--hyperparameters PATH` | `used_hyperparameters.json` to reconstruct the model architecture from. Default: `checkpoints/used_hyperparameters.json`. |
| `--seq-len N` | Must match the checkpoint's training `seq_len`. Default: `data.seq_len` in `config.yaml`. |

Output is the top predicted move (UCI, SAN, and probability) plus a
ranked candidate list, printed after every move in `--interactive` mode.

## Monitoring while it runs

Metrics to watch, per epoch, in `training_log.csv` (or the live progress
bar): `val_loss` should decrease (or at least not get worse) every few
epochs; `val_top1_acc`/`val_top5_acc` should climb; `perplexity`/
`val_perplexity` (== `exp(loss)`) should track `loss`/`val_loss` closely
and trend down alongside it. All four metrics are computed as
`weighted_metrics` in `model.py`'s `compile_default()`, so -- like the
loss itself -- they correctly exclude padding positions from a game's
final, shorter-than-`seq_len` window; if `perplexity` ever swings wildly
epoch to epoch while `loss` stays smooth, that's a sign this masking
regressed, not that training itself is unstable.

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
                            (the tf.data pipeline used by find_lr.py/tune.py/train.py)
  metrics.py               Perplexity metric, build_metrics()
  model.py                 ChessTransformerDecoder, WarmupCosineDecay, LinearWarmup, MoveLoss
  hypermodel.py             ChessHyperModel (keras_tuner.HyperModel wrapper)
  find_lr.py                optional LR range test -- sanity-checks the learning_rate ceiling before tuning
  tune.py                   Keras Tuner search CLI; also logs each trial's real per-epoch history
  plot_tuner_trials.py      ranks/plots tune.py's top trials by per-epoch trajectory, not just raw score
  train.py                  full training run CLI
  predict.py                loads a checkpoint and predicts the next move, one-shot or --interactive
  check_gpu.py              verifies tensorflow-metal is actually accelerating on the GPU
data/                      filtered PGN, vocab.json, TFRecord shards (gitignored -- regenerate, don't commit)
lr_range_test/             find_lr.py's output (results.csv, results.png)
checkpoints/               training outputs (weights, logs, used_hyperparameters.json)
tuner_runs/                Keras Tuner's own search state; *_best_hyperparameters.json if a search
                            actually finished; any hand-picked trial hyperparameters copied out of a
                            trial_*/trial.json yourself (e.g. chess_transformer_trial22_hyperparameters.json,
                            see "Picking a trial" above)
```
