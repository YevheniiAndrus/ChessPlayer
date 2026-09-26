#!/usr/bin/env python3
"""
find_lr.py

Runs a "learning rate range test" (Smith, 2015 -- arxiv.org/abs/1506.01186)
to find a good peak learning rate BEFORE committing to a real training or
tuning run, instead of trusting tune.py's own learning_rate search to find
it blind. Two separate tune.py searches on this project both picked a
learning_rate within ~4% of the search space's ceiling (0.0065, then
0.0096), and both looked great for a couple of epochs before destabilizing
-- see the comment on tuner_search_space.learning_rate in config.yaml for
why: Keras Tuner scores a trial by the BEST value its objective ever
reaches during that trial, not its sustained value, so an unstable LR that
dips low early before diverging scores identically to one that reaches the
same low point and stays there. This script sidesteps that failure mode
entirely by not treating "lowest loss reached" as the answer at all.

How it works: starting from a tiny learning rate (--min-lr) and ending at
a large one (--max-lr), this ramps the LR *exponentially* upward, batch by
batch, over --num-steps total training steps on a single freshly
-initialized model -- short enough to run in minutes, not the hours a real
training run takes. No validation pass, no checkpointing, no held-out
split -- this is a diagnostic sweep over the TRAINING loss only, and its
weights are discarded when it finishes. It automatically stops early if
the (smoothed) loss explodes past --diverge-factor times its best-seen
value, or turns non-finite, since there's nothing more to learn (and some
risk of wasted compute/NaN propagation) from continuing once the model has
clearly blown up.

The result is a loss-vs-learning-rate curve, written to a CSV and (if
matplotlib is installed) a PNG plot under --output-dir. Early on, loss
decreases as the LR climbs into a useful range; past some point it starts
increasing again as the LR gets too large for the model to handle stably
-- often sharply, sometimes after a brief further dip (exactly the kind of
transient dip that fools tune.py's best-epoch trial scoring). The
suggested learning rate this script prints is deliberately NOT the LR at
the lowest point of that curve -- training AT that LR is usually already
borderline unstable, since you're sitting right at the edge where things
start going wrong. Instead it's the LR where the smoothed loss is still
falling FASTEST (steepest descent), searched only over the region before
the minimum -- a standard, more conservative choice that leaves headroom
before the instability edge. Sanity-check the automatic pick against the
printed curve/plot yourself rather than trusting it blindly; this is meant
to narrow down a good optimizer_defaults.learning_rate and a sensible
tuner_search_space.learning_rate ceiling, not to replace your own
judgement.

Requires:
    pip install tensorflow pyyaml
    pip install matplotlib   # optional -- only needed for the saved plot

Usage:
    # Sweep using config.yaml's model_defaults/optimizer_defaults architecture:
    python scripts/find_lr.py

    # Sweep using a specific architecture (e.g. what tune.py last found, or
    # what train.py last actually trained with):
    python scripts/find_lr.py --hyperparameters tuner_runs/chess_transformer_best_hyperparameters.json
    python scripts/find_lr.py --hyperparameters checkpoints/used_hyperparameters.json

    # Narrower/wider sweep, more/fewer steps:
    python scripts/find_lr.py --min-lr 1e-6 --max-lr 1e-1 --num-steps 500
"""

import argparse
import csv
import json
import math
from pathlib import Path

import tensorflow as tf

from build_dataset import create_dataset
from config import load_config
from model import from_yaml_config

# Keys that can appear in a hyperparameters JSON (tune.py's
# *_best_hyperparameters.json, or train.py's used_hyperparameters.json)
# that are NOT ChessTransformerDecoder constructor arguments. Mirrors
# predict.py's _NON_CONSTRUCTOR_KEYS denylist. "dff_multiplier" is handled
# separately below (translated into "dff") rather than just dropped,
# since tune.py's file has it and the model needs "dff" itself.
_NON_CONSTRUCTOR_KEYS = {"batch_size", "seq_len", "lr_schedule", "total_steps", "resumed_from"}


def parse_args(cfg):
    parser = argparse.ArgumentParser(
        description="Sweep the learning rate from small to large over a short run to find a good peak LR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vocab", type=Path, default=cfg["paths"]["vocab"])
    parser.add_argument("--train-pattern", default=cfg["paths"]["train_tfrecord_pattern"])
    parser.add_argument(
        "--hyperparameters", type=Path, default=None,
        help="Optional hyperparameters JSON (tune.py's *_best_hyperparameters.json, or "
             "train.py's checkpoints/used_hyperparameters.json) to sweep with that exact "
             "architecture. Omit to use config.yaml's model_defaults/optimizer_defaults.",
    )
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Override the batch size. Default: from --hyperparameters if given, "
                              "else config.yaml's optimizer_defaults.batch_size. LR range test "
                              "results are somewhat batch-size-dependent, so matching whatever "
                              "size you actually intend to train with is worth doing.")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Sweep start. Default: 1e-7.")
    parser.add_argument("--max-lr", type=float, default=1.0, help="Sweep end (before any early stop). Default: 1.0.")
    parser.add_argument("--num-steps", type=int, default=400,
                         help="Total batches to sweep over (short on purpose -- minutes, not hours). Default: 400.")
    parser.add_argument("--smoothing", type=float, default=0.98,
                         help="EMA smoothing factor for the loss curve (fastai-style, bias-corrected). Default: 0.98.")
    parser.add_argument("--diverge-factor", type=float, default=4.0,
                         help="Stop early once smoothed loss exceeds this multiple of its best-seen value. Default: 4.0.")
    parser.add_argument("--output-dir", type=Path, default=Path("lr_range_test"),
                         help="Where to write results.csv and (if matplotlib is installed) results.png. "
                              "Separate from checkpoints/ -- this script never touches real training outputs.")
    return parser.parse_args()


def load_overrides(hp_path, cfg):
    """Load a hyperparameters JSON into a ChessTransformerDecoder-constructor
    -safe overrides dict, plus whatever batch_size it specified (or None)."""
    if hp_path is None:
        return {}, None
    with Path(hp_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    overrides = {k: v for k, v in raw.items() if k not in _NON_CONSTRUCTOR_KEYS and k != "dff_multiplier"}
    if "dff_multiplier" in raw and "dff" not in raw:
        # tune.py's best_hyperparameters.json format -- "dff" isn't
        # computed yet, unlike train.py's used_hyperparameters.json.
        d_model = raw.get("d_model", cfg["model_defaults"]["d_model"])
        overrides["dff"] = d_model * raw["dff_multiplier"]

    return overrides, raw.get("batch_size")


class LRRangeTest(tf.keras.callbacks.Callback):
    """Ramps the optimizer's learning rate exponentially from min_lr to
    max_lr over num_steps batches, recording (learning_rate, raw loss,
    EMA-smoothed loss) at every step. Stops training early if the
    smoothed loss turns non-finite or explodes past diverge_factor times
    its best-seen value -- there's nothing more to learn (and some risk
    of wasted compute/NaN propagation) from continuing once the sweep has
    clearly blown up.

    Requires the model to have been compiled with a plain, settable
    learning rate (ChessTransformerDecoder.compile_default() with
    total_steps=None -- see model.py's LinearWarmup for the same
    requirement and why).
    """

    def __init__(self, min_lr, max_lr, num_steps, smoothing=0.98, diverge_factor=4.0):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smoothing = smoothing
        self.diverge_factor = diverge_factor
        self.step = 0
        self._avg_loss = None
        self._current_lr = min_lr
        self.best_smoothed_loss = float("inf")
        self.history = []  # list of (learning_rate, raw_loss, smoothed_loss)

    def on_train_batch_begin(self, batch, logs=None):
        frac = self.step / max(1, self.num_steps - 1)
        self._current_lr = self.min_lr * (self.max_lr / self.min_lr) ** frac
        self.model.optimizer.learning_rate = self._current_lr

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        self.step += 1

        if loss is None or not math.isfinite(loss):
            print(f"\nStep {self.step}: non-finite loss ({loss}) at lr={self._current_lr:.3e} -- stopping early.")
            self.model.stop_training = True
            return

        if self._avg_loss is None:
            # Seed the recursion as if avg_loss started at 0 (avg_0 = 0),
            # matching what the bias-correction denominator below assumes:
            # avg_1 = smoothing * 0 + (1 - smoothing) * loss_1. Seeding
            # with the raw loss instead (the original bug here) makes
            # avg_1 come out 1/(1 - smoothing) too large -- with the
            # default smoothing=0.98 that's 50x -- and the inflated value
            # decays back to the true loss only over the next ~100-200
            # steps, which fakes a steep early "descent" right after the
            # skip_fraction cutoff and hijacks suggest_learning_rate().
            self._avg_loss = (1 - self.smoothing) * loss
        else:
            self._avg_loss = self.smoothing * self._avg_loss + (1 - self.smoothing) * loss
        # Bias-corrected the same way Adam corrects its own moment
        # estimates, so the first few steps aren't artificially dragged
        # toward 0 before the EMA has "warmed up".
        smoothed = self._avg_loss / (1 - self.smoothing ** self.step)

        self.history.append((self._current_lr, loss, smoothed))
        self.best_smoothed_loss = min(self.best_smoothed_loss, smoothed)

        if smoothed > self.diverge_factor * self.best_smoothed_loss:
            print(
                f"\nStep {self.step}: smoothed loss {smoothed:.4f} exceeds "
                f"{self.diverge_factor}x its best-seen value ({self.best_smoothed_loss:.4f}) "
                f"at lr={self._current_lr:.3e} -- sweep has clearly diverged, stopping early."
            )
            self.model.stop_training = True
            return

        if self.step >= self.num_steps:
            self.model.stop_training = True


def suggest_learning_rate(history, skip_fraction=0.1):
    """Pick the LR where the smoothed loss is falling fastest (steepest
    descent per decade of LR), searched only over the region from the
    (skip_fraction-trimmed) start up to the minimum smoothed loss --
    never past it, since slope naturally turns positive there as the
    sweep starts diverging, and training AT the loss-minimizing LR itself
    is usually already borderline unstable. Returns (lr, index) or
    (None, None) if there isn't enough history to make a call."""
    n = len(history)
    skip = int(n * skip_fraction)
    if n - skip < 3:
        return None, None

    smoothed = [h[2] for h in history]
    min_idx = min(range(skip, n), key=lambda i: smoothed[i])
    if min_idx - skip < 2:
        return None, None

    best_slope = 0.0
    best_idx = None
    for i in range(skip + 1, min_idx + 1):
        lr_prev, lr_cur = history[i - 1][0], history[i][0]
        d_log_lr = math.log10(lr_cur) - math.log10(lr_prev)
        if d_log_lr <= 0:
            continue
        slope = (smoothed[i] - smoothed[i - 1]) / d_log_lr
        if slope < best_slope:
            best_slope = slope
            best_idx = i

    if best_idx is None:
        return None, None
    return history[best_idx][0], best_idx


def write_csv(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "learning_rate", "loss", "smoothed_loss"])
        for step, (lr, loss, smoothed) in enumerate(history, start=1):
            writer.writerow([step, lr, loss, smoothed])


def write_plot(history, suggested_lr, path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed -- just save a PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n(matplotlib not installed -- skipping {path}; `pip install matplotlib` to get the plot too.)")
        return

    lrs = [h[0] for h in history]
    smoothed = [h[2] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, smoothed, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed training loss")
    ax.set_title("LR range test")
    if suggested_lr is not None:
        ax.axvline(suggested_lr, color="red", linestyle="--", linewidth=1,
                    label=f"suggested lr = {suggested_lr:.2e}")
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {path}")


def main():
    cfg = load_config()
    args = parse_args(cfg)

    if not Path(args.vocab).exists():
        raise SystemExit(f"Vocab file not found: {args.vocab} (run build_vocab.py first)")

    overrides, hp_batch_size = load_overrides(args.hyperparameters, cfg)
    batch_size = args.batch_size or hp_batch_size or cfg["optimizer_defaults"]["batch_size"]
    seq_len = cfg["data"]["seq_len"]

    with Path(args.vocab).open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    # total_steps=None is required -- it's what makes compile_default()
    # build a plain, settable-LR optimizer instead of a LearningRateSchedule
    # (see LinearWarmup's docstring in model.py). This callback needs to
    # be able to set model.optimizer.learning_rate every batch.
    model = from_yaml_config(cfg, str(args.vocab), total_steps=None, **overrides)
    model.compile_default()
    model(tf.zeros((1, seq_len), dtype=tf.int32))  # dummy forward pass -- builds every sub-layer

    print(f"Architecture: d_model={model.d_model}, num_layers={model.num_layers}, "
          f"num_heads={model.num_heads}, dff={model.dff}, batch_size={batch_size}")
    print(f"Sweeping learning_rate from {args.min_lr:.1e} to {args.max_lr:.1e} over {args.num_steps} steps...\n")

    train_ds = create_dataset(
        str(args.train_pattern), seq_len=seq_len, pad_id=vocab["pad_id"],
        batch_size=batch_size, shuffle=True,
    ).repeat()  # a short sweep can easily need more steps than one epoch has

    range_test = LRRangeTest(
        min_lr=args.min_lr, max_lr=args.max_lr, num_steps=args.num_steps,
        smoothing=args.smoothing, diverge_factor=args.diverge_factor,
    )
    model.fit(train_ds, epochs=1, steps_per_epoch=args.num_steps, callbacks=[range_test], verbose=1)

    if len(range_test.history) < 10:
        raise SystemExit(
            f"\nOnly {len(range_test.history)} steps completed before stopping -- too few to suggest "
            f"anything. The model likely diverged almost immediately; try a lower --max-lr."
        )

    write_csv(range_test.history, args.output_dir / "results.csv")
    suggested_lr, idx = suggest_learning_rate(range_test.history)
    write_plot(range_test.history, suggested_lr, args.output_dir / "results.png")

    print(f"\n{len(range_test.history)} steps completed.")
    print(f"Raw results: {args.output_dir / 'results.csv'}")
    if suggested_lr is None:
        print(
            "\nCouldn't confidently identify a steepest-descent point (not enough of a clean "
            "downward trend before the minimum) -- inspect results.csv/results.png by hand."
        )
    else:
        print(f"\nSuggested peak learning_rate: {suggested_lr:.2e}")
        print(
            "This is the point of steepest descent, not the lowest-loss point -- deliberately "
            "conservative, leaving headroom before the instability edge. Check results.png (or "
            "plot results.csv yourself) before trusting it: look for a clear, sustained downward "
            "slope up to roughly this point, followed by loss flattening out or rising. Use this "
            "as optimizer_defaults.learning_rate and/or to set a tighter "
            "tuner_search_space.learning_rate ceiling in config.yaml, rather than letting the "
            "tuner search blind up to a high, potentially-unstable value."
        )


if __name__ == "__main__":
    main()
