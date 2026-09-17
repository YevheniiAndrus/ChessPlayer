#!/usr/bin/env python3
"""
tune.py

Runs a Keras Tuner search over ChessHyperModel (hypermodel.py) and reports
the best hyperparameters found, using the train/validation TFRecord shards
produced by build_dataset.py. All defaults below come from config.yaml
(data/paths/tuner_run sections) -- CLI flags exist only to override the
config for a one-off run, not to hold hyperparameter values themselves.

Requires:
    pip install tensorflow keras-tuner pyyaml

Usage (everything from config.yaml):
    python tune.py

Usage (overriding a couple of things for one run):
    python tune.py --tuner random --max-trials 2 --epochs-per-trial 1

Which search strategy to pick (--tuner, default from config.yaml's
tuner_run.tuner):
  - bayesian: models the objective as it goes and proposes the next
    trial's hyperparameters based on what's worked so far. The most
    sample-efficient option, which matters here since every trial means
    training a transformer -- good default when trials are expensive.
  - hyperband: starts many trials with a small epoch budget and only lets
    the promising ones keep training longer. Good when you can afford lots
    of short trials and want to explore a wide space quickly; uses
    --max-epochs instead of --epochs-per-trial.
  - random: samples uniformly at random. Simplest, and a reasonable
    baseline to compare the other two against, but wastes trials on
    obviously-bad regions of the search space that bayesian would learn to
    avoid.

Remember: train/validation shards should be a whole-shard split (see the
note in build_dataset.py) so no game leaks between the two.
"""

import argparse
import json
from pathlib import Path

import keras_tuner as kt
import tensorflow as tf

from config import load_config
from hypermodel import ChessHyperModel


class _DeleteCheckpointAfterTrial:
    """Mixin: deletes each trial's checkpoint right after the trial finishes.

    keras_tuner.engine.tuner.Tuner.run_trial() -- inherited by
    BayesianOptimization, Hyperband, and RandomSearch alike -- always
    appends its own SaveBestEpoch callback that checkpoints the trial's
    best epoch to <project_dir>/<project_name>/trial_<id>/checkpoint.weights.h5.
    This is automatic, built-in Keras Tuner behavior (not something this
    project's code added) and it is never cleaned up on its own -- every
    trial leaves a full model checkpoint behind (tens to hundreds of MB
    each here, since tuner_search_space in config.yaml varies d_model up
    to 512 and num_layers up to 8), which is exactly what filled the disk
    partway through a 40-trial search.

    Those checkpoints exist so tuner.get_best_models() can reload a
    specific trial's trained weights later -- but this script never calls
    that. It only ever reads tuner.get_best_hyperparameters() (the
    hyperparameter VALUES, written to *_best_hyperparameters.json below),
    and train.py trains a brand new model from scratch with those values.
    So per-trial checkpoints are pure disk usage with no benefit here.

    Deleting them doesn't affect resuming an interrupted search
    (overwrite=False): Keras Tuner tracks which trials are done and their
    objective values via each trial's trial.json / the oracle's own state
    files, not via these checkpoints.
    """

    def run_trial(self, trial, *args, **kwargs):
        result = super().run_trial(trial, *args, **kwargs)
        Path(self._get_checkpoint_fname(trial.trial_id)).unlink(missing_ok=True)
        return result


class BayesianOptimization(_DeleteCheckpointAfterTrial, kt.BayesianOptimization):
    pass


class Hyperband(_DeleteCheckpointAfterTrial, kt.Hyperband):
    pass


class RandomSearch(_DeleteCheckpointAfterTrial, kt.RandomSearch):
    pass


def parse_args(cfg):
    tuner_run = cfg["tuner_run"]
    parser = argparse.ArgumentParser(description="Search for good hyperparameters with Keras Tuner.")
    parser.add_argument("--vocab", type=Path, default=cfg["paths"]["vocab"],
                         help="Path to vocab.json from build_vocab.py.")
    parser.add_argument("--train-pattern", default=cfg["paths"]["train_tfrecord_pattern"],
                         help="Glob pattern for training TFRecord shards.")
    parser.add_argument("--val-pattern", default=cfg["paths"]["val_tfrecord_pattern"],
                         help="Glob pattern for validation TFRecord shards.")
    parser.add_argument("--tuner", choices=["bayesian", "hyperband", "random"], default=tuner_run["tuner"])
    parser.add_argument("--max-trials", type=int, default=tuner_run["max_trials"],
                         help="Number of trials for bayesian/random search. Ignored for hyperband.")
    parser.add_argument("--epochs-per-trial", type=int, default=tuner_run["epochs_per_trial"],
                         help="Epochs per trial for bayesian/random search. Ignored for hyperband.")
    parser.add_argument("--max-epochs", type=int, default=tuner_run["max_epochs"],
                         help="Max epochs any single hyperband trial can reach. Ignored otherwise.")
    parser.add_argument("--early-stopping-patience", type=int, default=tuner_run["early_stopping_patience"])
    parser.add_argument("--project-dir", type=Path, default=cfg["paths"]["tuner_project_dir"],
                         help="Where Keras Tuner stores trial results/checkpoints.")
    parser.add_argument("--project-name", default=tuner_run["project_name"])
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help="Cap each trial's epoch at this many batches instead of a full pass over the "
             "training shards. Strongly recommended for a first search: hyperparameter search "
             "only needs a fast, consistent relative signal between configurations, not a full "
             "pass over the whole dataset every trial -- capping this is what keeps a 30-trial "
             "search to hours instead of days. Omit for a full-pass search once you've budgeted "
             "the time for it.",
    )
    parser.add_argument(
        "--validation-steps", type=int, default=None,
        help="Same idea as --steps-per-epoch, but for the validation pass each epoch.",
    )
    return parser.parse_args()


def build_tuner(args, hypermodel, objective):
    common_kwargs = dict(
        hypermodel=hypermodel,
        objective=objective,
        directory=str(args.project_dir),
        project_name=args.project_name,
        overwrite=False,  # resume a previous search in the same project dir if one exists
    )
    if args.tuner == "hyperband":
        return Hyperband(max_epochs=args.max_epochs, factor=3, **common_kwargs)
    if args.tuner == "bayesian":
        return BayesianOptimization(max_trials=args.max_trials, **common_kwargs)
    return RandomSearch(max_trials=args.max_trials, **common_kwargs)


def main():
    cfg = load_config()
    args = parse_args(cfg)

    with Path(args.vocab).open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    hypermodel = ChessHyperModel(
        cfg=cfg,
        vocab_size=vocab["vocab_size"],
        pad_id=vocab["pad_id"],
        eos_id=vocab["eos_id"],
    )
    # CLI overrides for the patterns take effect even though ChessHyperModel
    # already pulled its defaults from cfg["paths"] in __init__.
    hypermodel.train_tfrecord_pattern = str(args.train_pattern)
    hypermodel.val_tfrecord_pattern = str(args.val_pattern)

    objective = kt.Objective("val_loss", direction="min")
    tuner = build_tuner(args, hypermodel, objective)
    tuner.search_space_summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True
    )

    search_kwargs = {"callbacks": [early_stopping]}
    if args.tuner != "hyperband":
        search_kwargs["epochs"] = args.epochs_per_trial
    if args.steps_per_epoch is not None:
        search_kwargs["steps_per_epoch"] = args.steps_per_epoch
    if args.validation_steps is not None:
        search_kwargs["validation_steps"] = args.validation_steps

    tuner.search(**search_kwargs)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters found:")
    for key, value in best_hp.values.items():
        print(f"  {key}: {value}")

    out_path = Path(args.project_dir) / f"{args.project_name}_best_hyperparameters.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best_hp.values, f, indent=2)
    print(f"\nSaved to: {out_path}")
    print("Feed these into train.py to run the real, full-length training run.")


if __name__ == "__main__":
    main()
