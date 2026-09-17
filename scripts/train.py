#!/usr/bin/env python3
"""
train.py

The real, full-length training run: builds the model using the winning
hyperparameters from tune.py (if a best_hyperparameters.json exists),
falling back to config.yaml's model_defaults/optimizer_defaults for
anything the tuner didn't search over (or everything, if you skip tuning
entirely). Trains over the FULL training shards -- no --steps-per-epoch
cap like tune.py uses, since this is the run you actually want to trust.

Requires:
    pip install tensorflow keras-tuner pyyaml python-chess tqdm

Usage:
    python train.py --epochs 15
    python train.py --epochs 15 --no-tuned   # ignore tune.py's results entirely
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from build_dataset import create_dataset
from config import load_config
from model import ChessTransformerDecoder, LinearWarmup


def parse_args(cfg):
    tuner_run = cfg["tuner_run"]
    default_best_hp_path = Path(cfg["paths"]["tuner_project_dir"]) / f"{tuner_run['project_name']}_best_hyperparameters.json"

    parser = argparse.ArgumentParser(description="Run the full, non-tuning training loop.")
    parser.add_argument("--epochs", type=int, required=True,
                         help="Number of full passes over the training shards.")
    parser.add_argument("--vocab", type=Path, default=cfg["paths"]["vocab"])
    parser.add_argument("--train-pattern", default=cfg["paths"]["train_tfrecord_pattern"])
    parser.add_argument("--val-pattern", default=cfg["paths"]["val_tfrecord_pattern"])
    parser.add_argument(
        "--best-hyperparameters", type=Path, default=default_best_hp_path,
        help="JSON file written by tune.py. Used automatically if it exists; pass --no-tuned to skip it.",
    )
    parser.add_argument(
        "--no-tuned", action="store_true",
        help="Ignore --best-hyperparameters and use config.yaml's model_defaults/optimizer_defaults as-is.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=cfg["paths"]["checkpoints_dir"])
    parser.add_argument(
        "--early-stopping-patience", type=int, default=5,
        help="Full training can afford more patience than a short tuner trial. Default: 5.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help=(
            "Override the peak learning_rate that would otherwise come from "
            "--best-hyperparameters or config.yaml's optimizer_defaults. "
            "Useful when the tuner (short, few-epoch trials) picked an LR "
            "that's unstable over a full-length run -- everything else "
            "(architecture, warmup_steps, etc.) stays whatever was tuned."
        ),
    )
    parser.add_argument(
        "--lr-schedule", choices=["plateau", "cosine"], default="plateau",
        help=(
            "plateau (default): warmup then hold at peak learning_rate, "
            "only backing off -- via ReduceLROnPlateau -- once val_loss "
            "actually stops improving. Lets you keep training for as long "
            "as it's helping without committing to an epoch count up "
            "front. cosine: warmup then cosine-decay learning_rate down to "
            "learning_rate * min_lr_ratio by the end of --epochs, baked in "
            "as a fixed schedule (the original behavior) -- use this if "
            "you want the LR to have fully decayed by a specific epoch."
        ),
    )
    parser.add_argument(
        "--reduce-lr-factor", type=float, default=0.5,
        help="plateau mode only: multiply learning_rate by this once --reduce-lr-patience is exceeded. Default: 0.5.",
    )
    parser.add_argument(
        "--reduce-lr-patience", type=int, default=2,
        help="plateau mode only: epochs with no val_loss improvement before reducing learning_rate. Default: 2.",
    )
    parser.add_argument(
        "--min-lr", type=float, default=None,
        help="plateau mode only: floor for learning_rate reduction. Defaults to learning_rate * min_lr_ratio (same floor the cosine schedule would have used).",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help=(
            "Path to a .weights.h5 file to load before training (e.g. "
            "checkpoints/best.weights.h5), for continuing an earlier or "
            "interrupted run instead of starting from a random init. Only "
            "model weights are restored -- optimizer state (warmup "
            "progress, Adam moments, the ReduceLROnPlateau patience "
            "counter) is not, so training resumes with a fresh warmup."
        ),
    )
    return parser.parse_args()


def build_model_params(cfg, best_hp):
    """Merge config.yaml's model_defaults/optimizer_defaults with the
    winning hyperparameters from tune.py (if any), producing exactly the
    kwargs ChessTransformerDecoder's constructor expects, plus batch_size
    (not a constructor arg -- used for the dataset instead)."""
    params = dict(cfg["model_defaults"])
    params.update(cfg["optimizer_defaults"])

    batch_size = params.pop("batch_size")
    dff_multiplier = params.pop("dff_multiplier")

    if best_hp:
        batch_size = best_hp.get("batch_size", batch_size)
        dff_multiplier = best_hp.get("dff_multiplier", dff_multiplier)
        # Only overwrite keys that were actually part of the search space
        # (config.yaml's tuner_search_space) -- anything else (e.g.
        # layer_norm_epsilon, initializer_range, beta_1, adam_epsilon)
        # stays at its config.yaml model_defaults/optimizer_defaults value,
        # since those were deliberately held fixed during the search.
        for key in (
            "d_model", "num_layers", "num_heads", "dropout_rate", "attention_dropout_rate",
            "activation", "tie_embeddings", "learning_rate", "warmup_steps", "weight_decay",
            "beta_2", "label_smoothing", "gradient_clip_norm",
        ):
            if key in best_hp:
                params[key] = best_hp[key]

    params["dff"] = params["d_model"] * dff_multiplier
    params["top_k_metrics"] = tuple(params["top_k_metrics"])
    return params, batch_size


def main():
    cfg = load_config()
    args = parse_args(cfg)
    args.vocab = Path(args.vocab)
    args.best_hyperparameters = Path(args.best_hyperparameters)
    args.checkpoint_dir = Path(args.checkpoint_dir)

    with args.vocab.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    best_hp = None
    if not args.no_tuned and args.best_hyperparameters.exists():
        with args.best_hyperparameters.open("r", encoding="utf-8") as f:
            best_hp = json.load(f)
        print(f"Using tuned hyperparameters from: {args.best_hyperparameters}")
    else:
        print("Using config.yaml's model_defaults/optimizer_defaults (no tuned hyperparameters applied).")

    model_params, batch_size = build_model_params(cfg, best_hp)

    if args.learning_rate is not None:
        print(
            f"\nOverriding learning_rate: {model_params['learning_rate']} -> {args.learning_rate}"
        )
        model_params["learning_rate"] = args.learning_rate

    print("\nFinal hyperparameters for this run:")
    for key, value in sorted({**model_params, "batch_size": batch_size}.items()):
        print(f"  {key}: {value}")

    seq_len = cfg["data"]["seq_len"]

    train_ds = create_dataset(
        str(args.train_pattern), seq_len=seq_len, pad_id=vocab["pad_id"],
        batch_size=batch_size, shuffle=True,
    )
    val_ds = create_dataset(
        str(args.val_pattern), seq_len=seq_len, pad_id=vocab["pad_id"],
        batch_size=batch_size, shuffle=False,
    )

    # steps_per_epoch/validation_steps are needed up front for two reasons:
    # 1) the model's warmup-then-cosine-decay schedule (model.py's
    #    WarmupCosineDecay) needs the ACTUAL total step count to decay
    #    across.
    # 2) The train_ds pipeline (build_dataset.create_dataset) shuffles
    #    file order via tf.data.Dataset.list_files(shuffle=True) and reads
    #    shards with interleave(cycle_length=AUTOTUNE,
    #    num_parallel_calls=AUTOTUNE). Re-iterating that pipeline (once per
    #    epoch, as plain model.fit(train_ds, epochs=N) does under the hood)
    #    does not reliably yield the same number of batches on every pass,
    #    which is what caused the "ran out of data; interrupting training"
    #    warning -- the first epoch discovered N batches, a later epoch
    #    produced fewer, and Keras 3 treated that as the whole fit() call
    #    running dry rather than just moving on to the next epoch.
    #
    # The fix: count batches once (no model involved, cheap next to
    # training itself), then .repeat() both datasets so they never run dry
    # mid-epoch, and pass steps_per_epoch/validation_steps explicitly so
    # Keras always pulls exactly that many batches per epoch regardless of
    # where the underlying shuffled/interleaved stream happens to be.
    print("\nCounting training batches (one-time pass, no model involved yet)...")
    steps_per_epoch = sum(1 for _ in train_ds)
    total_steps = steps_per_epoch * args.epochs
    print(f"steps_per_epoch={steps_per_epoch:,}, total_steps={total_steps:,} over {args.epochs} epoch(s)")

    print("Counting validation batches (one-time pass)...")
    validation_steps = sum(1 for _ in val_ds)
    print(f"validation_steps={validation_steps:,}")

    train_ds = train_ds.repeat()
    val_ds = val_ds.repeat()

    # cosine: bake total_steps into the LR schedule so it fully decays by
    # the end of --epochs. plateau: pass None -- compile_default() then
    # compiles a settable LR, driven by LinearWarmup + ReduceLROnPlateau
    # below instead of a pre-committed schedule.
    model_total_steps = total_steps if args.lr_schedule == "cosine" else None

    model = ChessTransformerDecoder(
        vocab_size=vocab["vocab_size"], pad_id=vocab["pad_id"], eos_id=vocab["eos_id"],
        max_seq_len=seq_len, total_steps=model_total_steps, **model_params,
    )
    model.compile_default()
    model(tf.zeros((1, seq_len), dtype=tf.int32))  # builds every sub-layer via a real forward pass
    model.summary()

    if args.resume_from is not None:
        print(f"\nResuming from weights: {args.resume_from}")
        model.load_weights(str(args.resume_from))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact hyperparameters this run used -- reloading the model
    # later means reconstructing ChessTransformerDecoder with matching
    # arguments before load_weights() will work, so this file is what
    # makes that reproducible without having to remember which tune.py
    # run or config.yaml state produced these weights.
    used_hp_path = args.checkpoint_dir / "used_hyperparameters.json"
    with used_hp_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **model_params, "batch_size": batch_size, "seq_len": seq_len,
                "lr_schedule": args.lr_schedule, "total_steps": total_steps,
                "resumed_from": str(args.resume_from) if args.resume_from else None,
            },
            f, indent=2,
        )

    best_checkpoint_path = args.checkpoint_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_path), monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(args.checkpoint_dir / "training_log.csv")),
    ]

    if args.lr_schedule == "plateau":
        min_lr = args.min_lr if args.min_lr is not None else model_params["learning_rate"] * model_params["min_lr_ratio"]
        callbacks.append(LinearWarmup(peak_lr=model_params["learning_rate"], warmup_steps=model_params["warmup_steps"]))
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=args.reduce_lr_factor, patience=args.reduce_lr_patience,
                min_lr=min_lr, verbose=1,
            )
        )
        print(
            f"\nLR schedule: plateau -- warmup to {model_params['learning_rate']} over "
            f"{model_params['warmup_steps']} steps, then hold; ReduceLROnPlateau will cut LR by "
            f"{args.reduce_lr_factor}x after {args.reduce_lr_patience} epoch(s) with no val_loss "
            f"improvement, down to a floor of {min_lr}."
        )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    final_checkpoint_path = args.checkpoint_dir / "final.weights.h5"
    model.save_weights(str(final_checkpoint_path))

    print(f"\nTraining complete.")
    print(f"  Best-val-loss checkpoint: {best_checkpoint_path}")
    print(f"  Final weights (after EarlyStopping's restore_best_weights): {final_checkpoint_path}")
    print(f"  Hyperparameters used: {used_hp_path}")
    print(f"  Per-epoch log: {args.checkpoint_dir / 'training_log.csv'}")


if __name__ == "__main__":
    main()
