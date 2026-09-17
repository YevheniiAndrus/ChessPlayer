#!/usr/bin/env python3
"""
build_dataset.py

Two jobs live in this module:

1. A CLI (see main()) that converts a filtered PGN file (output of
   filter_pgn_by_elo.py) into TFRecord shards of fixed-length
   (input_ids, labels) windows.

2. create_dataset(), a function the training script imports directly to
   turn those TFRecord shards into a tf.data.Dataset ready for
   model.fit() -- no need to re-run the CLI or touch PGN/python-chess at
   training time, since that heavy parsing already happened once here.

Requires:
    pip install python-chess tensorflow tqdm

CLI usage:
    python build_dataset.py \
        --input filtered.pgn \
        --vocab ../data/vocab.json \
        --output-dir ../data/tfrecords \
        --seq-len 40

Library usage (from a training script):
    from build_dataset import create_dataset

    train_ds = create_dataset(
        "../data/tfrecords/shard-0000[0-8]*.tfrecord",
        seq_len=40,
        pad_id=0,
        batch_size=64,
    )

How the windows are built
--------------------------
Each game becomes a sequence of UCI move ids, e.g.
    [id(e2e4), id(e7e5), id(g1f3), ..., <EOS>]

That sequence is chopped into non-overlapping windows of length
(seq_len + 1), padding the final window with <PAD> if the game doesn't
divide evenly. Each window is then split into:
    input_ids = window[:-1]   # length seq_len
    labels    = window[1:]    # length seq_len

so that position t of `labels` is the token the model should predict having
seen input_ids[0:t+1] -- the standard causal language-model training setup.
Padding positions are stored as-is; create_dataset() builds the loss/metric
mask at read time from `labels != pad_id`, so nothing extra needs to be
stored in the TFRecords.

Note on train/validation splitting: because --games-per-shard writes whole
games into a shard before moving to the next one, every shard only ever
contains complete, distinct games. That means you can safely hold out one
or more whole shard files as a validation set (e.g. train on
"shard-0000[0-8]*.tfrecord" and validate on "shard-00009*.tfrecord")
without any game leaking across the split -- windows from the same game
never end up in both sets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import chess.pgn
import tensorflow as tf
from tqdm import tqdm

from config import load_config


def load_vocab(vocab_path: Path):
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["move_to_id"], vocab["pad_id"], vocab["eos_id"]


def game_to_ids(game, move_to_id):
    """Convert a parsed python-chess game into a list of move ids, or None
    if it contains a move outside the vocabulary (shouldn't happen given
    how build_vocab.py enumerates every possible move, but we skip
    defensively rather than crash a multi-hour preprocessing run)."""
    ids = []
    for move in game.mainline_moves():
        move_id = move_to_id.get(move.uci())
        if move_id is None:
            return None
        ids.append(move_id)
    return ids


def make_windows(ids, seq_len, pad_id):
    """Chop a token-id sequence into non-overlapping (input_ids, labels) windows."""
    window_len = seq_len + 1
    windows = []
    for start in range(0, len(ids), seq_len):
        chunk = ids[start:start + window_len]
        if len(chunk) < window_len:
            chunk = chunk + [pad_id] * (window_len - len(chunk))
        windows.append((chunk[:-1], chunk[1:]))
    return windows


def make_example(input_ids, labels):
    feature = {
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ---------------------------------------------------------------------------
# Training-time API
# ---------------------------------------------------------------------------

def _feature_description(seq_len: int):
    return {
        "input_ids": tf.io.FixedLenFeature([seq_len], tf.int64),
        "labels": tf.io.FixedLenFeature([seq_len], tf.int64),
    }


def create_dataset(
    tfrecord_pattern: Union[str, Path],
    seq_len: int,
    pad_id: int,
    batch_size: int = 64,
    shuffle_buffer: int = 50_000,
    shuffle: bool = True,
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset of (input_ids, labels, sample_weight) batches
    from TFRecord shards written by this module's CLI.

    input_ids:     int32 tensor, shape (batch, seq_len) -- the previous
                    moves (as token ids) the model conditions on.
    labels:        int32 tensor, shape (batch, seq_len) -- input_ids
                    shifted one position into the future; labels[t] is the
                    move the model should predict having seen
                    input_ids[0:t+1].
    sample_weight: float32 tensor, shape (batch, seq_len) -- 0.0 wherever
                    `labels` is a padding token, 1.0 elsewhere. Keras
                    applies this to both the loss AND any compiled
                    metrics automatically, so padded positions never
                    count toward training signal or reported numbers.
    """
    file_pattern = str(tfrecord_pattern)
    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    ds = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    feature_description = _feature_description(seq_len)

    def _parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        input_ids = tf.cast(parsed["input_ids"], tf.int32)
        labels = tf.cast(parsed["labels"], tf.int32)
        sample_weight = tf.cast(tf.not_equal(labels, pad_id), tf.float32)
        return input_ids, labels, sample_weight

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# CLI: PGN -> TFRecord shards
# ---------------------------------------------------------------------------

def parse_args(cfg):
    parser = argparse.ArgumentParser(description="Convert a filtered PGN file into TFRecord training windows.")
    parser.add_argument("--input", "-i", type=Path, default=cfg["paths"]["filtered_pgn"],
                         help="Default: paths.filtered_pgn in config.yaml.")
    parser.add_argument("--vocab", type=Path, default=cfg["paths"]["vocab"],
                         help="Default: paths.vocab in config.yaml.")
    parser.add_argument("--output-dir", "-o", type=Path, default=cfg["paths"]["tfrecords_dir"],
                         help="Default: paths.tfrecords_dir in config.yaml.")
    parser.add_argument(
        "--seq-len", type=int, default=cfg["data"]["seq_len"],
        help="Number of previous moves (context length) per training example. "
             "Default: data.seq_len in config.yaml.",
    )
    parser.add_argument(
        "--games-per-shard", type=int, default=cfg["data"]["games_per_shard"],
        help="How many source games to write per TFRecord shard file. "
             "Default: data.games_per_shard in config.yaml.",
    )
    parser.add_argument(
        "--total-games", type=int, default=None,
        help="Known/estimated total game count, for the tqdm progress bar only.",
    )
    return parser.parse_args()


def main():
    cfg = load_config()
    args = parse_args(cfg)
    args.input = Path(args.input)
    args.vocab = Path(args.vocab)
    args.output_dir = Path(args.output_dir)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.vocab.exists():
        print(f"Vocab file not found: {args.vocab} (run build_vocab.py first)", file=sys.stderr)
        sys.exit(1)

    move_to_id, pad_id, eos_id = load_vocab(args.vocab)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    games_in_shard = 0
    writer = None

    def open_new_shard():
        nonlocal writer, shard_idx, games_in_shard
        if writer is not None:
            writer.close()
        shard_path = args.output_dir / f"shard-{shard_idx:05d}.tfrecord"
        writer = tf.io.TFRecordWriter(str(shard_path))
        shard_idx += 1
        games_in_shard = 0

    open_new_shard()

    total_games = 0
    kept_games = 0
    skipped_games = 0
    total_windows = 0

    with args.input.open("r", encoding="utf-8", errors="replace") as pgn_in, \
         tqdm(total=args.total_games, unit="game", desc="Building dataset", dynamic_ncols=True) as bar:

        while True:
            game = chess.pgn.read_game(pgn_in)
            if game is None:
                break

            total_games += 1
            bar.update(1)

            ids = game_to_ids(game, move_to_id)
            if not ids:
                skipped_games += 1
                bar.set_postfix(kept=kept_games, windows=total_windows, refresh=False)
                continue

            ids.append(eos_id)
            for input_ids, labels in make_windows(ids, args.seq_len, pad_id):
                writer.write(make_example(input_ids, labels).SerializeToString())
                total_windows += 1

            kept_games += 1
            games_in_shard += 1
            if games_in_shard >= args.games_per_shard:
                open_new_shard()

            bar.set_postfix(kept=kept_games, windows=total_windows, refresh=False)

    writer.close()

    print(f"Done. Read {total_games:,} games, converted {kept_games:,}, "
          f"skipped {skipped_games:,} (moves outside the vocabulary).")
    print(f"Wrote {total_windows:,} training windows across {shard_idx:,} shard(s) to: {args.output_dir}")


if __name__ == "__main__":
    main()
