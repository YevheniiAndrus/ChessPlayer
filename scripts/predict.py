#!/usr/bin/env python3
"""
predict.py

Loads the trained model (checkpoints/best.weights.h5 by default) and
predicts the next move given a sequence of moves already played.

Requires:
    pip install tensorflow python-chess pyyaml
(see requirements.txt in the project root)

Usage:
    # Moves as plain text -- SAN ("Nf3", "O-O", "exd5", "e8=Q+", ...) and/or
    # UCI ("g1f3", "e1g1", "e7d5", "e7e8q", ...) are both accepted, move-by
    # -move, auto-detected; PGN move numbers ("1.", "1...") are stripped if
    # present so you can paste movetext directly.
    python scripts/predict.py --moves "e4 e5 Nf3 Nc6 Bb5 a6"
    python scripts/predict.py --moves "1. e4 e5 2. Nf3 Nc6 3. Bb5"

    # Or read the mainline moves out of a PGN file instead:
    python scripts/predict.py --pgn game_in_progress.pgn

    # Show more/fewer candidate moves, or see the model's raw preference
    # without restricting it to legal moves:
    python scripts/predict.py --moves "e4 e5 Nf3 Nc6" --top-k 10
    python scripts/predict.py --moves "e4 e5 Nf3 Nc6" --allow-illegal

    # Interactive mode: one process stays running, predicts, waits for you
    # to type the move actually played, appends it, and predicts again --
    # instead of re-running the script (and reloading the model) per move.
    # --moves/--pgn become optional starting context instead of the whole
    # game. See make_predict_fn()'s docstring for why this is also the mode
    # that benefits from a compiled prediction function.
    python scripts/predict.py --interactive
    python scripts/predict.py --interactive --moves "e4 e5 Nf3 Nc6"

How the prediction is made
----------------------------
Moves are parsed against a real chess.Board() (so SAN is disambiguated,
castling/promotions/checks are all handled the same way build_dataset.py's
game_to_ids() handles them while preparing training data), which also
gives us the current position for two things: knowing which of the
model's vocab entries are actually legal right now, and turning the
model's chosen move back into human-readable SAN.

The model was trained on non-overlapping windows of exactly
--seq-len (config.yaml: data.seq_len) moves, so it has no notion of
"context beyond that many moves back" -- each training window's
positions start counting from 0 regardless of where that window fell in
the source game. This script feeds the model the most recent
min(len(moves), seq_len) moves, right-padded with <PAD> the same way
build_dataset.py pads a game's final (partial) window -- there's no
alternative that's more "correct" given how it was trained, but note that
once more than seq_len moves have been played, the model is only ever
looking at a tail window it never specifically saw positioned that way
during training, so predictions that deep into a game are a bit more of
an extrapolation than early-game ones.

By default the model's output is masked down to only positions
corresponding to a currently-legal move before picking the top candidate
-- language-model training alone doesn't hard-constrain the model to
legal moves, so without this it will sometimes prefer an illegal one,
especially for less-common positions. Pass --allow-illegal to see the raw,
unmasked distribution instead (useful for sanity-checking the model
itself, not for actually playing).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import chess
import numpy as np
import tensorflow as tf

from config import load_config
from model import from_vocab_file

# Keys in checkpoints/used_hyperparameters.json (written by train.py) that
# are NOT ChessTransformerDecoder constructor arguments -- everything else
# in that file is passed straight through to from_vocab_file(). Keeping
# this as a denylist (rather than an allowlist of architecture keys) means
# a new *model* hyperparameter train.py starts recording automatically
# works here too, with no changes needed in this script.
_NON_CONSTRUCTOR_KEYS = {"batch_size", "seq_len", "lr_schedule", "total_steps", "resumed_from"}

_MOVE_NUMBER_RE = re.compile(r"^\d+\.+$")          # a lone "1." / "12..." token
_MOVE_NUMBER_PREFIX_RE = re.compile(r"^\d+\.+")    # a "1.e4" / "12...Nf3" token


def parse_args(cfg):
    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    parser = argparse.ArgumentParser(
        description="Predict the next chess move from a sequence of previous moves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    moves_group = parser.add_mutually_exclusive_group(required=False)
    moves_group.add_argument(
        "--moves",
        help="The moves played so far, as one space-separated string (SAN and/or UCI, "
             "move numbers like '1.'/'1...' are stripped automatically).",
    )
    moves_group.add_argument(
        "--pgn", type=Path,
        help="Path to a PGN file; the mainline moves of its first game are used as input.",
    )
    parser.add_argument("--vocab", type=Path, default=cfg["paths"]["vocab"],
                         help="Default: paths.vocab in config.yaml.")
    parser.add_argument(
        "--checkpoint", type=Path, default=checkpoints_dir / "best.weights.h5",
        help="Weights file written by train.py. Default: checkpoints/best.weights.h5.",
    )
    parser.add_argument(
        "--hyperparameters", type=Path, default=checkpoints_dir / "used_hyperparameters.json",
        help="Written by train.py alongside its checkpoints -- needed to reconstruct the "
             "model with matching architecture before --checkpoint's weights can be loaded. "
             "Default: checkpoints/used_hyperparameters.json.",
    )
    parser.add_argument("--seq-len", type=int, default=cfg["data"]["seq_len"],
                         help="Must match the seq_len the checkpoint was trained with. Default: data.seq_len in config.yaml.")
    parser.add_argument("--top-k", type=int, default=5,
                         help="Number of candidate next moves to show. Default: 5.")
    parser.add_argument(
        "--allow-illegal", action="store_true",
        help="Don't mask the model's output down to legal moves -- show its raw top-k "
             "preference over the whole vocabulary instead. For inspecting the model, not "
             "for actually picking a move to play.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help=(
            "Run as a persistent loop instead of a single one-shot prediction: predicts, "
            "waits for you to type the move that was actually played, appends it, and "
            "predicts again -- repeatedly, in the same process. --moves/--pgn become "
            "optional starting context instead of the whole game."
        ),
    )
    args = parser.parse_args()
    if not args.interactive and args.moves is None and args.pgn is None:
        parser.error("one of --moves or --pgn is required unless --interactive is set")
    return args


def clean_move_tokens(text):
    """Split a plain or PGN-movetext-style string into move tokens,
    dropping/stripping move-number annotations ("1.", "1...", "1.e4")."""
    tokens = []
    for raw in text.split():
        if _MOVE_NUMBER_RE.match(raw):
            continue  # a lone "1." / "1..." token -- not a move
        raw = _MOVE_NUMBER_PREFIX_RE.sub("", raw)  # "1.e4" -> "e4"
        if raw:
            tokens.append(raw)
    return tokens


def moves_from_pgn(pgn_path):
    import chess.pgn
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        game = chess.pgn.read_game(f)
    if game is None:
        print(f"No game found in {pgn_path}", file=sys.stderr)
        sys.exit(1)
    return [move.uci() for move in game.mainline_moves()]


def replay_moves(move_tokens, move_to_id):
    """Replay move_tokens (SAN and/or UCI strings, or already-UCI strings
    from moves_from_pgn) on a fresh board, returning (ids, board) where
    `board` reflects the position after the last move and `ids` is the
    corresponding list of vocabulary token ids -- the exact same encoding
    build_dataset.py's game_to_ids() produces, so the model sees input in
    the same form it was trained on."""
    board = chess.Board()
    ids = []
    for i, token in enumerate(move_tokens, start=1):
        move = None
        try:
            move = board.parse_san(token)
        except ValueError:
            try:
                move = board.parse_uci(token)
            except ValueError:
                pass
        if move is None:
            print(
                f"Move {i} ('{token}') isn't a legal SAN or UCI move in the position "
                f"reached after the previous moves.",
                file=sys.stderr,
            )
            sys.exit(1)

        uci = move.uci()
        move_id = move_to_id.get(uci)
        if move_id is None:
            # Shouldn't happen -- build_vocab.py enumerates every possible
            # (from_square, to_square) pair plus every promotion variant,
            # so every legal move's UCI is in the vocabulary. Defensive
            # only, same as build_dataset.py's game_to_ids().
            print(f"Move {i} ('{token}' -> {uci}) has no vocabulary entry in {{vocab}}.", file=sys.stderr)
            sys.exit(1)

        ids.append(move_id)
        board.push(move)

    return ids, board


def build_input_window(ids, seq_len, pad_id):
    """Return (input_ids, next_move_position): the fixed-length, right
    -padded window fed to the model, and the sequence position whose
    output distribution is the prediction for the move after the last one
    played. Mirrors build_dataset.py's make_windows() padding convention
    (pad at the end, never the start)."""
    if len(ids) >= seq_len:
        if len(ids) > seq_len:
            print(
                f"Note: {len(ids)} moves given, but the model only ever sees {seq_len} at a "
                f"time -- using the most recent {seq_len}. See this script's module "
                f"docstring for why that's an approximation past {seq_len} moves.",
                file=sys.stderr,
            )
        recent = ids[-seq_len:]
        next_move_position = seq_len - 1
    else:
        recent = ids + [pad_id] * (seq_len - len(ids))
        next_move_position = len(ids) - 1

    return recent, next_move_position


def make_predict_fn(model):
    """Wrap the model's forward pass in a tf.function so repeated calls
    with the same input shape only pay tracing/graph-compilation cost
    ONCE, then reuse the compiled graph on every later call instead of
    running eagerly (unoptimized, op-by-op) every time.

    This only pays off because build_input_window() always hands back a
    FIXED shape -- exactly (1, seq_len), padded or truncated as needed --
    no matter how long the actual game gets. That's different from naive
    autoregressive text generation, where the sequence grows by one token
    every step and would trigger a retrace on every single call, erasing
    the benefit (and, worse, spending MORE time retracing than eager
    execution would have taken). Because this project always predicts one
    fixed-length window at a time rather than growing the input, --interactive
    mode gets the full benefit: the first call traces, every call after
    that in the same session runs the already-compiled graph.

    Deliberately not model.predict(): that also compiles under the hood
    and would give the same one-time-tracing benefit here, but it also
    re-enters Keras's data-adapter/Dataset-wrapping and callback machinery
    on every single call -- overhead built for iterating many batches,
    paid again each time for what's always a single (1, seq_len) example
    here. A directly tf.function-wrapped call skips straight from "already
    traced" to "run the compiled graph", with none of that per-call cost.
    """
    @tf.function
    def predict_fn(input_tensor):
        return model(input_tensor, training=False)
    return predict_fn


def predict_next_move(predict_fn, input_ids, next_move_position, board, id_to_move, top_k, allow_illegal):
    input_tensor = tf.constant([input_ids], dtype=tf.int32)  # shape (1, seq_len) -- always the same, see make_predict_fn
    logits = predict_fn(input_tensor)[0, next_move_position, :].numpy()  # (vocab_size,)

    if not allow_illegal:
        legal_uci = {move.uci() for move in board.legal_moves}
        if not legal_uci:
            return None  # checkmate or stalemate -- no move to predict
        legal_ids = [move_id for move_id, uci in id_to_move.items() if uci in legal_uci]
        mask = np.zeros(logits.shape, dtype=bool)
        mask[legal_ids] = True
        logits = np.where(mask, logits, -np.inf)

    probs = tf.nn.softmax(logits).numpy()
    top_k = min(top_k, int(np.count_nonzero(probs)))
    top_indices = probs.argsort()[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        uci = id_to_move[int(idx)]
        try:
            san = board.san(chess.Move.from_uci(uci))
        except ValueError:
            san = None  # --allow-illegal can surface a move that isn't legal right now
        candidates.append({"uci": uci, "san": san, "probability": float(probs[idx])})
    return candidates


def print_prediction(candidates, top_k):
    if candidates is None:
        return
    best = candidates[0]
    best_label = f"{best['san']} ({best['uci']})" if best["san"] else best["uci"]
    print(f"Predicted move: {best_label}  [{best['probability']:.1%}]")
    if len(candidates) > 1:
        print(f"Top {len(candidates)} candidates:")
        for rank, c in enumerate(candidates, start=1):
            label = f"{c['san']} ({c['uci']})" if c["san"] else c["uci"]
            print(f"  {rank}. {label:<18} {c['probability']:.1%}")


def run_interactive(predict_fn, ids, board, move_to_id, id_to_move, pad_id, seq_len, top_k, allow_illegal):
    """Persistent loop: predict, wait for the move actually played, append
    it, predict again. One process, one model load, one trace of
    predict_fn -- see make_predict_fn()'s docstring for why every
    prediction after the first reuses that same compiled graph."""
    print("\nInteractive mode -- enter each move as it's played (SAN or UCI), or 'quit' to exit.")
    while True:
        if board.is_game_over():
            if board.is_checkmate():
                result = "Checkmate"
            elif board.is_stalemate():
                result = "Stalemate"
            else:
                result = "Game over"
            print(f"\n{result} ({board.result()}) -- nothing left to predict.")
            return

        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        print(f"\nPosition after {len(ids)} move(s) -- {side_to_move} to play (move {board.fullmove_number}).")
        if board.is_check():
            print("(in check)")

        input_ids, next_move_position = build_input_window(ids, seq_len, pad_id)
        candidates = predict_next_move(predict_fn, input_ids, next_move_position, board, id_to_move, top_k, allow_illegal)
        print_prediction(candidates, top_k)

        try:
            token = input("\nEnter the move actually played (or 'quit'): ").strip()
        except EOFError:
            print()
            return
        if token.lower() in ("quit", "exit", "q"):
            return
        if not token:
            continue

        move = None
        try:
            move = board.parse_san(token)
        except ValueError:
            try:
                move = board.parse_uci(token)
            except ValueError:
                pass
        if move is None:
            print(f"'{token}' isn't a legal SAN or UCI move in this position -- try again.")
            continue

        uci = move.uci()
        move_id = move_to_id.get(uci)
        if move_id is None:
            # Same defensive case as replay_moves() -- shouldn't happen given
            # how build_vocab.py enumerates the vocabulary.
            print(f"'{token}' -> {uci} has no vocabulary entry (unexpected) -- try again.")
            continue

        ids.append(move_id)
        board.push(move)


def main():
    cfg = load_config()
    args = parse_args(cfg)

    for path, label in ((args.vocab, "vocab"), (args.checkpoint, "checkpoint"), (args.hyperparameters, "hyperparameters")):
        if not Path(path).exists():
            hint = " (run train.py first)" if label != "vocab" else " (run build_vocab.py first)"
            print(f"{label.capitalize()} file not found: {path}{hint}", file=sys.stderr)
            sys.exit(1)

    with Path(args.vocab).open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    move_to_id = vocab["move_to_id"]
    id_to_move = {v: k for k, v in move_to_id.items()}
    pad_id = vocab["pad_id"]

    with args.hyperparameters.open("r", encoding="utf-8") as f:
        used_hp = json.load(f)
    model_kwargs = {k: v for k, v in used_hp.items() if k not in _NON_CONSTRUCTOR_KEYS}

    model = from_vocab_file(str(args.vocab), max_seq_len=args.seq_len, **model_kwargs)
    model(tf.zeros((1, args.seq_len), dtype=tf.int32))  # build every sub-layer via a real forward pass
    model.load_weights(str(args.checkpoint))

    predict_fn = make_predict_fn(model)
    predict_fn(tf.zeros((1, args.seq_len), dtype=tf.int32))  # trace once, up front, rather than on the first real move

    if args.pgn is not None:
        move_tokens = moves_from_pgn(args.pgn)
    elif args.moves is not None:
        move_tokens = clean_move_tokens(args.moves)
    else:
        move_tokens = []  # --interactive with no starting context -- fresh board

    if not move_tokens and not args.interactive:
        print("No moves given.", file=sys.stderr)
        sys.exit(1)

    ids, board = replay_moves(move_tokens, move_to_id) if move_tokens else ([], chess.Board())

    if args.interactive:
        run_interactive(predict_fn, ids, board, move_to_id, id_to_move, pad_id, args.seq_len, args.top_k, args.allow_illegal)
        return

    input_ids, next_move_position = build_input_window(ids, args.seq_len, pad_id)

    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    move_number = board.fullmove_number
    print(f"Position after {len(ids)} move(s) -- {side_to_move} to play (move {move_number}).")
    if board.is_check():
        print("(in check)")

    candidates = predict_next_move(
        predict_fn, input_ids, next_move_position, board, id_to_move, args.top_k, args.allow_illegal
    )

    if candidates is None:
        result = "Checkmate" if board.is_checkmate() else "Stalemate"
        print(f"\n{result} -- no legal moves, nothing to predict.")
        return

    if not candidates:
        print("\nNo legal moves found in the model's vocabulary (unexpected) -- try --allow-illegal to inspect raw output.")
        return

    print()
    print_prediction(candidates, args.top_k)


if __name__ == "__main__":
    main()
