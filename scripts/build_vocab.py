#!/usr/bin/env python3
"""
build_vocab.py

Builds the fixed move vocabulary used to turn chess moves into integer
token ids for the transformer.

Moves are encoded in UCI notation (e.g. "e2e4", "g1f3", "e7e8q" for a
promotion). This is context-free -- a move.uci() string can be produced
directly from a python-chess Move object without needing the board -- and
the set of possible strings is small and fixed, so we enumerate the whole
vocabulary once instead of building it from the dataset.

The vocabulary consists of:
  - every (from_square, to_square) pair with from != to -- this is a
    superset of every geometrically possible move for every piece type
    (sliding pieces, knights, and the king), so it can never miss a real
    move
  - every pawn-promotion variant of the pairs that plausibly represent a
    pawn advancing to the back rank (from the 2nd/7th rank to the 1st/8th
    rank, straight ahead or capturing one file to either side), each with
    the four possible promotion pieces (queen, rook, bishop, knight)
  - two special tokens: "<PAD>" (id 0) for padding, and "<EOS>" (id 1) to
    mark the end of a game's move sequence

Requires:
    pip install python-chess

Usage:
    python build_vocab.py --output ../data/vocab.json
"""

import argparse
import json
from pathlib import Path

import chess

from config import load_config

PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN]


def generate_uci_vocab():
    moves = set()

    # Every from/to square pair: a superset of every non-promotion move any
    # piece could ever make (rook/bishop/queen sliding moves, knight L-shapes,
    # one-square king moves, and plain pawn pushes/captures are all included).
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            moves.add(chess.Move(from_sq, to_sq).uci())

    # Promotion variants: a pawn moving from the 2nd/7th rank to the 1st/8th
    # rank, straight ahead or capturing diagonally (file differs by at most 1).
    promotion_pieces = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    for from_sq in chess.SQUARES:
        from_rank = chess.square_rank(from_sq)
        if from_rank not in (1, 6):
            continue
        from_file = chess.square_file(from_sq)
        to_rank = 0 if from_rank == 1 else 7
        for to_file in (from_file - 1, from_file, from_file + 1):
            if not 0 <= to_file <= 7:
                continue
            to_sq = chess.square(to_file, to_rank)
            for promo in promotion_pieces:
                moves.add(chess.Move(from_sq, to_sq, promotion=promo).uci())

    return sorted(moves)


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Build the UCI move vocabulary.")
    parser.add_argument("--output", "-o", type=Path, default=cfg["paths"]["vocab"],
                         help="Default: paths.vocab in config.yaml.")
    args = parser.parse_args()

    moves = generate_uci_vocab()
    vocab = SPECIAL_TOKENS + moves
    move_to_id = {token: idx for idx, token in enumerate(vocab)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pad_token": PAD_TOKEN,
                "eos_token": EOS_TOKEN,
                "pad_id": move_to_id[PAD_TOKEN],
                "eos_id": move_to_id[EOS_TOKEN],
                "vocab_size": len(move_to_id),
                "move_to_id": move_to_id,
            },
            f,
            indent=2,
        )

    print(f"Vocabulary size: {len(move_to_id):,} tokens "
          f"({len(SPECIAL_TOKENS)} special + {len(moves):,} UCI moves).")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
