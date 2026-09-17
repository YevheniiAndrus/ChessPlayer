#!/usr/bin/env python3
"""
filter_pgn_by_elo.py

Filter a PGN file (e.g. a Lichess games database dump) down to only the
games where BOTH players have a rating strictly greater than a given
threshold.

Requires:
    pip install python-chess tqdm pyyaml
(see requirements.txt in the project root)

--min-elo and --output default from config.yaml (data.min_elo and
paths.filtered_pgn) -- pass them explicitly to override for a one-off run
without editing the config.

Usage:
    python filter_pgn_by_elo.py --input games.pgn
    python filter_pgn_by_elo.py --input games.pgn --output filtered.pgn --min-elo 2200
    python filter_pgn_by_elo.py --input games.pgn --total-games 91201234

A game is kept only if it has BOTH a "WhiteElo" and a "BlackElo" header,
both parse as integers, and both are strictly greater than --min-elo.
Games with missing/unknown ratings (e.g. WhiteElo "?") are skipped.
Games whose movetext python-chess couldn't fully parse (game.errors is
non-empty) are also skipped, since they won't be usable in later steps
(move/board encoding) anyway.

Counting the exact number of games in a multi-gigabyte Lichess PGN dump
ahead of time requires a full pass over the file, so instead of doing that
automatically, you can pass --total-games with a known/estimated count
(Lichess publishes this figure alongside each monthly database dump) so
the tqdm progress bar can show a percentage and ETA. Without it, the bar
still shows a live count and processing rate, just no percentage/ETA.
"""

import argparse
import sys
from pathlib import Path

import chess.pgn
from tqdm import tqdm

from config import load_config


def parse_args(cfg):
    parser = argparse.ArgumentParser(
        description="Filter a PGN file to games where both players are rated above a threshold."
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Path to the source PGN file.")
    parser.add_argument(
        "--output", "-o", type=Path, default=cfg["paths"]["filtered_pgn"],
        help="Path to write the filtered PGN file. Default: paths.filtered_pgn in config.yaml.",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=cfg["data"]["min_elo"],
        help="Minimum rating (strictly greater than) required for BOTH players. "
             "Default: data.min_elo in config.yaml.",
    )
    parser.add_argument(
        "--total-games",
        type=int,
        default=None,
        help=(
            "Known/estimated total number of games in the input file, used only to show "
            "a percentage and ETA on the progress bar. Lichess publishes this figure "
            "alongside each monthly database dump. Optional -- omit it if unknown."
        ),
    )
    return parser.parse_args()


def elo_passes(headers: "chess.pgn.Headers", min_elo: int) -> bool:
    white_elo = headers.get("WhiteElo")
    black_elo = headers.get("BlackElo")
    if not white_elo or not black_elo:
        return False
    try:
        return int(white_elo) > min_elo and int(black_elo) > min_elo
    except ValueError:
        return False


def main():
    cfg = load_config()
    args = parse_args(cfg)
    args.output = Path(args.output)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    unparseable = 0

    with args.input.open("r", encoding="utf-8", errors="replace") as pgn_in, \
         args.output.open("w", encoding="utf-8") as pgn_out, \
         tqdm(total=args.total_games, unit="game", desc="Filtering games", dynamic_ncols=True) as bar:

        while True:
            try:
                game = chess.pgn.read_game(pgn_in)
            except (ValueError, UnicodeDecodeError) as exc:
                # A malformed game can leave the parser unable to find the next
                # header block reliably -- there's no fully safe way to resync,
                # so we stop rather than risk silently corrupting the output.
                tqdm.write(f"Stopped early after a malformed game: {exc}")
                break

            if game is None:
                break  # end of file

            total += 1
            bar.update(1)

            if game.errors:
                unparseable += 1
            elif elo_passes(game.headers, args.min_elo):
                kept += 1
                print(game, file=pgn_out, end="\n\n")

            bar.set_postfix(kept=kept, refresh=False)

    pct = (kept / total * 100) if total else 0.0
    print(f"Done. Scanned {total:,} games, kept {kept:,} ({pct:.2f}%) "
          f"with both players > {args.min_elo} ELO.")
    if unparseable:
        print(f"Note: {unparseable:,} games had move-parsing errors and were skipped.")
    print(f"Filtered games written to: {args.output}")


if __name__ == "__main__":
    main()
