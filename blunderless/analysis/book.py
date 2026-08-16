"""Opening book from the lichess-org/chess-openings dataset.

Every position reachable along a named ECO line counts as "in book";
moves played from a book position are excluded from error statistics
(§6.3 — theory is not a decision the player got wrong).
"""

from __future__ import annotations

import csv
import io
from functools import lru_cache
from pathlib import Path

import chess
import chess.pgn

from blunderless.analysis.keys import position_key

DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "openings"


@lru_cache(maxsize=1)
def book_keys(tsv_dir: str | None = None) -> frozenset[str]:
    """Set of Zobrist keys for every position along every book line."""
    directory = Path(tsv_dir) if tsv_dir else DEFAULT_DIR
    keys: set[str] = set()
    for tsv in sorted(directory.glob("*.tsv")):
        with open(tsv, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                game = chess.pgn.read_game(io.StringIO(row["pgn"]))
                if game is None:
                    continue
                board = game.board()
                keys.add(position_key(board))
                for move in game.mainline_moves():
                    board.push(move)
                    keys.add(position_key(board))
    return frozenset(keys)


def is_book_position(board: chess.Board) -> bool:
    return position_key(board) in book_keys()
