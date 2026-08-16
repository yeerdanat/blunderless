"""Position identity for caching: Zobrist hash of the full position.

Covers piece placement, side to move, castling rights, and en passant —
never the game or move number, so transpositions and repeat openings
across games hit the same cache entry.
"""

import chess
import chess.polyglot


def position_key(board: chess.Board) -> str:
    return f"{chess.polyglot.zobrist_hash(board):016x}"
