"""Exclusion filters and move-context tagging (§6.3).

A move only enters weakness statistics when it was a real decision:
not book theory, not the only legal move, not a dead-lost/dead-won
position, and not a time scramble (tagged separately — reflexes, not
understanding).
"""

from __future__ import annotations

import chess

DECIDED_CP = 800  # beyond this the game is over for statistics purposes
SCRAMBLE_SECONDS = 10.0
TIME_PRESSURE_SECONDS = 60.0

# Phase: simple material threshold. Endgame once total non-pawn, non-king
# material (both sides) drops to 6 points or less (Q=4 here, R=2, minor=1
# — a coarse scale, but phase only needs to be roughly right).
_PHASE_WEIGHTS = {
    chess.QUEEN: 4,
    chess.ROOK: 2,
    chess.BISHOP: 1,
    chess.KNIGHT: 1,
}
ENDGAME_THRESHOLD = 6
OPENING_MAX_PLY = 16  # opening ends by ply 16 unless book ended sooner


def phase(board: chess.Board, ply: int, in_book: bool) -> str:
    if in_book or ply <= OPENING_MAX_PLY:
        return "opening"
    material = sum(
        weight * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
        for pt, weight in _PHASE_WEIGHTS.items()
    )
    return "endgame" if material <= ENDGAME_THRESHOLD else "middlegame"


def is_forced(board: chess.Board) -> bool:
    """True when there was no decision to make: exactly one legal move."""
    moves = iter(board.legal_moves)
    first = next(moves, None)
    return first is not None and next(moves, None) is None


# wp equivalent of DECIDED_CP: win prob at +800cp for the side ahead.
DECIDED_WP = 0.95


def stayed_decided(wp_before_mover: float, wp_after_mover: float) -> bool:
    """Both sides of the move sit in the same decided zone.

    A position is only excluded as "already decided" when the move changed
    nothing: totally winning before AND after (or totally lost before and
    after). A mate or +900 advantage that gets thrown away is exactly the
    kind of decision this project exists to classify — the naive
    "exclude if decided before" gate silently deleted every missed mate.
    """
    both_winning = wp_before_mover >= DECIDED_WP and wp_after_mover >= DECIDED_WP
    both_lost = wp_before_mover <= 1 - DECIDED_WP and wp_after_mover <= 1 - DECIDED_WP
    return both_winning or both_lost


def time_bucket(clock_remaining_s: float | None) -> str:
    """Bucket by clock *after* the move; None = no clock data (unknown)."""
    if clock_remaining_s is None:
        return "unknown"
    if clock_remaining_s < SCRAMBLE_SECONDS:
        return "scramble"
    if clock_remaining_s < TIME_PRESSURE_SECONDS:
        return "lt60s"
    return "normal"


def should_classify(
    *,
    in_book: bool,
    forced: bool,
    wp_before_mover: float,
    wp_after_mover: float,
    bucket: str,
) -> bool:
    """Gate for whether this move participates in weakness statistics."""
    return (
        not in_book
        and not forced
        and not stayed_decided(wp_before_mover, wp_after_mover)
        and bucket != "scramble"
    )
