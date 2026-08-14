"""Centipawns → win probability, and severity classification on Δwin%.

Why not centipawns: a 300cp drop from +50 to -250 flips a drawn game into
a lost one; a 300cp drop from +1200 to +900 changes nothing. Classifying
on the win-probability delta makes those two events different sizes, and
automatically stops flagging "errors" in already-decided positions.
"""

from __future__ import annotations

import math
from enum import StrEnum

# Lichess's logistic mapping: win% = 50 + 50*(2/(1+exp(-K*cp)) - 1),
# which simplifies to the sigmoid below.
K = 0.00368208

BLUNDER_THRESHOLD = 0.20
MISTAKE_THRESHOLD = 0.10
INACCURACY_THRESHOLD = 0.05


class Severity(StrEnum):
    BLUNDER = "blunder"
    MISTAKE = "mistake"
    INACCURACY = "inaccuracy"
    FINE = "fine"


def cp_to_win_prob(cp: float) -> float:
    """White's win probability in [0, 1] for a White-POV centipawn eval."""
    return 1.0 / (1.0 + math.exp(-K * cp))


def eval_to_win_prob(eval_cp: int | None, mate_in: int | None) -> float:
    """White's win probability from an engine score (exactly one field set).

    Mate scores don't live on the centipawn scale: any forced mate is a
    certainty, regardless of distance.
    """
    if mate_in is not None:
        return 1.0 if mate_in > 0 else 0.0
    if eval_cp is None:
        raise ValueError("engine score has neither eval_cp nor mate_in")
    return cp_to_win_prob(eval_cp)


def pov(win_prob_white: float, color: str) -> float:
    """Flip a White-POV win probability to the given player's perspective."""
    if color == "white":
        return win_prob_white
    if color == "black":
        return 1.0 - win_prob_white
    raise ValueError(f"invalid color: {color!r}")


def delta_win_prob(win_prob_before_white: float, win_prob_after_white: float, color: str) -> float:
    """How much win probability the mover threw away with their move.

    Positive = the move made things worse for the mover. Clamped at 0:
    a move can't be better than the engine's best by definition, so any
    negative delta is search noise, not brilliance.
    """
    delta = pov(win_prob_before_white, color) - pov(win_prob_after_white, color)
    return max(0.0, delta)


def classify(delta: float) -> Severity:
    if delta >= BLUNDER_THRESHOLD:
        return Severity.BLUNDER
    if delta >= MISTAKE_THRESHOLD:
        return Severity.MISTAKE
    if delta >= INACCURACY_THRESHOLD:
        return Severity.INACCURACY
    return Severity.FINE
