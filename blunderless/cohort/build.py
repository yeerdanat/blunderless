"""Build cohort baselines from the sampled Lichess dump games.

The dump games carry [%eval] annotations, so severity classification is
free — no engine. The engine runs only on classified error positions
(shallow, single PV) to extract the line needed for motif detection.
Both players of each cohort game are in-band, so both sides' moves count.
"""

from __future__ import annotations

import io
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from blunderless.analysis.book import is_book_position
from blunderless.analysis.engine import Engine
from blunderless.analysis.filters import (
    is_forced,
    should_classify,
    time_bucket,
)
from blunderless.analysis.filters import (
    phase as detect_phase,
)
from blunderless.analysis.motifs import detect_motifs
from blunderless.analysis.winprob import classify, delta_win_prob, eval_to_win_prob, pov
from blunderless.db.models import CohortBaseline

COHORT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cohort"
MOTIF_NODES = 40_000
ERROR_SEVERITIES = ("blunder", "mistake", "inaccuracy")
ANY_ERROR = "__any_error__"
ELIGIBLE = "__eligible__"


@dataclass
class CellCounts:
    """Counters for one (band, tc): eligible moves and error/motif counts
    per (phase, bucket)."""

    eligible: Counter
    errors: Counter  # (motif_or_ANY, phase, bucket) -> count

    def __init__(self) -> None:
        self.eligible = Counter()
        self.errors = Counter()


def _score_to_wp(score: chess.engine.PovScore) -> float:
    white = score.white()
    return eval_to_win_prob(white.score(), white.mate())


def process_game(pgn_text: str, counts: CellCounts, engine: Engine | None) -> None:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return

    board = game.board()
    still_book = True
    prev_wp: float | None = None  # white POV, position before current node
    nodes = list(game.mainline())
    for i, node in enumerate(nodes):
        move = node.move
        mover = "white" if board.turn == chess.WHITE else "black"
        # book = stays in theory (see pipeline.py) — leaving book is a decision
        before_book = still_book and is_book_position(board)
        in_book = False
        if before_book:
            board.push(move)
            in_book = is_book_position(board)
            board.pop()
        still_book = in_book
        forced = is_forced(board)
        ply = i + 1
        ph = detect_phase(board, ply, in_book)
        clock = node.clock()
        bucket = time_bucket(clock)

        ev = node.eval()  # eval AFTER this move (lichess convention)
        wp_after = _score_to_wp(ev) if ev is not None else None
        board_before = board.copy() if engine is not None else None
        board.push(move)

        if prev_wp is None or wp_after is None:
            prev_wp = wp_after
            continue

        eligible = should_classify(
            in_book=in_book,
            forced=forced,
            wp_before_mover=pov(prev_wp, mover),
            wp_after_mover=pov(wp_after, mover),
            bucket=bucket,
        )
        if eligible:
            counts.eligible[(ph, bucket)] += 1
            delta = delta_win_prob(prev_wp, wp_after, mover)
            severity = classify(delta)
            if severity in ERROR_SEVERITIES:
                counts.errors[(ANY_ERROR, ph, bucket)] += 1
                if engine is not None and board_before is not None:
                    analysis = engine.analyse(board_before, nodes=MOTIF_NODES)
                    best = analysis.best
                    mate = best.mate_in
                    if mate is not None and mover == "black":
                        mate = -mate
                    for hit in detect_motifs(
                        board_before,
                        played_uci=move.uci(),
                        best_pv=best.pv,
                        best_mate_in=mate,
                    ):
                        counts.errors[(hit.motif, ph, bucket)] += 1
        prev_wp = wp_after


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def store_baseline(
    db: Session, band: str, tc: str, counts: CellCounts
) -> int:
    """Upsert cohort_baseline rows for one (band, tc). Returns rows written."""
    written = 0
    motifs = {key[0] for key in counts.errors}
    for motif in motifs | {ANY_ERROR}:
        for (ph, bucket), n_eligible in counts.eligible.items():
            k = counts.errors.get((motif, ph, bucket), 0)
            if n_eligible == 0:
                continue
            lo, hi = _wilson_ci(k, n_eligible)
            db.execute(
                insert(CohortBaseline)
                .values(
                    rating_band=band,
                    time_control=tc,
                    motif_type=motif,
                    phase=ph,
                    time_bucket=bucket,
                    error_rate=k / n_eligible,
                    n_samples=n_eligible,
                    ci_low=lo,
                    ci_high=hi,
                )
                .on_conflict_do_update(
                    index_elements=[
                        "rating_band", "time_control", "motif_type", "phase", "time_bucket"
                    ],
                    set_={
                        "error_rate": k / n_eligible,
                        "n_samples": n_eligible,
                        "ci_low": lo,
                        "ci_high": hi,
                    },
                )
            )
            written += 1
    db.commit()
    return written


def split_cell_file(path: Path) -> list[str]:
    text = path.read_text()
    import re

    starts = [m.start() for m in re.finditer(r"^\[Event ", text, re.MULTILINE)]
    return [
        text[s:e].strip()
        for s, e in zip(starts, [*starts[1:], len(text)], strict=False)
        if text[s:e].strip()
    ]
