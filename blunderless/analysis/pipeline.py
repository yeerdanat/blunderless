"""Two-pass game analysis (§6.5).

Pass 1: every position at a small fixed node count, single PV, cache-aware.
Pass 2: deep MultiPV only where pass 1 saw a meaningful win-prob drop,
plus a seeded random control sample (to measure pass 1's false-negative
rate — without it, quiet positional errors vanish silently).
"""

from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass, field

import chess
import chess.pgn
from sqlalchemy import delete
from sqlalchemy.orm import Session

from blunderless.analysis import cache
from blunderless.analysis.book import is_book_position
from blunderless.analysis.engine import Engine, PositionAnalysis
from blunderless.analysis.filters import is_forced, phase, should_classify, time_bucket
from blunderless.analysis.keys import position_key
from blunderless.analysis.winprob import (
    classify,
    delta_win_prob,
    eval_to_win_prob,
    pov,
)
from blunderless.db.models import Game, MoveAnalysis
from blunderless.ingest.pgn import move_clocks

PASS1_NODES = 40_000
PASS2_NODES = 350_000
PASS2_MULTIPV = 4
CANDIDATE_DELTA = 0.03  # pass-1 threshold for escalating to deep analysis
CONTROL_SAMPLE_RATE = 0.05


@dataclass
class GameStats:
    plies: int = 0
    player_moves: int = 0
    cache_hits: int = 0
    engine_calls_pass1: int = 0
    engine_calls_pass2: int = 0
    control_samples: int = 0
    seconds: float = 0.0

    def add(self, other: GameStats) -> None:
        for f in (
            "plies", "player_moves", "cache_hits", "engine_calls_pass1",
            "engine_calls_pass2", "control_samples", "seconds",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))


@dataclass
class _Ply:
    """Everything known about one half-move as analysis proceeds."""

    ply: int  # 1-based
    san: str
    uci: str
    mover: str  # "white" | "black"
    key_before: str
    fen_before: str
    in_book: bool
    forced: bool
    phase: str
    clock_after: float | None
    terminal_after: str | None = None  # "checkmate" | "draw" | None
    analysis_before: PositionAnalysis | None = None
    wp_before: float | None = None  # White POV
    wp_after: float | None = None
    delta: float | None = None
    complexity: float | None = None
    deep: bool = field(default=False)


def _terminal_wp(board_after: chess.Board, mover_is_white: bool) -> tuple[str, float] | None:
    """Win prob (White POV) when the move ends the game outright."""
    if board_after.is_checkmate():
        return "checkmate", 1.0 if mover_is_white else 0.0
    if (
        board_after.is_stalemate()
        or board_after.is_insufficient_material()
        or board_after.can_claim_fifty_moves()
    ):
        return "draw", 0.5
    return None


def _eval_position(
    db: Session,
    engine: Engine,
    board: chess.Board,
    key: str,
    nodes: int,
    multipv: int,
    stats: GameStats,
    pass_no: int,
    use_cache: bool = True,
) -> PositionAnalysis:
    if use_cache:
        cached = cache.get(db, key, nodes, multipv)
        if cached is not None:
            stats.cache_hits += 1
            return cached
    analysis = engine.analyse(board, nodes=nodes, multipv=multipv)
    if use_cache:
        cache.put(db, key, analysis)
    if pass_no == 1:
        stats.engine_calls_pass1 += 1
    else:
        stats.engine_calls_pass2 += 1
    return analysis


def _complexity(analysis: PositionAnalysis, side_to_move: str) -> float | None:
    """Gap in win prob between the best and 2nd-best move, mover's POV.

    Large gap = only-move position (sharp); small gap = many playable
    moves. None when MultiPV depth wasn't requested.
    """
    if len(analysis.lines) < 2:
        return None
    wps = [
        pov(eval_to_win_prob(ln.eval_cp, ln.mate_in), side_to_move)
        for ln in analysis.lines
    ]
    return max(0.0, wps[0] - wps[1])


def analyze_game(
    db: Session,
    engine: Engine,
    game: Game,
    *,
    naive: bool = False,
) -> GameStats:
    """Analyze one game's player moves; writes MoveAnalysis rows.

    naive=True disables tiering (every position deep) — kept for
    benchmarking the optimization story honestly.
    """
    started = time.monotonic()
    stats = GameStats()

    parsed = chess.pgn.read_game(io.StringIO(game.pgn))
    if parsed is None:
        return stats
    clocks = move_clocks(game.pgn)

    # --- replay: collect per-ply context -------------------------------
    plies: list[_Ply] = []
    board = parsed.board()
    still_book = True
    for i, move in enumerate(parsed.mainline_moves()):
        ply_no = i + 1
        mover = "white" if board.turn == chess.WHITE else "black"
        # A move is "book" only if it stays in theory: played from a book
        # position AND landing in one. A blunder played out of a known
        # opening leaves the book lines and must be classified.
        before_book = still_book and is_book_position(board)
        in_book = False
        if before_book:
            board.push(move)
            in_book = is_book_position(board)
            board.pop()
        still_book = in_book
        p = _Ply(
            ply=ply_no,
            san=board.san(move),
            uci=move.uci(),
            mover=mover,
            key_before=position_key(board),
            fen_before=board.fen(),
            in_book=in_book,
            forced=is_forced(board),
            phase=phase(board, ply_no, in_book),
            clock_after=clocks[i] if i < len(clocks) else None,
        )
        board.push(move)
        terminal = _terminal_wp(board, mover == "white")
        if terminal is not None:
            p.terminal_after, p.wp_after = terminal
        plies.append(p)
    stats.plies = len(plies)

    # --- pass 1: shallow eval of every unique position ------------------
    nodes1 = PASS2_NODES if naive else PASS1_NODES
    multipv1 = PASS2_MULTIPV if naive else 1
    boards: dict[str, chess.Board] = {}
    replay = parsed.board()
    keys_in_order: list[str] = [position_key(replay)]
    boards[keys_in_order[0]] = replay.copy()
    for move in parsed.mainline_moves():
        replay.push(move)
        k = position_key(replay)
        if k not in boards:
            boards[k] = replay.copy()
            keys_in_order.append(k)

    evals: dict[str, PositionAnalysis] = {}
    for k in keys_in_order:
        b = boards[k]
        if b.is_game_over(claim_draw=True):
            continue
        # naive benchmark mode: no cache, no tiering — the honest "before".
        evals[k] = _eval_position(
            db, engine, b, k, nodes1, multipv1, stats, 1, use_cache=not naive
        )

    # attach before/after win probs
    key_after: dict[int, str] = {}
    replay = parsed.board()
    for p in plies:
        replay.push(chess.Move.from_uci(p.uci))
        key_after[p.ply] = position_key(replay)

    for p in plies:
        a_before = evals.get(p.key_before)
        if a_before is not None:
            p.analysis_before = a_before
            p.wp_before = eval_to_win_prob(a_before.best.eval_cp, a_before.best.mate_in)
            p.complexity = _complexity(a_before, p.mover)
        if p.wp_after is None:
            a_after = evals.get(key_after[p.ply])
            if a_after is not None:
                p.wp_after = eval_to_win_prob(a_after.best.eval_cp, a_after.best.mate_in)
        if p.wp_before is not None and p.wp_after is not None:
            p.delta = delta_win_prob(p.wp_before, p.wp_after, p.mover)

    # --- pass 2: deep MultiPV on candidates + control sample -------------
    if not naive:
        rng = random.Random(game.id)  # deterministic per game
        for p in plies:
            if p.mover != game.player_color or p.delta is None:
                continue
            candidate = p.delta >= CANDIDATE_DELTA
            control = not candidate and rng.random() < CONTROL_SAMPLE_RATE
            if not (candidate or control):
                continue
            if control:
                stats.control_samples += 1
            b_before = boards[p.key_before]
            deep_before = _eval_position(
                db, engine, b_before, p.key_before, PASS2_NODES, PASS2_MULTIPV, stats, 2
            )
            p.analysis_before = deep_before
            p.wp_before = eval_to_win_prob(
                deep_before.best.eval_cp, deep_before.best.mate_in
            )
            p.complexity = _complexity(deep_before, p.mover)
            k_after = key_after[p.ply]
            if p.terminal_after is None and k_after in boards:
                b_after = boards[k_after]
                if not b_after.is_game_over(claim_draw=True):
                    deep_after = _eval_position(
                        db, engine, b_after, k_after, PASS2_NODES, 1, stats, 2
                    )
                    p.wp_after = eval_to_win_prob(
                        deep_after.best.eval_cp, deep_after.best.mate_in
                    )
            p.delta = delta_win_prob(p.wp_before, p.wp_after, p.mover)
            p.deep = True

    # --- persist player's moves -----------------------------------------
    db.execute(delete(MoveAnalysis).where(MoveAnalysis.game_id == game.id))
    for p in plies:
        if p.mover != game.player_color:
            continue
        stats.player_moves += 1
        bucket = time_bucket(p.clock_after)
        best_before = p.analysis_before.best if p.analysis_before else None
        eligible = (
            p.wp_before is not None
            and p.wp_after is not None
            and should_classify(
                in_book=p.in_book,
                forced=p.forced,
                wp_before_mover=pov(p.wp_before, p.mover),
                wp_after_mover=pov(p.wp_after, p.mover),
                bucket=bucket,
            )
        )
        severity = classify(p.delta) if (eligible and p.delta is not None) else None
        db.add(
            MoveAnalysis(
                game_id=game.id,
                ply=p.ply,
                san=p.san,
                position_key=p.key_before,
                win_prob_before=p.wp_before,
                win_prob_after=p.wp_after,
                delta_win_prob=p.delta,
                severity=severity,
                best_move=best_before.move if best_before else None,
                was_forced=p.forced,
                was_book=p.in_book,
                phase=p.phase,
                clock_remaining_s=p.clock_after,
                complexity_score=p.complexity,
            )
        )
    db.commit()
    stats.seconds = time.monotonic() - started
    return stats
