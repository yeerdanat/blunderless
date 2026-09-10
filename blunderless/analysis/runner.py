"""Parallel analysis of a player's corpus: one Stockfish per process.

Parallelism lives here, not inside a search (engines run Threads=1 for
reproducibility). Cache is shared through Postgres, so workers benefit
from each other's opening evaluations as they go.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from sqlalchemy import select

from blunderless.analysis.engine import Engine
from blunderless.analysis.pipeline import GameStats, analyze_game
from blunderless.db.models import Game, Player
from blunderless.db.session import make_session_factory

_worker_sessions = None


def _init_worker() -> None:
    global _worker_sessions
    _worker_sessions = make_session_factory()


def _analyze_one(game_id: int, naive: bool) -> GameStats:
    assert _worker_sessions is not None
    # Engine per task, not per worker: python-chess's UCI transport thread
    # is non-daemon, and interpreter shutdown joins threads BEFORE atexit
    # runs — a long-lived engine deadlocks worker exit. Startup is ~50ms
    # against seconds of analysis, and a fresh hash table per game keeps
    # fixed-node results reproducible regardless of scheduling order.
    #
    # Cache upserts from parallel workers can deadlock in Postgres when two
    # games share positions (transpositions); the loser is retried once.
    from sqlalchemy.exc import OperationalError

    for attempt in (1, 2):
        try:
            with _worker_sessions() as db, Engine() as engine:
                game = db.get(Game, game_id)
                if game is None:
                    return GameStats()
                return analyze_game(db, engine, game, naive=naive)
        except OperationalError:
            if attempt == 2:
                raise
    return GameStats()


def default_workers() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


def analyze_player(
    platform: str,
    username: str,
    *,
    workers: int | None = None,
    naive: bool = False,
    limit: int | None = None,
    progress: Callable[[int, int, GameStats], None] | None = None,
) -> GameStats:
    sessions = make_session_factory()
    with sessions() as db:
        player = db.execute(
            select(Player).where(Player.platform == platform, Player.username == username)
        ).scalar_one()
        game_ids = list(
            db.execute(
                select(Game.id).where(Game.player_id == player.id).order_by(Game.played_at)
            ).scalars()
        )
    if limit is not None:
        game_ids = game_ids[:limit]

    total = GameStats()
    n_workers = workers or default_workers()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_analyze_one, gid, naive): gid for gid in game_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            stats = fut.result()
            total.add(stats)
            if progress is not None:
                progress(i, len(game_ids), stats)
    return total
