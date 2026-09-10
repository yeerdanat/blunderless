import asyncio
import io
import json
from statistics import median

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from blunderless import __version__
from blunderless.api.jobs import JOBS, start_analysis_job
from blunderless.cohort.build import ANY_ERROR
from blunderless.db.models import Game, Motif, MoveAnalysis, Player, Weakness
from blunderless.db.session import make_session_factory
from blunderless.narrate.narrator import narrate
from blunderless.puzzles import recommend

app = FastAPI(title="Blunderless", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = make_session_factory()

ERROR_SEVERITIES = ("blunder", "mistake", "inaccuracy")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _get_player(db, platform: str, username: str) -> Player:
    player = db.execute(
        select(Player).where(Player.platform == platform, Player.username == username)
    ).scalar_one_or_none()
    if player is None:
        raise HTTPException(404, f"player {username} on {platform} not synced")
    return player


@app.post("/players/{platform}/{username}/sync")
def sync(platform: str, username: str, max_games: int | None = None) -> dict:
    from blunderless.ingest.sync import sync_player

    stats = sync_player(platform, username, sessions, max_games=max_games)
    return {"fetched": stats.fetched, "inserted": stats.inserted}


@app.post("/players/{platform}/{username}/analyze")
def analyze(platform: str, username: str) -> dict:
    with sessions() as db:
        _get_player(db, platform, username)
    job = start_analysis_job(platform, username)
    return {"job_id": job.id}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-sent events: analysis progress until the job finishes."""
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")

    async def stream():
        while True:
            job = JOBS[job_id]
            payload = {
                "status": job.status,
                "done": job.done,
                "total": job.total,
                "error": job.error,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if job.status in ("done", "error"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _player_rating(db, player: Player) -> int:
    ratings = [
        r
        for r in db.execute(
            select(Game.opponent_rating).where(
                Game.player_id == player.id, Game.opponent_rating.is_not(None)
            )
        ).scalars()
    ]
    return int(median(ratings)) if ratings else 1500


def _examples_for(db, player: Player, motif_type: str, limit: int = 3) -> list[dict]:
    """Worst examples of one motif: position FEN + facts + narration."""
    rows = db.execute(
        select(MoveAnalysis, Motif, Game)
        .join(Motif, Motif.move_analysis_id == MoveAnalysis.id)
        .join(Game, Game.id == MoveAnalysis.game_id)
        .where(Game.player_id == player.id, Motif.motif_type == motif_type)
        .order_by(MoveAnalysis.delta_win_prob.desc())
        .limit(limit)
    ).all()
    if motif_type == ANY_ERROR:
        rows = [
            (ma, None, g)
            for ma, g in db.execute(
                select(MoveAnalysis, Game)
                .join(Game, Game.id == MoveAnalysis.game_id)
                .where(
                    Game.player_id == player.id,
                    MoveAnalysis.severity.in_(ERROR_SEVERITIES),
                )
                .order_by(MoveAnalysis.delta_win_prob.desc())
                .limit(limit)
            ).all()
        ]

    examples = []
    for ma, motif, game in rows:
        fen = _fen_at_ply(game.pgn, ma.ply)
        best_san = None
        if fen and ma.best_move:
            board = chess.Board(fen)
            try:
                best_san = board.san(chess.Move.from_uci(ma.best_move))
            except ValueError:
                best_san = None
        facts = {
            "player_move_san": ma.san,
            "best_move_san": best_san,
            "delta_win_prob": ma.delta_win_prob,
            "motif": motif.motif_type if motif else None,
            "detail": motif.detail if motif else {},
            "phase": ma.phase,
        }
        text, source = narrate(facts)
        examples.append(
            {
                "game_id": game.id,
                "platform_game_id": game.platform_game_id,
                "played_at": game.played_at.isoformat() if game.played_at else None,
                "ply": ma.ply,
                "fen": fen,
                "san": ma.san,
                "best_move": ma.best_move,
                "best_move_san": best_san,
                "delta_win_prob": ma.delta_win_prob,
                "clock_remaining_s": ma.clock_remaining_s,
                "narration": text,
                "narration_source": source,
            }
        )
    return examples


def _fen_at_ply(pgn: str, ply: int) -> str | None:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if i + 1 == ply:
            return board.fen()
        board.push(move)
    return None


@app.get("/players/{platform}/{username}/report")
def report(platform: str, username: str) -> dict:
    with sessions() as db:
        player = _get_player(db, platform, username)
        rating = _player_rating(db, player)

        n_games = db.execute(
            select(func.count()).select_from(Game).where(Game.player_id == player.id)
        ).scalar_one()
        severity_counts = dict(
            db.execute(
                select(MoveAnalysis.severity, func.count())
                .join(Game, Game.id == MoveAnalysis.game_id)
                .where(Game.player_id == player.id, MoveAnalysis.severity.is_not(None))
                .group_by(MoveAnalysis.severity)
            ).all()
        )

        weaknesses = list(
            db.execute(
                select(Weakness)
                .where(Weakness.player_id == player.id)
                .order_by(Weakness.total_winprob_lost.desc())
            ).scalars()
        )

        out = []
        for w in weaknesses:
            out.append(
                {
                    "motif": w.motif_type,
                    "phase": w.phase,
                    "time_bucket": w.time_bucket,
                    "player_rate": w.player_rate,
                    "cohort_rate": w.cohort_rate,
                    "ratio": w.ratio,
                    "p_value": w.p_value,
                    "q_value": w.q_value,
                    "significant": w.q_value is not None and w.q_value < 0.10,
                    "total_winprob_lost": w.total_winprob_lost,
                    "n_observations": w.n_observations,
                    "examples": _examples_for(db, player, w.motif_type),
                    "puzzles": recommend(w.motif_type, rating)
                    if w.motif_type != ANY_ERROR
                    else [],
                }
            )

        return {
            "player": {
                "platform": platform,
                "username": username,
                "estimated_rating": rating,
                "games": n_games,
                "last_synced_at": player.last_synced_at.isoformat()
                if player.last_synced_at
                else None,
            },
            "severity_counts": severity_counts,
            "weaknesses": out,
        }
