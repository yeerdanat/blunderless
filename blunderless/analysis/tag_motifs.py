"""Tag classified errors with tactical motifs, from cached engine output.

Runs after the analysis pipeline: every blunder/mistake/inaccuracy gets
its position rebuilt, the deep MultiPV lines pulled from the cache, and
the geometric detectors applied. Pure post-processing — no engine calls.
"""

from __future__ import annotations

import io

import chess
import chess.pgn
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from blunderless.analysis.keys import position_key
from blunderless.analysis.motifs import detect_motifs
from blunderless.analysis.pipeline import PASS1_NODES, PASS2_NODES
from blunderless.db.models import Game, Motif, MoveAnalysis, PositionEval

ERROR_SEVERITIES = ("blunder", "mistake", "inaccuracy")


def _cached_lines(db: Session, key: str) -> list[dict] | None:
    for nodes in (PASS2_NODES, PASS1_NODES):
        row = db.get(PositionEval, (key, nodes))
        if row is not None and row.pv_moves:
            return row.pv_moves
    return None


def tag_game_motifs(db: Session, game: Game) -> int:
    """(Re)tag all error moves of one game. Returns motif rows written."""
    errors = list(
        db.execute(
            select(MoveAnalysis)
            .where(
                MoveAnalysis.game_id == game.id,
                MoveAnalysis.severity.in_(ERROR_SEVERITIES),
            )
            .order_by(MoveAnalysis.ply)
        ).scalars()
    )
    if not errors:
        return 0
    db.execute(
        delete(Motif).where(
            Motif.move_analysis_id.in_([e.id for e in errors])
        )
    )

    parsed = chess.pgn.read_game(io.StringIO(game.pgn))
    if parsed is None:
        return 0
    boards_by_ply: dict[int, chess.Board] = {}
    moves_by_ply: dict[int, chess.Move] = {}
    board = parsed.board()
    wanted = {e.ply for e in errors}
    for i, move in enumerate(parsed.mainline_moves()):
        ply = i + 1
        if ply in wanted:
            boards_by_ply[ply] = board.copy()
            moves_by_ply[ply] = move
        board.push(move)

    written = 0
    for err in errors:
        board_before = boards_by_ply.get(err.ply)
        if board_before is None:
            continue
        lines = _cached_lines(db, err.position_key)
        if not lines:
            continue
        best = lines[0]
        played = moves_by_ply[err.ply]

        board_after = board_before.copy(stack=False)
        board_after.push(played)
        after_lines = _cached_lines(db, position_key(board_after))
        reply_pv = after_lines[0]["pv"] if after_lines else None
        # Does the played move still lead to a mate for the mover?
        played_mate_in = None
        if after_lines and after_lines[0]["mate_in"] is not None:
            mate_white_pov = after_lines[0]["mate_in"]
            played_mate_in = (
                mate_white_pov if err.game.player_color == "white" else -mate_white_pov
            )
        best_mate_in = best["mate_in"]
        if best_mate_in is not None and game.player_color == "black":
            best_mate_in = -best_mate_in

        for hit in detect_motifs(
            board_before,
            played_uci=played.uci(),
            best_pv=best["pv"],
            best_mate_in=best_mate_in,
            played_mate_in=played_mate_in,
            reply_pv=reply_pv,
        ):
            db.add(
                Motif(
                    move_analysis_id=err.id,
                    motif_type=hit.motif,
                    confidence=hit.confidence,
                    detail=hit.detail,
                )
            )
            written += 1
    db.commit()
    return written


def tag_player_motifs(db: Session, player_id: int) -> int:
    total = 0
    game_ids = db.execute(
        select(Game.id).where(Game.player_id == player_id)
    ).scalars()
    for gid in game_ids:
        game = db.get(Game, gid)
        total += tag_game_motifs(db, game)
    return total
