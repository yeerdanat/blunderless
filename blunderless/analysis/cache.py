"""Durable position-eval cache in Postgres, keyed (zobrist, node_count).

Openings repeat heavily across one player's history; the cache turns the
first 10-15 plies of most games into free lookups. node_count is part of
the key so a deep request never gets a shallow hit.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from blunderless.analysis.engine import LineEval, PositionAnalysis
from blunderless.db.models import PositionEval


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _to_rows(analysis: PositionAnalysis) -> list[dict]:
    return [
        {"move": ln.move, "eval_cp": ln.eval_cp, "mate_in": ln.mate_in, "pv": ln.pv}
        for ln in analysis.lines
    ]


def _from_row(row: PositionEval, key: str) -> PositionAnalysis:
    lines = [
        LineEval(move=ln["move"], eval_cp=ln["eval_cp"], mate_in=ln["mate_in"], pv=ln["pv"])
        for ln in (row.pv_moves or [])
    ]
    return PositionAnalysis(
        fen="", nodes=row.node_count, multipv=len(lines) or 1, lines=lines
    )


def get(db: Session, key: str, nodes: int, multipv: int) -> PositionAnalysis | None:
    row = db.get(PositionEval, (key, nodes))
    # A cached single-PV entry can't satisfy a MultiPV request.
    if row is None or len(row.pv_moves or []) < multipv:
        return None
    return _from_row(row, key)


def put(db: Session, key: str, analysis: PositionAnalysis) -> None:
    best = analysis.best
    stmt = insert(PositionEval).values(
        position_key=key,
        node_count=analysis.nodes,
        eval_cp=best.eval_cp,
        mate_in=best.mate_in,
        pv_moves=_to_rows(analysis),
    )
    # A MultiPV result may replace a single-PV entry at the same key
    # (it strictly contains more information); otherwise first write wins.
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=["position_key", "node_count"],
            set_={
                "pv_moves": stmt.excluded.pv_moves,
                "eval_cp": stmt.excluded.eval_cp,
                "mate_in": stmt.excluded.mate_in,
            },
            where=func.jsonb_array_length(PositionEval.pv_moves)
            < len(analysis.lines),
        )
    )


def get_many(db: Session, keys: list[str], nodes: int) -> dict[str, PositionAnalysis]:
    """Batch lookup for a whole game's positions in one query."""
    rows = db.execute(
        select(PositionEval).where(
            PositionEval.position_key.in_(keys), PositionEval.node_count == nodes
        )
    ).scalars()
    return {row.position_key: _from_row(row, row.position_key) for row in rows}
