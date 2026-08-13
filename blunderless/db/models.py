"""Schema for games, engine output, classified errors, and cohort statistics.

Naming conventions used throughout:
- position_key: first four FEN fields (placement, side to move, castling,
  en passant) — the full position identity for caching. Game/move number
  must never leak into it.
- All probabilities are win probabilities in [0, 1]; deltas are from the
  moving player's perspective.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "player"
    __table_args__ = (UniqueConstraint("platform", "username"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    platform: Mapped[str] = mapped_column(String(16))  # "chesscom" | "lichess"
    username: Mapped[str] = mapped_column(String(64))
    current_rating: Mapped[int | None] = mapped_column(Integer)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    games: Mapped[list[Game]] = relationship(back_populates="player")


class Game(Base):
    __tablename__ = "game"
    __table_args__ = (UniqueConstraint("player_id", "platform_game_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.id"))
    platform_game_id: Mapped[str] = mapped_column(String(64))
    played_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    time_control: Mapped[str | None] = mapped_column(String(32))  # e.g. "300+3"
    player_color: Mapped[str] = mapped_column(String(5))  # "white" | "black"
    result: Mapped[str | None] = mapped_column(String(8))  # "win" | "loss" | "draw"
    opponent_rating: Mapped[int | None] = mapped_column(Integer)
    eco: Mapped[str | None] = mapped_column(String(3))
    pgn: Mapped[str] = mapped_column(Text)

    player: Mapped[Player] = relationship(back_populates="games")
    moves: Mapped[list[MoveAnalysis]] = relationship(back_populates="game")


class PositionEval(Base):
    """The durable engine cache. PK includes node_count so a deep request
    never gets a shallow hit."""

    __tablename__ = "position_eval"

    position_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    node_count: Mapped[int] = mapped_column(Integer, primary_key=True)
    depth: Mapped[int | None] = mapped_column(Integer)
    eval_cp: Mapped[int | None] = mapped_column(Integer)
    mate_in: Mapped[int | None] = mapped_column(Integer)
    pv_moves: Mapped[list[Any] | None] = mapped_column(JSONB)  # MultiPV line list
    complexity_score: Mapped[float | None] = mapped_column(Float)


class MoveAnalysis(Base):
    __tablename__ = "move_analysis"
    __table_args__ = (UniqueConstraint("game_id", "ply"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("game.id"))
    ply: Mapped[int] = mapped_column(Integer)  # 1-based half-move index
    san: Mapped[str] = mapped_column(String(16))
    position_key: Mapped[str] = mapped_column(String(100))  # position *before* the move
    win_prob_before: Mapped[float | None] = mapped_column(Float)
    win_prob_after: Mapped[float | None] = mapped_column(Float)
    delta_win_prob: Mapped[float | None] = mapped_column(Float)
    severity: Mapped[str | None] = mapped_column(String(12))  # blunder|mistake|inaccuracy|fine
    best_move: Mapped[str | None] = mapped_column(String(8))  # UCI
    was_forced: Mapped[bool] = mapped_column(Boolean, default=False)
    was_book: Mapped[bool] = mapped_column(Boolean, default=False)
    phase: Mapped[str | None] = mapped_column(String(12))  # opening|middlegame|endgame
    clock_remaining_s: Mapped[float | None] = mapped_column(Float)
    complexity_score: Mapped[float | None] = mapped_column(Float)

    game: Mapped[Game] = relationship(back_populates="moves")
    motifs: Mapped[list[Motif]] = relationship(back_populates="move_analysis")


class Motif(Base):
    __tablename__ = "motif"

    id: Mapped[int] = mapped_column(primary_key=True)
    move_analysis_id: Mapped[int] = mapped_column(ForeignKey("move_analysis.id"))
    motif_type: Mapped[str] = mapped_column(String(32))  # fork, pin, back_rank, ...
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    move_analysis: Mapped[MoveAnalysis] = relationship(back_populates="motifs")


class CohortBaseline(Base):
    __tablename__ = "cohort_baseline"
    __table_args__ = (
        UniqueConstraint("rating_band", "time_control", "motif_type", "phase", "time_bucket"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    rating_band: Mapped[str] = mapped_column(String(16))  # e.g. "1400-1500"
    time_control: Mapped[str] = mapped_column(String(16))  # blitz|rapid|classical
    motif_type: Mapped[str] = mapped_column(String(32))
    phase: Mapped[str] = mapped_column(String(12))
    time_bucket: Mapped[str] = mapped_column(String(16))  # e.g. "lt60s"
    error_rate: Mapped[float] = mapped_column(Float)
    n_samples: Mapped[int] = mapped_column(Integer)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)


class Weakness(Base):
    __tablename__ = "weakness"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.id"))
    motif_type: Mapped[str] = mapped_column(String(32))
    phase: Mapped[str | None] = mapped_column(String(12))
    time_bucket: Mapped[str | None] = mapped_column(String(16))
    player_rate: Mapped[float] = mapped_column(Float)
    cohort_rate: Mapped[float] = mapped_column(Float)
    ratio: Mapped[float] = mapped_column(Float)
    p_value: Mapped[float] = mapped_column(Float)
    q_value: Mapped[float | None] = mapped_column(Float)  # BH-adjusted
    total_winprob_lost: Mapped[float] = mapped_column(Float)
    n_observations: Mapped[int] = mapped_column(Integer)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
