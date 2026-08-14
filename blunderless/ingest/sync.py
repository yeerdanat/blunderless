"""Pull a player's history from a platform and upsert it into the DB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from blunderless.db.models import Game, Player
from blunderless.ingest import chesscom, lichess
from blunderless.ingest.pgn import GameRecord, normalize_pgn, split_pgns


@dataclass
class SyncStats:
    fetched: int = 0
    inserted: int = 0
    skipped_unparseable: int = 0


def _fetch_records(
    platform: str,
    username: str,
    max_games: int | None,
    client: httpx.Client | None,
) -> tuple[list[GameRecord], int]:
    if platform == "lichess":
        pgns = split_pgns(lichess.fetch_games_pgn(username, max_games=max_games, client=client))
    elif platform == "chesscom":
        pgns = list(chesscom.fetch_games_pgn(username, max_games=max_games, client=client))
    else:
        raise ValueError(f"unknown platform: {platform!r}")

    records = []
    unparseable = 0
    for pgn in pgns:
        record = normalize_pgn(pgn, platform, username)
        if record is None:
            unparseable += 1
        else:
            records.append(record)
    return records, unparseable


def _get_or_create_player(db: Session, platform: str, username: str) -> Player:
    player = db.execute(
        select(Player).where(Player.platform == platform, Player.username == username)
    ).scalar_one_or_none()
    if player is None:
        player = Player(platform=platform, username=username)
        db.add(player)
        db.flush()
    return player


def sync_player(
    platform: str,
    username: str,
    session_factory: sessionmaker[Session],
    *,
    max_games: int | None = None,
    client: httpx.Client | None = None,
) -> SyncStats:
    records, unparseable = _fetch_records(platform, username, max_games, client)
    stats = SyncStats(fetched=len(records) + unparseable, skipped_unparseable=unparseable)

    with session_factory() as db:
        player = _get_or_create_player(db, platform, username)
        for record in records:
            inserted_id = db.execute(
                insert(Game)
                .values(
                    player_id=player.id,
                    platform_game_id=record.platform_game_id,
                    played_at=record.played_at,
                    time_control=record.time_control,
                    player_color=record.player_color,
                    result=record.result,
                    opponent_rating=record.opponent_rating,
                    eco=record.eco,
                    pgn=record.pgn,
                )
                .on_conflict_do_nothing(index_elements=["player_id", "platform_game_id"])
                .returning(Game.id)
            ).scalar_one_or_none()
            stats.inserted += inserted_id is not None
        player.last_synced_at = datetime.now(UTC)
        db.commit()
    return stats
