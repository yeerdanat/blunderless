"""PGN utilities shared by both platform ingestors."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import chess.pgn

_GAME_START = re.compile(r"^\[Event ", re.MULTILINE)


def split_pgns(text: str) -> list[str]:
    """Split a multi-game PGN stream into individual game strings."""
    starts = [m.start() for m in _GAME_START.finditer(text)]
    return [
        text[start:end].strip()
        for start, end in zip(starts, [*starts[1:], len(text)], strict=False)
        if text[start:end].strip()
    ]


def move_clocks(pgn: str) -> list[float | None]:
    """Per-ply clock readings in seconds, from [%clk ...] annotations.

    Index i is the clock of the player who made ply i (0-based) *after*
    moving. None where the annotation is absent.
    """
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return []
    return [node.clock() for node in game.mainline()]


@dataclass(frozen=True)
class GameRecord:
    """Normalized form of one game, ready for the `game` table."""

    platform: str
    platform_game_id: str
    played_at: datetime | None
    time_control: str | None
    player_color: str
    result: str | None
    opponent_rating: int | None
    eco: str | None
    pgn: str


def _game_id_from_url(url: str) -> str | None:
    # https://lichess.org/AbCdEfGh  /  https://www.chess.com/game/live/123456
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def _parse_played_at(headers: chess.pgn.Headers) -> datetime | None:
    date = headers.get("UTCDate", headers.get("Date", ""))
    time = headers.get("UTCTime", "00:00:00")
    try:
        return datetime.strptime(f"{date} {time}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def normalize_pgn(pgn: str, platform: str, username: str) -> GameRecord | None:
    """Extract the `game` table fields for `username`'s side of one PGN.

    Returns None for games that can't be attributed (username on neither
    side) or that lack a usable game ID.
    """
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    headers = game.headers

    white = headers.get("White", "").lower()
    black = headers.get("Black", "").lower()
    if username.lower() == white:
        color, opponent_elo = "white", headers.get("BlackElo")
    elif username.lower() == black:
        color, opponent_elo = "black", headers.get("WhiteElo")
    else:
        return None

    game_id = _game_id_from_url(headers.get("Site", "") or headers.get("Link", ""))
    if not game_id:
        return None

    raw_result = headers.get("Result", "*")
    if raw_result == "1/2-1/2":
        result = "draw"
    elif raw_result in ("1-0", "0-1"):
        won_as = "white" if raw_result == "1-0" else "black"
        result = "win" if color == won_as else "loss"
    else:
        result = None

    return GameRecord(
        platform=platform,
        platform_game_id=game_id,
        played_at=_parse_played_at(headers),
        time_control=headers.get("TimeControl") or None,
        player_color=color,
        result=result,
        opponent_rating=_int_or_none(opponent_elo),
        eco=headers.get("ECO") or None,
        pgn=pgn.strip(),
    )
