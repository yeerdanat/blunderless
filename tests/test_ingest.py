from pathlib import Path

import httpx
import pytest
import sqlalchemy
from sqlalchemy import func, select

from blunderless.db.models import Game, Player
from blunderless.db.session import make_engine, make_session_factory
from blunderless.ingest.pgn import move_clocks, normalize_pgn, split_pgns
from blunderless.ingest.sync import sync_player

FIXTURE = (Path(__file__).parent / "fixtures" / "lichess_two_games.pgn").read_text()


def test_split_pgns():
    games = split_pgns(FIXTURE)
    assert len(games) == 2
    assert games[0].startswith('[Event "Rated blitz game"]')
    assert games[0].endswith("1-0")
    assert "efgh5678" in games[1]


def test_normalize_as_white_win():
    record = normalize_pgn(split_pgns(FIXTURE)[0], "lichess", "yerdanat")
    assert record is not None
    assert record.platform_game_id == "abcd1234"
    assert record.player_color == "white"
    assert record.result == "win"
    assert record.opponent_rating == 1455
    assert record.eco == "B01"
    assert record.time_control == "300+3"
    assert record.played_at is not None
    assert record.played_at.hour == 18


def test_normalize_as_black_loss():
    record = normalize_pgn(split_pgns(FIXTURE)[1], "lichess", "yerdanat")
    assert record is not None
    assert record.player_color == "black"
    assert record.result == "loss"
    assert record.opponent_rating == 1502


def test_normalize_unrelated_player_returns_none():
    assert normalize_pgn(split_pgns(FIXTURE)[0], "lichess", "someone_else") is None


def test_move_clocks():
    clocks = move_clocks(split_pgns(FIXTURE)[0])
    assert clocks == [300.0, 300.0, 298.0, 297.0, 295.0, 290.0]


def _db_reachable() -> bool:
    try:
        with make_engine().connect():
            return True
    except sqlalchemy.exc.OperationalError:
        return False


@pytest.mark.skipif(not _db_reachable(), reason="postgres not reachable")
def test_sync_lichess_end_to_end_with_dedupe():
    # Fixture PGNs mention "yerdanat", but we sync under a throwaway username
    # so test data never mixes with a real player's rows. normalize_pgn only
    # cares that the username matches a side in the PGN — patch the fixture.
    username = "blunderless_test_user"
    fixture = FIXTURE.replace("yerdanat", username)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=fixture))
    client = httpx.Client(transport=transport)
    factory = make_session_factory()

    try:
        stats = sync_player("lichess", username, factory, client=client)
        assert stats.fetched == 2
        assert stats.inserted == 2

        # Second run: nothing new.
        stats = sync_player("lichess", username, factory, client=client)
        assert stats.fetched == 2
        assert stats.inserted == 0

        with factory() as db:
            player = db.execute(
                select(Player).where(Player.platform == "lichess", Player.username == username)
            ).scalar_one()
            n_games = db.execute(
                select(func.count()).select_from(Game).where(Game.player_id == player.id)
            ).scalar_one()
            assert n_games == 2
            assert player.last_synced_at is not None
    finally:
        with factory() as db:
            player = db.execute(
                select(Player).where(Player.platform == "lichess", Player.username == username)
            ).scalar_one_or_none()
            if player is not None:
                for game in player.games:
                    db.delete(game)
                db.delete(player)
                db.commit()
