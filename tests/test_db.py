from datetime import UTC, datetime

import pytest
import sqlalchemy
from sqlalchemy import select

from blunderless.db.models import Game, Player, PositionEval
from blunderless.db.session import make_engine, make_session_factory


def _db_reachable() -> bool:
    try:
        with make_engine().connect():
            return True
    except sqlalchemy.exc.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="postgres not reachable")


@pytest.fixture(scope="module")
def sessions():
    from alembic.config import Config

    from alembic import command

    command.upgrade(Config("alembic.ini"), "head")
    return make_session_factory()


def test_player_game_roundtrip(sessions):
    with sessions() as db:
        player = Player(platform="lichess", username=f"test_{datetime.now(UTC).timestamp()}")
        db.add(player)
        db.flush()
        db.add(
            Game(
                player_id=player.id,
                platform_game_id="abc123",
                played_at=datetime.now(UTC),
                time_control="300+3",
                player_color="white",
                result="win",
                opponent_rating=1450,
                eco="B01",
                pgn="1. e4 d5 2. exd5 Qxd5 *",
            )
        )
        db.commit()

        loaded = db.execute(select(Player).where(Player.id == player.id)).scalar_one()
        assert loaded.games[0].platform_game_id == "abc123"
        assert loaded.games[0].result == "win"

        db.delete(loaded.games[0])
        db.delete(loaded)
        db.commit()


def test_position_eval_pk_includes_node_count(sessions):
    key = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    with sessions() as db:
        db.merge(PositionEval(position_key=key, node_count=10_000, eval_cp=30, pv_moves=[]))
        db.merge(PositionEval(position_key=key, node_count=100_000, eval_cp=25, pv_moves=[]))
        db.commit()

        rows = db.execute(
            select(PositionEval).where(PositionEval.position_key == key)
        ).scalars().all()
        assert len(rows) == 2  # same position, two depths — distinct cache entries

        for row in rows:
            db.delete(row)
        db.commit()
