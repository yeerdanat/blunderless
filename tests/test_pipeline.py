"""End-to-end pipeline test: a real engine analyzing a famous blunder.

Scholar's mate — 3...Nf6?? allows 4.Qxf7#. The pipeline must classify
Black's 3rd move as a blunder, mark the opening moves as book, and write
MoveAnalysis rows for Black's moves only.
"""

from datetime import UTC, datetime

import pytest
import sqlalchemy
from sqlalchemy import select

from blunderless.analysis.engine import Engine, EngineNotFoundError, find_stockfish
from blunderless.analysis.pipeline import analyze_game
from blunderless.db.models import Game, MoveAnalysis, Player
from blunderless.db.session import make_engine, make_session_factory

SCHOLARS_MATE = """[Event "test"]
[Site "https://lichess.org/pipetest1"]
[White "someone"]
[Black "pipeline_test_user"]
[Result "1-0"]
[TimeControl "300+0"]

1. e4 { [%clk 0:05:00] } e5 { [%clk 0:05:00] } 2. Bc4 { [%clk 0:04:58] } \
Nc6 { [%clk 0:04:57] } 3. Qh5 { [%clk 0:04:56] } Nf6 { [%clk 0:04:50] } \
4. Qxf7# { [%clk 0:04:55] } 1-0
"""


def _ready() -> bool:
    try:
        find_stockfish()
        with make_engine().connect():
            return True
    except (EngineNotFoundError, sqlalchemy.exc.OperationalError):
        return False


pytestmark = pytest.mark.skipif(not _ready(), reason="needs stockfish + postgres")


@pytest.fixture
def game():
    sessions = make_session_factory()
    with sessions() as db:
        player = Player(platform="lichess", username="pipeline_test_user")
        db.add(player)
        db.flush()
        game = Game(
            player_id=player.id,
            platform_game_id="pipetest1",
            played_at=datetime.now(UTC),
            time_control="300+0",
            player_color="black",
            result="loss",
            pgn=SCHOLARS_MATE,
        )
        db.add(game)
        db.commit()
        yield game
        db.execute(
            sqlalchemy.delete(MoveAnalysis).where(MoveAnalysis.game_id == game.id)
        )
        db.delete(db.get(Game, game.id))
        db.delete(db.get(Player, player.id))
        db.commit()


def test_scholars_mate_blunder_detected(game):
    sessions = make_session_factory()
    with sessions() as db, Engine() as engine:
        stats = analyze_game(db, engine, db.get(Game, game.id))

        assert stats.plies == 7
        assert stats.player_moves == 3  # black's moves only

        rows = {
            row.ply: row
            for row in db.execute(
                select(MoveAnalysis).where(MoveAnalysis.game_id == game.id)
            ).scalars()
        }
        assert set(rows) == {2, 4, 6}  # plies of black's moves

        # 1...e5 is theory
        assert rows[2].was_book

        # 3...Nf6?? hangs mate — must classify as a blunder with the clock read
        blunder = rows[6]
        assert blunder.san == "Nf6"
        assert blunder.severity == "blunder"
        assert blunder.delta_win_prob > 0.3
        assert blunder.best_move is not None
        assert blunder.clock_remaining_s == 290.0

        # engine evals were cached along the way
        assert stats.engine_calls_pass1 + stats.cache_hits > 0
