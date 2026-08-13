import pytest

from blunderless.analysis.engine import Engine, EngineNotFoundError, find_stockfish

try:
    find_stockfish()
    HAVE_STOCKFISH = True
except EngineNotFoundError:
    HAVE_STOCKFISH = False

pytestmark = pytest.mark.skipif(not HAVE_STOCKFISH, reason="stockfish binary not installed")

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# White to move: Ra8 is mate (lone black king on g8 boxed in by its own pawns).
MATE_IN_1 = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"
# Black to move and mates: back rank mirror.
BLACK_MATES = "r3k3/8/8/8/8/8/5PPP/6K1 b - - 0 1"


@pytest.fixture(scope="module")
def engine():
    with Engine() as eng:
        yield eng


def test_startpos_is_roughly_equal(engine):
    result = engine.analyse(STARTPOS, nodes=50_000)
    assert result.best.mate_in is None
    assert result.best.eval_cp is not None
    assert abs(result.best.eval_cp) < 150


def test_finds_mate_in_one(engine):
    result = engine.analyse(MATE_IN_1, nodes=20_000)
    assert result.best.move == "a1a8"
    assert result.best.mate_in == 1  # positive: White mates


def test_scores_are_white_pov(engine):
    result = engine.analyse(BLACK_MATES, nodes=20_000)
    assert result.best.mate_in == -1  # negative: Black mates


def test_multipv_returns_ordered_lines(engine):
    result = engine.analyse(STARTPOS, nodes=50_000, multipv=4)
    assert len(result.lines) == 4
    evals = [line.eval_cp for line in result.lines]
    assert all(cp is not None for cp in evals)
    assert evals == sorted(evals, reverse=True)  # best-first, White POV
    assert len({line.move for line in result.lines}) == 4  # distinct first moves


def test_missing_binary_raises():
    with pytest.raises(EngineNotFoundError):
        find_stockfish("/nonexistent/stockfish")
