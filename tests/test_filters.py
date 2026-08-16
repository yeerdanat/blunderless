import chess
import pytest

from blunderless.analysis.book import book_keys, is_book_position
from blunderless.analysis.filters import (
    is_forced,
    phase,
    should_classify,
    stayed_decided,
    time_bucket,
)
from blunderless.analysis.keys import position_key


def test_position_key_ignores_move_counters():
    a = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 40")
    assert position_key(a) == position_key(b)


def test_position_key_distinguishes_side_to_move():
    a = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    c = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert position_key(a) != position_key(c)


def test_book_contains_mainlines_not_garbage():
    assert len(book_keys()) > 5000
    board = chess.Board()
    assert is_book_position(board)  # startpos
    board.push_san("e4")
    assert is_book_position(board)  # 1.e4 is theory
    # A random shuffled middlegame is not theory.
    assert not is_book_position(
        chess.Board("r4rk1/1pp2ppp/p1np1n2/4p3/B3P1b1/2PP1N2/PP3PPP/RNBQR1K1 w - - 0 10")
    )


def test_forced_single_legal_move():
    # Kh8, white Qg7 undefended: Kxg7 is the only legal move.
    board = chess.Board("7k/6Q1/8/8/8/8/8/K7 b - - 0 1")
    assert board.legal_moves.count() == 1
    assert is_forced(board)
    assert not is_forced(chess.Board())


def test_phase_transitions():
    assert phase(chess.Board(), ply=1, in_book=True) == "opening"
    # Heavy material, past the opening plies -> middlegame
    mid = chess.Board("r4rk1/1pp2ppp/p1np1n2/4p3/B3P1b1/2PP1N2/PP3PPP/RNBQR1K1 w - - 0 10")
    assert phase(mid, ply=30, in_book=False) == "middlegame"
    # K+R vs K -> endgame
    end = chess.Board("8/8/8/4k3/8/8/8/4K2R w K - 0 1")
    assert phase(end, ply=60, in_book=False) == "endgame"


def test_time_buckets():
    assert time_bucket(None) == "unknown"
    assert time_bucket(4.0) == "scramble"
    assert time_bucket(45.0) == "lt60s"
    assert time_bucket(300.0) == "normal"


def test_stayed_decided_excludes_only_unchanged_outcomes():
    assert stayed_decided(0.98, 0.97)  # totally winning, still winning
    assert stayed_decided(0.02, 0.03)  # dead lost, still lost
    assert not stayed_decided(0.98, 0.60)  # threw away a won game — classify!
    assert not stayed_decided(1.0, 0.40)  # missed mate — classify!
    assert not stayed_decided(0.55, 0.50)  # normal position


@pytest.mark.parametrize(
    "in_book,forced,wp_before,wp_after,bucket,expected",
    [
        (True, False, 0.5, 0.4, "normal", False),  # book move
        (False, True, 0.5, 0.4, "normal", False),  # forced move
        (False, False, 0.5, 0.4, "scramble", False),  # time scramble
        (False, False, 0.98, 0.97, "normal", False),  # stayed decided
        (False, False, 0.98, 0.60, "normal", True),  # threw away a win
        (False, False, 0.5, 0.4, "lt60s", True),  # normal decision
    ],
)
def test_should_classify(in_book, forced, wp_before, wp_after, bucket, expected):
    result = should_classify(
        in_book=in_book,
        forced=forced,
        wp_before_mover=wp_before,
        wp_after_mover=wp_after,
        bucket=bucket,
    )
    assert result is expected
