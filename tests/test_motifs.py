import chess

from blunderless.analysis.motifs import (
    MotifHit,
    _fork_in_pv,
    _hanging_after,
    _missed_mate,
    detect_motifs,
)


def test_fork_detected_in_pv():
    # Nd4-c6+ forks the king on b8 and queen on d8.
    board = chess.Board("1k1q4/8/8/8/3N4/8/8/K7 w - - 0 1")
    hit = _fork_in_pv(board, ["d4c6"])
    assert hit is not None
    assert hit.motif == "fork"
    targets = {sq for sq, _ in hit.detail["targets"]}
    assert {"b8", "d8"} <= targets


def test_no_fork_without_two_targets():
    board = chess.Board("1k6/8/8/8/3N4/8/8/K7 w - - 0 1")  # no queen to fork
    assert _fork_in_pv(board, ["d4c6"]) is None


def test_hanging_piece_confirmed_by_reply():
    # Rd5?? puts the rook on the a5-queen's rank; Qxd5 wins it.
    board = chess.Board("k7/8/8/q7/8/8/8/K2R4 w - - 0 1")
    move = chess.Move.from_uci("d1d5")
    hit = _hanging_after(board, move, reply_pv=["a5d5"])
    assert hit is not None
    assert hit.motif == "hanging_piece"
    assert hit.detail["square"] == "d5"
    # Without the confirming reply the confidence drops but it still fires.
    weak = _hanging_after(board, move, reply_pv=None)
    assert weak is not None and weak.confidence < hit.confidence


def test_hanging_not_flagged_when_reply_goes_elsewhere():
    board = chess.Board("k7/8/8/q7/8/8/8/K2R4 w - - 0 1")
    move = chess.Move.from_uci("d1d5")
    assert _hanging_after(board, move, reply_pv=["a5a4"]) is None


def test_missed_mate_logic():
    assert isinstance(_missed_mate(2, played_leads_to_mate=False), MotifHit)
    assert _missed_mate(2, played_leads_to_mate=True) is None  # still mating
    assert _missed_mate(None, played_leads_to_mate=False) is None
    assert _missed_mate(-3, played_leads_to_mate=False) is None  # opponent mates


def test_detect_motifs_is_multilabel_and_skips_played_best():
    board = chess.Board("1k1q4/8/8/8/3N4/8/8/K7 w - - 0 1")
    hits = detect_motifs(board, played_uci="a1b1", best_pv=["d4c6"], best_mate_in=None)
    assert "fork" in {h.motif for h in hits}
    # If the player actually played the engine move, nothing was "missed".
    assert detect_motifs(board, played_uci="d4c6", best_pv=["d4c6"], best_mate_in=None) == []
