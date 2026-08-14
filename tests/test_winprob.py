import pytest

from blunderless.analysis.winprob import (
    Severity,
    classify,
    cp_to_win_prob,
    delta_win_prob,
    eval_to_win_prob,
    pov,
)


def test_sigmoid_anchors():
    assert cp_to_win_prob(0) == pytest.approx(0.5)
    assert cp_to_win_prob(1000) > 0.95
    assert cp_to_win_prob(-1000) < 0.05
    # Symmetry: P(+cp) + P(-cp) == 1
    assert cp_to_win_prob(300) + cp_to_win_prob(-300) == pytest.approx(1.0)


def test_same_cp_drop_different_meaning():
    """The reason this module exists: 300cp means opposite things at
    +50 and +1200."""
    drawn_to_lost = cp_to_win_prob(50) - cp_to_win_prob(-250)
    winning_to_winning = cp_to_win_prob(1200) - cp_to_win_prob(900)
    assert drawn_to_lost > 0.25  # a real blunder
    assert winning_to_winning < 0.03  # nothing happened
    assert classify(drawn_to_lost) == Severity.BLUNDER
    assert classify(winning_to_winning) == Severity.FINE


def test_mate_scores_are_certainties():
    assert eval_to_win_prob(None, mate_in=1) == 1.0
    assert eval_to_win_prob(None, mate_in=12) == 1.0
    assert eval_to_win_prob(None, mate_in=-3) == 0.0
    with pytest.raises(ValueError):
        eval_to_win_prob(None, None)


def test_pov_flip():
    assert pov(0.7, "white") == pytest.approx(0.7)
    assert pov(0.7, "black") == pytest.approx(0.3)
    with pytest.raises(ValueError):
        pov(0.5, "green")


def test_delta_is_mover_relative():
    # White blunders: White-POV 0.6 -> 0.2.
    assert delta_win_prob(0.6, 0.2, "white") == pytest.approx(0.4)
    # Black blunders: White-POV probability *rises* 0.4 -> 0.8.
    assert delta_win_prob(0.4, 0.8, "black") == pytest.approx(0.4)
    # The same White-POV rise is no loss at all from White's side.
    assert delta_win_prob(0.4, 0.8, "white") == 0.0


def test_negative_delta_clamped():
    # "Better than best" is engine noise, not brilliance.
    assert delta_win_prob(0.5, 0.55, "white") == 0.0


def test_severity_thresholds():
    assert classify(0.25) == Severity.BLUNDER
    assert classify(0.20) == Severity.BLUNDER  # boundary inclusive
    assert classify(0.15) == Severity.MISTAKE
    assert classify(0.10) == Severity.MISTAKE
    assert classify(0.07) == Severity.INACCURACY
    assert classify(0.05) == Severity.INACCURACY
    assert classify(0.049) == Severity.FINE
    assert classify(0.0) == Severity.FINE


def test_decided_positions_produce_no_flags():
    """A 400cp swing in a totally won game stays 'fine' — the #1 noise
    source in naive implementations."""
    delta = delta_win_prob(cp_to_win_prob(1500), cp_to_win_prob(1100), "white")
    assert classify(delta) == Severity.FINE
