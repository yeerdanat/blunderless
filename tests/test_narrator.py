from blunderless.narrate.narrator import narrate, render_template
from blunderless.narrate.validator import validate

FACTS = {
    "player_move_san": "Nf3",
    "best_move_san": "Bxf7+",
    "win_prob_before": 0.54,
    "win_prob_after": 0.21,
    "delta_win_prob": 0.33,
    "motif": "fork",
    "detail": {"forking_move": "d5e6", "targets": [["g8", "K"], ["d8", "Q"]]},
    "phase": "middlegame",
}


def test_validator_accepts_grounded_text():
    ok, offending = validate(
        "Nf3 missed the fork Bxf7+ hitting the king on g8 and queen on d8.", FACTS
    )
    assert ok, offending


def test_validator_rejects_invented_square():
    ok, offending = validate("Your knight on h5 was the problem.", FACTS)
    assert not ok
    assert "h5" in offending


def test_validator_rejects_invented_move():
    ok, offending = validate("You should have played Qxb7 instead.", FACTS)
    assert not ok
    assert "Qxb7" in offending


def test_validator_uci_tokens_are_grounded():
    # d5e6 appears in facts as UCI; both endpoint squares are fair game
    ok, _ = validate("The fork lands on e6, coming from d5.", FACTS)
    assert ok


def test_template_rendering():
    text = render_template(FACTS)
    assert "Nf3" in text
    assert "Bxf7+" in text
    assert "33%" in text


def test_narrate_without_api_key_uses_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text, source = narrate(FACTS)
    assert source == "template"
    assert "Nf3" in text
