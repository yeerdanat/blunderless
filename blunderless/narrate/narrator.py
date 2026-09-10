"""Constrained narration: engine-derived facts in, one or two sentences out.

The LLM never sees a board and never judges a move — it renders structured
facts as prose. Output is validated against the input (validator.py); any
reference to a square or move not in the facts is rejected and we fall
back to a deterministic template. With no API key configured, templates
are used directly — the system is fully functional without the LLM.
"""

from __future__ import annotations

import json
import os

from blunderless.narrate.validator import validate

NARRATOR_MODEL = os.environ.get("BLUNDERLESS_NARRATOR_MODEL", "claude-opus-5")

SYSTEM = (
    "You turn chess analysis facts into one or two plain-English coaching "
    "sentences for a club player. Rules: mention ONLY squares, moves, and "
    "pieces that appear in the provided facts. Never invent variations, "
    "squares, or evaluations. Never judge the position yourself - the "
    "facts are the judgment. Address the player as 'you'. Output only the "
    "sentence(s), no preamble."
)

TEMPLATES = {
    "hanging_piece": "{san} left a piece hanging{square_clause} — {delta_pct}% of your "
    "winning chances gone. The engine preferred {best_san}.",
    "fork": "After {san} you missed a fork with {best_san}, hitting multiple targets "
    "at once. That cost about {delta_pct}% in win probability.",
    "pin": "{san} overlooked a pin available with {best_san}; the position was worth "
    "about {delta_pct}% more with it.",
    "skewer": "{san} missed a skewer starting with {best_san} — about {delta_pct}% of "
    "win probability left on the table.",
    "back_rank": "{san} missed a back-rank tactic beginning with {best_san}. Your "
    "opponent's king had no escape squares — this was worth {delta_pct}%.",
    "discovered_attack": "{san} passed up a discovered attack with {best_san}, "
    "costing roughly {delta_pct}% in winning chances.",
    "missed_mate": "{san} missed a forced mate starting with {best_san}. "
    "The game was yours.",
    None: "{san} lost about {delta_pct}% of your win probability; the engine "
    "preferred {best_san}.",
}


def render_template(facts: dict) -> str:
    motif = facts.get("motif")
    template = TEMPLATES.get(motif, TEMPLATES[None])
    square = facts.get("detail", {}).get("square")
    return template.format(
        san=facts.get("player_move_san", "your move"),
        best_san=facts.get("best_move_san", "another move"),
        delta_pct=round(100 * facts.get("delta_win_prob", 0)),
        square_clause=f" on {square}" if square else "",
    )


def _llm_narrate(facts: dict) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=NARRATOR_MODEL,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(facts, indent=2)}],
        )
        if response.stop_reason == "refusal":
            return None
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception:
        return None  # any API failure degrades to the template


def narrate(facts: dict) -> tuple[str, str]:
    """Returns (text, source) where source is 'llm' or 'template'.

    LLM output that mentions any square or move absent from the facts is
    rejected — that's the hallucination gate.
    """
    text = _llm_narrate(facts)
    if text is not None:
        ok, _offending = validate(text, facts)
        if ok:
            return text.strip(), "llm"
    return render_template(facts), "template"
