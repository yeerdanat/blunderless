"""Hallucination validator for narrated explanations.

The narrator (LLM or template) may only mention squares and moves that
exist in its input facts. Anything else is a fabrication and the text is
rejected — the caller falls back to a deterministic template.
"""

from __future__ import annotations

import re
from typing import Any

SQUARE_RE = re.compile(r"\b([a-h][1-8])\b")
SAN_RE = re.compile(
    r"\b((?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|"
    r"[a-h]x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?)\b"
)
UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def _collect_strings(value: Any, out: set[str]) -> None:
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_strings(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_strings(v, out)


def allowed_tokens(facts: dict) -> set[str]:
    """Every square and move token derivable from the fact dict."""
    strings: set[str] = set()
    _collect_strings(facts, strings)
    tokens: set[str] = set()
    for s in strings:
        tokens.update(SQUARE_RE.findall(s))
        tokens.update(m.rstrip("+#") for m in SAN_RE.findall(s))
        for uci in UCI_RE.findall(s):
            tokens.add(uci)
            tokens.add(uci[:2])
            tokens.add(uci[2:4])
    return tokens


def validate(text: str, facts: dict) -> tuple[bool, list[str]]:
    """Check that the text references only squares/moves present in facts.

    Returns (ok, offending_tokens).
    """
    allowed = allowed_tokens(facts)
    mentioned: set[str] = set()
    mentioned.update(SQUARE_RE.findall(text))
    mentioned.update(m.rstrip("+#") for m in SAN_RE.findall(text))
    offending = sorted(t for t in mentioned if t not in allowed)
    return (not offending, offending)
