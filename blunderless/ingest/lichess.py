"""Lichess game export — one streaming PGN request for the full history."""

from __future__ import annotations

import time

import httpx

EXPORT_URL = "https://lichess.org/api/games/user/{username}"
HEADERS = {
    "Accept": "application/x-chess-pgn",
    "User-Agent": "blunderless/0.1 (chess weakness analysis)",
}
_MAX_RETRIES = 3


def fetch_games_pgn(
    username: str,
    *,
    max_games: int | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Full game history as one multi-game PGN string, newest first.

    clocks=true embeds [%clk] per move; opening=true adds ECO headers.
    Lichess answers 429 with a mandatory 60s cool-off.
    """
    params: dict[str, str | int] = {"clocks": "true", "opening": "true", "moves": "true"}
    if max_games is not None:
        params["max"] = max_games

    own_client = client is None
    client = client or httpx.Client(timeout=120)
    try:
        for attempt in range(_MAX_RETRIES):
            resp = client.get(
                EXPORT_URL.format(username=username),
                params=params,
                headers=HEADERS,
            )
            if resp.status_code == 429:
                if attempt == _MAX_RETRIES - 1:
                    break
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.text
        resp.raise_for_status()
        return resp.text
    finally:
        if own_client:
            client.close()
