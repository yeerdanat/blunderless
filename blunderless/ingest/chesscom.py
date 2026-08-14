"""Chess.com public API — monthly archive endpoints, no auth required."""

from __future__ import annotations

from collections.abc import Iterator

import httpx

API = "https://api.chess.com/pub"
# Chess.com rejects requests without a real User-Agent.
HEADERS = {"User-Agent": "blunderless/0.1 (chess weakness analysis; contact via github)"}


def fetch_games_pgn(
    username: str,
    *,
    max_games: int | None = None,
    client: httpx.Client | None = None,
) -> Iterator[str]:
    """Yield individual game PGNs, newest first."""
    own_client = client is None
    client = client or httpx.Client(timeout=60, headers=HEADERS)
    try:
        resp = client.get(f"{API}/player/{username}/games/archives")
        resp.raise_for_status()
        yielded = 0
        for archive_url in reversed(resp.json()["archives"]):  # newest month first
            resp = client.get(archive_url)
            resp.raise_for_status()
            for game in reversed(resp.json().get("games", [])):
                pgn = game.get("pgn")
                if not pgn:
                    continue  # e.g. abandoned games have no movetext
                yield pgn
                yielded += 1
                if max_games is not None and yielded >= max_games:
                    return
    finally:
        if own_client:
            client.close()
