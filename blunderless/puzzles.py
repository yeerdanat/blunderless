"""Puzzle recommendations from the Lichess puzzle database (§9).

A one-time index pass extracts the top puzzles per motif theme into JSON
(the full CSV is ~5M rows; scanning it per request is absurd). Difficulty
targets ~70% expected success: by the Elo expectation formula that means
puzzles rated ~150 below the player.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA / "puzzle_index.json"

MOTIF_TO_THEMES = {
    "fork": ["fork"],
    "pin": ["pin"],
    "skewer": ["skewer"],
    "back_rank": ["backRankMate"],
    "discovered_attack": ["discoveredAttack"],
    "hanging_piece": ["hangingPiece"],
    "missed_mate": ["mateIn1", "mateIn2"],
}
TARGET_OFFSET = -150  # ~70% expected success
WINDOW = 150
PER_MOTIF = 3000
MIN_POPULARITY = 70


def build_index(csv_path: Path | None = None) -> dict[str, int]:
    """Scan the puzzle CSV once; keep the most popular per motif."""
    csv_path = csv_path or DATA / "puzzles.csv"
    buckets: dict[str, list[dict]] = {m: [] for m in MOTIF_TO_THEMES}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                popularity = int(row["Popularity"])
                rating = int(row["Rating"])
            except ValueError:
                continue
            if popularity < MIN_POPULARITY or not 600 <= rating <= 2400:
                continue
            themes = set(row["Themes"].split())
            for motif, wanted in MOTIF_TO_THEMES.items():
                if any(t in themes for t in wanted):
                    buckets[motif].append(
                        {
                            "id": row["PuzzleId"],
                            "rating": rating,
                            "popularity": popularity,
                            "url": f"https://lichess.org/training/{row['PuzzleId']}",
                        }
                    )
    for motif, rows in buckets.items():
        rows.sort(key=lambda r: -r["popularity"])
        buckets[motif] = rows[:PER_MOTIF]
    INDEX_PATH.write_text(json.dumps(buckets))
    return {m: len(rows) for m, rows in buckets.items()}


def recommend(motif: str, player_rating: int, limit: int = 6) -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    index = json.loads(INDEX_PATH.read_text())
    rows = index.get(motif, [])
    target = player_rating + TARGET_OFFSET
    in_window = [r for r in rows if abs(r["rating"] - target) <= WINDOW]
    in_window.sort(key=lambda r: (abs(r["rating"] - target), -r["popularity"]))
    return in_window[:limit]
