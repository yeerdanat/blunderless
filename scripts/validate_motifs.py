"""Measure motif-detector precision/recall against Lichess puzzle themes.

The puzzle DB is community/engine-labeled ground truth: each puzzle has
theme tags (fork, pin, backRankMate, ...). Puzzle format: FEN is the
position BEFORE the setup move; Moves[0] is the setup (often the blunder),
the solution starts at Moves[1]. So the position our detectors see is
FEN+Moves[0], the "best PV" is Moves[1:], and the setup move is the
"played move" for hanging-piece detection.

Notes on the numbers: recall = of puzzles tagged T, how many we flag as T.
Precision here is a *lower bound* — puzzle themes aren't exhaustive, so a
correct detection on an untagged puzzle counts against us.

Usage: python scripts/validate_motifs.py [n_per_theme]
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blunderless.analysis.motifs import (  # noqa: E402
    _advance,
    _back_rank,
    _discovered_attack_in_pv,
    _existing_pin_exploited,
    _fork_in_pv,
    _hanging_after,
    _pin_or_skewer_in_pv,
)

PUZZLES = Path(__file__).resolve().parent.parent / "data" / "puzzles.csv"

THEME_TO_MOTIF = {
    "fork": "fork",
    "pin": "pin",
    "skewer": "skewer",
    "backRankMate": "back_rank",
    "discoveredAttack": "discovered_attack",
    "hangingPiece": "hanging_piece",
    "mate": "missed_mate",
}
N_PER_THEME = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEED = 20260815


def mate_in_from_solution(board: chess.Board, solution: list[str]) -> int | None:
    b = board.copy(stack=False)
    try:
        for uci in solution:
            b.push(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return None
    if b.is_checkmate():
        return (len(solution) + 1) // 2
    return None


def main() -> None:
    rng = random.Random(SEED)
    # reservoir-sample puzzles per theme
    samples: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    with open(PUZZLES, newline="") as fh:
        for row in csv.DictReader(fh):
            themes = set(row["Themes"].split())
            for theme in THEME_TO_MOTIF:
                if theme not in themes:
                    continue
                seen[theme] += 1
                if len(samples[theme]) < N_PER_THEME:
                    samples[theme].append(row)
                else:
                    j = rng.randrange(seen[theme])
                    if j < N_PER_THEME:
                        samples[theme][j] = row

    # deduplicate rows across themes; keep full theme sets
    by_id: dict[str, dict] = {}
    for rows in samples.values():
        for row in rows:
            by_id[row["PuzzleId"]] = row

    tp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    detected_total: dict[str, int] = defaultdict(int)

    evaluated = 0
    for row in by_id.values():
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        if not moves:
            continue
        try:
            setup = chess.Move.from_uci(moves[0])
            if not board.is_legal(setup):
                continue
        except ValueError:
            continue
        solution = moves[1:]
        if not solution:
            continue
        board_after_setup = board.copy(stack=False)
        board_after_setup.push(setup)

        mate_in = mate_in_from_solution(board_after_setup, solution)
        # Detectors see what the pipeline would see: the position the
        # solver faces, the solution as the engine PV, and the setup move
        # as the "played" move (with the solution confirming what it hung).
        hits: set[str] = set()
        if (h := _hanging_after(board, setup, reply_pv=solution)) is not None:
            hits.add(h.motif)
        for detector in (
            _fork_in_pv,
            _pin_or_skewer_in_pv,
            _discovered_attack_in_pv,
            _existing_pin_exploited,
        ):
            if (h := detector(board_after_setup, solution)) is not None:
                hits.add(h.motif)
        if not hits:
            board_later = _advance(board_after_setup, solution, 2)
            if board_later is not None and len(solution) > 2:
                for detector in (
                    _fork_in_pv,
                    _pin_or_skewer_in_pv,
                    _discovered_attack_in_pv,
                ):
                    if (h := detector(board_later, solution[2:])) is not None:
                        hits.add(h.motif)
        if (h := _back_rank(board_after_setup, solution, mate_in)) is not None:
            hits.add(h.motif)
        if mate_in is not None:
            hits.add("missed_mate")  # mate confirmed by replaying the solution

        evaluated += 1
        themes = set(row["Themes"].split())
        for theme, motif in THEME_TO_MOTIF.items():
            has_label = theme in themes
            detected = motif in hits
            if detected:
                detected_total[motif] += 1
            if has_label and detected:
                tp[motif] += 1
            elif has_label and not detected:
                fn[motif] += 1
            elif detected and not has_label:
                fp[motif] += 1

    print(f"evaluated={evaluated} puzzles (themes sampled at {N_PER_THEME} each)\n")
    print(f"{'motif':<20}{'recall':>8}{'precision*':>12}{'n_labeled':>11}{'n_detected':>12}")
    for motif in THEME_TO_MOTIF.values():
        n_labeled = tp[motif] + fn[motif]
        recall = tp[motif] / n_labeled if n_labeled else float("nan")
        prec = (
            tp[motif] / detected_total[motif] if detected_total[motif] else float("nan")
        )
        print(
            f"{motif:<20}{recall:>8.2f}{prec:>12.2f}{n_labeled:>11}{detected_total[motif]:>12}"
        )
    print("\n* precision is a lower bound: puzzle theme tags are not exhaustive.")


if __name__ == "__main__":
    main()
