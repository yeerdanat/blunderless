"""Parallel cohort-baseline build over the sampled dump files."""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from blunderless.analysis.engine import Engine
from blunderless.cohort.build import (
    COHORT_DIR,
    CellCounts,
    process_game,
    split_cell_file,
    store_baseline,
)
from blunderless.db.session import make_session_factory

CHUNK = 150  # games per worker task; each task owns one engine lifetime


def _process_chunk(pgns: list[str], with_motifs: bool) -> tuple[Counter, Counter]:
    counts = CellCounts()
    if with_motifs:
        with Engine() as engine:
            for pgn in pgns:
                process_game(pgn, counts, engine)
    else:
        for pgn in pgns:
            process_game(pgn, counts, None)
    return counts.eligible, counts.errors


def build_all(
    *,
    workers: int,
    games_per_cell: int | None = None,
    with_motifs: bool = True,
    cells: list[str] | None = None,
    progress=print,
) -> dict[str, int]:
    sessions = make_session_factory()
    results: dict[str, int] = {}
    for path in sorted(COHORT_DIR.glob("*.pgn")):
        if cells is not None and path.stem not in cells:
            continue
        tc, band = path.stem.split("_")  # e.g. blitz_1200-1400
        started = time.monotonic()
        pgns = split_cell_file(path)
        if games_per_cell is not None:
            pgns = pgns[:games_per_cell]
        chunks = [pgns[i : i + CHUNK] for i in range(0, len(pgns), CHUNK)]

        merged = CellCounts()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_chunk, c, with_motifs) for c in chunks]
            for i, fut in enumerate(as_completed(futures), 1):
                eligible, errors = fut.result()
                merged.eligible.update(eligible)
                merged.errors.update(errors)
                progress(f"  {path.stem}: chunk {i}/{len(chunks)}")

        with sessions() as db:
            written = store_baseline(db, band, tc, merged)
        results[path.stem] = written
        progress(
            f"{path.stem}: games={len(pgns)} eligible_moves={sum(merged.eligible.values())} "
            f"rows={written} in {time.monotonic() - started:.0f}s"
        )
    return results
