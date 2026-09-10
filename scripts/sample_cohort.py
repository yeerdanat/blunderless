"""Stream-filter the Lichess monthly dump for cohort-baseline games.

Reads decompressed PGN text on stdin (curl | zstd -dc | this), keeps rated
blitz/rapid games where both players sit in a target rating band AND the
game carries [%eval] annotations, and writes them to per-cell files under
data/cohort/. Exits once every cell quota is met (SIGPIPE stops upstream).

Usage: zstd -dc dump.pgn.zst | python scripts/sample_cohort.py [quota_per_cell]
"""

import re
import sys
from pathlib import Path

TCS = ("blitz", "rapid")
QUOTA = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
# optional second arg: comma-separated bands, e.g. "600-800,800-1000"
if len(sys.argv) > 2:
    BANDS = [tuple(map(int, b.split("-"))) for b in sys.argv[2].split(",")]
else:
    BANDS = [(1000, 1200), (1200, 1400), (1400, 1600)]
SCAN_CAP = 6_000_000  # absolute safety stop, games scanned

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "cohort"
ELO_RE = re.compile(r'\[(White|Black)Elo "(\d+)"\]')
EVENT_RE = re.compile(r'\[Event "[^"]*(Blitz|Rapid)[^"]*"\]', re.IGNORECASE)


def cell_for(game: str) -> str | None:
    event = EVENT_RE.search(game)
    if not event:
        return None
    if "%eval" not in game:
        return None
    elos = {m.group(1): int(m.group(2)) for m in ELO_RE.finditer(game)}
    if len(elos) != 2:
        return None
    lo, hi = min(elos.values()), max(elos.values())
    for band_lo, band_hi in BANDS:
        if band_lo <= lo and hi < band_hi:
            return f"{event.group(1).lower()}_{band_lo}-{band_hi}"
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = [f"{tc}_{lo}-{hi}" for tc in TCS for lo, hi in BANDS]
    counts = dict.fromkeys(cells, 0)
    files = {c: open(OUT_DIR / f"{c}.pgn", "w") for c in cells}

    scanned = 0
    buf: list[str] = []
    in_moves = False
    for line in sys.stdin:
        if line.startswith("[Event ") and in_moves:
            game = "".join(buf)
            scanned += 1
            cell = cell_for(game)
            if cell is not None and counts[cell] < QUOTA:
                files[cell].write(game + "\n")
                counts[cell] += 1
            buf, in_moves = [], False
            if scanned % 200_000 == 0:
                print(f"scanned={scanned} {counts}", file=sys.stderr, flush=True)
            if all(n >= QUOTA for n in counts.values()) or scanned >= SCAN_CAP:
                break
        buf.append(line)
        if not line.startswith("[") and line.strip():
            in_moves = True

    for f in files.values():
        f.close()
    print(f"DONE scanned={scanned} {counts}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
