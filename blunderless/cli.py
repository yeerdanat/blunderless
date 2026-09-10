import argparse
import time

from blunderless.db.session import make_session_factory
from blunderless.ingest.sync import sync_player


def main() -> None:
    parser = argparse.ArgumentParser(prog="blunderless")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="import a player's game history")
    sync.add_argument("platform", choices=["lichess", "chesscom"])
    sync.add_argument("username")
    sync.add_argument("--max", type=int, default=None, help="limit number of games")

    analyze = sub.add_parser("analyze", help="run engine analysis over a player's games")
    analyze.add_argument("platform", choices=["lichess", "chesscom"])
    analyze.add_argument("username")
    analyze.add_argument("--workers", type=int, default=None)
    analyze.add_argument("--limit", type=int, default=None, help="only first N games")
    analyze.add_argument(
        "--naive", action="store_true",
        help="disable two-pass tiering (benchmark baseline: deep MultiPV everywhere)",
    )

    args = parser.parse_args()
    if args.command == "sync":
        stats = sync_player(
            args.platform, args.username, make_session_factory(), max_games=args.max
        )
        print(
            f"fetched={stats.fetched} inserted={stats.inserted} "
            f"skipped_unparseable={stats.skipped_unparseable}"
        )
    elif args.command == "analyze":
        from blunderless.analysis.runner import analyze_player

        started = time.monotonic()

        def report(done: int, total: int, _stats) -> None:
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total} games", flush=True)

        total = analyze_player(
            args.platform,
            args.username,
            workers=args.workers,
            naive=args.naive,
            limit=args.limit,
            progress=report,
        )
        wall = time.monotonic() - started
        cached_pct = 100 * total.cache_hits / max(
            1, total.cache_hits + total.engine_calls_pass1 + total.engine_calls_pass2
        )
        print(
            f"wall={wall:.1f}s plies={total.plies} player_moves={total.player_moves} "
            f"pass1_calls={total.engine_calls_pass1} pass2_calls={total.engine_calls_pass2} "
            f"cache_hits={total.cache_hits} ({cached_pct:.1f}%) "
            f"control={total.control_samples}"
        )


if __name__ == "__main__":
    main()
