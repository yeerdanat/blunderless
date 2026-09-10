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

    motifs = sub.add_parser("motifs", help="tag classified errors with tactical motifs")
    motifs.add_argument("platform", choices=["lichess", "chesscom"])
    motifs.add_argument("username")

    baseline = sub.add_parser("baseline", help="build cohort baselines from the dump sample")
    baseline.add_argument("--workers", type=int, default=6)
    baseline.add_argument("--games-per-cell", type=int, default=None)
    baseline.add_argument("--no-motifs", action="store_true")
    baseline.add_argument("--cells", default=None, help="comma-separated cell names to build")

    weakness = sub.add_parser("weakness", help="compute weaknesses vs cohort baselines")
    weakness.add_argument("platform", choices=["lichess", "chesscom"])
    weakness.add_argument("username")

    holdout = sub.add_parser("holdout", help="chronological holdout validation")
    holdout.add_argument("platform", choices=["lichess", "chesscom"])
    holdout.add_argument("username")

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
    elif args.command == "motifs":
        from sqlalchemy import select

        from blunderless.analysis.tag_motifs import tag_player_motifs
        from blunderless.db.models import Player

        with make_session_factory()() as db:
            player = db.execute(
                select(Player).where(
                    Player.platform == args.platform, Player.username == args.username
                )
            ).scalar_one()
            written = tag_player_motifs(db, player.id)
        print(f"motif_rows={written}")
    elif args.command == "baseline":
        from blunderless.cohort.runner import build_all

        results = build_all(
            workers=args.workers,
            games_per_cell=args.games_per_cell,
            with_motifs=not args.no_motifs,
            cells=args.cells.split(",") if args.cells else None,
        )
        print(results)
    elif args.command == "weakness":
        from sqlalchemy import select

        from blunderless.db.models import Player
        from blunderless.stats.weakness import compute_weaknesses

        with make_session_factory()() as db:
            player = db.execute(
                select(Player).where(
                    Player.platform == args.platform, Player.username == args.username
                )
            ).scalar_one()
            written = compute_weaknesses(db, player)
        print(f"weakness_rows={written}")
    elif args.command == "holdout":
        from sqlalchemy import select

        from blunderless.db.models import Player
        from blunderless.stats.holdout import evaluate_holdout

        with make_session_factory()() as db:
            player = db.execute(
                select(Player).where(
                    Player.platform == args.platform, Player.username == args.username
                )
            ).scalar_one()
            result = evaluate_holdout(db, player.id)
        if result is None:
            print("not enough games for holdout")
        else:
            print(
                f"train_moves={result.n_train_moves} test_moves={result.n_test_moves} "
                f"test_error_rate={result.test_error_rate:.4f} "
                f"logloss_cohort={result.logloss_cohort:.5f} "
                f"logloss_profile={result.logloss_profile:.5f} "
                f"improvement={result.improvement_pct:.2f}%"
            )


if __name__ == "__main__":
    main()
