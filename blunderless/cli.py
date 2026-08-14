import argparse

from blunderless.db.session import make_session_factory
from blunderless.ingest.sync import sync_player


def main() -> None:
    parser = argparse.ArgumentParser(prog="blunderless")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="import a player's game history")
    sync.add_argument("platform", choices=["lichess", "chesscom"])
    sync.add_argument("username")
    sync.add_argument("--max", type=int, default=None, help="limit number of games")

    args = parser.parse_args()
    if args.command == "sync":
        stats = sync_player(
            args.platform, args.username, make_session_factory(), max_games=args.max
        )
        print(
            f"fetched={stats.fetched} inserted={stats.inserted} "
            f"skipped_unparseable={stats.skipped_unparseable}"
        )


if __name__ == "__main__":
    main()
