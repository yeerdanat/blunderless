"""In-process analysis jobs with progress streaming.

A job runs the full diagnosis pipeline (engine analysis → motif tagging →
weakness computation) in a background thread; progress is polled by the
SSE endpoint. Single-process by design — good enough for a local demo;
swap for Celery/Redis when multiple API workers are needed.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    platform: str
    username: str
    status: str = "queued"  # queued | analyzing | tagging | scoring | done | error
    done: int = 0
    total: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)


JOBS: dict[str, Job] = {}


def start_analysis_job(platform: str, username: str, workers: int | None = None) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], platform=platform, username=username)
    JOBS[job.id] = job

    def run() -> None:
        try:
            from sqlalchemy import select

            from blunderless.analysis.runner import analyze_player
            from blunderless.analysis.tag_motifs import tag_player_motifs
            from blunderless.db.models import Player
            from blunderless.db.session import make_session_factory
            from blunderless.stats.weakness import compute_weaknesses

            job.status = "analyzing"

            def progress(done: int, total: int, _stats) -> None:
                job.done, job.total = done, total

            analyze_player(platform, username, workers=workers, progress=progress)

            job.status = "tagging"
            with make_session_factory()() as db:
                player = db.execute(
                    select(Player).where(
                        Player.platform == platform, Player.username == username
                    )
                ).scalar_one()
                tag_player_motifs(db, player.id)
                job.status = "scoring"
                compute_weaknesses(db, player)
            job.status = "done"
        except Exception as exc:  # surfaced via SSE
            job.status = "error"
            job.error = str(exc)

    threading.Thread(target=run, daemon=True).start()
    return job
