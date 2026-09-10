"""Weakness detection: player error rates vs cohort, with FDR control (§8).

A weakness is a (motif, phase, time_bucket) cell where the player's rate
is significantly above the cohort's after Benjamini–Hochberg correction,
with a minimum-sample gate. Cells are ranked by total win probability
lost — the most expensive habit first, not the most frequent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from blunderless.cohort.build import ANY_ERROR
from blunderless.db.models import (
    CohortBaseline,
    Game,
    Motif,
    MoveAnalysis,
    Player,
    Weakness,
)
from blunderless.stats.tests import benjamini_hochberg, two_proportion_z

ERROR_SEVERITIES = ("blunder", "mistake", "inaccuracy")
MIN_OBSERVATIONS = 5  # min player errors in a cell to even test it
MIN_ELIGIBLE = 30  # min eligible player moves in the (phase, bucket)
Q_THRESHOLD = 0.10


def band_for_rating(rating: int | None) -> str | None:
    if rating is None:
        return None
    for lo in (600, 800, 1000, 1200, 1400):
        if lo <= rating < lo + 200:
            return f"{lo}-{lo + 200}"
    return None


def tc_class(time_control: str | None) -> str | None:
    """Initial clock < 8 min → blitz; 8–25 min → rapid (lichess bounds)."""
    if not time_control:
        return None
    base = time_control.split("+")[0]
    try:
        seconds = int(base)
    except ValueError:
        return None
    if seconds < 480:
        return "blitz"
    if seconds <= 1500:
        return "rapid"
    return None


@dataclass
class PlayerCells:
    eligible: Counter  # (tc, band, phase, bucket) -> n
    errors: Counter  # (tc, band, motif, phase, bucket) -> k
    winprob_lost: dict  # same key as errors -> summed delta


def collect_player_cells(db: Session, player_id: int) -> PlayerCells:
    cells = PlayerCells(Counter(), Counter(), defaultdict(float))
    games = db.execute(select(Game).where(Game.player_id == player_id)).scalars()
    for game in games:
        tc = tc_class(game.time_control)
        band = band_for_rating(game.opponent_rating)
        if tc is None or band is None:
            continue
        rows = db.execute(
            select(MoveAnalysis).where(MoveAnalysis.game_id == game.id)
        ).scalars()
        for row in rows:
            if row.severity is None:
                continue  # excluded from statistics by the filters
            key_base = (tc, band, row.phase, _bucket(row.clock_remaining_s))
            cells.eligible[key_base] += 1
            if row.severity not in ERROR_SEVERITIES:
                continue
            for motif_type in [ANY_ERROR] + [
                m.motif_type
                for m in db.execute(
                    select(Motif).where(Motif.move_analysis_id == row.id)
                ).scalars()
            ]:
                key = (tc, band, motif_type, row.phase, key_base[3])
                cells.errors[key] += 1
                cells.winprob_lost[key] += row.delta_win_prob or 0.0
    return cells


def _bucket(clock_s: float | None) -> str:
    from blunderless.analysis.filters import time_bucket

    return time_bucket(clock_s)


def compute_weaknesses(db: Session, player: Player) -> int:
    """(Re)compute weakness rows for a player. Returns rows written."""
    cells = collect_player_cells(db, player.id)

    baselines = {
        (b.time_control, b.rating_band, b.motif_type, b.phase, b.time_bucket): b
        for b in db.execute(select(CohortBaseline)).scalars()
    }

    # Aggregate over bands/TCs the player actually played: weight cohort
    # rates by the player's eligible-move counts in each (tc, band).
    tested: list[dict] = []
    motif_keys = {(k[2], k[3], k[4]) for k in cells.errors}
    for motif_type, ph, bucket in motif_keys:
        k_player = 0
        n_player = 0
        k_cohort = 0
        n_cohort = 0
        lost = 0.0
        for (tc, band, phase_, bucket_), n_elig in cells.eligible.items():
            if phase_ != ph or bucket_ != bucket:
                continue
            base = baselines.get((tc, band, motif_type, ph, bucket))
            if base is None:
                continue
            n_player += n_elig
            k_player += cells.errors.get((tc, band, motif_type, ph, bucket), 0)
            lost += cells.winprob_lost.get((tc, band, motif_type, ph, bucket), 0.0)
            n_cohort += base.n_samples
            k_cohort += round(base.error_rate * base.n_samples)
        if k_player < MIN_OBSERVATIONS or n_player < MIN_ELIGIBLE or n_cohort == 0:
            continue
        z, p = two_proportion_z(k_player, n_player, k_cohort, n_cohort)
        player_rate = k_player / n_player
        cohort_rate = k_cohort / n_cohort
        if player_rate <= cohort_rate:
            continue  # only elevated rates are "weaknesses"
        tested.append(
            {
                "motif_type": motif_type,
                "phase": ph,
                "time_bucket": bucket,
                "player_rate": player_rate,
                "cohort_rate": cohort_rate,
                "ratio": player_rate / cohort_rate if cohort_rate else float("inf"),
                "p_value": p,
                "total_winprob_lost": lost,
                "n_observations": k_player,
            }
        )

    q_values = benjamini_hochberg([t["p_value"] for t in tested])
    db.execute(delete(Weakness).where(Weakness.player_id == player.id))
    written = 0
    now = datetime.now(UTC)
    for t, q in zip(tested, q_values, strict=True):
        db.add(
            Weakness(
                player_id=player.id,
                computed_at=now,
                q_value=q,
                **t,
            )
        )
        written += 1
    db.commit()
    return written
