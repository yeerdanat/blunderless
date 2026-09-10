"""Predictive-validity check: does the diagnosed profile beat the cohort
baseline on unseen games? (§8.4)

Chronological 70/30 split. Both predictors assign every eligible held-out
move a probability of being an error, per (phase, time_bucket) cell:
  - cohort predictor: the rating-matched baseline rate
  - profile predictor: player's train-split rate, empirically shrunk
    toward the cohort rate (cells with little data defer to the cohort)
Scored by log-loss; the headline number is the improvement percentage.
If the profile can't beat the cohort on held-out data, the "diagnosis"
was noise — this is the test that keeps us honest.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from blunderless.cohort.build import ANY_ERROR
from blunderless.db.models import CohortBaseline, Game, MoveAnalysis
from blunderless.stats.weakness import ERROR_SEVERITIES, band_for_rating, tc_class

SHRINKAGE = 10.0  # pseudo-observations of the cohort rate
TRAIN_FRACTION = 0.7
EPS = 1e-6


@dataclass
class HoldoutResult:
    n_train_moves: int
    n_test_moves: int
    test_error_rate: float
    logloss_cohort: float
    logloss_profile: float

    @property
    def improvement_pct(self) -> float:
        if self.logloss_cohort == 0:
            return 0.0
        return 100 * (self.logloss_cohort - self.logloss_profile) / self.logloss_cohort


def _cell(row: MoveAnalysis, game: Game) -> tuple | None:
    from blunderless.analysis.filters import time_bucket

    tc = tc_class(game.time_control)
    band = band_for_rating(game.opponent_rating)
    if tc is None or band is None:
        return None
    return (tc, band, row.phase, time_bucket(row.clock_remaining_s))


def evaluate_holdout(db: Session, player_id: int) -> HoldoutResult | None:
    games = list(
        db.execute(
            select(Game)
            .where(Game.player_id == player_id)
            .order_by(Game.played_at)
        ).scalars()
    )
    if len(games) < 20:
        return None
    split = int(len(games) * TRAIN_FRACTION)
    train_games, test_games = games[:split], games[split:]

    baselines = {
        (b.time_control, b.rating_band, b.phase, b.time_bucket): b.error_rate
        for b in db.execute(
            select(CohortBaseline).where(CohortBaseline.motif_type == ANY_ERROR)
        ).scalars()
    }

    def moves_of(games_subset):
        for game in games_subset:
            rows = db.execute(
                select(MoveAnalysis).where(MoveAnalysis.game_id == game.id)
            ).scalars()
            for row in rows:
                if row.severity is None:
                    continue
                cell = _cell(row, game)
                if cell is None or cell not in baselines:
                    continue
                yield cell, row.severity in ERROR_SEVERITIES

    train_n: Counter = Counter()
    train_k: Counter = Counter()
    for cell, is_error in moves_of(train_games):
        train_n[cell] += 1
        train_k[cell] += is_error

    ll_cohort = ll_profile = 0.0
    n_test = 0
    k_test = 0
    for cell, is_error in moves_of(test_games):
        p_cohort = min(1 - EPS, max(EPS, baselines[cell]))
        p_profile = (train_k[cell] + SHRINKAGE * p_cohort) / (
            train_n[cell] + SHRINKAGE
        )
        p_profile = min(1 - EPS, max(EPS, p_profile))
        y = 1.0 if is_error else 0.0
        ll_cohort -= y * math.log(p_cohort) + (1 - y) * math.log(1 - p_cohort)
        ll_profile -= y * math.log(p_profile) + (1 - y) * math.log(1 - p_profile)
        n_test += 1
        k_test += is_error

    if n_test == 0:
        return None
    return HoldoutResult(
        n_train_moves=sum(train_n.values()),
        n_test_moves=n_test,
        test_error_rate=k_test / n_test,
        logloss_cohort=ll_cohort / n_test,
        logloss_profile=ll_profile / n_test,
    )
