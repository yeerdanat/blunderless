# Blunderless

Mines a chess player's **full game history** for the mistakes they make *repeatedly*,
benchmarked against players at the same rating — then routes them to targeted training.

Per-game annotation tells you what you already knew: you lost. Blunderless answers the
question a player actually has:

> Across my last 1,200 games, what do I get wrong over and over —
> and is it unusual for someone at my level?

On the reference corpus (1,243 real games) the answer comes out like:

> You miss **pins** 1.21× more often than the 800–1000 cohort in normal-time
> middlegames (6.0% vs 5.0% of decisions, n=885, BH-adjusted q < 0.0001), and it
> is your single most expensive habit — roughly 160 games' worth of win
> probability. Under 60 seconds, your missed-fork rate jumps to 1.47× cohort
> (q = 0.025).

## Design thesis

**Engine as oracle, LLM as narrator.** Every factual claim — evaluation, best move,
win-probability delta, missed tactical motif — comes from Stockfish and board geometry.
The LLM only turns structured facts into prose, and a validator rejects any output that
references squares, moves, or pieces absent from its input (falling back to templates).
The system is fully functional with no LLM configured.

## Architecture

```
 Chess.com / Lichess APIs ──▶ Ingestor (PGN + %clk, dedupe)
                                   │
                                   ▼
              ┌─────────────── Analysis pipeline ────────────────┐
              │ pass 1: every position @40k nodes, single PV     │
              │ pass 2: deep @350k nodes, MultiPV 4 — only where │
              │         pass 1 saw Δwin% ≥ 3%, + 5% control      │
              │ position cache: Zobrist-keyed, Postgres          │
              │ parallel: N processes × 1 engine (Threads=1)     │
              └───────────────────┬──────────────────────────────┘
                                  ▼
        Classifier: Δwin% severity · motif geometry · phase · clock
                                  ▼
        Aggregator: rates vs cohort · z-tests · BH FDR · rank by cost
                                  ▼
     FastAPI (SSE progress) ──▶ React + chessground dashboard
```

## Measured performance (Apple M-series, 6 workers, 1,243 real games / 72,968 plies)

| Metric | Value |
|---|---|
| Naive baseline (deep MultiPV everywhere, no cache) | 269.5s / 50 games ⇒ **~108 min** projected full corpus |
| Optimized cold run (two-pass tiering + cache) | **51.1 min** full corpus (2.1×) |
| Warm re-analysis (cache populated) | **84.7s** full corpus (**76×** vs naive, 99.7% cache hit rate) |
| Cold-run cache hit rate (opening transpositions) | 10.6% |
| Positions escalated to deep analysis | ~35% of player moves (blitz corpus — high error rate) |
| Reproducibility | fixed node counts (40k / 350k), Threads=1 per engine |

Engine analysis is **node-limited, never time-limited**, so results reproduce across
machines and load. All evaluations are stored White-POV in a cache keyed on
`(Zobrist hash, node_count)` — a deep request never gets a shallow hit.

## Motif classifier validation

Detectors are pure board geometry over engine PVs (python-chess attack maps + SEE) —
no ML, no LLM. Validated against **2,800 human/community-tagged Lichess puzzles**
(400 sampled per theme, seed 20260815):

| Motif | Recall | Precision* |
|---|---|---|
| missed mate | 1.00 | 1.00 |
| hanging piece | 1.00 | 0.75 |
| fork | 0.84 | 0.64 |
| skewer | 0.84 | 0.55 |
| back rank | 0.80 | 0.86 |
| discovered attack | 0.60 | 0.85 |
| pin | 0.51 | 0.37 |

\* Precision is a lower bound: puzzle theme tags are not exhaustive, so a correct
detection on an untagged puzzle counts against us. Pin is the known weak spot —
pin geometry overlaps heavily with other motifs and exploit-existing-pin cases are
hard to separate from incidental attacks. Reproduce with
`python scripts/validate_motifs.py`.

## Cohort baselines

Built from the **Lichess open database** (June 2026 standard dump): 7.0M games
streamed, keeping rated blitz/rapid games where both players sit in the target band
**and** the game carries `[%eval]` annotations — so severity classification of cohort
games needs no engine time at all. The engine runs only on cohort error positions
(shallow, single PV) to extract lines for motif detection.

- **29,662 cohort games**, ~1.5M eligible moves after exclusion filters
- 10 cells: {blitz, rapid} × {600–800, 800–1000, 1000–1200, 1200–1400, 1400–1600}
- Per-cell rates for every (motif × phase × time-pressure bucket), Wilson CIs

## Statistical rigor

- **Win probability, not centipawns** — a 300cp drop from +50 is a blunder; from
  +1200 it's nothing. Severity is classified on Δwin% (lichess logistic mapping).
- **Exclusion filters** — book moves (lichess-openings dataset), forced moves,
  time scrambles (<10s), and positions that were *and stayed* decided. (The naive
  "exclude if decided before the move" gate silently deletes every missed mate —
  found the hard way, fixed, tested.)
- **Two-proportion z-tests** per (motif × phase × clock) cell vs the cohort, with
  **Benjamini–Hochberg FDR control** across the grid and minimum-sample gates.
  Shuffled-label property tests verify the machinery doesn't invent weaknesses.
- **Chronological holdout** — profile fitted on the first 70% of games must predict
  errors in the held-out 30% better than the rating-matched cohort baseline does
  (log-loss). If it can't, the "diagnosis" was noise. Measured on the reference
  corpus: profile log-loss 0.5895 vs cohort 0.5938 on 8,613 held-out decisions —
  a **+0.7% improvement**, small but real; per-move error prediction is dominated
  by rating, and the profile's edge is exactly the recurring-weakness signal.
- **Small samples stay honest** — the 40-game test account produces zero
  FDR-surviving weaknesses (all q > 0.18). The machinery refuses to diagnose
  from thin data; on the 1,203-game account, five cells survive at q < 0.10.

## Constrained narration

The narrator receives only structured facts (`{player_move_san, best_move_san,
delta_win_prob, motif, targets, ...}`) and produces 1–2 sentences. A post-generation
validator parses every square/move token in the output and rejects any text
referencing entities absent from the input — the classic LLM-chess hallucination,
eliminated by construction. No `ANTHROPIC_API_KEY` → deterministic templates.

## Running it

```bash
docker compose up -d postgres redis     # infra
pip install -e ".[dev]" && alembic upgrade head
brew install stockfish zstd

blunderless sync chesscom <username>    # import full history
blunderless analyze chesscom <username> # engine pipeline (progress printed)
blunderless motifs chesscom <username>  # tag tactical motifs
blunderless weakness chesscom <username># z-tests + FDR vs cohort
blunderless holdout chesscom <username> # predictive-validity check

uvicorn blunderless.api.main:app --port 8000   # API
cd frontend && npm install && npm run dev      # dashboard at :5173
```

Cohort baselines ship as a build step (`blunderless baseline`) over
`data/cohort/*.pgn` sampled by `scripts/sample_cohort.py`.

## Honest limitations

- Cohort baselines are keyed to **lichess** ratings; Chess.com opponent ratings are
  used as a proxy band. The two pools differ (~200–300 points at the low end).
- Pin detection precision (0.37 lower-bound) is below the other detectors.
- Player rating is estimated as the median opponent rating of rated games.
- 40 games (the lichess test account) produce wide CIs everywhere — by design, the
  minimum-sample gates refuse to call those "weaknesses".

## Stack

Python 3.13 · python-chess · Stockfish 16+ · FastAPI · SQLAlchemy/Alembic ·
Postgres · Redis · React + Vite + chessground · Claude API (optional narration)
