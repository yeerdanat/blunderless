# Blunderless

Mines a chess player's **full game history** for the mistakes they make *repeatedly*,
benchmarked against players at the same rating — then routes them to targeted training.

Per-game annotation tells you what you already knew: you lost. Blunderless answers the
question a player actually has:

> Across my last 400 games, what do I get wrong over and over —
> and is it unusual for someone at my level?

## Design thesis

**Engine as oracle, LLM as narrator.** Every factual claim — evaluation, best move,
win-probability delta, missed tactical motif — comes from Stockfish and board geometry.
The LLM only turns structured facts into prose, and a validator rejects any output that
references squares, moves, or pieces absent from its input.

## Status

Early scaffold. See the commit log for the build plan unfolding.

## Development

```bash
pip install -e ".[dev]"
pytest
docker compose up --build   # postgres + redis + stockfish worker image
```
