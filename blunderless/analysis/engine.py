"""UCI engine wrapper.

The engine is the project's oracle: every factual claim downstream (eval,
best move, win-probability delta, missed motif) traces back to output from
this module. Two invariants live here:

- Search is limited by *node count*, never wall-clock time, so results are
  reproducible across machines and machine load (§6.4 of the design doc).
- Evaluations are always reported from White's point of view, so callers
  never have to guess whose perspective a score is in.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from types import TracebackType

import chess
import chess.engine

DEFAULT_NODES = 100_000


class EngineNotFoundError(RuntimeError):
    """No Stockfish binary could be located."""


@dataclass(frozen=True)
class LineEval:
    """One engine line (a single PV) for a position.

    Exactly one of eval_cp / mate_in is set. Scores are from White's POV:
    positive mate_in means White mates in N, negative means Black does.
    """

    move: str  # first move of the PV, UCI notation
    eval_cp: int | None
    mate_in: int | None
    pv: list[str] = field(default_factory=list)  # full PV, UCI notation


@dataclass(frozen=True)
class PositionAnalysis:
    fen: str
    nodes: int  # requested node limit, part of the cache identity later
    multipv: int
    lines: list[LineEval]  # ordered best-first

    @property
    def best(self) -> LineEval:
        return self.lines[0]


def find_stockfish(explicit_path: str | None = None) -> str:
    path = explicit_path or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not path or not os.path.exists(path):
        raise EngineNotFoundError(
            "Stockfish binary not found. Install it (brew install stockfish / "
            "apt install stockfish) or set STOCKFISH_PATH."
        )
    return path


class Engine:
    """A single Stockfish process speaking UCI.

    Defaults to Threads=1: multi-threaded search is nondeterministic even at
    a fixed node count. Parallelism belongs at the pool level (one process
    per worker), not inside a single search.
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        threads: int = 1,
        hash_mb: int = 128,
    ) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(find_stockfish(path))
        self._engine.configure({"Threads": threads, "Hash": hash_mb})

    def analyse(
        self,
        position: chess.Board | str,
        *,
        nodes: int = DEFAULT_NODES,
        multipv: int = 1,
    ) -> PositionAnalysis:
        board = chess.Board(position) if isinstance(position, str) else position
        infos = self._engine.analyse(
            board, chess.engine.Limit(nodes=nodes), multipv=multipv
        )
        lines = [self._to_line(info) for info in infos]
        return PositionAnalysis(
            fen=board.fen(), nodes=nodes, multipv=multipv, lines=lines
        )

    @staticmethod
    def _to_line(info: chess.engine.InfoDict) -> LineEval:
        score = info["score"].white()
        pv = [move.uci() for move in info.get("pv", [])]
        if not pv:
            raise chess.engine.EngineError(f"engine returned no PV: {info}")
        return LineEval(
            move=pv[0],
            eval_cp=score.score(),
            mate_in=score.mate(),
            pv=pv,
        )

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> Engine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
