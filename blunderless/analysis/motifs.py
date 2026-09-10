"""Tactical motif detection from board geometry and engine PVs (§7).

No machine learning, no LLM: every motif is a geometric property of the
position, the player's move, and the engine's principal variation.
Multi-label by design — a move can hang a piece AND miss a fork; each
detector reports independently with a confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


@dataclass(frozen=True)
class MotifHit:
    motif: str
    confidence: float
    detail: dict = field(default_factory=dict)


def _see_ge_zero(board: chess.Board, move: chess.Move) -> bool:
    """Static exchange evaluation: does this capture not lose material?"""
    try:
        return board.gives_check(move) or _see(board, move) >= 0
    except (ValueError, AssertionError):
        return False


def _see(board: chess.Board, move: chess.Move) -> int:
    """Simple SEE via recursive least-attacker exchange on the target square."""
    target = move.to_square
    b = board.copy(stack=False)
    captured = b.piece_type_at(target)
    gain = [PIECE_VALUES.get(captured, 0) if captured else 0]
    b.push(move)
    attacker_value = PIECE_VALUES[board.piece_type_at(move.from_square)]

    while True:
        # least valuable attacker of the side to move onto target
        attackers = b.attackers(b.turn, target)
        if not attackers:
            break
        least = min(attackers, key=lambda sq: PIECE_VALUES[b.piece_type_at(sq)])
        gain.append(attacker_value - gain[-1])
        attacker_value = PIECE_VALUES[b.piece_type_at(least)]
        recapture = chess.Move(least, target)
        if b.piece_type_at(least) == chess.PAWN and chess.square_rank(target) in (0, 7):
            recapture = chess.Move(least, target, promotion=chess.QUEEN)
        if not b.is_legal(recapture):
            break
        b.push(recapture)

    while len(gain) > 1:
        gain[-2] = -max(-gain[-2], gain[-1])
        gain.pop()
    return gain[0]


def _hanging_after(
    board: chess.Board, move: chess.Move, reply_pv: list[str] | None = None
) -> MotifHit | None:
    """Did the player's move leave a piece en prise?

    When the opponent's engine reply is known (reply_pv), require that it
    actually captures the hanging piece — SEE alone over-fires on pieces
    that are theoretically winnable but not the point of the position.
    """
    b = board.copy(stack=False)
    b.push(move)
    mover_color = board.turn
    reply_target: chess.Square | None = None
    if reply_pv:
        try:
            reply_target = chess.Move.from_uci(reply_pv[0]).to_square
        except ValueError:
            reply_target = None

    worst: tuple[int, chess.Square] | None = None
    for sq in chess.SquareSet(b.occupied_co[mover_color]):
        piece = b.piece_type_at(sq)
        if piece in (chess.KING, chess.PAWN):
            continue
        if reply_target is not None and sq != reply_target:
            continue
        attackers = b.attackers(not mover_color, sq)
        if not attackers:
            continue
        for att in attackers:
            cap = chess.Move(att, sq)
            if b.piece_type_at(att) == chess.PAWN and chess.square_rank(sq) in (0, 7):
                cap = chess.Move(att, sq, promotion=chess.QUEEN)
            if not b.is_legal(cap):
                continue
            if _see(b, cap) > 0:
                value = PIECE_VALUES[piece]
                if worst is None or value > worst[0]:
                    worst = (value, sq)
                break
    if worst is None:
        return None
    return MotifHit(
        "hanging_piece",
        confidence=0.9 if reply_target is not None else 0.6,
        detail={"square": chess.square_name(worst[1]), "value": worst[0]},
    )


def _fork_in_pv(board: chess.Board, pv: list[str]) -> MotifHit | None:
    """Engine's first move attacks >=2 targets worth more than the forker
    (or undefended)."""
    if not pv:
        return None
    move = chess.Move.from_uci(pv[0])
    if not board.is_legal(move):
        return None
    b = board.copy(stack=False)
    b.push(move)
    forker_sq = move.to_square
    forker = b.piece_type_at(forker_sq)
    if forker is None:
        return None
    forker_value = PIECE_VALUES[forker]
    targets = []
    for sq in b.attacks(forker_sq):
        victim = b.piece_type_at(sq)
        if victim is None or b.color_at(sq) == b.color_at(forker_sq):
            continue
        victim_value = PIECE_VALUES[victim]
        defended = bool(b.attackers(b.turn, sq))
        # A target must be worth forking: higher-valued than the forker,
        # or an undefended piece (not a stray pawn).
        if (
            victim == chess.KING
            or victim_value > forker_value
            or (not defended and victim_value >= 3)
        ):
            targets.append((chess.square_name(sq), b.piece_at(sq).symbol()))
    if len(targets) < 2:
        return None
    return MotifHit(
        "fork",
        confidence=0.85,
        detail={"forking_move": pv[0], "targets": targets},
    )


def _pin_or_skewer_in_pv(board: chess.Board, pv: list[str]) -> MotifHit | None:
    """Engine's first move creates a line attack through a piece to a more
    valuable one behind it (pin) or in front (skewer)."""
    if not pv:
        return None
    move = chess.Move.from_uci(pv[0])
    if not board.is_legal(move):
        return None
    b = board.copy(stack=False)
    b.push(move)
    sq = move.to_square
    piece = b.piece_type_at(sq)
    if piece not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None
    enemy = b.turn
    for ray_sq in b.attacks(sq):
        front = b.piece_at(ray_sq)
        if front is None or front.color != enemy:
            continue
        # look through 'front' along the same ray
        direction = _ray_direction(sq, ray_sq)
        if direction is None or not _piece_moves_along(piece, direction):
            continue
        behind_sq = _next_along(ray_sq, direction)
        while behind_sq is not None:
            behind = b.piece_at(behind_sq)
            if behind is not None:
                if behind.color == enemy:
                    v_front = PIECE_VALUES[front.piece_type]
                    v_behind = PIECE_VALUES[behind.piece_type]
                    if v_behind > v_front:
                        return MotifHit(
                            "pin",
                            confidence=0.75,
                            detail={
                                "move": pv[0],
                                "pinned": chess.square_name(ray_sq),
                                "behind": chess.square_name(behind_sq),
                            },
                        )
                    if v_front > v_behind:
                        return MotifHit(
                            "skewer",
                            confidence=0.75,
                            detail={
                                "move": pv[0],
                                "front": chess.square_name(ray_sq),
                                "behind": chess.square_name(behind_sq),
                            },
                        )
                break
            behind_sq = _next_along(behind_sq, direction)
    return None


def _existing_pin_exploited(board: chess.Board, pv: list[str]) -> MotifHit | None:
    """The PV wins by attacking/capturing a piece already pinned to its king."""
    if not pv:
        return None
    move = chess.Move.from_uci(pv[0])
    if not board.is_legal(move):
        return None
    enemy = not board.turn
    b = board.copy(stack=False)
    b.push(move)
    attacker_value = PIECE_VALUES.get(b.piece_type_at(move.to_square) or 0, 0)
    for sq in chess.SquareSet(board.occupied_co[enemy]):
        if not board.is_pinned(enemy, sq):
            continue
        victim_value = PIECE_VALUES.get(board.piece_type_at(sq) or 0, 0)
        captures_it = move.to_square == sq
        piles_on = (
            sq in b.attacks(move.to_square)
            and (victim_value > attacker_value or not b.attackers(enemy, sq))
        )
        if captures_it or piles_on:
            return MotifHit(
                "pin",
                confidence=0.7,
                detail={"move": pv[0], "pinned": chess.square_name(sq)},
            )
    return None


def _advance(board: chess.Board, pv: list[str], k: int) -> chess.Board | None:
    """Board after the first k PV moves (None if the line is malformed)."""
    b = board.copy(stack=False)
    try:
        for uci in pv[:k]:
            mv = chess.Move.from_uci(uci)
            if not b.is_legal(mv):
                return None
            b.push(mv)
    except ValueError:
        return None
    return b


def _ray_direction(a: chess.Square, b: chess.Square) -> tuple[int, int] | None:
    df = chess.square_file(b) - chess.square_file(a)
    dr = chess.square_rank(b) - chess.square_rank(a)
    if df == 0 and dr == 0:
        return None
    if df == 0 or dr == 0 or abs(df) == abs(dr):
        return (0 if df == 0 else df // abs(df), 0 if dr == 0 else dr // abs(dr))
    return None


def _piece_moves_along(piece: chess.PieceType, direction: tuple[int, int]) -> bool:
    diagonal = direction[0] != 0 and direction[1] != 0
    if piece == chess.BISHOP:
        return diagonal
    if piece == chess.ROOK:
        return not diagonal
    return piece == chess.QUEEN


def _next_along(sq: chess.Square, direction: tuple[int, int]) -> chess.Square | None:
    f = chess.square_file(sq) + direction[0]
    r = chess.square_rank(sq) + direction[1]
    if 0 <= f <= 7 and 0 <= r <= 7:
        return chess.square(f, r)
    return None


def _missed_mate(pv_mate_in: int | None, played_leads_to_mate: bool) -> MotifHit | None:
    if pv_mate_in is None or pv_mate_in <= 0:
        return None
    if played_leads_to_mate:
        return None
    return MotifHit("missed_mate", confidence=1.0, detail={"mate_in": pv_mate_in})


def _back_rank(board: chess.Board, pv: list[str], mate_in: int | None) -> MotifHit | None:
    """Missed back-rank tactic: PV delivers mate/threat on the opponent's
    home rank while their king is boxed in by its own pawns."""
    if not pv or mate_in is None or mate_in <= 0:
        return None
    enemy = not board.turn
    king_sq = board.king(enemy)
    if king_sq is None:
        return None
    home_rank = 7 if enemy == chess.BLACK else 0
    if chess.square_rank(king_sq) != home_rank:
        return None
    # escape squares directly in front of the king blocked by own pawns
    forward = -1 if enemy == chess.BLACK else 1
    blocked = 0
    escapes = 0
    for df in (-1, 0, 1):
        f = chess.square_file(king_sq) + df
        if not 0 <= f <= 7:
            continue
        escapes += 1
        front_sq = chess.square(f, home_rank + forward)
        piece = board.piece_at(front_sq)
        if piece is not None and piece.color == enemy and piece.piece_type == chess.PAWN:
            blocked += 1
    if blocked < escapes:
        return None
    # PV's mating line must involve a heavy piece landing on the home rank
    for uci in pv[:4]:
        mv = chess.Move.from_uci(uci)
        if chess.square_rank(mv.to_square) == home_rank:
            return MotifHit(
                "back_rank", confidence=0.9, detail={"mate_in": mate_in, "rank": home_rank}
            )
    return None


def _discovered_attack_in_pv(board: chess.Board, pv: list[str]) -> MotifHit | None:
    """Engine's first move unveils an attack from a piece behind it onto a
    high-value target."""
    if not pv:
        return None
    move = chess.Move.from_uci(pv[0])
    if not board.is_legal(move):
        return None
    mover_color = board.turn
    before_attacked = _attacked_squares_by_sliders(board, mover_color, exclude=move.from_square)
    b = board.copy(stack=False)
    b.push(move)
    after_attacked = _attacked_squares_by_sliders(b, mover_color, exclude=move.to_square)
    newly = after_attacked - before_attacked
    for sq in newly:
        victim = b.piece_at(sq)
        if victim is not None and victim.color != mover_color and (
            PIECE_VALUES[victim.piece_type] >= 5 or victim.piece_type == chess.KING
        ):
            return MotifHit(
                "discovered_attack",
                confidence=0.7,
                detail={"move": pv[0], "target": chess.square_name(sq)},
            )
    return None


def _attacked_squares_by_sliders(
    board: chess.Board, color: chess.Color, exclude: chess.Square
) -> set[chess.Square]:
    squares: set[chess.Square] = set()
    for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if sq == exclude:
                continue
            squares.update(board.attacks(sq))
    return squares


def detect_motifs(
    board_before: chess.Board,
    played_uci: str,
    best_pv: list[str],
    best_mate_in: int | None,
    played_mate_in: int | None = None,
    reply_pv: list[str] | None = None,
) -> list[MotifHit]:
    """All motifs applicable to one classified error.

    board_before: position the player faced. best_pv: engine's top line
    (what they missed). played_uci: what they did instead. reply_pv:
    engine's best answer to the played move (confirms hanging pieces).
    """
    hits: list[MotifHit] = []
    played = chess.Move.from_uci(played_uci)

    if (h := _hanging_after(board_before, played, reply_pv)) is not None:
        hits.append(h)

    # what the engine line exploited / would have exploited — checked on
    # the first move and (tactics often land one exchange later) on the
    # mover's second move of the line.
    if best_pv and best_pv[0] != played_uci:
        for detector in (
            _fork_in_pv,
            _pin_or_skewer_in_pv,
            _discovered_attack_in_pv,
            _existing_pin_exploited,
        ):
            if (h := detector(board_before, best_pv)) is not None:
                hits.append(h)
        # Second-move sweep only when the first move showed nothing —
        # tactics often land one exchange later, but sweeping every line
        # tags incidental geometry and costs precision.
        if not hits:
            board_later = _advance(board_before, best_pv, 2)
            if board_later is not None and len(best_pv) > 2:
                for detector in (
                    _fork_in_pv,
                    _pin_or_skewer_in_pv,
                    _discovered_attack_in_pv,
                ):
                    if (h := detector(board_later, best_pv[2:])) is not None:
                        hits.append(h)
        if (h := _back_rank(board_before, best_pv, best_mate_in)) is not None:
            hits.append(h)
        if (
            h := _missed_mate(
                best_mate_in, played_leads_to_mate=(played_mate_in or 0) > 0
            )
        ) is not None:
            hits.append(h)

    return hits
