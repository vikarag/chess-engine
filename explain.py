"""Turn Stockfish's lines into facts about WHY a move is good (or bad).

Stockfish only produces numbers and lines; it has no reasons. Everything the
narrator can say about "why" comes from here: what a line wins or loses, which
tactical patterns a move creates, what the opponent threatens, and how a
candidate move compares with the best one. All of it is computed from the
board and Stockfish output — nothing is guessed.
"""
import chess

import stats
from stockfish import label_loss

VAL, NAME = stats.VAL, stats.NAME


def _mat(b: chess.Board) -> int:
    """White-minus-Black material in pawns."""
    return sum((VAL.get(p.piece_type, 0) if p.color else -VAL.get(p.piece_type, 0))
               for p in b.piece_map().values())


def _who(color: bool) -> str:
    return "White" if color else "Black"


def _fmt(cp, human_color: str) -> str:
    return stats.fmt_cp(cp, 1 if human_color == "w" else -1)


def summarize_line(board: chess.Board, sans: list[str], max_plies: int = 8) -> str:
    """What a line does: captures, checks, mate, net material."""
    b = board.copy(stack=False)
    start, caps, checks, played, mate = _mat(b), [], 0, [], False
    for san in sans[:max_plies]:
        try:
            m = b.parse_san(san)
        except ValueError:
            break
        if b.is_capture(m):
            victim = chess.PAWN if b.is_en_passant(m) else b.piece_type_at(m.to_square)
            caps.append(f"{_who(b.turn)} takes a {NAME[victim]} ({san})")
        elif m.promotion:
            caps.append(f"{_who(b.turn)} promotes ({san})")
        b.push(m)
        played.append(san)
        if b.is_checkmate():
            mate = True
        elif b.is_check():
            checks += 1
    if not played:
        return "no line available"
    net = _mat(b) - start
    parts = [" ".join(played)]
    if caps:
        parts.append("; ".join(caps))
    if checks:
        parts.append(f"{checks} check{'s' if checks > 1 else ''} along the way")
    if mate:
        parts.append(f"ends in checkmate for {_who(not b.turn)}")
    elif net:
        parts.append(f"after it, material is {abs(net)} pawn-points better for {_who(net > 0)}")
    else:
        parts.append("material stays level")
    return " — ".join(parts)


def move_motifs(board: chess.Board, move: chess.Move) -> list[str]:
    """Tactical facts about one move, played from `board` (the position before it)."""
    b = board.copy(stack=False)
    mover = b.turn
    piece = b.piece_at(move.from_square)
    notes = []
    if b.is_castling(move):
        notes.append("castles")
    if b.is_capture(move):
        victim = chess.PAWN if b.is_en_passant(move) else b.piece_type_at(move.to_square)
        notes.append(f"captures a {NAME[victim]}")
    if move.promotion:
        notes.append(f"promotes to a {NAME[move.promotion]}")
    loose_before = set(stats.loose_pieces(b))
    pinned_before = {sq for sq in chess.SQUARES if b.piece_at(sq) and b.is_pinned(not mover, sq)}
    att_before = {sq: set(b.attackers(mover, sq)) for sq in chess.SQUARES
                  if (p := b.piece_at(sq)) and p.color != mover}
    b.push(move)
    if b.is_checkmate():
        notes.append("delivers checkmate")
    elif b.is_check():
        notes.append("gives double check" if len(b.checkers()) > 1 else "gives check")

    to = move.to_square
    targets = []
    for sq in b.attacks(to):
        p = b.piece_at(sq)
        if p and p.color != mover and p.piece_type != chess.KING:
            if VAL[p.piece_type] > VAL[piece.piece_type] or not b.attackers(not mover, sq):
                targets.append(f"{NAME[p.piece_type]} on {chess.square_name(sq)}")
    if len(targets) + (1 if b.is_check() else 0) >= 2 and targets:
        notes.append("forks " + (" and ".join(targets) + (" and the king" if b.is_check() else "")))
    elif targets:
        notes.append("attacks the " + targets[0])

    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p and p.color != mover and p.piece_type != chess.KING:
            if b.is_pinned(not mover, sq) and sq not in pinned_before:
                notes.append(f"pins the {NAME[p.piece_type]} on {chess.square_name(sq)} to its king")
            new = (set(b.attackers(mover, sq)) - {to}) - att_before.get(sq, set())
            if new and (VAL[p.piece_type] >= 3 or not b.attackers(not mover, sq)):
                notes.append(f"discovers an attack on the {NAME[p.piece_type]} on {chess.square_name(sq)}")

    loose_after = set(stats.loose_pieces(b))
    me = _who(mover)
    moved_from = f" on {chess.square_name(move.from_square)}"
    for x in sorted(loose_after - loose_before):
        tail = x.split(" ", 1)[1]
        if x.startswith(me):
            notes.append(f"puts its {tail} en prise" if tail.endswith(chess.square_name(to))
                         else f"leaves its {tail} en prise")
        elif not any(tail in n for n in notes):
            notes.append(f"attacks the loose {tail}")
    for x in sorted(loose_before - loose_after):
        if x.startswith(me) and not x.endswith(moved_from):
            notes.append("saves its " + x.split(" ", 1)[1])
    seen = []
    return [n for n in notes if not (n in seen or seen.append(n))]


def threats(sf, board: chess.Board, cur_sf: dict | None, human_color: str) -> str:
    """What the side NOT to move would do if the side to move passed (a null-move search)."""
    if board.is_check():
        return "The side to move is in check, so there is no threat search; it must answer the check."
    b = board.copy(stack=False)
    b.push(chess.Move.null())
    if b.is_game_over():
        return "Nothing to compute: the game is over."
    r = sf.analyse(b.fen(), depth=11, multipv=1)
    if not r["lines"]:
        return "No threat found."
    ln = r["lines"][0]
    cur = None if not cur_sf else (10000 if (cur_sf.get("mate") or 0) > 0 else
                                   -10000 if cur_sf.get("mate") is not None else cur_sf.get("eval_cp"))
    after = ln["cp"] if ln["mate"] is None else (10000 if ln["mate"] > 0 else -10000)
    other = _who(not board.turn)
    first = chess.Move.from_uci(ln["pv"][0])
    notes = move_motifs(b, first) if first in b.legal_moves else []
    return (f"THREAT CHECK: if {_who(board.turn)} passed, {other} would play {ln['pv_san'][0]} "
            f"(line: {summarize_line(b, ln['pv_san'])}). Eval now {_fmt(cur, human_color)}, "
            f"with that threat executed {_fmt(after, human_color)} (human's point of view). "
            + (f"That move {', '.join(notes)}." if notes else ""))


def analyze_move(sf, board: chess.Board, text: str, cur_sf: dict | None, human_color: str) -> str:
    """What-if: how good is this move here, and what happens after it?"""
    text = (text or "").strip()
    try:
        try:
            m = board.parse_san(text)
        except ValueError:
            m = board.parse_uci(text)
    except ValueError:
        legal = ", ".join(board.san(x) for x in board.legal_moves)
        return f"'{text}' is not a legal move here. Legal moves: {legal}."
    san, mover = board.san(m), board.turn
    notes = move_motifs(board, m)
    b = board.copy(stack=False)
    b.push(m)
    best_san = (cur_sf or {}).get("best_san")
    best_cp = None if not cur_sf else (10000 if (cur_sf.get("mate") or 0) > 0 else
                                       -10000 if cur_sf.get("mate") is not None else cur_sf.get("eval_cp"))
    if b.is_game_over(claim_draw=True):
        res = "checkmate" if b.is_checkmate() else "a draw"
        return f"MOVE {san}: ends the game in {res}. " + (f"It {', '.join(notes)}." if notes else "")
    r = sf.analyse(b.fen(), depth=12, multipv=1)
    ln = r["lines"][0] if r["lines"] else None
    after = None if not ln else (ln["cp"] if ln["mate"] is None else (10000 if ln["mate"] > 0 else -10000))
    out = [f"MOVE {san} (evals from the human's point of view, positive = good for the human)"]
    out.append(f"eval after it {_fmt(after, human_color)}"
               + (f"; Stockfish's best move here is {best_san} at {_fmt(best_cp, human_color)}" if best_san else ""))
    if after is not None and best_cp is not None:
        sign = 1 if mover else -1
        loss = int(max(0, min(sign * best_cp, 9000) - min(sign * after, 9000)))
        if san == best_san:
            out.append("this is Stockfish's best move")
        elif sign * after <= -9000:
            out.append("it walks into a forced mate")
        elif sign * best_cp >= 9000:
            out.append("it gives up a forced mate")
        else:
            out.append(f"it costs {loss} cp compared with the best move ({label_loss(loss)})")
    if ln:
        out.append(f"expected continuation: {summarize_line(b, ln['pv_san'])}")
    if notes:
        out.append(f"what the move itself does: {', '.join(notes)}")
    return out[0] + ": " + ". ".join(out[1:]) + "."


if __name__ == "__main__":      # self-check: python explain.py (needs no engine)
    # Nxe5 wins a pawn and attacks the c6 knight's... plain capture + attack facts
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert "captures a pawn" in move_motifs(b, b.parse_san("Nxe5"))
    # classic knight fork of king and queen: Nc7+ attacks e8 and a8
    g = chess.Board("q3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
    mo = move_motifs(g, g.parse_san("Nc7+"))
    assert any(n.startswith("forks") for n in mo) and "gives check" in mo, mo
    assert "puts its knight on f6 en prise" in move_motifs(chess.Board("3qk3/8/8/8/4N3/8/8/4K3 w - - 0 1"), chess.Move.from_uci("e4f6"))
    assert "delivers checkmate" in move_motifs(chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"), chess.Move.from_uci("a1a8"))
    s = summarize_line(b, ["Nxe5", "Nxe5", "d4"])
    assert "White takes a pawn (Nxe5)" in s and "Black takes a knight (Nxe5)" in s, s
    assert summarize_line(b, ["Bc4", "Bc5"]).endswith("material stays level")
    print("ok")
