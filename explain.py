"""Turn Stockfish's lines into facts about WHY a move is good (or bad).

Stockfish only produces numbers and lines; it has no reasons. Everything the
narrator can say about "why" comes from here: what a line wins or loses, which
tactical patterns a move creates, what the opponent threatens, and how a
candidate move compares with the best one. All of it is computed from the
board and Stockfish output — nothing is guessed.
"""
import chess

import stats
from stockfish import label_win_loss

VAL, NAME = stats.VAL, stats.NAME


def _mat(b: chess.Board) -> int:
    """White-minus-Black material in pawns."""
    return sum((VAL.get(p.piece_type, 0) if p.color else -VAL.get(p.piece_type, 0))
               for p in b.piece_map().values())


def _who(color: bool) -> str:
    return "White" if color else "Black"


def _say(cp, human_color: str) -> str:
    """Standing in plain words from the human's point of view (no numbers)."""
    return stats.verdict(cp, 1 if human_color == "w" else -1) + " for the human"


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


def _pn(b: chess.Board, sq: int) -> str:
    """Explicit piece name with colour and square: 'white knight on c3'."""
    p = b.piece_at(sq)
    return f"{'white' if p.color else 'black'} {NAME[p.piece_type]} on {chess.square_name(sq)}"


CENTRE = {chess.D4, chess.E4, chess.D5, chess.E5}


def line_net(board: chess.Board, sans: list[str], max_plies: int = 8):
    """Replay a line: (net material change White-minus-Black in pawns, ends in mate?, first move or None)."""
    b = board.copy(stack=False)
    start, first, mate = _mat(b), None, False
    for san in sans[:max_plies]:
        try:
            m = b.parse_san(san)
        except ValueError:
            break
        first = first or m
        b.push(m)
        if b.is_checkmate():
            mate = True
            break
    return _mat(b) - start, mate, first


def move_motifs(board: chess.Board, move: chess.Move) -> list[str]:
    """Explicit facts about one move, played from `board` (the position before it)."""
    b = board.copy(stack=False)
    mover = b.turn
    piece = b.piece_at(move.from_square)
    notes = []
    if b.is_castling(move):
        notes.append("castles" + (" kingside" if chess.square_file(move.to_square) > 4 else " queenside"))
    if b.is_capture(move):
        if b.is_en_passant(move):
            notes.append("captures a pawn en passant")
        else:
            notes.append(f"captures the {_pn(b, move.to_square)}")
    if move.promotion:
        notes.append(f"promotes to a {NAME[move.promotion]}")
    back_rank = 0 if mover else 7
    if (piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            and chess.square_rank(move.from_square) == back_rank
            and chess.square_rank(move.to_square) != back_rank):
        notes.append(f"develops the {NAME[piece.piece_type]}")
    if piece.piece_type == chess.PAWN and move.to_square in CENTRE and not b.is_capture(move):
        notes.append("puts a pawn in the centre")
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
    targets, attacked = [], []         # attacked = all enemy pieces hit; targets = those worth winning
    for sq in b.attacks(to):
        p = b.piece_at(sq)
        if p and p.color != mover and p.piece_type != chess.KING:
            undefended = not b.attackers(not mover, sq)
            text = _pn(b, sq) + (" (undefended)" if undefended else " (defended)")
            attacked.append(text)
            if VAL[p.piece_type] > VAL[piece.piece_type] or undefended:
                targets.append(text)
    if len(targets) + (1 if b.is_check() else 0) >= 2 and targets:
        notes.append("forks the " + " and the ".join(targets) + (" and the king" if b.is_check() else ""))
    elif attacked:
        notes.append("attacks the " + " and the ".join(attacked[:3]))
    protected = [_pn(b, sq) for sq in b.attacks(to)
                 if (p := b.piece_at(sq)) and p.color == mover and p.piece_type != chess.KING
                 and b.attackers(not mover, sq)]
    if protected:
        notes.append("defends the " + " and the ".join(protected[:2]) + ", which is under attack")

    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p and p.color != mover and p.piece_type != chess.KING:
            if b.is_pinned(not mover, sq) and sq not in pinned_before:
                notes.append(f"pins the {_pn(b, sq)} to its king")
            new = (set(b.attackers(mover, sq)) - {to}) - att_before.get(sq, set())
            if new and (VAL[p.piece_type] >= 3 or not b.attackers(not mover, sq)):
                notes.append(f"discovers an attack on the {_pn(b, sq)}")

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
        tail = x.split(" ", 1)[1]
        if x.startswith(me) and not x.endswith(moved_from) and not any(tail.lower() in n.lower() for n in notes):
            notes.append("saves its " + tail)
    seen = []
    return [n for n in notes if not (n in seen or seen.append(n))]


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """One explicit sentence: who moved what from where to where, and what it does."""
    piece = board.piece_at(move.from_square)
    base = (f"{_who(board.turn)}'s {NAME[piece.piece_type]} moves from "
            f"{chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}")
    notes = move_motifs(board, move)
    return base + (": " + "; ".join(notes) if notes else "; it has no immediate tactical effect (a quiet move)")


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
    what = describe_move(b, first) if first in b.legal_moves else ""
    return (f"THREAT CHECK: if {_who(board.turn)} passed, {other} would play {ln['pv_san'][0]} "
            f"(line: {summarize_line(b, ln['pv_san'])}). The position is now {_say(cur, human_color)}; "
            f"if that threat were carried out it would be {_say(after, human_color)}. "
            + (what + "." if what else ""))


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
    what = describe_move(board, m)
    b = board.copy(stack=False)
    b.push(m)
    best_san = (cur_sf or {}).get("best_san")
    best_cp = None if not cur_sf else (10000 if (cur_sf.get("mate") or 0) > 0 else
                                       -10000 if cur_sf.get("mate") is not None else cur_sf.get("eval_cp"))
    if b.is_game_over(claim_draw=True):
        res = "checkmate" if b.is_checkmate() else "a draw"
        return f"MOVE {san}: ends the game in {res}. {what}."
    r = sf.analyse(b.fen(), depth=12, multipv=1)
    ln = r["lines"][0] if r["lines"] else None
    after = None if not ln else (ln["cp"] if ln["mate"] is None else (10000 if ln["mate"] > 0 else -10000))
    out = [f"MOVE {san} (standing is always from the human's point of view)"]
    out.append(f"after it the position is {_say(after, human_color)}"
               + (f"; the best move here is {best_san} ({_say(best_cp, human_color)})" if best_san else ""))
    if after is not None and best_cp is not None:
        sign = 1 if mover else -1
        wl = max(0.0, stats.win_pct(sign * best_cp) - stats.win_pct(sign * after))
        label = label_win_loss(wl)
        swing = stats.swing(best_cp, after, 1 if human_color == "w" else -1)
        if san == best_san or label == "best":
            out.append("this is the best move")
        elif sign * after <= -9000:
            out.append("it walks into a forced mate")
        elif sign * best_cp >= 9000:
            out.append("it gives up a forced mate")
        elif label in ("good", "fine"):
            out.append(f"it is a {label} alternative that gives up almost nothing (the standing {swing})")
        else:
            out.append(f"it is an {label} compared with the best move (the standing {swing})")
    if ln:
        out.append(f"expected continuation: {summarize_line(b, ln['pv_san'])}")
    out.append(f"what the move itself does: {what}")
    return out[0] + ": " + ". ".join(out[1:]) + "."


if __name__ == "__main__":      # self-check: python explain.py (needs no engine)
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert "captures the black pawn on e5" in move_motifs(b, b.parse_san("Nxe5"))
    # Black's Nc6 attacks White's e5 pawn (the old brief gave no description at all)
    d = describe_move(chess.Board("rnbqkb1r/ppp1pppp/3p1n2/4P3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"), chess.Move.from_uci("b8c6"))
    assert d.startswith("Black's knight moves from b8 to c6") and "develops the knight" in d \
        and "the white pawn on e5 (defended)" in d, d
    # ...whereas in the normal Open Game Nc6 defends Black's own e5 pawn
    d2 = describe_move(chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"), chess.Move.from_uci("b8c6"))
    assert "defends the black pawn on e5" in d2 and "saves" not in d2, d2
    # classic knight fork of king and queen: Nc7+ attacks e8 and a8
    g = chess.Board("q3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
    mo = move_motifs(g, g.parse_san("Nc7+"))
    assert any(n.startswith("forks") for n in mo) and "gives check" in mo, mo
    assert "puts its knight on f6 en prise" in move_motifs(chess.Board("3qk3/8/8/8/4N3/8/8/4K3 w - - 0 1"), chess.Move.from_uci("e4f6"))
    assert "delivers checkmate" in move_motifs(chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"), chess.Move.from_uci("a1a8"))
    assert "puts a pawn in the centre" in move_motifs(chess.Board(), chess.Move.from_uci("e2e4"))
    assert "castles kingside" in move_motifs(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), chess.Move.from_uci("e1g1"))
    q = describe_move(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"), chess.Move.from_uci("a1a2"))
    assert "no immediate tactical effect" in q, q
    net, mate, first = line_net(b, ["Nxe5", "Nxe5", "d4"])
    assert net == -2 and not mate and first == chess.Move.from_uci("f3e5"), (net, mate, first)   # +1 pawn, -3 knight
    assert line_net(chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"), ["Ra8#"])[1] is True
    s = summarize_line(b, ["Nxe5", "Nxe5", "d4"])
    assert "White takes a pawn (Nxe5)" in s and "Black takes a knight (Nxe5)" in s, s
    assert summarize_line(b, ["Bc4", "Bc5"]).endswith("material stays level")
    print("ok")
