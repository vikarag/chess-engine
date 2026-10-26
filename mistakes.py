"""Name the TYPE of a weak move from the board and the judge's lines, and give the habit that prevents it.

All detection is plain board logic (python-chess) on the position before the move, the move played, the
judge's best line there, and the judge's best line after the move. Nothing is guessed: a type is only
reported when its condition holds, and when none does the move is called positional (and the coach is told
to say so).
"""
import chess

import explain
import stats

VAL, NAME = stats.VAL, stats.NAME

LESSONS = {
    "walked into mate": "Before every move, ask what forcing moves your opponent has (checks, captures, threats), "
                        "above all against your own king.",
    "hanging piece": "Before you move, check that the piece you move, and every piece you leave behind, cannot "
                     "simply be captured for free.",
    "undefended pawn": "Before you move, check which of your pawns and pieces are unprotected; protect them or make "
                       "sure they cannot be attacked.",
    "allowed a tactic": "After you pick a move, look at your opponent's checks, captures and threats in reply: "
                        "forks, pins and discovered attacks hide there.",
    "missed a chance": "Before settling on a quiet move, look at every capture, check and threat you have; one of "
                       "them may win material.",
    "king safety": "Keep your king safe: castle early, and don't push pawns in front of your king or give up "
                   "castling without a concrete reason.",
    "development": "In the opening, bring a new piece into play with almost every move, instead of moving the same "
                   "piece again, pushing edge pawns or sending the queen out early.",
    "positional": "",
    "unclear": "",
}
ORDER = ["walked into mate", "hanging piece", "undefended pawn", "allowed a tactic", "missed a chance",
         "king safety", "development", "positional", "unclear"]


def _line(board: chess.Board, sans: list[str], max_plies: int = 8):
    """Replay a line: (material change for the side to move on `board`, final board, first move)."""
    b = board.copy(stack=False)
    side, start, first = b.turn, explain._mat(b), None
    for san in sans[:max_plies]:
        try:
            m = b.parse_san(san)
        except ValueError:
            break
        first = first or m
        b.push(m)
        if b.is_checkmate():
            break
    return (explain._mat(b) - start) * (1 if side else -1), b, first


def _shield(b: chess.Board, color: bool):
    """(pawns directly in front of a castled-looking king, adjacent files with no pawns at all) or None."""
    ksq = b.king(color)
    if ksq is None or chess.square_rank(ksq) != (0 if color else 7) or chess.square_file(ksq) in (3, 4):
        return None
    kf, step = chess.square_file(ksq), 1 if color else -1
    shield, open_files = 0, 0
    for f in (kf - 1, kf, kf + 1):
        if not 0 <= f <= 7:
            continue
        own = [r for r in (chess.square_rank(ksq) + step, chess.square_rank(ksq) + 2 * step)
               if 0 <= r <= 7 and (p := b.piece_at(chess.square(f, r))) and p.piece_type == chess.PAWN and p.color == color]
        shield += 1 if own else 0
        if not any((p := b.piece_at(chess.square(f, r))) and p.piece_type == chess.PAWN for r in range(8)):
            open_files += 1
    return shield, open_files


def _undeveloped(b: chess.Board, color: bool) -> list[str]:
    home = {True: {chess.B1: "knight", chess.G1: "knight", chess.C1: "bishop", chess.F1: "bishop"},
            False: {chess.B8: "knight", chess.G8: "knight", chess.C8: "bishop", chess.F8: "bishop"}}[color]
    return [f"{n} on {chess.square_name(sq)}" for sq, n in home.items()
            if (p := b.piece_at(sq)) and p.color == color and NAME[p.piece_type] == n]


def lost_material(board: chess.Board, move: chess.Move, refut_pv: list[str]) -> float:
    """Pawn-points the mover loses along the opponent's best line after `move` (0 if none)."""
    a = board.copy(stack=False)
    a.push(move)
    return max(0.0, _line(a, refut_pv or [])[0])


def classify(board: chess.Board, move: chess.Move, best_san: str | None, refut_pv: list[str],
             best_pv: list[str], wl: float | None = None) -> list[dict]:
    """Types of mistake for `move` played on `board`: [{tag, detail, lesson}], most important first."""
    mover = board.turn
    a = board.copy(stack=False)
    a.push(move)
    found: dict[str, str] = {}

    opp_net, fin, first = _line(a, refut_pv or [])
    lost = opp_net                                    # the opponent's gain along its best line = what the mover loses (pawn-points)
    if fin.is_checkmate() and fin.turn == mover:
        found["walked into mate"] = "your opponent's best reply starts a forced line that ends in checkmate against you"
    elif first is not None and lost >= 1 and a.is_capture(first):
        sq = first.to_square
        victim = chess.Piece(chess.PAWN, mover) if a.is_en_passant(first) else a.piece_at(sq)
        if victim is not None and victim.color == mover:
            taker = a.piece_at(first.from_square)
            where = chess.square_name(sq if not a.is_en_passant(first) else first.to_square)
            undefended = not a.attackers(mover, sq)
            if victim.piece_type == chess.PAWN:
                found["undefended pawn"] = (f"your pawn on {where} can be captured by the {NAME[taker.piece_type]} "
                                            f"on {chess.square_name(first.from_square)}"
                                            + (" and nothing protects it" if undefended else ""))
            else:
                found["hanging piece"] = (f"your {NAME[victim.piece_type]} on {where} can be captured by the "
                                          f"{NAME[taker.piece_type]} on {chess.square_name(first.from_square)}"
                                          + (" and nothing protects it" if undefended
                                             else ", a cheaper piece, so the exchange loses material"))
    elif first is not None and (lost >= 2 or (lost >= 1 and (a.is_capture(first) or a.gives_check(first)))):
        # a forcing reply (check or capture) that wins material, even if a sacrifice is won back on the way
        motif = next((m for m in explain.move_motifs(a, first) if m.startswith(("forks", "pins", "discovers"))), None)
        seq = explain.summarize_line(a, (refut_pv or [])[:6])
        found["allowed a tactic"] = ((f"your opponent's best reply {motif}; " if motif else "")
                                     + "your opponent has a forcing sequence that wins material: " + seq)

    if best_pv:
        bnet, bfin, bfirst = _line(board, best_pv)
        if bfin.is_checkmate() and bfin.turn != mover:
            found["missed a chance"] = f"{best_san or best_pv[0]} started a forced checkmate"
        elif bnet >= 1 and bfirst is not None and (board.is_capture(bfirst) or board.gives_check(bfirst)):
            found["missed a chance"] = (f"{best_san or best_pv[0]} " +
                                        ("wins material" if board.is_capture(bfirst) else "starts a line that wins material"))

    # king safety: forfeited castling, ignored a castling move, or loosened the cover of a castled king
    if board.has_castling_rights(mover) and not a.has_castling_rights(mover) and not board.is_castling(move):
        found["king safety"] = "you gave up the right to castle without castling"
    elif best_san and best_san.startswith("O-O") and not board.is_castling(move):
        found["king safety"] = "your king is still in the centre; castling was the best move"
    else:
        s0, s1 = _shield(board, mover), _shield(a, mover)
        if s0 and s1 and (s1[0] < s0[0] or s1[1] > s0[1]) and board.piece_at(move.from_square).piece_type == chess.PAWN:
            found["king safety"] = "the pawn move loosens the pawns in front of your king"

    # development (opening only; captures and checks are exempt)
    p = board.piece_at(move.from_square)
    behind = _undeveloped(board, mover)
    if (stats.phase(board) == "opening" and len(behind) >= 2 and not board.is_capture(move)
            and not a.is_check() and not board.is_castling(move)):
        n = len(behind)
        if p.piece_type == chess.QUEEN:
            found["development"] = f"an early queen move while {n} minor pieces are still at home ({behind[0]}, ...)"
        elif p.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK) and move.from_square not in (
                chess.B1, chess.G1, chess.C1, chess.F1, chess.B8, chess.G8, chess.C8, chess.F8):
            found["development"] = (f"you moved an already developed {NAME[p.piece_type]} while {n} minor pieces "
                                    f"are still at home ({behind[0]}, ...)")
        elif p.piece_type == chess.PAWN and chess.square_file(move.from_square) in (0, 7):
            found["development"] = f"an edge-pawn move while {n} minor pieces are still at home ({behind[0]}, ...)"

    if not found and (wl or 0) >= 25:
        # a blunder-sized drop whose cause none of the patterns names: never call that "positional"
        found["unclear"] = ("a large loss whose cause is not one of the simple patterns; describe only the line "
                            "given after the move and the better move, and do not call it positional")
    elif not found:
        found["positional"] = ("no tactic or concrete weakness was found; the better move simply keeps a more "
                               "favourable position")
    return [{"tag": t, "detail": found[t], "lesson": LESSONS[t]} for t in ORDER if t in found][:2]


if __name__ == "__main__":      # self-check: python mistakes.py (needs no engine)
    def tags(fen, uci, best, refut, best_pv):
        b = chess.Board(fen)
        return {t["tag"]: t for t in classify(b, chess.Move.from_uci(uci), best, refut, best_pv)}
    # a queen put on a square a pawn attacks
    t = tags("4k3/8/2p5/8/8/8/8/3QK3 w - - 0 1", "d1d5", "Qd2", ["cxd5"], ["Qd2"])
    assert "hanging piece" in t and "queen on d5" in t["hanging piece"]["detail"] and "nothing protects" in t["hanging piece"]["detail"], t
    # a pawn lost for nothing
    t = tags("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1", "e2e3", "Kd2", ["dxe3"], ["Kd2"])
    assert "undefended pawn" in t or "positional" in t, t
    # missed a free queen
    t = tags("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1", "d1d2", "Qxd5", [], ["Qxd5"])
    assert "missed a chance" in t and "wins material" in t["missed a chance"]["detail"], t
    # walked into a back-rank mate
    t = tags("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1", "g8h8", "h6", ["Ra8#"], ["h6"])
    assert "walked into mate" in t, t
    # edge pawn move with all four minor pieces still at home
    t = tags("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "a2a3", "Nf3", ["Nf6"], ["Nf3"])
    assert "development" in t and "edge-pawn" in t["development"]["detail"], t
    # gave up castling by moving the king
    t = tags("r3k2r/pppqbppp/2n1bn2/3pp3/3PP3/2NBBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "e1d1", "O-O-O", [], ["O-O-O"])
    assert "king safety" in t, t
    # nothing concrete -> positional, with no lesson text
    t = tags("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e1d1", "e4", ["Kd7"], ["e4"])
    assert list(t) == ["positional"] and t["positional"]["lesson"] == "", t
    # the Qh6 game: a queen sacrifice (Qf8+) that wins the queen back is a tactic, not "positional"
    t = tags("4k1n1/p1pr2p1/5q2/2Q3p1/5P2/4p1Pr/1PP3K1/RNB2R2 b - - 0 20", "f6h6", "e2",
             ["Qf8+", "Kxf8", "fxg5+", "Nf6", "gxh6", "e2", "Re1", "Rd1"], ["e2"])
    assert "allowed a tactic" in t and "Qf8+ Kxf8 fxg5+" in t["allowed a tactic"]["detail"], t
    # an unexplained blunder-sized drop is never called positional
    b0 = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    assert [x["tag"] for x in classify(b0, chess.Move.from_uci("e1d1"), "e4", ["Kd7"], ["e4"], wl=30)] == ["unclear"]
    print("ok")
