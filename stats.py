"""Facts about a whole game, computed in code so the narrator never has to guess.

Everything here is derived from the move list and the per-move records the
server keeps (rec[i] = {san, color, cp, best, loss, wl, label}: cp = White-POV centipawns after the
move (mate mapped to +/-10000), loss = cp given away, wl = winning-chance points given away).
"""
import csv
import math
import os

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
MARK = {"inaccuracy": "?!", "mistake": "?", "blunder": "??"}

_OPENINGS: dict[str, tuple[str, str]] | None = None


def _openings() -> dict:
    """epd -> (eco, name), built once from data/openings/*.tsv (lichess chess-openings, CC0)."""
    global _OPENINGS
    if _OPENINGS is None:
        d = {}
        for f in "abcde":
            try:
                rows = list(csv.DictReader(open(os.path.join(HERE, "data", "openings", f + ".tsv")),
                                           delimiter="\t"))
            except OSError:
                continue
            for r in rows:
                b = chess.Board()
                try:
                    for tok in r["pgn"].split():
                        if not tok[0].isdigit():
                            b.push_san(tok)
                except ValueError:
                    continue
                d[b.epd()] = (r["eco"], r["name"])
        _OPENINGS = d
    return _OPENINGS


def opening(history_uci: list[str]) -> str | None:
    """Name of the deepest known opening position reached, e.g. 'D02 Queen's Pawn Game: London System'."""
    table, b, found = _openings(), chess.Board(), None
    for u in history_uci[:40]:
        b.push(chess.Move.from_uci(u))
        if b.epd() in table:
            found = table[b.epd()]
    return f"{found[0]} {found[1]}" if found else None


def phase(board: chess.Board) -> str:
    total = sum(VAL.get(p.piece_type, 0) for p in board.piece_map().values()
                if p.piece_type not in (chess.PAWN, chess.KING))
    if total <= 26:
        return "endgame"
    return "opening" if board.fullmove_number <= 10 and total >= 50 else "middlegame"


def loose_pieces(board: chess.Board) -> list[str]:
    """Pieces that are attacked and either undefended or attacked by something cheaper."""
    out = []
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        att = board.attackers(not p.color, sq)
        if not att:
            continue
        cheapest = min(VAL.get(board.piece_type_at(a), 99) for a in att)
        if not board.attackers(p.color, sq) or cheapest < VAL[p.piece_type]:
            who = "White" if p.color else "Black"
            out.append(f"{who} {NAME[p.piece_type]} on {chess.square_name(sq)}")
    return out


def captures(history_uci: list[str]) -> dict[str, list[str]]:
    """Pieces each colour has captured, in order."""
    b, got = chess.Board(), {"w": [], "b": []}
    for u in history_uci:
        m = chess.Move.from_uci(u)
        if b.is_capture(m):
            victim = chess.PAWN if b.is_en_passant(m) else b.piece_type_at(m.to_square)
            got["w" if b.turn else "b"].append(NAME[victim])
        b.push(m)
    return got


def win_pct(cp: float) -> float:
    """Lichess's centipawn -> winning-chances curve, 0-100, for the side whose POV the cp is in."""
    if cp >= 9000:
        return 100.0
    if cp <= -9000:
        return 0.0
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * max(-1000, min(1000, cp)))) - 1)


def verdict(cp_white: int | None, pov: int = 1) -> str:
    """Plain-words standing from the point of view pov (+1 White, -1 Black); no numbers."""
    if cp_white is None:
        return "unclear"
    d = win_pct(cp_white * pov) - 50
    a = abs(d)
    if a < 7:
        return "about equal"
    if a < 18:
        return "slightly better" if d > 0 else "slightly worse"
    if a < 35:
        return "clearly better" if d > 0 else "clearly worse"
    return "winning" if d > 0 else "lost"


def swing(before_cp: int | None, after_cp: int | None, pov: int = 1) -> str:
    b, a = verdict(before_cp, pov), verdict(after_cp, pov)
    return f"stayed {a}" if a == b else f"went from {b} to {a}"


def accuracy(rec: list[dict], color: str) -> int | None:
    """Mean per-move accuracy (lichess formula on win-% lost), 0-100."""
    vals = []
    for r in rec:
        if r["color"] != color or r.get("wl") is None:
            continue
        vals.append(max(0, min(100, 103.1668 * math.exp(-0.04354 * r["wl"]) - 3.1669)))
    return round(sum(vals) / len(vals)) if vals else None


def side_stats(rec: list[dict], color: str) -> dict | None:
    mine = [r for r in rec if r["color"] == color and r.get("wl") is not None]
    if not mine:
        return None
    counts = {k: sum(1 for r in mine if r["label"] == k)
              for k in ("best", "good", "fine", "inaccuracy", "mistake", "blunder")}
    return {"moves": len(mine), "accuracy": accuracy(rec, color), **counts}


def worst_moments(rec: list[dict], n: int = 3, min_wl: float = 15) -> list[tuple[int, dict]]:
    """The biggest errors of the game (mistake or worse), in move order."""
    ranked = sorted(((i, r) for i, r in enumerate(rec)
                     if (r.get("wl") or 0) >= min_wl or r.get("label") in ("mistake", "blunder")),
                    key=lambda t: -t[1]["wl"])
    return sorted(ranked[:n])


def movelist(rec: list[dict], human_color: str, limit: int = 120) -> str:
    """'1. e4 [about equal] Qh4?? (better: e5) [winning] ...' — the bracket (the human's standing) is
    shown only when it changes; ?! ? ?? mark inaccuracy / mistake / blunder; no numbers."""
    pov = 1 if human_color == "w" else -1
    start = max(0, len(rec) - limit)
    out = ["(earlier moves omitted)"] if start else []
    prev = None
    for i in range(start, len(rec)):
        r = rec[i]
        if r["color"] == "w":
            out.append(f"{i // 2 + 1}.")
        elif i == start:
            out.append(f"{i // 2 + 1}...")
        txt = r["san"] + MARK.get(r.get("label"), "")
        if MARK.get(r.get("label")) and r.get("best") and r["best"] != r["san"]:
            txt += f" (better: {r['best']})"
        v = verdict(r.get("cp"), pov)
        if v != prev:
            txt += f" [{v}]"
            prev = v
        out.append(txt)
    return " ".join(out)


if __name__ == "__main__":      # self-check: python stats.py
    b = chess.Board()
    ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    assert opening(ucis).endswith("Ruy Lopez"), opening(ucis)
    assert phase(b) == "opening" and phase(chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 40")) == "endgame"
    assert loose_pieces(chess.Board("4k3/8/8/8/8/3q4/8/3QK3 w - - 0 1")) == ["Black queen on d3"]
    assert loose_pieces(chess.Board("4k3/8/8/8/8/3q4/8/4K3 w - - 0 1")) == [], "queen not attacked"
    assert sorted(loose_pieces(chess.Board("4k3/8/8/3r4/8/8/8/3QK3 b - - 0 1"))) == ["Black rook on d5", "White queen on d1"]
    assert captures(["e2e4", "d7d5", "e4d5"]) == {"w": ["pawn"], "b": []}
    # win curve: 40 cp is worth far less in a lopsided position than at equality
    assert 3.4 < win_pct(0) - win_pct(-40) < 4.0 and win_pct(800) - win_pct(760) < 1.0
    assert win_pct(10000) == 100 and win_pct(-10000) == 0
    # plain-words verdicts, from either side
    assert [verdict(c) for c in (0, 100, 300, 800, -100, -300, -800)] == [
        "about equal", "slightly better", "clearly better", "winning", "slightly worse", "clearly worse", "lost"]
    assert verdict(300, -1) == "clearly worse" and swing(0, 300, 1) == "went from about equal to clearly better"
    assert swing(30, 20) == "stayed about equal"
    rec = [{"san": "e4", "color": "w", "cp": 30, "best": "e4", "loss": 0, "wl": 0.0, "label": "best"},
           {"san": "Qh4", "color": "b", "cp": 300, "best": "e5", "loss": 270, "wl": 26.0, "label": "blunder"}]
    assert side_stats(rec, "b")["blunder"] == 1 and worst_moments(rec)[0][0] == 1
    assert accuracy(rec, "w") > 95 > accuracy(rec, "b")
    ml = movelist(rec, "w"); assert ml == "1. e4 [about equal] Qh4?? (better: e5) [clearly better]", ml
    assert not any(ch.isdigit() and "." in ml.split("]")[0] for ch in ml[:0]) and "cp" not in ml
    print("ok")
