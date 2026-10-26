"""Facts about a whole game, computed in code so the narrator never has to guess.

Everything here is derived from the move list and the per-move records the
server keeps (rec[i] = {san, color, cp, best, loss, label}, cp = White-POV
centipawns after the move, mate mapped to +/-10000).
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


def _win_pct(cp: float) -> float:
    """Lichess's centipawn -> winning-chances curve (0-100) for the side whose POV cp is in."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * max(-1000, min(1000, cp)))) - 1)


def accuracy(rec: list[dict], color: str) -> int | None:
    """Mean per-move accuracy (lichess formula on win-% lost), 0-100."""
    vals = []
    for i, r in enumerate(rec):
        if r["color"] != color or r.get("cp") is None:
            continue
        sign = 1 if color == "w" else -1
        before = rec[i - 1]["cp"] if i and rec[i - 1].get("cp") is not None else 30
        drop = max(0, _win_pct(sign * before) - _win_pct(sign * r["cp"]))
        vals.append(max(0, min(100, 103.1668 * math.exp(-0.04354 * drop) - 3.1669)))
    return round(sum(vals) / len(vals)) if vals else None


def side_stats(rec: list[dict], color: str) -> dict | None:
    mine = [r for r in rec if r["color"] == color and r.get("loss") is not None]
    if not mine:
        return None
    counts = {k: sum(1 for r in mine if r["label"] == k)
              for k in ("best", "good", "inaccuracy", "mistake", "blunder")}
    return {"moves": len(mine), "accuracy": accuracy(rec, color), "avg_loss": round(sum(min(r["loss"], 500) for r in mine) / len(mine)),
            **counts}


def worst_moments(rec: list[dict], n: int = 3, min_loss: int = 100) -> list[tuple[int, dict]]:
    ranked = sorted(((i, r) for i, r in enumerate(rec) if (r.get("loss") or 0) >= min_loss),
                    key=lambda t: -t[1]["loss"])
    return sorted(ranked[:n])


def fmt_cp(cp: int | None, pov: int = 1) -> str:
    """Centipawns as pawns with a sign; pov=-1 flips White-POV numbers to Black's view."""
    if cp is None:
        return "?"
    if abs(cp) >= 9000:
        return "mate" if cp * pov > 0 else "-mate"
    return f"{cp * pov / 100:+.1f}"


def movelist(rec: list[dict], human_color: str, limit: int = 120) -> str:
    """'1. d4 (+0.3) Nf6 (+0.2)  2. Bxf7?? (-3.1; best Nxe5) ...' — evals from the human's point of view."""
    pov = 1 if human_color == "w" else -1
    start = max(0, len(rec) - limit)
    out = ["(earlier moves omitted)"] if start else []
    for i in range(start, len(rec)):
        r = rec[i]
        if r["color"] == "w":
            out.append(f"{i // 2 + 1}.")
        elif i == start:
            out.append(f"{i // 2 + 1}...")
        note = fmt_cp(r.get("cp"), pov)
        mark = MARK.get(r.get("label"), "")
        if mark and r.get("best") and r["best"] != r["san"]:
            note += f"; best {r['best']}"
        out.append(f"{r['san']}{mark} ({note})")
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
    rec = [{"san": "e4", "color": "w", "cp": 30, "best": "e4", "loss": 0, "label": "best"},
           {"san": "Qh4", "color": "b", "cp": 300, "best": "e5", "loss": 270, "label": "blunder"}]
    assert side_stats(rec, "b")["blunder"] == 1 and worst_moments(rec)[0][0] == 1
    assert accuracy(rec, "w") > 95 > accuracy(rec, "b")
    assert movelist(rec, "w") == "1. e4 (+0.3) Qh4?? (+3.0; best e5)", movelist(rec, "w")
    print("ok")
