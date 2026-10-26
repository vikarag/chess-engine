"""Thin UCI wrapper around a persistent Stockfish process.

Two instances are used by the server: a *judge* at full strength (evals, best
lines, move grading) and a *player* that only ever calls play(), with
UCI_LimitStrength set to the game's Elo.
"""
import subprocess
import threading

import chess


class Stockfish:
    def __init__(self, path: str = "/usr/games/stockfish", threads: int = 2, hash_mb: int = 64):
        self.lock = threading.Lock()
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._read_until("uciok")
        self._set("Threads", threads)
        self._set("Hash", hash_mb)
        self._set("Threads", threads)
        self._send("isready")
        self._read_until("readyok")

    # ---------- plumbing ----------
    def _send(self, cmd: str):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        out = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("stockfish exited")
            line = line.strip()
            out.append(line)
            if line.startswith(token):
                return out

    def _set(self, name: str, value):
        self._send(f"setoption name {name} value {value}")

    def quit(self):
        try:
            self._send("quit")
        except Exception:
            pass

    # ---------- analysis ----------
    def analyse(self, fen: str, depth: int = 12, multipv: int = 3) -> dict:
        board = chess.Board(fen)
        with self.lock:
            self._set("MultiPV", multipv)
            self._send("ucinewgame")
            self._send(f"position fen {fen}")
            self._send(f"go depth {depth}")
            best = None
            lines: dict[int, dict] = {}
            while True:
                line = self.p.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("bestmove"):
                    best = line.split()[1] if len(line.split()) > 1 else None
                    break
                if not line.startswith("info") or " pv " not in line + " ":
                    continue
                parts = line.split()
                d = {"pv": parts[parts.index("pv") + 1:]}
                if "multipv" in parts:
                    d["multipv"] = int(parts[parts.index("multipv") + 1])
                if "depth" in parts:
                    d["depth"] = int(parts[parts.index("depth") + 1])
                if "score" in parts:
                    kind = parts[parts.index("score") + 1]
                    val = int(parts[parts.index("score") + 2])
                    d[kind] = val
                if "pv" in d:
                    lines[d.get("multipv", 1)] = d
        out = []
        for i in sorted(lines):
            ln = lines[i]
            cp = ln.get("cp")
            mate = ln.get("mate")
            if cp is not None and board.turn == chess.BLACK:
                cp = -cp
            if mate is not None and board.turn == chess.BLACK:
                mate = -mate
            out.append({"rank": i, "cp": cp, "mate": mate,
                        "depth": ln.get("depth"), "pv": ln["pv"],
                        "pv_san": _pv_san(board, ln["pv"])})
        return {"best": best, "lines": out, "fen": fen,
                "eval_cp": out[0]["cp"] if out else None,
                "mate": out[0]["mate"] if out else None}

    def play(self, fen: str, elo: int, max_depth: int = 12, movetime_ms: int = 3000) -> str:
        """Best move (uci) at a limited strength (Stockfish 16 accepts UCI_Elo 1320-3190).
        The search ends at max_depth plies or movetime_ms, whichever comes first."""
        with self.lock:
            self._set("UCI_LimitStrength", "true")
            self._set("UCI_Elo", int(elo))
            self._set("MultiPV", 1)
            self._send(f"position fen {fen}")
            self._send(f"go depth {max_depth} movetime {movetime_ms}")
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("stockfish exited")
                if line.startswith("bestmove"):
                    mv = line.split()[1]
                    if mv == "(none)":
                        raise RuntimeError("no legal move")
                    return mv

    def eval_cp(self, fen: str, depth: int = 10) -> int | None:
        """Centipawns from White's point of view (mate mapped to +/-10000)."""
        r = self.analyse(fen, depth=depth, multipv=1)
        if r["mate"] is not None:
            return 10000 if r["mate"] > 0 else -10000
        return r["eval_cp"]

    def best_move(self, fen: str, depth: int = 12) -> dict:
        return self.analyse(fen, depth=depth, multipv=1)


def _pv_san(board: chess.Board, pv: list[str]) -> list[str]:
    b = board.copy(stack=False)
    out = []
    for u in pv[:10]:
        try:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                break
            out.append(b.san(m))
            b.push(m)
        except Exception:
            break
    return out


def label_win_loss(wl: float) -> str:
    """Label a move by the winning chances (percentage points, mover's point of view) it gave away.
    About 11 cp near equality is 1 point, and the same cp is worth less in a lopsided position.
    best <1, good <4, fine <8 (not worth a comment), inaccuracy <15, mistake <25, blunder beyond."""
    if wl < 1:
        return "best"
    if wl < 4:
        return "good"
    if wl < 8:
        return "fine"
    if wl < 15:
        return "inaccuracy"
    return "mistake" if wl < 25 else "blunder"
