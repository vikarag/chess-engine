"""
LFM2.5-230M-Chess inference core — faithful port of the reference protocol
used by the ChessLFM Space (mlabonne), running from GGUF on llama.cpp.

Protocol (80 tokens, fixed):
  <|pos|> <c:X>*64 <stm:w|b> <cast:...> <ep:...> <hm:n> <rep:n>
  <|hist|> <m:uci>*8  <|eval|>
Pass 1: logits at last position -> argmax over 64 <v:k> tokens -> win prob
Pass 2: append <v:k> <|bestmove|> (2 tokens, reuses KV cache) -> logits
        masked to legal moves -> policy distribution
"""
import json
import os
import time
from dataclasses import dataclass, field

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "meta", "spec_tokens.json")))
TOKENS = SPEC["tokens"]          # 2106 spec tokens, ids BASE_ID..BASE_ID+2105
BASE_ID = SPEC["base_id"]        # 64402
IDX = {t: i for i, t in enumerate(TOKENS)}

CASTLING = ["-", "q", "k", "kq", "Q", "Qq", "Qk", "Qkq",
            "K", "Kq", "Kk", "Kkq", "KQ", "KQq", "KQk", "KQkq"]
MAX_HIST = 8
NUM_VALUE_BINS = 64
PROMPT_LEN = 80


def _castling_bits(board: chess.Board) -> int:
    b = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        b |= 8
    if board.has_queenside_castling_rights(chess.WHITE):
        b |= 4
    if board.has_kingside_castling_rights(chess.BLACK):
        b |= 2
    if board.has_queenside_castling_rights(chess.BLACK):
        b |= 1
    return b


def _ep_field(board: chess.Board) -> str:
    """chess.js (used to build training prompts) only emits an ep square when
    an en-passant capture is actually legal, unlike python-chess's FEN."""
    if board.ep_square is None:
        return "-"
    for m in board.legal_moves:
        if m.to_square != board.ep_square:
            continue
        p = board.piece_at(m.from_square)
        if p and p.piece_type == chess.PAWN and not board.piece_at(m.to_square):
            return chess.square_name(board.ep_square)[0]
    return "-"


def build_prompt(board: chess.Board, history_uci: list[str], rep: int = 0) -> list[str]:
    out = ["<|pos|>"]
    for rank in range(7, -1, -1):
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            out.append(f"<c:{p.symbol() if p else '.'}>")
    out.append(f"<stm:{'w' if board.turn else 'b'}>")
    out.append(f"<cast:{CASTLING[_castling_bits(board)]}>")
    out.append(f"<ep:{_ep_field(board)}>")
    out.append(f"<hm:{min(board.halfmove_clock, 100) // 4}>")
    out.append(f"<rep:{min(max(rep, 0), 2)}>")
    out.append("<|hist|>")
    hist = history_uci[-MAX_HIST:]
    out.extend(["<m:0000>"] * (MAX_HIST - len(hist)))
    out.extend(f"<m:{u}>" for u in hist)
    out.append("<|eval|>")
    assert len(out) == PROMPT_LEN, len(out)
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


@dataclass
class Candidate:
    uci: str
    san: str
    logit: float
    prob: float
    value_after: float | None = None


@dataclass
class Analysis:
    best: Candidate
    candidates: list[Candidate]
    winprob: float
    value_token: str
    nodes: int = 2
    pv: list[str] = field(default_factory=list)


class ChessLFM:
    def __init__(self, model_path: str, n_threads: int = 2, n_ctx: int = 256, seed: int = 0):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_batch=n_ctx,
            logits_all=True, use_mmap=True, seed=seed, verbose=False,
        )
        self._id_cache: dict[str, int] = {}
        self._sanity()

    # ---------- tokenizer ----------
    def tid(self, tok: str) -> int:
        i = self._id_cache.get(tok)
        if i is None:
            ids = self.llm.tokenize(tok.encode(), add_bos=False, special=True)
            if len(ids) != 1:
                raise ValueError(f"{tok!r} is not a single token: {ids}")
            i = ids[0]
            self._id_cache[tok] = i
        return i

    def ids_for(self, toks: list[str]) -> list[int]:
        return [self.tid(t) for t in toks]

    def _sanity(self):
        bad = [t for t in ("<|pos|>", "<|eval|>", "<c:P>", "<m:e2e4>", "<v:0>", "<v:63>")
               if self.tid(t) != BASE_ID + IDX[t]]
        if bad:
            raise RuntimeError(f"tokenizer/vocab mismatch: {bad}")

    # ---------- core ----------
    def _pos_analysis(self, board: chess.Board, history_uci: list[str], rep: int = 0):
        """One 80-token prefll + 2-token continuation reusing the KV cache."""
        ids = self.ids_for(build_prompt(board, history_uci, rep))
        self.llm.reset()
        self.llm.eval(ids)
        lg1 = np.array(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float32)
        v_ids = np.array([BASE_ID + IDX[f"<v:{k}>"] for k in range(NUM_VALUE_BINS)])
        k = int(np.argmax(lg1[v_ids]))
        self.llm.eval([BASE_ID + IDX[f"<v:{k}>"], BASE_ID + IDX["<|bestmove|>"]])
        lg2 = np.array(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float32)
        return lg2, k

    def policy(self, board: chess.Board, history_uci: list[str], rep: int = 0) -> Analysis:
        lg, k = self._pos_analysis(board, history_uci, rep)
        return self._policy_from_logits(board, lg, (k + 0.5) / NUM_VALUE_BINS, k)

    def _policy_from_logits(self, board, lg, winprob, k) -> Analysis:
        ids, mvs, sans = [], [], []
        for m in board.legal_moves:
            u = m.uci()
            t = f"<m:{u}>"
            if t not in IDX:
                continue          # outside the model's action space (rare)
            ids.append(BASE_ID + IDX[t])
            mvs.append(u)
            sans.append(board.san(m))
        if not ids:
            raise RuntimeError("no legal move in the model action space")
        logits = lg[np.array(ids)].astype(np.float64)
        probs = softmax(logits)
        cands = sorted([Candidate(u, s, float(l), float(p))
                        for u, s, l, p in zip(mvs, sans, logits, probs)],
                       key=lambda c: -c.prob)
        return Analysis(best=cands[0], candidates=cands, winprob=winprob,
                        value_token=f"<v:{k}>", nodes=2)

    def winprob(self, board: chess.Board, history_uci: list[str], rep: int = 0) -> float:
        lg, k = self._pos_analysis(board, history_uci, rep)
        return (k + 0.5) / NUM_VALUE_BINS

    # ---------- search ----------
    def analyse(self, board: chess.Board, history_uci: list[str], rep: int = 0,
                depth: int = 1, root_k: int = 4, child_k: int = 3,
                time_budget: float | None = None) -> Analysis:
        """Value-head negamax. depth=1 is the raw policy (~1500 Elo).
        Each extra ply costs one 2.4s prefill per examined node on this CPU."""
        import time
        t0 = time.time()
        base = self.policy(board, history_uci, rep)
        if depth <= 1:
            return base
        nodes = 2
        ranked = []
        for c in base.candidates[:root_k]:
            if time_budget and time.time() - t0 > time_budget:
                break
            b = board.copy(stack=False)
            b.push(chess.Move.from_uci(c.uci))
            sc, n = self._negamax(b, history_uci + [c.uci], depth - 1, child_k, rep, t0, time_budget)
            nodes += n
            c.value_after = sc
            ranked.append(c)
        if not ranked:
            return base
        best = max(ranked, key=lambda c: c.value_after)
        # surface the searched moves on top, keeping policy order for the rest
        seen = {c.uci for c in ranked}
        out_c = sorted(ranked, key=lambda c: -c.value_after) + \
                [c for c in base.candidates if c.uci not in seen]
        return Analysis(best=best, candidates=out_c, winprob=base.winprob,
                        value_token=base.value_token, nodes=nodes)

    def _negamax(self, board, hist, depth, child_k, rep, t0, budget):
        if board.is_checkmate():
            return 0.0, 1
        if board.is_game_over(claim_draw=True):
            return 0.5, 1
        if budget and time.time() - t0 > budget:
            return 0.5, 0
        if depth <= 0:
            return self.winprob(board, hist, rep), 1
        pol = self.policy(board, hist, rep)
        vals, n = [], 1
        for c in pol.candidates[:child_k]:
            b = board.copy(stack=False)
            b.push(chess.Move.from_uci(c.uci))
            v, m = self._negamax(b, hist + [c.uci], depth - 1, child_k, rep, t0, budget)
            vals.append(1.0 - v)
            n += m
        return (max(vals) if vals else 0.5), n


def repetition_count(fen_history: list[str]) -> int:
    """How many times the current position already occurred (for <rep:>)."""
    if not fen_history:
        return 0
    key = " ".join(fen_history[-1].split()[:4])
    return max(0, sum(1 for f in fen_history if " ".join(f.split()[:4]) == key) - 1)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-230M-Chess.Q8_0.gguf"))
    ap.add_argument("--fen", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--depth", type=int, default=1)
    a = ap.parse_args()
    eng = ChessLFM(a.model, n_threads=a.threads)
    b = chess.Board(a.fen) if a.fen else chess.Board()
    t = time.time()
    an = eng.analyse(b, [], depth=a.depth)
    print(f"{time.time()-t:.2f}s  winprob={an.winprob:.3f}  best={an.best.san}  nodes={an.nodes}")
    for c in an.candidates[:6]:
        print(f"  {c.san:6s} {c.prob:6.1%}  logit={c.logit:+.2f}"
              + (f"  search={c.value_after:.3f}" if c.value_after is not None else ""))
