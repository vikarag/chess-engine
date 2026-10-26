"""The talking layer.

Verified fact: the 230M chess model itself cannot chat (it returns an empty
string for any natural-language prompt — its vocabulary tail was repurposed
for chess tokens and it was distilled only on positions). So conversation is
produced by a narrator LLM that is handed a factual brief assembled from

  * the model's own policy distribution + value head (its "intuition"),
  * Stockfish's eval, best move and principal variations (ground truth),
  * move grades and material.

The narrator is told to use only those facts, so tactics it mentions are real.
"""
import json
import os
import re
from typing import Iterator

import chess
import httpx

def _load_env() -> None:
    """Read KEY=value lines from ./.env (git-ignored) into os.environ, without overriding."""
    try:
        for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*['\"]?([^'\"#\n]*)", line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip())
    except OSError:
        pass


_load_env()
MODEL = os.environ.get("CHESSLFM_MODEL", "deepseek/deepseek-v4.1-flash-1017")
BASE_URL = os.environ.get("CHESSLFM_BASE_URL", "https://openrouter.ai/api/v1")


def _provider_pref() -> dict | None:
    """Optional OpenRouter provider pinning, e.g. CHESSLFM_PROVIDER=sail-research."""
    p = os.environ.get("CHESSLFM_PROVIDER")
    return {"order": [p], "allow_fallbacks": True} if p else None


def _api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or None


SYSTEM = """You are the opponent in a live casual chess game, chatting with the \
person you are playing. You are a strong club player (around 2000) with a dry, \
friendly, slightly playful manner.

You will receive an ANALYSIS BRIEF built from real engine output. Hard rules:
- Only make chess claims supported by the brief. Never invent variations, \
tactics, threats, or evaluations that are not in it.
- If you quote a line, quote the principal variation given in the brief.
- Numbers you cite (percentages, evaluations, centipawns) must come from the brief.
- Evaluations are given from your own point of view in a dedicated line; use that \
line, never call a White-POV score "mine".
- Length: 1-3 sentences for a quick remark or a one-line reply. When the user asks \
you to explain a position, plan, or tactic, give a fuller answer of about 120-200 \
words in at most two short paragraphs. No headers, no bullet lists, no tables.
- In an explanation, name the concrete moves from the brief (the main line, the \
candidate moves, the last move) and say what each one is for: what it attacks, \
defends, or prepares. One well-explained idea beats a vague list.
- You are only ONE side. Moves listed as legal for the side to move belong to \
whoever is on move; never present the human's moves or ideas as your own, and \
never promise a move you cannot legally play.
- Talk like a player at the board, not like a chess engine. First person.
- Reply in the language the user writes in (Korean if they write Korean).
- If the user asks for a hint or the best move, give it from the brief.
"""


def fmt_eval(cp: int | None, mate: int | None = None,
             pov: str = "White's point of view") -> str:
    if mate is not None:
        return f"mate in {abs(mate)} for {'White' if mate > 0 else 'Black'}"
    if cp is None:
        return "unknown"
    return f"{cp/100:+.2f} ({pov})"


def build_brief(state) -> str:
    """state is server.GameState; kept duck-typed to avoid a circular import."""
    b: chess.Board = state.board
    side = "White" if b.turn else "Black"
    bot_white = state.human_color == "b"
    bot = "White" if bot_white else "Black"
    human = "Black" if bot_white else "White"
    # _material() is human-POV, so negate for the bot's point of view
    lines = [
        f"FEN: {b.fen()}",
        f"Move number: {b.fullmove_number}. Side to move: {side}. "
        f"You are playing {bot}; the human is playing {human}.",
        f"Material (your side minus theirs, pawns): {-_material(b, state.human_color):+.1f}",
    ]
    if state.history_san:
        lines.append("Moves so far: " + " ".join(state.history_san[-24:]))
    if state.last_engine:
        le = state.last_engine
        top = ", ".join(f"{t['san']} {t['prob']*100:.0f}%" for t in le.get("top", [])[:4])
        lines.append(
            f"Your last move: {le['san']}. Your own policy preferred: {top}. "
            f"Your net's win probability for the side to move after it: {le.get('winprob', 0):.2f}."
        )
    if state.last_grade:
        g = state.last_grade
        lines.append(
            f"Human's last move: {g.get('san', '?')} — "
            f"Stockfish grades it '{g['label']}' ({g['loss_cp']} centipawn loss)."
        )
    if state.sf:
        sf = state.sf
        pv = " ".join(sf.get("pv_san") or [])
        cp, mate = sf.get("eval_cp"), sf.get("mate")
        # translate the White-POV number into the bot's own point of view
        sign = 1 if state.human_color == "b" else -1
        my_cp = None if cp is None else cp * sign
        my_mate = None if mate is None else mate * sign
        lines.append(
            f"Stockfish (depth {sf.get('depth', 12)}): eval {fmt_eval(cp, mate)}, "
            f"best move {sf.get('best_san') or sf.get('best')}"
            + (f", main line: {pv}" if pv else "") + "."
        )
        if my_mate is not None:
            pov_txt = f"mate in {abs(my_mate)} for {'you' if my_mate > 0 else 'the human'}"
        elif my_cp is not None:
            pov_txt = f"{my_cp/100:+.2f} (your point of view)"
        else:
            pov_txt = "unknown"
        lines.append(
            f"FROM YOUR OWN POINT OF VIEW the evaluation is {pov_txt} "
            f"(positive = good for you). Quote this when you say who stands better."
        )
        alt = []
        for ln in (sf.get("lines") or [])[1:3]:
            alt.append(f"{ln.get('pv_san', ['?'])[0]} ({fmt_eval(ln.get('cp'), ln.get('mate'))})")
        if alt:
            lines.append("Stockfish's other candidates: " + ", ".join(alt) + ".")
    legal = [b.san(m) for m in b.legal_moves]
    # "yours" = the bot's. If the side to move is the human's colour, these are HIS moves.
    yours = "the human's" if (("w" if b.turn else "b") == state.human_color) else "yours"
    lines.append(f"Legal moves for {side} (side to move, {yours}) ({len(legal)}): "
                 + ", ".join(legal))
    if b.is_checkmate():
        lines.append("The game is over by checkmate.")
    elif b.is_game_over():
        lines.append("The game is over.")
    return "\n".join(lines)


def _material(b: chess.Board, human_color: str) -> float:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    mine = sum(vals.get(p.piece_type, 0) for p in b.piece_map().values()
               if (p.color == chess.WHITE) == (human_color == "w"))
    theirs = sum(vals.get(p.piece_type, 0) for p in b.piece_map().values()
                 if (p.color == chess.BLACK) == (human_color == "w"))
    return mine - theirs


def templated_reply(brief_state, question: str) -> str:
    """Deterministic fallback so chat still answers without any LLM."""
    s = brief_state
    parts = []
    if s.last_engine:
        le = s.last_engine
        top = le.get("top", [])
        alt = top[1]["san"] if len(top) > 1 else None
        parts.append(f"I played {le['san']}"
                     + (f", though {alt} was close" if alt else "")
                     + f" — my net gave it {le['prob']*100:.0f}% and rated my position "
                       f"{le.get('winprob', 0)*100:.0f}%.")
    if s.sf:
        pv = " ".join((s.sf.get("pv_san") or [])[:4])
        parts.append(f"Stockfish says {fmt_eval(s.sf.get('eval_cp'), s.sf.get('mate'))}"
                     + (f", main line {pv}" if pv else "") + ".")
    if s.last_grade and s.last_grade.get("label") in ("mistake", "blunder", "inaccuracy"):
        parts.append(f"Your last move was an {s.last_grade['label']} "
                     f"({s.last_grade['loss_cp']} centipawns).")
    parts.append("(Chat narration is running in offline mode — the narrator did not answer.)")
    return " ".join(parts)


def stream_reply(state, question: str, history: list[dict]) -> Iterator[str]:
    key = _api_key()
    if not key:
        yield templated_reply(state, question)
        return
    brief = build_brief(state)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "system", "content": "ANALYSIS BRIEF\n" + brief}]
    for h in history[-8:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": question})

    payload = {"model": MODEL, "messages": msgs, "temperature": 0.6,
               "max_tokens": 600, "stream": True,
               # deepseek-v4.1-flash is a reasoning model: left alone it can burn the whole
               # budget on `reasoning` and emit content=null. Turn thinking off instead.
               "reasoning": {"enabled": False}}
    if _provider_pref():
        payload["provider"] = _provider_pref()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "http://localhost", "X-Title": "chess-lfm"}
    got = False
    try:
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", f"{BASE_URL}/chat/completions",
                               json=payload, headers=headers) as r:
                if r.status_code != 200:
                    body = r.read().decode(errors="replace")[:300]
                    yield f"[narrator error {r.status_code}: {body}] "
                    yield templated_reply(state, question)
                    return
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        # show only content; reasoning, if any, stays hidden
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    except Exception:
                        continue
                    if delta:
                        got = True
                        yield delta
        if not got:
            yield templated_reply(state, question)
    except httpx.TimeoutException:
        yield "[narrator did not answer in time] "
        yield templated_reply(state, question)
    except httpx.ConnectError:
        yield "[narrator unreachable] "
        yield templated_reply(state, question)
    except Exception as e:
        yield f"[narrator unavailable: {type(e).__name__}] "
        yield templated_reply(state, question)
