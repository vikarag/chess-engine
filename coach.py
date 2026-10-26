"""The talking layer.

The opponent is a strength-limited Stockfish; a separate full-strength Stockfish
judges every move. The narrator LLM never calculates anything: it is handed a
factual brief built from those records (see stats.py) — the whole annotated
game, per-side statistics, key mistakes, the current position — and is told to
use only that, so any tactic or number it mentions is real.
"""
import json
import os
import re
from typing import Iterator

import chess
import httpx

import explain
import stats

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
MODEL = os.environ.get("CHESS_ENGINE_MODEL", "deepseek/deepseek-v4.1-flash-1017")
BASE_URL = os.environ.get("CHESS_ENGINE_BASE_URL", "https://openrouter.ai/api/v1")


def _provider_pref() -> dict | None:
    """Optional OpenRouter provider pinning, e.g. CHESS_ENGINE_PROVIDER=sail-research."""
    p = os.environ.get("CHESS_ENGINE_PROVIDER")
    return {"order": [p], "allow_fallbacks": True} if p else None


def _api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or None


SYSTEM = """You are the opponent in a live casual chess game, chatting with the \
person you are playing. You are a strong club player with a dry, friendly, \
slightly playful manner. The brief states your playing strength (an Elo-limited \
engine); your moves are sometimes slightly inaccurate on purpose, and the brief \
shows how the judge graded each of them.

You will receive an ANALYSIS BRIEF built from real engine output. Hard rules:
- Only make chess claims supported by the brief. Never invent variations, \
tactics, threats, or evaluations that are not in it.
- If you quote a line, quote the principal variation given in the brief.
- Numbers you cite (percentages, evaluations, cp losses) must come from the brief.
- Always write the abbreviation "cp" (e.g. "lost 90 cp"); never write the word "centipawn" or "centipawns".
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
- Reply in English unless the user's latest message is written in another language; then reply in that language.
- If the user asks for a hint or the best move, give it from the brief.
- For questions about the whole game (accuracy, mistakes, turning points, the \
opening, who was better when), answer from the STATS, KEY MOMENTS, OPENING and \
MOVES sections. Move-list evaluations are from the HUMAN's point of view \
(positive = good for the human); flip the sign when speaking about yourself.
- Never mention a move, square or idea that is not in the brief. If the brief \
does not contain what the user asks, say you don't have it.
- To explain WHY (why a move is good or bad, why not another move, what the \
opponent threatens), use the LAST MOVE EXPLAINED and BEST LINE sections, and call \
the tools for anything else: analyze_move(move) evaluates any candidate move in \
the current position (use it for "what if I play X?" and "why not X?"), and \
get_threats() shows what the opponent would do if the side to move passed. \
Stockfish gives numbers and lines, not reasons: explain only what the tool \
results and sections show (captures, checks, material won or lost, patterns \
named there, the eval gap to other moves). If the real reason is positional and \
no pattern is listed, say Stockfish simply rates it higher by N cp rather than \
inventing a concept. For your own moves, remember you play at a limited \
strength: if the judge preferred another move, say so honestly.
- When you quote a move's grade, use the label and cp loss from the brief; \
"best X" in the move list is the move Stockfish preferred instead.
"""


def fmt_eval(cp: int | None, mate: int | None = None,
             pov: str = "White's point of view") -> str:
    if mate is not None:
        return f"mate in {abs(mate)} for {'White' if mate > 0 else 'Black'}"
    if cp is None:
        return "unknown"
    return f"{cp/100:+.2f} ({pov})"


def build_brief(state) -> str:
    """state is server.Game; kept duck-typed to avoid a circular import."""
    b: chess.Board = state.board
    side = "White" if b.turn else "Black"
    bot_white = state.human_color == "b"
    bot = "White" if bot_white else "Black"
    human = "Black" if bot_white else "White"
    hc, bc = state.human_color, ("w" if bot_white else "b")
    rec = state.rec
    pawns = _material(b, state.human_color)          # human-POV, pawns
    lines = [
        "GAME",
        f"You are playing {bot} at a limited strength of about Elo {state.elo}; "
        f"the human is playing {human}. Move {b.fullmove_number}, {side} to move, "
        f"phase: {stats.phase(b)}.",
        f"OPENING: {stats.opening(state.history_uci) or 'no named opening (out of book or too few moves)'}",
        f"FEN: {b.fen()}",
        f"Material (your side minus theirs, pawns): {-pawns:+.1f}",
    ]
    cap = stats.captures(state.history_uci)
    if cap["w"] or cap["b"]:
        lines.append(f"Captured so far — by {bot}: {', '.join(cap[bc]) or 'nothing'}; "
                     f"by {human}: {', '.join(cap[hc]) or 'nothing'}.")
    loose = stats.loose_pieces(b)
    if loose:
        lines.append("Attacked and loose (undefended, or attacked by a cheaper piece): "
                     + "; ".join(loose) + ".")
    if b.is_check():
        lines.append(f"{side} is in check.")

    if state.sf:
        sf = state.sf
        pv = " ".join(sf.get("pv_san") or [])
        cp, mate = sf.get("eval_cp"), sf.get("mate")
        sign = 1 if state.human_color == "b" else -1      # White-POV -> bot POV
        my_cp = None if cp is None else cp * sign
        my_mate = None if mate is None else mate * sign
        lines += ["", "STOCKFISH (full-strength judge, depth "
                  f"{sf.get('depth', 12)}) — current position",
                  f"Eval {fmt_eval(cp, mate)}; best move {sf.get('best_san') or sf.get('best')}"
                  + (f", main line: {pv}" if pv else "") + "."]
        if my_mate is not None:
            pov_txt = f"mate in {abs(my_mate)} for {'you' if my_mate > 0 else 'the human'}"
        elif my_cp is not None:
            pov_txt = f"{my_cp/100:+.2f} (your point of view)"
        else:
            pov_txt = "unknown"
        lines.append(f"FROM YOUR OWN POINT OF VIEW the evaluation is {pov_txt} "
                     "(positive = good for you). Quote this when you say who stands better.")
        if sf.get("pv_san"):
            lines.append("BEST LINE EXPLAINED: " + explain.summarize_line(b, sf["pv_san"]) + ".")
        alt = [f"{ln['pv_san'][0]} ({fmt_eval(ln.get('cp'), ln.get('mate'))})"
               for ln in (sf.get("lines") or [])[1:3] if ln.get("pv_san")]
        if alt:
            lines.append("Other candidates: " + ", ".join(alt) + ".")

    if rec:
        hs, bs = stats.side_stats(rec, hc), stats.side_stats(rec, bc)
        lines += ["", "STATS (cp loss = how much a move cost versus Stockfish's best, in hundredths of a pawn; lower is better)"]
        for who, st in ((f"Human ({human})", hs), (f"You ({bot})", bs)):
            if st:
                lines.append(f"{who}: {st['moves']} moves, accuracy {st['accuracy']}%, "
                             f"average loss {st['avg_loss']} cp — "
                             f"{st['best']} best, {st['good']} good, {st['inaccuracy']} inaccuracies, "
                             f"{st['mistake']} mistakes, {st['blunder']} blunders.")
        worst = stats.worst_moments(rec)
        if worst:
            lines += ["", "KEY MOMENTS (the biggest mistakes of the game, in move order)"]
            for i, r in worst:
                mine = r["color"] == bc
                lines.append(f"- Move {i // 2 + 1}: {'YOU' if mine else 'the HUMAN'} played {r['san']}, "
                             f"a {r['label']} costing {'you' if mine else 'the human'} {r['loss']} cp"
                             + (f" (Stockfish preferred {r['best']})" if r.get("best") and r["best"] != r["san"] else "")
                             + f" — this helped {'the human' if mine else 'you'}.")
        last = rec[-1]
        who = "Human's" if last["color"] == hc else "Your"
        lines.append(f"{who} last move: {last['san']} — graded '{last['label']}' "
                     f"({last['loss']} cp loss).")
        lines += ["", _last_move_explained(state)]
        lines += ["", "MOVES (eval after each move, human's point of view; ?! inaccuracy, ? mistake, ?? blunder)",
                  stats.movelist(rec, state.human_color)]

    legal = [b.san(m) for m in b.legal_moves]
    # "yours" = the bot's. If the side to move is the human's colour, these are HIS moves.
    yours = "the human's" if (("w" if b.turn else "b") == state.human_color) else "yours"
    lines += ["", f"Legal moves for {side} (side to move, {yours}) ({len(legal)}): " + ", ".join(legal)]
    if b.is_checkmate():
        lines.append("The game is over by checkmate.")
    elif b.is_game_over():
        lines.append("The game is over.")
    return "\n".join(lines)


def _last_move_explained(state) -> str:
    rec = state.rec[-1]
    b = chess.Board()
    for u in state.history_uci[:-1]:
        b.push(chess.Move.from_uci(u))
    m = chess.Move.from_uci(state.history_uci[-1])
    who = "Human" if rec["color"] == state.human_color else "You"
    out = [f"LAST MOVE EXPLAINED ({who} played {rec['san']})"]
    notes = explain.move_motifs(b, m)
    out.append("What the move itself does: " + (", ".join(notes) if notes else "nothing tactical; a quiet move") + ".")
    best = (rec.get("lines") or [None])[0]
    if best and best.get("pv_san"):
        if rec.get("best") and rec["best"] != rec["san"]:
            out.append(f"Stockfish preferred {rec['best']} ({rec['loss']} cp better). Its line: "
                       + explain.summarize_line(b, best["pv_san"]) + ".")
        else:
            out.append("It was Stockfish's top choice. Expected line: "
                       + explain.summarize_line(b, best["pv_san"]) + ".")
    return "\n".join(out)


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
    le = s.last_engine
    if le:
        parts.append(f"I played {le['san']}"
                     + (f", though Stockfish preferred {le['best']}" if le.get("best") and le["best"] != le["san"] else "")
                     + f" (rated {le['label']}).")
    if s.sf:
        pv = " ".join((s.sf.get("pv_san") or [])[:4])
        parts.append(f"Stockfish says {fmt_eval(s.sf.get('eval_cp'), s.sf.get('mate'))}"
                     + (f", main line {pv}" if pv else "") + ".")
    mine = [r for r in s.rec if r["color"] == s.human_color]
    if mine and mine[-1]["label"] in ("mistake", "blunder", "inaccuracy"):
        parts.append(f"Your last move was an {mine[-1]['label']} ({mine[-1]['loss']} cp).")
    parts.append("(Chat narration is running in offline mode — the narrator did not answer.)")
    return " ".join(parts)


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "analyze_move",
        "description": "Ask Stockfish to evaluate a candidate move in the CURRENT position "
                       "(for 'what if' and 'why not X' questions). Returns the eval after it, its "
                       "cost versus the best move, the expected continuation and what the move does.",
        "parameters": {"type": "object", "properties": {"move": {
            "type": "string", "description": "legal move for the side to move, SAN (Nf3) or UCI (g1f3)"}},
            "required": ["move"]}}},
    {"type": "function", "function": {
        "name": "get_threats",
        "description": "What the opponent would do if the side to move passed (null-move search): "
                       "reveals the threat the side to move has to deal with.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _run_tool(tools: dict, name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args or "{}")
        fn = tools[name]
        return fn(**args) if args else fn()
    except Exception as e:
        return f"tool error: {type(e).__name__}: {e}"


def _round(client, payload: dict, headers: dict):
    """One streamed completion. Yields ('text', str) and finally ('calls', [..]) or ('error', msg)."""
    calls: dict[int, dict] = {}
    with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload, headers=headers) as r:
        if r.status_code != 200:
            yield ("error", f"{r.status_code}: {r.read().decode(errors='replace')[:300]}")
            return
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[6:].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"]
            except Exception:
                continue
            if delta.get("content"):               # reasoning, if any, stays hidden
                yield ("text", delta["content"])
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function") or {}
                c["name"] = f.get("name") or c["name"]
                c["arguments"] += f.get("arguments") or ""
    yield ("calls", [calls[i] for i in sorted(calls)])


def stream_reply(state, question: str, history: list[dict], tools: dict | None = None) -> Iterator[str]:
    """Stream the narrator's answer. `tools` maps tool names to callables that run Stockfish."""
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
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "http://localhost", "X-Title": "chess-engine"}
    got, use_tools = False, bool(tools)
    try:
        with httpx.Client(timeout=60) as client:
            for rnd in range(4):                   # up to 3 tool rounds, then a final answer
                payload = {"model": MODEL, "messages": msgs, "temperature": 0.6,
                           "max_tokens": 700, "stream": True,
                           # a reasoning model left alone can burn the whole budget on
                           # `reasoning` and emit content=null: turn thinking off
                           "reasoning": {"enabled": False}}
                if _provider_pref():
                    payload["provider"] = _provider_pref()
                if use_tools and rnd < 3:
                    payload["tools"] = TOOL_SPECS
                text, calls = [], []
                for kind, val in _round(client, payload, headers):
                    if kind == "text":
                        text.append(val)
                        got = True
                        yield val
                    elif kind == "calls":
                        calls = val
                    else:                          # error
                        if "tools" in payload and not got:
                            use_tools = False      # provider may not support tools: retry plain
                            break
                        yield f"[narrator error {val}] "
                        yield templated_reply(state, question)
                        return
                else:
                    if not calls:
                        break
                    msgs.append({"role": "assistant", "content": "".join(text) or None,
                                 "tool_calls": [{"id": c["id"], "type": "function",
                                                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                                for c in calls]})
                    for c in calls:
                        msgs.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": _run_tool(tools or {}, c["name"], c["arguments"])})
                    continue
                continue                           # only reached after a tools error: retry plain
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
