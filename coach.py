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
import mistakes
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


SYSTEM = """You are a chess coach commenting on a live game between the human you are \
talking to and a computer opponent. You speak in a professional, courteous and \
helpful manner: clear, constructive and encouraging, but always honest about \
mistakes. No slang, jokes, sarcasm, teasing or banter.

You will receive an ANALYSIS BRIEF with the game, its statistics and your own \
analysis of the positions. Treat that analysis as YOUR OWN judgement: say "I would \
prefer c3", "you are slightly better", "the strongest move here is Nf3". Do not \
volunteer or mention where the analysis comes from (no Stockfish, engine, judge, \
brief or tools in your answers). Only if the user directly asks about its source, \
answer truthfully and briefly: it is computer analysis by the Stockfish chess \
engine. Hard rules:
- Only make chess claims supported by the brief or the tool results. Never invent \
variations, tactics, threats, or evaluations that are not in them. If the brief \
does not contain what the user asks, say you don't have it.
- If you quote a line, quote the principal variation you were given.
- WORDS, NOT NUMBERS. Never quote centipawns ("cp"), evaluation numbers or winning \
percentages, and do not use the words "centipawn" or "evaluation score". Describe \
who stands better in words (about equal, slightly better, clearly better, winning; \
slightly worse, clearly worse, lost) and sizes in material ("a pawn", "a piece"). \
You may quote the accuracy percentage and counts from STATS.
- SPEAK ONLY WHEN IT MATTERS. Comment on a move of the human's only if it is rated \
inaccuracy, mistake or blunder. A move rated best, good or fine gets at most one \
short clause ("a good move", "that's fine"): never criticise it, never say another \
move was better, and never suggest an alternative unless the user asks. Spend the \
rest of the answer on the position and a practical plan.
- WHEN A MOVE WAS WEAK, use the "WHY ... WAS WEAK" section: (1) name the type of \
mistake from its MISTAKE TYPE line, in plain words; (2) explain the concrete \
consequence (what the opponent can now do with which move, what was left \
unprotected or lost, or that the cost is positional); (3) say what the better move \
does instead; (4) end with the HABIT TO TEACH in your own words, in one sentence. \
If the type is "positional", say plainly that no tactic is involved and do not \
invent a concept or a habit. Never give only the rating. For an earlier move, call \
explain_past_move(move_number).
- You are the coach, not a player. Address the human as "you"; refer to the other \
side as "your opponent" or by colour ("Black's Nd7"). Never say "I played" about a \
game move. If asked "why did you play X?" about the opponent's move, say it was \
your opponent's move, then explain what the move does.
- Never state or guess your opponent's intentions or plans. Describe its moves only \
from the "THE OPPONENT'S LAST MOVE" line (piece, colour, squares, what it attacks or \
defends, whether those pieces are defended); do not add effects that are not listed \
there, and do not say a piece is "supported" or "pressuring" something unless the line \
says so. Mention the opponent's last move in a clause at most, unless asked. If it \
was a mistake or blunder, you may point out how the human can take advantage.
- The EVALUATION line and every standing in the brief are from the human's point of \
view; lead with the human's side when you say who stands better.
- Length: 1-3 sentences for a quick remark or a one-line reply. When the user asks \
you to explain a position, plan, or tactic, or a weak move needs its explanation, \
give a fuller answer of about 120-200 words in at most two short paragraphs. No \
headers, no bullet lists, no tables. You may use *italic* and **bold** sparingly.
- In an explanation, name the concrete moves from the brief (the main line, the \
candidate moves, the last move) and say what each one is for: what it attacks, \
defends, or prepares. One well-explained idea beats a vague list.
- FOCUS ON THE HUMAN: their moves, ideas, mistakes and chances, and what they could \
consider next. Give practical, actionable guidance rather than commentary for its \
own sake. When they play well, say so briefly and specifically.
- Reply in English unless the user's latest message is written in another language; \
then reply in that language.
- If the user asks for a hint or the best move, give it from the brief.
- For questions about the whole game (accuracy, mistakes, turning points, the \
opening, who was better when), answer from the STATS, KEY MOMENTS, OPENING and \
MOVES sections.
- To explain WHY (why a move is good or bad, why not another move, what the \
opponent threatens), use the sections of the brief, and call the tools for anything \
else: analyze_move(move) evaluates any candidate move in the current position (use \
it for "what if I play X?" and "why not X?"), and get_threats() shows what the \
opponent would do if the side to move passed. The analysis gives lines, not \
reasons: explain only what the tool results and sections show (captures, checks, \
material won or lost, patterns named there). If the real reason is positional and \
no pattern is listed, say that I simply rate it higher rather than inventing a concept.
"""


def _standing(cp: int | None, mate: int | None, pov: int) -> str:
    """How the human stands, in plain words (pov=+1 if the human is White)."""
    if mate is not None:
        return f"mate in {abs(mate)} for {'the human' if mate * pov > 0 else 'the opponent'}"
    v = stats.verdict(cp, pov)
    return v if v in ("about equal", "unclear") else f"{v} for the human"


def _before_cp(rec: list, i: int):
    """White-POV cp of the position before ply i (the previous ply's result; None for the first ply)."""
    return rec[i - 1].get("cp") if i > 0 else None


def build_brief(state) -> str:
    """state is server.Game; kept duck-typed to avoid a circular import."""
    b: chess.Board = state.board
    side = "White" if b.turn else "Black"
    bot_white = state.human_color == "b"
    bot = "White" if bot_white else "Black"
    human = "Black" if bot_white else "White"
    hc, bc = state.human_color, ("w" if bot_white else "b")
    pov = 1 if hc == "w" else -1                     # White-POV -> human-POV multiplier
    rec = state.rec
    lines = [
        "GAME",
        f"The human plays {human}. The opponent (a computer playing {bot}) plays at about "
        f"Elo {state.elo}. Move {b.fullmove_number}, {side} to move, phase: {stats.phase(b)}.",
        f"OPENING: {stats.opening(state.history_uci) or 'no named opening (out of book or too few moves)'}",
        f"FEN: {b.fen()}",
        f"Material (the human's side minus the opponent's, in pawns): {_material(b, hc):+.1f}",
    ]
    cap = stats.captures(state.history_uci)
    if cap["w"] or cap["b"]:
        lines.append(f"Captured so far — by the opponent ({bot}): {', '.join(cap[bc]) or 'nothing'}; "
                     f"by the human ({human}): {', '.join(cap[hc]) or 'nothing'}.")
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
        lines += ["", "MY ANALYSIS of the current position",
                  f"Best move: {sf.get('best_san') or sf.get('best')}" + (f"; main line: {pv}" if pv else "") + ".",
                  f"EVALUATION: {_standing(cp, mate, pov)}. When you say who stands better, lead with the human's side."]
        if sf.get("pv_san"):
            lines.append("BEST LINE EXPLAINED: " + explain.summarize_line(b, sf["pv_san"]) + ".")
        alt = [f"{ln['pv_san'][0]} ({_standing(ln.get('cp'), ln.get('mate'), pov)})"
               for ln in (sf.get("lines") or [])[1:3] if ln.get("pv_san")]
        if alt:
            lines.append("Other candidates: " + ", ".join(alt) + ".")

    if rec:
        hs, bs = stats.side_stats(rec, hc), stats.side_stats(rec, bc)
        lines += ["", "STATS (accuracy is out of 100; moves are rated best, good, fine, inaccuracy, mistake, blunder)"]
        for who, st in ((f"Human ({human})", hs), (f"Opponent ({bot})", bs)):
            if st:
                lines.append(f"{who}: {st['moves']} moves, accuracy {st['accuracy']}% — "
                             f"{st['best']} best, {st['good']} good, {st['fine']} fine, {st['inaccuracy']} inaccuracies, "
                             f"{st['mistake']} mistakes, {st['blunder']} blunders.")
        worst = stats.worst_moments(rec)
        if worst:
            lines += ["", "KEY MOMENTS (the biggest mistakes of the game, in move order)"]
            for i, r in worst:
                opp = r["color"] == bc
                prev = _before_cp(rec, i)
                lines.append(f"- Move {i // 2 + 1}: {'the OPPONENT' if opp else 'the HUMAN'} played {r['san']}, "
                             f"a {r['label']}"
                             + (f" (for the human the position {stats.swing(prev, r['cp'], pov)})" if prev is not None else "")
                             + (f"; the better move was {r['best']}" if r.get("best") and r["best"] != r["san"] else "")
                             + f" — this helped {'the human' if opp else 'the opponent'}.")
        lines += ["", _human_move_explained(state)]
        opp_move = _opponent_move_described(state)
        if opp_move:
            lines += ["", opp_move]
        lines += ["", "MOVES (the bracket is how the position stands for the human after that move, shown when it "
                      "changes; ?! inaccuracy, ? mistake, ?? blunder; 'better:' is the move I would have preferred)",
                  stats.movelist(rec, state.human_color)]

    legal = [b.san(m) for m in b.legal_moves]
    mover = "the human's" if (("w" if b.turn else "b") == state.human_color) else "the opponent's"
    lines += ["", f"Legal moves for {side} (side to move, {mover}) ({len(legal)}): " + ", ".join(legal)]
    if b.is_checkmate():
        lines.append("The game is over by checkmate.")
    elif b.is_game_over():
        lines.append("The game is over.")
    return "\n".join(lines)


RATING_NOTE = {
    "best": "the strongest move available. Praise it in a clause.",
    "good": "a good move (one of several fine moves). Praise it in a clause; do not suggest alternatives unless asked.",
    "fine": "fine: not the very best, but it costs nothing real. Say it is fine and move on; do not criticise it or "
            "suggest alternatives unless asked.",
}


def _human_move_explained(state) -> str:
    """The human's latest move in detail (the coach is meant to focus on it)."""
    rec, hc = state.rec, state.human_color
    idx = [i for i, r in enumerate(rec) if r["color"] == hc]
    if not idx:
        return "THE HUMAN'S LAST MOVE: the human has not moved yet."
    i = idx[-1]
    r = rec[i]
    b = chess.Board()
    for u in state.history_uci[:i]:
        b.push(chess.Move.from_uci(u))
    m = chess.Move.from_uci(state.history_uci[i])
    pov = 1 if hc == "w" else -1
    out = [f"THE HUMAN'S LAST MOVE EXPLAINED (move {i // 2 + 1}: the human played {r['san']})"]
    prev = _before_cp(rec, i)
    sw = f" For the human the position {stats.swing(prev, r['cp'], pov)}." if prev is not None and r.get("cp") is not None else ""
    if r["label"] in RATING_NOTE:
        out.append(f"Rating: {r['label']} — {RATING_NOTE[r['label']]}{sw}")
    else:
        out.append(f"Rating: {r['label']} — worth discussing.{sw}")
    out.append("What the move itself does: " + explain.describe_move(b, m) + ".")
    best = (r.get("lines") or [None])[0]
    if r["label"] == "best" and best and best.get("pv_san"):
        out.append("Expected continuation: " + explain.summarize_line(b, best["pv_san"]) + ".")
    why = _why_weak(state, i)
    if why:
        out += ["", why]
    return "\n".join(out)


WEAK = ("inaccuracy", "mistake", "blunder")


def _why_weak(state, i: int) -> str | None:
    """Concrete reasons ply i (a rated-weak move) was worse than the best move, from computed facts only."""
    rec, hc = state.rec, state.human_color
    if i < 0 or i >= len(rec) or rec[i].get("label") not in WEAK:
        return None
    r = rec[i]
    who = "the human" if r["color"] == hc else "the opponent"
    pov = 1 if hc == "w" else -1
    b = chess.Board()
    for u in state.history_uci[:i]:
        b.push(chess.Move.from_uci(u))
    played = chess.Move.from_uci(state.history_uci[i])
    after = b.copy(stack=False)
    after.push(played)
    best_before = (r.get("lines") or [None])[0]
    nxt = rec[i + 1].get("lines") if i + 1 < len(rec) else (state.sf or {}).get("lines")
    refut = (nxt or [None])[0]
    prev = _before_cp(rec, i)
    sw = f": the position {stats.swing(prev, r['cp'], pov)} for the human" if prev is not None and r.get("cp") is not None else ""
    out = [f"WHY {who.upper()}'S MOVE {r['san']} (move {i // 2 + 1}) WAS WEAK — rated '{r['label']}'{sw}. "
           "Explain this in your answer."]
    types = mistakes.classify(b, played, r.get("best"), (refut or {}).get("pv_san") or [],
                              (best_before or {}).get("pv_san") or [])
    for t in types:
        out.append(f"MISTAKE TYPE: {t['tag']} — {t['detail']}.")
    lesson = next((t["lesson"] for t in types if t["lesson"]), None)
    if lesson:
        out.append("HABIT TO TEACH (say it in your own words, in one sentence): " + lesson)
    if types and types[0]["tag"] == "positional":
        out.append("Cause: positional. Say so honestly; give a concrete reason only if one is listed below, "
                   "otherwise say the better move keeps a more favourable position.")
    if refut and refut.get("pv_san"):
        try:
            first = after.parse_san(refut["pv_san"][0])
            out.append(("The opponent's best reply: " if r["color"] == hc else "The human's best reply: ")
                       + explain.describe_move(after, first) + ".")
        except ValueError:
            pass
        out.append("Line after the weak move: " + explain.summarize_line(after, refut["pv_san"]) + ".")
    notes = explain.move_motifs(b, played)
    bad = [n for n in notes if "en prise" in n or "leaves" in n]
    if bad:
        out.append(f"What the move itself gave away: {', '.join(bad)}.")
    if r.get("best") and r["best"] != r["san"] and best_before and best_before.get("pv_san"):
        try:
            bm = b.parse_san(r["best"])
            out.append("The better move: " + explain.describe_move(b, bm) + ".")
        except ValueError:
            pass
        out.append("Its line: " + explain.summarize_line(b, best_before["pv_san"]) + ".")
    return "\n".join(out)


def _opponent_move_described(state) -> str | None:
    """The opponent's latest move as explicit facts, so the coach never has to guess what it does."""
    rec = state.rec
    idx = [i for i, r in enumerate(rec) if r["color"] != state.human_color]
    if not idx:
        return None
    i = idx[-1]
    b = chess.Board()
    for u in state.history_uci[:i]:
        b.push(chess.Move.from_uci(u))
    r = rec[i]
    d = explain.describe_move(b, chess.Move.from_uci(state.history_uci[i]))
    gift = (f", a {r['label']} by your opponent that the human may be able to exploit"
            if r["label"] in ("mistake", "blunder") else "")
    return (f"THE OPPONENT'S LAST MOVE (move {i // 2 + 1}: {r['san']}{gift})\n"
            f"{d}.\nDescribe it only from this line; its intentions are unknown. Mention it only in passing "
            "unless asked.")


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
    mine = [r for r in s.rec if r["color"] == s.human_color]
    if mine:
        r = mine[-1]
        if r["label"] in WEAK:
            parts.append(f"Your last move {r['san']} was an {r['label']}"
                         + (f"; I would have preferred {r['best']}." if r.get("best") and r["best"] != r["san"] else "."))
        else:
            parts.append(f"Your last move {r['san']} was {'the best move' if r['label'] == 'best' else 'fine'}.")
    if s.sf:
        pov = 1 if s.human_color == "w" else -1
        pv = " ".join((s.sf.get("pv_san") or [])[:4])
        parts.append(f"The position is {_standing(s.sf.get('eval_cp'), s.sf.get('mate'), pov)}"
                     + (f", main line {pv}" if pv else "") + ".")
    parts.append("(Chat narration is running in offline mode — the narrator did not answer.)")
    return " ".join(parts)


# reasoning is off by default (a reasoning model can spend the whole budget thinking and return no
# answer); experiments override these two module-level settings
REASONING = {"enabled": False}
MAX_TOKENS = 700

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "analyze_move",
        "description": "Evaluate a candidate move in the CURRENT position "
                       "(for 'what if' and 'why not X' questions). Returns the eval after it, its "
                       "cost versus the best move, the expected continuation and what the move does.",
        "parameters": {"type": "object", "properties": {"move": {
            "type": "string", "description": "legal move for the side to move, SAN (Nf3) or UCI (g1f3)"}},
            "required": ["move"]}}},
    {"type": "function", "function": {
        "name": "explain_past_move",
        "description": "Why a move already played in this game was weak (what it allowed or lost, "
                       "the opponent's best reply, the better move). Use it when asked why an EARLIER "
                       "inaccuracy, mistake or blunder was bad.",
        "parameters": {"type": "object", "properties": {"move_number": {
            "type": "integer", "description": "full-move number as shown in the move list (1, 2, 3...)"},
            "side": {"type": "string", "enum": ["human", "opponent"],
                     "description": "whose move; default human"}},
            "required": ["move_number"]}}},
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
                           "max_tokens": MAX_TOKENS, "stream": True,
                           # a reasoning model left alone can burn the whole budget on
                           # `reasoning` and emit content=null: turn thinking off
                           "reasoning": REASONING}
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
