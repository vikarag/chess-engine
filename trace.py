"""Show the three layers separately for the current position.

Run:  .venv/bin/python trace.py
"""
import os

import chess
import coach
import server

gid = None
if os.path.exists(os.path.join(server.GAMES, ".trace_game")):
    gid = open(os.path.join(server.GAMES, ".trace_game")).read().strip()
g = server.get_game(gid) if gid else None
if g is None:                       # newest saved game, else start one
    names = [n[:-5] for n in os.listdir(server.GAMES)
             if n.endswith(".json") and server.GAME_ID_RE.fullmatch(n[:-5])]
    if names:
        names.sort(key=lambda n: os.path.getmtime(server._path(n)), reverse=True)
        g = server.get_game(names[0])
    if g is None:
        g = server.create_game("w", "normal")
        open(os.path.join(server.GAMES, ".trace_game"), "w").write(g.id)
        print(f"[trace] started new game {g.id}")
server.state(g)             # first call lazily fills g.sf (Stockfish) after a restart
if not g.board.is_game_over(claim_draw=True) \
        and ("w" if g.board.turn else "b") != g.human_color:
    server._engine_play(g)  # let the model actually move, so the trace has its policy

b = g.board

print("=" * 72)
print("POSITION")
print("=" * 72)
print("FEN      :", b.fen())
print("to move  :", "White" if b.turn else "Black")
print("human is :", g.human_color, "| bot is:", "w" if g.human_color == "b" else "b")
print("moves    :", " ".join(g.history_san[-16:]) or "(none)")

print()
print("=" * 72)
print("LAYER 1 — LFM2.5-230M-CHESS (the player): its own policy")
print("=" * 72)
e = g.last_engine
if e:
    print(f"it played {e['san']} ({e['uci']}), its own probability {e['prob']*100:.1f}%")
    print(f"value head (win prob for side to move when it chose): {e['winprob']:.2f}")
    print(f"forward passes {e['nodes']}, {e['think_ms']} ms, mode {e['mode']}")
    for t in e["top"]:
        print(f"   {t['san']:>6}  {t['prob']*100:5.1f}%  {'#'*int(t['prob']*40)}")
else:
    print("(no engine move yet)")

print()
print("=" * 72)
print("LAYER 2 — STOCKFISH 16 (the judge)")
print("=" * 72)
print("eval (White POV):", coach.fmt_eval((g.sf or {}).get("eval_cp"), (g.sf or {}).get("mate")))
print("best move       :", (g.sf or {}).get("best_san"))
print("depth           :", (g.sf or {}).get("depth"))
print("main line       :", " ".join((g.sf or {}).get("pv_san") or []))
for i, ln in enumerate((g.sf or {}).get("lines") or []):
    print(f"  line {i+1}: {coach.fmt_eval(ln['cp'], ln['mate'])}  {' '.join(ln['pv_san'][:6])}")
if g.last_grade:
    print("your last move graded:", g.last_grade)

print()
print("=" * 72)
print("LAYER 3 — WHAT THE NARRATOR ACTUALLY RECEIVES")
print("=" * 72)
brief = coach.build_brief(g)
print(brief)

print()
print("=" * 72)
print("NARRATOR ANSWERS (it can only rephrase the block above)")
print("=" * 72)
for q in ["How do you stand, and what's the plan?",
          "What is my best move here?",
          "If I play h5 next, what is your tactical refutation?"]:
    print(f"\n--- Q: {q}")
    print("".join(coach.stream_reply(g, q, [])).strip())
    print()
