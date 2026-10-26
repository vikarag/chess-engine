"""Show what the narrator is given for the newest saved game.

Run:  .venv/bin/python trace.py
"""
import os

import coach
import server

names = [n[:-5] for n in os.listdir(server.GAMES)
         if n.endswith(".json") and server.GAME_ID_RE.fullmatch(n[:-5])]
if not names:
    raise SystemExit("no saved games")
names.sort(key=lambda n: os.path.getmtime(server._path(n)), reverse=True)
g = server.get_game(names[0])
server._ensure_rec(g)
print(f"game {g.id}: human {g.human_color}, opponent Elo {g.elo}, {len(g.history_uci)} plies\n")
print(coach.build_brief(g))
