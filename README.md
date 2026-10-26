# chess-engine

A web chess opponent you can talk to. It plays with Stockfish limited to the Elo you choose (1500–2500), a second full-strength Stockfish judges every move, and an LLM commentator explains the game using only facts computed from those two.

Earlier versions used the small LFM2.5-230M-Chess language model as the opponent; it was too weak, so it was replaced (see the git history).

## How it works

| layer | does | with |
|---|---|---|
| **plays** | picks the opponent's moves | Stockfish with `UCI_LimitStrength` + `UCI_Elo`, search capped at depth 7 |
| **judges** | evals, best lines, grades every move (yours and the opponent's), hints | a separate full-strength Stockfish (depth 12, 3 lines) |
| **talks** | table talk, whole-game stats, explanations of *why* | any OpenAI-compatible LLM via OpenRouter, handed only computed facts |

The judge never plays and the player never analyses, so the commentary can say "I played this, but Stockfish prefers that" and be right.

**What the commentator knows.** Each ply gets a record (eval, best move, centipawn loss, label). From those, code computes the opening name, game phase, accuracy and average loss per side, the biggest mistakes, captures, loose pieces and the annotated move list, and passes them to the LLM with strict "cite only the brief" rules.

**Why a move was played.** Stockfish gives numbers and lines, not reasons, so "why" is built from them: what the expected line wins or loses, tactical patterns the move creates (fork, pin, discovered attack, piece left en prise), the gap to other candidate moves, and what the opponent would do if the side to move passed. The LLM can also call two tools that run the judge: `analyze_move` ("what if I play X?", "why not X?") and `get_threats`. If the real reason is positional and no pattern applies, it is told to say so rather than invent one.

**Elo and depth.** In limited-strength mode Stockfish picks its move at a fixed depth of roughly the skill level + 1 (about 7 plies at Elo 2500), then adds randomness, which is why a depth cap of 7 matches the top of the slider. The Elo numbers are on Stockfish's engine scale; they are not verified against human ratings.

Other features: chessground board (the one lichess uses), Play Random/White/Black, hints, take-back, eval bar, per-game URLs (`/g/<id>`) that survive refresh, a saved-games list (games older than 24h are grouped, deleted after 15 days idle), and downloading a game as a `.json` file, deleting it, or loading a file back in.

## Run

```bash
sudo apt install stockfish                       # opponent (and judge, unless bin/stockfish-19 exists): /usr/games/stockfish
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                             # add your OpenRouter key (optional)
.venv/bin/python server.py                       # http://localhost:8975
```

Optional, recommended: use a newer Stockfish for the judge. Download the official Linux build from the [Stockfish releases](https://github.com/official-stockfish/Stockfish/releases) (`stockfish-linux-x86-64-universal`), check its sha256 against the release page, and save it as `bin/stockfish-19` (executable). It is picked up automatically and git-ignored.

Without `OPENROUTER_API_KEY` the app still plays and grades; chat falls back to a short templated reply.

Settings (environment variables or `.env`): `OPENROUTER_API_KEY`, `CHESS_ENGINE_MODEL`, `CHESS_ENGINE_BASE_URL`, `CHESS_ENGINE_PROVIDER` (optional OpenRouter provider pin).

The server listens on `0.0.0.0:8975` with no authentication, so keep it on a trusted network (anyone who can reach it can play and spend your API credits).

## Layout

```
server.py     FastAPI app, game storage (games/<id>.json), per-move judge records
stockfish.py  UCI wrapper: analyse() for the judge, play() for the limited-strength player
stats.py      whole-game facts: opening, phase, accuracy, key mistakes, annotated move list
explain.py    "why" facts: line summaries, tactical patterns, threat check, what-if analysis
coach.py      the commentator's brief, streaming, tool calls, offline fallback
trace.py      prints the brief the commentator receives for the newest game
static/       single-file UI + vendored chessground
data/openings the lichess opening table (CC0)
```

`python stats.py` and `python explain.py` run small self-checks (no engine needed).

## Licenses

- `static/vendor/chessground*` and the cburnett piece set are from [lichess chessground](https://github.com/lichess-org/chessground) (GPL-3.0) — redistribution of this repo must respect that.
- Stockfish is GPL-3.0; it is not bundled here.
- `data/openings` is from [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) (CC0).
