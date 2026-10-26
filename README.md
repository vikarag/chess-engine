# chess-engine

A web chess opponent you can talk to. It plays with Stockfish limited to the Elo you choose (1500–2800), a second full-strength Stockfish judges every move, and an LLM commentator explains the game using only facts computed from those two.

Earlier versions used the small LFM2.5-230M-Chess language model as the opponent; it was too weak, so it was replaced (see the git history).

## How it works

| layer | does | with |
|---|---|---|
| **plays** | picks the opponent's moves | Stockfish with `UCI_LimitStrength` + `UCI_Elo`, search capped at depth 12 |
| **judges** | evals, best lines, grades every move (yours and the opponent's), hints | a separate full-strength Stockfish (depth 12, 3 lines) |
| **talks** | coaching comments, whole-game stats, explanations of *why* | any OpenAI-compatible LLM via OpenRouter, handed only computed facts |

The judge never plays and the player never analyses, so every grade comes from a stronger, independent analysis than the one that chose the opponent's move.

**How moves are graded.** A move is rated by how many points of *winning chances* it gives away (lichess's centipawn-to-win-probability curve), not by raw centipawns, so the same slip counts for less when the game is already decided: best (<1), good (<4), fine (<8), inaccuracy (<15), mistake (<25), blunder. A move that lets the opponent win a piece is rated at least a mistake even in a lopsided position. Before an analysis is trusted, its top move is played and searched once more; if it collapses (a shallow search can miss a sacrifice), the position is searched deeper, so the move the coach recommends and the grade that move later receives always agree. "Fine" exists so the commentator stays quiet about small differences that are within the judge's own depth-to-depth wobble.

**The commentator is a separate coach, not the opponent.** It speaks professionally and supportively, focuses on *your* moves and position, refers to the opponent in the third person ("Black's Nd7"), and never claims to know the opponent's intentions. It presents the analysis as its own judgement ("I would prefer c3") and does not volunteer that Stockfish produced it, but it answers truthfully if you ask. It talks in words, not numbers ("slightly worse", "clearly better", "a piece down"); centipawns only appear in the small grade badge. It comments only on moves rated inaccuracy or worse, and for those it names the type of mistake (walked into mate, hanging piece, undefended pawn, allowed a tactic, missed a chance, king safety, development, or plainly positional), explains the concrete consequence and the better move, and ends with the habit that prevents it.

**What the commentator knows.** Each ply gets a record (eval, best move, winning chances lost, label). From those, code computes the opening name, game phase, accuracy per side, the biggest mistakes, captures, loose pieces and the annotated move list, and passes them to the LLM with strict "cite only the brief" rules. The last move of each side is also described explicitly in code ("Black's knight moves from b8 to c6: develops the knight; attacks the white pawn on d4 (defended) and the white pawn on e5 (defended)"): colour, piece, squares, captures, checks, what it attacks (defended or undefended), what it defends, development, castling and centre pawns. The LLM is told to describe moves only from these lines, so it does not have to guess what a move does.

**Why a move was played.** Stockfish gives numbers and lines, not reasons, so "why" is built from them: what the expected line wins or loses, tactical patterns the move creates (fork, pin, discovered attack, piece left en prise, piece defended or attacked), the gap to other candidate moves, and what the opponent would do if the side to move passed. When one of your moves is rated an inaccuracy, mistake or blunder, the commentator is handed a "why it was weak" section (the opponent's best reply described move by move, whether material is lost or the cost is only positional, what the move gave away, and what the better move does instead) and is required to explain it, not just quote the rating. If the real reason is positional and no pattern applies, it is told to say so rather than invent one.

**The commentator's tools.** Besides the brief, the LLM can call three tools that run the judge (Stockfish 19):

| Tool | Arguments | Returns |
|---|---|---|
| `analyze_move` | `move` (SAN `Nf3` or UCI `g1f3`) | For "what if I play X?" / "why not X?": the position after the move in the current position, how much it costs against the best move, the expected continuation and what the move does (forks, pins, threats). |
| `explain_past_move` | `move_number` (full-move number), `side` (`human` default, or `opponent`) | Why an earlier inaccuracy, mistake or blunder was weak: what it allowed or lost, the opponent's best reply, the better move, the mistake type and the habit that prevents it. Says so if the move wasn't weak. |
| `get_threats` | none | A null-move search: what the opponent would play if the side to move passed, i.e. the threat to deal with. |

It gets up to three tool rounds per answer, then must reply. The what-if and threat tools search at depth 16 on a single thread, so asking the same question twice gives the same answer. A failing tool returns its error to the model instead of breaking the reply, and if the provider rejects tool calls the request is retried without them. Every answer that used a tool carries a small "Engine-checked: …" tag in the chat. The tools only read the current game: they cannot load other positions or make or take back moves.

**Elo and depth.** In limited-strength mode Stockfish picks its move at a fixed depth of roughly the skill level + 1 (about 7 plies at Elo 2500 and 11 at Elo 2800), then adds randomness. The depth cap of 12 sits just above what the top of the slider needs, so it never weakens the engine. The Elo numbers are on Stockfish's engine scale; they are not verified against human ratings.

Other features: a phone-friendly single-column layout (on narrow screens the order is board, chat, moves, new game, saved games), chessground board (the one lichess uses), Play Random/White/Black, a clickable move list (click any move to see the board as it was after it), hints, take-back (the commentator is told about it and ignores its earlier remarks on the taken-back moves), eval bar, captured-pieces strips above and below the board (what each side has taken, with the material lead), per-game URLs (`/g/<id>`) that survive refresh, a saved-games list (games older than 24h are grouped, deleted after 15 days idle), and downloading a game as a `.json` file, deleting it, or loading a file back in.

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
mistakes.py   names the type of a weak move from board logic, with the habit that prevents it
coach.py      the commentator's brief, streaming, tool calls, offline fallback
trace.py      prints the brief the commentator receives for the newest game
static/       single-file UI + vendored chessground
data/openings the lichess opening table (CC0)
```

`python stats.py`, `python explain.py` and `python mistakes.py` run small self-checks (no engine needed).

## Licenses

- `static/vendor/chessground*` and the cburnett piece set are from [lichess chessground](https://github.com/lichess-org/chessground) (GPL-3.0) — redistribution of this repo must respect that.
- Stockfish is GPL-3.0; it is not bundled here.
- `data/openings` is from [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) (CC0).
