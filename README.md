# ChessLFM

A web chess opponent that runs entirely on a small CPU box, with a chat you can use to talk about the game.

Three layers, each doing one job:

| layer | does | with |
|---|---|---|
| **plays** | picks every move | [LFM2.5-230M-Chess](https://huggingface.co/mlabonne/LFM2.5-230M-Chess) (Q8_0 GGUF) via llama.cpp, CPU only |
| **judges** | eval, best line, grades your moves | Stockfish (never chooses the bot's move) |
| **talks** | table talk and explanations | any OpenAI-compatible LLM through OpenRouter, handed only Stockfish-grounded facts |

The 230M model cannot chat (it only emits chess tokens), so the narrator is a separate LLM and is told to cite only the analysis brief it is given.

## Why let a 230M model pick the moves?

Stockfish would be a stronger and faster player, and it can be capped to a human rating (`UCI_Elo`). Using the small model instead is a deliberate trade:

- **Mistakes that look human.** The model was trained on positions and plays plausible but imperfect moves, such as an odd developing move or a missed tactic. That can feel more like playing a person than a reduced-strength Stockfish.
- **A different judge from the player.** Because the model chooses and Stockfish only evaluates, the chat can say "I played this, but Stockfish prefers that". If Stockfish chose the moves too, it would always agree with itself.

### How the model differs from Stockfish

The model is a small language model, not a chess program: 14 layers (8 short-convolution, 6 attention), hidden width 1024, about 230M parameters. Its vocabulary was extended with 2,106 chess tokens (one per from-to move, square contents, position details, 64 win-probability buckets). Each position is flattened into an 80-token prompt.

Each move takes two forward passes:

1. **Value pass.** The model predicts one of the 64 eval buckets, a learned guess of the win probability.
2. **Policy pass.** After `<|bestmove|>` it predicts a move token. `engine.py` masks out illegal moves and takes a softmax over the rest. Normal mode plays the top move, Casual samples among the top 5, and Strong looks two plies ahead using the model's own win-probability readings.

| | LFM model | Stockfish |
|---|---|---|
| Core idea | Pattern recognition: positions like this were followed by moves like that | Calculation: look ahead and compare outcomes |
| Looks ahead? | No (one forward pass), apart from Strong mode's shallow look | Yes, typically 20+ plies with pruning |
| Knows the rules? | No. Legality comes from the move filter in `engine.py` | Yes, exactly |
| Evaluation | A learned guess in 64 buckets | A neural network (NNUE) backed by search |
| Errors | Can miss a tactic two moves deep because it never calculates it | Rarely blunders at normal depth |
| Cost | Fixed: one prefill (about 2.4 s on a 2-core CPU) | Scales with the time you give it |

It plays natural-looking moves from experience without verifying them, while Stockfish calculates everything and has little intuition apart from its evaluation. That is also why Stockfish judges every move here: the model cannot explain or check its own choice.

The cost is speed (about 2.5 s per move instead of milliseconds). The strength labels in the UI (Casual about 1200, Normal about 1500) are rough guesses, not measured ratings.

Features: chessground board (lichess's), Random/White/Black, three strengths, hints, take-back, eval bar, per-game URLs (`/g/<id>`) that survive refresh, saved-games list (games older than 24h are grouped, deleted after 15 days idle), download a game as a `.json` file, delete it, or load a file back in.

## Run

```bash
sudo apt install stockfish                       # expects /usr/games/stockfish
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p models && curl -L -o models/LFM2.5-230M-Chess.Q8_0.gguf \
  https://huggingface.co/mradermacher/LFM2.5-230M-Chess-GGUF/resolve/main/LFM2.5-230M-Chess.Q8_0.gguf
cp .env.example .env                             # add your OpenRouter key (optional)
.venv/bin/python server.py                       # http://localhost:8975
```

Without `OPENROUTER_API_KEY` the app still plays; chat falls back to a short templated reply.

`.env` / environment variables: `OPENROUTER_API_KEY`, `CHESSLFM_MODEL`, `CHESSLFM_BASE_URL`, `CHESSLFM_PROVIDER` (optional OpenRouter provider pin).

Speed: about 2.5 s per move on a 2-core CPU (the model reads an 80-token position prompt; Q4 is barely faster). The server listens on `0.0.0.0:8975` with no authentication, so keep it on a trusted network.

## Layout

```
engine.py     the model's 80-token position protocol, legal-move masking, KV reuse
stockfish.py  persistent UCI wrapper, MultiPV, move grading
coach.py      analysis brief + narrator (streaming) + offline fallback
server.py     FastAPI app and game storage (games/<id>.json)
static/       single-file UI + vendored chessground
meta/spec_tokens.json   token table from the model's reference Space
```

## Licenses

- `static/vendor/chessground*` and the cburnett piece set are from [lichess chessground](https://github.com/lichess-org/chessground) (GPL-3.0) — redistribution of this repo must respect that.
- The model weights are not included; see the LFM2.5-230M-Chess model card for its license.
