# ChessLFM

A web chess opponent that runs entirely on a small CPU box, with a chat you can use to talk about the game.

Three layers, each doing one job:

| layer | does | with |
|---|---|---|
| **plays** | picks every move | [LFM2.5-230M-Chess](https://huggingface.co/mlabonne/LFM2.5-230M-Chess) (Q8_0 GGUF) via llama.cpp, CPU only |
| **judges** | eval, best line, grades your moves | Stockfish (never chooses the bot's move) |
| **talks** | table talk and explanations | any OpenAI-compatible LLM through OpenRouter, handed only Stockfish-grounded facts |

The 230M model cannot chat (it only emits chess tokens), so the narrator is a separate LLM and is told to cite only the analysis brief it is given.

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
