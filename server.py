"""FastAPI app: play LFM2.5-230M-Chess in the browser and talk to it.

Every game has an id (10 hex chars), is saved as games/<id>.json and is
addressed by URL: /g/<id>. A browser refresh reloads that game from the
server. Games with no activity for 15 days are deleted. A game can be
downloaded as its JSON file, deleted, or imported from such a file.
"""
import json
import os
import random
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

import chess
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import coach
from engine import ChessLFM, repetition_count
from stockfish import Stockfish, grade

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "LFM2.5-230M-Chess.Q8_0.gguf")
GAMES = os.path.join(HERE, "games")
INDEX = os.path.join(HERE, "static", "index.html")
RETENTION_S = 15 * 24 * 3600
GAME_ID_RE = re.compile(r"[0-9a-f]{10}")
os.makedirs(GAMES, exist_ok=True)

_lock = threading.RLock()     # one engine, one Stockfish: compute is serialised
print("[startup] loading engine…", flush=True)
ENGINE = ChessLFM(MODEL, n_threads=2)
SF = Stockfish(threads=2, hash_mb=128)
print("[startup] ready", flush=True)

MODES = {"casual": "Casual (~1200, varied)", "normal": "Normal (policy, ~1500)",
         "strong": "Strong (1-ply search, slow)"}


@dataclass
class Game:
    id: str
    board: chess.Board = field(default_factory=chess.Board)
    human_color: str = "w"
    mode: str = "normal"
    history_uci: list = field(default_factory=list)
    history_san: list = field(default_factory=list)
    fens: list = field(default_factory=list)
    last_engine: dict | None = None
    last_grade: dict | None = None
    sf: dict | None = None
    chat: list = field(default_factory=list)
    result: str | None = None
    think_ms: int | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)


_GAMES: dict[str, Game] = {}   # in-memory cache of loaded games


# ---------------------------------------------------------------- storage
def _path(gid: str) -> str:
    return os.path.join(GAMES, gid + ".json")


def _new_id() -> str:
    while True:
        gid = secrets.token_hex(5)
        if not os.path.exists(_path(gid)):
            return gid


def _save(g: Game) -> None:
    g.updated = time.time()
    data = {"id": g.id, "created": g.created, "updated": g.updated,
            "human_color": g.human_color, "mode": g.mode,
            "uci": g.history_uci, "chat": g.chat[-40:]}
    tmp = _path(g.id) + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _path(g.id))
    except Exception as e:
        print("[save] failed:", e, flush=True)


def _build(d: dict) -> Game:
    b = chess.Board()
    fens, sans = [b.fen()], []
    for u in d.get("uci", []):
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            raise ValueError(f"illegal move {u}")
        sans.append(b.san(m))
        b.push(m)
        fens.append(b.fen())
    g = Game(id=d["id"], board=b, human_color=d.get("human_color", "w"),
             mode=d.get("mode", "normal"), history_uci=list(d.get("uci", [])),
             history_san=sans, fens=fens, chat=d.get("chat", []),
             created=d.get("created", time.time()),
             updated=d.get("updated", time.time()))
    g.result = _result_text(b)
    return g


def get_game(gid: str) -> Game | None:
    if gid in _GAMES:
        return _GAMES[gid]
    if not GAME_ID_RE.fullmatch(gid) or not os.path.exists(_path(gid)):
        return None
    try:
        with open(_path(gid)) as f:
            g = _build(json.load(f))
    except Exception as e:
        print(f"[games] could not load {gid}: {e}", flush=True)
        return None
    _GAMES[gid] = g
    return g


def _need(gid: str) -> Game:
    g = get_game(gid)
    if g is None:
        raise HTTPException(404, "game not found (games are kept 15 days after the last move)")
    return g


def create_game(color: str, mode: str) -> Game:
    g = Game(id=_new_id(), human_color=color, mode=mode)
    g.fens = [g.board.fen()]
    _GAMES[g.id] = g
    _save(g)
    return g


def prune() -> None:
    """Delete games whose file has not been written for RETENTION_S seconds."""
    now = time.time()
    for name in os.listdir(GAMES):
        if not name.endswith(".json"):
            continue
        p = os.path.join(GAMES, name)
        if now - os.path.getmtime(p) > RETENTION_S:
            os.remove(p)
            _GAMES.pop(name[:-5], None)
            print(f"[games] expired {name}", flush=True)


def _migrate_legacy() -> None:
    """Turn the old single-game file (games/current.json) into a normal game."""
    old = os.path.join(GAMES, "current.json")
    if not os.path.exists(old):
        return
    try:
        with open(old) as f:
            d = json.load(f)
        d["id"] = _new_id()
        d["created"] = d["updated"] = os.path.getmtime(old)
        with open(_path(d["id"]), "w") as f:
            json.dump(d, f)
        os.rename(old, old + ".migrated")
        print(f"[games] migrated current game -> {d['id']}", flush=True)
    except Exception as e:
        print("[games] migration failed:", e, flush=True)


# ---------------------------------------------------------------- helpers
def _material(b: chess.Board) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    s = 0
    for p in b.piece_map().values():
        v = vals.get(p.piece_type, 0)
        s += v if p.color == chess.WHITE else -v
    return s


def _legal_map(b: chess.Board) -> dict:
    out: dict[str, list[str]] = {}
    for m in b.legal_moves:
        out.setdefault(chess.square_name(m.from_square), []).append(
            chess.square_name(m.to_square))
    return out


def _result_text(b: chess.Board) -> str | None:
    if not b.is_game_over(claim_draw=True):
        return None
    if b.is_checkmate():
        return "Checkmate — Black wins" if b.turn == chess.WHITE else "Checkmate — White wins"
    if b.is_stalemate():
        return "Draw — stalemate"
    if b.is_insufficient_material():
        return "Draw — insufficient material"
    if b.is_seventyfive_moves():
        return "Draw — 75-move rule"
    if b.is_fivefold_repetition():
        return "Draw — fivefold repetition"
    return "Draw"


def _sf_block(g: Game, depth: int = 12) -> dict | None:
    try:
        r = SF.analyse(g.board.fen(), depth=depth, multipv=3)
        lines = r["lines"]
        best_san = lines[0]["pv_san"][0] if lines and lines[0]["pv_san"] else r["best"]
        return {"eval_cp": r["eval_cp"], "mate": r["mate"], "best": r["best"],
                "best_san": best_san, "depth": (lines[0].get("depth") if lines else depth),
                "pv_san": lines[0]["pv_san"] if lines else [],
                "lines": [{"cp": l["cp"], "mate": l["mate"], "pv_san": l["pv_san"]} for l in lines]}
    except Exception as e:
        print("[stockfish] error", e)
        return None


def _pick_engine_move(g: Game, an) -> object:
    if g.mode == "casual":
        pool = an.candidates[:5]
        w = [max(c.prob, 0.01) ** 0.7 for c in pool]
        return random.choices(pool, weights=w, k=1)[0]
    return an.best


def _engine_play(g: Game) -> None:
    if g.board.is_game_over(claim_draw=True):
        g.result = _result_text(g.board)
        return
    t0 = time.time()
    depth = 2 if g.mode == "strong" else 1
    an = ENGINE.analyse(g.board, g.history_uci, repetition_count(g.fens),
                        depth=depth, root_k=4, child_k=3, time_budget=25)
    mv = _pick_engine_move(g, an)
    g.think_ms = int((time.time() - t0) * 1000)
    m = chess.Move.from_uci(mv.uci)
    san = g.board.san(m)
    g.board.push(m)
    g.history_uci.append(mv.uci)
    g.history_san.append(san)
    g.fens.append(g.board.fen())
    g.last_engine = {
        "uci": mv.uci, "san": mv.san, "prob": mv.prob,
        "winprob": an.winprob, "nodes": an.nodes, "think_ms": g.think_ms,
        "top": [{"san": c.san, "prob": c.prob} for c in an.candidates[:5]],
        "mode": g.mode,
    }
    g.result = _result_text(g.board)
    g.sf = _sf_block(g, 12)
    _save(g)


def state(g: Game) -> dict:
    b = g.board
    if g.sf is None and not b.is_game_over(claim_draw=True):
        g.sf = _sf_block(g, 10)   # lazy: first load after a restart
    last = g.history_uci[-1] if g.history_uci else None
    return {
        "id": g.id,
        "fen": b.fen(),
        "turn": "w" if b.turn else "b",
        "human_color": g.human_color,
        "mode": g.mode,
        "modes": MODES,
        "legal": _legal_map(b),
        "moves": [{"san": s, "uci": u, "color": "w" if i % 2 == 0 else "b"}
                  for i, (s, u) in enumerate(zip(g.history_san, g.history_uci))],
        "last_move": [last[:2], last[2:4]] if last else None,
        "check": b.is_check(),
        "over": b.is_game_over(claim_draw=True),
        "result": g.result or _result_text(b),
        "material": _material(b),
        "eval": {"cp": g.sf["eval_cp"], "mate": g.sf["mate"]} if g.sf else None,
        "engine": g.last_engine,
        "grade": g.last_grade,
        "chat": g.chat[-40:],
        "thinking_ms": g.think_ms,
        "engine_turn": (not b.is_game_over(claim_draw=True)) and
                       (("w" if b.turn else "b") != g.human_color),
    }


def summary(g: Game) -> dict:
    return {"id": g.id, "created": g.created, "updated": g.updated,
            "human_color": g.human_color, "mode": g.mode,
            "plies": len(g.history_uci), "result": g.result or "in progress",
            "last_moves": " ".join(g.history_san[-4:])}


# ---------------------------------------------------------------- models
class NewReq(BaseModel):
    color: str = "w"
    mode: str = "normal"


class MoveReq(BaseModel):
    uci: str


class ChatReq(BaseModel):
    message: str
    silent: bool = False


# ---------------------------------------------------------------- app
app = FastAPI(title="ChessLFM")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

_migrate_legacy()
prune()


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/g/{gid}")
def game_page(gid: str):
    # the page reads the id from the URL and loads that game
    return FileResponse(INDEX)


@app.get("/api/games")
def list_games():
    with _lock:
        prune()
        out = []
        for name in os.listdir(GAMES):
            if name.endswith(".json") and GAME_ID_RE.fullmatch(name[:-5]):
                g = get_game(name[:-5])
                if g is not None:
                    out.append(summary(g))
    out.sort(key=lambda s: s["updated"], reverse=True)
    return {"games": out, "modes": MODES, "retention_days": RETENTION_S // 86400}


@app.post("/api/games")
def new_game(req: NewReq):
    with _lock:
        prune()
        g = create_game(req.color if req.color in ("w", "b") else "w",
                        req.mode if req.mode in MODES else "normal")
        g.sf = _sf_block(g, 12)
        if g.human_color == "b":
            _engine_play(g)
        _save(g)
        return state(g)


@app.post("/api/games/import")
def import_game(d: dict = Body(...)):
    """Turn an uploaded game file (what /file downloads) into a new game."""
    uci = d.get("uci")
    if not isinstance(uci, list) or len(uci) > 1000 or not all(isinstance(u, str) for u in uci):
        raise HTTPException(400, "not a ChessLFM game file (missing 'uci' move list)")
    chat = [{"role": c["role"], "content": c["content"][:4000]}
            for c in d.get("chat", [])[-40:]
            if isinstance(c, dict) and c.get("role") in ("user", "assistant")
            and isinstance(c.get("content"), str)]
    clean = {"id": _new_id(), "uci": uci, "chat": chat,
             "human_color": d.get("human_color") if d.get("human_color") in ("w", "b") else "w",
             "mode": d.get("mode") if d.get("mode") in MODES else "normal"}
    try:
        g = _build(clean)
    except ValueError as e:
        raise HTTPException(400, f"invalid game file: {e}")
    with _lock:
        _GAMES[g.id] = g
        g.sf = _sf_block(g, 12)
        if not g.board.is_game_over(claim_draw=True) and \
                ("w" if g.board.turn else "b") != g.human_color:
            _engine_play(g)
        _save(g)
        return state(g)


@app.get("/api/games/{gid}/file")
def download_game(gid: str):
    g = _need(gid)
    day = time.strftime("%Y%m%d", time.localtime(g.created))
    return FileResponse(_path(gid), media_type="application/json",
                        filename=f"chesslfm-{day}-{gid}.json")


@app.delete("/api/games/{gid}")
def delete_game(gid: str):
    with _lock:
        _need(gid)
        os.remove(_path(gid))
        _GAMES.pop(gid, None)
        return {"deleted": gid}


@app.get("/api/games/{gid}/state")
def get_state(gid: str):
    with _lock:
        return state(_need(gid))


@app.post("/api/games/{gid}/move")
def human_move(gid: str, req: MoveReq):
    with _lock:
        g = _need(gid)
        if g.board.is_game_over(claim_draw=True):
            raise HTTPException(400, "game over")
        if ("w" if g.board.turn else "b") != g.human_color:
            raise HTTPException(400, "not your turn")
        try:
            m = chess.Move.from_uci(req.uci)
        except ValueError:
            raise HTTPException(400, "bad move")
        if m not in g.board.legal_moves:
            raise HTTPException(400, "illegal move")
        before = g.board.fen()
        san = g.board.san(m)
        g.board.push(m)
        g.history_uci.append(m.uci())
        g.history_san.append(san)
        g.fens.append(g.board.fen())
        g.last_grade = grade(before, g.board.fen(), SF, depth=9)
        g.last_grade["san"] = san
        if g.board.is_game_over(claim_draw=True):
            g.result = _result_text(g.board)
            g.sf = _sf_block(g, 12)
            _save(g)
            return state(g)
        _engine_play(g)
        _save(g)
        return state(g)


@app.post("/api/games/{gid}/undo")
def undo(gid: str):
    with _lock:
        g = _need(gid)
        if not g.history_uci:
            raise HTTPException(400, "nothing to undo")

        def replay(n: int) -> chess.Board:
            b = chess.Board()
            for u in g.history_uci[:n]:
                b.push(chess.Move.from_uci(u))
            return b

        # step back to the position just before the human's last move
        keep = max(0, len(g.history_uci) - 2)
        b = replay(keep)
        if ("w" if b.turn else "b") != g.human_color and keep > 0:
            keep -= 1
            b = replay(keep)
        g.board = b
        g.history_uci = g.history_uci[:keep]
        g.history_san = g.history_san[:keep]
        g.fens = g.fens[:keep + 1]
        g.result = None
        g.last_engine = g.last_grade = None
        g.sf = _sf_block(g, 10)
        _save(g)
        return state(g)


@app.post("/api/games/{gid}/mode")
def set_mode(gid: str, req: NewReq):
    with _lock:
        g = _need(gid)
        if req.mode in MODES:
            g.mode = req.mode
            _save(g)
        return state(g)


@app.post("/api/games/{gid}/hint")
def hint(gid: str):
    with _lock:
        g = _need(gid)
        r = SF.analyse(g.board.fen(), depth=14, multipv=1)
        line = r["lines"][0] if r["lines"] else None
        return {"best": r["best"], "san": line["pv_san"][0] if line else None,
                "cp": r["eval_cp"], "mate": r["mate"],
                "pv": line["pv_san"] if line else []}


@app.post("/api/games/{gid}/chat")
def chat(gid: str, req: ChatReq):
    g = _need(gid)
    q = (req.message or "").strip()
    if not q:
        raise HTTPException(400, "empty message")
    hist = [c for c in g.chat[-8:] if c["role"] in ("user", "assistant")]
    if not req.silent:
        g.chat.append({"role": "user", "content": q})

    def gen():
        acc = []
        try:
            for delta in coach.stream_reply(g, q, hist):
                acc.append(delta)
                yield "data: " + json.dumps({"delta": delta}) + "\n\n"
            g.chat.append({"role": "assistant", "content": "".join(acc)})
            _save(g)
            yield "data: " + json.dumps({"done": True}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"delta": f"\n[error: {e}]"}) + "\n\n"
            yield "data: " + json.dumps({"done": True}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.on_event("shutdown")
def _shutdown():
    SF.quit()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8975, log_level="info")
