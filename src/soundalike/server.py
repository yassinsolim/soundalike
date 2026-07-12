"""Local recommendation server + web UI — the frictionless front door.

`soundalike serve` warm-loads the neural encoder and the deep-vibe index **once**,
then keeps them in memory so every subsequent lookup is fast. It exposes:

  * a small JSON API (`/api/recommend`, `/api/playlist`, `/api/seeds`, `/health`)
  * a self-contained web UI at `/`
  * permissive CORS, so a Spicetify extension running *inside* the Spotify
    desktop client can call it directly

A seed can be:
  * free text — ``Title — Artist`` (or just a title)
  * a Spotify track link — right-click any song in Spotify → Share → Copy Song
    Link → paste it (this is how the flow works with the Microsoft-Store build of
    Spotify, which Spicetify can't patch)
  * a Deezer track link

Tracks already in the library are matched instantly from their cached embedding;
anything else is embedded on the fly from a 30-second Deezer preview. No new
dependencies — everything here is Python standard library plus what the
recommender already uses.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import numpy as np


def _norm(s: str) -> str:
    """Canonical match key: strip accents, parenthetical credits/versions, and
    '- Remaster' suffixes — but keep ordinary words like 'with' (so 'Stay With Me'
    doesn't collapse to 'stay')."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.casefold()
    s = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", s)   # (feat...)/(Remaster)/[Explicit]
    s = re.sub(r"\s+-\s+.*$", "", s)               # "- 2011 Remaster" suffix
    for sep in (" feat. ", " feat ", " ft. ", " ft ", " featuring "):
        i = s.find(sep)
        if i > 0:
            s = s[:i]
    return " ".join(s.split())


def _version_penalty(title: str) -> Tuple[int, int]:
    derivative = int(bool(re.search(
        r"\b(?:karaoke|tribute|slowed|reverb|nightcore|instrumental|"
        r"remix|cover|live|acoustic)\b",
        str(title),
        re.IGNORECASE,
    )))
    return derivative, len(str(title))


_SPOTIFY_RE = re.compile(r"open\.spotify\.com/track/([A-Za-z0-9]+)")
_SPOTIFY_URI_RE = re.compile(r"spotify:track:([A-Za-z0-9]+)")
_DEEZER_RE = re.compile(r"deezer\.com/(?:[a-z]{2}/)?track/(\d+)")


def _split_text_query(q: str) -> Tuple[str, str]:
    """Split a free-text seed into (title, artist).

    Accepts the common human formats — "Title — Artist", "Title - Artist",
    "Title :: Artist", "Title by Artist" — and falls back to (whole, "") when
    there's no separator so a bare title still works.
    """
    q = (q or "").strip()
    for sep in (" — ", " – ", " :: ", " - ", " by "):
        if sep in q:
            a, b = q.split(sep, 1)
            return a.strip(), b.strip()
    return q, ""

class SoundalikeEngine:
    """Warm-loaded recommender: models stay in memory across requests."""

    def __init__(self, index_path: Optional[Path] = None,
                 model_dir: Optional[Path] = None, alpha: float = 0.8):
        from .ml.deepvibe import DeepVibeIndex, DeepVibeRecommender
        from .ml.encoder_infer import EncoderExtractor
        from .ml.index_store import ensure_pack
        from .ml.spectrogram import SpectrogramConfig

        enc_default, idx_default = (None, None)
        if index_path is None or model_dir is None:
            enc_default, idx_default = ensure_pack()
        ipath = Path(index_path) if index_path else (
            Path(idx_default) if idx_default else DeepVibeIndex.default_path())
        if not ipath or not Path(ipath).exists():
            raise FileNotFoundError(
                f"No deep-vibe library at {ipath}. Run `soundalike fetch-index` first.")

        self.index = DeepVibeIndex.load(ipath)
        self.extractor = EncoderExtractor(model_dir or enc_default)
        # enhance=True applies the quality filter and leakage-free guarded
        # artist-centroid reranker, identical to the hosted numpy path.
        self.recommender = DeepVibeRecommender(self.index, alpha=alpha, enhance=True)
        self.cfg = SpectrogramConfig()
        self._lock = threading.Lock()          # torch inference: serialize to be safe
        self._spotify_client = None
        self._spotify_tried = False

        # Build fast library-hit lookups so a track already in the library skips
        # the Deezer download entirely and reuses its cached embedding.
        self._by_pair: Dict[Tuple[str, str], int] = {}
        self._by_title: Dict[str, List[int]] = {}
        for i in range(len(self.index)):
            t, a = _norm(self.index.titles[i]), _norm(self.index.artists[i])
            previous = self._by_pair.get((t, a))
            if previous is None or _version_penalty(self.index.titles[i]) < _version_penalty(
                self.index.titles[previous]
            ):
                self._by_pair[(t, a)] = i
            self._by_title.setdefault(t, []).append(i)

    # ---------------------------------------------------------------- spotify
    def spotify(self):
        """Lazily build a Spotify client (for URL seeds, top tracks, playlists).

        Returns None if the user isn't logged in / no key configured — every
        Spotify-backed feature degrades gracefully to "unavailable".
        """
        if self._spotify_client is None and not self._spotify_tried:
            self._spotify_tried = True
            try:
                from .config import Config
                from .spotify.auth import SpotifyAuth
                from .spotify.client import SpotifyClient
                auth = SpotifyAuth(Config.from_env())
                auth.get_valid_token(interactive=False)
                self._spotify_client = SpotifyClient(auth)
            except Exception:
                self._spotify_client = None
        return self._spotify_client

    # ------------------------------------------------------------ seed resolve
    def _deezer(self):
        if getattr(self, "_deezer_client", None) is None:
            from .audio import DeezerClient
            self._deezer_client = DeezerClient()
        return self._deezer_client

    def resolve(self, query: str) -> Dict:
        """Turn a raw query (text / Spotify link / Deezer link) into a seed.

        Returns {title, artist, deezer_id, source}. `deezer_id` is set only when
        we already know the exact Deezer track (a Deezer link), letting us skip
        the search step.
        """
        q = (query or "").strip()
        m = _SPOTIFY_RE.search(q) or _SPOTIFY_URI_RE.search(q)
        if m:
            sp = self.spotify()
            if sp is None:
                raise RuntimeError("Spotify link needs login — run `soundalike login`.")
            tr = sp._get(f"/tracks/{m.group(1)}")
            artists = [a.get("name", "") for a in tr.get("artists", []) if a.get("name")]
            return {"title": tr.get("name", ""), "artist": artists[0] if artists else "",
                    "deezer_id": None, "source": "spotify"}
        m = _DEEZER_RE.search(q)
        if m:
            tr = self._deezer()._get(f"/track/{m.group(1)}")
            return {"title": tr.get("title", ""),
                    "artist": (tr.get("artist") or {}).get("name", ""),
                    "deezer_id": int(m.group(1)), "source": "deezer"}
        # Free text: split "Title — Artist" / "Title - Artist" / "Title by Artist".
        title, artist = _split_text_query(q)
        return {"title": title, "artist": artist, "deezer_id": None, "source": "text"}

    def _library_row(self, title: str, artist: str) -> Optional[int]:
        t, a = _norm(title), _norm(artist)
        if a and (t, a) in self._by_pair:
            return self._by_pair[(t, a)]
        if not a and len(self._by_title.get(t, [])) == 1:
            return self._by_title[t][0]  # unambiguous title-only hit
        return None

    # --------------------------------------------------------------- recommend
    def _seed_links(self, title: str, artist: str) -> Dict[str, str]:
        from urllib.parse import quote
        q = quote(f"{title} {artist}".strip())
        return {"spotify_url": f"https://open.spotify.com/search/{q}"}

    def recommend(self, query: str, n: int = 20, diversity: float = 0.15,
                  max_per_artist: int = 1) -> Dict:
        """Full pipeline: resolve → embed (library or preview) → rank → enrich."""
        from urllib.parse import quote

        from .audio.vibe import VibeFeatures, vibe_from_file
        from .ml.spectrogram import _fit_frames, load_audio, log_mel_full

        seed = self.resolve(query)
        with self._lock:
            row = self._library_row(seed["title"], seed["artist"])
            if row is not None:
                seed_neural = np.asarray(self.index.neural[row], dtype=np.float32)
                seed_vibe = VibeFeatures.from_vector(
                    np.asarray(self.index.vibe[row], dtype=np.float32))
                seed_title = str(self.index.titles[row])
                seed_artist = str(self.index.artists[row])
                exclude_ids = {int(self.index.track_ids[row])}
                matched = "library"
                seed_sonic = (
                    None if self.index.sonic is None
                    else np.asarray(self.index.sonic[row], dtype=np.float32)
                )
            else:
                dz = self._deezer()
                if seed["deezer_id"] is not None:
                    raw = dz._get(f"/track/{seed['deezer_id']}")
                    from .audio.previews import _parse_track
                    track = _parse_track(raw)
                else:
                    track = dz.search_track(seed["title"], seed["artist"] or None)
                if track is None or not track.has_preview:
                    return {"ok": False,
                            "error": f"No previewable track found for “{seed['title']}”.",
                            "seed": seed}
                with TemporaryDirectory() as tmp:
                    dest = Path(tmp) / f"{track.id}.mp3"
                    dz.download_preview(track, dest)
                    y = load_audio(dest, self.cfg.sample_rate)
                    spec = _fit_frames(log_mel_full(y, self.cfg), self.cfg.target_frames)
                    seed_neural = self.extractor.embed_spec(spec)
                    seed_vibe = vibe_from_file(str(dest))
                seed_title, seed_artist = track.title, track.artist
                exclude_ids = {int(track.id)}
                matched = "preview"
                seed_sonic = None

            results = self.recommender.recommend(
                seed_neural, seed_vibe, n=n, exclude_ids=exclude_ids,
                exclude_artist=seed_artist, seed_title=seed_title, diversity=diversity,
                max_per_artist=max_per_artist, seed_sonic=seed_sonic, seed_row=row)

        vibe = seed_vibe.describe()
        out = []
        for r in results:
            out.append({
                "title": r.title, "artist": r.artist,
                "neural_sim": round(float(r.neural_sim), 4),
                "vibe_sim": round(float(r.vibe_sim), 4),
                "score": round(float(r.score), 4),
                "deezer_url": f"https://www.deezer.com/track/{int(r.track_id)}",
                "spotify_url": f"https://open.spotify.com/search/{quote(f'{r.title} {r.artist}')}",
            })
        return {
            "ok": True,
            "seed": {"title": seed_title, "artist": seed_artist, "matched": matched,
                     "source": seed["source"], **self._seed_links(seed_title, seed_artist)},
            "vibe": {"tempo": vibe["tempo"], "dynamics": vibe["dynamics"],
                     "low_end": vibe["low_end"], "tone": vibe["tone"]},
            "results": out,
            "library_size": len(self.index),
            "retrieval_mode": self.recommender.last_retrieval_mode,
        }

# ============================================================ HTTP server layer
_ENGINE: Optional[SoundalikeEngine] = None
_TOKEN: str = ""  # per-process token gating the state-changing playlist endpoint

DEMO_SEEDS = [
    {"title": "Chamber of Reflection", "artist": "Mac DeMarco"},
    {"title": "Plastic Love", "artist": "Mariya Takeuchi"},
    {"title": "money machine", "artist": "100 gecs"},
    {"title": "Only Shallow", "artist": "My Bloody Valentine"},
    {"title": "OMG", "artist": "NewJeans"},
    {"title": "Redbone", "artist": "Childish Gambino"},
]


def _seed_suggestions() -> List[Dict]:
    """The user's own top tracks if logged in, else a curated demo set."""
    eng = _ENGINE
    sp = eng.spotify() if eng else None
    if sp is not None:
        try:
            top = sp.top_tracks(limit=12)
            seeds = [{"title": t["title"], "artist": t["primary_artist"]}
                     for t in top if t.get("title")]
            if seeds:
                return seeds
        except Exception:
            pass
    return DEMO_SEEDS


def _make_playlist(name: str, tracks: List[Dict]) -> Dict:
    """Resolve (title, artist) rows to Spotify URIs and save a playlist."""
    eng = _ENGINE
    sp = eng.spotify() if eng else None
    if sp is None:
        return {"ok": False, "error": "Not logged in to Spotify. Run `soundalike login`."}
    uris, missing = [], 0
    for t in tracks:
        try:
            hit = sp.search_track(t.get("title", ""), t.get("artist") or None)
        except Exception:
            hit = None
        if hit and hit.get("uri"):
            uris.append(hit["uri"])
        else:
            missing += 1
    if not uris:
        return {"ok": False, "error": "Couldn't resolve any of these tracks on Spotify."}
    try:
        pl = sp.create_playlist(name or "soundalike mix", uris,
                                description="Made with soundalike — vibe-matched recommendations.",
                                public=False)
    except Exception as e:
        # Track resolution already worked, so a failure here is Spotify declining
        # the *write* — almost always a Development-Mode app that hasn't been
        # granted playlist scope for this account. Return the resolved URIs so the
        # UI can still offer a no-write fallback (copy links / open all).
        msg = str(e)
        if "403" in msg:
            msg = ("Spotify blocked playlist creation (403). Your Spotify app is likely "
                   "in Development Mode — add your account under the app's Dashboard → "
                   "User Management (or request extended quota), then run `soundalike login` "
                   "again. Your soundalikes still resolved — use “Copy Spotify links” below.")
        return {"ok": False, "error": msg,
                "uris": uris, "resolved": len(uris), "missing": missing}
    return {"ok": True, "url": (pl.get("external_urls") or {}).get("spotify", ""),
            "added": len(uris), "missing": missing, "name": pl.get("name", name)}

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    server_version = "soundalike/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Dict):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            html = INDEX_HTML.replace("__SA_TOKEN__", _TOKEN)
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/health":
            self._json(200, {"ok": True, "library": len(_ENGINE.index) if _ENGINE else 0})
        elif path == "/api/seeds":
            self._json(200, {"ok": True, "seeds": _seed_suggestions()})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def _body(self) -> Dict:
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            return {}
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except ValueError:
            return {}

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            data = self._body()
            if path == "/api/recommend":
                q = (data.get("query") or "").strip()
                if not q:
                    return self._json(400, {"ok": False, "error": "empty query"})
                res = _ENGINE.recommend(
                    q, n=int(data.get("n", 20)),
                    diversity=float(data.get("diversity", 0.15)),
                    max_per_artist=int(data.get("max_per_artist", 1)))
                return self._json(200 if res.get("ok") else 422, res)
            if path == "/api/playlist":
                # State-changing (writes to the user's Spotify). Require the
                # per-process token that only the served UI knows, so a random
                # website open in the browser can't drive playlist creation.
                if _TOKEN and self.headers.get("X-Soundalike-Token") != _TOKEN:
                    return self._json(403, {"ok": False, "error": "bad or missing token"})
                return self._json(200, _make_playlist(
                    data.get("name", "soundalike mix"), data.get("tracks", [])))
            self._json(404, {"ok": False, "error": "not found"})
        except Exception as e:  # never crash the server on a bad request
            self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def serve(host: str = "127.0.0.1", port: int = 8787,
          index_path: Optional[Path] = None, model_dir: Optional[Path] = None,
          open_browser: bool = True, progress=print) -> int:
    """Warm-load the models and run the local server until interrupted."""
    global _ENGINE, _TOKEN
    import secrets
    _TOKEN = secrets.token_urlsafe(16)
    progress("Loading encoder + deep-vibe library (one-time)…")
    _ENGINE = SoundalikeEngine(index_path=index_path, model_dir=model_dir)
    progress(f"Ready — {len(_ENGINE.index):,} tracks in the library.")
    url = f"http://{host}:{port}/"
    httpd = ThreadingHTTPServer((host, port), _Handler)
    progress(f"\n  soundalike is live at  {url}\n"
             f"  Paste a song or a Spotify 'Copy Song Link', and get soundalikes.\n"
             f"  Press Ctrl+C to stop.\n")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        progress("\nStopping…")
    finally:
        httpd.server_close()
    return 0

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>soundalike — find songs that feel the same</title>
<style>
  :root{ --bg:#0a0d12; --card:#12161d; --card2:#171c24; --line:#232a34;
    --txt:#eef1f5; --mut:#8b93a1; --grn:#1db954; --vio:#8b6cff; }
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:radial-gradient(1200px 600px at 70% -10%,#15202c 0%,var(--bg) 55%);
    color:var(--txt);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:32px 20px 80px}
  .brand{display:flex;align-items:center;gap:12px;margin-bottom:6px}
  .logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,var(--grn),var(--vio));
    display:grid;place-items:center;font-weight:800;color:#0a0d12}
  h1{font-size:26px;margin:0;letter-spacing:-.02em}
  .sub{color:var(--mut);margin:2px 0 22px}
  .search{display:flex;gap:10px}
  .search input{flex:1;background:var(--card);border:1px solid var(--line);color:var(--txt);
    padding:14px 16px;border-radius:12px;font-size:15px;outline:none}
  .search input:focus{border-color:var(--grn)}
  .btn{background:var(--grn);color:#06210f;border:0;border-radius:12px;padding:0 20px;
    font-weight:700;font-size:15px;cursor:pointer;white-space:nowrap}
  .btn:disabled{opacity:.6;cursor:default}
  .hint{color:var(--mut);font-size:12.5px;margin:10px 2px 0}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 6px}
  .chip{background:var(--card);border:1px solid var(--line);color:var(--txt);
    padding:7px 12px;border-radius:999px;font-size:13px;cursor:pointer}
  .chip:hover{border-color:var(--vio)}
  .chip b{color:var(--mut);font-weight:500}
  .seed{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);
    border-radius:16px;padding:18px 18px;margin:22px 0 14px;display:flex;
    justify-content:space-between;align-items:flex-start;gap:16px}
  .seed h2{margin:0;font-size:19px}.seed .art{color:var(--mut)}
  .badge{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:999px;
    padding:3px 9px;margin-left:8px;vertical-align:middle}
  .tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
  .tag{background:#0e1319;border:1px solid var(--line);color:#c7cdd6;border-radius:8px;
    padding:4px 9px;font-size:12px}
  .save{background:transparent;border:1px solid var(--grn);color:var(--grn);border-radius:10px;
    padding:9px 13px;font-weight:600;cursor:pointer;font-size:13px;white-space:nowrap}
  .save:disabled{opacity:.55;cursor:default}
  ol{list-style:none;margin:0;padding:0}
  li.row{display:flex;align-items:center;gap:14px;padding:11px 12px;border-radius:12px}
  li.row:hover{background:var(--card)}
  .rank{width:22px;text-align:right;color:var(--mut);font-variant-numeric:tabular-nums}
  .meta{flex:1;min-width:0}
  .meta .t{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .meta .a{color:var(--mut);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bar{width:96px;height:6px;background:#0e1319;border-radius:6px;overflow:hidden}
  .bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--vio),var(--grn))}
  .play{background:var(--card2);border:1px solid var(--line);color:var(--txt);border-radius:9px;
    padding:8px 12px;font-size:13px;cursor:pointer;text-decoration:none;white-space:nowrap}
  .play:hover{border-color:var(--grn)}
  .state{color:var(--mut);padding:26px 4px}
  .err{color:#ff8a8a}
  .foot{color:var(--mut);font-size:12px;margin-top:28px;border-top:1px solid var(--line);padding-top:14px}
  .spin{display:inline-block;width:15px;height:15px;border:2px solid var(--line);
    border-top-color:var(--grn);border-radius:50%;animation:sp 0.8s linear infinite;vertical-align:-2px}
  @keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="brand"><div class="logo">◈</div>
    <div><h1>soundalike</h1></div></div>
  <div class="sub">Find songs that <i>feel</i> the same — matched by how they actually sound, not tags.</div>

  <div class="search">
    <input id="q" placeholder="Paste a Spotify song link, or type:  Title — Artist" autocomplete="off"/>
    <button class="btn" id="go">Find soundalikes</button>
  </div>
  <div class="hint">Tip: in Spotify, right-click a song → <b>Share → Copy Song Link</b>, then paste it here. Works with any Spotify.</div>

  <div class="chips" id="chips"></div>
  <div id="out"></div>

  <div class="foot" id="foot"></div>
</div>
<script>
const $=s=>document.querySelector(s);
const TOKEN="__SA_TOKEN__";
let lastResults=[], lastSeed=null;

async function loadSeeds(){
  try{ const r=await fetch('/api/seeds'); const d=await r.json();
    const c=$('#chips'); c.innerHTML='';
    (d.seeds||[]).slice(0,8).forEach(s=>{
      const b=document.createElement('button'); b.className='chip';
      b.innerHTML=`${esc(s.title)} <b>· ${esc(s.artist)}</b>`;
      b.onclick=()=>{ $('#q').value=`${s.title} — ${s.artist}`; run(); };
      c.appendChild(b);
    });
  }catch(e){}
}

function esc(s){ return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function run(){
  const q=$('#q').value.trim(); if(!q) return;
  $('#go').disabled=true;
  $('#out').innerHTML='<div class="state"><span class="spin"></span> listening & matching…</div>';
  try{
    const r=await fetch('/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q,n:20,diversity:0.15})});
    const d=await r.json();
    if(!d.ok){ $('#out').innerHTML=`<div class="state err">${esc(d.error||'No match found.')}</div>`; return; }
    render(d);
  }catch(e){ $('#out').innerHTML=`<div class="state err">Server error: ${esc(''+e)}</div>`; }
  finally{ $('#go').disabled=false; }
}

function render(d){
  lastResults=d.results; lastSeed=d.seed;
  const v=d.vibe, s=d.seed;
  const mx=Math.max(...d.results.map(x=>x.neural_sim),0.001);
  const badge = s.matched==='library' ? 'in library' : 'analyzed preview';
  let html=`<div class="seed"><div>
      <h2>${esc(s.title)} <span class="badge">${badge}</span></h2>
      <div class="art">${esc(s.artist)}</div>
      <div class="tags"><span class="tag">${esc(v.tempo)}</span><span class="tag">${esc(v.dynamics)}</span>
        <span class="tag">${esc(v.low_end)}</span><span class="tag">${esc(v.tone)}</span></div>
    </div><button class="save" id="save">＋ Save as Spotify playlist</button></div><ol>`;
  d.results.forEach((x,i)=>{
    const w=Math.round(100*x.neural_sim/mx);
    html+=`<li class="row"><div class="rank">${i+1}</div>
      <div class="meta"><div class="t">${esc(x.title)}</div><div class="a">${esc(x.artist)}</div></div>
      <div class="bar"><i style="width:${w}%"></i></div>
      <a class="play" href="${x.spotify_url}" target="_blank" rel="noopener">Open in Spotify ▸</a></li>`;
  });
  html+='</ol>';
  $('#out').innerHTML=html;
  $('#save').onclick=savePlaylist;
  $('#foot').textContent=`Matched from a ${d.library_size.toLocaleString()}-track library · seed source: ${s.source}`;
}

async function savePlaylist(){
  const b=$('#save'); b.disabled=true; const old=b.textContent; b.textContent='Saving…';
  try{
    const name=`soundalike · ${lastSeed.title}`;
    const r=await fetch('/api/playlist',{method:'POST',headers:{'Content-Type':'application/json','X-Soundalike-Token':TOKEN},
      body:JSON.stringify({name, tracks:lastResults})});
    const d=await r.json();
    if(d.ok){ b.textContent=`✓ Saved ${d.added} songs`;
      if(d.url){ window.open(d.url,'_blank'); } return; }
    // Graceful fallback: playlist write blocked (e.g. Spotify dev-mode). Offer copy.
    b.textContent='⧉ Copy tracklist'; b.disabled=false; b.onclick=copyTracklist;
    const note=document.createElement('div');
    note.className='hint err'; note.style.marginTop='8px'; note.id='savenote';
    note.textContent=d.error||'Playlist save was blocked by Spotify.';
    if(!$('#savenote')) $('.seed').after(note);
  }catch(e){ b.textContent='✕ error'; setTimeout(()=>{b.textContent=old;b.disabled=false;},2600); }
}

async function copyTracklist(){
  const lines=lastResults.map(x=>`${x.title} — ${x.artist}`).join('\n');
  try{ await navigator.clipboard.writeText(lines);
    const b=$('#save'); b.textContent='✓ Copied '+lastResults.length+' songs';
  }catch(e){ window.prompt('Copy your soundalikes:', lines); }
}

$('#go').onclick=run;
$('#q').addEventListener('keydown',e=>{ if(e.key==='Enter') run(); });
loadSeeds();
</script>
</body>
</html>"""
