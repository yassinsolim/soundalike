# Deploying soundalike as a hosted web app (Vercel)

This directory is a **self-contained Vercel deployment**: a static frontend + two
tiny Python serverless functions that recommend from the 272,853-song library using
**numpy only** (no PyTorch). You can host it on a subdomain like
`soundalike.yassin.app` and let anyone try it in the browser.

> **Release status (2026-07-12):** Production still serves the versioned
> `index-2026.07.11-dual-sonic64` asset and reports `dual_sonic64_guardrail`.
> A later audio-only method improved development but failed its once-opened final test, so it was
> not uploaded or deployed. The retained release previously passed 12 live search/recommendation/
> preview checks; that manual UX evidence is not a claim of significant retrieval improvement.

---

## Can this really run on Vercel? (the honest version)

**The full model can't** — embedding an *arbitrary* song needs PyTorch (~2.9 GB),
which is ~12× over Vercel's 250 MB serverless limit. **But it doesn't need to.**

Every song in the 272,853-row release already has precomputed neural, vibe,
EfficientNet PCA64, and CLAP PCA64 embeddings. Ranking is pure numpy (whiten →
cosine → guarded candidate union). The hosted app therefore needs only
numpy plus the 299 MB release index. `tests/test_webapp.py` pins the numpy path to
the desktop recommender so results are **byte-identical**.

| | Hosted (Vercel) | Desktop (`soundalike serve`) |
|---|---|---|
| Recommend from a library song (272,853) | ✅ numpy | ✅ |
| Recommend from *any* song (on-the-fly neural embedding) | ❌ needs torch | ✅ |
| Save to Spotify playlist | ✅ (browser → Spotify) | ✅ |
| Cost / maintenance | free, serverless | your machine |

So: **host the library demo on Vercel; keep the desktop app for arbitrary songs.**
The release catalogue contains 272,853 songs; misses are reported honestly.

---

## What runs where

```
webapp/
  index.html          # the whole UI (static) — search, results, Spotify login, save
  api/
    _reco.py          # numpy recommender (fetches the index from the GitHub Release)
    recommend.py      # POST /api/recommend
    search.py         # GET  /api/search?q=
  requirements.txt    # numpy   (that's the entire backend dependency)
  vercel.json
  dev_server.py       # local-only: mimics Vercel routing for testing
```

The index is **not** committed here. On first request the function downloads
`deepvibe_index.npz` (299,288,526 bytes) from the pinned public GitHub Release into
`/tmp`, verifies SHA-256 `f3ed57af…526fb9`, and atomically caches it for the warm
instance. A mismatch fails closed before numpy loads the file. Custom deployments
may override `SOUNDALIKE_INDEX_URL`, `SOUNDALIKE_INDEX_SHA256`, or
`SOUNDALIKE_INDEX_PATH`.

---

## Deploy it (≈5 minutes)

1. **Create the Vercel project** from your `soundalike` GitHub repo.
2. In **Project → Settings → General**, set **Root Directory = `webapp`**.
   (Framework preset: *Other*. Vercel auto-detects `api/*.py` as Python functions
   and installs `requirements.txt`.)
3. Deploy. You'll get `https://<project>.vercel.app`.
4. **Custom domain:** Project → Settings → Domains → add `soundalike.yassin.app`
   (Vercel shows the CNAME to add at your DNS provider). Your existing
   `yassin.app` / `os.yassin.app` / `strafe.yassin.app` projects are untouched —
   this is just another subdomain pointing at a different project.

That's the whole recommendation app. **No Spotify setup is needed** for search +
recommendations — only for the optional "Save as playlist".

---

## The "log in with Spotify, without giving us your password" part

Your instinct was exactly right — and it's a standard, safe flow called **OAuth
2.0 Authorization Code + PKCE**. Here's what actually happens when someone clicks
**Log in with Spotify**:

1. We send them to **accounts.spotify.com** (Spotify's own site).
2. If they're **already logged in** on spotify.com, Spotify just shows a small
   *"soundalike wants to create playlists — Agree?"* screen. If they're **not**,
   Spotify shows its own login page first.
3. They approve **on Spotify's site** and get redirected back to us with a
   one-time `code`, which the browser exchanges for a scoped **access token**.
4. **"Save as playlist" runs entirely in the browser → Spotify.** The token never
   touches our server (the frontend is static; there's no server to touch). Vercel
   never sees it.

**The user never gives us their password.** Credentials only ever go to Spotify;
we only ever receive a token limited to `playlist-modify-*`. That's the whole
point of OAuth, and it's what "Login with Spotify" buttons everywhere do.

Because it's PKCE (a *public* client), there is **no client secret** — nothing
secret ships in the frontend. A Spotify **Client ID is not a secret** (it's
visible in the OAuth URL by design).

### Enabling it
1. In the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard),
   open your app → **Settings**.
2. Add the **Redirect URI**: `https://soundalike.yassin.app/`
   (exactly your deployed URL, trailing slash included; add
   `http://127.0.0.1:8788/` too if you test locally).
3. Copy the **Client ID** and set it at the top of `webapp/index.html`:
   ```js
   const SPOTIFY_CLIENT_ID = "your_client_id_here";
   ```
   (Safe to commit — it's public. Leave it empty to ship a recommend-only demo
   with no login.)

### The one real limitation (be aware)
Spotify apps start in **Development Mode**, which only lets **up to 5 Spotify
accounts that you manually add** (Dashboard → *User Management*) log in and save
playlists. This is why the desktop "Save playlist" returned 403 earlier — your own
account just needs to be added there.

For the *general public* to log in and save, Spotify requires **Extended Quota
Mode** — and **as of May 15 2025 they only accept applications from organizations,
not individuals** (a registered business, a launched service with ≥250k monthly
active users, applied via a company email, ~6-week review). For a personal project
that's effectively unavailable. So realistically:

- **Recommendations: truly public** (no login, works for everyone). ✅
- **One-click Save-to-playlist: you + up to 4 accounts you allowlist** (5 total).
  Public one-click save isn't attainable for a solo dev under Spotify's policy.
- **Everyone else** gets a no-login **"Copy list"** button (paste into a new Spotify
  playlist) and an **Open in Spotify** link on every result. ✅

If you want *any* visitor to get a real playlist without logging in, the only route
is an **owner-account model**: store your own refresh token as a server-side secret
and have a serverless function create public playlists in your account, returning a
shareable link. It sidesteps the 5-user cap (visitors are listeners, not API users)
but every playlist lives under your account — a deliberate tradeoff, not enabled by
default.

---

## Release and verification procedure

The GitHub repository is already connected to the production Vercel project with
`webapp` as its root. For an index-backed ranking release:

1. Build the index and verify its row order, SHA-256, dimensions, and local parity.
2. Upload it as `deepvibe_index.npz` under the release tag named in
   `src/soundalike/data/index_manifest.json`.
3. Update `_INDEX_URL`, `_INDEX_VERSION`, and `_INDEX_SHA256` together.
4. Merge the verified code to `main`; the Git integration triggers production.
5. Cold-load `/api/stats`, then verify search, recommendations, and previews for at
   least ten diverse seeds. Confirm each response reports the expected retrieval
   mode and index version.

Run the same hosted code locally first:

```bash
python webapp/dev_server.py      # → http://127.0.0.1:8788/
```
