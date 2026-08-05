# Deploying soundalike as a hosted web app (Vercel)

This directory is a **self-contained Vercel deployment**: a static frontend,
small Python recommendation functions, and one Node ratings-ingest function. The
recommender serves the 272,853-song library using **numpy only** (no PyTorch). You
can host it on a subdomain like
`soundalike.yassin.app` and let anyone try it in the browser.

> **Release status (2026-07-11):** Dual-Sonic64 uses the versioned
> `index-2026.07.11-dual-sonic64` release asset. Desktop and hosted paths checksum the
> same 272,853-row index and expose `dual_sonic64_guardrail` plus the active index
> version in recommendation responses. Production was live-verified on 12 diverse
> seeds with search and fresh preview lookups passing.

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

## Always-on recommendation origin

Production recommendations use the same numpy endpoint on an Ubuntu VM so model
initialization happens once at service startup instead of during a user request.
The public hostname `soundalike-api.yassin.app` is a Cloudflare Tunnel route to
`http://127.0.0.1:8788`; the VM has no public application port.

The reproducible service files are in `deploy/homelab/`. On the VM:

```bash
sudo useradd --system --home /var/lib/soundalike --shell /usr/sbin/nologin soundalike
sudo install -d -o root -g soundalike -m 0750 /opt/soundalike /var/lib/soundalike
sudo git clone https://github.com/yassinsolim/soundalike.git /opt/soundalike
sudo python3 -m venv /opt/soundalike/.venv
sudo /opt/soundalike/.venv/bin/pip install -r /opt/soundalike/webapp/requirements.txt
curl --fail --location --output /tmp/deepvibe_index.npz \
  https://github.com/yassinsolim/soundalike/releases/download/index-2026.07.11-dual-sonic64/deepvibe_index.npz
echo 'f3ed57af1b8073f2872eed1e9192dee04d1089c7266fb98a157d1ea194526fb9  /tmp/deepvibe_index.npz' |
  sha256sum --check
sudo install -o root -g soundalike -m 0640 /tmp/deepvibe_index.npz \
  /var/lib/soundalike/deepvibe_index.npz
sudo install -o root -g root -m 0644 \
  /opt/soundalike/deploy/homelab/soundalike.service \
  /etc/systemd/system/soundalike.service
sudo systemctl daemon-reload
sudo systemctl enable --now soundalike.service
curl --fail http://127.0.0.1:8788/healthz
```

Keep the Cloudflare Tunnel token only in root-readable VM configuration. Never
place it in this repository. Configure the tunnel ingress to send
`soundalike-api.yassin.app` to `http://localhost:8788`, followed by a catch-all
404 rule.

The website and Spicetify extension try this always-on origin first and retain
Vercel's cacheable GET endpoint as a fallback. The Vercel POST endpoint remains
available for compatibility but is not the normal interactive path.

---

## What runs where

```
webapp/
  index.html          # the whole UI (static) — search, results, Spotify login, save
  search.js           # abortable autocomplete, prefix reuse, and idle prewarming
  build_search_catalog.py
  api/
    _search.py        # stdlib-only title/artist catalog and bounded query cache
    search_catalog.json.gz
    _reco.py          # numpy recommender (fetches the index from the GitHub Release)
    recommend.py      # POST /api/recommend
    search.py         # GET  /api/search?q=
    ratings.js        # POST-only private v17 ratings ingestion
    ratings-v2.js     # POST-only private full-track v2 ingestion
  evaluate/           # canonical full-track v2 evaluator + public locked pack
  evaluate-v1/        # byte-preserved v17 evaluator public payload
  package.json        # official @vercel/blob SDK
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

Autocomplete does not download or initialize that model. The row-aligned
`search_catalog.json.gz` is 3,961,198 bytes, contains only title/artist metadata,
and is pinned to SHA-256 `c9ce8b8f…adbdf`. `/api/search` loads this catalog with
the standard library, caches up to 256 normalized queries per warm instance, and
returns the original production row so `/api/recommend` can use it directly. The
full model remains lazy until a user selects a song.

When the production index changes, rebuild and commit the catalog from the exact
release index:

```bash
python webapp/build_search_catalog.py deepvibe_index.npz
```

Update the index version, index checksum, catalog checksum, and production row
count together in `api/_search.py`. Custom catalogs may instead set both
`SOUNDALIKE_SEARCH_CATALOG_PATH` and `SOUNDALIKE_SEARCH_CATALOG_SHA256`.

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

## Private ratings inboxes

`/evaluate` is the research-only 20-seed full-track v2 pilot. `/evaluate-v1`
retains the exact v17 browser application and files so existing
`soundalike-human-v17` autosaves remain resumable. The v2 page instead uses the
`soundalike-fulltrack-v2` namespace and never scans, migrates, or deletes v17
state.

Both evaluators submit only after the listener checks the consent box and presses
the explicit submit button. Neither submits on autosave, page unload, playback,
or export. JSON export/import remains a manual fallback. V2 attribution and
license links appear only after the corresponding list judgment; the public pack
contains no model identity or private unblinding document.

### Blob setup

1. Create a **private Vercel Blob store** and connect it only to this Vercel project.
   Do not use a public store.
2. Prefer Vercel OIDC credentials: enable project OIDC, set `BLOB_STORE_ID` to the
   private store ID, and let the runtime supply `VERCEL_OIDC_TOKEN`.
3. If OIDC is unavailable, set the store's `BLOB_READ_WRITE_TOKEN` as a sensitive
   server-side environment variable. `@vercel/blob` uses this official fallback
   automatically. Never expose either credential to browser code.
4. Deploy from `webapp`; `npm ci` installs the pinned official Blob SDK.
5. Add Vercel Firewall rate-limit rules for `POST /api/ratings` and
   `POST /api/ratings-v2`. Origin checks and
   the browser's local-key HMAC provide abuse resistance and integrity; they are
   not authentication, so application validation is not a replacement for rate
   limiting.

Accepted records use immutable, deduplicated private paths:
`human-ratings/v17/<session-id>/<canonical-payload-sha>.json`. A retry of the exact
snapshot returns the same receipt without overwriting. A later snapshot with added
ratings gets a different digest and may coexist.

V2 uses its own immutable prefix and schema:
`human-ratings/fulltrack-v2/<fulltrack-session-id>/<canonical-payload-sha>.json`.
A v1 payload cannot validate against the v2 endpoint, and a v2 payload cannot
validate against the v1 endpoint.

The stored record contains random anonymous/session IDs, ratings, rating timestamps
and durations, locked protocol/list hashes, server receipt time, canonical digest,
and server-derived counts. It strips the local HMAC key and HMAC. Application code
does not store IP addresses, Origin, user-agent, cookies, raw headers, Spotify
identity, email, or a Blob URL. Vercel may process request metadata in operational
infrastructure; review project log settings separately. There is no public GET, listing, admin, or unblinding endpoint for either inbox.

### Authorized analyst and retention workflow

Private downloads are deliberately local-only. From an authorized workstation:

```bash
cd webapp
npm ci
npm run ratings:inbox -- ../private-ratings-inbox --acknowledge-private-data
npm run ratings:v2-inbox -- ../private-ratings-v2-inbox --acknowledge-private-data
python ../tools/aggregate_ratings.py ../private-ratings-inbox \
  --output ../ratings-aggregate.local.json
```

The inbox command uses the same official SDK credential resolution (OIDC first,
static token fallback), validates every private object path, and never prints Blob
URLs or rating contents. The aggregator also accepts already-downloaded v16 signed
client exports and v17 sanitized server records. It deduplicates snapshots by digest
and session, merges additions, and stops on conflicting values rather than silently
choosing one. Neither private inputs nor local aggregates should be committed.

Before opening the pilot, an authorized analyst must use private Blob
credentials to record the v17 prefix count, submit one fresh v2 test snapshot,
download and validate that receipt only through `ratings:v2-inbox`, confirm it
does not appear under the v17 prefix, and delete the exact test object with the
official private Blob SDK/CLI. Record only counts and the deletion result, never
listener identifiers, credentials, private URLs, or rating contents.

Choose and document a retention period before collection. On review dates, an
authorized analyst should download and verify the private inbox, keep only the
approved encrypted analysis copy, and delete expired Blob objects with the official
Vercel Blob SDK/CLI. Record deletion totals without recording listener identifiers.
Do not retain credentials in shell history or analysis output.

For an end-to-end local function test use `vercel dev` with a dedicated test-only
private store. `python webapp/dev_server.py` remains useful for recommendation and
preview UI work, but intentionally does not emulate or weaken private ratings
validation.

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
