# Deploying Soundalike as a hosted web app (Vercel)

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

## Vercel limits

**The full model cannot run there.** Embedding an *arbitrary* song needs PyTorch
(~2.9 GB), which is about 12 times over Vercel's 250 MB serverless limit. The
hosted catalog does not need to embed songs during a request.

Every song in the 272,853-row release already has precomputed neural, vibe,
EfficientNet PCA64, and CLAP PCA64 embeddings. Ranking is pure numpy (whiten →
cosine → guarded candidate union). The hosted app therefore needs only
numpy plus the 299 MB release index. `tests/test_webapp.py` pins the numpy path to
the desktop recommender so results are **byte-identical**.

| | Hosted (Vercel) | Desktop (`soundalike serve`) |
|---|---|---|
| Recommend from a library song (272,853) | Yes, with numpy | Yes |
| Recommend from *any* song (on-the-fly neural embedding) | No, needs PyTorch | Yes |
| Save to Spotify playlist | Yes, through the browser | Yes |
| Cost / maintenance | free, serverless | your machine |

Vercel hosts the catalog app, while the desktop app handles arbitrary songs.
The release catalog contains 272,853 songs and reports tracks outside that
catalog as missing.

---

## Always-on recommendation origin

Production recommendations use the same numpy endpoint on an Ubuntu VM so model
initialization happens once at service startup instead of during a user request.
The public hostname `soundalike-api.yassin.app` is a Cloudflare Tunnel route to
`http://127.0.0.1:8788`; the VM has no public application port.

The reproducible service files are in `deploy/homelab/`. The public hostname
is **only** a Cloudflare Tunnel route to `http://127.0.0.1:8788`; do not expose
the application port on the VM firewall.

### Safe releases and rollback

The production unit always executes `/opt/soundalike/current`. The updater
builds an immutable, commit-named release in `/opt/soundalike/releases/`,
verifies the checked-out commit plus the committed runtime-file and
`webapp/requirements.txt` checksums, then downloads and verifies the pinned
release index. It only switches `current` after all checks and dependency
installation succeed. The old target remains both on disk and at
`/opt/soundalike/previous`.

After switching the symlink it reloads and restarts `soundalike.service`, checks
local `/healthz`, and runs a full strict API-v4 canary. The canary requires a
200 JSON response, `dual_sonic64_guardrail` method and retrieval mode, the
`2026.07.11-dual-sonic64` index, `spotify-lyrics-strict-v2`,
`model-quality-v1`, and non-empty results. Any restart or probe failure
atomically restores the prior symlink, reinstalls the prior unit, restarts it,
and verifies rollback health before reporting failure.

Initial bootstrap (replace the commit with an immutable 40-character SHA):

```bash
sudo groupadd --system soundalike
sudo useradd --system --gid soundalike --home /var/lib/soundalike --shell /usr/sbin/nologin soundalike
sudo install -d -o root -g soundalike -m 0750 /opt/soundalike /var/lib/soundalike
sudo git clone https://github.com/yassinsolim/soundalike.git /opt/soundalike/bootstrap
commit=PUT_A_FULL_40_CHARACTER_COMMIT_SHA_HERE
sudo python3 /opt/soundalike/bootstrap/deploy/homelab/update.py --commit "$commit"
sudo systemctl enable soundalike.service
```

For every later release, first inspect the no-change dry run, then run the same
command from `current`:

```bash
sudo /opt/soundalike/current/deploy/homelab/update.py \
  --commit PUT_A_FULL_40_CHARACTER_COMMIT_SHA_HERE --dry-run
sudo /opt/soundalike/current/deploy/homelab/update.py \
  --commit PUT_A_FULL_40_CHARACTER_COMMIT_SHA_HERE
```

The updater has no password argument and does not accept mutable refs. Use
normal `sudo` policy; grant an automation account access only to this updater,
not unrestricted root. It uses normal TLS certificate verification for the
public Git repository and index download. Keep the Cloudflare Tunnel token only
in root-readable VM configuration, never in this repository. Configure the
tunnel ingress to send `soundalike-api.yassin.app` to
`http://127.0.0.1:8788`, followed by a catch-all 404 rule.

When changing a checksummed runtime file or `webapp/requirements.txt`, regenerate
and commit the manifest in the same change:

```bash
python3 deploy/homelab/write_manifest.py
python3 deploy/homelab/write_manifest.py --check
```

`deploy/homelab/probe_v4.py` is also safe to run from an external monitor:

```bash
python3 deploy/homelab/probe_v4.py --url https://soundalike-api.yassin.app
```

The scheduled GitHub Actions workflow runs this probe every six hours. A failed
workflow uses GitHub's normal failed-run notifications. It sends a webhook only
when the optional protected `SOUNDALIKE_MONITOR_WEBHOOK` secret is set. Manual
deployment is disabled unless the protected production environment contains
the SSH key, host key, host, and user secrets **and**
`SOUNDALIKE_DEPLOY_ENABLED=true`; it uses `BatchMode` and
`StrictHostKeyChecking=yes`, never a password or relaxed host verification.

The website and Spicetify extension try this always-on origin first and retain
Vercel's cacheable GET endpoint as a fallback. The Vercel POST endpoint remains
available for compatibility but is not the normal interactive path.

---

## What runs where

```
webapp/
  index.html          # the whole static UI: search, results, Spotify login, save
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
    ratings-pacing-v3.js # POST-only private pacing V3 ingestion
    ratings-v5.js     # archived V5 private ratings ingestion
    ratings-v6.js     # active V6 development ratings ingestion
    spicetify-feedback.js # public-CORS, private-Blob extension feedback
  evaluate/           # active blinded V6 evaluator + compatibility V5 assets
  evaluate-v5/        # byte-preserved V5 evaluator
  evaluate-semantic-v2/ # byte-preserved semantic v2 evaluator
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
   `yassin.app` / `os.yassin.app` / `strafe.yassin.app` projects are untouched;
   this is just another subdomain pointing at a different project.

That is the complete recommendation app. **No Spotify setup is needed** for
search and recommendations. Spotify setup is only needed for the optional
"Save as playlist" feature.

## Private ratings inboxes

`/evaluate` is the V6 development/model-improvement full-ranking study. It uses
the isolated `soundalike-development-v6-ranking-v1` browser namespace and a
locked four-candidate pack. Listeners assign the most similar, next most
similar, second least similar, and least similar positions, then give the
existing closed worst-item reason. V6 is explicitly **not** an independent
promotion holdout and cannot authorize model promotion on its own.
Playback is limited to committed approximately 20-second strongest-recurrence
excerpts. The excerpt is a recurrence heuristic, not a verified chorus classifier.
Tasks prioritize disagreements among three frozen methods without using current
submitted ratings, include two repeated anchors, and allow adaptive stopping after 12
unique comparisons. All 80 unique study tracks have distinct artists, and every
track and artist exposed by earlier evaluator packs, including V5, is excluded.
The consumed, inconclusive V5 application is byte-preserved at `/evaluate-v5`
with its original `soundalike-strict-v5-ranking-v1` state, `/api/ratings-v5`
receipt behavior, and private analysis namespace. Strict V4 is preserved at
`/evaluate-v4` with its original
`soundalike-active-v4-ranking-v2` state. The pacing V3 study is preserved at
`/evaluate-pacing-v3` with its `soundalike-pacing-v3` state.
The semantic v2 study remains at `/evaluate-semantic-v2` with its isolated state,
and semantic v1 remains at `/evaluate-semantic-v1`. The prior
full-track V2 application is byte-preserved at `/evaluate-v2` so existing
`soundalike-fulltrack-v2` autosaves remain resumable. `/evaluate-v1` retains the
exact v17 browser application and its `soundalike-human-v17` state. None of the
eight pages scans, migrates, or deletes another study's state.

All evaluators submit only after the listener checks the consent box and presses
the explicit submit button. Neither submits on autosave, page unload, playback,
or export. JSON export/import remains a manual fallback. In V4, V5, and V6,
attribution appears only after a complete A-to-D ranking is saved or the task
is skipped. For pacing V3, attribution and
license links appear only after the list's overall
0 to 10 score and all five required result scores are complete. Optional mismatch reasons
use a closed enum and no free text. Public packs contain no method identity or private
unblinding document. V5 and V6 classify three separate positions for every plausible vocal
reserve track, requires all three decisions to agree, rejects vocal/instrumental
detector conflicts, and requires exact same-language candidates for vocal seeds. It
saves language decisions but no transcript.
The archived pacing study did not evaluate language.

### Blob setup

1. Create a **private Vercel Blob store** and connect it only to this Vercel project.
   Do not use a public store.
2. Prefer Vercel OIDC credentials: enable project OIDC, set `BLOB_STORE_ID` to the
   private store ID, and let the runtime supply `VERCEL_OIDC_TOKEN`.
3. If OIDC is unavailable, set the store's `BLOB_READ_WRITE_TOKEN` as a sensitive
   server-side environment variable. `@vercel/blob` uses this official fallback
   automatically. Never expose either credential to browser code.
4. Deploy from `webapp`; `npm ci` installs the pinned official Blob SDK.
5. Add Vercel Firewall rate-limit rules for `POST /api/ratings`,
   `POST /api/ratings-v2`, `POST /api/ratings-semantic-v1`,
   `POST /api/ratings-semantic-v2`, `POST /api/ratings-pacing-v3`,
   `POST /api/ratings-v4`, `POST /api/ratings-v5`, `POST /api/ratings-v6`,
   and `POST /api/spicetify-feedback`. Ratings origin checks and the browser's
   local-key HMAC provide abuse resistance and integrity; these controls are
   not authentication. The extension feedback endpoint is intentionally
   unauthenticated and allows public CORS because a Spicetify extension cannot
   safely hold a secret. Its exact schema, body limits, immutable digest
   deduplication, and private storage are still not substitutes for a Vercel
   Firewall rate limit.

Accepted records use immutable, deduplicated private paths:
`human-ratings/v17/<session-id>/<canonical-payload-sha>.json`. A retry of the exact
snapshot returns the same receipt without overwriting. A later snapshot with added
ratings gets a different digest and may coexist.

V2 uses its own immutable prefix and schema:
`human-ratings/fulltrack-v2/<fulltrack-session-id>/<canonical-payload-sha>.json`.
A v1 payload cannot validate against the v2 endpoint, and a v2 payload cannot
validate against the v1 endpoint.

The archived semantic v1 study remains isolated at:
`human-ratings/semantic-v1/<semantic-session-id>/<canonical-payload-sha>.json`.
Its schema, hashes, IDs, endpoint, and committed list set reject v1 and V2 payloads.
The archived semantic v2 study writes only to:
`human-ratings/semantic-v2/<semantic-session-id>/<canonical-payload-sha>.json`.
The archived pacing V3 study writes only to:
`human-ratings/pacing-v3/<pacing-session-id>/<canonical-payload-sha>.json`.
The archived V4 study writes only to:
`human-ratings/active-v4-ranking-v2/<v4-session-id>/<canonical-payload-sha>.json`.
The archived V5 study writes only to:
`human-ratings/strict-v5-ranking-v1/<v5-session-id>/<canonical-payload-sha>.json`.
The active V6 development study writes only to:
`human-ratings/development-v6-ranking-v1/<v6-session-id>/<canonical-payload-sha>.json`.

Optional Spicetify feedback uses its own immutable private prefix:
`spicetify-feedback/match-quality-v1/<canonical-payload-sha>.json`. The public
CORS endpoint accepts only `POST` and bounded `OPTIONS` preflight, requires
unencoded `application/json`, rejects unknown keys and invalid enum/count/string
bounds, and caps notes at 280 characters. The digest is computed from the
normalized accepted payload, so a retry receives the same receipt without an
overwrite. Responses contain only `receipt_sha256`, never a Blob URL.

The stored record contains random anonymous/session IDs, ratings, rating timestamps
and durations, locked protocol/list hashes, server receipt time, canonical digest,
and server-derived counts. It strips the local HMAC key and HMAC. Application code
does not store IP addresses, Origin, user-agent, cookies, raw headers, Spotify
identity, email, or a Blob URL. Vercel may process request metadata in operational
infrastructure; review project log settings separately. There is no public GET,
listing, admin, or unblinding endpoint for any inbox. Feedback application
records likewise omit account identity, credentials, library/history, headers,
IP addresses, user agent, and hidden candidates. They contain only the seed,
displayed rows in order, bounded policy labels, selected feedback, an optional
plain-text note, random anonymous nonces, receipt time, and digest.

### Authorized analyst and retention workflow

Private downloads are deliberately local-only. From an authorized workstation:

```bash
cd webapp
npm ci
PRIVATE_ROOT="${SOUNDALIKE_PRIVATE_ROOT:-$HOME/.soundalike/private}"
mkdir -p "$PRIVATE_ROOT"
npm run ratings:inbox -- "$PRIVATE_ROOT/ratings-v1" --acknowledge-private-data
npm run ratings:v2-inbox -- "$PRIVATE_ROOT/ratings-v2" --acknowledge-private-data
npm run ratings:semantic-inbox -- "$PRIVATE_ROOT/ratings-semantic-v1" --acknowledge-private-data
npm run ratings:semantic-v2-inbox -- "$PRIVATE_ROOT/ratings-semantic-v2" --acknowledge-private-data
npm run ratings:pacing-v3-inbox -- "$PRIVATE_ROOT/ratings-pacing-v3" --acknowledge-private-data
npm run ratings:v4-inbox -- "$PRIVATE_ROOT/ratings-v4" --acknowledge-private-data
npm run ratings:v4-analysis -- "$PRIVATE_ROOT/ratings-v4" \
  "$PRIVATE_ROOT/private-v4-unblinding.json" "$PRIVATE_ROOT/ratings-v4-analysis.local.json" \
  --acknowledge-private-data
npm run ratings:v5-inbox -- "$PRIVATE_ROOT/ratings-v5" --acknowledge-private-data
npm run ratings:v5-analysis -- "$PRIVATE_ROOT/ratings-v5" \
  "$PRIVATE_ROOT/private-v5-unblinding.json" "$PRIVATE_ROOT/ratings-v5-analysis.local.json" \
  --acknowledge-private-data
npm run ratings:v6-inbox -- "$PRIVATE_ROOT/ratings-v6" --acknowledge-private-data
npm run ratings:v6-analysis -- "$PRIVATE_ROOT/ratings-v6" \
  "$PRIVATE_ROOT/private-v6-unblinding.json" "$PRIVATE_ROOT/ratings-v6-analysis.local.json" \
  --acknowledge-private-data
npm run feedback:inbox -- "$PRIVATE_ROOT/spicetify-feedback" \
  --acknowledge-private-data --retention-days 90
python ../tools/aggregate_ratings.py "$PRIVATE_ROOT/ratings-v1" \
  --output "$PRIVATE_ROOT/ratings-aggregate.local.json"
```

The inbox commands use the same official SDK credential resolution (OIDC first,
static token fallback), validate every private object path and byte count, and
never print Blob URLs, free-text feedback comments, or rating contents. The
feedback downloader reports only downloaded/existing counts and how many records
are older than the configured retention window; it never deletes automatically.
The aggregator also accepts already-downloaded v16 signed
client exports and v17 sanitized server records. It deduplicates snapshots by digest
and session, merges additions, and stops on conflicting values rather than silently
choosing one. Neither private inputs nor local aggregates should be committed.
The V4 analyzer validates the active pack and private unblinding binding, uses only
the latest valid snapshot per session, excludes repeated anchors from primary
pairwise totals, and separately reports mismatch reasons, skips, and anchor
consistency. Repeated listener observations are averaged within task for primary
inference, with listener-clustered sensitivity reported separately. Its report never
makes an automatic promotion decision. The repository also ignores conventional
private inbox, unblinding, and local-analysis names as defense in depth, but the
external private root remains mandatory.
The V5 analyzer applies the same snapshot and anchor rules, then scores all six
pairwise predictions made by each frozen method for every complete A-D ranking.
Primary inference averages multiple listener observations of the same task into one
cluster before exact task-level sign-flip comparisons. Listener-clustered sensitivity
results are reported separately. The analyzer never makes an automatic promotion
decision.
The V6 analyzer keeps that method-order and task-clustered analysis behavior in
the new development-only namespace. It validates the V6 public/private binding
and does not reinterpret V6 as independent promotion evidence. Sign-flip
inference is exact through 16 nonzero clusters (the complete task population);
listener sensitivity uses 100,000 deterministic Monte Carlo draws above that
bound so a large response count cannot cause exponential memory growth.

Before opening a study, an authorized analyst must use private Blob credentials
to record the other prefix counts, submit one fresh test snapshot, download and
validate that receipt only through the matching inbox command, confirm it does not
appear under either other prefix, and delete the exact test object with the official
private Blob SDK/CLI. Record only counts and the deletion result, never listener
identifiers, credentials, private URLs, or rating contents.

Use **90 days** as the private Spicetify feedback record retention period unless
a documented legal or research requirement approves a different duration. On
review dates, an authorized analyst should download and verify the private
inbox, keep only the approved encrypted analysis copy, and delete expired Blob
objects with the official Vercel Blob SDK/CLI. Deletion is a separate,
deliberate maintainer action; the inbox tool only identifies the count eligible
for review. Record deletion totals without recording listener identifiers,
comments, or URLs. Do not retain credentials in shell history or analysis
output.

For an end-to-end local function test use `vercel dev` with a dedicated test-only
private store. `python webapp/dev_server.py` remains useful for recommendation and
preview UI work, but intentionally does not emulate or weaken private ratings
validation.

---

## Spotify OAuth flow

This uses **OAuth 2.0 Authorization Code + PKCE**. When someone clicks
**Log in with Spotify**:

1. We send them to **accounts.spotify.com** (Spotify's own site).
2. If they're **already logged in** on spotify.com, Spotify just shows a small
   *"Soundalike wants to create playlists. Agree?"* screen. If they are **not**,
   Spotify shows its own login page first.
3. They approve **on Spotify's site** and get redirected back to us with a
   one-time `code`, which the browser exchanges for a scoped **access token**.
4. **"Save as playlist" runs entirely in the browser → Spotify.** The token never
   touches our server (the frontend is static; there's no server to touch). Vercel
   never sees it.

**The user never gives Soundalike their password.** Credentials only go to
Spotify. The browser receives a token limited to `playlist-modify-*`.

Because it uses PKCE (a *public* client), there is **no client secret**. No
secret is included in the frontend. A Spotify **Client ID is not a secret** (it is
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
   (The Client ID is public and safe to commit. Leave it empty to ship a recommendation-only demo
   with no login.)

### Spotify Development Mode limits
Spotify apps start in **Development Mode**, which only lets **up to 5 Spotify
accounts that you manually add** (Dashboard → *User Management*) log in and save
playlists. This is why the desktop "Save playlist" returned 403 earlier: your own
account just needs to be added there.

For the *general public* to log in and save, Spotify requires **Extended Quota
Mode**. **As of May 15, 2025, Spotify only accepts applications from organizations,
not individuals** (a registered business, a launched service with ≥250k monthly
active users, applied via a company email, ~6-week review). For a personal project
that's effectively unavailable. So realistically:

- **Recommendations:** public, with no login required.
- **One-click Save-to-playlist: you + up to 4 accounts you allowlist** (5 total).
  Public one-click save isn't attainable for a solo dev under Spotify's policy.
- **Everyone else** gets a no-login **"Copy list"** button (paste into a new Spotify
  playlist) and an **Open in Spotify** link on every result.

If you want *any* visitor to get a real playlist without logging in, the only route
is an **owner-account model**: store your own refresh token as a server-side secret
and have a serverless function create public playlists in your account, returning a
shareable link. It sidesteps the 5-user cap (visitors are listeners, not API users)
but every playlist lives under your account. Soundalike does not enable this
behavior by default.

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
