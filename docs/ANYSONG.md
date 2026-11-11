# Any-song (beta)

The search on soundalike.yassin.app (the old `/anysong` page now redirects to the home page) takes any Deezer track,
including one that has never been indexed, and finds
catalog songs that share its **melody**, a **sample/loop** or its overall **sound**. Unseen songs are fetched
and embedded live on a GPU; everything else is a lookup.

| Channel | Model | Finds | Score |
| --- | --- | --- | --- |
| `melody` (Same melody) | CLEWS DVI | covers, interpolations, reused tunes | negative distance |
| `samples` (Samples & loops) | SampleID | shared samples, loops, chopped sections | cosine |
| `sonic` (Sounds like) | MuQ-MuLan | production, instrumentation, mood | cosine |

Scores are only comparable within one channel and one query; the page shows relative bars, not percentages.
Everything runs on 30-second Deezer previews, so a relationship outside the preview can be missed.

The page opens on **Best matches**, one list that merges the melody and samples lists by reciprocal-rank fusion:
a song scores `sum(1 / (5 + rank))` over the two channels' song lists, and is shown as the release found by the
channel that ranked it higher. On held-out benchmarks it finds covers as well as the melody tab and samples better
than the samples tab ([ANYSONG_EVAL.md](ANYSONG_EVAL.md#best-matches)). The channel tabs remain for browsing one
relation. The same list is the default view in the
[Spicetify extension](../integrations/spicetify/README.md), which names the song by title and artist instead of a
Deezer id. Best matches hides confidently detected other-language songs by default (`--languages`, see
[Language labels](#language-labels)); a **Show all languages** switch (`lang=all`) restores the unfiltered ranking,
and the individual melody, samples and sounds tabs are never filtered. A song without a confident label takes its
artist's majority language, a conservative prior from that artist's indexed songs; songs with neither (often
instrumentals) remain visible. A live seed outside the catalog uses the same prior; if it cannot establish a
language, the filter fails open. The filter costs some cross-language matches; the measured trade-offs are in
[ANYSONG_EVAL.md](ANYSONG_EVAL.md#language-gate-impact).

Other artists' versions of the seed's own song (the same title) are limited in Best matches, Same vibe and Sounds
like. One that scores 0.55 or more in the samples list reuses the seed's recording (a re-upload, a sped-up edit, the
same song under another credit) and is dropped. At most three of the rest stay, the best placed first, so covers of
a standard cannot fill the lists. The channel tabs keep them all.

Best matches also weighs the **artist scene** ([`scene.py`](../src/soundalike/anysong/scene.py)). Audio still decides
which songs can appear; the scene layer only arranges them. A candidate's affinity to the seed's artist combines the
same artist, closeness in Deezer's related-artist graph and a shared regional scene (Wikidata countries, mostly
those of the artist's graph neighbours). The first songs of the audio ranking stay on top: at least three, and
further down to the last top-10 song whose melody or samples score is strong enough to suggest a cover or sample.
After that, each slot takes the next **scene neighbour** with audio evidence in some channel (the sounds list counts
here), ordered by affinity times that evidence. When none is left, the audio ranking continues; it lifts scene
neighbours too, but never past a song whose melody or samples score clearly stands out. One artist can place at most
two songs in the top 20. Without metadata for the seed's artist the ranking is the plain fusion. The selection,
safeguards and results are in [ANYSONG_EVAL.md](ANYSONG_EVAL.md#scene-layer-2026-11-04) and, for the current slots,
[ANYSONG_EVAL.md](ANYSONG_EVAL.md#ranking-for-the-lists-users-see-pre-registered-2026-11-09-before-any-held-out-result).

With a **listener ranker** (`--ranker`, [`listener.py`](../src/soundalike/anysong/listener.py)), the slots above give
way to a learned order whenever the seed's artist is known:
- **Candidates.** The channel lists, plus, for each artist in the seed artist's Deezer neighbourhood, its three songs
  closest in sound. These are scored like the sonic channel and gated by the catalog tiers.
- **Model.** A small tree model estimates, for each candidate, whether listeners of the seed's artist also play the
  candidate's artist. It uses the audio scores, the Deezer artist graph (overlap, which artist lists the other, region,
  genre), tempo and whether the language is known. Popularity is deliberately not a feature. The model was trained on
  development seeds with ListenBrainz similar artists as labels.
- **Best matches** keeps its audio head: at least three songs, longer only for melody or samples matches that also
  stand out from the seed's other matches (a z-score of 2 or more among the first 20 songs of their list). The rest
  follow that estimate, the sound score and the melody/samples fusion
  score (z-scores weighted 1, 0.5 and 0.25). After the head, no artist plays twice in a row, and each artist appears
  once in the first 10 places; an artist's later songs wait for the next place that fits (`scene.arrange_artists`).
- **Sounds like** orders the sonic list by its sound score plus twice the estimate.
- **Same vibe** applies its rule to both new orders.
- Songs by the seed's own artist are placed by sound alone.

The selection and results are in
[ANYSONG_EVAL.md](ANYSONG_EVAL.md#listener-aware-ranking-pre-registered-2026-11-09-before-any-held-out-result).

Next to Best matches, **Same vibe** ([`samevibe.py`](../src/soundalike/anysong/samevibe.py)) lists songs a listener
hears as belonging together. It keeps the Best candidates that share the seed's tempo (the same, double or half,
within 6 %) and its overall sound, fills up from the sounds list, and takes one song per artist. Best stays the list
for covers, samples and shared melodies, which often cross genres. On 1,000 fresh test artists, Same vibe had fewer
genre-incoherent results than the Sounds like tab (39 % vs 50 %) and more ListenBrainz-similar artists
([ANYSONG_EVAL.md](ANYSONG_EVAL.md#same-vibe-2026-11-06)).

## Architecture

```
browser ── Cloudflare ── tunnel ── CT 408 Caddy :8080 ─┬─ /api/anysong/*  ─▶ CT 409 :8790  anysong.api (RTX 2070 SUPER)
                                                        └─ everything else ─▶ vercel_gateway (webapp/)
```

- The home page, [`webapp/index.html`](../webapp/index.html) with [`webapp/search.js`](../webapp/search.js), is the
  any-song search: suggestions from `/api/anysong/search`, results from `/api/anysong/similar?id=` in five tabs
  (Best matches, Same vibe, Same melody, Samples & loops, Sounds like). Share links are `/#id=<deezer id>&tab=<tab>`;
  older `#s=<title — artist>` links still resolve by name. Previews are resolved through the existing `/api/preview`.
- `/anysong` redirects to `/` (`redirects` in [`vercel.json`](../webapp/vercel.json), temporary, query kept).
  [`webapp/anysong/index.html`](../webapp/anysong/index.html) is a stub for servers that ignore `redirects`: it
  forwards the query and hash to `/`, where old `?id=` and `#id=` links (including `tab=sonic`, now Sounds like) open
  the same song.
- [`anysong/api.py`](../src/soundalike/anysong/api.py) serves `GET /api/anysong/search?q=&limit=` (Deezer search
  plus `in_catalog`), `GET /api/anysong/similar?id=&k=&lang=same|all&scene=1|0` and `GET /healthz`. With
  `--languages`, `lang=same` is the default and `lang=all` bypasses the Best-matches-only filter; without it, `lang`
  has no effect. `/similar` returns `results.best` followed by one
  unfiltered list per channel. Each channel ranks its top 200 rows and keeps one release per song; `best` fuses
  those song lists (`FUSE_K`, `FUSE_WEIGHTS`), and each of its items carries `ranks`, its 1-based position in each
  fused channel's list. Its `score` is the fused score. When `--languages` is configured, `query.language` reports
  the seed language (or `unknown`), `query.language_mode` reports `same` or `all`, and `query.hidden` counts
  confidently different-language songs among the unfiltered top k. Unknown candidate labels stay in the list; an
  unknown seed language disables filtering. With `--scene`, `query.scene` is `on`, `unknown_artist` (no metadata
  for the seed's artist: plain fusion) or `off` (`scene=0`, or no table), `query.artist_id` is the seed's Deezer
  artist, and a Best item's `ranks` include `sonic` when the sounds list placed it. In a scene list `score` is
  positional (`1 / (5 + position)`), because slots rather than summed scores decide the order. A live seed whose
  artist's related-artist list is not in the table costs one cached Deezer request (`/artist/{id}/related`), made in
  the background while the preview downloads; if it fails, the API uses what the table knows. With `--tempo`
  (a catalog tempo table) the results also hold `vibe`, the Same vibe list (items like `best` without `ranks`,
  positional `score`, language-filtered like `best`), and `query.bpm` gives the seed's tempo (`null` when unknown).
  `--tempo-model` points to the Beat This! checkpoint that estimates the tempo of seeds outside the catalog; without
  it their tempo is unknown and only the sound gate applies. Whenever the sonic channel is served, `sounds` is the
  Sounds like list that the main page and Spotify show: the sonic channel's songs, language-filtered like `best`,
  one song per artist (the scene table's Deezer artist, else the artist name), each with its sonic `score` and its
  catalog `bpm` (`null` when unknown or without `--tempo`). With the scene layer on, songs move up by their artist's
  scene affinity, `1 / (5 + rank) * (1 + 8 * affinity)`; with the listener ranker, they follow z(sound score) + 2
  z(listener estimate) instead. Either way the list draws only from the sonic channel's 200 songs. With the ranker, a
  Best item that came from the seed artist's neighbourhood rather than a channel list has empty `ranks`, and
  `timings_ms.listener` gives the time the ranker took.
- `/similar?title=&artist=&k=` (used by Spotify, which has no Deezer id, and by old `#s=` home-page links) resolves
  the name first.
  Songs match by song key (primary artist plus title without bracketed, dashed or feat. tags), ignoring accents on
  Latin letters, symbols and a leading "The" ("Beyoncé" = "Beyonce", "The Black Eyed Peas" = "Black Eyed Peas").
  Among a song's releases, one with the same version words (`deezer.version_words`: live, remix, acoustic, edit,
  sped up, ...) and the same content tags (bracketed or dashed parts that name other content, such as a medley's
  other songs) wins, so "Song" is never seeded with "Song (X Remix)" or a medley when the original exists, while
  "Song - Remastered" may use "Song". A catalog release with the requested version answers without any Deezer
  request (only indexed releases count: the rest usually had no preview). Otherwise Deezer is searched for the title
  and artist as given and, when that finds no release of the song, for the normalized song key, which finds
  symbol-named artists. The first of 10 hits with a preview, that version and those tags is ranked live. Failing
  that, an indexed release with that version, a hit with that version, the song's best indexed release (untagged
  first, then lowest id) or else its first search hit stands in, and no release at all is a 404. With `isrc=` (the
  Spotify extension reads the seed's ISRC from Spotify), an indexed release with the requested version answers
  first; then Deezer's ISRC lookup (cached), which also finds tracks that Deezer's search hides in the server's
  country; then the name lookup. `query.id`
  and `query.title` in the response name the resolved track. The service sends `Access-Control-Allow-Origin: *`
  (`--cors-origin`) so the Spotify desktop client can read these public, credential-free GET responses.
- [`anysong/serve.py`](../src/soundalike/anysong/serve.py) holds the two-stage index: a PCA projection of every
  window in VRAM ranks candidates, then the full-dimensional windows (memory-mapped `full.npy`) rerank them.
  A catalog song is queried with its stored windows (no audio work); an unseen song is downloaded, decoded and run
  through all three encoders, and the result is cached in memory.
- Rate limits are token buckets per client: `/similar` 20/min (burst 5), `/search` 60/min (burst 15). Cached
  `/similar` results are free, and a name lookup the catalog cannot answer spends one `/search` request (when that
  budget is spent, a song with another catalog release uses it instead of failing). Clients are
  keyed by `--client-ip-header` (`CF-Connecting-IP`) with IPv6 grouped per /64; only enable that flag while the
  port is reachable solely through the proxy chain. Excess concurrent live work returns 503 (`--max-pending`),
  which the page retries with backoff.

Errors are `{ok:false, error}` with 400 (bad parameters), 404 (unknown track or song / no preview), 422
(undecodable preview), 429 (rate limit), 502 (Deezer), 503 (busy) or 504 (timed out waiting for an identical
request).

## Growing the catalog

[`anysong.crawl`](../src/soundalike/anysong/crawl.py) adds songs along Deezer's related-artist graph, starting from
the scene table, and downloads their previews into the same cache as the catalog's:

```sh
python -m soundalike.anysong.crawl run --scene index/scene.npz --meta index/catalog_meta.npz \
  --db /var/lib/soundalike-crawl/crawl.sqlite --previews /data/previews --target 730000
python -m soundalike.anysong.crawl status --db /var/lib/soundalike-crawl/crawl.sqlite
python -m soundalike.anysong.crawl export --db /var/lib/soundalike-crawl/crawl.sqlite --out /data/crawl/export
```

Catalog artists are crawled first: they get topped up with popular songs the catalog lacks, and their related
lists bring fan counts for the artists around them. The rest of the frontier goes most-followed first, skipping
artists with fewer than 100 fans. Ordering by link count instead fills up on dense scenes; in a test, 10 of the
first 15 artists were classical performers who all list each other. An artist ends up with up to 8 songs at 100
fans, 12 at 1,000, 16 at 10,000 and up to 25, counting those already in the catalog, taken from its own top songs.
A song needs a playable preview, one to fifteen
minutes of audio and a Deezer rank of at least 10,000. Karaoke, tribute, sped-up, slowed, 8D, nightcore,
frequency, instrumental and a cappella uploads are left out, and so are karaoke and sleep or study-music
channels. Live, edit, version and remaster variants count as the original song (remixes stay separate), and a
catalogue-numbered classical work is kept for its first performer only.

Each artist costs two API requests (4 per second by default, of Deezer's ~10 per IP, leaving room for the live
API). Signed preview links expire in about fifteen minutes, so previews download straight away. The preview CDN
rate-limits each IP at roughly ten downloads a second, so the pace adapts: it starts at 12 per second, slows by a
fifth after each burst of HTTP 429s, and recovers while quiet. At about 470 KB per preview, 730k songs take
roughly a day and 340 GB. Everything is in SQLite, so stopping and rerunning resumes; failed downloads are retried.

`export` writes the new songs as `catalog_meta.npz` (same layout as the API's) and `ids.txt` for `embed_run`.
It also writes `deezer_tracks.jsonl` and `deezer_related.jsonl` (related lists with names and fan counts for
every crawled artist), which go into the scene data folder before the next `scene` build.

**Learning as it goes.** The API's `--learn-queue` file gets one line per song it answered live, meaning a song
not yet in the catalog: `{"id": <Deezer track>, "artist_id": <Deezer artist>, "day": "YYYY-MM-DD"}`.
- Nothing about the request is written: no address, title or time of day.
- Each song is written once per process, with at most 5,000 lines a day, and a failed write never fails the
  request.

`crawl run --learn-queue <file> --keep-learning` reads new lines every 10 minutes:
- The looked-up song is added itself, with the usual filters except the rank minimum.
- Its artist goes ahead of the bulk crawl. If that artist was crawled already, its related artists go ahead
  instead.
- Crawling a looked-up artist queues its related artists (with at least 100 fans) next, so the song's
  neighbourhood follows.

A Deezer failure leaves the rest of the queue for the next pass. Once the bulk target is reached, the crawl keeps
running for looked-up songs only. They are embedded with the rest of the crawl's songs and join the catalog at the
next index rebuild. `crawl status` reports learned songs and looked-up artists still waiting.

**Catalog tiers.** Songs the crawl added are recommended only for seeds by their own artist or by an artist whose
own Deezer related list names theirs; the songs the catalog held before (the core, listed in `core_ids.npy`) are
always candidates. Unrestricted, the grown catalog failed its go-live gate: crawled songs by artists outside the
seed's scene filled about 45 % of list rows, and listeners almost never played them (see
[ANYSONG_EVAL.md](ANYSONG_EVAL.md), "Catalog growth"). [`anysong.tiers`](../src/soundalike/anysong/tiers.py) turns
the rule into one row mask per channel, applied before the index picks its exact-scored candidates, so lists keep
their depth. The related lists come from the scene table, so the search never waits for Deezer: a seed whose artist
has no list there gets core songs and its own artist's, never fewer candidates than before the crawl (Deezer's live
list still feeds the scene layer). Every list is gated, the channel lists included, and `eval_index --core-ids`
measures the gated catalog. `/healthz` reports core and discovered rows per channel.

## Building an index

Extract window embeddings for the frozen catalog with
[`embed_run`](../src/soundalike/anysong/embed_run.py) (`clews_dvi`, `sampleid`, `muq_mulan`; shards are resumable
`part-NNNN.npz` files with `ids`, `track`, `win`), then:

```sh
python -m soundalike.anysong.index_build --emb emb_full --out index_full \
  --ids catalog_ids.txt --dims melody=256 samples=384 sonic=128 \
  --meta catalog_meta.npz --max-song-tracks 100
```

Each channel directory holds `basis.npy`, `proj.npy` (fp16, or int8 / packed 4-bit with `proj_scale.npy`),
`full.npy`, `ids.npy`, `norms.npy` and `meta.json`
(retained PCA energy, candidate counts). `--max-song-tracks` leaves out any song key (as the API groups
results) with more than that many tracks. In the frozen catalog this removes only 2,031 *A State Of Trance*
radio-show segments, which otherwise surface as one "song" in many unrelated result lists (see
[ANYSONG_EVAL.md](ANYSONG_EVAL.md)). `meta.json` records what was dropped. Then add the hub penalty:

```sh
python -m soundalike.anysong.hub_build --index index_full --meta catalog_meta.npz
```

This writes `hub.npy` per channel and records `hub` in `meta.json`. Serving subtracts `alpha * h(track)`
([CSLS](https://arxiv.org/abs/1710.04087)), where `h` is the track's mean similarity to its 10 nearest other
songs. That stops a few "sounds like everything" tracks from crowding every result list. Defaults are `alpha =
0.5` for samples and sonic, and 0 (off) for melody, because a family of songs sharing one melody is a genuine
dense cluster that the penalty would push down. The reasons are in [ANYSONG_EVAL.md](ANYSONG_EVAL.md).
`--hub-alpha` on `api` and `eval_index` overrides the recorded value; 0 disables it.

Checks before serving an index:

- `python -m soundalike.anysong.eval_index --index index_full --meta catalog_meta.npz --bench bench --exact 256
  --hubs 2000 --out eval.json` scores every benchmark split per channel, as the page's "any tab" and as the fused
  `best` list. It also
  compares the two-stage ranking with brute force (`--exact`) and measures hubness (`--hubs`). Choose settings
  on `development` splits; report `held_out`/`test`/`validation`. Add `--languages languages.npz` to also report
  `best_same_language` beside unfiltered `best`; `language_gate` records labeled-seed coverage and the number of
  other-language results in the unfiltered top 20. Add `--scene scene.npz` to also report `best_scene` (and
  `best_scene_same_language`), the Best list as served with the scene layer; `--scene-params` overrides its
  settings as JSON.
- `python -m soundalike.anysong.serve_check --index <dir> --previews <dir> --bench <dir> --out report.json`
  re-embeds a sample of indexed tracks live to check parity with the offline index, and reports VRAM, latency
  and benchmark recall.
- `python -m soundalike.anysong.robustness --n 200` takes catalog previews and edits them (sped up, slowed,
  ±2 semitones, tempo change, noise, phone band, a 10 s excerpt). It embeds each edit live, like an unseen
  query, and checks whether the original song still ranks first.

On the 8 GB 2070 SUPER, stop the API before running `serve_check` or `robustness`. The full 269k index fits:
5.4 GB of VRAM loaded, 6.3 GB peak.

Stage 1 is the part that has to fit in VRAM: about 3.9 GB of the API's footprint is models and working memory, so
the projections get roughly 3.8 GB. In fp16 at the default dimensions they take 8.4 GB per million tracks, so a
bigger catalog stores them as integers with one scale per window:

```sh
python -m soundalike.anysong.index_build ... --dims melody=256 samples=384 sonic=128 \
  --proj-bits melody=8 samples=4 sonic=8
```

That is 2.7 GB per million tracks (melody 0.77, samples 1.54, sound 0.38), plus 0.06 GB of scales. On the live
269k index, compared with an exact brute-force search over 300 test queries per channel, the served top 10 and
top 20 were identical in every channel. At top 100, one melody query and one sound query differed by one song
([ANYSONG_EVAL.md](ANYSONG_EVAL.md#compressed-stage-1-2026-11-07)). Re-run that comparison at the new size before
serving it.

Integer projections are exact in fp16, so `api --half-stage1 <channels>` (also on `serve_check` and `eval_index`)
runs a channel's stage 1 as an fp16 GEMM against the query projection scaled to |x| ≤ 1. On the 2070 SUPER that
made the 4-bit samples stage 1 23 % faster (196 against 256 ms per million tracks) but the 8-bit melody and sound
stages 6 % and 14 % slower, so the service enables it for samples only. Candidates are still re-scored exactly.

## Language labels

`python -m soundalike.anysong.languages` labels cached catalog previews with the pinned Whisper large-v3-turbo
revision used by the V4 language probe. It saves resumable `part-*.npz` shards before producing `languages.npz`;
the compact final file keeps the top five language probabilities per ID so the calibrated confidence/margin
thresholds can be reapplied without decoding audio. The serving thresholds are confidence ≥ 0.8 and margin ≥ 0.5.
Use `--fp16-accumulation` only on supported RTX 40/50 GPUs; it is faster but can change a marginal `unknown` to a
language decision. Decode failures are recorded and make the run fail rather than producing a success-shaped,
partially labelled catalog.

Example, with the project environment and local Whisper files available:

```powershell
python -m soundalike.anysong.languages `
  --ids C:/anysong/index/melody/ids.npy `
  --previews C:/anysong/previews `
  --out C:/anysong/languages `
  --model-dir C:/models/whisper-large-v3-turbo `
  --fp16-accumulation
```

Catalog seeds use their own label. A live seed outside the catalog uses a conservative artist-majority prior
(≥2 confidently labelled recordings and ≥80% agreement); if no prior is available, the filter fails open. Listed
songs without a confident label take the same prior, and stay when it is unknown. The lists users see (Best
matches, Same vibe, Sounds like) are filtered; the three channel lists are unchanged. `lang=all` returns the same
unfiltered Best ranking and still reports the other-language count in `query.hidden`.

## Scene table

The scene layer reads one table, `scene.npz`: each catalog track's primary Deezer artist, a symmetric related-artist
graph, and per-artist country and genre weights. Build it from collected sources:

```sh
python -m soundalike.anysong.scene_sources --ids index_full/samples/ids.npy \
  --catalog-artists catalog_artist_ids.json --out scene_data
python -m soundalike.anysong.scene --data scene_data --catalog-artists catalog_artist_ids.json --out scene.npz
```

`scene_sources` is resumable. It looks up artists for catalog tracks without one (Deezer `/track/{id}`), Deezer's
related-artist list for every catalog artist (`/artist/{id}/related`, up to 20 each), and Wikidata (CC0) facts
for catalog and related artists via the Deezer artist id property P2722: citizenship, origin, formation and
birth countries, and genres with their parent genres. It keeps to 5 Deezer requests per second by default
(Deezer allows about 10 per IP and the live API shares the address) and sends one Wikidata query at a time. Deezer's
API terms limit use to personal, non-commercial applications, as for the rest of this project.

`scene` symmetrises the graph (an artist's neighbourhood is the artist, its related artists and the artists that
list it), and also keeps each crawled artist's own related list in Deezer's order for the catalog tiers. An artist's region is 70 % the stated countries of its graph neighbours and 30 % its own, because
citizenship or birthplace alone misplaces artists (21 Savage, born in London, belongs to the Atlanta scene).
Artists without stated countries or genres inherit their neighbours' at 80 % weight when at least two neighbours
know them. Genres are IDF-weighted and include Wikidata parent genres at half weight; they are stored but weighted 0
in the shipped setting. ListenBrainz co-listening is used only to validate the layer, never as a feature.
Rebuild the table about monthly: related lists drift and new artists appear (artists missing from the table are
looked up live).

## Tempo table

Same vibe compares tempos from [Beat This!](https://github.com/CPJKU/beat_this) (`final0`, MIT licence for code
and weights; `pip install beat-this==1.1.0`). `anysong.tempo` decodes each preview with soundfile, resamples to
22,050 Hz with soxr, tracks beats on one 30-second chunk in fp16 and stores `60 / median(inter-beat interval)` (NaN
when fewer than four beats are found):

```sh
python -m soundalike.anysong.tempo --ids index_full/samples/ids.npy --previews previews \
  --model ckpt/beat_this/final0.ckpt --out tempo.npz
```

The full catalog took 0.7 GPU-hours on the RTX 5080; 266,945 of 269,278 tracks have a tempo. The API runs the same
model on the previews of seeds outside the catalog (`--tempo-model`).

## Listener ranker

`ranker.npz` holds the listener model as plain arrays: 100 trees of depth at most 6, about 120 KB. Serving
evaluates it with numpy (`listener.Ranker`), with no dependency on scikit-learn's pickles. It is trained in the
research lab (`grow/qa/ranker_dev.py`) on rows that come from `AnySong._listener` itself, the code the server runs,
on cached searches of development seeds:
- `rows` captures each seed's candidate features;
- `cv` compares out-of-fold variants with the served lists;
- `train` fits on all development rows and writes `listener.export(model)`.

Labels are ListenBrainz similar artists; they are never a feature. The model depends on the scene table and the sonic
index it was trained with, so it belongs with the index's tables. Retrain it when they change. Per seed the extra
work is about 650 candidates and up to a few thousand neighbour songs scored like the sonic channel. That took a median
of 16–31 ms on the RTX 5080. `/healthz` reports the model's description under `ranker`.

## Deployment (homelab)

GPU host CT 409 (`/opt/soundalike`): `src/`, `venv/`, checkpoints under `ckpt/`, upstream code under `ext/`,
offline Hugging Face cache in `hf/`, and indexes with `index/current` → the live index. Each index directory also
holds the catalog tables it was built with: `catalog_meta.npz` for titles, `languages.npz` for language labels
(`--languages`), `scene.npz` for the scene layer (`--scene`), `core_ids.npy` for the catalog tiers (`--core-ids`;
for an index without crawled songs, all of its ids) and `tempo.npz` (with `ckpt/beat_this/final0.ckpt`)
for Same vibe (`--tempo`, `--tempo-model`; about 160 MiB of VRAM, and 30 ms of GPU time per unseen seed), and
`ranker.npz`, the [listener ranker](#listener-ranker) trained on these tables (`--ranker`). One
symlink swap therefore switches the index and its tables together. Grown indexes live on
`/opt/soundalike/index-store`, a 96 GB NVMe volume (`mp1`, `backup=0`). The nightly vzdump to the bulk HDD skips it,
because each index is about 40 GB and can be rebuilt from its embeddings. The first full index stays at
`index/full-20261101` on the root disk, with its own `core_ids.npy`, as the rollback. Since 2026-11-09,
`index/current` → `index-store/grown-20261109`: 999,293 songs, the 269,278 core songs plus 730,015 crawled ones,
served with the catalog tiers and samples' stage 1 in fp16 (`--half-stage1 samples`). The previous grown index,
`index-store/grown-20261108` (889,277 songs), stays as the rollback. The service is
[`deploy/homelab/soundalike-anysong.service`](../deploy/homelab/soundalike-anysong.service) (user `anysong`,
`MemoryMax=6G`, enabled at boot; startup loads and warms the models and index). The process itself uses about
1.5 GB of RAM; the rest of the cap is page cache for the memory-mapped windows, which the kernel trims.

Swapping the index:

```sh
scp -r index_new root@ct409:/opt/soundalike/index-store/NAME          # parallel scp per file is fastest
cd /opt/soundalike/index-store/NAME && sha256sum -c ../NAME.sha256 && chmod -R go-w .
cp catalog_meta.npz languages.npz scene.npz tempo.npz core_ids.npy ranker.npz /opt/soundalike/index-store/NAME/   # its tables
systemctl stop soundalike-anysong
ln -sfn /opt/soundalike/index-store/NAME /opt/soundalike/index/current
systemctl start soundalike-anysong && journalctl -u soundalike-anysong -f   # wait for "ready in ..."
curl -s 192.168.1.47:8790/healthz
```

Keep the previous index directory until the new one has been checked; rolling back is the same symlink swap.

Public routing lives on CT 408: `"proxy_paths": {"/api/anysong/*": "192.168.1.47:8790"}` in the soundalike entry of
`/etc/web/sites.json`, applied with `web-caddy-gen`. The API then listens on the LAN address
(`ANYSONG_HOST=192.168.1.47`), and the Proxmox firewall for CT 409 only admits port 8790 from CT 408, which keeps
`CF-Connecting-IP` trustworthy. The page itself ships with the normal `main` deploy.

Catalog growth runs on the same host as
[`deploy/homelab/soundalike-crawl.service`](../deploy/homelab/soundalike-crawl.service), from its own copy of the
package in `/opt/soundalike/crawl-src` so API deploys never change it mid-crawl. Previews go to `/data/previews` on
the bulk HDD. The state database is `/var/lib/soundalike-crawl/crawl.sqlite` on the NVMe rootfs: on the HDD, its
small random writes made up more than half of the disk's write operations (57 writes/s before the move, 23 after, at
the same write volume), which left little room for reading previews while the crawl ran. Progress is in the
journal or `crawl status`. The API writes its learn queue to `/var/lib/anysong/learn-queue.jsonl`, and the crawl
reads it with `--keep-learning`, so it keeps running after the bulk target.

## Limitations

- Coverage is limited to the indexed catalog: 999,293 songs since 2026-11-09 (269,278 core tracks plus the crawl's
  songs, which only reach seeds in their artist's scene). Famous sources missing from it cannot be returned, however
  good the match.
- Preview-only audio: shared material outside the 30-second clip is invisible.
- Benchmarks are small for some relations (see the held-out notes in the README); treat the channels as beta.
