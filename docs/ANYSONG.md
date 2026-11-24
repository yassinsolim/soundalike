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

The API's **Best matches** list (`best`) merges the melody and samples lists by reciprocal-rank fusion:
a song scores `sum(1 / (5 + rank))` over the two channels' song lists, and is shown as the release found by the
channel that ranked it higher. On held-out benchmarks it finds covers as well as the melody tab and samples better
than the samples tab ([ANYSONG_EVAL.md](ANYSONG_EVAL.md#best-matches)). The page and the
[Spicetify extension](../integrations/spicetify/README.md), which names the song by title and artist instead of a
Deezer id, show three tabs: **Best matches** (`main`: likely covers, samples and shared melodies first, then the songs
that sound most alike), **Listeners also play** (`listeners`) and **Melody match** (`best`, the fused list above with
its listener picks); all are described with the API below. Best
matches hides confidently detected other-language songs by default (`--languages`, see
[Language labels](#language-labels)); a **Show all languages** switch (`lang=all`) restores the unfiltered ranking,
and the individual melody, samples and sonic lists are never filtered. A song without a confident label takes its
artist's majority language, a conservative prior from that artist's indexed songs; songs with neither (often
instrumentals) remain visible. A live seed outside the catalog uses the same prior; if it cannot establish a
language, the filter fails open. The filter costs some cross-language matches; the measured trade-offs are in
[ANYSONG_EVAL.md](ANYSONG_EVAL.md#language-gate-impact).

The seed's own song never appears in a list (`dedupe=1`, the default): any release by the seed's artist whose title
is the seed's without version tags, also when the names are spelled differently in accents, width forms or a
leading "The". A seed outside the catalog is named by Deezer today and the catalog's releases as they were when
indexed, so "Maria También" hides "Maria Tambien (Live at Sydney Opera House)". Another song by the seed's artist
may still lead Best matches (`main`): its audio head is for likely covers, samples and shared melodies whoever made
them (Khruangbin's dub version "Mary Always" for "Maria También"); the consensus body after the head and Listeners
also play hold other artists only.

Other artists' versions of the seed's own song (the same title) are limited in Best matches, Same vibe and Sounds
like. One that scores 0.55 or more in the samples list reuses the seed's recording (a re-upload, a sped-up edit, the
same song under another credit) and is dropped. At most three of the rest stay, the best placed first, so covers of
a standard cannot fill the lists. The channel lists keep them all.

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

**Same vibe** ([`samevibe.py`](../src/soundalike/anysong/samevibe.py)), served as `vibe`, lists songs a listener
hears as belonging together. It keeps the Best candidates that share the seed's tempo (the same, double or half,
within 6 %) and its overall sound, fills up from the sounds list, and takes one song per artist. Best stays the list
for covers, samples and shared melodies, which often cross genres. On 1,000 fresh test artists, Same vibe had fewer
genre-incoherent results than the Sounds like tab (39 % vs 50 %) and more ListenBrainz-similar artists
([ANYSONG_EVAL.md](ANYSONG_EVAL.md#same-vibe-2026-11-06)).

## Architecture

```
browser ── Cloudflare ── tunnel ── CT 408 Caddy :8080 ─┬─ /api/anysong/*  ─▶ CT 409 :8790  anysong.api (RTX 2070 SUPER)
                                                        └─ everything else ─▶ web gateway (webapp/)
```

- The home page, [`webapp/index.html`](../webapp/index.html) with [`webapp/search.js`](../webapp/search.js), is the
  any-song search: suggestions from `/api/anysong/search`, results from `/api/anysong/similar?id=` in the Spotify
  extension's three tabs. **Best matches** reads `main`; from a server without it, the page shows `best` and marks its
  rows with `basis: "listeners"` **Listeners also play**. **Listeners also play** reads `listeners`, and **Melody
  match** reads `best`. Requests carry
  `policy=main-listeners-v1`, which the API ignores, so HTTP-cached answers from the five-tab page are not reused.
  Share links are `/#id=<deezer id>&tab=best|listeners|melody`; links to the old tabs (`vibe`, `samples`,
  `sounds`, `sonic`) and unknown tabs open Best matches, old `tab=melody` links open Melody match, and older
  `#s=<title — artist>` links still resolve by name. The tab chosen last is kept in `localStorage`
  (`soundalike:tab:v1`) for the next visit; a link's tab wins for that visit.
  Previews are resolved through the existing `/api/preview`. The **Audio only** switch next to the language switch
  asks with `scene=0` and is remembered in `localStorage` (`soundalike:audio-only:v1`). With it on, or for an artist
  without listener data, there is no `listeners` list, and Listeners also play says why.
- `/anysong` redirects to `/` (`redirects` in [`site.json`](../webapp/site.json), temporary, query kept).
  [`webapp/anysong/index.html`](../webapp/anysong/index.html) is a stub for servers that ignore `redirects`: it
  forwards the query and hash to `/`, where old `?id=` and `#id=` links (including `tab=sonic`) open the same song
  on Best matches.
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
  Sounds like list: the sonic channel's songs, language-filtered like `best`,
  one song per artist (the scene table's Deezer artist, else the artist name), each with its sonic `score` and its
  catalog `bpm` (`null` when unknown or without `--tempo`). With the scene layer on, songs move up by their artist's
  scene affinity, `1 / (5 + rank) * (1 + 8 * affinity)`; with the listener ranker, they follow z(sound score) + 2
  z(listener estimate) instead. Either way the list draws only from the sonic channel's 200 songs. With the ranker, a
  Best item that came from the seed artist's neighbourhood rather than a channel list has empty `ranks`, each Best
  item has `basis`: `audio` for the audio head (likely covers, samples and shared melodies) and `listeners` for the
  songs placed after it by the listener blend, and `timings_ms.listener` gives the time the ranker took. Without the
  ranker (or with `scene=0`) Best items have no `basis`; the list is audio only. Whenever `sounds` is served, so is
  `main`, the Best matches list the page and Spotify show: the audio head (only songs with cover or sample evidence;
  `basis` `audio`), then the sonic channel's songs in Sounds like's order (`basis` `sound`), one song per artist, each
  with `ranks` and a positional `score`. With the ranker and the scene layer on, `listeners` is the Listeners also
  play list: Best's candidates by artists other than the seed's, ordered by z(listener estimate) + 0.25 z(sound
  score) (`LISTENERS_SOUND_WEIGHT`), one song per artist, `basis` `listeners`. Without the ranker, with `scene=0` or
  for an unknown artist there is no `listeners`. Items whose sound score is known carry `match`, the sound match from
  0 to 100 (`api.sound_match`). [ANYSONG_EVAL.md](ANYSONG_EVAL.md#one-main-list-and-listeners-also-play-pre-registered-2026-11-11-before-any-held-out-result)
  has the evaluation. With `--consensus-model` (the served configuration since 2026-11-22, "v12") the part of `main`
  after the audio head is the consensus body instead: other artists' songs ordered by a small model on five
  catalog-wide signals (`basis` `consensus`, `ranks_all` with each signal's rank, and a `match` that scores the
  body's ranking rather than the sound); `scene=0` still gives the sound order
  ([evaluation](ANYSONG_EVAL.md#consensus-body-of-the-main-list-2026-11-20-served-since-2026-11-22)).
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
  first, then lowest id) or else its first search hit stands in. When neither the catalog nor those searches know
  the song under the names as given, it is looked for under another credit or naming
  ([`anysong/lookup.py`](../src/soundalike/anysong/lookup.py)), on evidence only:
  - *the title* is the same key in any naming Deezer holds for the track: its default one, the one it gives with
    `Accept-Language` set to the language of the request's script ("Zankyosanka" is 残響散歌 for `ja`), or a bracketed
    part in another script ("Gangnam Style (강남스타일)"); Arabic letters written two ways are folded;
  - *the artist* is the track's credited artist or one name of that credit, or one of its contributors (Deezer's
    track details: "Ella Baila Sola" is credited to Eslabon Armado first and Peso Pluma second), or the Deezer
    artist that the artist search finds for the name (周杰倫 is Jay Chou), or spelled almost like the credit ("Pink"
    for "P!nk", "ACDC" for "AC/DC").

  The steps, each only if the one before found nothing: the hits already returned, the same search in the request
  script's language, the contributors of the three most popular hits with the title, Deezer's artist search, and,
  for a title outside the Latin script, the top tracks of those artists in that language. Last, a title one typing
  slip off (similarity 0.85 or more, at least six characters, the same numbers) counts when the artist evidence is
  better than a close spelling. Among several, the same title wins, then the better artist evidence, then Deezer's
  rank. The song found is then the catalog's release of it when there is one (under Deezer's own credit and
  naming), chosen as above. Anything less is a 404 as before: a known title by an artist who is not on the track
  is never answered with someone else's song. This costs up to about ten more Deezer requests (1 to 3 s) for a
  name that would otherwise be a 404, none for a name that is found, and the answers are cached for an hour.
  Those requests share the one Deezer limiter with every other lookup and the crawl, so each of them is paid from
  the client's `/search` budget like the first search: a client out of budget gets a 429 part-way (what was read
  stays cached, so asking again goes on from there), and no client can put more than its 60 requests a minute on
  Deezer this way. A name that was not found is remembered for 15 minutes (`RESOLVE_MISS_TTL`): asked again it is
  a 404 without any Deezer request and at no cost to the budget.
  What Deezer alone cannot settle stays a 404: a title in a script for which Deezer holds no naming (BTS "봄날",
  filed as "Spring Day" only) or an artist typed in another script than Deezer's without any shared naming
  ("Kino" for Кино). On 2026-11-23, of 81 real requests in many scripts 45 resolved before and 72 after, none
  changed, and 10 requests for songs that do not exist stayed 404 (`tests/test_anysong_lookup.py`, on recorded
  Deezer answers). With `isrc=` (the
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
  `/similar` results are free, and a name lookup the catalog cannot answer spends one `/search` request, plus one
  for each further Deezer request when the name is not found as given (when that budget is spent, a song with
  another catalog release uses it instead of failing). A `POST` gets a 404 unless `--excerpts` is on; its body is
  not read (up to 2 MiB is skimmed so the client sees the answer; after more the connection closes). Clients are
  keyed by `--client-ip-header` (`CF-Connecting-IP`) with IPv6 grouped per /64; only enable that flag while the
  port is reachable solely through the proxy chain. Excess concurrent live work returns 503 (`--max-pending`),
  which the page retries with backoff.

Errors are `{ok:false, error}` with 400 (bad parameters), 404 (unknown track or song / no preview), 422
(undecodable preview), 429 (rate limit), 502 (Deezer), 503 (busy) or 504 (timed out waiting for an identical
request).

## Mobile API

For phone clients (the iOS share extension and Shortcut, the Android share target, the phone page `/find`): give it
the link a listener shared from Spotify or Apple Music and get soundalikes back with links that open in the same
app. No login and no key. Both endpoints are public `GET`s with `Access-Control-Allow-Origin: *`, served by the
website's functions ([`webapp/server/mobile.js`](../webapp/server/mobile.js), routed in
[`site.json`](../webapp/site.json)); the ranking is the engine's `results.main` above.

### `GET /api/mobile/similar`

Name the song in one of these ways. A client that knows more may send several: `url`, then `spotify_id`, then
`apple_id` wins. An extra `isrc` is passed to the engine when the link itself has none, and an extra `title` +
`artist` is used when the link or id cannot be read (Spotify or Apple did not answer, or do not know it). `isrc`
together with `title` + `artist` uses the name as given and passes the ISRC to the engine.

| Parameter | Value |
|---|---|
| `url` | The shared link, or share-sheet text containing it: `https://open.spotify.com/track/<id>` (also `/intl-xx/track/`, any `?si=` query), `spotify:track:<id>`, `https://spotify.link/…` and `https://spotify.app.link/…` short links, `https://music.apple.com/<storefront>/album/<slug>/<albumId>?i=<songId>`, `https://music.apple.com/<storefront>/song/<slug>/<songId>` |
| `spotify_id` | A 22-character Spotify track id |
| `apple_id` | An Apple Music song id (pass `storefront` with it) |
| `isrc` | An ISRC, with or without hyphens |
| `title` + `artist` | Both, 1 to 300 characters each |

Optional: `platform=spotify\|apple\|both` (default: the platform of the link, otherwise `both`), `k` (1 to 50,
default 20), `lang=same\|all` (default `same`: Best matches in the seed's language), `list=main\|best\|listeners`
(default `main`; `best` is the fused melody and samples list the site calls Melody match, `listeners` is Listeners
also play; a list the answer lacks falls back to `main`, and `engine.list` names the one served; `/find` passes
`?list=` on), `part=<start>-<end>` (seconds of the song's 30-second preview, at least 10 apart, at most 30: match
that part instead of the whole preview; see [A part of a song](#a-part-of-a-song)), `storefront` (two-letter Apple
Music country; default: the link's storefront, then the listener's country, then `us`).

```json
{
  "ok": true,
  "query": {
    "title": "Du hast", "artist": "Rammstein", "isrc": "DED579704010",
    "artwork": "https://image-cdn-ak.spotifycdn.com/image/…", "platform": "spotify",
    "source_url": "https://open.spotify.com/track/5awDvzxWfd53SSrsRZ8pXO",
    "spotify_id": "5awDvzxWfd53SSrsRZ8pXO", "language": "de", "storefront": "us", "deezer_id": 630595142,
    "part": null, "parts": { "min_s": 10, "max_s": 30, "preview_s": 30, "song_offset_s": 48.0 }
  },
  "results": [
    {
      "title": "Gekreuzigt 2006", "artist": "OOMPH!", "match": 95, "deezer_id": 567060,
      "isrc": "DEC760600298", "artwork": "https://cdn-images.dzcdn.net/…/250x250-000000-80-0-0.jpg",
      "preview_url": "https://cdnt-preview.dzcdn.net/….mp3?hdnea=…",
      "spotify_url": "https://open.spotify.com/track/4dpWUmLyYPo4vMk5FTMnMv", "spotify_exact": true,
      "spotify_kind": "track", "spotify_id": "4dpWUmLyYPo4vMk5FTMnMv",
      "apple_music_url": "https://music.apple.com/us/search?term=OOMPH!%20Gekreuzigt%202006", "apple_exact": false
    }
  ],
  "engine": { "list": "main", "k": 20, "lang": "same", "source": "catalog",
              "matched_title": "Du hast", "matched_artist": "Rammstein", "hidden": 0 },
  "links": { "spotify_exact": 13, "spotify_isrc": 7, "apple_exact": 0, "spotify_lookup": true,
             "spotify_source": "metadata", "apple_eager_rows": 5 }
}
```

- `query`: the shared song. `artist` lists every artist; `isrc` and `artwork` are `null` when unknown; `platform`
  is the app the result links are meant for; `spotify_id` is the seed's Spotify track (from the link, otherwise
  looked up like a result's; `null` when unknown or when `platform` is `apple`); `source_url` is the song's own
  page (`null` for `isrc` and
  `title`+`artist` seeds); `language` is the engine's label (`unknown` possible); `deezer_id` is the recording the
  engine ranked from (`engine.matched_title` / `matched_artist` name it, and may be another release of the song).
  `parts` is `null` from an engine that cannot match a part of the preview; otherwise it gives the shortest and
  longest part and the preview's length in seconds, and `song_offset_s`, where the preview starts in the song
  (`null` when unknown). `part` is `{"start", "end"}` when the lists are for a part, and `null` otherwise: also
  when `part=` was sent to an engine without parts, which answers for the whole preview. A client shows its part
  control only when `parts` is not `null`.
- `results`: in rank order. `match` is the sound match from 0 to 100 (`null` when the engine has none). `isrc`,
  `artwork` and `preview_url` come from Deezer and are `null` when Deezer did not answer in time. `preview_url` is a
  30-second MP3 whose address **expires about 15 minutes after the answer was built**: play it soon, and on a
  playback error get a new one from `GET /api/preview?id=<deezer_id>` (from a browser, the website's own origin
  only) or by asking again.
- Every result has both links, and both always open something. `spotify_exact` / `apple_exact` are `true` only
  when the link is the song's own page (`open.spotify.com/track/<id>`, `music.apple.com/…?i=<id>`); `false` means
  a search link. A client with a better lookup of its own (MusicKit by ISRC, for example) should use it instead of
  a link marked `false`.
  - **Spotify**: `spotify_kind` says which of three links `spotify_url` is, and `spotify_id` is the track id or
    `null`.
    - `track`: `https://open.spotify.com/track/<spotify_id>`. On a phone with Spotify this opens the song and
      starts playing it (as does `spotify:track:<spotify_id>`). The id comes from one of two places
      (`links.spotify_source`): `credentials`, Spotify's own search by ISRC, then by title and artist, when
      `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` are set on the server (client-credentials flow, no user);
      otherwise `metadata`, described under [Spotify tracks without credentials](#spotify-tracks-without-credentials).
      A `metadata` id is matched on artist and title, so it **can be another release of the same song** (the
      album cut instead of the single, a remaster) than the recording the engine ranked; `isrc` still names that
      one.
    - `isrc_search`: `https://open.spotify.com/search/isrc%3A<ISRC>`, a search that lists exactly that recording,
      used when no track id was found but the ISRC is known. The listener taps once more to play it.
    - `search`: a search for artist and title, only when Deezer gave no ISRC in time.

    `links.spotify_exact` and `links.spotify_isrc` count the first two kinds.
  - **Apple Music**: matched through the iTunes Search API, which allows about 20 calls a minute for the whole
    server. So only the first `links.apple_eager_rows` results (5) are matched up front, and only when `platform`
    is `apple` or `both`; results matched earlier come from memory for free. The rest carry search links: call
    `/api/mobile/link` for a result when the listener taps it or it scrolls into view.
- `engine`: where the list came from (`list` is `main`, or `best` from an engine without it; `hidden` counts
  songs left out for their language).
- Caching: `Cache-Control: public, max-age=300` (`private` when the storefront was guessed from the listener's
  country). The server remembers seeds for a day, engine answers for 10 minutes and link matches for a week, so a
  repeated request is answered in a few milliseconds. A first request takes the engine's time (about 0.1 s for a
  cached song to several seconds for a song it has to download) plus at most 2 seconds for the lookups; whatever
  has not arrived by then is `null` or a search link, never an error.
- Deezer allows the server 50 lookups in 5 seconds, and a new 20-row answer needs 21. Under a burst of new
  songs the later answers therefore come back with fewer covers, previews and ISRCs (so plain search links), marked
  `max-age=20` so that asking again shortly afterwards fills them in. ISRCs and covers seen before are remembered
  for a week; previews never are.

Errors are `{"ok": false, "error": {"code", "message"}}`; `message` is fit to show to a listener.

| Status | `code` | Meaning |
|---|---|---|
| 400 | `missing_seed`, `invalid_parameter` | No song named, or a bad value |
| 400 | `invalid_link`, `not_a_track`, `unsupported_link` | Not a Spotify or Apple Music link; an album link without a song; a playlist, artist, podcast or a short link that leads elsewhere |
| 404 | `track_not_found` | Spotify, Apple Music or Deezer does not know the linked song |
| 404 | `song_not_found` | The engine found no recording of it with a preview, by name or by the fallback below |
| 429 | `rate_limited` | The engine's per-listener limit; honour `Retry-After` |
| 502 | `resolver_unavailable`, `engine_unavailable` | Spotify / iTunes / Deezer, or the engine, did not answer |
| 503 | `engine_busy` | The engine is busy; retry after `Retry-After` seconds |

### `GET /api/mobile/link`

`?deezer_id=<result's deezer_id>&platform=spotify|apple`, optionally `storefront=` (Apple) and the result's
`title=` and `artist=` (so a link can still be built if Deezer is down). Returns the best link for one result:

```json
{ "ok": true, "deezer_id": 567060, "platform": "apple", "exact": true, "kind": "track", "storefront": "de",
  "url": "https://music.apple.com/de/album/gekreuzigt-2006/…?i=…", "retry_after": null }
```

`kind` is `track`, `isrc_search` (Spotify only) or `search`, as for `spotify_kind` above; `exact` is `true` only
for `track`. With `platform=spotify` the answer also has `spotify_id` (the track id or `null`), and the server
tries harder than for a whole list: it checks up to five of the candidates it knows instead of the first, and
takes the one closest in length to Deezer's recording.
`exact: false` with `retry_after: null` means the store has no clear match (the search link is the answer; this
is remembered for a day). `retry_after: <seconds>` means the iTunes budget is spent: `url` is still a usable search
link, and asking again after that many seconds may return the song's page. Ask for at most a screenful of rows,
one at a time, and stop at the first `retry_after`. Errors: 400 `invalid_parameter`, 404 `track_not_found`,
502 `lookup_unavailable`.

### Spotify tracks without credentials

Opening `open.spotify.com/track/<id>` on a phone plays the song at once; an ISRC search needs one more tap, and
nothing on a phone can turn an ISRC into a track id without Spotify credentials. The website has none (only a
public PKCE client id), so the server finds track ids like this:

1. **One call per answer** to ListenBrainz Labs, `POST https://labs.api.listenbrainz.org/spotify-id-from-metadata/json`,
   with every result (and the seed, when it did not come as a Spotify link) as `{artist_name, release_name,
   track_name}`. It matches names as written, so each song is asked in up to four spellings in that same call:
   Deezer's full artist credit or the result's artist, with the title as given or without its bracketed and
   dashed tags. Deezer's album name is passed as `release_name`; in the sample below it changed nothing.
2. **Each first candidate is checked** on Spotify's public page `open.spotify.com/embed/track/<id>`: the title must
   be the same song and the same kind of version (live, remix, acoustic, ...), and the track must be playable.
   Artist names are not compared, because Spotify romanises them (Кино is "Kino" there). An answer waits 0.3 s
   for these checks; a candidate not yet checked is served as it is, the answer is marked `max-age=20`, and the
   check finishes for the next answer.
3. Candidates, misses and checked picks are remembered like the other link matches (a week for a hit, a day for a
   miss), so a repeated answer asks nobody. On 429 or a 5xx, ListenBrainz (or Spotify's embed pages) is left alone
   for `Retry-After`, else a minute, and rows fall back to ISRC search links. Embed checks are capped at 40 per
   10 seconds.

Measured on 2026-11-23 with the production engine's top 20 for 12 seeds in 12 languages (240 results):

| | Results with a track id |
|---|---|
| All 240 | 157 served on a first request (65%), 152 after the checks (63%) |
| By seed language | ko 16, ja 16, es 16, fr 15, de 14, tr 14, en 12, ru 12, hi 12, it 11, ar 9, pt 5 (of 20, after checks) |
| The result's own artist and title only | 123 (51%); Deezer's full credit adds the rest |

- **Right song:** all 154 ids of an earlier run of the same lists were compared with Spotify's page: 154 the same
  title, 147 the same artist text and the other 7 the same artist romanised or respelled; no wrong song. Two were
  another version (a studio cut for a live recording and the reverse), which the version rule now drops, and 3
  differed from Deezer's length by more than 5 seconds (another release). The checks also drop about 2% that
  Spotify marks unplayable, and a few right songs whose titles are written differently (`Ijazat From "One Night
  Stand"`); those keep their ISRC search link.
- **Time:** a first answer took a median 1.5 s after the engine (maximum 1.7 s), against 0.3 s before: 0.7 s is the
  ListenBrainz call, which can only start once Deezer has named the artists, and 0.3 s the wait for the checks. A
  repeated answer took 0 ms (maximum 0.3 s while checks were still running).
- **Not used:** MusicBrainz `GET /ws/2/isrc/<ISRC>?inc=url-rels`. Of 30 results ListenBrainz missed, it knew 8
  ISRCs and had a Spotify link for 1, at one request a second and with 503s at that pace.

ListenBrainz Labs is an experimental service without a stated limit or guarantee, and Spotify's embed page is
not a documented API; when either fails the links are what they were before, never an error.

### Where the song's identity comes from

- **Spotify link**: the public embed page `open.spotify.com/embed/track/<id>` gives title, artists, duration and
  cover, but no ISRC (oEmbed gives only the title). With the Spotify credentials above, the ISRC is read from the
  Web API and passed to the engine. Short links are expanded by following redirects on Spotify's short-link hosts
  only.
- **Apple Music link**: the iTunes Lookup API gives title, artist, duration and artwork; it never has an ISRC.
- In both cases `query.isrc` is then filled from Deezer's entry for the recording the engine matched, and every
  result's ISRC from Deezer's track API (50 requests per 5 seconds; the server stays under it and returns `null`
  rather than wait).
- **A shared link the engine does not find by name.** Spotify and Apple Music name a song as its market does;
  Deezer may hold another title or artist spelling (Jay Chou's 晴天 is "Sunny Day" there; Кино is "Kino" on Spotify).
  Only after the engine's 404 for the names, and only for links (they have a length), the server looks for the
  recording itself:
  1. the artist's name in the title's script, from Spotify's embed page asked in that script's language (`ru` for
     Cyrillic, `zh-TW` and `ja` for Han, `ar`, `ko`, ...);
  2. on Deezer, the first two artists its artist search gives for each name (it knows other spellings: 周杰倫 finds
     Jay Chou), their top 100 tracks in Deezer's default naming and in that language, and a title search;
  3. a track is accepted only when its length is within 2 seconds of the link's **and** its title in one of those
     namings is the link's title (same kind of version), or, for Apple Music links, it is the one recording of
     that length on the album the link names. Of several releases of the song, the recording (ISRC) most of them
     are wins, then the most played. The length alone is never enough, even for the only track that has it
     (three of Jay Chou's 63 tracks are within 2 seconds of 晴天). The engine is then asked with `id=`;
  4. failing that, the engine is asked once more by name with the artist from step 1; this is a match by name, as
     on the normal path, and may be another release (Spotify's 5:46 Кукушка answered from Deezer's 6:41 one).

  What worked is remembered for the link for 30 days (a miss for a day), so later requests go straight to it.
  Measured on 2026-11-23: 晴天 (Spotify and Apple Music links) → Deezer 2602471112 "Sunny Day"; Земфира's ХОЧЕШЬ? →
  1520555492; Кино's Группа крови and Кукушка by step 4. The lookups took 1.1 to 1.3 s (9 to 14 requests, 13 to
  Deezer at most) on top of the engine's two calls; songs found by name (アイドル, Wegz's البخت, Despacito, bad guy)
  made no extra request. A title that Deezer holds only in translation in every language asked stays a 404, and
  so do typed `title`+`artist` seeds. No credential-free way from a Spotify track id to an ISRC was found:
  ListenBrainz Labs maps names or MusicBrainz ids to Spotify ids, not back; MusicBrainz knows no URL for this
  track and lists its recording without an ISRC.
- The engine is reached at `SOUNDALIKE_ENGINE_URL` (default `https://soundalike.yassin.app/api/anysong`). Through
  the public address every phone request shares the web host's engine rate limit (cached answers are free). When
  the variable is a plain `http://` LAN address, the listener's `CF-Connecting-IP` is forwarded so the engine's
  per-listener limit applies; the engine host's firewall must admit the web host for that.

### The phone page

`/find?url=<shared link>` ([`webapp/find/index.html`](../webapp/find/index.html)) is the same thing as a page: it
accepts every `/api/mobile/similar` parameter (and `text=` for share-sheet text), or a link pasted into its box,
and lists the seed and its soundalikes with cover, match and a preview button. Tapping a row opens the song in
Spotify or Apple Music, by default the app the link came from, with a switch for the other. Rows navigate in the
same tab, without `target="_blank"`, so the phone's app links hand over to the installed app from an in-app
browser as well; coming back shows the saved list without a new request. Under the seed it has
[Match a part](#match-a-part-in-the-clients), and `part=<start>-<end>` in the page's address opens a part. For
local work, `node webapp/tools/mobile-dev-server.js` serves the home page, this page and their endpoints against
the engine named by `SOUNDALIKE_ENGINE_URL`.

So that a `/find` link opens the installed app instead of the page, the site serves
`/.well-known/apple-app-site-association` (iPhone app `FJH4UWVC4K.app.yassin.soundalike`, paths `/find*`) and
`/.well-known/assetlinks.json` (Android package `app.yassin.soundalike` with its release key's SHA-256). Both are
plain JSON files in [`webapp/well-known/`](../webapp/well-known), reached through `rewrites` in `site.json` so they
are served as `application/json` without a redirect, which both platforms require. A new signing key or app id
means editing those files.

## A part of a song

A listener can say which part of a song the matches should follow. Measurements: `anysong-lab/partmatch/REPORT.md`
(2026-11-23) and [ANYSONG_EVAL.md](ANYSONG_EVAL.md#a-part-of-a-song-as-the-seed-2026-11-23); local demo page:
[`demo/part-match/`](../demo/part-match) (`python demo/part-match/serve.py`, not deployed). Served: a part of the
preview (`--parts`, in the production unit). Not served: clips of the listener's own audio (`--excerpts`) and
naming a clip's song (`--identify-score`), which are off unless the engine is started with their flags; the
production unit must not carry them (`tests/test_anysong_api.py` checks the unit file), and `--identify-score`
without `--excerpts` refuses to start.

**`--parts`: a part of the preview.** `GET /api/anysong/similar?...&part=<start>-<end>` takes seconds of the song's
30 s preview, decimals allowed, `0 <= start < end <= 30` and at least 10 apart (the sound model's window); every
other parameter (`id` or `title` + `artist` or `isrc`, `k`, `lang`, `scene`, `dedupe`) works as without it.

| Request | Answer |
|---|---|
| `part=10-20` | 200, the lists of `similar` for that span |
| `part=` that is malformed, outside 0 to 30, or shorter than 10 s | 400, before the song is looked up |
| `part=` on an engine without `--parts` | 404 `choosing a part of the preview is not enabled`, before any lookup |

The seed is then only the windows of that span (`audio.part_windows`): the 10 s sound windows (and their scene-head
vectors) and the 6 s samples chunks it covers by 70 % or more, and the 20 s melody shingle around it, so a 10 s part
still hears 5 s either side in the melody model. A catalog song uses its stored windows (no download, no
embedding); a song outside the catalog is downloaded and embedded as before and cut to the span. Playlist vectors,
tempo, language (and so the language gate) and the artist's scene stay the song's own, the seed's own song and the
other lists' rules are as for the whole preview, and every list is returned. Stems (`/dissect`) do not take a
part.

- **Detection.** `/healthz` has `parts: true` (and `excerpts: false`), and every `similar` answer of such an engine
  carries `query.parts`: `{"min_s": 10.0, "max_s": 30.0, "preview_s": 30.0, "song_offset_s": 48.0}`.
  `song_offset_s` is where the preview starts in the song, `null` when the engine does not know the song's length
  (always known for a song outside the catalog; for a catalog song only when it came through `search` in the last
  ten minutes) or the song is shorter than 90 s. 16 of 17 previews measured against full recordings begin 47.5 to
  48.0 s into the song and one at 59.4 s, so clients should word it as the usual place; a client that knows the
  song's length itself may assume 48 s for songs of 90 s or more.
- **Echo.** An answer for a span carries `query.part`: `{"start": 10.0, "end": 20.0, "heard": {"sonic": [10.0,
  20.0], "samples": [9.19, 21.44], "melody": [5.0, 25.0]}}`, the span asked for and the seconds of the preview each
  model heard. Without `part=` there is no `query.part`.
- **Cache and rate limit.** Results are cached per window choice, apart from the whole preview's: spans that pick
  the same windows share one result (each answer echoes its own span), and a span that picks every window (such as
  `0-30`) is the whole-preview result, never the other way round. A span whose windows were not ranked yet costs
  one `/similar` request of the client's budget, like any new ranking; cached ones are free.
- Parts add no model and no table: measured under an 8 GB cap with part requests, a live seed and a Stems
  dissection, see [ANYSONG_EVAL.md](ANYSONG_EVAL.md#a-part-of-a-song-as-the-seed-2026-11-23).

**`--excerpts`: a clip of the listener's own audio.** `POST /api/anysong/excerpt` takes one clip of 5 to 30 s as the
request body (WAV, FLAC, Ogg or MP3: what libsndfile reads; at most `--excerpt-max-mb`, default 12) and the query
parameters `k`, `dedupe`, `lang`, `scene` of `similar`. The clip is decoded in memory with a header check first,
embedded like a song outside the catalog with as many windows as it fills (`audio.excerpt_clips`), ranked, and
dropped: nothing is written to disk, cached, logged or added to the catalog or the learn queue. The answer has the
shape of `similar` with `query.source` `excerpt` and `query.seconds`; `listeners` and the artist scene need a
known song. Optional `title` + `artist` or `isrc` name the song: it is then left out of the lists and its artist
gives the scene, the catalog tiers and the listener lists (a name that nothing knows is not an error). With
`--identify-score S` a clip without a name is first looked up in the whole samples index, and a best match scoring
at least `S` (0.55: the same recording) stands in for the name and is reported as `query.identified`. The clip is
brought to full scale before it is embedded, because the sound model hears level. The route has its own budget,
`--excerpt-per-min` (default 6 per client), answers `411` without a body, `413` over the size limit, `422` for
audio it cannot decode, clips outside 5 to 30 s and silence, and a CORS preflight when `--cors-origin` is set.
`--part-sound-weight` sets the consensus body's sound weight for part and excerpt seeds (default: the body's own;
not set in production).

### Match a part in the clients

The clients show a **Match a part** control (marked beta) only when the `similar` answer on screen has
`query.parts`, so it costs no request of its own and nothing changes with an engine without `--parts`. They keep
spans on a 5-second grid, 10, 15 or 20 seconds long. An answer to `part=` without `query.part` is for the whole
preview (an older engine ignores the parameter), so the client drops the part; an engine that refuses the
parameter (400 or 404) is asked once more without it.

- **Website** ([`webapp/search.js`](../webapp/search.js)): "Match a part" follows the seed's preview button. It
  opens a timeline of the preview (its waveform, read from Deezer's CDN with WebAudio, or a plain bar) with a span
  to drag, resize by its edges or by the length buttons, and move with the arrow keys (Shift changes the length).
  Times are song times when the place of the preview is known (`song_offset_s`, or the usual 0:48 for a song the
  search listed as 90 seconds or longer), else preview times. "Play this part" plays just the span, and Apply
  reloads the lists. A chip ("Matching 0:58–1:13 of the song · whole preview") leads back. The part is in the
  share link (`#id=<id>&tab=<tab>&part=<start>-<end>`, also as `/?id=…&part=…`) and in the request, so each part
  has its own cached answer. A newly picked song always starts from its whole preview.
  - **Any part of the song (website only).** The preview is all Soundalike holds, so any other part has to come
    from audio the listener is playing or has. When the answer on screen also has `query.parts.excerpt`
    (`{min_s, max_s, max_bytes}`: an engine with `--excerpts`), the panel gets a row of sources after "The
    30-second preview":
    - **Any part, from YouTube or another tab** (experimental; Chromium browsers on a computer). The listener
      opens the song in another tab at the part they want (a YouTube link and a time can be pasted and are opened
      as `youtube.com/watch?v=…&t=…s`; with a seed there is also a YouTube search link), presses Capture and
      picks that tab in the browser's own dialog with "Also share tab audio" on. The page takes the tab's sound
      with `getDisplayMedia` (voice processing off, the picture's track stopped at once, plain samples through the
      worklet [`webapp/clip-tap.js`](../webapp/clip-tap.js)) for 10, 20 or 30 seconds, and stops every track when
      it is done, cancelled, or the share has no sound. Soundalike asks YouTube for nothing: no server-side fetch,
      no player embed. It hears only what was captured, so there is no slider over the whole video, and the panel
      says that sharing a tab does not make copying allowed. Checked on 2026-11-23 in Chrome 156 with a CC BY
      video on YouTube and the page served with production's response headers (`Permissions-Policy` blocks the
      microphone, not tab sharing): three 20-second captures shared 8, 9 and 6 of their Best matches top 10 with
      the same seconds of the artist's own file.
    - **A file of the song** (chosen or dropped; decoded in the page with WebAudio, never uploaded).

    Either gives a timeline of the whole capture or file with the same span (10, 15 or 20 seconds, moved by the
    second) and "Play this part". Apply renders only the span to one channel at 24 kHz, brings it to full scale
    (the sound model hears level), and sends it as a 16-bit WAV (0.5 to 1 MB) with
    `POST /api/anysong/excerpt?id=<seed>`: the clip replaces the preview as the audio, while the song, its artist,
    language and links stay the seed's. The chip reads "Matching your clip (0:00–0:20 of your capture) · whole
    preview"; the language and Audio only switches send the same clip again. A clip is never in a link (the
    address stays the song's) and nothing of it is kept, by the page beyond the open result or by the server.
    Without a seed, "Match any part of a song, from YouTube or a file" under the search bar (shown once an answer
    has advertised clips; the advert is remembered in `localStorage`) matches the clip by sound alone: no
    Listeners also play, no share link, and a clip the server names as a catalog song takes that song as its
    seed. Firefox and Safari share no tab audio, so they get the file source only; a browser without WebAudio
    gets neither.
- **Spotify extension** (runtime 2.16.0): the same control in the page's header, drawn with the theme's colours.
  The page plays no previews, so "Play this part" plays the song in Spotify from about where the part starts and
  pauses when it is over. Saved lists are kept per part (`…:part:<start>-<end>` in the cache id), and nothing is
  remembered from one search to the next.
- **Phones**: `/api/mobile/similar` takes `part=` and reports `query.part` and `query.parts`
  ([Mobile API](#mobile-api)).
  - **`/find` page** ([`webapp/find/index.html`](../webapp/find/index.html)): it has the control in its own inline
    script, because its Content-Security-Policy admits no script file: "Match a part" under the seed opens a plain
    bar of the preview (no waveform, so nothing is fetched) with a span to drag by finger, move with the arrow
    keys, or place by tapping the bar, and 10 s, 15 s and 20 s buttons for its length. "Play this part" plays the
    span from the seed's preview (`/api/preview`), Apply reloads the list, and a chip ("Matching 0:58–1:13 of the
    song · whole preview") leads back. Times are song times only when the answer has `song_offset_s`; the page
    does not know the song's length, so it never assumes 0:48. The part is in the page's address
    (`/find?url=…&part=<start>-<end>`), so a shared link and the Back button keep it; a part that is not one
    (shorter than 10 s, past 30 s) is ignored and taken out of the address, and a newly pasted link starts from
    its whole preview.
  - **Android app** (2.1.0, [`android/`](../android/README.md#match-a-part)): "Match a part" sits under the song in
    the results sheet and in the app. It opens a bar of the preview with the part on it: drag the part, drag an
    edge, tap elsewhere, or use the −5 s / +5 s and 10 / 15 / 20 s buttons. "Play this part" plays the span from the
    preview and Apply reloads the list; a chip ("Matching 0:58–1:08 of the song", then "Whole preview") leads back.
    Song times are used when the server gives the preview's place or the song is known to be 90 s or longer. The
    part stays while the player moves on to other songs and ends when another song's list is loaded.
  - The iPhone app has no control yet.
- **Which lists follow.** Best matches, Listeners also play and Melody match are read from the one answer, so all
  three follow the part; Listeners also play still starts from the whole song's listeners. The extension's Sounds
  like list (local companion or the older hosted library) always uses the whole song.

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

**Adding a few songs to a built index.** `index_build` fits the samples mean and every stage-1 basis on the rows it
is given, so a rebuild changes all stored rows a little and needs every old shard.
[`index_append`](../src/soundalike/anysong/index_append.py) instead copies an index and appends new tracks prepared
with its `mu.npy` and `basis.npy` and the same quantizer, so scores between old tracks stay bit-identical:

```sh
python -m soundalike.anysong.previews --ids new_ids.txt --out previews --save-meta tracks.jsonl --adaptive
python -m soundalike.anysong.embed_run --model muq_mulan --ids new_ids.txt --previews previews --out emb_new
python -m soundalike.anysong.index_append --base index_1m --emb emb_new --ids new_ids.txt --out index_new
python -m soundalike.anysong.hub_build --index index_new --meta catalog_meta.npz
```

`previews --save-meta` keeps each track's Deezer metadata (title, artist, ISRC, rank, duration) from the request
that signs its preview link, and `--adaptive` paces downloads like the crawl. `embed_run` and `languages` take
`--device cpu` when the GPU is busy (per song on 12 threads: CLEWS 0.4 s, SampleID 1.3 s, MuQ-MuLan 3.7 s, Whisper
5.4 s; the vectors match the GPU's to cosine 0.99998). `hub.npy` is not copied, because new songs change their
neighbours' hub scores; rerun `hub_build`. Songs that should reach every seed, such as well-known songs the crawl
missed, also go into `core_ids.npy`. Refit with `index_build` once additions stop being small against the catalog.

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
songs without a confident label take the same prior, and stay when it is unknown. The ranked lists (Best matches,
Same vibe, Sounds like, and `main` and `listeners`, which the page and the Spotify extension show) are filtered; the
three channel lists are unchanged. `lang=all` returns the same unfiltered Best ranking and still reports the
other-language count in `query.hidden`.

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
`index/full-20261101` on the root disk, with its own `core_ids.npy`, as the rollback. From 2026-11-09 to
2026-11-22 `index/current` was `index-store/grown-20261109`: 999,293 songs, the 269,278 core songs plus 730,015
crawled ones, served with the catalog tiers and samples' stage 1 in fp16 (`--half-stage1 samples`). Since
2026-11-22 (v12) `index/current` is the 1,109,024-song index with the whitened sonic channel and the v12 tables
listed below. The service is
[`deploy/homelab/soundalike-anysong.service`](../deploy/homelab/soundalike-anysong.service) (user `anysong`,
`MemoryMax=14G` since v12, 6G before; enabled at boot; startup loads and warms the models and index). Before v12
the process itself used about 4 GB of RAM; the rest of the cap is page cache for the memory-mapped windows, which
the kernel trims.

**v12 (consensus body) on the 8 GB card.** At 1,109,024 songs the tables of the v12 service take 6.2 GB of VRAM and
its models 4.4 GB, so they cannot all live on the card. The unit therefore runs the service with

- `--host-tables samples sonic head`: the samples and sonic stage-1 projections and the blend's head table stay in
  host memory, in the file cache, and are read block by block for each request (`serve.HostTable`: reader threads
  fill small page-locked buffers ahead of the GPU). The numbers are the stored ones, so every list is byte for byte
  what the tables on the card give. `melody` can be added for 0.8 GB more room on the card;
- `--consensus-windows stream`: the consensus body's sonic windows (`sonic/full.npy`) are read the same way instead
  of being held page-locked (3.2 GB of host memory).

Measured on the RTX 5080 with the process capped at 8 GB (`--vram-cap-gb 8`), Stems view on: 5.9 GB allocated,
7.1 GB at peak (a live dissection; 6.4 GB for live seeds), against 10.7 and 11.5 GB without the flags; the process
holds about 5.3 GB of host memory (0.4 GB of it page-locked buffers). Every request reads 8.1 GB from the file
cache (`samples/proj.npy`, `sonic/proj.npy`, `sonic/full.npy`, `heads/h4.emb.npy`), so the container needs room
to keep those cached: 16 GB with `MemoryMax=14G`. If they fall out of the cache every request waits for the disk.

One directory holds everything of an index, with fixed names (the unit's flags point into `index/current`):
`melody/`, `samples/`, `sonic/` (index_build / index_whiten output), `heads/h2.*` and `heads/h4.*`,
`playlists/cooc.npz` and `cf128.npy`, `catalog_meta.npz`, `languages.npz`, `scene.npz`, `core_ids.npy`,
`tempo.npz`, `ranker.npz`, `popularity.npz`, `match_quantiles.json`, `overall.json`, `consensus.npz` and
`consensus_quantiles.json`. Before the symlink swap,

```sh
/opt/soundalike/venv/bin/python -m soundalike.anysong.index_check /opt/soundalike/index-store/NAME
```

checks, without the GPU, that every file is there and that they fit each other (rows, dimensions, track ids across
the channels and tables, the models load); it prints one `PROBLEM` line per fault and exits 1. After the restart,
`/healthz` shows v12 is live: `channels` with 1,109,024 songs each, a `consensus` block (model note, `sound_cap`
5011, `windows` `stream`), `blend.head_table` `host` and `host_tables` `["samples", "sonic", "head"]`. The
Whisper language model (`--language-model`) lives in `ckpt/whisper-large-v3-turbo`.

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

- Coverage is limited to the indexed catalog: 1,109,024 songs since 2026-11-22 (the core catalog plus the crawl's
  songs, which outside the consensus body only reach seeds in their artist's scene). Famous sources missing from it
  cannot be returned, however good the match.
- Preview-only audio: shared material outside the 30-second clip is invisible.
- Benchmarks are small for some relations (see the held-out notes in the README); treat the channels as beta.
