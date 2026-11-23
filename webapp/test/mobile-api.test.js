import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createDispatchHandler } from "../api/ratings.js";
import {
  createBucket,
  createMobileService,
  parseSpotifyEmbed,
  TtlCache,
} from "../server/mobile.js";

// Recorded on 2026-11-20: Spotify's embed page for "Du hast" (its anonymous
// session token removed) and for an unknown id, the iTunes Lookup answer for
// "99 Luftballons" in the German store, the engine's answer for "Du hast"
// (k=3), and Deezer's and iTunes Search's answers for its first result.
const fixture = (name) => readFileSync(new URL(`./fixtures/mobile/${name}`, import.meta.url), "utf8");
const EMBED_HTML = fixture("spotify-embed-track.html");
const EMBED_MISSING_HTML = fixture("spotify-embed-missing.html");
const ITUNES_LOOKUP = JSON.parse(fixture("itunes-lookup-song.json"));
const ITUNES_SEARCH = JSON.parse(fixture("itunes-search-song.json"));
const DEEZER_TRACK = JSON.parse(fixture("deezer-track.json"));
const ENGINE = JSON.parse(fixture("engine-similar.json"));
// Recorded on 2026-11-23: ListenBrainz's Spotify ids for the engine fixture's
// three results and for "99 Luftballons", and Spotify's embed pages of the ids
// it gave for the first two results.
const LISTENBRAINZ = JSON.parse(fixture("listenbrainz-spotify-ids.json"));
const OOMPH_IDS = ["4dpWUmLyYPo4vMk5FTMnMv", "0v7uLvgfSSwPuEtLg4eSGt"];
const EISBRECHER_ID = "5zP1tnylWUoVgfdQZiyACB";
const CANDIDATE_PAGES = Object.fromEntries(
  [...OOMPH_IDS, EISBRECHER_ID].map((id) => [id, fixture(`spotify-embed-${id}.html`)]),
);

const SPOTIFY_ID = "5awDvzxWfd53SSrsRZ8pXO";
const SPOTIFY_LINK = `https://open.spotify.com/intl-de/track/${SPOTIFY_ID}?si=abc`;
const APPLE_LINK = "https://music.apple.com/de/album/99-luftballons/1446013584?i=1446013598";
const ENGINE_URL = "http://engine.test/api/anysong";
const FIRST_ID = ENGINE.results.main[0].id;

function reply(body, status = 200, headers = {}) {
  const textBody = typeof body === "string" ? body : JSON.stringify(body);
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: new Headers(headers),
    json: async () => JSON.parse(textBody),
    text: async () => textBody,
  };
}

/** Waits for the caller's abort signal, like a request that never answers. */
function hang(options) {
  return new Promise((_, reject) => {
    const fail = () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    if (options?.signal?.aborted) fail();
    else options?.signal?.addEventListener("abort", fail, { once: true });
  });
}

function deezerTrack(key) {
  // `isrc:<ISRC>` lookups answer with the recorded track itself.
  if (key.startsWith("isrc")) return DEEZER_TRACK;
  return { ...DEEZER_TRACK, id: Number(key), isrc: key === String(FIRST_ID) ? DEEZER_TRACK.isrc : "DEA620000001" };
}

/**
 * A network made of the recorded answers. `routes` replaces one upstream by
 * host: a function (url, options) => response, e.g. to fail or hang it.
 */
function network(routes = {}) {
  const calls = [];
  const defaults = {
    "open.spotify.com": () => reply(EMBED_HTML),
    "itunes.apple.com": (url) =>
      reply(url.pathname === "/lookup" ? ITUNES_LOOKUP : ITUNES_SEARCH),
    "api.deezer.com": (url) => reply(deezerTrack(url.pathname.split("/").pop())),
    "engine.test": () => reply(ENGINE),
    // ListenBrainz knows none of the songs unless a test says otherwise.
    "labs.api.listenbrainz.org": (_, options) =>
      reply(JSON.parse(options.body).map((query) => ({ ...query, spotify_track_ids: [] }))),
  };
  const fetch = async (input, options = {}) => {
    const url = new URL(input);
    calls.push(url);
    const route = routes[url.hostname] ?? defaults[url.hostname];
    assert.ok(route, `unexpected request to ${url.hostname}`);
    return route(url, options);
  };
  return { fetch, calls, count: (host) => calls.filter((url) => url.hostname === host).length };
}

/** ListenBrainz and Spotify's embed pages as recorded; `pages` replaces or adds embed pages by track id. */
function trackLookup(pages = {}) {
  return {
    "labs.api.listenbrainz.org": (_, options) =>
      reply(
        JSON.parse(options.body).map((query) => ({
          ...query,
          spotify_track_ids:
            LISTENBRAINZ.find(
              (known) => known.artist_name === query.artist_name && known.track_name === query.track_name,
            )?.spotify_track_ids ?? [],
        })),
      ),
    "open.spotify.com": (url, options) => {
      const id = url.pathname.split("/").pop();
      const page = { ...CANDIDATE_PAGES, ...pages }[id];
      if (typeof page === "function") return page(url, options);
      return reply(page ?? EMBED_HTML);
    },
  };
}

function service(net, options = {}) {
  return createMobileService({
    fetch: net.fetch,
    env: { SOUNDALIKE_ENGINE_URL: ENGINE_URL, ...options.env },
    budgetMs: options.budgetMs ?? 500,
    seedTimeoutMs: 200,
    engineTimeoutMs: 200,
    now: options.now,
    appleEagerRows: options.appleEagerRows,
    lookupTimeoutMs: 200,
    verifyWaitMs: options.verifyWaitMs ?? 200,
  });
}

async function call(handler, path, { method = "GET", headers = {} } = {}) {
  const response = {
    statusCode: 200,
    headers: {},
    body: undefined,
    setHeader(name, value) {
      this.headers[name.toLowerCase()] = String(value);
    },
    end(body) {
      this.body = body;
    },
  };
  await handler({ method, url: path, headers }, response);
  return {
    status: response.statusCode,
    headers: response.headers,
    body: response.body === undefined ? null : JSON.parse(response.body),
  };
}

const similarPath = (params) => `/api/ratings?__soundalike_handler=mobile-similar&${new URLSearchParams(params)}`;

test("reads the recorded Spotify embed page", () => {
  const embed = parseSpotifyEmbed(EMBED_HTML);
  assert.equal(embed.title, "Du hast");
  assert.deepEqual(embed.artists, ["Rammstein"]);
  assert.ok(embed.durationMs > 200000 && embed.durationMs < 260000);
  assert.match(embed.artwork, /^https:\/\/[a-z0-9.-]+\.(spotifycdn\.com|scdn\.co)\//);
  assert.equal(/isrc/i.test(EMBED_HTML), false, "the embed page carries no ISRC");
  assert.equal(parseSpotifyEmbed(EMBED_MISSING_HTML), "missing");
  for (const broken of ["", "<html></html>", '<script id="__NEXT_DATA__">{not json</script>', null]) {
    assert.equal(parseSpotifyEmbed(broken), null);
  }
  const album = '<script id="__NEXT_DATA__">{"props":{"pageProps":{"state":{"data":{"entity":{"type":"album","name":"x"}}}}}}</script>';
  assert.equal(parseSpotifyEmbed(album), null);
});

test("a Spotify link returns the documented response shape", async () => {
  const net = network();
  const { similarHandler } = service(net);
  const { status, headers, body } = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(status, 200);
  assert.equal(headers["access-control-allow-origin"], "*");
  assert.equal(headers["cache-control"], "public, max-age=300");
  assert.equal(headers["content-type"], "application/json; charset=utf-8");
  assert.deepEqual(Object.keys(body), ["ok", "query", "results", "engine", "links"]);
  assert.deepEqual(body.query, {
    title: "Du hast",
    artist: "Rammstein",
    isrc: "DEA620000001", // from the Deezer release the engine matched
    artwork: body.query.artwork,
    platform: "spotify",
    source_url: `https://open.spotify.com/track/${SPOTIFY_ID}`,
    spotify_id: SPOTIFY_ID,
    language: "de",
    storefront: "us",
    deezer_id: ENGINE.query.id,
  });
  assert.match(body.query.artwork, /^https:\/\//);
  assert.equal(body.results.length, 3);
  assert.deepEqual(body.results[0], {
    title: "Gekreuzigt 2006",
    artist: "OOMPH!",
    match: ENGINE.results.main[0].match,
    deezer_id: FIRST_ID,
    isrc: DEEZER_TRACK.isrc,
    artwork: DEEZER_TRACK.album.cover_medium,
    preview_url: DEEZER_TRACK.preview,
    // Without credentials: a search for exactly this recording by its ISRC.
    spotify_url: `https://open.spotify.com/search/isrc%3A${DEEZER_TRACK.isrc}`,
    spotify_exact: false,
    spotify_kind: "isrc_search",
    spotify_id: null,
    apple_music_url: "https://music.apple.com/us/search?term=OOMPH!%20Gekreuzigt%202006",
    apple_exact: false,
  });
  assert.deepEqual(body.engine, {
    list: "main",
    k: 3,
    lang: "same",
    source: ENGINE.query.source,
    matched_title: ENGINE.query.title,
    matched_artist: ENGINE.query.artist,
    hidden: ENGINE.query.hidden,
  });
  assert.deepEqual(body.links, {
    spotify_exact: 0,
    spotify_isrc: 3,
    apple_exact: 0,
    spotify_lookup: true,
    spotify_source: "metadata",
    apple_eager_rows: 5,
  });

  const engineCall = net.calls.find((url) => url.hostname === "engine.test");
  assert.equal(engineCall.pathname, "/api/anysong/similar");
  assert.deepEqual(Object.fromEntries(engineCall.searchParams), {
    title: "Du hast",
    artist: "Rammstein",
    k: "3",
    lang: "same",
  });
  // A Spotify link asks for Spotify links only: no iTunes call is spent.
  assert.equal(net.count("itunes.apple.com"), 0);
});

test("an Apple Music link resolves through iTunes and maps the top rows", async () => {
  const net = network();
  const { similarHandler } = service(net, { appleEagerRows: 2 });
  const { status, body } = await call(similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  assert.equal(status, 200);
  assert.equal(body.query.title, "99 Luftballons");
  assert.equal(body.query.artist, "Nena");
  assert.equal(body.query.platform, "apple");
  assert.equal(body.query.storefront, "de");
  assert.equal(body.query.source_url, APPLE_LINK);
  assert.match(body.query.artwork, /\/300x300bb\.jpg$/);
  const lookup = net.calls.find((url) => url.pathname === "/lookup");
  assert.deepEqual(Object.fromEntries(lookup.searchParams), { id: "1446013598", country: "de", entity: "song" });

  // Row 1 is the recorded search answer's song; rows 2 and 3 are other songs
  // (the stub answers every search alike), and row 3 is past the eager rows.
  assert.equal(body.results[0].apple_exact, true);
  assert.match(body.results[0].apple_music_url, /^https:\/\/music\.apple\.com\/de\/album\/[^?]+\?i=\d+$/);
  assert.equal(body.results[1].apple_exact, false);
  assert.equal(body.results[1].apple_music_url, "https://music.apple.com/de/search?term=Eisbrecher%201000%20Flammen");
  assert.equal(body.results[2].apple_exact, false);
  assert.equal(net.calls.filter((url) => url.pathname === "/search").length, 2);
  const search = net.calls.find((url) => url.pathname === "/search");
  assert.equal(search.searchParams.get("term"), "OOMPH! Gekreuzigt 2006");
  assert.equal(search.searchParams.get("country"), "de");
  assert.equal(search.searchParams.get("entity"), "song");
  assert.deepEqual(body.links, {
    spotify_exact: 0,
    spotify_isrc: 3,
    apple_exact: 1,
    spotify_lookup: true,
    spotify_source: "metadata",
    apple_eager_rows: 2,
  });
});

test("repeat requests are answered from memory", async () => {
  const net = network();
  const { similarHandler } = service(net);
  const first = await call(similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  const before = net.calls.length;
  const second = await call(similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  assert.deepEqual(second.body, first.body);
  assert.equal(net.calls.length, before);
});

test("isrc and title+artist seeds, platform and storefront parameters", async () => {
  const net = network();
  const { similarHandler } = service(net, { appleEagerRows: 0 });
  const byIsrc = await call(similarHandler, similarPath({ isrc: "de-c76-06-00298", k: "3" }));
  assert.equal(byIsrc.status, 200);
  assert.equal(byIsrc.body.query.isrc, "DEC760600298");
  assert.equal(byIsrc.body.query.platform, "both");
  assert.equal(byIsrc.body.query.source_url, null);
  assert.ok(net.calls.some((url) => url.pathname === "/track/isrc%3ADEC760600298"));
  const engineCall = net.calls.find((url) => url.hostname === "engine.test");
  assert.equal(engineCall.searchParams.get("isrc"), "DEC760600298");

  const byName = await call(
    similarHandler,
    similarPath({ title: "Du hast", artist: "Rammstein", platform: "apple", storefront: "FR", lang: "all" }),
    { headers: { "cf-ipcountry": "DE" } },
  );
  assert.equal(byName.status, 200);
  assert.equal(byName.body.query.platform, "apple");
  assert.equal(byName.body.query.storefront, "fr");
  assert.equal(byName.body.engine.lang, "all");
  assert.equal(byName.headers["cache-control"], "public, max-age=300");
  assert.match(byName.body.results[0].apple_music_url, /^https:\/\/music\.apple\.com\/fr\/search\?term=/);

  // Without storefront= the listener's country (Cloudflare) picks the store,
  // and the answer is then only cacheable by that listener.
  const guessed = await call(similarHandler, similarPath({ title: "Du hast", artist: "Rammstein" }), {
    headers: { "cf-ipcountry": "DE" },
  });
  assert.equal(guessed.body.query.storefront, "de");
  assert.equal(guessed.headers["cache-control"], "private, max-age=300");
});

test("rejects bad requests with 400 and a code", async () => {
  const net = network();
  const { similarHandler } = service(net);
  const cases = [
    [{}, "missing_seed"],
    [{ url: SPOTIFY_LINK, isrc: "nope" }, "invalid_parameter"],
    [{ artist: "only an artist" }, "invalid_parameter"],
    [{ url: "https://example.com/track/1" }, "invalid_link"],
    [{ url: "https://music.apple.com/us/album/x/1559885420" }, "not_a_track"],
    [{ url: "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M" }, "unsupported_link"],
    [{ spotify_id: "short" }, "invalid_parameter"],
    [{ apple_id: "12ab" }, "invalid_parameter"],
    [{ isrc: "nope" }, "invalid_parameter"],
    [{ title: "only a title" }, "invalid_parameter"],
    [{ title: "x".repeat(301), artist: "y" }, "invalid_parameter"],
    [{ url: SPOTIFY_LINK, k: "0" }, "invalid_parameter"],
    [{ url: SPOTIFY_LINK, k: "51" }, "invalid_parameter"],
    [{ url: SPOTIFY_LINK, k: "ten" }, "invalid_parameter"],
    [{ url: SPOTIFY_LINK, lang: "en" }, "invalid_parameter"],
    [{ url: SPOTIFY_LINK, platform: "tidal" }, "invalid_parameter"],
    [{ url: SPOTIFY_LINK, storefront: "usa" }, "invalid_parameter"],
  ];
  for (const [params, code] of cases) {
    const { status, headers, body } = await call(similarHandler, similarPath(params));
    assert.equal(status, 400, JSON.stringify(params));
    assert.deepEqual(Object.keys(body), ["ok", "error"]);
    assert.equal(body.ok, false);
    assert.equal(body.error.code, code, JSON.stringify(params));
    assert.equal(typeof body.error.message, "string");
    assert.equal(headers["cache-control"], "no-store, max-age=0");
    assert.equal(headers["access-control-allow-origin"], "*");
  }
  const twice = await call(similarHandler, `${similarPath({ url: SPOTIFY_LINK })}&k=3&k=4`);
  assert.equal(twice.body.error.code, "invalid_parameter");
  assert.equal(net.calls.length, 0, "nothing is fetched for a bad request");

  const post = await call(similarHandler, similarPath({ url: SPOTIFY_LINK }), { method: "POST" });
  assert.equal(post.status, 405);
  assert.equal(post.headers.allow, "GET, OPTIONS");
  const preflight = await call(similarHandler, similarPath({}), { method: "OPTIONS" });
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers["access-control-allow-methods"], "GET, OPTIONS");
});

test("several names at once: the most specific wins, the rest are hints", async () => {
  // The iPhone app sends the id it has together with the ISRC, title and artist it knows.
  const net = network();
  const { similarHandler } = service(net);
  const all = { spotify_id: SPOTIFY_ID, isrc: "GBAAA0000001", title: "Typed Title", artist: "Typed Artist", platform: "spotify", k: "3" };
  const byId = await call(similarHandler, similarPath(all));
  assert.equal(byId.status, 200);
  assert.equal(byId.body.query.title, "Du hast", "the id names the song, not the typed title");
  assert.equal(byId.body.query.source_url, `https://open.spotify.com/track/${SPOTIFY_ID}`);
  assert.equal(byId.body.query.isrc, "GBAAA0000001", "the app's ISRC fills what Spotify's page lacks");
  const engineCall = net.calls.find((url) => url.hostname === "engine.test");
  assert.equal(engineCall.searchParams.get("title"), "Du hast");
  assert.equal(engineCall.searchParams.get("isrc"), "GBAAA0000001");

  const appleNet = network();
  const byApple = await call(
    service(appleNet).similarHandler,
    similarPath({ apple_id: "1446013598", storefront: "de", title: "x", artist: "y", platform: "apple", k: "3" }),
  );
  assert.equal(byApple.body.query.title, "99 Luftballons");

  const isrcNet = network();
  // A name with its ISRC: the name is used as typed, the ISRC goes to the engine.
  const byIsrc = await call(service(isrcNet).similarHandler, similarPath({ isrc: "DEC760600298", title: "x", artist: "y" }));
  assert.equal(byIsrc.body.query.title, "x");
  assert.equal(byIsrc.body.query.isrc, "DEC760600298");
  assert.equal(isrcNet.calls.find((url) => url.hostname === "engine.test").searchParams.get("isrc"), "DEC760600298");
  assert.equal(isrcNet.calls.some((url) => url.pathname.startsWith("/track/isrc")), false);

  // When the link cannot be read, the typed name still finds the song.
  for (const broken of [() => reply(EMBED_MISSING_HTML), () => reply("down", 503)]) {
    const down = network({ "open.spotify.com": broken });
    const byName = await call(service(down).similarHandler, similarPath(all));
    assert.equal(byName.status, 200);
    assert.equal(byName.body.query.title, "Typed Title");
    assert.equal(byName.body.query.artist, "Typed Artist");
    assert.equal(byName.body.query.platform, "spotify");
    assert.equal(byName.body.query.source_url, null);
    assert.equal(down.calls.find((url) => url.hostname === "engine.test").searchParams.get("title"), "Typed Title");
  }
  // A link that is not a song stays an error, whatever else is sent.
  const wrong = await call(similarHandler, similarPath({ url: "https://example.com/x", title: "a", artist: "b" }));
  assert.equal(wrong.status, 400);
  assert.equal(wrong.body.error.code, "invalid_link");
});

test("the engine's match is always a whole number", async () => {
  const engine = structuredClone(ENGINE);
  engine.results.main[0].match = 94.6;
  delete engine.results.main[1].match;
  const { body } = await call(service(network({ "engine.test": () => reply(engine) })).similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.deepEqual(body.results.map((row) => row.match).slice(0, 2), [95, null]);
});

test("songs that cannot be resolved are 404, upstream failures 502", async () => {
  const cases = [
    [{ "open.spotify.com": () => reply(EMBED_MISSING_HTML) }, { url: SPOTIFY_LINK }, 404, "track_not_found"],
    [{ "open.spotify.com": () => reply("gone", 404) }, { url: SPOTIFY_LINK }, 404, "track_not_found"],
    [{ "open.spotify.com": () => reply("<html>new layout</html>") }, { url: SPOTIFY_LINK }, 502, "resolver_unavailable"],
    [{ "open.spotify.com": () => reply("down", 503) }, { url: SPOTIFY_LINK }, 502, "resolver_unavailable"],
    [{ "open.spotify.com": (_, options) => hang(options) }, { url: SPOTIFY_LINK }, 502, "resolver_unavailable"],
    [{ "itunes.apple.com": () => reply({ resultCount: 0, results: [] }) }, { url: APPLE_LINK }, 404, "track_not_found"],
    [{ "itunes.apple.com": () => reply("slow down", 403) }, { url: APPLE_LINK }, 502, "resolver_unavailable"],
    [{ "api.deezer.com": () => reply({ error: { code: 800 } }) }, { isrc: "DEC760600298" }, 404, "track_not_found"],
    [{ "api.deezer.com": () => reply({ error: { code: 4 } }) }, { isrc: "DEC760600298" }, 502, "resolver_unavailable"],
    [{ "engine.test": () => reply({ ok: false, error: "song not found" }, 404) }, { url: SPOTIFY_LINK }, 404, "song_not_found"],
    [{ "engine.test": () => reply({ ok: false, error: "undecodable" }, 422) }, { url: SPOTIFY_LINK }, 404, "song_not_found"],
    [{ "engine.test": () => reply("bad gateway", 502) }, { url: SPOTIFY_LINK }, 502, "engine_unavailable"],
    [{ "engine.test": () => reply({ ok: true }) }, { url: SPOTIFY_LINK }, 502, "engine_unavailable"],
    [{ "engine.test": () => { throw new TypeError("connection refused"); } }, { url: SPOTIFY_LINK }, 502, "engine_unavailable"],
    [{ "engine.test": (_, options) => hang(options) }, { url: SPOTIFY_LINK }, 502, "engine_unavailable"],
  ];
  for (const [routes, params, status, code] of cases) {
    const { similarHandler } = service(network(routes));
    const answer = await call(similarHandler, similarPath(params));
    assert.equal(answer.status, status, code);
    assert.equal(answer.body.error.code, code);
  }

  const busy = service(network({ "engine.test": () => reply({ ok: false }, 503) }));
  const waiting = await call(busy.similarHandler, similarPath({ url: SPOTIFY_LINK }));
  assert.equal(waiting.status, 503);
  assert.equal(waiting.body.error.code, "engine_busy");
  assert.equal(waiting.headers["retry-after"], "5");
  const limited = service(network({ "engine.test": () => reply({ ok: false }, 429, { "retry-after": "12" }) }));
  const slow = await call(limited.similarHandler, similarPath({ url: SPOTIFY_LINK }));
  assert.equal(slow.status, 429);
  assert.equal(slow.body.error.code, "rate_limited");
  assert.equal(slow.headers["retry-after"], "12");
});

test("a failed engine answer is not remembered", async () => {
  let down = true;
  const net = network({ "engine.test": () => (down ? reply("down", 502) : reply(ENGINE)) });
  const { similarHandler } = service(net);
  assert.equal((await call(similarHandler, similarPath({ url: SPOTIFY_LINK }))).status, 502);
  down = false;
  assert.equal((await call(similarHandler, similarPath({ url: SPOTIFY_LINK }))).status, 200);
});

test("failed or slow lookups degrade to search links and nulls", async () => {
  // Deezer and iTunes Search both fail.
  const failing = network({
    "api.deezer.com": () => reply("down", 500),
    "itunes.apple.com": (url) => (url.pathname === "/lookup" ? reply(ITUNES_LOOKUP) : reply("down", 503)),
  });
  const failed = await call(service(failing).similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  assert.equal(failed.status, 200);
  assert.equal(failed.body.query.isrc, null);
  for (const row of failed.body.results) {
    assert.equal(row.isrc, null);
    assert.equal(row.artwork, null);
    assert.equal(row.preview_url, null);
    assert.equal(row.apple_exact, false);
    assert.match(row.apple_music_url, /^https:\/\/music\.apple\.com\/de\/search\?term=/);
    assert.equal(row.spotify_kind, "search");
    assert.equal(row.spotify_url, `https://open.spotify.com/search/${encodeURIComponent(`${row.artist} ${row.title}`)}`);
  }

  // Both never answer: the request still returns when the time budget ends.
  const hanging = network({
    "api.deezer.com": (_, options) => hang(options),
    "itunes.apple.com": (url, options) => (url.pathname === "/lookup" ? reply(ITUNES_LOOKUP) : hang(options)),
  });
  const started = Date.now();
  const late = await call(service(hanging, { budgetMs: 60 }).similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  assert.ok(Date.now() - started < 1500, "returns at the budget, not at the upstream timeout");
  assert.equal(late.status, 200);
  assert.equal(late.body.results.length, 3);
  assert.equal(late.headers["cache-control"], "public, max-age=20", "an incomplete answer is not kept for long");
  assert.equal(late.body.results.every((row) => row.preview_url === null && !row.apple_exact), true);

  // Only iTunes hangs: Deezer's details still arrive.
  const half = network({
    "itunes.apple.com": (url, options) => (url.pathname === "/lookup" ? reply(ITUNES_LOOKUP) : hang(options)),
  });
  const partial = await call(service(half, { budgetMs: 60 }).similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  assert.equal(partial.body.results[0].preview_url, DEEZER_TRACK.preview);
  assert.equal(partial.body.results[0].apple_exact, false);
});

test("when Deezer cannot be asked, what is remembered about a song is still used", async () => {
  let clock = 1_000_000;
  let down = false;
  const net = network({
    "api.deezer.com": (url) => (down ? reply("down", 500) : reply(deezerTrack(url.pathname.split("/").pop()))),
  });
  const { similarHandler } = service(net, { now: () => clock });
  const fresh = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(fresh.body.results[0].preview_url, DEEZER_TRACK.preview);

  clock += 20 * 60 * 1000; // previews and the engine answer have expired
  down = true;
  const later = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(later.status, 200);
  const [first] = later.body.results;
  assert.equal(first.preview_url, null, "an old preview address is never served");
  assert.equal(first.isrc, DEEZER_TRACK.isrc);
  assert.equal(first.artwork, DEEZER_TRACK.album.cover_medium);
  assert.equal(first.spotify_kind, "isrc_search");
  assert.equal(later.headers["cache-control"], "public, max-age=20");
});

test("untrusted media addresses from Deezer are dropped", async () => {
  const net = network({
    "api.deezer.com": () =>
      reply({ ...DEEZER_TRACK, preview: "https://evil.example/a.mp3", album: { cover_medium: "http://cdn-images.dzcdn.net/x.jpg" } }),
  });
  const { body } = await call(service(net).similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(body.results[0].preview_url, null);
  assert.equal(body.results[0].artwork, null);
});

test("iTunes Search calls are rationed and the link endpoint says when to retry", async () => {
  let clock = 1_000_000;
  const net = network();
  const { linkHandler } = service(net, { now: () => clock });
  const linkPath = (id, extra = {}) =>
    `/api/ratings?__soundalike_handler=mobile-link&${new URLSearchParams({ deezer_id: String(id), platform: "apple", storefront: "de", ...extra })}`;

  const first = await call(linkHandler, linkPath(FIRST_ID));
  assert.equal(first.status, 200);
  assert.deepEqual(Object.keys(first.body), ["ok", "deezer_id", "platform", "url", "exact", "kind", "storefront", "retry_after"]);
  assert.equal(first.body.exact, true);
  assert.match(first.body.url, /^https:\/\/music\.apple\.com\/de\/album\//);
  assert.equal(first.body.retry_after, null);
  assert.equal(first.headers["cache-control"], "public, max-age=86400");

  // The same song again costs nothing.
  const searches = () => net.calls.filter((url) => url.pathname === "/search").length;
  await call(linkHandler, linkPath(FIRST_ID));
  assert.equal(searches(), 1);

  // 18 calls a minute: the 19th different song inside the minute is refused.
  for (let id = 1; id <= 17; id += 1) await call(linkHandler, linkPath(id));
  assert.equal(searches(), 18);
  const refused = await call(linkHandler, linkPath(900));
  assert.equal(refused.status, 200);
  assert.equal(refused.body.exact, false);
  assert.match(refused.body.url, /^https:\/\/music\.apple\.com\/de\/search\?term=/);
  assert.ok(refused.body.retry_after >= 1 && refused.body.retry_after <= 4);
  assert.equal(refused.headers["cache-control"], "no-store, max-age=0");
  assert.equal(searches(), 18);

  clock += 4000;
  const later = await call(linkHandler, linkPath(900));
  assert.equal(later.body.retry_after, null);
  assert.equal(searches(), 19);
});

test("the link endpoint: Spotify without credentials, bad input, missing songs", async () => {
  const net = network();
  const { linkHandler } = service(net);
  const base = "/api/ratings?__soundalike_handler=mobile-link&";
  const spotify = await call(linkHandler, `${base}deezer_id=${FIRST_ID}&platform=spotify`);
  assert.equal(spotify.status, 200);
  assert.equal(spotify.body.exact, false);
  assert.equal(spotify.body.url, `https://open.spotify.com/search/isrc%3A${DEEZER_TRACK.isrc}`);
  assert.equal(spotify.body.kind, "isrc_search");
  assert.equal(spotify.body.storefront, null);
  assert.equal(net.count("accounts.spotify.com"), 0);

  for (const query of ["platform=apple", "deezer_id=0&platform=apple", "deezer_id=abc&platform=apple", `deezer_id=${FIRST_ID}`, `deezer_id=${FIRST_ID}&platform=both`]) {
    const bad = await call(linkHandler, base + query);
    assert.equal(bad.status, 400, query);
    assert.equal(bad.body.error.code, "invalid_parameter");
  }

  const missing = service(network({ "api.deezer.com": () => reply({ error: { code: 800 } }) }));
  assert.equal((await call(missing.linkHandler, `${base}deezer_id=5&platform=apple`)).status, 404);

  // Deezer down: with the row's title and artist the client still gets a link.
  const down = service(network({ "api.deezer.com": () => reply("down", 500) }));
  const blind = await call(down.linkHandler, `${base}deezer_id=5&platform=spotify`);
  assert.equal(blind.status, 502);
  assert.equal(blind.body.error.code, "lookup_unavailable");
  const named = await call(down.linkHandler, `${base}deezer_id=5&platform=spotify&title=Du+hast&artist=Rammstein`);
  assert.equal(named.status, 200);
  assert.equal(named.body.url, "https://open.spotify.com/search/Rammstein%20Du%20hast");
  assert.equal(named.body.kind, "search");
});

test("with Spotify client credentials results carry exact track links", async () => {
  const exactId = "1a2B3c4D5e6F7g8H9i0J1k";
  const seen = [];
  const net = network({
    "accounts.spotify.com": (_, options) => {
      assert.equal(options.method, "POST");
      assert.equal(options.headers.Authorization, `Basic ${Buffer.from("client:secret").toString("base64")}`);
      return reply({ access_token: "token-1", expires_in: 3600 });
    },
    "api.spotify.com": (url, options) => {
      assert.equal(options.headers.Authorization, "Bearer token-1");
      if (url.pathname === `/v1/tracks/${SPOTIFY_ID}`) return reply({ external_ids: { isrc: "DED579704010" } });
      const query = url.searchParams.get("q");
      seen.push(query);
      if (query === `isrc:${DEEZER_TRACK.isrc}`) {
        return reply({ tracks: { items: [{ id: exactId, name: "Gekreuzigt 2006", artists: [{ name: "OOMPH!" }] }] } });
      }
      if (query.startsWith("isrc:")) return reply({ tracks: { items: [] } });
      if (query === "track:1000 Flammen artist:Eisbrecher") {
        return reply({
          tracks: {
            items: [
              { id: "0000000000000000000000", name: "Something Else", artists: [{ name: "Eisbrecher" }] },
              { id: "9z8Y7x6W5v4U3t2S1r0Q9p", name: "1000 Flammen", artists: [{ name: "Eisbrecher" }], duration_ms: 217000 },
            ],
          },
        });
      }
      return reply({ tracks: { items: [] } });
    },
  });
  const { similarHandler } = service(net, { env: { SPOTIFY_CLIENT_ID: "client", SPOTIFY_CLIENT_SECRET: "secret" } });
  const { status, body } = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(status, 200);
  assert.equal(body.query.isrc, "DED579704010");
  assert.equal(net.calls.find((url) => url.hostname === "engine.test").searchParams.get("isrc"), "DED579704010");
  assert.deepEqual(
    body.results.map((row) => [row.spotify_exact, row.spotify_url]),
    [
      [true, `https://open.spotify.com/track/${exactId}`],
      [true, "https://open.spotify.com/track/9z8Y7x6W5v4U3t2S1r0Q9p"],
      [false, "https://open.spotify.com/search/isrc%3ADEA620000001"],
    ],
  );
  assert.deepEqual(body.links, {
    spotify_exact: 2,
    spotify_isrc: 1,
    apple_exact: 0,
    spotify_lookup: true,
    spotify_source: "credentials",
    apple_eager_rows: 5,
  });
  assert.equal(net.count("accounts.spotify.com"), 1, "one token serves every lookup");
  assert.ok(seen.includes("track:Willkommen artist:Unheilig"));

  // Credentials that Spotify rejects fall back to search links.
  const rejected = network({ "accounts.spotify.com": () => reply({ error: "invalid_client" }, 400) });
  const fallback = service(rejected, { env: { SPOTIFY_CLIENT_ID: "client", SPOTIFY_CLIENT_SECRET: "wrong" } });
  const answer = await call(fallback.similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(answer.status, 200);
  assert.equal(answer.body.results.every((row) => !row.spotify_exact && row.spotify_url.includes("/search/")), true);
  assert.equal(rejected.count("api.spotify.com"), 0);
});

test("short links are followed only on Spotify's hosts", async () => {
  const hops = {
    "spotify.link": () => reply("", 307, { location: "https://spotify.app.link/AbC?_p=1" }),
    "spotify.app.link": () => reply("", 307, { location: `https://open.spotify.com/track/${SPOTIFY_ID}?si=zz` }),
  };
  const net = network(hops);
  const good = await call(service(net).similarHandler, similarPath({ url: "https://spotify.link/AbC", k: "3" }));
  assert.equal(good.status, 200);
  assert.equal(good.body.query.source_url, `https://open.spotify.com/track/${SPOTIFY_ID}`);
  assert.equal(good.body.query.platform, "spotify");

  // A landing page that names the track in its markup.
  const page = network({
    "spotify.link": () => reply(`<a href="https:\\/\\/open.spotify.com\\/track\\/${SPOTIFY_ID}?si=1">Open</a>`),
  });
  assert.equal((await call(service(page).similarHandler, similarPath({ url: "https://spotify.link/AbC" }))).status, 200);

  const cases = [
    [{ "spotify.link": () => reply("", 307, { location: "https://open.spotify.com/" }) }, 400, "unsupported_link"],
    [{ "spotify.link": () => reply("", 307, { location: "https://open.spotify.com/album/6eUW0wxWtzkFdaEFsTJto6" }) }, 400, "unsupported_link"],
    [{ "spotify.link": () => reply("", 302, { location: "https://evil.example/steal" }) }, 400, "unsupported_link"],
    [{ "spotify.link": () => reply("", 302, { location: "http://169.254.169.254/latest" }) }, 400, "unsupported_link"],
    [{ "spotify.link": () => reply("", 302, { location: "https://spotify.link/AbC" }) }, 400, "unsupported_link"],
    [{ "spotify.link": () => reply("not found", 404) }, 400, "unsupported_link"],
    [{ "spotify.link": () => { throw new TypeError("network"); } }, 502, "resolver_unavailable"],
  ];
  for (const [routes, status, code] of cases) {
    const stub = network(routes);
    const answer = await call(service(stub).similarHandler, similarPath({ url: "https://spotify.link/AbC" }));
    assert.equal(answer.status, status, code);
    assert.equal(answer.body.error.code, code);
    assert.equal(stub.calls.every((url) => url.hostname === "spotify.link"), true, "never leaves the short-link host");
  }
});

test("the engine is asked directly with the listener's address when it is on the LAN", async () => {
  const seen = [];
  const net = network({
    "engine.test": (_, options) => {
      seen.push(options.headers["CF-Connecting-IP"]);
      return reply(ENGINE);
    },
    "soundalike.yassin.app": (_, options) => {
      seen.push(options.headers["CF-Connecting-IP"]);
      return reply(ENGINE);
    },
  });
  const headers = { "cf-connecting-ip": "203.0.113.7" };
  await call(service(net).similarHandler, similarPath({ url: SPOTIFY_LINK }), { headers });
  await call(service(net).similarHandler, similarPath({ url: SPOTIFY_LINK }), { headers: { "cf-connecting-ip": "bad header\n" } });
  // The default engine address is the public one, where Cloudflare sets it.
  const publicService = createMobileService({ fetch: net.fetch, env: {}, budgetMs: 100 });
  await call(publicService.similarHandler, similarPath({ url: SPOTIFY_LINK }), { headers });
  assert.deepEqual(seen, ["203.0.113.7", undefined, undefined]);
  assert.equal(net.count("soundalike.yassin.app"), 1);
});

test("site.json routes both endpoints through the shared dispatcher", async () => {
  const config = JSON.parse(readFileSync(new URL("../site.json", import.meta.url), "utf8"));
  const destination = (source) => config.rewrites.find((rewrite) => rewrite.source === source)?.destination;
  assert.equal(destination("/api/mobile/similar"), "/api/ratings?__soundalike_handler=mobile-similar");
  assert.equal(destination("/api/mobile/link"), "/api/ratings?__soundalike_handler=mobile-link");

  const net = network();
  const mobile = service(net);
  const dispatch = createDispatchHandler(
    () => assert.fail("the ratings handler must not see mobile requests"),
    { "mobile-similar": async () => mobile.similarHandler, "mobile-link": async () => mobile.linkHandler },
  );
  const similar = await call(dispatch, `${destination("/api/mobile/similar")}&${new URLSearchParams({ url: SPOTIFY_LINK, k: "3" })}`);
  assert.equal(similar.status, 200);
  assert.equal(similar.body.results.length, 3);
  const link = await call(dispatch, `${destination("/api/mobile/link")}&deezer_id=${FIRST_ID}&platform=spotify`);
  assert.equal(link.status, 200);
  assert.equal(link.body.platform, "spotify");
});

test("TtlCache expires and evicts; the bucket refills over its window", () => {
  let clock = 0;
  const cache = new TtlCache(2, () => clock);
  cache.set("a", 1, 100);
  cache.set("b", 2, 100);
  cache.set("c", 3, 100);
  assert.equal(cache.get("a"), undefined);
  assert.equal(cache.get("b"), 2);
  clock = 100;
  assert.equal(cache.get("b"), undefined);

  const bucket = createBucket(2, 1000, () => clock);
  assert.equal(bucket.take(), true);
  assert.equal(bucket.take(), true);
  assert.equal(bucket.take(), false);
  assert.equal(bucket.retryAfter(), 1);
  assert.equal(bucket.take(true), true);
  clock += 500;
  assert.equal(bucket.take(), false, "a forced call is paid back first");
  clock += 500;
  assert.equal(bucket.take(), true);
});

// ---- Spotify track ids without credentials ----------------------------------

const lookups = (net) => net.calls.filter((url) => url.hostname === "labs.api.listenbrainz.org").length;
const embeds = (net) => net.calls.filter((url) => url.pathname.startsWith("/embed/track/")).map((url) => url.pathname.split("/").pop());
const trackRows = (body) => body.results.map((row) => [row.spotify_kind, row.spotify_exact, row.spotify_id, row.spotify_url]);

test("without credentials one ListenBrainz call gives the rows their Spotify tracks", async () => {
  const sent = [];
  const routes = trackLookup();
  const net = network({
    ...routes,
    "labs.api.listenbrainz.org": (url, options) => {
      assert.equal(url.pathname, "/spotify-id-from-metadata/json");
      assert.equal(options.method, "POST");
      assert.equal(options.headers["Content-Type"], "application/json");
      assert.match(options.headers["User-Agent"], /^soundalike-mobile-api\/\d+\.\d+ \(\+https:\/\/soundalike\.yassin\.app; /);
      sent.push(JSON.parse(options.body));
      return routes["labs.api.listenbrainz.org"](url, options);
    },
  });
  const { similarHandler } = service(net);
  const { status, headers, body } = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(status, 200);
  assert.deepEqual(trackRows(body), [
    ["track", true, OOMPH_IDS[0], `https://open.spotify.com/track/${OOMPH_IDS[0]}`],
    ["track", true, EISBRECHER_ID, `https://open.spotify.com/track/${EISBRECHER_ID}`],
    ["isrc_search", false, null, "https://open.spotify.com/search/isrc%3ADEA620000001"],
  ]);
  assert.deepEqual(body.links, {
    spotify_exact: 2,
    spotify_isrc: 1,
    apple_exact: 0,
    spotify_lookup: true,
    spotify_source: "metadata",
    apple_eager_rows: 5,
  });
  assert.equal(headers["cache-control"], "public, max-age=300");
  // One call for the whole list. The seed came with its Spotify id, so it is not asked about.
  assert.deepEqual(sent, [
    [
      { artist_name: "OOMPH!", release_name: DEEZER_TRACK.album.title, track_name: "Gekreuzigt 2006" },
      { artist_name: "Eisbrecher", release_name: DEEZER_TRACK.album.title, track_name: "1000 Flammen" },
      { artist_name: "Unheilig", release_name: DEEZER_TRACK.album.title, track_name: "Willkommen" },
    ],
  ]);
  // Only each row's first candidate is checked on Spotify's embed page.
  assert.deepEqual(embeds(net).sort(), [SPOTIFY_ID, OOMPH_IDS[0], EISBRECHER_ID].sort());

  // Hits, misses and checks are remembered: the same answer again asks nobody.
  const before = net.calls.length;
  const again = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.deepEqual(again.body, body);
  assert.equal(net.calls.length, before);
});

test("a seed that came without a Spotify id is looked up with the rows", async () => {
  const sent = [];
  const routes = trackLookup({ "2BYgWai3FjLyyikzlzKfQf": EMBED_HTML.replaceAll("Du hast", "99 Luftballons") });
  const net = network({
    ...routes,
    "labs.api.listenbrainz.org": (url, options) => {
      sent.push(JSON.parse(options.body));
      return routes["labs.api.listenbrainz.org"](url, options);
    },
  });
  const { body } = await call(service(net, { appleEagerRows: 0 }).similarHandler, similarPath({ url: APPLE_LINK, platform: "both", k: "3" }));
  assert.equal(body.query.platform, "both");
  assert.equal(body.query.spotify_id, "2BYgWai3FjLyyikzlzKfQf");
  assert.equal(sent.length, 1);
  assert.deepEqual(sent[0].at(-1), { artist_name: "Nena", release_name: "", track_name: "99 Luftballons" });
  assert.equal(body.results[0].spotify_id, OOMPH_IDS[0]);

  // An Apple-only answer needs no Spotify links, so nothing is asked.
  const appleOnly = network(trackLookup());
  const apple = await call(service(appleOnly).similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  assert.equal(lookups(appleOnly), 0);
  assert.equal(apple.body.query.spotify_id, null);
  assert.equal(apple.body.results.every((row) => row.spotify_id === null), true);
});

test("a candidate that is not the song is dropped, and the link endpoint checks the others", async () => {
  const other = EMBED_HTML; // "Du hast": another song
  const unplayable = CANDIDATE_PAGES[EISBRECHER_ID].replace('"isPlayable":true', '"isPlayable":false');
  assert.notEqual(unplayable, CANDIDATE_PAGES[EISBRECHER_ID]);
  const net = network(trackLookup({ [OOMPH_IDS[0]]: other, [EISBRECHER_ID]: unplayable }));
  const { similarHandler, linkHandler } = service(net);
  const { body } = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.deepEqual(trackRows(body).map((row) => row.slice(0, 3)), [
    ["isrc_search", false, null],
    ["isrc_search", false, null],
    ["isrc_search", false, null],
  ]);

  const base = "/api/ratings?__soundalike_handler=mobile-link&";
  const harder = await call(linkHandler, `${base}deezer_id=${FIRST_ID}&platform=spotify`);
  assert.equal(harder.status, 200);
  assert.deepEqual(harder.body, {
    ok: true,
    deezer_id: FIRST_ID,
    platform: "spotify",
    url: `https://open.spotify.com/track/${OOMPH_IDS[1]}`,
    exact: true,
    kind: "track",
    spotify_id: OOMPH_IDS[1],
    storefront: null,
    retry_after: null,
  });
  assert.equal(harder.headers["cache-control"], "public, max-age=86400");
  // The next answer has it too.
  const next = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(next.body.results[0].spotify_id, OOMPH_IDS[1]);
  assert.equal(next.body.results[1].spotify_id, null);
});

test("of several candidates the link endpoint picks the closest in length, and never another version", async () => {
  const page = (id, changes) => {
    let html = CANDIDATE_PAGES[id];
    for (const [from, to] of changes) {
      assert.ok(html.includes(from), from);
      html = html.replaceAll(from, to);
    }
    return html;
  };
  // Deezer says 217 s. The first candidate is 215.1 s, the second 216.1 s.
  const closest = network(trackLookup());
  const base = "/api/ratings?__soundalike_handler=mobile-link&";
  const picked = await call(service(closest).linkHandler, `${base}deezer_id=${FIRST_ID}&platform=spotify`);
  assert.equal(picked.body.spotify_id, OOMPH_IDS[1]);
  assert.deepEqual(embeds(closest).sort(), [...OOMPH_IDS].sort());

  // A live recording is another version, however close its length.
  const live = network(
    trackLookup({ [OOMPH_IDS[1]]: page(OOMPH_IDS[1], [['"name":"Gekreuzigt 2006"', '"name":"Gekreuzigt 2006 - Live"']]) }),
  );
  const studio = await call(service(live).linkHandler, `${base}deezer_id=${FIRST_ID}&platform=spotify`);
  assert.equal(studio.body.spotify_id, OOMPH_IDS[0]);

  // Nothing known: the ISRC search, as before.
  const none = network();
  const fallback = await call(service(none).linkHandler, `${base}deezer_id=${FIRST_ID}&platform=spotify`);
  assert.deepEqual(
    [fallback.body.kind, fallback.body.exact, fallback.body.spotify_id, fallback.body.url],
    ["isrc_search", false, null, `https://open.spotify.com/search/isrc%3A${DEEZER_TRACK.isrc}`],
  );
});

test("a check that is slow does not hold the answer: the first candidate stands until it is checked", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const slow = async (url) => {
    await gate;
    return reply(CANDIDATE_PAGES[url.pathname.split("/").pop()]);
  };
  const net = network(trackLookup({ [OOMPH_IDS[0]]: slow, [EISBRECHER_ID]: slow }));
  const { similarHandler } = service(net, { verifyWaitMs: 30 });
  const started = Date.now();
  const first = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.ok(Date.now() - started < 400, "did not wait for the embed pages");
  assert.deepEqual(first.body.results.map((row) => row.spotify_id), [OOMPH_IDS[0], EISBRECHER_ID, null]);
  assert.equal(first.headers["cache-control"], "public, max-age=20", "an unchecked answer is not kept for long");

  release();
  await new Promise((resolve) => setTimeout(resolve, 20));
  const before = net.calls.length;
  const second = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(second.headers["cache-control"], "public, max-age=300");
  assert.equal(net.calls.length, before, "the check finished in the background");
});

test("ListenBrainz failing, throttling or answering nonsense leaves search links and is not hammered", async () => {
  const fallback = [
    ["isrc_search", false, null],
    ["isrc_search", false, null],
    ["isrc_search", false, null],
  ];
  const kinds = (body) => trackRows(body).map((row) => row.slice(0, 3));
  for (const broken of [
    () => reply("not json"),
    () => reply([{ spotify_track_ids: [OOMPH_IDS[0]] }]), // not one answer per question
    () => reply({ error: "no" }, 400),
    () => { throw new TypeError("network"); },
    (_, options) => hang(options),
  ]) {
    const net = network({ ...trackLookup(), "labs.api.listenbrainz.org": broken });
    const answer = await call(service(net).similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
    assert.equal(answer.status, 200);
    assert.deepEqual(kinds(answer.body), fallback);
  }
  // Ids that are not Spotify track ids are ignored.
  const odd = network({
    ...trackLookup(),
    "labs.api.listenbrainz.org": (_, options) =>
      reply(JSON.parse(options.body).map(() => ({ spotify_track_ids: ["javascript:alert(1)", 7, null, "short"] }))),
  });
  assert.deepEqual(kinds((await call(service(odd).similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }))).body), fallback);

  // 429 and 5xx pause the lookups: for Retry-After, else for a minute.
  for (const [response, pauseMs] of [
    [() => reply("slow down", 429, { "retry-after": "120" }), 120_000],
    [() => reply("down", 503), 60_000],
  ]) {
    let clock = 1_000_000;
    let failing = true;
    const routes = trackLookup();
    const net = network({
      ...routes,
      "labs.api.listenbrainz.org": (url, options) => (failing ? response() : routes["labs.api.listenbrainz.org"](url, options)),
    });
    const { similarHandler } = service(net, { now: () => clock });
    assert.deepEqual(kinds((await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }))).body), fallback);
    assert.equal(lookups(net), 1);
    failing = false;
    clock += pauseMs - 1000;
    assert.deepEqual(kinds((await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }))).body), fallback);
    assert.equal(lookups(net), 1, "not asked again while paused");
    clock += 2000;
    const resumed = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
    assert.equal(lookups(net), 2);
    assert.equal(resumed.body.results[0].spotify_id, OOMPH_IDS[0]);
  }
});

test("Spotify refusing its embed pages pauses the checks and the first candidate stands", async () => {
  let clock = 1_000_000;
  const refuse = () => reply("too many", 429);
  const net = network(trackLookup({ [OOMPH_IDS[0]]: refuse, [EISBRECHER_ID]: refuse }));
  const { similarHandler } = service(net, { now: () => clock });
  const first = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.deepEqual(first.body.results.map((row) => row.spotify_id), [OOMPH_IDS[0], EISBRECHER_ID, null]);
  assert.equal(first.headers["cache-control"], "public, max-age=20");
  const asked = embeds(net).length;
  clock += 30_000;
  await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(embeds(net).length, asked, "no embed page is fetched while paused");
});

test("with credentials Spotify's own search comes first and ListenBrainz gets only the rest", async () => {
  const sent = [];
  const routes = trackLookup();
  const net = network({
    ...routes,
    "accounts.spotify.com": () => reply({ access_token: "token-1", expires_in: 3600 }),
    "api.spotify.com": (url) => {
      if (url.pathname.startsWith("/v1/tracks/")) return reply({ external_ids: {} });
      const found = url.searchParams.get("q") === `isrc:${DEEZER_TRACK.isrc}`;
      return reply({ tracks: { items: found ? [{ id: "1a2B3c4D5e6F7g8H9i0J1k", name: "Gekreuzigt 2006", artists: [{ name: "OOMPH!" }] }] : [] } });
    },
    "labs.api.listenbrainz.org": (url, options) => {
      sent.push(JSON.parse(options.body).map((query) => query.artist_name));
      return routes["labs.api.listenbrainz.org"](url, options);
    },
  });
  const { similarHandler } = service(net, { env: { SPOTIFY_CLIENT_ID: "client", SPOTIFY_CLIENT_SECRET: "secret" } });
  const { body } = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.deepEqual(body.results.map((row) => row.spotify_id), ["1a2B3c4D5e6F7g8H9i0J1k", EISBRECHER_ID, null]);
  assert.deepEqual(sent, [["Eisbrecher", "Unheilig"]]);
  assert.equal(body.links.spotify_source, "credentials");
});

// ---- A shared song the engine does not find by name -------------------------

// Recorded on 2026-11-23: Jay Chou's 晴天 on Spotify (the embed page in English
// and in Chinese) and on Deezer, which lists it as "Sunny Day": the artist
// search, his top tracks in Deezer's default naming and for "zh-TW,ja" (cut
// to the three of the song's length and two others), a search, three tracks.
const JAY_ID = "5pIcwtJYNJx93l420oR2Vm";
const JAY_LINK = `https://open.spotify.com/track/${JAY_ID}?si=abc`;
const JAY_EMBED = { en: fixture(`spotify-embed-${JAY_ID}.en.html`), native: fixture(`spotify-embed-${JAY_ID}.zh-TW.html`) };
const JAY_DEEZER = JSON.parse(fixture("deezer-seed-fallback.json"));
const SUNNY_DAY_ID = 2602471112;

/** Spotify and Deezer as recorded for 晴天; `engine(params)` answers the engine's requests. */
function jayChou(engine, { topLocal = JAY_DEEZER.topLocal } = {}) {
  return {
    "open.spotify.com": (_, options) => reply(options.headers["Accept-Language"] === "en" ? JAY_EMBED.en : JAY_EMBED.native),
    "engine.test": (url) => engine(Object.fromEntries(url.searchParams)),
    "api.deezer.com": (url, options) => {
      const language = options.headers?.["Accept-Language"];
      if (url.pathname === "/search/artist") {
        return reply({ data: url.searchParams.get("q") === "Jay Chou" ? JAY_DEEZER.artists : JAY_DEEZER.artistsNative });
      }
      if (url.pathname === "/artist/69175/top") return reply({ data: language ? topLocal : JAY_DEEZER.top });
      if (url.pathname.startsWith("/artist/")) return reply({ data: [] });
      if (url.pathname === "/search") return reply({ data: JAY_DEEZER.search });
      const id = url.pathname.split("/").pop();
      return reply(JAY_DEEZER.tracks[id] ?? deezerTrack(id));
    },
  };
}

const notFound = () => reply({ ok: false, error: "song not found" }, 404);
const engineCalls = (net) =>
  net.calls.filter((url) => url.hostname === "engine.test").map((url) => url.searchParams.get("id") ?? `${url.searchParams.get("title")} / ${url.searchParams.get("artist")}`);
const deezerPaths = (net) => net.calls.filter((url) => url.hostname === "api.deezer.com").map((url) => url.pathname);

test("a song Deezer lists under another title is found among its artist's tracks and asked for by id", async () => {
  let clock = 1_000_000;
  const net = network(jayChou((params) => (params.id === String(SUNNY_DAY_ID) ? reply(ENGINE) : notFound())));
  const { similarHandler } = service(net, { now: () => clock });
  const { status, body } = await call(similarHandler, similarPath({ url: JAY_LINK, k: "3" }));
  assert.equal(status, 200);
  assert.equal(body.query.title, "晴天", "the song is shown as it was shared");
  assert.equal(body.query.artist, "Jay Chou");
  assert.equal(body.query.spotify_id, JAY_ID);
  assert.equal(body.results.length, 3);
  assert.deepEqual(engineCalls(net), ["晴天 / Jay Chou", String(SUNNY_DAY_ID)]);
  // His tracks were listed in Deezer's naming and in the title's languages.
  const tops = net.calls.filter((url) => url.pathname === "/artist/69175/top");
  assert.equal(tops.length, 2);
  assert.ok(net.calls.some((url) => url.pathname === "/search/artist" && url.searchParams.get("q") === "周杰倫"));

  // The link's recording is remembered: later the engine is asked by id at once.
  clock += 60 * 60 * 1000;
  const before = net.calls.length;
  const later = await call(similarHandler, similarPath({ url: JAY_LINK, k: "3" }));
  assert.equal(later.status, 200);
  assert.deepEqual(engineCalls(net).slice(2), [String(SUNNY_DAY_ID)]);
  const since = net.calls.slice(before);
  assert.equal(since.some((url) => url.pathname.startsWith("/artist/") || url.pathname.startsWith("/search")), false);
});

test("without evidence the song stays not found, and that is remembered", async () => {
  // Deezer has no title for it in the title's language: three of his tracks
  // have the length, and nothing says which one it is.
  const net = network(jayChou(notFound, { topLocal: JAY_DEEZER.top }));
  const { similarHandler } = service(net);
  const first = await call(similarHandler, similarPath({ url: JAY_LINK, k: "3" }));
  assert.equal(first.status, 404);
  assert.equal(first.body.error.code, "song_not_found");
  // Asked by name, then once more with the artist as written in Chinese; never by a guessed id.
  assert.deepEqual(engineCalls(net), ["晴天 / Jay Chou", "晴天 / 周杰倫"]);

  const before = net.calls.length;
  const again = await call(similarHandler, similarPath({ url: JAY_LINK, k: "3" }));
  assert.equal(again.status, 404);
  assert.deepEqual(net.calls.slice(before).map((url) => url.hostname), ["engine.test"], "no second search of Deezer");
});

test("an artist Spotify romanises is tried under the name in the title's script", async () => {
  const page = (artist) => EMBED_HTML.replaceAll("Du hast", "Группа крови").replaceAll("Rammstein", artist);
  const net = network({
    "open.spotify.com": (_, options) => reply(page(options.headers["Accept-Language"] === "ru" ? "Кино" : "Kino")),
    "engine.test": (url) => (url.searchParams.get("artist") === "Кино" ? reply(ENGINE) : notFound()),
    // Deezer's releases of the song have other lengths, so no track has the evidence.
    "api.deezer.com": (url) => (url.pathname.startsWith("/track/") ? reply(deezerTrack(url.pathname.split("/").pop())) : reply({ data: [] })),
  });
  const { similarHandler } = service(net);
  const { status, body } = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(status, 200);
  assert.equal(body.query.artist, "Kino");
  assert.deepEqual(engineCalls(net), ["Группа крови / Kino", "Группа крови / Кино"]);
  assert.deepEqual(
    net.calls.filter((url) => url.pathname === "/search/artist").map((url) => url.searchParams.get("q")),
    ["Kino", "Кино"],
  );
});

test("an Apple Music song is found by its album and length", async () => {
  // iTunes: "99 Luftballons" on the album "99 Luftballons", 233 s. Deezer here has it as "99 Red Balloons".
  const top = (tracks) => tracks.map(([id, title, duration, album]) => ({ id, title, duration, rank: 1, artist: { id: 77, name: "Nena" }, album: { title: album } }));
  const routes = (tracks) => ({
    "engine.test": (url) => (url.searchParams.get("id") === "501" ? reply(ENGINE) : notFound()),
    "api.deezer.com": (url) => {
      if (url.pathname === "/search/artist") return reply({ data: [{ id: 77, name: "Nena" }] });
      if (url.pathname === "/artist/77/top") return reply({ data: top(tracks) });
      if (url.pathname === "/search") return reply({ data: [] });
      return reply(deezerTrack(url.pathname.split("/").pop()));
    },
  });
  const found = network(routes([[501, "99 Red Balloons", 234, "99 Luftballons"], [502, "Leuchtturm", 233, "Nena"]]));
  const answer = await call(service(found).similarHandler, similarPath({ url: APPLE_LINK, k: "3" }));
  assert.equal(answer.status, 200);
  assert.equal(answer.body.query.title, "99 Luftballons");
  assert.deepEqual(engineCalls(found), ["99 Luftballons / Nena", "501"]);

  // Another album, or the wrong length: no evidence.
  for (const tracks of [
    [[501, "99 Red Balloons", 234, "Nena feat. Nena"]],
    [[501, "99 Red Balloons", 240, "99 Luftballons"]],
  ]) {
    const none = network(routes(tracks));
    assert.equal((await call(service(none).similarHandler, similarPath({ url: APPLE_LINK, k: "3" }))).status, 404);
    assert.deepEqual(engineCalls(none), ["99 Luftballons / Nena"]);
  }
});

test("a song the engine finds by name costs nothing more, and typed names are never looked up", async () => {
  const net = network();
  await call(service(net).similarHandler, similarPath({ url: SPOTIFY_LINK, k: "3" }));
  assert.equal(engineCalls(net).length, 1);
  assert.equal(deezerPaths(net).some((path) => path.startsWith("/search") || path.startsWith("/artist/")), false);
  assert.equal(net.calls.filter((url) => url.pathname === `/embed/track/${SPOTIFY_ID}`).length, 1);

  // title+artist and isrc seeds have no length to check a recording against.
  const typed = network({ "engine.test": notFound });
  const answer = await call(service(typed).similarHandler, similarPath({ title: "晴天", artist: "Jay Chou" }));
  assert.equal(answer.status, 404);
  assert.deepEqual(typed.calls.map((url) => url.hostname), ["engine.test"]);
});

test("the engine being down during the fallback is an error, not a wrong answer", async () => {
  const net = network(jayChou((params) => (params.id ? reply("bad gateway", 502) : notFound())));
  const answer = await call(service(net).similarHandler, similarPath({ url: JAY_LINK, k: "3" }));
  assert.equal(answer.status, 502);
  assert.equal(answer.body.error.code, "engine_unavailable");
});

test("list= picks the melody list or what listeners also play from the same answer", async () => {
  const titles = (list) => list.slice(0, 2).map((row) => row.title);
  const engine = {
    ...ENGINE,
    results: {
      ...ENGINE.results,
      best: [...ENGINE.results.main].reverse(),
      listeners: ENGINE.results.main.slice(1),
    },
  };
  const net = network({ "engine.test": () => reply(engine) });
  const { similarHandler } = service(net);
  const ask = (list) => call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "2", ...(list ? { list } : {}) }));

  const main = await ask();
  assert.equal(main.body.engine.list, "main");
  assert.deepEqual(titles(main.body.results), titles(engine.results.main));
  const best = await ask("best");
  assert.equal(best.body.engine.list, "best");
  assert.deepEqual(titles(best.body.results), titles(engine.results.best));
  const listeners = await ask("listeners");
  assert.equal(listeners.body.engine.list, "listeners");
  assert.deepEqual(titles(listeners.body.results), titles(engine.results.listeners));
  // One ranking serves all three.
  assert.equal(net.count("engine.test"), 1);

  const refused = await ask("vibe");
  assert.equal(refused.status, 400);
  assert.equal(refused.body.error.code, "invalid_parameter");
});

test("a list the answer lacks falls back to the main one and says so", async () => {
  const { listeners: _none, ...results } = ENGINE.results;
  const net = network({ "engine.test": () => reply({ ...ENGINE, results }) });
  const { similarHandler } = service(net);
  const { body } = await call(similarHandler, similarPath({ url: SPOTIFY_LINK, k: "2", list: "listeners" }));
  assert.equal(body.engine.list, "main");
  assert.equal(body.results[0].deezer_id, FIRST_ID);
});
