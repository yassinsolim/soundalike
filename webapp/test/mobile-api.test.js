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

function service(net, options = {}) {
  return createMobileService({
    fetch: net.fetch,
    env: { SOUNDALIKE_ENGINE_URL: ENGINE_URL, ...options.env },
    budgetMs: options.budgetMs ?? 500,
    seedTimeoutMs: 200,
    engineTimeoutMs: 200,
    now: options.now,
    appleEagerRows: options.appleEagerRows,
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
    spotify_url: "https://open.spotify.com/search/OOMPH!%20Gekreuzigt%202006",
    spotify_exact: false,
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
  assert.deepEqual(body.links, { spotify_exact: 0, apple_exact: 0, spotify_lookup: false, apple_eager_rows: 5 });

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
  assert.deepEqual(body.links, { spotify_exact: 0, apple_exact: 1, spotify_lookup: false, apple_eager_rows: 2 });
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
    [{ url: SPOTIFY_LINK, isrc: "DEC760600298" }, "ambiguous_seed"],
    [{ spotify_id: SPOTIFY_ID, title: "x", artist: "y" }, "ambiguous_seed"],
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
    assert.match(row.spotify_url, /^https:\/\/open\.spotify\.com\/search\//);
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
  assert.deepEqual(Object.keys(first.body), ["ok", "deezer_id", "platform", "url", "exact", "storefront", "retry_after"]);
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
  assert.equal(spotify.body.url, "https://open.spotify.com/search/OOMPH!%20Gekreuzigt%202006");
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
      [false, "https://open.spotify.com/search/Unheilig%20Willkommen"],
    ],
  );
  assert.deepEqual(body.links, { spotify_exact: 2, apple_exact: 0, spotify_lookup: true, apple_eager_rows: 5 });
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
