// GET /api/mobile/similar and GET /api/mobile/link: soundalikes for a song
// shared from Spotify or Apple Music, with links that open in the same app.
// No user login anywhere: the seed comes from public pages (Spotify's embed
// page, the iTunes Lookup API), the ranking from the any-song engine, and the
// per-result details from Deezer's and iTunes' public APIs.

import {
  appleArtwork,
  appleSearchUrl,
  cleanAppleUrl,
  isAppleId,
  coreTitle,
  isSpotifyId,
  normalize,
  normalizeIsrc,
  normalizeStorefront,
  parseSharedLink,
  pickBestMatch,
  pickSeedRecording,
  sameLength,
  scriptLanguages,
  searchTitle,
  SPOTIFY_SHORT_HOSTS,
  spotifyIsrcUrl,
  spotifyMetadataQueries,
  spotifySearchUrl,
  spotifyTrackUrl,
  versionWords,
} from "./mobile-links.js";

export const DEFAULT_ENGINE_URL = "https://soundalike.yassin.app/api/anysong";
export const DEFAULT_K = 20;
export const MAX_K = 50;
export const DEFAULT_STOREFRONT = "us";
// Rows of one answer that may spend an iTunes Search call up front. The rest
// carry search links until a client asks /api/mobile/link for them.
export const APPLE_EAGER_ROWS = 5;
export const RESPONSE_MAX_AGE_S = 300;

const MINUTE = 60 * 1000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const SEED_TTL_MS = DAY;
const ENGINE_TTL_MS = 10 * MINUTE;
// Deezer preview URLs are signed for about 15 minutes. A cached row plus the
// response's own max-age must stay inside that.
const DEEZER_TTL_MS = 5 * MINUTE;
const LINK_HIT_TTL_MS = 7 * DAY;
const LINK_MISS_TTL_MS = DAY;
// Which Deezer recording a link is does not change.
const SEED_RECORDING_TTL_MS = 30 * DAY;
const MAX_TEXT = 300;
const BROWSER_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";
// ListenBrainz Labs: Spotify track ids for artist and title, many songs per call, no key.
const METADATA_LOOKUP_URL = "https://labs.api.listenbrainz.org/spotify-id-from-metadata/json";
const METADATA_LOOKUP_UA =
  "soundalike-mobile-api/1.0 (+https://soundalike.yassin.app; https://github.com/yassinsolim/soundalike)";
// Candidates of one song that /api/mobile/similar and /api/mobile/link check on Spotify's embed page.
const VERIFY_EAGER = 1;
const VERIFY_ON_DEMAND = 5;
const NEXT_DATA = /<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/;
const CLIENT_IP = /^[0-9a-fA-F:.]{3,45}$/;

const MESSAGES = {
  missing_seed: "Name the song with url, spotify_id, apple_id, isrc, or title and artist.",
  invalid_parameter: "A parameter has an invalid value.",
  invalid_link: "This is not a Spotify or Apple Music song link.",
  not_a_track: "This link points to an album. Share a single song instead.",
  unsupported_link: "Only links to single songs are supported.",
  track_not_found: "The linked song could not be found.",
  song_not_found: "No recording of this song with a preview could be found.",
  resolver_unavailable: "The music service did not answer. Try again.",
  rate_limited: "Too many requests. Try again shortly.",
  engine_busy: "The recommender is busy. Try again shortly.",
  engine_unavailable: "The recommender did not answer. Try again.",
  lookup_unavailable: "The song lookup did not answer. Try again.",
  method_not_allowed: "Only GET is supported.",
};

export class MobileError extends Error {
  constructor(status, code, { message, retryAfter } = {}) {
    super(message || MESSAGES[code] || code);
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

/** Bounded map with per-entry expiry; the oldest entry leaves first. */
export class TtlCache {
  constructor(maxEntries, now = Date.now) {
    this.maxEntries = maxEntries;
    this.now = now;
    this.entries = new Map();
  }
  get(key) {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    if (entry.expires <= this.now()) {
      this.entries.delete(key);
      return undefined;
    }
    return entry.value;
  }
  set(key, value, ttlMs) {
    this.entries.delete(key);
    this.entries.set(key, { value, expires: this.now() + ttlMs });
    while (this.entries.size > this.maxEntries) {
      this.entries.delete(this.entries.keys().next().value);
    }
  }
}

/** Token bucket: `capacity` calls at once, refilled evenly over `windowMs`. */
export function createBucket(capacity, windowMs, now = Date.now) {
  let tokens = capacity;
  let stamp = now();
  const refill = () => {
    const current = now();
    tokens = Math.min(capacity, tokens + ((current - stamp) * capacity) / windowMs);
    stamp = current;
  };
  return {
    /** Takes a token. With `force` the call is counted even when none is left. */
    take(force = false) {
      refill();
      if (tokens >= 1 || force) {
        tokens -= 1;
        return true;
      }
      return false;
    },
    /** Seconds until one token is available again. */
    retryAfter() {
      refill();
      return tokens >= 1 ? 0 : Math.ceil(((1 - tokens) * windowMs) / capacity / 1000);
    },
  };
}

function text(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function httpsUrl(value) {
  return typeof value === "string" && value.startsWith("https://") ? value : null;
}

/** Deezer's media (covers, previews) is only ever served from dzcdn.net. */
function dzcdnUrl(value) {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    return url.protocol === "https:" && (host === "dzcdn.net" || host.endsWith(".dzcdn.net")) ? value : null;
  } catch {
    return null;
  }
}

function pickSpotifyImage(images) {
  if (!Array.isArray(images)) return null;
  const usable = images
    .filter((image) => httpsUrl(image?.url))
    .sort((a, b) => (a.maxWidth ?? 0) - (b.maxWidth ?? 0));
  return (usable.find((image) => (image.maxWidth ?? 0) >= 200) ?? usable.at(-1))?.url ?? null;
}

/**
 * Reads Spotify's public embed page (`open.spotify.com/embed/track/<id>`).
 * Returns {title, artists[], durationMs|null, artwork|null, playable},
 * "missing" when Spotify says the track does not exist, or null when the page
 * is unreadable. The page has no ISRC.
 */
export function parseSpotifyEmbed(html) {
  const match = NEXT_DATA.exec(typeof html === "string" ? html : "");
  if (!match) return null;
  let props;
  try {
    props = JSON.parse(match[1])?.props?.pageProps;
  } catch {
    return null;
  }
  if (props?.status === 404) return "missing";
  const entity = props?.state?.data?.entity;
  if (!entity || entity.type !== "track") return null;
  const title = text(entity.name) ?? text(entity.title);
  const artists = (Array.isArray(entity.artists) ? entity.artists : [])
    .map((artist) => text(artist?.name))
    .filter(Boolean);
  if (!title || !artists.length) return null;
  return {
    title,
    artists,
    durationMs: Number.isFinite(entity.duration) ? entity.duration : null,
    artwork: pickSpotifyImage(entity.visualIdentity?.image),
    playable: entity.isPlayable !== false,
  };
}

function firstParam(params, name) {
  const values = params.getAll(name);
  if (values.length > 1) throw new MobileError(400, "invalid_parameter", { message: `${name} is given more than once.` });
  return values.length ? values[0] : null;
}

function boundedText(params, name) {
  const value = firstParam(params, name);
  if (value === null) return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > MAX_TEXT) {
    throw new MobileError(400, "invalid_parameter", { message: `${name} must be 1-${MAX_TEXT} characters.` });
  }
  return trimmed;
}

function choice(params, name, allowed, fallback) {
  const value = firstParam(params, name);
  if (value === null) return fallback;
  if (!allowed.includes(value)) {
    throw new MobileError(400, "invalid_parameter", { message: `${name} must be one of: ${allowed.join(", ")}.` });
  }
  return value;
}

function storefrontParam(params, headers) {
  const given = firstParam(params, "storefront");
  if (given !== null) {
    const storefront = normalizeStorefront(given);
    if (!storefront) {
      throw new MobileError(400, "invalid_parameter", { message: "storefront must be a two-letter country code." });
    }
    return { storefront, guessed: false };
  }
  // Behind Cloudflare the listener's country is the best guess for the store.
  const country = normalizeStorefront(String(headers?.["cf-ipcountry"] ?? ""));
  if (country && country !== "xx" && country !== "t1") return { storefront: country, guessed: true };
  return { storefront: DEFAULT_STOREFRONT, guessed: false };
}

function parseSimilarParams(params, headers) {
  const url = firstParam(params, "url");
  const spotifyId = firstParam(params, "spotify_id");
  const appleId = firstParam(params, "apple_id");
  const isrcRaw = firstParam(params, "isrc");
  const title = boundedText(params, "title");
  const artist = boundedText(params, "artist");
  const named = title !== null && artist !== null;
  if (url === null && spotifyId === null && appleId === null && isrcRaw === null && !named) {
    if (title !== null || artist !== null) {
      throw new MobileError(400, "invalid_parameter", { message: "title and artist are both required." });
    }
    throw new MobileError(400, "missing_seed");
  }
  const isrc = isrcRaw === null ? null : normalizeIsrc(isrcRaw);
  if (isrcRaw !== null && !isrc) {
    throw new MobileError(400, "invalid_parameter", { message: "isrc must be a 12-character ISRC." });
  }

  const kText = firstParam(params, "k");
  let k = DEFAULT_K;
  if (kText !== null) {
    if (!/^[0-9]{1,2}$/.test(kText) || Number(kText) < 1 || Number(kText) > MAX_K) {
      throw new MobileError(400, "invalid_parameter", { message: `k must be between 1 and ${MAX_K}.` });
    }
    k = Number(kText);
  }
  const lang = choice(params, "lang", ["same", "all"], "same");
  const platform = choice(params, "platform", ["spotify", "apple", "both"], null);
  const store = storefrontParam(params, headers);

  // The most specific name wins. What else the client knows (the app's own
  // ISRC, title and artist) is kept as a hint, see `similar`.
  let seed;
  if (url !== null) {
    const parsed = parseSharedLink(url);
    if (parsed.error) throw new MobileError(400, parsed.error);
    seed = parsed;
  } else if (spotifyId !== null) {
    if (!isSpotifyId(spotifyId)) {
      throw new MobileError(400, "invalid_parameter", { message: "spotify_id must be a 22-character track id." });
    }
    seed = { kind: "spotify", id: spotifyId };
  } else if (appleId !== null) {
    if (!isAppleId(appleId)) {
      throw new MobileError(400, "invalid_parameter", { message: "apple_id must be a numeric song id." });
    }
    seed = { kind: "apple", id: appleId, storefront: null };
  } else if (isrc !== null && !named) {
    seed = { kind: "isrc", isrc };
  } else {
    seed = { kind: "name", title, artist };
  }
  const hints = {
    isrc: seed.kind === "isrc" ? null : isrc,
    name: named && seed.kind !== "name" ? { title, artist } : null,
  };
  // A link's own storefront wins unless the caller names one.
  if (seed.kind === "apple" && seed.storefront && firstParam(params, "storefront") === null) {
    store.storefront = seed.storefront;
    store.guessed = false;
  }
  return { seed, hints, k, lang, platform, storefront: store.storefront, storefrontGuessed: store.guessed };
}

export function createMobileService({
  fetch: fetchImpl = globalThis.fetch,
  env = process.env,
  now = Date.now,
  budgetMs = 2000,
  seedTimeoutMs = 5000,
  engineTimeoutMs = 25000,
  lookupTimeoutMs = 4000,
  verifyWaitMs = 300,
  appleEagerRows = APPLE_EAGER_ROWS,
} = {}) {
  const engineBase = String(env.SOUNDALIKE_ENGINE_URL || DEFAULT_ENGINE_URL).replace(/\/+$/, "");
  // Only an engine reached directly (plain HTTP on the LAN or loopback) keys
  // its rate limit on a forwarded client address; Cloudflare sets its own.
  const forwardClientIp = engineBase.startsWith("http://");
  const spotifyClientId = text(env.SPOTIFY_CLIENT_ID);
  const spotifyClientSecret = text(env.SPOTIFY_CLIENT_SECRET);
  const hasSpotifyCredentials = Boolean(spotifyClientId && spotifyClientSecret);

  const seeds = new TtlCache(5000, now);
  const engineAnswers = new TtlCache(500, now);
  const deezerTracks = new TtlCache(20000, now);
  // What does not expire about a track (ISRC, cover, names), for when Deezer cannot be asked.
  const deezerFacts = new TtlCache(100000, now);
  const appleLinks = new TtlCache(50000, now);
  const spotifyLinks = new TtlCache(50000, now);
  // Track ids ListenBrainz knows for a song (by Deezer id), and the one of them that was checked.
  const spotifyCandidates = new TtlCache(50000, now);
  const spotifyPicks = new TtlCache(50000, now);
  const spotifyPicking = new Map();
  // How a shared song that its names did not find was found after all: by
  // Deezer id, under another artist name, or not at all. Keyed by the link.
  const seedRecordings = new TtlCache(20000, now);
  // Deezer allows 50 requests per 5 seconds per address, iTunes Search about
  // 20 a minute. Both buckets stay a little under that. Spotify publishes no
  // limit for its embed pages; 40 per 10 seconds is two new answers.
  const deezerBucket = createBucket(45, 5000, now);
  const itunesBucket = createBucket(18, MINUTE, now);
  const embedBucket = createBucket(40, 10000, now);
  let spotifyToken = null;
  let spotifyPausedUntil = 0;
  let metadataPausedUntil = 0;
  let embedPausedUntil = 0;

  function withTimeout(signal, timeoutMs) {
    const timeout = AbortSignal.timeout(timeoutMs);
    return signal ? AbortSignal.any([signal, timeout]) : timeout;
  }

  async function getJson(url, { signal, timeoutMs, headers } = {}) {
    const response = await fetchImpl(url, {
      headers: { Accept: "application/json", ...headers },
      signal: withTimeout(signal, timeoutMs),
    });
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    return { status: response.status, ok: response.ok, body, headers: response.headers };
  }

  /** Runs `load` once per key while it is in flight and keeps what it returns. */
  async function cached(cache, key, load) {
    const hit = cache.get(key);
    if (hit !== undefined) return hit;
    const pending = (async () => {
      try {
        const { value, ttlMs } = await load();
        if (ttlMs > 0) cache.set(key, value, ttlMs);
        else cache.entries.delete(key);
        return value;
      } catch (error) {
        cache.entries.delete(key);
        throw error;
      }
    })();
    cache.set(key, pending, MINUTE);
    return pending;
  }

  // ---- Deezer -------------------------------------------------------------

  function deezerRow(body) {
    return {
      id: body.id,
      title: text(body.title),
      artist: text(body.artist?.name),
      isrc: normalizeIsrc(body.isrc),
      durationMs: Number.isFinite(body.duration) && body.duration > 0 ? body.duration * 1000 : null,
      album: text(body.album?.title),
      contributors: [
        ...new Set((Array.isArray(body.contributors) ? body.contributors : []).map((one) => text(one?.name)).filter(Boolean)),
      ],
      artwork: dzcdnUrl(body.album?.cover_medium),
      preview: dzcdnUrl(body.preview),
    };
  }

  /**
   * One Deezer track by id or `isrc:<ISRC>`: the row, "missing" when Deezer
   * has no such track, or null when it could not be asked (over the rate
   * limit, failed or timed out) and nothing is remembered about it. A
   * remembered row has no preview (`stale`): preview addresses expire.
   */
  async function deezerTrack(key, signal, { force = false } = {}) {
    const cacheKey = String(key);
    try {
      const hit = deezerTracks.get(cacheKey);
      if (hit !== undefined) return await hit;
      if (!deezerBucket.take(force)) return deezerFacts.get(cacheKey) ?? null;
      return await cached(deezerTracks, cacheKey, async () => {
        const { ok, body } = await getJson(`https://api.deezer.com/track/${encodeURIComponent(cacheKey)}`, {
          signal,
          timeoutMs: 4000,
        });
        if (ok && body?.error?.code === 800) return { value: "missing", ttlMs: HOUR };
        if (!ok || !body || body.error || !body.id) throw new Error("deezer");
        const row = deezerRow(body);
        deezerFacts.set(cacheKey, { ...row, preview: null, stale: true }, 7 * DAY);
        return { value: row, ttlMs: DEEZER_TTL_MS };
      });
    } catch {
      return deezerFacts.get(cacheKey) ?? null;
    }
  }

  // ---- Apple Music (iTunes Search / Lookup) --------------------------------

  /**
   * The Apple Music song page for a song, or null when the store has no clear
   * match. `undefined` means it could not be asked now (throttled, failed,
   * timed out): the caller then falls back to a search link.
   */
  async function appleLink(deezerId, wanted, storefront, signal) {
    const key = `${storefront}:${deezerId}`;
    try {
      const hit = appleLinks.get(key);
      if (hit !== undefined) return await hit;
      if (!itunesBucket.take()) return undefined;
      return await cached(appleLinks, key, async () => {
        const term = `${wanted.artist} ${searchTitle(wanted.title)}`;
        const { ok, body } = await getJson(
          `https://itunes.apple.com/search?term=${encodeURIComponent(term)}&entity=song&limit=10&country=${storefront}`,
          { signal, timeoutMs: 4000 },
        );
        if (!ok || !Array.isArray(body?.results)) throw new Error("itunes");
        const rows = body.results
          .filter((row) => row?.kind === "song" && cleanAppleUrl(row.trackViewUrl))
          .map((row) => ({
            title: row.trackName,
            artist: row.artistName,
            durationMs: row.trackTimeMillis,
            url: cleanAppleUrl(row.trackViewUrl),
          }));
        const best = pickBestMatch(rows, wanted);
        return best
          ? { value: best.url, ttlMs: LINK_HIT_TTL_MS }
          : { value: null, ttlMs: LINK_MISS_TTL_MS };
      });
    } catch {
      return undefined;
    }
  }

  // ---- Spotify (only with client credentials) ------------------------------

  async function spotifyAccessToken(signal) {
    if (!hasSpotifyCredentials || now() < spotifyPausedUntil) return null;
    if (spotifyToken && spotifyToken.expires > now()) return spotifyToken.value;
    try {
      const response = await fetchImpl("https://accounts.spotify.com/api/token", {
        method: "POST",
        headers: {
          Authorization: `Basic ${Buffer.from(`${spotifyClientId}:${spotifyClientSecret}`).toString("base64")}`,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: "grant_type=client_credentials",
        signal: withTimeout(signal, 4000),
      });
      const body = response.ok ? await response.json() : null;
      if (!text(body?.access_token)) {
        spotifyPausedUntil = now() + 5 * MINUTE;
        return null;
      }
      const lifetime = Number.isFinite(body.expires_in) ? body.expires_in : 3600;
      spotifyToken = { value: body.access_token, expires: now() + Math.max(60, lifetime - 60) * 1000 };
      return spotifyToken.value;
    } catch {
      return null;
    }
  }

  async function spotifyApi(path, signal) {
    const token = await spotifyAccessToken(signal);
    if (!token) return null;
    const result = await getJson(`https://api.spotify.com/v1/${path}`, {
      signal,
      timeoutMs: 4000,
      headers: { Authorization: `Bearer ${token}` },
    });
    if (result.status === 429) {
      const wait = Number(result.headers?.get?.("retry-after"));
      spotifyPausedUntil = now() + (Number.isFinite(wait) && wait > 0 ? Math.min(wait, 3600) : 30) * 1000;
      return null;
    }
    if (result.status === 401) spotifyToken = null;
    return result.ok ? result.body : null;
  }

  /** With client credentials: the Spotify track id of a song, null (not found) or undefined (not asked). */
  async function spotifyLink(deezerId, wanted, signal) {
    if (!hasSpotifyCredentials) return undefined;
    const key = String(deezerId);
    try {
      const hit = spotifyLinks.get(key);
      if (hit !== undefined) return await hit;
      return await cached(spotifyLinks, key, async () => {
        const queries = [];
        if (wanted.isrc) queries.push(`isrc:${wanted.isrc}`);
        queries.push(`track:${searchTitle(wanted.title)} artist:${wanted.artist}`);
        for (const query of queries) {
          const body = await spotifyApi(`search?type=track&limit=5&q=${encodeURIComponent(query)}`, signal);
          if (!body) throw new Error("spotify");
          const rows = (Array.isArray(body.tracks?.items) ? body.tracks.items : [])
            .filter((item) => isSpotifyId(item?.id))
            .map((item) => ({
              title: item.name,
              artist: item.artists?.[0]?.name,
              durationMs: item.duration_ms,
              id: item.id,
            }));
          const best = query.startsWith("isrc:") ? rows[0] : pickBestMatch(rows, wanted);
          if (best) return { value: best.id, ttlMs: LINK_HIT_TTL_MS };
        }
        return { value: null, ttlMs: LINK_MISS_TTL_MS };
      });
    } catch {
      return undefined;
    }
  }

  // ---- Spotify track ids without credentials -------------------------------

  function pauseFor(response, fallbackSeconds) {
    const wait = Number(response.headers?.get?.("retry-after"));
    return now() + (Number.isFinite(wait) && wait > 0 ? Math.min(wait, 3600) : fallbackSeconds) * 1000;
  }

  /**
   * Asks ListenBrainz for the Spotify track ids of several songs in one call
   * and remembers the answer per song, hits and misses alike. `entries` are
   * {key, wanted}. It runs on its own clock, not the request's, so an answer
   * that comes too late for one request still serves the next.
   */
  async function lookupSpotifyCandidates(entries) {
    const asked = entries.filter((entry) => spotifyCandidates.get(entry.key) === undefined);
    if (!asked.length || now() < metadataPausedUntil) return;
    const queries = [];
    const owners = [];
    for (const entry of asked) {
      for (const query of spotifyMetadataQueries(entry.wanted)) {
        queries.push(query);
        owners.push(entry);
      }
    }
    try {
      const response = await fetchImpl(METADATA_LOOKUP_URL, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json", "User-Agent": METADATA_LOOKUP_UA },
        body: JSON.stringify(queries),
        signal: AbortSignal.timeout(lookupTimeoutMs),
      });
      if (response.status === 429 || response.status >= 500) {
        metadataPausedUntil = pauseFor(response, 60);
        return;
      }
      const body = response.ok ? await response.json() : null;
      if (!Array.isArray(body) || body.length !== queries.length) return;
      const found = new Map(asked.map((entry) => [entry, []]));
      body.forEach((item, index) => {
        const ids = found.get(owners[index]);
        for (const id of Array.isArray(item?.spotify_track_ids) ? item.spotify_track_ids : []) {
          if (isSpotifyId(id) && !ids.includes(id)) ids.push(id);
        }
      });
      for (const [entry, ids] of found) {
        spotifyCandidates.set(entry.key, ids, ids.length ? LINK_HIT_TTL_MS : LINK_MISS_TTL_MS);
      }
    } catch {
      // Unreachable or too slow: nothing is remembered, the next answer asks again.
    }
  }

  /** One track's embed page: the parsed page, "missing", or undefined when it cannot be asked now. */
  async function readEmbed(id, language = "en") {
    if (now() < embedPausedUntil || !embedBucket.take()) return undefined;
    try {
      const response = await fetchImpl(`https://open.spotify.com/embed/track/${id}`, {
        headers: { "User-Agent": BROWSER_UA, "Accept-Language": language },
        signal: AbortSignal.timeout(lookupTimeoutMs),
      });
      if (response.status === 404) return "missing";
      if (response.status === 429 || response.status >= 500) {
        embedPausedUntil = pauseFor(response, 60);
        return undefined;
      }
      if (!response.ok) return undefined;
      return parseSpotifyEmbed(await response.text()) ?? undefined;
    } catch {
      return undefined;
    }
  }

  /**
   * Checks the first `limit` candidates on Spotify's public embed page and
   * picks the one that is this song: the same title and kind of version (live,
   * remix, ...), playable, and closest in length when several are. The lookup matches on names, so the pick can be
   * another release of the song (album or single, a remaster). Artist names
   * are not compared: Spotify romanises them (Кино is "Kino" there).
   * Returns the id, null (none of them is the song) or undefined (not known yet).
   */
  function pickSpotifyCandidate(key, wanted, ids, limit) {
    const count = Math.min(limit, ids.length);
    const known = spotifyPicks.get(key);
    if (known !== undefined && known.checked >= count) return Promise.resolve(known.id);
    // One check per song at a time, however many answers are waiting for it.
    const flight = `${key}:${count}`;
    let pending = spotifyPicking.get(flight);
    if (!pending) {
      pending = checkSpotifyCandidates(key, wanted, ids, count).finally(() => spotifyPicking.delete(flight));
      spotifyPicking.set(flight, pending);
    }
    return pending;
  }

  async function checkSpotifyCandidates(key, wanted, ids, count) {
    const pages = await Promise.all(ids.slice(0, count).map(readEmbed));
    const wantedTitle = coreTitle(wanted.title);
    const wantedVersion = versionWords(wanted.title);
    let best = null;
    let bestApart = Infinity;
    pages.forEach((page, index) => {
      if (!page || page === "missing" || !page.playable || coreTitle(page.title) !== wantedTitle) return;
      if (versionWords(page.title) !== wantedVersion) return;
      const apart =
        Number.isFinite(wanted.durationMs) && Number.isFinite(page.durationMs)
          ? Math.abs(wanted.durationMs - page.durationMs)
          : 0;
      if (apart < bestApart) {
        best = ids[index];
        bestApart = apart;
      }
    });
    if (pages.includes(undefined)) return best ?? undefined;
    spotifyPicks.set(key, { id: best, checked: count }, best ? LINK_HIT_TTL_MS : LINK_MISS_TTL_MS);
    return best;
  }

  // ---- Seed ----------------------------------------------------------------

  async function expandShortLink(url, signal) {
    let current = url;
    for (let hop = 0; hop < 5; hop += 1) {
      let response;
      try {
        response = await fetchImpl(current, {
          redirect: "manual",
          headers: { "User-Agent": BROWSER_UA },
          signal: withTimeout(signal, seedTimeoutMs),
        });
      } catch {
        throw new MobileError(502, "resolver_unavailable");
      }
      const location = response.headers?.get?.("location");
      if (response.status >= 300 && response.status < 400 && location) {
        let next;
        try {
          next = new URL(location, current);
        } catch {
          break;
        }
        const parsed = parseSharedLink(next.toString());
        if (parsed.kind === "spotify") return parsed;
        // Never follow a redirect off Spotify's short-link hosts.
        if (next.protocol !== "https:" || !SPOTIFY_SHORT_HOSTS.has(next.hostname)) break;
        current = next.toString();
        continue;
      }
      if (response.ok) {
        // Branch pages name the destination in their markup.
        const parsed = parseSharedLink((await response.text()).slice(0, 200000).replace(/\\\//g, "/"));
        if (parsed.kind === "spotify") return parsed;
      }
      break;
    }
    throw new MobileError(400, "unsupported_link", { message: "This short link does not lead to a song." });
  }

  function spotifySeed(id, signal) {
    return cached(seeds, `spotify:${id}`, async () => {
      let html;
      try {
        const response = await fetchImpl(`https://open.spotify.com/embed/track/${id}`, {
          headers: { "User-Agent": BROWSER_UA, "Accept-Language": "en" },
          signal: withTimeout(signal, seedTimeoutMs),
        });
        if (response.status === 404) throw new MobileError(404, "track_not_found");
        if (!response.ok) throw new MobileError(502, "resolver_unavailable");
        html = await response.text();
      } catch (error) {
        if (error instanceof MobileError) throw error;
        throw new MobileError(502, "resolver_unavailable");
      }
      const embed = parseSpotifyEmbed(html);
      if (embed === "missing") throw new MobileError(404, "track_not_found");
      if (!embed) throw new MobileError(502, "resolver_unavailable");
      let isrc = null;
      if (hasSpotifyCredentials) {
        try {
          isrc = normalizeIsrc((await spotifyApi(`tracks/${id}`, signal))?.external_ids?.isrc);
        } catch {
          isrc = null;
        }
      }
      return {
        value: {
          platform: "spotify",
          spotifyId: id,
          key: `spotify:${id}`,
          title: embed.title,
          artist: embed.artists[0],
          artists: embed.artists,
          artistLine: embed.artists.join(", "),
          isrc,
          artwork: embed.artwork,
          durationMs: embed.durationMs,
          sourceUrl: spotifyTrackUrl(id),
        },
        ttlMs: SEED_TTL_MS,
      };
    });
  }

  function appleSeed(id, storefront, signal) {
    return cached(seeds, `apple:${storefront}:${id}`, async () => {
      itunesBucket.take(true);
      let result;
      try {
        result = await getJson(
          `https://itunes.apple.com/lookup?id=${id}&country=${storefront}&entity=song`,
          { signal, timeoutMs: seedTimeoutMs },
        );
      } catch {
        throw new MobileError(502, "resolver_unavailable");
      }
      if (!result.ok || !Array.isArray(result.body?.results)) {
        throw new MobileError(502, "resolver_unavailable");
      }
      const song = result.body.results.find(
        (row) => row?.wrapperType === "track" && row.kind === "song" && String(row.trackId) === id,
      );
      const title = text(song?.trackName);
      const artist = text(song?.artistName);
      if (!title || !artist) throw new MobileError(404, "track_not_found");
      return {
        value: {
          platform: "apple",
          key: `apple:${id}`,
          title,
          artist,
          artists: [artist],
          album: text(song.collectionName),
          artistLine: artist,
          isrc: null, // the iTunes API does not publish ISRCs
          artwork: appleArtwork(song.artworkUrl100),
          durationMs: Number.isFinite(song.trackTimeMillis) ? song.trackTimeMillis : null,
          sourceUrl: cleanAppleUrl(song.trackViewUrl),
        },
        ttlMs: SEED_TTL_MS,
      };
    });
  }

  async function isrcSeed(isrc, signal) {
    const track = await deezerTrack(`isrc:${isrc}`, withTimeout(signal, seedTimeoutMs), { force: true });
    if (track === "missing") throw new MobileError(404, "track_not_found");
    if (!track?.title || !track.artist) throw new MobileError(502, "resolver_unavailable");
    return {
      platform: null,
      title: track.title,
      artist: track.artist,
      artistLine: track.artist,
      isrc,
      artwork: track.artwork,
      durationMs: track.durationMs,
      sourceUrl: null,
    };
  }

  async function resolveSeed(seed, storefront, signal) {
    let target = seed;
    if (target.kind === "spotify-short") target = await expandShortLink(target.url, signal);
    if (target.kind === "spotify") return spotifySeed(target.id, signal);
    if (target.kind === "apple") return appleSeed(target.id, storefront, signal);
    if (target.kind === "isrc") return isrcSeed(target.isrc, signal);
    return {
      platform: null,
      title: target.title,
      artist: target.artist,
      artistLine: target.artist,
      isrc: null,
      artwork: null,
      durationMs: null,
      sourceUrl: null,
    };
  }

  // ---- Engine --------------------------------------------------------------

  /** `song` is {id} (a Deezer track) or {title, artist, isrc?}. */
  function askEngine(song, k, lang, clientIp) {
    const query = new URLSearchParams(song.id ? { id: String(song.id) } : { title: song.title, artist: song.artist });
    if (!song.id && song.isrc) query.set("isrc", song.isrc);
    query.set("k", String(k));
    query.set("lang", lang);
    const url = `${engineBase}/similar?${query}`;
    return cached(engineAnswers, url, async () => {
      let result;
      try {
        result = await getJson(url, {
          timeoutMs: engineTimeoutMs,
          headers: forwardClientIp && clientIp ? { "CF-Connecting-IP": clientIp } : {},
        });
      } catch {
        throw new MobileError(502, "engine_unavailable");
      }
      const { status, body } = result;
      if (status === 200 && body?.ok && body.query && body.results) {
        return { value: body, ttlMs: ENGINE_TTL_MS };
      }
      if (status === 404 || status === 422) throw new MobileError(404, "song_not_found");
      const wait = Number(result.headers?.get?.("retry-after"));
      const retryAfter = Number.isFinite(wait) && wait > 0 ? Math.min(Math.ceil(wait), 120) : 5;
      if (status === 429) throw new MobileError(429, "rate_limited", { retryAfter });
      if (status === 503 || status === 504) throw new MobileError(503, "engine_busy", { retryAfter });
      throw new MobileError(502, "engine_unavailable");
    });
  }

  // ---- A shared song the engine does not find by name ------------------------

  async function deezerList(path, language) {
    deezerBucket.take(true);
    try {
      const { ok, body } = await getJson(`https://api.deezer.com/${path}`, {
        timeoutMs: lookupTimeoutMs,
        headers: language ? { "Accept-Language": language } : {},
      });
      return ok && Array.isArray(body?.data) ? body.data : [];
    } catch {
      return [];
    }
  }

  /** The artist names the link's own service has in the script of the title ("Kino" is Кино). */
  async function alternateArtists(seed, language) {
    if (!language || !seed.spotifyId) return [];
    const page = await readEmbed(seed.spotifyId, language);
    if (!page || page === "missing") return [];
    const known = new Set(seed.artists.map(normalize));
    return page.artists.filter((name) => !known.has(normalize(name)));
  }

  /**
   * Looks for the shared song among the Deezer tracks of its artists, in
   * Deezer's own naming and in the language of the title's script, and
   * returns the track id when `pickSeedRecording` has the evidence, else null.
   */
  async function findDeezerRecording(seed, names, language) {
    const found = await Promise.all(
      names.slice(0, 4).map((name) => deezerList(`search/artist?q=${encodeURIComponent(name)}&limit=5`)),
    );
    // Deezer's artist search knows other spellings (周杰倫 finds Jay Chou), so
    // the names are not compared: the track evidence decides.
    const artistIds = [...new Set(found.flatMap((list) => list.slice(0, 2).map((artist) => artist?.id)))]
      .filter(Number.isSafeInteger)
      .slice(0, 4);
    if (!artistIds.length) return null;
    const lists = [];
    for (const id of artistIds) {
      lists.push(deezerList(`artist/${id}/top?limit=100`));
      if (language) lists.push(deezerList(`artist/${id}/top?limit=100`, language));
    }
    for (const name of names.slice(0, 2)) {
      lists.push(deezerList(`search?q=${encodeURIComponent(`${seed.title} ${name}`)}&limit=25`, language));
    }
    const tracks = new Map();
    for (const row of (await Promise.all(lists)).flat()) {
      if (!Number.isSafeInteger(row?.id) || !artistIds.includes(row.artist?.id) || !text(row.title)) continue;
      const track = tracks.get(row.id) ?? {
        id: row.id,
        titles: [],
        durationS: row.duration,
        album: text(row.album?.title),
        rank: Number.isFinite(row.rank) ? row.rank : 0,
        isrc: null,
      };
      if (!track.titles.includes(row.title)) track.titles.push(row.title);
      tracks.set(row.id, track);
    }
    const fitting = [...tracks.values()].filter((track) => sameLength(seed.durationMs, track.durationS)).slice(0, 8);
    // Which releases are one recording is told by their ISRCs.
    await Promise.all(
      fitting.map(async (track) => {
        const details = await deezerTrack(track.id, AbortSignal.timeout(lookupTimeoutMs), { force: true });
        if (details && details !== "missing") track.isrc = details.isrc;
      }),
    );
    return pickSeedRecording(seed, fitting)?.id ?? null;
  }

  /**
   * The engine's answer for a shared song. Its names are tried first, as
   * always. Only when the engine does not know the song by them (Deezer has
   * 晴天 as "Sunny Day", Кино's songs under Кино and not "Kino"), the Deezer
   * recording is looked for with evidence and the engine asked by its id;
   * failing that, the names are tried once more with the artist as written
   * in the title's script. What worked is remembered for the link.
   */
  async function engineAnswer(seed, k, lang, clientIp) {
    const known = seed.key ? seedRecordings.get(seed.key) : undefined;
    if (known?.id || known?.artist) {
      try {
        return await askEngine(known.id ? { id: known.id } : { title: seed.title, artist: known.artist }, k, lang, clientIp);
      } catch (error) {
        if (!(error instanceof MobileError) || error.code !== "song_not_found") throw error;
        seedRecordings.entries.delete(seed.key);
      }
    }
    let notFound;
    try {
      return await askEngine(seed, k, lang, clientIp);
    } catch (error) {
      if (!(error instanceof MobileError) || error.code !== "song_not_found") throw error;
      if (!seed.key || !Number.isFinite(seed.durationMs) || known?.miss) throw error;
      notFound = error;
    }
    const language = scriptLanguages(`${seed.title} ${seed.artists.join(" ")}`);
    const alternates = await alternateArtists(seed, language);
    const attempts = [];
    const recording = await findDeezerRecording(seed, [...seed.artists, ...alternates], language);
    if (recording) attempts.push({ remember: { id: recording }, song: { id: recording } });
    if (alternates.length) {
      attempts.push({ remember: { artist: alternates[0] }, song: { title: seed.title, artist: alternates[0] } });
    }
    for (const attempt of attempts) {
      try {
        const answer = await askEngine(attempt.song, k, lang, clientIp);
        seedRecordings.set(seed.key, attempt.remember, SEED_RECORDING_TTL_MS);
        return answer;
      } catch (error) {
        if (!(error instanceof MobileError) || error.code !== "song_not_found") throw error;
      }
    }
    seedRecordings.set(seed.key, { miss: true }, LINK_MISS_TTL_MS);
    throw notFound;
  }

  // ---- /api/mobile/similar -------------------------------------------------

  async function similar(params, headers = {}) {
    const request = parseSimilarParams(params, headers);
    const { hints } = request;
    let seed;
    try {
      seed = await resolveSeed(request.seed, request.storefront);
    } catch (error) {
      // The link could not be read, but the client told us the song's name.
      if (!hints.name || !(error instanceof MobileError) || error.status < 404) throw error;
      seed = {
        platform: request.seed.kind.startsWith("spotify") ? "spotify" : request.seed.kind === "apple" ? "apple" : null,
        title: hints.name.title,
        artist: hints.name.artist,
        artistLine: hints.name.artist,
        isrc: null,
        artwork: null,
        durationMs: null,
        sourceUrl: null,
      };
    }
    if (hints.isrc && !seed.isrc) seed = { ...seed, isrc: hints.isrc };
    const platform = request.platform ?? seed.platform ?? "both";
    const wantApple = platform !== "spotify";
    const wantSpotify = platform !== "apple";
    const { storefront } = request;

    const clientIp = String(headers["cf-connecting-ip"] ?? "");
    const answer = await engineAnswer(seed, request.k, request.lang, CLIENT_IP.test(clientIp) ? clientIp : null);
    const list = Array.isArray(answer.results.main) ? "main" : "best";
    const rows = (Array.isArray(answer.results[list]) ? answer.results[list] : [])
      .filter((row) => Number.isSafeInteger(row?.id) && text(row.title) && text(row.artist))
      .slice(0, request.k);

    // Everything below is best effort inside one time budget: whatever has
    // not arrived when it ends is served as a search link or null.
    const details = rows.map(() => ({}));
    const seedDetails = { spotify: seed.spotifyId ?? null };
    const controller = new AbortController();
    const { signal } = controller;
    const wantedOf = (row, track) => ({
      title: row.title,
      artist: row.artist,
      durationMs: track?.durationMs ?? null,
      isrc: track?.isrc ?? null,
      album: track?.album ?? null,
      contributors: track?.contributors ?? [],
    });
    // Per row: Deezer's details, then (with credentials) Spotify's own search.
    const ready = rows.map(async (row, index) => {
      const state = details[index];
      const track = await deezerTrack(row.id, signal);
      if (track && track !== "missing") state.track = track;
      if (wantSpotify) state.spotify = (await spotifyLink(row.id, wantedOf(row, state.track), signal)) || null;
    });
    const tasks = rows.map(async (row, index) => {
      if (!wantApple) return;
      await ready[index];
      const state = details[index];
      const known = appleLinks.get(`${storefront}:${row.id}`);
      if (known !== undefined && !(known instanceof Promise)) state.apple = known;
      else if (index < appleEagerRows) {
        state.apple = await appleLink(row.id, wantedOf(row, state.track), storefront, signal);
      }
    });
    tasks.push(...ready);
    const seedTrack =
      Number.isSafeInteger(answer.query.id) && (!seed.isrc || !seed.artwork || (wantSpotify && !seedDetails.spotify))
        ? (async () => {
            const track = await deezerTrack(answer.query.id, signal, { force: true });
            if (track && track !== "missing") seedDetails.track = track;
          })()
        : Promise.resolve();
    tasks.push(seedTrack);
    if (wantSpotify) {
      // Everything credentials did not find goes to ListenBrainz in one call,
      // then each first candidate is checked on Spotify's embed page.
      tasks.push(
        (async () => {
          await Promise.allSettled([...ready, seedTrack]);
          const entries = rows
            .map((row, index) => ({ key: String(row.id), wanted: wantedOf(row, details[index].track), state: details[index] }))
            .filter((entry) => !entry.state.spotify);
          if (!seedDetails.spotify) {
            // The engine's release of the seed names everyone on it ("Luis Fonsi, Daddy Yankee").
            const sameSong = seedDetails.track && coreTitle(seedDetails.track.title) === coreTitle(seed.title);
            entries.push({
              key: `seed:${normalize(seed.artist)}:${normalize(seed.title)}`,
              wanted: {
                title: seed.title,
                artist: seed.artist,
                durationMs: seed.durationMs,
                album: null,
                contributors: sameSong ? seedDetails.track.contributors : [],
              },
              state: seedDetails,
            });
          }
          await lookupSpotifyCandidates(entries);
          const checks = Promise.all(
            entries.map(async (entry) => {
              const ids = spotifyCandidates.get(entry.key);
              if (!ids?.length) return;
              // Until it is checked the first id stands: of 157 first ids measured, all were the song.
              entry.state.spotify = ids[0];
              entry.state.unchecked = true;
              const pick = await pickSpotifyCandidate(entry.key, entry.wanted, ids, VERIFY_EAGER);
              if (pick === undefined) return;
              entry.state.spotify = pick;
              entry.state.unchecked = false;
            }),
          );
          // The checks finish on their own for the next answer; this one waits only briefly.
          let wait;
          await Promise.race([
            checks,
            new Promise((resolve) => {
              wait = setTimeout(resolve, verifyWaitMs);
            }),
          ]);
          clearTimeout(wait);
        })(),
      );
    }
    let timer;
    await Promise.race([
      Promise.allSettled(tasks),
      new Promise((resolve) => {
        timer = setTimeout(resolve, budgetMs);
      }),
    ]);
    clearTimeout(timer);
    controller.abort();

    const results = rows.map((row, index) => {
      const { track, apple, spotify } = details[index];
      const isrc = track?.isrc ?? null;
      const spotifyKind = spotify ? "track" : isrc ? "isrc_search" : "search";
      return {
        title: row.title,
        artist: row.artist,
        match: Number.isFinite(row.match) ? Math.round(row.match) : null,
        deezer_id: row.id,
        isrc,
        artwork: track?.artwork ?? null,
        preview_url: track?.preview ?? null,
        spotify_url: spotify
          ? spotifyTrackUrl(spotify)
          : isrc
            ? spotifyIsrcUrl(isrc)
            : spotifySearchUrl(row.artist, row.title),
        spotify_exact: Boolean(spotify),
        spotify_kind: spotifyKind,
        spotify_id: spotify || null,
        apple_music_url: apple || appleSearchUrl(storefront, row.artist, row.title),
        apple_exact: Boolean(apple),
      };
    });
    return {
      cacheable: !request.storefrontGuessed,
      // An answer missing Deezer's details (rate limit, time budget) is worth asking for again soon.
      maxAge:
        details.every((state) => state.track && !state.track.stale && !state.unchecked) ? RESPONSE_MAX_AGE_S : 20,
      body: {
        ok: true,
        query: {
          title: seed.title,
          artist: seed.artistLine,
          isrc: seed.isrc ?? seedDetails.track?.isrc ?? null,
          artwork: seed.artwork ?? seedDetails.track?.artwork ?? null,
          platform,
          source_url: seed.sourceUrl,
          spotify_id: seedDetails.spotify ?? null,
          language: text(answer.query.language),
          storefront,
          deezer_id: Number.isSafeInteger(answer.query.id) ? answer.query.id : null,
        },
        results,
        engine: {
          list,
          k: request.k,
          lang: request.lang,
          source: text(answer.query.source),
          matched_title: text(answer.query.title),
          matched_artist: text(answer.query.artist),
          hidden: Number.isFinite(answer.query.hidden) ? answer.query.hidden : null,
        },
        links: {
          spotify_exact: results.filter((row) => row.spotify_exact).length,
          spotify_isrc: results.filter((row) => row.spotify_kind === "isrc_search").length,
          apple_exact: results.filter((row) => row.apple_exact).length,
          spotify_lookup: true,
          spotify_source: hasSpotifyCredentials ? "credentials" : "metadata",
          apple_eager_rows: appleEagerRows,
        },
      },
    };
  }

  // ---- /api/mobile/link ----------------------------------------------------

  async function link(params, headers = {}) {
    const idText = firstParam(params, "deezer_id");
    if (idText === null || !/^[1-9][0-9]{0,15}$/.test(idText)) {
      throw new MobileError(400, "invalid_parameter", { message: "deezer_id must be a Deezer track id." });
    }
    const deezerId = Number(idText);
    const platform = choice(params, "platform", ["spotify", "apple"], null);
    if (!platform) {
      throw new MobileError(400, "invalid_parameter", { message: "platform must be one of: spotify, apple." });
    }
    const title = boundedText(params, "title");
    const artist = boundedText(params, "artist");
    const store = storefrontParam(params, headers);
    const { storefront } = store;

    const signal = AbortSignal.timeout(Math.max(budgetMs, 4000));
    let wanted = title && artist ? { title, artist, durationMs: null, isrc: null, album: null, contributors: [] } : null;
    const track = await deezerTrack(deezerId, signal, { force: true });
    if (track === "missing") throw new MobileError(404, "track_not_found");
    if (track?.title && track.artist) {
      wanted = {
        title: track.title,
        artist: track.artist,
        durationMs: track.durationMs,
        isrc: track.isrc,
        album: track.album ?? null,
        contributors: track.contributors ?? [],
      };
    }
    if (!wanted) throw new MobileError(502, "lookup_unavailable");

    let url;
    let exact = false;
    let kind = "search";
    let retryAfter = null;
    let spotifyId = null;
    if (platform === "apple") {
      const found = await appleLink(deezerId, wanted, storefront, signal);
      if (found === undefined) retryAfter = Math.max(1, itunesBucket.retryAfter());
      exact = Boolean(found);
      if (exact) kind = "track";
      url = found || appleSearchUrl(storefront, wanted.artist, wanted.title);
    } else {
      let found = await spotifyLink(deezerId, wanted, signal);
      if (!found) {
        // One row may take longer: every candidate ListenBrainz has is checked, not only the first.
        const key = String(deezerId);
        await lookupSpotifyCandidates([{ key, wanted }]);
        const ids = spotifyCandidates.get(key);
        if (ids?.length) found = await pickSpotifyCandidate(key, wanted, ids, VERIFY_ON_DEMAND);
      }
      spotifyId = found || null;
      exact = Boolean(found);
      if (exact) kind = "track";
      else if (wanted.isrc) kind = "isrc_search";
      url = found
        ? spotifyTrackUrl(found)
        : wanted.isrc
          ? spotifyIsrcUrl(wanted.isrc)
          : spotifySearchUrl(wanted.artist, wanted.title);
    }
    return {
      // An answer that may improve on the next call must not be cached.
      cacheable: retryAfter === null && !(store.guessed && platform === "apple"),
      noStore: retryAfter !== null,
      // A song's own page does not change; a search link may still become one.
      maxAge: exact ? 24 * 60 * 60 : RESPONSE_MAX_AGE_S,
      body: {
        ok: true,
        deezer_id: deezerId,
        platform,
        url,
        exact,
        kind,
        ...(platform === "spotify" ? { spotify_id: spotifyId } : {}),
        storefront: platform === "apple" ? storefront : null,
        retry_after: retryAfter,
      },
    };
  }

  // ---- HTTP ----------------------------------------------------------------

  function send(response, status, body, cacheControl, retryAfter) {
    response.statusCode = status;
    response.setHeader("Access-Control-Allow-Origin", "*");
    response.setHeader("Cache-Control", cacheControl);
    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("X-Content-Type-Options", "nosniff");
    if (retryAfter) response.setHeader("Retry-After", String(retryAfter));
    response.end(status === 204 ? undefined : JSON.stringify(body));
  }

  function handler(run) {
    return async function mobileHandler(request, response) {
      if (request.method === "OPTIONS") {
        response.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
        response.setHeader("Access-Control-Max-Age", "86400");
        return send(response, 204, null, "public, max-age=86400");
      }
      try {
        if (request.method !== "GET" && request.method !== "HEAD") {
          response.setHeader("Allow", "GET, OPTIONS");
          throw new MobileError(405, "method_not_allowed");
        }
        const params = new URL(request.url || "/", "http://localhost").searchParams;
        const { body, cacheable, noStore, maxAge = RESPONSE_MAX_AGE_S } = await run(params, request.headers || {});
        const cacheControl = noStore
          ? "no-store, max-age=0"
          : `${cacheable ? "public" : "private"}, max-age=${maxAge}`;
        return send(response, 200, body, cacheControl);
      } catch (error) {
        const known = error instanceof MobileError;
        if (!known) console.error("mobile api: unexpected error", error?.name || "Error");
        const status = known ? error.status : 500;
        const code = known ? error.code : "internal_error";
        const message = known ? error.message : "Something went wrong.";
        return send(response, status, { ok: false, error: { code, message } }, "no-store, max-age=0", known ? error.retryAfter : null);
      }
    };
  }

  return {
    similar,
    link,
    similarHandler: handler(similar),
    linkHandler: handler(link),
  };
}

let shared;
/** The process-wide service, so both functions share one set of caches. */
export function sharedMobileService() {
  shared ??= createMobileService();
  return shared;
}
