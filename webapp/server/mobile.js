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
  isSpotifyId,
  normalizeIsrc,
  normalizeStorefront,
  parseSharedLink,
  pickBestMatch,
  searchTitle,
  SPOTIFY_SHORT_HOSTS,
  spotifySearchUrl,
  spotifyTrackUrl,
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
const MAX_TEXT = 300;
const BROWSER_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";
const NEXT_DATA = /<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/;
const CLIENT_IP = /^[0-9a-fA-F:.]{3,45}$/;

const MESSAGES = {
  missing_seed: "Name the song with url, spotify_id, apple_id, isrc, or title and artist.",
  ambiguous_seed: "Use exactly one of url, spotify_id, apple_id, isrc, or title and artist.",
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
 * Returns {title, artists[], durationMs|null, artwork|null}, "missing" when
 * Spotify says the track does not exist, or null when the page is unreadable.
 * The page has no ISRC.
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
  const named = title !== null || artist !== null;
  const kinds = [url, spotifyId, appleId, isrcRaw].filter((value) => value !== null).length + (named ? 1 : 0);
  if (kinds === 0) throw new MobileError(400, "missing_seed");
  if (kinds > 1) throw new MobileError(400, "ambiguous_seed");

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
  } else if (isrcRaw !== null) {
    const isrc = normalizeIsrc(isrcRaw);
    if (!isrc) throw new MobileError(400, "invalid_parameter", { message: "isrc must be a 12-character ISRC." });
    seed = { kind: "isrc", isrc };
  } else {
    if (title === null || artist === null) {
      throw new MobileError(400, "invalid_parameter", { message: "title and artist are both required." });
    }
    seed = { kind: "name", title, artist };
  }
  // A link's own storefront wins unless the caller names one.
  if (seed.kind === "apple" && seed.storefront && firstParam(params, "storefront") === null) {
    store.storefront = seed.storefront;
    store.guessed = false;
  }
  return { seed, k, lang, platform, storefront: store.storefront, storefrontGuessed: store.guessed };
}

export function createMobileService({
  fetch: fetchImpl = globalThis.fetch,
  env = process.env,
  now = Date.now,
  budgetMs = 2000,
  seedTimeoutMs = 5000,
  engineTimeoutMs = 25000,
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
  const appleLinks = new TtlCache(50000, now);
  const spotifyLinks = new TtlCache(50000, now);
  // Deezer allows 50 requests per 5 seconds per address, iTunes Search about
  // 20 a minute. Both buckets stay a little under that.
  const deezerBucket = createBucket(45, 5000, now);
  const itunesBucket = createBucket(18, MINUTE, now);
  let spotifyToken = null;
  let spotifyPausedUntil = 0;

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
      artwork: dzcdnUrl(body.album?.cover_medium),
      preview: dzcdnUrl(body.preview),
    };
  }

  /**
   * One Deezer track by id or `isrc:<ISRC>`: the row, "missing" when Deezer
   * has no such track, or null when it could not be asked (over the rate
   * limit, failed or timed out).
   */
  async function deezerTrack(key, signal, { force = false } = {}) {
    const cacheKey = String(key);
    try {
      const hit = deezerTracks.get(cacheKey);
      if (hit !== undefined) return await hit;
      if (!deezerBucket.take(force)) return null;
      return await cached(deezerTracks, cacheKey, async () => {
        const { ok, body } = await getJson(`https://api.deezer.com/track/${encodeURIComponent(cacheKey)}`, {
          signal,
          timeoutMs: 4000,
        });
        if (ok && body?.error?.code === 800) return { value: "missing", ttlMs: HOUR };
        if (!ok || !body || body.error || !body.id) throw new Error("deezer");
        return { value: deezerRow(body), ttlMs: DEEZER_TTL_MS };
      });
    } catch {
      return null;
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

  /** The Spotify track id of a song, null (not found) or undefined (not asked). */
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
          title: embed.title,
          artist: embed.artists[0],
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
          title,
          artist,
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

  function askEngine(seed, k, lang, clientIp) {
    const query = new URLSearchParams({ title: seed.title, artist: seed.artist });
    if (seed.isrc) query.set("isrc", seed.isrc);
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

  // ---- /api/mobile/similar -------------------------------------------------

  async function similar(params, headers = {}) {
    const request = parseSimilarParams(params, headers);
    const seed = await resolveSeed(request.seed, request.storefront);
    const platform = request.platform ?? seed.platform ?? "both";
    const wantApple = platform !== "spotify";
    const wantSpotify = platform !== "apple";
    const { storefront } = request;

    const clientIp = String(headers["cf-connecting-ip"] ?? "");
    const answer = await askEngine(seed, request.k, request.lang, CLIENT_IP.test(clientIp) ? clientIp : null);
    const list = Array.isArray(answer.results.main) ? "main" : "best";
    const rows = (Array.isArray(answer.results[list]) ? answer.results[list] : [])
      .filter((row) => Number.isSafeInteger(row?.id) && text(row.title) && text(row.artist))
      .slice(0, request.k);

    // Everything below is best effort inside one time budget: whatever has
    // not arrived when it ends is served as a search link or null.
    const details = rows.map(() => ({}));
    const seedDetails = {};
    const controller = new AbortController();
    const { signal } = controller;
    const tasks = rows.map(async (row, index) => {
      const state = details[index];
      const track = await deezerTrack(row.id, signal);
      if (track && track !== "missing") state.track = track;
      const wanted = {
        title: row.title,
        artist: row.artist,
        durationMs: state.track?.durationMs ?? null,
        isrc: state.track?.isrc ?? null,
      };
      await Promise.all([
        (async () => {
          if (!wantApple) return;
          const key = `${storefront}:${row.id}`;
          const known = appleLinks.get(key);
          if (known !== undefined && !(known instanceof Promise)) state.apple = known;
          else if (index < appleEagerRows) state.apple = await appleLink(row.id, wanted, storefront, signal);
        })(),
        (async () => {
          if (wantSpotify) state.spotify = await spotifyLink(row.id, wanted, signal);
        })(),
      ]);
    });
    if (Number.isSafeInteger(answer.query.id) && (!seed.isrc || !seed.artwork)) {
      tasks.push(
        (async () => {
          const track = await deezerTrack(answer.query.id, signal, { force: true });
          if (track && track !== "missing") seedDetails.track = track;
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
      return {
        title: row.title,
        artist: row.artist,
        match: Number.isFinite(row.match) ? row.match : null,
        deezer_id: row.id,
        isrc: track?.isrc ?? null,
        artwork: track?.artwork ?? null,
        preview_url: track?.preview ?? null,
        spotify_url: spotify ? spotifyTrackUrl(spotify) : spotifySearchUrl(row.artist, row.title),
        spotify_exact: Boolean(spotify),
        apple_music_url: apple || appleSearchUrl(storefront, row.artist, row.title),
        apple_exact: Boolean(apple),
      };
    });
    return {
      cacheable: !request.storefrontGuessed,
      // An answer missing Deezer's details (rate limit, time budget) is worth asking for again soon.
      maxAge: details.every((state) => state.track) ? RESPONSE_MAX_AGE_S : 20,
      body: {
        ok: true,
        query: {
          title: seed.title,
          artist: seed.artistLine,
          isrc: seed.isrc ?? seedDetails.track?.isrc ?? null,
          artwork: seed.artwork ?? seedDetails.track?.artwork ?? null,
          platform,
          source_url: seed.sourceUrl,
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
          apple_exact: results.filter((row) => row.apple_exact).length,
          spotify_lookup: hasSpotifyCredentials,
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
    let wanted = title && artist ? { title, artist, durationMs: null, isrc: null } : null;
    const track = await deezerTrack(deezerId, signal, { force: true });
    if (track === "missing") throw new MobileError(404, "track_not_found");
    if (track?.title && track.artist) {
      wanted = { title: track.title, artist: track.artist, durationMs: track.durationMs, isrc: track.isrc };
    }
    if (!wanted) throw new MobileError(502, "lookup_unavailable");

    let url;
    let exact = false;
    let retryAfter = null;
    if (platform === "apple") {
      const found = await appleLink(deezerId, wanted, storefront, signal);
      if (found === undefined) retryAfter = Math.max(1, itunesBucket.retryAfter());
      exact = Boolean(found);
      url = found || appleSearchUrl(storefront, wanted.artist, wanted.title);
    } else {
      const found = await spotifyLink(deezerId, wanted, signal);
      exact = Boolean(found);
      url = found ? spotifyTrackUrl(found) : spotifySearchUrl(wanted.artist, wanted.title);
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
