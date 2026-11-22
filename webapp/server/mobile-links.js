// Pure helpers for the mobile API: shared-link parsing, title/artist
// normalisation, catalogue matching and the links handed back to phones.
// Nothing here touches the network.

const SPOTIFY_ID = "[A-Za-z0-9]{22}";
const SPOTIFY_TRACK = new RegExp(
  `(?:open\\.spotify\\.com/(?:intl-[A-Za-z-]{2,8}/)?(?:embed/)?track/|spotify:track:)(${SPOTIFY_ID})(?![A-Za-z0-9])`,
  "i",
);
const SPOTIFY_OTHER =
  /(?:open\.spotify\.com\/(?:intl-[A-Za-z-]{2,8}\/)?(?:embed\/)?|spotify:)(album|playlist|artist|episode|show|user)[/:]/i;
const SPOTIFY_SHORT =
  /https?:\/\/(?:spotify\.link|spotify\.app\.link|link\.tospotify\.com)\/[A-Za-z0-9_-]{1,64}/i;
const URL_TOKEN = /https?:\/\/[^\s<>"'`]+/gi;
const APPLE_HOSTS = new Set([
  "music.apple.com",
  "geo.music.apple.com",
  "embed.music.apple.com",
  "itunes.apple.com",
]);
const APPLE_ID = /^[1-9][0-9]{0,19}$/;
const STOREFRONT = /^[a-z]{2}$/;

export const SPOTIFY_SHORT_HOSTS = new Set([
  "spotify.link",
  "spotify.app.link",
  "link.tospotify.com",
]);
export const MAX_LINK_LENGTH = 2000;

export function isSpotifyId(value) {
  return typeof value === "string" && new RegExp(`^${SPOTIFY_ID}$`).test(value);
}

export function isAppleId(value) {
  return typeof value === "string" && APPLE_ID.test(value);
}

export function normalizeStorefront(value) {
  if (typeof value !== "string") return null;
  const storefront = value.trim().toLowerCase();
  return STOREFRONT.test(storefront) ? storefront : null;
}

/** ISRC without hyphens, upper case; null when it is not one. */
export function normalizeIsrc(value) {
  if (typeof value !== "string") return null;
  const isrc = value.replace(/[-\s]/g, "").toUpperCase();
  return /^[A-Z]{2}[A-Z0-9]{3}[0-9]{7}$/.test(isrc) ? isrc : null;
}

function parseAppleUrl(url) {
  const segments = url.pathname.split("/").filter(Boolean);
  let storefront = null;
  if (segments.length && STOREFRONT.test(segments[0].toLowerCase())) {
    storefront = segments.shift().toLowerCase();
  }
  const kind = (segments[0] || "").toLowerCase();
  const last = segments[segments.length - 1] || "";
  if (kind === "song") {
    const id = last.replace(/^id/, "");
    return segments.length >= 2 && APPLE_ID.test(id)
      ? { kind: "apple", id, storefront }
      : { error: "invalid_link" };
  }
  if (kind === "album") {
    const song = url.searchParams.get("i");
    if (song !== null) {
      return APPLE_ID.test(song)
        ? { kind: "apple", id: song, storefront }
        : { error: "invalid_link" };
    }
    return { error: "not_a_track" };
  }
  return { error: "unsupported_link" };
}

/**
 * Reads a shared link, or share-sheet text that contains one.
 * Returns {kind:"spotify", id}, {kind:"spotify-short", url},
 * {kind:"apple", id, storefront|null} or {error}: `invalid_link` (nothing we
 * recognise), `not_a_track` (an album link without a song) or
 * `unsupported_link` (a playlist, artist, podcast, ...).
 */
export function parseSharedLink(text) {
  if (typeof text !== "string") return { error: "invalid_link" };
  const input = text.trim();
  if (!input || input.length > MAX_LINK_LENGTH) return { error: "invalid_link" };

  const track = SPOTIFY_TRACK.exec(input);
  if (track) return { kind: "spotify", id: track[1] };
  const short = SPOTIFY_SHORT.exec(input);
  if (short) return { kind: "spotify-short", url: short[0].replace(/^http:/i, "https:") };

  let appleError = null;
  for (const token of input.match(URL_TOKEN) || []) {
    let url;
    try {
      url = new URL(token);
    } catch {
      continue;
    }
    if (!APPLE_HOSTS.has(url.hostname.toLowerCase())) continue;
    const parsed = parseAppleUrl(url);
    if (!parsed.error) return parsed;
    appleError ??= parsed.error;
  }
  if (appleError) return { error: appleError };
  if (SPOTIFY_OTHER.test(input)) return { error: "unsupported_link" };
  return { error: "invalid_link" };
}

const DIACRITICS = /[\u0300-\u036f]/g;
const NON_ALPHANUMERIC = /[^\p{L}\p{N}]+/gu;
const TRAILING_VARIANT = /\s*[([][^)\]]*[)\]]\s*$|\s+-\s+[^-]*$/u;
const FEATURED = /\s*[([]?\s*\b(?:feat\.?|ft\.?|featuring)\s+.*$/iu;
const ARTIST_SPLIT = /\s*(?:,|&|;|\bfeat\.?|\bft\.?|\bfeaturing\b|\bx\b|\band\b|\bwith\b)\s+/iu;

export function normalize(value) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(DIACRITICS, "")
    .toLowerCase()
    .replace(NON_ALPHANUMERIC, " ")
    .trim();
}

/** Drops "(Remastered)", "- Radio Edit" and "feat. X" so releases of a song agree. */
export function coreTitle(value) {
  return normalize(searchTitle(value)) || normalize(value);
}

/** The title as typed into a catalogue search: without trailing version tags. */
export function searchTitle(value) {
  let working = String(value ?? "").replace(FEATURED, "");
  let previous = "";
  while (working !== previous) {
    previous = working;
    working = working.replace(TRAILING_VARIANT, "");
  }
  return working.trim() || String(value ?? "").trim();
}

export function primaryArtist(value) {
  const text = String(value ?? "");
  return normalize(text.split(ARTIST_SPLIT)[0]) || normalize(text);
}

function artistsAgree(wanted, candidate) {
  const a = primaryArtist(wanted);
  const b = primaryArtist(candidate);
  if (!a || !b) return false;
  if (a === b) return true;
  const full = ` ${normalize(candidate)} `;
  const wantedFull = ` ${normalize(wanted)} `;
  return full.includes(` ${a} `) || wantedFull.includes(` ${b} `);
}

/**
 * Picks the catalogue row that is the wanted song: the core title and an
 * artist must agree; the same full title, then the closest duration, win.
 * `rows` are {title, artist, durationMs}; returns the row or null.
 */
export function pickBestMatch(rows, wanted) {
  const wantedCore = coreTitle(wanted.title);
  const wantedFull = normalize(wanted.title);
  if (!wantedCore) return null;
  let best = null;
  let bestScore = -1;
  for (const row of Array.isArray(rows) ? rows : []) {
    if (!row || typeof row.title !== "string" || typeof row.artist !== "string") continue;
    if (coreTitle(row.title) !== wantedCore) continue;
    if (!artistsAgree(wanted.artist, row.artist)) continue;
    let score = 0;
    if (normalize(row.title) === wantedFull) score += 4;
    if (primaryArtist(row.artist) === primaryArtist(wanted.artist)) score += 2;
    if (Number.isFinite(wanted.durationMs) && Number.isFinite(row.durationMs)) {
      const apart = Math.abs(wanted.durationMs - row.durationMs);
      if (apart <= 3000) score += 3;
      else if (apart <= 10000) score += 1;
      else score -= 2;
    }
    if (score > bestScore) {
      best = row;
      bestScore = score;
    }
  }
  return best;
}

function searchTerm(artist, title) {
  return `${String(artist ?? "").trim()} ${String(title ?? "").trim()}`.trim();
}

export function spotifyTrackUrl(id) {
  return `https://open.spotify.com/track/${id}`;
}

export function spotifySearchUrl(artist, title) {
  return `https://open.spotify.com/search/${encodeURIComponent(searchTerm(artist, title))}`;
}

/**
 * A Spotify search for one recording by ISRC. It lists exactly that recording
 * (the form the iPhone app opens), but it is a search page, not the track.
 */
export function spotifyIsrcUrl(isrc) {
  return `https://open.spotify.com/search/isrc%3A${isrc}`;
}

export function appleSearchUrl(storefront, artist, title) {
  return `https://music.apple.com/${storefront}/search?term=${encodeURIComponent(searchTerm(artist, title))}`;
}

/** iTunes puts an affiliate marker (`uo=4`) on its links; phones do not need it. */
export function cleanAppleUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || !APPLE_HOSTS.has(url.hostname)) return null;
    url.searchParams.delete("uo");
    return url.toString();
  } catch {
    return null;
  }
}

/** A larger square of an iTunes artwork URL (they come as .../100x100bb.jpg). */
export function appleArtwork(value, size = 300) {
  if (typeof value !== "string" || !value.startsWith("https://")) return null;
  return value.replace(/\/\d+x\d+bb\.(jpg|png|webp)$/i, `/${size}x${size}bb.$1`);
}
