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

const VERSION_WORDS =
  /\b(live|ao vivo|en vivo|en directo|remix|mix|acoustic|acustico|unplugged|instrumental|karaoke|demo|sped up|slowed|cover)\b/g;

/** The words in a title that make it another recording ("live", "remix"), sorted; "" for the plain song. */
export function versionWords(value) {
  return [...new Set(normalize(value).match(VERSION_WORDS) ?? [])].sort().join(" ");
}

// The language a title or name in another script is most likely written in.
// Sent as Accept-Language, Spotify's embed page then gives the artist's name
// in that script and Deezer the title it holds for it. Deezer answers 晴天 for
// "ja" and "Sunny Day" for every Chinese locale, so Han asks for both.
const SCRIPT_LANGUAGES = [
  [/[\u3040-\u30ff]/u, ["ja"]],
  [/\p{Script=Han}/u, ["zh-TW", "ja"]],
  [/\p{Script=Hangul}/u, ["ko"]],
  [/\p{Script=Cyrillic}/u, ["ru"]],
  [/\p{Script=Arabic}/u, ["ar"]],
  [/\p{Script=Hebrew}/u, ["he"]],
  [/\p{Script=Greek}/u, ["el"]],
  [/\p{Script=Thai}/u, ["th"]],
  [/\p{Script=Devanagari}/u, ["hi"]],
];

/** Languages for the scripts in `text`, as an Accept-Language value ("zh-TW,ja;q=0.9"); null for Latin text. */
export function scriptLanguages(text) {
  const languages = [];
  for (const [script, list] of SCRIPT_LANGUAGES) {
    if (!script.test(String(text ?? ""))) continue;
    for (const language of list) if (!languages.includes(language)) languages.push(language);
  }
  if (!languages.length) return null;
  return languages.map((language, index) => (index ? `${language};q=${(1 - index / 10).toFixed(1)}` : language)).join(",");
}

// Letters that are written two ways in the same word. Decomposed, the Arabic
// alef forms are an alef and a mark, so dropping the marks (with the vowel
// marks and the tatweel) folds them; yeh and teh marbuta have a second
// letter. Cyrillic yo folds with the other accents in `normalize`.
const LETTER_FOLDS = [
  [/[\u064b-\u065f\u0670\u0640]/g, ""],
  [/\u0671/g, "\u0627"],
  [/\u0649/g, "\u064a"],
  [/\u0629/g, "\u0647"],
];

/** A title for comparison across spellings: its core, with interchangeable letters folded. */
export function looseTitle(value) {
  let folded = String(value ?? "").normalize("NFD");
  for (const [pattern, replacement] of LETTER_FOLDS) folded = folded.replace(pattern, replacement);
  return coreTitle(folded);
}

export const SEED_DURATION_TOLERANCE_S = 2;

/** Whether a Deezer length (whole seconds) is the seed's length, give or take two seconds. */
export function sameLength(seedDurationMs, deezerSeconds) {
  return (
    Number.isFinite(seedDurationMs) &&
    Number.isFinite(deezerSeconds) &&
    Math.abs(Math.round(seedDurationMs / 1000) - deezerSeconds) <= SEED_DURATION_TOLERANCE_S
  );
}

/**
 * Picks the Deezer recording that is the shared song, when the names alone
 * did not find it (Deezer lists 晴天 as "Sunny Day"). It never guesses: a
 * candidate needs the seed's length and one more piece of evidence.
 *
 * `seed` is {title, durationMs, album?}. `candidates` are Deezer tracks of
 * the seed's artists: {id, titles[] (in every language asked), durationS,
 * album, isrc, rank}.
 *
 * Evidence, stronger first; the first kind that any candidate has decides:
 *  1. the same title in one of the languages, and the same kind of version;
 *  2. the same album name (Apple Music links name the album).
 * With title evidence several releases of the song may remain: the ISRC most
 * of them carry wins, then the most played. With album evidence more than
 * one recording remaining means it is not known which, and the answer is
 * null. The length alone is never enough, however few tracks have it: an
 * artist's other songs are too often within two seconds (three of Jay Chou's
 * 63 are, for 晴天), and the shared song may be missing from Deezer.
 */
export function pickSeedRecording(seed, candidates) {
  const fitting = (Array.isArray(candidates) ? candidates : []).filter((candidate) =>
    sameLength(seed.durationMs, candidate?.durationS),
  );
  if (!fitting.length) return null;
  const wantedTitle = looseTitle(seed.title);
  const wantedVersion = versionWords(seed.title);
  const byTitle = fitting.filter((candidate) =>
    (candidate.titles || []).some(
      (title) => wantedTitle && looseTitle(title) === wantedTitle && versionWords(title) === wantedVersion,
    ),
  );
  const recordingOf = (candidate) => candidate.isrc || `id:${candidate.id}`;
  if (byTitle.length) {
    const counts = new Map();
    for (const candidate of byTitle) counts.set(recordingOf(candidate), (counts.get(recordingOf(candidate)) || 0) + 1);
    return [...byTitle].sort(
      (a, b) => counts.get(recordingOf(b)) - counts.get(recordingOf(a)) || (b.rank || 0) - (a.rank || 0) || a.id - b.id,
    )[0];
  }
  const wantedAlbum = normalize(seed.album);
  const byAlbum = wantedAlbum ? fitting.filter((candidate) => normalize(candidate.album) === wantedAlbum) : [];
  if (byAlbum.length && new Set(byAlbum.map(recordingOf)).size === 1) {
    return [...byAlbum].sort((a, b) => (b.rank || 0) - (a.rank || 0) || a.id - b.id)[0];
  }
  return null;
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

/**
 * The ways one song is put to ListenBrainz's Spotify lookup, best first. It
 * matches the artist credit and the title as written, so each spelling finds
 * songs the others miss. Measured on 240 results of 12 seeds in 12 languages:
 * Deezer's full credit with the title as given found 152, the result's own
 * artist and title 123, the full credit with the plain title 140, the first
 * artist with the plain title 117; together 157. The album name is passed
 * along but changed nothing in that sample.
 * `wanted` is {title, artist, album?, contributors?}.
 */
export function spotifyMetadataQueries(wanted) {
  const title = String(wanted.title ?? "").trim();
  const artist = String(wanted.artist ?? "").trim();
  const album = String(wanted.album ?? "").trim();
  const credit = Array.isArray(wanted.contributors) && wanted.contributors.length ? wanted.contributors.join(", ") : artist;
  const plain = searchTitle(title);
  const first = artist.split(ARTIST_SPLIT)[0].trim() || artist;
  const seen = new Set();
  const queries = [];
  for (const [artistName, trackName, releaseName] of [
    [credit, title, album],
    [artist, title, album],
    [credit, plain, ""],
    [first, plain, ""],
  ]) {
    const key = `${artistName.toLowerCase()}\n${trackName.toLowerCase()}`;
    if (!artistName || !trackName || seen.has(key)) continue;
    seen.add(key);
    // release_name is required by the service, even when empty.
    queries.push({ artist_name: artistName, release_name: releaseName, track_name: trackName });
  }
  return queries;
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
