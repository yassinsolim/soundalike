import { RESOLVE_TIMEOUT_MS } from "./config";
import { fetchWithTimeout, searchCatalog } from "./api";
import { findShortLink, parseSharedLabel, parseTrackId, trackUrl } from "./spotify";
import type { CatalogTrack, SharedTrack } from "./types";

const DIACRITICS = /[\u0300-\u036f]/g;
const NON_ALPHANUMERIC = /[^\p{L}\p{N}]+/gu;
const TRAILING_VARIANT = /\s*[([][^)\]]*[)\]]\s*$|\s+-\s+[^-]*$/u;
const FEATURED = /\s*(?:feat\.?|ft\.?|featuring|with)\s+.*$/iu;

export function normalize(value: string): string {
  return value
    .normalize("NFD")
    .replace(DIACRITICS, "")
    .toLowerCase()
    .replace(NON_ALPHANUMERIC, " ")
    .trim();
}

/** Drops "(Remastered)", "- Radio Edit" and "feat. X" so variants still match. */
export function coreTitle(value: string): string {
  let working = value.replace(FEATURED, "");
  let previous = "";
  while (working !== previous) {
    previous = working;
    working = working.replace(TRAILING_VARIANT, "");
  }
  return normalize(working || value);
}

export function primaryArtist(value: string): string {
  return normalize(value.split(/\s*(?:,|&|feat\.?|ft\.?|featuring|x|and)\s+/iu)[0] || value);
}

/** Expands spotify.link short links by reading the final URL after redirects. */
export async function expandShortLink(url: string, signal?: AbortSignal): Promise<string | null> {
  try {
    const response = await fetchWithTimeout(url, RESOLVE_TIMEOUT_MS, signal);
    return parseTrackId(response.url ?? "");
  } catch {
    return null;
  }
}

type OEmbed = { title?: string; thumbnail_url?: string };

/** Spotify's public oEmbed endpoint. No account, no token, no user login. */
export async function fetchOEmbed(
  trackId: string,
  signal?: AbortSignal
): Promise<OEmbed | null> {
  try {
    const response = await fetchWithTimeout(
      `https://open.spotify.com/oembed?url=${encodeURIComponent(trackUrl(trackId))}`,
      RESOLVE_TIMEOUT_MS,
      signal
    );
    if (!response.ok) return null;
    const payload = await response.json();
    return {
      title: typeof payload?.title === "string" ? payload.title : undefined,
      thumbnail_url:
        typeof payload?.thumbnail_url === "string" ? payload.thumbnail_url : undefined,
    };
  } catch {
    return null;
  }
}

export type SharedMeta = { title?: string; artist?: string; artworkUrl?: string };

const NEXT_DATA = /<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/;

function pickImage(images: unknown): string | undefined {
  if (!Array.isArray(images)) return undefined;
  const usable = images
    .filter(
      (image): image is { url: string; maxWidth?: number } =>
        typeof (image as { url?: unknown })?.url === "string"
    )
    .sort((a, b) => (a.maxWidth ?? 0) - (b.maxWidth ?? 0));
  return (usable.find((image) => (image.maxWidth ?? 0) >= 200) ?? usable[usable.length - 1])?.url;
}

/**
 * Reads the public embed page, which unlike oEmbed also names the artist. That
 * turns most shares into a certain match instead of a pick-one prompt.
 */
export function parseEmbedPage(html: string): SharedMeta | null {
  const match = NEXT_DATA.exec(html);
  if (!match) return null;
  try {
    const entity = JSON.parse(match[1])?.props?.pageProps?.state?.data?.entity;
    if (!entity) return null;
    const title = typeof entity.name === "string" ? entity.name : undefined;
    const artist =
      typeof entity.artists?.[0]?.name === "string" ? entity.artists[0].name : undefined;
    if (!title && !artist) return null;
    return { title, artist, artworkUrl: pickImage(entity.visualIdentity?.image) };
  } catch {
    return null;
  }
}

export async function fetchEmbedMetadata(
  trackId: string,
  signal?: AbortSignal
): Promise<SharedMeta | null> {
  try {
    const response = await fetchWithTimeout(
      `https://open.spotify.com/embed/track/${encodeURIComponent(trackId)}`,
      RESOLVE_TIMEOUT_MS,
      signal
    );
    if (!response.ok) return null;
    return parseEmbedPage(await response.text());
  } catch {
    return null;
  }
}

export async function readSharedTrack(
  text: string,
  signal?: AbortSignal
): Promise<SharedTrack | null> {
  let trackId = parseTrackId(text);
  if (!trackId) {
    const short = findShortLink(text);
    if (short) trackId = await expandShortLink(short, signal);
  }
  if (!trackId) return null;

  const label = parseSharedLabel(text);
  const embed = await fetchEmbedMetadata(trackId, signal);
  const oembed = embed?.title ? null : await fetchOEmbed(trackId, signal);
  return {
    trackId,
    title: embed?.title ?? oembed?.title ?? label?.title,
    artist: embed?.artist ?? label?.artist,
    artworkUrl: embed?.artworkUrl ?? oembed?.thumbnail_url,
  };
}

export type SeedMatch = {
  track: CatalogTrack;
  exact: boolean;
};

/**
 * Ranks catalog hits against what the share sheet gave us. An exact title and
 * artist agreement is treated as certain; everything else is offered as a choice.
 */
export function rankCatalogMatches(
  shared: SharedTrack,
  candidates: CatalogTrack[]
): SeedMatch[] {
  const wantedTitle = shared.title ? coreTitle(shared.title) : "";
  const wantedArtist = shared.artist ? primaryArtist(shared.artist) : "";

  const scored = candidates.map((track) => {
    const titleMatch = wantedTitle ? coreTitle(track.title) === wantedTitle : false;
    const artistMatch = wantedArtist ? primaryArtist(track.artist) === wantedArtist : false;
    const titleContains =
      wantedTitle && !titleMatch
        ? coreTitle(track.title).includes(wantedTitle) || wantedTitle.includes(coreTitle(track.title))
        : false;
    let score = 0;
    if (titleMatch) score += 4;
    else if (titleContains) score += 1;
    if (artistMatch) score += 3;
    return { track, exact: titleMatch && (artistMatch || !wantedArtist), score };
  });

  return scored
    .filter((entry) => entry.score > 0 || !wantedTitle)
    .sort((a, b) => b.score - a.score)
    .map(({ track, exact }) => ({ track, exact }));
}

export type Resolution =
  | { kind: "resolved"; shared: SharedTrack; track: CatalogTrack }
  | { kind: "choose"; shared: SharedTrack; matches: CatalogTrack[] }
  | { kind: "missing"; shared: SharedTrack }
  | { kind: "unsupported" };

/**
 * Turns shared text into a catalog seed. Auto-continues when exactly one track
 * agrees on title and artist, otherwise asks the listener to pick.
 */
export async function resolveSharedText(
  text: string,
  signal?: AbortSignal
): Promise<Resolution> {
  const shared = await readSharedTrack(text, signal);
  if (!shared) return { kind: "unsupported" };
  if (!shared.title) return { kind: "missing", shared };

  const query = shared.artist ? `${shared.title} ${shared.artist}` : shared.title;
  let candidates = await searchCatalog(query, 20, signal);
  if (!candidates.length && shared.artist) {
    candidates = await searchCatalog(shared.title, 20, signal);
  }
  const ranked = rankCatalogMatches(shared, candidates);
  if (!ranked.length) return { kind: "missing", shared };

  const exact = ranked.filter((entry) => entry.exact);
  if (exact.length === 1) return { kind: "resolved", shared, track: exact[0].track };
  if (exact.length > 1) {
    return { kind: "choose", shared, matches: exact.map((entry) => entry.track) };
  }
  return { kind: "choose", shared, matches: ranked.slice(0, 8).map((entry) => entry.track) };
}
