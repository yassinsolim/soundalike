const TRACK_ID = "([A-Za-z0-9]{22})";
const TRACK_LINK = new RegExp(
  `(?:open\\.spotify\\.com/(?:[A-Za-z0-9-]+/)?track/|spotify:track:)${TRACK_ID}`
);
const SHORT_LINK = /https?:\/\/(?:spotify\.link|link\.tospotify\.com)\/[A-Za-z0-9]+/i;
const BY_LINE = /^\s*(.+?)\s+by\s+(.+?)\s*$/i;

/** Pulls a Spotify track id out of a shared link or an arbitrary block of text. */
export function parseTrackId(text: string): string | null {
  if (typeof text !== "string" || !text) return null;
  const match = TRACK_LINK.exec(text);
  return match ? match[1] : null;
}

/** Finds a Spotify short link that has to be expanded before the id is visible. */
export function findShortLink(text: string): string | null {
  if (typeof text !== "string" || !text) return null;
  const match = SHORT_LINK.exec(text);
  return match ? match[0] : null;
}

/**
 * Spotify's share sheet often prefixes the link with "Title by Artist".
 * That line is a hint only, so anything unparseable is silently ignored.
 */
export function parseSharedLabel(
  text: string
): { title: string; artist: string } | null {
  if (typeof text !== "string" || !text) return null;
  for (const rawLine of text.split(/[\r\n]+/)) {
    const line = rawLine.trim();
    if (!line || TRACK_LINK.test(line) || /^https?:\/\//i.test(line)) continue;
    const match = BY_LINE.exec(line);
    if (!match) continue;
    const title = match[1].trim();
    const artist = match[2].trim();
    if (title && artist && title.length <= 300 && artist.length <= 300) {
      return { title, artist };
    }
  }
  return null;
}

export function trackUrl(trackId: string): string {
  return `https://open.spotify.com/track/${trackId}`;
}

export function searchQuery(title: string, artist: string): string {
  return `${title} ${artist}`.trim();
}

/** App URI first so tapping a result lands in Spotify rather than a browser tab. */
export function searchUris(title: string, artist: string): string[] {
  const query = searchQuery(title, artist);
  return [
    `spotify:search:${encodeURIComponent(query)}`,
    `https://open.spotify.com/search/${encodeURIComponent(query)}`,
  ];
}
