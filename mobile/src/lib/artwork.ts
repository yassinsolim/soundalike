const cache = new Map<number, string | null>();
const nameCache = new Map<string, string | null>();
const MAX_CACHE = 400;

/**
 * Cover art from Deezer's public catalog, matched by the same track ids the
 * recommender already returns. Purely decorative, so failures stay silent.
 */
export async function fetchCover(deezerId: number): Promise<string | null> {
  if (!Number.isFinite(deezerId) || deezerId <= 0) return null;
  if (cache.has(deezerId)) return cache.get(deezerId) ?? null;
  try {
    const response = await fetch(`https://api.deezer.com/track/${deezerId}`);
    if (!response.ok) throw new Error(String(response.status));
    const payload = await response.json();
    const url =
      typeof payload?.album?.cover_medium === "string"
        ? payload.album.cover_medium
        : null;
    if (cache.size >= MAX_CACHE) cache.clear();
    cache.set(deezerId, url);
    return url;
  } catch {
    cache.set(deezerId, null);
    return null;
  }
}

/** Cover art for a seed that came from in-app search, matched by name. */
export async function fetchCoverByName(
  title: string,
  artist: string
): Promise<string | null> {
  const query = `track:"${title.replace(/"/g, "")}" artist:"${artist.replace(/"/g, "")}"`;
  const key = `name:${query}`;
  if (nameCache.has(key)) return nameCache.get(key) ?? null;
  try {
    const response = await fetch(
      `https://api.deezer.com/search?q=${encodeURIComponent(query)}&limit=1`
    );
    if (!response.ok) throw new Error(String(response.status));
    const payload = await response.json();
    const url =
      typeof payload?.data?.[0]?.album?.cover_medium === "string"
        ? payload.data[0].album.cover_medium
        : null;
    if (nameCache.size >= MAX_CACHE) nameCache.clear();
    nameCache.set(key, url);
    return url;
  } catch {
    nameCache.set(key, null);
    return null;
  }
}

/** Resolves covers a few at a time so a full page does not stall the network. */
export async function fetchCovers(
  ids: number[],
  onCover: (id: number, url: string | null) => void,
  concurrency = 4
): Promise<void> {
  const queue = [...ids];
  const workers = Array.from({ length: Math.min(concurrency, queue.length) }, async () => {
    while (queue.length) {
      const id = queue.shift();
      if (id === undefined) return;
      onCover(id, await fetchCover(id));
    }
  });
  await Promise.all(workers);
}
