import {
  DIVERSITY,
  FALLBACK_API,
  FALLBACK_TIMEOUT_MS,
  HOSTED_API_VERSION,
  LANGUAGE_POLICY,
  PRIMARY_API,
  PRIMARY_TIMEOUT_MS,
  RANKING_POLICY,
  RESULT_COUNT,
} from "./config";
import type { CatalogTrack, RecommendationSet } from "./types";

/** The hosted ranker keys off "Title — Artist" with a spaced em dash. */
export const SEED_SEPARATOR = " \u2014 ";

export class ApiError extends Error {
  readonly kind: "offline" | "not_in_catalog" | "server";

  constructor(kind: ApiError["kind"], message: string) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
  }
}

export async function fetchWithTimeout(
  url: string,
  timeoutMs: number,
  signal?: AbortSignal
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort);
  try {
    return await fetch(url, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

export function buildRecommendUrl(base: string, seed: string): string {
  const params = new URLSearchParams({
    query: seed,
    n: String(RESULT_COUNT),
    diversity: String(DIVERSITY),
    v: HOSTED_API_VERSION,
    language_policy: LANGUAGE_POLICY,
    ranking_policy: RANKING_POLICY,
  });
  return `${base}/api/spicetify_recommend?${params.toString()}`;
}

export function parseRecommendations(payload: any): RecommendationSet {
  if (!payload || payload.ok !== true || !Array.isArray(payload.results)) {
    throw new ApiError("server", "The recommendation service returned nothing usable.");
  }
  const results = payload.results
    .filter((item: any) => item && typeof item.title === "string" && typeof item.artist === "string")
    .map((item: any, index: number) => ({
      position: index + 1,
      title: item.title,
      artist: item.artist,
      deezerId: Number.isFinite(item.deezer_id) ? Number(item.deezer_id) : undefined,
      bpm: Number.isFinite(item.bpm) && item.bpm > 0 ? Math.round(item.bpm) : undefined,
      spotifyUrl: typeof item.spotify_url === "string" ? item.spotify_url : undefined,
    }));
  if (!results.length) {
    throw new ApiError("server", "No similar tracks came back for that song.");
  }
  return {
    seed: {
      title: String(payload.seed?.title ?? ""),
      artist: String(payload.seed?.artist ?? ""),
    },
    vibe: {
      tempo: payload.vibe?.tempo,
      tone: payload.vibe?.tone,
      dynamics: payload.vibe?.dynamics,
      lowEnd: payload.vibe?.low_end,
    },
    results,
    method: typeof payload.method === "string" ? payload.method : "unknown",
    indexVersion: typeof payload.index_version === "string" ? payload.index_version : "unknown",
    librarySize: Number.isFinite(payload.library_size) ? Number(payload.library_size) : 0,
  };
}

async function requestRecommendations(
  base: string,
  seed: string,
  timeoutMs: number,
  signal?: AbortSignal
): Promise<RecommendationSet> {
  const response = await fetchWithTimeout(buildRecommendUrl(base, seed), timeoutMs, signal);
  if (response.status === 422) {
    throw new ApiError("not_in_catalog", "That song is not in the Soundalike library yet.");
  }
  if (!response.ok) {
    throw new ApiError("server", `The recommendation service replied with ${response.status}.`);
  }
  return parseRecommendations(await response.json());
}

/**
 * Always-on host first, Vercel second. A missing catalog entry is a real answer,
 * so it is never retried against the second host.
 */
export async function fetchRecommendations(
  title: string,
  artist: string,
  signal?: AbortSignal
): Promise<RecommendationSet> {
  const seed = `${title.trim()}${SEED_SEPARATOR}${artist.trim()}`;
  try {
    return await requestRecommendations(PRIMARY_API, seed, PRIMARY_TIMEOUT_MS, signal);
  } catch (error) {
    if (error instanceof ApiError && error.kind === "not_in_catalog") throw error;
    if (signal?.aborted) throw error;
  }
  try {
    return await requestRecommendations(FALLBACK_API, seed, FALLBACK_TIMEOUT_MS, signal);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("offline", "Could not reach Soundalike. Check your connection and try again.");
  }
}

export function parseCatalogTracks(payload: any): CatalogTrack[] {
  if (!payload || payload.ok !== true || !Array.isArray(payload.results)) return [];
  return payload.results
    .filter(
      (item: any) =>
        item && typeof item.title === "string" && typeof item.artist === "string"
    )
    .map((item: any) => ({
      row: Number.isFinite(item.row) ? Number(item.row) : -1,
      title: item.title,
      artist: item.artist,
    }));
}

export async function searchCatalog(
  query: string,
  limit = 12,
  signal?: AbortSignal
): Promise<CatalogTrack[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  const params = new URLSearchParams({
    q: trimmed.slice(0, 120),
    limit: String(limit),
  });
  const response = await fetchWithTimeout(
    `${FALLBACK_API}/api/search?${params.toString()}`,
    PRIMARY_TIMEOUT_MS,
    signal
  );
  if (!response.ok) return [];
  return parseCatalogTracks(await response.json());
}
