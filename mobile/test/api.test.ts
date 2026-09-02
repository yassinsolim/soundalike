import { describe, expect, test } from "vitest";

import {
  ApiError,
  SEED_SEPARATOR,
  buildRecommendUrl,
  parseCatalogTracks,
  parseRecommendations,
} from "../src/lib/api";

describe("buildRecommendUrl", () => {
  test("sends the canonical parameters the hosted ranker expects", () => {
    const url = new URL(
      buildRecommendUrl("https://example.test", `Higher${SEED_SEPARATOR}Creed`)
    );
    expect(url.pathname).toBe("/api/spicetify_recommend");
    expect(url.searchParams.get("query")).toBe("Higher \u2014 Creed");
    expect(url.searchParams.get("n")).toBe("20");
    expect(url.searchParams.get("diversity")).toBe("0.15");
    expect(url.searchParams.get("v")).toBe("4");
    expect(url.searchParams.get("language_policy")).toBe("spotify-lyrics-strict-v2");
    expect(url.searchParams.get("ranking_policy")).toBe("model-quality-v1");
  });
});

describe("parseRecommendations", () => {
  const payload = {
    ok: true,
    seed: { title: "Blinding Lights", artist: "The Weeknd" },
    vibe: { tempo: "86 BPM", tone: "neutral", low_end: "balanced low-end" },
    results: [
      {
        title: "Jealous",
        artist: "Nick Jonas",
        deezer_id: 113416766,
        bpm: 92,
        spotify_url: "https://open.spotify.com/search/Jealous",
      },
      { title: "Thinkin", artist: "Mario", deezer_id: 4273654, bpm: null },
    ],
    method: "dual_sonic64_guardrail",
    index_version: "2026.07.11-dual-sonic64",
    library_size: 272853,
  };

  test("numbers results from one and keeps the fields the app renders", () => {
    const set = parseRecommendations(payload);
    expect(set.results.map((result) => result.position)).toEqual([1, 2]);
    expect(set.results[0].bpm).toBe(92);
    expect(set.results[1].bpm).toBeUndefined();
    expect(set.results[0].deezerId).toBe(113416766);
    expect(set.seed).toEqual({ title: "Blinding Lights", artist: "The Weeknd" });
    expect(set.vibe.lowEnd).toBe("balanced low-end");
    expect(set.librarySize).toBe(272853);
  });

  test("rejects malformed and empty responses", () => {
    expect(() => parseRecommendations({ ok: false })).toThrow(ApiError);
    expect(() => parseRecommendations({ ok: true, results: [] })).toThrow(ApiError);
    expect(() =>
      parseRecommendations({ ok: true, results: [{ title: 5, artist: null }] })
    ).toThrow(ApiError);
  });
});

describe("parseCatalogTracks", () => {
  test("keeps well formed rows and drops the rest", () => {
    const tracks = parseCatalogTracks({
      ok: true,
      results: [
        { row: 12, title: "Higher", artist: "Creed" },
        { row: 13, title: 5, artist: "Creed" },
        null,
      ],
    });
    expect(tracks).toEqual([{ row: 12, title: "Higher", artist: "Creed" }]);
  });

  test("returns nothing for a failed response", () => {
    expect(parseCatalogTracks({ ok: false })).toEqual([]);
    expect(parseCatalogTracks(null)).toEqual([]);
  });
});
