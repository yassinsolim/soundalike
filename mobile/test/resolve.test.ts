import { afterEach, describe, expect, test, vi } from "vitest";

import {
  coreTitle,
  normalize,
  parseEmbedPage,
  primaryArtist,
  rankCatalogMatches,
  readSharedTrack,
} from "../src/lib/resolve";
import type { CatalogTrack, SharedTrack } from "../src/lib/types";

describe("normalize", () => {
  test("folds case, accents, and punctuation", () => {
    expect(normalize("Déjà Vu!")).toBe("deja vu");
    expect(normalize("Thinkin' About You")).toBe("thinkin about you");
  });
});

describe("coreTitle", () => {
  test("drops version suffixes so variants still line up", () => {
    expect(coreTitle("Higher (Remastered 2019)")).toBe("higher");
    expect(coreTitle("Higher - Radio Edit")).toBe("higher");
    expect(coreTitle("Higher [Live]")).toBe("higher");
    expect(coreTitle("Higher (feat. Someone)")).toBe("higher");
    expect(coreTitle("Higher feat. Someone")).toBe("higher");
  });

  test("never reduces a title to nothing", () => {
    expect(coreTitle("(Interlude)")).toBe("interlude");
  });
});

describe("primaryArtist", () => {
  test("keeps the lead artist only", () => {
    expect(primaryArtist("The Weeknd, Daft Punk")).toBe("the weeknd");
    expect(primaryArtist("Nick Jonas feat. Tinashe")).toBe("nick jonas");
    expect(primaryArtist("Jay-Z & Kanye West")).toBe("jay z");
  });
});

const catalog: CatalogTrack[] = [
  { row: 1, title: "Higher", artist: "Creed" },
  { row: 2, title: "Higher", artist: "Clean Bandit" },
  { row: 3, title: "Higher Love", artist: "Kygo" },
];

function shared(partial: Partial<SharedTrack>): SharedTrack {
  return { trackId: "0VjIjW4GlUZAMYd2vXMi3b", ...partial };
}

describe("rankCatalogMatches", () => {
  test("marks the title and artist agreement as exact", () => {
    const ranked = rankCatalogMatches(
      shared({ title: "Higher", artist: "Creed" }),
      catalog
    );
    expect(ranked[0].track.artist).toBe("Creed");
    expect(ranked[0].exact).toBe(true);
    expect(ranked.filter((entry) => entry.exact)).toHaveLength(1);
  });

  test("treats every title match as exact when no artist is known", () => {
    const ranked = rankCatalogMatches(shared({ title: "Higher" }), catalog);
    const exact = ranked.filter((entry) => entry.exact);
    expect(exact).toHaveLength(2);
    expect(exact.map((entry) => entry.track.artist).sort()).toEqual([
      "Clean Bandit",
      "Creed",
    ]);
  });

  test("still offers near matches when nothing agrees exactly", () => {
    const ranked = rankCatalogMatches(
      shared({ title: "Higher Love", artist: "Whitney Houston" }),
      catalog
    );
    expect(ranked.length).toBeGreaterThan(0);
    expect(ranked.every((entry) => entry.exact)).toBe(false);
    expect(ranked[0].track.title).toBe("Higher Love");
  });

  test("matches a remastered share against the plain catalog title", () => {
    const ranked = rankCatalogMatches(
      shared({ title: "Higher (Remastered)", artist: "Creed" }),
      catalog
    );
    expect(ranked[0].exact).toBe(true);
    expect(ranked[0].track.row).toBe(1);
  });
});

function embedHtml(entity: unknown): string {
  const payload = JSON.stringify({ props: { pageProps: { state: { data: { entity } } } } });
  return `<html><body><script id="__NEXT_DATA__" type="application/json">${payload}</script></body></html>`;
}

const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

describe("parseEmbedPage", () => {
  test("reads the title, lead artist, and a usable cover", () => {
    const html = embedHtml({
      name: "Blinding Lights",
      artists: [{ name: "The Weeknd" }],
      visualIdentity: {
        image: [
          { url: "https://cdn/64.jpg", maxWidth: 64 },
          { url: "https://cdn/300.jpg", maxWidth: 300 },
          { url: "https://cdn/640.jpg", maxWidth: 640 },
        ],
      },
    });

    expect(parseEmbedPage(html)).toEqual({
      title: "Blinding Lights",
      artist: "The Weeknd",
      artworkUrl: "https://cdn/300.jpg",
    });
  });

  test("falls back to the largest cover when none reaches the preferred size", () => {
    const html = embedHtml({
      name: "Tiny",
      artists: [{ name: "Someone" }],
      visualIdentity: { image: [{ url: "https://cdn/64.jpg", maxWidth: 64 }] },
    });

    expect(parseEmbedPage(html)?.artworkUrl).toBe("https://cdn/64.jpg");
  });

  test("returns null for a page without the embedded payload", () => {
    expect(parseEmbedPage("<html><body>nope</body></html>")).toBeNull();
  });

  test("returns null when the payload is not the shape we expect", () => {
    expect(parseEmbedPage(embedHtml({ nothing: true }))).toBeNull();
    expect(
      parseEmbedPage('<script id="__NEXT_DATA__" type="application/json">{broken</script>')
    ).toBeNull();
  });
});

describe("readSharedTrack", () => {
  test("prefers the embed page and never calls oEmbed when it succeeds", async () => {
    const calls: string[] = [];
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      return {
        ok: true,
        status: 200,
        text: async () =>
          embedHtml({
            name: "Blinding Lights",
            artists: [{ name: "The Weeknd" }],
            visualIdentity: { image: [{ url: "https://cdn/300.jpg", maxWidth: 300 }] },
          }),
      } as unknown as Response;
    }) as unknown as typeof fetch;

    const shared = await readSharedTrack(
      "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b"
    );

    expect(shared).toEqual({
      trackId: "0VjIjW4GlUZAMYd2vXMi3b",
      title: "Blinding Lights",
      artist: "The Weeknd",
      artworkUrl: "https://cdn/300.jpg",
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toContain("/embed/track/");
  });

  test("falls back to oEmbed and the shared label when the embed page fails", async () => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/embed/track/")) {
        return { ok: false, status: 503, text: async () => "" } as unknown as Response;
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          title: "Blinding Lights",
          thumbnail_url: "https://i.scdn.co/image/fallback",
        }),
      } as unknown as Response;
    }) as unknown as typeof fetch;

    const shared = await readSharedTrack(
      "Blinding Lights by The Weeknd\nhttps://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b"
    );

    expect(shared?.title).toBe("Blinding Lights");
    expect(shared?.artist).toBe("The Weeknd");
    expect(shared?.artworkUrl).toBe("https://i.scdn.co/image/fallback");
  });

  test("rejects text that holds no Spotify track", async () => {
    globalThis.fetch = vi.fn() as unknown as typeof fetch;
    await expect(readSharedTrack("just some words")).resolves.toBeNull();
  });
});
