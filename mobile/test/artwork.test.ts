import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchCover, fetchCoverByName, fetchCovers } from "../src/lib/artwork";

const originalFetch = globalThis.fetch;

function jsonResponse(body: unknown, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
  } as unknown as Response;
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("fetchCover", () => {
  it("returns the medium cover for a track id", async () => {
    globalThis.fetch = vi.fn(async () =>
      jsonResponse({ album: { cover_medium: "https://cdn/1.jpg" } })
    ) as unknown as typeof fetch;

    await expect(fetchCover(101)).resolves.toBe("https://cdn/1.jpg");
  });

  it("caches a result so a repeat lookup makes no request", async () => {
    const spy = vi.fn(async () =>
      jsonResponse({ album: { cover_medium: "https://cdn/2.jpg" } })
    );
    globalThis.fetch = spy as unknown as typeof fetch;

    await fetchCover(202);
    await fetchCover(202);

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("stays silent when the request fails", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new Error("network down");
    }) as unknown as typeof fetch;

    await expect(fetchCover(303)).resolves.toBeNull();
  });

  it("rejects ids that cannot be real without making a request", async () => {
    const spy = vi.fn();
    globalThis.fetch = spy as unknown as typeof fetch;

    await expect(fetchCover(0)).resolves.toBeNull();
    await expect(fetchCover(Number.NaN)).resolves.toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("fetchCoverByName", () => {
  it("searches Deezer with a quoted track and artist query", async () => {
    const spy = vi.fn(async (_input?: unknown) =>
      jsonResponse({ data: [{ album: { cover_medium: "https://cdn/seed.jpg" } }] })
    );
    globalThis.fetch = spy as unknown as typeof fetch;

    await expect(fetchCoverByName("Blinding Lights", "The Weeknd")).resolves.toBe(
      "https://cdn/seed.jpg"
    );

    const url = String(spy.mock.calls[0]?.[0]);
    expect(url).toContain("api.deezer.com/search");
    expect(decodeURIComponent(url)).toContain('track:"Blinding Lights"');
    expect(decodeURIComponent(url)).toContain('artist:"The Weeknd"');
  });

  it("returns null when the search has no hits", async () => {
    globalThis.fetch = vi.fn(async () => jsonResponse({ data: [] })) as unknown as typeof fetch;

    await expect(fetchCoverByName("Nothing Here", "Nobody")).resolves.toBeNull();
  });
});

describe("fetchCovers", () => {
  it("reports every id and never runs more workers than requested", async () => {
    let inFlight = 0;
    let peak = 0;
    globalThis.fetch = vi.fn(async () => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 5));
      inFlight -= 1;
      return jsonResponse({ album: { cover_medium: "https://cdn/x.jpg" } });
    }) as unknown as typeof fetch;

    const ids = [11, 12, 13, 14, 15, 16, 17];
    const seen: number[] = [];
    await fetchCovers(ids, (id) => seen.push(id), 3);

    expect(seen.sort((a, b) => a - b)).toEqual(ids);
    expect(peak).toBeLessThanOrEqual(3);
  });
});
