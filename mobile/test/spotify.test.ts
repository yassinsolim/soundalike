import { describe, expect, test } from "vitest";

import {
  findShortLink,
  parseSharedLabel,
  parseTrackId,
  searchUris,
} from "../src/lib/spotify";

describe("parseTrackId", () => {
  test("reads a plain share link", () => {
    expect(
      parseTrackId("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=abc123")
    ).toBe("0VjIjW4GlUZAMYd2vXMi3b");
  });

  test("reads a localized share link", () => {
    expect(
      parseTrackId("https://open.spotify.com/intl-de/track/0VjIjW4GlUZAMYd2vXMi3b")
    ).toBe("0VjIjW4GlUZAMYd2vXMi3b");
  });

  test("reads the app uri scheme", () => {
    expect(parseTrackId("spotify:track:0VjIjW4GlUZAMYd2vXMi3b")).toBe(
      "0VjIjW4GlUZAMYd2vXMi3b"
    );
  });

  test("finds a link buried in shared text", () => {
    const text =
      "Blinding Lights by The Weeknd\nhttps://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=x";
    expect(parseTrackId(text)).toBe("0VjIjW4GlUZAMYd2vXMi3b");
  });

  test("ignores albums, playlists, and junk", () => {
    expect(parseTrackId("https://open.spotify.com/album/0VjIjW4GlUZAMYd2vXMi3b")).toBe(
      null
    );
    expect(parseTrackId("https://example.com/track/abc")).toBe(null);
    expect(parseTrackId("")).toBe(null);
  });
});

describe("findShortLink", () => {
  test("detects links that need expanding", () => {
    expect(findShortLink("check this https://spotify.link/aBc123 out")).toBe(
      "https://spotify.link/aBc123"
    );
  });

  test("returns null for a normal link", () => {
    expect(findShortLink("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b")).toBe(
      null
    );
  });
});

describe("parseSharedLabel", () => {
  test("pulls title and artist from the share sheet line", () => {
    const text =
      "Blinding Lights by The Weeknd\nhttps://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b";
    expect(parseSharedLabel(text)).toEqual({
      title: "Blinding Lights",
      artist: "The Weeknd",
    });
  });

  test("ignores the url line itself", () => {
    expect(parseSharedLabel("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b")).toBe(
      null
    );
  });

  test("returns null when there is no by line", () => {
    expect(parseSharedLabel("just some text")).toBe(null);
  });
});

describe("searchUris", () => {
  test("prefers the app scheme and falls back to the web player", () => {
    const [app, web] = searchUris("Take My Breath", "The Weeknd");
    expect(app.startsWith("spotify:search:")).toBe(true);
    expect(web.startsWith("https://open.spotify.com/search/")).toBe(true);
    expect(app).toContain("Take%20My%20Breath");
  });
});
