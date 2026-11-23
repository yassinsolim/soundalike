import assert from "node:assert/strict";
import test from "node:test";

import {
  appleArtwork,
  appleSearchUrl,
  cleanAppleUrl,
  coreTitle,
  normalizeIsrc,
  parseSharedLink,
  pickBestMatch,
  primaryArtist,
  searchTitle,
  spotifyMetadataQueries,
  spotifySearchUrl,
  versionWords,
} from "../server/mobile-links.js";

const ID = "4cOdK2wGLETKBW3PvgPWqT";
const spotify = { kind: "spotify", id: ID };
const apple = (id, storefront) => ({ kind: "apple", id, storefront });

const LINKS = [
  // Spotify track links
  [`https://open.spotify.com/track/${ID}`, spotify],
  [`https://open.spotify.com/track/${ID}?si=0a1b2c3d4e5f6a7b`, spotify],
  [`https://open.spotify.com/track/${ID}?si=abc&utm_source=copy-link&nd=1`, spotify],
  [`https://open.spotify.com/intl-de/track/${ID}?si=abc`, spotify],
  [`https://open.spotify.com/intl-pt-BR/track/${ID}`, spotify],
  [`http://open.spotify.com/track/${ID}/`, spotify],
  [`open.spotify.com/track/${ID}`, spotify],
  [`https://open.spotify.com/embed/track/${ID}?utm_source=oembed`, spotify],
  [`spotify:track:${ID}`, spotify],
  [`  spotify:track:${ID}\n`, spotify],
  [`Never Gonna Give You Up by Rick Astley\nhttps://open.spotify.com/track/${ID}?si=x`, spotify],
  // Spotify short links
  ["https://spotify.link/AbCdEf123", { kind: "spotify-short", url: "https://spotify.link/AbCdEf123" }],
  ["https://spotify.app.link/AbCdEf123?_p=c1", { kind: "spotify-short", url: "https://spotify.app.link/AbCdEf123" }],
  ["http://link.tospotify.com/xYz", { kind: "spotify-short", url: "https://link.tospotify.com/xYz" }],
  ["Listen to this https://spotify.link/AbCdEf123 now", { kind: "spotify-short", url: "https://spotify.link/AbCdEf123" }],
  // Apple Music song links
  ["https://music.apple.com/us/album/never-gonna-give-you-up/1559885420?i=1559885421", apple("1559885421", "us")],
  ["https://music.apple.com/us/album/never-gonna-give-you-up/1559885420?i=1559885421&uo=4", apple("1559885421", "us")],
  ["https://music.apple.com/jp/album/%E3%82%A2%E3%82%A4%E3%83%89%E3%83%AB/1679278166?i=1679278167", apple("1679278167", "jp")],
  ["https://music.apple.com/DE/album/99-luftballons/1446013584?i=1446013598&l=en", apple("1446013598", "de")],
  ["https://music.apple.com/de/song/99-luftballons/1446013598", apple("1446013598", "de")],
  ["https://music.apple.com/fr/song/alors-on-danse-radio-edit/1440520983?l=en-GB", apple("1440520983", "fr")],
  ["https://music.apple.com/album/envolver/1593185587?i=1593185593", apple("1593185593", null)],
  ["https://geo.music.apple.com/us/album/x/1559885420?i=1559885421&app=music", apple("1559885421", "us")],
  ["https://itunes.apple.com/us/album/never-gonna-give-you-up/id1559885420?i=1559885421", apple("1559885421", "us")],
  ["Envolver by Anitta https://music.apple.com/br/album/envolver/1593185587?i=1593185593", apple("1593185593", "br")],
  // Things that are not one song
  ["https://music.apple.com/us/album/whenever-you-need-somebody/1559885420", { error: "not_a_track" }],
  ["https://music.apple.com/us/playlist/todays-hits/pl.f4d106fed2bd41149aaacabb233eb5eb", { error: "unsupported_link" }],
  ["https://music.apple.com/us/artist/rick-astley/669771", { error: "unsupported_link" }],
  ["https://music.apple.com/us/album/x/1559885420?i=abc", { error: "invalid_link" }],
  ["https://music.apple.com/us/song/slug-only", { error: "invalid_link" }],
  ["https://open.spotify.com/album/6eUW0wxWtzkFdaEFsTJto6", { error: "unsupported_link" }],
  ["https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=x", { error: "unsupported_link" }],
  ["spotify:album:6eUW0wxWtzkFdaEFsTJto6", { error: "unsupported_link" }],
  ["https://open.spotify.com/episode/4cOdK2wGLETKBW3PvgPWqT", { error: "unsupported_link" }],
  // Bad input
  ["https://open.spotify.com/track/short", { error: "invalid_link" }],
  [`https://open.spotify.com/track/${ID}extra`, { error: "invalid_link" }],
  [`https://open.spotify.com.evil.example/nothing`, { error: "invalid_link" }],
  ["https://music.apple.com.evil.example/us/song/x/1446013598", { error: "invalid_link" }],
  ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", { error: "invalid_link" }],
  ["https://www.deezer.com/track/3135556", { error: "invalid_link" }],
  ["javascript:alert(1)", { error: "invalid_link" }],
  ["never gonna give you up", { error: "invalid_link" }],
  ["", { error: "invalid_link" }],
  ["   ", { error: "invalid_link" }],
  [`https://open.spotify.com/track/${ID}?${"x".repeat(2100)}`, { error: "invalid_link" }],
];

test("parses every supported shared-link shape and rejects the rest", () => {
  for (const [input, expected] of LINKS) {
    assert.deepEqual(parseSharedLink(input), expected, input.slice(0, 90));
  }
  for (const input of [null, undefined, 42, {}, [`spotify:track:${ID}`]]) {
    assert.deepEqual(parseSharedLink(input), { error: "invalid_link" });
  }
});

test("normalises titles, artists and ISRCs", () => {
  assert.equal(coreTitle("I Don't Want To Be A Hero (2008 Digital Remaster)"), "i don t want to be a hero");
  assert.equal(coreTitle("Song - Radio Edit"), "song");
  assert.equal(coreTitle("Canción (feat. Otro) [Live]"), "cancion");
  assert.equal(coreTitle("夜に駆ける"), "夜に駆ける");
  assert.equal(coreTitle("(Intro)"), "intro");
  assert.equal(searchTitle("Alors on danse (Radio Edit)"), "Alors on danse");
  assert.equal(searchTitle("(Intro)"), "(Intro)");
  assert.equal(primaryArtist("Luis Fonsi, Daddy Yankee"), "luis fonsi");
  assert.equal(primaryArtist("Beyoncé feat. JAY-Z"), "beyonce");
  assert.equal(primaryArtist("Lil Nas X"), "lil nas x");
  assert.equal(normalizeIsrc("us-um7-16-07007"), "USUM71607007");
  assert.equal(normalizeIsrc("USUM7160700"), null);
  assert.equal(normalizeIsrc(123), null);
});

test("picks the catalogue row that is the same song", () => {
  const rows = [
    { title: "Gekreuzigt 2006", artist: "OOMPH!", durationMs: 250000, url: "long" },
    { title: "Gekreuzigt 2006", artist: "OOMPH!", durationMs: 216587, url: "right" },
    { title: "Gekreuzigt 2006 (Live)", artist: "OOMPH!", durationMs: 217000, url: "live" },
    { title: "The Power of Love", artist: "OOMPH!", durationMs: 217000, url: "other-song" },
    { title: "Gekreuzigt 2006", artist: "Tribute Band", durationMs: 217000, url: "other-artist" },
  ];
  const wanted = { title: "Gekreuzigt 2006", artist: "OOMPH!", durationMs: 217000 };
  assert.equal(pickBestMatch(rows, wanted).url, "right");
  assert.equal(pickBestMatch(rows.slice(2), wanted).url, "live");
  assert.equal(pickBestMatch(rows.slice(3), wanted), null);
  assert.equal(pickBestMatch(rows, { ...wanted, durationMs: null }).url, "long");
  assert.equal(
    pickBestMatch([{ title: "Despacito", artist: "Luis Fonsi & Daddy Yankee" }], { title: "Despacito", artist: "Daddy Yankee" })
      .title,
    "Despacito",
  );
  assert.equal(pickBestMatch([{ title: "Hello", artist: "Adele" }], { title: "", artist: "Adele" }), null);
  assert.equal(pickBestMatch(null, wanted), null);
  assert.equal(pickBestMatch([null, {}, { title: 1, artist: 2 }], wanted), null);
});

test("builds search links and cleans Apple links", () => {
  assert.equal(
    spotifySearchUrl("AC/DC", "T.N.T. & more?"),
    "https://open.spotify.com/search/AC%2FDC%20T.N.T.%20%26%20more%3F",
  );
  assert.equal(
    appleSearchUrl("jp", "YOASOBI", "アイドル"),
    "https://music.apple.com/jp/search?term=YOASOBI%20%E3%82%A2%E3%82%A4%E3%83%89%E3%83%AB",
  );
  assert.equal(
    cleanAppleUrl("https://music.apple.com/de/album/99-luftballons/1446013584?i=1446013598&uo=4"),
    "https://music.apple.com/de/album/99-luftballons/1446013584?i=1446013598",
  );
  assert.equal(cleanAppleUrl("https://evil.example/de/album/x/1?i=2"), null);
  assert.equal(cleanAppleUrl("http://music.apple.com/de/album/x/1?i=2"), null);
  assert.equal(cleanAppleUrl(undefined), null);
  assert.equal(
    appleArtwork("https://is1-ssl.mzstatic.com/image/thumb/Music/a/b.jpg/100x100bb.jpg"),
    "https://is1-ssl.mzstatic.com/image/thumb/Music/a/b.jpg/300x300bb.jpg",
  );
  assert.equal(appleArtwork("http://is1-ssl.mzstatic.com/a/100x100bb.jpg"), null);
});

test("builds the ways a song is put to the Spotify id lookup, best first, without repeats", () => {
  const q = (artist_name, track_name, release_name = "") => ({ artist_name, release_name, track_name });
  assert.deepEqual(
    spotifyMetadataQueries({
      title: "Hey DJ (Remix)",
      artist: "CNCO",
      album: "Hey DJ",
      contributors: ["CNCO", "Meghan Trainor", "Sean Paul"],
    }),
    [
      q("CNCO, Meghan Trainor, Sean Paul", "Hey DJ (Remix)", "Hey DJ"),
      q("CNCO", "Hey DJ (Remix)", "Hey DJ"),
      q("CNCO, Meghan Trainor, Sean Paul", "Hey DJ"),
      q("CNCO", "Hey DJ"),
    ],
  );
  // One artist and a plain title is one question.
  assert.deepEqual(spotifyMetadataQueries({ title: "Lethal", artist: "Vera Blue", album: null, contributors: ["Vera Blue"] }), [
    q("Vera Blue", "Lethal"),
  ]);
  // No Deezer details: the result's own names, then the plain title, then the first artist with it.
  assert.deepEqual(spotifyMetadataQueries({ title: "Loca (feat. Cali y El Dandee)", artist: "Maite Perroni & Friends" }), [
    q("Maite Perroni & Friends", "Loca (feat. Cali y El Dandee)"),
    q("Maite Perroni & Friends", "Loca"),
    q("Maite Perroni", "Loca"),
  ]);
  assert.deepEqual(spotifyMetadataQueries({ title: "", artist: "x" }), []);
  assert.deepEqual(spotifyMetadataQueries({ title: "x", artist: " " }), []);
});

test("names the kind of version in a title", () => {
  assert.equal(versionWords("American Spirit (Live)"), "live");
  assert.equal(versionWords("American Spirit"), "");
  assert.equal(versionWords("Chora Boy (Ao Vivo)"), versionWords("Chora Boy - Ao Vivo"));
  assert.equal(versionWords("Neele Neele Ambar Par (The Unwind Mix)"), versionWords("Neele Neele Ambar Par - The Unwind Mix"));
  assert.equal(versionWords("Song (Live Acoustic Remix)"), "acoustic live remix");
  assert.equal(versionWords("Kleine geile Nonne (Remastered)"), "", "a remaster is the same recording");
  assert.equal(versionWords("Alive"), "", "whole words only");
});
