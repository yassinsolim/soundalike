import assert from "node:assert/strict";
import test from "node:test";

import {
  appleArtwork,
  appleSearchUrl,
  cleanAppleUrl,
  coreTitle,
  looseTitle,
  normalizeIsrc,
  parseSharedLink,
  pickBestMatch,
  pickSeedRecording,
  primaryArtist,
  sameLength,
  scriptLanguages,
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

test("names the languages of a title's script", () => {
  assert.equal(scriptLanguages("晴天 Jay Chou"), "zh-TW,ja;q=0.9");
  assert.equal(scriptLanguages("アイドル"), "ja");
  assert.equal(scriptLanguages("夜に駆ける"), "ja,zh-TW;q=0.9");
  assert.equal(scriptLanguages("Группа крови Kino"), "ru");
  assert.equal(scriptLanguages("البخت"), "ar");
  assert.equal(scriptLanguages("강남스타일"), "ko");
  assert.equal(scriptLanguages("Despacito — Luis Fonsi"), null);
  assert.equal(scriptLanguages("Zitti e buoni Måneskin"), null);
  assert.equal(scriptLanguages(null), null);
});

test("compares titles across spellings of the same letters", () => {
  assert.equal(looseTitle("تملي معاك"), looseTitle("تملى معاك"));
  assert.equal(looseTitle("أواخر الشتا"), looseTitle("اواخر الشتا"));
  assert.equal(looseTitle("Ёлка"), looseTitle("елка"));
  assert.equal(looseTitle("晴天 (Remastered)"), "晴天");
  assert.notEqual(looseTitle("晴天"), looseTitle("Sunny Day"));
});

test("a Deezer length is the seed's within two seconds", () => {
  assert.equal(sameLength(269746, 270), true);
  assert.equal(sameLength(269746, 272), true);
  assert.equal(sameLength(269746, 268), true);
  assert.equal(sameLength(269746, 273), false);
  assert.equal(sameLength(269746, 267), false);
  assert.equal(sameLength(null, 270), false);
  assert.equal(sameLength(269746, undefined), false);
});

test("picks the shared song's Deezer recording only on evidence", () => {
  const seed = { title: "晴天", durationMs: 269746 };
  const track = (id, titles, durationS, extra = {}) => ({ id, titles, durationS, album: "Yeh, Hwei-Mei", isrc: null, rank: 0, ...extra });
  const sunny = track(1, ["Sunny Day", "晴天"], 270, { isrc: "TWK970300503" });
  const waiting = track(2, ["Waiting For You"], 270, { album: "Greatest Works Of Art", isrc: "TWK971801101" });
  const simple = track(3, ["Simple Love", "シンプルな愛"], 271, { album: "Fantasy", isrc: "TWK970100203" });
  assert.equal(pickSeedRecording(seed, [waiting, simple, sunny]).id, 1);

  // The length alone never decides, not even for the only track that has it.
  assert.equal(pickSeedRecording(seed, [waiting]), null);
  assert.equal(pickSeedRecording(seed, [waiting, simple]), null);
  assert.equal(pickSeedRecording(seed, [track(1, ["Sunny Day"], 270)]), null);
  // The title alone does not either.
  assert.equal(pickSeedRecording(seed, [track(1, ["晴天"], 300)]), null);
  assert.equal(pickSeedRecording({ title: "晴天", durationMs: null }, [sunny]), null);
  // A live recording of the same length is another recording.
  assert.equal(pickSeedRecording(seed, [track(1, ["晴天 (Live)"], 270)]), null);
  assert.equal(pickSeedRecording({ title: "晴天 (Live)", durationMs: 269746 }, [track(1, ["晴天 - Live"], 270)]).id, 1);

  // Several releases of the song: the recording most of them are wins, then the most played.
  const single = track(4, ["晴天"], 270, { isrc: "TWK970300503", rank: 900 });
  const rerecorded = track(5, ["晴天"], 269, { isrc: "TWK972400001", rank: 5000 });
  assert.equal(pickSeedRecording(seed, [rerecorded, sunny, single]).id, 4);
  assert.equal(pickSeedRecording(seed, [rerecorded, sunny]).id, 5);

  // Apple Music names the album: with the length, one recording on it is enough.
  const apple = { title: "晴天", durationMs: 269747, album: "Yeh, Hwei-Mei" };
  assert.equal(pickSeedRecording(apple, [track(1, ["Sunny Day"], 270), waiting]).id, 1);
  // Two recordings of that length on the album: not known which.
  assert.equal(pickSeedRecording(apple, [track(1, ["Sunny Day"], 270), track(6, ["Class Is Over"], 271)]), null);
  // The same recording listed twice on it is one recording.
  assert.equal(
    pickSeedRecording(apple, [
      track(1, ["Sunny Day"], 270, { isrc: "TWK970300503", rank: 1 }),
      track(7, ["Sunny Day"], 270, { isrc: "TWK970300503", rank: 2 }),
    ]).id,
    7,
  );
  assert.equal(pickSeedRecording({ ...apple, album: "葉惠美" }, [track(1, ["Sunny Day"], 270)]), null);
  assert.equal(pickSeedRecording(seed, []), null);
  assert.equal(pickSeedRecording(seed, null), null);
});
