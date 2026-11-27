import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../find/index.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const config = JSON.parse(readFileSync(new URL("../site.json", import.meta.url), "utf8"));

// Just enough DOM for the page's own script: elements by id, children,
// attributes, classes and events.
class FakeElement {
  constructor(tag, id = "") {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.className = "";
    this.textContent = "";
    this.hidden = false;
    this.value = "";
    this.style = {};
    this.classList = {
      add: (name) => { this.className = [...new Set([...this.className.split(" ").filter(Boolean), name])].join(" "); },
      remove: (name) => { this.className = this.className.split(" ").filter((part) => part && part !== name).join(" "); },
      contains: (name) => this.className.split(" ").includes(name),
    };
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  dispatch(type, event = {}) {
    for (const listener of this.listeners[type] || []) listener({ preventDefault() {}, ...event });
  }
  focus() {}
  blur() {}
  // Every element is 300 px wide at the left edge, so 10 px is one second of a 30-second preview.
  getBoundingClientRect() { return { left: 0, width: 300 }; }
  setPointerCapture() {}
}

const IDS = ["form", "link", "status", "list", "seed", "seed-art", "seed-title", "seed-artist", "bar", "count", "p-spotify", "p-apple",
  "part", "part-open", "part-chip", "part-chip-text", "part-clear", "part-panel", "part-track", "part-span", "part-head",
  "part-from", "part-to", "part-label", "part-10", "part-15", "part-20", "part-play", "part-apply"];
// What the page's markup hides until its script shows it.
const HIDDEN = ["seed", "bar", "part", "part-chip", "part-panel", "part-head"];

const row = (n, extra = {}) => ({
  title: `Song ${n}`,
  artist: `Artist ${n}`,
  match: 90 - n,
  deezer_id: 1000 + n,
  isrc: null,
  artwork: `https://cdn-images.dzcdn.net/images/cover/${n}/250x250.jpg`,
  preview_url: `https://cdnt-preview.dzcdn.net/${n}.mp3`,
  spotify_url: `https://open.spotify.com/search/Artist%20${n}%20Song%20${n}`,
  spotify_exact: false,
  apple_music_url: `https://music.apple.com/us/search?term=Artist%20${n}%20Song%20${n}`,
  apple_exact: false,
  ...extra,
});

const answer = (platform, results, links = {}) => ({
  ok: true,
  query: {
    title: "Seed <b>Song</b>",
    artist: "Seed Artist",
    isrc: null,
    artwork: "https://image-cdn-ak.spotifycdn.com/image/abc",
    platform,
    source_url: null,
    language: "en",
    storefront: "us",
    deezer_id: 1,
  },
  results,
  engine: { list: "main", k: results.length },
  links: { spotify_exact: 0, apple_exact: 0, spotify_lookup: false, apple_eager_rows: 5, ...links },
});

const REFUSED = Symbol("refused");
const refused = (body) => ({ ...body, [REFUSED]: true });

/** Loads the page at `search` with `respond(url)` standing in for the server. */
async function open(search, respond, { userAgent = "Mozilla/5.0 (Linux; Android 14)", session = {} } = {}) {
  const elements = Object.fromEntries(IDS.map((id) => [id, new FakeElement("div", id)]));
  for (const id of HIDDEN) elements[id].hidden = true;
  const requests = [];
  const addresses = [];
  const audio = new FakeElement("audio");
  Object.assign(audio, {
    played: [],
    currentTime: 0,
    paused: true,
    pause() { this.paused = true; },
    async play() { this.paused = false; this.played.push(this.src); },
  });
  const storage = (data) => ({
    getItem: (key) => data[key] ?? null,
    setItem: (key, value) => { data[key] = String(value); },
  });
  const windowObject = new FakeElement("window");
  Object.assign(windowObject, { localStorage: storage({}), sessionStorage: storage(session) });
  const context = {
    window: windowObject,
    document: { getElementById: (id) => elements[id], createElement: (tag) => new FakeElement(tag), title: "" },
    location: { search },
    history: {
      pushState: (_, __, url) => addresses.push(["push", url]),
      replaceState: (_, __, url) => addresses.push(["replace", url]),
    },
    navigator: { userAgent },
    Audio: function Audio() { return audio; },
    URLSearchParams,
    Date,
    JSON,
    fetch: async (url) => {
      requests.push(url);
      const body = await respond(url);
      // `refused(body)` stands for a 400 answer.
      return { status: body?.[REFUSED] ? 400 : 200, json: async () => body };
    },
  };
  vm.runInNewContext(script, context);
  const settle = async () => { for (let turn = 0; turn < 40; turn += 1) await new Promise((resolve) => setImmediate(resolve)); };
  await settle();
  const rows = () =>
    elements.list.children.filter((item) => item.className.split(" ").includes("row")).map((item) => ({
      item,
      button: item.children[0],
      anchor: item.children[1],
    }));
  return { elements, requests, addresses, rows, settle, audio, context };
}

test("the page is self-contained and routed with a strict policy", () => {
  assert.match(html, /<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"\/>/);
  assert.equal((html.match(/<script/g) || []).length, 1, "one inline script, no framework");
  assert.doesNotMatch(html, /<script[^>]+src=|<link[^>]+rel="stylesheet"|@import|<iframe/i);
  assert.doesNotMatch(html, /innerHTML|insertAdjacentHTML|document\.write|eval\(/);
  assert.doesNotMatch(script, /target\s*=\s*"_blank"|window\.open/, "taps navigate directly so app links open the app");
  assert.ok(Buffer.byteLength(html) < 30000, "small enough for a fast first paint");

  const rewrites = Object.fromEntries(config.rewrites.map((rewrite) => [rewrite.source, rewrite.destination]));
  assert.equal(rewrites["/find"], "/find/index.html");
  assert.equal(rewrites["/find/"], "/find/index.html");
  for (const source of ["/find", "/find/(.*)"]) {
    const headers = config.headers.find((entry) => entry.source === source)?.headers ?? [];
    const policy = headers.find((header) => header.key === "Content-Security-Policy")?.value ?? "";
    assert.match(policy, /default-src 'none'/);
    assert.match(policy, /connect-src 'self'/);
    assert.match(policy, /img-src [^;]*https:\/\/\*\.dzcdn\.net[^;]*https:\/\/\*\.spotifycdn\.com[^;]*https:\/\/\*\.mzstatic\.com/);
    assert.match(policy, /media-src [^;]*https:\/\/\*\.dzcdn\.net/);
    assert.match(policy, /frame-ancestors 'none'/);
  }
});

test("without a song the page waits for a link and asks nothing", async () => {
  const page = await open("", () => assert.fail("no request expected"));
  assert.equal(page.requests.length, 0);
  assert.equal(page.elements.seed.hidden, true);
  assert.equal(page.elements.bar.hidden, true);

  page.elements.link.value = "  ";
  page.elements.form.dispatch("submit");
  await page.settle();
  assert.equal(page.requests.length, 0);
});

test("a shared Spotify link shows the seed and rows that open in Spotify", async () => {
  const link = "https://open.spotify.com/track/5awDvzxWfd53SSrsRZ8pXO?si=abc";
  const page = await open(`?url=${encodeURIComponent(link)}`, () =>
    answer("spotify", [row(1, { spotify_exact: true, spotify_url: "https://open.spotify.com/track/1a2B3c4D5e6F7g8H9i0J1k" }), row(2)]),
  );
  assert.deepEqual(page.requests, [`/api/mobile/similar?url=${encodeURIComponent(link).replace(/%20/g, "+")}`]);
  assert.equal(page.elements.link.value, link);
  assert.equal(page.elements.seed.hidden, false);
  assert.equal(page.elements["seed-title"].textContent, "Seed <b>Song</b>", "text only, never markup");
  assert.equal(page.elements["seed-artist"].textContent, "Seed Artist");
  assert.equal(page.elements["seed-art"].src, "https://image-cdn-ak.spotifycdn.com/image/abc");
  assert.equal(page.elements.count.textContent, "2 songs");
  assert.equal(page.elements.status.hidden, true);
  assert.equal(page.elements["p-spotify"].attributes["aria-pressed"], "true");
  assert.equal(page.elements["p-apple"].attributes["aria-pressed"], "false");

  const rows = page.rows();
  assert.equal(rows.length, 2);
  assert.equal(rows[0].anchor.href, "https://open.spotify.com/track/1a2B3c4D5e6F7g8H9i0J1k");
  assert.equal(rows[1].anchor.href, "https://open.spotify.com/search/Artist%202%20Song%202");
  assert.equal(rows[0].anchor.attributes["aria-label"], "Song 1 by Artist 1, open in Spotify");
  assert.equal(rows[0].anchor.rel, "noopener noreferrer");
  assert.equal(rows[0].anchor.children[1].textContent, "89%");
  assert.equal(rows[0].button.children[0].src, "https://cdn-images.dzcdn.net/images/cover/1/250x250.jpg");
  // Without Spotify credentials on the server no row is looked up again.
  assert.equal(page.requests.length, 1);
});

test("Spotify rows are asked about again only when the server has Spotify credentials", async () => {
  const respond = (source) => (url) => {
    if (url.startsWith("/api/mobile/similar")) {
      return answer(
        "spotify",
        [row(1, { spotify_exact: true, spotify_kind: "track", spotify_id: "1a2B3c4D5e6F7g8H9i0J1k", spotify_url: "https://open.spotify.com/track/1a2B3c4D5e6F7g8H9i0J1k" }), row(2)],
        { spotify_lookup: true, spotify_source: source },
      );
    }
    return { ok: true, exact: true, kind: "track", url: "https://open.spotify.com/track/9z8Y7x6W5v4U3t2S1r0Q9p", retry_after: null };
  };
  // Without credentials the answer already holds every track the server can find.
  const metadata = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", respond("metadata"));
  assert.equal(metadata.requests.length, 1);
  // A track link is a plain same-tab link, so the phone hands it to Spotify, which plays it.
  const [track, search] = metadata.rows();
  assert.equal(track.anchor.href, "https://open.spotify.com/track/1a2B3c4D5e6F7g8H9i0J1k");
  assert.equal(track.anchor.target, undefined);
  assert.equal(search.anchor.href, "https://open.spotify.com/search/Artist%202%20Song%202");

  const credentials = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", respond("credentials"));
  assert.deepEqual(credentials.requests.slice(1), [
    "/api/mobile/link?deezer_id=1002&platform=spotify&title=Song+2&artist=Artist+2&storefront=us",
  ]);
  assert.equal(credentials.rows()[1].anchor.href, "https://open.spotify.com/track/9z8Y7x6W5v4U3t2S1r0Q9p");
});

test("the toggle switches every row to Apple Music and looks up exact links", async () => {
  const page = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO&k=3&lang=all", (url) => {
    if (url.startsWith("/api/mobile/similar")) return answer("spotify", [row(1), row(2), row(3)]);
    const params = new URL(url, "https://soundalike.yassin.app").searchParams;
    const id = params.get("deezer_id");
    if (id === "1001") return { ok: true, exact: true, url: "https://music.apple.com/us/album/song-1/1?i=2", retry_after: null };
    if (id === "1002") return { ok: true, exact: false, url: "https://music.apple.com/us/search?term=x", retry_after: 3 };
    return assert.fail("no lookups after the server asks to wait");
  });
  assert.equal(page.requests[0], "/api/mobile/similar?spotify_id=5awDvzxWfd53SSrsRZ8pXO&k=3&lang=all");

  page.elements["p-apple"].dispatch("click");
  await page.settle();
  assert.equal(page.elements["p-apple"].attributes["aria-pressed"], "true");
  const rows = page.rows();
  assert.equal(rows[0].anchor.href, "https://music.apple.com/us/album/song-1/1?i=2");
  assert.equal(rows[1].anchor.href, "https://music.apple.com/us/search?term=Artist%202%20Song%202");
  assert.equal(rows[2].anchor.href, "https://music.apple.com/us/search?term=Artist%203%20Song%203");
  assert.equal(rows[2].anchor.attributes["aria-label"], "Song 3 by Artist 3, open in Apple Music");
  assert.deepEqual(page.requests.slice(1), [
    "/api/mobile/link?deezer_id=1001&platform=apple&title=Song+1&artist=Artist+1&storefront=us",
    "/api/mobile/link?deezer_id=1002&platform=apple&title=Song+2&artist=Artist+2&storefront=us",
  ]);
  assert.equal(page.context.window.localStorage.getItem("soundalike:find:platform:v1"), "apple");

  page.elements["p-spotify"].dispatch("click");
  await page.settle();
  assert.equal(page.rows()[0].anchor.href, "https://open.spotify.com/search/Artist%201%20Song%201");
  assert.equal(page.requests.length, 3);
});

test("share text, a seed without a platform, and unsafe addresses", async () => {
  const text = "Envolver by Anitta https://music.apple.com/br/album/envolver/1593185587?i=1593185593";
  const page = await open(
    `?text=${encodeURIComponent(text)}&title=ignored`,
    (url) => (url.startsWith("/api/mobile/similar")
      ? answer("both", [row(1, { apple_music_url: "javascript:alert(1)", artwork: "http://insecure.example/a.jpg", apple_exact: true })])
      : assert.fail("unexpected")),
    { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)" },
  );
  assert.equal(new URLSearchParams(page.requests[0].split("?")[1]).get("url"), text);
  assert.equal(new URLSearchParams(page.requests[0].split("?")[1]).has("title"), false);
  // On an iPhone a seed that came from neither app opens in Apple Music.
  assert.equal(page.elements["p-apple"].attributes["aria-pressed"], "true");
  const [first] = page.rows();
  assert.equal(first.anchor.href, undefined, "an address that is not https is never used");
  assert.equal(first.button.children.length, 1, "no image for an insecure address");
});

test("errors from the API and from the network are shown", async () => {
  const refused = await open("?url=https%3A%2F%2Fexample.com", () => ({
    ok: false,
    error: { code: "invalid_link", message: "This is not a Spotify or Apple Music song link." },
  }));
  assert.equal(refused.elements.status.textContent, "This is not a Spotify or Apple Music song link.");
  assert.equal(refused.elements.status.className, "hint err");
  assert.equal(refused.rows().length, 0);
  assert.equal(refused.elements.list.children.length, 0);

  const offline = await open("?isrc=DEC760600298", () => { throw new TypeError("offline"); });
  assert.match(offline.elements.status.textContent, /could not be reached/);

  const empty = await open("?title=a&artist=b", () => answer("both", []));
  assert.equal(empty.elements.status.textContent, "No soundalikes were found for this song.");
  assert.equal(empty.elements.bar.hidden, true);
});

test("previews play one at a time and recover from an expired address", async () => {
  const page = await open("?title=a&artist=b&platform=spotify", (url) => {
    if (url.startsWith("/api/mobile/similar")) return answer("spotify", [row(1), row(2, { preview_url: null })]);
    assert.equal(url, "/api/preview?id=1002");
    return { ok: true, preview: "https://cdnt-preview.dzcdn.net/fresh.mp3", cover: "" };
  });
  const [first, second] = page.rows();
  first.button.dispatch("click");
  await page.settle();
  assert.deepEqual(page.audio.played, ["https://cdnt-preview.dzcdn.net/1.mp3"]);
  assert.equal(first.item.classList.contains("playing"), true);

  second.button.dispatch("click");
  await page.settle();
  assert.equal(first.item.classList.contains("playing"), false);
  assert.equal(second.item.classList.contains("playing"), true);
  assert.deepEqual(page.audio.played.at(-1), "https://cdnt-preview.dzcdn.net/fresh.mp3");

  second.button.dispatch("click");
  assert.equal(second.item.classList.contains("playing"), false);
});

test("coming back from the music app reuses the answer", async () => {
  const session = {};
  const search = "?spotify_id=5awDvzxWfd53SSrsRZ8pXO";
  const first = await open(search, () => answer("spotify", [row(1)]), { session });
  assert.equal(first.requests.length, 1);
  const again = await open(search, () => assert.fail("served from the saved answer"), { session });
  assert.equal(again.requests.length, 0);
  assert.equal(again.rows().length, 1);
  const other = await open("?spotify_id=2Fxmhks0bxGSBdJ92vM42m", () => answer("spotify", [row(1), row(2)]), { session });
  assert.equal(other.requests.length, 1);
});

test("association files let /find links open the iPhone and Android apps", () => {
  const destination = (source) => config.rewrites.find((rewrite) => rewrite.source === source)?.destination;
  const header = (source, key) =>
    config.headers.find((entry) => entry.source === source)?.headers.find((item) => item.key === key)?.value;
  const load = (source) => {
    const target = destination(source);
    assert.match(target ?? "", /^\/well-known\/[a-z-]+\.json$/, "a plain .json file, served as application/json");
    assert.equal(header(source, "Content-Type"), "application/json");
    return JSON.parse(readFileSync(new URL(`..${target}`, import.meta.url), "utf8"));
  };

  assert.deepEqual(load("/.well-known/apple-app-site-association"), {
    applinks: {
      details: [
        {
          appIDs: ["FJH4UWVC4K.app.yassin.soundalike"],
          components: [{ "/": "/find*", comment: "Soundalikes for a shared song link" }],
        },
      ],
    },
  });

  const links = load("/.well-known/assetlinks.json");
  assert.equal(links.length, 1);
  assert.deepEqual(links[0].relation, ["delegate_permission/common.handle_all_urls"]);
  assert.equal(links[0].target.namespace, "android_app");
  assert.equal(links[0].target.package_name, "app.yassin.soundalike");
  assert.equal(links[0].target.sha256_cert_fingerprints.length, 1);
  assert.match(links[0].target.sha256_cert_fingerprints[0], /^([0-9A-F]{2}:){31}[0-9A-F]{2}$/);
});

// ---- Match a part ------------------------------------------------------------

const PARTS = { min_s: 10, max_s: 30, preview_s: 30, song_offset_s: 48 };
/** An answer from a server that can match a part; `part` when the lists are for one. */
const partAnswer = (part = null, parts = PARTS) => {
  const body = answer("spotify", [row(1), row(2)]);
  Object.assign(body.query, { parts, part: part ? { start: part[0], end: part[1] } : null });
  return body;
};
const similarParams = (url) => Object.fromEntries(new URLSearchParams(url.split("?")[1]));
const press = (element) => element.dispatch("click");
/** A finger (or mouse) on the track at `second` of the preview: the fake track is 10 px a second. */
const pointer = (second) => ({ clientX: second * 10, pointerId: 1 });
const span = (page) => [page.elements["part-span"].style.left, page.elements["part-span"].style.width];

test("the page's markup carries the part control, hidden, and its policy is unchanged", () => {
  for (const id of ["part", "part-chip", "part-panel", "part-head"]) {
    assert.match(html, new RegExp(`<[a-z]+ [^>]*id="${id}"[^>]* hidden>`), id);
  }
  for (const id of IDS) assert.ok(html.includes(`id="${id}"`), id);
  assert.match(html, /\.trk\{[^}]*touch-action:none/, "dragging the span must not scroll the page");
  assert.doesNotMatch(html, /<canvas|AudioContext|decodeAudioData/, "a plain bar: no waveform, nothing to fetch");
  const policy = config.headers.find((entry) => entry.source === "/find").headers.find((header) => header.key === "Content-Security-Policy").value;
  assert.match(policy, /script-src 'unsafe-inline';/);
  assert.match(policy, /connect-src 'self'$/);
  assert.doesNotMatch(policy, /blob:|unsafe-eval|script-src[^;]*https:/);
});

test("Match a part shows only when the server offers parts", async () => {
  const without = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", () => answer("spotify", [row(1)]));
  assert.equal(without.elements.part.hidden, true);
  press(without.elements["part-open"]);
  assert.equal(without.elements["part-panel"].hidden, true, "nothing opens without parts");

  const offered = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", () => partAnswer());
  assert.equal(offered.elements.part.hidden, false);
  assert.equal(offered.elements["part-chip"].hidden, true);
  assert.equal(offered.elements["part-panel"].hidden, true);
  assert.equal(offered.requests.length, 1, "the control costs no request of its own");

  // Limits the page cannot serve (a longer shortest part than its longest) hide it too.
  const odd = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", () => partAnswer(null, { ...PARTS, min_s: 25 }));
  assert.equal(odd.elements.part.hidden, true);
});

test("the span is dragged on a 5-second grid, resized by the length buttons, and applied", async () => {
  const link = "https://open.spotify.com/track/5awDvzxWfd53SSrsRZ8pXO";
  const page = await open(`?url=${encodeURIComponent(link)}&lang=all`, (url) =>
    partAnswer(similarParams(url).part ? similarParams(url).part.split("-").map(Number) : null),
  );
  const { elements } = page;
  press(elements["part-open"]);
  assert.equal(elements["part-panel"].hidden, false);
  assert.equal(elements["part-open"].attributes["aria-expanded"], "true");
  // Starts at 10 to 20 seconds of the preview, named in song time (the preview starts at 0:48).
  assert.deepEqual(span(page), [`${100 / 3}%`, `${100 / 3}%`]);
  assert.equal(elements["part-label"].textContent, "0:58–1:08 of the song · 10 seconds");
  assert.equal(elements["part-from"].textContent, "0:48");
  assert.equal(elements["part-to"].textContent, "1:18");
  assert.deepEqual(["10", "15", "20"].map((n) => elements[`part-${n}`].attributes["aria-pressed"]), ["true", "false", "false"]);
  assert.equal(elements["part-span"].attributes["aria-valuetext"], "0:58–1:08 of the song");

  // A finger inside the span drags it from where it was put down; it snaps to 5 seconds.
  const track = elements["part-track"];
  let prevented = false;
  track.dispatch("pointerdown", { ...pointer(13), preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  track.dispatch("pointermove", pointer(15));
  assert.deepEqual(span(page), [`${100 / 3}%`, `${100 / 3}%`], "2 seconds is not yet a step");
  track.dispatch("pointermove", pointer(19));
  assert.deepEqual(span(page), ["50%", `${100 / 3}%`]);
  track.dispatch("pointermove", pointer(40));
  assert.deepEqual(span(page), [`${200 / 3}%`, `${100 / 3}%`], "it stops at the end of the preview");
  track.dispatch("pointermove", pointer(-20));
  assert.deepEqual(span(page), ["0%", `${100 / 3}%`], "and at its start");
  track.dispatch("pointerup", pointer(-20));
  track.dispatch("pointermove", pointer(25));
  assert.deepEqual(span(page), ["0%", `${100 / 3}%`], "a lifted finger moves nothing");

  // A tap outside the span brings its middle there.
  track.dispatch("pointerdown", pointer(21));
  track.dispatch("pointercancel", pointer(21));
  assert.deepEqual(span(page), ["50%", `${100 / 3}%`]);

  // The length buttons keep the start and stay inside the preview.
  press(elements["part-20"]);
  assert.deepEqual(span(page), [`${100 / 3}%`, `${200 / 3}%`], "20 seconds from 0:15 would run past the end");
  assert.equal(elements["part-label"].textContent, "0:58–1:18 of the song · 20 seconds");
  assert.equal(elements["part-20"].attributes["aria-pressed"], "true");
  press(elements["part-15"]);
  // Arrow keys move it one step.
  let keyPrevented = false;
  elements["part-span"].dispatch("keydown", { key: "ArrowLeft", preventDefault() { keyPrevented = true; } });
  assert.equal(keyPrevented, true);
  assert.equal(elements["part-label"].textContent, "0:53–1:08 of the song · 15 seconds");
  elements["part-span"].dispatch("keydown", { key: "Enter" });
  assert.equal(elements["part-label"].textContent, "0:53–1:08 of the song · 15 seconds");
  assert.equal(page.requests.length, 1, "nothing is asked until Apply");

  press(elements["part-apply"]);
  await page.settle();
  assert.deepEqual(similarParams(page.requests[1]), { url: link, lang: "all", part: "5-20" });
  // The part is in the page's address, so a shared link keeps it.
  assert.deepEqual(page.addresses, [["push", `/find?url=${encodeURIComponent(link)}&lang=all&part=5-20`]]);
  assert.equal(elements["part-panel"].hidden, true);
  assert.equal(elements["part-chip"].hidden, false);
  assert.equal(elements["part-chip-text"].textContent, "Matching 0:53–1:08 of the song");

  // Reopened, the picker starts from the part on screen.
  press(elements["part-open"]);
  assert.equal(elements["part-label"].textContent, "0:53–1:08 of the song · 15 seconds");
  press(elements["part-open"]);
  assert.equal(elements["part-panel"].hidden, true, "the button also closes it");

  // "whole preview" leads back.
  press(elements["part-clear"]);
  await page.settle();
  assert.deepEqual(similarParams(page.requests[2]), { url: link, lang: "all" });
  assert.deepEqual(page.addresses.at(-1), ["push", `/find?url=${encodeURIComponent(link)}&lang=all`]);
  assert.equal(elements["part-chip"].hidden, true);
});

test("a link with a part opens on that part; a bad part is ignored", async () => {
  const page = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO&part=15-30", () => partAnswer([15, 30], { ...PARTS, song_offset_s: null }));
  assert.deepEqual(similarParams(page.requests[0]), { spotify_id: "5awDvzxWfd53SSrsRZ8pXO", part: "15-30" });
  // Without the preview's place in the song, times are the preview's own.
  assert.equal(page.elements["part-chip-text"].textContent, "Matching 0:15–0:30 of the preview");
  assert.deepEqual(page.addresses, []);
  press(page.elements["part-open"]);
  assert.equal(page.elements["part-label"].textContent, "0:15–0:30 of the preview · 15 seconds");
  assert.equal(page.elements["part-to"].textContent, "0:30");

  for (const bad of ["5-12", "20-10", "0-31", "abc", "10-20-30", "-5-20", ""]) {
    const ignored = await open(`?spotify_id=5awDvzxWfd53SSrsRZ8pXO&part=${bad}`, () => partAnswer());
    assert.deepEqual(similarParams(ignored.requests[0]), { spotify_id: "5awDvzxWfd53SSrsRZ8pXO" }, bad);
    assert.deepEqual(ignored.addresses, [["replace", "/find?spotify_id=5awDvzxWfd53SSrsRZ8pXO"]], bad);
    assert.equal(ignored.elements["part-chip"].hidden, true);
  }

  // A newly pasted link starts from its whole preview.
  page.elements.link.value = "https://open.spotify.com/track/2Fxmhks0bxGSBdJ92vM42m";
  page.elements.form.dispatch("submit");
  await page.settle();
  assert.equal(similarParams(page.requests.at(-1)).part, undefined);
});

test("a server without parts answers for the whole preview and the part is dropped", async () => {
  // An older engine ignores the parameter: no query.part in the answer.
  const session = {};
  const ignored = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO&part=10-20&lang=all", () => answer("spotify", [row(1)]), { session });
  assert.equal(ignored.elements.part.hidden, true);
  assert.equal(ignored.elements["part-chip"].hidden, true);
  assert.deepEqual(ignored.addresses, [["replace", "/find?spotify_id=5awDvzxWfd53SSrsRZ8pXO&lang=all"]]);
  assert.equal(ignored.rows().length, 1);
  assert.equal(Object.keys(session).length, 0, "not kept under the part's address");

  // A server that refuses the part is asked once more without it.
  const page = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO&part=10-20", (url) =>
    similarParams(url).part
      ? refused({ ok: false, error: { code: "invalid_parameter", message: "part is not supported." } })
      : partAnswer(),
  );
  assert.deepEqual(page.requests.map((url) => similarParams(url).part), ["10-20", undefined]);
  assert.equal(page.rows().length, 2);
  assert.equal(page.elements.status.hidden, true);
  assert.equal(page.elements["part-chip"].hidden, true);
  assert.deepEqual(page.addresses, [["replace", "/find?spotify_id=5awDvzxWfd53SSrsRZ8pXO"]]);

  // Refused for another reason: the error is shown, after one retry only.
  const broken = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO&part=10-20", () =>
    refused({ ok: false, error: { code: "invalid_parameter", message: "Nope." } }),
  );
  assert.equal(broken.requests.length, 2);
  assert.equal(broken.elements.status.textContent, "Nope.");
});

test("Play this part plays the seed's preview from the part's start and stops at its end", async () => {
  const page = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", (url) => {
    if (url.startsWith("/api/mobile/similar")) return partAnswer();
    assert.equal(url, "/api/preview?id=1");
    return { ok: true, preview: "https://cdnt-preview.dzcdn.net/seed.mp3", cover: "" };
  });
  const { elements, audio } = page;
  press(elements["part-open"]);
  assert.equal(elements["part-play"].hidden, false);
  press(elements["part-15"]);
  press(elements["part-play"]);
  assert.equal(elements["part-play"].textContent, "❚❚ Stop");
  await page.settle();
  assert.deepEqual(audio.played, ["https://cdnt-preview.dzcdn.net/seed.mp3"]);
  assert.equal(audio.currentTime, 10);

  audio.currentTime = 17.5;
  audio.dispatch("timeupdate");
  assert.equal(elements["part-head"].hidden, false);
  assert.equal(elements["part-head"].style.left, `${(100 * 17.5) / 30}%`);
  audio.currentTime = 25.1;
  audio.dispatch("timeupdate");
  assert.equal(audio.paused, true, "stopped at the end of the part");
  assert.equal(elements["part-play"].textContent, "▶ Play this part");
  assert.equal(elements["part-head"].hidden, true);

  // Again: the address is reused; moving the span stops it; a row's preview takes over.
  press(elements["part-play"]);
  await page.settle();
  assert.equal(page.requests.filter((url) => url.startsWith("/api/preview")).length, 1);
  assert.equal(audio.paused, false);
  elements["part-track"].dispatch("pointerdown", pointer(27));
  assert.equal(audio.paused, true);
  press(elements["part-play"]);
  await page.settle();
  page.rows()[0].button.dispatch("click");
  await page.settle();
  assert.equal(elements["part-play"].textContent, "▶ Play this part");
  assert.equal(audio.played.at(-1), "https://cdnt-preview.dzcdn.net/1.mp3");
  audio.currentTime = 29;
  audio.dispatch("timeupdate");
  assert.equal(audio.paused, false, "a row's preview is not cut at the part's end");

  // No preview to be had: the button is switched off, the rest still works.
  const silent = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", (url) =>
    url.startsWith("/api/mobile/similar") ? partAnswer() : { ok: false, error: "no preview" },
  );
  press(silent.elements["part-open"]);
  press(silent.elements["part-play"]);
  await silent.settle();
  assert.equal(silent.elements["part-play"].disabled, true);
  assert.equal(silent.elements["part-play"].textContent, "▶ Play this part");
  assert.deepEqual(silent.audio.played, []);

  // A seed without a Deezer id has nothing to play.
  const unknown = await open("?spotify_id=5awDvzxWfd53SSrsRZ8pXO", () => {
    const body = partAnswer();
    body.query.deezer_id = null;
    return body;
  });
  press(unknown.elements["part-open"]);
  assert.equal(unknown.elements["part-play"].hidden, true);
});
