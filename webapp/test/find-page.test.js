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
}

const IDS = ["form", "link", "status", "list", "seed", "seed-art", "seed-title", "seed-artist", "bar", "count", "p-spotify", "p-apple"];

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

/** Loads the page at `search` with `respond(url)` standing in for the server. */
async function open(search, respond, { userAgent = "Mozilla/5.0 (Linux; Android 14)", session = {} } = {}) {
  const elements = Object.fromEntries(IDS.map((id) => [id, new FakeElement("div", id)]));
  const requests = [];
  const audio = new FakeElement("audio");
  Object.assign(audio, { played: [], pause() {}, async play() { this.played.push(this.src); } });
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
    history: { pushState() {} },
    navigator: { userAgent },
    Audio: function Audio() { return audio; },
    URLSearchParams,
    Date,
    JSON,
    fetch: async (url) => {
      requests.push(url);
      const body = await respond(url);
      return { json: async () => body };
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
  return { elements, requests, rows, settle, audio, context };
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
