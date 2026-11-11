import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const anysongHtml = read("../anysong/index.html");
const mainHtml = read("../index.html");
const searchJs = read("../search.js");
const inlineScript = (html) => html.match(/<script>([\s\S]*?)<\/script>/)[1];
const ORIGIN = "https://soundalike.yassin.app";

// Just enough DOM to run the pages' own scripts: HTML parsing, simple
// selectors (tag, #id, .class, [attr] and descendants), focus and clicks.
const VOID_TAGS = new Set(["br", "hr", "img", "input", "link", "meta"]);
const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'" };
const decode = (text) =>
  text.replace(/&(amp|lt|gt|quot|#39);/g, (_, name) => ENTITIES[name]);
const encode = (text) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
const dataAttribute = (key) =>
  `data-${String(key).replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`;

class FakeEvents {
  constructor() {
    this.listeners = new Map();
  }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }
  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }
  dispatch(type, extra = {}) {
    const event = { type, target: this, currentTarget: this, preventDefault() {}, ...extra };
    this[`on${type}`]?.(event);
    for (const listener of this.listeners.get(type) || []) listener(event);
  }
}

class FakeText {
  constructor(text) {
    this.textContent = text;
    this.parentElement = null;
  }
  get outerHTML() {
    return encode(this.textContent);
  }
}

class FakeElement extends FakeEvents {
  constructor(document, tag, attributes = {}) {
    super();
    this.ownerDocument = document;
    this.tagName = tag.toUpperCase();
    this.attributes = attributes;
    this.childNodes = [];
    this.parentElement = null;
    this.style = {};
    this.value = attributes.value || "";
    this.disabled = Object.hasOwn(attributes, "disabled");
    this.dataset = new Proxy({}, {
      get: (_, key) => this.attributes[dataAttribute(key)],
      set: (_, key, value) => {
        this.attributes[dataAttribute(key)] = String(value);
        return true;
      },
    });
  }
  get id() {
    return this.attributes.id || "";
  }
  get className() {
    return this.attributes.class || "";
  }
  set className(value) {
    this.attributes.class = String(value);
  }
  get hidden() {
    return Object.hasOwn(this.attributes, "hidden");
  }
  set hidden(value) {
    if (value) this.attributes.hidden = "";
    else delete this.attributes.hidden;
  }
  get tabIndex() {
    return Number(this.attributes.tabindex ?? -1);
  }
  set tabIndex(value) {
    this.attributes.tabindex = String(value);
  }
  get classList() {
    const names = () => this.className.split(/\s+/).filter(Boolean);
    const set = (name, on) => {
      const rest = names().filter((item) => item !== name);
      this.className = (on ? [...rest, name] : rest).join(" ");
      return on;
    };
    return {
      contains: (name) => names().includes(name),
      add: (name) => set(name, true),
      remove: (name) => set(name, false),
      toggle: (name, force = !names().includes(name)) => set(name, force),
    };
  }
  get children() {
    return this.childNodes.filter((node) => node instanceof FakeElement);
  }
  get textContent() {
    return this.childNodes.map((node) => node.textContent).join("");
  }
  set textContent(value) {
    this.replaceChildren(new FakeText(String(value)));
  }
  get innerHTML() {
    return this.childNodes.map((node) => node.outerHTML).join("");
  }
  set innerHTML(html) {
    this.replaceChildren();
    parseInto(this, String(html));
  }
  get outerHTML() {
    const tag = this.tagName.toLowerCase();
    const attributes = Object.entries(this.attributes)
      .map(([name, value]) => ` ${name}="${encode(value)}"`)
      .join("");
    return VOID_TAGS.has(tag)
      ? `<${tag}${attributes}>`
      : `<${tag}${attributes}>${this.innerHTML}</${tag}>`;
  }
  getAttribute(name) {
    return this.attributes[name] ?? null;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  removeAttribute(name) {
    delete this.attributes[name];
  }
  appendChild(node) {
    node.parentElement = this;
    this.childNodes.push(node);
    return node;
  }
  replaceChildren(...nodes) {
    this.childNodes.forEach((node) => {
      node.parentElement = null;
    });
    this.childNodes = [];
    nodes.forEach((node) => this.appendChild(node));
  }
  after(node) {
    const siblings = this.parentElement.childNodes;
    node.parentElement = this.parentElement;
    siblings.splice(siblings.indexOf(this) + 1, 0, node);
  }
  closest(selector) {
    for (let node = this; node; node = node.parentElement) {
      if (matches(node, selector)) return node;
    }
    return null;
  }
  querySelectorAll(selector) {
    const found = [];
    const visit = (element) => element.children.forEach((child) => {
      if (matches(child, selector)) found.push(child);
      visit(child);
    });
    visit(this);
    return found;
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  click() {
    this.dispatch("click");
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  blur() {
    if (this.ownerDocument.activeElement === this) {
      this.ownerDocument.activeElement = this.ownerDocument.body;
    }
  }
  scrollIntoView() {}
}

class FakeDocument extends FakeEvents {
  constructor(bodyHtml) {
    super();
    this.documentElement = new FakeElement(this, "html");
    this.body = this.documentElement.appendChild(new FakeElement(this, "body"));
    this.body.innerHTML = bodyHtml;
    this.activeElement = this.body;
  }
  querySelector(selector) {
    return this.documentElement.querySelector(selector);
  }
  querySelectorAll(selector) {
    return this.documentElement.querySelectorAll(selector);
  }
  getElementById(id) {
    return this.querySelector(`#${id}`);
  }
  createElement(tag) {
    return new FakeElement(this, tag);
  }
}

class FakeAudio extends FakeEvents {
  constructor() {
    super();
    this.paused = true;
  }
  pause() {
    this.paused = true;
  }
  async play() {
    this.paused = false;
  }
}

function parseInto(parent, html) {
  const stack = [parent];
  const tokens =
    /<!--[\s\S]*?-->|<\/([a-z][\w-]*)\s*>|<([a-z][\w-]*)((?:\s+[^\s"'>/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))?)*)\s*\/?>|([^<]+)/gi;
  for (const [, closing, opening, rawAttributes = "", text] of html.matchAll(tokens)) {
    const top = stack.at(-1);
    if (closing) {
      const index = stack.findLastIndex((element, position) =>
        position > 0 && element.tagName === closing.toUpperCase());
      if (index > 0) stack.length = index;
    } else if (opening) {
      const attributes = {};
      for (const [, name, quoted, single, bare] of rawAttributes.matchAll(
        /([^\s"'>/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g,
      )) {
        attributes[name.toLowerCase()] = decode(quoted ?? single ?? bare ?? "");
      }
      const element = top.appendChild(
        new FakeElement(parent.ownerDocument, opening, attributes),
      );
      if (!VOID_TAGS.has(opening.toLowerCase())) stack.push(element);
    } else if (text) {
      top.appendChild(new FakeText(decode(text)));
    }
  }
}

function compound(text) {
  const parts = { tag: "", classes: [], attributes: [] };
  for (const [, tag, id, name, attribute, value] of text.matchAll(
    /^([a-z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:="?([^"\]]*)"?)?\]/gi,
  )) {
    if (tag) parts.tag = tag.toUpperCase();
    else if (id) parts.attributes.push(["id", id]);
    else if (name) parts.classes.push(name);
    else parts.attributes.push([attribute, value]);
  }
  return parts;
}

function matchesCompound(element, { tag, classes, attributes }) {
  const names = element.className.split(/\s+/);
  return (!tag || element.tagName === tag) &&
    classes.every((name) => names.includes(name)) &&
    attributes.every(([name, value]) =>
      Object.hasOwn(element.attributes, name) &&
      (value === undefined || element.attributes[name] === value));
}

function matches(element, selector) {
  const chain = selector.trim().split(/\s+/).map(compound);
  if (!matchesCompound(element, chain.at(-1))) return false;
  let index = chain.length - 2;
  for (let node = element.parentElement; node && index >= 0; node = node.parentElement) {
    if (matchesCompound(node, chain[index])) index -= 1;
  }
  return index < 0;
}

function json(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function loadPage(html, scripts, { path, hash = "", search = "", storage = {}, local = {}, routes }) {
  const body = html.match(/<body>([\s\S]*)<\/body>/)[1]
    .replace(/<script[\s\S]*?<\/script>/g, "");
  const document = new FakeDocument(body);
  const windowEvents = new FakeEvents();
  const requests = [];
  const errors = [];
  const stored = new Map(Object.entries(storage));
  const kept = new Map(Object.entries(local));
  let currentHash = hash;
  let currentSearch = search;
  const context = {
    AbortController,
    Audio: FakeAudio,
    URL,
    URLSearchParams,
    clearTimeout,
    console: { log() {}, warn() {}, error: (...args) => errors.push(args) },
    document,
    navigator: {},
    sessionStorage: {
      getItem: (key) => stored.get(key) ?? null,
      setItem: (key, value) => stored.set(key, String(value)),
      removeItem: (key) => stored.delete(key),
    },
    localStorage: {
      getItem: (key) => kept.get(key) ?? null,
      setItem: (key, value) => kept.set(key, String(value)),
      removeItem: (key) => kept.delete(key),
    },
    setTimeout,
    scrollTo() {},
    addEventListener: (type, listener) => windowEvents.addEventListener(type, listener),
    removeEventListener: (type, listener) =>
      windowEvents.removeEventListener(type, listener),
    history: {
      replaceState(_state, _title, url) {
        const next = new URL(url, `${ORIGIN}${path}${currentSearch}${currentHash}`);
        currentSearch = next.search;
        currentHash = next.hash;
      },
    },
    location: {
      origin: ORIGIN,
      pathname: path,
      get search() {
        return currentSearch;
      },
      get href() {
        return `${ORIGIN}${path}${currentSearch}${currentHash}`;
      },
      get hash() {
        return currentHash;
      },
      set hash(value) {
        const next = value.startsWith("#") ? value : `#${value}`;
        if (next === currentHash) return;
        currentHash = next;
        setTimeout(() => windowEvents.dispatch("hashchange"), 0);
      },
    },
    fetch: async (url) => {
      requests.push(String(url));
      return routes(new URL(String(url), ORIGIN));
    },
  };
  context.window = context;
  vm.createContext(context);
  for (const script of scripts) vm.runInContext(script, context);
  document.dispatch("DOMContentLoaded");
  return {
    context,
    document,
    errors,
    requests,
    get hash() {
      return currentHash;
    },
    get search() {
      return currentSearch;
    },
    stored,
    kept,
    text: (selector) => document.querySelector(selector)?.textContent,
    texts: (selector) =>
      document.querySelectorAll(selector).map((node) => node.textContent),
    similar: () =>
      requests
        .filter((url) => url.startsWith("/api/anysong/similar"))
        .map((url) => new URL(url, ORIGIN).searchParams),
  };
}

async function waitFor(predicate, timeoutMs = 1000) {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) assert.fail("Timed out waiting for the page");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

const AMANGA_ID = 2206566227;

function anysongBody({ lang = "same", language = "en", hidden = 2, vibe = true, bpm = 121.6 } = {}) {
  return {
    ok: true,
    query: {
      id: AMANGA_ID,
      title: "Amanga",
      artist: "Josiah De Disciple",
      source: "catalog",
      language,
      language_mode: lang,
      hidden,
      scene: "on",
      ...(vibe ? { bpm } : {}),
    },
    results: {
      best: [
        { id: 11, title: "Bar A Thym", artist: "Kerri Chandler", score: 0.2, ranks: { melody: 1 } },
        lang === "all"
          ? { id: 12, title: "Ngiyakhuluma", artist: "Mduduzi", score: 0.1, ranks: { samples: 1 } }
          : { id: 13, title: "Sunshine", artist: "Robert Glasper", score: 0.1, ranks: { samples: 1 } },
      ],
      ...(vibe ? {
        vibe: [
          { id: 71, title: "Deep Vibe", artist: "Black Coffee", score: 0.16667 },
          lang === "all"
            ? { id: 72, title: "Wena", artist: "Mduduzi", score: 0.14286 }
            : { id: 73, title: "Kitchen Floor", artist: "Moodymann", score: 0.14286 },
        ],
      } : {}),
      melody: [{ id: 21, title: "Melody Match", artist: "Someone", score: 0.9 }],
      samples: [{ id: 31, title: "Sample Match", artist: "Someone Else", score: 0.8 }],
      sonic: [{ id: 41, title: "Gumbaya Chords", artist: "Mdu a.k.a TRP", score: 0.7 }],
    },
    timings_ms: { total: 554.3 },
    cached: true,
  };
}

function mainBestBody({ lang = "same", language = "en", hidden = 1, vibe = true } = {}) {
  return {
    ok: true,
    query: {
      id: 3588588931,
      title: "8PM",
      artist: "Nemzzz",
      source: "catalog",
      language,
      language_mode: lang,
      hidden,
      scene: "on",
      ...(vibe ? { bpm: 140.2 } : {}),
    },
    results: {
      best: [
        { id: 51, title: "n.h.i.e.", artist: "21 Savage", score: 0.2, ranks: { melody: 1 } },
        lang === "all"
          ? { id: 52, title: "Airforce Negras", artist: "Spiff TV", score: 0.1, ranks: { samples: 2 } }
          : { id: 53, title: "AHHH HA", artist: "Lil Durk", score: 0.1, ranks: { samples: 1 } },
      ],
      ...(vibe ? {
        vibe: [
          { id: 81, title: "Kept Going", artist: "Unknown T", score: 0.16667 },
          lang === "all"
            ? { id: 82, title: "Booba Vibe", artist: "Booba", score: 0.14286 }
            : { id: 83, title: "Ain't It Different", artist: "Headie One", score: 0.14286 },
        ],
      } : {}),
      melody: [],
      samples: [],
      sonic: [],
    },
    timings_ms: { total: 120 },
    cached: true,
  };
}

function mainSoundsBody({ lang = "same" } = {}) {
  const body = mainBestBody({ lang });
  body.results.sounds = [
    { id: 91, title: "Sounds Match", artist: "Kidwild", score: 0.82, bpm: 141 },
    lang === "all"
      ? { id: 92, title: "Autre Son", artist: "Ninho", score: 0.8, bpm: null }
      : { id: 93, title: "Close Sound", artist: "AntsLive", score: 0.8, bpm: 70.1 },
  ];
  return body;
}

const EIGHT_PM_ID = 3588588931;
const COVER = "https://cdn-images.dzcdn.net/images/cover/0a1b2c/250x250-000000-80-0-0.jpg";
const PREVIEW = "https://cdnt-preview.dzcdn.net/api/1/1/a/b/c/0/preview.mp3";
const TRACKS = [
  {
    id: EIGHT_PM_ID,
    title: "8PM",
    artist: "Nemzzz",
    album: "8PM",
    cover: COVER,
    duration: 150,
    has_preview: true,
    in_catalog: false,
  },
  {
    id: 3588588932,
    title: "8PM (Sped Up)",
    artist: "Nemzzz",
    album: "",
    cover: "https://evil.example/cover.jpg",
    duration: 120,
    has_preview: false,
    in_catalog: false,
  },
];
const LEGACY_APIS = ["/api/search", "/api/recommend", "/api/stats", "/api/spicetify_recommend"];

function loadMain({ hash = "", search = "", storage, local, body = mainBestBody, best, tracks = TRACKS } = {}) {
  return loadPage(mainHtml, [inlineScript(mainHtml), searchJs], {
    path: "/",
    hash,
    search,
    storage,
    local,
    routes: (url) => {
      if (url.hostname === "accounts.spotify.com") {
        return json(200, { access_token: "spotify-token" });
      }
      if (url.pathname === "/api/anysong/similar") {
        return best
          ? best(url.searchParams)
          : json(200, body({ lang: url.searchParams.get("lang") }));
      }
      if (url.pathname === "/api/anysong/search") return json(200, { tracks });
      if (url.pathname === "/api/preview") {
        return json(200, { ok: true, preview: PREVIEW, cover: COVER });
      }
      return json(404, { ok: false });
    },
  });
}

const loadAnysong = ({ hash = `#id=${AMANGA_ID}`, search = "", body = anysongBody, local } = {}) =>
  loadMain({ hash, search, body, local });
const shown = (page) => page.document.querySelector("#matches li.row");
const selected = (page, key) =>
  page.document.querySelector(`#tab-${key}`).getAttribute("aria-selected");
const legacyCalls = (page) =>
  page.requests.filter((url) => LEGACY_APIS.some((api) => url.startsWith(api)));

const BEST_CAPTION =
  "Likely covers, samples and shared melodies first, then similar-sounding songs " +
  "by artists this song's listeners also play, marked Listeners also play. " +
  "Turn on Audio only to leave those out.";
const AUDIO_BEST_CAPTION =
  "Likely covers, samples and shared melodies first, then the songs that sound most alike.";
const SOUNDS_CAPTION =
  "Songs with the same overall sound (instruments, production and mood), favouring " +
  "artists this song's listeners also play. Turn on Audio only to leave that out.";
const AUDIO_SOUNDS_CAPTION =
  "Songs with the same overall sound (instruments, production and mood).";
const AUDIO_ONLY_KEY = "soundalike:audio-only:v1";
const VIBE_CAPTION =
  "Songs from these matches that share this song's tempo and overall sound, one per artist.";
const TAB_LABELS = ["Best matches", "Same vibe", "Same melody", "Samples & loops", "Sounds like"];

test("home page has one any-song search bar and no library promo or size line", async () => {
  const page = loadMain();
  await new Promise((resolve) => setTimeout(resolve, 10));

  assert.equal(page.document.querySelectorAll("input").length, 1);
  assert.equal(
    page.document.querySelector("#q").getAttribute("placeholder"),
    "Search any song or artist…  (e.g. Redbone Childish Gambino)",
  );
  assert.deepEqual(
    page.document.querySelectorAll(".promos a").map((link) => link.getAttribute("href")),
    ["https://github.com/yassinsolim/soundalike/tree/main/integrations/spicetify", "/evaluate"],
  );
  assert.equal(
    page.text("#hint"),
    "Search any song on Deezer, even a brand-new one. A song we haven't heard before takes a few seconds the first time.",
  );
  assert.doesNotMatch(mainHtml, /272,853|library_size|\/api\/stats|href="\/anysong"|Try any song<\/strong>/);
  assert.equal(page.document.querySelectorAll("#chips .chip").length, 7);
  assert.deepEqual(page.requests, []);
  assert.equal(page.text("#out"), "");
  assert.deepEqual(page.errors, []);
});

test("the one search bar suggests any song on Deezer and opens the pick", async () => {
  const page = loadMain();
  const input = page.document.querySelector("#q");
  input.focus();
  input.value = "8pm nemz";
  input.dispatch("input");
  await waitFor(() => page.document.querySelector("#ac .opt"));

  assert.deepEqual(page.requests, ["/api/anysong/search?limit=8&q=8pm%20nemz"]);
  assert.equal(input.getAttribute("aria-expanded"), "true");
  const options = page.document.querySelectorAll("#ac .opt");
  assert.deepEqual(options.map((option) => option.getAttribute("aria-disabled")), ["false", "true"]);
  assert.equal(options[0].querySelector("img.thumb").getAttribute("src"), COVER);
  assert.equal(options[1].querySelector("img"), null, "only Deezer covers are shown");
  assert.match(options[1].textContent, /· no preview$/);

  options[1].click();
  assert.equal(page.hash, "", "a song without a preview cannot be matched");
  options[0].click();
  assert.equal(page.hash, `#id=${EIGHT_PM_ID}&tab=best`);
  assert.equal(input.value, "8PM — Nemzzz");
  assert.equal(page.document.querySelector("#ac").style.display, "none");
  await waitFor(() => shown(page));

  const [params] = page.similar();
  assert.equal(params.get("id"), String(EIGHT_PM_ID));
  assert.equal(params.get("k"), "20");
  assert.equal(params.get("lang"), "same");
  assert.equal(params.get("policy"), "best-vibe-sounds-v4");
  assert.equal(params.get("scene"), null);
  assert.equal(page.text(".seed h2"), "8PM");
  assert.equal(page.document.querySelector(".seed .cover img").getAttribute("src"), COVER);
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "AHHH HA"]);
  assert.deepEqual(legacyCalls(page), []);
  assert.deepEqual(page.errors, []);
});

test("arrow keys wrap through suggestions and Enter picks one with a preview", async () => {
  const page = loadMain({ tracks: [TRACKS[1], TRACKS[0]] });
  const input = page.document.querySelector("#q");
  input.focus();
  input.value = "8pm";
  input.dispatch("input");
  await waitFor(() => page.document.querySelector("#ac .opt"));

  input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(input.getAttribute("aria-activedescendant"), "ac-0");
  input.dispatch("keydown", { key: "ArrowDown" });
  input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(input.getAttribute("aria-activedescendant"), "ac-0");
  input.dispatch("keydown", { key: "Escape" });
  assert.equal(input.getAttribute("aria-expanded"), "false");

  input.value = "8pm";
  input.dispatch("input");
  assert.equal(page.document.querySelectorAll("#ac .opt").length, 2, "repeat queries come from the cache");
  input.dispatch("keydown", { key: "Enter" });
  assert.equal(page.hash, `#id=${EIGHT_PM_ID}&tab=best`);
  assert.equal(
    page.requests.filter((url) => url.startsWith("/api/anysong/search")).length,
    1,
  );
});

test("the /anysong stub forwards old links to the home page", () => {
  for (const [search, hash, expected] of [
    [`?id=${AMANGA_ID}&tab=sonic`, "", `/?id=${AMANGA_ID}&tab=sonic`],
    ["", `#id=${AMANGA_ID}&tab=vibe`, `/#id=${AMANGA_ID}&tab=vibe`],
    ["", "", "/"],
  ]) {
    let target = null;
    vm.runInNewContext(inlineScript(anysongHtml), {
      location: { search, hash, replace: (url) => { target = url; } },
    });
    assert.equal(target, expected);
  }
  assert.match(anysongHtml, /<meta http-equiv="refresh" content="0; url=\/">/);
  assert.match(anysongHtml, /<a href="\/">soundalike home page<\/a>/);
});

test("old /anysong?id= links land on the same song's results", async () => {
  const page = loadAnysong({ hash: "", search: `?id=${AMANGA_ID}&tab=sonic&ref=share` });
  await waitFor(() => shown(page));

  assert.equal(page.search, "?ref=share");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=sounds`);
  assert.equal(page.similar()[0].get("id"), String(AMANGA_ID));
  assert.equal(selected(page, "sounds"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Gumbaya Chords"]);
  assert.equal(page.document.querySelector("#language").hidden, true);
  assert.deepEqual(page.errors, []);
});

test("old any-song #id= links to Sounds like still open the sound channel", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=sonic` });
  await waitFor(() => shown(page));

  assert.equal(selected(page, "sounds"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Gumbaya Chords"]);
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=sounds`);
});

test("the home page's own #s= links still open the song", async () => {
  const page = loadMain({ hash: `#s=${encodeURIComponent("8PM — Nemzzz")}` });
  await waitFor(() => shown(page));

  const [params] = page.similar();
  assert.equal(params.get("title"), "8PM");
  assert.equal(params.get("artist"), "Nemzzz");
  assert.equal(params.get("id"), null);
  assert.equal(page.hash, `#id=${EIGHT_PM_ID}&tab=best`);
  assert.equal(page.document.querySelector("#q").value, "8PM — Nemzzz");

  const titleOnly = loadMain({ hash: "#s=8PM" });
  await waitFor(() => shown(titleOnly));
  assert.equal(titleOnly.requests[0], "/api/anysong/search?limit=8&q=8PM");
  assert.equal(titleOnly.similar()[0].get("id"), String(EIGHT_PM_ID));
});

test("example chips and Surprise me open songs through the any-song search", async () => {
  const page = loadMain();
  const chips = page.document.querySelectorAll("#chips .chip");
  assert.equal(chips[0].id, "surprise");
  assert.equal(chips[2].textContent, "Redbone · Childish Gambino");

  chips[2].click();
  assert.equal(page.hash, `#s=${encodeURIComponent("Redbone — Childish Gambino")}`);
  await waitFor(() => shown(page));
  assert.equal(page.similar()[0].get("title"), "Redbone");
  assert.equal(page.similar()[0].get("artist"), "Childish Gambino");

  page.document.querySelector("#surprise").click();
  await waitFor(() => page.similar().length === 2);
  assert.deepEqual(legacyCalls(page), []);
  assert.deepEqual(page.errors, []);
});

test("all five any-song tabs read one response", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));

  assert.deepEqual(page.texts("#out .tab"), TAB_LABELS);
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  const expected = {
    best: [["Bar A Thym", "Sunshine"], BEST_CAPTION, false],
    vibe: [["Deep Vibe", "Kitchen Floor"], VIBE_CAPTION, false],
    melody: [["Melody Match"], "Covers, interpolations and songs that reuse the tune.", true],
    samples: [
      ["Sample Match"],
      "Songs that may share a sample, a loop or a chopped-up section.",
      true,
    ],
    sounds: [["Gumbaya Chords"], "A similar overall sound: production, instruments and mood.", true],
  };
  for (const [key, [titles, caption, unfiltered]] of Object.entries(expected)) {
    page.document.querySelector(`#tab-${key}`).click();
    assert.equal(selected(page, key), "true");
    assert.equal(page.document.querySelector("#matches").getAttribute("aria-labelledby"), `tab-${key}`);
    assert.deepEqual(page.texts("#matches .t"), titles, key);
    assert.equal(page.text("#matchblurb"), caption, key);
    assert.equal(page.document.querySelector("#language").hidden, unfiltered, key);
    assert.equal(page.hash, `#id=${AMANGA_ID}&tab=${key}`);
  }
  assert.deepEqual(page.texts("#matches .why"), []);
  assert.deepEqual(
    page.document.querySelectorAll("#matches a.play").map((link) => link.getAttribute("href")),
    ["https://open.spotify.com/search/Gumbaya%20Chords%20Mdu%20a.k.a%20TRP"],
  );
  assert.equal(page.similar().length, 1);
  assert.deepEqual(page.errors, []);
});

test("Best matches ask for one language and say what was hidden", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));

  assert.equal(
    page.requests[0],
    `/api/anysong/similar?id=${AMANGA_ID}&k=20&lang=same&policy=best-vibe-sounds-v4`,
  );
  assert.equal(page.text("#matchblurb"), BEST_CAPTION);
  const box = page.document.querySelector("#language");
  assert.equal(box.hidden, false);
  assert.equal(box.getAttribute("role"), "status");
  assert.equal(
    page.text("#language-status"),
    "English · 2 other-language matches hidden from the top 20",
  );
  const toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.hidden, false);
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(toggle.getAttribute("aria-pressed"), "false");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Sunshine"]);
  assert.deepEqual(page.texts("#matches .why"), [" · melody #1", " · samples #1"]);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library"]);
  assert.deepEqual(page.errors, []);
});

test("the language switch reloads Best matches and keeps focus on it", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));
  const toggle = page.document.querySelector("#language-toggle");
  toggle.focus();
  toggle.click();

  assert.equal(toggle.textContent, "Show same language");
  assert.match(page.text("#matches"), /matching…/);
  await waitFor(() =>
    page.text("#language-status") === "English · 2 other-language matches in the top 20");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Ngiyakhuluma"]);

  toggle.click();
  await waitFor(() => page.similar().length === 3 &&
    page.text("#language-status")?.endsWith("hidden from the top 20"));
  assert.equal(page.similar()[2].get("lang"), "same");
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(page.document.activeElement, toggle);
  assert.equal(page.document.querySelector(".seed h2").textContent, "Amanga");
});

test("an uncertain language shows all languages without a switch", async () => {
  const page = loadAnysong({
    body: ({ lang }) => anysongBody({ lang, language: "unknown", hidden: 0 }),
  });
  await waitFor(() => shown(page));

  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  page.document.querySelector("#tab-vibe").click();
  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
});

test("Same vibe sits next to Best matches and opens from a share link", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=vibe` });
  await waitFor(() => shown(page));

  assert.deepEqual(page.texts("#out .tab"), TAB_LABELS);
  assert.equal(selected(page, "vibe"), "true");
  assert.equal(selected(page, "best"), "false");
  assert.equal(page.text("#matchblurb"), VIBE_CAPTION);
  assert.deepEqual(page.texts("#matches .t"), ["Deep Vibe", "Kitchen Floor"]);
  assert.deepEqual(page.texts("#matches .why"), []);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library", "122 BPM"]);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.text("#language-status"), "English · other languages left out");
  const toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.hidden, false);
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(toggle.getAttribute("aria-pressed"), "false");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=vibe`);
  assert.equal(page.similar().length, 1);
  assert.deepEqual(page.errors, []);

  page.document.querySelector("#tab-best").click();
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  assert.equal(page.text("#matchblurb"), BEST_CAPTION);
  page.document.querySelector("#tab-best").onkeydown({ key: "ArrowRight" });
  assert.equal(selected(page, "vibe"), "true");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=vibe`);
  page.document.querySelector("#tab-vibe").onkeydown({ key: "ArrowRight", preventDefault() {} });
  assert.equal(selected(page, "melody"), "true");
  assert.equal(page.document.querySelector("#language").hidden, true);
  page.document.querySelector("#tab-melody").onkeydown({ key: "ArrowLeft", preventDefault() {} });
  assert.equal(selected(page, "vibe"), "true");
});

test("the language switch on Same vibe is shared with Best matches", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=vibe` });
  await waitFor(() => shown(page));
  const toggle = page.document.querySelector("#language-toggle");
  toggle.focus();
  toggle.click();

  await waitFor(() => page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(toggle.textContent, "Show same language");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.equal(selected(page, "vibe"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Deep Vibe", "Wena"]);

  page.document.querySelector("#tab-best").click();
  assert.equal(
    page.text("#language-status"),
    "English · 2 other-language matches in the top 20",
  );
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Ngiyakhuluma"]);
  assert.equal(page.similar().length, 2);
});

test("Same vibe and the tempo are hidden when the server has neither", async () => {
  const page = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=vibe`,
    body: ({ lang }) => anysongBody({ lang, vibe: false }),
  });
  await waitFor(() => shown(page));

  assert.equal(page.document.querySelector("#tab-vibe").hidden, true);
  assert.equal(selected(page, "best"), "true");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  assert.equal(page.text("#matchblurb"), BEST_CAPTION);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library"]);
  assert.equal(
    page.text("#language-status"),
    "English · 2 other-language matches hidden from the top 20",
  );
  page.document.querySelector("#tab-best").onkeydown({ key: "ArrowRight" });
  assert.equal(selected(page, "melody"), "true");
  page.document.querySelector("#tab-melody").onkeydown({ key: "ArrowLeft" });
  assert.equal(selected(page, "best"), "true");
  assert.deepEqual(page.errors, []);
});

test("Same vibe shows all languages without a switch for an uncertain seed", async () => {
  const page = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=vibe`,
    body: ({ lang }) => anysongBody({ lang, language: "unknown", hidden: 0, bpm: null }),
  });
  await waitFor(() => shown(page));

  assert.equal(selected(page, "vibe"), "true");
  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library"]);
});

test("Sounds like reads the fused list and shares the language switch", async () => {
  const page = loadMain({
    hash: `#id=${EIGHT_PM_ID}`,
    best: (params) => json(200, mainSoundsBody({ lang: params.get("lang") })),
  });
  await waitFor(() => shown(page));

  page.document.querySelector("#tab-sounds").click();
  assert.deepEqual(page.texts("#matches .t"), ["Sounds Match", "Close Sound"]);
  assert.equal(page.text("#matchblurb"), SOUNDS_CAPTION);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library", "140 BPM"]);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.text("#language-status"), "English · other languages left out");

  page.document.querySelector("#language-toggle").click();
  await waitFor(() => page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(selected(page, "sounds"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Sounds Match", "Autre Son"]);
  assert.deepEqual(legacyCalls(page), []);
  assert.deepEqual(page.errors, []);
});

test("Best matches label the songs placed by what listeners also play", async () => {
  const page = loadAnysong({
    body: ({ lang }) => {
      const body = anysongBody({ lang });
      body.results.best[0].basis = "audio";
      body.results.best[1].basis = "listeners";
      return body;
    },
  });
  await waitFor(() => shown(page));

  assert.deepEqual(page.texts("#matches .why"), [
    " · melody #1",
    " · Listeners also play · samples #1",
  ]);
  const labels = page.document.querySelectorAll("#matches .basis");
  assert.equal(labels.length, 1);
  assert.equal(
    labels[0].getAttribute("title"),
    "Placed by what this song's listeners also play, not by audio alone",
  );
  page.document.querySelector("#tab-melody").click();
  assert.deepEqual(page.texts("#matches .basis"), []);
  assert.deepEqual(page.errors, []);
});

test("Audio only reloads the lists without listeners and is remembered", async () => {
  const page = loadMain({
    hash: `#id=${EIGHT_PM_ID}`,
    best: (params) => json(200, mainSoundsBody({ lang: params.get("lang") })),
  });
  await waitFor(() => shown(page));
  const audio = page.document.querySelector("#audio-only");
  assert.equal(audio.textContent, "Audio only");
  assert.equal(audio.getAttribute("aria-pressed"), "false");
  assert.equal(page.similar()[0].get("scene"), null);

  audio.focus();
  audio.click();
  assert.equal(audio.getAttribute("aria-pressed"), "true");
  assert.match(page.text("#matches"), /matching…/);
  await waitFor(() => page.similar().length === 2 && shown(page));
  assert.equal(page.similar()[1].get("scene"), "0");
  assert.equal(page.similar()[1].get("lang"), "same");
  assert.equal(page.kept.get(AUDIO_ONLY_KEY), "true");
  assert.equal(page.document.activeElement, audio);
  assert.equal(page.text("#matchblurb"), AUDIO_BEST_CAPTION);
  page.document.querySelector("#tab-sounds").click();
  assert.equal(page.text("#matchblurb"), AUDIO_SOUNDS_CAPTION);
  assert.equal(page.document.querySelector("#language").hidden, false);

  page.document.querySelector("#language-toggle").click();
  await waitFor(() => page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[2].get("lang"), "all");
  assert.equal(page.similar()[2].get("scene"), "0");

  audio.click();
  await waitFor(() => page.similar().length === 4 && shown(page));
  assert.equal(page.similar()[3].get("scene"), null);
  assert.equal(page.similar()[3].get("lang"), "all");
  assert.equal(audio.getAttribute("aria-pressed"), "false");
  assert.equal(page.kept.has(AUDIO_ONLY_KEY), false);
  assert.equal(page.text("#matchblurb"), SOUNDS_CAPTION);
  assert.deepEqual(page.errors, []);
});

test("a saved Audio only choice applies to the first request", async () => {
  const page = loadAnysong({ local: { [AUDIO_ONLY_KEY]: "true" } });
  await waitFor(() => shown(page));

  assert.equal(page.similar().length, 1);
  assert.equal(page.similar()[0].get("scene"), "0");
  assert.equal(page.document.querySelector("#audio-only").getAttribute("aria-pressed"), "true");
  assert.equal(page.text("#matchblurb"), AUDIO_BEST_CAPTION);
});

test("Audio only stays reachable when its reload fails", async () => {
  const page = loadMain({
    hash: `#id=${EIGHT_PM_ID}`,
    best: (params) => params.get("scene") === "0"
      ? json(500, { ok: false })
      : json(200, mainBestBody({ lang: params.get("lang") })),
  });
  await waitFor(() => shown(page));
  const audio = page.document.querySelector("#audio-only");
  audio.click();

  await waitFor(() => page.document.querySelector("#matches .state.err"));
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  assert.equal(audio.getAttribute("aria-pressed"), "true");

  audio.click();
  await waitFor(() => shown(page));
  assert.equal(page.similar().at(-1).get("scene"), null);
  assert.equal(page.document.querySelector("#language-toggle").hidden, false);
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "AHHH HA"]);
});

test("a reload without a Same vibe list returns to Best matches", async () => {
  const page = loadMain({
    hash: `#id=${EIGHT_PM_ID}`,
    best: (params) => {
      const lang = params.get("lang");
      return json(200, mainBestBody({ lang, vibe: lang !== "all" }));
    },
  });
  await waitFor(() => shown(page));
  page.document.querySelector("#tab-vibe").click();
  page.document.querySelector("#language-toggle").click();

  await waitFor(() => page.text("#language-status") === "English · 1 other-language match in the top 20");
  assert.equal(page.document.querySelector("#tab-vibe").hidden, true);
  assert.equal(selected(page, "best"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "Airforce Negras"]);
  assert.equal(page.hash, `#id=${EIGHT_PM_ID}&tab=best`);
});

test("a song the matcher cannot use shows the reason instead of tabs", async () => {
  const page = loadMain({
    hash: `#id=${EIGHT_PM_ID}`,
    best: () => json(404, { ok: false, error: "no preview available for this track" }),
  });
  await waitFor(() => page.document.querySelector("#out .state.err"));

  assert.equal(page.text("#out .state.err"), "No preview available for this track.");
  assert.equal(page.document.querySelector("#out .tab"), null);
  assert.deepEqual(legacyCalls(page), []);
});

test("previews play from the seed card and from each match", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));
  await waitFor(() => page.document.querySelector(".seed .cover img"));
  assert.equal(page.document.querySelector(".seed .cover img").getAttribute("src"), COVER);

  const seedButton = page.document.querySelector(".seed .pv");
  seedButton.click();
  await waitFor(() => seedButton.textContent === "⏸");
  assert.ok(page.requests.includes(`/api/preview?id=${AMANGA_ID}`));
  assert.equal(seedButton.classList.contains("playing"), true);

  const rowButton = page.document.querySelector("#matches .pv");
  rowButton.click();
  await waitFor(() => rowButton.textContent === "⏸");
  assert.ok(page.requests.includes("/api/preview?id=11"));
  assert.equal(seedButton.textContent, "▶");
  assert.equal(shown(page).classList.contains("playing"), true);
  rowButton.click();
  assert.equal(rowButton.textContent, "▶");
});

test("Copy list, Share, Open in Spotify and Save use the list on screen", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));
  let copied = null;
  page.context.navigator.clipboard = { writeText: async (text) => { copied = text; } };

  page.document.querySelector("#copy").click();
  await waitFor(() => copied !== null);
  assert.equal(copied, "Bar A Thym — Kerri Chandler\nSunshine — Robert Glasper");
  page.document.querySelector("#tab-melody").click();
  page.document.querySelector("#copy").click();
  await waitFor(() => copied === "Melody Match — Someone");

  page.document.querySelector("#share").click();
  await waitFor(() => page.text("#share") === "✓ Link copied");
  assert.equal(copied, `${ORIGIN}/#id=${AMANGA_ID}&tab=melody`);
  assert.deepEqual(
    page.document.querySelectorAll("#matches a.play").map((link) => link.getAttribute("href")),
    ["https://open.spotify.com/search/Melody%20Match%20Someone"],
  );
  const save = page.document.querySelector("#save");
  assert.equal(save.disabled, false);
  assert.equal(save.onclick, page.context.savePlaylist);
  assert.equal(vm.runInContext("lastSeed.title", page.context), "Amanga");
  assert.equal(vm.runInContext("lastResults.length", page.context), 1);

  page.document.querySelector("#tab-samples").click();
  assert.equal(vm.runInContext("lastResults[0].title", page.context), "Sample Match");
  assert.deepEqual(page.errors, []);
});

test("a match's title opens songs like it, and clearing the link clears the page", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));

  page.document.querySelector("#matches .t").click();
  assert.equal(page.hash, "#id=11&tab=best");
  assert.equal(page.document.querySelector("#q").value, "Bar A Thym — Kerri Chandler");
  await waitFor(() => page.similar().length === 2);
  assert.equal(page.similar()[1].get("id"), "11");
  await waitFor(() => shown(page));

  page.context.location.hash = "";
  await waitFor(() => page.text("#out") === "");
  assert.equal(page.document.querySelector("#q").value, "");
});

test("returning from the Spotify login reopens the song that was open", async () => {
  const page = loadMain({
    search: "?code=auth-code&state=expected",
    storage: {
      sa_state: "expected",
      sa_verifier: "verifier",
      sa_return: `#id=${AMANGA_ID}&tab=vibe`,
    },
    body: anysongBody,
  });
  assert.equal(page.search, "");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=vibe`);
  await waitFor(() => shown(page));
  assert.equal(selected(page, "vibe"), "true");
  await waitFor(() => page.text("#sp") === "✓ Spotify connected");
  assert.equal(page.stored.get("sa_token"), "spotify-token");
  assert.equal(page.stored.has("sa_return"), false);
  assert.ok(page.requests.includes("https://accounts.spotify.com/api/token"));
  assert.deepEqual(page.errors, []);
});
