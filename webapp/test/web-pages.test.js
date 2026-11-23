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
  // Every element is 300 px wide, so 10 px are one second of a 30 s timeline.
  getBoundingClientRect() {
    return { left: 0, top: 0, width: 300, height: 60 };
  }
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
  const audios = [];
  const stored = new Map(Object.entries(storage));
  const kept = new Map(Object.entries(local));
  let currentHash = hash;
  let currentSearch = search;
  const context = {
    AbortController,
    Audio: class extends FakeAudio {
      constructor() {
        super();
        audios.push(this);
      }
    },
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
    audios,
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

// /api/anysong/similar answers. The page shows main (Best matches) and
// listeners (Listeners also play) and ignores the other lists. Audio only
// (scene "off") and artists without listener data ("unknown_artist") come
// without listeners; older servers send neither, only Best matches with
// listener picks in it. After the audio head the main list's rows come from
// the five-signal model (basis "consensus"), or in sound order ("sound")
// with Audio only; `body` overrides that for other servers.
function anysongBody({
  lang = "same", language = "en", hidden = 2, scene = "on", older = false,
  body = scene === "on" ? "consensus" : "sound",
} = {}) {
  const listeners = scene === "on";
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
      scene,
      bpm: 121.6,
    },
    results: {
      ...(older ? {} : {
        main: [
          { id: 11, title: "Bar A Thym", artist: "Kerri Chandler", score: 0.16667, basis: "audio", match: 41, ranks: { melody: 1 } },
          lang === "all"
            ? { id: 12, title: "Ngiyakhuluma", artist: "Mduduzi", score: 0.14286, basis: body, match: 88, ranks: { sonic: 2 } }
            : { id: 13, title: "Sunshine", artist: "Robert Glasper", score: 0.14286, basis: body, match: 90, ranks: { sonic: 1 } },
        ],
      }),
      ...(older || !listeners ? {} : {
        listeners: [
          { id: 71, title: "Deep Vibe", artist: "Black Coffee", score: 0.16667, basis: "listeners", match: 70, ranks: { sonic: 12 } },
          lang === "all"
            ? { id: 72, title: "Wena", artist: "Mduduzi", score: 0.14286, basis: "listeners", match: 40, ranks: {} }
            : { id: 73, title: "Kitchen Floor", artist: "Moodymann", score: 0.14286, basis: "listeners", match: 52, ranks: { samples: 3 } },
        ],
      }),
      best: [
        { id: 11, title: "Bar A Thym", artist: "Kerri Chandler", score: 0.2, basis: "audio", ranks: { melody: 1 } },
        listeners
          ? { id: 14, title: "Listener Pick", artist: "Culoe De Song", score: 0.1, basis: "listeners", ranks: { samples: 1 } }
          : { id: 13, title: "Sunshine", artist: "Robert Glasper", score: 0.1, ranks: { samples: 1 } },
      ],
      vibe: [{ id: 81, title: "Vibe Match", artist: "Black Motion", score: 0.16667 }],
      melody: [{ id: 21, title: "Melody Match", artist: "Someone", score: 0.9 }],
      samples: [{ id: 31, title: "Sample Match", artist: "Someone Else", score: 0.8 }],
      sounds: [{ id: 91, title: "Sounds Match", artist: "Kidwild", score: 0.82 }],
      sonic: [{ id: 41, title: "Gumbaya Chords", artist: "Mdu a.k.a TRP", score: 0.7 }],
    },
    timings_ms: { total: 554.3 },
    cached: true,
  };
}

function mainBestBody({ lang = "same", language = "en", hidden = 1, scene = "on" } = {}) {
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
      scene,
      bpm: 140.2,
    },
    results: {
      main: [
        { id: 51, title: "n.h.i.e.", artist: "21 Savage", score: 0.16667, basis: "audio", match: 35, ranks: { melody: 1 } },
        lang === "all"
          ? { id: 52, title: "Airforce Negras", artist: "Spiff TV", score: 0.14286, basis: scene === "on" ? "consensus" : "sound", match: 77, ranks: { sonic: 3 } }
          : { id: 53, title: "AHHH HA", artist: "Lil Durk", score: 0.14286, basis: scene === "on" ? "consensus" : "sound", match: 81, ranks: { sonic: 2 } },
      ],
      ...(scene === "on" ? {
        listeners: [
          { id: 81, title: "Kept Going", artist: "Unknown T", score: 0.16667, basis: "listeners", match: 66, ranks: { sonic: 9 } },
          lang === "all"
            ? { id: 82, title: "Autre Son", artist: "Ninho", score: 0.14286, basis: "listeners", match: 31, ranks: {} }
            : { id: 83, title: "Ain't It Different", artist: "Headie One", score: 0.14286, basis: "listeners", match: 58, ranks: {} },
        ],
      } : {}),
      best: [
        { id: 51, title: "n.h.i.e.", artist: "21 Savage", score: 0.2, basis: "audio", ranks: { melody: 1 } },
        { id: 53, title: "AHHH HA", artist: "Lil Durk", score: 0.1, ranks: { samples: 1 } },
      ],
      melody: [],
      samples: [],
      sonic: [],
    },
    timings_ms: { total: 120 },
    cached: true,
  };
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
          : json(200, body({
            lang: url.searchParams.get("lang"),
            scene: url.searchParams.get("scene") === "0" ? "off" : "on",
          }));
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
  "Likely covers, samples and shared melodies first, then other artists' songs a model ranks closest, " +
  "from five signals: sound, scene, playlists, melody and samples. Turn on Audio only to rank by sound alone.";
const NUDGED_BEST_CAPTION =
  "Likely covers, samples and shared melodies first, then the songs that sound most alike, " +
  "nudged toward artists this song's listeners also play. Turn on Audio only to rank by sound alone.";
const AUDIO_BEST_CAPTION =
  "Likely covers, samples and shared melodies first, then the songs that sound most alike.";
const MIXED_BEST_CAPTION =
  "Likely covers, samples and shared melodies first, then similar-sounding songs " +
  "by artists this song's listeners also play, marked Listeners also play. " +
  "Turn on Audio only to leave those out.";
const LISTENERS_CAPTION =
  "Songs by other artists that this song's listeners also play, one per artist. " +
  "The likelier its listeners are to play a song, and the closer it sounds, the higher it ranks.";
const AUDIO_ONLY_LISTENERS_NOTE =
  "Audio only is on, and this list comes from what this song's listeners also play. " +
  "Turn Audio only off to see it.";
const NO_LISTENERS_NOTE =
  "We don't know what this artist's listeners play yet, so there's no list here. " +
  "Best matches still has this song's closest matches.";
const AUDIO_ONLY_KEY = "soundalike:audio-only:v1";
const TAB_KEY = "soundalike:tab:v1";
const TAB_LABELS = ["Best matches", "Listeners also play", "Melody match"];
const DROPPED_TABS = ["vibe", "samples", "sounds", "sonic"];
const MELODY_CAPTION =
  "Likely covers, samples and shared melodies first, from the melody and sample matches " +
  "fused into one ranking, then similar-sounding songs by artists this song's listeners " +
  "also play, marked Listeners also play. Turn on Audio only to leave those out.";
const AUDIO_MELODY_CAPTION =
  "Likely covers, samples and shared melodies: the melody and sample matches fused into " +
  "one ranking, by audio alone.";

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
  assert.equal(params.get("policy"), "main-consensus-v4");
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

test("old /anysong?id= links land on the same song's Best matches", async () => {
  const page = loadAnysong({ hash: "", search: `?id=${AMANGA_ID}&tab=sonic&ref=share` });
  await waitFor(() => shown(page));

  assert.equal(page.search, "?ref=share");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  assert.equal(page.similar()[0].get("id"), String(AMANGA_ID));
  assert.equal(selected(page, "best"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Sunshine"]);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.deepEqual(page.errors, []);

  const listeners = loadAnysong({ hash: "", search: `?id=${AMANGA_ID}&tab=listeners` });
  await waitFor(() => shown(listeners));
  assert.equal(listeners.search, "");
  assert.equal(listeners.hash, `#id=${AMANGA_ID}&tab=listeners`);
  assert.equal(selected(listeners, "listeners"), "true");
  assert.deepEqual(listeners.texts("#matches .t"), ["Deep Vibe", "Kitchen Floor"]);
});

test("old #id= links to the five tabs, and unknown tabs, open Best matches", async () => {
  for (const old of [...DROPPED_TABS, "best", "lyrics", ""]) {
    const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=${old}` });
    await waitFor(() => shown(page));
    assert.equal(selected(page, "best"), "true", old);
    assert.equal(selected(page, "listeners"), "false", old);
    assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Sunshine"], old);
    assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`, old);
  }

  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=listeners` });
  await waitFor(() => shown(page));
  assert.equal(selected(page, "listeners"), "true");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=listeners`);
  page.context.location.hash = `#id=${AMANGA_ID}&tab=samples`;
  await waitFor(() => selected(page, "best") === "true");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Sunshine"]);
  assert.equal(page.similar().length, 1);
  assert.deepEqual(page.errors, []);
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

test("three any-song tabs read one response, as in the Spotify extension", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));

  assert.deepEqual(page.texts("#out .tab"), TAB_LABELS);
  for (const old of DROPPED_TABS) {
    assert.equal(page.document.querySelector(`#tab-${old}`), null, old);
  }
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  const expected = {
    best: [["Bar A Thym", "Sunshine"], BEST_CAPTION],
    listeners: [["Deep Vibe", "Kitchen Floor"], LISTENERS_CAPTION],
  };
  for (const [key, [titles, caption]] of Object.entries(expected)) {
    page.document.querySelector(`#tab-${key}`).click();
    assert.equal(selected(page, key), "true");
    assert.equal(page.document.querySelector("#matches").getAttribute("aria-labelledby"), `tab-${key}`);
    assert.deepEqual(page.texts("#matches .t"), titles, key);
    assert.equal(page.text("#matchblurb"), caption, key);
    assert.equal(page.document.querySelector("#language").hidden, false, key);
    assert.equal(page.hash, `#id=${AMANGA_ID}&tab=${key}`);
  }
  // Like the extension, Listeners also play rows carry no ranks or labels.
  assert.deepEqual(page.texts("#matches .why"), []);
  assert.deepEqual(
    page.document.querySelectorAll("#matches a.play").map((link) => link.getAttribute("href")),
    [
      "https://open.spotify.com/search/Deep%20Vibe%20Black%20Coffee",
      "https://open.spotify.com/search/Kitchen%20Floor%20Moodymann",
    ],
  );
  // Melody match is the server's "best" list: the fused melody and sample
  // matches with their ranks, then the listener picks, marked.
  page.document.querySelector("#tab-melody").click();
  assert.equal(selected(page, "melody"), "true");
  assert.equal(page.document.querySelector("#matches").getAttribute("aria-labelledby"), "tab-melody");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Listener Pick"]);
  assert.equal(page.text("#matchblurb"), MELODY_CAPTION);
  assert.deepEqual(page.texts("#matches .why"), [
    " · melody #1",
    " · Listeners also play · samples #1",
  ]);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=melody`);
  assert.equal(page.similar().length, 1);
  assert.deepEqual(page.errors, []);
});

test("the tab chosen last is where the next visit opens", async () => {
  const first = loadAnysong();
  await waitFor(() => shown(first));
  assert.equal(first.kept.has(TAB_KEY), false);
  first.document.querySelector("#tab-melody").click();
  assert.equal(first.kept.get(TAB_KEY), "melody");

  const next = loadAnysong({ local: { [TAB_KEY]: "melody" } });
  await waitFor(() => shown(next));
  assert.equal(selected(next, "melody"), "true");
  assert.equal(next.hash, `#id=${AMANGA_ID}&tab=melody`);
  assert.deepEqual(next.texts("#matches .t"), ["Bar A Thym", "Listener Pick"]);

  // A share link's tab wins for that visit and isn't kept.
  const linked = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=listeners`, local: { [TAB_KEY]: "melody" } });
  await waitFor(() => shown(linked));
  assert.equal(selected(linked, "listeners"), "true");
  assert.equal(linked.kept.get(TAB_KEY), "melody");

  const unknown = loadAnysong({ local: { [TAB_KEY]: "vibe" } });
  await waitFor(() => shown(unknown));
  assert.equal(selected(unknown, "best"), "true");
});

test("Melody match says it ranks by audio alone with Audio only or no listener data", async () => {
  const audio = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=melody`,
    local: { [AUDIO_ONLY_KEY]: "true" },
    body: (options) => anysongBody({ ...options, scene: "off" }),
  });
  await waitFor(() => shown(audio));
  assert.equal(audio.similar()[0].get("scene"), "0");
  assert.equal(audio.text("#matchblurb"), AUDIO_MELODY_CAPTION);
  assert.deepEqual(audio.texts("#matches .t"), ["Bar A Thym", "Sunshine"]);
  assert.deepEqual(audio.texts("#matches .why"), [" · melody #1", " · samples #1"]);

  const unknown = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=melody`,
    body: (options) => anysongBody({ ...options, scene: "unknown_artist" }),
  });
  await waitFor(() => shown(unknown));
  assert.equal(unknown.text("#matchblurb"), AUDIO_MELODY_CAPTION);
});

test("Best matches read the main list, ask for one language and say other languages are left out", async () => {
  const page = loadAnysong();
  await waitFor(() => shown(page));

  assert.equal(
    page.requests[0],
    `/api/anysong/similar?id=${AMANGA_ID}&k=20&lang=same&policy=main-consensus-v4`,
  );
  assert.equal(page.text("#matchblurb"), BEST_CAPTION);
  const box = page.document.querySelector("#language");
  assert.equal(box.hidden, false);
  assert.equal(box.getAttribute("role"), "status");
  // The server's hidden count (2 here) is for its fused order of every channel,
  // not for the main list, so it is not shown.
  assert.equal(page.text("#language-status"), "English · other languages left out");
  const toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.hidden, false);
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(toggle.getAttribute("aria-pressed"), "false");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Sunshine"]);
  // Only the audio head's rows show the ranks that put them there.
  assert.deepEqual(page.texts("#matches .why"), [" · melody #1"]);
  assert.deepEqual(page.texts("#matches .basis"), []);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library"]);
  assert.deepEqual(page.errors, []);
});

test("the Best matches caption follows what placed the main list's rows", async () => {
  // A server whose main list follows sound and the listener ranker.
  const nudged = loadAnysong({ body: (options) => anysongBody({ ...options, body: "sound" }) });
  await waitFor(() => shown(nudged));
  assert.equal(nudged.text("#matchblurb"), NUDGED_BEST_CAPTION);

  // The model needs no listener data, so an unknown artist keeps its caption.
  const unknown = loadAnysong({
    body: (options) => anysongBody({ ...options, scene: "unknown_artist", body: "consensus" }),
  });
  await waitFor(() => shown(unknown));
  assert.equal(unknown.text("#matchblurb"), BEST_CAPTION);
  assert.deepEqual(unknown.texts("#matches .basis"), []);
  assert.deepEqual(unknown.texts("#matches .why"), [" · melody #1"]);
});

test("a consensus row's bar shows its match without calling it a sound match", async () => {
  const page = loadAnysong({
    body: (options) => {
      const body = anysongBody(options);
      body.results.main = body.results.main.map((item, index) => ({ ...item, overall: [93, 90][index] }));
      return body;
    },
  });
  await waitFor(() => shown(page));

  const bars = page.document.querySelectorAll("#matches .bar");
  assert.match(bars[0].getAttribute("aria-label"), /Sound match 41%/);
  assert.equal(
    bars[1].getAttribute("aria-label"),
    "Match 90%: how strongly Soundalike ranks it for this song",
  );
  assert.equal(bars[1].getAttribute("title"), bars[1].getAttribute("aria-label"));
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
    page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Ngiyakhuluma"]);

  toggle.click();
  await waitFor(() => page.similar().length === 3 &&
    page.text("#language-status") === "English · other languages left out");
  assert.equal(page.similar()[2].get("lang"), "same");
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(page.document.activeElement, toggle);
  assert.equal(page.document.querySelector(".seed h2").textContent, "Amanga");
});

test("bars show the overall match with the sound match in the tooltip, and a guessed language says so", async () => {
  const page = loadAnysong({
    body: ({ lang }) => {
      const body = anysongBody({ lang });
      body.query.language_guessed = true;
      body.results.main = body.results.main.map((item, index) => ({ ...item, overall: [93, 101][index] }));
      return body;
    },
  });
  await waitFor(() => shown(page));

  assert.equal(page.text("#language-status"), "English (best guess) · other languages left out");
  const bars = page.document.querySelectorAll("#matches .bar");
  assert.equal(
    bars[0].getAttribute("aria-label"),
    "Match 93%: how strongly Soundalike ranks it for this song. " +
      "Sound match 41%: sounds more alike than 41% of typical matches",
  );
  assert.equal(bars[0].getAttribute("title"), bars[0].getAttribute("aria-label"));
  assert.equal(bars[0].getAttribute("role"), "img");
  // An overall match out of range keeps the bar relative to the list's scores.
  assert.equal(bars[1].getAttribute("aria-label"), null);
  assert.equal(bars[1].getAttribute("aria-hidden"), "true");
  assert.deepEqual(
    page.document.querySelectorAll("#matches .bar i").map((bar) => bar.getAttribute("style")),
    ["width:93%", "width:25%"],
  );
  assert.deepEqual(page.errors, []);
});

test("an uncertain language shows all languages without a switch", async () => {
  const page = loadAnysong({
    body: ({ lang }) => anysongBody({ lang, language: "unknown", hidden: 0 }),
  });
  await waitFor(() => shown(page));

  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  page.document.querySelector("#tab-listeners").click();
  assert.deepEqual(page.texts("#matches .t"), ["Deep Vibe", "Kitchen Floor"]);
  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
});

test("a main-only language guess labels Best matches and leaves Listeners also play uncertain", async () => {
  const page = loadAnysong({
    body: ({ lang }) => {
      const body = anysongBody({ lang, language: "unknown", hidden: 0 });
      body.query.main_language_guess = "en";
      return body;
    },
  });
  await waitFor(() => shown(page));

  assert.equal(page.text("#language-status"), "English (best guess) · other languages left out");
  assert.equal(page.document.querySelector("#language-toggle").hidden, false);
  page.document.querySelector("#tab-listeners").click();
  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  page.document.querySelector("#tab-best").click();
  page.document.querySelector("#language-toggle").click();
  await waitFor(() => page.text("#language-status") === "English (best guess) · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.deepEqual(page.errors, []);
});

test("Listeners also play opens from a share link and arrow keys switch tabs", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=listeners` });
  await waitFor(() => shown(page));

  assert.deepEqual(page.texts("#out .tab"), TAB_LABELS);
  assert.equal(selected(page, "listeners"), "true");
  assert.equal(selected(page, "best"), "false");
  assert.equal(page.text("#matchblurb"), LISTENERS_CAPTION);
  assert.deepEqual(page.texts("#matches .t"), ["Deep Vibe", "Kitchen Floor"]);
  assert.deepEqual(page.texts("#matches .why"), []);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library"]);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.text("#language-status"), "English · other languages left out");
  const toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.hidden, false);
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(toggle.getAttribute("aria-pressed"), "false");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=listeners`);
  assert.equal(page.similar().length, 1);
  assert.deepEqual(page.errors, []);

  page.document.querySelector("#tab-best").click();
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  assert.equal(page.text("#matchblurb"), BEST_CAPTION);
  page.document.querySelector("#tab-best").onkeydown({ key: "ArrowRight" });
  assert.equal(selected(page, "listeners"), "true");
  assert.equal(page.document.activeElement, page.document.querySelector("#tab-listeners"));
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=listeners`);
  page.document.querySelector("#tab-listeners").onkeydown({ key: "ArrowRight", preventDefault() {} });
  assert.equal(selected(page, "melody"), "true");
  page.document.querySelector("#tab-melody").onkeydown({ key: "ArrowRight", preventDefault() {} });
  assert.equal(selected(page, "best"), "true");
  page.document.querySelector("#tab-best").onkeydown({ key: "ArrowLeft", preventDefault() {} });
  assert.equal(selected(page, "melody"), "true");
});

test("the language switch on Listeners also play is shared with Best matches", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=listeners` });
  await waitFor(() => shown(page));
  const toggle = page.document.querySelector("#language-toggle");
  toggle.focus();
  toggle.click();

  await waitFor(() => page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(toggle.textContent, "Show same language");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.equal(selected(page, "listeners"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Deep Vibe", "Wena"]);

  page.document.querySelector("#tab-best").click();
  assert.equal(page.text("#language-status"), "English · all languages");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Ngiyakhuluma"]);
  assert.equal(page.similar().length, 2);
});

test("Best matches fall back to the older Best list from a server without main", async () => {
  const page = loadAnysong({ body: (options) => anysongBody({ ...options, older: true }) });
  await waitFor(() => shown(page));

  assert.deepEqual(page.texts("#out .tab"), TAB_LABELS);
  assert.equal(selected(page, "best"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Listener Pick"]);
  assert.equal(page.text("#matchblurb"), MIXED_BEST_CAPTION);
  // That list mixes in listener picks, so they are marked.
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
  assert.equal(page.text("#language-status"), "English · other languages left out");
  // Such a server sends no listeners list either.
  page.document.querySelector("#tab-listeners").click();
  assert.equal(page.text("#matches .state"), NO_LISTENERS_NOTE);

  page.document.querySelector("#tab-best").click();
  page.document.querySelector("#audio-only").click();
  await waitFor(() => page.similar().length === 2 && shown(page));
  assert.equal(page.text("#matchblurb"), AUDIO_BEST_CAPTION);
  assert.deepEqual(page.texts("#matches .basis"), []);
  assert.deepEqual(page.errors, []);

  const emptyMain = loadAnysong({
    body: (options) => {
      const body = anysongBody(options);
      body.results.main = [];
      return body;
    },
  });
  await waitFor(() => shown(emptyMain));
  assert.deepEqual(emptyMain.texts("#matches .t"), ["Bar A Thym", "Listener Pick"]);
  assert.equal(emptyMain.text("#matchblurb"), MIXED_BEST_CAPTION);

  const neither = loadAnysong({
    body: (options) => {
      const body = anysongBody({ ...options, older: true });
      delete body.results.best;
      return body;
    },
  });
  await waitFor(() => neither.document.querySelector("#matches .state"));
  assert.equal(selected(neither, "best"), "true");
  assert.equal(neither.text("#matches .state"), "No matches in this list.");
  assert.deepEqual(neither.errors, []);
});

test("an artist without listener data says so on Listeners also play", async () => {
  const page = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=listeners`,
    body: (options) => anysongBody({ ...options, scene: "unknown_artist" }),
  });
  await waitFor(() => page.document.querySelector("#matches .state"));

  assert.deepEqual(page.texts("#out .tab"), TAB_LABELS);
  assert.equal(selected(page, "listeners"), "true");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=listeners`);
  assert.equal(page.text("#matches .state"), NO_LISTENERS_NOTE);
  // An empty list has no ranking to describe, so only its note shows.
  assert.equal(page.text("#matchblurb"), "");
  assert.equal(page.document.querySelector("#matchblurb").hidden, true);
  // Only Audio only is left to change on an empty list.
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.text("#language-status"), "");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  assert.equal(page.document.querySelector("#audio-only").getAttribute("aria-pressed"), "false");
  assert.equal(page.document.querySelector("#copy").disabled, true);
  assert.equal(vm.runInContext("lastResults.length", page.context), 0);

  // Best matches are in sound order, so they get the audio-only caption.
  page.document.querySelector("#tab-best").click();
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Sunshine"]);
  assert.equal(page.text("#matchblurb"), AUDIO_BEST_CAPTION);
  assert.equal(page.document.querySelector("#matchblurb").hidden, false);
  assert.equal(page.document.querySelector("#language-toggle").hidden, false);
  assert.equal(page.text("#language-status"), "English · other languages left out");
  assert.equal(page.similar().length, 1);
  assert.deepEqual(page.errors, []);

  const emptyList = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=listeners`,
    body: (options) => {
      const body = anysongBody(options);
      body.results.listeners = [];
      return body;
    },
  });
  await waitFor(() => emptyList.document.querySelector("#matches .state"));
  assert.equal(emptyList.text("#matches .state"), NO_LISTENERS_NOTE);
  assert.equal(emptyList.document.querySelector("#matchblurb").hidden, true);
  assert.deepEqual(emptyList.errors, []);
});

test("Audio only reloads without the listeners and is remembered", async () => {
  const page = loadMain({ hash: `#id=${EIGHT_PM_ID}` });
  await waitFor(() => shown(page));
  const audio = page.document.querySelector("#audio-only");
  assert.equal(audio.textContent, "Audio only");
  assert.equal(
    audio.getAttribute("title"),
    "Rank by audio alone, leaving out what this song's listeners also play",
  );
  assert.equal(audio.getAttribute("aria-pressed"), "false");
  assert.equal(page.similar()[0].get("scene"), null);
  assert.equal(page.text("#matchblurb"), BEST_CAPTION);

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
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "AHHH HA"]);

  // Listeners also play comes from the listeners, so it says how to get it back.
  page.document.querySelector("#tab-listeners").click();
  assert.equal(page.text("#matches .state"), AUDIO_ONLY_LISTENERS_NOTE);
  assert.equal(page.text("#matchblurb"), "");
  assert.equal(page.document.querySelector("#matchblurb").hidden, true);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.text("#language-status"), "");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  assert.equal(audio.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.querySelector("#copy").disabled, true);

  page.document.querySelector("#tab-best").click();
  page.document.querySelector("#language-toggle").click();
  await waitFor(() =>
    page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[2].get("lang"), "all");
  assert.equal(page.similar()[2].get("scene"), "0");

  page.document.querySelector("#tab-listeners").click();
  audio.click();
  await waitFor(() => page.similar().length === 4 && shown(page));
  assert.equal(page.similar()[3].get("scene"), null);
  assert.equal(page.similar()[3].get("lang"), "all");
  assert.equal(audio.getAttribute("aria-pressed"), "false");
  assert.equal(page.kept.has(AUDIO_ONLY_KEY), false);
  assert.equal(selected(page, "listeners"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Kept Going", "Autre Son"]);
  assert.equal(page.text("#language-status"), "English · all languages");
  assert.equal(page.document.querySelector("#language-toggle").hidden, false);
  assert.equal(page.text("#matchblurb"), LISTENERS_CAPTION);
  assert.equal(page.document.querySelector("#matchblurb").hidden, false);
  assert.equal(page.document.querySelector("#copy").disabled, false);
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

// A server that can match a part of the preview says so in every answer
// (query.parts) and names the part an answer is for (query.part); its lists
// then differ from the whole preview's.
const PARTS_INFO = { min_s: 10, max_s: 30, preview_s: 30, song_offset_s: 48 };
const PART_INTRO = "Soundalike hears a 30-second preview. Pick the part of it you care about.";

function partsBest({ parts = PARTS_INFO, refuse = false } = {}) {
  return (params) => {
    const part = params.get("part");
    if (part && refuse) {
      return json(404, { ok: false, error: "choosing a part of the preview is not enabled" });
    }
    const body = anysongBody({
      lang: params.get("lang"),
      scene: params.get("scene") === "0" ? "off" : "on",
    });
    if (parts) body.query.parts = parts;
    if (part && parts) {
      const [start, end] = part.split("-").map(Number);
      body.query.part = {
        start, end, heard: { sonic: [10, 30], samples: [15.31, 30.63], melody: [10, 30] },
      };
      body.results.main.reverse();
    }
    return json(200, body);
  };
}

const partParams = (page) => page.similar().map((params) => params.get("part"));
const key = (page, name, shiftKey = false) =>
  page.document.querySelector("#part-span").dispatch("keydown", { key: name, shiftKey });

test("Match a part stays hidden on a server that cannot match parts", async () => {
  // A link to a part opens the whole preview there, and the link is put right.
  const page = loadMain({ hash: `#id=${AMANGA_ID}&tab=best&part=10-25`, body: anysongBody });
  await waitFor(() => shown(page));
  assert.equal(page.document.querySelector("#part-open").hidden, true);
  assert.equal(page.document.querySelector("#part-chip").hidden, true);
  assert.equal(page.document.querySelector("#part-panel").hidden, true);
  assert.deepEqual(partParams(page), ["10-25"]);
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  assert.deepEqual(page.errors, []);
});

test("Match a part opens a picker by the seed's preview button and Apply reloads the lists", async () => {
  const page = loadMain({ hash: `#id=${AMANGA_ID}`, best: partsBest() });
  await waitFor(() => shown(page));
  const open = page.document.querySelector("#part-open");
  assert.equal(open.hidden, false);
  assert.equal(open.textContent, "Match a part beta");
  assert.equal(open.getAttribute("aria-expanded"), "false");
  // It follows the seed's preview button in the seed card's actions.
  assert.deepEqual(
    page.document.querySelectorAll(".seed .actions button").map((button) => button.id || button.className),
    ["pv", "part-open", "share", "copy", "save"],
  );
  assert.equal(page.document.querySelector("#part-chip").hidden, true);

  open.click();
  const panel = page.document.querySelector("#part-panel");
  assert.equal(panel.hidden, false);
  assert.equal(open.getAttribute("aria-expanded"), "true");
  assert.equal(page.text("#part-panel .part-intro"), `New · beta ${PART_INTRO}`);
  const span = page.document.querySelector("#part-span");
  assert.equal(span.getAttribute("role"), "slider");
  assert.equal(span.getAttribute("tabindex"), "0");
  assert.equal(span.getAttribute("aria-valuetext"), "0:58–1:08 of the song · 10 s");
  assert.deepEqual(page.texts("#part-panel .part-ticks span"), ["0:48", "0:58", "1:08", "1:18"]);
  assert.deepEqual(page.texts("#part-panel .part-lengths button"), ["10 s", "15 s", "20 s"]);
  assert.match(page.text("#part-panel .part-note"), /^All three tabs follow the part\. /);
  assert.match(page.text("#part-panel .part-note"), /in a few songs it is elsewhere/);

  // Arrow keys move the span in 5-second steps; Shift changes its length.
  key(page, "ArrowRight");
  assert.equal(page.text("#part-label"), "1:03–1:13 of the song · 10 s");
  assert.equal(span.style.left, "50%");
  key(page, "ArrowRight", true);
  assert.equal(page.text("#part-label"), "1:03–1:18 of the song · 15 s");
  // At the end of the preview a longer part grows backwards, up to 20 s.
  key(page, "ArrowRight", true);
  key(page, "ArrowRight", true);
  assert.equal(page.text("#part-label"), "0:58–1:18 of the song · 20 s");
  key(page, "Home");
  assert.equal(page.text("#part-label"), "0:48–1:08 of the song · 20 s");
  // A length button keeps the start.
  page.document.querySelectorAll("#part-panel .part-lengths button")[1].click();
  assert.equal(page.text("#part-label"), "0:48–1:03 of the song · 15 s");
  assert.deepEqual(
    page.document.querySelectorAll("#part-panel .part-lengths button")
      .map((button) => button.getAttribute("aria-pressed")),
    ["false", "true", "false"],
  );
  // Dragging the span moves it (touch and mouse are both pointers).
  const track = page.document.querySelector("#part-track");
  track.dispatch("pointerdown", { clientX: 50, pointerId: 1, target: span });
  track.dispatch("pointermove", { clientX: 152 });
  track.dispatch("pointerup");
  assert.equal(page.text("#part-label"), "0:58–1:13 of the song · 15 s");
  // Dragging an edge resizes it, never under 10 s.
  const edge = page.document.querySelectorAll("#part-span .part-edge")[1];
  track.dispatch("pointerdown", { clientX: 250, pointerId: 1, target: edge });
  track.dispatch("pointermove", { clientX: 120 });
  track.dispatch("pointerup");
  assert.equal(page.text("#part-label"), "0:58–1:08 of the song · 10 s");
  // A press outside the span brings its middle there.
  track.dispatch("pointerdown", { clientX: 290, pointerId: 1, target: track });
  track.dispatch("pointerup");
  assert.equal(page.text("#part-label"), "1:08–1:18 of the song · 10 s");
  assert.deepEqual(partParams(page), [null]);

  const before = shown(page).dataset.id;
  page.document.querySelector("#part-apply").click();
  await waitFor(() => page.similar().length === 2 && shown(page) && shown(page).dataset.id !== before);
  assert.deepEqual(partParams(page), [null, "20-30"]);
  // The part is in the share link, and the picker gives way to a chip.
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best&part=20-30`);
  assert.equal(panel.hidden, true);
  const chip = page.document.querySelector("#part-chip");
  assert.equal(chip.hidden, false);
  assert.equal(chip.textContent, "Matching 1:08–1:18 of the song · whole preview");
  assert.equal(
    page.document.querySelector("#part-chip-text").getAttribute("title"),
    "Heard: sound 0:58–1:18, melody 0:58–1:18, samples 1:03–1:19",
  );

  // The other tabs read the same answer; switching keeps the part in the link.
  page.document.querySelector("#tab-listeners").click();
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=listeners&part=20-30`);
  assert.equal(chip.hidden, false);
  assert.equal(page.similar().length, 2);
  // So do the language and Audio only switches.
  page.document.querySelector("#language-toggle").click();
  await waitFor(() => page.similar().length === 3);
  assert.equal(page.similar()[2].get("part"), "20-30");
  assert.equal(page.similar()[2].get("lang"), "all");

  // "whole preview" goes back.
  await waitFor(() => page.document.querySelector("#part-clear"));
  page.document.querySelector("#part-clear").click();
  await waitFor(() => page.similar().length === 4 && chip.hidden);
  assert.equal(page.similar()[3].get("part"), null);
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=listeners`);
  assert.equal(open.hidden, false);
  assert.deepEqual(page.errors, []);
});

test("a link with a part opens that part, from the hash or an old /anysong query", async () => {
  const page = loadMain({ hash: `#id=${AMANGA_ID}&tab=best&part=5-20`, best: partsBest() });
  await waitFor(() => shown(page));
  assert.deepEqual(partParams(page), ["5-20"]);
  assert.equal(page.text("#part-chip"), "Matching 0:53–1:08 of the song · whole preview");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best&part=5-20`);
  // The picker opens on the part being matched, with nothing to apply yet.
  page.document.querySelector("#part-open").click();
  assert.equal(page.text("#part-label"), "0:53–1:08 of the song · 15 s");
  assert.equal(page.document.querySelector("#part-apply").disabled, true);
  page.document.querySelector("#part-cancel").click();
  assert.equal(page.document.querySelector("#part-panel").hidden, true);
  // Picking another song starts from its whole preview.
  page.document.querySelector("#matches .t").click();
  await waitFor(() => page.similar().length === 2);
  assert.equal(page.similar()[1].get("part"), null);

  const forwarded = loadMain({ search: `?id=${AMANGA_ID}&tab=melody&part=10-20`, best: partsBest() });
  await waitFor(() => shown(forwarded));
  assert.equal(forwarded.search, "");
  assert.equal(forwarded.hash, `#id=${AMANGA_ID}&tab=melody&part=10-20`);
  assert.deepEqual(partParams(forwarded), ["10-20"]);

  // Spans the picker cannot make are not sent.
  for (const bad of ["12", "20-10", "5-12", "10-40", "a-b"]) {
    const other = loadMain({ hash: `#id=${AMANGA_ID}&part=${bad}`, best: partsBest() });
    await waitFor(() => shown(other));
    assert.deepEqual(partParams(other), [null]);
  }
});

test("a server that refuses the part answers for the whole preview and no chip stays", async () => {
  const page = loadMain({ hash: `#id=${AMANGA_ID}`, best: partsBest({ refuse: true }) });
  await waitFor(() => shown(page));
  page.document.querySelector("#part-open").click();
  page.document.querySelector("#part-apply").click();
  await waitFor(() => page.similar().length === 3 && shown(page));
  assert.deepEqual(partParams(page), [null, "10-20", null]);
  assert.equal(page.document.querySelector("#part-chip").hidden, true);
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  assert.deepEqual(page.errors, []);
});

test("a part is given in preview time when the song's is unknown, and plays just that span", async () => {
  const page = loadMain({
    hash: `#id=${AMANGA_ID}`,
    best: partsBest({ parts: { ...PARTS_INFO, song_offset_s: null } }),
  });
  await waitFor(() => shown(page));
  page.document.querySelector("#part-open").click();
  assert.equal(page.text("#part-label"), "0:10–0:20 of the preview · 10 s");
  assert.deepEqual(page.texts("#part-panel .part-ticks span"), ["0:00", "0:10", "0:20", "0:30"]);
  assert.doesNotMatch(page.text("#part-panel .part-note"), /in a few songs it is elsewhere/);

  const [audio] = page.audios;
  const play = page.document.querySelector("#part-play");
  assert.equal(play.textContent, "▶ Play this part");
  play.click();
  await waitFor(() => !audio.paused);
  assert.equal(audio.src, PREVIEW);
  assert.equal(audio.currentTime, 10);
  assert.equal(play.textContent, "⏸ Stop");
  audio.currentTime = 14;
  audio.dispatch("timeupdate");
  assert.equal(audio.paused, false);
  assert.equal(page.document.querySelector("#part-head").style.left, `${(100 * 14) / 30}%`);
  // It stops at the end of the span, not of the preview.
  audio.currentTime = 20.1;
  audio.dispatch("timeupdate");
  assert.equal(audio.paused, true);
  assert.equal(play.textContent, "▶ Play this part");
  // The seed's own preview button still plays the whole preview.
  play.click();
  await waitFor(() => !audio.paused);
  page.document.querySelector(".seed .pv").click();
  assert.equal(audio.paused, true);
  assert.deepEqual(page.errors, []);
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
  page.document.querySelector("#tab-listeners").click();
  page.document.querySelector("#copy").click();
  await waitFor(() => copied === "Deep Vibe — Black Coffee\nKitchen Floor — Moodymann");

  page.document.querySelector("#share").click();
  await waitFor(() => page.text("#share") === "✓ Link copied");
  assert.equal(copied, `${ORIGIN}/#id=${AMANGA_ID}&tab=listeners`);
  assert.deepEqual(
    page.document.querySelectorAll("#matches a.play").map((link) => link.getAttribute("href")),
    [
      "https://open.spotify.com/search/Deep%20Vibe%20Black%20Coffee",
      "https://open.spotify.com/search/Kitchen%20Floor%20Moodymann",
    ],
  );
  const save = page.document.querySelector("#save");
  assert.equal(save.disabled, false);
  assert.equal(save.onclick, page.context.savePlaylist);
  assert.equal(vm.runInContext("lastSeed.title", page.context), "Amanga");
  assert.equal(vm.runInContext("lastResults.length", page.context), 2);
  assert.equal(vm.runInContext("lastResults[0].title", page.context), "Deep Vibe");

  page.document.querySelector("#tab-best").click();
  assert.equal(vm.runInContext("lastResults[0].title", page.context), "Bar A Thym");
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
  // A login started on an old five-tab link comes back to Best matches.
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
  assert.equal(selected(page, "best"), "true");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=best`);
  assert.deepEqual(page.texts("#matches .t"), ["Bar A Thym", "Sunshine"]);
  await waitFor(() => page.text("#sp") === "✓ Spotify connected");
  assert.equal(page.stored.get("sa_token"), "spotify-token");
  assert.equal(page.stored.has("sa_return"), false);
  assert.ok(page.requests.includes("https://accounts.spotify.com/api/token"));
  assert.deepEqual(page.errors, []);

  const listeners = loadMain({
    search: "?code=auth-code&state=expected",
    storage: {
      sa_state: "expected",
      sa_verifier: "verifier",
      sa_return: `#id=${AMANGA_ID}&tab=listeners`,
    },
    body: anysongBody,
  });
  await waitFor(() => shown(listeners));
  assert.equal(selected(listeners, "listeners"), "true");
  assert.equal(listeners.hash, `#id=${AMANGA_ID}&tab=listeners`);
});
