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
  dispatch(type) {
    const event = { type, target: this, currentTarget: this, preventDefault() {} };
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

function loadPage(html, scripts, { path, hash = "", routes }) {
  const body = html.match(/<body>([\s\S]*)<\/body>/)[1]
    .replace(/<script[\s\S]*?<\/script>/g, "");
  const document = new FakeDocument(body);
  const windowEvents = new FakeEvents();
  const requests = [];
  const errors = [];
  let currentHash = hash;
  const context = {
    AbortController,
    Audio: FakeAudio,
    URL,
    URLSearchParams,
    clearTimeout,
    console: { log() {}, warn() {}, error: (...args) => errors.push(args) },
    document,
    navigator: {},
    sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    setTimeout,
    scrollTo() {},
    addEventListener: (type, listener) => windowEvents.addEventListener(type, listener),
    removeEventListener: (type, listener) =>
      windowEvents.removeEventListener(type, listener),
    history: {
      replaceState(_state, _title, url) {
        currentHash = new URL(url, `${ORIGIN}${path}`).hash;
      },
    },
    location: {
      origin: ORIGIN,
      pathname: path,
      search: "",
      get href() {
        return `${ORIGIN}${path}${currentHash}`;
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

function loadAnysong({ hash = `#id=${AMANGA_ID}`, body = anysongBody } = {}) {
  return loadPage(anysongHtml, [inlineScript(anysongHtml)], {
    path: "/anysong",
    hash,
    routes: (url) => {
      if (url.pathname === "/api/anysong/similar") {
        return json(200, body({ lang: url.searchParams.get("lang") }));
      }
      if (url.pathname === "/api/preview") return json(200, { ok: true, preview: "", cover: "" });
      return json(404, { ok: false });
    },
  });
}

const BEST_CAPTION =
  "Likely covers, samples and shared melodies first, then similar-sounding songs " +
  "by artists this song's listeners also play.";

test("any-song Best matches ask for one language and say what was hidden", async () => {
  const page = loadAnysong();
  await waitFor(() => page.document.querySelector("#list li.row"));

  assert.equal(page.requests[0], `/api/anysong/similar?id=${AMANGA_ID}&k=20&lang=same`);
  assert.equal(page.text("#blurb"), BEST_CAPTION);
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
  assert.deepEqual(page.texts("#list .t"), ["Bar A Thym", "Sunshine"]);
  assert.deepEqual(page.texts("#list .why"), [" · melody #1", " · samples #1"]);
  assert.deepEqual(page.errors, []);
});

test("any-song language switch reloads Best matches and keeps focus on it", async () => {
  const page = loadAnysong();
  await waitFor(() => page.document.querySelector("#list li.row"));

  page.document.querySelector("#language-toggle").click();
  await waitFor(() =>
    page.text("#language-status") === "English · 2 other-language matches in the top 20");
  assert.equal(page.similar()[1].get("lang"), "all");
  let toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.textContent, "Show same language");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.deepEqual(page.texts("#list .t"), ["Bar A Thym", "Ngiyakhuluma"]);

  toggle.click();
  await waitFor(() => page.similar().length === 3 &&
    page.text("#language-status")?.endsWith("hidden from the top 20"));
  assert.equal(page.similar()[2].get("lang"), "same");
  toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(page.document.activeElement, toggle);
});

test("any-song channel tabs and Sounds like stay unfiltered", async () => {
  const page = loadAnysong();
  await waitFor(() => page.document.querySelector("#list li.row"));

  page.document.querySelector("#tab-melody").click();
  assert.equal(page.document.querySelector("#language").hidden, true);
  assert.deepEqual(page.texts("#list .t"), ["Melody Match"]);

  page.document.querySelector("#tab-sonic").click();
  assert.equal(page.document.querySelector("#language").hidden, true);
  assert.equal(
    page.text("#blurb"),
    "A similar overall sound: production, instruments and mood.",
  );
  assert.deepEqual(page.texts("#list .t"), ["Gumbaya Chords"]);
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=sonic`);

  page.document.querySelector("#tab-best").click();
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.similar().length, 1);
});

test("any-song share links to Sounds like still open the sound channel", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=sonic` });
  await waitFor(() => page.document.querySelector("#list li.row"));

  assert.equal(page.document.querySelector("#tab-sonic").getAttribute("aria-selected"), "true");
  assert.deepEqual(page.texts("#list .t"), ["Gumbaya Chords"]);
  assert.equal(page.document.querySelector("#language").hidden, true);
});

test("any-song shows all languages without a switch for an uncertain seed", async () => {
  const page = loadAnysong({
    body: ({ lang }) => anysongBody({ lang, language: "unknown", hidden: 0 }),
  });
  await waitFor(() => page.document.querySelector("#list li.row"));

  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
});

const VIBE_CAPTION =
  "Songs from these matches that share this song's tempo and overall sound, one per artist.";

test("any-song Same vibe sits next to Best matches and opens from a share link", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=vibe` });
  await waitFor(() => page.document.querySelector("#list li.row"));

  assert.deepEqual(page.texts(".tab"),
    ["Best matches", "Same vibe", "Same melody", "Samples & loops", "Sounds like"]);
  assert.equal(page.document.querySelector("#tab-vibe").getAttribute("aria-selected"), "true");
  assert.equal(page.document.querySelector("#tab-best").getAttribute("aria-selected"), "false");
  assert.equal(page.text("#blurb"), VIBE_CAPTION);
  assert.deepEqual(page.texts("#list .t"), ["Deep Vibe", "Kitchen Floor"]);
  assert.deepEqual(page.texts("#list .why"), []);
  assert.deepEqual(page.texts(".seed .tag"), ["in our library", "122 BPM"]);
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
  assert.equal(page.text("#blurb"), BEST_CAPTION);
  page.document.querySelector("#tab-best").onkeydown({ key: "ArrowRight" });
  assert.equal(page.document.querySelector("#tab-vibe").getAttribute("aria-selected"), "true");
  assert.equal(page.hash, `#id=${AMANGA_ID}&tab=vibe`);
});

test("any-song language switch on Same vibe is shared with Best matches", async () => {
  const page = loadAnysong({ hash: `#id=${AMANGA_ID}&tab=vibe` });
  await waitFor(() => page.document.querySelector("#list li.row"));

  page.document.querySelector("#language-toggle").click();
  await waitFor(() => page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  const toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.textContent, "Show same language");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.equal(page.document.querySelector("#tab-vibe").getAttribute("aria-selected"), "true");
  assert.deepEqual(page.texts("#list .t"), ["Deep Vibe", "Wena"]);

  page.document.querySelector("#tab-best").click();
  assert.equal(
    page.text("#language-status"),
    "English · 2 other-language matches in the top 20",
  );
  assert.equal(page.document.querySelector("#language-toggle").getAttribute("aria-pressed"), "true");
  assert.deepEqual(page.texts("#list .t"), ["Bar A Thym", "Ngiyakhuluma"]);
  assert.equal(page.similar().length, 2);
});

test("any-song hides Same vibe and the tempo when the server has neither", async () => {
  const page = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=vibe`,
    body: ({ lang }) => anysongBody({ lang, vibe: false }),
  });
  await waitFor(() => page.document.querySelector("#list li.row"));

  assert.ok(!page.document.querySelector("#tab-vibe"), "Same vibe tab is left out");
  assert.deepEqual(page.texts(".tab"),
    ["Best matches", "Same melody", "Samples & loops", "Sounds like"]);
  assert.equal(page.document.querySelector("#tab-best").getAttribute("aria-selected"), "true");
  assert.equal(page.text("#blurb"), BEST_CAPTION);
  assert.deepEqual(page.texts(".seed .tag"), ["in our library"]);
  assert.equal(
    page.text("#language-status"),
    "English · 2 other-language matches hidden from the top 20",
  );
  page.document.querySelector("#tab-best").onkeydown({ key: "ArrowRight" });
  assert.equal(page.document.querySelector("#tab-melody").getAttribute("aria-selected"), "true");
  assert.deepEqual(page.errors, []);
});

test("any-song Same vibe shows all languages without a switch for an uncertain seed", async () => {
  const page = loadAnysong({
    hash: `#id=${AMANGA_ID}&tab=vibe`,
    body: ({ lang }) => anysongBody({ lang, language: "unknown", hidden: 0, bpm: null }),
  });
  await waitFor(() => page.document.querySelector("#list li.row"));

  assert.equal(page.document.querySelector("#tab-vibe").getAttribute("aria-selected"), "true");
  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
  assert.deepEqual(page.texts(".seed .tag"), ["in our library"]);
});

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

const libraryBody = {
  ok: true,
  seed: { title: "8PM", artist: "Nemzzz" },
  vibe: { low_end: "deep low end", dynamics: "punchy", tone: "dark" },
  results: [{
    deezer_id: 61,
    title: "Library Match",
    artist: "Library Artist",
    neural_sim: 0.9,
    spotify_url: "https://open.spotify.com/track/library",
  }],
  library_size: 272853,
};

function loadMain({ best = (params) => json(200, mainBestBody({ lang: params.get("lang") })) } = {}) {
  return loadPage(mainHtml, [inlineScript(mainHtml), searchJs], {
    path: "/",
    routes: (url) => {
      if (url.pathname === "/api/anysong/similar") return best(url.searchParams);
      if (url.pathname === "/api/spicetify_recommend") return json(200, libraryBody);
      if (url.pathname === "/api/stats") return json(200, { ok: true, library_size: 272853 });
      if (url.pathname === "/api/search") return json(200, { results: [] });
      return json(404, { ok: false });
    },
  });
}

const PICK = { title: "8PM", artist: "Nemzzz", row: null };

test("main page Best matches ask for one language and offer the switch", async () => {
  const page = loadMain();
  await page.context.pick(PICK);

  const [params] = page.similar();
  assert.equal(params.get("title"), "8PM");
  assert.equal(params.get("artist"), "Nemzzz");
  assert.equal(params.get("lang"), "same");
  assert.equal(params.get("policy"), "best-vibe-sounds-v2");
  assert.equal(page.document.querySelector("#tab-best").getAttribute("aria-selected"), "true");
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(
    page.text("#language-status"),
    "English · 1 other-language match hidden from the top 20",
  );
  const toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(toggle.getAttribute("aria-pressed"), "false");
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "AHHH HA"]);
  assert.deepEqual(page.errors, []);
});

test("main page language switch reloads only Best matches", async () => {
  const page = loadMain();
  await page.context.pick(PICK);
  const toggle = page.document.querySelector("#language-toggle");
  toggle.focus();
  toggle.click();

  assert.equal(toggle.textContent, "Show same language");
  assert.match(page.text("#matches"), /matching…/);
  await waitFor(() =>
    page.text("#language-status") === "English · 1 other-language match in the top 20");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "Airforce Negras"]);
  assert.equal(
    page.requests.filter((url) => url.includes("/api/spicetify_recommend")).length,
    1,
  );
});

test("main page Sounds like uses the library list when the response has none", async () => {
  const page = loadMain();
  await page.context.pick(PICK);

  page.document.querySelector("#tab-sounds").click();
  await waitFor(() => page.texts("#matches .t").includes("Library Match"));
  assert.equal(page.document.querySelector("#language").hidden, true);
  assert.equal(
    page.text("#matchblurb"),
    "Songs with the same overall sound: low end, dynamics and tone.",
  );
  page.document.querySelector("#tab-best").click();
  assert.equal(page.document.querySelector("#language").hidden, false);
});

test("main page falls back to Sounds like without the switch when Best fails", async () => {
  const page = loadMain({ best: () => json(404, { ok: false, error: "not found" }) });
  await page.context.pick(PICK);

  assert.equal(page.document.querySelector("#tab-sounds").getAttribute("aria-selected"), "true");
  assert.equal(page.document.querySelector("#tab-vibe").hidden, true);
  assert.equal(page.document.querySelector("#language").hidden, true);
  assert.deepEqual(page.texts("#matches .t"), ["Library Match"]);
});

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

test("main page Sounds like comes with Best matches and shares the switch", async () => {
  const page = loadMain({
    best: (params) => json(200, mainSoundsBody({ lang: params.get("lang") })),
  });
  await page.context.pick(PICK);
  const libraryAsked = () =>
    page.requests.filter((url) => url.includes("/api/spicetify_recommend")).length;
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(libraryAsked(), 0);

  page.document.querySelector("#tab-sounds").click();
  assert.deepEqual(page.texts("#matches .t"), ["Sounds Match", "Close Sound"]);
  assert.equal(
    page.text("#matchblurb"),
    "Songs with the same overall sound (instruments, production and mood), favouring artists this song's listeners also play.",
  );
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library", "140 BPM"]);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.text("#language-status"), "English · other languages left out");
  assert.equal(
    page.text("#foot"),
    "Sounds like comes from the same matcher as Best matches. ▶ = 30-second preview.",
  );

  page.document.querySelector("#language-toggle").click();
  await waitFor(() => page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(page.document.querySelector("#tab-sounds").getAttribute("aria-selected"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Sounds Match", "Autre Son"]);
  assert.equal(libraryAsked(), 0);
  assert.deepEqual(page.errors, []);
});

test("main page shows all languages without a switch for an uncertain seed", async () => {
  const page = loadMain({
    best: () => json(200, mainBestBody({ language: "unknown", hidden: 0 })),
  });
  await page.context.pick(PICK);

  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);

  page.document.querySelector("#tab-vibe").click();
  assert.equal(page.text("#language-status"), "Language uncertain; showing all languages.");
  assert.equal(page.document.querySelector("#language-toggle").hidden, true);
});

test("main page offers Same vibe next to Best matches from the same response", async () => {
  const page = loadMain();
  await page.context.pick(PICK);

  assert.deepEqual(page.texts("#out .tab"), ["Best matches", "Same vibe", "Sounds like"]);
  const vibeTab = page.document.querySelector("#tab-vibe");
  assert.equal(vibeTab.hidden, false);
  assert.equal(vibeTab.getAttribute("aria-selected"), "false");
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library"]);

  vibeTab.click();
  assert.equal(vibeTab.getAttribute("aria-selected"), "true");
  assert.equal(page.document.querySelector("#matches").getAttribute("aria-labelledby"), "tab-vibe");
  assert.equal(page.text("#matchblurb"), VIBE_CAPTION);
  assert.deepEqual(page.texts("#matches .t"), ["Kept Going", "Ain't It Different"]);
  assert.deepEqual(page.texts("#matches .why"), []);
  assert.deepEqual(page.texts("#seedtags .tag"), ["in our library", "140 BPM"]);
  assert.equal(page.document.querySelector("#language").hidden, false);
  assert.equal(page.text("#language-status"), "English · other languages left out");
  const toggle = page.document.querySelector("#language-toggle");
  assert.equal(toggle.hidden, false);
  assert.equal(toggle.textContent, "Show all languages");
  assert.equal(
    page.text("#foot"),
    "Same vibe comes from the same matcher as Best matches. ▶ = 30-second preview.",
  );
  assert.equal(page.similar().length, 1);
  assert.deepEqual(page.errors, []);

  vibeTab.onkeydown({ key: "ArrowRight", preventDefault() {} });
  assert.equal(page.document.querySelector("#tab-sounds").getAttribute("aria-selected"), "true");
  assert.equal(page.document.querySelector("#language").hidden, true);
  page.document.querySelector("#tab-sounds").onkeydown({ key: "ArrowLeft", preventDefault() {} });
  assert.equal(vibeTab.getAttribute("aria-selected"), "true");
});

test("main page language switch on Same vibe reloads the shared response", async () => {
  const page = loadMain();
  await page.context.pick(PICK);
  page.document.querySelector("#tab-vibe").click();
  const toggle = page.document.querySelector("#language-toggle");
  toggle.focus();
  toggle.click();

  assert.equal(toggle.textContent, "Show same language");
  assert.match(page.text("#matches"), /matching…/);
  await waitFor(() => page.text("#language-status") === "English · all languages");
  assert.equal(page.similar()[1].get("lang"), "all");
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.equal(page.document.activeElement, toggle);
  assert.equal(page.document.querySelector("#tab-vibe").getAttribute("aria-selected"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["Kept Going", "Booba Vibe"]);

  page.document.querySelector("#tab-best").click();
  assert.equal(
    page.text("#language-status"),
    "English · 1 other-language match in the top 20",
  );
  assert.equal(toggle.getAttribute("aria-pressed"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "Airforce Negras"]);
  assert.equal(page.similar().length, 2);
  assert.equal(
    page.requests.filter((url) => url.includes("/api/spicetify_recommend")).length,
    1,
  );
});

test("main page hides Same vibe when the server sends no vibe list", async () => {
  const page = loadMain({
    best: (params) => json(200, mainBestBody({ lang: params.get("lang"), vibe: false })),
  });
  await page.context.pick(PICK);

  assert.equal(page.document.querySelector("#tab-vibe").hidden, true);
  assert.equal(page.document.querySelector("#tab-best").getAttribute("aria-selected"), "true");
  assert.equal(
    page.text("#language-status"),
    "English · 1 other-language match hidden from the top 20",
  );
  page.document.querySelector("#tab-best").onkeydown({ key: "ArrowRight", preventDefault() {} });
  assert.equal(page.document.querySelector("#tab-sounds").getAttribute("aria-selected"), "true");
  page.document.querySelector("#tab-sounds").onkeydown({ key: "ArrowLeft", preventDefault() {} });
  assert.equal(page.document.querySelector("#tab-best").getAttribute("aria-selected"), "true");
  assert.deepEqual(page.errors, []);
});

test("main page returns to Best matches when a reload brings no Same vibe list", async () => {
  const page = loadMain({
    best: (params) => {
      const lang = params.get("lang");
      return json(200, mainBestBody({ lang, vibe: lang !== "all" }));
    },
  });
  await page.context.pick(PICK);
  page.document.querySelector("#tab-vibe").click();
  page.document.querySelector("#language-toggle").click();

  await waitFor(() => page.text("#language-status") === "English · 1 other-language match in the top 20");
  assert.equal(page.document.querySelector("#tab-vibe").hidden, true);
  assert.equal(page.document.querySelector("#tab-best").getAttribute("aria-selected"), "true");
  assert.deepEqual(page.texts("#matches .t"), ["n.h.i.e.", "Airforce Negras"]);
});
