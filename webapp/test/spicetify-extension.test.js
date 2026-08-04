import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const extensionSource = fs.readFileSync(
  new URL("../../integrations/spicetify/soundalike.js", import.meta.url),
  "utf8",
);

const recommendation = {
  ok: true,
  seed: { title: "Blinding Lights", artist: "The Weeknd" },
  vibe: { low_end: "balanced", dynamics: "moderate", tone: "neutral" },
  results: [],
};

function response(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return body;
    },
  };
}

function loadExtension(fetchImpl, options = {}) {
  let handler;
  let currentPage;
  const notifications = [];
  const history = [];
  const played = [];
  const graphqlRequests = [];
  const cosmosRequests = [];
  const historyListeners = [];
  const storage = options.storage || new Map();
  const rows = (options.results || []).map((_, index) => ({
    dataset: { index: String(index) },
    hidden: index >= 20,
    querySelector() {
      return null;
    },
  }));
  const languageStatus = { textContent: "" };
  function makeDiv() {
    return {
      className: "",
      innerHTML: "",
      style: {},
      children: [],
      parentElement: null,
      appendChild(value) {
        value.parentElement = this;
        this.children.push(value);
      },
      remove() {
        if (this.parentElement?.children) {
          this.parentElement.children = this.parentElement.children.filter(
            (child) => child !== this,
          );
        }
        if (this.className === "sa-results-host") currentPage = undefined;
      },
      querySelector(selector) {
        if (selector === ".sa-language-status") return languageStatus;
        const match = selector.match(/data-index="(\d+)"/);
        return match ? rows[Number(match[1])] : null;
      },
      querySelectorAll(selector) {
        return selector === ".sa-row" ? rows : [];
      },
    };
  }
  const nativeChild = { id: "spotify-owned-content" };
  let pageContainer;
  const body = {
    children: [],
    appendChild(value) {
      value.parentElement = this;
      this.children.push(value);
      if (value.className === "sa-results-host") {
        currentPage = value.children.find(
          (child) => child.className === "sa-results",
        );
      }
    },
  };
  const wrap = makeDiv();
  Object.assign(wrap, {
    querySelector(selector) {
      if (selector === ".sa-language-status") return languageStatus;
      const match = selector.match(/data-index="(\d+)"/);
      return match ? rows[Number(match[1])] : null;
    },
    querySelectorAll(selector) {
      return selector === ".sa-row" ? rows : [];
    },
  });
  const registry = {};
  const platform = {};
  const AppProvider = function AppProvider() {};
  const PlatformContextProvider = function PlatformContextProvider() {};
  const RegistryProvider = function RegistryProvider() {};
  const StoreProvider = function StoreProvider() {};
  const storeValue = {
    store: { getState() {} },
    subscription: {},
  };
  const nativeHost = {
    "__reactFiber$test": {
      tag: 10,
      memoizedProps: {
        value: { isDesktop: true, isWeb: false, ui: {} },
      },
      elementType: AppProvider,
      return: {
        tag: 10,
        memoizedProps: { value: platform },
        elementType: PlatformContextProvider,
        return: {
          tag: 10,
          memoizedProps: { value: registry },
          elementType: RegistryProvider,
          return: {
            tag: 10,
            memoizedProps: { value: storeValue },
            elementType: StoreProvider,
            return: null,
          },
        },
      },
    },
  };
  pageContainer = {
    parentElement: null,
    children: [nativeChild],
    getBoundingClientRect() {
      return { left: 72, top: 64, width: 1200, height: 720 };
    },
    replaceChildren() {
      throw new Error("Soundalike must not replace Spotify-owned children");
    },
    ...(options.nativeMenus ? nativeHost : {}),
  };
  const RightClickMenu = function RightClickMenu() {};
  const TrackMenu = function TrackMenu() {};
  const PlatformProvider = function PlatformProvider() {};
  const React = {
    Suspense: Symbol("Suspense"),
    createElement(type, props, ...children) {
      return {
        type,
        props: {
          ...(props || {}),
          children: children.length <= 1 ? children[0] : children,
        },
      };
    },
  };
  const context = {
    AbortController,
    Date,
    URLSearchParams,
    clearTimeout,
    fetch: fetchImpl,
    setTimeout,
    console: { error() {}, log() {}, warn() {} },
    document: {
      body,
      createElement(tag) {
        if (tag !== "div") return {};
        if (!wrap.className && !wrap.innerHTML) return wrap;
        return makeDiv();
      },
      querySelector(selector) {
        if (
          selector === "main" ||
          selector === '[data-testid="main-view-container"]'
        ) {
          return pageContainer;
        }
        return options.nativeMenus && selector === '[data-testid="now-playing-bar"]'
          ? nativeHost
          : null;
      },
    },
  };
  const getTrack = {};
  const searchModalResults = {};
  context.Spicetify = {
    ContextMenu: {
      Item: class {
        constructor(_label, callback) {
          handler = callback;
        }
        register() {}
      },
    },
    GraphQL: {
      Definitions: { getTrack, searchModalResults },
      async Request(definition, variables) {
        graphqlRequests.push({ definition, variables });
        if (definition === getTrack) {
          if (
            options.spotifyTrackDetails &&
            variables?.uri === options.spotifyTrackDetails.uri
          ) {
            return {
              data: { trackUnion: options.spotifyTrackDetails },
            };
          }
          return {
            data: {
              trackUnion: {
                name: "Blinding Lights",
                firstArtist: { items: [{ profile: { name: "The Weeknd" } }] },
              },
            },
          };
        }
        assert.equal(definition, searchModalResults);
        return {
          data: {
            searchV2: {
              topResultsV2: {
                itemsV2: (
                  options.spotifyTracks?.[variables.searchTerm] ||
                  options.spotifyTrack
                )
                  ? [{
                    item: {
                      data: options.spotifyTracks?.[variables.searchTerm] ||
                        options.spotifyTrack,
                    },
                  }]
                  : [],
              },
            },
          },
        };
      },
    },
    ...(options.languageByTrackId ? {
      CosmosAsync: {
        async get(url) {
          cosmosRequests.push(url);
          const trackId = url.match(/\/track\/([^?]+)/)?.[1];
          const language = options.languageByTrackId[trackId];
          return language
            ? { lyrics: { language } }
            : { code: 404, error: "lyrics unavailable" };
        },
      },
    } : {}),
    Platform: {
      History: {
        location: { pathname: "/" },
        listen(callback) {
          historyListeners.push(callback);
        },
        push(value) {
          const path = typeof value === "string" ? value : value.pathname;
          history.push(path);
          this.location.pathname = path;
          if (path !== "/soundalike") currentPage = undefined;
          for (const listener of historyListeners) {
            listener({ pathname: path });
          }
        },
      },
      Registry: registry,
    },
    Player: {
      async playUri(uri) {
        played.push(uri);
      },
    },
    LocalStorage: {
      get(key) {
        return storage.get(key) ?? null;
      },
      set(key, value) {
        storage.set(key, value);
      },
    },
    React,
    ReactComponent: { PlatformProvider, RightClickMenu, TrackMenu },
    ReactDOM: {
      createRoot(row) {
        return {
          render(tree) {
            row.tree = tree;
          },
          unmount() {
            delete row.tree;
          },
        };
      },
    },
    ReactJSX: { jsx() {} },
    URI: {},
    _platform: platform,
    showNotification(...args) {
      notifications.push(args);
    },
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(extensionSource, context);
  return {
    components: {
      AppProvider,
      PlatformProvider,
      PlatformContextProvider,
      RegistryProvider,
      RightClickMenu,
      StoreProvider,
      TrackMenu,
    },
    history,
    notifications,
    played,
    graphqlRequests,
    cosmosRequests,
    nativeChildren: pageContainer.children,
    languageStatus,
    rows,
    get currentPage() {
      return currentPage;
    },
    navigate(path) {
      context.Spicetify.Platform.History.push(path);
    },
    async run(uri = "spotify:track:test") {
      await handler([uri]);
      await new Promise((resolve) => setTimeout(resolve, 0));
      return currentPage;
    },
  };
}

function findElement(node, predicate) {
  if (!node || typeof node !== "object") return null;
  if (predicate(node)) return node;
  const children = node.props?.children;
  for (const child of Array.isArray(children) ? children : [children]) {
    const match = findElement(child, predicate);
    if (match) return match;
  }
  return null;
}

test("uses the hosted library when the local companion is unavailable", async () => {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url === "http://127.0.0.1:8787/health") {
      throw new TypeError("connection refused");
    }
    assert.match(
      url,
      /^https:\/\/soundalike\.yassin\.app\/api\/spicetify_recommend\?/,
    );
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.equal(urls[0], "http://127.0.0.1:8787/health");
  assert.equal(urls.length, 2);
  assert.match(
    urls[1],
    /^https:\/\/soundalike\.yassin\.app\/api\/spicetify_recommend\?/,
  );
  assert.equal(new URL(urls[1]).searchParams.get("v"), "3");
  assert.equal(
    new URL(urls[1]).searchParams.get("language_policy"),
    "spotify-lyrics-v1",
  );
  assert.equal(new URL(urls[1]).searchParams.get("n"), "40");
  assert.match(page.innerHTML, /HOSTED LIBRARY/);
  assert.deepEqual(app.history, ["/soundalike"]);
  assert.equal(
    app.notifications.some(([message]) => message.includes("first request after idle")),
    true,
  );
});

test("prefers a healthy local companion over the hosted library", async () => {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url.endsWith("/health")) {
      return response(200, { ok: true });
    }
    assert.equal(url, "http://127.0.0.1:8787/api/recommend");
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.deepEqual(urls, [
    "http://127.0.0.1:8787/health",
    "http://127.0.0.1:8787/api/recommend",
  ]);
  assert.match(page.innerHTML, /LOCAL ENGINE/);
  assert.equal(
    app.notifications.some(([message]) => message.includes("first request after idle")),
    false,
  );
});

test("falls back to hosted results when a healthy local companion later fails", async () => {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url.endsWith("/health")) {
      return response(200, { ok: true });
    }
    if (url.startsWith("http://127.0.0.1")) {
      return response(500, { ok: false, error: "local engine failed" });
    }
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.deepEqual(urls.slice(0, 2), [
    "http://127.0.0.1:8787/health",
    "http://127.0.0.1:8787/api/recommend",
  ]);
  assert.equal(urls.length, 3);
  assert.match(
    urls[2],
    /^https:\/\/soundalike\.yassin\.app\/api\/spicetify_recommend\?/,
  );
  assert.match(page.innerHTML, /HOSTED LIBRARY/);
});

test("uses the legacy hosted POST during a deployment race", async () => {
  const urls = [];
  const app = loadExtension(async (url, options) => {
    urls.push(url);
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    if (url.includes("/api/spicetify_recommend")) {
      return response(404, { error: "not found" });
    }
    assert.equal(url, "https://soundalike.yassin.app/api/recommend");
    assert.equal(options.method, "POST");
    return response(200, recommendation);
  });

  const page = await app.run();
  await app.run();

  assert.equal(urls.length, 5);
  assert.match(page.innerHTML, /HOSTED LIBRARY/);
});

test("shows the hosted library miss without opening an empty page", async () => {
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) {
      throw new TypeError("connection refused");
    }
    return response(422, {
      ok: false,
      error: "This track is not in the hosted library.",
    });
  });

  const page = await app.run();

  assert.equal(page, undefined);
  assert.equal(
    app.notifications.some(([message, isError]) =>
      message.includes("not in the hosted library") && isError === true),
    true,
  );
});

test("renders playlist metadata, plays on double-click, and exposes the native menu", async () => {
  const result = { title: "Take My Breath", artist: "The Weeknd", bpm: 121.7 };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:verified",
    albumOfTrack: {
      name: "Dawn FM",
      coverArt: { sources: [] },
    },
    artists: {
      items: [{
        profile: { name: result.artist },
      }],
    },
  };
  const spotifyTrackDetails = {
    __typename: "Track",
    name: result.title,
    uri: spotifyTrack.uri,
    albumOfTrack: {
      name: "Dawn FM",
      uri: "spotify:album:verified",
      coverArt: { sources: [] },
    },
    firstArtist: {
      items: [{
        uri: "spotify:artist:verified",
        profile: { name: result.artist },
      }],
    },
    otherArtists: { items: [] },
  };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, {
      ...recommendation,
      results: [result],
    });
  }, {
    nativeMenus: true,
    results: [result],
    spotifyTrack,
    spotifyTrackDetails,
  });

  await app.run();

  const rowTree = app.rows[0].tree;
  const rightClickMenu = findElement(
    rowTree,
    (node) => node.type === app.components.RightClickMenu,
  );
  assert.ok(rightClickMenu);
  assert.ok(findElement(
    rowTree,
    (node) => node.type === app.components.StoreProvider,
  ));
  const trackMenu = findElement(
    rightClickMenu.props.menu,
    (node) => node.type === app.components.TrackMenu,
  );
  assert.equal(trackMenu.props.uri, spotifyTrack.uri);
  assert.equal(trackMenu.props.albumUri, spotifyTrackDetails.albumOfTrack.uri);
  assert.deepEqual(JSON.parse(JSON.stringify(trackMenu.props.artists)), [{
    type: "artist",
    name: result.artist,
    uri: "spotify:artist:verified",
  }]);

  const playButton = findElement(
    rowTree,
    (node) => node.props?.className === "sa-play",
  );
  const leadingCell = findElement(
    rowTree,
    (node) => node.props?.className === "sa-leading",
  );
  assert.equal(findElement(leadingCell, (node) => node === playButton), playButton);
  assert.equal(
    findElement(rowTree, (node) => node.props?.className === "sa-album").props.children,
    "Dawn FM",
  );
  assert.equal(
    findElement(rowTree, (node) => node.props?.className === "sa-bpm").props.children,
    "122",
  );
  const rowContent = findElement(
    rowTree,
    (node) => node.props?.className === "sa-row-content",
  );
  assert.equal(rowContent.props.onClick, undefined);
  await rowContent.props.onDoubleClick();
  assert.deepEqual(app.played, [spotifyTrack.uri]);

  let propagationStopped = false;
  await playButton.props.onClick({
    stopPropagation() {
      propagationStopped = true;
    },
  });
  assert.equal(propagationStopped, true);
  assert.deepEqual(app.played, [spotifyTrack.uri, spotifyTrack.uri]);
  assert.deepEqual(app.history, ["/soundalike"]);
  assert.ok(app.currentPage);
});

test("keeps Spotify search as the fallback for unresolved result rows", async () => {
  const result = { title: "Catalog Miss", artist: "Unknown Artist" };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, {
      ...recommendation,
      results: [result],
    });
  }, {
    nativeMenus: true,
    results: [result],
  });

  await app.run();

  const rowTree = app.rows[0].tree;
  assert.notEqual(rowTree.type, app.components.RightClickMenu);
  const trigger = findElement(
    rowTree,
    (node) => node.props?.className === "sa-row-content",
  );
  assert.equal(trigger.props.onClick, undefined);
  await trigger.props.onDoubleClick();
  assert.deepEqual(app.played, []);
  assert.deepEqual(app.history, [
    "/soundalike",
    "/search/Catalog%20Miss%20Unknown%20Artist",
  ]);
});

test("restores cached results when navigating back to the Soundalike page", async () => {
  const result = { title: "Take My Breath", artist: "The Weeknd" };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:verified",
    albumOfTrack: { coverArt: { sources: [] } },
    artists: { items: [{ profile: { name: result.artist } }] },
  };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    nativeMenus: true,
    results: [result],
    spotifyTrack,
  });

  const firstPage = await app.run();
  app.navigate("/home");
  assert.equal(app.currentPage, undefined);

  app.navigate("/soundalike");
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.ok(app.currentPage);
  assert.equal(firstPage.innerHTML, app.currentPage.innerHTML);
  assert.equal(app.rows[0].dataset.uri, spotifyTrack.uri);
  assert.deepEqual(app.nativeChildren, [{ id: "spotify-owned-content" }]);
});

test("filters confident cross-language results and keeps unknown lyrics as fallback", async () => {
  const results = [
    { title: "Même amour", artist: "Artiste Français" },
    { title: "English Song", artist: "English Artist" },
    { title: "Sans paroles", artist: "Artiste Inconnu" },
  ];
  const spotifyTracks = Object.fromEntries(results.map((result, index) => [
    `${result.title} ${result.artist}`,
    {
      __typename: "Track",
      name: result.title,
      uri: `spotify:track:result${index}`,
      albumOfTrack: { coverArt: { sources: [] } },
      artists: { items: [{ profile: { name: result.artist } }] },
    },
  ]));
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results });
  }, {
    results,
    spotifyTracks,
    languageByTrackId: {
      test: "fr",
      result0: "fr",
      result1: "en",
    },
  });

  await app.run();
  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.match(app.languageStatus.textContent, /French lyrics/);
  assert.equal(app.cosmosRequests.length, 4);
  assert.equal(app.rows[0].hidden, false);
  assert.equal(app.rows[1].hidden, true);
  assert.equal(app.rows[2].hidden, false);
});

test("reuses persisted recommendations and Spotify metadata on repeated tracks", async () => {
  const storage = new Map();
  const result = { title: "Take My Breath", artist: "The Weeknd", bpm: 122 };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:verified",
    albumOfTrack: {
      name: "Dawn FM",
      uri: "spotify:album:verified",
      coverArt: { sources: [] },
    },
    artists: { items: [{ profile: { name: result.artist } }] },
  };
  const urls = [];
  const first = loadExtension(async (url) => {
    urls.push(url);
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    nativeMenus: true,
    results: [result],
    spotifyTrack,
    storage,
  });

  await first.run();
  await new Promise((resolve) => setTimeout(resolve, 250));
  assert.equal(
    urls.filter((url) => url.includes("/api/spicetify_recommend")).length,
    1,
  );
  const persisted = JSON.parse(storage.get("soundalike:spicetify-cache:v3"));
  assert.ok(persisted.spotifyTracks["spotify:track:test"]);

  const second = loadExtension(async (url) => {
    throw new Error(`unexpected network request: ${url}`);
  }, {
    nativeMenus: true,
    results: [result],
    spotifyTrack,
    storage,
  });
  await second.run();

  assert.deepEqual(second.graphqlRequests, []);
  assert.equal(second.rows[0].dataset.uri, spotifyTrack.uri);
  assert.equal(second.currentPage !== undefined, true);
});
