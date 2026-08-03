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
  const historyListeners = [];
  const rows = (options.results || []).map((_, index) => ({
    dataset: { index: String(index) },
    querySelector() {
      return null;
    },
  }));
  const wrap = {
    className: "",
    innerHTML: "",
    querySelector(selector) {
      const match = selector.match(/data-index="(\d+)"/);
      return match ? rows[Number(match[1])] : null;
    },
    querySelectorAll(selector) {
      return selector === ".sa-row" ? rows : [];
    },
  };
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
  const pageContainer = {
    parentElement: null,
    replaceChildren(value) {
      currentPage = value;
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
    clearTimeout,
    fetch: fetchImpl,
    setTimeout,
    console: { error() {}, log() {}, warn() {} },
    document: {
      createElement(tag) {
        return tag === "div" ? wrap : {};
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
                itemsV2: options.spotifyTrack
                  ? [{ item: { data: options.spotifyTrack } }]
                  : [],
              },
            },
          },
        };
      },
    },
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
    rows,
    get currentPage() {
      return currentPage;
    },
    navigate(path) {
      context.Spicetify.Platform.History.push(path);
    },
    async run() {
      await handler(["spotify:track:test"]);
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
    assert.equal(url, "https://soundalike.yassin.app/api/recommend");
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.deepEqual(urls, [
    "http://127.0.0.1:8787/health",
    "https://soundalike.yassin.app/api/recommend",
  ]);
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

  assert.deepEqual(urls, [
    "http://127.0.0.1:8787/health",
    "http://127.0.0.1:8787/api/recommend",
    "https://soundalike.yassin.app/api/recommend",
  ]);
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

test("plays verified Spotify tracks and exposes their native track menu", async () => {
  const result = { title: "Take My Breath", artist: "The Weeknd" };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:verified",
    albumOfTrack: {
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
  let propagationStopped = false;
  await playButton.props.onClick({
    stopPropagation() {
      propagationStopped = true;
    },
  });
  assert.equal(propagationStopped, true);
  assert.deepEqual(app.played, [spotifyTrack.uri]);
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
  await trigger.props.onClick();
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
});
