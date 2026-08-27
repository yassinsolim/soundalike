import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
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
  method: "dual_sonic64_guardrail",
  ranking_policy: "model-quality-v1",
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

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function waitFor(predicate, timeoutMs = 500) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.fail("Timed out waiting for condition");
}

function loadExtension(fetchImpl, options = {}) {
  let handler;
  let currentPage;
  const notifications = [];
  const history = [];
  const played = [];
  const graphqlRequests = [];
  const cosmosRequests = [];
  const logs = [];
  const historyListeners = [];
  const storage = options.storage || new Map();
  const rows = (options.results || []).map((_, index) => ({
    dataset: { index: String(index) },
    hidden: index >= 20,
    style: {},
    querySelector() {
      return null;
    },
  }));
  const languageStatus = { textContent: "" };
  const languageLoading = { hidden: false };
  function control(value = "") {
    const classes = new Set();
    return {
      value,
      checked: false,
      disabled: false,
      hidden: false,
      textContent: "",
      classList: {
        add(name) {
          classes.add(name);
        },
        remove(name) {
          classes.delete(name);
        },
        contains(name) {
          return classes.has(name);
        },
      },
    };
  }
  const feedback = {
    panel: { ...control(), hidden: true, dataset: {} },
    details: { ...control(), hidden: true },
    ratings: ["good", "mixed", "off"].map(control),
    reasons: [
      "style",
      "mood_energy",
      "tempo",
      "vocals_language",
      "instruments_timbre",
    ].map(control),
    note: { ...control(), maxLength: 280 },
    count: control(),
    question: control(),
    actions: control(),
    preference: { ...control(), hidden: true },
    keepShowing: control(),
    stopShowing: control(),
    send: { ...control(), disabled: true },
    dismiss: control(),
    status: control(),
  };
  function queryPage(selector) {
    if (selector === ".sa-language-status") return languageStatus;
    if (selector === ".sa-language-loading") return languageLoading;
    const controls = {
      ".sa-feedback": feedback.panel,
      ".sa-feedback-details": feedback.details,
      ".sa-feedback-note": feedback.note,
      ".sa-feedback-count": feedback.count,
      ".sa-feedback-question": feedback.question,
      ".sa-feedback-actions": feedback.actions,
      ".sa-feedback-preference": feedback.preference,
      ".sa-feedback-keep-showing": feedback.keepShowing,
      ".sa-feedback-stop-showing": feedback.stopShowing,
      ".sa-feedback-send": feedback.send,
      ".sa-feedback-dismiss": feedback.dismiss,
      ".sa-feedback-status": feedback.status,
    };
    if (controls[selector]) return controls[selector];
    const match = selector.match(/data-index="(\d+)"/);
    return match ? rows[Number(match[1])] : null;
  }
  function queryAllPage(selector) {
    if (selector === ".sa-row") return rows;
    if (selector === 'input[name="sa-feedback-rating"]') {
      return feedback.ratings;
    }
    if (selector === 'input[name="sa-feedback-reason"]') {
      return feedback.reasons;
    }
    return [];
  }
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
        return queryPage(selector);
      },
      querySelectorAll(selector) {
        return queryAllPage(selector);
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
    crypto: webcrypto,
    fetch: fetchImpl,
    setTimeout,
    console: {
      error() {},
      log(...values) {
        logs.push(values.join(" "));
      },
      warn() {},
    },
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
  class GraphQLDefinition {
    constructor(name, operation, sha256Hash, value) {
      this.name = name;
      this.operation = operation;
      this.sha256Hash = sha256Hash;
      this.value = value;
    }
  }
  const getTrack = new GraphQLDefinition("getTrack", "query", "get-track", null);
  const searchModalResults = new GraphQLDefinition(
    "searchModalResults",
    "query",
    "search-modal",
    null,
  );
  const searchTracks = new GraphQLDefinition(
    "searchTracks",
    "query",
    "search-tracks",
    null,
  );
  const queryArtistRelated = new GraphQLDefinition(
    "queryArtistRelated",
    "query",
    "artist-related",
    null,
  );
  const definitions = { getTrack, searchModalResults };
  if (options.includeSearchTracksDefinition !== false) {
    definitions.searchTracks = searchTracks;
  }
  if (options.includeRelatedArtistsDefinition !== false) {
    definitions.queryArtistRelated = queryArtistRelated;
  }
  const languageAttempts = new Map();
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
      Definitions: definitions,
      async Request(definition, variables) {
        graphqlRequests.push({ definition, variables });
        await options.beforeGraphqlRequest?.(definition, variables);
        if (definition === getTrack) {
          if (options.seedTrack && variables?.uri === "spotify:track:test") {
            return {
              data: { trackUnion: options.seedTrack },
            };
          }
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
        if (definition === queryArtistRelated) {
          return {
            data: {
              artistUnion: {
                relatedContent: {
                  relatedArtists: {
                    items: options.relatedArtists || [],
                  },
                },
              },
            },
          };
        }
        if (definition === searchTracks || definition?.name === "searchTracks") {
          const tracks = options.spotifyTrackSearchResults?.[variables.searchTerm] || [];
          return {
            data: {
              searchV2: {
                tracksV2: {
                  items: tracks.map((track) => ({
                    item: { data: track },
                  })),
                },
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
          const configured = options.languageByTrackId[trackId];
          const attempt = languageAttempts.get(trackId) || 0;
          languageAttempts.set(trackId, attempt + 1);
          const language = Array.isArray(configured)
            ? configured[Math.min(attempt, configured.length - 1)]
            : configured;
          if (language?.error) throw new Error(language.error);
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
      remove(key) {
        storage.delete(key);
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
    logs,
    nativeChildren: pageContainer.children,
    languageStatus,
    languageLoading,
    feedback,
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

test("starts the optional local probe before the first search", async () => {
  const probe = deferred();
  const urls = [];
  loadExtension((url) => {
    urls.push(url);
    return probe.promise;
  });

  assert.deepEqual(urls, ["http://127.0.0.1:8787/health"]);

  probe.resolve(response(200, { ok: false }));
  await new Promise((resolve) => setTimeout(resolve, 0));
});

test("uses the always-on hosted library when the local companion is unavailable", async () => {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url === "http://127.0.0.1:8787/health") {
      throw new TypeError("connection refused");
    }
    assert.match(
      url,
      /^https:\/\/soundalike-api\.yassin\.app\/api\/spicetify_recommend\?/,
    );
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.equal(urls[0], "http://127.0.0.1:8787/health");
  assert.equal(urls.length, 2);
  assert.match(
    urls[1],
    /^https:\/\/soundalike-api\.yassin\.app\/api\/spicetify_recommend\?/,
  );
  assert.match(page.innerHTML, /Dual-Sonic64/);
  assert.match(page.innerHTML, /V5 strict \+ artist context/);
  assert.equal(new URL(urls[1]).searchParams.get("v"), "4");
  assert.equal(
    new URL(urls[1]).searchParams.get("language_policy"),
    "spotify-lyrics-strict-v2",
  );
  assert.equal(
    new URL(urls[1]).searchParams.get("ranking_policy"),
    "model-quality-v1",
  );
  assert.equal(new URL(urls[1]).searchParams.get("n"), "20");
  assert.match(page.innerHTML, /HOSTED LIBRARY/);
  assert.deepEqual(app.history, ["/soundalike"]);
  assert.equal(
    app.notifications.some(([message]) => message.includes("first request after idle")),
    false,
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

test("uses Vercel when the primary lacks the strict contract", async () => {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    if (url.startsWith("https://soundalike-api.yassin.app")) {
      return response(400, { ok: false, error: "unsupported API version" });
    }
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.equal(urls.length, 3);
  assert.match(
    urls[2],
    /^https:\/\/soundalike\.yassin\.app\/api\/spicetify_recommend\?/,
  );
  assert.equal(new URL(urls[2]).searchParams.get("v"), "4");
  assert.equal(
    new URL(urls[2]).searchParams.get("language_policy"),
    "spotify-lyrics-strict-v2",
  );
  assert.match(page.innerHTML, /HOSTED LIBRARY/);
});

test("uses Vercel when the primary returns an outdated ranking policy", async () => {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    if (url.startsWith("https://soundalike-api.yassin.app")) {
      return response(200, {
        ...recommendation,
        ranking_policy: "legacy-v1",
      });
    }
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.equal(urls.length, 3);
  assert.match(
    urls[2],
    /^https:\/\/soundalike\.yassin\.app\/api\/spicetify_recommend\?/,
  );
  assert.match(page.innerHTML, /HOSTED LIBRARY/);
});

test("uses Vercel when the primary service fails", async () => {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    if (url.startsWith("https://soundalike-api.yassin.app")) {
      return response(503, {
        ok: false,
        error: "service unavailable",
      });
    }
    return response(200, recommendation);
  });

  const page = await app.run();

  assert.equal(urls.length, 3);
  assert.match(
    urls[2],
    /^https:\/\/soundalike\.yassin\.app\/api\/spicetify_recommend\?/,
  );
  assert.equal(new URL(urls[2]).searchParams.get("v"), "4");
  assert.match(page.innerHTML, /HOSTED LIBRARY/);
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
    /^https:\/\/soundalike-api\.yassin\.app\/api\/spicetify_recommend\?/,
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

  assert.equal(urls.length, 7);
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
  assert.equal(
    rightClickMenu.props.menu.props.uri,
    spotifyTrack.uri,
    "registered context actions need the result track URI",
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
  const albumCell = findElement(rowTree, (node) => node.props?.className === "sa-album");
  const albumLink = findElement(albumCell, (node) => node.props?.className === "sa-album-link");
  assert.ok(albumLink, "album name should be a link when the album URI is available");
  assert.equal(albumLink.props.children, "Dawn FM");
  {
    let defaultPrevented = false;
    let propagationStopped = false;
    albumLink.props.onClick({
      preventDefault: () => { defaultPrevented = true; },
      stopPropagation: () => { propagationStopped = true; },
    });
    assert.ok(defaultPrevented, "album link click should prevent default");
    assert.ok(propagationStopped, "album link click should stop propagation");
    assert.equal(app.history.at(-1), "/album/verified", "album link should navigate to the album page");
    app.navigate("/soundalike");
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
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
  assert.deepEqual(app.history, ["/soundalike", "/album/verified", "/soundalike"]);
  assert.ok(app.currentPage);
});

test("keeps Spotify search as the fallback for unresolved result rows", async () => {
  const result = { title: "Catalog Miss", artist: "Unknown Artist" };
  const wrongTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:wrong",
    albumOfTrack: { coverArt: { sources: [] } },
    artists: { items: [{ profile: { name: "Unknown Artist Tribute" } }] },
  };
  const splitNameTrack = {
    ...wrongTrack,
    uri: "spotify:track:split-name",
    artists: {
      items: [
        { profile: { name: "Unknown" } },
        { profile: { name: "Artist" } },
      ],
    },
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
    spotifyTracks: {
      [`${result.title} ${result.artist}`]: wrongTrack,
    },
    spotifyTrackSearchResults: {
      [`${result.title} ${result.artist}`]: [wrongTrack, splitNameTrack],
    },
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

test("falls back to Spotify's song-only search when top results omit a track", async () => {
  const result = { title: "Echo (feat. Richard Caddock)", artist: "WRLD" };
  const spotifyTrack = {
    __typename: "Track",
    name: "Echo",
    uri: "spotify:track:69b9S93kHT979Iw3rvev89",
    albumOfTrack: { coverArt: { sources: [] } },
    artists: {
      items: [
        { profile: { name: "WRLD" } },
        { profile: { name: "Richard Caddock" } },
      ],
    },
  };
  const searchTerm = `${result.title} ${result.artist}`;
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    includeSearchTracksDefinition: false,
    results: [result],
    spotifyTrackSearchResults: { [searchTerm]: [spotifyTrack] },
  });

  await app.run();

  assert.equal(app.rows[0].dataset.uri, spotifyTrack.uri);
  const searches = app.graphqlRequests.filter(
    ({ variables }) => variables.searchTerm === searchTerm,
  );
  assert.deepEqual(searches.map(({ variables }) => variables.limit), [5, 20]);
  assert.equal(searches[1].definition.name, "searchTracks");
  assert.equal(
    searches[1].definition.sha256Hash,
    "59ee4a659c32e9ad894a71308207594a65ba67bb6b632b183abe97303a51fa55",
  );
});

test("matches combined credits without splitting punctuation inside artist names", async () => {
  const result = {
    title: "Potato Salad",
    artist: "Tyler, The Creator & A$AP Rocky",
  };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:combined",
    albumOfTrack: { coverArt: { sources: [] } },
    artists: {
      items: [
        { profile: { name: "Tyler, The Creator" } },
        { profile: { name: "A$AP Rocky" } },
      ],
    },
  };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    results: [result],
    spotifyTrackSearchResults: {
      [`${result.title} ${result.artist}`]: [spotifyTrack],
    },
  });

  await app.run();

  assert.equal(app.rows[0].dataset.uri, spotifyTrack.uri);
});

test("does not persist unresolved Spotify searches", async () => {
  const storage = new Map();
  const result = { title: "Catalog Miss", artist: "Unknown Artist" };
  const fetchImpl = async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  };
  const first = loadExtension(fetchImpl, { results: [result], storage });

  await first.run();
  await new Promise((resolve) => setTimeout(resolve, 150));

  const persisted = JSON.parse(storage.get("soundalike:spicetify-cache:v9"));
  const cacheId = "result:catalog miss::unknown artist";
  assert.equal(Object.hasOwn(persisted.spotifyTracks, cacheId), false);

  const second = loadExtension(async (url) => {
    throw new Error(`unexpected network request: ${url}`);
  }, { results: [result], storage });
  await second.run();

  assert.equal(
    second.graphqlRequests.filter(({ variables }) => variables.searchTerm).length,
    2,
  );
});

test("invalidates legacy negative caches before resolving a later Spotify match", async () => {
  const result = { title: "Recovered Song", artist: "Recovered Artist" };
  const cacheId = "result:recovered song::recovered artist";
  const storage = new Map([[
    "soundalike:spicetify-cache:v4",
    JSON.stringify({
      recommendations: {},
      spotifyTracks: {
        [cacheId]: {
          value: null,
          cachedAt: Date.now(),
          lastUsedAt: Date.now(),
        },
      },
    }),
  ]]);
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:recovered",
    albumOfTrack: { coverArt: { sources: [] } },
    artists: { items: [{ profile: { name: result.artist } }] },
  };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    results: [result],
    spotifyTrackSearchResults: {
      [`${result.title} ${result.artist}`]: [spotifyTrack],
    },
    storage,
  });

  await app.run();

  assert.equal(storage.has("soundalike:spicetify-cache:v4"), false);
  assert.equal(app.rows[0].dataset.uri, spotifyTrack.uri);
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

test("opens the results page without waiting for related-artist context", async () => {
  const relatedArtists = deferred();
  const result = { title: "Fast Result", artist: "Fast Artist" };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:fast-result",
    albumOfTrack: {
      name: "Fast Album",
      uri: "spotify:album:fast",
      coverArt: {
        sources: [{ url: "https://i.scdn.co/image/fast", width: 64 }],
      },
    },
    artists: {
      items: [{
        uri: "spotify:artist:fast",
        profile: { name: result.artist },
      }],
    },
  };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    results: [result],
    seedTrack: {
      __typename: "Track",
      name: "Blinding Lights",
      uri: "spotify:track:test",
      firstArtist: {
        items: [{
          uri: "spotify:artist:seed",
          profile: { name: "The Weeknd" },
        }],
      },
      albumOfTrack: { coverArt: { sources: [] } },
    },
    spotifyTrack,
    languageByTrackId: {
      test: "en",
      "fast-result": "en",
    },
    beforeGraphqlRequest(definition) {
      return definition?.name === "queryArtistRelated"
        ? relatedArtists.promise
        : undefined;
    },
  });

  const run = app.run();
  await new Promise((resolve) => setTimeout(resolve, 50));
  const navigatedBeforeRelatedArtists = app.history.at(-1) === "/soundalike";
  relatedArtists.resolve();
  await run;
  await waitFor(() => app.languageLoading.hidden);

  assert.equal(navigatedBeforeRelatedArtists, true);
  assert.match(app.logs.join("\n"), /results page ready in \d+ ms/);
});

test("reveals verified rows progressively and skips redundant detail lookups", async () => {
  const delayedSearch = deferred();
  const results = [
    { title: "Ready First", artist: "Ready Artist" },
    { title: "Delayed Second", artist: "Delayed Artist" },
  ];
  const spotifyTracks = Object.fromEntries(results.map((result, index) => [
    `${result.title} ${result.artist}`,
    {
      __typename: "Track",
      name: result.title,
      uri: `spotify:track:progressive${index}`,
      albumOfTrack: {
        name: `Album ${index}`,
        uri: `spotify:album:progressive${index}`,
        coverArt: {
          sources: [{
            url: `https://i.scdn.co/image/progressive${index}`,
            width: 64,
          }],
        },
      },
      artists: {
        items: [{
          uri: `spotify:artist:progressive${index}`,
          profile: { name: result.artist },
        }],
      },
    },
  ]));
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results });
  }, {
    results,
    seedTrack: {
      __typename: "Track",
      name: "Blinding Lights",
      uri: "spotify:track:test",
      firstArtist: {
        items: [{
          uri: "spotify:artist:seed",
          profile: { name: "The Weeknd" },
        }],
      },
      albumOfTrack: { coverArt: { sources: [] } },
    },
    spotifyTracks,
    languageByTrackId: {
      test: "en",
      progressive0: "en",
      progressive1: "en",
    },
    beforeGraphqlRequest(definition, variables) {
      return definition?.name === "searchModalResults" &&
          variables?.searchTerm === "Delayed Second Delayed Artist"
        ? delayedSearch.promise
        : undefined;
    },
  });

  await app.run();
  await waitFor(() => app.rows[0].hidden === false);

  assert.equal(app.rows[1].hidden, true);
  assert.equal(app.languageLoading.hidden, false);

  delayedSearch.resolve();
  await waitFor(() => app.languageLoading.hidden);

  assert.equal(app.rows[1].hidden, false);
  assert.equal(
    app.graphqlRequests.filter(({ definition }) => definition?.name === "getTrack")
      .length,
    1,
    "complete search metadata should avoid per-result detail requests",
  );
  assert.match(app.logs.join("\n"), /Spotify metadata settled in \d+ ms/);
});

test("shows model-order rows immediately when seed language is unavailable", async () => {
  const delayedSearch = deferred();
  const result = { title: "No Gate Result", artist: "No Gate Artist" };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    results: [result],
    seedTrack: {
      __typename: "Track",
      name: "Instrumental Seed",
      uri: "spotify:track:test",
      firstArtist: {
        items: [{
          uri: "spotify:artist:seed",
          profile: { name: "Seed Artist" },
        }],
      },
      albumOfTrack: { coverArt: { sources: [] } },
    },
    beforeGraphqlRequest(definition, variables) {
      return definition?.name === "searchModalResults" && variables?.searchTerm
        ? delayedSearch.promise
        : undefined;
    },
  });

  await app.run();

  assert.equal(app.rows[0].hidden, false);
  assert.equal(app.languageLoading.hidden, false);

  delayedSearch.resolve();
  await waitFor(() => app.languageLoading.hidden);
});

test("strictly filters cross-language and unavailable-language results", async () => {
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
  await new Promise((resolve) => setTimeout(resolve, 450));

  assert.match(app.languageStatus.textContent, /French lyrics/);
  assert.equal(app.cosmosRequests.length, 5);
  assert.equal(app.rows[0].hidden, false);
  assert.equal(app.rows[1].hidden, true);
  assert.equal(app.rows[2].hidden, true);
  assert.equal(app.languageLoading.hidden, true);
  assert.equal(
    (app.currentPage.innerHTML.match(/class="sa-row"[^>]* hidden/g) || []).length,
    3,
  );
  assert.match(app.languageStatus.textContent, /1 exact match/);
  assert.match(app.languageStatus.textContent, /1 different hidden/);
  assert.match(app.languageStatus.textContent, /1 no-lyrics hidden/);
});

test("retries transient lyrics misses before hiding a candidate", async () => {
  const result = { title: "Recovered Song", artist: "Related Artist" };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:result0",
    albumOfTrack: { coverArt: { sources: [] } },
    artists: {
      items: [{
        uri: "spotify:artist:related",
        profile: { name: result.artist },
      }],
    },
  };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    results: [result],
    spotifyTrack,
    languageByTrackId: {
      test: "en",
      result0: [null, "en"],
    },
  });

  await app.run();
  await new Promise((resolve) => setTimeout(resolve, 450));

  assert.equal(app.rows[0].hidden, false);
  assert.equal(app.cosmosRequests.length, 3);
  assert.match(app.languageStatus.textContent, /1 exact match in top 1/);
});

test("does not persist failed language lookups as unavailable", async () => {
  const storage = new Map();
  const result = { title: "Retry Song", artist: "Retry Artist" };
  const spotifyTrack = {
    __typename: "Track",
    name: result.title,
    uri: "spotify:track:result0",
    albumOfTrack: { coverArt: { sources: [] } },
    artists: { items: [{ profile: { name: result.artist } }] },
  };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, {
    results: [result],
    spotifyTrack,
    storage,
    languageByTrackId: {
      test: "en",
      result0: [
        { error: "temporary failure" },
        { error: "temporary failure" },
        "en",
      ],
    },
  });

  await app.run();
  await new Promise((resolve) => setTimeout(resolve, 450));
  assert.equal(app.rows[0].hidden, true);
  assert.match(app.languageStatus.textContent, /1 checks failed/);
  await new Promise((resolve) => setTimeout(resolve, 150));
  const firstCache = JSON.parse(storage.get("soundalike:spicetify-cache:v9"));
  const cacheId = "result:retry song::retry artist";
  assert.equal(
    Object.hasOwn(firstCache.spotifyTracks[cacheId].value, "soundalikeLanguage"),
    false,
  );

  await app.run();
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(app.rows[0].hidden, false);
  assert.equal(app.cosmosRequests.length, 4);
});

test("prioritizes related artists only inside the original quality head", async () => {
  const results = [
    { title: "Unrelated One", artist: "Unrelated Artist" },
    { title: "Related One", artist: "Related Artist" },
    ...Array.from({ length: 18 }, (_, index) => ({
      title: `Head ${index}`,
      artist: `Head Artist ${index}`,
    })),
    { title: "Related Tail", artist: "Related Artist" },
  ];
  const spotifyTracks = Object.fromEntries(results.map((result, index) => [
    `${result.title} ${result.artist}`,
    {
      __typename: "Track",
      name: result.title,
      uri: `spotify:track:result${index}`,
      albumOfTrack: { coverArt: { sources: [] } },
      artists: {
        items: [{
          uri: index === 1 || index === 20
            ? "spotify:artist:related"
            : `spotify:artist:unrelated${index}`,
          profile: { name: result.artist },
        }],
      },
    },
  ]));
  const languageByTrackId = Object.fromEntries([
    ["test", "en"],
    ...results.map((_, index) => [`result${index}`, "en"]),
  ]);
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results });
  }, {
    results,
    seedTrack: {
      __typename: "Track",
      name: "Blinding Lights",
      uri: "spotify:track:test",
      firstArtist: {
        items: [{
          uri: "spotify:artist:seed",
          profile: { name: "The Weeknd" },
        }],
      },
      albumOfTrack: { coverArt: { sources: [] } },
    },
    spotifyTracks,
    languageByTrackId,
    relatedArtists: [{
      id: "related",
      uri: "spotify:artist:related",
      profile: { name: "Related Artist" },
    }],
  });

  await app.run();
  await new Promise((resolve) => setTimeout(resolve, 150));

  assert.equal(
    app.graphqlRequests.some(
      ({ definition }) => definition?.name === "queryArtistRelated",
    ),
    true,
  );
  assert.equal(app.rows[1].hidden, false);
  assert.equal(app.rows[1].style.order, "1");
  assert.equal(app.rows[0].style.order, "2");
  assert.equal(app.rows[20].hidden, true);
  assert.match(app.languageStatus.textContent, /1 related prioritized/);
});

test("reuses persisted recommendations and Spotify metadata on repeated tracks", async () => {
  const storage = new Map();
  storage.set("soundalike:spicetify-cache:v2", "{\"stale\":true}");
  storage.set("soundalike:spicetify-cache:v3", "{\"oversized\":true}");
  storage.set("soundalike:spicetify-cache:v4", "{\"negative\":true}");
  storage.set("soundalike:spicetify-cache:v5", "{\"permissive\":true}");
  storage.set("soundalike:spicetify-cache:v6", "{\"deepTail\":true}");
  storage.set("soundalike:spicetify-cache:v7", "{\"compatibility\":true}");
  storage.set("soundalike:spicetify-cache:v8", "{\"ranking\":true}");
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
    artists: {
      items: [{
        profile: { name: result.artist },
        debugPayload: "x".repeat(50000),
      }],
    },
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
  assert.equal(storage.has("soundalike:spicetify-cache:v2"), false);
  assert.equal(storage.has("soundalike:spicetify-cache:v3"), false);
  assert.equal(storage.has("soundalike:spicetify-cache:v4"), false);
  assert.equal(storage.has("soundalike:spicetify-cache:v5"), false);
  assert.equal(storage.has("soundalike:spicetify-cache:v6"), false);
  assert.equal(storage.has("soundalike:spicetify-cache:v7"), false);
  assert.equal(storage.has("soundalike:spicetify-cache:v8"), false);

  await first.run();
  await new Promise((resolve) => setTimeout(resolve, 250));
  assert.equal(
    urls.filter((url) => url.includes("/api/spicetify_recommend")).length,
    1,
  );
  assert.equal(
    urls.some((url) => url.includes("ranking_policy=model-quality-v1")),
    true,
  );
  const persisted = JSON.parse(storage.get("soundalike:spicetify-cache:v9"));
  assert.ok(storage.get("soundalike:spicetify-cache:v9").length < 10000);
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

test("offers feedback only after filtering settles and sends displayed order only", async () => {
  const results = [
    { title: "Model First", artist: "Artist One" },
    { title: "Related First", artist: "Related Artist" },
    { title: "French Hidden", artist: "Artiste Trois" },
  ];
  const spotifyTracks = Object.fromEntries(results.map((result, index) => [
    `${result.title} ${result.artist}`,
    {
      __typename: "Track",
      name: result.title,
      uri: `spotify:track:feedback${index}`,
      albumOfTrack: { coverArt: { sources: [] } },
      artists: {
        items: [{
          uri: index === 1
            ? "spotify:artist:related-feedback"
            : `spotify:artist:feedback${index}`,
          profile: { name: result.artist },
        }],
      },
    },
  ]));
  const feedbackRequests = [];
  const app = loadExtension(async (url, options) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    if (url.endsWith("/api/spicetify-feedback")) {
      feedbackRequests.push(options);
      return response(200, { receipt_sha256: "a".repeat(64) });
    }
    return response(200, {
      ...recommendation,
      index_version: "index-2026.07.11-dual-sonic64",
      results,
    });
  }, {
    results,
    seedTrack: {
      __typename: "Track",
      name: "Blinding Lights",
      uri: "spotify:track:test",
      firstArtist: {
        items: [{
          uri: "spotify:artist:seed-feedback",
          profile: { name: "The Weeknd" },
        }],
      },
      albumOfTrack: { coverArt: { sources: [] } },
    },
    spotifyTracks,
    languageByTrackId: {
      test: "en",
      feedback0: [null, "en"],
      feedback1: "en",
      feedback2: "fr",
    },
    relatedArtists: [{
      id: "related-feedback",
      uri: "spotify:artist:related-feedback",
      profile: { name: "Related Artist" },
    }],
  });

  await app.run();
  assert.equal(app.feedback.panel.hidden, true);
  assert.equal(feedbackRequests.length, 0, "feedback must never auto-submit");
  await new Promise((resolve) => setTimeout(resolve, 450));
  assert.equal(app.feedback.panel.hidden, false);

  app.feedback.ratings[1].checked = true;
  app.feedback.ratings[1].onchange();
  assert.equal(app.feedback.details.hidden, false);
  app.feedback.reasons[0].checked = true;
  app.feedback.reasons[0].onchange();
  app.feedback.reasons[2].checked = true;
  app.feedback.reasons[2].onchange();
  app.feedback.note.value = "Tempo drifted after the first result.";
  app.feedback.note.oninput();
  await app.feedback.send.onclick();

  assert.equal(feedbackRequests.length, 1);
  const sent = JSON.parse(feedbackRequests[0].body);
  assert.deepEqual(sent.seed, {
    title: "Blinding Lights",
    artist: "The Weeknd",
  });
  assert.deepEqual(sent.displayed_results, [
    { position: 1, title: "Related First", artist: "Related Artist" },
    { position: 2, title: "Model First", artist: "Artist One" },
  ]);
  assert.equal(
    sent.displayed_results.some((row) => row.title === "French Hidden"),
    false,
  );
  assert.equal(sent.method, "dual_sonic64_guardrail");
  assert.equal(sent.index_version, "index-2026.07.11-dual-sonic64");
  assert.equal(sent.api_version, "4");
  assert.equal(sent.language_policy, "spotify-lyrics-strict-v2");
  assert.equal(
    sent.selection_policy,
    "top-20-strict-language-related-artist-model-quality-v1",
  );
  assert.equal(sent.source, "hosted");
  assert.equal(sent.selection, "mixed");
  assert.deepEqual(sent.reasons, ["style", "tempo"]);
  assert.equal(sent.note, "Tempo drifted after the first result.");
  for (const forbidden of [
    "account",
    "credentials",
    "headers",
    "history",
    "hidden_candidates",
    "ip",
    "library",
  ]) {
    assert.equal(Object.hasOwn(sent, forbidden), false);
  }
  assert.match(sent.install_nonce, /^[a-f0-9]{32}$/);
  assert.match(sent.session_nonce, /^[a-f0-9]{32}$/);
  assert.match(app.feedback.status.textContent, /feedback received.*Receipt a{12}/);
  assert.equal(app.feedback.send.hidden, true);
  assert.equal(app.feedback.dismiss.hidden, true);
});

test("reveals optional details conditionally and caps reasons and note", async () => {
  const result = { title: "Candidate", artist: "Artist" };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, { results: [result] });

  await app.run();
  assert.equal(app.feedback.panel.hidden, false);
  assert.equal(app.feedback.details.hidden, true);
  assert.equal(app.feedback.send.disabled, true);

  app.feedback.ratings[2].checked = true;
  app.feedback.ratings[2].onchange();
  assert.equal(app.feedback.details.hidden, false);
  for (const reason of app.feedback.reasons.slice(0, 2)) {
    reason.checked = true;
    reason.onchange();
  }
  assert.equal(app.feedback.reasons[2].disabled, true);
  app.feedback.reasons[2].checked = true;
  app.feedback.reasons[2].onchange();
  assert.equal(app.feedback.reasons[2].checked, false);
  assert.match(app.feedback.status.textContent, /up to two/);

  app.feedback.note.value = "x".repeat(300);
  app.feedback.note.oninput();
  assert.equal(app.feedback.note.value.length, 280);
  assert.equal(app.feedback.count.textContent, "280/280");

  app.feedback.ratings.forEach((rating) => {
    rating.checked = rating.value === "good";
  });
  app.feedback.ratings[0].onchange();
  assert.equal(app.feedback.details.hidden, true);
  assert.equal(app.feedback.reasons.some((reason) => reason.checked), false);
  assert.equal(app.feedback.note.value, "");
});

test("keeps failed feedback retryable and asks again after successful feedback", async () => {
  const result = { title: "Candidate", artist: "Artist" };
  const feedbackBodies = [];
  const storage = new Map();
  const fetchImpl = async (url, options) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    if (url.endsWith("/api/spicetify-feedback")) {
      feedbackBodies.push(options.body);
      return feedbackBodies.length === 1
        ? response(503, { error: "storage unavailable" })
        : response(200, { receipt_sha256: "b".repeat(64) });
    }
    return response(200, { ...recommendation, results: [result] });
  };
  const app = loadExtension(fetchImpl, { results: [result], storage });

  await app.run();
  app.feedback.ratings[0].checked = true;
  app.feedback.ratings[0].onchange();
  await app.feedback.send.onclick();
  assert.equal(app.feedback.send.hidden, false);
  assert.equal(app.feedback.send.disabled, false);
  assert.equal(app.feedback.send.textContent, "Retry feedback");
  assert.match(app.feedback.status.textContent, /retry/);
  assert.equal(
    app.feedback.status.classList.contains("sa-feedback-error"),
    true,
  );
  assert.equal(storage.has("soundalike:feedback-preference:v2"), false);

  await app.feedback.send.onclick();
  assert.equal(feedbackBodies.length, 2);
  assert.equal(feedbackBodies[0], feedbackBodies[1]);
  assert.equal(app.feedback.send.hidden, true);
  const later = loadExtension(fetchImpl, { results: [result], storage });
  await later.run();
  assert.equal(later.feedback.panel.hidden, false);
});

test("asks after the second dismissal and keeps showing after Yes", async () => {
  const storage = new Map();
  const result = { title: "Candidate", artist: "Artist" };
  const feedbackUrls = [];
  const fetchImpl = async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    if (url.endsWith("/api/spicetify-feedback")) feedbackUrls.push(url);
    return response(200, { ...recommendation, results: [result] });
  };
  const first = loadExtension(fetchImpl, { results: [result], storage });
  await first.run();
  assert.equal(first.feedback.panel.hidden, false);
  first.feedback.dismiss.onclick();
  assert.equal(first.feedback.panel.hidden, true);
  assert.deepEqual(
    JSON.parse(storage.get("soundalike:feedback-preference:v2")),
    { dismissals: 1, showAgain: null },
  );
  assert.equal(feedbackUrls.length, 0);

  const second = loadExtension(fetchImpl, { results: [result], storage });
  await second.run();
  assert.equal(second.feedback.panel.hidden, false);
  second.feedback.dismiss.onclick();
  assert.equal(second.feedback.panel.hidden, false);
  assert.equal(second.feedback.question.hidden, true);
  assert.equal(second.feedback.actions.hidden, true);
  assert.equal(second.feedback.preference.hidden, false);
  second.feedback.keepShowing.onclick();
  assert.equal(second.feedback.panel.hidden, true);
  assert.deepEqual(
    JSON.parse(storage.get("soundalike:feedback-preference:v2")),
    { dismissals: 2, showAgain: true },
  );

  const third = loadExtension(fetchImpl, { results: [result], storage });
  await third.run();
  assert.equal(third.feedback.panel.hidden, false);
  third.feedback.dismiss.onclick();
  assert.equal(third.feedback.panel.hidden, true);
  assert.equal(feedbackUrls.length, 0);
});

test("stops future surveys when No is chosen after the second dismissal", async () => {
  const storage = new Map();
  const result = { title: "Candidate", artist: "Artist" };
  const fetchImpl = async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  };

  const first = loadExtension(fetchImpl, { results: [result], storage });
  await first.run();
  first.feedback.dismiss.onclick();

  const second = loadExtension(fetchImpl, { results: [result], storage });
  await second.run();
  second.feedback.dismiss.onclick();
  second.feedback.stopShowing.onclick();
  assert.deepEqual(
    JSON.parse(storage.get("soundalike:feedback-preference:v2")),
    { dismissals: 2, showAgain: false },
  );

  const third = loadExtension(fetchImpl, { results: [result], storage });
  await third.run();
  assert.equal(third.feedback.panel.hidden, true);
});

test("removes legacy cooldowns and shows the survey again", async () => {
  const storage = new Map([[
    "soundalike:feedback-suppression:v1",
    JSON.stringify({ outcome: "dismissed", at: Date.now() }),
  ]]);
  const result = { title: "Candidate", artist: "Artist" };
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, { ...recommendation, results: [result] });
  }, { results: [result], storage });

  await app.run();

  assert.equal(storage.has("soundalike:feedback-suppression:v1"), false);
  assert.equal(app.feedback.panel.hidden, false);
});

test("feedback records local and legacy API sources accurately", async () => {
  const result = { title: "Candidate", artist: "Artist" };
  for (const mode of ["local", "legacy"]) {
    let payload;
    const app = loadExtension(async (url, options) => {
      if (url.endsWith("/api/spicetify-feedback")) {
        payload = JSON.parse(options.body);
        return response(200, { receipt_sha256: "c".repeat(64) });
      }
      if (url.endsWith("/health")) {
        if (mode === "local") return response(200, { ok: true });
        throw new TypeError("connection refused");
      }
      if (mode === "legacy" && url.includes("/api/spicetify_recommend")) {
        return response(404, { error: "not found" });
      }
      return response(200, { ...recommendation, results: [result] });
    }, { results: [result] });
    await app.run();
    app.feedback.ratings[0].checked = true;
    app.feedback.ratings[0].onchange();
    await app.feedback.send.onclick();
    assert.equal(payload.source, mode === "local" ? "local" : "hosted");
    assert.equal(payload.api_version, mode);
  }
});

test("feedback markup is inline and exposes accessible form semantics", () => {
  assert.match(
    extensionSource,
    /<section class="sa-feedback" hidden aria-labelledby="sa-feedback-question">/,
  );
  assert.match(
    extensionSource,
    /<legend id="sa-feedback-question">How close were these matches\?<\/legend>/,
  );
  assert.match(
    extensionSource,
    /Do you want this survey to keep showing after future searches\?/,
  );
  assert.match(
    extensionSource,
    /class="sa-feedback-keep-showing" type="button">Yes<\/button>/,
  );
  assert.match(
    extensionSource,
    /class="sa-feedback-stop-showing" type="button">No<\/button>/,
  );
  assert.equal(
    (extensionSource.match(/name="sa-feedback-rating" value=/g) || []).length,
    3,
  );
  assert.equal(
    (extensionSource.match(/name="sa-feedback-reason" value=/g) || []).length,
    5,
  );
  assert.match(extensionSource, /maxlength="280"/);
  assert.match(extensionSource, /Please don’t include personal information\./);
  assert.match(
    extensionSource,
    /class="sa-feedback-status" role="status" aria-live="polite"/,
  );
  assert.equal(extensionSource.includes("dialog"), false);
});
