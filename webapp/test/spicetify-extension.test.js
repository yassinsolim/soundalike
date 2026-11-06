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
  let registeredMenuItems = 0;
  let currentPage;
  const notifications = [];
  const history = [];
  const played = [];
  const contextPlays = [];
  const playerState = { context: null, item: null };
  const queuedSongs = [];
  const queueState = [];
  const nowPlayingMenuItems = [];
  const songChangeListeners = [];
  const graphqlRequests = [];
  const cosmosRequests = [];
  const logs = [];
  const historyListeners = [];
  const storage = options.storage || new Map();
  const bestRequests = [];
  const tabFocus = [];
  // Indexes 0 and 1 stay Best matches and Sounds like; a page exposes only the
  // tabs its markup draws, in page order.
  const tabs = ["best", "sounds", "vibe"].map((view) => ({
    dataset: { view },
    onclick: null,
    onkeydown: null,
    focus() {
      tabFocus.push(view);
      context.document.activeElement = this;
    },
  }));
  const offerButton = { onclick: null };
  const languageToggleButton = { onclick: null };
  // Best matches are offline unless a test opts in, so the older tests keep
  // exercising the Sounds like flow they were written for.
  function fetchWithBestMatches(url, init) {
    if (String(url).startsWith("https://soundalike.yassin.app/api/anysong/")) {
      bestRequests.push(String(url));
      return options.bestFetch
        ? options.bestFetch(String(url), init)
        : Promise.reject(new TypeError("Best matches are offline in this test"));
    }
    return fetchImpl(url, init);
  }
  const scaledSetTimeout = options.timerScale
    ? (callback, ms = 0, ...args) =>
      setTimeout(callback, Math.round(ms * options.timerScale), ...args)
    : setTimeout;
  const rows = (options.results || []).map((_, index) => {
    const classes = new Set();
    return {
      dataset: { index: String(index) },
      hidden: index >= 20,
      style: {},
      classList: {
        toggle(name, force = !classes.has(name)) {
          if (force) classes.add(name);
          else classes.delete(name);
          return force;
        },
        contains(name) {
          return classes.has(name);
        },
      },
      querySelector() {
        return null;
      },
    };
  });
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
    if (selector === ".sa-language-toggle") return languageToggleButton;
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
    const tab = selector.match(/^\.sa-tab\[data-view="(\w+)"\]$/);
    if (tab) return tabs.find((item) => item.dataset.view === tab[1]) || null;
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
        if (selector === ".sa-best-offer") {
          return this.innerHTML.includes('class="sa-best-offer"')
            ? offerButton
            : null;
        }
        if (selector === ".sa-language-toggle") {
          return this.innerHTML.includes('class="sa-language-toggle"')
            ? languageToggleButton
            : null;
        }
        return queryPage(selector);
      },
      querySelectorAll(selector) {
        if (selector === ".sa-tab") {
          const at = (tab) => this.innerHTML.indexOf(`data-view="${tab.dataset.view}"`);
          return tabs.filter((tab) => at(tab) >= 0).sort((left, right) => at(left) - at(right));
        }
        return queryAllPage(selector);
      },
      contains(node) {
        return this.className === "sa-results-host" &&
          (tabs.includes(node) || node === offerButton);
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
    fetch: fetchWithBestMatches,
    setTimeout: scaledSetTimeout,
    console: {
      error() {},
      log(...values) {
        logs.push(values.join(" "));
      },
      warn() {},
    },
    document: {
      activeElement: null,
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
  const player = {
    data: options.playingTrackUri ? { item: { uri: options.playingTrackUri } } : null,
    addEventListener(type, callback) {
      if (type === "songchange") songChangeListeners.push(callback);
    },
    async playUri(uri) {
      played.push(uri);
    },
  };
  context.Spicetify = {
    ContextMenu: {
      Item: class {
        constructor(_label, callback) {
          handler = callback;
        }
        register() {
          registeredMenuItems += 1;
        }
      },
    },
    // Spicetify 2.46's low-level menu items, which see each menu's raw props.
    ...(options.contextMenuV2 ? {
      ContextMenuV2: {
        Item: class {
          constructor(config) {
            this.config = config;
          }
          register() {
            nowPlayingMenuItems.push(this.config);
          }
        },
        parseProps(props) {
          if (!props) return undefined;
          const uri = props.uri ?? props.item?.uri ?? props.reference?.uri;
          const uris = props.uris ?? (uri ? [uri] : undefined);
          return uris ? [uris, undefined, props.contextUri ?? props.context?.uri] : undefined;
        },
      },
    } : {}),
    ...(options.spicetifyVersion === undefined
      ? {}
      : { Config: { version: options.spicetifyVersion } }),
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
      ...(options.spotifyVersion === undefined
        ? {}
        : { version: options.spotifyVersion }),
      ...(options.playerApi ? {
        PlayerAPI: {
          async play(playContext, origin, playOptions) {
            contextPlays.push({ context: playContext, origin, options: playOptions });
            if (options.playerApi === "fails") {
              throw new Error("Command failed with reasons 'not_playing_context'");
            }
            playerState.context = { uri: playContext.uri };
            playerState.item = { uri: playOptions?.skipTo?.uri };
          },
          getState() {
            return playerState;
          },
          getQueue() {
            return { queued: queueState };
          },
          async addToQueue(items) {
            queuedSongs.push(items);
            // Spotify shows queued songs a moment later, and can drop songs
            // queued just as a song starts.
            await new Promise((resolve) => setTimeout(resolve, 5));
            if (options.dropFirstQueueAdd && queuedSongs.length === 1) return;
            queueState.push(...items);
          },
        },
      } : {}),
    },
    Player: player,
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
    contextPlays,
    playerState,
    queuedSongs,
    nowPlayingMenuItems,
    player,
    songChangeListeners,
    graphqlRequests,
    cosmosRequests,
    logs,
    nativeChildren: pageContainer.children,
    languageStatus,
    languageLoading,
    feedback,
    rows,
    bestRequests,
    tabs,
    tab(view) {
      return tabs.find((item) => item.dataset.view === view);
    },
    tabFocus,
    offerButton,
    get activeElement() {
      return context.document.activeElement;
    },
    get registeredMenuItems() {
      return registeredMenuItems;
    },
    historyListeners,
    focusOn(node) {
      context.document.activeElement = node;
    },
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
  assert.deepEqual(app.bestRequests, []);
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
    playerApi: true,
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

  // Playing a row follows the order on screen, not the model order.
  await findElement(
    app.rows[0].tree,
    (node) => node.props?.className === "sa-row-content",
  ).props.onDoubleClick();
  const [play] = JSON.parse(JSON.stringify(app.contextPlays));
  assert.deepEqual(
    play.context.pages[0].items.map((item) => item.uri),
    [1, 0, ...Array.from({ length: 18 }, (_, index) => index + 2)]
      .map((index) => `spotify:track:result${index}`),
  );
  assert.deepEqual(play.options, { skipTo: { index: 1, uri: "spotify:track:result0" } });
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

const bestSeedQuery = {
  id: 908604612,
  title: "Blinding Lights",
  artist: "The Weeknd",
  source: "catalog",
};

const bestFixture = [
  {
    title: "Save Your Tears",
    artist: "The Weeknd",
    ranks: { melody: 1, samples: 3, "<b>": 2, sonic: 0 },
  },
  { title: "Chanson Proche", artist: "Artiste Deux", ranks: { samples: 2 } },
  { title: " ", artist: "Nobody", ranks: { melody: 4 } },
];

const soundsFixture = [
  { title: "Take My Breath", artist: "The Weeknd", bpm: 121.7 },
];

const bestCacheKey = "best:anysong-best-rrf-language-vibe-v1:same:spotify:track:test";

function bestMatchesBody(best = bestFixture, overrides = {}) {
  return {
    ok: true,
    cached: false,
    query: bestSeedQuery,
    results: { best, melody: [], samples: [], sonic: [] },
    timings_ms: { total: 2345 },
    ...overrides,
  };
}

function spotifyTrackFixture(title, artist, id) {
  return {
    __typename: "Track",
    name: title,
    uri: `spotify:track:${id}`,
    albumOfTrack: {
      name: `${title} (Single)`,
      uri: `spotify:album:${id}`,
      coverArt: {
        sources: [{ url: `https://i.scdn.co/image/${id}`, width: 300 }],
      },
    },
    artists: {
      items: [{ uri: `spotify:artist:${id}`, profile: { name: artist } }],
    },
  };
}

// Both lists succeed unless a test overrides them. Every track has a known
// lyrics language, so no lookup waits for a retry.
function loadBestMatchesExtension({
  bestFetch = async () => response(200, bestMatchesBody()),
  soundsResponse = async () =>
    response(200, { ...recommendation, results: soundsFixture }),
  localCompanion = false,
  ...options
} = {}) {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    if (url.endsWith("/health")) {
      if (localCompanion) return response(200, { ok: true });
      throw new TypeError("connection refused");
    }
    return soundsResponse(url);
  }, {
    results: bestFixture,
    spotifyTracks: {
      "Save Your Tears The Weeknd":
        spotifyTrackFixture("Save Your Tears", "The Weeknd", "best0"),
      "Chanson Proche Artiste Deux":
        spotifyTrackFixture("Chanson Proche", "Artiste Deux", "best1"),
      "Take My Breath The Weeknd":
        spotifyTrackFixture("Take My Breath", "The Weeknd", "sounds0"),
    },
    languageByTrackId: { test: "en", best0: "en", best1: "fr", sounds0: "en" },
    bestFetch,
    ...options,
  });
  return Object.assign(app, { urls });
}

function hasLog(app, text) {
  return app.logs.some((line) => line.includes(text));
}

function errorNotifications(app) {
  return app.notifications.filter(([, isError]) => isError === true);
}

const noPreview = () => response(404, { ok: false, error: "not found" });
const hostedMiss = async () => response(422, {
  ok: false,
  error: "This track is not in the hosted library.",
});

test("opens on Best matches with Sounds like one tab away", async () => {
  const app = loadBestMatchesExtension();

  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  await waitFor(() =>
    app.urls.some((url) => url.includes("/api/spicetify_recommend")));

  assert.equal(app.bestRequests.length, 1);
  const bestUrl = new URL(app.bestRequests[0]);
  assert.equal(
    `${bestUrl.origin}${bestUrl.pathname}`,
    "https://soundalike.yassin.app/api/anysong/similar",
  );
  assert.equal(bestUrl.searchParams.get("title"), "Blinding Lights");
  assert.equal(bestUrl.searchParams.get("artist"), "The Weeknd");
  assert.equal(bestUrl.searchParams.get("k"), "20");
  assert.equal(bestUrl.searchParams.get("lang"), "same");
  assert.equal(
    app.urls.filter((url) => url.includes("/api/spicetify_recommend")).length,
    1,
  );
  assert.deepEqual(app.history, ["/soundalike"]);
  assert.deepEqual(errorNotifications(app), []);

  const html = app.currentPage.innerHTML;
  assert.match(html, /<div class="sa-kicker">BEST MATCHES · HOSTED LIBRARY<\/div>/);
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(html, /id="sa-tab-best"[^>]*aria-selected="true" tabindex="0"/);
  assert.match(html, /id="sa-tab-sounds"[^>]*aria-selected="false" tabindex="-1"/);
  assert.match(html, /<span class="sa-tag">Melody \+ samples<\/span>/);
  assert.match(html, /<span class="sa-tag">In our library<\/span>/);
  assert.match(html, /<span class="sa-tag">Matched in 2\.3 s<\/span>/);
  assert.doesNotMatch(html, /Matched to/);
  assert.match(html, /<span class="sa-why"> · melody #1 · samples #3<\/span>/);
  assert.doesNotMatch(html, /&lt;b&gt;/);
  assert.equal((html.match(/<div class="sa-row"/g) || []).length, 2);
  assert.doesNotMatch(html, /<div class="sa-row"[^>]* hidden>/);
  assert.doesNotMatch(html, /<div class="sa-bpm">BPM<\/div>/);
  assert.doesNotMatch(html, /<section class="sa-feedback"/);

  const why = findElement(
    app.rows[0].tree,
    (node) => node.props?.className === "sa-why",
  );
  assert.equal(why.props.children, " · melody #1 · samples #3");
  assert.equal(
    findElement(app.rows[0].tree, (node) => node.props?.className === "sa-bpm"),
    null,
  );
  assert.equal(app.rows[0].dataset.uri, "spotify:track:best0");
  // Best matches skip the lyrics-language gate.
  assert.equal(app.rows[1].hidden, false);
  assert.equal(app.rows[1].dataset.uri, "spotify:track:best1");
  assert.equal(app.feedback.panel.dataset.ready, undefined);
  assert.equal(app.feedback.panel.hidden, true);
});

test("Best matches language switch requests and caches all-language results separately", async () => {
  const app = loadBestMatchesExtension({
    bestFetch: async (url) => {
      const langMode = new URL(url).searchParams.get("lang");
      return response(200, bestMatchesBody(bestFixture, {
        query: { ...bestSeedQuery, language: "en", language_mode: langMode, hidden: 2 },
      }));
    },
  });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.match(
    app.currentPage.innerHTML,
    /English · 2 other-language matches hidden from the top 20/,
  );
  const toggle = app.currentPage.querySelector(".sa-language-toggle");
  assert.ok(toggle);
  toggle.onclick();
  await waitFor(() => app.bestRequests.length === 2 &&
    app.currentPage.innerHTML.includes("Use same-language matches"));
  assert.equal(new URL(app.bestRequests[1]).searchParams.get("lang"), "all");
  assert.match(app.currentPage.innerHTML, /English · 2 other-language matches in the top 20/);
});

test("switching back to same-language Best matches reuses the saved list", async () => {
  const storage = new Map();
  const app = loadBestMatchesExtension({
    storage,
    bestFetch: async (url) => {
      const langMode = new URL(url).searchParams.get("lang");
      return response(200, bestMatchesBody(bestFixture, {
        query: { ...bestSeedQuery, language: "en", language_mode: langMode, hidden: 1 },
      }));
    },
  });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  const toggle = app.currentPage.querySelector(".sa-language-toggle");
  toggle.onclick();
  await waitFor(() => app.currentPage.innerHTML.includes("Use same-language matches"));
  assert.match(app.currentPage.innerHTML, /English · 1 other-language match in the top 20/);
  toggle.onclick();
  await waitFor(() =>
    app.currentPage.innerHTML.includes('aria-pressed="false">Show all languages'));
  assert.match(
    app.currentPage.innerHTML,
    /English · 1 other-language match hidden from the top 20/,
  );
  assert.deepEqual(
    app.bestRequests.map((url) => new URL(url).searchParams.get("lang")),
    ["same", "all"],
  );
  await waitFor(() => {
    const saved = JSON.parse(storage.get("soundalike:spicetify-cache:v9") || "{}");
    return Boolean(
      saved.recommendations?.[bestCacheKey] &&
      saved.recommendations?.[bestCacheKey.replace(":same:", ":all:")],
    );
  }, 1000);
});

test("leaves Sounds like unfiltered and without the language switch", async () => {
  const app = loadBestMatchesExtension({
    bestFetch: async () => response(200, bestMatchesBody(bestFixture, {
      query: { ...bestSeedQuery, language: "en", language_mode: "same", hidden: 2 },
    })),
  });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.match(app.currentPage.innerHTML, /<div class="sa-language-filter" role="status">/);

  app.tabs[1].onclick();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");
  const html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-sounds"/);
  assert.match(html, /Dual-Sonic64/);
  assert.doesNotMatch(html, /<div class="sa-language-filter"/);
  assert.equal(app.currentPage.querySelector(".sa-language-toggle"), null);
  assert.equal(app.bestRequests.length, 1);
});

test("shows all languages without a switch when the seed language is unknown", async () => {
  const app = loadBestMatchesExtension({
    bestFetch: async () => response(200, bestMatchesBody(bestFixture, {
      query: { ...bestSeedQuery, language: "unknown", language_mode: "same", hidden: 0 },
    })),
  });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  const html = app.currentPage.innerHTML;
  assert.match(
    html,
    /<div class="sa-language-filter" role="status"><span>Language uncertain; showing all languages\.<\/span><\/div>/,
  );
  assert.equal(app.currentPage.querySelector(".sa-language-toggle"), null);
});

test("switches result lists with tabs and arrow keys and keeps the choice", async () => {
  const app = loadBestMatchesExtension();
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  app.tabs[1].onclick();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");
  const html = app.currentPage.innerHTML;
  assert.match(html, /<div class="sa-kicker">SOUNDS LIKE · HOSTED LIBRARY<\/div>/);
  assert.match(html, /class="sa-panel sa-view-sounds"/);
  assert.match(html, /id="sa-tab-sounds"[^>]*aria-selected="true"/);
  assert.match(html, /Dual-Sonic64/);
  assert.match(html, /<div class="sa-bpm">BPM<\/div>/);
  assert.match(html, /<section class="sa-feedback"/);
  assert.doesNotMatch(html, /Showing Sounds like instead/);
  assert.equal(app.feedback.panel.hidden, false);
  assert.equal(app.rows[0].dataset.uri, "spotify:track:sounds0");
  assert.notEqual(
    findElement(app.rows[0].tree, (node) => node.props?.className === "sa-bpm"),
    null,
  );
  assert.deepEqual(app.tabFocus, ["sounds"]);

  const handledKeys = [];
  const press = (tab, key) => {
    app.tabs[tab].onkeydown({
      key,
      preventDefault() {
        handledKeys.push(key);
      },
    });
  };
  const view = () =>
    app.currentPage.innerHTML.match(/class="sa-panel sa-view-(\w+)"/)[1];
  press(1, "ArrowLeft");
  assert.equal(view(), "best");
  assert.equal(app.rows[0].dataset.uri, "spotify:track:best0");
  press(0, "End");
  assert.equal(view(), "sounds");
  press(1, "Home");
  assert.equal(view(), "best");
  press(0, "ArrowRight");
  assert.equal(view(), "sounds");
  press(1, "a");
  assert.equal(view(), "sounds");
  assert.deepEqual(handledKeys, ["ArrowLeft", "End", "Home", "ArrowRight"]);
  assert.deepEqual(
    app.tabFocus,
    ["sounds", "best", "sounds", "best", "sounds"],
  );

  app.navigate("/album/elsewhere");
  assert.equal(app.currentPage, undefined);
  app.navigate("/soundalike");
  await waitFor(() => app.currentPage !== undefined);
  assert.equal(view(), "sounds");
  assert.deepEqual(app.history, ["/soundalike", "/album/elsewhere", "/soundalike"]);
});

test("falls back to Sounds like when a song has no Deezer preview", async () => {
  const app = loadBestMatchesExtension({ bestFetch: async () => noPreview() });

  await app.run();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");

  assert.equal(app.bestRequests.length, 1);
  assert.deepEqual(errorNotifications(app), []);
  let html = app.currentPage.innerHTML;
  assert.match(html, /<div class="sa-kicker">SOUNDS LIKE · HOSTED LIBRARY<\/div>/);
  assert.match(html, /class="sa-panel sa-view-sounds"/);
  assert.match(
    html,
    /<p class="sa-note">This song isn't on Deezer with a preview, so it has no best matches\. Showing Sounds like instead\.<\/p>/,
  );
  assert.equal(app.feedback.panel.hidden, false);

  app.tabs[0].onclick();
  html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(
    html,
    /<div class="sa-state sa-state-error" role="alert">This song isn't on Deezer with a preview, so it has no best matches\.<\/div>/,
  );
  assert.doesNotMatch(html, /<div class="sa-row"/);

  app.tabs[1].onclick();
  assert.doesNotMatch(app.currentPage.innerHTML, /Showing Sounds like instead/);
});

test("reports both failures once without opening an empty page", async () => {
  const app = loadBestMatchesExtension({
    bestFetch: async () => noPreview(),
    soundsResponse: hostedMiss,
  });

  const page = await app.run();

  assert.equal(page, undefined);
  assert.deepEqual(app.history, []);
  const errors = errorNotifications(app);
  assert.equal(errors.length, 1);
  assert.match(errors[0][0], /not in the hosted library/);
  assert.match(errors[0][0], /isn't on Deezer with a preview/);
});

test("keeps Best matches open when the hosted library misses the song", async () => {
  const app = loadBestMatchesExtension({ soundsResponse: hostedMiss });

  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  assert.match(app.currentPage.innerHTML, /class="sa-panel sa-view-best"/);
  app.tabs[1].onclick();
  await waitFor(() => /role="alert"/.test(app.currentPage.innerHTML));
  const html = app.currentPage.innerHTML;
  assert.match(html, /<div class="sa-kicker">SOUNDS LIKE<\/div>/);
  assert.match(
    html,
    /<div class="sa-state sa-state-error" role="alert">[^<]*not in the hosted library[^<]*<\/div>/,
  );
  assert.doesNotMatch(html, /<section class="sa-feedback"/);
  assert.deepEqual(errorNotifications(app), []);
});

test("shows Sounds like loading until its list arrives", async () => {
  const gate = deferred();
  const app = loadBestMatchesExtension({
    soundsResponse: async () => {
      await gate.promise;
      return response(200, { ...recommendation, results: soundsFixture });
    },
  });

  await app.run();
  assert.match(app.currentPage.innerHTML, /class="sa-panel sa-view-best"/);
  app.tabs[1].onclick();
  assert.match(app.currentPage.innerHTML, /<div class="sa-kicker">SOUNDS LIKE<\/div>/);
  assert.match(
    app.currentPage.innerHTML,
    /<div class="sa-state" role="status">Finding songs that sound alike…<\/div>/,
  );

  gate.resolve();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");
  assert.match(app.currentPage.innerHTML, /Dual-Sonic64/);
  assert.equal(app.rows[0].dataset.uri, "spotify:track:sounds0");
});

test("opens with Best matches loading once Sounds like has waited", async () => {
  const gate = deferred();
  const app = loadBestMatchesExtension({
    timerScale: 0.01,
    bestFetch: async () => {
      await gate.promise;
      return response(200, bestMatchesBody(bestFixture, {
        query: { ...bestSeedQuery, title: "Blinding Lights - Live" },
      }));
    },
  });

  await app.run();
  let html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(
    html,
    /<div class="sa-state" role="status">Listening for best matches… songs outside the library take a few seconds\.<\/div>/,
  );
  assert.equal(app.rows[0].tree, undefined);

  gate.resolve();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  html = app.currentPage.innerHTML;
  assert.match(html, /<span class="sa-tag">Melody \+ samples<\/span>/);
  assert.match(html, /Matched to “Blinding Lights - Live” by The Weeknd\./);
  assert.equal(app.rows[0].dataset.uri, "spotify:track:best0");
});

test("switches to Sounds like when slow Best matches fail", async () => {
  const gate = deferred();
  const app = loadBestMatchesExtension({
    timerScale: 0.01,
    bestFetch: async () => {
      await gate.promise;
      return noPreview();
    },
  });

  await app.run();
  assert.match(app.currentPage.innerHTML, /class="sa-panel sa-view-best"/);
  gate.resolve();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");

  const html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-sounds"/);
  assert.match(html, /no best matches\. Showing Sounds like instead\./);
  assert.equal(app.feedback.panel.hidden, false);
});

test("keeps a chosen list when Best matches fail later", async () => {
  const gate = deferred();
  const app = loadBestMatchesExtension({
    timerScale: 0.01,
    bestFetch: async () => {
      await gate.promise;
      return noPreview();
    },
  });

  await app.run();
  app.tabs[0].onclick();
  assert.deepEqual(app.tabFocus, ["best"]);
  gate.resolve();
  await waitFor(() => /role="alert"/.test(app.currentPage.innerHTML));

  const html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(html, /isn't on Deezer with a preview/);
  assert.doesNotMatch(html, /Showing Sounds like instead/);
});

test("retries a busy matcher before falling back to Sounds like", async () => {
  let calls = 0;
  const recovered = loadBestMatchesExtension({
    timerScale: 0.01,
    bestFetch: async () => (++calls === 1
      ? response(503, { ok: false, error: "busy" })
      : response(200, bestMatchesBody(bestFixture, { cached: true }))),
  });
  await recovered.run();
  await waitFor(() => hasLog(recovered, "Best matches metadata settled"));
  assert.equal(recovered.bestRequests.length, 2);
  let html = recovered.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(html, /<span class="sa-tag">In our library<\/span>/);
  assert.doesNotMatch(html, /Matched in/);

  const busy = loadBestMatchesExtension({
    timerScale: 0.01,
    bestFetch: async () => response(503, { ok: false, error: "busy" }),
  });
  await busy.run();
  await waitFor(() =>
    /Showing Sounds like instead/.test(busy.currentPage?.innerHTML || ""));
  assert.equal(busy.bestRequests.length, 3);
  html = busy.currentPage.innerHTML;
  assert.match(
    html,
    /The matcher is busy right now\. Try again in a moment\. Showing Sounds like instead\./,
  );
});

test("restores cached Best matches without asking the matcher again", async () => {
  const storage = new Map();
  const first = loadBestMatchesExtension({
    storage,
    bestFetch: async () => response(200, bestMatchesBody(bestFixture, {
      query: { ...bestSeedQuery, source: "live" },
    })),
  });
  await first.run();
  await waitFor(() => hasLog(first, "Best matches metadata settled"));
  let html = first.currentPage.innerHTML;
  assert.match(html, /<span class="sa-tag">Heard live just now<\/span>/);
  assert.match(html, /<span class="sa-tag">Matched in 2\.3 s<\/span>/);

  const saved = () => {
    const raw = storage.get("soundalike:spicetify-cache:v9");
    return raw ? JSON.parse(raw) : null;
  };
  await waitFor(() =>
    Boolean(
      saved()?.recommendations?.[bestCacheKey] &&
      saved().spotifyTracks?.["result:chanson proche::artiste deux"],
    ), 1000);
  assert.ok(storage.get("soundalike:spicetify-cache:v9").length < 10000);
  const entry = saved().recommendations[bestCacheKey].value;
  assert.deepEqual(
    Object.keys(entry).sort(),
    ["query", "results", "serverCached", "totalMs", "vibe"],
  );
  assert.deepEqual(
    entry.query,
    { title: "Blinding Lights", artist: "The Weeknd", source: "live",
      language: "unknown", languageMode: "same", hidden: 0, bpm: null },
  );
  assert.deepEqual(entry.vibe, []);
  assert.equal(entry.results.length, 2);
  assert.deepEqual(entry.results[0], {
    title: "Save Your Tears",
    artist: "The Weeknd",
    matchNote: "melody #1 · samples #3",
  });

  const second = loadBestMatchesExtension({
    storage,
    bestFetch: async () => response(500, { ok: false }),
  });
  await second.run();
  await waitFor(() => hasLog(second, "Best matches metadata settled"));

  assert.deepEqual(second.bestRequests, []);
  assert.deepEqual(second.urls.filter((url) => !url.endsWith("/health")), []);
  assert.deepEqual(second.graphqlRequests, []);
  assert.deepEqual(second.cosmosRequests, []);
  html = second.currentPage.innerHTML;
  assert.match(html, /<span class="sa-tag">Heard live earlier<\/span>/);
  assert.doesNotMatch(html, /Matched in/);
  assert.equal(second.rows[1].dataset.uri, "spotify:track:best1");
});

test("ignores an older search that finishes after a newer one opened", async () => {
  const soundsGate = deferred();
  const bestGate = deferred();
  const app = loadBestMatchesExtension({
    seedTrack: {
      name: "Song A",
      firstArtist: { items: [{ profile: { name: "Artist A" } }] },
    },
    languageByTrackId: {
      test: "en",
      other: "en",
      best0: "en",
      best1: "fr",
      sounds0: "en",
    },
    soundsResponse: async (url) => {
      if (url.includes("Song+A")) await soundsGate.promise;
      return response(200, { ...recommendation, results: soundsFixture });
    },
    bestFetch: async (url) => {
      if (url.includes("title=Song+A")) await bestGate.promise;
      return response(200, bestMatchesBody());
    },
  });

  const older = app.run("spotify:track:test");
  await waitFor(() => app.bestRequests.some((url) => url.includes("title=Song+A")));
  await app.run("spotify:track:other");
  const page = app.currentPage;
  assert.match(page.innerHTML, /<div class="sa-seed-title">Blinding Lights<\/div>/);
  assert.deepEqual(app.history, ["/soundalike"]);

  soundsGate.resolve();
  bestGate.resolve();
  await older;
  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.deepEqual(app.history, ["/soundalike"]);
  assert.equal(app.currentPage, page);
  assert.doesNotMatch(app.currentPage.innerHTML, /Song A/);
  assert.deepEqual(errorNotifications(app), []);
});

test("keeps a dismissed survey closed when switching lists", async () => {
  const app = loadBestMatchesExtension({ bestFetch: async () => noPreview() });
  await app.run();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");
  assert.equal(app.feedback.panel.hidden, false);

  app.feedback.dismiss.onclick();
  assert.equal(app.feedback.panel.hidden, true);
  // The harness reuses one panel; a real re-render creates a fresh one.
  app.feedback.panel.dataset = {};
  const settledCount = () =>
    app.logs.filter((line) => line.includes("Spotify metadata settled")).length;
  const before = settledCount();
  app.tabs[0].onclick();
  app.tabs[1].onclick();
  await waitFor(() => settledCount() > before);

  assert.equal(app.feedback.panel.dataset.ready, undefined);
  assert.equal(app.feedback.panel.hidden, true);
});

const localMiss = async () => response(200, {
  ok: false,
  error: "The local engine couldn't analyze this song.",
});

test("asks before sending a local companion search to Best matches", async () => {
  const app = loadBestMatchesExtension({ localCompanion: true });
  await app.run();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");

  assert.deepEqual(app.bestRequests, []);
  assert.deepEqual(
    app.urls.filter((url) => !url.endsWith("/health")),
    ["http://127.0.0.1:8787/api/recommend"],
  );
  let html = app.currentPage.innerHTML;
  assert.match(html, /<div class="sa-kicker">SOUNDS LIKE · LOCAL ENGINE<\/div>/);
  assert.match(html, /id="sa-tab-sounds"[^>]*aria-selected="true"/);
  assert.doesNotMatch(html, /<p class="sa-note">/);

  app.tabs[0].onclick();
  html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(html, /Your local companion keeps searches on this computer\./);
  assert.match(
    html,
    /<button class="sa-best-offer" type="button">Get best matches<\/button>/,
  );
  assert.deepEqual(app.bestRequests, []);

  app.offerButton.onclick();
  assert.match(app.currentPage.innerHTML, /Listening for best matches/);
  assert.deepEqual(app.tabFocus, ["best", "best"]);
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  assert.equal(app.bestRequests.length, 1);
  const bestUrl = new URL(app.bestRequests[0]);
  assert.equal(bestUrl.searchParams.get("title"), "Blinding Lights");
  assert.equal(bestUrl.searchParams.get("artist"), "The Weeknd");
  html = app.currentPage.innerHTML;
  assert.doesNotMatch(html, /<button class="sa-best-offer"/);
  assert.match(html, /<span class="sa-why"> · melody #1 · samples #3<\/span>/);
  assert.equal(app.rows[0].dataset.uri, "spotify:track:best0");
  assert.deepEqual(errorNotifications(app), []);
});

test("keeps the Best matches tab open when an offered search fails", async () => {
  const app = loadBestMatchesExtension({
    localCompanion: true,
    bestFetch: async () => noPreview(),
  });
  await app.run();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");

  app.tabs[0].onclick();
  app.offerButton.onclick();
  await waitFor(() => /role="alert"/.test(app.currentPage.innerHTML));

  const html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(html, /This song isn't on Deezer with a preview/);
  assert.equal(app.bestRequests.length, 1);
  assert.deepEqual(errorNotifications(app), []);
});

test("shows saved Best matches to a local companion without asking", async () => {
  const storage = new Map();
  const first = loadBestMatchesExtension({ storage });
  await first.run();
  await waitFor(() => hasLog(first, "Best matches metadata settled"));
  await waitFor(() =>
    Boolean(
      JSON.parse(storage.get("soundalike:spicetify-cache:v9") || "{}")
        .recommendations?.[bestCacheKey],
    ), 1000);

  const second = loadBestMatchesExtension({ storage, localCompanion: true });
  await second.run();
  await waitFor(() => second.feedback.panel.dataset.ready === "true");
  second.tabs[0].onclick();
  await waitFor(() => hasLog(second, "Best matches metadata settled"));

  assert.deepEqual(second.bestRequests, []);
  const html = second.currentPage.innerHTML;
  assert.doesNotMatch(html, /<button class="sa-best-offer"/);
  assert.match(html, /<span class="sa-tag">In our library<\/span>/);
  assert.equal(second.rows[0].dataset.uri, "spotify:track:best0");
});

test("offers Best matches when the local companion finds no Sounds like list", async () => {
  const app = loadBestMatchesExtension({
    localCompanion: true,
    soundsResponse: localMiss,
  });
  const page = await app.run();

  assert.notEqual(page, undefined);
  assert.deepEqual(app.bestRequests, []);
  assert.deepEqual(errorNotifications(app), []);
  assert.match(page.innerHTML, /class="sa-panel sa-view-best"/);
  assert.match(
    page.innerHTML,
    /<p class="sa-note">The local engine couldn't analyze this song\.<\/p>/,
  );
  assert.match(
    page.innerHTML,
    /<button class="sa-best-offer" type="button">Get best matches<\/button>/,
  );

  app.offerButton.onclick();
  assert.deepEqual(app.tabFocus, ["best"]);
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.equal(app.bestRequests.length, 1);
  assert.doesNotMatch(app.currentPage.innerHTML, /<p class="sa-note">/);
  assert.equal(app.activeElement, app.tabs[0]);
});

test("keeps keyboard focus on the tabs while a list arrives", async () => {
  const app = loadBestMatchesExtension({ localCompanion: true });
  await app.run();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");

  app.tabs[0].onclick();
  app.focusOn(app.offerButton);
  app.offerButton.onclick();
  assert.equal(app.activeElement, app.tabs[0]);
  assert.deepEqual(app.tabFocus, ["best", "best"]);

  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.equal(app.rows[0].dataset.uri, "spotify:track:best0");
  assert.equal(app.activeElement, app.tabs[0]);
  assert.deepEqual(app.tabFocus, ["best", "best", "best"]);
});

test("leaves focus outside the page alone when a list arrives", async () => {
  const gate = deferred();
  const app = loadBestMatchesExtension({
    timerScale: 0.01,
    bestFetch: async () => {
      await gate.promise;
      return response(200, bestMatchesBody());
    },
  });
  await app.run();
  assert.match(app.currentPage.innerHTML, /Listening for best matches/);

  const searchBox = { id: "spotify-search" };
  app.focusOn(searchBox);
  gate.resolve();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.equal(app.rows[0].dataset.uri, "spotify:track:best0");
  assert.equal(app.activeElement, searchBox);
  assert.deepEqual(app.tabFocus, []);
});

// Same vibe rows carry no ranks; the blank title is dropped like Best's.
const vibeFixture = [
  { id: 41, title: "Take My Breath", artist: "The Weeknd", score: 0.16667 },
  { id: 42, title: "Chanson Proche", artist: "Artiste Deux", score: 0.14286 },
  { id: 43, title: " ", artist: "Nobody", score: 0.125 },
];

function vibeMatchesBody(langMode = "same") {
  return bestMatchesBody(bestFixture, {
    query: {
      ...bestSeedQuery,
      language: "en",
      language_mode: langMode,
      hidden: 2,
      bpm: 171.4,
    },
    results: {
      best: bestFixture,
      vibe: vibeFixture,
      melody: [],
      samples: [],
      sonic: [],
    },
  });
}

const vibeFetch = async (url) =>
  response(200, vibeMatchesBody(new URL(url).searchParams.get("lang")));

test("offers Same vibe next to Best matches from the same hosted response", async () => {
  const app = loadBestMatchesExtension({ bestFetch: vibeFetch });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  let html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.match(
    html,
    />Best matches<\/button><button [^>]*id="sa-tab-vibe"[^>]*aria-selected="false" tabindex="-1">Same vibe<\/button><button [^>]*id="sa-tab-sounds"[^>]*>Sounds like<\/button>/,
  );
  assert.match(html, /English · 2 other-language matches hidden from the top 20/);

  app.tab("vibe").onclick();
  await waitFor(() => hasLog(app, "Same vibe metadata settled"));
  html = app.currentPage.innerHTML;
  assert.match(html, /<div class="sa-kicker">SAME VIBE · HOSTED LIBRARY<\/div>/);
  assert.match(html, /class="sa-panel sa-view-vibe" id="sa-panel" role="tabpanel" aria-labelledby="sa-tab-vibe"/);
  assert.match(html, /id="sa-tab-vibe"[^>]*aria-selected="true" tabindex="0"/);
  assert.match(
    html,
    /<p class="sa-blurb">Songs from these matches that share this song's tempo and overall sound, one per artist\.<\/p>/,
  );
  assert.match(
    html,
    /<span class="sa-tag">Tempo \+ sound<\/span><span class="sa-tag">171 BPM<\/span><span class="sa-tag">In our library<\/span><span class="sa-tag">Matched in 2\.3 s<\/span>/,
  );
  assert.match(
    html,
    /<div class="sa-language-filter" role="status"><span>English · other languages left out<\/span><button class="sa-language-toggle" type="button" aria-pressed="false">Show all languages<\/button><\/div>/,
  );
  assert.doesNotMatch(html, /hidden from the top/);
  assert.doesNotMatch(html, /class="sa-why"/);
  assert.doesNotMatch(html, /<div class="sa-bpm">BPM<\/div>/);
  assert.doesNotMatch(html, /<section class="sa-feedback"/);
  assert.equal((html.match(/<div class="sa-row"/g) || []).length, 2);
  assert.equal(app.rows[0].dataset.uri, "spotify:track:sounds0");
  assert.equal(app.rows[1].dataset.uri, "spotify:track:best1");
  for (const className of ["sa-why", "sa-bpm"]) {
    assert.equal(
      findElement(app.rows[0].tree, (node) => node.props?.className === className),
      null,
    );
  }
  // Both lists come from one any-song request; Same vibe asks nothing new.
  assert.equal(app.bestRequests.length, 1);
  assert.deepEqual(
    app.urls.filter((url) => !url.endsWith("/health") &&
      !url.includes("/api/spicetify_recommend")),
    [],
  );
  assert.deepEqual(app.tabFocus, ["vibe"]);
  assert.equal(app.feedback.panel.dataset.ready, undefined);
  assert.deepEqual(errorNotifications(app), []);

  const view = () =>
    app.currentPage.innerHTML.match(/class="sa-panel sa-view-(\w+)"/)[1];
  const press = (tab, key) => app.tab(tab).onkeydown({ key, preventDefault() {} });
  press("vibe", "ArrowRight");
  assert.equal(view(), "sounds");
  press("sounds", "ArrowRight");
  assert.equal(view(), "best");
  press("best", "ArrowRight");
  assert.equal(view(), "vibe");
  press("vibe", "ArrowLeft");
  assert.equal(view(), "best");
  press("best", "End");
  assert.equal(view(), "sounds");
  press("sounds", "Home");
  assert.equal(view(), "best");
});

test("Same vibe shares the Best matches language switch", async () => {
  const app = loadBestMatchesExtension({ bestFetch: vibeFetch });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  app.tab("vibe").onclick();
  app.currentPage.querySelector(".sa-language-toggle").onclick();
  assert.match(app.currentPage.innerHTML, /class="sa-panel sa-view-vibe"/);
  assert.match(
    app.currentPage.innerHTML,
    /<div class="sa-state" role="status">Finding songs with the same vibe…<\/div>/,
  );

  await waitFor(() => app.currentPage.innerHTML.includes("English · all languages"));
  let html = app.currentPage.innerHTML;
  assert.equal(new URL(app.bestRequests[1]).searchParams.get("lang"), "all");
  assert.match(html, /class="sa-panel sa-view-vibe"/);
  assert.match(html, /aria-pressed="true">Use same-language matches<\/button>/);
  assert.equal(app.tabFocus.at(-1), "vibe");

  app.tab("best").onclick();
  html = app.currentPage.innerHTML;
  assert.match(html, /English · 2 other-language matches in the top 20/);
  assert.match(html, /aria-pressed="true">Use same-language matches<\/button>/);

  app.tab("vibe").onclick();
  app.currentPage.querySelector(".sa-language-toggle").onclick();
  await waitFor(() =>
    app.currentPage.innerHTML.includes("English · other languages left out"));
  assert.deepEqual(
    app.bestRequests.map((url) => new URL(url).searchParams.get("lang")),
    ["same", "all"],
  );
});

test("leaves Same vibe out when the matcher sends no vibe list", async () => {
  const app = loadBestMatchesExtension();
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  assert.doesNotMatch(app.currentPage.innerHTML, /sa-tab-vibe|Same vibe/);
  assert.equal(app.tab("vibe").onclick, null);
  app.tabs[0].onkeydown({ key: "ArrowRight", preventDefault() {} });
  assert.match(app.currentPage.innerHTML, /class="sa-panel sa-view-sounds"/);
});

test("adds the Same vibe tab once slow Best matches arrive", async () => {
  const gate = deferred();
  const app = loadBestMatchesExtension({
    timerScale: 0.01,
    bestFetch: async (url) => {
      await gate.promise;
      return vibeFetch(url);
    },
  });
  await app.run();
  assert.match(app.currentPage.innerHTML, /Listening for best matches/);
  assert.doesNotMatch(app.currentPage.innerHTML, /sa-tab-vibe/);

  gate.resolve();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.match(app.currentPage.innerHTML, /id="sa-tab-vibe"[^>]*aria-selected="false"/);
});

test("returns to Best matches when a language reload brings no Same vibe list", async () => {
  const app = loadBestMatchesExtension({
    bestFetch: async (url) => {
      const langMode = new URL(url).searchParams.get("lang");
      return response(200, langMode === "all"
        ? bestMatchesBody(bestFixture, {
          query: { ...bestSeedQuery, language: "en", language_mode: "all", hidden: 2 },
        })
        : vibeMatchesBody(langMode));
    },
  });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  app.tab("vibe").onclick();
  app.currentPage.querySelector(".sa-language-toggle").onclick();

  await waitFor(() => app.bestRequests.length === 2 &&
    /class="sa-panel sa-view-best"/.test(app.currentPage.innerHTML));
  const html = app.currentPage.innerHTML;
  assert.doesNotMatch(html, /sa-tab-vibe/);
  assert.match(html, /English · 2 other-language matches in the top 20/);
  assert.equal(app.tabFocus.at(-1), "best");
});

test("keeps Same vibe out of local companion searches", async () => {
  const app = loadBestMatchesExtension({ localCompanion: true, bestFetch: vibeFetch });
  await app.run();
  await waitFor(() => app.feedback.panel.dataset.ready === "true");
  assert.doesNotMatch(app.currentPage.innerHTML, /sa-tab-vibe/);

  app.tabs[0].onclick();
  app.offerButton.onclick();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.equal(app.bestRequests.length, 1);
  const html = app.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-best"/);
  assert.doesNotMatch(html, /sa-tab-vibe|Same vibe/);
  assert.equal(app.tab("vibe").onclick, null);
});

test("restores a saved Same vibe list without asking the matcher again", async () => {
  const storage = new Map();
  const first = loadBestMatchesExtension({ storage, bestFetch: vibeFetch });
  await first.run();
  await waitFor(() => hasLog(first, "Best matches metadata settled"));
  first.tab("vibe").onclick();
  await waitFor(() => hasLog(first, "Same vibe metadata settled"));
  const saved = () =>
    JSON.parse(storage.get("soundalike:spicetify-cache:v9") || "{}");
  await waitFor(() => Boolean(
    saved().recommendations?.[bestCacheKey] &&
    saved().spotifyTracks?.["result:take my breath::the weeknd"] &&
    saved().spotifyTracks?.["result:chanson proche::artiste deux"],
  ), 1000);
  const entry = saved().recommendations[bestCacheKey].value;
  assert.deepEqual(entry.vibe, [
    { title: "Take My Breath", artist: "The Weeknd" },
    { title: "Chanson Proche", artist: "Artiste Deux" },
  ]);
  assert.equal(entry.query.bpm, 171.4);

  const second = loadBestMatchesExtension({
    storage,
    bestFetch: async () => response(500, { ok: false }),
  });
  await second.run();
  await waitFor(() => hasLog(second, "Best matches metadata settled"));
  second.tab("vibe").onclick();
  await waitFor(() => hasLog(second, "Same vibe metadata settled"));

  assert.deepEqual(second.bestRequests, []);
  assert.deepEqual(second.urls.filter((url) => !url.endsWith("/health")), []);
  assert.deepEqual(second.graphqlRequests, []);
  const html = second.currentPage.innerHTML;
  assert.match(html, /class="sa-panel sa-view-vibe"/);
  assert.match(html, /<span class="sa-tag">171 BPM<\/span><span class="sa-tag">In our library<\/span>/);
  assert.doesNotMatch(html, /Matched in/);
  assert.equal(second.rows[0].dataset.uri, "spotify:track:sounds0");
  assert.equal(second.rows[1].dataset.uri, "spotify:track:best1");
});

function loadVersionedExtension(spicetifyVersion, spotifyVersion) {
  const urls = [];
  const app = loadExtension(async (url) => {
    urls.push(url);
    return response(200, { ok: false });
  }, { spicetifyVersion, spotifyVersion, contextMenuV2: true });
  return { app, urls };
}

test("stays off where old Spicetify would empty Spotify's right-click menu", async () => {
  for (const [spicetifyVersion, spotifyVersion] of [
    ["2.45.3", "1.3.5.3058"],
    ["2.46.0", "1.3.5.3058"],
    ["2.45.9", "1.3.6.100.g1234abcd"],
    ["1.2.0", "1.4.0.1"],
    ["v2.40.0", "2.0.0.1"],
  ]) {
    const { app, urls } = loadVersionedExtension(spicetifyVersion, spotifyVersion);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const label = `${spicetifyVersion} on ${spotifyVersion}`;
    assert.equal(app.registeredMenuItems, 0, label);
    assert.equal(app.nowPlayingMenuItems.length, 0, label);
    assert.equal(app.historyListeners.length, 0, label);
    assert.equal(app.songChangeListeners.length, 0, label);
    assert.deepEqual(urls, [], label);
    assert.equal(app.notifications.length, 1, label);
    const [message, isError, timeout] = app.notifications[0];
    assert.equal(
      message,
      `Find soundalikes is off: Spicetify ${spicetifyVersion} can't add ` +
        'right-click items on this Spotify version. Run "spicetify update" ' +
        "to turn it on.",
      label,
    );
    assert.equal(isError, false, label);
    assert.equal(timeout, 10000, label);
  }
});

test("adds the menu item when Spicetify can draw it or the versions are unknown", async () => {
  for (const [spicetifyVersion, spotifyVersion] of [
    ["2.46.1", "1.3.5.3058"],
    ["2.46.4", "1.3.7.1234.gabcdef01"],
    ["3.0.0-rc.1", "1.3.6.100"],
    ["2.45.3", "1.3.4.479"],
    ["2.45.3", "1.2.99.1"],
    ["2.45.3", "0.9.9"],
    [undefined, "1.3.5.3058"],
    ["Dev", "1.3.5.3058"],
    ["2.45.3", undefined],
    ["2.45.3", "unknown"],
  ]) {
    const { app, urls } = loadVersionedExtension(spicetifyVersion, spotifyVersion);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const label = `${spicetifyVersion} on ${spotifyVersion}`;
    assert.equal(app.registeredMenuItems, 1, label);
    assert.equal(app.nowPlayingMenuItems.length, 1, label);
    assert.equal(app.historyListeners.length, 1, label);
    assert.equal(app.songChangeListeners.length, 1, label);
    assert.deepEqual(urls, ["http://127.0.0.1:8787/health"], label);
    assert.deepEqual(app.notifications, [], label);
  }
});

const plain = (value) => JSON.parse(JSON.stringify(value));
const rowContent = (row) =>
  findElement(row.tree, (node) => node.props?.className === "sa-row-content");

test("plays Best matches on in the shown order from the chosen row", async () => {
  const app = loadBestMatchesExtension({ playerApi: true });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  await rowContent(app.rows[1]).props.onDoubleClick();
  assert.deepEqual(plain(app.contextPlays), [{
    context: {
      uri: "spotify:app:soundalike",
      pages: [{
        items: [{ uri: "spotify:track:best0" }, { uri: "spotify:track:best1" }],
      }],
    },
    origin: { featureIdentifier: "soundalike" },
    options: { skipTo: { index: 1, uri: "spotify:track:best1" } },
  }]);

  const playButton = findElement(
    app.rows[0].tree,
    (node) => node.props?.className === "sa-play",
  );
  await playButton.props.onClick({ stopPropagation() {} });
  await waitFor(() => app.contextPlays.length === 2);
  assert.deepEqual(
    plain(app.contextPlays[1].options),
    { skipTo: { index: 0, uri: "spotify:track:best0" } },
  );
  assert.deepEqual(app.played, []);
  assert.deepEqual(errorNotifications(app), []);
});

test("plays the Same vibe list from its own tab", async () => {
  const app = loadBestMatchesExtension({ bestFetch: vibeFetch, playerApi: true });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  app.tab("vibe").onclick();
  await waitFor(() => hasLog(app, "Same vibe metadata settled"));

  await rowContent(app.rows[1]).props.onDoubleClick();
  assert.deepEqual(plain(app.contextPlays), [{
    context: {
      uri: "spotify:app:soundalike",
      pages: [{
        items: [{ uri: "spotify:track:sounds0" }, { uri: "spotify:track:best1" }],
      }],
    },
    origin: { featureIdentifier: "soundalike" },
    options: { skipTo: { index: 1, uri: "spotify:track:best1" } },
  }]);
});

test("leaves hidden, unmatched and repeated songs out of the played list", async () => {
  const app = loadBestMatchesExtension({ playerApi: true });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.equal(app.rows[2].dataset.uri, undefined, "row 2 has no Spotify match");

  // A second row resolving to the same Spotify song plays only once.
  app.rows[2].dataset.uri = "spotify:track:best0";
  await rowContent(app.rows[1]).props.onDoubleClick();
  assert.deepEqual(
    plain(app.contextPlays[0].context.pages[0].items),
    [{ uri: "spotify:track:best0" }, { uri: "spotify:track:best1" }],
  );

  app.rows[0].hidden = true;
  app.rows[2].hidden = true;
  await rowContent(app.rows[1]).props.onDoubleClick();
  assert.deepEqual(plain(app.contextPlays[1]), {
    context: {
      uri: "spotify:app:soundalike",
      pages: [{ items: [{ uri: "spotify:track:best1" }] }],
    },
    origin: { featureIdentifier: "soundalike" },
    options: { skipTo: { index: 0, uri: "spotify:track:best1" } },
  });
});

test("plays the song alone when Spotify refuses the list", async () => {
  const app = loadBestMatchesExtension({ playerApi: "fails" });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  await rowContent(app.rows[1]).props.onDoubleClick();
  assert.equal(app.contextPlays.length, 1);
  assert.deepEqual(app.played, ["spotify:track:best1"]);
  assert.deepEqual(errorNotifications(app), []);
});

// Three Best matches; the third is still being matched to Spotify until
// `gate` resolves.
function loadLateMatchExtension(options = {}) {
  const slowSearch = "Late Song Slow Artist";
  const gate = deferred();
  const app = loadBestMatchesExtension({
    playerApi: true,
    timerScale: 0.01,
    bestFetch: async () => response(200, bestMatchesBody([
      { title: "Save Your Tears", artist: "The Weeknd", ranks: { melody: 1 } },
      { title: "Chanson Proche", artist: "Artiste Deux", ranks: { samples: 1 } },
      { title: "Late Song", artist: "Slow Artist", ranks: { melody: 2 } },
    ])),
    spotifyTracks: {
      "Save Your Tears The Weeknd":
        spotifyTrackFixture("Save Your Tears", "The Weeknd", "best0"),
      "Chanson Proche Artiste Deux":
        spotifyTrackFixture("Chanson Proche", "Artiste Deux", "best1"),
      [slowSearch]: spotifyTrackFixture("Late Song", "Slow Artist", "late0"),
    },
    async beforeGraphqlRequest(_definition, variables) {
      if (variables?.searchTerm === slowSearch) await gate.promise;
    },
    ...options,
  });
  return Object.assign(app, { gate });
}

async function playBeforeLastRowMatches(app) {
  await app.run();
  await waitFor(() => app.rows[1].dataset.uri === "spotify:track:best1");
  await rowContent(app.rows[0]).props.onDoubleClick();
  app.gate.resolve();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  assert.equal(app.rows[2].dataset.uri, "spotify:track:late0");
}

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test("starts the list at once and queues rows matched later after its last song", async () => {
  const app = loadLateMatchExtension();
  await playBeforeLastRowMatches(app);
  assert.deepEqual(plain(app.contextPlays), [{
    context: {
      uri: "spotify:app:soundalike",
      pages: [{
        items: [{ uri: "spotify:track:best0" }, { uri: "spotify:track:best1" }],
      }],
    },
    origin: { featureIdentifier: "soundalike" },
    options: { skipTo: { index: 0, uri: "spotify:track:best0" } },
  }], "playback starts without waiting for the row still matching");
  await pause(80);
  assert.deepEqual(app.queuedSongs, [], "the list still has songs to play first");

  // Another context reaching the same song leaves the queue alone.
  app.playerState.context = { uri: "spotify:playlist:elsewhere" };
  app.playerState.item = { uri: "spotify:track:best1" };
  app.songChangeListeners[0]();
  await pause(80);
  assert.deepEqual(app.queuedSongs, []);

  // So does skipping past the list's last song before the queue settles.
  app.playerState.context = { uri: "spotify:app:soundalike" };
  app.songChangeListeners[0]();
  app.playerState.item = { uri: "spotify:track:autoplay" };
  await pause(80);
  assert.deepEqual(app.queuedSongs, []);

  app.playerState.item = { uri: "spotify:track:best1" };
  app.songChangeListeners[0]();
  app.songChangeListeners[0]();
  await waitFor(() => app.queuedSongs.length === 1);
  assert.deepEqual(plain(app.queuedSongs), [[{ uri: "spotify:track:late0" }]]);
  await pause(80);
  app.songChangeListeners[0]();
  await pause(80);
  assert.equal(app.queuedSongs.length, 1, "late songs are queued once");
});

test("queues late rows again once when Spotify drops them", async () => {
  const app = loadLateMatchExtension({ dropFirstQueueAdd: true });
  await playBeforeLastRowMatches(app);
  app.playerState.item = { uri: "spotify:track:best1" };
  app.songChangeListeners[0]();
  await waitFor(() => app.queuedSongs.length === 2);
  assert.deepEqual(plain(app.queuedSongs), [
    [{ uri: "spotify:track:late0" }],
    [{ uri: "spotify:track:late0" }],
  ]);
  await pause(80);
  app.songChangeListeners[0]();
  await pause(80);
  assert.equal(app.queuedSongs.length, 2, "one retry at most");
});

test("queues nothing when every shown row was in the list", async () => {
  const app = loadBestMatchesExtension({ playerApi: true });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));

  await rowContent(app.rows[0]).props.onDoubleClick();
  app.playerState.item = { uri: "spotify:track:best1" };
  app.songChangeListeners[0]();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(app.queuedSongs, []);
});

test("marks the row of the song that is playing", async () => {
  const app = loadBestMatchesExtension({ playingTrackUri: "spotify:track:best1" });
  await app.run();
  await waitFor(() => hasLog(app, "Best matches metadata settled"));
  const playing = () =>
    app.rows.map((row) => row.classList.contains("sa-row-playing"));
  assert.deepEqual(playing(), [false, true, false]);
  assert.match(
    app.currentPage.innerHTML,
    /\.sa-row\.sa-row-playing \.sa-title,\.sa-row\.sa-row-playing \.sa-rank\{color:var\(--spice-button,#1ed760\)\}/,
  );

  assert.equal(app.songChangeListeners.length, 1);
  app.player.data = { item: { uri: "spotify:track:best0" } };
  app.songChangeListeners[0]();
  assert.deepEqual(playing(), [true, false, false]);
  app.player.data = { item: { uri: "spotify:episode:podcast" } };
  app.songChangeListeners[0]();
  assert.deepEqual(playing(), [false, false, false]);
});

test("adds Find soundalikes to the player bar's song menu, which names no song", async () => {
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, recommendation);
  }, { contextMenuV2: true, playingTrackUri: "spotify:track:test" });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(app.registeredMenuItems, 1);
  assert.equal(app.nowPlayingMenuItems.length, 1);
  const [item] = app.nowPlayingMenuItems;
  assert.equal(item.children, "Find soundalikes");
  assert.equal(item.leadingIcon, "enhance");
  // Menus that name their song keep the main item, including the player bar
  // once Spicetify passes its song.
  assert.equal(item.shouldAdd({ uri: "spotify:track:other" }), false);
  assert.equal(
    item.shouldAdd({ canToggleCanvas: true, item: { uri: "spotify:track:test" } }),
    false,
  );
  // Menus about no song at all stay untouched.
  assert.equal(item.shouldAdd({}), false);
  assert.equal(item.shouldAdd(undefined), false);
  // A podcast, or nothing, in the player bar is not a song.
  app.player.data = { item: { uri: "spotify:episode:podcast" } };
  assert.equal(item.shouldAdd({ canToggleCanvas: false }), false);
  app.player.data = null;
  assert.equal(item.shouldAdd({ canToggleCanvas: false }), false);

  app.player.data = { item: { uri: "spotify:track:test" } };
  assert.equal(item.shouldAdd({ canToggleCanvas: false }), true);
  // Clicks on other menus' contexts do nothing.
  item.onClick({ props: { uri: "spotify:track:other" } });
  item.onClick(undefined);
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.deepEqual(app.history, []);
  item.onClick({ props: { canToggleCanvas: false }, trigger: "right-click" });
  await waitFor(() => app.history.includes("/soundalike"));
  assert.ok(app.graphqlRequests.some(
    ({ definition, variables }) =>
      definition?.name === "getTrack" && variables?.uri === "spotify:track:test",
  ));
});

test("keeps the main menu item when Spicetify has no low-level menu items", async () => {
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) throw new TypeError("connection refused");
    return response(200, recommendation);
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(app.registeredMenuItems, 1);
  assert.equal(app.nowPlayingMenuItems.length, 0);
});
