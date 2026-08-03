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

function loadExtension(fetchImpl) {
  let handler;
  let modal;
  const notifications = [];
  const context = {
    AbortController,
    Date,
    clearTimeout,
    fetch: fetchImpl,
    setTimeout,
    console: { error() {}, log() {}, warn() {} },
    document: {
      createElement() {
        return {
          className: "",
          innerHTML: "",
          querySelectorAll() {
            return [];
          },
        };
      },
    },
  };
  const getTrack = {};
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
      Definitions: { getTrack },
      async Request(definition) {
        assert.equal(definition, getTrack);
        return {
          data: {
            trackUnion: {
              name: "Blinding Lights",
              firstArtist: { items: [{ profile: { name: "The Weeknd" } }] },
            },
          },
        };
      },
    },
    Platform: { History: { push() {} } },
    PopupModal: {
      display(value) {
        modal = value;
      },
      hide() {},
    },
    ReactJSX: { jsx() {} },
    URI: {},
    showNotification(...args) {
      notifications.push(args);
    },
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(extensionSource, context);
  return {
    notifications,
    async run() {
      await handler(["spotify:track:test"]);
      return modal;
    },
  };
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

  const modal = await app.run();

  assert.deepEqual(urls, [
    "http://127.0.0.1:8787/health",
    "https://soundalike.yassin.app/api/recommend",
  ]);
  assert.match(modal.content.innerHTML, /HOSTED LIBRARY/);
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

  const modal = await app.run();

  assert.deepEqual(urls, [
    "http://127.0.0.1:8787/health",
    "http://127.0.0.1:8787/api/recommend",
  ]);
  assert.match(modal.content.innerHTML, /LOCAL ENGINE/);
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

  const modal = await app.run();

  assert.deepEqual(urls, [
    "http://127.0.0.1:8787/health",
    "http://127.0.0.1:8787/api/recommend",
    "https://soundalike.yassin.app/api/recommend",
  ]);
  assert.match(modal.content.innerHTML, /HOSTED LIBRARY/);
});

test("shows the hosted library miss without opening an empty modal", async () => {
  const app = loadExtension(async (url) => {
    if (url.endsWith("/health")) {
      throw new TypeError("connection refused");
    }
    return response(422, {
      ok: false,
      error: "This track is not in the hosted library.",
    });
  });

  const modal = await app.run();

  assert.equal(modal, undefined);
  assert.equal(
    app.notifications.some(([message, isError]) =>
      message.includes("not in the hosted library") && isError === true),
    true,
  );
});
