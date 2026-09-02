import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import test from "node:test";

import ratingsDispatch, {
  createDispatchHandler,
} from "../api/ratings.js";
import {
  createDiscordWebhookSender,
  formatFeedbackDigest,
} from "../server/discord-feedback.js";
import {
  FEEDBACK_BLOB_PREFIX,
  FEEDBACK_DIGEST_INPUT_PREFIX,
  MAX_FEEDBACK_BODY_BYTES,
  MAX_FEEDBACK_STORED_BYTES,
  createFeedbackHandler,
  parseFeedbackStoredRecordBytes,
  validateFeedbackPayload,
} from "../server/spicetify-feedback.js";
import {
  FEEDBACK_DIGEST_BLOB_PREFIX,
  createFeedbackDigestHandler,
  summarizeFeedback,
} from "../server/spicetify-feedback-digest.js";
import { downloadSpicetifyFeedback } from "../tools/spicetify-feedback-inbox.js";

function validPayload() {
  return {
    schema_version: 1,
    survey_version: "spicetify-match-feedback-v1",
    install_nonce: "1".repeat(32),
    session_nonce: "2".repeat(32),
    seed: { title: "Blinding Lights", artist: "The Weeknd" },
    displayed_results: [
      { position: 1, title: "Take My Breath", artist: "The Weeknd" },
      { position: 2, title: "Midnight City", artist: "M83" },
    ],
    method: "dual_sonic64_guardrail",
    index_version: "index-2026.07.11-dual-sonic64",
    api_version: "4",
    language_policy: "spotify-lyrics-strict-v2",
    selection_policy: "top-20-strict-language-related-artist-v1",
    source: "hosted",
    selection: "mixed",
    reasons: ["mood_energy", "tempo"],
    note: "Close overall, but the second result felt slower.",
  };
}

function clone(value) {
  return structuredClone(value);
}

function mobilePayload() {
  const payload = validPayload();
  payload.source = "mobile";
  payload.language_policy = "none";
  payload.selection_policy = "mobile-top-20-model-quality-v1";
  return payload;
}

test("accepts mobile feedback with its own language and selection policy", () => {
  const accepted = validateFeedbackPayload(mobilePayload());
  assert.notEqual(accepted, null);
  assert.equal(accepted.source, "mobile");
  assert.equal(accepted.language_policy, "none");
  assert.equal(accepted.selection_policy, "mobile-top-20-model-quality-v1");
});

test("rejects mobile and desktop policy fields crossing over", () => {
  const mobileWithDesktopLanguage = mobilePayload();
  mobileWithDesktopLanguage.language_policy = "spotify-lyrics-strict-v2";
  assert.equal(validateFeedbackPayload(mobileWithDesktopLanguage), null);

  const mobileWithDesktopSelection = mobilePayload();
  mobileWithDesktopSelection.selection_policy =
    "top-20-strict-language-related-artist-v1";
  assert.equal(validateFeedbackPayload(mobileWithDesktopSelection), null);

  const desktopClaimingNoLanguage = validPayload();
  desktopClaimingNoLanguage.language_policy = "none";
  assert.equal(validateFeedbackPayload(desktopClaimingNoLanguage), null);

  const desktopWithMobileSelection = validPayload();
  desktopWithMobileSelection.selection_policy =
    "mobile-top-20-model-quality-v1";
  assert.equal(validateFeedbackPayload(desktopWithMobileSelection), null);
});

function response() {
  return {
    headers: {},
    setHeader(name, value) {
      this.headers[name] = value;
    },
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(value) {
      this.body = value;
      return this;
    },
    end() {
      this.ended = true;
      return this;
    },
  };
}

class MemoryStorage {
  constructor() {
    this.objects = new Map();
    this.lists = [];
    this.puts = [];
    this.failPut = false;
  }
  async head(pathname) {
    if (!this.objects.has(pathname)) {
      const error = new Error("missing");
      error.name = "BlobNotFoundError";
      throw error;
    }
    const body = this.objects.get(pathname);
    return {
      pathname,
      size: Buffer.byteLength(body),
      contentType: "application/json",
    };
  }
  async put(pathname, body, options) {
    if (this.failPut) throw new Error("offline");
    if (this.objects.has(pathname)) throw new Error("exists");
    this.objects.set(pathname, body);
    this.puts.push({ pathname, body, options });
  }
  async list(options) {
    this.lists.push(options);
    const blobs = [...this.objects.entries()]
      .filter(([pathname]) => pathname.startsWith(options.prefix))
      .map(([pathname, body]) => ({
        pathname,
        size: Buffer.byteLength(body),
      }));
    return { blobs, hasMore: false };
  }
  async get(pathname, options) {
    assert.deepEqual(options, { access: "private", useCache: false });
    const body = this.objects.get(pathname);
    if (body === undefined) throw new Error("missing");
    const size = Buffer.byteLength(body);
    return {
      statusCode: 200,
      stream: Readable.from([Buffer.from(body)]),
      blob: { pathname, size, contentType: "application/json" },
    };
  }
  async del(pathname) {
    this.objects.delete(pathname);
  }
}

async function submit(
  body = validPayload(),
  storage = new MemoryStorage(),
  options = {},
) {
  const res = response();
  const headers = {
    "content-type": "application/json",
    ...(options.headers || {}),
  };
  if (options.contentLength !== undefined) {
    headers["content-length"] = String(options.contentLength);
  }
  const supplied = options.rawBody ?? body;
  const raw = Buffer.isBuffer(supplied)
    ? supplied
    : Buffer.from(
        typeof supplied === "string" ? supplied : JSON.stringify(supplied),
        "utf8",
      );
  const request = options.replayed ? new EventEmitter() : Readable.from([raw]);
  request.method = options.method || "POST";
  request.headers = headers;
  if (options.replayed) {
    Object.defineProperty(request, "body", {
      get() {
        throw new Error("request.body must remain untouched");
      },
    });
    queueMicrotask(() => {
      request.emit("data", raw);
      request.emit("end");
    });
  }
  const pending = [];
  const handlerOptions = { ...(options.handlerOptions || {}) };
  if (!handlerOptions.waitUntil) {
    handlerOptions.waitUntil = (promise) => pending.push(promise);
  }
  await createFeedbackHandler(storage, handlerOptions)(request, res);
  await Promise.all(pending);
  return { res, storage };
}

test("stores a strict private record and returns only a receipt", async () => {
  const { res, storage } = await submit();
  assert.equal(res.statusCode, 200);
  assert.deepEqual(Object.keys(res.body), ["receipt_sha256"]);
  assert.match(res.body.receipt_sha256, /^[a-f0-9]{64}$/);
  assert.equal(JSON.stringify(res.body).includes("url"), false);
  assert.equal(JSON.stringify(res.body).includes("note"), false);
  assert.equal(storage.puts.length, 2);
  const put = storage.puts[0];
  assert.equal(
    put.pathname,
    `${FEEDBACK_BLOB_PREFIX}${res.body.receipt_sha256}.json`,
  );
  assert.deepEqual(put.options, {
    access: "private",
    addRandomSuffix: false,
    allowOverwrite: false,
    contentType: "application/json",
  });
  assert.match(
    storage.puts[1].pathname,
    new RegExp(
      `^${FEEDBACK_DIGEST_INPUT_PREFIX}` +
        `\\d{4}-\\d{2}-\\d{2}/${res.body.receipt_sha256}\\.json$`,
    ),
  );
  const parsed = parseFeedbackStoredRecordBytes(put.body, put.pathname);
  assert.equal(parsed.digest, res.body.receipt_sha256);
  assert.equal(parsed.payload.note, validPayload().note);
  for (const forbidden of [
    "spotify_account",
    "credentials",
    "headers",
    "history",
    "ip",
    "library",
    "user_agent",
    "url",
  ]) {
    assert.equal(Object.hasOwn(parsed.document, forbidden), false);
  }
});

test("requires exact keys and validates nested types and enums", async () => {
  const mutations = [
    (value) => {
      value.extra = true;
    },
    (value) => {
      value.seed.extra = true;
    },
    (value) => {
      value.displayed_results[0].hidden_candidate = true;
    },
    (value) => {
      value.schema_version = "1";
    },
    (value) => {
      value.survey_version = "v2";
    },
    (value) => {
      value.install_nonce = "not-a-nonce";
    },
    (value) => {
      value.install_nonce = ["1".repeat(32)];
    },
    (value) => {
      value.session_nonce = true;
    },
    (value) => {
      value.method = "arbitrary";
    },
    (value) => {
      value.index_version = "bad index";
    },
    (value) => {
      value.index_version = false;
    },
    (value) => {
      value.api_version = "5";
    },
    (value) => {
      value.language_policy = "permissive";
    },
    (value) => {
      value.selection_policy = "rerank-everything";
    },
    (value) => {
      value.source = "browser";
    },
    (value) => {
      value.source = "local";
    },
    (value) => {
      value.selection = "excellent";
    },
  ];
  for (const mutate of mutations) {
    const value = validPayload();
    mutate(value);
    assert.equal((await submit(value)).res.statusCode, 400);
  }
  const legacy = validPayload();
  legacy.api_version = "legacy";
  assert.equal((await submit(legacy)).res.statusCode, 200);
  const local = validPayload();
  local.source = "local";
  local.api_version = "local";
  assert.equal((await submit(local)).res.statusCode, 200);
  const modelQuality = validPayload();
  modelQuality.selection_policy =
    "top-20-strict-language-related-artist-model-quality-v1";
  assert.equal((await submit(modelQuality)).res.statusCode, 200);
});

test("enforces displayed order, result count, labels, and reason rules", async () => {
  const mutations = [
    (value) => {
      value.displayed_results = [];
    },
    (value) => {
      value.displayed_results = Array.from({ length: 21 }, (_, index) => ({
        position: index + 1,
        title: `Track ${index}`,
        artist: `Artist ${index}`,
      }));
    },
    (value) => {
      value.displayed_results[1].position = 1;
    },
    (value) => {
      value.seed.title = "";
    },
    (value) => {
      value.displayed_results[0].artist = "x".repeat(301);
    },
    (value) => {
      value.reasons = ["style", "tempo", "mood_energy"];
    },
    (value) => {
      value.reasons = ["style", "style"];
    },
    (value) => {
      value.reasons = ["personal_profile"];
    },
    (value) => {
      value.selection = "good";
    },
  ];
  for (const mutate of mutations) {
    const value = validPayload();
    mutate(value);
    assert.equal((await submit(value)).res.statusCode, 400);
  }
  const good = validPayload();
  good.selection = "good";
  good.reasons = [];
  good.note = "";
  assert.equal((await submit(good)).res.statusCode, 200);
});

test("accepts a 280-character note, rejects 281, and sanitizes controls", async () => {
  const boundary = validPayload();
  boundary.note = "n".repeat(280);
  assert.equal((await submit(boundary)).res.statusCode, 200);

  const over = validPayload();
  over.note = "n".repeat(281);
  assert.equal((await submit(over)).res.statusCode, 400);

  const controls = validPayload();
  controls.note = "First line\nSecond\tline";
  const { res, storage } = await submit(controls);
  assert.equal(res.statusCode, 200);
  const parsed = parseFeedbackStoredRecordBytes(
    storage.puts[0].body,
    storage.puts[0].pathname,
  );
  assert.equal(parsed.payload.note, "First line Second line");
});

test("supports public CORS preflight and constrains method and content type", async () => {
  const storage = new MemoryStorage();
  const preflight = await submit(undefined, storage, { method: "OPTIONS" });
  assert.equal(preflight.res.statusCode, 204);
  assert.equal(preflight.res.headers["Access-Control-Allow-Origin"], "*");
  assert.equal(
    preflight.res.headers["Access-Control-Allow-Methods"],
    "POST, OPTIONS",
  );
  assert.equal(
    preflight.res.headers["Access-Control-Allow-Headers"],
    "Content-Type",
  );
  assert.equal(storage.puts.length, 0);

  const get = await submit(validPayload(), storage, { method: "GET" });
  assert.equal(get.res.statusCode, 405);
  assert.equal(get.res.headers.Allow, "POST, OPTIONS");

  for (const headers of [
    { "content-type": "text/plain" },
    { "content-type": "application/json; charset=latin1" },
    { "content-encoding": "gzip" },
  ]) {
    assert.equal(
      (await submit(validPayload(), new MemoryStorage(), { headers })).res
        .statusCode,
      415,
    );
  }
});

test("enforces declared and actual request size and strict JSON parsing", async () => {
  assert.equal(
    (await submit(validPayload(), new MemoryStorage(), {
      contentLength: MAX_FEEDBACK_BODY_BYTES + 1,
    })).res.statusCode,
    413,
  );
  assert.equal(
    (await submit(validPayload(), new MemoryStorage(), {
      rawBody: Buffer.alloc(MAX_FEEDBACK_BODY_BYTES + 1, 0x20),
    })).res.statusCode,
    413,
  );
  assert.equal(
    (await submit(validPayload(), new MemoryStorage(), {
      rawBody: '{"schema_version":1,"schema_version":1}',
    })).res.statusCode,
    400,
  );
});

test("accepts a Vercel-style replay without touching request.body", async () => {
  const { res, storage } = await submit(
    validPayload(),
    new MemoryStorage(),
    { replayed: true },
  );
  assert.equal(res.statusCode, 200);
  assert.equal(storage.puts.length, 2);
});

test("shared ratings function dispatches feedback preflight", async () => {
  const request = Readable.from([]);
  request.method = "OPTIONS";
  request.url = "/api/ratings?__soundalike_handler=spicetify-feedback";
  request.headers = {};
  const res = response();

  await ratingsDispatch(request, res);
  assert.equal(res.statusCode, 204);
  assert.equal(res.headers["Access-Control-Allow-Origin"], "*");
});

test("shared ratings function dispatches the daily digest route", async () => {
  let called = false;
  const dispatch = createDispatchHandler(
    async () => {
      throw new Error("legacy handler must not run");
    },
    {
      "feedback-digest": async () => async (_request, response) => {
        called = true;
        return response.status(200).json({ ok: true });
      },
    },
  );
  const request = Readable.from([]);
  request.method = "GET";
  request.url = "/api/ratings?__soundalike_handler=feedback-digest";
  request.headers = {};
  const res = response();
  await dispatch(request, res);
  assert.equal(called, true);
  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body, { ok: true });
});

test("deduplicates deterministically without overwrite", async () => {
  const storage = new MemoryStorage();
  const first = await submit(validPayload(), storage);
  const second = await submit(validPayload(), storage);
  assert.equal(first.res.statusCode, 200);
  assert.equal(second.res.statusCode, 200);
  assert.equal(
    first.res.body.receipt_sha256,
    second.res.body.receipt_sha256,
  );
  assert.equal(storage.puts.length, 2);

  const changed = validPayload();
  changed.note = "A different observation.";
  const third = await submit(changed, storage);
  assert.notEqual(
    first.res.body.receipt_sha256,
    third.res.body.receipt_sha256,
  );
  assert.equal(storage.puts.length, 4);
});

test("reports storage failure without leaking storage details", async () => {
  const storage = new MemoryStorage();
  storage.failPut = true;
  const { res } = await submit(validPayload(), storage);
  assert.equal(res.statusCode, 503);
  assert.deepEqual(res.body, { error: "storage unavailable" });
});

test("notifies Discord once with a redacted summary after storage", async () => {
  const storage = new MemoryStorage();
  const messages = [];
  const handlerOptions = {
    sendDiscord: async (content) => {
      messages.push(content);
      return true;
    },
  };
  const first = await submit(validPayload(), storage, { handlerOptions });
  const retry = await submit(validPayload(), storage, { handlerOptions });
  assert.equal(first.res.statusCode, 200);
  assert.equal(retry.res.statusCode, 200);
  assert.equal(messages.length, 1);
  assert.match(messages[0], /New Soundalike feedback/);
  assert.match(messages[0], /Rating: \*\*Mixed\*\*/);
  assert.match(messages[0], /Blinding Lights - The Weeknd/);
  assert.doesNotMatch(messages[0], /Close overall/);
  assert.doesNotMatch(messages[0], /Take My Breath/);
  assert.doesNotMatch(messages[0], /1111111111111111/);
  assert.doesNotMatch(messages[0], /2222222222222222/);
});

test("keeps stored feedback successful when Discord is unavailable", async () => {
  const messages = [];
  const { res, storage } = await submit(validPayload(), new MemoryStorage(), {
    handlerOptions: {
      sendDiscord: async () => {
        throw new Error("offline");
      },
      logger: {
        error(message) {
          messages.push(message);
        },
      },
    },
  });
  assert.equal(res.statusCode, 200);
  assert.equal(storage.puts.length, 2);
  assert.deepEqual(messages, [
    "Soundalike feedback Discord notification failed.",
  ]);
});

test("Discord sender validates its URL and disables mentions and embeds", async () => {
  const requests = [];
  const sender = createDiscordWebhookSender({
    webhookUrl:
      "https://discord.com/api/webhooks/123456789012345678/" +
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN",
    fetchImpl: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, status: 204 };
    },
  });

  assert.equal(await sender("New feedback from @everyone"), true);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].options.redirect, "error");
  const body = JSON.parse(requests[0].options.body);
  assert.deepEqual(body.allowed_mentions, { parse: [] });
  assert.equal(body.flags, 4);
  assert.equal(body.username, "Soundalike Feedback");
  await assert.rejects(
    createDiscordWebhookSender({
      webhookUrl: "https://example.com/webhook",
      fetchImpl: async () => ({ ok: true }),
    })("test"),
    /Invalid Discord webhook configuration/,
  );
});

test("returns the feedback receipt before Discord delivery finishes", async () => {
  let finish;
  let pending;
  const delivery = new Promise((resolve) => {
    finish = resolve;
  });
  const result = await submit(validPayload(), new MemoryStorage(), {
    handlerOptions: {
      sendDiscord: async () => delivery,
      waitUntil(promise) {
        pending = promise;
      },
    },
  });
  assert.equal(result.res.statusCode, 200);
  assert.ok(pending instanceof Promise);
  finish(true);
  await pending;
});

test("summarizes feedback without notes, nonces, or result lists", () => {
  const mixed = validPayload();
  const off = validPayload();
  off.selection = "off";
  off.reasons = ["style"];
  const summary = summarizeFeedback("2026-09-01", [mixed, off]);
  assert.equal(summary.total, 2);
  assert.deepEqual(summary.selections, { good: 0, mixed: 1, off: 1 });
  assert.equal(summary.reasons.mood_energy, 1);
  assert.equal(summary.reasons.tempo, 1);
  assert.equal(summary.reasons.style, 1);
  const serialized = JSON.stringify(summary);
  assert.doesNotMatch(serialized, /Close overall/);
  assert.doesNotMatch(serialized, /Take My Breath/);
  assert.doesNotMatch(serialized, /1111111111111111/);
  assert.doesNotMatch(formatFeedbackDigest(summary), /Close overall/);
});

test("daily digest is authenticated, redacted, and processed once", async () => {
  const storage = new MemoryStorage();
  const recordTime = Date.UTC(2026, 8, 1, 12);
  const digestTime = Date.UTC(2026, 8, 2, 15);
  const mixed = validPayload();
  const good = validPayload();
  good.selection = "good";
  good.reasons = [];
  good.note = "";
  const off = validPayload();
  off.selection = "off";
  off.reasons = ["style"];
  for (const payload of [mixed, good, off]) {
    await submit(payload, storage, {
      handlerOptions: {
        now: () => recordTime,
        sendDiscord: async () => false,
      },
    });
  }
  const messages = [];
  const handler = createFeedbackDigestHandler(storage, {
    cronSecret: "test-cron-secret",
    now: () => digestTime,
    sendDiscord: async (content) => {
      assert.equal(
        storage.objects.has(
          `${FEEDBACK_DIGEST_BLOB_PREFIX}2026-09-01.json`,
        ),
        true,
      );
      messages.push(content);
      return true;
    },
  });
  const request = {
    method: "GET",
    headers: { authorization: "Bearer test-cron-secret" },
  };
  const first = response();
  await handler(request, first);
  assert.equal(first.statusCode, 200);
  assert.deepEqual(first.body, {
    date: "2026-09-01",
    record_count: 3,
    sent: true,
  });
  assert.equal(messages.length, 1);
  assert.deepEqual(storage.lists, [{
    prefix: `${FEEDBACK_DIGEST_INPUT_PREFIX}2026-09-01/`,
    limit: 1000,
    cursor: undefined,
  }]);
  assert.match(messages[0], /Total: \*\*3\*\* \(Good 1, Mixed 1, Off 1\)/);
  assert.doesNotMatch(messages[0], /Close overall/);
  assert.doesNotMatch(messages[0], /Take My Breath/);
  assert.doesNotMatch(messages[0], /1111111111111111/);
  assert.equal(
    storage.objects.has(
      `${FEEDBACK_DIGEST_BLOB_PREFIX}2026-09-01.json`,
    ),
    true,
  );
  const retry = response();
  await handler(request, retry);
  assert.equal(retry.statusCode, 200);
  assert.deepEqual(retry.body, {
    date: "2026-09-01",
    sent: false,
    already_processed: true,
  });
  assert.equal(messages.length, 1);
});

test("daily digest rejects unauthorized calls before reading storage", async () => {
  let listed = false;
  const storage = {
    async list() {
      listed = true;
      throw new Error("must not list");
    },
  };
  const handler = createFeedbackDigestHandler(storage, {
    cronSecret: "test-cron-secret",
    sendDiscord: async () => true,
  });
  const res = response();
  await handler(
    { method: "GET", headers: { authorization: "Bearer wrong-secret" } },
    res,
  );
  assert.equal(res.statusCode, 401);
  assert.deepEqual(res.body, { error: "unauthorized" });
  assert.equal(listed, false);
});

test("daily digest remains retryable when Discord delivery fails", async () => {
  const storage = new MemoryStorage();
  await submit(validPayload(), storage, {
    handlerOptions: {
      now: () => Date.UTC(2026, 8, 1, 12),
      sendDiscord: async () => false,
    },
  });
  const logs = [];
  const handler = createFeedbackDigestHandler(storage, {
    cronSecret: "test-cron-secret",
    now: () => Date.UTC(2026, 8, 2, 15),
    sendDiscord: async () => {
      throw new Error("offline");
    },
    logger: { error: (message) => logs.push(message) },
  });
  const res = response();
  await handler(
    {
      method: "GET",
      headers: { authorization: "Bearer test-cron-secret" },
    },
    res,
  );
  assert.equal(res.statusCode, 502);
  assert.deepEqual(res.body, { error: "digest failed" });
  assert.deepEqual(logs, ["Soundalike feedback digest failed."]);
  assert.equal(
    storage.objects.has(
      `${FEEDBACK_DIGEST_BLOB_PREFIX}2026-09-01.json`,
    ),
    false,
  );
});

test("inbox validates private paths and bytes and only flags retention", async () => {
  const submitted = await submit();
  const put = submitted.storage.puts[0];
  const bytes = Buffer.from(put.body);
  const storage = {
    async list(options) {
      assert.deepEqual(options, {
        prefix: FEEDBACK_BLOB_PREFIX,
        limit: 1000,
        cursor: undefined,
      });
      return {
        blobs: [{ pathname: put.pathname, size: bytes.length }],
        hasMore: false,
      };
    },
    async get(pathname, options) {
      assert.equal(pathname, put.pathname);
      assert.deepEqual(options, { access: "private", useCache: false });
      return {
        statusCode: 200,
        stream: Readable.from([bytes]),
        blob: {
          pathname,
          size: bytes.length,
          contentType: "application/json",
        },
      };
    },
  };
  const directory = await mkdtemp(join(tmpdir(), "soundalike-feedback-inbox-"));
  try {
    const future = Date.now() + 91 * 24 * 60 * 60 * 1000;
    assert.deepEqual(
      await downloadSpicetifyFeedback(directory, storage, {
        now: future,
        retentionDays: 90,
      }),
      {
        downloaded: 1,
        existing: 0,
        retentionCandidates: 1,
        retentionDays: 90,
      },
    );
    assert.deepEqual(
      await downloadSpicetifyFeedback(directory, storage, {
        now: future,
        retentionDays: 90,
      }),
      {
        downloaded: 0,
        existing: 1,
        retentionCandidates: 1,
        retentionDays: 90,
      },
    );
    const saved = await readFile(
      join(directory, `${submitted.res.body.receipt_sha256}.json`),
    );
    assert.equal(saved.equals(bytes), true);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("inbox rejects an unexpected path before downloading", async () => {
  const storage = {
    async list() {
      return {
        blobs: [{ pathname: `${FEEDBACK_BLOB_PREFIX}../escape.json`, size: 10 }],
        hasMore: false,
      };
    },
    async get() {
      throw new Error("must not download");
    },
  };
  const directory = await mkdtemp(join(tmpdir(), "soundalike-feedback-bad-"));
  try {
    await assert.rejects(
      downloadSpicetifyFeedback(directory, storage),
      /Unexpected object/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("inbox rejects an oversized listing and has no content logging path", async () => {
  let downloaded = false;
  const storage = {
    async list() {
      return {
        blobs: [{
          pathname: `${FEEDBACK_BLOB_PREFIX}${"a".repeat(64)}.json`,
          size: MAX_FEEDBACK_STORED_BYTES + 1,
        }],
        hasMore: false,
      };
    },
    async get() {
      downloaded = true;
      throw new Error("must not download");
    },
  };
  const directory = await mkdtemp(join(tmpdir(), "soundalike-feedback-large-"));
  try {
    await assert.rejects(
      downloadSpicetifyFeedback(directory, storage),
      /Unexpected object/,
    );
    assert.equal(downloaded, false);
    const downloader = downloadSpicetifyFeedback.toString();
    assert.equal(downloader.includes("console."), false);
    assert.equal(downloader.includes(".url"), false);
    assert.equal(downloader.includes(".note"), false);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Vercel routes feedback through the shared bounded ratings function", () => {
  const config = JSON.parse(
    readFileSync(new URL("../vercel.json", import.meta.url), "utf8"),
  );
  assert.equal(Object.keys(config.functions).length, 12);
  assert.equal(config.functions["api/spicetify-feedback.js"], undefined);
  assert.equal(
    config.rewrites.find(
      (rewrite) => rewrite.source === "/api/spicetify-feedback",
    )?.destination,
    "/api/ratings?__soundalike_handler=spicetify-feedback",
  );
  assert.deepEqual(config.functions["api/ratings.js"], { maxDuration: 60 });
  assert.equal(config.functions["api/feedback-digest.js"], undefined);
  assert.equal(
    config.rewrites.find(
      (rewrite) => rewrite.source === "/api/feedback-digest",
    )?.destination,
    "/api/ratings?__soundalike_handler=feedback-digest",
  );
  assert.deepEqual(config.crons, [
    { path: "/api/feedback-digest", schedule: "0 15 * * *" },
  ]);
});
