import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MAX_BODY_BYTES,
  V17_PROTOCOL_SHA256,
  V17_SERVED_LISTS_SHA256,
  createHandler,
} from "../api/ratings.js";

const lists = JSON.parse(
  readFileSync(new URL("../evaluate/served-lists.json", import.meta.url), "utf8"),
);
const resultIds = lists.seeds.flatMap((seed) =>
  seed.results.map((result) => result.result_id),
);
const listIds = lists.seeds.flatMap((seed) =>
  seed.lists.map((list) => list.list_id),
);
const KEY = "a".repeat(64);

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function sign(ratings) {
  const payload = { ...ratings };
  delete payload.integrity_hmac_sha256;
  ratings.integrity_hmac_sha256 = createHmac("sha256", KEY)
    .update(canonical(payload), "utf8")
    .digest("hex");
  return ratings;
}

function validExport() {
  return sign({
    schema_version: 17,
    source_kind: "human_listener",
    provider: "hosted_private_submission_evaluator",
    anonymous_rater_id: `anon-${"1".repeat(24)}`,
    session_id: `session-${"2".repeat(24)}`,
    protocol_sha256: V17_PROTOCOL_SHA256,
    served_lists_sha256: V17_SERVED_LISTS_SHA256,
    local_session_key: KEY,
    started_at: "2026-07-14T00:00:00.000Z",
    last_activity_at: "2026-07-14T00:00:01.000Z",
    migration: {
      from_schema_version: 16,
      from_provider: "hosted_client_only_evaluator",
      from_protocol_sha256:
        "c94ce615c68cde595b4e48ac5010297d76bedbed52948b10d315a39286117727",
      from_served_lists_sha256:
        "809b98ae4314b396ffb33f7349fee72c94e1a80a33d84b1661ab83166a52b9e9",
      migrated_at: "2026-07-14T00:00:00.500Z",
    },
    result_ratings: {
      [resultIds[0]]: {
        similarity: "very_similar",
        score_0_10: 9,
        junk_or_version: false,
        rated_at: "2026-07-14T00:00:01.000Z",
        interaction_ms: 1000,
      },
    },
    list_ratings: {},
    exported_at: "2026-07-14T00:00:02.000Z",
    duration_ms: 2000,
    integrity_notice:
      "Local-key HMAC provides integrity, not identity or authenticity; the key is included in this export.",
  });
}

function request(body, options = {}) {
  const headers = {
    origin: "https://soundalike.yassin.app",
    "content-type": "application/json",
    ...options.headers,
  };
  if (!Object.hasOwn(headers, "host") && typeof headers.origin === "string") {
    try {
      headers.host = new URL(headers.origin).host;
    } catch {
      // Invalid origins are intentionally sent without a trusted Host fallback.
    }
  }
  if (options.contentLength !== undefined) {
    headers["content-length"] = String(options.contentLength);
  }
  return {
    method: options.method || "POST",
    headers,
    body: options.raw ? body : body,
  };
}

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
  };
}

class MemoryStorage {
  constructor() {
    this.objects = new Map();
    this.puts = [];
  }

  async head(pathname) {
    if (!this.objects.has(pathname)) {
      const error = new Error("missing");
      error.name = "BlobNotFoundError";
      throw error;
    }
    return { pathname };
  }

  async put(pathname, body, options) {
    if (this.objects.has(pathname)) throw new Error("already exists");
    this.puts.push({ pathname, body, options });
    this.objects.set(pathname, body);
    return { pathname, url: "private-url-must-not-escape" };
  }
}

async function submit(ratings = validExport(), storage = new MemoryStorage(), options = {}) {
  const res = response();
  const wrapper =
    options.wrapper === undefined
      ? { consent: true, ratings }
      : options.wrapper;
  await createHandler(storage)(
    request(options.rawBody ?? wrapper, options),
    res,
  );
  return { res, storage };
}

test("accepts a valid v16-migrated export and stores only a sanitized record", async () => {
  const { res, storage } = await submit();
  assert.equal(res.statusCode, 200);
  assert.deepEqual(Object.keys(res.body).sort(), [
    "counts",
    "duplicate",
    "receipt_sha256",
  ]);
  assert.deepEqual(res.body.counts, {
    complete_list_ratings: 0,
    complete_result_ratings: 1,
  });
  assert.equal(res.body.duplicate, false);
  assert.match(res.body.receipt_sha256, /^[a-f0-9]{64}$/);
  assert.equal(storage.puts.length, 1);
  const put = storage.puts[0];
  assert.equal(
    put.pathname,
    `human-ratings/v17/session-${"2".repeat(24)}/${res.body.receipt_sha256}.json`,
  );
  assert.deepEqual(put.options, {
    access: "private",
    addRandomSuffix: false,
    allowOverwrite: false,
    contentType: "application/json",
  });
  const stored = JSON.parse(put.body);
  assert.equal(stored.local_session_key, undefined);
  assert.equal(stored.integrity_hmac_sha256, undefined);
  assert.equal(stored.integrity_notice, undefined);
  assert.equal(stored.canonical_payload_sha256, res.body.receipt_sha256);
  assert.deepEqual(stored.counts, res.body.counts);
  assert.match(stored.received_at, /^\d{4}-\d{2}-\d{2}T/);
  for (const forbidden of [
    "private-url-must-not-escape",
    "user-agent",
    "cookie",
    "spotify",
    "email",
  ]) {
    assert.equal(JSON.stringify(res.body).toLowerCase().includes(forbidden), false);
  }
  assert.equal(res.headers["Cache-Control"], "no-store, max-age=0");
  assert.equal(res.headers["X-Content-Type-Options"], "nosniff");
  assert.equal(res.headers["Access-Control-Allow-Origin"], undefined);
});

test("rejects a tampered local-key HMAC", async () => {
  const ratings = validExport();
  ratings.result_ratings[resultIds[0]].similarity = "not_similar";
  const { res, storage } = await submit(ratings);
  assert.equal(res.statusCode, 400);
  assert.deepEqual(res.body, { error: "invalid request" });
  assert.equal(storage.puts.length, 0);
});

test("rejects invented result and list IDs", async () => {
  for (const [field, invented, value] of [
    [
      "result_ratings",
      `T14-${"f".repeat(24)}`,
      {
        similarity: "very_similar",
        score_0_10: null,
        junk_or_version: false,
        rated_at: "2026-07-14T00:00:01.000Z",
        interaction_ms: 1000,
      },
    ],
    [
      "list_ratings",
      `L14-${"f".repeat(24)}`,
      {
        whole_list_coherence: "very_coherent",
        unrelated_positions_1_to_3: 0,
        rated_at: "2026-07-14T00:00:01.000Z",
        interaction_ms: 1000,
      },
    ],
  ]) {
    const ratings = validExport();
    ratings[field][invented] = value;
    sign(ratings);
    const { res } = await submit(ratings);
    assert.equal(res.statusCode, 400);
  }
});

test("rejects malformed fields and untrusted hashes", async () => {
  const mutations = [
    (value) => {
      value.schema_version = 16;
    },
    (value) => {
      value.session_id = "../escape";
    },
    (value) => {
      value.protocol_sha256 = "0".repeat(64);
    },
    (value) => {
      value.duration_ms = -1;
    },
    (value) => {
      value.exported_at = "not-a-timestamp";
    },
    (value) => {
      value.result_ratings[resultIds[0]].score_0_10 = 11;
    },
    (value) => {
      value.result_ratings[resultIds[0]].extra = true;
    },
    (value) => {
      value.extra = true;
    },
  ];
  for (const mutate of mutations) {
    const ratings = validExport();
    mutate(ratings);
    sign(ratings);
    const { res } = await submit(ratings);
    assert.equal(res.statusCode, 400);
  }
});

test("rejects declared and actual payloads over 512 KiB", async () => {
  let result = await submit(validExport(), new MemoryStorage(), {
    contentLength: MAX_BODY_BYTES + 1,
  });
  assert.equal(result.res.statusCode, 413);

  const rawBody = `${" ".repeat(MAX_BODY_BYTES)}{}`;
  result = await submit(validExport(), new MemoryStorage(), {
    rawBody,
    raw: true,
  });
  assert.equal(result.res.statusCode, 413);
});

test("rejects evil and missing origins before storage", async () => {
  for (const origin of [
    undefined,
    "https://evil.example",
    "https://soundalike.yassin.app.evil.example",
    "null",
    "http://localhost.evil.example:3000",
  ]) {
    const storage = new MemoryStorage();
    const { res } = await submit(validExport(), storage, {
      headers: { origin },
    });
    assert.equal(res.statusCode, 403);
    assert.equal(storage.puts.length, 0);
  }
});

test("accepts production and loopback origins only", async () => {
  for (const origin of [
    "https://soundalike.yassin.app",
    "http://localhost:8788",
    "https://127.0.0.1:3000",
    "http://[::1]:5173",
  ]) {
    const { res } = await submit(validExport(), new MemoryStorage(), {
      headers: { origin },
    });
    assert.equal(res.statusCode, 200);
  }
});

test("requires explicit consent and at least one complete rating", async () => {
  let result = await submit(validExport(), new MemoryStorage(), {
    wrapper: { consent: false, ratings: validExport() },
  });
  assert.equal(result.res.statusCode, 400);

  const empty = validExport();
  empty.result_ratings = {};
  sign(empty);
  result = await submit(empty);
  assert.equal(result.res.statusCode, 400);
});

test("deduplicates exact snapshots without overwriting", async () => {
  const storage = new MemoryStorage();
  const ratings = validExport();
  const first = await submit(ratings, storage);
  const second = await submit(ratings, storage);
  assert.equal(first.res.body.duplicate, false);
  assert.equal(second.res.body.duplicate, true);
  assert.equal(first.res.body.receipt_sha256, second.res.body.receipt_sha256);
  assert.equal(storage.puts.length, 1);
});

test("keeps a distinct later non-conflicting snapshot from the same session", async () => {
  const storage = new MemoryStorage();
  const first = validExport();
  const later = validExport();
  later.result_ratings[resultIds[1]] = {
    similarity: "somewhat_similar",
    score_0_10: null,
    junk_or_version: false,
    rated_at: "2026-07-14T00:00:02.000Z",
    interaction_ms: 1000,
  };
  later.last_activity_at = "2026-07-14T00:00:02.000Z";
  later.exported_at = "2026-07-14T00:00:03.000Z";
  later.duration_ms = 3000;
  sign(later);
  const firstResult = await submit(first, storage);
  const laterResult = await submit(later, storage);
  assert.notEqual(
    firstResult.res.body.receipt_sha256,
    laterResult.res.body.receipt_sha256,
  );
  assert.equal(storage.puts.length, 2);
});

test("redacts storage failures and never returns a Blob URL", async () => {
  const storage = {
    async head() {
      throw new Error("BLOB_READ_WRITE_TOKEN=secret private-url");
    },
    async put() {
      throw new Error("should not run");
    },
  };
  const { res } = await submit(validExport(), storage);
  assert.equal(res.statusCode, 503);
  assert.deepEqual(res.body, { error: "storage unavailable" });
  assert.equal(JSON.stringify(res.body).includes("secret"), false);
});

test("is POST/JSON only and exposes no method-key material", async () => {
  let result = await submit(validExport(), new MemoryStorage(), {
    method: "GET",
  });
  assert.equal(result.res.statusCode, 405);
  assert.equal(result.res.headers.Allow, "POST");

  result = await submit(validExport(), new MemoryStorage(), {
    headers: { "content-type": "text/plain" },
  });
  assert.equal(result.res.statusCode, 415);

  const publicText = [
    readFileSync(new URL("../api/ratings.js", import.meta.url), "utf8"),
    readFileSync(new URL("../evaluate/index.html", import.meta.url), "utf8"),
    readFileSync(new URL("../evaluate/protocol.json", import.meta.url), "utf8"),
    readFileSync(new URL("../evaluate/served-lists.json", import.meta.url), "utf8"),
  ].join("\n");
  for (const marker of [
    "BEGIN OPENSSH PRIVATE KEY",
    '"unblinding_map"',
    '"private_method_key"',
    '"method_role"',
  ]) {
    assert.equal(publicText.includes(marker), false);
  }
  assert.equal(publicText.includes("BLOB_READ_WRITE_TOKEN="), false);
});
