import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import test from "node:test";

import { canonical } from "../api/ratings.js";
import {
  MAX_SEMANTIC_BODY_BYTES,
  SEMANTIC_BLOB_PREFIX,
  SEMANTIC_PACK_SHA256,
  SEMANTIC_PROTOCOL_SHA256,
  createSemanticHandler,
  parseSemanticStoredRecordBytes,
} from "../api/ratings-semantic-v2.js";
import { downloadSemantic } from "../tools/ratings-semantic-v2-inbox.js";

const pack = JSON.parse(
  readFileSync(new URL("../evaluate/semantic-pack.json", import.meta.url), "utf8"),
);
const listId = pack.seeds[0].lists[0].list_id;
const KEY = "a".repeat(64);

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
    schema_version: 2,
    submission_schema: "repeated_excerpt_semantic_listener_submission_v2",
    source_kind: "human_listener",
    provider: "hosted_private_semantic_v2_evaluator",
    anonymous_rater_id: `anon-semantic-${"1".repeat(24)}`,
    session_id: `semantic-session-${"2".repeat(24)}`,
    protocol_sha256: SEMANTIC_PROTOCOL_SHA256,
    pilot_pack_sha256: SEMANTIC_PACK_SHA256,
    local_session_key: KEY,
    started_at: "2026-07-30T00:00:00.000Z",
    last_activity_at: "2026-07-30T00:00:01.000Z",
    list_ratings: {
      [listId]: {
        similarity: "very_similar",
        score_0_10: 9,
        unrelated_positions_1_to_5: [4],
        rated_at: "2026-07-30T00:00:01.000Z",
        interaction_ms: 1000,
      },
    },
    exported_at: "2026-07-30T00:00:02.000Z",
    duration_ms: 2000,
    integrity_notice:
      "Local-key HMAC provides integrity, not identity or authenticity; the key is included in this export.",
  });
}

function request(body, options = {}) {
  const headers = {
    origin: "https://soundalike.yassin.app",
    host: "soundalike.yassin.app",
    "content-type": "application/json",
    ...options.headers,
  };
  if (options.contentLength !== undefined) {
    headers["content-length"] = String(options.contentLength);
  }
  return {
    method: options.method || "POST",
    headers,
    body,
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
    this.objects.set(pathname, body);
    this.puts.push({ pathname, body, options });
    return { pathname, url: "private-url-must-not-escape" };
  }
}

async function submit(
  ratings = validExport(),
  storage = new MemoryStorage(),
  options = {},
) {
  const res = response();
  const wrapper =
    options.wrapper === undefined
      ? { consent: true, study: "semantic-repeated-excerpt-v2", ratings }
      : options.wrapper;
  await createSemanticHandler(storage, options.deploymentHost)(
    request(options.rawBody ?? wrapper, options),
    res,
  );
  return { res, storage };
}

test("accepts semantic evidence only in its immutable private prefix", async () => {
  const { res, storage } = await submit();
  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body.counts, { complete_list_ratings: 1 });
  assert.equal(res.body.duplicate, false);
  assert.match(res.body.receipt_sha256, /^[a-f0-9]{64}$/);
  assert.equal(storage.puts.length, 1);
  const put = storage.puts[0];
  assert.equal(
    put.pathname,
    `${SEMANTIC_BLOB_PREFIX}semantic-session-${"2".repeat(24)}/${res.body.receipt_sha256}.json`,
  );
  assert.equal(put.pathname.startsWith("human-ratings/fulltrack-v2/"), false);
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
  assert.equal(
    parseSemanticStoredRecordBytes(put.body, put.pathname).digest,
    res.body.receipt_sha256,
  );
  assert.equal(JSON.stringify(res.body).includes("private-url"), false);
});

test("rejects hash, schema, ID, rating and HMAC tampering", async () => {
  const mutations = [
    (value) => {
      value.schema_version = 17;
    },
    (value) => {
      value.protocol_sha256 = "0".repeat(64);
    },
    (value) => {
      value.pilot_pack_sha256 = "0".repeat(64);
    },
    (value) => {
      value.session_id = "../escape";
    },
    (value) => {
      value.list_ratings[`list-${"f".repeat(24)}`] =
        value.list_ratings[listId];
    },
    (value) => {
      value.list_ratings[listId].score_0_10 = 11;
    },
    (value) => {
      value.list_ratings[listId].unrelated_positions_1_to_5 = [2, 2];
    },
    (value) => {
      value.extra = true;
    },
  ];
  for (const mutate of mutations) {
    const ratings = validExport();
    mutate(ratings);
    sign(ratings);
    const { res, storage } = await submit(ratings);
    assert.equal(res.statusCode, 400);
    assert.equal(storage.puts.length, 0);
  }
  const ratings = validExport();
  ratings.list_ratings[listId].similarity = "not_similar";
  const { res } = await submit(ratings);
  assert.equal(res.statusCode, 400);
});

test("requires exact wrapper consent, study and at least one rating", async () => {
  for (const wrapper of [
    {
      consent: false,
      study: "semantic-repeated-excerpt-v2",
      ratings: validExport(),
    },
    { consent: true, study: "v17", ratings: validExport() },
    {
      consent: true,
      study: "semantic-repeated-excerpt-v2",
      ratings: validExport(),
      extra: true,
    },
  ]) {
    const { res } = await submit(validExport(), new MemoryStorage(), { wrapper });
    assert.equal(res.statusCode, 400);
  }
  const empty = validExport();
  empty.list_ratings = {};
  sign(empty);
  const { res } = await submit(empty);
  assert.equal(res.statusCode, 400);
});

test("deduplicates exact snapshots without overwrite", async () => {
  const storage = new MemoryStorage();
  const ratings = validExport();
  const first = await submit(ratings, storage);
  const second = await submit(ratings, storage);
  assert.equal(first.res.body.duplicate, false);
  assert.equal(second.res.body.duplicate, true);
  assert.equal(first.res.body.receipt_sha256, second.res.body.receipt_sha256);
  assert.equal(storage.puts.length, 1);
});

test("enforces bounded strict JSON and POST-only private ingestion", async () => {
  let result = await submit(validExport(), new MemoryStorage(), {
    contentLength: MAX_SEMANTIC_BODY_BYTES + 1,
  });
  assert.equal(result.res.statusCode, 413);

  result = await submit(validExport(), new MemoryStorage(), { method: "GET" });
  assert.equal(result.res.statusCode, 405);
  assert.equal(result.res.headers.Allow, "POST");

  const rawBody =
    `{"consent":true,"consent":true,"study":"semantic-repeated-excerpt-v2",` +
    `"ratings":${JSON.stringify(validExport())}}`;
  result = await submit(validExport(), new MemoryStorage(), { rawBody });
  assert.equal(result.res.statusCode, 400);
});

test("rejects untrusted origins and never exposes list or read behavior", async () => {
  const { res, storage } = await submit(validExport(), new MemoryStorage(), {
    headers: {
      origin: "https://evil.example",
      host: "evil.example",
    },
  });
  assert.equal(res.statusCode, 403);
  assert.equal(storage.puts.length, 0);

  const source = readFileSync(
    new URL("../api/ratings-semantic-v2.js", import.meta.url),
    "utf8",
  );
  assert.equal(source.includes("blobList"), false);
  assert.equal(source.includes("blobGet"), false);
  assert.equal(source.includes("access: \"public\""), false);
});

test("analyst-only tool lists the semantic prefix and validates downloads", async () => {
  const submitted = await submit();
  const put = submitted.storage.puts[0];
  const bytes = Buffer.from(put.body);
  const calls = [];
  const analystStorage = {
    async list(options) {
      calls.push(options);
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
  const directory = await mkdtemp(join(tmpdir(), "soundalike-semantic-inbox-"));
  try {
    const first = await downloadSemantic(directory, analystStorage);
    const second = await downloadSemantic(directory, analystStorage);
    assert.deepEqual(first, { downloaded: 1, existing: 0 });
    assert.deepEqual(second, { downloaded: 0, existing: 1 });
    assert.equal(
      calls.every((call) => call.prefix === SEMANTIC_BLOB_PREFIX),
      true,
    );
    const destination = join(
      directory,
      `semantic-session-${"2".repeat(24)}`,
      `${submitted.res.body.receipt_sha256}.json`,
    );
    assert.deepEqual(await readFile(destination), bytes);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("analyst tool accepts private get metadata without a content length", async () => {
  const submitted = await submit();
  const put = submitted.storage.puts[0];
  const bytes = Buffer.from(put.body);
  const analystStorage = {
    async list() {
      return {
        blobs: [{ pathname: put.pathname, size: bytes.length }],
        hasMore: false,
      };
    },
    async get(pathname) {
      return {
        statusCode: 200,
        stream: Readable.from([bytes]),
        blob: {
          pathname,
          size: 0,
          contentType: "application/json",
        },
      };
    },
  };
  const directory = await mkdtemp(join(tmpdir(), "soundalike-semantic-inbox-"));
  try {
    assert.deepEqual(await downloadSemantic(directory, analystStorage), {
      downloaded: 1,
      existing: 0,
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
