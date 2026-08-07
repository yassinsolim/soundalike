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
  MAX_PACING_BODY_BYTES,
  PACING_BLOB_PREFIX,
  PACING_PACK_SHA256,
  PACING_PROTOCOL_SHA256,
  createPacingHandler,
  parsePacingStoredRecordBytes,
} from "../api/ratings-pacing-v3.js";
import { downloadPacing } from "../tools/ratings-pacing-v3-inbox.js";

const pack = JSON.parse(
  readFileSync(
    new URL("../evaluate-pacing-v3/pacing-pack.json", import.meta.url),
    "utf8",
  ),
);
const listId = pack.seeds[0].lists[0].list_id;
const resultId = pack.seeds[0].lists[0].ranking[0].result_id;
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
    schema_version: 3,
    submission_schema: "pacing_v3_listener_submission_v1",
    source_kind: "human_listener",
    provider: "hosted_private_pacing_v3_evaluator",
    anonymous_rater_id: `anon-pacing-${"1".repeat(24)}`,
    session_id: `pacing-session-${"2".repeat(24)}`,
    protocol_sha256: PACING_PROTOCOL_SHA256,
    pilot_pack_sha256: PACING_PACK_SHA256,
    local_session_key: KEY,
    started_at: "2026-07-30T00:00:00.000Z",
    last_activity_at: "2026-07-30T00:00:01.000Z",
    list_ratings: {
      [listId]: {
        score_0_10: 9,
        rated_at: "2026-07-30T00:00:01.000Z",
        interaction_ms: 1000,
      },
    },
    result_ratings: {
      [resultId]: {
        score_0_10: 8,
        mismatch_reasons: ["tempo_pacing", "tone_timbre"],
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
      ? { consent: true, study: "pacing-v3-blind", ratings }
      : options.wrapper;
  await createPacingHandler(storage, options.deploymentHost)(
    request(options.rawBody ?? wrapper, options),
    res,
  );
  return { res, storage };
}

test("accepts pacing evidence only in its immutable private prefix", async () => {
  const { res, storage } = await submit();
  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body.counts, {
    complete_list_ratings: 1,
    complete_result_ratings: 1,
  });
  assert.equal(res.body.duplicate, false);
  assert.match(res.body.receipt_sha256, /^[a-f0-9]{64}$/);
  assert.equal(storage.puts.length, 1);
  const put = storage.puts[0];
  assert.equal(
    put.pathname,
    `${PACING_BLOB_PREFIX}pacing-session-${"2".repeat(24)}/${res.body.receipt_sha256}.json`,
  );
  assert.equal(put.pathname.startsWith("human-ratings/semantic-v2/"), false);
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
    parsePacingStoredRecordBytes(put.body, put.pathname).digest,
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
      value.result_ratings[resultId].mismatch_reasons = [
        "tone_timbre",
        "tempo_pacing",
      ];
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
  ratings.result_ratings[resultId].mismatch_reasons = ["free text"];
  const { res } = await submit(ratings);
  assert.equal(res.statusCode, 400);
});

test("requires exact wrapper consent, study and at least one rating", async () => {
  for (const wrapper of [
    {
      consent: false,
      study: "pacing-v3-blind",
      ratings: validExport(),
    },
    { consent: true, study: "v17", ratings: validExport() },
    {
      consent: true,
      study: "pacing-v3-blind",
      ratings: validExport(),
      extra: true,
    },
  ]) {
    const { res } = await submit(validExport(), new MemoryStorage(), { wrapper });
    assert.equal(res.statusCode, 400);
  }
  const empty = validExport();
  empty.list_ratings = {};
  empty.result_ratings = {};
  sign(empty);
  const { res } = await submit(empty);
  assert.equal(res.statusCode, 400);
});

test("accepts independently complete list-only or result-only partial evidence", async () => {
  const listOnly = validExport();
  listOnly.result_ratings = {};
  sign(listOnly);
  let result = await submit(listOnly);
  assert.equal(result.res.statusCode, 200);
  assert.deepEqual(result.res.body.counts, {
    complete_list_ratings: 1,
    complete_result_ratings: 0,
  });

  const resultOnly = validExport();
  resultOnly.list_ratings = {};
  sign(resultOnly);
  result = await submit(resultOnly);
  assert.equal(result.res.statusCode, 200);
  assert.deepEqual(result.res.body.counts, {
    complete_list_ratings: 0,
    complete_result_ratings: 1,
  });
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
    contentLength: MAX_PACING_BODY_BYTES + 1,
  });
  assert.equal(result.res.statusCode, 413);

  result = await submit(validExport(), new MemoryStorage(), { method: "GET" });
  assert.equal(result.res.statusCode, 405);
  assert.equal(result.res.headers.Allow, "POST");

  const rawBody =
    `{"consent":true,"consent":true,"study":"pacing-v3-blind",` +
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
    new URL("../api/ratings-pacing-v3.js", import.meta.url),
    "utf8",
  );
  assert.equal(source.includes("blobList"), false);
  assert.equal(source.includes("blobGet"), false);
  assert.equal(source.includes("access: \"public\""), false);
});

test("analyst-only tool lists the pacing prefix and validates downloads", async () => {
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
  const directory = await mkdtemp(join(tmpdir(), "soundalike-pacing-inbox-"));
  try {
    const first = await downloadPacing(directory, analystStorage);
    const second = await downloadPacing(directory, analystStorage);
    assert.deepEqual(first, { downloaded: 1, existing: 0 });
    assert.deepEqual(second, { downloaded: 0, existing: 1 });
    assert.equal(
      calls.every((call) => call.prefix === PACING_BLOB_PREFIX),
      true,
    );
    const destination = join(
      directory,
      `pacing-session-${"2".repeat(24)}`,
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
  const directory = await mkdtemp(join(tmpdir(), "soundalike-pacing-inbox-"));
  try {
    assert.deepEqual(await downloadPacing(directory, analystStorage), {
      downloaded: 1,
      existing: 0,
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
