import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { EventEmitter } from "node:events";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import test from "node:test";

import ratingsDispatch, { canonical } from "../api/ratings.js";
import {
  MAX_V6_BODY_BYTES,
  V6_BLOB_PREFIX,
  V6_PACK_SHA256,
  V6_PROTOCOL_SHA256,
  createV6Handler,
  parseV6StoredRecordBytes,
} from "../server/ratings-v6.js";
import { downloadV6 } from "../tools/ratings-v6-inbox.js";

const pack = JSON.parse(
  readFileSync(
    new URL("../evaluate/active-pack-v6.json", import.meta.url),
    "utf8",
  ),
);
const task = pack.tasks[0];
const KEY = "a".repeat(64);

function sign(ratings) {
  const payload = { ...ratings };
  delete payload.integrity_hmac_sha256;
  ratings.integrity_hmac_sha256 = createHmac(
    "sha256",
    Buffer.from(KEY, "hex"),
  )
    .update(canonical(payload), "utf8")
    .digest("hex");
  return ratings;
}

function validExport(outcome = "rated") {
  return sign({
    schema_version: 1,
    submission_schema: "v6_development_listener_submission_v1",
    source_kind: "human_listener",
    provider: "hosted_private_development_v6_evaluator",
    anonymous_rater_id: `anon-v6-${"1".repeat(24)}`,
    session_id: `v6-session-${"2".repeat(24)}`,
    protocol_sha256: V6_PROTOCOL_SHA256,
    pilot_pack_sha256: V6_PACK_SHA256,
    local_session_key: KEY,
    started_at: "2026-07-30T00:00:00.000Z",
    last_activity_at: "2026-07-30T00:00:01.000Z",
    task_ratings: {
      [task.task_id]: {
        outcome,
        ranked_choice_ids:
          outcome === "rated"
            ? task.candidates.map((choice) => choice.choice_id)
            : null,
        worst_primary_reason: outcome === "rated" ? "tempo_pacing" : null,
        skip_reason: outcome === "skipped" ? "out_of_scope" : null,
        completed_at: "2026-07-30T00:00:01.000Z",
        interaction_ms: 1000,
      },
    },
    exported_at: "2026-07-30T00:00:02.000Z",
    duration_ms: 2000,
    integrity_notice:
      "Local-key HMAC provides integrity, not identity or authenticity; the key is included in this export.",
  });
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
    if (this.objects.has(pathname)) throw new Error("exists");
    this.objects.set(pathname, body);
    this.puts.push({ pathname, body, options });
  }
}

async function submit(ratings = validExport(), storage = new MemoryStorage(), options = {}) {
  const res = response();
  const supplied =
    options.rawBody ??
    options.wrapper ?? {
      consent: true,
      study: "development-v6-ranking",
      ratings,
    };
  const raw = Buffer.isBuffer(supplied)
    ? supplied
    : Buffer.from(
        typeof supplied === "string" ? supplied : JSON.stringify(supplied),
        "utf8",
      );
  const request = options.replayed ? new EventEmitter() : Readable.from([raw]);
  request.method = options.method || "POST";
  request.headers = {
    origin: options.origin || "https://soundalike.yassin.app",
    host: "soundalike.yassin.app",
    "content-type": "application/json",
    ...(options.contentLength === undefined
      ? {}
      : { "content-length": String(options.contentLength) }),
  };
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
  await createV6Handler(storage)(request, res);
  return { res, storage };
}

test("stores only sanitized V6 evidence in its private prefix", async () => {
  const { res, storage } = await submit();
  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body.counts, {
    complete_task_ratings: 1,
    rated_tasks: 1,
    skipped_tasks: 0,
    unique_comparisons: 1,
  });
  assert.equal(storage.puts.length, 1);
  const put = storage.puts[0];
  assert.equal(
    put.pathname,
    `${V6_BLOB_PREFIX}v6-session-${"2".repeat(24)}/${res.body.receipt_sha256}.json`,
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
  assert.equal(
    parseV6StoredRecordBytes(put.body, put.pathname).digest,
    res.body.receipt_sha256,
  );
});

test("accepts skip evidence and rejects choice, reason, schema and HMAC tampering", async () => {
  let result = await submit(validExport("skipped"));
  assert.equal(result.res.statusCode, 200);
  assert.equal(result.res.body.counts.skipped_tasks, 1);

  const mutations = [
    (value) => {
      value.schema_version = 3;
    },
    (value) => {
      value.protocol_sha256 = "0".repeat(64);
    },
    (value) => {
      value.task_ratings[task.task_id].ranked_choice_ids[3] =
        task.candidates[0].choice_id;
    },
    (value) => {
      value.task_ratings[task.task_id].worst_primary_reason = "free text";
    },
    (value) => {
      value.extra = true;
    },
    (value) => {
      value.anonymous_rater_id = [`anon-v6-${"1".repeat(24)}`];
    },
    (value) => {
      value.session_id = [`v6-session-${"2".repeat(24)}`];
    },
  ];
  for (const mutate of mutations) {
    const value = validExport();
    mutate(value);
    sign(value);
    result = await submit(value);
    assert.equal(result.res.statusCode, 400);
  }
  const badHmac = validExport();
  badHmac.integrity_hmac_sha256 = "0".repeat(64);
  result = await submit(badHmac);
  assert.equal(result.res.statusCode, 400);
});

test("requires exact consent and enforces origin, body and method boundaries", async () => {
  for (const wrapper of [
    { consent: false, study: "development-v6-ranking", ratings: validExport() },
    { consent: true, study: "v3", ratings: validExport() },
    {
      consent: true,
      study: "development-v6-ranking",
      ratings: validExport(),
      extra: true,
    },
  ]) {
    const result = await submit(validExport(), new MemoryStorage(), { wrapper });
    assert.equal(result.res.statusCode, 400);
  }
  assert.equal(
    (await submit(validExport(), new MemoryStorage(), {
      origin: "https://evil.example",
    })).res.statusCode,
    403,
  );
  assert.equal(
    (await submit(validExport(), new MemoryStorage(), {
      contentLength: MAX_V6_BODY_BYTES + 1,
    })).res.statusCode,
    413,
  );
  assert.equal(
    (await submit(validExport(), new MemoryStorage(), {
      rawBody: Buffer.alloc(MAX_V6_BODY_BYTES + 1, 0x20),
    })).res.statusCode,
    413,
  );
  assert.equal(
    (await submit(validExport(), new MemoryStorage(), {
      rawBody: '{"consent":true,"consent":true}',
    })).res.statusCode,
    400,
  );
  assert.equal(
    (await submit(validExport(), new MemoryStorage(), { method: "GET" })).res
      .statusCode,
    405,
  );
});

test("accepts a Vercel-style replay without touching request.body", async () => {
  const result = await submit(validExport(), new MemoryStorage(), {
    replayed: true,
  });
  assert.equal(result.res.statusCode, 200);
  assert.equal(result.storage.puts.length, 1);
});

test("shared ratings function dispatches the public V6 route", async () => {
  const request = Readable.from([Buffer.from("{}")]);
  request.method = "POST";
  request.url = "/api/ratings?__soundalike_handler=ratings-v6";
  request.headers = {
    origin: "https://soundalike.yassin.app",
    host: "soundalike.yassin.app",
    "content-type": "application/json",
  };
  const res = response();

  await ratingsDispatch(request, res);
  assert.equal(res.statusCode, 400);
  assert.equal(res.body.code, "invalid_wrapper");
});

test("deduplicates exact snapshots without overwrite", async () => {
  const storage = new MemoryStorage();
  const ratings = validExport();
  const first = await submit(ratings, storage);
  const second = await submit(ratings, storage);
  assert.equal(first.res.body.duplicate, false);
  assert.equal(second.res.body.duplicate, true);
  assert.equal(storage.puts.length, 1);
});

test("analyst tool validates bounded private downloads", async () => {
  const submitted = await submit();
  const put = submitted.storage.puts[0];
  const bytes = Buffer.from(put.body);
  const analystStorage = {
    async list(options) {
      assert.equal(options.prefix, V6_BLOB_PREFIX);
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
  const directory = await mkdtemp(join(tmpdir(), "soundalike-v6-inbox-"));
  try {
    assert.deepEqual(await downloadV6(directory, analystStorage), {
      downloaded: 1,
      existing: 0,
    });
    assert.deepEqual(await downloadV6(directory, analystStorage), {
      downloaded: 0,
      existing: 1,
    });
    const saved = await readFile(
      join(
        directory,
        `v6-session-${"2".repeat(24)}`,
        `${submitted.res.body.receipt_sha256}.json`,
      ),
    );
    assert.equal(saved.equals(bytes), true);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("submission endpoint exposes no private listing or read capability", () => {
  const source = readFileSync(
    new URL("../server/ratings-v6.js", import.meta.url),
    "utf8",
  );
  assert.equal(source.includes("blobList"), false);
  assert.equal(source.includes("blobGet"), false);
  assert.equal(source.includes('access: "public"'), false);
});
