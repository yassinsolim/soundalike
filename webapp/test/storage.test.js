import assert from "node:assert/strict";
import {
  mkdtemp,
  readdir,
  readFile,
  rm,
  symlink,
  writeFile,
  mkdir,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  ObjectExistsError,
  createDiskStorage,
  head as defaultHead,
  isObjectNotFound,
} from "../server/storage.js";

const JSON_PUT = {
  access: "private",
  addRandomSuffix: false,
  allowOverwrite: false,
  contentType: "application/json",
};

async function withStorage(run) {
  const root = await mkdtemp(join(tmpdir(), "soundalike-storage-"));
  try {
    await run(createDiskStorage(root), root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

async function readStream(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

test("put, head and get round-trip an object", async () => {
  await withStorage(async (storage, root) => {
    const pathname = "human-ratings/v17/session-abc/digest.json";
    const body = '{"a":1}\n';
    await storage.put(pathname, body, JSON_PUT);
    const metadata = await storage.head(pathname);
    assert.equal(metadata.pathname, pathname);
    assert.equal(metadata.size, Buffer.byteLength(body));
    assert.equal(metadata.contentType, "application/json");
    const result = await storage.get(pathname, { access: "private", useCache: false });
    assert.equal(result.statusCode, 200);
    assert.deepEqual(result.blob.pathname, pathname);
    assert.equal(result.blob.size, Buffer.byteLength(body));
    assert.equal(result.blob.contentType, "application/json");
    assert.equal((await readStream(result.stream)).toString("utf8"), body);
    assert.deepEqual(
      await readdir(join(root, "human-ratings", "v17", "session-abc")),
      ["digest.json"],
    );
  });
});

test("put never overwrites unless allowOverwrite is set", async () => {
  await withStorage(async (storage, root) => {
    const pathname = "spicetify-feedback/digest-v1/2026-11-11.json";
    await storage.put(pathname, "first\n", JSON_PUT);
    await assert.rejects(
      storage.put(pathname, "second\n", JSON_PUT),
      (error) => error instanceof ObjectExistsError && error.code === "EEXIST",
    );
    const file = join(root, "spicetify-feedback", "digest-v1", "2026-11-11.json");
    assert.equal(await readFile(file, "utf8"), "first\n");
    await storage.put(pathname, "third\n", { ...JSON_PUT, allowOverwrite: true });
    assert.equal(await readFile(file, "utf8"), "third\n");
    assert.deepEqual(
      await readdir(join(root, "spicetify-feedback", "digest-v1")),
      ["2026-11-11.json"],
    );
  });
});

test("concurrent puts of one pathname create it exactly once", async () => {
  await withStorage(async (storage) => {
    const pathname = "human-ratings/v17/session-race/digest.json";
    const results = await Promise.allSettled(
      Array.from({ length: 8 }, (_, index) =>
        storage.put(pathname, `${index}\n`, JSON_PUT),
      ),
    );
    assert.equal(results.filter((r) => r.status === "fulfilled").length, 1);
    assert.ok(
      results
        .filter((r) => r.status === "rejected")
        .every((r) => r.reason instanceof ObjectExistsError),
    );
  });
});

test("missing objects raise a recognizable not-found error", async () => {
  await withStorage(async (storage) => {
    await assert.rejects(storage.head("a/missing.json"), isObjectNotFound);
    await assert.rejects(storage.get("a/missing.json"), isObjectNotFound);
    await storage.put("a/b.json", "{}", JSON_PUT);
    await assert.rejects(storage.head("a/b.json/c.json"), isObjectNotFound);
    await assert.rejects(storage.head("a"), isObjectNotFound);
  });
});

test("list filters by string prefix, sorts, pages and skips temporary files", async () => {
  await withStorage(async (storage, root) => {
    const names = [
      "human-ratings/v17/session-b/2.json",
      "human-ratings/v17/session-a/1.json",
      "human-ratings/v17/session-a/3.json",
      "human-ratings/v1/session-z/9.json",
      "human-ratings/v170/session-y/8.json",
      "spicetify-feedback/match-quality-v1/x.json",
    ];
    for (const name of names) await storage.put(name, "{}\n", JSON_PUT);
    await writeFile(
      join(root, "human-ratings", "v17", "session-a", ".tmp-1-abc"),
      "partial",
    );

    const seen = [];
    let cursor;
    let pages = 0;
    do {
      const page = await storage.list({
        prefix: "human-ratings/v17/",
        limit: 2,
        cursor,
      });
      pages += 1;
      for (const blob of page.blobs) {
        assert.equal(blob.size, 3);
        seen.push(blob.pathname);
      }
      assert.equal(typeof page.hasMore, "boolean");
      cursor = page.hasMore ? page.cursor : undefined;
    } while (cursor);
    assert.equal(pages, 2);
    assert.deepEqual(seen, [
      "human-ratings/v17/session-a/1.json",
      "human-ratings/v17/session-a/3.json",
      "human-ratings/v17/session-b/2.json",
    ]);

    const partial = await storage.list({ prefix: "human-ratings/v1" });
    assert.equal(partial.blobs.length, 5);
    assert.equal(partial.hasMore, false);
    const empty = await storage.list({ prefix: "spicetify-feedback/digest-v1/" });
    assert.deepEqual(empty, { blobs: [], hasMore: false, cursor: undefined });
  });
});

test("del removes objects and ignores missing ones", async () => {
  await withStorage(async (storage) => {
    await storage.put("a/b.json", "{}", JSON_PUT);
    await storage.del("a/b.json");
    await assert.rejects(storage.head("a/b.json"), isObjectNotFound);
    await storage.del("a/b.json");
  });
});

test("unsafe pathnames and prefixes are rejected before touching disk", async () => {
  await withStorage(async (storage, root) => {
    for (const pathname of [
      "",
      "../escape.json",
      "a/../../escape.json",
      "/absolute.json",
      "a//b.json",
      "a/./b.json",
      ".hidden/b.json",
      "a/.tmp-1",
      "a\\b.json",
      "C:/windows.json",
      `${"a/".repeat(9)}b.json`,
      `${"a".repeat(129)}.json`,
      42,
    ]) {
      await assert.rejects(storage.put(pathname, "{}", JSON_PUT), TypeError);
      await assert.rejects(storage.head(pathname), TypeError);
      await assert.rejects(storage.get(pathname), TypeError);
      await assert.rejects(storage.del(pathname), TypeError);
    }
    await assert.rejects(storage.list({ prefix: "../" }), TypeError);
    await assert.rejects(storage.list({ prefix: "a/", cursor: 5 }), TypeError);
    await assert.rejects(
      storage.put("a/b.json", "{}", { ...JSON_PUT, addRandomSuffix: true }),
      TypeError,
    );
    assert.deepEqual(await readdir(root), []);
  });
});

test("symlinks inside the root are not followed", async (t) => {
  await withStorage(async (storage, root) => {
    const outside = await mkdtemp(join(tmpdir(), "soundalike-outside-"));
    try {
      await writeFile(join(outside, "secret.json"), "{}");
      await mkdir(join(root, "a"));
      try {
        await symlink(join(outside, "secret.json"), join(root, "a", "link.json"));
      } catch {
        t.skip("symlinks are not permitted on this machine");
        return;
      }
      await assert.rejects(storage.head("a/link.json"), isObjectNotFound);
      if (process.platform !== "win32") {
        await assert.rejects(storage.get("a/link.json"), isObjectNotFound);
      }
      const page = await storage.list({ prefix: "a/" });
      assert.deepEqual(page.blobs, []);
    } finally {
      await rm(outside, { recursive: true, force: true });
    }
  });
});

test("storage fails closed when the root is not configured", async () => {
  const saved = process.env.SOUNDALIKE_STORAGE_DIR;
  delete process.env.SOUNDALIKE_STORAGE_DIR;
  try {
    await assert.rejects(defaultHead("a/b.json"), (error) => !isObjectNotFound(error));
    await assert.rejects(
      createDiskStorage("relative/path").put("a/b.json", "{}", JSON_PUT),
      /not configured/,
    );
  } finally {
    if (saved !== undefined) process.env.SOUNDALIKE_STORAGE_DIR = saved;
  }
});
