import { get as blobGet, list as blobList } from "@vercel/blob";
import { lstat, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  MAX_STORED_BYTES,
  parseStoredRecordBytes,
} from "../api/ratings.js";

const PREFIX = "human-ratings/v17/";
const PATHNAME =
  /^human-ratings\/v17\/(session-[a-f0-9]{24})\/([a-f0-9]{64})\.json$/;

function usage() {
  console.error(
    "Usage: npm run ratings:inbox -- <output-dir> --acknowledge-private-data",
  );
}

async function readBounded(stream) {
  const chunks = [];
  let size = 0;
  for await (const chunk of stream) {
    const buffer = Buffer.from(chunk);
    size += buffer.length;
    if (size > MAX_STORED_BYTES) {
      throw new Error("A private ratings object exceeds the expected size.");
    }
    chunks.push(buffer);
  }
  return Buffer.concat(chunks);
}

async function destinationFor(root, sessionId, digest) {
  const directory = resolve(root, sessionId);
  const destination = resolve(directory, `${digest}.json`);
  if (
    relative(root, directory).startsWith("..") ||
    relative(directory, destination).startsWith("..")
  ) {
    throw new Error("Unsafe private inbox destination.");
  }
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const metadata = await lstat(directory);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw new Error("Private inbox session path is not a safe directory.");
  }
  return destination;
}

export async function download(
  outputDirectory,
  storage = { get: blobGet, list: blobList },
) {
  await mkdir(outputDirectory, { recursive: true, mode: 0o700 });
  const root = await realpath(outputDirectory);
  let cursor;
  let downloaded = 0;
  let existing = 0;
  do {
    const page = await storage.list({ prefix: PREFIX, limit: 1000, cursor });
    if (!page || !Array.isArray(page.blobs) || typeof page.hasMore !== "boolean") {
      throw new Error("Private inbox listing returned an invalid page.");
    }
    for (const blob of page.blobs) {
      const match = PATHNAME.exec(blob.pathname);
      if (!match) throw new Error("Unexpected object path in the private inbox.");
      if (
        !Number.isInteger(blob.size) ||
        blob.size < 2 ||
        blob.size > MAX_STORED_BYTES
      ) {
        throw new Error("A private ratings object has an invalid listed size.");
      }
      const result = await storage.get(blob.pathname, {
        access: "private",
        useCache: false,
      });
      if (
        !result ||
        result.statusCode !== 200 ||
        !result.stream ||
        result.blob?.pathname !== blob.pathname ||
        result.blob?.size !== blob.size ||
        result.blob?.contentType !== "application/json"
      ) {
        throw new Error("A private ratings object could not be downloaded.");
      }
      const bytes = await readBounded(result.stream);
      if (bytes.length !== blob.size) {
        throw new Error("A private ratings object size changed during download.");
      }
      parseStoredRecordBytes(bytes, blob.pathname);
      const destination = await destinationFor(root, match[1], match[2]);
      try {
        await writeFile(destination, bytes, { flag: "wx", mode: 0o600 });
        downloaded += 1;
      } catch (error) {
        if (error?.code !== "EEXIST") throw error;
        const local = await readFile(destination);
        if (!local.equals(bytes)) {
          throw new Error("An existing private inbox file failed integrity comparison.");
        }
        existing += 1;
      }
    }
    if (page.hasMore && typeof page.cursor !== "string") {
      throw new Error("Private inbox listing omitted its continuation cursor.");
    }
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return { downloaded, existing };
}

async function main() {
  const args = process.argv.slice(2);
  const acknowledged = args.includes("--acknowledge-private-data");
  const positional = args.filter((arg) => arg !== "--acknowledge-private-data");
  if (!acknowledged || positional.length !== 1) {
    usage();
    process.exitCode = 2;
    return;
  }
  try {
    const result = await download(positional[0]);
    console.log(
      `Private inbox sync complete: ${result.downloaded} downloaded, ${result.existing} already present.`,
    );
  } catch {
    console.error(
      "Private inbox download failed. Check authorized Blob credentials and retry.",
    );
    process.exitCode = 1;
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  await main();
}
