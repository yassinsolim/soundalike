import { get as blobGet, list as blobList } from "@vercel/blob";
import { lstat, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  FEEDBACK_BLOB_PREFIX,
  MAX_FEEDBACK_STORED_BYTES,
  parseFeedbackStoredRecordBytes,
} from "../api/spicetify-feedback.js";

const PATHNAME =
  /^spicetify-feedback\/match-quality-v1\/([a-f0-9]{64})\.json$/;
const DEFAULT_RETENTION_DAYS = 90;

async function readBounded(stream) {
  const chunks = [];
  let size = 0;
  for await (const chunk of stream) {
    const buffer = Buffer.from(chunk);
    size += buffer.length;
    if (size > MAX_FEEDBACK_STORED_BYTES) {
      throw new Error("A private feedback object exceeds the expected size.");
    }
    chunks.push(buffer);
  }
  return Buffer.concat(chunks);
}

async function destinationFor(root, digest) {
  const destination = resolve(root, `${digest}.json`);
  if (relative(root, destination).startsWith("..")) {
    throw new Error("Unsafe private feedback inbox destination.");
  }
  return destination;
}

export async function downloadSpicetifyFeedback(
  outputDirectory,
  storage = { get: blobGet, list: blobList },
  options = {},
) {
  const retentionDays = options.retentionDays ?? DEFAULT_RETENTION_DAYS;
  const now = options.now ?? Date.now();
  if (
    !Number.isInteger(retentionDays) ||
    retentionDays < 1 ||
    retentionDays > 3650 ||
    !Number.isFinite(now)
  ) {
    throw new Error("Invalid private feedback retention review options.");
  }
  await mkdir(outputDirectory, { recursive: true, mode: 0o700 });
  const root = await realpath(outputDirectory);
  const rootMetadata = await lstat(root);
  if (!rootMetadata.isDirectory() || rootMetadata.isSymbolicLink()) {
    throw new Error("Private feedback inbox path is not a safe directory.");
  }
  const retentionCutoff = now - retentionDays * 24 * 60 * 60 * 1000;
  let cursor;
  let downloaded = 0;
  let existing = 0;
  let retentionCandidates = 0;
  do {
    const page = await storage.list({
      prefix: FEEDBACK_BLOB_PREFIX,
      limit: 1000,
      cursor,
    });
    if (
      !page ||
      !Array.isArray(page.blobs) ||
      typeof page.hasMore !== "boolean"
    ) {
      throw new Error("Private feedback inbox listing returned an invalid page.");
    }
    for (const blob of page.blobs) {
      const match = PATHNAME.exec(blob.pathname);
      if (
        !match ||
        !Number.isInteger(blob.size) ||
        blob.size < 2 ||
        blob.size > MAX_FEEDBACK_STORED_BYTES
      ) {
        throw new Error("Unexpected object in the private feedback inbox.");
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
        !Number.isInteger(result.blob?.size) ||
        (result.blob.size !== 0 && result.blob.size !== blob.size) ||
        result.blob?.contentType !== "application/json"
      ) {
        throw new Error("A private feedback object could not be downloaded.");
      }
      const bytes = await readBounded(result.stream);
      if (bytes.length !== blob.size) {
        throw new Error("A private feedback object changed during download.");
      }
      const validated = parseFeedbackStoredRecordBytes(bytes, blob.pathname);
      if (Date.parse(validated.document.received_at) < retentionCutoff) {
        retentionCandidates += 1;
      }
      const destination = await destinationFor(root, match[1]);
      try {
        await writeFile(destination, bytes, { flag: "wx", mode: 0o600 });
        downloaded += 1;
      } catch (error) {
        if (error?.code !== "EEXIST") throw error;
        const local = await readFile(destination);
        if (!local.equals(bytes)) {
          throw new Error(
            "An existing feedback inbox file failed integrity comparison.",
          );
        }
        existing += 1;
      }
    }
    if (page.hasMore && typeof page.cursor !== "string") {
      throw new Error(
        "Private feedback inbox listing omitted its continuation cursor.",
      );
    }
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return { downloaded, existing, retentionCandidates, retentionDays };
}

function parseArguments(args) {
  const positional = [];
  let acknowledged = false;
  let retentionDays = DEFAULT_RETENTION_DAYS;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--acknowledge-private-data") {
      acknowledged = true;
    } else if (argument === "--retention-days") {
      retentionDays = Number(args[index + 1]);
      index += 1;
    } else {
      positional.push(argument);
    }
  }
  return { acknowledged, positional, retentionDays };
}

async function main() {
  const { acknowledged, positional, retentionDays } = parseArguments(
    process.argv.slice(2),
  );
  if (
    !acknowledged ||
    positional.length !== 1 ||
    !Number.isInteger(retentionDays) ||
    retentionDays < 1 ||
    retentionDays > 3650
  ) {
    console.error(
      "Usage: npm run feedback:inbox -- <output-dir> " +
        "--acknowledge-private-data [--retention-days 90]",
    );
    process.exitCode = 2;
    return;
  }
  try {
    const result = await downloadSpicetifyFeedback(positional[0], undefined, {
      retentionDays,
    });
    console.log(
      `Private feedback inbox sync complete: ${result.downloaded} downloaded, ` +
        `${result.existing} already present, ${result.retentionCandidates} ` +
        `older than ${result.retentionDays} days for manual retention review. ` +
        "No objects were deleted.",
    );
  } catch {
    console.error(
      "Private feedback inbox download failed. Check authorized Blob " +
        "credentials and retry.",
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
