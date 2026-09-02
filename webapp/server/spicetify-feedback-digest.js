import {
  BlobNotFoundError,
  del as blobDel,
  get as blobGet,
  head as blobHead,
  list as blobList,
  put as blobPut,
} from "@vercel/blob";
import { timingSafeEqual } from "node:crypto";

import {
  createDiscordWebhookSender,
  formatFeedbackDigest,
  validateDiscordWebhookUrl,
} from "./discord-feedback.js";
import {
  FEEDBACK_BLOB_PREFIX,
  FEEDBACK_DIGEST_INPUT_PREFIX,
  MAX_FEEDBACK_STORED_BYTES,
  parseFeedbackStoredRecordBytes,
} from "./spicetify-feedback.js";

export const FEEDBACK_DIGEST_BLOB_PREFIX =
  "spicetify-feedback/digest-v1/";
const DIGEST_INPUT_PATH =
  /^spicetify-feedback\/digest-input-v1\/(\d{4}-\d{2}-\d{2})\/([a-f0-9]{64})\.json$/u;
const DIGEST_INPUT_KEYS = ["date", "record_path", "schema_version"].sort();
const MAX_DIGEST_INPUT_BYTES = 1024;
const DAY_MS = 24 * 60 * 60 * 1000;
const MAX_RECORDS = 10_000;

function header(request, name) {
  if (typeof request.headers?.get === "function") {
    const value = request.headers.get(name);
    return typeof value === "string" ? value : undefined;
  }
  const found = Object.entries(request.headers || {}).find(
    ([key]) => key.toLowerCase() === name.toLowerCase(),
  );
  return found && typeof found[1] === "string" ? found[1] : undefined;
}

function setHeaders(response) {
  response.setHeader("Cache-Control", "no-store, max-age=0");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'none'; frame-ancestors 'none'; sandbox",
  );
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader("X-Content-Type-Options", "nosniff");
}

function send(response, status, body) {
  setHeaders(response);
  if (typeof response.status === "function") {
    return response.status(status).json(body);
  }
  response.statusCode = status;
  return response.end(JSON.stringify(body));
}

function authorized(actual, secret) {
  if (typeof actual !== "string" || typeof secret !== "string" || !secret) {
    return false;
  }
  const expectedBytes = Buffer.from(`Bearer ${secret}`, "utf8");
  const actualBytes = Buffer.from(actual, "utf8");
  return (
    expectedBytes.length === actualBytes.length &&
    timingSafeEqual(expectedBytes, actualBytes)
  );
}

function previousUtcDay(now) {
  const date = new Date(now);
  if (!Number.isFinite(date.getTime())) {
    throw new Error("Invalid digest time");
  }
  const end = Date.UTC(
    date.getUTCFullYear(),
    date.getUTCMonth(),
    date.getUTCDate(),
  );
  const start = end - DAY_MS;
  return {
    start,
    end,
    date: new Date(start).toISOString().slice(0, 10),
  };
}

function isNotFound(error) {
  return (
    error instanceof BlobNotFoundError ||
    error?.name === "BlobNotFoundError"
  );
}

async function exists(storage, pathname) {
  try {
    const metadata = await storage.head(pathname);
    if (metadata?.pathname !== pathname) {
      throw new Error("Digest storage returned an unexpected object");
    }
    return true;
  } catch (error) {
    if (isNotFound(error)) return false;
    throw error;
  }
}

async function readBounded(stream) {
  const chunks = [];
  let size = 0;
  for await (const chunk of stream) {
    const bytes = Buffer.from(chunk);
    size += bytes.length;
    if (size > MAX_FEEDBACK_STORED_BYTES) {
      throw new Error("A private feedback object exceeds the expected size");
    }
    chunks.push(bytes);
  }
  return Buffer.concat(chunks);
}

function exactKeys(value, expected) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    JSON.stringify(Object.keys(value).sort()) === JSON.stringify(expected)
  );
}

async function readObject(storage, blob, maximumBytes) {
  if (
    !Number.isInteger(blob.size) ||
    blob.size < 2 ||
    blob.size > maximumBytes
  ) {
    throw new Error("Unexpected object in the private feedback digest");
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
    throw new Error("A private feedback object could not be downloaded");
  }
  const bytes = await readBounded(result.stream);
  if (bytes.length !== blob.size) {
    throw new Error("A private feedback object changed during download");
  }
  return bytes;
}

async function recordsForDay(storage, period) {
  const records = [];
  let cursor;
  do {
    const page = await storage.list({
      prefix: `${FEEDBACK_DIGEST_INPUT_PREFIX}${period.date}/`,
      limit: 1000,
      cursor,
    });
    if (
      !page ||
      !Array.isArray(page.blobs) ||
      typeof page.hasMore !== "boolean"
    ) {
      throw new Error("Feedback digest listing returned an invalid page");
    }
    for (const blob of page.blobs) {
      if (records.length >= MAX_RECORDS) {
        throw new Error("Feedback digest record limit exceeded");
      }
      const match = DIGEST_INPUT_PATH.exec(blob.pathname);
      if (!match || match[1] !== period.date) {
        throw new Error("Unexpected object in the private feedback digest");
      }
      const indexBytes = await readObject(
        storage,
        blob,
        MAX_DIGEST_INPUT_BYTES,
      );
      const indexText = new TextDecoder("utf-8", { fatal: true }).decode(
        indexBytes,
      );
      const index = JSON.parse(indexText);
      const expectedRecordPath = `${FEEDBACK_BLOB_PREFIX}${match[2]}.json`;
      if (
        !exactKeys(index, DIGEST_INPUT_KEYS) ||
        index.schema_version !== 1 ||
        index.date !== period.date ||
        index.record_path !== expectedRecordPath ||
        indexText !== `${JSON.stringify({
          date: period.date,
          record_path: expectedRecordPath,
          schema_version: 1,
        })}\n`
      ) {
        throw new Error("Invalid private feedback digest index");
      }
      const metadata = await storage.head(expectedRecordPath);
      const recordBytes = await readObject(
        storage,
        metadata,
        MAX_FEEDBACK_STORED_BYTES,
      );
      const validated = parseFeedbackStoredRecordBytes(
        recordBytes,
        expectedRecordPath,
      );
      const receivedAt = Date.parse(validated.document.received_at);
      if (receivedAt < period.start || receivedAt >= period.end) {
        throw new Error("Feedback digest index date does not match its record");
      }
      records.push(validated.payload);
    }
    if (page.hasMore && typeof page.cursor !== "string") {
      throw new Error("Feedback digest listing omitted its continuation cursor");
    }
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return records;
}

export function summarizeFeedback(date, records) {
  const summary = {
    date,
    total: records.length,
    selections: { good: 0, mixed: 0, off: 0 },
    reasons: {
      style: 0,
      mood_energy: 0,
      tempo: 0,
      vocals_language: 0,
      instruments_timbre: 0,
    },
    flagged: [],
  };
  for (const record of records) {
    summary.selections[record.selection] += 1;
    for (const reason of record.reasons) summary.reasons[reason] += 1;
    if (record.selection !== "good") {
      summary.flagged.push({
        selection: record.selection,
        reasons: [...record.reasons],
        seed: { ...record.seed },
      });
    }
  }
  return summary;
}

async function claimDigest(storage, pathname, date, now) {
  if (await exists(storage, pathname)) return false;
  try {
    await storage.put(
      pathname,
      `${JSON.stringify({
        schema_version: 1,
        date,
        claimed_at: new Date(now).toISOString(),
      })}\n`,
      {
        access: "private",
        addRandomSuffix: false,
        allowOverwrite: false,
        contentType: "application/json",
      },
    );
    return true;
  } catch (error) {
    if (await exists(storage, pathname)) return false;
    throw error;
  }
}

export function createFeedbackDigestHandler(
  storage = {
    get: blobGet,
    head: blobHead,
    list: blobList,
    put: blobPut,
    del: blobDel,
  },
  options = {},
) {
  const now = options.now ?? (() => Date.now());
  const logger = options.logger ?? console;
  return async function feedbackDigestHandler(request, response) {
    if (request.method !== "GET") {
      response.setHeader("Allow", "GET");
      return send(response, 405, { error: "method not allowed" });
    }
    const cronSecret = options.cronSecret ?? process.env.CRON_SECRET;
    if (!cronSecret) {
      return send(response, 503, { error: "digest is not configured" });
    }
    if (!authorized(header(request, "authorization"), cronSecret)) {
      return send(response, 401, { error: "unauthorized" });
    }
    let sendDiscord = options.sendDiscord;
    if (!sendDiscord) {
      try {
        const webhookUrl = validateDiscordWebhookUrl(
          options.webhookUrl ??
            process.env.SOUNDALIKE_FEEDBACK_DISCORD_WEBHOOK,
        );
        if (!webhookUrl) {
          return send(response, 503, { error: "digest is not configured" });
        }
        sendDiscord = createDiscordWebhookSender({ webhookUrl });
      } catch {
        return send(response, 503, { error: "digest is not configured" });
      }
    }
    const timestamp = now();
    const period = previousUtcDay(timestamp);
    const markerPath = `${FEEDBACK_DIGEST_BLOB_PREFIX}${period.date}.json`;
    let claimed = false;
    try {
      claimed = await claimDigest(
        storage,
        markerPath,
        period.date,
        timestamp,
      );
      if (!claimed) {
        return send(response, 200, {
          date: period.date,
          sent: false,
          already_processed: true,
        });
      }
      const records = await recordsForDay(storage, period);
      const summary = summarizeFeedback(period.date, records);
      if (summary.total > 0) {
        await sendDiscord(formatFeedbackDigest(summary));
      }
      return send(response, 200, {
        date: period.date,
        record_count: summary.total,
        sent: summary.total > 0,
      });
    } catch {
      if (claimed) {
        try {
          await storage.del(markerPath);
        } catch {
          logger.error("Soundalike feedback digest claim cleanup failed.");
        }
      }
      logger.error("Soundalike feedback digest failed.");
      return send(response, 502, { error: "digest failed" });
    }
  };
}

export default createFeedbackDigestHandler();
