import {
  BlobNotFoundError,
  get as blobGet,
  head as blobHead,
  put as blobPut,
} from "@vercel/blob";
import { waitUntil as vercelWaitUntil } from "@vercel/functions";
import { createHash } from "node:crypto";
import { TextDecoder } from "node:util";

import { canonical, strictJsonParse } from "../api/ratings.js";
import {
  createDiscordWebhookSender,
  formatFeedbackNotification,
} from "./discord-feedback.js";

export const FEEDBACK_BLOB_PREFIX = "spicetify-feedback/match-quality-v1/";
export const FEEDBACK_DIGEST_INPUT_PREFIX =
  "spicetify-feedback/digest-input-v1/";
export const MAX_FEEDBACK_BODY_BYTES = 32 * 1024;
export const MAX_FEEDBACK_STORED_BYTES = 40 * 1024;

const SURVEY_VERSION = "spicetify-match-feedback-v1";
const SELECTION_POLICIES = new Set([
  "top-20-strict-language-related-artist-v1",
  "top-20-strict-language-related-artist-model-quality-v1",
]);
const LANGUAGE_POLICY = "spotify-lyrics-strict-v2";
const HEX_32 = /^[a-f0-9]{32}$/;
const INDEX_VERSION = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const CONTROL_CHARACTERS = /[\u0000-\u001f\u007f-\u009f]/u;
const NOTE_CONTROL_CHARACTERS = /[\u0000-\u001f\u007f-\u009f]+/gu;
const METHODS = new Set([
  "dual_sonic64_guardrail",
  "sonic64_stable_head",
  "legacy_no_sonic_seed",
  "unknown",
]);
const API_VERSIONS = new Set(["4", "legacy", "local", "unknown"]);
const SOURCES = new Set(["hosted", "local"]);
const SELECTIONS = new Set(["good", "mixed", "off"]);
const REASONS = new Set([
  "style",
  "mood_energy",
  "tempo",
  "vocals_language",
  "instruments_timbre",
]);
const PAYLOAD_KEYS = [
  "api_version",
  "displayed_results",
  "index_version",
  "install_nonce",
  "language_policy",
  "method",
  "note",
  "reasons",
  "schema_version",
  "seed",
  "selection",
  "selection_policy",
  "session_nonce",
  "source",
  "survey_version",
].sort();
const TRACK_KEYS = ["artist", "title"].sort();
const RESULT_KEYS = ["artist", "position", "title"].sort();
const STORED_KEYS = [
  ...PAYLOAD_KEYS,
  "canonical_payload_sha256",
  "received_at",
].sort();

function sha256(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function isRecord(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (Object.getPrototypeOf(value) === Object.prototype ||
      Object.getPrototypeOf(value) === null)
  );
}

function hasExactKeys(value, expected) {
  return (
    isRecord(value) &&
    JSON.stringify(Object.keys(value).sort()) === JSON.stringify(expected)
  );
}

function cleanLabel(value, maximumLength) {
  if (
    typeof value !== "string" ||
    value.length > maximumLength ||
    CONTROL_CHARACTERS.test(value)
  ) {
    return null;
  }
  const cleaned = value.normalize("NFC").trim();
  return cleaned && cleaned.length <= maximumLength ? cleaned : null;
}

function cleanNote(value) {
  if (typeof value !== "string" || value.length > 280) return null;
  const cleaned = value
    .normalize("NFC")
    .replace(NOTE_CONTROL_CHARACTERS, " ")
    .replace(/\s{2,}/gu, " ")
    .trim();
  return cleaned.length <= 280 ? cleaned : null;
}

function parseTimestamp(value) {
  if (typeof value !== "string" || !ISO_TIMESTAMP.test(value)) return null;
  const milliseconds = Date.parse(value);
  return Number.isFinite(milliseconds) &&
    new Date(milliseconds).toISOString() === value
    ? milliseconds
    : null;
}

export function validateFeedbackPayload(value) {
  if (
    !hasExactKeys(value, PAYLOAD_KEYS) ||
    value.schema_version !== 1 ||
    value.survey_version !== SURVEY_VERSION ||
    typeof value.install_nonce !== "string" ||
    !HEX_32.test(value.install_nonce) ||
    typeof value.session_nonce !== "string" ||
    !HEX_32.test(value.session_nonce) ||
    !METHODS.has(value.method) ||
    typeof value.index_version !== "string" ||
    !INDEX_VERSION.test(value.index_version) ||
    !API_VERSIONS.has(value.api_version) ||
    value.language_policy !== LANGUAGE_POLICY ||
    !SELECTION_POLICIES.has(value.selection_policy) ||
    !SOURCES.has(value.source) ||
    (value.source === "local") !== (value.api_version === "local") ||
    !SELECTIONS.has(value.selection) ||
    !hasExactKeys(value.seed, TRACK_KEYS)
  ) {
    return null;
  }
  const seed = {
    title: cleanLabel(value.seed.title, 300),
    artist: cleanLabel(value.seed.artist, 300),
  };
  if (!seed.title || !seed.artist) return null;

  if (
    !Array.isArray(value.displayed_results) ||
    value.displayed_results.length < 1 ||
    value.displayed_results.length > 20
  ) {
    return null;
  }
  const displayedResults = [];
  for (const [index, result] of value.displayed_results.entries()) {
    if (
      !hasExactKeys(result, RESULT_KEYS) ||
      result.position !== index + 1
    ) {
      return null;
    }
    const title = cleanLabel(result.title, 300);
    const artist = cleanLabel(result.artist, 300);
    if (!title || !artist) return null;
    displayedResults.push({ position: index + 1, title, artist });
  }

  if (
    !Array.isArray(value.reasons) ||
    value.reasons.length > 2 ||
    new Set(value.reasons).size !== value.reasons.length ||
    !value.reasons.every((reason) => REASONS.has(reason))
  ) {
    return null;
  }
  const note = cleanNote(value.note);
  if (
    note === null ||
    (value.selection === "good" &&
      (value.reasons.length !== 0 || note !== ""))
  ) {
    return null;
  }
  return {
    schema_version: 1,
    survey_version: SURVEY_VERSION,
    install_nonce: value.install_nonce,
    session_nonce: value.session_nonce,
    seed,
    displayed_results: displayedResults,
    method: value.method,
    index_version: value.index_version,
    api_version: value.api_version,
    language_policy: LANGUAGE_POLICY,
    selection_policy: value.selection_policy,
    source: value.source,
    selection: value.selection,
    reasons: [...value.reasons],
    note,
  };
}

export function validateFeedbackStoredRecord(document, pathname) {
  if (!hasExactKeys(document, STORED_KEYS)) return null;
  const payload = Object.fromEntries(
    PAYLOAD_KEYS.map((key) => [key, document[key]]),
  );
  const accepted = validateFeedbackPayload(payload);
  const receivedAt = parseTimestamp(document.received_at);
  if (!accepted || receivedAt === null) return null;
  const digest = sha256(canonical(accepted));
  const expectedPath = `${FEEDBACK_BLOB_PREFIX}${digest}.json`;
  if (
    document.canonical_payload_sha256 !== digest ||
    (pathname !== undefined && pathname !== expectedPath)
  ) {
    return null;
  }
  return { digest, pathname: expectedPath, payload: accepted };
}

export function parseFeedbackStoredRecordBytes(value, pathname) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  if (bytes.length < 2 || bytes.length > MAX_FEEDBACK_STORED_BYTES) {
    throw new Error("Invalid private Spicetify feedback record size");
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  const document = strictJsonParse(text);
  const validated = validateFeedbackStoredRecord(document, pathname);
  if (!validated || text !== `${canonical(document)}\n`) {
    throw new Error("Invalid private Spicetify feedback record");
  }
  return { document, ...validated };
}

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

async function readBody(request) {
  const length = header(request, "content-length");
  if (
    length !== undefined &&
    (!/^(0|[1-9]\d*)$/.test(length) ||
      Number(length) > MAX_FEEDBACK_BODY_BYTES)
  ) {
    const error = new Error("payload");
    error.statusCode = 413;
    throw error;
  }
  // Vercel replays its lazily buffered request through data/end listeners.
  // Avoid request.body so duplicate JSON keys remain observable.
  const raw = await new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let settled = false;
    const finish = (error, value) => {
      if (settled) return;
      settled = true;
      if (error) reject(error);
      else resolve(value);
    };
    request.on("data", (chunk) => {
      if (settled) return;
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      size += bytes.length;
      if (size > MAX_FEEDBACK_BODY_BYTES) {
        const error = new Error("payload");
        error.statusCode = 413;
        finish(error);
        return;
      }
      chunks.push(bytes);
    });
    request.on("end", () => finish(null, Buffer.concat(chunks)));
    request.on("error", () => finish(new Error("request stream failed")));
    request.on("aborted", () => finish(new Error("request stream aborted")));
  });
  if (raw.length > MAX_FEEDBACK_BODY_BYTES) {
    const error = new Error("payload");
    error.statusCode = 413;
    throw error;
  }
  try {
    return strictJsonParse(
      new TextDecoder("utf-8", { fatal: true }).decode(raw),
    );
  } catch {
    const error = new Error("json");
    error.statusCode = 400;
    throw error;
  }
}

function setResponseHeaders(response) {
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  response.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Max-Age", "86400");
  response.setHeader("Cache-Control", "no-store, max-age=0");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'none'; frame-ancestors 'none'; sandbox",
  );
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader("X-Content-Type-Options", "nosniff");
}

function send(response, status, body) {
  setResponseHeaders(response);
  if (typeof response.status === "function") {
    return response.status(status).json(body);
  }
  response.statusCode = status;
  return response.end(JSON.stringify(body));
}

function sendEmpty(response, status) {
  setResponseHeaders(response);
  if (typeof response.status === "function") {
    const selected = response.status(status);
    return typeof selected.end === "function"
      ? selected.end()
      : selected.json(null);
  }
  response.statusCode = status;
  return response.end();
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
      throw new Error("storage returned an unexpected object");
    }
    return true;
  } catch (error) {
    if (isNotFound(error)) return false;
    throw error;
  }
}

async function persist(storage, pathname, body) {
  if (await exists(storage, pathname)) return false;
  try {
    await storage.put(pathname, body, {
      access: "private",
      addRandomSuffix: false,
      allowOverwrite: false,
      contentType: "application/json",
    });
    return true;
  } catch (error) {
    if (await exists(storage, pathname)) return false;
    throw new Error("storage unavailable", { cause: error });
  }
}

async function readPersisted(storage, pathname) {
  const result = await storage.get(pathname, {
    access: "private",
    useCache: false,
  });
  if (
    !result ||
    result.statusCode !== 200 ||
    !result.stream ||
    result.blob?.pathname !== pathname ||
    !Number.isInteger(result.blob?.size) ||
    result.blob.size < 2 ||
    result.blob.size > MAX_FEEDBACK_STORED_BYTES ||
    result.blob?.contentType !== "application/json"
  ) {
    throw new Error("A private feedback object could not be downloaded");
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of result.stream) {
    const bytes = Buffer.from(chunk);
    size += bytes.length;
    if (size > MAX_FEEDBACK_STORED_BYTES) {
      throw new Error("A private feedback object exceeds the expected size");
    }
    chunks.push(bytes);
  }
  const body = Buffer.concat(chunks);
  if (body.length !== result.blob.size) {
    throw new Error("A private feedback object changed during download");
  }
  return parseFeedbackStoredRecordBytes(body, pathname);
}

async function persistDigestInput(storage, stored, pathname, receiptHash) {
  const date = stored.received_at.slice(0, 10);
  const indexPath =
    `${FEEDBACK_DIGEST_INPUT_PREFIX}${date}/${receiptHash}.json`;
  const index = {
    date,
    record_path: pathname,
    schema_version: 1,
  };
  return persist(storage, indexPath, `${canonical(index)}\n`);
}

export function createFeedbackHandler(
  storage = { get: blobGet, head: blobHead, put: blobPut },
  options = {},
) {
  const sendDiscord = options.sendDiscord ?? createDiscordWebhookSender();
  const logger = options.logger ?? console;
  const now = options.now ?? (() => Date.now());
  const waitUntil = options.waitUntil ?? vercelWaitUntil;
  return async function spicetifyFeedbackHandler(request, response) {
    if (request.method === "OPTIONS") return sendEmpty(response, 204);
    if (request.method !== "POST") {
      response.setHeader("Allow", "POST, OPTIONS");
      return send(response, 405, { error: "method not allowed" });
    }
    const contentType = header(request, "content-type");
    if (
      typeof contentType !== "string" ||
      !/^application\/json(?:\s*;\s*charset=utf-8)?$/iu.test(contentType) ||
      header(request, "content-encoding") !== undefined
    ) {
      return send(response, 415, { error: "invalid request" });
    }
    let body;
    try {
      body = await readBody(request);
    } catch (error) {
      const status = error?.statusCode === 413 ? 413 : 400;
      return send(response, status, {
        error: status === 413 ? "payload too large" : "invalid request",
      });
    }
    const accepted = validateFeedbackPayload(body);
    if (!accepted) {
      return send(response, 400, {
        error: "invalid request",
        code: "validation_failed",
      });
    }
    const receiptHash = sha256(canonical(accepted));
    const pathname = `${FEEDBACK_BLOB_PREFIX}${receiptHash}.json`;
    const stored = {
      ...accepted,
      received_at: new Date(now()).toISOString(),
      canonical_payload_sha256: receiptHash,
    };
    if (!validateFeedbackStoredRecord(stored, pathname)) {
      return send(response, 500, { error: "internal validation failed" });
    }
    let created;
    let indexed;
    try {
      created = await persist(storage, pathname, `${canonical(stored)}\n`);
      const persisted = created
        ? { document: stored }
        : await readPersisted(storage, pathname);
      indexed = await persistDigestInput(
        storage,
        persisted.document,
        pathname,
        receiptHash,
      );
    } catch {
      return send(response, 503, { error: "storage unavailable" });
    }
    if (created || indexed) {
      const notification = (async () => {
        try {
          await sendDiscord(
            formatFeedbackNotification({
              selection: accepted.selection,
              reasons: [...accepted.reasons],
              seed: { ...accepted.seed },
              result_count: accepted.displayed_results.length,
              receipt: receiptHash.slice(0, 12),
            }),
          );
        } catch {
          logger.error("Soundalike feedback Discord notification failed.");
        }
      })();
      try {
        waitUntil(notification);
      } catch {
        logger.error("Soundalike feedback notification scheduling failed.");
      }
    }
    return send(response, 200, { receipt_sha256: receiptHash });
  };
}

export default createFeedbackHandler();
