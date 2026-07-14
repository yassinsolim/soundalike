import {
  BlobNotFoundError,
  head as blobHead,
  put as blobPut,
} from "@vercel/blob";
import {
  createHash,
  createHmac,
  timingSafeEqual,
} from "node:crypto";
import { readFileSync } from "node:fs";
import { TextDecoder } from "node:util";

export const MAX_BODY_BYTES = 512 * 1024;
export const MAX_STORED_BYTES = 600 * 1024;
export const PRODUCTION_ORIGIN = "https://soundalike.yassin.app";
export const V17_PROTOCOL_SHA256 =
  "5b20dc6a1399959b3afe246743b2c76c20cb652078c9938c80a6316377a32eb5";
export const V17_SERVED_LISTS_SHA256 =
  "2311a7f3dc3b84452060e7ba1c42ed33cd886d602caeb2511363dd8cb90e2eeb";

const RESULT_ID = /^T14-[a-f0-9]{24}$/;
const LIST_ID = /^L14-[a-f0-9]{24}$/;
const RATER_ID = /^anon-[a-f0-9]{24}$/;
const SESSION_ID = /^session-[a-f0-9]{24}$/;
const HEX_64 = /^[a-f0-9]{64}$/;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const MAX_DURATION_MS = 366 * 24 * 60 * 60 * 1000;
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "[::1]", "localhost"]);
const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);
const SIMILARITY = new Set([
  "not_similar",
  "somewhat_similar",
  "very_similar",
]);
const COHERENCE = new Set([
  "not_coherent",
  "somewhat_coherent",
  "very_coherent",
]);
const EXPORT_KEYS = [
  "anonymous_rater_id",
  "duration_ms",
  "exported_at",
  "integrity_hmac_sha256",
  "integrity_notice",
  "last_activity_at",
  "list_ratings",
  "local_session_key",
  "migration",
  "protocol_sha256",
  "provider",
  "result_ratings",
  "schema_version",
  "served_lists_sha256",
  "session_id",
  "source_kind",
  "started_at",
].sort();
const SANITIZED_KEYS = EXPORT_KEYS.filter(
  (key) =>
    !["integrity_hmac_sha256", "integrity_notice", "local_session_key"].includes(
      key,
    ),
);
const STORED_KEYS = [
  ...SANITIZED_KEYS,
  "canonical_payload_sha256",
  "counts",
  "received_at",
].sort();
const COUNT_KEYS = [
  "complete_list_ratings",
  "complete_result_ratings",
].sort();
const RESULT_KEYS = [
  "interaction_ms",
  "junk_or_version",
  "rated_at",
  "score_0_10",
  "similarity",
].sort();
const LIST_KEYS = [
  "interaction_ms",
  "rated_at",
  "unrelated_positions_1_to_3",
  "whole_list_coherence",
].sort();
const MIGRATION_KEYS = [
  "from_protocol_sha256",
  "from_provider",
  "from_schema_version",
  "from_served_lists_sha256",
  "migrated_at",
].sort();
const V16_PROTOCOL_SHA256 =
  "c94ce615c68cde595b4e48ac5010297d76bedbed52948b10d315a39286117727";
const V16_SERVED_LISTS_SHA256 =
  "809b98ae4314b396ffb33f7349fee72c94e1a80a33d84b1661ab83166a52b9e9";
const INTEGRITY_NOTICE =
  "Local-key HMAC provides integrity, not identity or authenticity; the key is included in this export.";

const servedLists = strictJsonParse(
  readFileSync(new URL("../evaluate/served-lists.json", import.meta.url), "utf8"),
);
const protocol = strictJsonParse(
  readFileSync(new URL("../evaluate/protocol.json", import.meta.url), "utf8"),
);

export function canonical(value) {
  if (Array.isArray(value)) {
    return `[${value.map(canonical).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export function strictJsonParse(text) {
  if (typeof text !== "string") throw new SyntaxError("JSON text required");
  let index = 0;

  function whitespace() {
    while (
      index < text.length &&
      (text[index] === " " ||
        text[index] === "\n" ||
        text[index] === "\r" ||
        text[index] === "\t")
    ) {
      index += 1;
    }
  }

  function string() {
    const start = index;
    index += 1;
    while (index < text.length) {
      const code = text.charCodeAt(index);
      if (code === 0x22) {
        index += 1;
        return JSON.parse(text.slice(start, index));
      }
      if (code < 0x20) throw new SyntaxError("Invalid JSON string");
      if (code === 0x5c) {
        index += 1;
        if (index >= text.length) throw new SyntaxError("Invalid JSON escape");
        const escape = text[index];
        if (escape === "u") {
          if (!/^[a-fA-F0-9]{4}$/.test(text.slice(index + 1, index + 5))) {
            throw new SyntaxError("Invalid JSON unicode escape");
          }
          index += 5;
          continue;
        }
        if (!'"\\/bfnrt'.includes(escape)) {
          throw new SyntaxError("Invalid JSON escape");
        }
      }
      index += 1;
    }
    throw new SyntaxError("Unterminated JSON string");
  }

  function value() {
    whitespace();
    if (text[index] === "{") return object();
    if (text[index] === "[") return array();
    if (text[index] === '"') return string();
    for (const [literal, parsed] of [
      ["true", true],
      ["false", false],
      ["null", null],
    ]) {
      if (text.startsWith(literal, index)) {
        index += literal.length;
        return parsed;
      }
    }
    const match = text
      .slice(index)
      .match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
    if (!match) throw new SyntaxError("Invalid JSON value");
    index += match[0].length;
    const parsed = Number(match[0]);
    if (!Number.isFinite(parsed)) throw new SyntaxError("Invalid JSON number");
    return parsed;
  }

  function object() {
    const parsed = Object.create(null);
    const keys = new Set();
    index += 1;
    whitespace();
    if (text[index] === "}") {
      index += 1;
      return parsed;
    }
    while (index < text.length) {
      if (text[index] !== '"') throw new SyntaxError("JSON object key required");
      const key = string();
      if (keys.has(key) || FORBIDDEN_KEYS.has(key)) {
        throw new SyntaxError("Unsafe or duplicate JSON key");
      }
      keys.add(key);
      whitespace();
      if (text[index] !== ":") throw new SyntaxError("JSON colon required");
      index += 1;
      parsed[key] = value();
      whitespace();
      if (text[index] === "}") {
        index += 1;
        return parsed;
      }
      if (text[index] !== ",") throw new SyntaxError("JSON comma required");
      index += 1;
      whitespace();
    }
    throw new SyntaxError("Unterminated JSON object");
  }

  function array() {
    const parsed = [];
    index += 1;
    whitespace();
    if (text[index] === "]") {
      index += 1;
      return parsed;
    }
    while (index < text.length) {
      parsed.push(value());
      whitespace();
      if (text[index] === "]") {
        index += 1;
        return parsed;
      }
      if (text[index] !== ",") throw new SyntaxError("JSON comma required");
      index += 1;
      whitespace();
    }
    throw new SyntaxError("Unterminated JSON array");
  }

  const parsed = value();
  whitespace();
  if (index !== text.length) throw new SyntaxError("Trailing JSON content");
  return parsed;
}

function documentHash(document) {
  const payload = { ...document };
  delete payload.content_sha256;
  return sha256(canonical(payload));
}

function sha256(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function buildCommittedIds() {
  if (
    servedLists.schema_version !== 17 ||
    protocol.schema_version !== 17 ||
    servedLists.rankings_state !== "RANKINGS_LOCKED" ||
    protocol.rankings_state !== "RANKINGS_LOCKED" ||
    servedLists.ratings_count_at_freeze !== 0 ||
    protocol.ratings_count_at_freeze !== 0 ||
    servedLists.seed_count !== servedLists.seeds?.length ||
    servedLists.content_sha256 !== V17_SERVED_LISTS_SHA256 ||
    protocol.content_sha256 !== V17_PROTOCOL_SHA256 ||
    protocol.served_lists_sha256 !== V17_SERVED_LISTS_SHA256 ||
    documentHash(servedLists) !== V17_SERVED_LISTS_SHA256 ||
    documentHash(protocol) !== V17_PROTOCOL_SHA256
  ) {
    throw new Error("Committed v17 ratings protocol is inconsistent");
  }
  const resultIds = new Set();
  const listIds = new Set();
  for (const seed of servedLists.seeds) {
    const seedResultIds = new Set();
    for (const result of seed.results) {
      if (
        !RESULT_ID.test(result.result_id) ||
        seedResultIds.has(result.result_id) ||
        resultIds.has(result.result_id)
      ) {
        throw new Error("Committed v17 result identity is inconsistent");
      }
      seedResultIds.add(result.result_id);
      resultIds.add(result.result_id);
    }
    for (const list of seed.lists) {
      if (
        !LIST_ID.test(list.list_id) ||
        listIds.has(list.list_id) ||
        !Array.isArray(list.ranking) ||
        list.ranking.length !== 5
      ) {
        throw new Error("Committed v17 list identity is inconsistent");
      }
      listIds.add(list.list_id);
      const rankedIds = new Set();
      list.ranking.forEach((row, index) => {
        if (
          row.position !== index + 1 ||
          !seedResultIds.has(row.result_id) ||
          rankedIds.has(row.result_id)
        ) {
          throw new Error("Committed v17 candidate membership is inconsistent");
        }
        rankedIds.add(row.result_id);
      });
    }
  }
  return { resultIds, listIds };
}

let committedIds;
function ids() {
  if (!committedIds) committedIds = buildCommittedIds();
  return committedIds;
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

function parseTimestamp(value) {
  if (typeof value !== "string" || !ISO_TIMESTAMP.test(value)) return null;
  const milliseconds = Date.parse(value);
  return Number.isFinite(milliseconds) &&
    new Date(milliseconds).toISOString() === value
    ? milliseconds
    : null;
}

function validMigration(value, startedAt, exportedAt) {
  if (value === null) return true;
  if (!hasExactKeys(value, MIGRATION_KEYS)) return false;
  const migratedAt = parseTimestamp(value.migrated_at);
  return (
    value.from_schema_version === 16 &&
    value.from_provider === "hosted_client_only_evaluator" &&
    value.from_protocol_sha256 === V16_PROTOCOL_SHA256 &&
    value.from_served_lists_sha256 === V16_SERVED_LISTS_SHA256 &&
    migratedAt !== null &&
    migratedAt >= startedAt &&
    migratedAt <= exportedAt
  );
}

function validResultRating(value, startedAt, exportedAt, duration) {
  if (!hasExactKeys(value, RESULT_KEYS)) return false;
  const ratedAt = parseTimestamp(value.rated_at);
  return (
    SIMILARITY.has(value.similarity) &&
    (value.score_0_10 === null ||
      (Number.isInteger(value.score_0_10) &&
        value.score_0_10 >= 0 &&
        value.score_0_10 <= 10)) &&
    typeof value.junk_or_version === "boolean" &&
    Number.isInteger(value.interaction_ms) &&
    value.interaction_ms >= 1 &&
    value.interaction_ms <= duration &&
    ratedAt !== null &&
    ratedAt >= startedAt &&
    ratedAt <= exportedAt
  );
}

function validListRating(value, startedAt, exportedAt, duration) {
  if (!hasExactKeys(value, LIST_KEYS)) return false;
  const ratedAt = parseTimestamp(value.rated_at);
  return (
    COHERENCE.has(value.whole_list_coherence) &&
    Number.isInteger(value.unrelated_positions_1_to_3) &&
    value.unrelated_positions_1_to_3 >= 0 &&
    value.unrelated_positions_1_to_3 <= 3 &&
    Number.isInteger(value.interaction_ms) &&
    value.interaction_ms >= 1 &&
    value.interaction_ms <= duration &&
    ratedAt !== null &&
    ratedAt >= startedAt &&
    ratedAt <= exportedAt
  );
}

function validateEvidence(ratings) {
  const startedAt = parseTimestamp(ratings.started_at);
  const lastActivityAt = parseTimestamp(ratings.last_activity_at);
  const exportedAt = parseTimestamp(ratings.exported_at);
  if (
    ratings.schema_version !== 17 ||
    ratings.source_kind !== "human_listener" ||
    ratings.provider !== "hosted_private_submission_evaluator" ||
    !RATER_ID.test(ratings.anonymous_rater_id) ||
    !SESSION_ID.test(ratings.session_id) ||
    ratings.protocol_sha256 !== V17_PROTOCOL_SHA256 ||
    ratings.served_lists_sha256 !== V17_SERVED_LISTS_SHA256 ||
    startedAt === null ||
    lastActivityAt === null ||
    exportedAt === null ||
    startedAt > lastActivityAt ||
    lastActivityAt > exportedAt ||
    !Number.isInteger(ratings.duration_ms) ||
    ratings.duration_ms < 1 ||
    ratings.duration_ms > MAX_DURATION_MS ||
    Math.abs(exportedAt - startedAt - ratings.duration_ms) > 1000 ||
    !validMigration(ratings.migration, startedAt, exportedAt) ||
    !isRecord(ratings.result_ratings) ||
    !isRecord(ratings.list_ratings)
  ) {
    return null;
  }

  const { resultIds, listIds } = ids();
  let resultCount = 0;
  let listCount = 0;
  for (const [id, rating] of Object.entries(ratings.result_ratings)) {
    if (
      !RESULT_ID.test(id) ||
      !resultIds.has(id) ||
      !validResultRating(rating, startedAt, exportedAt, ratings.duration_ms)
    ) {
      return null;
    }
    resultCount += 1;
  }
  for (const [id, rating] of Object.entries(ratings.list_ratings)) {
    if (
      !LIST_ID.test(id) ||
      !listIds.has(id) ||
      !validListRating(rating, startedAt, exportedAt, ratings.duration_ms)
    ) {
      return null;
    }
    listCount += 1;
  }
  if (resultCount + listCount < 1) return null;
  return {
    complete_list_ratings: listCount,
    complete_result_ratings: resultCount,
  };
}

export function validateExport(ratings) {
  if (
    !hasExactKeys(ratings, EXPORT_KEYS) ||
    !HEX_64.test(ratings.local_session_key) ||
    !HEX_64.test(ratings.integrity_hmac_sha256) ||
    ratings.integrity_notice !== INTEGRITY_NOTICE
  ) {
    return null;
  }
  const counts = validateEvidence(ratings);
  if (!counts) return null;

  const signedPayload = { ...ratings };
  delete signedPayload.integrity_hmac_sha256;
  const expected = createHmac("sha256", ratings.local_session_key)
    .update(canonical(signedPayload), "utf8")
    .digest();
  const supplied = Buffer.from(ratings.integrity_hmac_sha256, "hex");
  if (
    supplied.length !== expected.length ||
    !timingSafeEqual(supplied, expected)
  ) {
    return null;
  }

  return {
    counts,
    ratings,
  };
}

function sanitizedEvidence(ratings) {
  return Object.fromEntries(SANITIZED_KEYS.map((key) => [key, ratings[key]]));
}

export function validateStoredRecord(document, pathname) {
  if (!hasExactKeys(document, STORED_KEYS)) return null;
  const ratings = sanitizedEvidence(document);
  const counts = validateEvidence(ratings);
  const receivedAt = parseTimestamp(document.received_at);
  if (
    !counts ||
    receivedAt === null ||
    !hasExactKeys(document.counts, COUNT_KEYS) ||
    document.counts.complete_list_ratings !== counts.complete_list_ratings ||
    document.counts.complete_result_ratings !== counts.complete_result_ratings
  ) {
    return null;
  }
  const digest = sha256(canonical(ratings));
  if (document.canonical_payload_sha256 !== digest) return null;
  const expectedPath =
    `human-ratings/v17/${ratings.session_id}/${digest}.json`;
  if (pathname !== undefined && pathname !== expectedPath) return null;
  return { counts, digest, pathname: expectedPath, ratings };
}

export function parseStoredRecordBytes(value, pathname) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  if (bytes.length < 2 || bytes.length > MAX_STORED_BYTES) {
    throw new Error("Invalid private ratings record size");
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  const document = strictJsonParse(text);
  const validated = validateStoredRecord(document, pathname);
  if (!validated || text !== `${canonical(document)}\n`) {
    throw new Error("Invalid private ratings record");
  }
  return { document, ...validated };
}

function parsedOrigin(value) {
  if (typeof value !== "string" || value.length > 200) return false;
  if ([...value].some((character) => {
    const code = character.charCodeAt(0);
    return code < 32 || code === 127;
  })) {
    return false;
  }
  try {
    const url = new URL(value);
    if (
      (url.protocol !== "http:" && url.protocol !== "https:") ||
      url.origin !== value ||
      url.username !== "" ||
      url.password !== "" ||
      url.pathname !== "/" ||
      url.search !== "" ||
      url.hash !== ""
    ) {
      return false;
    }
    return url;
  } catch {
    return false;
  }
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

function previewDeploymentOrigin(value) {
  if (
    typeof value !== "string" ||
    value.length > 200 ||
    !/^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+vercel\.app$/.test(
      value,
    )
  ) {
    return null;
  }
  try {
    const url = new URL(`https://${value}`);
    return url.host === value && url.hostname.endsWith(".vercel.app")
      ? url.origin
      : null;
  } catch {
    return null;
  }
}

export function allowedRequestOrigin(
  request,
  deploymentHost = process.env.VERCEL_URL,
) {
  const origin = parsedOrigin(header(request, "origin"));
  const host = header(request, "host");
  const forwardedProto = header(request, "x-forwarded-proto");
  const fetchSite = header(request, "sec-fetch-site");
  if (!origin || typeof host !== "string" || host.length > 200) return false;
  if (fetchSite !== undefined && fetchSite !== "same-origin") return false;
  if (
    forwardedProto !== undefined &&
    forwardedProto !== origin.protocol.slice(0, -1)
  ) {
    return false;
  }
  let target;
  try {
    target = new URL(`${origin.protocol}//${host}`);
  } catch {
    return false;
  }
  if (
    target.host !== origin.host ||
    target.username !== "" ||
    target.password !== "" ||
    target.pathname !== "/" ||
    target.search !== "" ||
    target.hash !== ""
  ) {
    return false;
  }
  return (
    origin.origin === PRODUCTION_ORIGIN ||
    origin.origin === previewDeploymentOrigin(deploymentHost) ||
    LOOPBACK_HOSTS.has(origin.hostname)
  );
}

async function readBody(request) {
  const length = header(request, "content-length");
  if (length !== undefined) {
    if (
      typeof length !== "string" ||
      !/^(0|[1-9]\d*)$/.test(length) ||
      Number(length) > MAX_BODY_BYTES
    ) {
      const error = new Error("payload");
      error.statusCode = 413;
      throw error;
    }
  }

  let raw;
  if (request.body !== undefined) {
    if (Buffer.isBuffer(request.body)) raw = request.body;
    else if (typeof request.body === "string")
      raw = Buffer.from(request.body, "utf8");
    else raw = Buffer.from(JSON.stringify(request.body), "utf8");
  } else {
    const chunks = [];
    let size = 0;
    for await (const chunk of request) {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      size += buffer.length;
      if (size > MAX_BODY_BYTES) {
        const error = new Error("payload");
        error.statusCode = 413;
        throw error;
      }
      chunks.push(buffer);
    }
    raw = Buffer.concat(chunks);
  }
  if (raw.length > MAX_BODY_BYTES) {
    const error = new Error("payload");
    error.statusCode = 413;
    throw error;
  }
  let value;
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
    value = strictJsonParse(text);
  } catch {
    const error = new Error("json");
    error.statusCode = 400;
    throw error;
  }
  if (Buffer.byteLength(JSON.stringify(value), "utf8") > MAX_BODY_BYTES) {
    const error = new Error("payload");
    error.statusCode = 413;
    throw error;
  }
  return value;
}

function setSecurityHeaders(response) {
  response.setHeader("Cache-Control", "no-store, max-age=0");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'none'; frame-ancestors 'none'; sandbox",
  );
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  response.setHeader("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader("Vary", "Origin");
  response.setHeader("X-Frame-Options", "DENY");
  response.setHeader("X-Content-Type-Options", "nosniff");
}

function send(response, status, body) {
  setSecurityHeaders(response);
  if (typeof response.status === "function") {
    return response.status(status).json(body);
  }
  response.statusCode = status;
  return response.end(JSON.stringify(body));
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
  if (await exists(storage, pathname)) return true;
  try {
    await storage.put(pathname, body, {
      access: "private",
      addRandomSuffix: false,
      allowOverwrite: false,
      contentType: "application/json",
    });
    return false;
  } catch {
    // A concurrent immutable write of the same digest is a successful duplicate.
    if (await exists(storage, pathname)) return true;
    throw new Error("storage unavailable");
  }
}

export function createHandler(
  storage = { head: blobHead, put: blobPut },
  deploymentHost = process.env.VERCEL_URL,
) {
  return async function ratingsHandler(request, response) {
    if (request.method !== "POST") {
      response.setHeader("Allow", "POST");
      return send(response, 405, { error: "method not allowed" });
    }
    if (!allowedRequestOrigin(request, deploymentHost)) {
      return send(response, 403, { error: "forbidden" });
    }
    const contentType = header(request, "content-type");
    if (
      typeof contentType !== "string" ||
      !/^application\/json(?:\s*;\s*charset=utf-8)?$/i.test(contentType) ||
      header(request, "content-encoding") !== undefined
    ) {
      return send(response, 415, { error: "invalid request" });
    }

    let wrapper;
    try {
      wrapper = await readBody(request);
    } catch (error) {
      const status = error?.statusCode === 413 ? 413 : 400;
      return send(response, status, {
        error: status === 413 ? "payload too large" : "invalid request",
      });
    }
    if (
      !hasExactKeys(wrapper, ["consent", "ratings"]) ||
      wrapper.consent !== true
    ) {
      return send(response, 400, { error: "invalid request" });
    }

    let accepted;
    try {
      accepted = validateExport(wrapper.ratings);
    } catch {
      accepted = null;
    }
    if (!accepted) {
      return send(response, 400, { error: "invalid request" });
    }

    const sanitized = sanitizedEvidence(accepted.ratings);
    const receiptHash = sha256(canonical(sanitized));
    const stored = {
      ...sanitized,
      received_at: new Date().toISOString(),
      canonical_payload_sha256: receiptHash,
      counts: accepted.counts,
    };
    const pathname =
      `human-ratings/v17/${sanitized.session_id}/${receiptHash}.json`;
    if (!validateStoredRecord(stored, pathname)) {
      return send(response, 500, { error: "internal validation failed" });
    }

    let duplicate;
    try {
      duplicate = await persist(storage, pathname, `${canonical(stored)}\n`);
    } catch {
      return send(response, 503, { error: "storage unavailable" });
    }
    return send(response, 200, {
      receipt_sha256: receiptHash,
      counts: accepted.counts,
      duplicate,
    });
  };
}

export default createHandler();
