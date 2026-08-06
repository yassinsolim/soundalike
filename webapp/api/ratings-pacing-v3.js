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

import {
  allowedRequestOrigin,
  canonical,
  strictJsonParse,
} from "./ratings.js";

export const MAX_PACING_BODY_BYTES = 256 * 1024;
export const MAX_PACING_STORED_BYTES = 300 * 1024;
export const PACING_PROTOCOL_SHA256 =
  "69aaba1238ec7fdc567f384001812cdbedc3423710ae13ca8139c2cac6d7d387";
export const PACING_PACK_SHA256 =
  "6d6dd1c03412b057e14d52d29ee775e5a4c62eea63c76f7d19c46f60f1942a5c";
export const PACING_BLOB_PREFIX = "human-ratings/pacing-v3/";

const SUBMISSION_SCHEMA = "pacing_v3_listener_submission_v1";
const PROVIDER = "hosted_private_pacing_v3_evaluator";
const INTEGRITY_NOTICE =
  "Local-key HMAC provides integrity, not identity or authenticity; the key is included in this export.";
const MAX_DURATION_MS = 366 * 24 * 60 * 60 * 1000;
const HEX_64 = /^[a-f0-9]{64}$/;
const RATER_ID = /^anon-pacing-[a-f0-9]{24}$/;
const SESSION_ID = /^pacing-session-[a-f0-9]{24}$/;
const LIST_ID = /^pacing-list-[a-f0-9]{24}$/;
const RESULT_ID = /^pacing-result-[a-f0-9]{24}$/;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const MISMATCH_REASONS = new Set([
  "tempo_pacing",
  "tone_timbre",
  "instrumentation",
  "vocals",
  "mood_feeling",
  "genre",
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
  "pilot_pack_sha256",
  "protocol_sha256",
  "provider",
  "result_ratings",
  "schema_version",
  "session_id",
  "source_kind",
  "started_at",
  "submission_schema",
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
const RATING_KEYS = [
  "interaction_ms",
  "rated_at",
  "score_0_10",
].sort();
const RESULT_RATING_KEYS = [
  "interaction_ms",
  "mismatch_reasons",
  "rated_at",
  "score_0_10",
].sort();
const COUNT_KEYS = ["complete_list_ratings", "complete_result_ratings"];

const protocol = strictJsonParse(
  readFileSync(
    new URL("../evaluate/protocol-pacing-v3.json", import.meta.url),
    "utf8",
  ),
);
const pilotPack = strictJsonParse(
  readFileSync(new URL("../evaluate/pacing-pack.json", import.meta.url), "utf8"),
);

function sha256(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function documentHash(document) {
  const payload = { ...document };
  delete payload.content_sha256;
  return sha256(canonical(payload));
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

function buildCommittedIds() {
  if (
    protocol.schema_version !== 3 ||
    protocol.protocol_kind !==
      "repeated_excerpt_blind_listener_v3_private_submission" ||
    protocol.submission_schema !== SUBMISSION_SCHEMA ||
    protocol.content_sha256 !== PACING_PROTOCOL_SHA256 ||
    protocol.pilot_pack_sha256 !== PACING_PACK_SHA256 ||
    protocol.submission_endpoint !== "/api/ratings-pacing-v3" ||
    protocol.private_blob_prefix !== PACING_BLOB_PREFIX ||
    protocol.explicit_consent_required !== true ||
    protocol.automatic_submission !== false ||
    protocol.partial_submission_allowed !== true ||
    protocol.research_only !== true ||
    protocol.promotion_allowed !== false ||
    protocol.production_recommendation_changed !== false ||
    protocol.language_evaluated !== false ||
    documentHash(protocol) !== PACING_PROTOCOL_SHA256 ||
    pilotPack.schema_version !== 3 ||
    pilotPack.pack_kind !== "blinded_repeated_excerpt_comparison_v3" ||
    pilotPack.pack_id !== "pacing-v3-blind-20" ||
    pilotPack.rankings_state !== "LOCKED_BEFORE_RATINGS" ||
    pilotPack.ratings_count_at_freeze !== 0 ||
    pilotPack.seed_count !== 20 ||
    pilotPack.method_count !== 2 ||
    pilotPack.results_per_method !== 5 ||
    pilotPack.source_semantic_v2_pack_sha256 !==
      "939b639abb6d6c6b2c7ba20ae570ff7ae9d06ee67254c219d6e5f61975403347" ||
    pilotPack.source_fulltrack_v2_pack_sha256 !==
      "1980da60810959e7cdd24f39bd7142c8e34c76dab633c705976b85e49b297023" ||
    pilotPack.language_policy?.evaluated_here !== false ||
    pilotPack.matched_design?.candidate_pool !== 200 ||
    pilotPack.matched_design?.one_result_per_artist !== true ||
    pilotPack.playback_policy?.kind !==
      "strongest_nonlocal_recurrence_excerpt" ||
    pilotPack.playback_policy?.excerpt_seconds !== 20 ||
    pilotPack.playback_policy?.verified_chorus_labels !== false ||
    pilotPack.playback_policy?.full_track_seeking_allowed !== false ||
    pilotPack.seed_order_policy?.randomized !== false ||
    pilotPack.seed_order_policy?.ratings_used !== false ||
    pilotPack.research_only !== true ||
    pilotPack.promotion_allowed !== false ||
    pilotPack.production_recommendation_changed !== false ||
    pilotPack.provenance?.ratings_used !== false ||
    pilotPack.content_sha256 !== PACING_PACK_SHA256 ||
    documentHash(pilotPack) !== PACING_PACK_SHA256 ||
    !Array.isArray(pilotPack.seeds) ||
    pilotPack.seeds.length !== 20
  ) {
    throw new Error("Committed pacing ratings protocol is inconsistent");
  }
  const listIds = new Set();
  const resultIds = new Set();
  for (const seed of pilotPack.seeds) {
    if (!Array.isArray(seed.lists) || seed.lists.length !== 2) {
      throw new Error("Committed pacing list cardinality is inconsistent");
    }
    for (const list of seed.lists) {
      if (
        !LIST_ID.test(list.list_id) ||
        listIds.has(list.list_id) ||
        !Array.isArray(list.ranking) ||
        list.ranking.length !== 5
      ) {
        throw new Error("Committed pacing list identity is inconsistent");
      }
      list.ranking.forEach((row, index) => {
        if (
          row.position !== index + 1 ||
          !RESULT_ID.test(row.result_id)
        ) {
          throw new Error("Committed pacing list ranking is inconsistent");
        }
        resultIds.add(row.result_id);
      });
      listIds.add(list.list_id);
    }
  }
  if (listIds.size !== 40) {
    throw new Error("Committed pacing list count is inconsistent");
  }
  const forbidden = [
    "fulltrack_audio_study_v2",
    "pacing_tone_study_v3",
    "method_bindings",
    "blinding_key_hex",
  ];
  if (forbidden.some((marker) => canonical(pilotPack).includes(marker))) {
    throw new Error("Committed pacing study is not blinded");
  }
  return { listIds, resultIds };
}

let committedIds;
function listIds() {
  if (!committedIds) committedIds = buildCommittedIds();
  return committedIds.listIds;
}
function resultIds() {
  if (!committedIds) committedIds = buildCommittedIds();
  return committedIds.resultIds;
}

function parseTimestamp(value) {
  if (typeof value !== "string" || !ISO_TIMESTAMP.test(value)) return null;
  const milliseconds = Date.parse(value);
  return Number.isFinite(milliseconds) &&
    new Date(milliseconds).toISOString() === value
    ? milliseconds
    : null;
}

function validListRating(value, startedAt, exportedAt, duration) {
  if (!hasExactKeys(value, RATING_KEYS)) return false;
  const ratedAt = parseTimestamp(value.rated_at);
  return (
    Number.isInteger(value.score_0_10) &&
    value.score_0_10 >= 0 &&
    value.score_0_10 <= 10 &&
    Number.isInteger(value.interaction_ms) &&
    value.interaction_ms >= 1 &&
    value.interaction_ms <= duration &&
    ratedAt !== null &&
    ratedAt >= startedAt &&
    ratedAt <= exportedAt
  );
}

function validResultRating(value, startedAt, exportedAt, duration) {
  if (!hasExactKeys(value, RESULT_RATING_KEYS)) return false;
  const ratedAt = parseTimestamp(value.rated_at);
  return (
    Number.isInteger(value.score_0_10) &&
    value.score_0_10 >= 0 &&
    value.score_0_10 <= 10 &&
    Array.isArray(value.mismatch_reasons) &&
    value.mismatch_reasons.length <= MISMATCH_REASONS.size &&
    value.mismatch_reasons.every(
      (reason, index, array) =>
        MISMATCH_REASONS.has(reason) &&
        (index === 0 || array[index - 1] < reason),
    ) &&
    Number.isInteger(value.interaction_ms) &&
    value.interaction_ms >= 1 &&
    value.interaction_ms <= duration &&
    ratedAt !== null &&
    ratedAt >= startedAt &&
    ratedAt <= exportedAt
  );
}

function sanitizedEvidence(ratings) {
  return Object.fromEntries(SANITIZED_KEYS.map((key) => [key, ratings[key]]));
}

function validateEvidence(ratings, requireRating = true) {
  const startedAt = parseTimestamp(ratings.started_at);
  const lastActivityAt = parseTimestamp(ratings.last_activity_at);
  const exportedAt = parseTimestamp(ratings.exported_at);
  if (
    ratings.schema_version !== 3 ||
    ratings.submission_schema !== SUBMISSION_SCHEMA ||
    ratings.source_kind !== "human_listener" ||
    ratings.provider !== PROVIDER ||
    !RATER_ID.test(ratings.anonymous_rater_id) ||
    !SESSION_ID.test(ratings.session_id) ||
    ratings.protocol_sha256 !== PACING_PROTOCOL_SHA256 ||
    ratings.pilot_pack_sha256 !== PACING_PACK_SHA256 ||
    startedAt === null ||
    lastActivityAt === null ||
    exportedAt === null ||
    startedAt > lastActivityAt ||
    lastActivityAt > exportedAt ||
    !Number.isInteger(ratings.duration_ms) ||
    ratings.duration_ms < 1 ||
    ratings.duration_ms > MAX_DURATION_MS ||
    Math.abs(exportedAt - startedAt - ratings.duration_ms) > 1000 ||
    !isRecord(ratings.list_ratings) ||
    Object.keys(ratings.list_ratings).length > 40 ||
    !isRecord(ratings.result_ratings) ||
    Object.keys(ratings.result_ratings).length > 200
  ) {
    return null;
  }
  let listCount = 0;
  for (const [id, rating] of Object.entries(ratings.list_ratings)) {
    if (
      !LIST_ID.test(id) ||
      !listIds().has(id) ||
      !validListRating(rating, startedAt, exportedAt, ratings.duration_ms)
    ) {
      return null;
    }
    listCount += 1;
  }
  let resultCount = 0;
  for (const [id, rating] of Object.entries(ratings.result_ratings)) {
    if (
      !RESULT_ID.test(id) ||
      !resultIds().has(id) ||
      !validResultRating(rating, startedAt, exportedAt, ratings.duration_ms)
    ) {
      return null;
    }
    resultCount += 1;
  }
  if (requireRating && listCount + resultCount === 0) return null;
  return {
    complete_list_ratings: listCount,
    complete_result_ratings: resultCount,
  };
}

export function validatePacingExport(ratings) {
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
  return { counts, ratings };
}

export function validatePacingStoredRecord(document, pathname) {
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
  const expectedPath =
    `${PACING_BLOB_PREFIX}${ratings.session_id}/${digest}.json`;
  if (
    document.canonical_payload_sha256 !== digest ||
    (pathname !== undefined && pathname !== expectedPath)
  ) {
    return null;
  }
  return { counts, digest, pathname: expectedPath, ratings };
}

export function parsePacingStoredRecordBytes(value, pathname) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  if (bytes.length < 2 || bytes.length > MAX_PACING_STORED_BYTES) {
    throw new Error("Invalid private pacing ratings record size");
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  const document = strictJsonParse(text);
  const validated = validatePacingStoredRecord(document, pathname);
  if (!validated || text !== `${canonical(document)}\n`) {
    throw new Error("Invalid private pacing ratings record");
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
      Number(length) > MAX_PACING_BODY_BYTES)
  ) {
    const error = new Error("payload");
    error.statusCode = 413;
    throw error;
  }
  let raw;
  if (request.body !== undefined) {
    raw = Buffer.isBuffer(request.body)
      ? request.body
      : Buffer.from(
          typeof request.body === "string"
            ? request.body
            : JSON.stringify(request.body),
          "utf8",
        );
  } else {
    const chunks = [];
    let size = 0;
    for await (const chunk of request) {
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      size += bytes.length;
      if (size > MAX_PACING_BODY_BYTES) {
        const error = new Error("payload");
        error.statusCode = 413;
        throw error;
      }
      chunks.push(bytes);
    }
    raw = Buffer.concat(chunks);
  }
  if (raw.length > MAX_PACING_BODY_BYTES) {
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
    if (await exists(storage, pathname)) return true;
    throw new Error("storage unavailable");
  }
}

export function createPacingHandler(
  storage = { head: blobHead, put: blobPut },
  deploymentHost = process.env.VERCEL_URL,
) {
  return async function ratingsPacingHandler(request, response) {
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
      !hasExactKeys(wrapper, ["consent", "ratings", "study"]) ||
      wrapper.consent !== true ||
      wrapper.study !== "pacing-v3-blind"
    ) {
      return send(response, 400, { error: "invalid request" });
    }
    let accepted;
    try {
      accepted = validatePacingExport(wrapper.ratings);
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
      `${PACING_BLOB_PREFIX}${sanitized.session_id}/${receiptHash}.json`;
    if (!validatePacingStoredRecord(stored, pathname)) {
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

export default createPacingHandler();
