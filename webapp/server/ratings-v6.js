import {
  BlobNotFoundError,
  head as blobHead,
  put as blobPut,
} from "@vercel/blob";
import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import { readFileSync } from "node:fs";
import { TextDecoder } from "node:util";

import {
  allowedRequestOrigin,
  canonical,
  strictJsonParse,
} from "../api/ratings.js";

export const MAX_V6_BODY_BYTES = 128 * 1024;
export const MAX_V6_STORED_BYTES = 160 * 1024;
export const V6_PROTOCOL_SHA256 =
  "084c25271bd8630949dacf50bfa8670328afcbab197303a7c79af7f95801d0f1";
export const V6_PACK_SHA256 =
  "38cf7b0a4c035b27237288c9e4022a2b44d73ad82a0f3bd9085a2f862bea9637";
export const V6_BLOB_PREFIX = "human-ratings/development-v6-ranking-v1/";

const SUBMISSION_SCHEMA = "v6_development_listener_submission_v1";
const PROVIDER = "hosted_private_development_v6_evaluator";
const INTEGRITY_NOTICE =
  "Local-key HMAC provides integrity, not identity or authenticity; the key is included in this export.";
const MAX_DURATION_MS = 366 * 24 * 60 * 60 * 1000;
const HEX_64 = /^[a-f0-9]{64}$/;
const RATER_ID = /^anon-v6-[a-f0-9]{24}$/;
const SESSION_ID = /^v6-session-[a-f0-9]{24}$/;
const TASK_ID = /^v6-(task|anchor)-[a-f0-9]{24}$/;
const CHOICE_ID = /^v6-choice-[a-f0-9]{24}$/;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const REASONS = new Set([
  "genre",
  "instrumentation",
  "mood_feeling",
  "none_uncertain",
  "overall_structure",
  "tempo_pacing",
  "tone_timbre",
  "vocals_language",
]);
const SKIP_REASONS = new Set([
  "audio_problem",
  "cannot_decide",
  "out_of_scope",
  "unfamiliar_style",
]);
const EXPORT_KEYS = [
  "anonymous_rater_id",
  "duration_ms",
  "exported_at",
  "integrity_hmac_sha256",
  "integrity_notice",
  "last_activity_at",
  "local_session_key",
  "pilot_pack_sha256",
  "protocol_sha256",
  "provider",
  "schema_version",
  "session_id",
  "source_kind",
  "started_at",
  "submission_schema",
  "task_ratings",
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
  "completed_at",
  "interaction_ms",
  "outcome",
  "ranked_choice_ids",
  "skip_reason",
  "worst_primary_reason",
].sort();
const COUNT_KEYS = [
  "complete_task_ratings",
  "rated_tasks",
  "skipped_tasks",
  "unique_comparisons",
].sort();

const protocol = strictJsonParse(
  readFileSync(new URL("../evaluate/protocol-v6.json", import.meta.url), "utf8"),
);
const pilotPack = strictJsonParse(
  readFileSync(new URL("../evaluate/active-pack-v6.json", import.meta.url), "utf8"),
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

function buildCommittedTasks() {
  if (
    protocol.schema_version !== 1 ||
    protocol.protocol_kind !==
      "development_complete_ranking_v6_private_submission" ||
    protocol.submission_schema !== SUBMISSION_SCHEMA ||
    protocol.content_sha256 !== V6_PROTOCOL_SHA256 ||
    protocol.pilot_pack_sha256 !== V6_PACK_SHA256 ||
    protocol.submission_endpoint !== "/api/ratings-v6" ||
    protocol.private_blob_prefix !== V6_BLOB_PREFIX ||
    protocol.explicit_consent_required !== true ||
    protocol.automatic_submission !== false ||
    protocol.partial_submission_allowed !== true ||
    protocol.skip_allowed !== true ||
    protocol.research_only !== true ||
    protocol.development_evidence !== true ||
    protocol.independent_holdout !== false ||
    protocol.evidence_role !== "development_model_improvement" ||
    protocol.promotion_allowed !== false ||
    protocol.production_recommendation_changed !== false ||
    documentHash(protocol) !== V6_PROTOCOL_SHA256 ||
    protocol.unknown_language_allowed !== false ||
    protocol.language_segments_per_track !== 3 ||
    protocol.pairwise_predictions_per_rated_task !== 6 ||
    canonical(protocol.ranking_slots) !==
      canonical([
        "most_similar",
        "next_most_similar",
        "second_least_similar",
        "least_similar",
      ]) ||
    protocol.worst_item_reason_required !== true ||
    pilotPack.schema_version !== 1 ||
    pilotPack.pack_kind !== "soundalike_v6_development_full_ranking" ||
    pilotPack.pack_id !== "v6-development-full-ranking-1" ||
    pilotPack.research_only !== true ||
    pilotPack.development_evidence !== true ||
    pilotPack.independent_holdout !== false ||
    pilotPack.evidence_role !== "development_model_improvement" ||
    pilotPack.promotion_allowed !== false ||
    pilotPack.production_recommendation_changed !== false ||
    pilotPack.task_format?.candidates !== 4 ||
    pilotPack.task_format?.adaptive_stop_after_unique_tasks !== 12 ||
    canonical(pilotPack.task_format?.ranking_slots) !==
      canonical([
        "most_similar",
        "next_most_similar",
        "second_least_similar",
        "least_similar",
      ]) ||
    pilotPack.provenance?.includes_v5_exposure !== true ||
    pilotPack.provenance?.excludes_all_prior_exposed_tracks_and_artists !==
      true ||
    pilotPack.content_sha256 !== V6_PACK_SHA256 ||
    documentHash(pilotPack) !== V6_PACK_SHA256 ||
    !Array.isArray(pilotPack.tasks) ||
    pilotPack.tasks.length !== 18
  ) {
    throw new Error("Committed V6 ratings protocol is inconsistent");
  }
  const tasks = new Map();
  const choiceIds = new Set();
  for (const [index, task] of pilotPack.tasks.entries()) {
    if (
      task.priority_rank !== index + 1 ||
      typeof task.task_id !== "string" ||
      !TASK_ID.test(task.task_id) ||
      tasks.has(task.task_id) ||
      !Array.isArray(task.candidates) ||
      task.candidates.length !== 4
    ) {
      throw new Error("Committed V6 task identity is inconsistent");
    }
    const choices = new Set();
    for (const choice of task.candidates) {
      if (
        typeof choice.choice_id !== "string" ||
        !CHOICE_ID.test(choice.choice_id) ||
        choices.has(choice.choice_id) ||
        choiceIds.has(choice.choice_id)
      ) {
        throw new Error("Committed V6 choice identity is inconsistent");
      }
      choices.add(choice.choice_id);
      choiceIds.add(choice.choice_id);
    }
    const signature =
      `${task.seed_track_id}:` +
      task.candidates
        .map((choice) => choice.track_id)
        .sort((left, right) => left - right)
        .join(",");
    tasks.set(task.task_id, { choices, signature });
  }
  return tasks;
}

let committedTasks;
function tasks() {
  if (!committedTasks) committedTasks = buildCommittedTasks();
  return committedTasks;
}

function parseTimestamp(value) {
  if (typeof value !== "string" || !ISO_TIMESTAMP.test(value)) return null;
  const milliseconds = Date.parse(value);
  return Number.isFinite(milliseconds) &&
    new Date(milliseconds).toISOString() === value
    ? milliseconds
    : null;
}

function validTaskRating(taskId, value, startedAt, exportedAt, duration) {
  const committed = tasks().get(taskId);
  const completedAt = parseTimestamp(value?.completed_at);
  if (
    !committed ||
    !hasExactKeys(value, RATING_KEYS) ||
    completedAt === null ||
    completedAt < startedAt ||
    completedAt > exportedAt ||
    !Number.isInteger(value.interaction_ms) ||
    value.interaction_ms < 1 ||
    value.interaction_ms > duration
  ) {
    return false;
  }
  if (value.outcome === "rated") {
    const ranking = value.ranked_choice_ids;
    return (
      Array.isArray(ranking) &&
      ranking.length === 4 &&
      new Set(ranking).size === 4 &&
      ranking.every((choiceId) => committed.choices.has(choiceId)) &&
      REASONS.has(value.worst_primary_reason) &&
      value.skip_reason === null
    );
  }
  return (
    value.outcome === "skipped" &&
    value.ranked_choice_ids === null &&
    value.worst_primary_reason === null &&
    SKIP_REASONS.has(value.skip_reason)
  );
}

function sanitizedEvidence(ratings) {
  return Object.fromEntries(SANITIZED_KEYS.map((key) => [key, ratings[key]]));
}

function validateEvidenceDetailed(ratings) {
  const startedAt = parseTimestamp(ratings.started_at);
  const lastActivityAt = parseTimestamp(ratings.last_activity_at);
  const exportedAt = parseTimestamp(ratings.exported_at);
  if (
    ratings.schema_version !== 1 ||
    ratings.submission_schema !== SUBMISSION_SCHEMA ||
    ratings.source_kind !== "human_listener" ||
    ratings.provider !== PROVIDER ||
    typeof ratings.anonymous_rater_id !== "string" ||
    !RATER_ID.test(ratings.anonymous_rater_id) ||
    typeof ratings.session_id !== "string" ||
    !SESSION_ID.test(ratings.session_id) ||
    ratings.protocol_sha256 !== V6_PROTOCOL_SHA256 ||
    ratings.pilot_pack_sha256 !== V6_PACK_SHA256 ||
    startedAt === null ||
    lastActivityAt === null ||
    exportedAt === null ||
    startedAt > lastActivityAt ||
    lastActivityAt > exportedAt ||
    !Number.isInteger(ratings.duration_ms) ||
    ratings.duration_ms < 1 ||
    ratings.duration_ms > MAX_DURATION_MS ||
    Math.abs(exportedAt - startedAt - ratings.duration_ms) > 1000
  ) {
    return { counts: null, error: "invalid_snapshot_metadata" };
  }
  if (!isRecord(ratings.task_ratings)) {
    return { counts: null, error: "invalid_task_collection" };
  }
  if (Object.keys(ratings.task_ratings).length < 1) {
    return { counts: null, error: "no_complete_tasks" };
  }
  if (Object.keys(ratings.task_ratings).length > 18) {
    return { counts: null, error: "too_many_tasks" };
  }
  let rated = 0;
  let skipped = 0;
  const signatures = new Set();
  for (const [taskId, rating] of Object.entries(ratings.task_ratings)) {
    if (
      !TASK_ID.test(taskId) ||
      !validTaskRating(
        taskId,
        rating,
        startedAt,
        exportedAt,
        ratings.duration_ms,
      )
    ) {
      return { counts: null, error: "invalid_task_rating" };
    }
    rated += rating.outcome === "rated" ? 1 : 0;
    skipped += rating.outcome === "skipped" ? 1 : 0;
    signatures.add(tasks().get(taskId).signature);
  }
  return {
    counts: {
      complete_task_ratings: rated + skipped,
      rated_tasks: rated,
      skipped_tasks: skipped,
      unique_comparisons: signatures.size,
    },
    error: null,
  };
}

function validateEvidence(ratings) {
  return validateEvidenceDetailed(ratings).counts;
}

export function validateV6ExportDetailed(ratings) {
  if (
    !hasExactKeys(ratings, EXPORT_KEYS) ||
    typeof ratings.local_session_key !== "string" ||
    !HEX_64.test(ratings.local_session_key) ||
    typeof ratings.integrity_hmac_sha256 !== "string" ||
    !HEX_64.test(ratings.integrity_hmac_sha256) ||
    ratings.integrity_notice !== INTEGRITY_NOTICE
  ) {
    return { accepted: null, error: "invalid_snapshot_shape" };
  }
  const validation = validateEvidenceDetailed(ratings);
  if (!validation.counts) {
    return { accepted: null, error: validation.error };
  }
  const signedPayload = { ...ratings };
  delete signedPayload.integrity_hmac_sha256;
  const expected = createHmac(
    "sha256",
    Buffer.from(ratings.local_session_key, "hex"),
  )
    .update(canonical(signedPayload), "utf8")
    .digest();
  const supplied = Buffer.from(ratings.integrity_hmac_sha256, "hex");
  if (
    supplied.length !== expected.length ||
    !timingSafeEqual(supplied, expected)
  ) {
    return { accepted: null, error: "integrity_check_failed" };
  }
  return {
    accepted: { counts: validation.counts, ratings },
    error: null,
  };
}

export function validateV6Export(ratings) {
  return validateV6ExportDetailed(ratings).accepted;
}

export function validateV6StoredRecord(document, pathname) {
  if (!hasExactKeys(document, STORED_KEYS)) return null;
  const ratings = sanitizedEvidence(document);
  const counts = validateEvidence(ratings);
  const receivedAt = parseTimestamp(document.received_at);
  if (
    !counts ||
    receivedAt === null ||
    !hasExactKeys(document.counts, COUNT_KEYS) ||
    COUNT_KEYS.some((key) => document.counts[key] !== counts[key])
  ) {
    return null;
  }
  const digest = sha256(canonical(ratings));
  const expectedPath =
    `${V6_BLOB_PREFIX}${ratings.session_id}/${digest}.json`;
  if (
    document.canonical_payload_sha256 !== digest ||
    (pathname !== undefined && pathname !== expectedPath)
  ) {
    return null;
  }
  return { counts, digest, pathname: expectedPath, ratings };
}

export function parseV6StoredRecordBytes(value, pathname) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  if (bytes.length < 2 || bytes.length > MAX_V6_STORED_BYTES) {
    throw new Error("Invalid private V6 ratings record size");
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  const document = strictJsonParse(text);
  const validated = validateV6StoredRecord(document, pathname);
  if (!validated || text !== `${canonical(document)}\n`) {
    throw new Error("Invalid private V6 ratings record");
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
    (!/^(0|[1-9]\d*)$/.test(length) || Number(length) > MAX_V6_BODY_BYTES)
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
      if (size > MAX_V6_BODY_BYTES) {
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
  if (raw.length > MAX_V6_BODY_BYTES) {
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
  response.setHeader(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=()",
  );
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

export function createV6Handler(
  storage = { head: blobHead, put: blobPut },
  deploymentHost = process.env.VERCEL_URL,
) {
  return async function ratingsV6Handler(request, response) {
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
      wrapper.study !== "development-v6-ranking"
    ) {
      return send(response, 400, {
        error: "invalid request",
        code: "invalid_wrapper",
      });
    }
    let validation;
    try {
      validation = validateV6ExportDetailed(wrapper.ratings);
    } catch {
      validation = { accepted: null, error: "validation_failed" };
    }
    const accepted = validation.accepted;
    if (!accepted) {
      return send(response, 400, {
        error: "invalid request",
        code: validation.error,
      });
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
      `${V6_BLOB_PREFIX}${sanitized.session_id}/${receiptHash}.json`;
    if (!validateV6StoredRecord(stored, pathname)) {
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

export default createV6Handler();
