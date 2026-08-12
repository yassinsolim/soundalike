import { createHash } from "node:crypto";
import { lstat, readFile, readdir, realpath, writeFile } from "node:fs/promises";
import { basename, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  V5_BLOB_PREFIX,
  V5_PACK_SHA256,
  parseV5StoredRecordBytes,
} from "../api/ratings-v5.js";
import { canonical } from "../api/ratings.js";

const METHODS = ["acoustic_control", "fixed_v4", "frozen_preference_v1"];
const SESSION_ID = /^v5-session-[a-f0-9]{24}$/;
const RECORD_FILE = /^[a-f0-9]{64}\.json$/;

function contentHash(document) {
  const payload = { ...document };
  delete payload.content_sha256;
  return createHash("sha256").update(canonical(payload), "utf8").digest("hex");
}

function taskSignature(task) {
  const candidates = task.candidates
    .map((row) => row.track_id)
    .sort((left, right) => left - right);
  return `${task.seed_track_id}:${candidates.join(",")}`;
}

function choiceTrack(task, choiceId) {
  const choice = task.candidates.find((row) => row.choice_id === choiceId);
  if (!choice) throw new Error("A V5 rating references an unknown choice.");
  return choice.track_id;
}

function count(target, key) {
  target[key] = (target[key] ?? 0) + 1;
}

function exactSignFlipOneSided(deltas) {
  const observed = deltas.reduce((total, value) => total + value, 0);
  const nonzero = deltas.filter((value) => value !== 0).map(Math.abs);
  if (!nonzero.length) return 1;
  let distribution = new Map([[0, 1]]);
  for (const delta of nonzero) {
    const next = new Map();
    for (const [total, probability] of distribution) {
      next.set(total + delta, (next.get(total + delta) ?? 0) + probability / 2);
      next.set(total - delta, (next.get(total - delta) ?? 0) + probability / 2);
    }
    distribution = next;
  }
  return [...distribution.entries()]
    .filter(([total]) => total >= observed)
    .reduce((probability, [, value]) => probability + value, 0);
}

function exactSignFlipTwoSided(deltas) {
  const observed = Math.abs(deltas.reduce((total, value) => total + value, 0));
  const nonzero = deltas.filter((value) => value !== 0).map(Math.abs);
  if (!nonzero.length) return 1;
  let distribution = new Map([[0, 1]]);
  for (const delta of nonzero) {
    const next = new Map();
    for (const [total, probability] of distribution) {
      next.set(total + delta, (next.get(total + delta) ?? 0) + probability / 2);
      next.set(total - delta, (next.get(total - delta) ?? 0) + probability / 2);
    }
    distribution = next;
  }
  return Math.min(
    1,
    [...distribution.entries()]
      .filter(([total]) => Math.abs(total) >= observed)
      .reduce((probability, [, value]) => probability + value, 0),
  );
}

function validateTaskMappings(pack, privateMap) {
  if (!Array.isArray(pack.tasks) || !Array.isArray(privateMap.tasks)) {
    throw new Error("V5 analysis task mappings are invalid.");
  }
  const publicTasks = new Map(pack.tasks.map((task) => [task.task_id, task]));
  const privateTasks = new Map(
    privateMap.tasks.map((task) => [task.task_id, task]),
  );
  if (
    publicTasks.size !== pack.tasks.length ||
    privateTasks.size !== privateMap.tasks.length ||
    publicTasks.size !== privateTasks.size ||
    [...publicTasks.keys()].some((taskId) => !privateTasks.has(taskId))
  ) {
    throw new Error("V5 analysis task identities do not match.");
  }
  for (const task of pack.tasks) {
    const mapping = privateTasks.get(task.task_id);
    if (mapping.seed_track_id !== task.seed_track_id) {
      throw new Error("V5 analysis seed identities do not match.");
    }
    if (mapping.anchor_of) {
      const sourceMapping = privateTasks.get(mapping.anchor_of);
      const sourceTask = publicTasks.get(mapping.anchor_of);
      if (
        mapping.anchor_of === mapping.task_id ||
        !sourceMapping ||
        sourceMapping.anchor_of ||
        !sourceTask ||
        taskSignature(task) !== taskSignature(sourceTask)
      ) {
        throw new Error("V5 analysis anchor mapping is invalid.");
      }
      continue;
    }
    const expected = task.candidates
      .map((row) => row.track_id)
      .sort((left, right) => left - right);
    if (
      canonical(Object.keys(mapping.method_orders ?? {}).sort()) !==
        canonical(METHODS.slice().sort()) ||
      METHODS.some((method) => {
        const order = mapping.method_orders[method];
        return (
          !Array.isArray(order) ||
          order.length !== expected.length ||
          new Set(order).size !== expected.length ||
          canonical(order.slice().sort((left, right) => left - right)) !==
            canonical(expected)
        );
      })
    ) {
      throw new Error("V5 analysis method orders are invalid.");
    }
  }
}

export function validateV5AnalysisArtifacts(pack, privateMap) {
  if (
    !pack ||
    !privateMap ||
    pack.content_sha256 !== V5_PACK_SHA256 ||
    contentHash(pack) !== pack.content_sha256 ||
    contentHash(privateMap) !== privateMap.content_sha256 ||
    pack.private_unblinding_sha256 !== privateMap.content_sha256 ||
    canonical(Object.keys(privateMap.method_bindings ?? {}).sort()) !==
      canonical(METHODS.slice().sort())
  ) {
    throw new Error("V5 analysis artifacts are not bound to the active study.");
  }
  validateTaskMappings(pack, privateMap);
}

function latestSnapshots(records) {
  const latest = new Map();
  for (const record of records) {
    const current = latest.get(record.session_id);
    const identity = [
      record.exported_at,
      record.received_at,
      record.canonical_payload_sha256,
    ].join(":");
    const currentIdentity = current
      ? [
          current.exported_at,
          current.received_at,
          current.canonical_payload_sha256,
        ].join(":")
      : "";
    if (!current || identity > currentIdentity) latest.set(record.session_id, record);
  }
  return latest;
}

function correctPairs(humanOrder, methodOrder) {
  const humanPosition = new Map(
    humanOrder.map((trackId, position) => [trackId, position]),
  );
  const methodPosition = new Map(
    methodOrder.map((trackId, position) => [trackId, position]),
  );
  let correct = 0;
  for (const [left, right] of humanOrder
    .slice()
    .sort((a, b) => a - b)
    .flatMap((left, index, values) =>
      values.slice(index + 1).map((right) => [left, right]),
    )) {
    correct +=
      (humanPosition.get(left) < humanPosition.get(right)) ===
      (methodPosition.get(left) < methodPosition.get(right))
        ? 1
        : 0;
  }
  return correct;
}

function clusteredDeltas(observations, clusterKey, value) {
  const clusters = new Map();
  for (const observation of observations) {
    const key = clusterKey(observation);
    const values = clusters.get(key) ?? [];
    values.push(value(observation));
    clusters.set(key, values);
  }
  return [...clusters.values()].map(
    (values) =>
      values.reduce((total, current) => total + current, 0) / values.length,
  );
}

export function analyzeV5Snapshots(records, pack, privateMap) {
  validateTaskMappings(pack, privateMap);
  const latest = latestSnapshots(records);
  const publicTasks = new Map(pack.tasks.map((task) => [task.task_id, task]));
  const privateTasks = new Map(
    privateMap.tasks.map((task) => [task.task_id, task]),
  );
  const primaryTasks = new Map();
  const observations = [];
  const mismatchReasons = {};
  const skipReasons = {};
  const methodStats = Object.fromEntries(
    METHODS.map((method) => [
      method,
      {
        correct_pairs: 0,
        compared_pairs: 0,
        exact_full_rankings: 0,
        first_place_matches: 0,
        last_place_matches: 0,
        task_correct_counts: [],
      },
    ]),
  );
  const anchors = {
    completed_pairs: 0,
    rated_pairs: 0,
    exact_matches: 0,
    full_ranking_matches: 0,
    most_similar_matches: 0,
    least_similar_matches: 0,
    mismatch_reason_matches: 0,
  };
  let rated = 0;
  let skipped = 0;

  for (const record of latest.values()) {
    const seen = new Set();
    for (const task of pack.tasks) {
      const rating = record.task_ratings[task.task_id];
      if (!rating) continue;
      const signature = taskSignature(task);
      if (!seen.has(signature)) {
        seen.add(signature);
        primaryTasks.set(`${record.session_id}:${signature}`, {
          task,
          rating,
          session_id: record.session_id,
          signature,
        });
      }
    }
    for (const mapping of privateMap.tasks.filter((task) => task.anchor_of)) {
      const first = record.task_ratings[mapping.anchor_of];
      const repeat = record.task_ratings[mapping.task_id];
      if (!first || !repeat) continue;
      anchors.completed_pairs += 1;
      if (first.outcome !== "rated" || repeat.outcome !== "rated") continue;
      anchors.rated_pairs += 1;
      const firstTask = publicTasks.get(mapping.anchor_of);
      const repeatTask = publicTasks.get(mapping.task_id);
      const firstRanking = first.ranked_choice_ids.map((choiceId) =>
        choiceTrack(firstTask, choiceId),
      );
      const repeatRanking = repeat.ranked_choice_ids.map((choiceId) =>
        choiceTrack(repeatTask, choiceId),
      );
      const rankingMatch = canonical(firstRanking) === canonical(repeatRanking);
      const reasonMatch =
        first.worst_primary_reason === repeat.worst_primary_reason;
      anchors.full_ranking_matches += rankingMatch ? 1 : 0;
      anchors.most_similar_matches +=
        firstRanking[0] === repeatRanking[0] ? 1 : 0;
      anchors.least_similar_matches +=
        firstRanking.at(-1) === repeatRanking.at(-1) ? 1 : 0;
      anchors.mismatch_reason_matches += reasonMatch ? 1 : 0;
      anchors.exact_matches += rankingMatch && reasonMatch ? 1 : 0;
    }
  }

  for (const { task, rating, session_id, signature } of primaryTasks.values()) {
    if (rating.outcome === "skipped") {
      skipped += 1;
      count(skipReasons, rating.skip_reason);
      continue;
    }
    rated += 1;
    count(mismatchReasons, rating.worst_primary_reason);
    const mapping = privateTasks.get(task.task_id);
    const source = mapping.anchor_of
      ? privateTasks.get(mapping.anchor_of)
      : mapping;
    const humanOrder = rating.ranked_choice_ids.map((choiceId) =>
      choiceTrack(task, choiceId),
    );
    const correctByMethod = {};
    for (const method of METHODS) {
      const methodOrder = source.method_orders[method];
      const correct = correctPairs(humanOrder, methodOrder);
      const stats = methodStats[method];
      stats.correct_pairs += correct;
      stats.compared_pairs += 6;
      stats.exact_full_rankings +=
        canonical(humanOrder) === canonical(methodOrder) ? 1 : 0;
      stats.first_place_matches += humanOrder[0] === methodOrder[0] ? 1 : 0;
      stats.last_place_matches += humanOrder.at(-1) === methodOrder.at(-1) ? 1 : 0;
      stats.task_correct_counts.push(correct);
      correctByMethod[method] = correct;
    }
    observations.push({ session_id, signature, correctByMethod });
  }

  const methodEvidence = Object.fromEntries(
    METHODS.map((method) => {
      const stats = methodStats[method];
      const taskDeltas = clusteredDeltas(
        observations,
        (observation) => observation.signature,
        (observation) => observation.correctByMethod[method] - 3,
      );
      const listenerDeltas = clusteredDeltas(
        observations,
        (observation) => observation.session_id,
        (observation) => observation.correctByMethod[method] - 3,
      );
      return [
        method,
        {
          correct_pairs: stats.correct_pairs,
          compared_pairs: stats.compared_pairs,
          pair_accuracy: stats.compared_pairs
            ? stats.correct_pairs / stats.compared_pairs
            : null,
          exact_full_rankings: stats.exact_full_rankings,
          first_place_matches: stats.first_place_matches,
          last_place_matches: stats.last_place_matches,
          rated_task_observations: stats.task_correct_counts.length,
          unique_task_clusters: taskDeltas.length,
          listener_clusters: listenerDeltas.length,
          observed_correct_minus_chance:
            stats.correct_pairs - 3 * stats.task_correct_counts.length,
          exact_task_sign_flip_over_chance_one_sided_p:
            exactSignFlipOneSided(taskDeltas),
          exact_listener_sign_flip_over_chance_one_sided_p:
            exactSignFlipOneSided(listenerDeltas),
        },
      ];
    }),
  );
  const pairedMethodComparisons = {};
  for (const [left, right] of METHODS.flatMap((left, index) =>
    METHODS.slice(index + 1).map((right) => [left, right]),
  )) {
    const taskDeltas = clusteredDeltas(
      observations,
      (observation) => observation.signature,
      (observation) =>
        observation.correctByMethod[left] - observation.correctByMethod[right],
    );
    const listenerDeltas = clusteredDeltas(
      observations,
      (observation) => observation.session_id,
      (observation) =>
        observation.correctByMethod[left] - observation.correctByMethod[right],
    );
    const observed =
      methodStats[left].correct_pairs - methodStats[right].correct_pairs;
    pairedMethodComparisons[`${left}__vs__${right}`] = {
      rated_task_observations: observations.length,
      task_clusters: taskDeltas.length,
      listener_clusters: listenerDeltas.length,
      observed_left_minus_right_correct_pairs: observed,
      exact_task_sign_flip_left_over_right_one_sided_p:
        exactSignFlipOneSided(taskDeltas),
      exact_task_sign_flip_right_over_left_one_sided_p:
        exactSignFlipOneSided(taskDeltas.map((value) => -value)),
      exact_task_sign_flip_two_sided_p: exactSignFlipTwoSided(taskDeltas),
      exact_listener_sign_flip_left_over_right_one_sided_p:
        exactSignFlipOneSided(listenerDeltas),
      exact_listener_sign_flip_right_over_left_one_sided_p:
        exactSignFlipOneSided(listenerDeltas.map((value) => -value)),
      exact_listener_sign_flip_two_sided_p:
        exactSignFlipTwoSided(listenerDeltas),
    };
  }

  const completed = rated + skipped;
  return {
    schema_version: 1,
    report_kind: "soundalike_v5_human_evidence_analysis",
    pilot_pack_sha256: pack.content_sha256,
    private_unblinding_sha256: privateMap.content_sha256,
    record_selection: {
      valid_snapshot_files: records.length,
      unique_sessions: latest.size,
      superseded_snapshots_ignored: records.length - latest.size,
      policy: "latest exported valid snapshot per session",
    },
    unique_comparisons: {
      completed,
      rated,
      skipped,
      skip_rate: completed ? skipped / completed : 0,
    },
    method_pairwise_prediction_evidence: methodEvidence,
    paired_method_inference: pairedMethodComparisons,
    worst_primary_mismatch_reasons: mismatchReasons,
    skip_reasons: skipReasons,
    repeated_anchor_consistency: anchors,
    automatic_promotion_allowed: false,
    promotion_decision: "not_evaluated",
  };
}

async function loadSnapshots(rootPath) {
  const root = await realpath(rootPath);
  const rootMetadata = await lstat(root);
  if (!rootMetadata.isDirectory() || rootMetadata.isSymbolicLink()) {
    throw new Error("The V5 inbox root must be a real directory.");
  }
  const records = [];
  for (const sessionEntry of await readdir(root, { withFileTypes: true })) {
    if (
      !sessionEntry.isDirectory() ||
      sessionEntry.isSymbolicLink() ||
      !SESSION_ID.test(sessionEntry.name)
    ) {
      throw new Error("The V5 inbox contains an unexpected session entry.");
    }
    const sessionPath = resolve(root, sessionEntry.name);
    if (relative(root, sessionPath).startsWith("..")) {
      throw new Error("Unsafe V5 inbox session path.");
    }
    for (const fileEntry of await readdir(sessionPath, { withFileTypes: true })) {
      if (
        !fileEntry.isFile() ||
        fileEntry.isSymbolicLink() ||
        !RECORD_FILE.test(fileEntry.name)
      ) {
        throw new Error("The V5 inbox contains an unexpected record entry.");
      }
      const filePath = resolve(sessionPath, fileEntry.name);
      const pathname = `${V5_BLOB_PREFIX}${sessionEntry.name}/${fileEntry.name}`;
      const bytes = await readFile(filePath);
      parseV5StoredRecordBytes(bytes, pathname);
      records.push(JSON.parse(bytes.toString("utf8")));
    }
  }
  return records;
}

async function main() {
  const args = process.argv.slice(2);
  const acknowledged = args.includes("--acknowledge-private-data");
  const positional = args.filter((arg) => arg !== "--acknowledge-private-data");
  if (!acknowledged || positional.length < 2 || positional.length > 3) {
    console.error(
      "Usage: npm run ratings:v5-analysis -- <inbox-dir> <private-map.json> [output.json] --acknowledge-private-data",
    );
    process.exitCode = 2;
    return;
  }
  const pack = JSON.parse(
    await readFile(new URL("../evaluate/active-pack.json", import.meta.url), "utf8"),
  );
  const privateMap = JSON.parse(await readFile(resolve(positional[1]), "utf8"));
  validateV5AnalysisArtifacts(pack, privateMap);
  const report = analyzeV5Snapshots(
    await loadSnapshots(positional[0]),
    pack,
    privateMap,
  );
  const output = `${JSON.stringify(report, null, 2)}\n`;
  if (positional[2]) {
    await writeFile(resolve(positional[2]), output, { flag: "wx", mode: 0o600 });
    console.log(`Private V5 evidence report written to ${basename(positional[2])}.`);
  } else {
    process.stdout.write(output);
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  await main();
}
