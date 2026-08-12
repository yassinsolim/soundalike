import { createHash } from "node:crypto";
import { lstat, readFile, readdir, realpath, writeFile } from "node:fs/promises";
import { basename, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  V4_BLOB_PREFIX,
  V4_PACK_SHA256,
  parseV4StoredRecordBytes,
} from "../api/ratings-v4.js";
import { canonical } from "../api/ratings.js";

const SESSION_ID = /^v4-session-[a-f0-9]{24}$/;
const RECORD_FILE = /^[a-f0-9]{64}\.json$/;

function contentHash(document) {
  const payload = { ...document };
  delete payload.content_sha256;
  return createHash("sha256").update(canonical(payload), "utf8").digest("hex");
}

function taskSignature(task) {
  const candidates = task.candidates.map((row) => row.track_id).sort((a, b) => a - b);
  return `${task.seed_track_id}:${candidates.join(",")}`;
}

function count(target, key) {
  target[key] = (target[key] ?? 0) + 1;
}

function choiceTrack(task, choiceId) {
  const choice = task.candidates.find((row) => row.choice_id === choiceId);
  if (!choice) throw new Error("A V4 rating references an unknown choice.");
  return String(choice.track_id);
}

function exclusiveMethod(origins) {
  return origins.length === 1 && ["control", "challenger"].includes(origins[0])
    ? origins[0]
    : null;
}

export function validateV4AnalysisArtifacts(pack, privateMap) {
  if (
    !pack ||
    !privateMap ||
    pack.content_sha256 !== V4_PACK_SHA256 ||
    contentHash(pack) !== pack.content_sha256 ||
    contentHash(privateMap) !== privateMap.content_sha256 ||
    pack.private_unblinding_sha256 !== privateMap.content_sha256 ||
    !Array.isArray(pack.tasks) ||
    !Array.isArray(privateMap.tasks)
  ) {
    throw new Error("V4 analysis artifacts are not bound to the active study.");
  }
  const publicTasks = new Map(pack.tasks.map((task) => [task.task_id, task]));
  const privateTasks = new Map(privateMap.tasks.map((task) => [task.task_id, task]));
  if (
    publicTasks.size !== pack.tasks.length ||
    privateTasks.size !== privateMap.tasks.length ||
    publicTasks.size !== privateTasks.size ||
    [...publicTasks.keys()].some((taskId) => !privateTasks.has(taskId))
  ) {
    throw new Error("V4 analysis task identities do not match.");
  }
  for (const task of pack.tasks) {
    const mapping = privateTasks.get(task.task_id);
    if (mapping.anchor_of) {
      if (!privateTasks.has(mapping.anchor_of)) {
        throw new Error("V4 analysis anchor mapping is invalid.");
      }
      continue;
    }
    const expected = task.candidates.map((row) => String(row.track_id)).sort();
    const actual = Object.keys(mapping.candidate_origins ?? {}).sort();
    if (
      canonical(expected) !== canonical(actual) ||
      actual.some((trackId) => {
        const origins = mapping.candidate_origins[trackId];
        return (
          !Array.isArray(origins) ||
          origins.length < 1 ||
          origins.some(
            (origin) => !["control", "challenger", "fill"].includes(origin),
          )
        );
      })
    ) {
      throw new Error("V4 analysis candidate origins are invalid.");
    }
  }
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

export function analyzeV4Snapshots(records, pack, privateMap) {
  const latest = latestSnapshots(records);
  const publicTasks = new Map(pack.tasks.map((task) => [task.task_id, task]));
  const privateTasks = new Map(privateMap.tasks.map((task) => [task.task_id, task]));
  const primaryTasks = new Map();
  const mismatchReasons = {};
  const skipReasons = {};
  const mostSelections = { control: 0, challenger: 0, fill: 0 };
  const leastSelections = { control: 0, challenger: 0, fill: 0 };
  const pairwise = {
    challenger_over_control: 0,
    control_over_challenger: 0,
    ambiguous_or_within_method: 0,
  };
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
        primaryTasks.set(`${record.session_id}:${signature}`, { task, rating });
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
      const mostMatch =
        firstRanking[0] === repeatRanking[0];
      const leastMatch =
        firstRanking.at(-1) === repeatRanking.at(-1);
      const reasonMatch = first.worst_primary_reason === repeat.worst_primary_reason;
      anchors.full_ranking_matches += rankingMatch ? 1 : 0;
      anchors.most_similar_matches += mostMatch ? 1 : 0;
      anchors.least_similar_matches += leastMatch ? 1 : 0;
      anchors.mismatch_reason_matches += reasonMatch ? 1 : 0;
      anchors.exact_matches += rankingMatch && reasonMatch ? 1 : 0;
    }
  }

  for (const { task, rating } of primaryTasks.values()) {
    if (rating.outcome === "skipped") {
      skipped += 1;
      count(skipReasons, rating.skip_reason);
      continue;
    }
    rated += 1;
    count(mismatchReasons, rating.worst_primary_reason);
    const privateTask = privateTasks.get(task.task_id);
    const source = privateTask.anchor_of
      ? privateTasks.get(privateTask.anchor_of)
      : privateTask;
    const rankedTracks = rating.ranked_choice_ids.map((choiceId) =>
      choiceTrack(task, choiceId),
    );
    const rankedOrigins = rankedTracks.map(
      (trackId) => source.candidate_origins[trackId],
    );
    const mostOrigins = rankedOrigins[0];
    const leastOrigins = rankedOrigins.at(-1);
    for (const origin of mostOrigins) count(mostSelections, origin);
    for (const origin of leastOrigins) count(leastSelections, origin);
    for (let higher = 0; higher < rankedOrigins.length; higher += 1) {
      for (let lower = higher + 1; lower < rankedOrigins.length; lower += 1) {
        const higherMethod = exclusiveMethod(rankedOrigins[higher]);
        const lowerMethod = exclusiveMethod(rankedOrigins[lower]);
        if (higherMethod === "challenger" && lowerMethod === "control") {
          pairwise.challenger_over_control += 1;
        } else if (
          higherMethod === "control" &&
          lowerMethod === "challenger"
        ) {
          pairwise.control_over_challenger += 1;
        } else {
          pairwise.ambiguous_or_within_method += 1;
        }
      }
    }
  }

  const completed = rated + skipped;
  return {
    schema_version: 1,
    report_kind: "soundalike_v4_human_evidence_analysis",
    pilot_pack_sha256: pack.content_sha256,
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
    method_pairwise_evidence: pairwise,
    method_selection_counts: {
      most_similar: mostSelections,
      least_similar: leastSelections,
    },
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
    throw new Error("The V4 inbox root must be a real directory.");
  }
  const records = [];
  for (const sessionEntry of await readdir(root, { withFileTypes: true })) {
    if (
      !sessionEntry.isDirectory() ||
      sessionEntry.isSymbolicLink() ||
      !SESSION_ID.test(sessionEntry.name)
    ) {
      throw new Error("The V4 inbox contains an unexpected session entry.");
    }
    const sessionPath = resolve(root, sessionEntry.name);
    if (relative(root, sessionPath).startsWith("..")) {
      throw new Error("Unsafe V4 inbox session path.");
    }
    for (const fileEntry of await readdir(sessionPath, { withFileTypes: true })) {
      if (
        !fileEntry.isFile() ||
        fileEntry.isSymbolicLink() ||
        !RECORD_FILE.test(fileEntry.name)
      ) {
        throw new Error("The V4 inbox contains an unexpected record entry.");
      }
      const filePath = resolve(sessionPath, fileEntry.name);
      const pathname = `${V4_BLOB_PREFIX}${sessionEntry.name}/${fileEntry.name}`;
      const bytes = await readFile(filePath);
      parseV4StoredRecordBytes(bytes, pathname);
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
      "Usage: npm run ratings:v4-analysis -- <inbox-dir> <private-map.json> [output.json] --acknowledge-private-data",
    );
    process.exitCode = 2;
    return;
  }
  const pack = JSON.parse(
    await readFile(new URL("../evaluate/active-pack.json", import.meta.url), "utf8"),
  );
  const privateMap = JSON.parse(await readFile(resolve(positional[1]), "utf8"));
  validateV4AnalysisArtifacts(pack, privateMap);
  const report = analyzeV4Snapshots(await loadSnapshots(positional[0]), pack, privateMap);
  const output = `${JSON.stringify(report, null, 2)}\n`;
  if (positional[2]) {
    await writeFile(resolve(positional[2]), output, { flag: "wx", mode: 0o600 });
    console.log(`Private V4 evidence report written to ${basename(positional[2])}.`);
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
