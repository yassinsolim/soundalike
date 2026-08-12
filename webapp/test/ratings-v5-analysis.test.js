import assert from "node:assert/strict";
import test from "node:test";

import { analyzeV5Snapshots } from "../tools/ratings-v5-analysis.js";

const candidates = [1, 2, 3, 4].map((trackId) => ({
  choice_id: `choice-${trackId}`,
  track_id: trackId,
}));
const anchorCandidates = candidates
  .slice()
  .reverse()
  .map((row) => ({ ...row, choice_id: `anchor-${row.track_id}` }));
const pack = {
  content_sha256: "pack",
  tasks: [
    {
      task_id: "task",
      seed_track_id: 10,
      candidates,
    },
    {
      task_id: "anchor",
      seed_track_id: 10,
      candidates: anchorCandidates,
    },
  ],
};
const privateMap = {
  content_sha256: "private",
  method_bindings: {
    acoustic_control: {},
    fixed_v4: {},
    frozen_preference_v1: {},
  },
  tasks: [
    {
      task_id: "task",
      seed_track_id: 10,
      method_orders: {
        acoustic_control: [1, 2, 3, 4],
        fixed_v4: [4, 3, 2, 1],
        frozen_preference_v1: [1, 3, 2, 4],
      },
    },
    {
      task_id: "anchor",
      anchor_of: "task",
      seed_track_id: 10,
    },
  ],
};

function rating(choiceIds) {
  return {
    outcome: "rated",
    ranked_choice_ids: choiceIds,
    worst_primary_reason: "genre",
    skip_reason: null,
  };
}

function skipped(reason = "not_enough_information") {
  return {
    outcome: "skipped",
    ranked_choice_ids: [],
    worst_primary_reason: null,
    skip_reason: reason,
  };
}

test("scores all six method predictions and repeated-anchor consistency", () => {
  const older = {
    session_id: "v5-session-111111111111111111111111",
    exported_at: "2026-08-13T00:00:00.000Z",
    received_at: "2026-08-13T00:00:01.000Z",
    canonical_payload_sha256: "a".repeat(64),
    task_ratings: {
      task: rating(["choice-4", "choice-3", "choice-2", "choice-1"]),
    },
  };
  const latest = {
    ...older,
    exported_at: "2026-08-13T00:01:00.000Z",
    received_at: "2026-08-13T00:01:01.000Z",
    canonical_payload_sha256: "b".repeat(64),
    task_ratings: {
      task: rating(["choice-1", "choice-2", "choice-3", "choice-4"]),
      anchor: rating(["anchor-1", "anchor-2", "anchor-3", "anchor-4"]),
    },
  };
  const report = analyzeV5Snapshots([older, latest], pack, privateMap);
  assert.deepEqual(report.record_selection, {
    valid_snapshot_files: 2,
    unique_sessions: 1,
    superseded_snapshots_ignored: 1,
    policy: "latest exported valid snapshot per session",
  });
  assert.equal(
    report.method_pairwise_prediction_evidence.acoustic_control.correct_pairs,
    6,
  );
  assert.equal(
    report.method_pairwise_prediction_evidence.fixed_v4.correct_pairs,
    0,
  );
  assert.equal(
    report.method_pairwise_prediction_evidence.frozen_preference_v1.correct_pairs,
    5,
  );
  assert.equal(
    report.paired_method_inference[
      "acoustic_control__vs__fixed_v4"
    ].observed_left_minus_right_correct_pairs,
    6,
  );
  assert.deepEqual(report.repeated_anchor_consistency, {
    completed_pairs: 1,
    rated_pairs: 1,
    exact_matches: 1,
    full_ranking_matches: 1,
    most_similar_matches: 1,
    least_similar_matches: 1,
    mismatch_reason_matches: 1,
  });
  assert.deepEqual(report.worst_primary_mismatch_reasons, { genre: 1 });
});

test("counts skips without treating repeated anchors as independent evidence", () => {
  const record = {
    session_id: "v5-session-222222222222222222222222",
    exported_at: "2026-08-13T00:00:00.000Z",
    received_at: "2026-08-13T00:00:01.000Z",
    canonical_payload_sha256: "c".repeat(64),
    task_ratings: {
      task: skipped(),
      anchor: skipped(),
    },
  };
  const report = analyzeV5Snapshots([record], pack, privateMap);
  assert.deepEqual(report.unique_comparisons, {
    completed: 1,
    rated: 0,
    skipped: 1,
    skip_rate: 1,
  });
  assert.deepEqual(report.skip_reasons, { not_enough_information: 1 });
  assert.equal(
    report.method_pairwise_prediction_evidence.acoustic_control.compared_pairs,
    0,
  );
  assert.equal(report.repeated_anchor_consistency.completed_pairs, 1);
  assert.equal(report.repeated_anchor_consistency.rated_pairs, 0);
});

test("rejects tampered private method orders before scoring", () => {
  const tampered = structuredClone(privateMap);
  tampered.tasks[0].method_orders.fixed_v4 = [4, 3, 2, 99];
  assert.throws(
    () => analyzeV5Snapshots([], pack, tampered),
    /method orders are invalid/,
  );
});

test("uses whole tasks as inference clusters", () => {
  const clusteredPack = { content_sha256: "cluster-pack", tasks: [] };
  const clusteredPrivate = {
    content_sha256: "cluster-private",
    method_bindings: privateMap.method_bindings,
    tasks: [],
  };
  const taskRatings = {};
  for (let taskIndex = 0; taskIndex < 3; taskIndex += 1) {
    const offset = taskIndex * 10;
    const rows = [1, 2, 3, 4].map((value) => ({
      choice_id: `task-${taskIndex}-choice-${value}`,
      track_id: offset + value,
    }));
    const taskId = `task-${taskIndex}`;
    clusteredPack.tasks.push({
      task_id: taskId,
      seed_track_id: 100 + taskIndex,
      candidates: rows,
    });
    clusteredPrivate.tasks.push({
      task_id: taskId,
      seed_track_id: 100 + taskIndex,
      method_orders: {
        acoustic_control: rows.map((row) => row.track_id),
        fixed_v4: rows.map((row) => row.track_id).reverse(),
        frozen_preference_v1: [
          offset + 1,
          offset + 3,
          offset + 2,
          offset + 4,
        ],
      },
    });
    taskRatings[taskId] = rating(rows.map((row) => row.choice_id));
  }
  const report = analyzeV5Snapshots(
    [
      {
        session_id: "v5-session-333333333333333333333333",
        exported_at: "2026-08-13T00:00:00.000Z",
        received_at: "2026-08-13T00:00:01.000Z",
        canonical_payload_sha256: "d".repeat(64),
        task_ratings: taskRatings,
      },
    ],
    clusteredPack,
    clusteredPrivate,
  );
  const comparison =
    report.paired_method_inference.acoustic_control__vs__fixed_v4;
  assert.equal(comparison.task_clusters, 3);
  assert.equal(comparison.observed_left_minus_right_correct_pairs, 18);
  assert.equal(comparison.exact_task_sign_flip_two_sided_p, 0.25);
  assert.equal(
    report.method_pairwise_prediction_evidence.acoustic_control
      .exact_task_sign_flip_over_chance_one_sided_p,
    0.125,
  );
});

test("does not multiply task clusters when listeners rate the same task", () => {
  const records = ["4", "5"].map((digit, index) => ({
    session_id: `v5-session-${digit.repeat(24)}`,
    exported_at: `2026-08-13T00:0${index}:00.000Z`,
    received_at: `2026-08-13T00:0${index}:01.000Z`,
    canonical_payload_sha256: digit.repeat(64),
    task_ratings: {
      task: rating(["choice-1", "choice-2", "choice-3", "choice-4"]),
    },
  }));
  const report = analyzeV5Snapshots(records, pack, privateMap);
  const acoustic =
    report.method_pairwise_prediction_evidence.acoustic_control;
  const comparison =
    report.paired_method_inference.acoustic_control__vs__fixed_v4;

  assert.equal(acoustic.rated_task_observations, 2);
  assert.equal(acoustic.unique_task_clusters, 1);
  assert.equal(acoustic.listener_clusters, 2);
  assert.equal(acoustic.exact_task_sign_flip_over_chance_one_sided_p, 0.5);
  assert.equal(acoustic.exact_listener_sign_flip_over_chance_one_sided_p, 0.25);
  assert.equal(comparison.rated_task_observations, 2);
  assert.equal(comparison.task_clusters, 1);
  assert.equal(comparison.listener_clusters, 2);
  assert.equal(comparison.exact_task_sign_flip_two_sided_p, 1);
  assert.equal(comparison.exact_listener_sign_flip_two_sided_p, 0.5);
});
