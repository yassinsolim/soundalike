import assert from "node:assert/strict";
import test from "node:test";

import { analyzeV4Snapshots } from "../tools/ratings-v4-analysis.js";

const pack = {
  content_sha256: "pack",
  tasks: [
    {
      task_id: "v4-task-a",
      seed_track_id: 1,
      candidates: [
        { choice_id: "a-control", track_id: 10 },
        { choice_id: "a-challenger", track_id: 11 },
        { choice_id: "a-fill", track_id: 12 },
        { choice_id: "a-shared", track_id: 13 },
      ],
    },
    {
      task_id: "v4-anchor-a",
      seed_track_id: 1,
      candidates: [
        { choice_id: "r-control", track_id: 10 },
        { choice_id: "r-challenger", track_id: 11 },
        { choice_id: "r-fill", track_id: 12 },
        { choice_id: "r-shared", track_id: 13 },
      ],
    },
    {
      task_id: "v4-task-b",
      seed_track_id: 2,
      candidates: [
        { choice_id: "b-control", track_id: 20 },
        { choice_id: "b-challenger", track_id: 21 },
        { choice_id: "b-fill", track_id: 22 },
        { choice_id: "b-shared", track_id: 23 },
      ],
    },
  ],
};

const privateMap = {
  tasks: [
    {
      task_id: "v4-task-a",
      candidate_origins: {
        10: ["control"],
        11: ["challenger"],
        12: ["fill"],
        13: ["control", "challenger"],
      },
    },
    { task_id: "v4-anchor-a", anchor_of: "v4-task-a" },
    {
      task_id: "v4-task-b",
      candidate_origins: {
        20: ["control"],
        21: ["challenger"],
        22: ["fill"],
        23: ["control", "challenger"],
      },
    },
  ],
};

function rated(ranking, reason) {
  return {
    outcome: "rated",
    ranked_choice_ids: ranking,
    worst_primary_reason: reason,
    skip_reason: null,
  };
}

test("uses the latest snapshot, avoids anchor double counting, and reports consistency", () => {
  const records = [
    {
      session_id: "session-1",
      exported_at: "2026-01-01T00:00:01.000Z",
      received_at: "2026-01-01T00:00:02.000Z",
      canonical_payload_sha256: "a",
      task_ratings: {
        "v4-task-a": rated(
          ["a-control", "a-fill", "a-shared", "a-challenger"],
          "tempo_pacing",
        ),
      },
    },
    {
      session_id: "session-1",
      exported_at: "2026-01-01T00:00:03.000Z",
      received_at: "2026-01-01T00:00:04.000Z",
      canonical_payload_sha256: "b",
      task_ratings: {
        "v4-task-a": rated(
          ["a-challenger", "a-fill", "a-shared", "a-control"],
          "tone_timbre",
        ),
        "v4-anchor-a": rated(
          ["r-challenger", "r-fill", "r-shared", "r-control"],
          "tone_timbre",
        ),
        "v4-task-b": {
          outcome: "skipped",
          ranked_choice_ids: null,
          worst_primary_reason: null,
          skip_reason: "audio_problem",
        },
      },
    },
  ];
  const report = analyzeV4Snapshots(records, pack, privateMap);
  assert.deepEqual(report.record_selection, {
    valid_snapshot_files: 2,
    unique_sessions: 1,
    superseded_snapshots_ignored: 1,
    policy: "latest exported valid snapshot per session",
  });
  assert.deepEqual(report.unique_comparisons, {
    completed: 2,
    rated: 1,
    skipped: 1,
    skip_rate: 0.5,
  });
  assert.deepEqual(report.method_pairwise_evidence, {
    challenger_over_control: 1,
    control_over_challenger: 0,
    ambiguous_or_within_method: 5,
  });
  assert.equal(report.worst_primary_mismatch_reasons.tone_timbre, 1);
  assert.equal(report.skip_reasons.audio_problem, 1);
  assert.deepEqual(report.repeated_anchor_consistency, {
    completed_pairs: 1,
    rated_pairs: 1,
    exact_matches: 1,
    full_ranking_matches: 1,
    most_similar_matches: 1,
    least_similar_matches: 1,
    mismatch_reason_matches: 1,
  });
  assert.equal(report.automatic_promotion_allowed, false);
  assert.equal(report.promotion_decision, "not_evaluated");
});
