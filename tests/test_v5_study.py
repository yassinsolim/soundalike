"""Focused tests for strict, three-method V5 study construction."""
from __future__ import annotations

import pytest

from soundalike.ml import v5_study


def test_collect_track_ids_walks_old_and_new_pack_shapes():
    document = {
        "tasks": [
            {
                "seed_track_id": 1,
                "candidates": [{"track_id": 2}, {"track_id": 3}],
            }
        ],
        "seeds": [
            {
                "seed_track_id": 4,
                "lists": [{"results": [{"track_id": 5}]}],
            }
        ],
    }
    assert v5_study._collect_track_ids(document) == {1, 2, 3, 4, 5}


def test_collect_track_ids_rejects_boolean_ids():
    with pytest.raises(v5_study.V5StudyError, match="invalid track ID"):
        v5_study._collect_track_ids({"track_id": True})


def test_select_candidates_uses_one_distinct_top_pick_per_method():
    rankings = {
        "acoustic_control": [1, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        + list(range(100, 100 + v5_study.RANKING_DEPTH - 10)),
        "fixed_v4": [2, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        + list(range(120, 120 + v5_study.RANKING_DEPTH - 10)),
        "frozen_preference_v1": [3, 30, 31, 32, 33, 34, 35, 36, 37, 38]
        + list(range(140, 140 + v5_study.RANKING_DEPTH - 10)),
    }
    artists = {track_id: track_id for ranking in rankings.values() for track_id in ranking}
    chosen, origins, selected_for = v5_study._select_candidates(
        rankings,
        artists,
        seed_artist=999,
        blocked_tracks=set(),
        blocked_artists=set(),
    )
    assert chosen[:3] == [1, 2, 3]
    assert origins[1] == ["acoustic_control"]
    assert origins[2] == ["fixed_v4"]
    assert origins[3] == ["frozen_preference_v1"]
    assert selected_for == {
        "acoustic_control": 1,
        "fixed_v4": 2,
        "frozen_preference_v1": 3,
    }
    assert len(chosen) == 4
    assert len(set(chosen)) == 4


def test_select_candidates_fails_without_artist_distinct_top_picks():
    shared = list(range(1, v5_study.RANKING_DEPTH + 1))
    rankings = {method: shared for method in v5_study.METHODS}
    artists = {track_id: 1 for track_id in shared}
    with pytest.raises(v5_study.V5StudyError, match="distinct"):
        v5_study._select_candidates(
            rankings,
            artists,
            seed_artist=999,
            blocked_tracks=set(),
            blocked_artists=set(),
        )


def test_select_candidates_excludes_artists_used_by_prior_tasks():
    rankings = {
        method: list(range(1, v5_study.RANKING_DEPTH + 1))
        for method in v5_study.METHODS
    }
    artists = {track_id: track_id for track_id in rankings["acoustic_control"]}
    chosen, _, _ = v5_study._select_candidates(
        rankings,
        artists,
        seed_artist=999,
        blocked_tracks=set(),
        blocked_artists={1, 2, 3},
    )
    assert not {artists[track_id] for track_id in chosen} & {1, 2, 3}


def _synthetic_artifacts():
    tasks = []
    private_tasks = []
    tracks = {}
    gate_rows = {}
    for task_index in range(v5_study.UNIQUE_TASKS):
        seed_id = task_index * 5 + 1
        candidate_ids = list(range(seed_id + 1, seed_id + 5))
        task_id = f"task-{task_index}"
        candidates = [
            {"choice_id": f"choice-{task_index}-{index}", "track_id": track_id}
            for index, track_id in enumerate(candidate_ids)
        ]
        tasks.append(
            {
                "task_id": task_id,
                "priority_rank": len(tasks) + 1,
                "seed_track_id": seed_id,
                "candidates": candidates,
            }
        )
        rankings = {
            method: [
                *candidate_ids,
                *range(
                    1000 + task_index * 100 + method_index * 20,
                    1000
                    + task_index * 100
                    + method_index * 20
                    + v5_study.RANKING_DEPTH
                    - len(candidate_ids),
                ),
            ]
            for method_index, method in enumerate(v5_study.METHODS)
        }
        private_tasks.append(
            {
                "task_id": task_id,
                "seed_track_id": seed_id,
                "candidate_origins": {
                    str(track_id): list(v5_study.METHODS)
                    for track_id in candidate_ids
                },
                "candidate_selection_sources": dict(
                    zip(
                        v5_study.METHODS,
                        candidate_ids[: len(v5_study.METHODS)],
                        strict=True,
                    )
                ),
                "method_orders": {
                    method: list(candidate_ids) for method in v5_study.METHODS
                },
                "method_rankings": rankings,
            }
        )
        for track_id in [seed_id, *candidate_ids]:
            tracks[str(track_id)] = {
                "source_identity": {"artist_id": track_id}
            }
            gate_rows[str(track_id)] = {
                "vocal_state": "instrumental",
                "language": "unknown",
            }

    for anchor_index, source_index in enumerate((0, 8)):
        source = tasks[source_index]
        anchor_id = f"anchor-{anchor_index}"
        tasks.append(
            {
                "task_id": anchor_id,
                "priority_rank": len(tasks) + 1,
                "seed_track_id": source["seed_track_id"],
                "candidates": [
                    {
                        "choice_id": f"anchor-choice-{anchor_index}-{index}",
                        "track_id": candidate["track_id"],
                    }
                    for index, candidate in enumerate(
                        reversed(source["candidates"])
                    )
                ],
            }
        )
        private_tasks.append(
            {
                "task_id": anchor_id,
                "anchor_of": source["task_id"],
                "seed_track_id": source["seed_track_id"],
            }
        )

    gate = {"content_sha256": "gate", "tracks": gate_rows}
    private = {
        "schema_version": v5_study.SCHEMA_VERSION,
        "private_kind": v5_study.PRIVATE_KIND,
        "pack_id": v5_study.PACK_ID,
        "tasks": private_tasks,
    }
    public = {
        "schema_version": v5_study.SCHEMA_VERSION,
        "pack_kind": v5_study.PACK_KIND,
        "pack_id": v5_study.PACK_ID,
        "provenance": {"detector_gate_sha256": gate["content_sha256"]},
        "tasks": tasks,
        "tracks": tracks,
    }
    plan = {
        "plan_kind": v5_study.PLAN_KIND,
        "public_pack_sha256": "",
    }
    return public, private, plan, gate


def _bind_artifacts(public, private, plan):
    private["content_sha256"] = v5_study._content_sha256(private)
    public["private_unblinding_sha256"] = private["content_sha256"]
    public["content_sha256"] = v5_study._content_sha256(public)
    plan["public_pack_sha256"] = public["content_sha256"]
    plan["content_sha256"] = v5_study._content_sha256(plan)


def test_artifact_validator_rejects_malformed_types_fail_closed():
    public, private, plan, gate = _synthetic_artifacts()
    _bind_artifacts(public, private, plan)
    v5_study.validate_study_artifacts(
        public, private, plan, gate_cache=gate
    )

    public["tasks"][0]["candidates"][0]["track_id"] = []
    _bind_artifacts(public, private, plan)
    with pytest.raises(v5_study.V5StudyError, match="candidate identity"):
        v5_study.validate_study_artifacts(
            public, private, plan, gate_cache=gate
        )

    public, private, plan, gate = _synthetic_artifacts()
    private["tasks"][0]["method_orders"]["acoustic_control"][0] = []
    _bind_artifacts(public, private, plan)
    with pytest.raises(v5_study.V5StudyError, match="method order"):
        v5_study.validate_study_artifacts(
            public, private, plan, gate_cache=gate
        )
