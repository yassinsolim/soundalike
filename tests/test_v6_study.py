"""Focused tests for the exposure-disjoint V6 development study."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from soundalike.ml import v5_study, v6_study
from soundalike.ml.v4_study import _content_sha256


ROOT = Path(__file__).resolve().parents[1]
V5_PACK = json.loads(
    (ROOT / "webapp" / "evaluate-v5" / "active-pack.json").read_text(
        encoding="utf-8"
    )
)


def _write_bound(path: Path, document: dict[str, object]) -> Path:
    document["content_sha256"] = _content_sha256(document)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _exposure_paths(tmp_path: Path) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, kind in enumerate(sorted(v6_study.REQUIRED_EXPOSURE_KINDS)):
        document: dict[str, object] = {
            "schema_version": 1,
            "pack_kind": kind,
            "pack_id": f"prior-{index}",
        }
        if kind in v6_study.SAME_CORPUS_EXPOSURE_KINDS:
            document["tasks"] = [
                {
                    "seed_track_id": 100 + index,
                    "candidates": [{"track_id": 200 + index}],
                }
            ]
        else:
            document["seeds"] = [
                {"query": {"title": "Prior song", "artist": "Prior Artist"}}
            ]
        paths.append(_write_bound(tmp_path / f"exposure-{index}.json", document))
    return paths


def _private_and_plan(public: dict[str, object]) -> tuple[dict, dict]:
    private_tasks = []
    signature_sources: dict[tuple[int, tuple[int, ...]], str] = {}
    for task_index, task in enumerate(public["tasks"]):
        candidate_ids = [
            candidate["track_id"] for candidate in task["candidates"]
        ]
        signature = (task["seed_track_id"], tuple(sorted(candidate_ids)))
        source = signature_sources.get(signature)
        if source:
            private_tasks.append(
                {
                    "task_id": task["task_id"],
                    "anchor_of": source,
                    "seed_track_id": task["seed_track_id"],
                }
            )
            continue
        signature_sources[signature] = task["task_id"]
        rankings = {}
        for method_index, method in enumerate(v5_study.METHODS):
            extras = list(
                range(
                    2_000_000 + task_index * 100 + method_index * 20,
                    2_000_000
                    + task_index * 100
                    + method_index * 20
                    + v5_study.RANKING_DEPTH
                    - len(candidate_ids),
                )
            )
            rankings[method] = [*candidate_ids, *extras]
        private_tasks.append(
            {
                "task_id": task["task_id"],
                "seed_track_id": task["seed_track_id"],
                "candidate_origins": {
                    str(track_id): list(v5_study.METHODS)
                    for track_id in candidate_ids
                },
                "candidate_selection_sources": dict(
                    zip(v5_study.METHODS, candidate_ids[:3])
                ),
                "method_orders": {
                    method: list(candidate_ids) for method in v5_study.METHODS
                },
                "method_rankings": rankings,
            }
        )
    private = {
        "schema_version": 1,
        "private_kind": v5_study.PRIVATE_KIND,
        "pack_id": v5_study.PACK_ID,
        "population_sha256": public["provenance"]["population_sha256"],
        "method_bindings": {
            method: {"frozen": True} for method in v5_study.METHODS
        },
        "tasks": private_tasks,
    }
    private["content_sha256"] = _content_sha256(private)
    plan = {
        "schema_version": 1,
        "plan_kind": v5_study.PLAN_KIND,
        "source_fingerprint": public["provenance"]["source_fingerprint"],
        "population_sha256": public["provenance"]["population_sha256"],
        "detector_gate_sha256": public["provenance"]["detector_gate_sha256"],
        "vibe_cache_sha256": "1" * 64,
        "preference_model_sha256": "2" * 64,
        "exposure_pack_sha256s": {},
        "excluded_track_count": 0,
        "excluded_artist_count": 0,
        "eligible_reserve_track_count": 100,
        "shortlisted_seed_ids": [],
        "rankings": [],
        "public_pack_sha256": public["content_sha256"],
    }
    plan["content_sha256"] = _content_sha256(plan)
    return private, plan


def _inventory() -> v6_study.ExposureInventory:
    return v6_study.ExposureInventory(
        hashes={
            kind: f"{index:064x}"
            for index, kind in enumerate(
                sorted(v6_study.REQUIRED_EXPOSURE_KINDS), 1
            )
        },
        same_corpus_paths=(),
        same_corpus_labels=(),
        external_artist_names=("Prior Artist",),
    )


def _converted():
    public_v5 = copy.deepcopy(V5_PACK)
    private_v5, plan_v5 = _private_and_plan(public_v5)
    return v6_study.convert_v5_artifacts(
        public_v5, private_v5, plan_v5, _inventory()
    )


def test_exposure_contract_requires_every_prior_kind_including_v5(tmp_path):
    paths = _exposure_paths(tmp_path)
    inventory = v6_study.load_v6_exposures(paths)

    assert set(inventory.hashes) == v6_study.REQUIRED_EXPOSURE_KINDS
    assert (
        "soundalike_v5_strict_three_method_ranking" in inventory.hashes
    )
    assert inventory.external_artist_names == ("Prior Artist",)
    assert len(inventory.same_corpus_paths) == 6
    assert len(set(inventory.same_corpus_labels)) == 6

    without_v5 = [
        path
        for path in paths
        if json.loads(path.read_text(encoding="utf-8"))["pack_kind"]
        != "soundalike_v5_strict_three_method_ranking"
    ]
    with pytest.raises(v6_study.V6StudyError, match="missing=.*v5"):
        v6_study.load_v6_exposures(without_v5)


def test_exposure_contract_rejects_hash_drift_and_duplicate_kinds(tmp_path):
    paths = _exposure_paths(tmp_path)
    drifted = json.loads(paths[0].read_text(encoding="utf-8"))
    drifted["pack_id"] = "tampered"
    paths[0].write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(v6_study.V6StudyError, match="binding failed"):
        v6_study.load_v6_exposures(paths)

    paths = _exposure_paths(tmp_path / "fresh")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(paths[0].read_bytes())
    with pytest.raises(v6_study.V6StudyError, match="duplicate"):
        v6_study.load_v6_exposures([*paths, duplicate])


def test_v6_conversion_is_deterministic_blinded_and_development_only():
    first = _converted()
    second = _converted()
    assert first == second
    public, private, plan = first
    v6_study.validate_v6_artifacts(
        public, private, plan, exposure=_inventory()
    )

    assert public["development_evidence"] is True
    assert public["independent_holdout"] is False
    assert public["promotion_allowed"] is False
    assert public["evidence_role"] == "development_model_improvement"
    assert public["task_format"]["ranking_slots"] == list(
        v6_study.RANKING_SLOTS
    )
    assert all(len(task["candidates"]) == 4 for task in public["tasks"])
    assert all(task["task_id"].startswith("v6-") for task in public["tasks"])
    assert all(
        candidate["choice_id"].startswith("v6-choice-")
        for task in public["tasks"]
        for candidate in task["candidates"]
    )
    public_text = json.dumps(public, sort_keys=True)
    for marker in (
        "method_bindings",
        "method_orders",
        "method_rankings",
        "candidate_selection_sources",
    ):
        assert marker not in public_text
    assert plan["exposure_pack_sha256s"] == _inventory().hashes


def test_v6_protocol_has_new_isolated_schema_hash_and_storage_names():
    public, _, _ = _converted()
    protocol = v6_study.build_v6_protocol(public["content_sha256"])
    v6_study.validate_v6_protocol(
        protocol, pack_sha256=public["content_sha256"]
    )

    assert protocol["submission_schema"] == (
        "v6_development_listener_submission_v1"
    )
    assert protocol["local_storage_namespace"] == (
        "soundalike-development-v6-ranking-v1"
    )
    assert protocol["private_blob_prefix"] == (
        "human-ratings/development-v6-ranking-v1/"
    )
    assert protocol["submission_endpoint"] == "/api/ratings-v6"
    assert protocol["ranking_slots"] == list(v6_study.RANKING_SLOTS)
    assert protocol["development_evidence"] is True
    assert protocol["independent_holdout"] is False
    assert protocol["content_sha256"] == _content_sha256(protocol)


@pytest.mark.parametrize("tamper", ["role", "playback", "artist", "leak"])
def test_v6_validator_fails_closed_on_role_media_diversity_and_blinding(tamper):
    public, private, plan = map(copy.deepcopy, _converted())
    if tamper == "role":
        public["independent_holdout"] = True
    elif tamper == "playback":
        track = next(iter(public["tracks"].values()))
        track["audio"]["url"] = "https://example.invalid/audio.mp3"
    elif tamper == "artist":
        first_task = next(
            task
            for task in public["tasks"]
            if not next(
                row for row in private["tasks"] if row["task_id"] == task["task_id"]
            ).get("anchor_of")
        )
        seed = public["tracks"][str(first_task["seed_track_id"])]
        candidate = public["tracks"][
            str(first_task["candidates"][0]["track_id"])
        ]
        candidate["source_identity"]["artist_id"] = seed["source_identity"][
            "artist_id"
        ]
    else:
        public["provenance"]["method_orders"] = {}
    public["content_sha256"] = _content_sha256(public)
    plan["public_pack_sha256"] = public["content_sha256"]
    plan["content_sha256"] = _content_sha256(plan)
    with pytest.raises(v6_study.V6StudyError):
        v6_study.validate_v6_artifacts(
            public, private, plan, exposure=_inventory()
        )


def test_v6_preflight_reports_missing_private_inputs_without_building(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(v6_study.V6StudyError) as raised:
        v6_study.build_study(
            metadata_root=missing / "metadata",
            audio_root=missing / "audio",
            state_root=missing / "state",
            population_path=missing / "population.json",
            store_root=missing / "store",
            predictor_model=missing / "predictor.npz",
            predictor_metadata=missing / "predictor.json",
            semantic_cache_path=missing / "semantic.npz",
            semantic_metadata_path=missing / "semantic.json",
            vibe_cache_path=missing / "vibe.npz",
            preference_model_path=missing / "preference.json",
            blinding_key_path=missing / "blinding-key.private",
            gate_cache_path=missing / "gate.json",
            exposure_pack_paths=[],
            workers=1,
        )
    message = str(raised.value)
    assert "private input preflight failed" in message
    assert "population is not a file" in message
    assert "gate_cache is not a file" in message
    assert not any(tmp_path.rglob("*")), "preflight must not fabricate inputs"


def test_private_analyst_outputs_stay_outside_and_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    deploy = (ROOT / "webapp" / "DEPLOY.md").read_text(encoding="utf-8")

    for pattern in (
        "private-ratings-*/",
        "private-spicetify-feedback/",
        "private-*-unblinding.json",
        "ratings-*-analysis.local.json",
    ):
        assert pattern in ignore
    assert 'PRIVATE_ROOT="${SOUNDALIKE_PRIVATE_ROOT:-$HOME/.soundalike/private}"' in deploy
    assert "../private-ratings-v6-inbox" not in deploy
    assert "../private-spicetify-feedback" not in deploy
    assert "../private-v6-unblinding.json" not in deploy
