"""Build a strict, exposure-disjoint three-method V5 listening study."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import itertools
import json
import secrets
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np

from .fulltrack_extract import normalize_rows
from .fulltrack_store import FullTrackStoreReader
from .jamendo_fulltrack import JamendoTrack, load_jamendo_context
from .pacing_eval import (
    acoustic_scores,
    compatibility_components,
    percentile_scores,
    robust_standardize_vibe,
)
from .semantic_predictor import load_predictor
from .v4_gates import UNKNOWN, VOCAL, compatibility_allowed
from .v4_population import validate_population_manifest
from .v4_preference import FEATURE_NAMES, MODEL_KIND, score_features
from .v4_study import (
    CANDIDATE_POOL,
    MAXIMUM_SEED_SECONDS,
    MINIMUM_SEED_SECONDS,
    NICHE_TAGS,
    RERANK_WEIGHTS,
    SEED_SHORTLIST,
    _artist_diverse,
    _artist_unique_pool,
    _choose_seeds,
    _content_sha256,
    _effective_gate,
    _excerpt,
    _load_gate_cache,
    _load_or_extract_vibe,
    _nested_keys,
    _sha256,
    _track_record,
    _write,
)
from .v4_features import load_semantic_cache
from .v5_track_gates import V5TrackGateError, validate_multisegment_gate_rows


SCHEMA_VERSION = 1
PACK_KIND = "soundalike_v5_strict_three_method_ranking"
PRIVATE_KIND = "soundalike_v5_strict_three_method_ranking_private"
PLAN_KIND = "soundalike_v5_strict_three_method_study_plan"
PACK_ID = "v5-strict-three-method-ranking-1"
METHODS = ("acoustic_control", "fixed_v4", "frozen_preference_v1")
RANKING_DEPTH = 15
EXCLUSIVE_DEPTH = 10
UNIQUE_TASKS = 16
ANCHOR_TASKS = 2
CANDIDATES_PER_TASK = 4


class V5StudyError(RuntimeError):
    """The V5 study inputs or generated artifacts are invalid."""


def _collect_track_ids(value: object) -> set[int]:
    track_ids: set[int] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"track_id", "seed_track_id"}:
                if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                    raise V5StudyError("exposure pack contains an invalid track ID")
                track_ids.add(item)
            else:
                track_ids.update(_collect_track_ids(item))
    elif isinstance(value, list):
        for item in value:
            track_ids.update(_collect_track_ids(item))
    return track_ids


def _load_exposure_packs(
    paths: Sequence[Path],
) -> tuple[set[int], Mapping[str, str]]:
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise V5StudyError("V5 requires distinct prior exposure packs")
    track_ids: set[int] = set()
    hashes = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("content_sha256") != _content_sha256(document):
            raise V5StudyError(f"exposure pack binding failed: {path.name}")
        current_ids = _collect_track_ids(document)
        if not current_ids:
            raise V5StudyError(f"exposure pack contains no tracks: {path.name}")
        track_ids.update(current_ids)
        if path.name in hashes:
            raise V5StudyError("V5 exposure pack filenames must be distinct")
        hashes[path.name] = str(document["content_sha256"])
    return track_ids, dict(sorted(hashes.items()))


def _load_preference_artifact(
    path: Path,
    *,
    source_fingerprint: str,
    semantic_cache_path: Path,
    predictor_model: Path,
) -> Mapping[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    bindings = artifact.get("bindings")
    validation = artifact.get("validation")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("model_kind") != MODEL_KIND
        or artifact.get("content_sha256") != _content_sha256(artifact)
        or not isinstance(bindings, Mapping)
        or not isinstance(validation, Mapping)
        or bindings.get("source_fingerprint") != source_fingerprint
        or bindings.get("semantic_cache_sha256") != _sha256(semantic_cache_path)
        or bindings.get("predictor_model_sha256") != _sha256(predictor_model)
    ):
        raise V5StudyError("frozen preference model binding failed")
    model = validation.get("final_model")
    if (
        not isinstance(model, Mapping)
        or model.get("feature_names") != list(FEATURE_NAMES)
        or len(cast(Sequence[object], model.get("coefficients", ())))
        != len(FEATURE_NAMES)
    ):
        raise V5StudyError("frozen preference model coefficients are invalid")
    return artifact


def _select_candidates(
    rankings: Mapping[str, Sequence[int]],
    artist_by_track: Mapping[int, int],
    *,
    seed_artist: int,
    blocked_tracks: set[int],
    blocked_artists: set[int],
) -> tuple[list[int], Mapping[int, list[str]], Mapping[str, int]]:
    if set(rankings) != set(METHODS):
        raise V5StudyError("V5 method rankings are incomplete")
    origins: dict[int, list[str]] = {}
    for method in METHODS:
        ranking = rankings[method]
        if len(ranking) != RANKING_DEPTH or len(set(ranking)) != RANKING_DEPTH:
            raise V5StudyError("V5 method ranking identity drift")
        for track_id in ranking:
            origins.setdefault(int(track_id), []).append(method)

    chosen: list[int] = []
    chosen_artists = {seed_artist}
    selected_for = {}
    for method in METHODS:
        for track_id in rankings[method][:EXCLUSIVE_DEPTH]:
            track_id = int(track_id)
            artist_id = artist_by_track[track_id]
            if (
                track_id not in chosen
                and track_id not in blocked_tracks
                and artist_id not in chosen_artists
                and artist_id not in blocked_artists
            ):
                chosen.append(track_id)
                chosen_artists.add(artist_id)
                selected_for[method] = track_id
                break
        else:
            raise V5StudyError(
                f"no distinct top-{EXCLUSIVE_DEPTH} candidate for {method}"
            )

    rank_positions = {
        method: {int(track_id): index for index, track_id in enumerate(ranking)}
        for method, ranking in rankings.items()
    }
    fill_candidates = []
    for track_id, track_origins in origins.items():
        artist_id = artist_by_track[track_id]
        if (
            track_id in chosen
            or track_id in blocked_tracks
            or artist_id in chosen_artists
            or artist_id in blocked_artists
        ):
            continue
        positions = [
            rank_positions[method].get(track_id, RANKING_DEPTH) for method in METHODS
        ]
        fill_candidates.append(
            (
                -max(positions) + min(positions),
                min(positions),
                -len(track_origins),
                track_id,
            )
        )
    if not fill_candidates:
        raise V5StudyError("V5 task has no artist-distinct disagreement fill")
    fill = min(fill_candidates)[-1]
    chosen.append(fill)
    return (
        chosen,
        {track_id: origins[track_id] for track_id in chosen},
        selected_for,
    )


def _method_disagreement(rankings: Mapping[str, Sequence[int]]) -> float:
    overlaps = [
        len(set(rankings[left][:4]) & set(rankings[right][:4])) / 4.0
        for left, right in itertools.combinations(METHODS, 2)
    ]
    return 1.0 - float(np.mean(overlaps))


def build_study(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    population_path: Path,
    store_root: Path,
    predictor_model: Path,
    predictor_metadata: Path,
    semantic_cache_path: Path,
    semantic_metadata_path: Path,
    vibe_cache_path: Path,
    preference_model_path: Path,
    blinding_key_path: Path,
    gate_cache_path: Path,
    exposure_pack_paths: Sequence[Path],
    workers: int,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if workers <= 0:
        raise V5StudyError("V5 worker count is invalid")
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    population = json.loads(population_path.read_text(encoding="utf-8"))
    validate_population_manifest(population, context)
    original_reserve_ids = set(population["human_reserve"]["track_ids"])
    all_tracks = {int(track.track_id): track for track in context.tracks}

    exposed_track_ids, exposure_hashes = _load_exposure_packs(exposure_pack_paths)
    if not exposed_track_ids.issubset(all_tracks):
        raise V5StudyError("an exposure pack references an unknown corpus track")
    exposed_artist_ids = {
        int(all_tracks[track_id].artist_id) for track_id in exposed_track_ids
    }
    reserve_tracks = tuple(
        track
        for track in context.tracks
        if int(track.track_id) in original_reserve_ids
        and int(track.artist_id) not in exposed_artist_ids
    )
    by_id = {int(track.track_id): track for track in reserve_tracks}
    artist_by_track = {
        int(track.track_id): int(track.artist_id) for track in reserve_tracks
    }
    if len(reserve_tracks) < CANDIDATE_POOL:
        raise V5StudyError("exposure-disjoint reserve is too small")

    predictor = load_predictor(predictor_model, predictor_metadata)
    gate_cache = _load_gate_cache(
        gate_cache_path,
        source_fingerprint=context.source_fingerprint,
    )
    if gate_cache is None or set(gate_cache["tracks"]) != {
        str(track_id) for track_id in original_reserve_ids
    }:
        raise V5StudyError("V5 detector gate cache does not cover the full reserve")
    try:
        validate_multisegment_gate_rows(
            cast(Mapping[str, Mapping[str, object]], gate_cache["tracks"])
        )
    except V5TrackGateError as error:
        raise V5StudyError("V5 detector gate cache violates strict policy") from error
    preference = _load_preference_artifact(
        preference_model_path,
        source_fingerprint=context.source_fingerprint,
        semantic_cache_path=semantic_cache_path,
        predictor_model=predictor_model,
    )

    if not blinding_key_path.exists():
        blinding_key_path.parent.mkdir(parents=True, exist_ok=True)
        blinding_key_path.write_bytes(secrets.token_bytes(32))
    key = blinding_key_path.read_bytes()
    if len(key) != 32:
        raise V5StudyError("V5 blinding key must contain 32 bytes")

    with FullTrackStoreReader(
        store_root, expected_source_fingerprint=context.source_fingerprint
    ) as reader:
        store_ids = np.asarray(reader.track_ids, dtype=np.int64)
        store_row = {
            int(track_id): row for row, track_id in enumerate(store_ids)
        }
        reserve_rows = np.asarray(
            [store_row[int(track.track_id)] for track in reserve_tracks],
            dtype=np.int64,
        )
        globals_matrix = normalize_rows(
            np.asarray(reader.global_embeddings, dtype=np.float32)
        )
        reserve_globals = globals_matrix[reserve_rows]
        probabilities, voice_scores, _ = load_semantic_cache(
            semantic_cache_path,
            semantic_metadata_path,
            expected_source_fingerprint=context.source_fingerprint,
            expected_track_ids=store_ids,
        )
        seed_ids = _choose_seeds(reserve_tracks, reserve_globals, gate_cache)
        seed_artist_ids = {artist_by_track[seed_id] for seed_id in seed_ids}
        reserve_position = {
            int(track.track_id): position
            for position, track in enumerate(reserve_tracks)
        }
        pools = {}
        all_ids = set(seed_ids)
        for seed_id in seed_ids:
            seed_position = reserve_position[seed_id]
            query_gate = _effective_gate(seed_id, UNKNOWN, gate_cache)
            eligible_positions = []
            for position, candidate in enumerate(reserve_tracks):
                candidate_id = int(candidate.track_id)
                if (
                    candidate_id in seed_ids
                    or int(candidate.artist_id) in seed_artist_ids
                    or int(candidate.artist_id) == artist_by_track[seed_id]
                ):
                    continue
                candidate_gate = _effective_gate(
                    candidate_id, UNKNOWN, gate_cache
                )
                if not compatibility_allowed(
                    query_gate[0],
                    candidate_gate[0],
                    query_gate[1],
                    candidate_gate[1],
                ):
                    continue
                eligible_positions.append(position)
            if len(eligible_positions) < RANKING_DEPTH:
                raise V5StudyError("seed lacks enough strict-compatible candidates")
            eligible = np.asarray(eligible_positions, dtype=np.int64)
            similarities = (
                reserve_globals[eligible] @ reserve_globals[seed_position]
            )
            pool_positions = _artist_unique_pool(
                eligible,
                similarities,
                reserve_tracks,
                CANDIDATE_POOL,
            )
            pool_ids = tuple(
                int(reserve_tracks[position].track_id)
                for position in pool_positions
            )
            pools[seed_id] = pool_ids
            all_ids.update(pool_ids)

        ordered_ids = np.asarray(sorted(all_ids), dtype=np.int64)
        excerpts = {
            int(track_id): _excerpt(reader, int(track_id))
            for track_id in ordered_ids
        }
        vibe = _load_or_extract_vibe(
            vibe_cache_path,
            ordered_ids,
            excerpts,
            by_id,
            workers,
        )
        standardized = robust_standardize_vibe(vibe)
        vibe_row = {
            int(track_id): position
            for position, track_id in enumerate(ordered_ids)
        }
        ranking_rows = []
        for seed_id in seed_ids:
            pool_ids = pools[seed_id]
            candidate_rows = np.asarray(
                [store_row[track_id] for track_id in pool_ids], dtype=np.int64
            )
            query_row = store_row[seed_id]
            global_scores = (
                globals_matrix[candidate_rows] @ globals_matrix[query_row]
            )
            acoustic = acoustic_scores(
                reader, seed_id, pool_ids, global_scores
            )
            pool_vibe_rows = np.asarray(
                [vibe_row[track_id] for track_id in pool_ids], dtype=np.int64
            )
            query_vibe_row = vibe_row[seed_id]
            components = compatibility_components(
                vibe[pool_vibe_rows],
                vibe[query_vibe_row],
                standardized[pool_vibe_rows],
                standardized[query_vibe_row],
                probabilities[candidate_rows],
                probabilities[query_row],
                predictor,
            )
            voice_compatibility = np.exp(
                -np.abs(
                    voice_scores[candidate_rows] - voice_scores[query_row]
                )
                / 0.05
            )
            fixed_v4 = RERANK_WEIGHTS["acoustic"] * percentile_scores(acoustic)
            for name in (
                "pacing",
                "tone",
                "dynamics",
                "instrument",
                "mood_theme",
                "genre",
            ):
                fixed_v4 += RERANK_WEIGHTS[name] * percentile_scores(
                    components[name]
                )
            fixed_v4 += RERANK_WEIGHTS[
                "voice_compatibility"
            ] * percentile_scores(voice_compatibility)
            feature_matrix = np.column_stack(
                [
                    acoustic,
                    components["pacing"],
                    components["tone"],
                    components["dynamics"],
                    components["instrument"],
                    components["mood_theme"],
                    components["genre"],
                    voice_compatibility,
                ]
            )
            preference_scores = score_features(feature_matrix, preference)
            method_score_arrays = {
                "acoustic_control": acoustic,
                "fixed_v4": fixed_v4,
                "frozen_preference_v1": preference_scores,
            }
            rankings = {
                "acoustic_control": list(
                    _artist_diverse(
                        pool_ids, acoustic, artist_by_track, RANKING_DEPTH
                    )
                ),
                "fixed_v4": list(
                    _artist_diverse(
                        pool_ids, fixed_v4, artist_by_track, RANKING_DEPTH
                    )
                ),
                "frozen_preference_v1": list(
                    _artist_diverse(
                        pool_ids,
                        preference_scores,
                        artist_by_track,
                        RANKING_DEPTH,
                    )
                ),
            }
            ranking_rows.append(
                {
                    "seed_track_id": seed_id,
                    "pool_ids": list(pool_ids),
                    "method_scores": {
                        method: [float(value) for value in values]
                        for method, values in method_score_arrays.items()
                    },
                    "method_rankings": rankings,
                    "disagreement": _method_disagreement(rankings),
                    "niche": bool(set(by_id[seed_id].tags) & NICHE_TAGS),
                }
            )

    ranking_rows.sort(
        key=lambda row: (
            row["niche"],
            -row["disagreement"],
            row["seed_track_id"],
        )
    )
    selected = []
    used_candidates: set[int] = set()
    used_candidate_artists: set[int] = set()
    blocked_seeds = set(seed_ids)
    construction_failures: Counter[str] = Counter()
    for row in ranking_rows:
        try:
            candidates, origins, selected_for = _select_candidates(
                row["method_rankings"],
                artist_by_track,
                seed_artist=artist_by_track[row["seed_track_id"]],
                blocked_tracks=blocked_seeds | used_candidates,
                blocked_artists=used_candidate_artists,
            )
        except V5StudyError as error:
            construction_failures[str(error)] += 1
            continue
        score_by_method = {
            method: {
                int(track_id): float(score)
                for track_id, score in zip(
                    row["pool_ids"], row["method_scores"][method]
                )
            }
            for method in METHODS
        }
        method_orders = {
            method: sorted(
                candidates,
                key=lambda track_id: (
                    -score_by_method[method][track_id],
                    track_id,
                ),
            )
            for method in METHODS
        }
        selected.append(
            {
                **row,
                "candidates": candidates,
                "origins": origins,
                "selected_for": selected_for,
                "method_orders": method_orders,
            }
        )
        used_candidates.update(candidates)
        used_candidate_artists.update(artist_by_track[value] for value in candidates)
        if len(selected) == UNIQUE_TASKS:
            break
    if len(selected) != UNIQUE_TASKS:
        raise V5StudyError(
            "cannot construct 16 independent V5 tasks: "
            f"built {len(selected)}; failures={dict(construction_failures)}"
        )

    def opaque_id(prefix: str, *parts: object) -> str:
        payload = "\0".join(str(part) for part in parts).encode("utf-8")
        return f"{prefix}-{hmac.new(key, payload, hashlib.sha256).hexdigest()[:24]}"

    tasks = []
    private_tasks = []
    used_tracks = set()
    for priority, row in enumerate(selected, 1):
        task_id = opaque_id("v5-task", row["seed_track_id"], priority)
        order = sorted(
            row["candidates"],
            key=lambda track_id: hmac.new(
                key,
                f"{task_id}\0{track_id}".encode(),
                hashlib.sha256,
            ).hexdigest(),
        )
        tasks.append(
            {
                "task_id": task_id,
                "priority_rank": priority,
                "seed_track_id": row["seed_track_id"],
                "candidates": [
                    {
                        "choice_id": opaque_id(
                            "v5-choice", task_id, track_id
                        ),
                        "track_id": track_id,
                    }
                    for track_id in order
                ],
            }
        )
        private_tasks.append(
            {
                "task_id": task_id,
                "seed_track_id": row["seed_track_id"],
                "candidate_origins": {
                    str(track_id): row["origins"][track_id] for track_id in order
                },
                "candidate_selection_sources": row["selected_for"],
                "method_orders": row["method_orders"],
                "method_rankings": row["method_rankings"],
            }
        )
        used_tracks.update([row["seed_track_id"], *row["candidates"]])

    for anchor_index, source_index in enumerate((0, len(tasks) // 2), 1):
        source = tasks[source_index]
        task_id = opaque_id("v5-anchor", source["task_id"], anchor_index)
        tasks.append(
            {
                **source,
                "task_id": task_id,
                "priority_rank": len(tasks) + 1,
                "candidates": [
                    {
                        "choice_id": opaque_id(
                            "v5-choice", task_id, candidate["track_id"]
                        ),
                        "track_id": candidate["track_id"],
                    }
                    for candidate in reversed(source["candidates"])
                ],
            }
        )
        private_tasks.append(
            {
                "task_id": task_id,
                "anchor_of": source["task_id"],
                "seed_track_id": source["seed_track_id"],
            }
        )

    presentation_order = [
        *tasks[:6],
        tasks[UNIQUE_TASKS],
        *tasks[6:12],
        tasks[UNIQUE_TASKS + 1],
        *tasks[12:UNIQUE_TASKS],
    ]
    private_by_task = {row["task_id"]: row for row in private_tasks}
    tasks = [
        {**task, "priority_rank": position}
        for position, task in enumerate(presentation_order, 1)
    ]
    private_tasks = [private_by_task[task["task_id"]] for task in tasks]

    public: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pack_kind": PACK_KIND,
        "pack_id": PACK_ID,
        "research_only": True,
        "promotion_allowed": False,
        "production_recommendation_changed": False,
        "task_format": {
            "seed": 1,
            "candidates": CANDIDATES_PER_TASK,
            "questions": ["full_similarity_ranking", "worst_primary_reason"],
            "partial_submission_allowed": True,
            "adaptive_stop_after_unique_tasks": 12,
        },
        "provenance": {
            "source_fingerprint": context.source_fingerprint,
            "population_sha256": population["content_sha256"],
            "detector_gate_sha256": gate_cache["content_sha256"],
            "prior_exposure_pack_sha256s": exposure_hashes,
            "frozen_method_count": len(METHODS),
            "method_identity_public": False,
            "listener_ratings_used_for_pack_selection": False,
            "candidate_gate": (
                "known voice class required; vocal tracks require three stable "
                "language decisions and exact seed-language compatibility"
            ),
        },
        "tasks": tasks,
        "tracks": {
            str(track_id): _track_record(by_id[track_id], excerpts[track_id])
            for track_id in sorted(used_tracks)
        },
    }
    private: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "private_kind": PRIVATE_KIND,
        "pack_id": PACK_ID,
        "population_sha256": population["content_sha256"],
        "method_bindings": {
            "acoustic_control": "frozen 30/25/25/20 acoustic score",
            "fixed_v4": {
                "weights": RERANK_WEIGHTS,
                "detector_gate_sha256": gate_cache["content_sha256"],
            },
            "frozen_preference_v1": {
                "artifact_sha256": preference["content_sha256"],
                "feature_names": list(FEATURE_NAMES),
                "detector_gate_sha256": gate_cache["content_sha256"],
            },
        },
        "tasks": private_tasks,
    }
    private["content_sha256"] = _content_sha256(private)
    public["private_unblinding_sha256"] = private["content_sha256"]
    public["content_sha256"] = _content_sha256(public)
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "plan_kind": PLAN_KIND,
        "source_fingerprint": context.source_fingerprint,
        "population_sha256": population["content_sha256"],
        "detector_gate_sha256": gate_cache["content_sha256"],
        "vibe_cache_sha256": _sha256(vibe_cache_path),
        "preference_model_sha256": preference["content_sha256"],
        "exposure_pack_sha256s": exposure_hashes,
        "excluded_track_count": len(exposed_track_ids),
        "excluded_artist_count": len(exposed_artist_ids),
        "eligible_reserve_track_count": len(reserve_tracks),
        "shortlisted_seed_ids": list(seed_ids),
        "rankings": ranking_rows,
        "public_pack_sha256": public["content_sha256"],
    }
    plan["content_sha256"] = _content_sha256(plan)
    validate_study_artifacts(public, private, plan, gate_cache=gate_cache)
    return public, private, plan


def validate_study_artifacts(
    public: Mapping[str, object],
    private: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    gate_cache: Mapping[str, object],
) -> None:
    provenance = public.get("provenance")
    if (
        public.get("schema_version") != SCHEMA_VERSION
        or public.get("pack_kind") != PACK_KIND
        or public.get("pack_id") != PACK_ID
        or public.get("content_sha256") != _content_sha256(public)
        or private.get("private_kind") != PRIVATE_KIND
        or private.get("pack_id") != PACK_ID
        or private.get("content_sha256") != _content_sha256(private)
        or plan.get("plan_kind") != PLAN_KIND
        or plan.get("content_sha256") != _content_sha256(plan)
        or public.get("private_unblinding_sha256")
        != private.get("content_sha256")
        or plan.get("public_pack_sha256") != public.get("content_sha256")
        or not isinstance(provenance, Mapping)
        or provenance.get("detector_gate_sha256")
        != gate_cache.get("content_sha256")
    ):
        raise V5StudyError("V5 study artifact binding failed")
    forbidden = set(METHODS) | {
        "candidate_selection_sources",
        "method_bindings",
        "method_orders",
        "method_rankings",
    }
    if forbidden & _nested_keys(public):
        raise V5StudyError("V5 public artifact leaks method identity")

    tasks = public.get("tasks")
    private_tasks = private.get("tasks")
    tracks = public.get("tracks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != UNIQUE_TASKS + ANCHOR_TASKS
        or not isinstance(private_tasks, list)
        or len(private_tasks) != len(tasks)
        or not isinstance(tracks, Mapping)
    ):
        raise V5StudyError("V5 study task coverage drift")
    if any(not isinstance(task, Mapping) for task in [*tasks, *private_tasks]):
        raise V5StudyError("V5 study task rows are invalid")
    priorities = [task.get("priority_rank") for task in tasks]
    if (
        any(
            isinstance(priority, bool) or not isinstance(priority, int)
            for priority in priorities
        )
        or priorities != list(range(1, len(tasks) + 1))
    ):
        raise V5StudyError("V5 priority order drift")
    public_task_ids = [task.get("task_id") for task in tasks]
    private_task_ids = [task.get("task_id") for task in private_tasks]
    if (
        any(not isinstance(task_id, str) or not task_id for task_id in public_task_ids)
        or any(
            not isinstance(task_id, str) or not task_id
            for task_id in private_task_ids
        )
        or len(set(public_task_ids)) != len(tasks)
        or len(set(private_task_ids)) != len(tasks)
        or set(public_task_ids) != set(private_task_ids)
    ):
        raise V5StudyError("V5 public/private task IDs are invalid")
    private_by_id = {task.get("task_id"): task for task in private_tasks}
    public_by_id = {task.get("task_id"): task for task in tasks}

    used_tracks = set()
    seen_candidates = set()
    seen_artists = set()
    seen_choice_ids = set()
    anchor_sources = set()
    unique_task_count = 0
    anchor_task_count = 0

    def track_artist(track_id: int) -> int:
        track = tracks.get(str(track_id))
        identity = track.get("source_identity") if isinstance(track, Mapping) else None
        artist_id = identity.get("artist_id") if isinstance(identity, Mapping) else None
        if (
            isinstance(artist_id, bool)
            or not isinstance(artist_id, int)
            or artist_id <= 0
        ):
            raise V5StudyError("V5 public track identity is invalid")
        return artist_id

    for task in tasks:
        task_id = task.get("task_id")
        seed_id = task.get("seed_track_id")
        candidates = task.get("candidates")
        private_task = private_by_id.get(task_id)
        if (
            isinstance(seed_id, bool)
            or not isinstance(seed_id, int)
            or seed_id <= 0
            or not isinstance(candidates, list)
            or len(candidates) != CANDIDATES_PER_TASK
            or any(not isinstance(row, Mapping) for row in candidates)
            or not isinstance(private_task, Mapping)
            or private_task.get("seed_track_id") != seed_id
        ):
            raise V5StudyError("V5 task structure drift")
        candidate_ids = [row.get("track_id") for row in candidates]
        choice_ids = [row.get("choice_id") for row in candidates]
        if (
            any(
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id <= 0
                for track_id in candidate_ids
            )
            or len(set(candidate_ids)) != CANDIDATES_PER_TASK
            or any(
                not isinstance(choice_id, str) or not choice_id
                for choice_id in choice_ids
            )
            or len(set(choice_ids)) != CANDIDATES_PER_TASK
            or seen_choice_ids & set(choice_ids)
        ):
            raise V5StudyError("V5 candidate identity drift")
        seen_choice_ids.update(choice_ids)
        used_tracks.update([seed_id, *candidate_ids])
        track_artist(seed_id)
        for candidate_id in candidate_ids:
            track_artist(candidate_id)
        seed_gate = _effective_gate(seed_id, UNKNOWN, gate_cache)
        if seed_gate[0] == UNKNOWN or (
            seed_gate[0] == VOCAL and seed_gate[1] == UNKNOWN
        ):
            raise V5StudyError("V5 task exposes an uncertain seed")
        for candidate_id in candidate_ids:
            candidate_gate = _effective_gate(
                candidate_id, UNKNOWN, gate_cache
            )
            if not compatibility_allowed(
                seed_gate[0],
                candidate_gate[0],
                seed_gate[1],
                candidate_gate[1],
            ):
                raise V5StudyError("V5 task violates the strict gate")
        if "anchor_of" not in private_task:
            unique_task_count += 1
            if seen_candidates & set(candidate_ids):
                raise V5StudyError("V5 unique tasks reuse a candidate")
            seen_candidates.update(candidate_ids)
            task_artists = {
                track_artist(track_id)
                for track_id in [seed_id, *candidate_ids]
            }
            if len(task_artists) != 5 or seen_artists & task_artists:
                raise V5StudyError("V5 unique tasks reuse an artist")
            seen_artists.update(task_artists)
            origins = private_task.get("candidate_origins")
            rankings = private_task.get("method_rankings")
            selection_sources = private_task.get("candidate_selection_sources")
            method_orders = private_task.get("method_orders")
            if (
                not isinstance(origins, Mapping)
                or set(origins) != {str(value) for value in candidate_ids}
                or not isinstance(rankings, Mapping)
                or set(rankings) != set(METHODS)
                or not isinstance(selection_sources, Mapping)
                or set(selection_sources) != set(METHODS)
                or not isinstance(method_orders, Mapping)
                or set(method_orders) != set(METHODS)
            ):
                raise V5StudyError("V5 private method mapping drift")
            selected_values = list(selection_sources.values())
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in candidate_ids
                    for value in selected_values
                )
                or len(set(selected_values)) != len(METHODS)
                or any(
                    not isinstance(origins[str(candidate_id)], list)
                    or not origins[str(candidate_id)]
                    or any(
                        not isinstance(method, str)
                        for method in origins[str(candidate_id)]
                    )
                    or len(set(origins[str(candidate_id)]))
                    != len(origins[str(candidate_id)])
                    or not set(origins[str(candidate_id)]).issubset(METHODS)
                    for candidate_id in candidate_ids
                )
            ):
                raise V5StudyError("V5 private candidate selection drift")
            for method in METHODS:
                order = method_orders[method]
                ranking = rankings[method]
                if (
                    not isinstance(order, list)
                    or len(order) != CANDIDATES_PER_TASK
                    or any(
                        isinstance(track_id, bool)
                        or not isinstance(track_id, int)
                        or track_id <= 0
                        for track_id in order
                    )
                    or set(order) != set(candidate_ids)
                    or not isinstance(ranking, list)
                    or len(ranking) != RANKING_DEPTH
                    or any(
                        isinstance(track_id, bool)
                        or not isinstance(track_id, int)
                        or track_id <= 0
                        for track_id in ranking
                    )
                    or len(set(ranking)) != RANKING_DEPTH
                ):
                    raise V5StudyError("V5 private method order drift")
        else:
            anchor_task_count += 1
            anchor_of = private_task.get("anchor_of")
            if (
                not isinstance(anchor_of, str)
                or anchor_of == task_id
                or anchor_of in anchor_sources
            ):
                raise V5StudyError("V5 repeated anchor identity drift")
            anchor_sources.add(anchor_of)
            source = private_by_id.get(anchor_of)
            source_task = public_by_id.get(anchor_of)
            source_candidates = (
                source_task.get("candidates")
                if isinstance(source_task, Mapping)
                else None
            )
            if (
                not isinstance(source, Mapping)
                or source.get("seed_track_id") != seed_id
                or source.get("anchor_of")
                or not isinstance(source_candidates, list)
                or [
                    row.get("track_id") for row in candidates
                ]
                != [
                    row.get("track_id")
                    for row in reversed(source_candidates)
                ]
            ):
                raise V5StudyError("V5 repeated anchor binding drift")
    if (
        unique_task_count != UNIQUE_TASKS
        or anchor_task_count != ANCHOR_TASKS
        or len(seen_artists) != UNIQUE_TASKS * (CANDIDATES_PER_TASK + 1)
    ):
        raise V5StudyError("V5 unique-task or anchor coverage drift")
    if set(tracks) != {str(track_id) for track_id in used_tracks}:
        raise V5StudyError("V5 public track table coverage drift")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the strict three-method V5 listening study."
    )
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "population",
        "store_root",
        "predictor_model",
        "predictor_metadata",
        "semantic_cache",
        "semantic_metadata",
        "vibe_cache",
        "preference_model",
        "blinding_key",
        "gate_cache",
        "public_output",
        "private_output",
        "plan_output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument(
        "--exposure-pack",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    public, private, plan = build_study(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        population_path=args.population,
        store_root=args.store_root,
        predictor_model=args.predictor_model,
        predictor_metadata=args.predictor_metadata,
        semantic_cache_path=args.semantic_cache,
        semantic_metadata_path=args.semantic_metadata,
        vibe_cache_path=args.vibe_cache,
        preference_model_path=args.preference_model,
        blinding_key_path=args.blinding_key,
        gate_cache_path=args.gate_cache,
        exposure_pack_paths=args.exposure_pack,
        workers=args.workers,
    )
    _write(args.public_output, public)
    _write(args.private_output, private)
    _write(args.plan_output, plan)
    print(
        json.dumps(
            {
                "pack_id": public["pack_id"],
                "content_sha256": public["content_sha256"],
                "private_unblinding_sha256": private["content_sha256"],
                "unique_tasks": UNIQUE_TASKS,
                "anchor_tasks": ANCHOR_TASKS,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
