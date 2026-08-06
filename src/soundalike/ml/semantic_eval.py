"""Build an exploratory full-track audio-versus-semantic listening study."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_eval import (
    HYBRID_WEIGHTS,
    batch_fixed_budget_maxsim,
    freeze_fixed_budget,
    freeze_ranked_section_budget,
)
from .fulltrack_extract import normalize_rows
from .fulltrack_pilot import (
    _canonical_bytes,
    _content_sha256,
    _opaque_id,
    _read_blinding_key,
    _track_record,
    _write_json,
    lawful_stream_url,
    validate_blinded_documents as validate_v2_documents,
    verify_public_audio_urls,
)
from .fulltrack_store import FullTrackStoreReader, sha256_path, stable_json_sha256
from .jamendo_fulltrack import EVIDENCE_SCOPE, JamendoContext, load_jamendo_context
from .semantic_predictor import (
    CalibratedSemanticPredictor,
    feature_domain_diagnostics,
    load_predictor,
)


SCHEMA_VERSION = 2
PACK_KIND = "fulltrack_semantic_repeated_excerpt_pilot_v2"
PRIVATE_KIND = "fulltrack_semantic_repeated_excerpt_pilot_v2_private_unblinding"
PACK_ID = "semantic-repeated-excerpt-v2-20"
METHODS = ("fulltrack_audio_control_v1", "semantic_fulltrack_v1")
RESULTS_PER_METHOD = 5
EXPECTED_V2_PACK_SHA256 = (
    "1980da60810959e7cdd24f39bd7142c8e34c76dab633c705976b85e49b297023"
)
EXPECTED_PREDICTOR_SHA256 = (
    "fb96562b4e257681924d7cb0eec5f98e9dfd17da541889a82c3b80ff00c54b56"
)
EXPECTED_PREDICTOR_PAYLOAD_SHA256 = (
    "d3749cba791b8df00edaffe8230e217ef58652fc3c51381fb4dfd86be4fe1905"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "060f43ed0fa12e5a583e26a7728be14a5334c7daffebe2289f08875e9ec0c709"
)
EXPECTED_STORE_BINDING_SHA256 = (
    "66baa07c058d842d5a5a7f068a3ea80070d5c43a4818a7a36f0192cb868de98a"
)
EXPECTED_PUBLIC_PACK_SHA256 = (
    "939b639abb6d6c6b2c7ba20ae570ff7ae9d06ee67254c219d6e5f61975403347"
)
EXPECTED_PRIVATE_UNBLINDING_SHA256 = (
    "368cb4796a167d321037a978e5673ebc87237cb78825099030f9bebc267a2d23"
)
EXPECTED_STORE_TRACKS = 55_701
EXPECTED_TEST_TRACKS = 11_565
SEMANTIC_WEIGHT = 0.25
AUDIO_WEIGHT = 0.75
CANDIDATE_POOL = 200
SECTION_BUDGET = 32
MAX_RESULTS_PER_ARTIST = 1
PLAYBACK_EXCERPT_SECONDS = 20
SOURCE_WINDOW_SECONDS = 10
SOURCE_SAMPLE_RATE = 48_000
CORE_SCENES = (
    "dance",
    "house",
    "hiphop",
    "electronic",
    "rock",
    "indie",
    "techno",
    "funk",
    "jazz",
    "trance",
    "alternative",
    "metal",
    "pop",
    "reggae",
    "folk",
)
_HEX64 = frozenset("0123456789abcdef")
_PUBLIC_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "pack_kind",
        "pack_id",
        "rankings_state",
        "ratings_count_at_freeze",
        "seed_count",
        "method_count",
        "results_per_method",
        "source_v2_pack_sha256",
        "source_fingerprint",
        "store_binding",
        "store_binding_sha256",
        "blinding",
        "audio_delivery",
        "section_coverage",
        "playback_policy",
        "seed_order_policy",
        "language_policy",
        "tracks",
        "seeds",
        "research_only",
        "promotion_allowed",
        "production_recommendation_changed",
        "notice",
        "content_sha256",
    }
)
_PRIVATE_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "pack_id",
        "source_v2_pack_sha256",
        "source_fingerprint",
        "store_binding_sha256",
        "blinding_key_hex",
        "blinding_key_sha256",
        "methods",
        "method_bindings",
        "seeds",
        "research_only",
        "promotion_allowed",
        "content_sha256",
    }
)
_BLINDING_KEYS = frozenset(
    {
        "opaque_per_seed_list_ids",
        "method_identity_public",
        "method_order_randomized_per_session",
        "blinding_key_sha256",
        "private_unblinding_sha256",
    }
)
_PUBLIC_SEED_KEYS = frozenset(
    {
        "seed_id",
        "scene",
        "seed_track_id",
        "tempo_bpm",
        "tempo_region",
        "clap_texture_region",
        "priority_rank",
        "matched_list_overlap",
        "result_ids",
        "lists",
    }
)
_PRIVATE_SEED_KEYS = frozenset({"seed_id", "seed_track_id", "lists"})
_PUBLIC_LIST_KEYS = frozenset(
    {"list_id", "binding_commitment_sha256", "ranking"}
)
_PRIVATE_LIST_KEYS = frozenset(
    {
        "pack_kind",
        "seed_id",
        "list_id",
        "method_binding",
        "ranking_track_ids",
        "binding_commitment_sha256",
    }
)
_RANKING_ROW_KEYS = frozenset({"position", "result_id", "track_id"})
_RESULT_ID_KEYS = frozenset({"result_id", "track_id"})
_SHARED_METHOD_KEYS = frozenset(
    {
        "method",
        "store_binding_sha256",
        "candidate_pool",
        "section_budget",
        "max_results_per_artist",
        "published_v2_results_excluded",
        "source_seed_pack_sha256",
        "test_labels_used_for_ranking",
        "language_metadata_used_for_ranking",
        "promoted",
        "audio_weight",
        "semantic_weight",
        "semantic_profile_used_for_ranking",
    }
)
_DOMAIN_DIAGNOSTIC_KEYS = frozenset(
    {
        "rows",
        "dimensions",
        "standardized_coordinate_mean_abs",
        "standardized_coordinate_scale_mean",
        "maximum_standardized_coordinate_mean_abs",
        "minimum_standardized_coordinate_scale_mean",
        "maximum_standardized_coordinate_scale_mean",
        "passed",
    }
)


class SemanticEvalError(RuntimeError):
    """The semantic listening study is unsafe, inconsistent, or non-reproducible."""


@dataclass(frozen=True)
class SemanticEvalConfig:
    fold_index: int = 0
    part: str = "test"
    semantic_weight: float = SEMANTIC_WEIGHT
    candidate_pool: int = CANDIDATE_POOL
    section_budget: int = SECTION_BUDGET
    results_per_method: int = RESULTS_PER_METHOD
    max_results_per_artist: int = MAX_RESULTS_PER_ARTIST

    def validate(self) -> None:
        if self.fold_index != 0 or self.part != "test":
            raise SemanticEvalError("semantic study must use the frozen fold-0 test partition")
        if self.semantic_weight != SEMANTIC_WEIGHT:
            raise SemanticEvalError("semantic study weight is frozen at 0.25")
        if self.candidate_pool != CANDIDATE_POOL:
            raise SemanticEvalError("semantic study candidate pool is frozen at 200")
        if self.section_budget != SECTION_BUDGET:
            raise SemanticEvalError("semantic study section budget is frozen at 32")
        if self.results_per_method != RESULTS_PER_METHOD:
            raise SemanticEvalError("semantic study requires five outputs per method")
        if self.max_results_per_artist != MAX_RESULTS_PER_ARTIST:
            raise SemanticEvalError("semantic study permits one result per artist")


def percentile_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.all(np.isfinite(scores)):
        raise SemanticEvalError("percentile score input is invalid")
    order = np.argsort(scores, kind="stable")
    result = np.empty(len(scores), dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(scores), dtype=np.float32)
    return result


def semantic_blend_scores(
    audio_scores: np.ndarray,
    semantic_scores: np.ndarray,
    *,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> np.ndarray:
    if semantic_weight != SEMANTIC_WEIGHT:
        raise SemanticEvalError("semantic blend weight is frozen at 0.25")
    audio = percentile_scores(audio_scores)
    semantic = percentile_scores(semantic_scores)
    if audio.shape != semantic.shape:
        raise SemanticEvalError("audio and semantic score shapes differ")
    return AUDIO_WEIGHT * audio + SEMANTIC_WEIGHT * semantic


def artist_diverse_top(
    track_ids: Sequence[int],
    scores: np.ndarray,
    artist_by_track: Mapping[int, int],
    *,
    count: int = RESULTS_PER_METHOD,
) -> Tuple[int, ...]:
    ids = tuple(int(track_id) for track_id in track_ids)
    values = np.asarray(scores, dtype=np.float64)
    if (
        count != RESULTS_PER_METHOD
        or values.shape != (len(ids),)
        or len(set(ids)) != len(ids)
        or not np.all(np.isfinite(values))
        or not set(ids).issubset(artist_by_track)
    ):
        raise SemanticEvalError("artist-diverse selection input is invalid")
    selected = []
    seen_artists = set()
    for index in np.argsort(-values, kind="stable"):
        track_id = ids[int(index)]
        artist_id = int(artist_by_track[track_id])
        if artist_id in seen_artists:
            continue
        selected.append(track_id)
        seen_artists.add(artist_id)
        if len(selected) == count:
            return tuple(selected)
    raise SemanticEvalError("candidate pool has fewer than five distinct artists")


def prioritize_source_seeds(
    source_seeds: Sequence[Mapping[str, object]],
    audio_control_rankings: Mapping[int, Sequence[int]],
    semantic_rankings: Mapping[int, Sequence[int]],
) -> Tuple[Mapping[str, object], ...]:
    seed_ids = tuple(int(seed["seed_track_id"]) for seed in source_seeds)
    if (
        len(seed_ids) != 20
        or len(set(seed_ids)) != 20
        or set(audio_control_rankings) != set(seed_ids)
        or set(semantic_rankings) != set(seed_ids)
    ):
        raise SemanticEvalError("semantic priority seed/ranking set drift")
    core_positions = {scene: position for position, scene in enumerate(CORE_SCENES)}

    def priority(seed: Mapping[str, object]) -> Tuple[int, int, int, int]:
        track_id = int(seed["seed_track_id"])
        control = {int(value) for value in audio_control_rankings[track_id]}
        challenger = {int(value) for value in semantic_rankings[track_id]}
        if len(control) != RESULTS_PER_METHOD or len(challenger) != RESULTS_PER_METHOD:
            raise SemanticEvalError("semantic priority ranking cardinality drift")
        scene = str(seed["scene"])
        return (
            0 if scene in core_positions else 1,
            len(control & challenger),
            core_positions.get(scene, len(CORE_SCENES)),
            track_id,
        )

    return tuple(sorted(source_seeds, key=priority))


def repeated_section_excerpt(
    reader: FullTrackStoreReader,
    track_id: int,
) -> Mapping[str, object]:
    track = reader.read_track(int(track_id))
    if (
        not len(track.repeated_indices)
        or not len(track.window_starts)
        or track.decoded_samples <= 0
    ):
        raise SemanticEvalError("track lacks repeated-section playback evidence")
    repeated_index = int(track.repeated_indices[0])
    if not 0 <= repeated_index < len(track.window_starts):
        raise SemanticEvalError("repeated-section playback index drift")
    source_start = float(track.window_starts[repeated_index]) / SOURCE_SAMPLE_RATE
    track_duration = float(track.decoded_samples) / SOURCE_SAMPLE_RATE
    excerpt_duration = min(PLAYBACK_EXCERPT_SECONDS, track_duration)
    context = max(0.0, (excerpt_duration - SOURCE_WINDOW_SECONDS) / 2.0)
    start = min(max(0.0, source_start - context), track_duration - excerpt_duration)
    end = start + excerpt_duration

    def json_seconds(value: float) -> float | int:
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded

    return {
        "kind": "strongest_nonlocal_recurrence",
        "start_seconds": json_seconds(start),
        "end_seconds": json_seconds(end),
        "source_window_start_seconds": json_seconds(source_start),
        "source_window_seconds": SOURCE_WINDOW_SECONDS,
    }


def _valid_playback_excerpt(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "kind",
        "start_seconds",
        "end_seconds",
        "source_window_start_seconds",
        "source_window_seconds",
    }:
        return False
    numeric = (
        value["start_seconds"],
        value["end_seconds"],
        value["source_window_start_seconds"],
        value["source_window_seconds"],
    )
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in numeric
    ):
        return False
    start = float(value["start_seconds"])
    end = float(value["end_seconds"])
    source_start = float(value["source_window_start_seconds"])
    return (
        value["kind"] == "strongest_nonlocal_recurrence"
        and start >= 0.0
        and 0.0 < end - start <= PLAYBACK_EXCERPT_SECONDS
        and start <= source_start < end
        and float(value["source_window_seconds"]) == SOURCE_WINDOW_SECONDS
    )


def _audio_scores(
    reader: FullTrackStoreReader,
    query_track_id: int,
    candidate_track_ids: Sequence[int],
    global_scores: np.ndarray,
    *,
    section_budget: int,
) -> np.ndarray:
    query = reader.read_track(query_track_id)
    candidates = [reader.read_track(int(track_id)) for track_id in candidate_track_ids]
    uniform = batch_fixed_budget_maxsim(
        freeze_fixed_budget(query.window_embeddings, section_budget),
        np.stack(
            [
                freeze_fixed_budget(candidate.window_embeddings, section_budget)
                for candidate in candidates
            ]
        ),
    )
    repeated = batch_fixed_budget_maxsim(
        freeze_ranked_section_budget(query.repeated_sections, section_budget),
        np.stack(
            [
                freeze_ranked_section_budget(
                    candidate.repeated_sections, section_budget
                )
                for candidate in candidates
            ]
        ),
    )
    salient = batch_fixed_budget_maxsim(
        freeze_ranked_section_budget(query.salient_sections, section_budget),
        np.stack(
            [
                freeze_ranked_section_budget(candidate.salient_sections, section_budget)
                for candidate in candidates
            ]
        ),
    )
    return (
        HYBRID_WEIGHTS["global_cosine"] * np.asarray(global_scores)
        + HYBRID_WEIGHTS["uniform_window_maxsim"] * uniform
        + HYBRID_WEIGHTS["section_maxsim"] * 0.5 * (repeated + salient)
    )


def rank_study_methods(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    predictor: CalibratedSemanticPredictor,
    seed_track_ids: Sequence[int],
    excluded_track_ids_by_seed: Mapping[int, Sequence[int]],
    config: SemanticEvalConfig,
) -> Tuple[
    Mapping[int, Tuple[int, ...]],
    Mapping[int, Tuple[int, ...]],
    Mapping[str, object],
]:
    config.validate()
    if context.source_fingerprint != EXPECTED_SOURCE_FINGERPRINT:
        raise SemanticEvalError("Jamendo source fingerprint drift")
    fold = next(item for item in context.folds if item.index == config.fold_index)
    partition = [
        track
        for track in context.tracks
        if fold.track_parts.get(int(track.track_id)) == config.part
    ]
    if len(partition) != EXPECTED_TEST_TRACKS:
        raise SemanticEvalError("fold-0 test partition size drift")
    by_id = {int(track.track_id): track for track in partition}
    seeds = tuple(int(track_id) for track_id in seed_track_ids)
    if len(seeds) != 20 or len(set(seeds)) != len(seeds) or not set(seeds).issubset(by_id):
        raise SemanticEvalError("semantic study seed set is invalid")
    if set(excluded_track_ids_by_seed) != set(seeds):
        raise SemanticEvalError("published-result exclusion seed set is invalid")

    store_rows = {int(track_id): row for row, track_id in enumerate(reader.track_ids)}
    if not set(by_id).issubset(store_rows):
        raise SemanticEvalError("sealed store does not cover the study partition")
    partition_rows = np.asarray(
        [store_rows[int(track.track_id)] for track in partition], dtype=np.int64
    )
    globals_matrix = normalize_rows(
        np.asarray(reader.global_embeddings[partition_rows], dtype=np.float32)
    )
    domain = feature_domain_diagnostics(predictor, globals_matrix)
    if domain["rows"] != EXPECTED_TEST_TRACKS or domain["passed"] is not True:
        raise SemanticEvalError("semantic predictor rejected the study feature domain")
    profiles = predictor.semantic_profiles(globals_matrix)
    id_to_position = {
        int(track.track_id): position for position, track in enumerate(partition)
    }
    artist_by_track = {
        int(track.track_id): int(track.artist_id) for track in partition
    }

    audio_rankings: Dict[int, Tuple[int, ...]] = {}
    semantic_rankings: Dict[int, Tuple[int, ...]] = {}
    for seed_track_id in seeds:
        query_position = id_to_position[seed_track_id]
        seed = by_id[seed_track_id]
        excluded = {
            int(track_id) for track_id in excluded_track_ids_by_seed[seed_track_id]
        }
        if seed_track_id in excluded or not excluded.issubset(by_id):
            raise SemanticEvalError("published-result exclusion set is invalid")
        eligible = np.asarray(
            [
                index
                for index, candidate in enumerate(partition)
                if int(candidate.track_id) != seed_track_id
                and int(candidate.artist_id) != int(seed.artist_id)
                and int(candidate.track_id) not in excluded
            ],
            dtype=np.int64,
        )
        initial_scores = globals_matrix[eligible] @ globals_matrix[query_position]
        pool = eligible[
            np.lexsort((eligible, -initial_scores))[: config.candidate_pool]
        ]
        pool_ids = tuple(int(partition[index].track_id) for index in pool)
        audio = _audio_scores(
            reader,
            seed_track_id,
            pool_ids,
            globals_matrix[pool] @ globals_matrix[query_position],
            section_budget=config.section_budget,
        )
        semantic = profiles[pool] @ profiles[query_position]
        audio_rankings[seed_track_id] = artist_diverse_top(
            pool_ids,
            percentile_scores(audio),
            artist_by_track,
        )
        semantic_rankings[seed_track_id] = artist_diverse_top(
            pool_ids,
            semantic_blend_scores(
                audio, semantic, semantic_weight=config.semantic_weight
            ),
            artist_by_track,
        )
    return audio_rankings, semantic_rankings, domain


def _validate_source_v2_pack(
    v2_public: Mapping[str, object],
    v2_private: Mapping[str, object],
) -> None:
    validate_v2_documents(v2_public, v2_private)
    if (
        v2_public.get("content_sha256") != EXPECTED_V2_PACK_SHA256
        or v2_public.get("ratings_count_at_freeze") != 0
        or v2_public.get("rankings_state") != "LOCKED_BEFORE_RATINGS"
    ):
        raise SemanticEvalError("source v2 pack is not the trusted zero-rating freeze")


def _method_bindings(
    predictor_metadata: Mapping[str, object],
    predictor_model_path: Path,
    domain: Mapping[str, object],
    store_binding_sha256: str,
) -> Mapping[str, Mapping[str, object]]:
    model_hash = sha256_path(predictor_model_path)
    if (
        predictor_metadata.get("payload_sha256") != EXPECTED_PREDICTOR_PAYLOAD_SHA256
        or predictor_metadata.get("model_npz_sha256") != model_hash
        or model_hash != EXPECTED_PREDICTOR_SHA256
        or domain.get("passed") is not True
    ):
        raise SemanticEvalError("semantic predictor binding drift")
    shared = {
        "store_binding_sha256": store_binding_sha256,
        "candidate_pool": CANDIDATE_POOL,
        "section_budget": SECTION_BUDGET,
        "max_results_per_artist": MAX_RESULTS_PER_ARTIST,
        "published_v2_results_excluded": True,
        "source_seed_pack_sha256": EXPECTED_V2_PACK_SHA256,
        "test_labels_used_for_ranking": False,
        "language_metadata_used_for_ranking": False,
        "promoted": False,
    }
    return {
        "fulltrack_audio_control_v1": {
            **shared,
            "method": "fulltrack_audio_control_v1",
            "audio_weight": 1.0,
            "semantic_weight": 0.0,
            "semantic_profile_used_for_ranking": False,
        },
        "semantic_fulltrack_v1": {
            **shared,
            "method": "semantic_fulltrack_v1",
            "predictor_payload_sha256": EXPECTED_PREDICTOR_PAYLOAD_SHA256,
            "predictor_model_sha256": model_hash,
            "audio_weight": AUDIO_WEIGHT,
            "semantic_weight": SEMANTIC_WEIGHT,
            "semantic_profile_used_for_ranking": True,
            "domain_diagnostics": dict(domain),
        },
    }


def _valid_opaque_id(value: object, prefix: str) -> bool:
    marker = f"{prefix}-"
    return (
        isinstance(value, str)
        and value.startswith(marker)
        and len(value) == len(marker) + 24
        and not (set(value[len(marker) :]) - _HEX64)
    )


def _validate_method_binding(
    value: object,
    *,
    expected_method: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SemanticEvalError("private semantic method binding is missing")
    expected_keys = set(_SHARED_METHOD_KEYS)
    if expected_method == METHODS[1]:
        expected_keys.update(
            {
                "predictor_payload_sha256",
                "predictor_model_sha256",
                "domain_diagnostics",
            }
        )
    if set(value) != expected_keys:
        raise SemanticEvalError("private semantic method binding schema drift")
    if (
        value.get("method") != expected_method
        or value.get("store_binding_sha256") != EXPECTED_STORE_BINDING_SHA256
        or value.get("candidate_pool") != CANDIDATE_POOL
        or value.get("section_budget") != SECTION_BUDGET
        or value.get("max_results_per_artist") != MAX_RESULTS_PER_ARTIST
        or value.get("published_v2_results_excluded") is not True
        or value.get("source_seed_pack_sha256") != EXPECTED_V2_PACK_SHA256
        or value.get("test_labels_used_for_ranking") is not False
        or value.get("language_metadata_used_for_ranking") is not False
        or value.get("promoted") is not False
    ):
        raise SemanticEvalError("private semantic method binding policy drift")
    if expected_method == METHODS[0]:
        if (
            value.get("audio_weight") != 1.0
            or value.get("semantic_weight") != 0.0
            or value.get("semantic_profile_used_for_ranking") is not False
        ):
            raise SemanticEvalError("audio control method binding drift")
        return value

    diagnostics = value.get("domain_diagnostics")
    if (
        value.get("predictor_payload_sha256")
        != EXPECTED_PREDICTOR_PAYLOAD_SHA256
        or value.get("predictor_model_sha256") != EXPECTED_PREDICTOR_SHA256
        or value.get("audio_weight") != AUDIO_WEIGHT
        or value.get("semantic_weight") != SEMANTIC_WEIGHT
        or value.get("semantic_profile_used_for_ranking") is not True
        or not isinstance(diagnostics, Mapping)
        or set(diagnostics) != _DOMAIN_DIAGNOSTIC_KEYS
        or diagnostics.get("rows") != EXPECTED_TEST_TRACKS
        or diagnostics.get("dimensions") != 512
        or diagnostics.get("maximum_standardized_coordinate_mean_abs") != 0.25
        or diagnostics.get("minimum_standardized_coordinate_scale_mean") != 0.8
        or diagnostics.get("maximum_standardized_coordinate_scale_mean") != 1.2
        or diagnostics.get("passed") is not True
    ):
        raise SemanticEvalError("semantic challenger method binding drift")
    mean_abs = diagnostics.get("standardized_coordinate_mean_abs")
    scale_mean = diagnostics.get("standardized_coordinate_scale_mean")
    if (
        isinstance(mean_abs, bool)
        or not isinstance(mean_abs, (int, float))
        or not math.isfinite(float(mean_abs))
        or not 0.0 <= float(mean_abs) <= 0.25
        or isinstance(scale_mean, bool)
        or not isinstance(scale_mean, (int, float))
        or not math.isfinite(float(scale_mean))
        or not 0.8 <= float(scale_mean) <= 1.2
    ):
        raise SemanticEvalError("semantic challenger domain evidence drift")
    return value


def build_blinded_documents(
    *,
    source_seeds: Sequence[Mapping[str, object]],
    track_records: Mapping[str, Mapping[str, object]],
    audio_control_rankings: Mapping[int, Sequence[int]],
    semantic_rankings: Mapping[int, Sequence[int]],
    method_bindings: Mapping[str, Mapping[str, object]],
    store_binding: Mapping[str, object],
    source_fingerprint: str,
    blinding_key: bytes,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    if len(blinding_key) != 32:
        raise SemanticEvalError("blinding key must contain exactly 32 bytes")
    if len(source_seeds) != 20:
        raise SemanticEvalError("semantic study requires the existing 20 seed set")
    if tuple(method_bindings) != METHODS:
        raise SemanticEvalError("semantic study method binding order drift")

    public_seeds = []
    private_seeds = []
    all_track_ids = set()
    for priority_rank, source_seed in enumerate(source_seeds, 1):
        seed_track_id = int(source_seed["seed_track_id"])
        rankings = {
            METHODS[0]: tuple(
                int(value) for value in audio_control_rankings[seed_track_id]
            ),
            METHODS[1]: tuple(int(value) for value in semantic_rankings[seed_track_id]),
        }
        seed_id = _opaque_id(blinding_key, "semantic-seed", seed_track_id)
        public_lists = []
        private_lists = []
        result_ids: Dict[int, str] = {}
        for ranking in rankings.values():
            if len(ranking) != RESULTS_PER_METHOD or len(set(ranking)) != len(ranking):
                raise SemanticEvalError("each semantic study list requires five tracks")
            for track_id in ranking:
                result_ids.setdefault(
                    track_id,
                    _opaque_id(
                        blinding_key, "semantic-result", seed_track_id, track_id
                    ),
                )
        for method in METHODS:
            ranking = rankings[method]
            list_id = _opaque_id(
                blinding_key, "semantic-list", seed_track_id, method
            )
            commitment_payload = {
                "pack_kind": PACK_KIND,
                "seed_id": seed_id,
                "list_id": list_id,
                "method_binding": method_bindings[method],
                "ranking_track_ids": list(ranking),
            }
            commitment = hmac.new(
                blinding_key,
                _canonical_bytes(commitment_payload),
                hashlib.sha256,
            ).hexdigest()
            public_lists.append(
                {
                    "list_id": list_id,
                    "binding_commitment_sha256": commitment,
                    "ranking": [
                        {
                            "position": position,
                            "result_id": result_ids[track_id],
                            "track_id": track_id,
                        }
                        for position, track_id in enumerate(ranking, 1)
                    ],
                }
            )
            private_lists.append(
                {
                    **commitment_payload,
                    "binding_commitment_sha256": commitment,
                }
            )
        public_lists.sort(key=lambda item: item["list_id"])
        private_lists.sort(key=lambda item: item["list_id"])
        all_track_ids.add(seed_track_id)
        all_track_ids.update(result_ids)
        public_seeds.append(
            {
                "seed_id": seed_id,
                "scene": str(source_seed["scene"]),
                "seed_track_id": seed_track_id,
                "tempo_bpm": float(source_seed["tempo_bpm"]),
                "tempo_region": str(source_seed["tempo_region"]),
                "clap_texture_region": int(source_seed["clap_texture_region"]),
                "priority_rank": priority_rank,
                "matched_list_overlap": len(
                    set(rankings[METHODS[0]]) & set(rankings[METHODS[1]])
                ),
                "result_ids": [
                    {"result_id": result_ids[track_id], "track_id": track_id}
                    for track_id in sorted(result_ids)
                ],
                "lists": public_lists,
            }
        )
        private_seeds.append(
            {
                "seed_id": seed_id,
                "seed_track_id": seed_track_id,
                "lists": private_lists,
            }
        )

    if {str(track_id) for track_id in all_track_ids} != set(track_records):
        raise SemanticEvalError("public track records do not exactly cover the study")
    store_binding_sha256 = stable_json_sha256(store_binding)
    private: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PRIVATE_KIND,
        "pack_id": PACK_ID,
        "source_v2_pack_sha256": EXPECTED_V2_PACK_SHA256,
        "source_fingerprint": source_fingerprint,
        "store_binding_sha256": store_binding_sha256,
        "blinding_key_hex": blinding_key.hex(),
        "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
        "methods": list(METHODS),
        "method_bindings": dict(method_bindings),
        "seeds": private_seeds,
        "research_only": True,
        "promotion_allowed": False,
    }
    private["content_sha256"] = _content_sha256(private)

    public: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pack_kind": PACK_KIND,
        "pack_id": PACK_ID,
        "rankings_state": "LOCKED_BEFORE_RATINGS",
        "ratings_count_at_freeze": 0,
        "seed_count": 20,
        "method_count": 2,
        "results_per_method": RESULTS_PER_METHOD,
        "source_v2_pack_sha256": EXPECTED_V2_PACK_SHA256,
        "source_fingerprint": source_fingerprint,
        "store_binding": dict(store_binding),
        "store_binding_sha256": store_binding_sha256,
        "blinding": {
            "opaque_per_seed_list_ids": True,
            "method_identity_public": False,
            "method_order_randomized_per_session": True,
            "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
            "private_unblinding_sha256": private["content_sha256"],
        },
        "audio_delivery": {
            "kind": "Jamendo first-party full-track MP3",
            "host": "prod-1.storage.jamendo.com",
            "repository_contains_audio": False,
            "commercial_preview": False,
        },
        "section_coverage": {
            "global_embedding": "coverage-weighted pooling over all 10-second windows",
            "uniform_window_budget": SECTION_BUDGET,
            "repeated_section_budget": SECTION_BUDGET,
            "salient_section_budget": SECTION_BUDGET,
        },
        "playback_policy": {
            "kind": "strongest_nonlocal_recurrence_excerpt",
            "excerpt_seconds": PLAYBACK_EXCERPT_SECONDS,
            "source_window_seconds": SOURCE_WINDOW_SECONDS,
            "verified_chorus_labels": False,
            "full_track_seeking_allowed": False,
        },
        "seed_order_policy": {
            "randomized": False,
            "ratings_used": False,
            "core_scenes": list(CORE_SCENES),
            "within_group": "ascending matched top-five overlap, then fixed scene order and track ID",
            "edge_scenes_after_core": True,
        },
        "language_policy": {
            "evaluated_here": False,
            "reason": "MTG-Jamendo provides no trustworthy track-language field",
            "production_policy": "Spotify lyrics-language gating remains a separate unchanged layer",
        },
        "tracks": dict(track_records),
        "seeds": public_seeds,
        "research_only": True,
        "promotion_allowed": False,
        "production_recommendation_changed": False,
        "notice": (
            "Exploratory blinded comparison of matched full-track retrieval methods "
            "using repeated-section listening excerpts. One listener cannot promote "
            "a model."
        ),
    }
    public["content_sha256"] = _content_sha256(public)
    validate_blinded_documents(public, private, require_frozen_artifacts=False)
    return public, private


def validate_blinded_documents(
    public: Mapping[str, object],
    private: Mapping[str, object],
    *,
    require_frozen_artifacts: bool = True,
) -> None:
    if (
        set(public) != _PUBLIC_DOCUMENT_KEYS
        or public.get("schema_version") != SCHEMA_VERSION
        or public.get("pack_kind") != PACK_KIND
        or public.get("pack_id") != PACK_ID
        or public.get("content_sha256") != _content_sha256(public)
        or public.get("ratings_count_at_freeze") != 0
        or public.get("rankings_state") != "LOCKED_BEFORE_RATINGS"
        or public.get("method_count") != 2
        or public.get("seed_count") != 20
        or public.get("results_per_method") != RESULTS_PER_METHOD
        or public.get("research_only") is not True
        or public.get("promotion_allowed") is not False
        or public.get("production_recommendation_changed") is not False
        or public.get("notice")
        != (
            "Exploratory blinded comparison of matched full-track retrieval methods "
            "using repeated-section listening excerpts. One listener cannot promote "
            "a model."
        )
    ):
        raise SemanticEvalError("public semantic study document drift")
    if (
        set(private) != _PRIVATE_DOCUMENT_KEYS
        or private.get("schema_version") != SCHEMA_VERSION
        or private.get("artifact_kind") != PRIVATE_KIND
        or private.get("pack_id") != PACK_ID
        or private.get("content_sha256") != _content_sha256(private)
        or private.get("methods") != list(METHODS)
        or private.get("research_only") is not True
        or private.get("promotion_allowed") is not False
    ):
        raise SemanticEvalError("private semantic study document drift")
    if require_frozen_artifacts and (
        public.get("content_sha256") != EXPECTED_PUBLIC_PACK_SHA256
        or private.get("content_sha256") != EXPECTED_PRIVATE_UNBLINDING_SHA256
    ):
        raise SemanticEvalError("frozen semantic study artifact drift")
    store_binding = public.get("store_binding")
    if (
        public.get("source_v2_pack_sha256") != EXPECTED_V2_PACK_SHA256
        or private.get("source_v2_pack_sha256") != EXPECTED_V2_PACK_SHA256
        or public.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT
        or private.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT
        or not isinstance(store_binding, Mapping)
        or store_binding.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT
        or stable_json_sha256(store_binding) != EXPECTED_STORE_BINDING_SHA256
        or public.get("store_binding_sha256") != EXPECTED_STORE_BINDING_SHA256
        or private.get("store_binding_sha256") != EXPECTED_STORE_BINDING_SHA256
    ):
        raise SemanticEvalError("semantic study source/store provenance drift")
    if public.get("audio_delivery") != {
        "kind": "Jamendo first-party full-track MP3",
        "host": "prod-1.storage.jamendo.com",
        "repository_contains_audio": False,
        "commercial_preview": False,
    }:
        raise SemanticEvalError("semantic study audio delivery policy drift")
    if public.get("section_coverage") != {
        "global_embedding": "coverage-weighted pooling over all 10-second windows",
        "uniform_window_budget": SECTION_BUDGET,
        "repeated_section_budget": SECTION_BUDGET,
        "salient_section_budget": SECTION_BUDGET,
    }:
        raise SemanticEvalError("semantic study section policy drift")
    if public.get("playback_policy") != {
        "kind": "strongest_nonlocal_recurrence_excerpt",
        "excerpt_seconds": PLAYBACK_EXCERPT_SECONDS,
        "source_window_seconds": SOURCE_WINDOW_SECONDS,
        "verified_chorus_labels": False,
        "full_track_seeking_allowed": False,
    }:
        raise SemanticEvalError("semantic study playback policy drift")
    if public.get("seed_order_policy") != {
        "randomized": False,
        "ratings_used": False,
        "core_scenes": list(CORE_SCENES),
        "within_group": "ascending matched top-five overlap, then fixed scene order and track ID",
        "edge_scenes_after_core": True,
    }:
        raise SemanticEvalError("semantic study seed order policy drift")
    if public.get("language_policy") != {
        "evaluated_here": False,
        "reason": "MTG-Jamendo provides no trustworthy track-language field",
        "production_policy": "Spotify lyrics-language gating remains a separate unchanged layer",
    }:
        raise SemanticEvalError("semantic study language policy drift")
    blinding = public.get("blinding")
    if (
        not isinstance(blinding, Mapping)
        or set(blinding) != _BLINDING_KEYS
        or blinding.get("opaque_per_seed_list_ids") is not True
        or blinding.get("method_identity_public") is not False
        or blinding.get("method_order_randomized_per_session") is not True
    ):
        raise SemanticEvalError("semantic study blinding declaration drift")

    method_bindings = private.get("method_bindings")
    if not isinstance(method_bindings, Mapping) or set(method_bindings) != set(METHODS):
        raise SemanticEvalError("private semantic method map drift")
    validated_methods = {
        method: _validate_method_binding(
            method_bindings[method],
            expected_method=method,
        )
        for method in METHODS
    }
    try:
        key = bytes.fromhex(str(private["blinding_key_hex"]))
    except ValueError as exc:
        raise SemanticEvalError("private blinding key is malformed") from exc
    if (
        len(key) != 32
        or hashlib.sha256(key).hexdigest() != private.get("blinding_key_sha256")
        or blinding.get("blinding_key_sha256") != private.get("blinding_key_sha256")
        or blinding.get("private_unblinding_sha256") != private.get("content_sha256")
    ):
        raise SemanticEvalError("semantic study blinding binding drift")

    public_seed_items = public.get("seeds", [])
    private_seed_items = private.get("seeds", [])
    tracks = public.get("tracks")
    if (
        not isinstance(public_seed_items, list)
        or not isinstance(private_seed_items, list)
        or len(public_seed_items) != 20
        or len(private_seed_items) != 20
        or not isinstance(tracks, Mapping)
        or any(
            not isinstance(seed, Mapping) or set(seed) != _PUBLIC_SEED_KEYS
            for seed in public_seed_items
        )
        or any(
            not isinstance(seed, Mapping) or set(seed) != _PRIVATE_SEED_KEYS
            for seed in private_seed_items
        )
    ):
        raise SemanticEvalError("semantic study seed count drift")
    private_seeds = {seed["seed_id"]: seed for seed in private_seed_items}
    if (
        len(private_seeds) != 20
        or len({seed["seed_id"] for seed in public_seed_items}) != 20
        or [
            (seed["seed_id"], seed["seed_track_id"]) for seed in public_seed_items
        ]
        != [
            (seed["seed_id"], seed["seed_track_id"]) for seed in private_seed_items
        ]
    ):
        raise SemanticEvalError("public/private semantic seed set drift")

    all_track_ids: set[int] = set()
    seen_seed_ids: set[str] = set()
    seen_seed_tracks: set[int] = set()
    seen_list_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    core_positions = {scene: position for position, scene in enumerate(CORE_SCENES)}
    priority_keys = []
    for priority_rank, public_seed in enumerate(public_seed_items, 1):
        seed_id = public_seed["seed_id"]
        seed_track_id = public_seed["seed_track_id"]
        private_seed = private_seeds.get(seed_id)
        if (
            not _valid_opaque_id(seed_id, "semantic-seed")
            or seed_id in seen_seed_ids
            or isinstance(seed_track_id, bool)
            or not isinstance(seed_track_id, int)
            or seed_track_id <= 0
            or seed_track_id in seen_seed_tracks
            or private_seed is None
            or private_seed["seed_track_id"] != seed_track_id
            or not isinstance(public_seed["scene"], str)
            or not public_seed["scene"]
            or isinstance(public_seed["tempo_bpm"], bool)
            or not isinstance(public_seed["tempo_bpm"], (int, float))
            or not math.isfinite(float(public_seed["tempo_bpm"]))
            or not 0.0 < float(public_seed["tempo_bpm"]) <= 400.0
            or not isinstance(public_seed["tempo_region"], str)
            or isinstance(public_seed["clap_texture_region"], bool)
            or not isinstance(public_seed["clap_texture_region"], int)
            or public_seed["clap_texture_region"] not in range(5)
            or public_seed["priority_rank"] != priority_rank
            or isinstance(public_seed["matched_list_overlap"], bool)
            or not isinstance(public_seed["matched_list_overlap"], int)
            or public_seed["matched_list_overlap"] not in range(RESULTS_PER_METHOD + 1)
        ):
            raise SemanticEvalError("public/private semantic seed binding drift")
        seen_seed_ids.add(seed_id)
        seen_seed_tracks.add(seed_track_id)

        result_rows = public_seed["result_ids"]
        if (
            not isinstance(result_rows, list)
            or not result_rows
            or any(
                not isinstance(row, Mapping) or set(row) != _RESULT_ID_KEYS
                for row in result_rows
            )
        ):
            raise SemanticEvalError("semantic result identity map schema drift")
        result_ids_by_track: Dict[int, str] = {}
        for row in result_rows:
            track_id = row["track_id"]
            result_id = row["result_id"]
            if (
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id <= 0
                or track_id in result_ids_by_track
                or not _valid_opaque_id(result_id, "semantic-result")
                or result_id in seen_result_ids
            ):
                raise SemanticEvalError("semantic result identity map drift")
            result_ids_by_track[track_id] = result_id
            seen_result_ids.add(result_id)
        if [row["track_id"] for row in result_rows] != sorted(result_ids_by_track):
            raise SemanticEvalError("semantic result identity map ordering drift")

        public_lists = public_seed["lists"]
        private_list_items = private_seed["lists"]
        if (
            not isinstance(public_lists, list)
            or len(public_lists) != len(METHODS)
            or not isinstance(private_list_items, list)
            or len(private_list_items) != len(METHODS)
            or any(
                not isinstance(item, Mapping) or set(item) != _PUBLIC_LIST_KEYS
                for item in public_lists
            )
            or any(
                not isinstance(item, Mapping) or set(item) != _PRIVATE_LIST_KEYS
                for item in private_list_items
            )
        ):
            raise SemanticEvalError("semantic seed must expose exactly two lists")
        public_list_ids = [item["list_id"] for item in public_lists]
        private_list_ids = [item["list_id"] for item in private_list_items]
        private_lists = {item["list_id"]: item for item in private_list_items}
        if (
            public_list_ids != sorted(public_list_ids)
            or private_list_ids != sorted(private_list_ids)
            or public_list_ids != private_list_ids
            or len(private_lists) != len(METHODS)
        ):
            raise SemanticEvalError("semantic list identity map drift")

        all_track_ids.add(seed_track_id)
        method_names = set()
        ranked_track_union: set[int] = set()
        rankings_by_method: Dict[str, set[int]] = {}
        for public_list in public_lists:
            list_id = public_list["list_id"]
            if (
                not _valid_opaque_id(list_id, "semantic-list")
                or list_id in seen_list_ids
            ):
                raise SemanticEvalError("public semantic list identity drift")
            seen_list_ids.add(list_id)
            private_list = private_lists.get(public_list["list_id"])
            if private_list is None:
                raise SemanticEvalError("public semantic list lacks private binding")
            method_binding = private_list["method_binding"]
            method = (
                method_binding.get("method")
                if isinstance(method_binding, Mapping)
                else None
            )
            if (
                method not in METHODS
                or method in method_names
                or method_binding != validated_methods[method]
            ):
                raise SemanticEvalError("semantic list method map drift")
            method_names.add(method)
            expected = hmac.new(
                key,
                _canonical_bytes(
                    {
                        name: private_list[name]
                        for name in (
                            "pack_kind",
                            "seed_id",
                            "list_id",
                            "method_binding",
                            "ranking_track_ids",
                        )
                    }
                ),
                hashlib.sha256,
            ).hexdigest()
            ranking = public_list["ranking"]
            if (
                not isinstance(ranking, list)
                or len(ranking) != RESULTS_PER_METHOD
                or any(
                    not isinstance(row, Mapping) or set(row) != _RANKING_ROW_KEYS
                    for row in ranking
                )
            ):
                raise SemanticEvalError("semantic ranking schema drift")
            public_ids = []
            artist_ids = set()
            for position, row in enumerate(ranking, 1):
                track_id = row["track_id"]
                track = tracks.get(str(track_id))
                source_identity = (
                    track.get("source_identity")
                    if isinstance(track, Mapping)
                    else None
                )
                if (
                    row["position"] != position
                    or isinstance(track_id, bool)
                    or not isinstance(track_id, int)
                    or track_id == seed_track_id
                    or result_ids_by_track.get(track_id) != row["result_id"]
                    or not isinstance(track, Mapping)
                    or track.get("track_id") != track_id
                    or not isinstance(source_identity, Mapping)
                    or source_identity.get("fold") != 0
                    or source_identity.get("fold_part") != "test"
                    or isinstance(source_identity.get("artist_id"), bool)
                    or not isinstance(source_identity.get("artist_id"), int)
                ):
                    raise SemanticEvalError("semantic ranking result identity drift")
                public_ids.append(track_id)
                artist_ids.add(source_identity["artist_id"])
            if (
                expected != private_list["binding_commitment_sha256"]
                or expected != public_list["binding_commitment_sha256"]
                or public_ids != private_list["ranking_track_ids"]
                or len(public_ids) != RESULTS_PER_METHOD
                or len(set(public_ids)) != RESULTS_PER_METHOD
                or len(artist_ids) != RESULTS_PER_METHOD
                or {
                    name: private_list[name]
                    for name in (
                        "pack_kind",
                        "seed_id",
                        "list_id",
                        "method_binding",
                        "ranking_track_ids",
                    )
                }
                != {
                    "pack_kind": PACK_KIND,
                    "seed_id": seed_id,
                    "list_id": list_id,
                    "method_binding": method_binding,
                    "ranking_track_ids": public_ids,
                }
            ):
                raise SemanticEvalError("semantic list commitment or ranking drift")
            all_track_ids.update(public_ids)
            ranked_track_union.update(public_ids)
            rankings_by_method[method] = set(public_ids)
        if method_names != set(METHODS):
            raise SemanticEvalError("semantic seed does not bind both study methods")
        if ranked_track_union != set(result_ids_by_track):
            raise SemanticEvalError("semantic result identity map differs from rankings")
        overlap = len(rankings_by_method[METHODS[0]] & rankings_by_method[METHODS[1]])
        if public_seed["matched_list_overlap"] != overlap:
            raise SemanticEvalError("semantic seed priority evidence drift")
        scene = public_seed["scene"]
        priority_keys.append(
            (
                0 if scene in core_positions else 1,
                overlap,
                core_positions.get(scene, len(CORE_SCENES)),
                seed_track_id,
            )
        )
    if priority_keys != sorted(priority_keys):
        raise SemanticEvalError("semantic seed priority ordering drift")
    if set(tracks) != {str(track_id) for track_id in all_track_ids}:
        raise SemanticEvalError("semantic study track coverage drift")
    for track_id, track in tracks.items():
        if (
            not isinstance(track, Mapping)
            or str(track.get("track_id")) != track_id
            or not _valid_playback_excerpt(track.get("playback_excerpt"))
        ):
            raise SemanticEvalError("semantic study track identity drift")
    public_text = json.dumps(public, sort_keys=True)
    if any(method in public_text for method in METHODS):
        raise SemanticEvalError("public semantic study reveals method identity")
    if not all(
        len(str(value)) == 64 and set(str(value)) <= _HEX64
        for value in (
            public["content_sha256"],
            private["content_sha256"],
            private["blinding_key_sha256"],
        )
    ):
        raise SemanticEvalError("semantic study hash encoding drift")


def build_production_semantic_eval(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    store_root: Path,
    predictor_model: Path,
    predictor_metadata: Path,
    v2_public_path: Path,
    v2_private_path: Path,
    public_output: Path,
    private_output: Path,
    blinding_key_path: Path,
    create_blinding_key: bool,
    verify_audio: bool,
    audio_workers: int,
    config: SemanticEvalConfig,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    config.validate()
    v2_public = json.loads(Path(v2_public_path).read_text(encoding="utf-8"))
    v2_private = json.loads(Path(v2_private_path).read_text(encoding="utf-8"))
    _validate_source_v2_pack(v2_public, v2_private)
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    if context.evidence_scope != EVIDENCE_SCOPE:
        raise SemanticEvalError("source evidence scope is not full-track Jamendo")
    predictor_meta = json.loads(Path(predictor_metadata).read_text(encoding="utf-8"))
    predictor = load_predictor(predictor_model, predictor_metadata)
    source_seeds = tuple(v2_public["seeds"])
    seed_ids = tuple(int(seed["seed_track_id"]) for seed in source_seeds)
    excluded_by_seed = {
        int(seed["seed_track_id"]): tuple(
            sorted(
                {
                    int(row["track_id"])
                    for candidate in seed["lists"]
                    for row in candidate["ranking"]
                }
            )
        )
        for seed in source_seeds
    }
    tracks_by_id = {int(track.track_id): track for track in context.tracks}

    with FullTrackStoreReader(
        store_root, expected_source_fingerprint=context.source_fingerprint
    ) as reader:
        if reader.binding.track_count != EXPECTED_STORE_TRACKS:
            raise SemanticEvalError("sealed store track count drift")
        store_binding = dict(reader.binding.as_dict())
        store_binding["sealed_manifest_sha256"] = stable_json_sha256(reader.manifest)
        store_rows = {
            int(track_id): row for row, track_id in enumerate(reader.track_ids)
        }
        audio_control, semantic, domain = rank_study_methods(
            context, reader, predictor, seed_ids, excluded_by_seed, config
        )
        source_seeds = prioritize_source_seeds(
            source_seeds,
            audio_control,
            semantic,
        )
        method_bindings = _method_bindings(
            predictor_meta,
            predictor_model,
            domain,
            stable_json_sha256(store_binding),
        )
        all_track_ids = set(seed_ids)
        for seed_id in seed_ids:
            all_track_ids.update(audio_control[seed_id])
            all_track_ids.update(semantic[seed_id])
        if not all_track_ids.issubset(tracks_by_id) or not all_track_ids.issubset(
            store_rows
        ):
            raise SemanticEvalError("semantic study track metadata/store coverage drift")

        existing_records = dict(v2_public["tracks"])
        audio_evidence = {
            record["audio"]["url"]: record["audio"]["verification"]
            for record in existing_records.values()
        }
        missing_urls = [
            lawful_stream_url(track_id)
            for track_id in sorted(all_track_ids)
            if lawful_stream_url(track_id) not in audio_evidence
        ]
        if missing_urls and not verify_audio:
            raise SemanticEvalError(
                "semantic study generation requires public-audio verification"
            )
        if missing_urls:
            audio_evidence.update(
                verify_public_audio_urls(missing_urls, workers=audio_workers)
            )
        track_records = {}
        for track_id in sorted(all_track_ids):
            existing = existing_records.get(str(track_id))
            if existing is not None:
                record = dict(existing)
            else:
                record = _track_record(
                    tracks_by_id[track_id],
                    fold_index=config.fold_index,
                    fold_part=config.part,
                    store_row=store_rows[track_id],
                    audio_verification=audio_evidence[lawful_stream_url(track_id)],
                )
            record["playback_excerpt"] = repeated_section_excerpt(reader, track_id)
            track_records[str(track_id)] = record

    key = _read_blinding_key(blinding_key_path, create_blinding_key)
    public, private = build_blinded_documents(
        source_seeds=source_seeds,
        track_records=track_records,
        audio_control_rankings=audio_control,
        semantic_rankings=semantic,
        method_bindings=method_bindings,
        store_binding=store_binding,
        source_fingerprint=context.source_fingerprint,
        blinding_key=key,
    )
    _write_json(public_output, public, private=False)
    _write_json(private_output, private, private=True)
    return public, private


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the exploratory semantic listening study."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "store_root",
        "predictor_model",
        "predictor_metadata",
        "v2_public",
        "v2_private",
        "public_output",
        "private_output",
        "blinding_key",
    ):
        build.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    build.add_argument("--create-blinding-key", action="store_true")
    build.add_argument("--verify-public-audio", action="store_true")
    build.add_argument("--audio-workers", type=int, default=8)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--public", type=Path, required=True)
    verify.add_argument("--private", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        public = json.loads(args.public.read_text(encoding="utf-8"))
        private = json.loads(args.private.read_text(encoding="utf-8"))
        validate_blinded_documents(public, private)
        print(json.dumps({"status": "ok", "content_sha256": public["content_sha256"]}))
        return 0
    build_production_semantic_eval(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        store_root=args.store_root,
        predictor_model=args.predictor_model,
        predictor_metadata=args.predictor_metadata,
        v2_public_path=args.v2_public,
        v2_private_path=args.v2_private,
        public_output=args.public_output,
        private_output=args.private_output,
        blinding_key_path=args.blinding_key,
        create_blinding_key=args.create_blinding_key,
        verify_audio=args.verify_public_audio,
        audio_workers=args.audio_workers,
        config=SemanticEvalConfig(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
