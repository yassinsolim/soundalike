"""Build the isolated blinded audio-vs-pacing repeated-excerpt study.

This module deliberately does not alter :mod:`semantic_eval`.  It consumes the
sealed semantic-v2 study as source provenance and a private, pre-extracted
29-dimensional repeated-section vibe cache.  Listener ratings are never inputs
to retrieval, reranking, seed priority, or artifact construction.
"""
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
    batch_fixed_budget_maxsim,
    freeze_fixed_budget,
    freeze_ranked_section_budget,
)
from .fulltrack_extract import normalize_rows
from .fulltrack_pilot import (
    _canonical_bytes,
    _content_sha256,
    _read_blinding_key,
    _track_record,
    _write_json,
    lawful_stream_url,
    verify_public_audio_urls,
)
from .fulltrack_store import FullTrackStoreReader, stable_json_sha256
from .jamendo_fulltrack import EVIDENCE_SCOPE, JamendoContext, load_jamendo_context
from .semantic_predictor import CalibratedSemanticPredictor, load_predictor


SCHEMA_VERSION = 3
PACK_KIND = "blinded_repeated_excerpt_comparison_v3"
PRIVATE_KIND = "blinded_repeated_excerpt_comparison_v3_private_unblinding"
PACK_ID = "pacing-v3-blind-20"
METHODS = ("fulltrack_audio_study_v2", "pacing_tone_study_v3")
RESULTS_PER_METHOD = 5
CANDIDATE_POOL = 200
SECTION_BUDGET = 32
MAX_RESULTS_PER_ARTIST = 1
VIBE_DIMENSIONS = 29
VIBE_CACHE_TRACKS = 3478
PLAYBACK_EXCERPT_SECONDS = 20
SOURCE_WINDOW_SECONDS = 10

ACOUSTIC_WEIGHTS = {
    "global_cosine": 0.30,
    "uniform_window_maxsim": 0.25,
    "repeated_section_maxsim": 0.25,
    "salient_section_maxsim": 0.20,
}
RERANK_WEIGHTS = {
    "acoustic": 0.75,
    "pacing": 0.10,
    "tone": 0.05,
    "dynamics": 0.05,
    "instrument": 0.03,
    "mood_theme": 0.01,
    "genre": 0.01,
}
PACING_WEIGHTS = {"tempo": 0.60, "onset_rate": 0.40}
TEMPO_WEIGHTS = {"direct": 0.70, "octave_folded": 0.30}
TEMPO_SCALE = 0.25
ONSET_RATIO_SCALE = math.log(1.35)

EXPECTED_SOURCE_FINGERPRINT = (
    "060f43ed0fa12e5a583e26a7728be14a5334c7daffebe2289f08875e9ec0c709"
)
EXPECTED_STORE_BINDING_SHA256 = (
    "66baa07c058d842d5a5a7f068a3ea80070d5c43a4818a7a36f0192cb868de98a"
)
EXPECTED_SEMANTIC_V2_PACK_SHA256 = (
    "939b639abb6d6c6b2c7ba20ae570ff7ae9d06ee67254c219d6e5f61975403347"
)
EXPECTED_SEMANTIC_V2_PACK_FILE_SHA256 = (
    "f07bf814eab2a363aa9fbec5acd946e57cfad3d3c3eef6dea4027a190d0e13b3"
)
EXPECTED_FULLTRACK_V2_PACK_SHA256 = (
    "1980da60810959e7cdd24f39bd7142c8e34c76dab633c705976b85e49b297023"
)
EXPECTED_FULLTRACK_V2_PACK_FILE_SHA256 = (
    "d23d66768f15fd5e37e01ad2a8905d181b4ff278c85674386edcd7dc50b267d3"
)
EXPECTED_PROBE_REPORT_FILE_SHA256 = (
    "1117240842a10824d296770f3c64631ea716d0096aaa966112b85e9928a6da89"
)
EXPECTED_VIBE_CACHE_FILE_SHA256 = (
    "e2cad7bdc7ffd5a35b09c2147d7d9148857d5d45e6e6590170fafd6e4df65b63"
)
EXPECTED_PUBLIC_PACK_SHA256 = (
    "6d6dd1c03412b057e14d52d29ee775e5a4c62eea63c76f7d19c46f60f1942a5c"
)
EXPECTED_PRIVATE_UNBLINDING_SHA256 = (
    "d8d2cc5ffb066d24a5990d0bc177dfeecad12dade5f28742c9aff421117ae1e9"
)

_HEX = frozenset("0123456789abcdef")
_TONE_POSITIONS = np.asarray([1, 2, *range(16, 29)], dtype=np.int64)
_DYNAMICS_POSITIONS = np.asarray([4, 5, 6, 7, 8, *range(9, 16)], dtype=np.int64)


class PacingEvalError(RuntimeError):
    """The pacing study is inconsistent, unblinded, or non-reproducible."""


@dataclass(frozen=True)
class PacingEvalConfig:
    fold_index: int = 0
    part: str = "test"
    candidate_pool: int = CANDIDATE_POOL
    section_budget: int = SECTION_BUDGET
    results_per_method: int = RESULTS_PER_METHOD
    max_results_per_artist: int = MAX_RESULTS_PER_ARTIST

    def validate(self) -> None:
        if self.fold_index != 0 or self.part != "test":
            raise PacingEvalError("pacing study must use the frozen fold-0 test partition")
        if self.candidate_pool != CANDIDATE_POOL:
            raise PacingEvalError("pacing study candidate pool is frozen at 200")
        if self.section_budget != SECTION_BUDGET:
            raise PacingEvalError("pacing study section budget is frozen at 32")
        if self.results_per_method != RESULTS_PER_METHOD:
            raise PacingEvalError("pacing study requires five results per method")
        if self.max_results_per_artist != MAX_RESULTS_PER_ARTIST:
            raise PacingEvalError("pacing study permits one result per artist")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_pack(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
    label: str,
) -> Mapping[str, object]:
    source = Path(path)
    if _sha256_path(source) != expected_file_sha256:
        raise PacingEvalError(f"source {label} file drift")
    document = json.loads(source.read_text(encoding="utf-8"))
    if (
        document.get("content_sha256") != expected_content_sha256
        or _content_sha256(document) != expected_content_sha256
    ):
        raise PacingEvalError(f"source {label} content drift")
    return document


def _published_result_exclusions(
    *packs: Mapping[str, object],
) -> Mapping[int, Tuple[int, ...]]:
    excluded: Dict[int, set[int]] = {}
    for pack in packs:
        seeds = pack.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != 20:
            raise PacingEvalError("published study seed coverage drift")
        seen_seeds = set()
        for seed in seeds:
            if not isinstance(seed, Mapping):
                raise PacingEvalError("published study seed is invalid")
            seed_track_id = seed.get("seed_track_id")
            lists = seed.get("lists")
            if (
                not isinstance(seed_track_id, int)
                or isinstance(seed_track_id, bool)
                or seed_track_id in seen_seeds
                or not isinstance(lists, list)
                or not lists
            ):
                raise PacingEvalError("published study seed identity drift")
            seen_seeds.add(seed_track_id)
            track_ids = excluded.setdefault(seed_track_id, set())
            for candidate_list in lists:
                ranking = (
                    candidate_list.get("ranking")
                    if isinstance(candidate_list, Mapping)
                    else None
                )
                if not isinstance(ranking, list) or len(ranking) != RESULTS_PER_METHOD:
                    raise PacingEvalError("published study ranking coverage drift")
                for row in ranking:
                    track_id = row.get("track_id") if isinstance(row, Mapping) else None
                    if not isinstance(track_id, int) or isinstance(track_id, bool):
                        raise PacingEvalError("published study track identity drift")
                    track_ids.add(track_id)
    if len(excluded) != 20:
        raise PacingEvalError("published study seed union drift")
    return {
        seed_track_id: tuple(sorted(track_ids))
        for seed_track_id, track_ids in excluded.items()
    }


def percentile_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.all(np.isfinite(scores)):
        raise PacingEvalError("percentile score input is invalid")
    order = np.argsort(scores, kind="stable")
    result = np.empty(len(scores), dtype=np.float64)
    result[order] = np.linspace(0.0, 1.0, len(scores), dtype=np.float64)
    return result


def artist_diverse_top(
    track_ids: Sequence[int],
    scores: np.ndarray,
    artist_by_track: Mapping[int, int],
) -> Tuple[int, ...]:
    ids = tuple(int(track_id) for track_id in track_ids)
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.shape != (len(ids),)
        or len(set(ids)) != len(ids)
        or not np.all(np.isfinite(values))
        or not set(ids).issubset(artist_by_track)
    ):
        raise PacingEvalError("artist-diverse selection input is invalid")
    selected = []
    seen_artists = set()
    for index in np.argsort(-values, kind="stable"):
        track_id = ids[int(index)]
        artist_id = int(artist_by_track[track_id])
        if artist_id in seen_artists:
            continue
        selected.append(track_id)
        seen_artists.add(artist_id)
        if len(selected) == RESULTS_PER_METHOD:
            return tuple(selected)
    raise PacingEvalError("candidate pool has fewer than five distinct artists")


def acoustic_scores(
    reader: FullTrackStoreReader,
    query_track_id: int,
    candidate_track_ids: Sequence[int],
    global_scores: np.ndarray,
    *,
    section_budget: int = SECTION_BUDGET,
) -> np.ndarray:
    """Return the frozen 30/25/25/20 acoustic score."""
    if section_budget != SECTION_BUDGET:
        raise PacingEvalError("acoustic section budget drift")
    query = reader.read_track(int(query_track_id))
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
                freeze_ranked_section_budget(
                    candidate.salient_sections, section_budget
                )
                for candidate in candidates
            ]
        ),
    )
    return (
        ACOUSTIC_WEIGHTS["global_cosine"] * np.asarray(global_scores)
        + ACOUSTIC_WEIGHTS["uniform_window_maxsim"] * uniform
        + ACOUSTIC_WEIGHTS["repeated_section_maxsim"] * repeated
        + ACOUSTIC_WEIGHTS["salient_section_maxsim"] * salient
    )


def robust_standardize_vibe(values: np.ndarray) -> np.ndarray:
    """Robust-standardize cached vibe features without changing their semantics."""
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != VIBE_DIMENSIONS
        or not np.all(np.isfinite(matrix))
    ):
        raise PacingEvalError("repeated-section vibe matrix is invalid")
    transformed = matrix.copy()
    # Positive, strongly skewed descriptors benefit from a deterministic log transform.
    transformed[:, :16] = np.log1p(np.maximum(transformed[:, :16], 0.0))
    center = np.median(transformed, axis=0)
    q25, q75 = np.percentile(transformed, [25.0, 75.0], axis=0)
    robust_sigma = (q75 - q25) / 1.349
    if np.any(robust_sigma < 0.0):
        raise PacingEvalError("repeated-section robust scale is invalid")
    return (transformed - center) / np.maximum(robust_sigma, 1e-6)


def tempo_compatibility(values: np.ndarray, query: float) -> np.ndarray:
    tempo = np.asarray(values, dtype=np.float64)
    if (
        tempo.ndim != 1
        or not np.all(np.isfinite(tempo))
        or np.any(tempo < 0.0)
        or not math.isfinite(query)
        or query < 0.0
    ):
        raise PacingEvalError("tempo compatibility input is invalid")
    log_ratio = np.log2(np.maximum(tempo, 1e-6) / max(query, 1e-6))
    direct = np.abs(log_ratio)
    folded = np.minimum.reduce(
        [np.abs(log_ratio + shift) for shift in range(-2, 3)]
    )
    distance = (
        TEMPO_WEIGHTS["direct"] * direct
        + TEMPO_WEIGHTS["octave_folded"] * folded
    )
    return np.exp(-distance / TEMPO_SCALE)


def _category_profiles(
    probabilities: np.ndarray,
    idf: np.ndarray,
    categories: Sequence[str],
    category: str,
) -> np.ndarray:
    positions = np.asarray(
        [index for index, value in enumerate(categories) if value == category],
        dtype=np.int64,
    )
    if not len(positions):
        raise PacingEvalError(f"semantic predictor lacks {category} labels")
    weighted = np.asarray(probabilities[:, positions], dtype=np.float64) * np.asarray(
        idf[positions], dtype=np.float64
    )
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    return weighted / np.maximum(norms, 1e-12)


def compatibility_components(
    pool_vibe: np.ndarray,
    query_vibe: np.ndarray,
    standardized_pool: np.ndarray,
    standardized_query: np.ndarray,
    pool_probabilities: np.ndarray,
    query_probabilities: np.ndarray,
    predictor: CalibratedSemanticPredictor,
) -> Mapping[str, np.ndarray]:
    pool = np.asarray(pool_vibe, dtype=np.float64)
    query = np.asarray(query_vibe, dtype=np.float64)
    standardized_pool = np.asarray(standardized_pool, dtype=np.float64)
    standardized_query = np.asarray(standardized_query, dtype=np.float64)
    if (
        pool.ndim != 2
        or pool.shape[1] != VIBE_DIMENSIONS
        or query.shape != (VIBE_DIMENSIONS,)
        or standardized_pool.shape != pool.shape
        or standardized_query.shape != query.shape
    ):
        raise PacingEvalError("compatibility vibe shapes differ")

    tempo = tempo_compatibility(pool[:, 0], float(query[0]))
    onset_delta = np.abs(
        np.log(np.maximum(pool[:, 3], 1e-6) / max(float(query[3]), 1e-6))
    )
    onset = np.exp(-onset_delta / ONSET_RATIO_SCALE)
    pacing = PACING_WEIGHTS["tempo"] * tempo + PACING_WEIGHTS["onset_rate"] * onset
    tone = -np.sqrt(
        np.mean(
            np.square(
                standardized_pool[:, _TONE_POSITIONS]
                - standardized_query[_TONE_POSITIONS]
            ),
            axis=1,
        )
    )
    dynamics = -np.sqrt(
        np.mean(
            np.square(
                standardized_pool[:, _DYNAMICS_POSITIONS]
                - standardized_query[_DYNAMICS_POSITIONS]
            ),
            axis=1,
        )
    )

    probabilities = np.vstack([query_probabilities, pool_probabilities])
    components: Dict[str, np.ndarray] = {
        "pacing": pacing,
        "tone": tone,
        "dynamics": dynamics,
    }
    for category, key in (
        ("genre", "genre"),
        ("instrument", "instrument"),
        ("mood/theme", "mood_theme"),
    ):
        profiles = _category_profiles(
            probabilities, predictor.idf, predictor.categories, category
        )
        similarity = profiles[1:] @ profiles[0]
        if category == "instrument":
            positions = np.asarray(
                [
                    index
                    for index, value in enumerate(predictor.categories)
                    if value == category
                ],
                dtype=np.int64,
            )
            query_mass = float(np.sum(query_probabilities[positions]))
            pool_mass = np.sum(pool_probabilities[:, positions], axis=1)
            mass_compatibility = np.exp(
                -np.abs(
                    np.log(
                        np.maximum(pool_mass, 1e-6) / max(query_mass, 1e-6)
                    )
                )
            )
            similarity = 0.75 * similarity + 0.25 * mass_compatibility
        components[key] = similarity
    if any(not np.all(np.isfinite(value)) for value in components.values()):
        raise PacingEvalError("compatibility component is non-finite")
    return components


def pacing_rerank_scores(
    acoustic: np.ndarray, components: Mapping[str, np.ndarray]
) -> np.ndarray:
    expected = {"pacing", "tone", "dynamics", "instrument", "mood_theme", "genre"}
    if set(components) != expected:
        raise PacingEvalError("pacing component set drift")
    result = RERANK_WEIGHTS["acoustic"] * percentile_scores(acoustic)
    for name in expected:
        result += RERANK_WEIGHTS[name] * percentile_scores(components[name])
    return result


def _excerpt(reader: FullTrackStoreReader, track_id: int) -> Mapping[str, object]:
    track = reader.read_track(int(track_id))
    if not len(track.repeated_indices) or not len(track.window_starts):
        raise PacingEvalError("track lacks repeated-section evidence")
    index = int(track.repeated_indices[0])
    if not 0 <= index < len(track.window_starts) or track.decoded_samples <= 0:
        raise PacingEvalError("track repeated-section identity drift")
    source_start = float(track.window_starts[index]) / 48_000
    duration = float(track.decoded_samples) / 48_000
    excerpt_duration = min(float(PLAYBACK_EXCERPT_SECONDS), duration)
    context = max(0.0, (excerpt_duration - SOURCE_WINDOW_SECONDS) / 2.0)
    start = min(max(0.0, source_start - context), duration - excerpt_duration)

    def clean(value: float) -> float | int:
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded

    return {
        "kind": "strongest_nonlocal_recurrence",
        "start_seconds": clean(start),
        "end_seconds": clean(start + excerpt_duration),
        "source_window_start_seconds": clean(source_start),
        "source_window_seconds": SOURCE_WINDOW_SECONDS,
    }


def load_probe_report(path: Path) -> Mapping[str, object]:
    if _sha256_path(path) != EXPECTED_PROBE_REPORT_FILE_SHA256:
        raise PacingEvalError("private pacing probe report hash drift")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(report)
        != {
            "aggregate_top5",
            "mean_top5_overlap",
            "ratings_used_for_weight_selection",
            "seeds",
            "study",
            "tracks_measured",
            "weights",
        }
        or report["study"] != "pacing-v3-development-probe"
        or report["tracks_measured"] != 3495
        or report["ratings_used_for_weight_selection"] is not False
        or report["weights"] != {
            "audio": 0.75,
            "dynamics": 0.05,
            "genre": 0.01,
            "instrument": 0.03,
            "mood_theme": 0.01,
            "pacing": 0.10,
            "tone": 0.05,
        }
        or not isinstance(report["seeds"], list)
        or len(report["seeds"]) != 20
    ):
        raise PacingEvalError("private pacing probe report schema drift")
    return report


def load_vibe_cache(
    path: Path,
    expected_track_ids: np.ndarray,
    expected_starts: np.ndarray,
    expected_ends: np.ndarray,
) -> np.ndarray:
    if _sha256_path(path) != EXPECTED_VIBE_CACHE_FILE_SHA256:
        raise PacingEvalError("private repeated-vibe cache hash drift")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"track_ids", "starts", "ends", "vibe"}:
            raise PacingEvalError("private repeated-vibe cache schema drift")
        track_ids = np.asarray(archive["track_ids"], dtype=np.int64)
        starts = np.asarray(archive["starts"], dtype=np.float64)
        ends = np.asarray(archive["ends"], dtype=np.float64)
        vibe = np.asarray(archive["vibe"], dtype=np.float64)
    if (
        not np.array_equal(track_ids, expected_track_ids)
        or not np.array_equal(starts, expected_starts)
        or not np.array_equal(ends, expected_ends)
        or vibe.shape != (len(track_ids), VIBE_DIMENSIONS)
        or not np.all(np.isfinite(vibe))
    ):
        raise PacingEvalError("private repeated-vibe cache identity drift")
    return vibe


def rank_study_methods(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    predictor: CalibratedSemanticPredictor,
    source_seeds: Sequence[Mapping[str, object]],
    excluded_track_ids_by_seed: Mapping[int, Sequence[int]],
    *,
    vibe_cache_path: Path,
    probe_report_path: Path,
    config: PacingEvalConfig,
) -> Tuple[Mapping[int, Tuple[int, ...]], Mapping[int, Tuple[int, ...]]]:
    """Rebuild both frozen rankings from the sealed store and private cache."""
    config.validate()
    report = load_probe_report(probe_report_path)
    if context.source_fingerprint != EXPECTED_SOURCE_FINGERPRINT:
        raise PacingEvalError("Jamendo source fingerprint drift")
    fold = next(item for item in context.folds if item.index == config.fold_index)
    partition = [
        track
        for track in context.tracks
        if fold.track_parts.get(int(track.track_id)) == config.part
    ]
    by_id = {int(track.track_id): track for track in partition}
    seed_ids = tuple(int(seed["seed_track_id"]) for seed in source_seeds)
    if (
        len(seed_ids) != 20
        or len(set(seed_ids)) != 20
        or not set(seed_ids).issubset(by_id)
        or set(excluded_track_ids_by_seed) != set(seed_ids)
        or {
            int(seed["seed_track_id"]) for seed in report["seeds"]
        }
        != set(seed_ids)
    ):
        raise PacingEvalError("pacing study source seed set drift")
    store_rows = {int(track_id): row for row, track_id in enumerate(reader.track_ids)}
    if not set(by_id).issubset(store_rows):
        raise PacingEvalError("sealed store does not cover the test partition")
    partition_rows = np.asarray(
        [store_rows[int(track.track_id)] for track in partition], dtype=np.int64
    )
    globals_matrix = normalize_rows(
        np.asarray(reader.global_embeddings[partition_rows], dtype=np.float32)
    )
    probabilities = predictor.predict_proba(globals_matrix)
    id_to_position = {
        int(track.track_id): position for position, track in enumerate(partition)
    }
    artist_by_track = {
        int(track.track_id): int(track.artist_id) for track in partition
    }

    pools: Dict[int, Tuple[Tuple[int, ...], np.ndarray]] = {}
    all_track_ids = set(seed_ids)
    for seed_id in seed_ids:
        seed = by_id[seed_id]
        excluded = {int(value) for value in excluded_track_ids_by_seed[seed_id]}
        eligible = np.asarray(
            [
                position
                for position, candidate in enumerate(partition)
                if int(candidate.track_id) != seed_id
                and int(candidate.artist_id) != int(seed.artist_id)
                and int(candidate.track_id) not in excluded
            ],
            dtype=np.int64,
        )
        query_position = id_to_position[seed_id]
        initial = globals_matrix[eligible] @ globals_matrix[query_position]
        pool = eligible[np.lexsort((eligible, -initial))[:CANDIDATE_POOL]]
        pool_ids = tuple(int(partition[index].track_id) for index in pool)
        pools[seed_id] = (pool_ids, pool)
        all_track_ids.update(pool_ids)

    ordered_ids = np.asarray(sorted(all_track_ids), dtype=np.int64)
    excerpts = {int(track_id): _excerpt(reader, int(track_id)) for track_id in ordered_ids}
    starts = np.asarray(
        [float(excerpts[int(track_id)]["start_seconds"]) for track_id in ordered_ids]
    )
    ends = np.asarray(
        [float(excerpts[int(track_id)]["end_seconds"]) for track_id in ordered_ids]
    )
    vibe = load_vibe_cache(vibe_cache_path, ordered_ids, starts, ends)
    standardized = robust_standardize_vibe(vibe)
    id_to_vibe = {int(track_id): index for index, track_id in enumerate(ordered_ids)}

    audio_rankings: Dict[int, Tuple[int, ...]] = {}
    pacing_rankings: Dict[int, Tuple[int, ...]] = {}
    for seed_id in seed_ids:
        pool_ids, pool = pools[seed_id]
        query_position = id_to_position[seed_id]
        pool_vibe_rows = np.asarray([id_to_vibe[value] for value in pool_ids])
        query_vibe_row = id_to_vibe[seed_id]
        acoustic = acoustic_scores(
            reader,
            seed_id,
            pool_ids,
            globals_matrix[pool] @ globals_matrix[query_position],
            section_budget=config.section_budget,
        )
        components = compatibility_components(
            vibe[pool_vibe_rows],
            vibe[query_vibe_row],
            standardized[pool_vibe_rows],
            standardized[query_vibe_row],
            probabilities[pool],
            probabilities[query_position],
            predictor,
        )
        audio_rankings[seed_id] = artist_diverse_top(
            pool_ids, acoustic, artist_by_track
        )
        pacing_rankings[seed_id] = artist_diverse_top(
            pool_ids, pacing_rerank_scores(acoustic, components), artist_by_track
        )
    return audio_rankings, pacing_rankings


def prioritize_source_seeds(
    source_seeds: Sequence[Mapping[str, object]],
    audio_rankings: Mapping[int, Sequence[int]],
    pacing_rankings: Mapping[int, Sequence[int]],
) -> Tuple[Mapping[str, object], ...]:
    """Put highest method disagreement first without consulting listener ratings."""
    seed_ids = tuple(int(seed["seed_track_id"]) for seed in source_seeds)
    if (
        len(seed_ids) != 20
        or set(audio_rankings) != set(seed_ids)
        or set(pacing_rankings) != set(seed_ids)
    ):
        raise PacingEvalError("pacing seed priority inputs drift")

    def key(seed: Mapping[str, object]) -> Tuple[int, int]:
        seed_id = int(seed["seed_track_id"])
        overlap = len(set(audio_rankings[seed_id]) & set(pacing_rankings[seed_id]))
        return overlap, seed_id

    return tuple(sorted(source_seeds, key=key))


def _opaque_id(key: bytes, prefix: str, *parts: object) -> str:
    message = "\0".join([PACK_KIND, prefix, *(str(part) for part in parts)])
    digest = hmac.new(key, message.encode(), hashlib.sha256).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _method_bindings() -> Mapping[str, Mapping[str, object]]:
    shared = {
        "candidate_retrieval": "top-200 global-embedding cosine",
        "candidate_pool": CANDIDATE_POOL,
        "section_budget": SECTION_BUDGET,
        "max_results_per_artist": MAX_RESULTS_PER_ARTIST,
        "published_v2_results_excluded": True,
        "test_labels_used_for_ranking": False,
        "listener_ratings_used_for_ranking": False,
        "language_metadata_used_for_ranking": False,
        "acoustic_weights": ACOUSTIC_WEIGHTS,
        "store_binding_sha256": EXPECTED_STORE_BINDING_SHA256,
        "source_semantic_v2_pack_sha256": EXPECTED_SEMANTIC_V2_PACK_SHA256,
        "source_fulltrack_v2_pack_sha256": EXPECTED_FULLTRACK_V2_PACK_SHA256,
        "promoted": False,
    }
    return {
        METHODS[0]: {
            **shared,
            "method": METHODS[0],
            "reranking": "acoustic score only",
        },
        METHODS[1]: {
            **shared,
            "method": METHODS[1],
            "reranking": {
                "normalization": "within-top-200 percentile",
                "weights": RERANK_WEIGHTS,
                "pacing_weights": PACING_WEIGHTS,
                "tempo_weights": TEMPO_WEIGHTS,
                "tempo_scale": TEMPO_SCALE,
                "onset_ratio_scale": ONSET_RATIO_SCALE,
                "tone_positions": [1, 2, *range(16, 29)],
                "dynamics_positions": [4, 5, 6, 7, 8, *range(9, 16)],
                "instrument_profile_weight": 0.75,
                "instrument_mass_weight": 0.25,
                "vibe_dimensions": VIBE_DIMENSIONS,
                "vibe_extractor": "vibe_from_signal(y,sr)",
                "exact_repeated_section_timestamps": True,
                "robust_standardization": "log1p positive descriptors; median and IQR/1.349",
                "semantic_predictors_reused": ["genre", "mood/theme", "instrument"],
            },
        },
    }


def build_blinded_documents(
    *,
    source_seeds: Sequence[Mapping[str, object]],
    track_records: Mapping[str, Mapping[str, object]],
    audio_rankings: Mapping[int, Sequence[int]],
    pacing_rankings: Mapping[int, Sequence[int]],
    store_binding: Mapping[str, object],
    blinding_key: bytes,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    if len(source_seeds) != 20 or len(blinding_key) != 32:
        raise PacingEvalError("pacing study requires 20 seeds and a 32-byte key")
    method_bindings = _method_bindings()
    public_seeds = []
    private_seeds = []
    all_tracks = set()
    for priority, seed in enumerate(source_seeds, 1):
        seed_track_id = int(seed["seed_track_id"])
        rankings = {
            METHODS[0]: tuple(int(value) for value in audio_rankings[seed_track_id]),
            METHODS[1]: tuple(int(value) for value in pacing_rankings[seed_track_id]),
        }
        seed_id = _opaque_id(blinding_key, "pacing-seed", seed_track_id)
        result_ids = {
            track_id: _opaque_id(
                blinding_key, "pacing-result", seed_track_id, track_id
            )
            for track_id in sorted(set(rankings[METHODS[0]]) | set(rankings[METHODS[1]]))
        }
        public_lists = []
        private_lists = []
        for method in METHODS:
            ranking = rankings[method]
            if len(ranking) != 5 or len(set(ranking)) != 5:
                raise PacingEvalError("each pacing study list requires five tracks")
            list_id = _opaque_id(blinding_key, "pacing-list", seed_track_id, method)
            commitment_payload = {
                "pack_kind": PACK_KIND,
                "seed_id": seed_id,
                "list_id": list_id,
                "method_binding": method_bindings[method],
                "ranking_track_ids": list(ranking),
            }
            commitment = hmac.new(
                blinding_key, _canonical_bytes(commitment_payload), hashlib.sha256
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
                {**commitment_payload, "binding_commitment_sha256": commitment}
            )
        public_lists.sort(key=lambda item: item["list_id"])
        private_lists.sort(key=lambda item: item["list_id"])
        overlap = len(set(rankings[METHODS[0]]) & set(rankings[METHODS[1]]))
        public_seeds.append(
            {
                "seed_id": seed_id,
                "seed_track_id": seed_track_id,
                "priority_rank": priority,
                "matched_list_overlap": overlap,
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
        all_tracks.add(seed_track_id)
        all_tracks.update(result_ids)
    if set(track_records) != {str(track_id) for track_id in all_tracks}:
        raise PacingEvalError("public track records do not exactly cover the study")

    store_hash = stable_json_sha256(store_binding)
    private: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PRIVATE_KIND,
        "pack_id": PACK_ID,
        "source_semantic_v2_pack_sha256": EXPECTED_SEMANTIC_V2_PACK_SHA256,
        "source_fulltrack_v2_pack_sha256": EXPECTED_FULLTRACK_V2_PACK_SHA256,
        "source_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "store_binding_sha256": store_hash,
        "probe_report_file_sha256": EXPECTED_PROBE_REPORT_FILE_SHA256,
        "vibe_cache_file_sha256": EXPECTED_VIBE_CACHE_FILE_SHA256,
        "blinding_key_hex": blinding_key.hex(),
        "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
        "methods": list(METHODS),
        "method_bindings": method_bindings,
        "seeds": private_seeds,
        "research_only": True,
        "promotion_allowed": False,
        "production_recommendation_changed": False,
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
        "results_per_method": 5,
        "source_semantic_v2_pack_sha256": EXPECTED_SEMANTIC_V2_PACK_SHA256,
        "source_fulltrack_v2_pack_sha256": EXPECTED_FULLTRACK_V2_PACK_SHA256,
        "source_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "store_binding": dict(store_binding),
        "store_binding_sha256": store_hash,
        "provenance": {
            "probe_report_file_sha256": EXPECTED_PROBE_REPORT_FILE_SHA256,
            "vibe_cache_file_sha256": EXPECTED_VIBE_CACHE_FILE_SHA256,
            "development_probe_tracks_measured": 3495,
            "ranking_cache_tracks": VIBE_CACHE_TRACKS,
            "signal_dimensions": VIBE_DIMENSIONS,
            "exact_repeated_section_timestamps": True,
            "ratings_used": False,
        },
        "blinding": {
            "opaque_per_seed_list_ids": True,
            "opaque_per_seed_result_ids": True,
            "method_identity_public": False,
            "method_order_randomized_per_seed_session": True,
            "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
            "private_unblinding_sha256": private["content_sha256"],
        },
        "matched_design": {
            "candidate_pool": CANDIDATE_POOL,
            "one_result_per_artist": True,
            "same_candidate_pool_per_seed": True,
            "whole_track_ranking": True,
        },
        "audio_delivery": {
            "kind": "Jamendo first-party full-track MP3",
            "host": "prod-1.storage.jamendo.com",
            "repository_contains_audio": False,
            "commercial_preview": False,
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
            "priority": "ascending anonymous top-five overlap, then track ID",
        },
        "language_policy": {
            "evaluated_here": False,
            "reason": "MTG-Jamendo provides no trustworthy track-language field",
            "production_policy": "Spotify lyrics-language gating remains separate and unchanged",
        },
        "tracks": dict(track_records),
        "seeds": public_seeds,
        "research_only": True,
        "promotion_allowed": False,
        "production_recommendation_changed": False,
        "notice": (
            "Research-only blinded comparison using repeated-section recurrence "
            "excerpts; these excerpts are not verified chorus labels."
        ),
    }
    public["content_sha256"] = _content_sha256(public)
    validate_blinded_documents(public, private, require_frozen_artifacts=False)
    return public, private


def _valid_opaque(value: object, prefix: str) -> bool:
    marker = f"{prefix}-"
    return (
        isinstance(value, str)
        and value.startswith(marker)
        and len(value) == len(marker) + 24
        and set(value[len(marker) :]) <= _HEX
    )


def validate_blinded_documents(
    public: Mapping[str, object],
    private: Mapping[str, object],
    *,
    require_frozen_artifacts: bool = True,
) -> None:
    """Fail closed on hashes, provenance, commitments, coverage, and blinding."""
    if (
        public.get("schema_version") != SCHEMA_VERSION
        or public.get("pack_kind") != PACK_KIND
        or public.get("pack_id") != PACK_ID
        or public.get("content_sha256") != _content_sha256(public)
        or public.get("rankings_state") != "LOCKED_BEFORE_RATINGS"
        or public.get("ratings_count_at_freeze") != 0
        or public.get("seed_count") != 20
        or public.get("method_count") != 2
        or public.get("results_per_method") != 5
        or public.get("research_only") is not True
        or public.get("promotion_allowed") is not False
        or public.get("production_recommendation_changed") is not False
    ):
        raise PacingEvalError("public pacing study document drift")
    if (
        private.get("schema_version") != SCHEMA_VERSION
        or private.get("artifact_kind") != PRIVATE_KIND
        or private.get("pack_id") != PACK_ID
        or private.get("content_sha256") != _content_sha256(private)
        or private.get("methods") != list(METHODS)
        or private.get("research_only") is not True
        or private.get("promotion_allowed") is not False
        or private.get("production_recommendation_changed") is not False
    ):
        raise PacingEvalError("private pacing study document drift")
    if require_frozen_artifacts:
        if not EXPECTED_PUBLIC_PACK_SHA256 or not EXPECTED_PRIVATE_UNBLINDING_SHA256:
            raise PacingEvalError("frozen pacing study hashes are not configured")
        if (
            public["content_sha256"] != EXPECTED_PUBLIC_PACK_SHA256
            or private["content_sha256"] != EXPECTED_PRIVATE_UNBLINDING_SHA256
        ):
            raise PacingEvalError("frozen pacing study artifact drift")
    if (
        public.get("source_semantic_v2_pack_sha256")
        != EXPECTED_SEMANTIC_V2_PACK_SHA256
        or private.get("source_semantic_v2_pack_sha256")
        != EXPECTED_SEMANTIC_V2_PACK_SHA256
        or public.get("source_fulltrack_v2_pack_sha256")
        != EXPECTED_FULLTRACK_V2_PACK_SHA256
        or private.get("source_fulltrack_v2_pack_sha256")
        != EXPECTED_FULLTRACK_V2_PACK_SHA256
        or public.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT
        or private.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT
        or public.get("store_binding_sha256") != EXPECTED_STORE_BINDING_SHA256
        or private.get("store_binding_sha256") != EXPECTED_STORE_BINDING_SHA256
        or stable_json_sha256(public.get("store_binding", {}))
        != EXPECTED_STORE_BINDING_SHA256
        or private.get("probe_report_file_sha256")
        != EXPECTED_PROBE_REPORT_FILE_SHA256
        or private.get("vibe_cache_file_sha256") != EXPECTED_VIBE_CACHE_FILE_SHA256
    ):
        raise PacingEvalError("pacing source provenance drift")
    if public.get("provenance") != {
        "probe_report_file_sha256": EXPECTED_PROBE_REPORT_FILE_SHA256,
        "vibe_cache_file_sha256": EXPECTED_VIBE_CACHE_FILE_SHA256,
        "development_probe_tracks_measured": 3495,
        "ranking_cache_tracks": VIBE_CACHE_TRACKS,
        "signal_dimensions": VIBE_DIMENSIONS,
        "exact_repeated_section_timestamps": True,
        "ratings_used": False,
    }:
        raise PacingEvalError("public pacing provenance drift")
    bindings = private.get("method_bindings")
    if bindings != _method_bindings():
        raise PacingEvalError("private pacing method binding drift")
    try:
        key = bytes.fromhex(str(private["blinding_key_hex"]))
    except (KeyError, ValueError) as exc:
        raise PacingEvalError("private pacing blinding key is malformed") from exc
    blinding = public.get("blinding")
    if (
        len(key) != 32
        or not isinstance(blinding, Mapping)
        or hashlib.sha256(key).hexdigest() != private.get("blinding_key_sha256")
        or blinding.get("blinding_key_sha256") != private.get("blinding_key_sha256")
        or blinding.get("private_unblinding_sha256") != private.get("content_sha256")
        or blinding.get("method_identity_public") is not False
        or blinding.get("method_order_randomized_per_seed_session") is not True
    ):
        raise PacingEvalError("pacing blinding binding drift")
    public_seeds = public.get("seeds")
    private_seeds = private.get("seeds")
    tracks = public.get("tracks")
    if (
        not isinstance(public_seeds, list)
        or not isinstance(private_seeds, list)
        or len(public_seeds) != 20
        or len(private_seeds) != 20
        or not isinstance(tracks, Mapping)
    ):
        raise PacingEvalError("pacing study seed count drift")
    private_by_seed = {item.get("seed_id"): item for item in private_seeds}
    seen_lists = set()
    seen_results = set()
    all_tracks = set()
    priority_keys = []
    for priority, seed in enumerate(public_seeds, 1):
        seed_id = seed.get("seed_id")
        seed_track_id = seed.get("seed_track_id")
        private_seed = private_by_seed.get(seed_id)
        if (
            not _valid_opaque(seed_id, "pacing-seed")
            or not isinstance(seed_track_id, int)
            or isinstance(seed_track_id, bool)
            or seed.get("priority_rank") != priority
            or private_seed is None
            or private_seed.get("seed_track_id") != seed_track_id
            or len(seed.get("lists", [])) != 2
            or len(private_seed.get("lists", [])) != 2
        ):
            raise PacingEvalError("public/private pacing seed binding drift")
        result_rows = seed.get("result_ids", [])
        result_map = {
            row.get("track_id"): row.get("result_id")
            for row in result_rows
            if isinstance(row, Mapping)
        }
        seed_result_ids = set(result_map.values())
        if (
            len(result_map) != len(result_rows)
            or len(seed_result_ids) != len(result_rows)
            or any(
                not _valid_opaque(result_id, "pacing-result")
                for result_id in seed_result_ids
            )
            or seen_results & seed_result_ids
        ):
            raise PacingEvalError("pacing result identity map drift")
        seen_results.update(seed_result_ids)
        private_lists = {item.get("list_id"): item for item in private_seed["lists"]}
        rankings = {}
        for public_list in seed["lists"]:
            list_id = public_list.get("list_id")
            private_list = private_lists.get(list_id)
            if (
                not _valid_opaque(list_id, "pacing-list")
                or list_id in seen_lists
                or private_list is None
                or len(public_list.get("ranking", [])) != 5
            ):
                raise PacingEvalError("pacing list identity drift")
            seen_lists.add(list_id)
            method = private_list.get("method_binding", {}).get("method")
            if method not in METHODS or private_list["method_binding"] != bindings[method]:
                raise PacingEvalError("pacing private method identity drift")
            ranking = []
            artist_ids = set()
            for position, row in enumerate(public_list["ranking"], 1):
                track_id = row.get("track_id")
                result_id = row.get("result_id")
                track = tracks.get(str(track_id), {})
                if (
                    row.get("position") != position
                    or result_map.get(track_id) != result_id
                    or track.get("track_id") != track_id
                ):
                    raise PacingEvalError("pacing ranking result identity drift")
                artist_ids.add(track["source_identity"]["artist_id"])
                ranking.append(track_id)
            payload = {
                name: private_list[name]
                for name in (
                    "pack_kind",
                    "seed_id",
                    "list_id",
                    "method_binding",
                    "ranking_track_ids",
                )
            }
            commitment = hmac.new(
                key, _canonical_bytes(payload), hashlib.sha256
            ).hexdigest()
            if (
                commitment != public_list.get("binding_commitment_sha256")
                or commitment != private_list.get("binding_commitment_sha256")
                or private_list.get("ranking_track_ids") != ranking
                or len(set(ranking)) != 5
                or len(artist_ids) != 5
            ):
                raise PacingEvalError("pacing list commitment or diversity drift")
            rankings[method] = set(ranking)
            all_tracks.update(ranking)
        if set(rankings) != set(METHODS) or set(result_map) != set.union(*rankings.values()):
            raise PacingEvalError("pacing seed method/result coverage drift")
        overlap = len(rankings[METHODS[0]] & rankings[METHODS[1]])
        if seed.get("matched_list_overlap") != overlap:
            raise PacingEvalError("pacing disagreement evidence drift")
        priority_keys.append((overlap, seed_track_id))
        all_tracks.add(seed_track_id)
    if priority_keys != sorted(priority_keys):
        raise PacingEvalError("pacing uncertainty priority drift")
    if set(tracks) != {str(track_id) for track_id in all_tracks}:
        raise PacingEvalError("pacing public track coverage drift")
    public_text = json.dumps(public, sort_keys=True)
    if any(method in public_text for method in METHODS):
        raise PacingEvalError("public pacing study reveals method identity")


def build_production_pacing_eval(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    store_root: Path,
    predictor_model: Path,
    predictor_metadata: Path,
    semantic_v2_public_path: Path,
    vibe_cache_path: Path,
    probe_report_path: Path,
    public_output: Path,
    private_output: Path,
    blinding_key_path: Path,
    create_blinding_key: bool,
    verify_audio: bool,
    audio_workers: int,
    config: PacingEvalConfig,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    config.validate()
    semantic_v2_public_path = Path(semantic_v2_public_path)
    semantic_public = _load_frozen_pack(
        semantic_v2_public_path,
        expected_file_sha256=EXPECTED_SEMANTIC_V2_PACK_FILE_SHA256,
        expected_content_sha256=EXPECTED_SEMANTIC_V2_PACK_SHA256,
        label="semantic-v2 pack",
    )
    fulltrack_public = _load_frozen_pack(
        semantic_v2_public_path.parent / "pilot-pack.json",
        expected_file_sha256=EXPECTED_FULLTRACK_V2_PACK_FILE_SHA256,
        expected_content_sha256=EXPECTED_FULLTRACK_V2_PACK_SHA256,
        label="fulltrack-v2 pack",
    )
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    if context.evidence_scope != EVIDENCE_SCOPE:
        raise PacingEvalError("source evidence scope is not full-track Jamendo")
    predictor = load_predictor(predictor_model, predictor_metadata)
    source_seeds = tuple(semantic_public["seeds"])
    excluded = _published_result_exclusions(fulltrack_public, semantic_public)
    if {int(seed["seed_track_id"]) for seed in source_seeds} != set(excluded):
        raise PacingEvalError("source and published-exclusion seed coverage drift")
    tracks_by_id = {int(track.track_id): track for track in context.tracks}
    with FullTrackStoreReader(
        store_root, expected_source_fingerprint=context.source_fingerprint
    ) as reader:
        store_binding = dict(reader.binding.as_dict())
        store_binding["sealed_manifest_sha256"] = stable_json_sha256(reader.manifest)
        if stable_json_sha256(store_binding) != EXPECTED_STORE_BINDING_SHA256:
            raise PacingEvalError("sealed store binding drift")
        audio_rankings, pacing_rankings = rank_study_methods(
            context,
            reader,
            predictor,
            source_seeds,
            excluded,
            vibe_cache_path=vibe_cache_path,
            probe_report_path=probe_report_path,
            config=config,
        )
        source_seeds = prioritize_source_seeds(
            source_seeds, audio_rankings, pacing_rankings
        )
        all_track_ids = {
            int(seed["seed_track_id"]) for seed in source_seeds
        } | {
            int(track_id)
            for rankings in (audio_rankings, pacing_rankings)
            for ranking in rankings.values()
            for track_id in ranking
        }
        store_rows = {
            int(track_id): row for row, track_id in enumerate(reader.track_ids)
        }
        if not all_track_ids.issubset(tracks_by_id) or not all_track_ids.issubset(
            store_rows
        ):
            raise PacingEvalError("pacing study track metadata/store coverage drift")
        existing_records = dict(semantic_public["tracks"])
        audio_evidence = {
            row["audio"]["url"]: row["audio"]["verification"]
            for row in existing_records.values()
        }
        missing_urls = [
            lawful_stream_url(track_id)
            for track_id in sorted(all_track_ids)
            if lawful_stream_url(track_id) not in audio_evidence
        ]
        if missing_urls and not verify_audio:
            raise PacingEvalError("new pacing tracks require public-audio verification")
        if missing_urls:
            audio_evidence.update(
                verify_public_audio_urls(missing_urls, workers=audio_workers)
            )
        track_records = {}
        for track_id in sorted(all_track_ids):
            existing = existing_records.get(str(track_id))
            record = (
                dict(existing)
                if existing is not None
                else _track_record(
                    tracks_by_id[track_id],
                    fold_index=config.fold_index,
                    fold_part=config.part,
                    store_row=store_rows[track_id],
                    audio_verification=audio_evidence[lawful_stream_url(track_id)],
                )
            )
            record["playback_excerpt"] = _excerpt(reader, track_id)
            track_records[str(track_id)] = record
    key = _read_blinding_key(blinding_key_path, create_blinding_key)
    public, private = build_blinded_documents(
        source_seeds=source_seeds,
        track_records=track_records,
        audio_rankings=audio_rankings,
        pacing_rankings=pacing_rankings,
        store_binding=store_binding,
        blinding_key=key,
    )
    validate_blinded_documents(public, private)
    _write_json(public_output, public, private=False)
    _write_json(private_output, private, private=True)
    return public, private


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or verify the pacing V3 study.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "store_root",
        "predictor_model",
        "predictor_metadata",
        "semantic_v2_public",
        "vibe_cache",
        "probe_report",
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
        validate_blinded_documents(
            json.loads(args.public.read_text(encoding="utf-8")),
            json.loads(args.private.read_text(encoding="utf-8")),
        )
        print(json.dumps({"status": "ok"}))
        return 0
    build_production_pacing_eval(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        store_root=args.store_root,
        predictor_model=args.predictor_model,
        predictor_metadata=args.predictor_metadata,
        semantic_v2_public_path=args.semantic_v2_public,
        vibe_cache_path=args.vibe_cache,
        probe_report_path=args.probe_report,
        public_output=args.public_output,
        private_output=args.private_output,
        blinding_key_path=args.blinding_key,
        create_blinding_key=args.create_blinding_key,
        verify_audio=args.verify_public_audio,
        audio_workers=args.audio_workers,
        config=PacingEvalConfig(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
