"""Build the blinded active best/worst V4 listening study."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans

from soundalike.audio.vibe import vibe_from_signal

from .fulltrack_extract import normalize_rows
from .fulltrack_pilot import _content_sha256, lawful_stream_url
from .fulltrack_store import FullTrackStoreReader
from .jamendo_fulltrack import JamendoTrack, load_jamendo_context
from .pacing_eval import (
    _excerpt,
    acoustic_scores,
    compatibility_components,
    percentile_scores,
    robust_standardize_vibe,
)
from .semantic_predictor import load_predictor
from .v4_features import STATE_CODES, load_semantic_cache
from .v4_gates import INSTRUMENTAL, UNKNOWN, VOCAL, compatibility_allowed
from .v4_population import validate_population_manifest


SCHEMA_VERSION = 2
PACK_KIND = "soundalike_v4_active_full_ranking"
PRIVATE_KIND = "soundalike_v4_active_full_ranking_private"
PLAN_KIND = "soundalike_v4_active_study_plan"
PACK_ID = "v4-active-full-ranking-2"
CANDIDATE_POOL = 200
SEED_SHORTLIST = 24
UNIQUE_TASKS = 16
ANCHOR_TASKS = 2
CANDIDATES_PER_TASK = 4
RANKING_DEPTH = 15
MINIMUM_SEED_SECONDS = 90.0
MAXIMUM_SEED_SECONDS = 480.0
RERANK_WEIGHTS = {
    "acoustic": 0.70,
    "pacing": 0.10,
    "tone": 0.05,
    "dynamics": 0.04,
    "instrument": 0.05,
    "mood_theme": 0.03,
    "genre": 0.01,
    "voice_compatibility": 0.02,
}
CODE_STATES = {value: key for key, value in STATE_CODES.items()}
NICHE_TAGS = frozenset(
    {
        "genre---ambient",
        "genre---classical",
        "genre---ebm",
        "genre---emocore",
        "genre---experimental",
        "genre---industrial",
        "genre---metalcore",
        "genre---soundtrack",
        "genre---technoindustrial",
    }
)


class V4StudyError(RuntimeError):
    """The V4 study is overlapping, unblinded, or non-reproducible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_vibe(task: tuple[int, str, float, float]) -> tuple[int, np.ndarray]:
    import librosa

    track_id, path, start, end = task
    waveform, sample_rate = librosa.load(
        path,
        sr=22_050,
        mono=True,
        offset=start,
        duration=end - start,
    )
    return track_id, vibe_from_signal(waveform, sample_rate).vector()


def _artist_diverse(
    track_ids: Sequence[int],
    scores: np.ndarray,
    artist_by_track: Mapping[int, int],
    count: int,
) -> tuple[int, ...]:
    values = np.asarray(scores, dtype=np.float64)
    selected = []
    artists = set()
    for position in np.argsort(-values, kind="stable"):
        if not np.isfinite(values[int(position)]):
            continue
        track_id = int(track_ids[int(position)])
        artist = int(artist_by_track[track_id])
        if artist in artists:
            continue
        selected.append(track_id)
        artists.add(artist)
        if len(selected) == count:
            return tuple(selected)
    raise V4StudyError("ranking lacks enough artist-diverse candidates")


def _artist_unique_pool(
    positions: np.ndarray,
    similarities: np.ndarray,
    tracks: Sequence[JamendoTrack],
    count: int,
) -> np.ndarray:
    """Keep the nearest track per artist so the shared pool is diverse."""
    if count <= 0 or len(positions) != len(similarities):
        raise V4StudyError("candidate pool inputs are invalid")
    ordered = np.lexsort(
        (
            np.asarray(
                [int(tracks[int(position)].track_id) for position in positions]
            ),
            -np.asarray(similarities, dtype=np.float64),
        )
    )
    selected = []
    artists = set()
    for local_position in ordered:
        position = int(positions[int(local_position)])
        artist_id = int(tracks[position].artist_id)
        if artist_id in artists:
            continue
        selected.append(position)
        artists.add(artist_id)
        if len(selected) == count:
            break
    if len(selected) < RANKING_DEPTH:
        raise V4StudyError("candidate pool lacks enough compatible artists")
    return np.asarray(selected, dtype=np.int64)


def _choose_seeds(
    tracks: Sequence[JamendoTrack],
    embeddings: np.ndarray,
    gate_cache: Mapping[str, object] | None,
) -> tuple[int, ...]:
    matrix = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    compatible_artists: dict[tuple[str, str], set[int]] = {}
    if gate_cache is not None:
        for track in tracks:
            gate = _effective_gate(int(track.track_id), UNKNOWN, gate_cache)
            compatible_artists.setdefault(gate, set()).add(int(track.artist_id))
    one_per_artist = {}
    for position, track in enumerate(tracks):
        if not MINIMUM_SEED_SECONDS <= track.duration_seconds <= MAXIMUM_SEED_SECONDS:
            continue
        if gate_cache is not None:
            vocal_state, language = _effective_gate(
                int(track.track_id), UNKNOWN, gate_cache
            )
            if vocal_state == UNKNOWN or (
                vocal_state == VOCAL and language == UNKNOWN
            ):
                continue
            if (
                len(
                    compatible_artists[(vocal_state, language)]
                    - {int(track.artist_id)}
                )
                < RANKING_DEPTH
            ):
                continue
        current = one_per_artist.get(int(track.artist_id))
        quality = (
            len(track.tags),
            min(float(track.duration_seconds), 600.0),
            -int(track.track_id),
        )
        if current is None or quality > current[0]:
            one_per_artist[int(track.artist_id)] = (quality, position)
    positions = np.asarray(
        sorted(value[1] for value in one_per_artist.values()), dtype=np.int64
    )
    if len(positions) < SEED_SHORTLIST:
        raise V4StudyError("reserve has insufficient seed artists")
    model = KMeans(
        n_clusters=SEED_SHORTLIST,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=44,
        algorithm="lloyd",
    )
    labels = model.fit_predict(matrix[positions])
    selected = []
    for cluster in range(SEED_SHORTLIST):
        local = np.flatnonzero(labels == cluster)
        distances = np.sum(
            np.square(
                matrix[positions[local]] - model.cluster_centers_[cluster]
            ),
            axis=1,
        )
        tied = local[
            np.flatnonzero(
                np.isclose(distances, np.min(distances), rtol=0.0, atol=1e-9)
            )
        ]
        chosen = min(
            tied,
            key=lambda value: int(tracks[int(positions[value])].track_id),
        )
        selected.append(int(tracks[int(positions[chosen])].track_id))
    return tuple(sorted(selected))


def _load_or_extract_vibe(
    cache_path: Path,
    ordered_ids: np.ndarray,
    excerpts: Mapping[int, Mapping[str, object]],
    tracks: Mapping[int, JamendoTrack],
    workers: int,
) -> np.ndarray:
    starts = np.asarray(
        [float(excerpts[int(track_id)]["start_seconds"]) for track_id in ordered_ids]
    )
    ends = np.asarray(
        [float(excerpts[int(track_id)]["end_seconds"]) for track_id in ordered_ids]
    )
    cached_vectors: dict[int, np.ndarray] = {}
    rewrite_cache = True
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != {"track_ids", "starts", "ends", "vibe"}:
                raise V4StudyError("V4 repeated-vibe cache identity drift")
            cached_ids = np.asarray(archive["track_ids"], dtype=np.int64)
            cached_starts = np.asarray(archive["starts"], dtype=np.float64)
            cached_ends = np.asarray(archive["ends"], dtype=np.float64)
            cached_vibe = np.asarray(archive["vibe"], dtype=np.float64)
            if (
                cached_ids.ndim != 1
                or len(set(cached_ids.tolist())) != len(cached_ids)
                or cached_starts.shape != cached_ids.shape
                or cached_ends.shape != cached_ids.shape
                or cached_vibe.shape != (len(cached_ids), 29)
                or not np.all(np.isfinite(cached_vibe))
            ):
                raise V4StudyError("V4 repeated-vibe cache arrays are invalid")
            desired = {
                int(track_id): (float(start), float(end))
                for track_id, start, end in zip(ordered_ids, starts, ends)
            }
            for row, (track_id, start, end) in enumerate(
                zip(cached_ids, cached_starts, cached_ends)
            ):
                track_id = int(track_id)
                if track_id not in desired:
                    continue
                if desired[track_id] != (float(start), float(end)):
                    raise V4StudyError("V4 repeated-vibe excerpt identity drift")
                cached_vectors[track_id] = cached_vibe[row]
            rewrite_cache = not (
                np.array_equal(cached_ids, ordered_ids)
                and np.array_equal(cached_starts, starts)
                and np.array_equal(cached_ends, ends)
            )
    missing_ids = [
        int(track_id)
        for track_id in ordered_ids
        if int(track_id) not in cached_vectors
    ]
    if missing_ids:
        desired_rows = {
            int(track_id): (float(start), float(end))
            for track_id, start, end in zip(ordered_ids, starts, ends)
        }
        tasks = [
            (
                int(track_id),
                str(tracks[int(track_id)].audio_path),
                desired_rows[int(track_id)][0],
                desired_rows[int(track_id)][1],
            )
            for track_id in missing_ids
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, (track_id, vector) in enumerate(
                executor.map(_extract_vibe, tasks, chunksize=4), 1
            ):
                cached_vectors[track_id] = vector
                if index % 250 == 0:
                    print(f"vibe {index}/{len(tasks)}", flush=True)
        rewrite_cache = True
    vibe = np.stack(
        [cached_vectors[int(track_id)] for track_id in ordered_ids]
    ).astype(np.float32, copy=False)
    if rewrite_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                track_ids=ordered_ids,
                starts=starts,
                ends=ends,
                vibe=vibe,
            )
        temporary.replace(cache_path)
    if vibe.shape != (len(ordered_ids), 29) or not np.all(np.isfinite(vibe)):
        raise V4StudyError("V4 repeated-vibe cache shape drift")
    return vibe


def _effective_gate(
    track_id: int,
    semantic_state: str,
    gate_cache: Mapping[str, object] | None,
) -> tuple[str, str]:
    if not gate_cache:
        return semantic_state, UNKNOWN
    row = gate_cache.get("tracks", {}).get(str(track_id))
    if not isinstance(row, Mapping):
        return UNKNOWN, UNKNOWN
    return str(row.get("vocal_state", UNKNOWN)), str(
        row.get("language", UNKNOWN)
    )


def _load_gate_cache(
    path: Path | None,
    *,
    source_fingerprint: str,
) -> Mapping[str, object] | None:
    if path is None:
        return None
    cache = json.loads(path.read_text(encoding="utf-8"))
    rows = cache.get("tracks")
    valid_states = {VOCAL, INSTRUMENTAL, UNKNOWN}
    if (
        cache.get("schema_version") != 2
        or cache.get("gate_kind") != "soundalike_v4_study_track_gates_v2"
        or cache.get("source_fingerprint") != source_fingerprint
        or cache.get("content_sha256") != _content_sha256(cache)
        or not isinstance(rows, Mapping)
        or any(
            not isinstance(track_id, str)
            or not track_id.isdigit()
            or not isinstance(row, Mapping)
            or row.get("vocal_state") not in valid_states
            or not isinstance(row.get("language"), str)
            or (
                row.get("vocal_state") != VOCAL
                and row.get("language") != UNKNOWN
            )
            for track_id, row in rows.items()
        )
    ):
        raise V4StudyError("V4 detector gate cache binding failed")
    return cache


def _track_record(
    track: JamendoTrack,
    excerpt: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "track_id": int(track.track_id),
        "title": track.title,
        "artist": track.artist_name,
        "album": track.album_name,
        "source_identity": {
            "artist_id": int(track.artist_id),
            "album_id": int(track.album_id),
            "source_audio_sha256": track.expected_audio_sha256,
            "source_audio_bytes": int(track.expected_audio_bytes),
        },
        "audio": {
            "url": lawful_stream_url(int(track.track_id)),
            "excerpt": dict(excerpt),
        },
        "attribution": {
            "track_url": track.jamendo_url.replace("http://", "https://", 1),
            "license_name": track.license.name,
            "license_url": track.license.url,
            "credit": track.license.attribution,
        },
    }


def _opaque_id(key: bytes, prefix: str, *parts: object) -> str:
    message = "\0".join([PACK_KIND, prefix, *(str(part) for part in parts)])
    return f"{prefix}-{hmac.new(key, message.encode(), hashlib.sha256).hexdigest()[:24]}"


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
    blinding_key_path: Path,
    gate_cache_path: Path | None,
    workers: int,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    population = json.loads(population_path.read_text(encoding="utf-8"))
    validate_population_manifest(population, context)
    reserve_ids = set(population["human_reserve"]["track_ids"])
    reserve_tracks = tuple(
        track for track in context.tracks if int(track.track_id) in reserve_ids
    )
    by_id = {int(track.track_id): track for track in reserve_tracks}
    artist_by_track = {
        int(track.track_id): int(track.artist_id) for track in reserve_tracks
    }
    predictor = load_predictor(predictor_model, predictor_metadata)
    gate_cache = _load_gate_cache(
        gate_cache_path,
        source_fingerprint=context.source_fingerprint,
    )
    if gate_cache is not None and set(gate_cache["tracks"]) != {
        str(track_id) for track_id in reserve_ids
    }:
        raise V4StudyError("V4 detector gate cache does not cover the reserve")
    if not blinding_key_path.exists():
        blinding_key_path.parent.mkdir(parents=True, exist_ok=True)
        blinding_key_path.write_bytes(secrets.token_bytes(32))
    key = blinding_key_path.read_bytes()
    if len(key) != 32:
        raise V4StudyError("V4 blinding key must contain 32 bytes")

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
        probabilities, voice_scores, voice_states = load_semantic_cache(
            semantic_cache_path,
            semantic_metadata_path,
            expected_source_fingerprint=context.source_fingerprint,
            expected_track_ids=store_ids,
        )
        seed_ids = _choose_seeds(reserve_tracks, reserve_globals, gate_cache)
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
                    candidate_id == seed_id
                    or int(candidate.artist_id) == artist_by_track[seed_id]
                ):
                    continue
                candidate_gate = _effective_gate(
                    candidate_id, UNKNOWN, gate_cache
                )
                if gate_cache is not None and not compatibility_allowed(
                    query_gate[0],
                    candidate_gate[0],
                    query_gate[1],
                    candidate_gate[1],
                ):
                    continue
                eligible_positions.append(position)
            if len(eligible_positions) < RANKING_DEPTH:
                raise V4StudyError(
                    "seed lacks enough language-compatible candidates"
                )
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
            score = RERANK_WEIGHTS["acoustic"] * percentile_scores(acoustic)
            for name in (
                "pacing",
                "tone",
                "dynamics",
                "instrument",
                "mood_theme",
                "genre",
            ):
                score += RERANK_WEIGHTS[name] * percentile_scores(
                    components[name]
                )
            score += RERANK_WEIGHTS["voice_compatibility"] * percentile_scores(
                voice_compatibility
            )
            control = _artist_diverse(
                pool_ids, acoustic, artist_by_track, RANKING_DEPTH
            )
            challenger = _artist_diverse(
                pool_ids, score, artist_by_track, RANKING_DEPTH
            )
            overlap = len(set(control[:4]) & set(challenger[:4]))
            niche = bool(set(by_id[seed_id].tags) & NICHE_TAGS)
            ranking_rows.append(
                {
                    "seed_track_id": seed_id,
                    "control": list(control),
                    "challenger": list(challenger),
                    "top_four_overlap": overlap,
                    "disagreement": 1.0 - overlap / 4.0,
                    "niche": niche,
                }
            )

    ranking_rows.sort(
        key=lambda row: (
            row["niche"],
            -row["disagreement"],
            row["seed_track_id"],
        )
    )
    selected = ranking_rows[:UNIQUE_TASKS]
    tasks = []
    private_tasks = []
    used_tracks = set()
    for priority, row in enumerate(selected, 1):
        chosen = []
        origins = {}
        chosen_artists = {artist_by_track[row["seed_track_id"]]}
        for method in ("control", "challenger"):
            for track_id in row[method]:
                if track_id in chosen:
                    if method not in origins[track_id]:
                        origins[track_id].append(method)
                elif artist_by_track[track_id] not in chosen_artists:
                    chosen.append(track_id)
                    chosen_artists.add(artist_by_track[track_id])
                    origins[track_id] = [method]
                if sum(method in origins[value] for value in chosen) == 2:
                    break
        for track_id in [*row["control"], *row["challenger"]]:
            if (
                track_id not in chosen
                and artist_by_track[track_id] not in chosen_artists
            ):
                chosen.append(track_id)
                chosen_artists.add(artist_by_track[track_id])
                origins[track_id] = ["fill"]
            if len(chosen) == CANDIDATES_PER_TASK:
                break
        if len(chosen) != CANDIDATES_PER_TASK:
            raise V4StudyError("cannot construct a four-item active task")
        task_id = _opaque_id(key, "v4-task", row["seed_track_id"], priority)
        order = sorted(
            chosen,
            key=lambda track_id: hmac.new(
                key,
                f"{task_id}\0{track_id}".encode(),
                hashlib.sha256,
            ).hexdigest(),
        )
        candidate_rows = [
            {
                "choice_id": _opaque_id(key, "v4-choice", task_id, track_id),
                "track_id": track_id,
            }
            for track_id in order
        ]
        tasks.append(
            {
                "task_id": task_id,
                "priority_rank": priority,
                "seed_track_id": row["seed_track_id"],
                "candidates": candidate_rows,
            }
        )
        private_tasks.append(
            {
                "task_id": task_id,
                "seed_track_id": row["seed_track_id"],
                "candidate_origins": {
                    str(track_id): origins[track_id] for track_id in order
                },
                "control_ranking": row["control"],
                "challenger_ranking": row["challenger"],
            }
        )
        used_tracks.update([row["seed_track_id"], *chosen])
    for anchor_index, source_index in enumerate((0, len(tasks) // 2), 1):
        source = tasks[source_index]
        task_id = _opaque_id(key, "v4-anchor", source["task_id"], anchor_index)
        anchor = {
            **source,
            "task_id": task_id,
            "priority_rank": len(tasks) + 1,
            "candidates": [
                {
                    "choice_id": _opaque_id(
                        key, "v4-choice", task_id, row["track_id"]
                    ),
                    "track_id": row["track_id"],
                }
                for row in reversed(source["candidates"])
            ],
        }
        tasks.append(anchor)
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
            "detector_gate_sha256": (
                gate_cache["content_sha256"] if gate_cache is not None else None
            ),
            "listener_ratings_used_for_ranking": False,
            "learned_preference_head_used": False,
            "candidate_gate": (
                "known voice class required; vocal candidates must match the "
                "seed's known language"
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
            "control": "frozen 30/25/25/20 acoustic score",
            "challenger": {
                "weights": RERANK_WEIGHTS,
                "strict_vocal_language_gate": True,
                "detector_gate_sha256": (
                    gate_cache["content_sha256"]
                    if gate_cache is not None
                    else None
                ),
                "unknown_fallback": False,
                "learned_preference_head": "rejected by grouped gate",
            },
        },
        "tasks": private_tasks,
    }
    private["content_sha256"] = _content_sha256(private)
    public["private_unblinding_sha256"] = private["content_sha256"]
    public["content_sha256"] = _content_sha256(public)
    gate_track_ids = sorted(reserve_ids)
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "plan_kind": PLAN_KIND,
        "source_fingerprint": context.source_fingerprint,
        "population_sha256": population["content_sha256"],
        "vibe_cache_sha256": _sha256(vibe_cache_path),
        "detector_gate_sha256": (
            gate_cache["content_sha256"] if gate_cache is not None else None
        ),
        "gate_track_ids": gate_track_ids,
        "rankings": ranking_rows,
        "public_pack_sha256": public["content_sha256"],
    }
    plan["content_sha256"] = _content_sha256(plan)
    validate_study_artifacts(public, private, plan, gate_cache=gate_cache)
    return public, private, plan


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return set(value) | {
            key
            for item in value.values()
            for key in _nested_keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def validate_study_artifacts(
    public: Mapping[str, object],
    private: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    gate_cache: Mapping[str, object] | None,
) -> None:
    if (
        public.get("pack_kind") != PACK_KIND
        or private.get("private_kind") != PRIVATE_KIND
        or plan.get("plan_kind") != PLAN_KIND
        or public.get("content_sha256") != _content_sha256(public)
        or private.get("content_sha256") != _content_sha256(private)
        or plan.get("content_sha256") != _content_sha256(plan)
        or public.get("private_unblinding_sha256") != private.get("content_sha256")
        or public.get("pack_id") != private.get("pack_id")
        or public.get("provenance", {}).get("population_sha256")
        != private.get("population_sha256")
    ):
        raise V4StudyError("V4 study artifact binding failed")
    if {"control", "challenger", "method_bindings"} & _nested_keys(public):
        raise V4StudyError("V4 public artifact leaks method identity")

    tasks = public.get("tasks")
    tracks = public.get("tracks")
    private_tasks = private.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != UNIQUE_TASKS + ANCHOR_TASKS
        or not isinstance(private_tasks, list)
        or len(private_tasks) != len(tasks)
        or not isinstance(tracks, Mapping)
    ):
        raise V4StudyError("V4 study task coverage is invalid")
    if [task.get("priority_rank") for task in tasks] != list(
        range(1, len(tasks) + 1)
    ):
        raise V4StudyError("V4 study priorities are invalid")
    task_by_id = {str(task.get("task_id")): task for task in tasks}
    private_by_id = {str(task.get("task_id")): task for task in private_tasks}
    if len(task_by_id) != len(tasks) or set(task_by_id) != set(private_by_id):
        raise V4StudyError("V4 study task identities are invalid")

    used_tracks = set()
    choice_ids = set()
    for task in tasks:
        seed_id = int(task["seed_track_id"])
        candidates = task.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != CANDIDATES_PER_TASK:
            raise V4StudyError("V4 active task must contain four candidates")
        candidate_ids = [int(row["track_id"]) for row in candidates]
        current_choice_ids = [str(row["choice_id"]) for row in candidates]
        if (
            len(set(candidate_ids)) != CANDIDATES_PER_TASK
            or len(set(current_choice_ids)) != CANDIDATES_PER_TASK
            or choice_ids.intersection(current_choice_ids)
            or seed_id in candidate_ids
        ):
            raise V4StudyError("V4 active task choices are invalid")
        choice_ids.update(current_choice_ids)
        used_tracks.update([seed_id, *candidate_ids])
        records = [tracks.get(str(track_id)) for track_id in [seed_id, *candidate_ids]]
        if any(not isinstance(record, Mapping) for record in records):
            raise V4StudyError("V4 task references an absent track")
        artist_ids = [
            int(record["source_identity"]["artist_id"]) for record in records
        ]
        if len(set(artist_ids)) != len(artist_ids):
            raise V4StudyError("V4 task repeats an artist")
        if gate_cache is not None:
            query_state, query_language = _effective_gate(
                seed_id, UNKNOWN, gate_cache
            )
            for candidate_id in candidate_ids:
                candidate_state, candidate_language = _effective_gate(
                    candidate_id, UNKNOWN, gate_cache
                )
                if not compatibility_allowed(
                    query_state,
                    candidate_state,
                    query_language,
                    candidate_language,
                ):
                    raise V4StudyError("V4 task retains a known gate mismatch")
    if set(tracks) != {str(track_id) for track_id in used_tracks}:
        raise V4StudyError("V4 public track coverage is not exact")

    anchors = [task for task in private_tasks if "anchor_of" in task]
    if len(anchors) != ANCHOR_TASKS:
        raise V4StudyError("V4 repeated-anchor coverage is invalid")
    for anchor in anchors:
        source = task_by_id.get(str(anchor.get("anchor_of")))
        current = task_by_id[str(anchor["task_id"])]
        if (
            source is None
            or current["seed_track_id"] != source["seed_track_id"]
            or [row["track_id"] for row in current["candidates"]]
            != [row["track_id"] for row in reversed(source["candidates"])]
        ):
            raise V4StudyError("V4 repeated anchor drift")


def _write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the V4 active study.")
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "population",
        "store_root",
        "predictor_model",
        "predictor_metadata",
        "semantic_cache",
        "semantic_cache_metadata",
        "vibe_cache",
        "blinding_key",
        "public_output",
        "private_output",
        "plan_output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--gate-cache", type=Path)
    parser.add_argument("--workers", type=int, default=12)
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
        semantic_metadata_path=args.semantic_cache_metadata,
        vibe_cache_path=args.vibe_cache,
        blinding_key_path=args.blinding_key,
        gate_cache_path=args.gate_cache,
        workers=args.workers,
    )
    _write(args.public_output, public)
    _write(args.private_output, private)
    _write(args.plan_output, plan)
    print(
        json.dumps(
            {
                "public_sha256": public["content_sha256"],
                "private_sha256": private["content_sha256"],
                "plan_sha256": plan["content_sha256"],
                "tasks": len(public["tasks"]),
                "tracks": len(public["tracks"]),
                "gate_tracks": len(plan["gate_track_ids"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
