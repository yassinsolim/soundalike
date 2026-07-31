"""Build the blinded full-track Jamendo v2 listening pilot.

This is a post-training, read-only workflow.  It opens the sealed 55,701-track
store, one official test fold, and one exact artifact from each trained family.
It never trains, changes production recommendation code, or writes audio.

The public pack contains opaque list identifiers and keyed commitments.  The
method map and commitment key are written only to an explicitly separate
private path.  That lets an analyst prove the exact model/ranking binding after
ratings are frozen without exposing model identity to a listener beforehand.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import secrets
import statistics
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_eval import (
    HYBRID_WEIGHTS,
    OFFICIAL_FOLDS,
    _score_trained_candidate_pool,
    _trained_result_model_binding,
    batch_fixed_budget_maxsim,
    freeze_fixed_budget,
    freeze_ranked_section_budget,
    load_trained_model_for_fold,
)
from .fulltrack_extract import normalize_rows
from .fulltrack_fusion import CANDIDATE_KINDS
from .fulltrack_store import (
    STORE_SCHEMA_VERSION,
    FullTrackStoreReader,
    stable_json_sha256,
)
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    JamendoContext,
    JamendoTrack,
    load_jamendo_context,
)


PILOT_SCHEMA_VERSION = 2
PILOT_PACK_KIND = "fulltrack_jamendo_blind_pilot_v2"
PRIVATE_KIND = "fulltrack_jamendo_blind_pilot_v2_private_unblinding"
MODEL_FAMILIES = tuple(CANDIDATE_KINDS)
METHODS = ("frozen_hybrid",) + MODEL_FAMILIES
RESULTS_PER_METHOD = 5
PILOT_MODEL_SEED = 17
PILOT_MAXSIM_BUDGET = 8
PUBLIC_AUDIO_HOST = "prod-1.storage.jamendo.com"
PUBLIC_AUDIO_FORMAT = "mp31"
PILOT_USE_NOTICE = (
    "verbatim playback for this non-commercial research pilot from "
    "Jamendo's first-party HTTPS stream with track-specific attribution; "
    "no audio redistribution by this repository"
)
TEMPO_BINS = (
    ("slow", 0.0, 90.0),
    ("medium", 90.0, 130.0),
    ("fast", 130.0, math.inf),
)
PILOT_SCENES = (
    "genre---alternative",
    "genre---ambient",
    "genre---classical",
    "genre---dance",
    "genre---electronic",
    "genre---experimental",
    "genre---folk",
    "genre---funk",
    "genre---hiphop",
    "genre---house",
    "genre---indie",
    "genre---jazz",
    "genre---metal",
    "genre---pop",
    "genre---reggae",
    "genre---rock",
    "genre---soundtrack",
    "genre---techno",
    "genre---trance",
    "genre---world",
)
_HEX64 = frozenset("0123456789abcdef")


class FullTrackPilotError(RuntimeError):
    """Unsafe, inconsistent, unlicensed, unblinded, or non-reproducible pilot."""


@dataclass(frozen=True)
class PilotConfig:
    fold_index: int = 0
    part: str = "test"
    seed_count: int = 20
    results_per_method: int = RESULTS_PER_METHOD
    candidate_pool: int = 200
    maxsim_budget: int = PILOT_MAXSIM_BUDGET
    model_seed: int = PILOT_MODEL_SEED
    shortlist_per_scene: int = 8
    texture_regions: int = 5

    def validate(self) -> None:
        if self.fold_index not in OFFICIAL_FOLDS:
            raise FullTrackPilotError("pilot fold must be an official fold")
        if self.part != "test":
            raise FullTrackPilotError("the pilot may select only held-out test tracks")
        if self.seed_count != len(PILOT_SCENES) or self.seed_count != 20:
            raise FullTrackPilotError("the v2 pilot requires exactly 20 scenes/seeds")
        if self.results_per_method != RESULTS_PER_METHOD:
            raise FullTrackPilotError("the v2 pilot requires five outputs per method")
        for label, value in (
            ("candidate_pool", self.candidate_pool),
            ("maxsim_budget", self.maxsim_budget),
            ("shortlist_per_scene", self.shortlist_per_scene),
            ("texture_regions", self.texture_regions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FullTrackPilotError(f"{label} must be a positive integer")
        if self.candidate_pool < self.results_per_method:
            raise FullTrackPilotError("candidate_pool is smaller than the output list")
        if self.texture_regions > self.seed_count:
            raise FullTrackPilotError("texture_regions exceeds the seed count")
        if self.maxsim_budget != PILOT_MAXSIM_BUDGET:
            raise FullTrackPilotError("pilot MaxSim budget must match the sealed artifacts")
        if self.model_seed != PILOT_MODEL_SEED:
            raise FullTrackPilotError("pilot model seed must match the sealed artifacts")


@dataclass(frozen=True)
class SeedCandidate:
    track_id: int
    artist_id: int
    tags: Tuple[str, ...]
    tempo_bpm: float
    texture_region: int


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FullTrackPilotError("pilot document is not canonical JSON") from exc


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - _HEX64)
    )


def _valid_opaque_id(value: object, prefix: str) -> bool:
    marker = f"{prefix}-"
    return (
        isinstance(value, str)
        and value.startswith(marker)
        and len(value) == len(marker) + 24
        and not (set(value[len(marker) :]) - _HEX64)
    )


def _tempo_bin(tempo: float) -> str:
    if not math.isfinite(tempo) or tempo <= 0:
        raise FullTrackPilotError("tempo values must be finite and positive")
    return next(name for name, low, high in TEMPO_BINS if low < tempo <= high)


def lawful_stream_url(track_id: int) -> str:
    if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id <= 0:
        raise FullTrackPilotError("Jamendo track ID must be a positive integer")
    return (
        f"https://{PUBLIC_AUDIO_HOST}/?trackid={track_id}"
        f"&format={PUBLIC_AUDIO_FORMAT}"
    )


def _normalized_embeddings(embeddings: np.ndarray, count: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != count or matrix.shape[1] < 2:
        raise FullTrackPilotError("CLAP embedding matrix shape is invalid")
    if not np.all(np.isfinite(matrix)):
        raise FullTrackPilotError("CLAP embedding matrix contains non-finite values")
    return normalize_rows(matrix)


def assign_texture_regions(
    track_ids: Sequence[int],
    embeddings: np.ndarray,
    region_count: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Assign deterministic farthest-first CLAP texture regions."""
    matrix = _normalized_embeddings(embeddings, len(track_ids))
    if not 1 <= region_count <= len(track_ids):
        raise FullTrackPilotError("invalid texture region count")
    ids = np.asarray([int(value) for value in track_ids], dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise FullTrackPilotError("texture input track IDs must be unique")

    centroid = normalize_rows(np.mean(matrix, axis=0, keepdims=True))[0]
    first = int(np.lexsort((ids, matrix @ centroid))[0])
    anchors = [first]
    nearest = 1.0 - np.clip(matrix @ matrix[first], -1.0, 1.0)
    while len(anchors) < region_count:
        nearest[np.asarray(anchors, dtype=np.int64)] = -1.0
        maximum = float(np.max(nearest))
        tied = np.flatnonzero(np.isclose(nearest, maximum, rtol=0.0, atol=1e-8))
        chosen = int(tied[np.argmin(ids[tied])])
        anchors.append(chosen)
        distance = 1.0 - np.clip(matrix @ matrix[chosen], -1.0, 1.0)
        nearest = np.minimum(nearest, distance)

    similarities = matrix @ matrix[np.asarray(anchors, dtype=np.int64)].T
    regions = np.argmax(similarities, axis=1)
    return tuple(int(value) for value in regions), tuple(
        int(ids[index]) for index in anchors
    )


def build_tempo_shortlist(
    tracks: Sequence[JamendoTrack],
    tags_by_id: Mapping[int, Sequence[str]],
    embeddings: np.ndarray,
    *,
    per_scene: int,
) -> Tuple[int, ...]:
    """Choose bounded CLAP-diverse tempo-analysis candidates for every scene."""
    matrix = _normalized_embeddings(embeddings, len(tracks))
    if per_scene <= 0:
        raise FullTrackPilotError("per_scene must be positive")
    selected: set[int] = set()
    ids = np.asarray([int(track.track_id) for track in tracks], dtype=np.int64)
    for scene in PILOT_SCENES:
        positions = np.asarray(
            [
                index
                for index, track in enumerate(tracks)
                if scene in set(tags_by_id.get(int(track.track_id), ()))
            ],
            dtype=np.int64,
        )
        if not len(positions):
            raise FullTrackPilotError(f"held-out fold has no candidate for {scene}")
        scene_matrix = matrix[positions]
        centroid = normalize_rows(np.mean(scene_matrix, axis=0, keepdims=True))[0]
        first_local = int(
            np.lexsort((ids[positions], scene_matrix @ centroid))[0]
        )
        chosen = [first_local]
        nearest = 1.0 - np.clip(
            scene_matrix @ scene_matrix[first_local], -1.0, 1.0
        )
        while len(chosen) < min(per_scene, len(positions)):
            nearest[np.asarray(chosen, dtype=np.int64)] = -1.0
            maximum = float(np.max(nearest))
            tied = np.flatnonzero(
                np.isclose(nearest, maximum, rtol=0.0, atol=1e-8)
            )
            next_local = int(tied[np.argmin(ids[positions[tied]])])
            chosen.append(next_local)
            nearest = np.minimum(
                nearest,
                1.0
                - np.clip(
                    scene_matrix @ scene_matrix[next_local],
                    -1.0,
                    1.0,
                ),
            )
        selected.update(int(ids[positions[index]]) for index in chosen)
    return tuple(sorted(selected))


def select_diverse_seeds(
    candidates: Sequence[SeedCandidate],
    embeddings: np.ndarray,
) -> Tuple[SeedCandidate, ...]:
    """Select one unique-artist seed per scene with tempo/texture coverage."""
    matrix = _normalized_embeddings(embeddings, len(candidates))
    if len({item.track_id for item in candidates}) != len(candidates):
        raise FullTrackPilotError("seed candidates contain duplicate tracks")
    positions = {item.track_id: index for index, item in enumerate(candidates)}
    selected: list[SeedCandidate] = []
    selected_positions: list[int] = []
    used_tracks: set[int] = set()
    used_artists: set[int] = set()
    covered_tags: set[str] = set()

    for scene_index, scene in enumerate(PILOT_SCENES):
        target_tempo = TEMPO_BINS[scene_index % len(TEMPO_BINS)][0]
        target_region = scene_index % 5
        available = [
            item
            for item in candidates
            if scene in item.tags
            and item.track_id not in used_tracks
            and item.artist_id not in used_artists
        ]
        if not available:
            raise FullTrackPilotError(
                f"cannot select a unique-track/artist seed for {scene}"
            )
        preferred = [
            item
            for item in available
            if _tempo_bin(item.tempo_bpm) == target_tempo
            and item.texture_region == target_region
        ]
        if not preferred:
            preferred = [
                item
                for item in available
                if item.texture_region
                not in {value.texture_region for value in selected}
                and _tempo_bin(item.tempo_bpm) == target_tempo
            ]
        if not preferred:
            preferred = [
                item
                for item in available
                if item.texture_region
                not in {value.texture_region for value in selected}
            ]
        if not preferred:
            preferred = [
                item
                for item in available
                if _tempo_bin(item.tempo_bpm) == target_tempo
            ]
        if not preferred:
            preferred = available

        scored = []
        for item in preferred:
            position = positions[item.track_id]
            if selected_positions:
                texture_distance = float(
                    np.min(
                        1.0
                        - np.clip(
                            matrix[selected_positions] @ matrix[position],
                            -1.0,
                            1.0,
                        )
                    )
                )
            else:
                texture_distance = 1.0
            novelty = len(set(item.tags) - covered_tags) / max(1, len(item.tags))
            region_novelty = float(
                item.texture_region not in {value.texture_region for value in selected}
            )
            scored.append(
                (
                    -(0.65 * texture_distance + 0.25 * novelty + 0.10 * region_novelty),
                    int(item.track_id),
                    item,
                )
            )
        chosen = min(scored)[2]
        selected.append(chosen)
        selected_positions.append(positions[chosen.track_id])
        used_tracks.add(chosen.track_id)
        used_artists.add(chosen.artist_id)
        covered_tags.update(chosen.tags)

    if len(selected) != 20:
        raise FullTrackPilotError("seed selection did not produce exactly 20 seeds")
    if {item.texture_region for item in selected} != set(range(5)):
        raise FullTrackPilotError(
            "seed selection did not cover all five CLAP texture regions"
        )
    if {_tempo_bin(item.tempo_bpm) for item in selected} != {
        name for name, _, _ in TEMPO_BINS
    }:
        raise FullTrackPilotError("seed selection did not cover all tempo regions")
    return tuple(selected)


def diversity_evidence(
    selected: Sequence[SeedCandidate],
    embeddings: np.ndarray,
    *,
    texture_anchor_track_ids: Sequence[int],
) -> Mapping[str, object]:
    matrix = _normalized_embeddings(embeddings, len(selected))
    tempos = [float(item.tempo_bpm) for item in selected]
    distances = [
        float(1.0 - np.clip(np.dot(matrix[left], matrix[right]), -1.0, 1.0))
        for left in range(len(matrix))
        for right in range(left + 1, len(matrix))
    ]
    tempo_counts = {
        name: sum(_tempo_bin(value) == name for value in tempos)
        for name, _, _ in TEMPO_BINS
    }
    region_counts = {
        str(region): sum(item.texture_region == region for item in selected)
        for region in sorted({item.texture_region for item in selected})
    }
    return {
        "scene_count": len(PILOT_SCENES),
        "scenes": [scene.removeprefix("genre---") for scene in PILOT_SCENES],
        "unique_seed_tracks": len({item.track_id for item in selected}),
        "unique_seed_artists": len({item.artist_id for item in selected}),
        "unique_tags": len({tag for item in selected for tag in item.tags}),
        "tempo_bpm": {
            "minimum": min(tempos),
            "median": float(statistics.median(tempos)),
            "maximum": max(tempos),
            "bin_boundaries": {
                "slow": "(0,90]",
                "medium": "(90,130]",
                "fast": "(130,+inf)",
            },
            "bin_counts": tempo_counts,
            "measurement": (
                "librosa beat tracking over the first 30 seconds of the verified "
                "local official source; machine descriptor, not a listening judgment"
            ),
        },
        "clap_texture": {
            "region_method": "deterministic farthest-first cosine anchors",
            "anchor_track_ids": [int(value) for value in texture_anchor_track_ids],
            "region_counts": region_counts,
            "pairwise_cosine_distance": {
                "minimum": min(distances),
                "median": float(statistics.median(distances)),
                "maximum": max(distances),
            },
        },
    }


def measure_tempo_bpm(track: JamendoTrack) -> float:
    """Compute a deterministic machine tempo descriptor without retaining audio."""
    from soundalike.audio.features import features_from_file

    value = float(features_from_file(str(track.audio_path)).tempo)
    if not math.isfinite(value) or value <= 0.0 or value > 400.0:
        raise FullTrackPilotError(
            f"invalid tempo descriptor for Jamendo track {track.track_id}"
        )
    return value


def verify_public_audio_url(url: str, timeout: float = 30.0) -> Mapping[str, object]:
    """Verify a first-party full-track response using HEAD; no audio is downloaded."""
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "soundalike-fulltrack-pilot-v2/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final = response.geturl()
            status = int(response.status)
            content_type = response.headers.get_content_type()
            raw_length = response.headers.get("Content-Length")
            accept_ranges = response.headers.get("Accept-Ranges")
    except Exception as exc:
        raise FullTrackPilotError("Jamendo audio HEAD verification failed") from exc
    parsed = urllib.parse.urlsplit(final)
    if (
        status != 200
        or parsed.scheme != "https"
        or parsed.hostname != PUBLIC_AUDIO_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or content_type != "audio/mpeg"
        or raw_length is None
        or not raw_length.isascii()
        or not raw_length.isdigit()
        or int(raw_length) < 100_000
        or accept_ranges != "bytes"
    ):
        raise FullTrackPilotError("Jamendo audio response is not a bounded MP3 stream")
    return {
        "status": status,
        "content_type": content_type,
        "content_length": int(raw_length),
        "accept_ranges": accept_ranges,
        "final_host": PUBLIC_AUDIO_HOST,
    }


def verify_public_audio_urls(
    urls: Iterable[str],
    *,
    workers: int = 8,
    verifier: Callable[[str], Mapping[str, object]] = verify_public_audio_url,
) -> Mapping[str, Mapping[str, object]]:
    ordered = tuple(sorted(set(urls)))
    if not ordered:
        raise FullTrackPilotError("no public audio URLs were supplied")
    if not 1 <= workers <= 16:
        raise FullTrackPilotError("audio verification workers must be in [1,16]")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = tuple(pool.map(verifier, ordered))
    return {url: dict(result) for url, result in zip(ordered, results)}


def _license_record(track: JamendoTrack) -> Mapping[str, object]:
    license_url = str(track.license.url)
    parsed = urllib.parse.urlsplit(license_url)
    host = (parsed.hostname or "").casefold()
    creative_commons = host in {
        "creativecommons.org",
        "www.creativecommons.org",
    } and parsed.path.startswith("/licenses/")
    art_libre = host in {"artlibre.org", "www.artlibre.org"} and (
        "licence/lal" in parsed.path.casefold()
    )
    if (
        parsed.scheme not in {"http", "https"}
        or not (creative_commons or art_libre)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 80, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise FullTrackPilotError(
            f"track {track.track_id} lacks a verified public-use license"
        )
    if track.license.path != track.relative_path:
        raise FullTrackPilotError("license/source path identity drift")
    secure_license_url = urllib.parse.urlunsplit(
        ("https", host.removeprefix("www."), parsed.path, "", "")
    )
    return {
        "name": track.license.name,
        "url": secure_license_url,
        "attribution": track.license.attribution.replace("http://", "https://"),
        "permits_commercial_use": bool(track.license.permits_commercial_use),
        "permits_derivatives": bool(track.license.permits_derivatives),
        "pilot_use": PILOT_USE_NOTICE,
    }


def _validate_diversity_evidence(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "scene_count",
        "scenes",
        "unique_seed_tracks",
        "unique_seed_artists",
        "unique_tags",
        "tempo_bpm",
        "clap_texture",
    }:
        raise FullTrackPilotError("pilot diversity evidence is missing or unexpected")
    tempo = value.get("tempo_bpm")
    texture = value.get("clap_texture")
    unique_tags = value.get("unique_tags")
    if (
        value.get("scene_count") != 20
        or value.get("scenes")
        != [scene.removeprefix("genre---") for scene in PILOT_SCENES]
        or value.get("unique_seed_tracks") != 20
        or value.get("unique_seed_artists") != 20
        or isinstance(unique_tags, bool)
        or not isinstance(unique_tags, int)
        or unique_tags < 20
        or not isinstance(tempo, Mapping)
        or set(tempo)
        != {
            "minimum",
            "median",
            "maximum",
            "bin_boundaries",
            "bin_counts",
            "measurement",
        }
        or not isinstance(texture, Mapping)
        or set(texture)
        != {
            "region_method",
            "anchor_track_ids",
            "region_counts",
            "pairwise_cosine_distance",
        }
    ):
        raise FullTrackPilotError("pilot scene diversity evidence is invalid")
    tempo_values = [tempo.get(name) for name in ("minimum", "median", "maximum")]
    if (
        any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 < float(item) <= 400.0
            for item in tempo_values
        )
        or not float(tempo_values[0])
        <= float(tempo_values[1])
        <= float(tempo_values[2])
        or tempo.get("bin_boundaries")
        != {"slow": "(0,90]", "medium": "(90,130]", "fast": "(130,+inf)"}
        or tempo.get("measurement")
        != (
            "librosa beat tracking over the first 30 seconds of the verified "
            "local official source; machine descriptor, not a listening judgment"
        )
    ):
        raise FullTrackPilotError("pilot tempo diversity evidence is invalid")
    bin_counts = tempo.get("bin_counts")
    if (
        not isinstance(bin_counts, Mapping)
        or set(bin_counts) != {name for name, _, _ in TEMPO_BINS}
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in bin_counts.values()
        )
        or sum(bin_counts.values()) != 20
    ):
        raise FullTrackPilotError("pilot tempo diversity evidence is invalid")
    region_counts = texture.get("region_counts")
    anchors = texture.get("anchor_track_ids")
    distances = texture.get("pairwise_cosine_distance")
    if (
        texture.get("region_method")
        != "deterministic farthest-first cosine anchors"
        or not isinstance(region_counts, Mapping)
        or set(region_counts) != {str(index) for index in range(5)}
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in region_counts.values()
        )
        or sum(region_counts.values()) != 20
        or not isinstance(anchors, list)
        or len(anchors) != 5
        or len(set(anchors)) != 5
        or any(
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id <= 0
            for track_id in anchors
        )
        or not isinstance(distances, Mapping)
        or set(distances) != {"minimum", "median", "maximum"}
    ):
        raise FullTrackPilotError("pilot CLAP texture diversity evidence is invalid")
    distance_values = [
        distances.get(name) for name in ("minimum", "median", "maximum")
    ]
    if (
        any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 2.0
            for item in distance_values
        )
        or not float(distance_values[0])
        <= float(distance_values[1])
        <= float(distance_values[2])
    ):
        raise FullTrackPilotError("pilot CLAP texture distances are invalid")
    return value


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
        "fold",
        "fold_part",
        "evidence_scope",
        "source_fingerprint",
        "store_binding",
        "store_binding_sha256",
        "blinding",
        "audio_delivery",
        "diversity_evidence",
        "tracks",
        "seeds",
        "research_only",
        "promotion_allowed",
        "notice",
        "content_sha256",
    }
)
_PRIVATE_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "pack_id",
        "source_fingerprint",
        "store_binding",
        "store_binding_sha256",
        "fold",
        "fold_part",
        "blinding_key_hex",
        "blinding_key_sha256",
        "methods",
        "seeds",
        "public_evidence_commitment_sha256",
        "research_only",
        "promotion_allowed",
        "content_sha256",
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
_RESULT_ID_KEYS = frozenset({"result_id", "track_id"})
_RANKING_ROW_KEYS = frozenset({"position", "result_id", "track_id"})
_STORE_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "source_fingerprint",
        "config_sha256",
        "model_sha256",
        "model_id",
        "embedding_dim",
        "track_count",
        "shard_tracks",
        "repetition_sections",
        "salient_sections",
        "track_plan_sha256",
        "sealed_manifest_sha256",
    }
)


def _validate_store_binding(
    binding: object, *, expected_source_fingerprint: object
) -> Mapping[str, object]:
    if not isinstance(binding, Mapping) or set(binding) != _STORE_BINDING_KEYS:
        raise FullTrackPilotError("sealed store binding has unexpected fields")
    if (
        binding.get("schema_version") != STORE_SCHEMA_VERSION
        or binding.get("track_count") != 55_701
        or binding.get("source_fingerprint") != expected_source_fingerprint
        or not _valid_hash(expected_source_fingerprint)
        or not isinstance(binding.get("model_id"), str)
        or not binding["model_id"]
    ):
        raise FullTrackPilotError("sealed store identity is invalid")
    for field in (
        "config_sha256",
        "model_sha256",
        "track_plan_sha256",
        "sealed_manifest_sha256",
    ):
        if not _valid_hash(binding.get(field)):
            raise FullTrackPilotError("sealed store checksum binding is invalid")
    for field in (
        "embedding_dim",
        "shard_tracks",
        "repetition_sections",
        "salient_sections",
    ):
        value = binding.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FullTrackPilotError("sealed store dimensions are invalid")
    if binding["repetition_sections"] < 8 or binding["salient_sections"] < 8:
        raise FullTrackPilotError("sealed store section budgets are too small")
    return binding


def _track_record(
    track: JamendoTrack,
    *,
    fold_index: int,
    fold_part: str,
    store_row: int,
    audio_verification: Mapping[str, object],
) -> Mapping[str, object]:
    audio_url = lawful_stream_url(int(track.track_id))
    if not _valid_hash(track.expected_audio_sha256):
        raise FullTrackPilotError("official source audio hash is invalid")
    return {
        "track_id": int(track.track_id),
        "source_identity": {
            "artist_id": int(track.artist_id),
            "album_id": int(track.album_id),
            "relative_path": track.relative_path,
            "source_audio_sha256": track.expected_audio_sha256,
            "source_audio_bytes": int(track.expected_audio_bytes),
            "store_row": int(store_row),
            "fold": int(fold_index),
            "fold_part": fold_part,
        },
        "title": track.title,
        "artist": track.artist_name,
        "album": track.album_name,
        "jamendo_url": track.jamendo_url.replace("http://", "https://", 1),
        "audio": {
            "url": audio_url,
            "delivery": "Jamendo first-party full-track MP3",
            "verification": dict(audio_verification),
        },
        "license": _license_record(track),
    }


def _opaque_id(key: bytes, prefix: str, *parts: object) -> str:
    message = "\0".join([PILOT_PACK_KIND, prefix, *(str(part) for part in parts)])
    digest = hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _validate_method_binding(
    value: object,
    *,
    expected_method: str,
    expected_source_fingerprint: str,
    expected_store_binding_sha256: str,
    expected_fold: int,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or value.get("method") != expected_method:
        raise FullTrackPilotError("private method binding identity is invalid")
    if expected_method == "frozen_hybrid":
        if (
            set(value) != {"method", "definition", "trained", "promoted"}
            or value.get("definition") != {
                "global_cosine": HYBRID_WEIGHTS["global_cosine"],
                "uniform_window_maxsim": HYBRID_WEIGHTS[
                    "uniform_window_maxsim"
                ],
                "section_maxsim": HYBRID_WEIGHTS["section_maxsim"],
            }
            or value.get("trained") is not False
            or value.get("promoted") is not False
        ):
            raise FullTrackPilotError("frozen hybrid binding drift")
        return value

    artifact = value.get("artifact")
    artifact_fields = {
        "candidate_kind",
        "seed",
        "fold_index",
        "ablation",
        "model_artifact_sha256",
        "model_json_sha256",
        "weights_npz_sha256",
        "report_sha256",
        "source_fingerprint",
        "store_binding_sha256",
        "training_config_sha256",
        "job_config_sha256",
        "maxsim_budget",
        "embedding_dim",
        "promoted",
    }
    if (
        set(value) != {"method", "trained", "promoted", "artifact"}
        or value.get("trained") is not True
        or value.get("promoted") is not False
        or not isinstance(artifact, Mapping)
        or set(artifact) != artifact_fields
        or expected_method not in MODEL_FAMILIES
        or artifact.get("candidate_kind") != expected_method
        or artifact.get("fold_index") != expected_fold
        or artifact.get("ablation") != "none"
        or artifact.get("source_fingerprint") != expected_source_fingerprint
        or artifact.get("store_binding_sha256")
        != expected_store_binding_sha256
        or artifact.get("seed") != PILOT_MODEL_SEED
        or artifact.get("maxsim_budget") != PILOT_MAXSIM_BUDGET
        or artifact.get("promoted") is not False
    ):
        raise FullTrackPilotError("trained artifact identity/fold/store binding drift")
    for field in (
        "model_artifact_sha256",
        "model_json_sha256",
        "weights_npz_sha256",
        "report_sha256",
        "source_fingerprint",
        "store_binding_sha256",
        "training_config_sha256",
        "job_config_sha256",
    ):
        if not _valid_hash(artifact.get(field)):
            raise FullTrackPilotError("trained artifact checksum binding is invalid")
    for field in ("seed", "embedding_dim"):
        item = artifact.get(field)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < (0 if field == "seed" else 1)
        ):
            raise FullTrackPilotError("trained artifact dimensions are invalid")
    return value


def _method_binding(
    method: str,
    model_bindings: Mapping[str, object],
    *,
    source_fingerprint: str,
    store_binding_sha256: str,
    fold_index: int,
) -> Mapping[str, object]:
    if method == "frozen_hybrid":
        result = {
            "method": method,
            "definition": {
                "global_cosine": HYBRID_WEIGHTS["global_cosine"],
                "uniform_window_maxsim": HYBRID_WEIGHTS["uniform_window_maxsim"],
                "section_maxsim": HYBRID_WEIGHTS["section_maxsim"],
            },
            "trained": False,
            "promoted": False,
        }
        _validate_method_binding(
            result,
            expected_method=method,
            expected_source_fingerprint=source_fingerprint,
            expected_store_binding_sha256=store_binding_sha256,
            expected_fold=fold_index,
        )
        return result
    binding = model_bindings.get(method)
    if binding is None:
        raise FullTrackPilotError(f"missing exact model binding for {method}")
    result = {
        "method": method,
        "trained": True,
        "promoted": False,
        "artifact": _trained_result_model_binding(
            {
                "candidate_kind": binding.candidate_kind,
                "seed": binding.seed,
                "fold_index": binding.fold_index,
                "report_sha256": binding.report_sha256,
                "model_artifact_sha256": binding.model_artifact_sha256,
                "model_json_sha256": binding.model_json_sha256,
                "weights_npz_sha256": binding.weights_npz_sha256,
                "source_fingerprint": binding.source_fingerprint,
                "store_binding_sha256": binding.store_binding_sha256,
                "training_config_sha256": binding.training_config_sha256,
                "job_config_sha256": binding.job_config_sha256,
                "maxsim_budget": binding.maxsim_budget,
                "embedding_dim": binding.embedding_dim,
            },
            "none",
        ),
    }
    _validate_method_binding(
        result,
        expected_method=method,
        expected_source_fingerprint=source_fingerprint,
        expected_store_binding_sha256=store_binding_sha256,
        expected_fold=fold_index,
    )
    return result


def _seed_evidence_descriptors(
    seeds: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Return the public seed descriptors protected by the private commitment."""
    descriptors = []
    for seed in seeds:
        if not isinstance(seed, Mapping):
            raise FullTrackPilotError("public seed descriptor is invalid")
        descriptors.append(
            {
                "seed_id": seed.get("seed_id"),
                "seed_track_id": seed.get("seed_track_id"),
                "scene": seed.get("scene"),
                "tempo_bpm": seed.get("tempo_bpm"),
                "tempo_region": seed.get("tempo_region"),
                "clap_texture_region": seed.get("clap_texture_region"),
            }
        )
    return descriptors


def build_blinded_documents(
    *,
    rankings: Mapping[int, Mapping[str, Sequence[int]]],
    selected_seeds: Sequence[SeedCandidate],
    tracks_by_id: Mapping[int, JamendoTrack],
    store_rows: Mapping[int, int],
    fold_track_parts: Mapping[int, str],
    store_binding: Mapping[str, object],
    source_fingerprint: str,
    fold_index: int,
    model_bindings: Mapping[str, object],
    blinding_key: bytes,
    diversity: Mapping[str, object],
    audio_verification: Mapping[str, Mapping[str, object]],
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    """Create a public pack and private exact method map with keyed commitments."""
    if len(blinding_key) != 32:
        raise FullTrackPilotError("blinding key must contain exactly 32 bytes")
    if len(selected_seeds) != 20:
        raise FullTrackPilotError("exactly 20 selected seeds are required")
    if fold_index not in OFFICIAL_FOLDS:
        raise FullTrackPilotError("pilot fold must be an official fold")
    if tuple(sorted(model_bindings)) != tuple(sorted(MODEL_FAMILIES)):
        raise FullTrackPilotError("exactly one artifact from each trained family is required")
    _validate_store_binding(
        store_binding, expected_source_fingerprint=source_fingerprint
    )
    _validate_diversity_evidence(diversity)
    store_binding_sha256 = stable_json_sha256(store_binding)

    all_track_ids = {
        int(seed.track_id) for seed in selected_seeds
    } | {
        int(track_id)
        for methods in rankings.values()
        for ranked in methods.values()
        for track_id in ranked
    }
    if not all_track_ids.issubset(tracks_by_id):
        raise FullTrackPilotError("a ranked source track is missing metadata")
    if any(fold_track_parts.get(track_id) != "test" for track_id in all_track_ids):
        raise FullTrackPilotError("a pilot track is not in the official held-out fold")
    if not all_track_ids.issubset(store_rows):
        raise FullTrackPilotError("a pilot track is missing from the sealed store")
    selected_store_rows = [store_rows[track_id] for track_id in all_track_ids]
    if (
        any(
            isinstance(row, bool) or not isinstance(row, int) or row < 0
            for row in selected_store_rows
        )
        or len(set(selected_store_rows)) != len(selected_store_rows)
    ):
        raise FullTrackPilotError("pilot track/store-row identity drift")
    track_records = {}
    for track_id in sorted(all_track_ids):
        track = tracks_by_id[track_id]
        url = lawful_stream_url(track_id)
        verification = audio_verification.get(url)
        if verification is None:
            raise FullTrackPilotError("a pilot track lacks public-audio verification")
        track_records[str(track_id)] = _track_record(
            track,
            fold_index=fold_index,
            fold_part="test",
            store_row=store_rows[track_id],
            audio_verification=verification,
        )

    private_seeds = []
    public_seeds = []
    seen_lists: set[str] = set()
    for scene_index, seed in enumerate(selected_seeds):
        methods = rankings.get(seed.track_id)
        if methods is None or set(methods) != set(METHODS):
            raise FullTrackPilotError("each seed must contain all four methods")
        seed_id = _opaque_id(blinding_key, "seed", seed.track_id)
        result_ids: Dict[int, str] = {}
        for ranked in methods.values():
            if len(ranked) != RESULTS_PER_METHOD or len(set(ranked)) != len(ranked):
                raise FullTrackPilotError("every method needs five unique ranked outputs")
            for track_id in ranked:
                result_ids.setdefault(
                    int(track_id),
                    _opaque_id(blinding_key, "result", seed.track_id, track_id),
                )

        public_lists = []
        private_lists = []
        for method in METHODS:
            method_binding = _method_binding(
                method,
                model_bindings,
                source_fingerprint=source_fingerprint,
                store_binding_sha256=store_binding_sha256,
                fold_index=fold_index,
            )
            ranking = [
                {
                    "position": position,
                    "result_id": result_ids[int(track_id)],
                    "track_id": int(track_id),
                }
                for position, track_id in enumerate(methods[method], 1)
            ]
            list_id = _opaque_id(blinding_key, "list", seed.track_id, method)
            if list_id in seen_lists:
                raise FullTrackPilotError("opaque list ID collision")
            seen_lists.add(list_id)
            commitment_payload = {
                "pack_kind": PILOT_PACK_KIND,
                "seed_id": seed_id,
                "list_id": list_id,
                "method_binding": method_binding,
                "ranking_track_ids": [row["track_id"] for row in ranking],
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
                    "ranking": ranking,
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
        public_seeds.append(
            {
                "seed_id": seed_id,
                "scene": PILOT_SCENES[scene_index].removeprefix("genre---"),
                "seed_track_id": int(seed.track_id),
                "tempo_bpm": float(seed.tempo_bpm),
                "tempo_region": _tempo_bin(seed.tempo_bpm),
                "clap_texture_region": int(seed.texture_region),
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
                "seed_track_id": int(seed.track_id),
                "lists": private_lists,
            }
        )

    evidence_payload = {
        "source_fingerprint": source_fingerprint,
        "store_binding_sha256": store_binding_sha256,
        "fold": fold_index,
        "fold_part": "test",
        "diversity_evidence": dict(diversity),
        "seed_descriptors": _seed_evidence_descriptors(public_seeds),
        "tracks": track_records,
    }
    evidence_commitment = hmac.new(
        blinding_key, _canonical_bytes(evidence_payload), hashlib.sha256
    ).hexdigest()
    private_document: Dict[str, object] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "artifact_kind": PRIVATE_KIND,
        "pack_id": "fulltrack-v2-pilot-20",
        "source_fingerprint": source_fingerprint,
        "store_binding": dict(store_binding),
        "store_binding_sha256": stable_json_sha256(store_binding),
        "fold": fold_index,
        "fold_part": "test",
        "blinding_key_hex": blinding_key.hex(),
        "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
        "methods": list(METHODS),
        "seeds": private_seeds,
        "public_evidence_commitment_sha256": evidence_commitment,
        "research_only": True,
        "promotion_allowed": False,
    }
    private_document["content_sha256"] = _content_sha256(private_document)

    public_document: Dict[str, object] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "pack_kind": PILOT_PACK_KIND,
        "pack_id": "fulltrack-v2-pilot-20",
        "rankings_state": "LOCKED_BEFORE_RATINGS",
        "ratings_count_at_freeze": 0,
        "seed_count": 20,
        "method_count": 4,
        "results_per_method": RESULTS_PER_METHOD,
        "fold": fold_index,
        "fold_part": "test",
        "evidence_scope": EVIDENCE_SCOPE,
        "source_fingerprint": source_fingerprint,
        "store_binding": dict(store_binding),
        "store_binding_sha256": stable_json_sha256(store_binding),
        "blinding": {
            "opaque_per_seed_list_ids": True,
            "method_identity_public": False,
            "method_order_randomized_per_session": True,
            "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
            "private_unblinding_sha256": private_document["content_sha256"],
            "public_evidence_commitment_sha256": evidence_commitment,
        },
        "audio_delivery": {
            "kind": "Jamendo first-party full-track MP3",
            "host": PUBLIC_AUDIO_HOST,
            "verification": "HEAD 200, HTTPS, audio/mpeg, byte ranges, >=100000 bytes",
            "repository_contains_audio": False,
            "commercial_preview": False,
        },
        "diversity_evidence": dict(diversity),
        "tracks": track_records,
        "seeds": public_seeds,
        "research_only": True,
        "promotion_allowed": False,
        "notice": (
            "This pilot collects human research evidence. Automated metrics and one "
            "rater cannot promote a model or change production recommendations."
        ),
    }
    public_document["content_sha256"] = _content_sha256(public_document)
    validate_blinded_documents(public_document, private_document)
    return public_document, private_document


def _validate_track_document(
    key: object,
    value: object,
    *,
    expected_fold: int,
) -> int:
    if not isinstance(key, str) or not key.isascii() or not key.isdigit():
        raise FullTrackPilotError("public track key is invalid")
    track_id = int(key)
    if (
        track_id <= 0
        or not isinstance(value, Mapping)
        or set(value)
        != {
            "track_id",
            "source_identity",
            "title",
            "artist",
            "album",
            "jamendo_url",
            "audio",
            "license",
        }
        or value.get("track_id") != track_id
    ):
        raise FullTrackPilotError("public source track identity is invalid")
    source = value.get("source_identity")
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "artist_id",
            "album_id",
            "relative_path",
            "source_audio_sha256",
            "source_audio_bytes",
            "store_row",
            "fold",
            "fold_part",
        }
        or source.get("fold") != expected_fold
        or source.get("fold_part") != "test"
        or not _valid_hash(source.get("source_audio_sha256"))
    ):
        raise FullTrackPilotError("public source track fold/hash binding is invalid")
    for field in ("artist_id", "album_id", "source_audio_bytes"):
        item = source.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise FullTrackPilotError("public source track dimensions are invalid")
    store_row = source.get("store_row")
    if isinstance(store_row, bool) or not isinstance(store_row, int) or store_row < 0:
        raise FullTrackPilotError("public source store row is invalid")
    relative_path = source.get("relative_path")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or relative_path.startswith("/")
        or ":" in relative_path
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
    ):
        raise FullTrackPilotError("public source relative path is invalid")
    for field in ("title", "artist", "album"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise FullTrackPilotError("public track attribution text is invalid")

    jamendo = urllib.parse.urlsplit(str(value.get("jamendo_url", "")))
    if (
        jamendo.scheme != "https"
        or (jamendo.hostname or "").casefold()
        not in {"jamendo.com", "www.jamendo.com"}
        or jamendo.username is not None
        or jamendo.password is not None
        or jamendo.port not in (None, 443)
    ):
        raise FullTrackPilotError("public Jamendo track URL is invalid")
    audio = value.get("audio")
    verification = audio.get("verification") if isinstance(audio, Mapping) else None
    if (
        not isinstance(audio, Mapping)
        or set(audio) != {"url", "delivery", "verification"}
        or audio.get("url") != lawful_stream_url(track_id)
        or audio.get("delivery") != "Jamendo first-party full-track MP3"
        or not isinstance(verification, Mapping)
        or set(verification)
        != {
            "status",
            "content_type",
            "content_length",
            "accept_ranges",
            "final_host",
        }
        or verification.get("status") != 200
        or verification.get("content_type") != "audio/mpeg"
        or isinstance(verification.get("content_length"), bool)
        or not isinstance(verification.get("content_length"), int)
        or verification["content_length"] < 100_000
        or verification.get("accept_ranges") != "bytes"
        or verification.get("final_host") != PUBLIC_AUDIO_HOST
    ):
        raise FullTrackPilotError("public audio delivery evidence is invalid")
    license_record = value.get("license")
    if (
        not isinstance(license_record, Mapping)
        or set(license_record)
        != {
            "name",
            "url",
            "attribution",
            "permits_commercial_use",
            "permits_derivatives",
            "pilot_use",
        }
        or not isinstance(license_record.get("name"), str)
        or not license_record["name"].strip()
        or not isinstance(license_record.get("attribution"), str)
        or not license_record["attribution"].strip()
        or "http://" in license_record["attribution"].casefold()
        or not isinstance(license_record.get("permits_commercial_use"), bool)
        or not isinstance(license_record.get("permits_derivatives"), bool)
        or license_record.get("pilot_use") != PILOT_USE_NOTICE
    ):
        raise FullTrackPilotError("public license/attribution evidence is invalid")
    parsed_license = urllib.parse.urlsplit(str(license_record.get("url", "")))
    license_host = (parsed_license.hostname or "").casefold()
    if (
        parsed_license.scheme != "https"
        or (
            not (
                license_host == "creativecommons.org"
                and parsed_license.path.startswith("/licenses/")
            )
            and not (
                license_host == "artlibre.org"
                and "licence/lal" in parsed_license.path.casefold()
            )
        )
        or parsed_license.username is not None
        or parsed_license.password is not None
        or parsed_license.port not in (None, 443)
        or parsed_license.query
        or parsed_license.fragment
    ):
        raise FullTrackPilotError("public license authority/path is invalid")
    return store_row


def validate_blinded_documents(
    public_document: Mapping[str, object],
    private_document: Mapping[str, object],
) -> None:
    """Strictly verify public/private bindings and absence of public model identity."""
    if (
        set(public_document) != _PUBLIC_DOCUMENT_KEYS
        or public_document.get("schema_version") != PILOT_SCHEMA_VERSION
        or public_document.get("pack_kind") != PILOT_PACK_KIND
        or public_document.get("pack_id") != "fulltrack-v2-pilot-20"
        or public_document.get("rankings_state") != "LOCKED_BEFORE_RATINGS"
        or public_document.get("ratings_count_at_freeze") != 0
        or public_document.get("seed_count") != 20
        or public_document.get("method_count") != len(METHODS)
        or public_document.get("results_per_method") != RESULTS_PER_METHOD
        or public_document.get("fold") not in OFFICIAL_FOLDS
        or public_document.get("fold_part") != "test"
        or public_document.get("evidence_scope") != EVIDENCE_SCOPE
        or public_document.get("research_only") is not True
        or public_document.get("promotion_allowed") is not False
        or public_document.get("notice")
        != (
            "This pilot collects human research evidence. Automated metrics and one "
            "rater cannot promote a model or change production recommendations."
        )
        or _content_sha256(public_document) != public_document.get("content_sha256")
    ):
        raise FullTrackPilotError("public pilot document failed strict validation")
    if (
        set(private_document) != _PRIVATE_DOCUMENT_KEYS
        or private_document.get("schema_version") != PILOT_SCHEMA_VERSION
        or private_document.get("artifact_kind") != PRIVATE_KIND
        or private_document.get("pack_id") != "fulltrack-v2-pilot-20"
        or private_document.get("fold") not in OFFICIAL_FOLDS
        or private_document.get("fold_part") != "test"
        or private_document.get("methods") != list(METHODS)
        or private_document.get("research_only") is not True
        or private_document.get("promotion_allowed") is not False
        or _content_sha256(private_document) != private_document.get("content_sha256")
    ):
        raise FullTrackPilotError("private pilot document failed strict validation")
    public_source = public_document.get("source_fingerprint")
    private_source = private_document.get("source_fingerprint")
    public_store = _validate_store_binding(
        public_document.get("store_binding"),
        expected_source_fingerprint=public_source,
    )
    private_store = _validate_store_binding(
        private_document.get("store_binding"),
        expected_source_fingerprint=private_source,
    )
    if (
        public_store != private_store
        or stable_json_sha256(public_store)
        != public_document.get("store_binding_sha256")
        or stable_json_sha256(private_store)
        != private_document.get("store_binding_sha256")
    ):
        raise FullTrackPilotError("public/private sealed store binding differs")
    if (
        public_source
        != private_source
        or public_document.get("store_binding_sha256")
        != private_document.get("store_binding_sha256")
        or public_document.get("fold") != private_document.get("fold")
    ):
        raise FullTrackPilotError("public/private source or fold binding differs")
    blinding = public_document.get("blinding")
    if (
        not isinstance(blinding, Mapping)
        or set(blinding)
        != {
            "opaque_per_seed_list_ids",
            "method_identity_public",
            "method_order_randomized_per_session",
            "blinding_key_sha256",
            "private_unblinding_sha256",
            "public_evidence_commitment_sha256",
        }
        or blinding.get("opaque_per_seed_list_ids") is not True
        or blinding.get("method_identity_public") is not False
        or blinding.get("method_order_randomized_per_session") is not True
    ):
        raise FullTrackPilotError("public blinding declaration is missing or invalid")
    if blinding.get("private_unblinding_sha256") != private_document.get(
        "content_sha256"
    ):
        raise FullTrackPilotError("private unblinding content hash is not committed")
    key_hex = private_document.get("blinding_key_hex")
    if not isinstance(key_hex, str) or len(key_hex) != 64:
        raise FullTrackPilotError("private blinding key is malformed")
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise FullTrackPilotError("private blinding key is malformed") from exc
    if (
        hashlib.sha256(key).hexdigest() != blinding.get("blinding_key_sha256")
        or hashlib.sha256(key).hexdigest()
        != private_document.get("blinding_key_sha256")
    ):
        raise FullTrackPilotError("blinding key commitment differs")
    diversity = _validate_diversity_evidence(
        public_document.get("diversity_evidence")
    )
    audio_delivery = public_document.get("audio_delivery")
    if (
        not isinstance(audio_delivery, Mapping)
        or set(audio_delivery)
        != {
            "kind",
            "host",
            "verification",
            "repository_contains_audio",
            "commercial_preview",
        }
        or audio_delivery.get("kind") != "Jamendo first-party full-track MP3"
        or audio_delivery.get("host") != PUBLIC_AUDIO_HOST
        or audio_delivery.get("verification")
        != "HEAD 200, HTTPS, audio/mpeg, byte ranges, >=100000 bytes"
        or audio_delivery.get("repository_contains_audio") is not False
        or audio_delivery.get("commercial_preview") is not False
    ):
        raise FullTrackPilotError("public audio delivery declaration is invalid")

    forbidden_public = (
        "nonnegative_linear",
        "monotonic_network",
        "channel_gated_embedding",
        "frozen_hybrid",
        "model_artifact_sha256",
        "weights_npz_sha256",
        "report_sha256",
        "method_binding",
        "blinding_key_hex",
    )
    public_text = _canonical_bytes(public_document).decode("utf-8")
    if any(marker in public_text for marker in forbidden_public):
        raise FullTrackPilotError("public pilot leaks method identity or private key")

    public_seeds = public_document.get("seeds")
    private_seeds = private_document.get("seeds")
    tracks = public_document.get("tracks")
    if (
        not isinstance(public_seeds, list)
        or len(public_seeds) != 20
        or not isinstance(private_seeds, list)
        or len(private_seeds) != 20
        or not isinstance(tracks, Mapping)
    ):
        raise FullTrackPilotError("pilot seeds/tracks have invalid cardinality")
    store_rows = [
        _validate_track_document(
            track_id,
            track,
            expected_fold=int(public_document["fold"]),
        )
        for track_id, track in tracks.items()
    ]
    if len(store_rows) != len(set(store_rows)):
        raise FullTrackPilotError("public source store rows are not unique")
    evidence_payload = {
        "source_fingerprint": public_source,
        "store_binding_sha256": public_document.get("store_binding_sha256"),
        "fold": public_document.get("fold"),
        "fold_part": "test",
        "diversity_evidence": dict(diversity),
        "seed_descriptors": _seed_evidence_descriptors(public_seeds),
        "tracks": tracks,
    }
    evidence_commitment = hmac.new(
        key, _canonical_bytes(evidence_payload), hashlib.sha256
    ).hexdigest()
    if (
        evidence_commitment
        != blinding.get("public_evidence_commitment_sha256")
        or evidence_commitment
        != private_document.get("public_evidence_commitment_sha256")
    ):
        raise FullTrackPilotError(
            "public source/license/diversity evidence commitment differs"
        )
    if any(
        not isinstance(seed, Mapping) or set(seed) != _PUBLIC_SEED_KEYS
        for seed in public_seeds
    ) or any(
        not isinstance(seed, Mapping) or set(seed) != _PRIVATE_SEED_KEYS
        for seed in private_seeds
    ):
        raise FullTrackPilotError("pilot seed schema is invalid")
    private_by_seed = {item.get("seed_id"): item for item in private_seeds}
    if len(private_by_seed) != 20:
        raise FullTrackPilotError("private seed identities are not unique")
    seen_seed_ids: set[str] = set()
    seen_seed_tracks: set[int] = set()
    seen_seed_artists: set[int] = set()
    seen_lists: set[str] = set()
    seen_result_ids: set[str] = set()
    seed_tempos: list[float] = []
    seed_tempo_counts = {name: 0 for name, _, _ in TEMPO_BINS}
    seed_texture_counts = {str(index): 0 for index in range(5)}
    for seed_index, seed in enumerate(public_seeds):
        seed_id = seed.get("seed_id")
        seed_track_id = seed.get("seed_track_id")
        private_seed = private_by_seed.get(seed_id)
        scene = seed.get("scene")
        tempo = seed.get("tempo_bpm")
        tempo_region = seed.get("tempo_region")
        texture_region = seed.get("clap_texture_region")
        if (
            not _valid_opaque_id(seed_id, "seed")
            or seed_id in seen_seed_ids
            or isinstance(seed_track_id, bool)
            or not isinstance(seed_track_id, int)
            or seed_track_id <= 0
            or private_seed is None
            or private_seed.get("seed_track_id") != seed_track_id
            or seed_track_id in seen_seed_tracks
            or scene != PILOT_SCENES[seed_index].removeprefix("genre---")
            or isinstance(tempo, bool)
            or not isinstance(tempo, (int, float))
            or not math.isfinite(float(tempo))
            or not 0.0 < float(tempo) <= 400.0
            or tempo_region != _tempo_bin(float(tempo))
            or isinstance(texture_region, bool)
            or not isinstance(texture_region, int)
            or texture_region not in range(5)
        ):
            raise FullTrackPilotError("seed identity or diversity descriptor is invalid")
        seed_record = tracks.get(str(seed_track_id))
        if not isinstance(seed_record, Mapping):
            raise FullTrackPilotError("seed source track record is missing")
        artist_id = seed_record.get("source_identity", {}).get("artist_id")
        if not isinstance(artist_id, int) or artist_id in seen_seed_artists:
            raise FullTrackPilotError("pilot seeds must be artist-disjoint")
        seen_seed_ids.add(seed_id)
        seen_seed_tracks.add(seed_track_id)
        seen_seed_artists.add(artist_id)
        seed_tempos.append(float(tempo))
        seed_tempo_counts[str(tempo_region)] += 1
        seed_texture_counts[str(texture_region)] += 1

        result_rows = seed.get("result_ids")
        if not isinstance(result_rows, list) or not result_rows:
            raise FullTrackPilotError("seed result identity map is invalid")
        result_ids_by_track: Dict[int, str] = {}
        for result_row in result_rows:
            if not isinstance(result_row, Mapping) or set(result_row) != _RESULT_ID_KEYS:
                raise FullTrackPilotError("seed result identity schema is invalid")
            track_id = result_row.get("track_id")
            result_id = result_row.get("result_id")
            if (
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id in result_ids_by_track
                or not _valid_opaque_id(result_id, "result")
                or result_id in seen_result_ids
                or not isinstance(tracks.get(str(track_id)), Mapping)
            ):
                raise FullTrackPilotError("seed result identity map is invalid")
            result_ids_by_track[track_id] = result_id
            seen_result_ids.add(result_id)
        if [row["track_id"] for row in result_rows] != sorted(result_ids_by_track):
            raise FullTrackPilotError("seed result identity map is not canonical")

        public_lists = seed.get("lists")
        private_lists = private_seed.get("lists")
        if (
            not isinstance(public_lists, list)
            or len(public_lists) != len(METHODS)
            or not isinstance(private_lists, list)
            or len(private_lists) != len(METHODS)
            or any(
                not isinstance(item, Mapping) or set(item) != _PUBLIC_LIST_KEYS
                for item in public_lists
            )
            or any(
                not isinstance(item, Mapping) or set(item) != _PRIVATE_LIST_KEYS
                for item in private_lists
            )
            or any(
                not _valid_opaque_id(item.get("list_id"), "list")
                for item in public_lists
            )
            or any(
                not _valid_opaque_id(item.get("list_id"), "list")
                for item in private_lists
            )
        ):
            raise FullTrackPilotError("each seed must contain four exact blinded lists")
        public_list_ids = [item.get("list_id") for item in public_lists]
        private_list_ids = [item.get("list_id") for item in private_lists]
        if public_list_ids != sorted(public_list_ids) or private_list_ids != sorted(
            private_list_ids
        ):
            raise FullTrackPilotError("blinded list ordering is not canonical")
        private_by_list = {item.get("list_id"): item for item in private_lists}
        if len(private_by_list) != len(METHODS):
            raise FullTrackPilotError("private list identities are not unique")
        method_names = set()
        ranked_track_union: set[int] = set()
        for item in public_lists:
            list_id = item.get("list_id")
            private_item = private_by_list.get(list_id)
            ranking = item.get("ranking")
            if (
                not _valid_opaque_id(list_id, "list")
                or list_id in seen_lists
                or private_item is None
                or not isinstance(ranking, list)
                or len(ranking) != RESULTS_PER_METHOD
            ):
                raise FullTrackPilotError("blinded list identity/cardinality is invalid")
            seen_lists.add(list_id)
            method_binding = private_item.get("method_binding")
            if not isinstance(method_binding, Mapping):
                raise FullTrackPilotError("private method binding is missing")
            method = method_binding.get("method")
            if not isinstance(method, str):
                raise FullTrackPilotError("private method binding is missing")
            _validate_method_binding(
                method_binding,
                expected_method=method,
                expected_source_fingerprint=str(public_source),
                expected_store_binding_sha256=str(
                    public_document.get("store_binding_sha256")
                ),
                expected_fold=int(public_document["fold"]),
            )
            method_names.add(method)
            if any(
                not isinstance(row, Mapping) or set(row) != _RANKING_ROW_KEYS
                for row in ranking
            ):
                raise FullTrackPilotError("ranking row schema is invalid")
            payload = {
                "pack_kind": PILOT_PACK_KIND,
                "seed_id": seed_id,
                "list_id": list_id,
                "method_binding": method_binding,
                "ranking_track_ids": [row.get("track_id") for row in ranking],
            }
            commitment = hmac.new(
                key, _canonical_bytes(payload), hashlib.sha256
            ).hexdigest()
            if (
                commitment != item.get("binding_commitment_sha256")
                or commitment != private_item.get("binding_commitment_sha256")
                or payload
                != {key_name: private_item.get(key_name) for key_name in payload}
            ):
                raise FullTrackPilotError("method/ranking commitment verification failed")
            ranked_ids = []
            for position, row in enumerate(ranking, 1):
                track_id = row.get("track_id")
                result_id = row.get("result_id")
                track = tracks.get(str(track_id))
                if (
                    row.get("position") != position
                    or result_ids_by_track.get(track_id) != result_id
                    or not isinstance(track, Mapping)
                    or track_id == seed_track_id
                    or track.get("source_identity", {}).get("artist_id") == artist_id
                    or track.get("source_identity", {}).get("fold_part") != "test"
                ):
                    raise FullTrackPilotError("ranked source/fold/artist binding is invalid")
                ranked_ids.append(track_id)
                ranked_track_union.add(track_id)
            if len(set(ranked_ids)) != RESULTS_PER_METHOD:
                raise FullTrackPilotError("a method ranking contains duplicate tracks")
        if method_names != set(METHODS):
            raise FullTrackPilotError("private seed map does not bind all methods")
        if ranked_track_union != set(result_ids_by_track):
            raise FullTrackPilotError("seed result identity map differs from rankings")

    tempo_evidence = diversity["tempo_bpm"]
    texture_evidence = diversity["clap_texture"]
    if (
        tempo_evidence["bin_counts"] != seed_tempo_counts
        or tempo_evidence["minimum"] != min(seed_tempos)
        or tempo_evidence["median"] != float(statistics.median(seed_tempos))
        or tempo_evidence["maximum"] != max(seed_tempos)
        or texture_evidence["region_counts"] != seed_texture_counts
    ):
        raise FullTrackPilotError("seed descriptors differ from diversity evidence")


def _rank_methods(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    selected_seeds: Sequence[SeedCandidate],
    model_bindings: Mapping[str, object],
    config: PilotConfig,
) -> Mapping[int, Mapping[str, Tuple[int, ...]]]:
    fold = next(item for item in context.folds if item.index == config.fold_index)
    partition = [
        track
        for track in context.tracks
        if fold.track_parts.get(int(track.track_id)) == config.part
    ]
    id_to_position = {
        int(track.track_id): position for position, track in enumerate(partition)
    }
    store_rows = {int(track_id): row for row, track_id in enumerate(reader.track_ids)}
    if not set(id_to_position).issubset(store_rows):
        raise FullTrackPilotError("sealed store does not cover the pilot fold")
    globals_matrix = normalize_rows(
        np.asarray(
            reader.global_embeddings[
                [store_rows[int(track.track_id)] for track in partition]
            ],
            dtype=np.float32,
        )
    )

    pools: Dict[int, np.ndarray] = {}
    required_ids = {int(item.track_id) for item in selected_seeds}
    for seed in selected_seeds:
        query_position = id_to_position[seed.track_id]
        eligible = np.asarray(
            [
                index
                for index, candidate in enumerate(partition)
                if candidate.track_id != seed.track_id
                and candidate.artist_id != seed.artist_id
            ],
            dtype=np.int64,
        )
        scores = globals_matrix[eligible] @ globals_matrix[query_position]
        order = eligible[np.lexsort((eligible, -scores))]
        pool = order[: min(config.candidate_pool, len(order))]
        if len(pool) < config.results_per_method:
            raise FullTrackPilotError("pilot candidate pool is too small")
        pools[seed.track_id] = pool
        required_ids.update(int(partition[index].track_id) for index in pool)

    stored_tracks: Dict[int, object] = {}
    for track_id in sorted(required_ids, key=store_rows.__getitem__):
        stored_tracks[track_id] = reader.read_track(track_id)

    trained_map = {
        f"trained_{family}": (model_bindings[family], "none")
        for family in MODEL_FAMILIES
    }
    rankings: Dict[int, Dict[str, Tuple[int, ...]]] = {}
    for seed in selected_seeds:
        query_position = id_to_position[seed.track_id]
        pool = pools[seed.track_id]
        pool_ids = [int(partition[index].track_id) for index in pool]
        query = stored_tracks[seed.track_id]
        candidates = [stored_tracks[track_id] for track_id in pool_ids]
        query_uniform = freeze_fixed_budget(
            query.window_embeddings, config.maxsim_budget
        ).astype(np.float16)
        candidate_uniform = np.stack(
            [
                freeze_fixed_budget(
                    item.window_embeddings, config.maxsim_budget
                ).astype(np.float16)
                for item in candidates
            ]
        ).astype(np.float32)
        uniform_scores = batch_fixed_budget_maxsim(
            query_uniform, candidate_uniform
        )
        query_repeated = freeze_ranked_section_budget(
            query.repeated_sections, config.maxsim_budget
        ).astype(np.float16)
        query_salient = freeze_ranked_section_budget(
            query.salient_sections, config.maxsim_budget
        ).astype(np.float16)
        repeated_scores = batch_fixed_budget_maxsim(
            query_repeated,
            np.stack(
                [
                    freeze_ranked_section_budget(
                        item.repeated_sections, config.maxsim_budget
                    ).astype(np.float16)
                    for item in candidates
                ]
            ).astype(np.float32),
        )
        salient_scores = batch_fixed_budget_maxsim(
            query_salient,
            np.stack(
                [
                    freeze_ranked_section_budget(
                        item.salient_sections, config.maxsim_budget
                    ).astype(np.float16)
                    for item in candidates
                ]
            ).astype(np.float32),
        )
        section_scores = 0.5 * (repeated_scores + salient_scores)
        global_scores = globals_matrix[pool] @ globals_matrix[query_position]
        hybrid_scores = (
            HYBRID_WEIGHTS["global_cosine"] * global_scores
            + HYBRID_WEIGHTS["uniform_window_maxsim"] * uniform_scores
            + HYBRID_WEIGHTS["section_maxsim"] * section_scores
        )
        methods: Dict[str, Tuple[int, ...]] = {
            "frozen_hybrid": tuple(
                pool_ids[index]
                for index in np.argsort(-hybrid_scores, kind="stable")[
                    : config.results_per_method
                ]
            )
        }
        trained_scores, _ = _score_trained_candidate_pool(
            query, candidates, trained_map
        )
        for family in MODEL_FAMILIES:
            scores = trained_scores[f"trained_{family}"]
            methods[family] = tuple(
                pool_ids[index]
                for index in np.argsort(-scores, kind="stable")[
                    : config.results_per_method
                ]
            )
        rankings[seed.track_id] = methods
    return rankings


def _load_or_measure_tempos(
    path: Path,
    *,
    source_fingerprint: str,
    shortlist: Sequence[JamendoTrack],
) -> Mapping[int, float]:
    expected_ids = [int(track.track_id) for track in shortlist]
    if path.exists():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FullTrackPilotError("tempo cache is invalid") from exc
        if (
            document.get("schema_version") != 2
            or document.get("artifact_kind") != "fulltrack_pilot_tempo_cache"
            or document.get("source_fingerprint") != source_fingerprint
            or document.get("track_ids") != expected_ids
            or _content_sha256(document) != document.get("content_sha256")
        ):
            raise FullTrackPilotError("tempo cache source/order/hash drift")
        values = document.get("tempo_bpm")
        rejected = document.get("rejected_track_ids")
        if not isinstance(values, Mapping) or not isinstance(rejected, list):
            raise FullTrackPilotError("tempo cache values are invalid")
        try:
            result = {int(key): float(value) for key, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise FullTrackPilotError("tempo cache values are invalid") from exc
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in rejected)
            or len(set(rejected)) != len(rejected)
            or set(result).intersection(rejected)
            or set(result).union(rejected) != set(expected_ids)
        ):
            raise FullTrackPilotError("tempo cache coverage differs")
        for value in result.values():
            _tempo_bin(value)
        return result

    measured: Dict[int, float] = {}
    rejected_ids: list[int] = []
    for track in shortlist:
        try:
            measured[int(track.track_id)] = measure_tempo_bpm(track)
        except FullTrackPilotError:
            rejected_ids.append(int(track.track_id))
    document: Dict[str, object] = {
        "schema_version": 2,
        "artifact_kind": "fulltrack_pilot_tempo_cache",
        "source_fingerprint": source_fingerprint,
        "track_ids": expected_ids,
        "tempo_bpm": {str(key): measured[key] for key in expected_ids if key in measured},
        "rejected_track_ids": rejected_ids,
        "measurement": "soundalike.audio.features first 30 seconds",
        "invalid_descriptor_policy": (
            "non-finite or out-of-range descriptors are excluded before selection"
        ),
    }
    document["content_sha256"] = _content_sha256(document)
    _write_json(path, document, private=True)
    return measured


def _read_blinding_key(path: Path, create: bool) -> bytes:
    if path.exists():
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise FullTrackPilotError("cannot read blinding key") from exc
        if len(raw) != 65 or not raw.endswith(b"\n"):
            raise FullTrackPilotError("blinding key file must be 64 hex chars plus LF")
        try:
            key = bytes.fromhex(raw[:-1].decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise FullTrackPilotError("blinding key file is malformed") from exc
        if len(key) != 32:
            raise FullTrackPilotError("blinding key must contain 32 bytes")
        return key
    if not create:
        raise FullTrackPilotError(
            "blinding key is missing; use --create-blinding-key only once"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    try:
        with path.open("xb") as handle:
            os.chmod(path, 0o600)
            handle.write(key.hex().encode("ascii") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FullTrackPilotError("cannot create private blinding key") from exc
    return key


def _write_json(path: Path, document: Mapping[str, object], *, private: bool) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.parent.is_symlink():
        raise FullTrackPilotError("pilot output may not use symlinks")
    raw = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            if private:
                os.chmod(temporary, 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_production_pilot(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    store_root: Path,
    trained_root: Path,
    public_output: Path,
    private_output: Path,
    tempo_cache: Path,
    blinding_key_path: Path,
    create_blinding_key: bool,
    verify_audio: bool,
    audio_workers: int,
    config: PilotConfig,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    config.validate()
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    if context.evidence_scope != EVIDENCE_SCOPE:
        raise FullTrackPilotError("source evidence scope is not full-track Jamendo")
    fold = next(item for item in context.folds if item.index == config.fold_index)
    partition = [
        track
        for track in context.tracks
        if fold.track_parts.get(int(track.track_id)) == config.part
    ]
    tracks_by_id = {int(track.track_id): track for track in context.tracks}
    if len(partition) < config.seed_count:
        raise FullTrackPilotError("held-out partition is too small")

    with FullTrackStoreReader(
        store_root, expected_source_fingerprint=context.source_fingerprint
    ) as reader:
        store_binding = dict(reader.binding.as_dict())
        if (
            int(store_binding.get("track_count", 0)) != 55_701
            or not _valid_hash(reader.manifest.get("global_sha256"))
        ):
            raise FullTrackPilotError("sealed production store binding is invalid")
        store_binding["sealed_manifest_sha256"] = stable_json_sha256(
            reader.manifest
        )
        partition_rows = {
            int(track_id): row for row, track_id in enumerate(reader.track_ids)
        }
        partition_embeddings = np.asarray(
            reader.global_embeddings[
                [partition_rows[int(track.track_id)] for track in partition]
            ],
            dtype=np.float32,
        )
        regions, anchors = assign_texture_regions(
            [track.track_id for track in partition],
            partition_embeddings,
            config.texture_regions,
        )
        shortlist_ids = build_tempo_shortlist(
            partition,
            fold.track_tags,
            partition_embeddings,
            per_scene=config.shortlist_per_scene,
        )
        shortlist = [tracks_by_id[track_id] for track_id in shortlist_ids]
        tempos = _load_or_measure_tempos(
            tempo_cache,
            source_fingerprint=context.source_fingerprint,
            shortlist=shortlist,
        )
        region_by_id = {
            int(track.track_id): int(regions[index])
            for index, track in enumerate(partition)
        }
        shortlist_candidates = [
            SeedCandidate(
                track_id=int(track.track_id),
                artist_id=int(track.artist_id),
                tags=tuple(fold.track_tags[int(track.track_id)]),
                tempo_bpm=float(tempos[int(track.track_id)]),
                texture_region=region_by_id[int(track.track_id)],
            )
            for track in shortlist
            if int(track.track_id) in tempos
        ]
        if len(shortlist_candidates) < config.seed_count:
            raise FullTrackPilotError(
                "too few valid tempo descriptors for the 20-seed pilot"
            )
        shortlist_embeddings = np.asarray(
            reader.global_embeddings[
                [partition_rows[item.track_id] for item in shortlist_candidates]
            ],
            dtype=np.float32,
        )
        selected = select_diverse_seeds(
            shortlist_candidates, shortlist_embeddings
        )
        selected_embeddings = np.asarray(
            reader.global_embeddings[
                [partition_rows[item.track_id] for item in selected]
            ],
            dtype=np.float32,
        )
        diversity = diversity_evidence(
            selected,
            selected_embeddings,
            texture_anchor_track_ids=anchors,
        )
        store_hash = stable_json_sha256(store_binding)
        models = {
            family: load_trained_model_for_fold(
                trained_root,
                fold_index=config.fold_index,
                candidate_kind=family,
                seed=config.model_seed,
                expected_source_fingerprint=context.source_fingerprint,
                expected_store_binding_sha256=store_hash,
                store_embedding_dim=reader.binding.embedding_dim,
                store_repetition_sections=reader.binding.repetition_sections,
                store_salient_sections=reader.binding.salient_sections,
            )
            for family in MODEL_FAMILIES
        }
        rankings = _rank_methods(context, reader, selected, models, config)
        all_ids = {
            item.track_id for item in selected
        } | {
            track_id
            for methods in rankings.values()
            for ranked in methods.values()
            for track_id in ranked
        }
        urls = [lawful_stream_url(int(track_id)) for track_id in all_ids]
        if not verify_audio:
            raise FullTrackPilotError(
                "production pilot generation requires --verify-public-audio"
            )
        audio_evidence = verify_public_audio_urls(
            urls, workers=audio_workers
        )
        key = _read_blinding_key(blinding_key_path, create_blinding_key)
        public, private = build_blinded_documents(
            rankings=rankings,
            selected_seeds=selected,
            tracks_by_id=tracks_by_id,
            store_rows=partition_rows,
            fold_track_parts=fold.track_parts,
            store_binding=store_binding,
            source_fingerprint=context.source_fingerprint,
            fold_index=config.fold_index,
            model_bindings=models,
            blinding_key=key,
            diversity=diversity,
            audio_verification=audio_evidence,
        )
    _write_json(public_output, public, private=False)
    _write_json(private_output, private, private=True)
    return public, private


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build and verify the v2 pilot")
    for name in (
        "metadata-root",
        "audio-root",
        "state-root",
        "store",
        "trained-root",
        "public-output",
        "private-output",
        "tempo-cache",
        "blinding-key",
    ):
        build.add_argument(f"--{name}", required=True)
    build.add_argument("--fold", type=int, default=0)
    build.add_argument("--model-seed", type=int, default=17)
    build.add_argument("--candidate-pool", type=int, default=200)
    build.add_argument("--shortlist-per-scene", type=int, default=8)
    build.add_argument("--create-blinding-key", action="store_true")
    build.add_argument("--verify-public-audio", action="store_true")
    build.add_argument("--audio-workers", type=int, default=8)

    validate = subparsers.add_parser(
        "validate", help="validate an existing public/private pilot pair"
    )
    validate.add_argument("--public", required=True)
    validate.add_argument("--private", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        public = json.loads(Path(args.public).read_text(encoding="utf-8"))
        private = json.loads(Path(args.private).read_text(encoding="utf-8"))
        validate_blinded_documents(public, private)
        print(
            json.dumps(
                {
                    "public_content_sha256": public["content_sha256"],
                    "private_content_sha256": private["content_sha256"],
                    "valid": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    config = PilotConfig(
        fold_index=args.fold,
        candidate_pool=args.candidate_pool,
        model_seed=args.model_seed,
        shortlist_per_scene=args.shortlist_per_scene,
    )
    public, private = build_production_pilot(
        metadata_root=Path(args.metadata_root),
        audio_root=Path(args.audio_root),
        state_root=Path(args.state_root),
        store_root=Path(args.store),
        trained_root=Path(args.trained_root),
        public_output=Path(args.public_output),
        private_output=Path(args.private_output),
        tempo_cache=Path(args.tempo_cache),
        blinding_key_path=Path(args.blinding_key),
        create_blinding_key=bool(args.create_blinding_key),
        verify_audio=bool(args.verify_public_audio),
        audio_workers=int(args.audio_workers),
        config=config,
    )
    print(
        json.dumps(
            {
                "public_output": args.public_output,
                "public_content_sha256": public["content_sha256"],
                "private_output": args.private_output,
                "private_content_sha256": private["content_sha256"],
                "seed_count": public["seed_count"],
                "method_count": public["method_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
