"""Compact catalogue-wide scene/style features from MusicBrainz tags.

MusicBrainz community tags are an independent metadata source.  They are used
only to label audio-nearest anchors; this module does not consume Last.fm,
Music4All, benchmark identities, or popularity signals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from .real_benchmark import normalize_text


# Deliberately broad and artist-agnostic.  A tag may activate several scenes
# (for example, "folk metal" and "electropop").
SCENE_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "pop": (
        "pop", "synthpop", "electropop", "dream pop", "art pop",
        "power pop", "city pop", "k pop", "j pop",
    ),
    "rock": (
        "rock", "indie", "alternative rock", "grunge", "post rock",
        "psychedelic rock", "new wave", "britpop",
    ),
    "art_rock": (
        "art rock", "experimental rock", "progressive rock", "post rock",
    ),
    "shoegaze_dream_pop": (
        "shoegaze", "dream pop", "ethereal wave", "noise pop",
    ),
    "hyperpop_digicore": (
        "hyperpop", "digicore", "glitch pop", "bubblegum bass", "pc music",
    ),
    "city_pop": (
        "city pop", "japanese city pop", "kayokyoku",
    ),
    "punk": (
        "punk", "hardcore", "emo", "post punk", "pop punk",
    ),
    "metal": (
        "metal", "doom", "black metal", "death metal", "heavy metal",
        "metalcore", "grindcore",
    ),
    "hip_hop": (
        "hip hop", "hiphop", "rap", "trap", "grime", "drill",
        "boom bap",
    ),
    "rnb_soul": (
        "r b", "rnb", "rhythm and blues", "soul", "neo soul",
        "motown", "quiet storm",
    ),
    "electronic": (
        "electronic", "electronica", "dance", "edm", "house", "techno",
        "trance", "ambient", "idm", "dubstep", "drum and bass", "garage",
        "breakbeat", "disco", "synthwave", "industrial", "electropop",
    ),
    "experimental": (
        "experimental", "avant garde", "noise", "musique concrete",
        "free improvisation",
    ),
    "jazz": (
        "jazz", "bebop", "swing", "fusion", "bossa nova",
    ),
    "blues": (
        "blues", "delta blues", "electric blues", "blues rock",
    ),
    "folk_country": (
        "folk", "country", "americana", "bluegrass", "singer songwriter",
        "roots", "traditional",
    ),
    "classical": (
        "classical", "baroque", "romantic", "opera", "orchestral",
        "chamber music", "contemporary classical",
    ),
    "latin": (
        "latin", "salsa", "reggaeton", "cumbia", "bachata", "merengue",
        "tango", "mariachi",
    ),
    "reggae_caribbean": (
        "reggae", "dub", "ska", "dancehall", "calypso", "soca",
    ),
    "african": (
        "afrobeat", "afrobeats", "highlife", "amapiano", "african",
        "rai", "soukous",
    ),
    "asian_pop": (
        "k pop", "j pop", "c pop", "city pop", "mandopop", "cantopop",
    ),
    "gospel": (
        "gospel", "spiritual", "christian music",
    ),
    "soundtrack": (
        "soundtrack", "film score", "video game music", "musical",
    ),
}
SCENE_NAMES: Tuple[str, ...] = tuple(SCENE_KEYWORDS)


def tags_to_scene_vector(
    tags: Sequence[Union[str, Mapping[str, Any]]],
    scene_names: Sequence[str] = SCENE_NAMES,
) -> np.ndarray:
    """Map community tags to an L2-normalized, multi-label broad scene vector."""
    vector = np.zeros(len(scene_names), dtype=np.float32)
    scene_positions = {name: position for position, name in enumerate(scene_names)}
    for raw in tags:
        value = raw.get("name", "") if isinstance(raw, Mapping) else raw
        tag = normalize_text(str(value))
        if not tag:
            continue
        padded = " " + tag + " "
        for scene, keywords in SCENE_KEYWORDS.items():
            position = scene_positions.get(scene)
            if position is None:
                continue
            if any(" " + normalize_text(keyword) + " " in padded for keyword in keywords):
                vector[position] = 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _unit_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("artist_audio must be a two-dimensional array")
    return result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-8)


def _nearest_labelled(
    audio: np.ndarray,
    labelled_ids: np.ndarray,
    query_ids: np.ndarray,
    anchors: int,
    chunk_size: int,
    anchor_block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k with memory bounded by the two block sizes."""
    count = min(max(int(anchors), 1), len(labelled_ids))
    positions = np.empty((len(query_ids), count), dtype=np.int32)
    similarities = np.empty((len(query_ids), count), dtype=np.float32)
    labelled_audio = audio[labelled_ids]
    for query_start in range(0, len(query_ids), chunk_size):
        query_stop = min(query_start + chunk_size, len(query_ids))
        query = audio[query_ids[query_start:query_stop]]
        best_positions = np.empty((len(query), 0), dtype=np.int32)
        best_values = np.empty((len(query), 0), dtype=np.float32)
        for anchor_start in range(0, len(labelled_ids), anchor_block_size):
            anchor_stop = min(anchor_start + anchor_block_size, len(labelled_ids))
            scores = query @ labelled_audio[anchor_start:anchor_stop].T
            block_positions = np.broadcast_to(
                np.arange(anchor_start, anchor_stop, dtype=np.int32),
                scores.shape,
            )
            merged_values = np.concatenate((best_values, scores), axis=1)
            merged_positions = np.concatenate(
                (best_positions, block_positions), axis=1
            )
            keep = min(count, merged_values.shape[1])
            next_positions = np.empty((len(query), keep), dtype=np.int32)
            next_values = np.empty((len(query), keep), dtype=np.float32)
            for row in range(len(query)):
                # Anchor position is the stable tie breaker.
                order = np.lexsort(
                    (merged_positions[row], -merged_values[row])
                )[:keep]
                next_positions[row] = merged_positions[row, order]
                next_values[row] = merged_values[row, order]
            best_positions, best_values = next_positions, next_values
        positions[query_start:query_stop] = best_positions
        similarities[query_start:query_stop] = best_values
    return positions, similarities


def propagate_scene_vectors(
    artist_audio: np.ndarray,
    direct_vectors: np.ndarray,
    direct_mask: np.ndarray,
    anchors: int = 8,
    chunk_size: int = 256,
    anchor_block_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate labels to every artist while leaving anchor rows unchanged."""
    audio = _unit_rows(artist_audio)
    vectors = np.asarray(direct_vectors, dtype=np.float32)
    mask = np.asarray(direct_mask, dtype=bool)
    if len(audio) != len(vectors) or mask.shape != (len(audio),):
        raise ValueError("artist audio, vectors, and direct mask are misaligned")
    if not np.any(mask):
        raise ValueError("at least one MusicBrainz-tagged anchor is required")
    result = vectors.copy()
    confidence = np.zeros(len(audio), dtype=np.float32)
    confidence[mask] = 1.0
    labelled_ids = np.flatnonzero(mask).astype(np.int32)
    query_ids = np.flatnonzero(~mask).astype(np.int32)
    if not len(query_ids):
        return result, confidence
    positions, similarities = _nearest_labelled(
        audio,
        labelled_ids,
        query_ids,
        anchors,
        max(int(chunk_size), 1),
        max(int(anchor_block_size), 1),
    )
    weights = np.maximum(similarities, 0.0)
    empty = ~np.any(weights > 0.0, axis=1)
    weights[empty] = 1.0
    mixtures = np.sum(
        vectors[labelled_ids[positions]] * weights[:, :, None], axis=1
    ) / weights.sum(axis=1, keepdims=True)
    mixture_norm = np.linalg.norm(mixtures, axis=1)
    mixtures /= mixture_norm[:, None].clip(min=1e-8)
    result[query_ids] = mixtures
    positive_similarity = np.sum(
        weights * np.maximum(similarities, 0.0), axis=1
    ) / weights.sum(axis=1)
    confidence[query_ids] = np.clip(
        positive_similarity * mixture_norm, 0.0, 1.0
    )
    # Explicitly restore anchors to protect this contract from future changes.
    result[mask] = vectors[mask]
    return result, confidence


def build_catalog_style_asset(
    graph_path: Union[str, Path],
    tag_cache_path: Union[str, Path],
    output_path: Union[str, Path],
    anchors: int = 8,
    chunk_size: int = 256,
    anchor_block_size: int = 2048,
) -> Dict[str, Any]:
    """Build and save catalogue-aligned float16 style vectors."""
    with np.load(graph_path, allow_pickle=False) as graph:
        artist_names = np.asarray(graph["artist_names"])
        artist_audio = np.asarray(graph["artist_audio"], dtype=np.float32)
    cache = json.loads(Path(tag_cache_path).read_text(encoding="utf-8"))
    if not isinstance(cache, dict):
        raise ValueError("MusicBrainz tag cache must be a JSON object")
    direct = np.zeros((len(artist_names), len(SCENE_NAMES)), dtype=np.float32)
    cache_matches = 0
    for row, artist in enumerate(artist_names):
        key = normalize_text(str(artist))
        if key not in cache:
            continue
        cache_matches += 1
        tags = cache[key] if isinstance(cache[key], list) else []
        direct[row] = tags_to_scene_vector(tags)
    direct_mask = np.linalg.norm(direct, axis=1) > 0.0
    vectors, confidence = propagate_scene_vectors(
        artist_audio,
        direct,
        direct_mask,
        anchors=anchors,
        chunk_size=chunk_size,
        anchor_block_size=anchor_block_size,
    )
    direct_count = int(direct_mask.sum())
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "provider": "MusicBrainz community artist tags",
            "cache": str(tag_cache_path),
            "relationship": "independent scene/style metadata",
            "graph_source_independent": True,
            "used_by_graph_or_benchmark": False,
            "used_by_catalog_graph": False,
            "used_by_music4all": False,
            "uses_lastfm": False,
            "uses_music4all": False,
        },
        "taxonomy": "general broad scene/style multi-label",
        "normalization": "l2",
        "propagation": {
            "method": "audio-nearest-labelled-anchors",
            "anchors": min(max(int(anchors), 1), direct_count),
            "chunk_size": max(int(chunk_size), 1),
            "anchor_block_size": max(int(anchor_block_size), 1),
            "deterministic": True,
            "anchor_labels_preserved": True,
        },
        "coverage": {
            "catalogue_artists": len(artist_names),
            "cache_name_matches": cache_matches,
            "direct_tag_artists": direct_count,
            "direct_tag_fraction": direct_count / max(len(artist_names), 1),
            "propagated_artists": len(artist_names) - direct_count,
            "covered_artists": int(np.count_nonzero(np.linalg.norm(vectors, axis=1))),
            "covered_fraction": float(
                np.count_nonzero(np.linalg.norm(vectors, axis=1))
                / max(len(artist_names), 1)
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        artist_names=artist_names,
        scene_names=np.asarray(SCENE_NAMES),
        style_vectors=vectors.astype(np.float16),
        confidence=confidence.astype(np.float16),
        direct_mask=direct_mask.astype(np.uint8),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


# A convenient plural spelling for callers that think of this as a feature set.
build_catalog_styles = build_catalog_style_asset


class CatalogStyleIndex:
    """Runtime artist lookup and style-overlap feature for graph-first ranking."""

    def __init__(self, path: Union[str, Path]):
        with np.load(path, allow_pickle=False) as asset:
            self.artist_names = np.asarray(asset["artist_names"])
            self.scene_names = tuple(str(value) for value in asset["scene_names"])
            self.vectors = np.asarray(asset["style_vectors"], dtype=np.float32)
            self.style_vectors = self.vectors
            self.confidence = np.asarray(asset["confidence"], dtype=np.float32)
            self.direct_mask = np.asarray(asset["direct_mask"], dtype=bool)
            self.metadata = json.loads(str(asset["metadata"]))
        self.artist_lookup = {
            normalize_text(str(name)): row
            for row, name in enumerate(self.artist_names)
        }

    def artist_id(self, artist: str) -> Union[int, None]:
        """Return the catalogue row for an artist, if present."""
        return self.artist_lookup.get(normalize_text(artist))

    def artist_vector(self, artist: str) -> np.ndarray:
        """Return a copy of an artist vector, or an all-zero unknown vector."""
        row = self.artist_lookup.get(normalize_text(artist))
        if row is None:
            return np.zeros(len(self.scene_names), dtype=np.float32)
        return self.vectors[row].copy()

    vector = artist_vector

    def style_overlap(self, query_artist: str, candidate_artist: str) -> float:
        """Cosine-like multi-label overlap in [0, 1]; unknown artists score zero."""
        query = self.artist_lookup.get(normalize_text(query_artist))
        candidate = self.artist_lookup.get(normalize_text(candidate_artist))
        if query is None or candidate is None:
            return 0.0
        return float(np.clip(self.vectors[query] @ self.vectors[candidate], 0.0, 1.0))

    overlap = style_overlap

    def style_overlaps(
        self, query_artist: str, candidate_artists: Sequence[str]
    ) -> np.ndarray:
        query = self.artist_lookup.get(normalize_text(query_artist))
        result = np.zeros(len(candidate_artists), dtype=np.float32)
        if query is None:
            return result
        rows = [
            self.artist_lookup.get(normalize_text(candidate))
            for candidate in candidate_artists
        ]
        for position, row in enumerate(rows):
            if row is not None:
                result[position] = np.clip(
                    self.vectors[query] @ self.vectors[row], 0.0, 1.0
                )
        return result


def audit_catalog_styles(
    style: Union[str, Path, CatalogStyleIndex],
    pairs: Sequence[Union[Tuple[str, str], Mapping[str, str]]] = (),
    threshold: float = 0.25,
) -> Dict[str, Any]:
    """Report coverage and threshold false exclusions for known-positive pairs."""
    index = style if isinstance(style, CatalogStyleIndex) else CatalogStyleIndex(style)
    total = len(index.artist_names)
    direct = int(index.direct_mask.sum())
    covered = int(np.count_nonzero(np.linalg.norm(index.vectors, axis=1)))
    exclusions: List[Dict[str, Any]] = []
    unresolved = 0
    evaluated = 0
    for pair in pairs:
        if isinstance(pair, Mapping):
            query = str(pair["query"])
            candidate = str(pair["candidate"])
        else:
            query, candidate = str(pair[0]), str(pair[1])
        if (
            normalize_text(query) not in index.artist_lookup
            or normalize_text(candidate) not in index.artist_lookup
        ):
            unresolved += 1
            continue
        evaluated += 1
        overlap = index.style_overlap(query, candidate)
        if overlap < threshold:
            exclusions.append(
                {"query": query, "candidate": candidate, "overlap": overlap}
            )
    return {
        "coverage": {
            "catalogue_artists": total,
            "direct_tag_artists": direct,
            "direct_tag_fraction": direct / max(total, 1),
            "propagated_artists": total - direct,
            "covered_artists": covered,
            "covered_fraction": covered / max(total, 1),
        },
        "false_exclusions": {
            "threshold": float(threshold),
            "supplied_pairs": len(pairs),
            "evaluated_pairs": evaluated,
            "unresolved_pairs": unresolved,
            "count": len(exclusions),
            "rate": len(exclusions) / max(evaluated, 1),
            "pairs": exclusions,
        },
        "source": index.metadata.get("source", {}),
    }


audit_catalog_style = audit_catalog_styles
