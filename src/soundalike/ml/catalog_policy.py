"""Compact graph-first catalogue ranking for the runtime recommender."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .quality_filter import TitleQualityFilter
from .real_benchmark import normalize_text


@dataclass(frozen=True)
class CatalogPolicy:
    """The complete tunable surface of the catalogue ranker."""

    audio_weight: float
    style_weight: float
    style_guard_min: float

    def __post_init__(self) -> None:
        for name in ("audio_weight", "style_weight", "style_guard_min"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be a finite non-negative number" % name)
        if self.style_guard_min > 1.0:
            raise ValueError("style_guard_min must be in [0, 1]")


GRAPH_ONLY_POLICY = CatalogPolicy(0.0, 0.0, 0.0)
GRAPH_AUDIO_SCENE_POLICY = CatalogPolicy(0.35, 0.25, 0.20)


def graph_score(normalized_graph_edge_weight: float, graph_edge_rank: int) -> float:
    """Return the fixed graph component for a one-based edge rank."""
    if graph_edge_rank < 1:
        raise ValueError("graph_edge_rank must be one-based")
    weight = float(np.clip(normalized_graph_edge_weight, 0.0, 1.0))
    return 0.7 * weight + 0.3 / math.log2(graph_edge_rank + 1.0)


def policy_score(
    graph: float, audio: float, style: float, policy: CatalogPolicy
) -> float:
    """Apply the exact low-capacity scoring formula."""
    return (
        float(graph)
        + policy.audio_weight * float(audio)
        + policy.style_weight * float(style)
    )


def _cosine01(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity mapped to [0, 1], with bounded block temporaries."""
    query32 = np.asarray(query, dtype=np.float32)
    query32 = query32 / max(float(np.linalg.norm(query32)), 1e-9)
    result = np.empty(len(matrix), dtype=np.float32)
    for start in range(0, len(result), 16384):
        stop = min(start + 16384, len(result))
        block = np.asarray(matrix[start:stop], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1)
        cosine = (block @ query32) / np.maximum(norms, 1e-9)
        result[start:stop] = np.clip(0.5 * (cosine + 1.0), 0.0, 1.0)
    return result


def _top_rows(values: np.ndarray, count: int) -> np.ndarray:
    count = min(max(int(count), 0), len(values))
    if count == 0:
        return np.empty(0, dtype=np.int32)
    if count == len(values):
        rows = np.arange(len(values), dtype=np.int32)
    else:
        rows = np.argpartition(values, -count)[-count:].astype(np.int32)
    return rows[np.argsort(values[rows], kind="stable")[::-1]]


def _artist_parts(value: str) -> set:
    text = normalize_text(value)
    return {
        part.strip()
        for part in re.split(r"\s+(?:feat|featuring|ft|x)\s+|[,;&]", text)
        if part.strip()
    }


class CatalogPolicyRanker:
    """Rank a WebRecommender-like catalogue from compact graph/style assets.

    The graph's unmasked ``full`` edges produce the primary candidate set.
    Audio is evaluated once per query and otherwise only selects tracks inside
    graph artists or supplies a small bridge/fallback pool.
    """

    def __init__(
        self,
        recommender: Any,
        graph: Any,
        styles: Any,
        policy: CatalogPolicy = GRAPH_AUDIO_SCENE_POLICY,
        quality_filter: Optional[TitleQualityFilter] = None,
    ):
        self.rec = recommender
        self.graph = graph
        self.styles = styles
        self.policy = policy
        self.quality_filter = quality_filter or TitleQualityFilter()
        self.titles = recommender.titles
        self.artists = recommender.artists
        self.track_ids = recommender.track_ids
        size = len(self.titles)
        if len(self.artists) != size or len(self.track_ids) != size:
            raise ValueError("recommender catalogue arrays are misaligned")
        if len(graph.track_artist_ids) != size:
            raise ValueError("graph and recommender track rows are misaligned")

    def audio_scores(self, query_row: int) -> np.ndarray:
        """Fixed equal blend of sonic, CLAP, and vibe similarities in [0, 1]."""
        sonic = getattr(self.rec, "_sonic", None)
        clap = getattr(self.rec, "_clap", None)
        vibe = getattr(self.rec, "_vscaled", None)
        if sonic is None or clap is None or vibe is None:
            raise ValueError("catalogue policy requires sonic, CLAP, and vibe features")
        score = _cosine01(sonic, sonic[query_row])
        score += _cosine01(clap, clap[query_row])
        query_vibe = np.asarray(vibe[query_row], dtype=np.float32)
        vibe_score = np.empty(len(vibe), dtype=np.float32)
        for start in range(0, len(vibe_score), 16384):
            stop = min(start + 16384, len(vibe_score))
            delta = np.asarray(vibe[start:stop], dtype=np.float32) - query_vibe
            vibe_score[start:stop] = 1.0 / (1.0 + np.linalg.norm(delta, axis=1))
        score += vibe_score
        score *= 1.0 / 3.0
        return np.clip(score, 0.0, 1.0)

    def _graph_candidates(
        self, query_row: int, audio: np.ndarray
    ) -> Tuple[List[Tuple[int, float, str]], str]:
        query_artist = str(self.artists[query_row])
        artist_audio = np.concatenate(
            (
                np.asarray(self.rec._sonic[query_row], dtype=np.float32),
                np.asarray(self.rec._clap[query_row], dtype=np.float32),
                np.asarray(self.rec._vscaled[query_row], dtype=np.float32),
            )
        )
        # Existing graph assets use a compact artist-audio dimension which can
        # differ from track features in test or transitional indexes.
        expected = int(self.graph.artist_audio.shape[1])
        if artist_audio.shape != (expected,):
            artist_id = self.graph.artist_lookup.get(normalize_text(query_artist))
            if artist_id is not None:
                artist_audio = np.asarray(self.graph.artist_audio[artist_id])
            else:
                artist_audio = np.zeros(expected, dtype=np.float32)
        neighbors, weights, mode = self.graph.artist_neighbors(
            query_artist, artist_audio, variant="full"
        )
        positive_max = max(
            (max(float(weight), 0.0) for weight in weights), default=0.0
        )
        graph_signal = mode != "audio_artist_bridge" and positive_max > 0.0
        candidates: List[Tuple[int, float, str]] = []
        for rank, (artist_id, edge_weight) in enumerate(
            zip(neighbors, weights), start=1
        ):
            start = int(self.graph.track_indptr[int(artist_id)])
            stop = int(self.graph.track_indptr[int(artist_id) + 1])
            rows = self.graph.track_rows[start:stop]
            if not len(rows):
                continue
            selected = _top_rows(audio[rows], min(16, len(rows)))
            normalized = (
                max(float(edge_weight), 0.0) / positive_max if graph_signal else 0.0
            )
            component = graph_score(normalized, rank) if graph_signal else 0.0
            source = "graph" if graph_signal else "audio_bridge"
            candidates.extend(
                (int(rows[position]), component, source) for position in selected
            )
        return candidates, mode

    def recommend(self, query_row: int, n: int = 20) -> Dict[str, Any]:
        """Return target-blind serialized recommendations and rationales."""
        if query_row < 0 or query_row >= len(self.titles):
            raise IndexError("query_row is outside the catalogue")
        n = max(int(n), 0)
        audio = self.audio_scores(query_row)
        raw, query_mode = self._graph_candidates(query_row, audio)
        graph_rows = {row for row, _, _ in raw}
        # Fixed target-blind audio pool.  Candidate capacity is not a tuned
        # ranking parameter and does not grow with requested output length.
        fallback_count = min(len(audio), 1000)
        raw.extend(
            (int(row), 0.0, "audio_fallback")
            for row in _top_rows(audio, fallback_count)
            if int(row) not in graph_rows
        )

        seed_title = str(self.titles[query_row])
        seed_artists = _artist_parts(str(self.artists[query_row]))
        seen_rows = set()
        eligible: List[Tuple[int, float, float, float, float, str]] = []
        for row, graph_component, source in raw:
            if row == query_row or row in seen_rows:
                continue
            seen_rows.add(row)
            title, artist = str(self.titles[row]), str(self.artists[row])
            if seed_artists & _artist_parts(artist):
                continue
            if self.quality_filter.is_junk(title, artist):
                continue
            if self.quality_filter.seed_title_in_result(seed_title, title):
                continue
            style = float(self.styles.style_overlap(str(self.artists[query_row]), artist))
            a_value = float(audio[row])
            total = policy_score(graph_component, a_value, style, self.policy)
            eligible.append((row, total, graph_component, a_value, style, source))

        eligible.sort(key=lambda item: (-item[1], item[0]))
        deduped: List[Tuple[int, float, float, float, float, str]] = []
        recordings, used_artists = set(), set()
        for item in eligible:
            row = item[0]
            artist_key = normalize_text(str(self.artists[row]))
            recording = (normalize_text(str(self.titles[row])), artist_key)
            if recording in recordings or artist_key in used_artists:
                continue
            recordings.add(recording)
            used_artists.add(artist_key)
            deduped.append(item)

        guarded_count = min(3, n)
        safe = [item for item in deduped if item[4] >= self.policy.style_guard_min]
        if len(safe) >= guarded_count:
            guarded = safe[:guarded_count]
            guarded_rows = {item[0] for item in guarded}
            deduped = guarded + [
                item for item in deduped if item[0] not in guarded_rows
            ]

        results = []
        for position, item in enumerate(deduped[:n], start=1):
            row, total, graph_component, a_value, style, source = item
            track_id = self.track_ids[row]
            track_id = track_id.item() if isinstance(track_id, np.generic) else track_id
            results.append(
                {
                    "position": position,
                    "row": row,
                    "title": str(self.titles[row]),
                    "artist": str(self.artists[row]),
                    "track_id": track_id,
                    "score": float(total),
                    "rationale": {
                        "G": float(graph_component),
                        "A": float(a_value),
                        "S": float(style),
                        "source": source,
                        "query_mode": query_mode,
                    },
                }
            )
        return {
            "query": {
                "row": int(query_row),
                "title": seed_title,
                "artist": str(self.artists[query_row]),
            },
            "query_mode": query_mode,
            "results": results,
        }

    rank = recommend


GraphFirstCatalogRanker = CatalogPolicyRanker
