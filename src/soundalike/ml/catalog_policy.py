"""Confidence-gated dual-source catalogue ranking."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .quality_filter import TitleQualityFilter
from .real_benchmark import normalize_text


@dataclass(frozen=True)
class CatalogPolicy:
    """The complete numeric policy surface for graph-head replacement."""

    tau: float
    sigma: float
    audio_weight: float

    def __post_init__(self) -> None:
        for name in ("tau", "sigma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        weight = float(self.audio_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("audio_weight must be a finite non-negative number")
        object.__setattr__(self, "audio_weight", weight)


# Compatibility names for callers; both still obey the confidence gate.
GRAPH_ONLY_POLICY = CatalogPolicy(0.55, 0.55, 0.0)
GRAPH_AUDIO_SCENE_POLICY = CatalogPolicy(0.55, 0.55, 0.35)


def graph_score(normalized_graph_edge_weight: float, graph_edge_rank: int) -> float:
    """Fixed per-source graph component for a one-based source rank."""
    if graph_edge_rank < 1:
        raise ValueError("graph_edge_rank must be one-based")
    weight = float(np.clip(normalized_graph_edge_weight, 0.0, 1.0))
    return 0.7 * weight + 0.3 / math.log2(graph_edge_rank + 1.0)


def policy_score(
    graph: float, audio: float, style: float, policy: CatalogPolicy
) -> float:
    """Rank by graph plus audio; style is intentionally gate-only."""
    del style
    return float(graph) + policy.audio_weight * float(audio)


def _cosine01(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    query32 = np.asarray(query, dtype=np.float32)
    query32 = query32 / max(float(np.linalg.norm(query32)), 1e-9)
    result = np.empty(len(matrix), dtype=np.float32)
    for start in range(0, len(result), 16384):
        stop = min(start + 16384, len(result))
        block = np.asarray(matrix[start:stop], dtype=np.float32)
        cosine = (block @ query32) / np.maximum(
            np.linalg.norm(block, axis=1), 1e-9
        )
        result[start:stop] = np.clip(0.5 * (cosine + 1.0), 0.0, 1.0)
    return result


def _top_rows(values: np.ndarray, count: int) -> np.ndarray:
    count = min(max(int(count), 0), len(values))
    if not count:
        return np.empty(0, dtype=np.int32)
    rows = np.arange(len(values), dtype=np.int32)
    return rows[np.lexsort((rows, -values))[:count]]


def _artist_parts(value: str) -> set:
    text = normalize_text(value)
    return {
        part.strip()
        for part in re.split(r"\s+(?:feat|featuring|ft|x)\s+|[,;&]", text)
        if part.strip()
    }


def _source_components(source: Mapping[str, np.ndarray]) -> Dict[int, float]:
    ids = np.asarray(source["artist_ids"], dtype=np.int32)
    weights = np.maximum(
        np.asarray(source["weights"], dtype=np.float32), 0.0
    )
    maximum = float(weights.max()) if len(weights) else 0.0
    if maximum <= 0.0:
        return {}
    return {
        int(artist_id): graph_score(float(weight) / maximum, rank)
        for rank, (artist_id, weight) in enumerate(zip(ids, weights), 1)
        if weight > 0.0
    }


def source_agreement(
    lastfm: Mapping[str, np.ndarray],
    music4all: Mapping[str, np.ndarray],
) -> Tuple[float, int]:
    """Return fixed independent-source agreement and shared-neighbor count.

    At least five shared artist neighbors are required.  Above that floor:
    agreement = equal mean of (intersection/min(source sizes)) and the mean
    paired normalized positive edge strength across shared neighbors.
    """
    def normalized(source: Mapping[str, np.ndarray]) -> Dict[int, float]:
        ids = np.asarray(source["artist_ids"], dtype=np.int32)
        weights = np.maximum(
            np.asarray(source["weights"], dtype=np.float32), 0.0
        )
        maximum = float(weights.max()) if len(weights) else 0.0
        if maximum <= 0.0:
            return {}
        return {
            int(artist_id): float(weight) / maximum
            for artist_id, weight in zip(ids, weights)
            if weight > 0.0
        }

    left, right = normalized(lastfm), normalized(music4all)
    shared = sorted(set(left) & set(right))
    count = len(shared)
    if count < 5:
        return 0.0, count
    intersection = count / max(min(len(left), len(right)), 1)
    strength = float(
        np.mean([(left[key] + right[key]) * 0.5 for key in shared])
    )
    return float(0.5 * intersection + 0.5 * strength), count


class CatalogPolicyRanker:
    """Keep production dual_sonic unless two independent gates both pass."""

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
        self.titles = np.asarray(recommender.titles)
        self.artists = np.asarray(recommender.artists)
        self.track_ids = np.asarray(recommender.track_ids)
        size = len(self.titles)
        if len(self.artists) != size or len(self.track_ids) != size:
            raise ValueError("recommender catalogue arrays are misaligned")
        if len(graph.track_artist_ids) != size:
            raise ValueError("graph and recommender track rows are misaligned")
        self._rows_by_track_id: Dict[Any, List[int]] = {}
        for row, track_id in enumerate(self.track_ids):
            key = track_id.item() if isinstance(track_id, np.generic) else track_id
            self._rows_by_track_id.setdefault(key, []).append(row)

    def audio_scores(self, query_row: int) -> np.ndarray:
        """Fixed mean of Sonic cosine01, CLAP cosine01 and inverse vibe distance."""
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
            vibe_score[start:stop] = 1.0 / (
                1.0 + np.linalg.norm(delta, axis=1)
            )
        return np.clip((score + vibe_score) / 3.0, 0.0, 1.0)

    def _production(self, query_row: int, n: int) -> List[Tuple[int, Mapping[str, Any]]]:
        """Run the wrapped production path and resolve served items to exact rows."""
        if (
            getattr(self.rec, "_centroid_idx", None) is None
            and hasattr(self.rec, "_load_enhancements")
        ):
            self.rec._load_enhancements(None)
        output = self.rec.recommend(
            query_row,
            n=n,
            alpha=getattr(self.rec, "alpha", 0.8),
            diversity=0.15,
            max_per_artist=1,
            quality_filter=True,
            genre_rerank=True,
        )
        served = output.get("results", output) if isinstance(output, Mapping) else output
        resolved = []
        for item in served:
            if "row" in item:
                row = int(item["row"])
            else:
                raw_id = item.get("deezer_id", item.get("track_id"))
                key = raw_id.item() if isinstance(raw_id, np.generic) else raw_id
                candidates = self._rows_by_track_id.get(key, [])
                row = next(
                    (
                        candidate
                        for candidate in candidates
                        if str(self.titles[candidate]) == str(item.get("title"))
                        and str(self.artists[candidate]) == str(item.get("artist"))
                    ),
                    candidates[0] if candidates else -1,
                )
                if row < 0:
                    raise ValueError(f"Served track id {raw_id!r} is absent from the index")
            resolved.append((row, item))
        return resolved

    def _serialize(
        self,
        row: int,
        position: int,
        source: str,
        *,
        graph: float = 0.0,
        audio: float = 0.0,
        style: float = 0.0,
        score: Optional[float] = None,
        query_mode: str,
        lastfm_graph: float = 0.0,
        music4all_graph: float = 0.0,
    ) -> Dict[str, Any]:
        track_id = self.track_ids[row]
        track_id = track_id.item() if isinstance(track_id, np.generic) else track_id
        return {
            "position": position,
            "row": int(row),
            "title": str(self.titles[row]),
            "artist": str(self.artists[row]),
            "track_id": track_id,
            "score": float(score if score is not None else 0.0),
            "rationale": {
                "G": float(graph),
                "A": float(audio),
                "S": float(style),
                "lastfm_G": float(lastfm_graph),
                "music4all_G": float(music4all_graph),
                "source": source,
                "query_mode": query_mode,
            },
        }

    def _raw_graph_candidates(
        self, query_row: int, audio: np.ndarray, neighborhood: Mapping[str, Any]
    ) -> List[Tuple[int, float, float, float, float, float]]:
        lastfm = _source_components(neighborhood["lastfm"])
        music4all = _source_components(neighborhood["music4all"])
        seed_title = str(self.titles[query_row])
        seed_artists = _artist_parts(str(self.artists[query_row]))
        candidates = []
        for artist_id in neighborhood["union_artist_ids"]:
            artist_id = int(artist_id)
            start = int(self.graph.track_indptr[artist_id])
            stop = int(self.graph.track_indptr[artist_id + 1])
            rows = self.graph.track_rows[start:stop]
            for selected in _top_rows(audio[rows], min(16, len(rows))):
                row = int(rows[int(selected)])
                title, artist = str(self.titles[row]), str(self.artists[row])
                if row == query_row or seed_artists & _artist_parts(artist):
                    continue
                if self.quality_filter.is_junk(title, artist):
                    continue
                if self.quality_filter.seed_title_in_result(seed_title, title):
                    continue
                left = lastfm.get(artist_id, 0.0)
                right = music4all.get(artist_id, 0.0)
                graph = 0.5 * (left + right)
                a_value = float(audio[row])
                style = float(
                    self.styles.style_overlap(str(self.artists[query_row]), artist)
                )
                candidates.append((row, graph, a_value, style, left, right))
        return candidates

    def _rank_graph_candidates(
        self,
        candidates: Sequence[Tuple[int, float, float, float, float, float]],
        policy: CatalogPolicy,
    ) -> List[Tuple[int, float, float, float, float, float, float]]:
        candidates = [
            (
                row, policy_score(graph, audio, style, policy),
                graph, audio, style, left, right,
            )
            for row, graph, audio, style, left, right in candidates
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        deduped = []
        recordings, used_artists = set(), set()
        for item in candidates:
            row = item[0]
            artist = normalize_text(str(self.artists[row]))
            recording = (normalize_text(str(self.titles[row])), artist)
            if recording in recordings or artist in used_artists:
                continue
            recordings.add(recording)
            used_artists.add(artist)
            deduped.append(item)
        return deduped

    def _graph_candidates(
        self, query_row: int, audio: np.ndarray, neighborhood: Mapping[str, Any]
    ) -> List[Tuple[int, float, float, float, float, float, float]]:
        return self._rank_graph_candidates(
            self._raw_graph_candidates(query_row, audio, neighborhood), self.policy
        )

    @staticmethod
    def _gate_reason(
        coverage: Mapping[str, Any],
        shared_count: int,
        agreement: float,
        consistency: float,
        policy: CatalogPolicy,
    ) -> str:
        if not all(bool(coverage.get(name)) for name in ("lastfm", "music4all")):
            return "missing_independent_source"
        if shared_count < 5:
            return "fewer_than_five_shared_neighbors"
        if agreement < policy.tau:
            return "agreement_below_tau"
        if consistency < policy.sigma:
            return "consistency_below_sigma"
        return "both_gates_passed"

    def precompute_policy_query(
        self,
        query_row: int,
        *,
        production_rows: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Cache all target-blind data needed to apply any three-field policy."""
        if query_row < 0 or query_row >= len(self.titles):
            raise IndexError("query_row is outside the catalogue")
        audio = self.audio_scores(query_row)
        if hasattr(self.graph, "dual_source_neighbors"):
            neighborhood = self.graph.dual_source_neighbors(
                str(self.artists[query_row])
            )
        else:
            neighborhood = {
                "lastfm": {"artist_ids": np.empty(0), "weights": np.empty(0)},
                "music4all": {"artist_ids": np.empty(0), "weights": np.empty(0)},
                "union_artist_ids": np.empty(0, dtype=np.int32),
                "source_coverage": {"lastfm": False, "music4all": False},
            }
        coverage = dict(neighborhood["source_coverage"])
        agreement, shared_count = source_agreement(
            neighborhood["lastfm"], neighborhood["music4all"]
        )
        raw = (
            self._raw_graph_candidates(query_row, audio, neighborhood)
            if all(coverage.values()) else []
        )
        components = [
            {
                "row": int(row), "G": float(graph), "A": float(a_value),
                "S": float(style), "lastfm_G": float(left),
                "music4all_G": float(right), "source": "dual_source_graph",
            }
            for row, graph, a_value, style, left, right in raw
        ]
        return {
            "query_row": int(query_row),
            "production_method": "dual_sonic",
            "production_rows": list(map(int, production_rows or [])),
            "components": components,
            "graph_union_rows": list(dict.fromkeys(
                int(component["row"]) for component in components
            )),
            "gate_components": {
                "agreement": float(agreement),
                "shared_count": int(shared_count),
                "source_coverage": coverage,
            },
            "policy_application": lambda policy, n=10: self.apply_precomputed_policy(
                {
                    "production_rows": list(map(int, production_rows or [])),
                    "components": components,
                    "graph_union_rows": list(dict.fromkeys(
                        int(component["row"]) for component in components
                    )),
                    "gate_components": {
                        "agreement": float(agreement),
                        "shared_count": int(shared_count),
                        "source_coverage": coverage,
                    },
                },
                policy,
                n,
            ),
        }

    def apply_precomputed_policy(
        self,
        cached: Mapping[str, Any],
        policy: CatalogPolicy,
        n: int = 10,
    ) -> Dict[str, Any]:
        """Apply the exact runtime gate to cached components without audio work."""
        raw = [
            (
                int(item["row"]), float(item["G"]), float(item["A"]),
                float(item["S"]), float(item.get("lastfm_G", 0.0)),
                float(item.get("music4all_G", 0.0)),
            )
            for item in cached.get("components", [])
        ]
        ranked = self._rank_graph_candidates(raw, policy)
        # Candidate recall uses the actual one-per-artist graph pool from which
        # the replacement head is chosen, not every alternate track considered
        # while selecting the best recording for an artist.
        all_graph_rows = [int(item[0]) for item in ranked]
        if len(ranked) >= 5:
            top5 = ranked[:5]
            consistency = float(min(
                np.mean([item[3] for item in top5]),
                np.mean([item[4] for item in top5]),
                min(item[4] for item in top5[:3]),
            ))
        else:
            consistency = 0.0
        gate = cached.get("gate_components", {})
        coverage = dict(gate.get("source_coverage", {}))
        agreement = float(gate.get("agreement", 0.0))
        shared_count = int(gate.get("shared_count", 0))
        reason = self._gate_reason(
            coverage, shared_count, agreement, consistency, policy
        )
        fired = reason == "both_gates_passed"
        production = list(map(int, cached.get("production_rows", [])))
        head = [int(item[0]) for item in ranked[:min(5, max(0, int(n)))]] if fired else []
        ranking = head + [
            row for row in production if row not in set(head)
        ][:max(0, int(n)-len(head))]
        return {
            "ranking_rows": ranking,
            "candidate_rows": (
                list(dict.fromkeys(all_graph_rows + production)) if fired
                else production
            ),
            "fired": fired,
            "reason": reason,
            "agreement": agreement,
            "consistency": consistency,
            "shared_count": shared_count,
            "source_coverage": coverage,
        }

    def recommend(self, query_row: int, n: int = 20) -> Dict[str, Any]:
        if query_row < 0 or query_row >= len(self.titles):
            raise IndexError("query_row is outside the catalogue")
        n = max(int(n), 0)
        production = self._production(query_row, n)
        audio = self.audio_scores(query_row)
        if hasattr(self.graph, "dual_source_neighbors"):
            neighborhood = self.graph.dual_source_neighbors(
                str(self.artists[query_row])
            )
        else:
            neighborhood = {
                "lastfm": {"artist_ids": np.empty(0), "weights": np.empty(0)},
                "music4all": {"artist_ids": np.empty(0), "weights": np.empty(0)},
                "union_artist_ids": np.empty(0, dtype=np.int32),
                "source_coverage": {"lastfm": False, "music4all": False},
                "mode": "dual_source_unavailable",
            }
        coverage = dict(neighborhood["source_coverage"])
        agreement, shared_count = source_agreement(
            neighborhood["lastfm"], neighborhood["music4all"]
        )
        graph_candidates = (
            self._graph_candidates(query_row, audio, neighborhood)
            if all(coverage.values())
            else []
        )
        if len(graph_candidates) >= 5:
            top5 = graph_candidates[:5]
            consistency = float(
                min(
                    np.mean([item[3] for item in top5]),
                    np.mean([item[4] for item in top5]),
                    min(item[4] for item in top5[:3]),
                )
            )
        else:
            consistency = 0.0

        reason = self._gate_reason(
            coverage, shared_count, agreement, consistency, self.policy
        )
        fired = reason == "both_gates_passed"
        gate = {
            "fired": fired,
            "reason": reason,
            "agreement": float(agreement),
            "consistency": float(consistency),
            "thresholds": {
                "tau": float(self.policy.tau),
                "sigma": float(self.policy.sigma),
            },
            "shared_count": int(shared_count),
            "source_coverage": {
                **coverage,
                "lastfm_candidates": int(
                    len(neighborhood["lastfm"]["artist_ids"])
                ),
                "music4all_candidates": int(
                    len(neighborhood["music4all"]["artist_ids"])
                ),
            },
        }
        if not fired:
            results = [
                self._serialize(
                    row,
                    position,
                    "production_abstention",
                    audio=float(audio[row]),
                    style=float(
                        self.styles.style_overlap(
                            str(self.artists[query_row]), str(self.artists[row])
                        )
                    ),
                    query_mode="production_abstention",
                )
                for position, (row, _) in enumerate(production, 1)
            ]
            mode = "production_abstention"
        else:
            head_count = min(5, n, len(graph_candidates))
            results = []
            used_rows, used_artists = set(), set()
            for item in graph_candidates[:head_count]:
                row, total, graph, a_value, style, left, right = item
                results.append(
                    self._serialize(
                        row,
                        len(results) + 1,
                        "dual_source_graph",
                        graph=graph,
                        audio=a_value,
                        style=style,
                        score=total,
                        query_mode="dual_source_graph",
                        lastfm_graph=left,
                        music4all_graph=right,
                    )
                )
                used_rows.add(row)
                used_artists.add(normalize_text(str(self.artists[row])))
            for row, _ in production:
                artist = normalize_text(str(self.artists[row]))
                if len(results) >= n:
                    break
                if row in used_rows or artist in used_artists:
                    continue
                results.append(
                    self._serialize(
                        row,
                        len(results) + 1,
                        "production_tail",
                        audio=float(audio[row]),
                        style=float(
                            self.styles.style_overlap(
                                str(self.artists[query_row]), str(self.artists[row])
                            )
                        ),
                        query_mode="dual_source_graph",
                    )
                )
                used_rows.add(row)
                used_artists.add(artist)
            mode = "dual_source_graph"
        return {
            "query": {
                "row": int(query_row),
                "title": str(self.titles[query_row]),
                "artist": str(self.artists[query_row]),
            },
            "query_mode": mode,
            "mode": mode,
            "gate": gate,
            "results": results,
        }

    rank = recommend


GraphFirstCatalogRanker = CatalogPolicyRanker
