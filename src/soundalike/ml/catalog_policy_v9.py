"""Low-capacity Last.fm-first policy for powered served-list evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog_policy import (
    CatalogPolicyRanker,
    _artist_parts,
    _source_components,
    _top_rows,
)
from .quality_filter import TitleQualityFilter
from .real_benchmark import normalize_text


OPTIONAL_MUSIC4ALL_WEIGHT = 0.15
OPTIONAL_MUSIC4ALL_CONFIDENCE_BONUS = 0.05
HEAD_SIZE = 5


@dataclass(frozen=True)
class ListPolicy:
    """The entire tunable surface: confidence, consistency, and audio tie-break."""

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
            raise ValueError("audio_weight must be finite and non-negative")
        object.__setattr__(self, "audio_weight", weight)


DEFAULT_LIST_POLICY = ListPolicy(0.55, 0.30, 0.05)
DEFAULT_LIST_POLICY_GRID = tuple(
    ListPolicy(tau, sigma, audio_weight)
    for tau in (0.35, 0.50, 0.65, 0.80)
    for sigma in (0.15, 0.25, 0.35, 0.45)
    for audio_weight in (0.0, 0.05, 0.10)
)


def _source_confidence(
    lastfm: Mapping[str, np.ndarray],
    music4all: Mapping[str, np.ndarray],
) -> Tuple[float, float, int]:
    """Return Last.fm strength and a fixed optional cross-source bonus."""
    weights = np.maximum(
        np.asarray(lastfm.get("weights", ()), dtype=np.float32), 0.0
    )
    if not len(weights) or float(weights.max()) <= 0.0:
        return 0.0, 0.0, 0
    normalized = weights / float(weights.max())
    base = float(np.mean(normalized[: min(5, len(normalized))]))
    left = set(map(int, np.asarray(lastfm.get("artist_ids", ())).tolist()))
    right = set(map(int, np.asarray(music4all.get("artist_ids", ())).tolist()))
    shared = len(left & right)
    bonus = OPTIONAL_MUSIC4ALL_CONFIDENCE_BONUS if shared else 0.0
    return float(min(1.0, base + bonus)), float(base), int(shared)


def _policy_score(graph: float, audio: float, policy: ListPolicy) -> float:
    return float(graph) + float(policy.audio_weight) * float(audio)


class LastfmListRanker(CatalogPolicyRanker):
    """Last.fm confidence gate with optional Music4All corroboration.

    Music4All is never required for coverage. Sigma is applied to every
    candidate track as ``min(track_audio_similarity, artist_style_overlap)``.
    """

    def __init__(
        self,
        recommender: Any,
        graph: Any,
        styles: Any,
        policy: ListPolicy = DEFAULT_LIST_POLICY,
        quality_filter: Optional[TitleQualityFilter] = None,
    ):
        super().__init__(
            recommender,
            graph,
            styles,
            policy=None,  # type: ignore[arg-type]
            quality_filter=quality_filter,
        )
        self.policy = policy

    def _raw_list_candidates(
        self,
        query_row: int,
        audio: np.ndarray,
        neighborhood: Mapping[str, Any],
    ) -> List[Dict[str, float | int]]:
        lastfm = _source_components(neighborhood["lastfm"])
        music4all = _source_components(neighborhood["music4all"])
        seed_title = str(self.titles[query_row])
        seed_artists = _artist_parts(str(self.artists[query_row]))
        candidates: List[Dict[str, float | int]] = []
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
                if not self.quality_filter.is_eligible_for_query(
                    seed_title,
                    str(self.artists[query_row]),
                    title,
                    artist,
                ):
                    continue
                if self.quality_filter.seed_title_in_result(seed_title, title):
                    continue
                left = float(lastfm.get(artist_id, 0.0))
                right = float(music4all.get(artist_id, 0.0))
                graph = left + OPTIONAL_MUSIC4ALL_WEIGHT * right
                a_value = float(audio[row])
                style = float(
                    self.styles.style_overlap(
                        str(self.artists[query_row]), str(self.artists[row])
                    )
                )
                candidates.append({
                    "row": row,
                    "title": title,
                    "artist": artist,
                    "G": graph,
                    "A": a_value,
                    "S": style,
                    "song_consistency": min(a_value, style),
                    "lastfm_G": left,
                    "music4all_G": right,
                })
        return candidates

    @staticmethod
    def _neighborhood(graph: Any, query_artist: str) -> Dict[str, Any]:
        if hasattr(graph, "dual_source_neighbors"):
            return dict(graph.dual_source_neighbors(query_artist))
        return {
            "lastfm": {
                "artist_ids": np.empty(0, dtype=np.int32),
                "weights": np.empty(0, dtype=np.float32),
            },
            "music4all": {
                "artist_ids": np.empty(0, dtype=np.int32),
                "weights": np.empty(0, dtype=np.float32),
            },
            "union_artist_ids": np.empty(0, dtype=np.int32),
            "source_coverage": {"lastfm": False, "music4all": False},
        }

    def precompute_list_query(
        self,
        query_row: int,
        *,
        production_rows: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        if query_row < 0 or query_row >= len(self.titles):
            raise IndexError("query_row is outside the catalogue")
        audio = self.audio_scores(query_row)
        neighborhood = self._neighborhood(
            self.graph, str(self.artists[query_row])
        )
        coverage = dict(neighborhood["source_coverage"])
        confidence, base, shared = _source_confidence(
            neighborhood["lastfm"], neighborhood["music4all"]
        )
        raw = (
            self._raw_list_candidates(query_row, audio, neighborhood)
            if bool(coverage.get("lastfm"))
            else []
        )
        return {
            "query_row": int(query_row),
            "production_method": "current_production_dual_sonic",
            "production_rows": list(map(int, production_rows or ())),
            "components": raw,
            "graph_union_rows": list(dict.fromkeys(
                int(item["row"]) for item in raw
            )),
            "gate_components": {
                "lastfm_confidence": confidence,
                "lastfm_base_confidence": base,
                "music4all_shared_neighbors": shared,
                "source_coverage": coverage,
            },
        }

    def apply_precomputed_list_policy(
        self,
        cached: Mapping[str, Any],
        policy: ListPolicy,
        n: int = 10,
    ) -> Dict[str, Any]:
        candidates = [
            {
                **dict(item),
                "score": _policy_score(
                    float(item["G"]), float(item["A"]), policy
                ),
            }
            for item in cached.get("components", ())
            if float(item["song_consistency"]) >= policy.sigma
        ]
        candidates.sort(key=lambda item: (-float(item["score"]), int(item["row"])))
        candidates = [
            dict(item) for item in self.quality_filter.prefer_canonical(candidates)
        ]
        ranked: List[Dict[str, Any]] = []
        used_artists = set()
        for item in candidates:
            artist = normalize_text(str(self.artists[int(item["row"])]))
            if artist in used_artists:
                continue
            used_artists.add(artist)
            ranked.append(item)
        gate = dict(cached.get("gate_components", {}))
        coverage = dict(gate.get("source_coverage", {}))
        confidence = float(gate.get("lastfm_confidence", 0.0))
        if not bool(coverage.get("lastfm")):
            reason = "missing_lastfm_source"
        elif confidence < policy.tau:
            reason = "lastfm_confidence_below_tau"
        elif len(ranked) < HEAD_SIZE:
            reason = "fewer_than_five_song_consistent_candidates"
        else:
            reason = "lastfm_and_song_consistency_passed"
        fired = reason == "lastfm_and_song_consistency_passed"
        production = list(map(int, cached.get("production_rows", ())))
        head = [int(item["row"]) for item in ranked[: min(HEAD_SIZE, n)]] if fired else []
        head_set = set(head)
        head_artists = {
            normalize_text(str(self.artists[row])) for row in head
        }
        tail = [
            row for row in production
            if row not in head_set
            and normalize_text(str(self.artists[row])) not in head_artists
        ]
        ranking = (head + tail)[: max(0, int(n))]
        return {
            "ranking_rows": ranking,
            "candidate_rows": (
                list(dict.fromkeys(
                    [int(item["row"]) for item in ranked] + production
                ))
                if fired else production
            ),
            "ranked_components": ranked,
            "fired": fired,
            "reason": reason,
            "lastfm_confidence": confidence,
            "lastfm_base_confidence":
                float(gate.get("lastfm_base_confidence", 0.0)),
            "music4all_shared_neighbors":
                int(gate.get("music4all_shared_neighbors", 0)),
            "source_coverage": coverage,
            "eligible_candidate_count": len(ranked),
            "policy": asdict(policy),
        }

    def recommend(self, query_row: int, n: int = 10) -> Dict[str, Any]:
        production = self._production(query_row, n)
        cached = self.precompute_list_query(
            query_row, production_rows=[row for row, _ in production]
        )
        applied = self.apply_precomputed_list_policy(cached, self.policy, n)
        by_row = {
            int(item["row"]): item for item in applied["ranked_components"]
        }
        results = []
        for position, row in enumerate(applied["ranking_rows"], start=1):
            item = by_row.get(int(row), {})
            source = (
                "lastfm_graph_optional_music4all"
                if int(row) in by_row and applied["fired"]
                else "production_abstention_or_tail"
            )
            results.append(self._serialize(
                int(row),
                position,
                source,
                graph=float(item.get("G", 0.0)),
                audio=float(item.get("A", 0.0)),
                style=float(item.get("S", 0.0)),
                score=float(item.get("score", 0.0)),
                query_mode=(
                    "lastfm_list_policy" if applied["fired"]
                    else "production_abstention"
                ),
                lastfm_graph=float(item.get("lastfm_G", 0.0)),
                music4all_graph=float(item.get("music4all_G", 0.0)),
            ))
            results[-1]["rationale"]["song_consistency"] = float(
                item.get("song_consistency", 0.0)
            )
        return {
            "query": {
                "row": int(query_row),
                "title": str(self.titles[query_row]),
                "artist": str(self.artists[query_row]),
            },
            "query_mode": (
                "lastfm_list_policy" if applied["fired"]
                else "production_abstention"
            ),
            "gate": {
                key: value for key, value in applied.items()
                if key not in {
                    "ranking_rows", "candidate_rows", "ranked_components"
                }
            },
            "results": results,
        }


__all__ = [
    "DEFAULT_LIST_POLICY",
    "DEFAULT_LIST_POLICY_GRID",
    "HEAD_SIZE",
    "LastfmListRanker",
    "ListPolicy",
    "OPTIONAL_MUSIC4ALL_CONFIDENCE_BONUS",
    "OPTIONAL_MUSIC4ALL_WEIGHT",
]
