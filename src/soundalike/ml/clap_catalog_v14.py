"""CLAP catalogue ranker v14 — development-only, identity-guarded challenger.

Extends :mod:`clap_catalog_v13` with four targeted improvements, all
development-only and **not** wired into production:

1. **Stable row identity** — dedup/filter by primary Deezer artist ID (from
   ``IdentityAsset``), with contributor-intersection cross-check, and plain
   normalised-name fallback only for rows that have no resolved Deezer ID.
   Distinct same-name artists (different Deezer IDs) are never collapsed in
   MMR; the same artist appearing under multiple spellings/casing/multi-credit
   attributions (same Deezer ID) is correctly deduped.

2. **Generic source-profile ambiguity guard** — uses only predeclared graph
   artist-audio centroids (``CatalogArtistGraph.artist_audio``) and style
   vectors (``CatalogStyleIndex``) with fixed constants.  No artist-specific
   conditions; the confidence is a deterministic weighted average over top-K
   graph neighbours.  When confidence falls below threshold, or query-ID
   resolution confidence/margin is insufficient, the conservative challenger
   abstains to exact current production with an explicit ``fallback_reason``.

   Generic default thresholds (documented here, not tuned to any artist):

   * ``GUARD_AUDIO_WEIGHT = 0.55`` — audio centroid weight (majority because
     only 1.1% of catalog style vectors are direct metadata)
   * ``GUARD_STYLE_WEIGHT = 0.45`` — style-overlap weight
   * ``GUARD_SOURCE_PROFILE_MIN = 0.62`` — combined source-profile minimum
     (broad positive-cosine agreement across top-K neighbours)
   * ``GUARD_CANDIDATE_MIN = 0.60`` — minimum cosine for candidate ID
     centroid resolution
   * ``GUARD_CENTROID_MIN = 0.60``  — ``IdentityAsset.disambiguate`` min
     confidence
   * ``GUARD_CENTROID_MARGIN = 0.05`` — ``IdentityAsset.disambiguate`` min
     margin

3. **Identity-guarded graph expansion** — for every graph-neighbour name
   group, tracks are grouped by stable Deezer ID.  Unresolved rows are
   excluded from mixed groups.  When multiple Deezer IDs share a name
   (homonym), only the cluster closest to the query-artist centroid is used if
   the confidence/margin gates pass; otherwise the whole name group is omitted.
   Entirely unresolved name groups are omitted.  At most one cluster per name
   is ever expanded.

4. **v14 diagnostics** — ``run_variant_diagnostics_v14`` follows the v13
   60-seed proxy methodology with ``commercial_human_ratings_used=0``,
   ``proxy_evidence_is_deciding=False``, ``production=False``, and adds:
   identity-guard counts, semantic diff against v13 selected rows, candidate
   coverage, and a deterministic 20-seed direct actual-list diagnostic
   spanning all 13 v13 scene labels (diagnostic only, not human gold).

v13 and production are **not modified**.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .artist_identity_v14 import IdentityAsset, normalize_key
from .clap_catalog_v13 import (
    VARIANT_ORDER,
    ClapCatalogError,
    ClapDevelopmentRanker,
    _deezer_related,
    _list_metrics,
    _proxy_passes,
    _sha256_path,
    _top_indices,
    _write_json,
)
from .human_eval_v10 import content_hash
from .real_benchmark import normalize_text

# ---------------------------------------------------------------------------
# Schema & generic threshold constants (documented defaults, not tuned to any
# artist or dataset-specific observation)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 14

# Source-profile ambiguity guard weights (audio + style = 1.0)
GUARD_AUDIO_WEIGHT: float = 0.55
"""Weight of the audio-centroid cosine in the source-profile confidence score."""

GUARD_STYLE_WEIGHT: float = 0.45
"""Weight of the style-overlap score in the source-profile confidence score."""

GUARD_SOURCE_PROFILE_MIN: float = 0.62
"""Minimum combined source-profile confidence to proceed with graph signal.

Based on broad positive-cosine agreement across top-K graph neighbours.
When the combined score falls below this value the conservative challenger
abstains to exact production output.
"""

GUARD_CANDIDATE_MIN: float = 0.60
"""Minimum cosine confidence from ``IdentityAsset.disambiguate`` for a
candidate name-cluster ID to be considered resolved.
"""

GUARD_CENTROID_MIN: float = 0.60
"""Minimum cosine confidence required by ``IdentityAsset.disambiguate``
for stable centroid-based ID selection.
"""

GUARD_CENTROID_MARGIN: float = 0.05
"""Minimum cosine margin (best − second-best) required by
``IdentityAsset.disambiguate`` for stable centroid-based ID selection.
"""

GUARD_TOP_K: int = 5
"""Number of top-weighted graph neighbours used to estimate source-profile
confidence.  Constant; no artist-specific adjustment.
"""

VARIANT_ORDER_V14 = VARIANT_ORDER  # same conservative → graph → pure order


# ---------------------------------------------------------------------------
# Standalone guard helpers (testable without the full ranker)
# ---------------------------------------------------------------------------


def _normalise(vec: np.ndarray) -> Optional[np.ndarray]:
    """Return L2-normalised float32 vector or None if zero."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    return (v / norm) if norm >= 1e-8 else None


def compute_source_profile_confidence(
    query_audio: np.ndarray,
    query_artist_name: str,
    top_neighbor_graph_ids: Sequence[int],
    graph_artist_audio: np.ndarray,
    graph_artist_names: np.ndarray,
    style_overlap_fn: Callable[[str, str], float],
    *,
    top_k: int = GUARD_TOP_K,
) -> Tuple[float, str]:
    """Generic source-profile ambiguity confidence over top-K graph neighbours.

    Uses only predeclared graph artist-audio centroids and the style-overlap
    function.  Constants: ``GUARD_AUDIO_WEIGHT`` and ``GUARD_STYLE_WEIGHT``.
    No artist-specific conditions; the result is deterministic.

    Parameters
    ----------
    query_audio:
        L2-normalised audio centroid for the query (from graph artist_audio or
        CLAP compact embedding — must be in the same space as
        ``graph_artist_audio``).
    query_artist_name:
        Raw artist name string for style-overlap lookup.
    top_neighbor_graph_ids:
        Pre-sorted graph artist IDs (highest Last.fm weight first), up to
        ``top_k`` used.
    graph_artist_audio:
        2-D float array of shape ``(num_artists, dim)`` — the graph's
        precomputed audio centroids.
    graph_artist_names:
        1-D string array of length ``num_artists`` — graph artist name labels.
    style_overlap_fn:
        Callable ``(query_name, candidate_name) -> float`` in ``[0, 1]``.
    top_k:
        Maximum neighbours to evaluate.

    Returns
    -------
    (confidence, reason) where confidence ∈ [0, 1] and reason is a
    diagnostic string.
    """
    q = _normalise(query_audio)
    if q is None:
        return 0.0, "query_audio_zero"

    candidates = [
        int(aid)
        for aid in top_neighbor_graph_ids[:top_k]
        if 0 <= int(aid) < len(graph_artist_audio)
    ]
    if not candidates:
        return 0.0, "no_graph_neighbors"

    audio_scores: List[float] = []
    style_scores: List[float] = []
    for aid in candidates:
        n_vec = _normalise(graph_artist_audio[aid])
        if n_vec is not None and n_vec.shape == q.shape:
            raw_cos = float(q @ n_vec)
            # Negative cosine is no agreement; positive cosine keeps its
            # native scale so the audio/style threshold remains interpretable.
            audio_scores.append(float(np.clip(raw_cos, 0.0, 1.0)))
        cand_name = str(graph_artist_names[aid])
        style_scores.append(float(style_overlap_fn(query_artist_name, cand_name)))

    if not audio_scores:
        return 0.0, "no_valid_audio_centroids"

    audio_mean = float(np.mean(audio_scores))
    style_mean = float(np.mean(style_scores)) if style_scores else 0.0
    confidence = GUARD_AUDIO_WEIGHT * audio_mean + GUARD_STYLE_WEIGHT * style_mean
    return confidence, "ok"


def apply_identity_guard_to_name_group(
    artist_name_str: str,
    track_rows: Sequence[int],
    row_to_stable_id: Mapping[int, Optional[int]],
    name_to_deezer_ids_fn: Callable[[str], List[int]],
    disambiguate_fn: Callable[
        [str, np.ndarray], Optional[Tuple[int, float, float]]
    ],
    query_vec: Optional[np.ndarray],
    unresolved_matches_fn: Optional[Callable[[int, int], bool]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Return at most one evidence-backed Deezer-ID cluster for a graph name."""
    id_to_rows: Dict[Optional[int], List[int]] = defaultdict(list)
    for row in track_rows:
        id_to_rows[row_to_stable_id.get(row)].append(row)
    resolved_ids = {sid for sid in id_to_rows if sid is not None}
    unresolved = id_to_rows.get(None, [])
    name_ids = set(name_to_deezer_ids_fn(artist_name_str))

    if not resolved_ids:
        return [], {
            "artist_name": artist_name_str,
            "action": "omitted_all_unresolved",
            "unresolved_count": len(unresolved),
        }

    if len(resolved_ids) == 1:
        chosen_id = next(iter(resolved_ids))
        rows_out = list(id_to_rows[chosen_id])
        inferred_rows = [
            row
            for row in unresolved
            if unresolved_matches_fn is not None
            and unresolved_matches_fn(row, chosen_id)
        ]
        rows_out.extend(inferred_rows)
        return rows_out, {
            "artist_name": artist_name_str,
            "action": "resolved_single_id",
            "deezer_id": chosen_id,
            "name_cluster_ids": sorted(name_ids),
            "tracks_added": len(rows_out),
            "unresolved_inferred_by_audio": len(inferred_rows),
            "unresolved_excluded": len(unresolved) - len(inferred_rows),
        }

    if query_vec is None:
        return [], {
            "artist_name": artist_name_str,
            "action": "omitted_homonym_no_query_vec",
            "candidate_ids": sorted(resolved_ids),
            "name_cluster_ids": sorted(name_ids),
        }
    result = disambiguate_fn(artist_name_str, query_vec)
    if result is None:
        return [], {
            "artist_name": artist_name_str,
            "action": "omitted_homonym_disambiguation_failed",
            "candidate_ids": sorted(resolved_ids),
            "name_cluster_ids": sorted(name_ids),
        }
    chosen_id, confidence, margin = result
    rows_out = list(id_to_rows.get(chosen_id, []))
    if not rows_out:
        return [], {
            "artist_name": artist_name_str,
            "action": "omitted_chosen_cluster_empty",
            "chosen_id": chosen_id,
            "confidence": confidence,
            "margin": margin,
        }
    return rows_out, {
        "artist_name": artist_name_str,
        "action": "homonym_disambiguated",
        "chosen_deezer_id": chosen_id,
        "confidence": confidence,
        "margin": margin,
        "tracks_added": len(rows_out),
        "unresolved_excluded": len(unresolved),
    }


# ---------------------------------------------------------------------------
# IdentityGuardedRanker
# ---------------------------------------------------------------------------


class IdentityGuardedRanker(ClapDevelopmentRanker):
    """v14 development-only ranker with identity guard and stable-ID MMR.

    Subclasses :class:`ClapDevelopmentRanker`.  All v13 weights, gates, and
    variant order are preserved; the only behavioural changes are:

    * MMR deduplication uses stable Deezer artist ID (or normalised name as
      fallback) rather than raw normalised-name key.
    * Same-artist seed exclusion uses primary Deezer ID / contributor
      intersection before falling back to name-part matching.
    * Graph candidates are filtered through the identity guard before use.
    * The conservative challenger abstains to exact production when the
      source-profile confidence or query-ID resolution is insufficient.

    Parameters
    ----------
    index_path, compact_path, status_path, graph_path:
        Same as :class:`ClapDevelopmentRanker`.
    style_path:
        Path to the ``CatalogStyleIndex`` NPZ used for the style component of
        the source-profile guard.
    identity_asset:
        Optional pre-loaded :class:`IdentityAsset`.  When ``None`` the ranker
        falls back gracefully to v13 name-based behaviour.
    """

    def __init__(
        self,
        index_path: Path,
        compact_path: Path,
        status_path: Path,
        graph_path: Path,
        style_path: Path,
        identity_asset: Optional[IdentityAsset] = None,
    ):
        super().__init__(index_path, compact_path, status_path, graph_path)
        from .catalog_style import CatalogStyleIndex

        self.style = CatalogStyleIndex(style_path)
        self.identity: Optional[IdentityAsset] = identity_asset

        # Sparse cache only for audio-inferred IDs; direct IDs remain in the
        # compact aligned numpy array owned by IdentityAsset.
        self._stable_id_for_row: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Stable key helpers
    # ------------------------------------------------------------------

    def _resolved_id_for_row(self, row: int) -> Optional[int]:
        """Return stable Deezer ID, or an audio-confident catalog inference."""
        cached = self._stable_id_for_row.get(row)
        if cached is not None:
            return cached
        if self.identity is None:
            return None
        direct = self.identity.primary_artist_id(row)
        if direct is not None:
            return direct
        vector = self._query_centroid(row)
        if vector is None:
            return None
        result = self.identity.disambiguate(
            str(self.artists[row]),
            vector,
            min_confidence=GUARD_CENTROID_MIN,
            min_margin=GUARD_CENTROID_MARGIN,
        )
        if result is None:
            return None
        inferred = int(result[0])
        self._stable_id_for_row[row] = inferred
        return inferred

    def _stable_key(self, row: int) -> str:
        """Use stable or confidently inferred Deezer ID before name fallback."""
        stable_id = self._resolved_id_for_row(row)
        if stable_id is not None:
            return f"did:{stable_id}"
        return f"name:{normalize_text(str(self.artists[row]))}"

    def _query_centroid(self, query_row: int) -> Optional[np.ndarray]:
        """Audio centroid for the query row (CLAP compact embedding)."""
        if not self.available[query_row]:
            return None
        return _normalise(np.asarray(self.compact[query_row], dtype=np.float32))

    def _is_same_artist_stable(self, query_row: int, candidate_row: int) -> bool:
        """Use stable primary/contributor IDs, then unresolved-name fallback."""
        from .catalog_policy import _artist_parts

        if self.identity is not None:
            query_id = self._resolved_id_for_row(query_row)
            candidate_id = self._resolved_id_for_row(candidate_row)
            if (
                query_id is not None
                and candidate_id is not None
                and query_id == candidate_id
            ):
                return True
            if self.identity.contributor_intersection(
                [query_row], [candidate_row]
            ):
                return True
            if query_id is not None and candidate_id is not None:
                return False

        return bool(
            _artist_parts(str(self.artists[query_row]))
            & _artist_parts(str(self.artists[candidate_row]))
        )

    # ------------------------------------------------------------------
    # Source-profile ambiguity guard
    # ------------------------------------------------------------------

    def _source_profile_confidence(
        self,
        query_row: int,
        top_neighbor_graph_ids: Sequence[int],
    ) -> Tuple[float, str]:
        """Compute source-profile confidence for the query's top graph neighbours.

        Returns ``(confidence, reason)``.  Constants: ``GUARD_AUDIO_WEIGHT``,
        ``GUARD_STYLE_WEIGHT``.  No artist-specific conditions.
        """
        query_artist_name = str(self.artists[query_row])
        q_graph_id = self.graph.artist_lookup.get(normalize_text(query_artist_name))
        if q_graph_id is None:
            return 0.0, "query_not_in_graph"

        q_audio = _normalise(np.asarray(self.graph.artist_audio[q_graph_id], dtype=np.float32))
        if q_audio is None:
            return 0.0, "query_audio_zero"

        return compute_source_profile_confidence(
            q_audio,
            query_artist_name,
            top_neighbor_graph_ids,
            self.graph.artist_audio,
            self.graph.artist_names,
            self.style.style_overlap,
            top_k=GUARD_TOP_K,
        )

    # ------------------------------------------------------------------
    # Eligible / MMR overrides using stable ID
    # ------------------------------------------------------------------

    def _eligible_v14(
        self,
        query_row: int,
        candidates: Sequence[int],
        relevance: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Like ``_eligible`` but uses stable Deezer ID for same-artist check."""
        seed_title = str(self.titles[query_row])
        seed_artist = str(self.artists[query_row])
        values: List[Dict[str, Any]] = []
        for raw in candidates:
            row = int(raw)
            title, artist = str(self.titles[row]), str(self.artists[row])
            if (
                row == query_row
                or not self.available[row]
                or self._is_same_artist_stable(query_row, row)
                or not self.quality.is_eligible_for_query(
                    seed_title, seed_artist, title, artist
                )
                or self.quality.seed_title_in_result(seed_title, title)
            ):
                continue
            values.append(
                {
                    "row": row,
                    "title": title,
                    "artist": artist,
                    "artist_key": normalize_text(artist),
                    "stable_key": self._stable_key(row),
                    "relevance": float(relevance[row]),
                }
            )
        values.sort(key=lambda item: (-item["relevance"], item["row"]))
        return [dict(item) for item in self.quality.prefer_canonical(values)]

    def _mmr_v14(
        self, query_row: int, candidates: List[Dict[str, Any]], n: int = 5
    ) -> List[int]:
        """Run v13 MMR weights while deduplicating stable artist identities."""
        selected: List[Dict[str, Any]] = []
        used_keys: Set[str] = set()
        remaining = list(candidates)
        while remaining and len(selected) < n:
            best: Optional[Dict[str, Any]] = None
            best_score_key: Optional[Tuple[float, int]] = None
            for item in remaining:
                if item["stable_key"] in used_keys:
                    continue
                row = int(item["row"])
                diversity = 0.0
                if selected:
                    candidate = _normalise(
                        np.asarray(self.compact[row], dtype=np.float32)
                    )
                    if candidate is not None:
                        similarities: List[float] = []
                        for chosen in selected:
                            chosen_vector = _normalise(
                                np.asarray(
                                    self.compact[int(chosen["row"])],
                                    dtype=np.float32,
                                )
                            )
                            if chosen_vector is not None:
                                similarities.append(
                                    (float(candidate @ chosen_vector) + 1.0) / 2.0
                                )
                        diversity = max(similarities) if similarities else 0.0
                score = 0.85 * float(item["relevance"]) - 0.15 * diversity
                key = (score, -row)
                if best_score_key is None or key > best_score_key:
                    best, best_score_key = item, key
            if best is None:
                break
            selected.append(best)
            used_keys.add(best["stable_key"])
            remaining.remove(best)
        return [int(item["row"]) for item in selected]

    # ------------------------------------------------------------------
    # Identity-guarded graph candidate expansion
    # ------------------------------------------------------------------

    def _graph_candidates_v14(
        self, query_row: int
    ) -> Tuple[
        np.ndarray,
        Dict[int, float],
        Dict[int, float],
        float,
        bool,
        Dict[str, Any],
    ]:
        """Identity-guarded graph candidate expansion.

        Wraps v13 ``_graph_candidates`` logic with:
        - Per-neighbour-name Deezer ID grouping and unresolved exclusion
        - Homonym disambiguation via ``IdentityAsset.disambiguate``
        - One-cluster-per-name policy
        - Source-profile ambiguity guard

        Returns ``(graph_rows, last, music, confidence, has_lastfm, guard_diag)``.
        When the guard abstains, ``graph_rows`` is empty and
        ``guard_diag["guard_abstained"]`` is ``True``.
        """
        # --- Retrieve raw graph neighbours (same as v13) ---
        neighborhood = self.graph.dual_source_neighbors(str(self.artists[query_row]))
        last_ids = np.asarray(neighborhood["lastfm"]["artist_ids"], dtype=np.int64)
        last_weights = np.asarray(
            neighborhood["lastfm"]["weights"], dtype=np.float32
        )
        music_ids = np.asarray(
            neighborhood["music4all"]["artist_ids"], dtype=np.int64
        )
        music_weights = np.asarray(
            neighborhood["music4all"]["weights"], dtype=np.float32
        )
        last_max = max(
            float(last_weights.max()) if len(last_weights) else 0.0, 1e-8
        )
        music_max = max(
            float(music_weights.max()) if len(music_weights) else 0.0, 1e-8
        )
        last: Dict[int, float] = {
            int(a): float(w / last_max) for a, w in zip(last_ids, last_weights)
        }
        music: Dict[int, float] = {
            int(a): float(w / music_max)
            for a, w in zip(music_ids, music_weights)
        }
        has_lastfm = bool(neighborhood["source_coverage"]["lastfm"])
        confidence = (
            float(np.mean(np.sort(last_weights / last_max)[-5:]))
            if len(last_weights)
            else 0.0
        )

        # --- Source-profile ambiguity guard ---
        top_last_aids = [
            int(a)
            for a, _ in sorted(last.items(), key=lambda x: -x[1])[:GUARD_TOP_K]
        ]
        source_conf, source_reason = self._source_profile_confidence(
            query_row, top_last_aids
        )

        # --- Query ID resolution confidence ---
        query_resolve_conf = 1.0
        query_resolve_margin = 1.0
        query_resolve_reason = "primary_id_available"
        query_centroid_vec: Optional[np.ndarray] = None

        if self.identity is not None:
            q_sid = self.identity.primary_artist_id(query_row)
            if q_sid is not None:
                # Primary ID known — try to get centroid
                c = self.identity.artist_centroid(q_sid)
                query_centroid_vec = c if c is not None else self._query_centroid(query_row)
            else:
                # Attempt to resolve query name to a Deezer ID
                query_centroid_vec = self._query_centroid(query_row)
                q_name = str(self.artists[query_row])
                name_dids = self.identity.name_to_deezer_ids(q_name)
                if name_dids and query_centroid_vec is not None:
                    result = self.identity.disambiguate(
                        q_name,
                        query_centroid_vec,
                        min_confidence=GUARD_CENTROID_MIN,
                        min_margin=GUARD_CENTROID_MARGIN,
                    )
                    if result is None:
                        query_resolve_conf = 0.0
                        query_resolve_margin = 0.0
                        query_resolve_reason = "query_disambiguation_failed"
                    else:
                        chosen_id, query_resolve_conf, query_resolve_margin = result
                        query_resolve_reason = (
                            "single_cluster_resolved"
                            if len(name_dids) == 1
                            else "ambiguous_cluster_disambiguated"
                        )
                        chosen_centroid = self.identity.artist_centroid(chosen_id)
                        if chosen_centroid is not None:
                            query_centroid_vec = chosen_centroid
                else:
                    query_resolve_conf = 0.0
                    query_resolve_margin = 0.0
                    query_resolve_reason = "unresolved_name"

        guard_abstained = (
            source_conf < GUARD_SOURCE_PROFILE_MIN
            or query_resolve_conf < GUARD_CENTROID_MIN
        )

        if guard_abstained:
            abstain_reason = (
                f"source_profile_below_{GUARD_SOURCE_PROFILE_MIN}"
                if source_conf < GUARD_SOURCE_PROFILE_MIN
                else f"query_resolution_below_{GUARD_CENTROID_MIN}"
            )
            diag: Dict[str, Any] = {
                "source_profile_confidence": source_conf,
                "source_profile_reason": source_reason,
                "query_resolution_confidence": query_resolve_conf,
                "query_resolution_margin": query_resolve_margin,
                "query_resolution_reason": query_resolve_reason,
                "guard_abstained": True,
                "abstain_reason": abstain_reason,
                "name_group_diagnostics": [],
                "filtered_row_count": 0,
                "omitted_name_groups": 0,
            }
            return (
                np.empty(0, dtype=np.int64),
                last,
                music,
                confidence,
                has_lastfm,
                diag,
            )

        # --- Identity-guarded row expansion ---
        all_artist_graph_ids = sorted(set(last) | set(music))
        filtered_rows: List[int] = []
        name_group_diags: List[Dict[str, Any]] = []
        omitted_count = 0

        for g_aid in all_artist_graph_ids:
            # Collect raw track rows for this graph artist
            g_start = int(self.graph.track_indptr[g_aid])
            g_stop = int(self.graph.track_indptr[g_aid + 1])
            track_rows_for_artist = [
                int(self.graph.track_rows[i]) for i in range(g_start, g_stop)
            ]

            if self.identity is None:
                # No identity asset: use all rows (v13-compatible)
                filtered_rows.extend(track_rows_for_artist)
                continue

            artist_name_str = str(self.graph.artist_names[g_aid])

            def _name_to_dids(name: str) -> List[int]:
                return self.identity.name_to_deezer_ids(name)  # type: ignore[union-attr]

            def _disambig(name: str, vec: np.ndarray) -> Optional[Tuple[int, float, float]]:
                return self.identity.disambiguate(  # type: ignore[union-attr]
                    name,
                    vec,
                    min_confidence=GUARD_CANDIDATE_MIN,
                    min_margin=GUARD_CENTROID_MARGIN,
                )

            def _unresolved_matches(row: int, artist_id: int) -> bool:
                centroid = self.identity.artist_centroid(artist_id)  # type: ignore[union-attr]
                row_vector = _normalise(
                    np.asarray(self.compact[row], dtype=np.float32)
                )
                return bool(
                    centroid is not None
                    and row_vector is not None
                    and float(centroid @ row_vector) >= GUARD_CANDIDATE_MIN
                )

            resolved_row_ids = {
                row: self._resolved_id_for_row(row)
                for row in track_rows_for_artist
            }
            new_rows, grp_diag = apply_identity_guard_to_name_group(
                artist_name_str,
                track_rows_for_artist,
                resolved_row_ids,
                _name_to_dids,
                _disambig,
                query_centroid_vec,
                _unresolved_matches,
            )
            filtered_rows.extend(new_rows)
            name_group_diags.append(grp_diag)
            if grp_diag["action"].startswith("omitted"):
                omitted_count += 1

        diag = {
            "source_profile_confidence": source_conf,
            "source_profile_reason": source_reason,
            "query_resolution_confidence": query_resolve_conf,
            "query_resolution_margin": query_resolve_margin,
            "query_resolution_reason": query_resolve_reason,
            "guard_abstained": False,
            "abstain_reason": None,
            "name_group_diagnostics": name_group_diags,
            "filtered_row_count": len(set(filtered_rows)),
            "omitted_name_groups": omitted_count,
        }

        return (
            np.asarray(sorted(set(filtered_rows)), dtype=np.int64),
            last,
            music,
            confidence,
            has_lastfm,
            diag,
        )

    # ------------------------------------------------------------------
    # rank_all override
    # ------------------------------------------------------------------

    def rank_all(self, query_row: int) -> Dict[str, Dict[str, Any]]:
        """v14 rank_all.

        Conservative behaviour, ordering, and weights are identical to v13
        *except*:

        * Same-artist filtering and MMR deduplication use stable Deezer ID
          (identity collision correction).
        * Graph candidates are filtered through the identity guard.
        * When the identity guard abstains, the conservative challenger returns
          exact production rows with an explicit ``fallback_reason`` starting
          with ``"identity_guard_abstained:"``; pure_clap and graph_clap_union
          proceed without graph signal.

        ``identity_guard`` diagnostics are appended to every variant result.
        """
        production = self.production_rows(query_row, 5)
        clap = self.clap_scores(query_row)

        # Base identity guard diagnostics
        base_guard_diag: Dict[str, Any] = {
            "identity_asset_available": self.identity is not None,
        }

        if clap is None:
            fallback_diag = {**base_guard_diag, "guard_abstained": False}
            return {
                name: {
                    "rows": list(production),
                    "query_available": False,
                    "gate_fired": False,
                    "fallback_reason": "seed_preview_unavailable",
                    "candidate_count": 0,
                    "candidate_rows": [],
                    "identity_guard": fallback_diag,
                }
                for name in VARIANT_ORDER_V14
            }

        clap01 = (clap + 1.0) / 2.0

        # --- Pure CLAP (identity dedup only, no graph guard) ---
        pure_candidate_rows = _top_indices(clap, 500)
        pure_eligible = self._eligible_v14(query_row, pure_candidate_rows, clap01)
        pure = self._mmr_v14(query_row, pure_eligible)

        # --- Identity-guarded graph candidates ---
        (
            graph_rows,
            last,
            music,
            confidence,
            has_lastfm,
            guard_diag,
        ) = self._graph_candidates_v14(query_row)

        full_guard_diag = {**base_guard_diag, **guard_diag}
        guard_abstained = bool(guard_diag["guard_abstained"])

        # --- Graph + CLAP union (uses guarded rows; pure CLAP fills gap if empty) ---
        union_rows = np.asarray(
            sorted(
                set(map(int, graph_rows))
                | set(map(int, _top_indices(clap, 200)))
            ),
            dtype=np.int64,
        )
        graph_relevance = np.full(len(self.track_ids), -np.inf, dtype=np.float32)
        graph_only_relevance = np.full(
            len(self.track_ids), -np.inf, dtype=np.float32
        )
        for row_idx in union_rows:
            r = int(row_idx)
            artist_id = int(self.graph.track_artist_ids[r])
            graph_relevance[r] = (
                0.70 * float(clap01[r])
                + 0.25 * last.get(artist_id, 0.0)
                + 0.05 * music.get(artist_id, 0.0)
            )
        for row_idx in graph_rows:
            r = int(row_idx)
            artist_id = int(self.graph.track_artist_ids[r])
            graph_only_relevance[r] = (
                0.80 * last.get(artist_id, 0.0)
                + 0.20 * float(clap01[r])
            )

        graph_eligible = self._eligible_v14(query_row, union_rows, graph_relevance)
        graph_ranked = self._mmr_v14(query_row, graph_eligible)

        # --- Conservative challenger ---
        if guard_abstained:
            conservative = list(production)
            conservative_fired = False
            conservative_fallback = (
                f"identity_guard_abstained:{guard_diag.get('abstain_reason', 'unknown')}"
            )
            consistent_count = 0
            consistent_rows: List[int] = []
        else:
            conservative_eligible = self._eligible_v14(
                query_row, graph_rows, graph_only_relevance
            )
            consistent = [
                item
                for item in conservative_eligible
                if float(clap01[int(item["row"])]) >= 0.65
            ]
            conservative_fired = (
                has_lastfm and confidence >= 0.55 and len(consistent) >= 5
            )
            conservative = (
                self._mmr_v14(query_row, consistent)
                if conservative_fired
                else list(production)
            )
            consistent_count = len(consistent)
            consistent_rows = [int(item["row"]) for item in consistent[:200]]
            conservative_fallback = (
                None
                if conservative_fired
                else (
                    "missing_lastfm"
                    if not has_lastfm
                    else (
                        "lastfm_confidence"
                        if confidence < 0.55
                        else "fewer_than_five_consistent_candidates"
                    )
                )
            )

        return {
            "pure_clap": {
                "rows": pure,
                "query_available": True,
                "gate_fired": True,
                "fallback_reason": None,
                "candidate_count": len(pure_eligible),
                "candidate_rows": [
                    int(item["row"]) for item in pure_eligible[:200]
                ],
                "identity_guard": full_guard_diag,
            },
            "graph_clap_union": {
                "rows": graph_ranked,
                "query_available": True,
                "gate_fired": True,
                "fallback_reason": (
                    f"identity_guard_abstained:{guard_diag.get('abstain_reason')}"
                    if guard_abstained
                    else None
                ),
                "candidate_count": len(graph_eligible),
                "candidate_rows": [
                    int(item["row"]) for item in graph_eligible[:200]
                ],
                "identity_guard": full_guard_diag,
            },
            "conservative_clap_fallback": {
                "rows": conservative,
                "query_available": True,
                "gate_fired": conservative_fired,
                "fallback_reason": conservative_fallback,
                "candidate_count": consistent_count,
                "candidate_rows": consistent_rows,
                "lastfm_confidence": confidence,
                "identity_guard": full_guard_diag,
            },
        }


# ---------------------------------------------------------------------------
# v14 diagnostics
# ---------------------------------------------------------------------------

# Canonical 13 scene labels used in the v13 60-seed suite
_V13_SCENES = (
    "rap",
    "rnb",
    "indie",
    "shoegaze",
    "hyperpop",
    "electronic",
    "metal",
    "jazz",
    "city_pop",
    "latin_afrobeats",
    "difficult_blend",
    "pop",
    "rock",
)
_DIRECT_DIAG_TARGET = 20  # deterministic 20-seed scene-spanning direct list


def _select_direct_diagnostic_seeds(
    seeds: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select exactly ``_DIRECT_DIAG_TARGET`` seeds covering all v13 scenes.

    Algorithm (deterministic):
    1. Normalise each seed's scene label: strip trailing digits/whitespace,
       lower.
    2. Round-robin over scenes in canonical order: each round takes the next
       unseen seed for that scene until 20 are collected.
    """
    # Normalise scene labels to canonical keys
    def _norm_scene(raw: str) -> str:
        s = str(raw).strip().lower()
        # Map common suffixes/variants
        for key in _V13_SCENES:
            if s == key or s.startswith(key):
                return key
        # Partial matches
        for key in _V13_SCENES:
            if key in s:
                return key
        return s

    # Index seeds by normalised scene
    scene_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        sc = _norm_scene(str(seed.get("scene", "")))
        scene_queues[sc].append(seed)

    selected: List[Dict[str, Any]] = []
    scene_order = list(_V13_SCENES) + sorted(
        s for s in scene_queues if s not in _V13_SCENES
    )
    pointers: Dict[str, int] = {sc: 0 for sc in scene_order}

    while len(selected) < _DIRECT_DIAG_TARGET:
        added_this_round = False
        for sc in scene_order:
            if len(selected) >= _DIRECT_DIAG_TARGET:
                break
            q = scene_queues.get(sc, [])
            ptr = pointers.get(sc, 0)
            if ptr < len(q):
                selected.append(q[ptr])
                pointers[sc] = ptr + 1
                added_this_round = True
        if not added_this_round:
            break  # exhausted all seeds

    return selected


def _semantic_diff(
    v14_records: List[Dict[str, Any]],
    v13_records: List[Dict[str, Any]],
    ranker: IdentityGuardedRanker,
) -> Dict[str, Any]:
    """Record exact list/position changes between frozen v13 and v14."""
    old_by_seed = {
        str(record["seed_id"]): list(map(int, record.get("rows", [])))
        for record in v13_records
    }
    new_by_seed = {
        str(record["seed_id"]): list(map(int, record.get("rows", [])))
        for record in v14_records
    }
    diffs: List[Dict[str, Any]] = []
    overlap_sum = 0.0
    exact_matches = 0
    changed_position_total = 0
    for seed_id in sorted(new_by_seed):
        old_rows = old_by_seed.get(seed_id, [])
        new_rows = new_by_seed[seed_id]
        old_set, new_set = set(old_rows), set(new_rows)
        overlap = len(old_set & new_set) / max(len(old_set | new_set), 1)
        exact = old_rows == new_rows
        exact_matches += int(exact)
        overlap_sum += overlap
        changed_positions = [
            position
            for position in range(1, max(len(old_rows), len(new_rows)) + 1)
            if (old_rows[position - 1] if position <= len(old_rows) else None)
            != (new_rows[position - 1] if position <= len(new_rows) else None)
        ]
        changed_position_total += len(changed_positions)

        def _describe(rows: Sequence[int]) -> List[Dict[str, Any]]:
            return [
                {
                    "position": position,
                    "row": row,
                    "deezer_track_id": int(ranker.track_ids[row]),
                    "title": str(ranker.titles[row]),
                    "artist": str(ranker.artists[row]),
                }
                for position, row in enumerate(rows, start=1)
            ]

        diffs.append(
            {
                "seed_id": seed_id,
                "exact_match": exact,
                "overlap": overlap,
                "changed_positions": changed_positions,
                "added_rows": sorted(new_set - old_set),
                "dropped_rows": sorted(old_set - new_set),
                "old_list": _describe(old_rows),
                "new_list": _describe(new_rows),
            }
        )
    count = len(diffs)
    return {
        "seed_count": count,
        "exact_match_count": exact_matches,
        "changed_seed_count": count - exact_matches,
        "changed_position_count": changed_position_total,
        "mean_overlap": overlap_sum / max(count, 1),
        "per_seed_diff": diffs,
    }


def run_variant_diagnostics_v14(
    index_path: Path,
    compact_path: Path,
    status_path: Path,
    graph_path: Path,
    style_path: Path,
    seed_lists_path: Path,
    output_path: Path,
    *,
    identity_asset: Optional[IdentityAsset] = None,
    deezer_fetcher: Callable[[str], List[str]] = _deezer_related,
    v13_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """v14 non-deciding proxy safety diagnostics.

    Matches the v13 60-seed methodology exactly (``commercial_human_ratings_used=0``,
    ``proxy_evidence_is_deciding=False``, ``production=False``), with four
    additional fields:

    * ``identity_guard_summary`` — guard-fired counts per variant
    * ``semantic_diff_vs_v13`` — diff against the v13 selected challenger rows
      (requires ``v13_diagnostics`` to be supplied)
    * ``candidate_coverage`` — recall@50/200 per variant (v13 compatible)
    * ``direct_diagnostic_20seed`` — actual lists for 20 deterministic seeds
      spanning all 13 v13 scene labels (diagnostic only, not human gold)

    Parameters
    ----------
    identity_asset:
        Optional pre-loaded :class:`IdentityAsset`.  When ``None``, the ranker
        falls back to v13 name-based behaviour.
    v13_diagnostics:
        Parsed content of the v13 variant-diagnostics JSON.  Used for semantic
        diff and as the v13 production baseline source.
    """
    from .catalog_style import CatalogStyleIndex
    from .clap_catalog_v13 import PREREGISTRATION_SHA256, TRACK_IDS_SHA256

    ranker = IdentityGuardedRanker(
        index_path,
        compact_path,
        status_path,
        graph_path,
        style_path,
        identity_asset=identity_asset,
    )
    style = ranker.style
    source = json.loads(seed_lists_path.read_text(encoding="utf-8"))
    if source.get("seed_count") != 60 or source.get("scene_count") != 13:
        raise ClapCatalogError(
            "v14 diagnostics require the frozen 60-seed/13-scene suite"
        )

    # Deezer affinity (same fresh-fetch approach as v13)
    deezer_truth: Dict[str, Set[str]] = {}
    for seed in source["seeds"]:
        try:
            deezer_truth[str(seed["seed_id"])] = set(
                deezer_fetcher(str(seed["query"]["artist"]))
            )
        except Exception:
            deezer_truth[str(seed["seed_id"])] = set()

    # --- Main 60-seed loop ---
    baseline_records: List[Dict[str, Any]] = []
    variants: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in VARIANT_ORDER_V14
    }
    scene_counts: Counter = Counter()
    latencies: List[float] = []
    guard_abstained_counts: Dict[str, int] = {n: 0 for n in VARIANT_ORDER_V14}

    for seed in source["seeds"]:
        scene_counts[str(seed["scene"])] += 1
        query_row = ranker.query_row(int(seed["query"]["deezer_track_id"]))
        baseline = ranker.production_rows(query_row, 5)
        baseline_records.append(
            {
                "seed_id": seed["seed_id"],
                "scene": seed["scene"],
                "query_row": query_row,
                "rows": baseline,
            }
        )
        t0 = time.perf_counter()
        ranked = ranker.rank_all(query_row)
        latencies.append(time.perf_counter() - t0)
        for name in VARIANT_ORDER_V14:
            record = {
                "seed_id": seed["seed_id"],
                "scene": seed["scene"],
                "query_row": query_row,
                **ranked[name],
            }
            variants[name].append(record)
            if (ranked[name].get("fallback_reason") or "").startswith(
                "identity_guard_abstained"
            ):
                guard_abstained_counts[name] += 1

    baseline_metrics = _list_metrics(
        baseline_records, ranker, style, deezer_truth
    )
    variant_metrics: Dict[str, Any] = {}
    selected: Optional[str] = None
    for name in VARIANT_ORDER_V14:
        metrics = _list_metrics(variants[name], ranker, style, deezer_truth)
        metrics["style_delta_vs_production"] = (
            metrics["mean_style_overlap"] - baseline_metrics["mean_style_overlap"]
        )
        metrics["deezer_affinity_delta_vs_production"] = (
            metrics["deezer_related_artist_hit_rate"]
            - baseline_metrics["deezer_related_artist_hit_rate"]
        )
        metrics["gate_fired_count"] = sum(
            bool(item.get("gate_fired")) for item in variants[name]
        )
        metrics["exact_production_fallback_count"] = sum(
            item["rows"] == baseline_records[pos]["rows"]
            for pos, item in enumerate(variants[name])
        )
        metrics["identity_guard_abstained_count"] = guard_abstained_counts[name]
        metrics["passes_proxy_safety"] = _proxy_passes(metrics, baseline_metrics)
        variant_metrics[name] = metrics
        if selected is None and metrics["passes_proxy_safety"]:
            selected = name

    if selected is None:
        raise ClapCatalogError(
            "all three v14 CLAP variants failed proxy collapse gates"
        )

    # --- Identity guard summary ---
    identity_guard_summary: Dict[str, Any] = {
        name: {
            "guard_abstained_count": guard_abstained_counts[name],
            "guard_abstained_fraction": guard_abstained_counts[name] / 60,
        }
        for name in VARIANT_ORDER_V14
    }

    # --- Semantic diff vs v13 selected rows ---
    semantic_diff: Optional[Dict[str, Any]] = None
    if v13_diagnostics is not None:
        v13_selected = v13_diagnostics.get("selected_challenger")
        if v13_selected and v13_selected in (v13_diagnostics.get("variants") or {}):
            v13_rec = v13_diagnostics["variants"][v13_selected].get("records", [])
            semantic_diff = _semantic_diff(
                variants.get(selected, []), v13_rec, ranker
            )

    # --- Candidate recall (v13-compatible; diagnostic only) ---
    try:
        from .catalog_list_gold_v9 import load_seed_specs
        from .real_benchmark import PairResolver

        seeds_spec = load_seed_specs(
            "benchmarks/soundalike_pairs.v6.json",
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-gated-direct-seeds-v8.json",
        )
        resolver = PairResolver(ranker.titles, ranker.artists)
        target_rows: Dict[str, List[int]] = {}
        for sp in seeds_spec:
            pair = sp.get("category_a_pair")
            if pair:
                resolved = resolver.target_rows(pair["target"])
                target_rows[str(sp["id"])] = list(map(int, resolved))
    except Exception:
        target_rows = {}

    candidate_coverage: Dict[str, Any] = {}
    for name in VARIANT_ORDER_V14:
        found50 = found200 = total = 0
        for record in variants[name]:
            targets = set(target_rows.get(str(record["seed_id"]), ()))
            if not targets:
                continue
            total += 1
            cands = list(map(int, record.get("candidate_rows", ())))
            found50 += int(bool(targets & set(cands[:50])))
            found200 += int(bool(targets & set(cands[:200])))
        candidate_coverage[name] = {
            "known_category_a_targets": total,
            "recall_at_50": found50 / total if total else None,
            "recall_at_200": found200 / total if total else None,
            "selection_use": False,
        }

    # --- 20-seed direct actual-list diagnostic ---
    direct_seeds = _select_direct_diagnostic_seeds(source["seeds"])
    direct_records: List[Dict[str, Any]] = []
    for seed in direct_seeds:
        qr = ranker.query_row(int(seed["query"]["deezer_track_id"]))
        ranked_direct = ranker.rank_all(qr)
        direct_records.append(
            {
                "seed_id": seed["seed_id"],
                "scene": seed["scene"],
                "query_row": qr,
                "query_title": str(seed["query"].get("title", "")),
                "query_artist": str(seed["query"].get("artist", "")),
                "variants": {
                    name: {
                        "rows": ranked_direct[name]["rows"],
                        "fallback_reason": ranked_direct[name].get("fallback_reason"),
                        "gate_fired": ranked_direct[name].get("gate_fired"),
                        "titles": [
                            str(ranker.titles[r]) for r in ranked_direct[name]["rows"]
                        ],
                        "artists": [
                            str(ranker.artists[r]) for r in ranked_direct[name]["rows"]
                        ],
                    }
                    for name in VARIANT_ORDER_V14
                },
            }
        )

    # Assemble report
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "clap_catalog_v14_proxy_safety_and_variant_selection",
        "created_at": _now(),
        "preregistration_content_sha256": PREREGISTRATION_SHA256,
        "commercial_human_ratings_used": 0,
        "proxy_evidence_is_deciding": False,
        "production": False,
        "compact_asset_sha256": _sha256_path(compact_path),
        "identity_asset_available": identity_asset is not None,
        "catalog": {
            "rows": len(ranker.track_ids),
            "track_ids_tobytes_sha256": TRACK_IDS_SHA256,
        },
        "scene_distribution": dict(sorted(scene_counts.items())),
        "deezer_affinity": {
            "seeds_requested": 60,
            "seeds_with_related_artists": sum(
                bool(v) for v in deezer_truth.values()
            ),
            "fresh_supporting_only": True,
        },
        "production_baseline": {
            "metrics": baseline_metrics,
            "records": baseline_records,
        },
        "variants": {
            name: {"metrics": variant_metrics[name], "records": variants[name]}
            for name in VARIANT_ORDER_V14
        },
        "candidate_coverage": candidate_coverage,
        "selection_order": list(VARIANT_ORDER_V14),
        "selected_challenger": selected,
        "selection_rule": (
            "first pre-registered variant in conservative, graph, pure order "
            "passing every proxy safety gate (identical to v13)"
        ),
        "identity_guard_summary": identity_guard_summary,
        "semantic_diff_vs_v13": semantic_diff,
        "direct_diagnostic_20seed": {
            "seed_count": len(direct_records),
            "scene_count": len({r["scene"] for r in direct_records}),
            "diagnostic_only": True,
            "not_human_gold": True,
            "not_selection_input": True,
            "records": direct_records,
        },
        "latency": {
            "queries": len(latencies),
            "mean_ms": float(np.mean(latencies) * 1000),
            "p50_ms": float(np.quantile(latencies, 0.50) * 1000),
            "p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        },
        "safety": {
            "human_ab_required": True,
            "production_changed": False,
            "deployed": False,
            "commercial_final_opened": False,
            "ac3_claimed": False,
        },
    }
    report["content_sha256"] = content_hash(report)
    _write_json(output_path, report)
    return report


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    diag = sub.add_parser("diagnose", help="Run v14 proxy diagnostics")
    diag.add_argument(
        "--index", type=Path, default=root / "ml_data/deepvibe_index_v5.npz"
    )
    diag.add_argument("--compact", type=Path, required=True)
    diag.add_argument(
        "--status",
        type=Path,
        default=root / "ml_data/clap_v13/status.sqlite3",
    )
    diag.add_argument(
        "--graph",
        type=Path,
        default=root / "ml_data/iteration8/catalog-artist-graph-dual-v8.npz",
    )
    diag.add_argument(
        "--style",
        type=Path,
        default=root / "ml_data/iteration7/catalog-style-v8.npz",
    )
    diag.add_argument(
        "--seeds",
        type=Path,
        default=(
            root
            / ".goals/human-quality-recommendations/"
            "protocol-v11-audio-access-erratum/served-lists-v11.json"
        ),
    )
    diag.add_argument("--identity-npz", type=Path, default=None)
    diag.add_argument("--v13-diagnostics", type=Path, default=None)
    diag.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    identity: Optional[IdentityAsset] = None
    if args.identity_npz is not None and args.identity_npz.is_file():
        identity = IdentityAsset.load(args.identity_npz)
    v13_diag: Optional[Dict[str, Any]] = None
    if args.v13_diagnostics is not None and args.v13_diagnostics.is_file():
        v13_diag = json.loads(args.v13_diagnostics.read_text(encoding="utf-8"))
    report = run_variant_diagnostics_v14(
        args.index,
        args.compact,
        args.status,
        args.graph,
        args.style,
        args.seeds,
        args.output,
        identity_asset=identity,
        v13_diagnostics=v13_diag,
    )
    print(f"selected challenger: {report['selected_challenger']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
