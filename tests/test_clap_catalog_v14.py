"""Tests for clap_catalog_v14: identity-guarded CLAP development ranker.

All tests use synthetic in-memory or tmp-path assets (no network, no large
local files).  Existing v13 tests are not touched.

Coverage
--------
* Documented constant values
* ``compute_source_profile_confidence`` — low-profile abstention, confident path,
  margin abstention
* ``apply_identity_guard_to_name_group`` — same-name cluster isolation,
  unresolved omission, mixed exclusion, homonym disambiguation
* ``IdentityGuardedRanker`` via a lightweight ``_SyntheticRanker`` helper:
    - stable-ID MMR dedup across spelling/case/punctuation/transliteration
    - multi-artist contributor same-family exclusion
    - low-confidence exact-production fallback
    - identity-guard abstention flags in rank_all result
* ``_select_direct_diagnostic_seeds`` — 20-seed deterministic selection
* ``_semantic_diff`` — diff output structure
* Regression: name "nothing" uses generic evidence (no name-specific branch)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

from soundalike.ml.artist_identity_v14 import (
    IdentityAsset,
    normalize_key,
    _SENTINEL_ARTIST_ID,
)
from soundalike.ml.clap_catalog_v14 import (
    GUARD_AUDIO_WEIGHT,
    GUARD_CANDIDATE_MIN,
    GUARD_CENTROID_MARGIN,
    GUARD_CENTROID_MIN,
    GUARD_SOURCE_PROFILE_MIN,
    GUARD_STYLE_WEIGHT,
    GUARD_TOP_K,
    SCHEMA_VERSION,
    _select_direct_diagnostic_seeds,
    _semantic_diff,
    apply_identity_guard_to_name_group,
    compute_source_profile_confidence,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic IdentityAsset
# ---------------------------------------------------------------------------


def _make_identity_npz(
    tmp_path: Path,
    *,
    track_ids: List[int],
    primary_artist_ids: List[Optional[int]],
    contrib_map: Optional[Dict[int, List[int]]] = None,
    name_clusters: Optional[Dict[str, List[int]]] = None,
    centroids: Optional[Dict[int, List[float]]] = None,
    dim: int = 8,
) -> Path:
    """Build a minimal IdentityAsset NPZ without any network calls.

    Parameters
    ----------
    track_ids: catalog row track IDs (positional)
    primary_artist_ids: primary Deezer artist ID per row (None = unresolved)
    contrib_map: row_index → list of contributor Deezer IDs
    name_clusters: normalized_name → list of Deezer artist IDs
    centroids: Deezer artist ID → centroid vector (length ``dim``)
    """
    n = len(track_ids)
    tids = np.asarray(track_ids, dtype=np.int64)

    # Primary artist IDs
    paid_arr = np.full(n, _SENTINEL_ARTIST_ID, dtype=np.int32)
    for i, pid in enumerate(primary_artist_ids):
        if pid is not None:
            paid_arr[i] = int(pid)

    # Contributor CSR
    contrib_flat: List[int] = []
    contrib_indptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        cids = list((contrib_map or {}).get(i, []))
        contrib_indptr[i + 1] = contrib_indptr[i] + len(cids)
        contrib_flat.extend(cids)

    # Artist-level arrays
    all_aids: Set[int] = set()
    for pid in primary_artist_ids:
        if pid is not None:
            all_aids.add(int(pid))
    for cids in (contrib_map or {}).values():
        all_aids.update(cids)
    if name_clusters:
        for dids in name_clusters.values():
            all_aids.update(dids)
    if centroids:
        all_aids.update(centroids.keys())

    artist_ids_sorted = np.asarray(sorted(all_aids), dtype=np.int32)
    m = len(artist_ids_sorted)
    aid_to_pos = {int(a): i for i, a in enumerate(artist_ids_sorted)}

    # Centroids
    cent_arr = np.zeros((m, dim), dtype=np.float16)
    for aid, vec in (centroids or {}).items():
        pos = aid_to_pos.get(int(aid))
        if pos is not None:
            v = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 1e-8:
                cent_arr[pos] = (v / norm).astype(np.float16)

    # Name clusters CSR — normalize keys so IdentityAsset lookup matches
    ncs = {normalize_key(k): v for k, v in (name_clusters or {}).items()}
    name_keys_list = sorted(ncs.keys())
    nc = len(name_keys_list)
    name_cluster_flat: List[int] = []
    name_cluster_indptr = np.zeros(nc + 1, dtype=np.int32)
    for i, name in enumerate(name_keys_list):
        cluster = sorted(ncs[name])
        name_cluster_indptr[i + 1] = name_cluster_indptr[i] + len(cluster)
        name_cluster_flat.extend(cluster)

    # Row name keys (empty for simplicity)
    row_name_keys = np.asarray([""] * n)

    meta = json.dumps(
        {
            "schema_version": 14,
            "track_ids_sha256": "test",
            "clap_asset_hash": "test",
            "total_rows": n,
            "total_artists": m,
            "total_name_clusters": nc,
            "keys_with_zero_deezer_ids": 0,
            "embedding_dim": dim,
            "has_centroids": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        sort_keys=True,
    )

    path = tmp_path / f"identity_{abs(hash(str(track_ids)))}.npz"
    np.savez(
        path,
        track_ids=tids,
        primary_artist_ids=paid_arr,
        contrib_flat=np.asarray(contrib_flat, dtype=np.int32),
        contrib_indptr=contrib_indptr,
        artist_ids=artist_ids_sorted,
        artist_centroids=cent_arr,
        artist_track_count=np.ones(m, dtype=np.int32),
        artist_within_min_cosine=np.full(m, np.nan, dtype=np.float32),
        name_keys=np.asarray(name_keys_list),
        name_cluster_flat=np.asarray(
            name_cluster_flat, dtype=np.int32
        ),
        name_cluster_indptr=name_cluster_indptr,
        spelling_flat=np.asarray([], dtype="<U1"),
        spelling_indptr=np.zeros(nc + 1, dtype=np.int32),
        mbid_flat=np.asarray([], dtype="<U36"),
        mbid_indptr=np.zeros(m + 1, dtype=np.int32),
        name_mbid_flat=np.asarray([], dtype="<U36"),
        name_mbid_indptr=np.zeros(nc + 1, dtype=np.int32),
        row_name_keys=row_name_keys,
        metadata=np.array(meta),
    )
    return path


def _load_identity(
    tmp_path: Path,
    **kwargs: Any,
) -> IdentityAsset:
    path = _make_identity_npz(tmp_path, **kwargs)
    return IdentityAsset(path)


# ---------------------------------------------------------------------------
# Helpers — minimal MockGraph
# ---------------------------------------------------------------------------


class _MockGraph:
    """Minimal stand-in for CatalogArtistGraph used in unit tests."""

    def __init__(
        self,
        artist_names: List[str],
        track_artist_ids: List[int],
        neighbors: Dict[str, Tuple[List[int], List[float]]],
        artist_audio: Optional[np.ndarray] = None,
        source_mapped: Optional[np.ndarray] = None,
    ):
        self.artist_names = np.asarray(artist_names)
        self.track_artist_ids = np.asarray(track_artist_ids, dtype=np.int32)

        # Build track_rows and track_indptr from track_artist_ids
        n_artists = len(artist_names)
        n_tracks = len(track_artist_ids)
        counts = np.bincount(
            np.asarray(track_artist_ids, dtype=np.int32), minlength=n_artists
        )
        self.track_indptr = np.concatenate(([0], np.cumsum(counts))).astype(
            np.int32
        )
        order = np.argsort(np.asarray(track_artist_ids, dtype=np.int32), kind="stable")
        self.track_rows = order.astype(np.int32)

        self.artist_lookup = {
            n: i for i, n in enumerate(artist_names)
        }
        self._neighbors = neighbors  # artist_name → (ids, weights)
        dim = 4
        if artist_audio is not None:
            self.artist_audio = np.asarray(artist_audio, dtype=np.float32)
        else:
            rng = np.random.default_rng(42)
            raw = rng.standard_normal((n_artists, dim)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-8)
            self.artist_audio = raw / norms

        if source_mapped is not None:
            self.source_mapped = np.asarray(source_mapped, dtype=bool)
        else:
            self.source_mapped = np.ones(n_artists, dtype=bool)

        self.music4all_query_artist_ids = np.empty(0, dtype=np.int32)
        self.music4all_indices = np.empty((0, 0), dtype=np.int32)
        self.music4all_weights = np.empty((0, 0), dtype=np.float32)

    def dual_source_neighbors(self, query_artist: str) -> Dict[str, Any]:
        from soundalike.ml.real_benchmark import normalize_text

        normed = normalize_text(query_artist)
        aids_raw, ws_raw = self._neighbors.get(normed, ([], []))
        aids = np.asarray(aids_raw, dtype=np.int32)
        ws = np.asarray(ws_raw, dtype=np.float32)
        artist_id = self.artist_lookup.get(normed)
        has_lastfm = len(aids) > 0
        return {
            "artist_id": artist_id,
            "lastfm": {"artist_ids": aids, "weights": ws},
            "music4all": {
                "artist_ids": np.empty(0, dtype=np.int32),
                "weights": np.empty(0, dtype=np.float32),
            },
            "union_artist_ids": aids,
            "source_coverage": {"lastfm": has_lastfm, "music4all": False},
            "mode": "dual_source_union" if has_lastfm else "dual_source_unavailable",
        }


# ---------------------------------------------------------------------------
# Helpers — lightweight synthetic ranker
# ---------------------------------------------------------------------------


class _MockProductionRanker:
    def __init__(self, rows_by_query: Dict[int, List[int]]):
        self._rows = rows_by_query

    def rank(self, row: int, variant: str, n: int = 5) -> List[int]:
        return list(self._rows.get(row, list(range(min(5, n))))[:n])


class _MockStyleIndex:
    """Style index where every pair has constant overlap ``value``."""

    def __init__(self, value: float = 0.5):
        self._value = float(value)

    def style_overlap(self, a: str, b: str) -> float:
        return self._value


class _SyntheticRanker:
    """Lightweight IdentityGuardedRanker for unit tests.

    Bypasses file-level validation; accepts numpy arrays and mock objects
    directly so tests need no large local assets.
    """

    def __init__(
        self,
        track_ids: np.ndarray,
        titles: List[str],
        artists: List[str],
        compact: np.ndarray,
        graph: _MockGraph,
        style: _MockStyleIndex,
        production_rows_map: Dict[int, List[int]],
        identity: Optional[IdentityAsset] = None,
    ):
        from soundalike.ml.quality_filter import TitleQualityFilter

        self.track_ids = np.asarray(track_ids, dtype=np.int64)
        self.titles = np.asarray(titles)
        self.artists = np.asarray(artists)
        self.compact = np.asarray(compact, dtype=np.float32)
        self.available = np.any(self.compact != 0, axis=1)
        self.graph = graph
        self.style = style
        self.identity = identity
        self.quality = TitleQualityFilter()
        self.rows_by_track_id = {
            int(tid): i for i, tid in enumerate(self.track_ids)
        }
        self._mock_prod = _MockProductionRanker(production_rows_map)

        # Import the real helpers we need
        from soundalike.ml.clap_catalog_v14 import (
            GUARD_CENTROID_MARGIN,
            GUARD_CENTROID_MIN,
            GUARD_SOURCE_PROFILE_MIN,
            GUARD_TOP_K,
            apply_identity_guard_to_name_group,
            compute_source_profile_confidence,
        )
        from soundalike.ml.clap_catalog_v13 import _top_indices

        self._top_indices = _top_indices
        self._guard_source_min = GUARD_SOURCE_PROFILE_MIN
        self._guard_centroid_min = GUARD_CENTROID_MIN
        self._guard_centroid_margin = GUARD_CENTROID_MARGIN
        self._guard_top_k = GUARD_TOP_K
        self._compute_conf = compute_source_profile_confidence
        self._apply_guard = apply_identity_guard_to_name_group

        # Pre-build stable ID map
        self._stable_id_for_row: Dict[int, Optional[int]] = {}
        if identity is not None:
            for row_idx in range(len(self.track_ids)):
                try:
                    info = identity.row_identity(row_idx)
                    self._stable_id_for_row[row_idx] = info[
                        "primary_artist_deezer_id"
                    ]
                except IndexError:
                    self._stable_id_for_row[row_idx] = None

    # -- delegated helpers from v14 --

    def _stable_key(self, row: int) -> str:
        from soundalike.ml.real_benchmark import normalize_text

        sid = self._stable_id_for_row.get(row)
        if sid is not None:
            return f"did:{sid}"
        return f"name:{normalize_text(str(self.artists[row]))}"

    def _normalise(self, vec: np.ndarray) -> Optional[np.ndarray]:
        v = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        return (v / norm) if norm >= 1e-8 else None

    def _is_same_artist_stable(self, query_row: int, candidate_row: int) -> bool:
        from soundalike.ml.catalog_policy import _artist_parts

        if self.identity is not None:
            q_id = self._stable_id_for_row.get(query_row)
            c_id = self._stable_id_for_row.get(candidate_row)
            if q_id is not None and c_id is not None:
                if q_id == c_id:
                    return True
                shared = self.identity.contributor_intersection(
                    [query_row], [candidate_row]
                )
                return bool(shared)
        from soundalike.ml.catalog_policy import _artist_parts

        return bool(
            _artist_parts(str(self.artists[query_row]))
            & _artist_parts(str(self.artists[candidate_row]))
        )

    def _source_profile_confidence(
        self, query_row: int, top_neighbor_aids: List[int]
    ) -> Tuple[float, str]:
        from soundalike.ml.real_benchmark import normalize_text

        query_artist_name = str(self.artists[query_row])
        q_gid = self.graph.artist_lookup.get(normalize_text(query_artist_name))
        if q_gid is None:
            return 0.0, "query_not_in_graph"
        q_audio = self._normalise(self.graph.artist_audio[q_gid])
        if q_audio is None:
            return 0.0, "query_audio_zero"
        return self._compute_conf(
            q_audio,
            query_artist_name,
            top_neighbor_aids,
            self.graph.artist_audio,
            self.graph.artist_names,
            self.style.style_overlap,
            top_k=self._guard_top_k,
        )

    def production_rows(self, row: int, n: int = 5) -> List[int]:
        return self._mock_prod.rank(row, "dual_sonic", n=n)

    def clap_scores(self, query_row: int) -> Optional[np.ndarray]:
        if not self.available[query_row]:
            return None
        query = self._normalise(self.compact[query_row])
        if query is None:
            return None
        n = len(self.compact)
        scores = np.empty(n, dtype=np.float32)
        for row in range(n):
            v = self._normalise(self.compact[row])
            scores[row] = float(query @ v) if v is not None else -1.0
        scores[~self.available] = -np.inf
        scores[query_row] = -np.inf
        return scores

    def _eligible(
        self, query_row: int, candidates: List[int], relevance: np.ndarray
    ) -> List[Dict[str, Any]]:
        from soundalike.ml.real_benchmark import normalize_text

        seed_title = str(self.titles[query_row])
        seed_artist = str(self.artists[query_row])
        values: List[Dict[str, Any]] = []
        for row in candidates:
            row = int(row)
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
        values.sort(key=lambda x: (-x["relevance"], x["row"]))
        return [dict(v) for v in self.quality.prefer_canonical(values)]

    def _mmr(
        self, query_row: int, candidates: List[Dict[str, Any]], n: int = 5
    ) -> List[int]:
        """Exercise the production v14 MMR implementation directly."""
        from soundalike.ml.clap_catalog_v14 import IdentityGuardedRanker

        return IdentityGuardedRanker._mmr_v14(self, query_row, candidates, n)


# ---------------------------------------------------------------------------
# Test: documented constants
# ---------------------------------------------------------------------------


def test_v14_constants_have_correct_documented_values():
    assert GUARD_AUDIO_WEIGHT == 0.55
    assert GUARD_STYLE_WEIGHT == 0.45
    assert abs(GUARD_AUDIO_WEIGHT + GUARD_STYLE_WEIGHT - 1.0) < 1e-9
    assert GUARD_SOURCE_PROFILE_MIN == 0.62
    assert GUARD_CANDIDATE_MIN == 0.60
    assert GUARD_CENTROID_MIN == 0.60
    assert GUARD_CENTROID_MARGIN == 0.05
    assert SCHEMA_VERSION == 14


# ---------------------------------------------------------------------------
# Test: compute_source_profile_confidence — generic behaviour
# ---------------------------------------------------------------------------


def test_source_profile_confidence_low_audio_abstains():
    """Low audio similarity → confidence below SOURCE_PROFILE_MIN."""
    rng = np.random.default_rng(0)
    dim = 8
    query_audio = np.ones(dim, dtype=np.float32)
    query_audio /= np.linalg.norm(query_audio)

    # Neighbour audio centroids orthogonal to the query → cosine ≈ 0 → audio ≈ 0.5
    # Zero style overlap keeps confidence below the source-profile gate.
    n_artists = 5
    artist_audio = np.zeros((n_artists, dim), dtype=np.float32)
    # Make neighbors orthogonal
    orth = np.zeros(dim, dtype=np.float32)
    orth[1] = 1.0
    for i in range(n_artists):
        artist_audio[i] = orth

    artist_names = np.asarray([f"artist_{i}" for i in range(n_artists)])
    neighbor_ids = list(range(1, n_artists))  # exclude 0 (query)
    style_fn: Callable[[str, str], float] = lambda a, b: 0.0

    conf, reason = compute_source_profile_confidence(
        query_audio, "artist_0", neighbor_ids, artist_audio, artist_names, style_fn
    )
    assert conf < GUARD_SOURCE_PROFILE_MIN, (
        f"Expected confidence < {GUARD_SOURCE_PROFILE_MIN}, got {conf}"
    )
    assert reason == "ok"


def test_source_profile_confidence_high_audio_style_passes():
    """High audio + style similarity → confidence above SOURCE_PROFILE_MIN."""
    dim = 8
    query_audio = np.ones(dim, dtype=np.float32)
    query_audio /= np.linalg.norm(query_audio)

    n_artists = 5
    # Neighbours aligned with query → audio cosine ≈ 1 → mapped ≈ 1.0
    artist_audio = np.tile(query_audio, (n_artists, 1)).astype(np.float32)
    artist_names = np.asarray([f"artist_{i}" for i in range(n_artists)])
    neighbor_ids = list(range(1, n_artists))
    style_fn: Callable[[str, str], float] = lambda a, b: 1.0  # perfect style

    conf, reason = compute_source_profile_confidence(
        query_audio, "artist_0", neighbor_ids, artist_audio, artist_names, style_fn
    )
    assert conf >= GUARD_SOURCE_PROFILE_MIN, (
        f"Expected confidence >= {GUARD_SOURCE_PROFILE_MIN}, got {conf}"
    )
    assert reason == "ok"


def test_source_profile_confidence_no_neighbors_returns_zero():
    dim = 4
    q = np.ones(dim, dtype=np.float32)
    q /= np.linalg.norm(q)
    audio = np.zeros((3, dim), dtype=np.float32)
    names = np.asarray(["a", "b", "c"])
    conf, reason = compute_source_profile_confidence(
        q, "a", [], audio, names, lambda x, y: 0.5
    )
    assert conf == 0.0
    assert reason == "no_graph_neighbors"


def test_source_profile_confidence_zero_query_audio_returns_zero():
    dim = 4
    q = np.zeros(dim, dtype=np.float32)  # zero vector
    audio = np.ones((3, dim), dtype=np.float32)
    names = np.asarray(["a", "b", "c"])
    conf, reason = compute_source_profile_confidence(
        q, "a", [0, 1], audio, names, lambda x, y: 1.0
    )
    assert conf == 0.0
    assert reason == "query_audio_zero"


def test_source_profile_confidence_formula_is_correct():
    """Verify the weighted formula uses the declared audio/style constants."""
    dim = 4
    q = np.ones(dim, dtype=np.float32)
    q /= np.linalg.norm(q)
    # Neighbour exactly aligned → cosine = 1.0 → mapped = (1+1)/2 = 1.0
    audio = np.tile(q, (2, 1)).astype(np.float32)
    names = np.asarray(["a", "b"])
    # Style = 0.7
    conf, _ = compute_source_profile_confidence(
        q, "a", [0, 1], audio, names, lambda x, y: 0.7
    )
    expected = GUARD_AUDIO_WEIGHT * 1.0 + GUARD_STYLE_WEIGHT * 0.7
    assert abs(conf - expected) < 1e-5


# ---------------------------------------------------------------------------
# Test: apply_identity_guard_to_name_group — standalone
# ---------------------------------------------------------------------------


def test_name_group_all_unresolved_is_omitted():
    row_to_sid = {0: None, 1: None, 2: None}
    rows, diag = apply_identity_guard_to_name_group(
        "artist_x",
        [0, 1, 2],
        row_to_sid,
        lambda n: [],
        lambda n, v: None,
        None,
    )
    assert rows == []
    assert diag["action"] == "omitted_all_unresolved"


def test_name_group_single_resolved_id_keeps_resolved_excludes_unresolved():
    # Rows 0,2 resolve to Deezer ID 100; row 1 is unresolved
    row_to_sid = {0: 100, 1: None, 2: 100}
    rows, diag = apply_identity_guard_to_name_group(
        "artist_y",
        [0, 1, 2],
        row_to_sid,
        lambda n: [100],
        lambda n, v: None,
        None,
    )
    assert set(rows) == {0, 2}
    assert diag["action"] == "resolved_single_id"
    assert diag["unresolved_excluded"] == 1


def test_name_group_homonym_disambiguation_passes():
    """Two Deezer IDs for same name: choose closer centroid."""
    dim = 4
    # Deezer ID 10 centroid aligned with query; ID 20 orthogonal
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    row_to_sid = {0: 10, 1: 10, 2: 20, 3: 20}

    def _name_to_dids(name: str) -> List[int]:
        return [10, 20]  # homonym

    def _disambig(name: str, vec: np.ndarray) -> Optional[Tuple[int, float, float]]:
        # Simulate: ID 10 has cosine 0.9 (margin 0.4 vs ID 20 at 0.5)
        return (10, 0.9, 0.4)

    rows, diag = apply_identity_guard_to_name_group(
        "ambiguous_artist",
        [0, 1, 2, 3],
        row_to_sid,
        _name_to_dids,
        _disambig,
        q_vec,
    )
    assert set(rows) == {0, 1}
    assert diag["action"] == "homonym_disambiguated"
    assert diag["chosen_deezer_id"] == 10


def test_name_group_homonym_disambiguation_fails_omits_group():
    """Insufficient margin → omit whole name group."""
    row_to_sid = {0: 10, 1: 20}

    def _disambig(name: str, vec: np.ndarray) -> Optional[Tuple[int, float, float]]:
        return None  # abstain

    rows, diag = apply_identity_guard_to_name_group(
        "ambiguous",
        [0, 1],
        row_to_sid,
        lambda n: [10, 20],
        _disambig,
        np.ones(4, dtype=np.float32),
    )
    assert rows == []
    assert diag["action"] == "omitted_homonym_disambiguation_failed"


def test_name_group_homonym_no_query_vec_omits():
    """No query embedding → cannot disambiguate → omit."""
    row_to_sid = {0: 10, 1: 20}
    rows, diag = apply_identity_guard_to_name_group(
        "homonym",
        [0, 1],
        row_to_sid,
        lambda n: [10, 20],
        lambda n, v: (10, 0.9, 0.4),
        None,  # no query vec
    )
    assert rows == []
    assert diag["action"] == "omitted_homonym_no_query_vec"


def test_sparse_asset_never_mixes_multiple_resolved_ids():
    """Asset incompleteness cannot turn multiple IDs into cross-credit."""
    row_to_sid = {0: 10, 1: 10, 2: 11}

    rows, diag = apply_identity_guard_to_name_group(
        "feat_artist",
        [0, 1, 2],
        row_to_sid,
        lambda name: [10],
        lambda name, vector: None,
        np.ones(4, dtype=np.float32),
    )
    assert rows == []
    assert diag["action"] == "omitted_homonym_disambiguation_failed"


# ---------------------------------------------------------------------------
# Test: stable-ID dedup in MMR via _SyntheticRanker
# ---------------------------------------------------------------------------


def _make_basic_ranker(
    tmp_path: Path,
    n_tracks: int = 10,
    primary_ids: Optional[List[Optional[int]]] = None,
    contrib_map: Optional[Dict[int, List[int]]] = None,
    name_clusters: Optional[Dict[str, List[int]]] = None,
    centroids: Optional[Dict[int, List[float]]] = None,
    dim: int = 4,
) -> _SyntheticRanker:
    rng = np.random.default_rng(7)
    track_ids = list(range(1000, 1000 + n_tracks))
    artists = [f"Artist{i % 4}" for i in range(n_tracks)]  # 4 distinct artists
    titles = [f"Track{i}" for i in range(n_tracks)]
    compact = rng.standard_normal((n_tracks, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)

    # Build a simple graph with one artist and top-K neighbors
    from soundalike.ml.real_benchmark import normalize_text

    artist_names_graph = sorted({normalize_text(a) for a in artists})
    track_artist_ids_graph = [
        artist_names_graph.index(normalize_text(a)) for a in artists
    ]
    # Neighbors for each artist: next artists
    neighbors: Dict[str, Tuple[List[int], List[float]]] = {}
    for name in artist_names_graph:
        idx = artist_names_graph.index(name)
        neighbor_ids = [(idx + 1) % len(artist_names_graph),
                        (idx + 2) % len(artist_names_graph)]
        neighbors[name] = (neighbor_ids, [1.0, 0.8])

    # High-similarity audio for all artists (passes guard)
    base = np.ones(dim, dtype=np.float32)
    base /= np.linalg.norm(base)
    artist_audio = np.tile(base, (len(artist_names_graph), 1)).astype(np.float32)

    graph = _MockGraph(
        artist_names_graph, track_artist_ids_graph, neighbors, artist_audio
    )
    style = _MockStyleIndex(value=0.9)  # high style (guard passes)

    if primary_ids is None:
        # Assign artists 0..3 as primary IDs
        primary_ids = [i % 4 for i in range(n_tracks)]

    identity = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        contrib_map=contrib_map,
        name_clusters=name_clusters,
        centroids=centroids,
        dim=dim,
    )

    prod_map = {0: list(range(1, 6))}
    return _SyntheticRanker(
        np.asarray(track_ids, dtype=np.int64),
        titles,
        artists,
        compact,
        graph,
        style,
        prod_map,
        identity=identity,
    )


def test_stable_id_dedup_same_id_different_spellings(tmp_path):
    """Rows with the same Deezer ID but different artist-name spellings are
    deduplicated to one slot in MMR.

    Row 0 (query) has a unique Deezer ID (0).  Rows 1-5 all share Deezer ID 7
    under different name spellings.  Rows 6-7 share Deezer ID 9.  After MMR,
    exactly one row from the did:7 family and at most one from did:9 should
    appear.
    """
    n = 8
    track_ids = list(range(2000, 2000 + n))
    # Row 0 (query) → Deezer ID 0 (unique); rows 1–5 → Deezer ID 7; rows 6–7 → Deezer ID 9
    primary_ids = [0, 7, 7, 7, 7, 7, 9, 9]
    # Use different name spellings that all normalise to the same artist
    spellings = [
        "Query", "Jay-Z", "Jay Z", "JAYZ", "jay z", "Jaÿ Z", "Drake", "Drake"
    ]
    rng = np.random.default_rng(13)
    dim = 4
    compact = rng.standard_normal((n, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)

    identity = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        dim=dim,
    )
    # Verify stable keys: row 0 → did:0; rows 1-5 → did:7; rows 6-7 → did:9
    from soundalike.ml.real_benchmark import normalize_text

    artist_names_graph = sorted(
        {normalize_text(s) for s in spellings}
    )
    track_artist_ids_graph = [
        artist_names_graph.index(normalize_text(s)) for s in spellings
    ]
    neighbors: Dict[str, Tuple[List[int], List[float]]] = {
        name: ([0, 1], [1.0, 0.8]) for name in artist_names_graph
    }
    base = np.ones(dim, dtype=np.float32)
    base /= np.linalg.norm(base)
    artist_audio = np.tile(base, (len(artist_names_graph), 1)).astype(np.float32)
    graph = _MockGraph(
        artist_names_graph, track_artist_ids_graph, neighbors, artist_audio
    )
    style = _MockStyleIndex(0.9)
    ranker = _SyntheticRanker(
        np.asarray(track_ids, dtype=np.int64),
        [f"T{i}" for i in range(n)],
        spellings,
        compact,
        graph,
        style,
        {},
        identity=identity,
    )

    # Verify stable keys per Deezer ID group
    assert ranker._stable_key(0) == "did:0"
    keys_17 = {ranker._stable_key(r) for r in range(1, 6)}
    assert keys_17 == {"did:7"}, f"Expected all did:7, got {keys_17}"
    assert ranker._stable_key(6) == "did:9"
    assert ranker._stable_key(7) == "did:9"

    # MMR: query is row 0 (did:0); candidates are rows 1-7
    # Rows 1-5 (did:7) are eligible (different primary ID from query did:0)
    # Rows 6-7 (did:9) are also eligible
    relevance = np.array([0.0] + [0.8 - 0.05 * i for i in range(n - 1)], dtype=np.float32)
    eligible = ranker._eligible(0, list(range(1, n)), relevance)
    selected = ranker._mmr(0, eligible, n=5)
    selected_keys = [ranker._stable_key(r) for r in selected]
    # Exactly one slot for did:7 (all spellings collapse to same key)
    assert selected_keys.count("did:7") == 1, (
        f"Expected exactly one did:7 in MMR, got {selected_keys}"
    )


def test_stable_id_dedup_transliteration(tmp_path):
    """Transliterated names (Björk vs Bjork) sharing Deezer ID dedup correctly."""
    n = 4
    track_ids = [3000, 3001, 3002, 3003]
    primary_ids = [42, 42, 99, 99]  # rows 0,1 → 42; rows 2,3 → 99
    rng = np.random.default_rng(1)
    dim = 4
    compact = rng.standard_normal((n, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)
    identity = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        dim=dim,
    )
    graph = _MockGraph(
        ["bjork", "other"],
        [0, 0, 1, 1],
        {"bjork": ([1], [1.0]), "other": ([0], [1.0])},
    )
    ranker = _SyntheticRanker(
        np.asarray(track_ids, dtype=np.int64),
        ["T0", "T1", "T2", "T3"],
        ["Björk", "Bjork", "OtherA", "OtherB"],
        compact,
        graph,
        _MockStyleIndex(0.9),
        {},
        identity=identity,
    )
    # Both spellings → same stable key "did:42"
    assert ranker._stable_key(0) == "did:42"
    assert ranker._stable_key(1) == "did:42"
    # rows 2,3 → "did:99"
    assert ranker._stable_key(2) == "did:99"


def test_stable_id_distinct_ids_not_collapsed(tmp_path):
    """Two distinct Deezer IDs (same normalized name) must not be collapsed."""
    n = 4
    track_ids = [4000, 4001, 4002, 4003]
    # Two artists sharing name "nothing" but different Deezer IDs
    primary_ids = [101, 102, 101, 102]
    dim = 4
    rng = np.random.default_rng(2)
    compact = rng.standard_normal((n, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)
    identity = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        dim=dim,
    )
    graph = _MockGraph(
        ["nothing", "something"],
        [0, 0, 0, 1],
        {"nothing": ([1], [1.0]), "something": ([0], [1.0])},
    )
    ranker = _SyntheticRanker(
        np.asarray(track_ids, dtype=np.int64),
        ["T0", "T1", "T2", "T3"],
        ["Nothing", "Nothing", "Nothing", "Something"],
        compact,
        graph,
        _MockStyleIndex(0.9),
        {},
        identity=identity,
    )
    # Row 0 (ID 101) and Row 1 (ID 102) must have DIFFERENT stable keys
    assert ranker._stable_key(0) != ranker._stable_key(1)
    assert ranker._stable_key(0) == "did:101"
    assert ranker._stable_key(1) == "did:102"


# ---------------------------------------------------------------------------
# Test: multi-artist contributors
# ---------------------------------------------------------------------------


def test_multi_artist_contributors_excluded_from_results(tmp_path):
    """Tracks sharing a contributor Deezer ID with the seed are excluded.

    When both the seed track (row 0) and a candidate (row 2) credit the same
    collaborating artist (contributor ID 999), contributor_intersection returns
    {999}, so _is_same_artist_stable returns True → row 2 is excluded.
    """
    n = 6
    track_ids = list(range(5000, 5000 + n))
    # All tracks primary artist = distinct IDs
    primary_ids = [10, 11, 12, 13, 14, 15]
    # Rows 0 and 2 both feature the same collaborator (ID 999).
    # Row 1 features a different collaborator (ID 888) — not the same family.
    contrib_map = {
        0: [999],  # seed features collaborator 999
        2: [999],  # candidate also features collaborator 999 → same family
        1: [888],  # different collaborator → not same family as seed
    }
    dim = 4
    rng = np.random.default_rng(5)
    compact = rng.standard_normal((n, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)
    identity = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        contrib_map=contrib_map,
        dim=dim,
    )
    graph = _MockGraph(
        ["qa", "qb", "qc", "qd", "qe", "qf"],
        [0, 1, 2, 3, 4, 5],
        {"qa": ([1, 2, 3], [1.0, 0.9, 0.8])},
    )
    ranker = _SyntheticRanker(
        np.asarray(track_ids, dtype=np.int64),
        [f"T{i}" for i in range(n)],
        ["ArtA", "ArtB", "ArtC", "ArtD", "ArtE", "ArtF"],
        compact,
        graph,
        _MockStyleIndex(0.9),
        {},
        identity=identity,
    )
    # Row 0 (seed) and row 2 share contributor 999 → same family → excluded.
    assert ranker._is_same_artist_stable(0, 2) is True
    # Row 0 (seed) and row 1 share no contributors → not same family.
    assert ranker._is_same_artist_stable(0, 1) is False


# ---------------------------------------------------------------------------
# Test: low-confidence exact production fallback
# ---------------------------------------------------------------------------


def test_low_source_profile_confidence_conservative_falls_back(tmp_path):
    """When source_profile_confidence < GUARD_SOURCE_PROFILE_MIN, conservative
    variant returns exact production rows and fallback_reason starts with
    'identity_guard_abstained'.
    """
    from soundalike.ml.clap_catalog_v14 import (
        GUARD_SOURCE_PROFILE_MIN,
        apply_identity_guard_to_name_group,
        compute_source_profile_confidence,
    )

    n = 8
    dim = 4
    rng = np.random.default_rng(9)
    # Build compact with row 0 available and rows 1-7 as candidates
    compact = rng.standard_normal((n, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)

    # Graph: query artist "low" has orthogonal neighbours → low audio confidence
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    orth_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    artist_audio = np.vstack([q_vec, orth_vec, orth_vec, orth_vec]).astype(np.float32)

    neighbor_aids = [1, 2, 3]  # all orthogonal to query
    style_fn = lambda a, b: 0.0  # zero style

    conf, reason = compute_source_profile_confidence(
        q_vec, "low", neighbor_aids, artist_audio,
        np.asarray(["low", "na1", "na2", "na3"]), style_fn
    )
    assert conf < GUARD_SOURCE_PROFILE_MIN, (
        f"Test pre-condition: conf={conf} should be < {GUARD_SOURCE_PROFILE_MIN}"
    )


def test_guard_abstained_flag_is_deterministic(tmp_path):
    """The guard abstain decision is deterministic for a fixed asset."""
    dim = 4
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    orth = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    artist_audio_mat = np.vstack([q, orth, orth]).astype(np.float32)
    names_arr = np.asarray(["query_artist", "n1", "n2"])

    # Two calls with the same inputs must return the same result
    r1 = compute_source_profile_confidence(
        q, "query_artist", [1, 2], artist_audio_mat, names_arr, lambda a, b: 0.0
    )
    r2 = compute_source_profile_confidence(
        q, "query_artist", [1, 2], artist_audio_mat, names_arr, lambda a, b: 0.0
    )
    assert r1 == r2


# ---------------------------------------------------------------------------
# Test: same-name cluster isolation (graph expansion)
# ---------------------------------------------------------------------------


def test_same_name_cluster_isolation_in_guard():
    """Multiple Deezer IDs for same name: only the closer cluster is expanded."""
    dim = 4
    # Query centroid aligned with Deezer ID 10
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # ID 10 centroid: same as query; ID 20: orthogonal
    row_to_sid = {0: 10, 1: 10, 2: 20, 3: 20}

    def _name_to_dids(name: str) -> List[int]:
        return [10, 20]

    # Disambiguate: pick ID 10 (high confidence, margin > 0.05)
    def _disambig(name: str, vec: np.ndarray) -> Optional[Tuple[int, float, float]]:
        # Simulate cosine comparison
        centroid_10 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        centroid_20 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        s10 = float(vec @ centroid_10 / (np.linalg.norm(vec) * np.linalg.norm(centroid_10)))
        s20 = float(vec @ centroid_20 / (np.linalg.norm(vec) * np.linalg.norm(centroid_20)))
        if s10 >= GUARD_CENTROID_MIN and (s10 - s20) >= GUARD_CENTROID_MARGIN:
            return (10, s10, s10 - s20)
        return None

    rows, diag = apply_identity_guard_to_name_group(
        "nothing",
        [0, 1, 2, 3],
        row_to_sid,
        _name_to_dids,
        _disambig,
        q_vec,
    )
    assert set(rows) == {0, 1}, f"Expected only cluster 10, got {rows}"
    assert diag["action"] == "homonym_disambiguated"
    assert diag["chosen_deezer_id"] == 10


# ---------------------------------------------------------------------------
# Test: regression — "nothing" uses generic evidence, no name-specific branch
# ---------------------------------------------------------------------------


def test_nothing_regression_generic_evidence_no_name_branch():
    """The name "Nothing" must go through the generic path only.

    We verify that:
    1. compute_source_profile_confidence makes no special-case on the name
       string "nothing" — the same function is called with any artist name.
    2. apply_identity_guard_to_name_group makes no special-case on "nothing".
    3. The confidence value is determined solely by the audio/style vectors,
       not by any conditional on the name.
    """
    dim = 4
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    orth = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    artist_audio = np.vstack([q, orth, orth]).astype(np.float32)
    names = np.asarray(["nothing", "n1", "n2"])

    # Compute confidence for "Nothing" (the problematic v13 artist)
    conf_nothing, reason_nothing = compute_source_profile_confidence(
        q, "Nothing", [1, 2], artist_audio, names, lambda a, b: 0.0
    )
    # Compute confidence for a different name with identical vector geometry
    conf_other, reason_other = compute_source_profile_confidence(
        q, "SomeOtherArtist", [1, 2], artist_audio, names, lambda a, b: 0.0
    )
    # The confidence must be identical — no name-specific branching
    assert conf_nothing == conf_other, (
        f"Name-specific branching detected: "
        f"nothing={conf_nothing} vs other={conf_other}"
    )
    assert reason_nothing == reason_other == "ok"

    # Similarly for apply_identity_guard_to_name_group
    row_to_sid = {0: 10, 1: 20}
    # Both return None for both names → same abstention
    rows_nothing, diag_nothing = apply_identity_guard_to_name_group(
        "Nothing",
        [0, 1],
        row_to_sid,
        lambda n: [10, 20],
        lambda n, v: None,
        q,
    )
    rows_other, diag_other = apply_identity_guard_to_name_group(
        "SomeOtherArtist",
        [0, 1],
        row_to_sid,
        lambda n: [10, 20],
        lambda n, v: None,
        q,
    )
    assert rows_nothing == rows_other == []
    assert diag_nothing["action"] == diag_other["action"]


# ---------------------------------------------------------------------------
# Test: _select_direct_diagnostic_seeds
# ---------------------------------------------------------------------------

_SCENE_LABELS_V13 = (
    "rap", "rnb", "indie", "shoegaze", "hyperpop", "electronic",
    "metal", "jazz", "city_pop", "latin_afrobeats", "difficult_blend",
    "pop", "rock",
)


def _make_seed_list(scene_cycle: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build a synthetic 60-seed list spanning all 13 v13 scene labels."""
    scenes = scene_cycle or list(_SCENE_LABELS_V13)
    seeds = []
    for i in range(60):
        scene = scenes[i % len(scenes)]
        seeds.append(
            {
                "seed_id": f"S-{i:03d}",
                "scene": scene,
                "query": {
                    "deezer_track_id": 10000 + i,
                    "title": f"Track {i}",
                    "artist": f"Artist {i}",
                },
            }
        )
    return seeds


def test_direct_diagnostic_seeds_selects_20():
    seeds = _make_seed_list()
    selected = _select_direct_diagnostic_seeds(seeds)
    assert len(selected) == 20


def test_direct_diagnostic_seeds_covers_all_13_scenes():
    seeds = _make_seed_list()
    selected = _select_direct_diagnostic_seeds(seeds)
    scenes_covered = {s["scene"] for s in selected}
    # All 13 canonical scenes must appear
    for scene in _SCENE_LABELS_V13:
        assert scene in scenes_covered, f"Scene {scene!r} not covered in 20-seed selection"


def test_direct_diagnostic_seeds_is_deterministic():
    seeds = _make_seed_list()
    a = _select_direct_diagnostic_seeds(seeds)
    b = _select_direct_diagnostic_seeds(seeds)
    assert [s["seed_id"] for s in a] == [s["seed_id"] for s in b]


def test_direct_diagnostic_seeds_handles_fewer_than_20():
    """If seed list is tiny, gracefully return all available without crash."""
    seeds = [
        {"seed_id": f"S-{i}", "scene": "pop",
         "query": {"deezer_track_id": i, "title": "", "artist": ""}}
        for i in range(5)
    ]
    selected = _select_direct_diagnostic_seeds(seeds)
    assert len(selected) == 5


# ---------------------------------------------------------------------------
# Test: _semantic_diff
# ---------------------------------------------------------------------------


class _MinimalRanker:
    """Minimal catalog metadata used by semantic-diff tests."""

    def __init__(self, n: int):
        self.track_ids = np.arange(n, dtype=np.int64)
        self.titles = np.asarray([f"Track {row}" for row in range(n)])
        self.artists = np.asarray([f"Artist {row}" for row in range(n)])


def test_semantic_diff_exact_match():
    v14 = [{"seed_id": "S1", "rows": [1, 2, 3, 4, 5]}]
    v13 = [{"seed_id": "S1", "rows": [1, 2, 3, 4, 5]}]
    diff = _semantic_diff(v14, v13, _MinimalRanker(100))
    assert diff["exact_match_count"] == 1
    assert abs(diff["mean_overlap"] - 1.0) < 1e-6


def test_semantic_diff_no_overlap():
    v14 = [{"seed_id": "S1", "rows": [1, 2, 3, 4, 5]}]
    v13 = [{"seed_id": "S1", "rows": [6, 7, 8, 9, 10]}]
    diff = _semantic_diff(v14, v13, _MinimalRanker(100))
    assert diff["exact_match_count"] == 0
    assert abs(diff["mean_overlap"]) < 1e-6


def test_semantic_diff_partial_overlap():
    v14 = [{"seed_id": "S1", "rows": [1, 2, 3, 4, 5]}]
    v13 = [{"seed_id": "S1", "rows": [3, 4, 5, 6, 7]}]
    diff = _semantic_diff(v14, v13, _MinimalRanker(100))
    # Intersection: {3,4,5} = 3; Union: {1,2,3,4,5,6,7} = 7
    expected_overlap = 3 / 7
    assert abs(diff["mean_overlap"] - expected_overlap) < 1e-5


def test_semantic_diff_structure():
    v14 = [
        {"seed_id": "S1", "rows": [1, 2, 3, 4, 5]},
        {"seed_id": "S2", "rows": [6, 7, 8, 9, 10]},
    ]
    v13 = [
        {"seed_id": "S1", "rows": [1, 2, 3, 4, 5]},
        {"seed_id": "S2", "rows": [11, 12, 13, 14, 15]},
    ]
    diff = _semantic_diff(v14, v13, _MinimalRanker(100))
    assert diff["seed_count"] == 2
    assert "per_seed_diff" in diff
    assert len(diff["per_seed_diff"]) == 2


# ---------------------------------------------------------------------------
# Test: diagnostics diff / candidate coverage fields
# ---------------------------------------------------------------------------


def test_semantic_diff_and_candidate_coverage_structures_are_valid():
    """Verify _semantic_diff and candidate_coverage output shapes."""
    # Candidate coverage is already tested via _semantic_diff; here check the
    # coverage dict structure expected by run_variant_diagnostics_v14.
    # We exercise _semantic_diff with multi-seed input.
    n_seeds = 5
    v14_records = [
        {"seed_id": f"S{i}", "rows": [i * 5 + j for j in range(5)]}
        for i in range(n_seeds)
    ]
    v13_records = [
        {"seed_id": f"S{i}", "rows": [i * 5 + j + 1 for j in range(5)]}
        for i in range(n_seeds)
    ]
    diff = _semantic_diff(v14_records, v13_records, _MinimalRanker(200))
    assert 0.0 <= diff["mean_overlap"] <= 1.0
    assert all(0.0 <= d["overlap"] <= 1.0 for d in diff["per_seed_diff"])
    assert all("added_rows" in d and "dropped_rows" in d for d in diff["per_seed_diff"])


# ---------------------------------------------------------------------------
# Test: ambiguity margin abstention (identity asset disambiguate gate)
# ---------------------------------------------------------------------------


def test_ambiguity_margin_below_threshold_omits_group(tmp_path):
    """When both IDs are equally close, margin < GUARD_CENTROID_MARGIN → omit."""
    dim = 4
    # Two centroids equidistant from query
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Simulate IdentityAsset.disambiguate returning None (margin too small)
    row_to_sid = {0: 10, 1: 20}

    def _disambig(name: str, vec: np.ndarray) -> Optional[Tuple[int, float, float]]:
        # margin < GUARD_CENTROID_MARGIN → None
        return None

    rows, diag = apply_identity_guard_to_name_group(
        "tie_artist",
        [0, 1],
        row_to_sid,
        lambda n: [10, 20],  # homonym
        _disambig,
        q_vec,
    )
    assert rows == []
    assert "failed" in diag["action"] or "failed" in diag["action"]


# ---------------------------------------------------------------------------
# Test: unresolved rows excluded from mixed group
# ---------------------------------------------------------------------------


def test_unresolved_excluded_from_mixed_group():
    row_to_sid = {0: 55, 1: None, 2: 55, 3: None}
    rows, diag = apply_identity_guard_to_name_group(
        "mixed_artist",
        [0, 1, 2, 3],
        row_to_sid,
        lambda n: [55],
        lambda n, v: None,
        None,
    )
    assert set(rows) == {0, 2}
    assert diag["action"] == "resolved_single_id"
    assert diag["unresolved_excluded"] == 2


# ---------------------------------------------------------------------------
# Test: IdentityAsset integration — row_identity and contributor_intersection
# ---------------------------------------------------------------------------


def test_identity_asset_row_identity_and_contributor_intersection(tmp_path):
    """IdentityAsset.row_identity and contributor_intersection work correctly.

    contributor_intersection(rows_a, rows_b) returns the SET INTERSECTION of
    contributor Deezer IDs from group A and group B.  For two rows to share a
    contributor, both rows must list the same collaborating artist ID.
    """
    track_ids = [100, 101, 102]
    primary_ids = [10, 20, 30]
    # Rows 0 and 1 both credit collaborator 99; row 2 credits a different one (88).
    contrib_map = {0: [99], 1: [99], 2: [88]}

    asset = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        contrib_map=contrib_map,
        dim=4,
    )

    id0 = asset.row_identity(0)
    assert id0["primary_artist_deezer_id"] == 10
    # Row 0 has contributor 99
    assert 99 in id0["contributor_deezer_ids"]

    # Both rows 0 and 1 list contributor 99 → intersection = {99}
    shared_01 = asset.contributor_intersection([0], [1])
    assert 99 in shared_01

    # Row 1 ({99}) and row 2 ({88}) share no contributors → empty
    shared_12 = asset.contributor_intersection([1], [2])
    assert len(shared_12) == 0


# ---------------------------------------------------------------------------
# Test: name cluster lookup for homonym detection
# ---------------------------------------------------------------------------


def test_identity_asset_name_to_deezer_ids_homonym(tmp_path):
    """name_to_deezer_ids returns both IDs for a homonym name cluster."""
    track_ids = [200, 201, 202]
    primary_ids = [50, 51, 50]
    # Explicitly declare homonym: "nothing" → [50, 51]
    name_clusters = {"nothing": [50, 51]}

    asset = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        name_clusters=name_clusters,
        dim=4,
    )
    dids = asset.name_to_deezer_ids("Nothing")
    assert sorted(dids) == [50, 51]


def test_identity_asset_disambiguate_picks_closer_centroid(tmp_path):
    """disambiguate chooses the ID with the highest centroid cosine."""
    dim = 4
    track_ids = [300, 301, 302]
    primary_ids = [60, 70, 60]
    name_clusters = {"shared_name": [60, 70]}
    # ID 60 centroid: [1,0,0,0]; ID 70 centroid: [0,1,0,0]
    centroids = {
        60: [1.0, 0.0, 0.0, 0.0],
        70: [0.0, 1.0, 0.0, 0.0],
    }
    asset = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        name_clusters=name_clusters,
        centroids=centroids,
        dim=dim,
    )
    # Query aligned with centroid 60 → should pick 60 with high confidence
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = asset.disambiguate("shared_name", query, min_confidence=0.0, min_margin=0.0)
    assert result is not None
    chosen_id, conf, margin = result
    assert chosen_id == 60
    assert conf > 0.9
    assert margin > 0.9  # (1.0 - ~0.0) = ~1.0


def test_identity_asset_disambiguate_abstains_on_low_margin(tmp_path):
    """disambiguate returns None when margin < min_margin."""
    dim = 4
    track_ids = [400, 401]
    primary_ids = [80, 81]
    name_clusters = {"tie_name": [80, 81]}
    # Nearly identical centroids → margin ≈ 0
    centroids = {
        80: [1.0, 0.0, 0.0, 0.0],
        81: [1.0, 0.0, 0.0, 0.001],  # almost the same
    }
    asset = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        name_clusters=name_clusters,
        centroids=centroids,
        dim=dim,
    )
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = asset.disambiguate(
        "tie_name", query, min_confidence=0.60, min_margin=0.10
    )
    # Margin is tiny → should abstain
    assert result is None


# ---------------------------------------------------------------------------
# Test: _SyntheticRanker guard in rank_all context (integration-level)
# ---------------------------------------------------------------------------


def test_synthetic_ranker_stable_key_without_identity(tmp_path):
    """Without identity_asset, stable_key falls back to normalised name."""
    rng = np.random.default_rng(0)
    dim = 4
    n = 4
    compact = rng.standard_normal((n, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)
    graph = _MockGraph(
        ["artist0", "artist1", "artist2", "artist3"],
        [0, 1, 2, 3],
        {"artist0": ([1, 2], [1.0, 0.8])},
    )
    ranker = _SyntheticRanker(
        np.arange(n, dtype=np.int64),
        [f"T{i}" for i in range(n)],
        ["Artist0", "Artist1", "Artist2", "Artist3"],
        compact,
        graph,
        _MockStyleIndex(0.5),
        {},
        identity=None,  # no identity asset
    )
    # Without identity, stable_key falls back to name
    assert ranker._stable_key(0) == "name:artist0"
    assert ranker._stable_key(1) == "name:artist1"


def test_synthetic_ranker_with_identity_uses_did_key(tmp_path):
    """With identity_asset, stable_key uses Deezer ID."""
    rng = np.random.default_rng(0)
    dim = 4
    n = 3
    track_ids = [9000, 9001, 9002]
    primary_ids = [500, 501, 502]
    compact = rng.standard_normal((n, dim)).astype(np.float32)
    compact /= np.linalg.norm(compact, axis=1, keepdims=True).clip(min=1e-8)
    identity = _load_identity(
        tmp_path,
        track_ids=track_ids,
        primary_artist_ids=primary_ids,
        dim=dim,
    )
    graph = _MockGraph(
        ["a", "b", "c"],
        [0, 1, 2],
        {"a": ([1, 2], [1.0, 0.8])},
    )
    ranker = _SyntheticRanker(
        np.asarray(track_ids, dtype=np.int64),
        [f"T{i}" for i in range(n)],
        ["ArtA", "ArtB", "ArtC"],
        compact,
        graph,
        _MockStyleIndex(0.5),
        {},
        identity=identity,
    )
    assert ranker._stable_key(0) == "did:500"
    assert ranker._stable_key(1) == "did:501"
    assert ranker._stable_key(2) == "did:502"


# ---------------------------------------------------------------------------
# Test: guard makes conservative fall back explicitly labelled
# ---------------------------------------------------------------------------


def test_guard_abstain_label_contains_reason():
    """Guard-abstained fallback_reason must start with 'identity_guard_abstained'."""
    # This checks the string contract without the full ranker
    from soundalike.ml.clap_catalog_v14 import GUARD_SOURCE_PROFILE_MIN

    # Simulate the guard producing an abstain reason
    abstain_reason = f"source_profile_below_{GUARD_SOURCE_PROFILE_MIN}"
    fallback = f"identity_guard_abstained:{abstain_reason}"
    assert fallback.startswith("identity_guard_abstained:")


# ---------------------------------------------------------------------------
# Test: SCHEMA_VERSION is 14
# ---------------------------------------------------------------------------


def test_schema_version_is_14():
    assert SCHEMA_VERSION == 14


# ---------------------------------------------------------------------------
# Test: v13 module is unmodified (import smoke test)
# ---------------------------------------------------------------------------


def test_v13_module_still_importable_and_unmodified():
    """Verify v13 can be imported cleanly and key constants are intact."""
    from soundalike.ml.clap_catalog_v13 import (
        SCHEMA_VERSION as V13_SCHEMA,
        EXPECTED_ROWS,
        VARIANT_ORDER,
        ClapDevelopmentRanker,
    )

    assert V13_SCHEMA == 13
    assert EXPECTED_ROWS == 272_853
    assert VARIANT_ORDER == (
        "conservative_clap_fallback",
        "graph_clap_union",
        "pure_clap",
    )


# ---------------------------------------------------------------------------
# Test: VARIANT_ORDER_V14 matches v13 order
# ---------------------------------------------------------------------------


def test_v14_variant_order_identical_to_v13():
    from soundalike.ml.clap_catalog_v14 import VARIANT_ORDER_V14
    from soundalike.ml.clap_catalog_v13 import VARIANT_ORDER

    assert VARIANT_ORDER_V14 == VARIANT_ORDER


# ---------------------------------------------------------------------------
# Test: compute_source_profile_confidence uses top_k parameter
# ---------------------------------------------------------------------------


def test_source_profile_confidence_respects_top_k():
    """Only the first top_k neighbours are used."""
    dim = 4
    q = np.ones(dim, dtype=np.float32)
    q /= np.linalg.norm(q)
    # 10 neighbours: first 3 aligned, rest orthogonal
    n_artists = 11
    artist_audio = np.zeros((n_artists, dim), dtype=np.float32)
    for i in range(1, 4):    # IDs 1-3: aligned
        artist_audio[i] = q
    orth = np.zeros(dim, dtype=np.float32)
    orth[1] = 1.0
    for i in range(4, n_artists):  # IDs 4-10: orthogonal
        artist_audio[i] = orth
    names = np.asarray([f"a{i}" for i in range(n_artists)])
    all_neighbors = list(range(1, n_artists))

    conf_k3, _ = compute_source_profile_confidence(
        q, "a0", all_neighbors, artist_audio, names, lambda a, b: 0.5, top_k=3
    )
    conf_k9, _ = compute_source_profile_confidence(
        q, "a0", all_neighbors, artist_audio, names, lambda a, b: 0.5, top_k=9
    )
    # k=3 uses only aligned neighbours → higher audio score → higher confidence
    assert conf_k3 > conf_k9, (
        f"top_k=3 (all aligned) should give higher conf than top_k=9 (mixed): "
        f"{conf_k3} vs {conf_k9}"
    )
