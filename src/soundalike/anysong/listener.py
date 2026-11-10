"""Listener-aware ranking for the lists users see (docs/ANYSONG_EVAL.md, "Listener-aware ranking").

A small gradient-boosted tree ensemble estimates, for each candidate song, how likely listeners of the seed's artist
also play the candidate's artist. It is trained offline on development seeds with ListenBrainz similar artists as
labels; ListenBrainz is never a feature. The features describe:
- audio closeness: the served sound score, and the song's rank and score in the melody, samples and sonic lists;
- the seed artist's Deezer neighbourhood: graph overlap, whether either artist lists the other as related, shared
  region and genre, the scene affinity;
- tempo and whether the song's language is known.
Popularity is deliberately not a feature: with it, lists concentrate on famous artists.

Candidates are the channel lists plus, for each artist in the seed artist's neighbourhood, that artist's songs closest
in sound (``SCENE_SONGS``, chosen among at most ``SCENE_MAX_TRACKS`` of its catalog songs). The model is stored as
plain arrays (``export``) and evaluated with numpy (``Ranker``), so serving does not depend on scikit-learn's pickles.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

from .scene import ArtistProfile, SceneParams, SceneTable, affinity, components

FEATURES = ("son", "r_melody", "s_melody", "r_samples", "s_samples", "r_sonic", "in_pool", "from_scene", "graph",
            "region", "genre", "aff", "fwd", "rev", "nb", "tempo_gap", "lang_known")
SCENE_SONGS = 3  # songs per neighbour artist added as candidates: its closest in sound
SCENE_MAX_TRACKS = 150  # catalog songs per neighbour artist scored for that choice (lowest Deezer ids first)
CHANNELS = ("melody", "samples", "sonic")


class ArtistTracks:
    """Catalog tracks per artist from the scene table, in ascending track id."""

    def __init__(self, table: SceneTable):
        order = np.argsort(table.track_artist, kind="stable")
        artists = table.track_artist[order]
        self.tracks = table.track_ids[order]
        self.artists, self.start, self.count = np.unique(artists, return_index=True, return_counts=True)

    def __call__(self, artist_id: int, limit: int = SCENE_MAX_TRACKS) -> np.ndarray:
        k = int(np.searchsorted(self.artists, artist_id))
        if k < len(self.artists) and self.artists[k] == artist_id and artist_id > 0:
            s = int(self.start[k])
            return self.tracks[s:s + min(int(self.count[k]), limit)]
        return self.tracks[:0]


def scene_candidates(profile: ArtistProfile, tracks: ArtistTracks, score: Callable[[list[int]], Mapping[int, float]],
                     artist_of: Callable[[int], int | None],
                     allowed: Callable[[int], bool] | None = None) -> dict[int, float]:
    """Each neighbour artist's ``SCENE_SONGS`` songs closest in sound: track id -> sound score. ``score`` returns the
    served sound score (hub penalty included) of indexed tracks and leaves the others out; ``allowed`` (the catalog
    tiers) drops tracks the seed may not get."""
    wanted = [int(t) for a in sorted(profile.neighbours) if a != profile.artist_id for t in tracks(int(a))]
    if allowed is not None:
        wanted = [t for t in wanted if allowed(t)]
    scored = score(wanted) if wanted else {}
    by_artist: dict[int, list[tuple[float, int]]] = {}
    for tid, s in scored.items():
        a = artist_of(int(tid))
        if a:
            by_artist.setdefault(a, []).append((float(s), int(tid)))
    out: dict[int, float] = {}
    for rows in by_artist.values():
        for s, tid in sorted(rows, key=lambda r: (-r[0], r[1]))[:SCENE_SONGS]:
            out[tid] = s
    return out


def tempo_gap(seed_bpm: float | None, bpm: float | None) -> float:
    """|log tempo ratio| to the nearest of the same, double or half tempo; NaN when either is unknown."""
    if not seed_bpm or not bpm:
        return math.nan
    return min(abs(math.log(bpm * m / seed_bpm)) for m in (1.0, 2.0, 0.5))


def features(raw: Mapping[str, Iterable[tuple[int, float]]], son: Mapping[int, float], scene: Iterable[int],
             query_id: int, profile: ArtistProfile, related: Iterable[int], table: SceneTable, params: SceneParams,
             seed_bpm: float | None, bpm_of: Callable[[int], float | None],
             language_known: Callable[[int], bool]) -> tuple[list[int], np.ndarray]:
    """Feature rows (``FEATURES`` order) for every candidate with a sound score: the channel lists' songs and the
    scene candidates, except the seed itself. Missing values are NaN."""
    pool: dict[int, dict[str, float]] = {}
    for channel in CHANNELS:
        for rank, (tid, s) in enumerate(raw.get(channel, ()), 1):
            entry = pool.setdefault(int(tid), {})
            entry.setdefault(f"r_{channel}", float(rank))
            entry.setdefault(f"s_{channel}", float(s))
    scene = {int(t) for t in scene}
    related = {int(a) for a in related}
    qa = profile.artist_id
    per_artist: dict[int, tuple[float, ...]] = {}
    tids, rows = [], []
    for tid in sorted((set(pool) | scene) - {int(query_id)}):
        if tid not in son:
            continue
        a = table.artist(tid) or 0
        if a not in per_artist:
            stored = table.stored(a) if a else None
            g, rg, gn = components(profile, stored) if stored is not None else (math.nan,) * 3
            rev = float(qa in (table.related(a) or ())) if a else 0.0
            per_artist[a] = (g, rg, gn, affinity(profile, stored, a, params) if a else 0.0, float(a in related),
                             rev, float(a in profile.neighbours))
        g, rg, gn, aff, fwd, rev, nb = per_artist[a]
        entry = pool.get(tid, {})
        rows.append((son[tid], entry.get("r_melody", math.nan), entry.get("s_melody", math.nan),
                     entry.get("r_samples", math.nan), entry.get("s_samples", math.nan),
                     entry.get("r_sonic", math.nan), float(tid in pool), float(tid in scene), g, rg, gn, aff, fwd,
                     rev, nb, tempo_gap(seed_bpm, bpm_of(tid)), float(language_known(tid))))
        tids.append(tid)
    return tids, np.asarray(rows, dtype=np.float64).reshape(len(rows), len(FEATURES))


def blend(primary: np.ndarray, secondary: np.ndarray, weight: float) -> np.ndarray:
    """z(primary) + weight * z(secondary) over one seed's candidates (constant columns contribute 0)."""
    return zscore(primary) + weight * zscore(secondary)


def zscore(x) -> np.ndarray:
    """Standard scores over one seed's candidates; a constant column scores 0."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Ranker:
    """A binary gradient-boosted tree ensemble as arrays: P(listeners connect the artists) for feature rows.

    Splits send a row left when its value is at most the threshold, and missing values (NaN) the way the tree
    learned (scikit-learn's HistGradientBoosting rules). Leaves point to themselves, so every row can descend
    ``depth`` steps."""

    def __init__(self, features: Iterable[str], baseline: float, roots: np.ndarray, feature: np.ndarray,
                 threshold: np.ndarray, missing_left: np.ndarray, left: np.ndarray, right: np.ndarray,
                 value: np.ndarray, depth: int, meta: str = ""):
        self.features = tuple(features)
        if self.features != FEATURES:
            raise ValueError(f"ranker features {self.features} differ from this code's {FEATURES}")
        n = len(feature)
        if not all(len(a) == n for a in (threshold, missing_left, left, right, value)):
            raise ValueError("inconsistent ranker arrays")
        if len(roots) == 0 or roots.max() >= n or left.max() >= n or right.max() >= n or \
                feature.max() >= len(FEATURES) or depth < 0:
            raise ValueError("ranker arrays point outside the model")
        self.baseline, self.roots, self.feature = float(baseline), roots.astype(np.int64), feature.astype(np.int64)
        self.threshold, self.missing_left = threshold.astype(np.float64), missing_left.astype(bool)
        self.left, self.right, self.value = left.astype(np.int64), right.astype(np.int64), value.astype(np.float64)
        self.depth, self.meta = int(depth), meta

    @classmethod
    def load(cls, path: Path | str) -> Ranker:
        with np.load(path, allow_pickle=False) as z:
            return cls([str(f) for f in z["features"]], float(z["baseline"]), z["roots"], z["feature"],
                       z["threshold"], z["missing_left"], z["left"], z["right"], z["value"], int(z["depth"]),
                       str(z["meta"]) if "meta" in z else "")

    def save(self, path: Path | str) -> None:
        np.savez(path, features=np.array(self.features), baseline=np.float64(self.baseline), roots=self.roots,
                 feature=self.feature, threshold=self.threshold, missing_left=self.missing_left, left=self.left,
                 right=self.right, value=self.value, depth=np.int64(self.depth), meta=np.array(self.meta))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError(f"expected rows of {len(self.features)} features")
        if not len(X):
            return np.zeros(0)
        node = np.broadcast_to(self.roots, (len(X), len(self.roots))).copy()
        rows = np.arange(len(X))[:, None]
        for _ in range(self.depth):
            x = X[rows, self.feature[node]]
            go_left = np.where(np.isnan(x), self.missing_left[node], x <= self.threshold[node])
            node = np.where(go_left, self.left[node], self.right[node])
        raw = self.baseline + self.value[node].sum(axis=1)
        return 1.0 / (1.0 + np.exp(-raw))


def export(model, meta: str = "") -> Ranker:
    """A fitted binary scikit-learn ``HistGradientBoostingClassifier`` (numeric features only) as a ``Ranker``."""
    if getattr(model, "n_trees_per_iteration_", 1) != 1:
        raise ValueError("only binary classifiers can be exported")
    feature, threshold, missing_left, left, right, value, roots = [], [], [], [], [], [], []
    depth = 0
    for (tree,) in model._predictors:
        nodes = tree.nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical splits are not supported")
        base = len(feature)
        roots.append(base)
        leaf = nodes["is_leaf"].astype(bool)
        idx = np.arange(len(nodes)) + base
        feature += np.where(leaf, 0, nodes["feature_idx"]).tolist()
        threshold += nodes["num_threshold"].tolist()
        missing_left += nodes["missing_go_to_left"].astype(bool).tolist()
        left += np.where(leaf, idx, nodes["left"] + base).tolist()
        right += np.where(leaf, idx, nodes["right"] + base).tolist()
        value += np.where(leaf, nodes["value"], 0.0).tolist()
        depth = max(depth, int(nodes["depth"].max()))
    return Ranker(FEATURES, float(np.ravel(model._baseline_prediction)[0]), np.array(roots), np.array(feature),
                  np.array(threshold), np.array(missing_left), np.array(left), np.array(right), np.array(value),
                  depth, meta)
