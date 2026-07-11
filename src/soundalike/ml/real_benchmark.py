"""Leakage-audited evaluation on sourced, real-world soundalike pairs.

This module intentionally does not use the old scene-label score.  Its relevance
labels are recording pairs asserted by public editorial, artist, listener, or
reference sources in ``benchmarks/soundalike_pairs.v1.json``.  Unknown artists do
not receive the benefit of the doubt.

The production baseline is the July 4, 2026 272,853-row ranking path with all
iteration-1 enhancements disabled.  Every report stores the actual top 50 so a
scalar score can always be audited against the ranked songs that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .genre_rerank import ArtistCentroidIndex
from .quality_filter import TitleQualityFilter

RECALL_CUTOFFS = (1, 5, 10, 20, 50)
PRIMARY_CUTOFF = 50
BENCHMARK_VERSION = "soundalike-real-pairs-v1"


def normalize_text(value: str) -> str:
    """Accent/punctuation-insensitive comparison key."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.casefold()
    value = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", value)
    value = re.sub(r"\s+-\s+(?:\d{4}\s+)?(?:re)?master(?:ed)?(?:\s+\d{4})?.*$", "", value)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def primary_artist(value: str) -> str:
    value = normalize_text(value)
    for separator in (" featuring ", " feat ", " ft ", " x ", " and ", " & ", ", "):
        if separator in value:
            value = value.split(separator, 1)[0]
    return value.strip()


def credited_artists(value: str) -> Set[str]:
    """Conservative artist tokens for split leakage audits."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(
        r"\s*(?:,|&|\+|\bx\b|\bwith\b|\band\b|\bfeaturing\b|\bfeat\.?\b|\bft\.?\b)\s*",
        ",",
        value.casefold(),
    )
    return {
        normalize_text(part)
        for part in value.split(",")
        if len(normalize_text(part)) > 1
    }


def load_benchmark(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("benchmark_id") != BENCHMARK_VERSION:
        raise ValueError(f"Unexpected benchmark id: {data.get('benchmark_id')!r}")
    if len(data.get("pairs", [])) < 50:
        raise ValueError("The real-world benchmark must contain at least 50 pairs")
    return data


def held_out_artists(benchmark: Mapping[str, Any]) -> Set[str]:
    artists: Set[str] = set()
    for pair in benchmark["pairs"]:
        if pair["split"] != "held_out":
            continue
        artists.update(credited_artists(pair["query"]["artist"]))
        artists.update(credited_artists(pair["target"]["artist"]))
    return artists


def audit_leakage(
    benchmark: Mapping[str, Any],
    manual_pairs: Sequence[Tuple[str, str]] = (),
    graph_artists: Iterable[str] = (),
) -> Dict[str, Any]:
    """Fail closed when any held-out artist reaches development or graph data."""
    dev: Set[str] = set()
    test = held_out_artists(benchmark)
    pair_ids: Set[str] = set()
    duplicate_ids: List[str] = []
    for pair in benchmark["pairs"]:
        if pair["id"] in pair_ids:
            duplicate_ids.append(pair["id"])
        pair_ids.add(pair["id"])
        if pair["split"] == "development":
            dev.update(credited_artists(pair["query"]["artist"]))
            dev.update(credited_artists(pair["target"]["artist"]))

    dev_overlap = sorted(dev & test)
    manual_artists = {
        artist
        for edge in manual_pairs
        for raw in edge
        for artist in credited_artists(raw)
    }
    manual_overlap = sorted(manual_artists & test)
    graph_keys = {
        artist
        for raw in graph_artists
        for artist in credited_artists(raw)
    }
    graph_overlap = sorted(graph_keys & test)
    held_pairs = [p for p in benchmark["pairs"] if p["split"] == "held_out"]
    audit = {
        "passed": not (dev_overlap or manual_overlap or graph_overlap or duplicate_ids),
        "development_artist_count": len(dev),
        "held_out_artist_count": len(test),
        "held_out_pair_count": len(held_pairs),
        "development_held_out_overlap": dev_overlap,
        "manual_pair_held_out_overlap": manual_overlap,
        "graph_held_out_overlap": graph_overlap,
        "duplicate_pair_ids": duplicate_ids,
    }
    if len(held_pairs) != 20:
        audit["passed"] = False
        audit["held_out_count_error"] = f"expected 20, found {len(held_pairs)}"
    return audit


@dataclass
class ResolvedPair:
    pair: Mapping[str, Any]
    query_row: Optional[int]
    baseline_query_row: Optional[int]
    target_rows: Set[int]


class PairResolver:
    """Resolve canonical benchmark names to exact production catalogue rows."""

    def __init__(self, titles: Sequence[str], artists: Sequence[str]):
        self.titles = np.asarray(titles).astype(str)
        self.artists = np.asarray(artists).astype(str)
        self.normal_titles = np.asarray([normalize_text(x) for x in self.titles])
        self.primary_artists = np.asarray([primary_artist(x) for x in self.artists])
        self.by_title: Dict[str, List[int]] = {}
        for row, title in enumerate(self.normal_titles):
            self.by_title.setdefault(str(title), []).append(row)

    @staticmethod
    def _artist_match(canonical: str, catalogue: str) -> bool:
        canonical_parts = credited_artists(canonical)
        catalog_parts = credited_artists(catalogue)
        if canonical_parts & catalog_parts:
            return True
        cp = primary_artist(canonical)
        ap = primary_artist(catalogue)
        if not cp or not ap:
            return False
        if cp in ap or ap in cp:
            return min(len(cp), len(ap)) >= 4
        # Handles replacement-character damage in a few old catalogue names,
        # e.g. "Beyonc�" normalising to "beyonc".
        return len(cp) >= 5 and len(ap) >= 5 and cp[:5] == ap[:5]

    def rows(self, song: Mapping[str, str]) -> List[int]:
        title_key = normalize_text(song["title"])
        rows = self.by_title.get(title_key, [])
        matched = [
            int(row) for row in rows
            if self._artist_match(song["artist"], self.artists[row])
        ]
        return matched

    def query_row(self, song: Mapping[str, str]) -> Optional[int]:
        rows = self.rows(song)
        if not rows:
            return None

        def version_penalty(row: int) -> Tuple[int, int]:
            title = self.titles[row].casefold()
            derivative = int(bool(re.search(
                r"\b(?:karaoke|tribute|slowed|reverb|nightcore|instrumental|"
                r"remix|cover|live|acoustic)\b", title
            )))
            return derivative, len(title)

        return min(rows, key=version_penalty)

    def resolve(self, pairs: Sequence[Mapping[str, Any]]) -> List[ResolvedPair]:
        resolved = []
        for pair in pairs:
            query_rows = self.rows(pair["query"])
            resolved.append(ResolvedPair(
                pair=pair,
                query_row=self.query_row(pair["query"]),
                baseline_query_row=query_rows[0] if query_rows else None,
                target_rows=set(self.rows(pair["target"])),
            ))
        return resolved


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-9)


class ProductionRanker:
    """Scores the immutable production index under independent approaches."""

    def __init__(self, recommender, heldout: Set[str], seed: int = 20260711):
        self.rec = recommender
        self.titles = recommender.titles
        self.artists = recommender.artists
        self._heldout = heldout
        self._quality_filter = TitleQualityFilter()
        self._quality_mask = self._quality_filter.keep_mask(self.titles, self.artists)
        self._centroids: Optional[ArtistCentroidIndex] = None
        self._hubness: Optional[np.ndarray] = None
        self._seed = seed

    def _base_parts(self, row: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        neural = self.rec._neural @ self.rec._neural[row]
        query_vibe = self.rec._vscaled[row]
        vibe = 1.0 / (
            1.0 + np.linalg.norm(self.rec._vscaled - query_vibe, axis=1)
        )
        blend = self.rec.alpha * _zscore(neural) + (1 - self.rec.alpha) * _zscore(vibe)
        return neural, vibe, blend

    def fit_hubness(self, n_reference: int = 384, temperature: float = 8.0) -> Dict[str, Any]:
        """Fit an unsupervised inverted-softmax hub penalty.

        Reference rows exclude every held-out benchmark artist.  Only the
        resulting one-float-per-catalogue-row density vector is used at ranking.
        """
        if self._hubness is not None:
            return {"n_reference": n_reference, "temperature": temperature}
        eligible = np.asarray([
            i for i, artist in enumerate(self.artists)
            if primary_artist(str(artist)) not in self._heldout
        ], dtype=np.int64)
        rng = np.random.default_rng(self._seed)
        refs = rng.choice(eligible, size=min(n_reference, len(eligible)), replace=False)
        reference = self.rec._neural[refs]
        density = np.empty(len(self.titles), dtype=np.float32)
        chunk = 8192
        for start in range(0, len(density), chunk):
            stop = min(start + chunk, len(density))
            similarities = self.rec._neural[start:stop] @ reference.T
            scaled = temperature * similarities
            maximum = scaled.max(axis=1, keepdims=True)
            density[start:stop] = (
                (maximum[:, 0] + np.log(np.exp(scaled - maximum).mean(axis=1)))
                / temperature
            )
        self._hubness = density
        return {
            "n_reference": int(len(refs)),
            "temperature": temperature,
            "bytes": int(density.nbytes),
            "held_out_reference_rows": 0,
        }

    def scores(self, row: int, method: str, hub_beta: float = 0.5) -> np.ndarray:
        neural, _, blend = self._base_parts(row)
        if method in {
            "raw_encoder", "production_baseline", "quality_filter",
            "guarded_centroid",
        }:
            return neural if method == "raw_encoder" else blend
        if method == "artist_centroid":
            if self._centroids is None:
                self._centroids = ArtistCentroidIndex(
                    self.rec._neural, self.artists, min_songs=2
                )
            return self._centroids.blend_with_genre(
                blend, str(self.artists[row]), self.rec._neural[row], gamma=0.25
            )
        if method in {"hubness", "quality_hubness"}:
            self.fit_hubness()
            corrected = neural - float(hub_beta) * self._hubness
            return (
                self.rec.alpha * _zscore(corrected)
                + (1 - self.rec.alpha) * _zscore(self._base_parts(row)[1])
            )
        if method == "query_expansion":
            candidates = np.argpartition(-neural, min(20, len(neural) - 1))[:20]
            seed_artist = primary_artist(str(self.artists[row]))
            neighbors = [
                int(i) for i in candidates
                if int(i) != row
                and primary_artist(str(self.artists[i])) != seed_artist
            ]
            neighbors = sorted(neighbors, key=lambda i: float(neural[i]), reverse=True)[:3]
            query = self.rec._neural[[row] + neighbors].mean(axis=0)
            query /= np.linalg.norm(query) + 1e-9
            expanded = self.rec._neural @ query
            return (
                self.rec.alpha * _zscore(expanded)
                + (1 - self.rec.alpha) * _zscore(self._base_parts(row)[1])
            )
        raise ValueError(f"Unknown benchmark method: {method}")

    def rank(
        self,
        row: int,
        method: str,
        n: int = 50,
        hub_beta: float = 0.5,
    ) -> List[int]:
        if method == "guarded_centroid":
            # Re-rank only the first 20 already-retrieved production candidates.
            # The tail is frozen, so a known counterpart at rank 21-50 cannot be
            # lost while top-five scene coherence is improved.  Quality filtering
            # happens during candidate generation, before this guarded re-rank.
            baseline = self.rank(
                row, "quality_filter", n=max(n, 50), hub_beta=hub_beta
            )
            if self._centroids is None:
                self._centroids = ArtistCentroidIndex(
                    self.rec._neural, self.artists, min_songs=2
                )
            _, _, blend = self._base_parts(row)
            centroid_score = self._centroids.blend_with_genre(
                blend, str(self.artists[row]), self.rec._neural[row], gamma=0.25
            )
            boundary = min(20, len(baseline))
            head = sorted(
                baseline[:boundary],
                key=lambda candidate: float(centroid_score[candidate]),
                reverse=True,
            )
            return (head + baseline[boundary:])[:n]

        score = self.scores(row, method, hub_beta=hub_beta)
        order = np.argsort(score)[::-1]
        seed_artist = str(self.artists[row]).casefold()
        seed_title = str(self.titles[row])
        quality = method in {"quality_filter", "quality_hubness"}
        candidates: List[int] = []
        seen_recordings: Set[Tuple[str, str]] = set()
        seen_artists: Set[str] = set()
        pool_cap = max(n * 25, 500)
        for raw in order:
            candidate = int(raw)
            if candidate == row:
                continue
            artist = str(self.artists[candidate]).casefold()
            if seed_artist and seed_artist in artist:
                continue
            if quality and not bool(self._quality_mask[candidate]):
                continue
            if quality and self._quality_filter.seed_title_in_result(
                seed_title, str(self.titles[candidate])
            ):
                continue
            recording = (
                str(self.titles[candidate]).casefold(),
                artist,
            )
            if recording in seen_recordings or artist in seen_artists:
                continue
            seen_recordings.add(recording)
            seen_artists.add(artist)
            candidates.append(candidate)
            if len(candidates) >= pool_cap:
                break
        return self.rec._mmr(candidates, score, n, diversity=0.15)


def _rank_for_target(rows: Sequence[int], targets: Set[int]) -> int:
    for position, row in enumerate(rows, 1):
        if row in targets:
            return position
    return 0


def _method_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(records)
    ranks = np.asarray([int(record["target_rank"]) for record in records], dtype=int)
    reciprocal = np.asarray([1.0 / rank if rank else 0.0 for rank in ranks])
    metrics: Dict[str, Any] = {
        f"recall_at_{cutoff}": float(np.mean((ranks > 0) & (ranks <= cutoff)))
        for cutoff in RECALL_CUTOFFS
    }
    metrics["mrr"] = float(reciprocal.mean()) if count else 0.0
    metrics["ndcg_at_50"] = float(np.mean([
        1.0 / math.log2(rank + 1) if 0 < rank <= 50 else 0.0
        for rank in ranks
    ])) if count else 0.0
    metrics["primary_score"] = (
        0.5 * metrics["recall_at_50"] + 0.5 * metrics["mrr"]
    )
    metrics["reciprocal_rank_distribution"] = [round(float(x), 8) for x in reciprocal]
    metrics["query_missing_rate"] = float(np.mean([
        not record["query_found"] for record in records
    ])) if count else 0.0
    metrics["target_missing_rate"] = float(np.mean([
        not record["target_found"] for record in records
    ])) if count else 0.0
    metrics["missing_catalogue_rate"] = float(np.mean([
        not record["query_found"] or not record["target_found"] for record in records
    ])) if count else 0.0
    metrics["n_pairs"] = count
    metrics["n_rankable"] = int(sum(
        record["query_found"] and record["target_found"] for record in records
    ))
    return metrics


def _per_scene(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    scenes: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        scenes.setdefault(record["scene"], []).append(record)
    return {
        scene: _method_metrics(group)
        for scene, group in sorted(scenes.items())
    }


def evaluate_method(
    ranker: ProductionRanker,
    resolved: Sequence[ResolvedPair],
    method: str,
    hub_beta: float = 0.5,
) -> Dict[str, Any]:
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    latencies: List[float] = []
    for item in resolved:
        pair = item.pair
        query_row = (
            item.query_row
            if method == "guarded_centroid"
            else item.baseline_query_row
        )
        ranked: List[int] = []
        elapsed_ms = 0.0
        if query_row is not None:
            call_started = time.perf_counter()
            ranked = ranker.rank(
                query_row, method, n=50, hub_beta=hub_beta
            )
            elapsed_ms = (time.perf_counter() - call_started) * 1000
            latencies.append(elapsed_ms)
        target_rank = _rank_for_target(ranked, item.target_rows)
        records.append({
            "pair_id": pair["id"],
            "scene": pair["scene"],
            "query": dict(pair["query"]),
            "target": dict(pair["target"]),
            "query_found": query_row is not None,
            "query_catalogue": None if query_row is None else {
                "row": query_row,
                "title": str(ranker.titles[query_row]),
                "artist": str(ranker.artists[query_row]),
            },
            "target_found": bool(item.target_rows),
            "target_catalogue_rows": sorted(item.target_rows),
            "target_rank": target_rank,
            "reciprocal_rank": 1.0 / target_rank if target_rank else 0.0,
            "latency_ms": round(elapsed_ms, 3),
            "ranked_outputs": [
                {
                    "rank": position,
                    "row": row,
                    "title": str(ranker.titles[row]),
                    "artist": str(ranker.artists[row]),
                    "is_target": row in item.target_rows,
                }
                for position, row in enumerate(ranked, 1)
            ],
        })
    metrics = _method_metrics(records)
    return {
        "method": method,
        "parameters": {"hub_beta": hub_beta} if "hubness" in method else {},
        "metrics": metrics,
        "per_scene": _per_scene(records),
        "latency": {
            "mean_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "wall_seconds": time.perf_counter() - started,
        },
        "pairs": records,
    }


def pair_contributions(report: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([
        0.5 * float(0 < pair["target_rank"] <= PRIMARY_CUTOFF)
        + 0.5 * float(pair["reciprocal_rank"])
        for pair in report["pairs"]
    ], dtype=np.float64)


def contribution_by_pair(report: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for pair in report["pairs"]:
        pair_id = pair["pair_id"]
        if pair_id in result:
            raise ValueError(f"Duplicate pair id in report: {pair_id}")
        result[pair_id] = (
            0.5 * float(0 < pair["target_rank"] <= PRIMARY_CUTOFF)
            + 0.5 * float(pair["reciprocal_rank"])
        )
    return result


def judged_top5_fingerprint(
    report: Mapping[str, Any],
    method_names: Sequence[str] = ("production_baseline", "guarded_centroid"),
) -> str:
    """Stable digest binding human labels to exact seed rows and top-five rows."""
    payload = []
    for method_name in method_names:
        method = report["methods"][method_name]
        for pair in method["pairs"]:
            payload.append([
                method_name,
                pair["pair_id"],
                pair["query_catalogue"]["row"] if pair["query_catalogue"] else None,
                [item["row"] for item in pair["ranked_outputs"][:5]],
            ])
    encoded = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bootstrap_delta(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    iterations: int = 10_000,
    seed: int = 20260711,
) -> Dict[str, float]:
    base_map = contribution_by_pair(baseline)
    test_map = contribution_by_pair(challenger)
    if set(base_map) != set(test_map):
        raise ValueError("Paired bootstrap requires identical pair ids")
    pair_ids = sorted(base_map)
    base = np.asarray([base_map[pair_id] for pair_id in pair_ids])
    test = np.asarray([test_map[pair_id] for pair_id in pair_ids])
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = rng.integers(0, len(base), size=len(base))
        deltas[iteration] = (test[sample] - base[sample]).mean()
    absolute = float((test - base).mean())
    relative = absolute / (float(base.mean()) + 1e-12)
    return {
        "absolute_delta": absolute,
        "relative_gain": relative,
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "probability_positive": float(np.mean(deltas > 0)),
        "bootstrap_iterations": iterations,
    }


def compare_to_baseline(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> Dict[str, Any]:
    delta = bootstrap_delta(baseline, challenger)
    scene_regressions: Dict[str, float] = {}
    for scene, base_metrics in baseline["per_scene"].items():
        challenger_metrics = challenger["per_scene"].get(scene)
        if not challenger_metrics:
            continue
        base = float(base_metrics["primary_score"])
        current = float(challenger_metrics["primary_score"])
        scene_regressions[scene] = (current - base) / (base + 1e-12)
    delta["per_scene_relative_delta"] = scene_regressions
    delta["passes_20pct_gain"] = bool(delta["relative_gain"] >= 0.20)
    delta["passes_scene_guardrail"] = all(
        change >= -0.10 for change in scene_regressions.values()
    )
    return delta


def combine_ranked_list_judgments(
    report: Mapping[str, Any],
    judgments_path: Path,
    baseline_name: str = "production_baseline",
    winner_name: str = "guarded_centroid",
    iterations: int = 10_000,
) -> Dict[str, Any]:
    """Combine sourced-pair retrieval and explicit top-five judgments 50/50."""
    judgments = json.loads(Path(judgments_path).read_text(encoding="utf-8"))
    expected_fingerprint = judgments.get("ranked_top5_fingerprint")
    actual_fingerprint = judged_top5_fingerprint(report)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Human judgments do not match these exact ranked outputs: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )
    by_id = {item["pair_id"]: item for item in judgments["judgments"]}
    baseline = report["methods"][baseline_name]
    winner = report["methods"][winner_name]
    base_pairs = {item["pair_id"]: item for item in baseline["pairs"]}
    winner_pairs = {item["pair_id"]: item for item in winner["pairs"]}
    ids = sorted(base_pairs)
    if set(ids) != set(by_id) or set(ids) != set(winner_pairs):
        raise ValueError("Judgments and ranked pair outputs must have identical pair ids")

    base_pair_map = contribution_by_pair(baseline)
    winner_pair_map = contribution_by_pair(winner)
    base_pair = np.asarray([base_pair_map[pair_id] for pair_id in ids])
    winner_pair = np.asarray([winner_pair_map[pair_id] for pair_id in ids])
    base_direct = np.asarray(
        [float(by_id[pair_id]["baseline_pass"]) for pair_id in ids], dtype=np.float64
    )
    winner_direct = np.asarray(
        [float(by_id[pair_id]["winner_pass"]) for pair_id in ids], dtype=np.float64
    )
    base_combined = 0.5 * base_pair + 0.5 * base_direct
    winner_combined = 0.5 * winner_pair + 0.5 * winner_direct

    rng = np.random.default_rng(20260711)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = rng.integers(0, len(ids), size=len(ids))
        deltas[iteration] = (
            winner_combined[sample] - base_combined[sample]
        ).mean()
    base_score = float(base_combined.mean())
    winner_score = float(winner_combined.mean())
    per_scene: Dict[str, List[float]] = {}
    for position, pair_id in enumerate(ids):
        scene = base_pairs[pair_id]["scene"]
        per_scene.setdefault(scene, []).append(
            float(winner_combined[position] - base_combined[position])
        )
    scene_delta = {
        scene: float(np.mean(values)) for scene, values in per_scene.items()
    }
    return {
        "definition": (
            "mean per seed of 0.5 * sourced-pair contribution "
            "(0.5*hit@50 + 0.5*reciprocal-rank) + 0.5 * direct-top5-pass"
        ),
        "baseline": baseline_name,
        "winner": winner_name,
        "baseline_pair_primary": float(base_pair.mean()),
        "winner_pair_primary": float(winner_pair.mean()),
        "baseline_direct_pass_rate": float(base_direct.mean()),
        "winner_direct_pass_rate": float(winner_direct.mean()),
        "baseline_primary": base_score,
        "winner_primary": winner_score,
        "absolute_gain": winner_score - base_score,
        "relative_gain": (winner_score - base_score) / (base_score + 1e-12),
        "ci95_absolute_low": float(np.percentile(deltas, 2.5)),
        "ci95_absolute_high": float(np.percentile(deltas, 97.5)),
        "probability_positive": float(np.mean(deltas > 0)),
        "per_scene_absolute_delta": scene_delta,
        "passes_20pct_gain": bool(
            (winner_score - base_score) / (base_score + 1e-12) >= 0.20
        ),
        "passes_scene_guardrail": all(delta >= -0.10 for delta in scene_delta.values()),
        "held_out_seeds_passing": int(winner_direct.sum()),
        "held_out_seed_count": len(ids),
        "passes_80pct_direct": bool(winner_direct.mean() >= 0.80),
        "judgments_path": str(judgments_path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def run_benchmark(
    index_path: Path,
    benchmark_path: Path,
    methods: Sequence[str],
    split: str = "held_out",
    hub_beta: float = 0.5,
) -> Dict[str, Any]:
    # Importing here keeps unit tests independent of the hosted module.
    from webapp.api._reco import WebRecommender
    from .related_artists_rerank import MANUAL_PAIRS

    benchmark = load_benchmark(benchmark_path)
    leakage = audit_leakage(benchmark, manual_pairs=MANUAL_PAIRS)
    if not leakage["passed"]:
        raise RuntimeError(f"Benchmark leakage audit failed: {leakage}")

    load_started = time.perf_counter()
    recommender = WebRecommender(str(index_path), enhance=False)
    load_seconds = time.perf_counter() - load_started
    rss_after_index = _rss_bytes()
    resolver = PairResolver(recommender.titles, recommender.artists)
    selected = [pair for pair in benchmark["pairs"] if pair["split"] == split]
    resolved = resolver.resolve(selected)
    # Seed exactly the row selected by the deployed query resolver.  Several
    # catalogue titles have multiple remaster/version rows; choosing a cleaner
    # version for evaluation would not freeze the actual product behavior.
    for item in resolved:
        item.query_row = recommender.find_row(
            item.pair["query"]["title"], item.pair["query"]["artist"]
        )
    rss_before_reranker = _rss_bytes()
    ranker = ProductionRanker(recommender, held_out_artists(benchmark))
    rss_after_quality = _rss_bytes()

    reports = {
        method: evaluate_method(ranker, resolved, method, hub_beta=hub_beta)
        for method in methods
    }
    rss_after_methods = _rss_bytes()
    baseline = reports.get("production_baseline")
    comparisons = {}
    if baseline:
        comparisons = {
            method: compare_to_baseline(baseline, report)
            for method, report in reports.items()
            if method != "production_baseline"
        }
    neural_bytes = int(recommender._neural.nbytes)
    vibe_bytes = int(recommender._vscaled.nbytes)
    centroid_song_map_bytes = (
        int(ranker._centroids._song_centroid.nbytes)
        if ranker._centroids is not None else 0
    )
    centroid_table_bytes = (
        int(ranker._centroids._centroid_matrix.nbytes)
        if ranker._centroids is not None else 0
    )
    return {
        "schema_version": 1,
        "created_at": "2026-07-11",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_version": benchmark["benchmark_version"],
        "split": split,
        "index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
            "tracks": len(recommender),
            "neural_dimension": int(recommender._neural.shape[1]),
            "file_bytes": int(Path(index_path).stat().st_size),
        },
        "leakage_audit": leakage,
        "resources": {
            "cold_load_seconds": load_seconds,
            "neural_matrix_bytes": neural_bytes,
            "vibe_matrix_bytes": vibe_bytes,
            "minimum_ranker_array_bytes": neural_bytes + vibe_bytes,
            "quality_mask_bytes": int(ranker._quality_mask.nbytes),
            "centroid_count": (
                int(ranker._centroids.n_centroids)
                if ranker._centroids is not None else 0
            ),
            "centroid_song_map_bytes": centroid_song_map_bytes,
            "centroid_table_bytes": centroid_table_bytes,
            "reranker_bytes": (
                int(ranker._quality_mask.nbytes)
                + centroid_song_map_bytes
                + centroid_table_bytes
            ),
            "rss_after_index_bytes": rss_after_index,
            "rss_before_reranker_bytes": rss_before_reranker,
            "rss_after_quality_filter_bytes": rss_after_quality,
            "rss_after_methods_bytes": rss_after_methods,
            "measured_reranker_rss_delta_bytes": (
                max(0, rss_after_methods - rss_before_reranker)
                if rss_after_methods is not None and rss_before_reranker is not None
                else None
            ),
            "hubness_bytes": int(ranker._hubness.nbytes) if ranker._hubness is not None else 0,
        },
        "methods": reports,
        "comparisons_to_production_baseline": comparisons,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/soundalike_pairs.v1.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "held_out"), default="held_out")
    parser.add_argument(
        "--methods",
        default=(
            "raw_encoder,production_baseline,quality_filter,artist_centroid,"
            "hubness,query_expansion,quality_hubness"
        ),
    )
    parser.add_argument("--hub-beta", type=float, default=0.5)
    parser.add_argument("--judgments", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.index,
        args.benchmark,
        [part.strip() for part in args.methods.split(",") if part.strip()],
        split=args.split,
        hub_beta=args.hub_beta,
    )
    if args.judgments:
        report["human_aligned_primary"] = combine_ranked_list_judgments(
            report, args.judgments
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {args.split} benchmark: {args.out}")
    for method, result in report["methods"].items():
        metrics = result["metrics"]
        print(
            f"{method:>20}: primary={metrics['primary_score']:.4f} "
            f"R@10={metrics['recall_at_10']:.3f} "
            f"R@50={metrics['recall_at_50']:.3f} MRR={metrics['mrr']:.4f} "
            f"missing={metrics['missing_catalogue_rate']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
