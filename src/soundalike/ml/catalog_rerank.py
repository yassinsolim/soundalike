"""DEV-select and lock catalogue-wide hybrid candidate retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .catalog_graph import CatalogArtistGraph
from .catalog_protocol import (
    _aggregate,
    _bootstrap,
    _graded_rows,
    _per_seed,
)
from .collaborative import CollaborativeIndex
from .final_protocol import content_sha256, file_sha256
from .quality_filter import TitleQualityFilter
from .real_benchmark import PairResolver, ProductionRanker, normalize_text

FEATURE_NAMES = (
    "audio_blend",
    "sonic_cosine",
    "clap_cosine",
    "vibe_cosine",
    "catalog_graph_strength",
    "catalog_graph_reciprocal_rank",
    "music4all_strength",
    "music4all_reciprocal_rank",
    "source_agreement",
    "scene_consistency",
    "global_popularity_zero",
)


@dataclass(frozen=True)
class CandidateConfig:
    audio_candidates: int = 1000
    sparse_candidates: int = 1000
    catalog_candidates: int = 1000
    union_candidates: int = 1800
    max_candidates_per_artist: int = 16
    final_max_per_artist: int = 1
    bridge_anchors: int = 8
    scene_guard_positions: int = 3
    scene_guard_quantile: float = 0.35


@dataclass(frozen=True)
class HybridScorer:
    coefficients: Tuple[float, ...]
    scene_guard: bool = True

    def score(self, features: np.ndarray) -> np.ndarray:
        return features @ np.asarray(self.coefficients, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "coefficients": list(self.coefficients),
            "scene_guard": self.scene_guard,
            "global_popularity_weight": 0.0,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HybridScorer":
        if tuple(value["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Scorer feature schema mismatch")
        return cls(
            tuple(float(item) for item in value["coefficients"]),
            bool(value.get("scene_guard", True)),
        )


@dataclass
class QueryContext:
    query_row: int
    query_mode: str
    audio_rows: np.ndarray
    sparse_rows: np.ndarray
    catalog_rows: np.ndarray
    union_rows: np.ndarray
    features: np.ndarray


class CatalogHybridRanker:
    """Audio + Music4All + catalogue graph retrieval without popularity."""

    def __init__(
        self,
        recommender: Any,
        sparse: CollaborativeIndex,
        catalog: CatalogArtistGraph,
        scorer: HybridScorer | None = None,
        config: CandidateConfig | None = None,
    ):
        self.rec = recommender
        self.sparse = sparse
        self.catalog = catalog
        self.scorer = scorer
        self.config = config or CandidateConfig()
        self.quality = TitleQualityFilter()
        self.quality_mask = self.quality.keep_mask(
            recommender.titles, recommender.artists
        )
        self.production = ProductionRanker(recommender, heldout=set())
        raw_vibe = (
            np.asarray(recommender._vscaled, dtype=np.float32)
            / (np.asarray(recommender._w, dtype=np.float32) + 1e-9)
        ) * np.asarray(recommender._vstd, dtype=np.float32) + np.asarray(
            recommender._vmean, dtype=np.float32
        )
        self._vibe_unit = np.array(raw_vibe, dtype=np.float32, copy=True)
        self._vibe_unit /= np.linalg.norm(
            self._vibe_unit, axis=1, keepdims=True
        ).clip(min=1e-8)

    @staticmethod
    def _z(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - values.mean()) / (values.std() + 1e-8)

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return value / max(float(np.linalg.norm(value)), 1e-8)

    def _audio(
        self, row: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sonic = self.rec._compact_cosine(
            self.rec._sonic, np.asarray(self.rec._sonic[row], dtype=np.float32)
        )
        clap = self.rec._compact_cosine(
            self.rec._clap, np.asarray(self.rec._clap[row], dtype=np.float32)
        )
        query_vibe = self._vibe_unit[row]
        vibe = self._vibe_unit @ query_vibe
        audio = 0.20 * self._z(sonic) + 0.65 * self._z(clap) + 0.15 * self._z(vibe)
        graph_query = self._unit(
            np.concatenate(
                (
                    self._unit(self.rec._sonic[row]),
                    self._unit(self.rec._clap[row]),
                    query_vibe,
                )
            )
        )
        return audio, sonic, clap, vibe, graph_query

    def _filter(
        self,
        query_row: int,
        rows: Iterable[int],
        limit: int,
        max_per_artist: int,
    ) -> np.ndarray:
        seed_artist = normalize_text(str(self.rec.artists[query_row]))
        seed_title = str(self.rec.titles[query_row])
        seen_tracks: Set[Tuple[str, str]] = set()
        artist_counts: Dict[str, int] = {}
        result = []
        for raw in rows:
            row = int(raw)
            artist = normalize_text(str(self.rec.artists[row]))
            title = str(self.rec.titles[row])
            track = (normalize_text(title), artist)
            if (
                row == query_row
                or artist == seed_artist
                or not self.quality_mask[row]
                or self.quality.seed_title_in_result(seed_title, title)
                or track in seen_tracks
                or artist_counts.get(artist, 0) >= max_per_artist
            ):
                continue
            result.append(row)
            seen_tracks.add(track)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            if len(result) >= limit:
                break
        return np.asarray(result, dtype=np.int32)

    @staticmethod
    def _round_robin(
        pools: Sequence[np.ndarray], limit: int
    ) -> np.ndarray:
        result: List[int] = []
        seen: Set[int] = set()
        depth = 0
        while len(result) < limit:
            added = False
            for pool in pools:
                if depth >= len(pool):
                    continue
                row = int(pool[depth])
                if row not in seen:
                    result.append(row)
                    seen.add(row)
                    added = True
                    if len(result) >= limit:
                        break
            if not added and all(depth >= len(pool) for pool in pools):
                break
            depth += 1
        return np.asarray(result, dtype=np.int32)

    @staticmethod
    def _ranks(rows: np.ndarray) -> Dict[int, int]:
        return {int(row): rank for rank, row in enumerate(rows, 1)}

    def context(self, row: int, variant: str = "twohop") -> QueryContext:
        audio, sonic, clap, vibe, graph_query = self._audio(row)
        audio_rows = self._filter(
            row,
            np.argsort(audio)[::-1],
            self.config.audio_candidates,
            self.config.max_candidates_per_artist,
        )
        sparse_raw, sparse_score, sparse_mode = self.sparse.candidates(
            row,
            str(self.rec.artists[row]),
            audio_scores=audio,
            n=self.config.sparse_candidates * 2,
        )
        sparse_lookup = {
            int(candidate): float(score)
            for candidate, score in zip(sparse_raw, sparse_score)
        }
        sparse_rows = self._filter(
            row,
            sparse_raw,
            self.config.sparse_candidates,
            self.config.max_candidates_per_artist,
        )
        catalog_raw, catalog_score, catalog_mode = self.catalog.candidates(
            row,
            str(self.rec.artists[row]),
            graph_query,
            audio,
            n=self.config.catalog_candidates * 2,
            variant=variant,
            max_tracks_per_artist=self.config.max_candidates_per_artist,
        )
        catalog_lookup = {
            int(candidate): float(score)
            for candidate, score in zip(catalog_raw, catalog_score)
        }
        catalog_rows = self._filter(
            row,
            catalog_raw,
            self.config.catalog_candidates,
            self.config.max_candidates_per_artist,
        )
        union = self._round_robin(
            (catalog_rows, sparse_rows, audio_rows),
            self.config.union_candidates,
        )
        audio_rank = self._ranks(audio_rows)
        sparse_rank = self._ranks(sparse_rows)
        catalog_rank = self._ranks(catalog_rows)
        features = np.zeros((len(union), len(FEATURE_NAMES)), dtype=np.float32)
        for position, candidate in enumerate(union):
            candidate = int(candidate)
            arank = audio_rank.get(candidate, 0)
            srank = sparse_rank.get(candidate, 0)
            crank = catalog_rank.get(candidate, 0)
            catalog_value = catalog_lookup.get(candidate, 0.0)
            sparse_value = sparse_lookup.get(candidate, 0.0)
            scene = 0.5 * float(vibe[candidate]) + 0.5 * float(
                clap[candidate]
            )
            features[position] = (
                float(audio[candidate]),
                float(sonic[candidate]),
                float(clap[candidate]),
                float(vibe[candidate]),
                catalog_value,
                1.0 / crank if crank else 0.0,
                sparse_value,
                1.0 / srank if srank else 0.0,
                float(bool(crank and srank)),
                scene,
                0.0,
            )
        query_mode = (
            f"catalog={catalog_mode};music4all={sparse_mode}"
        )
        return QueryContext(
            query_row=row,
            query_mode=query_mode,
            audio_rows=audio_rows,
            sparse_rows=sparse_rows,
            catalog_rows=catalog_rows,
            union_rows=union,
            features=features,
        )

    def _cap(self, query_row: int, rows: Iterable[int], n: int) -> List[int]:
        return self._filter(
            query_row, rows, n, self.config.final_max_per_artist
        ).tolist()

    def _scene_guard(
        self,
        context: QueryContext,
        ordered: np.ndarray,
    ) -> np.ndarray:
        if self.scorer is None or not self.scorer.scene_guard or not len(ordered):
            return ordered
        feature_positions = {
            int(row): position for position, row in enumerate(context.union_rows)
        }
        scene_values = context.features[:, 9]
        threshold = float(
            np.quantile(scene_values, self.config.scene_guard_quantile)
        )
        safe = [
            int(row)
            for row in ordered
            if context.features[feature_positions[int(row)], 9] >= threshold
        ]
        guarded = safe[: self.config.scene_guard_positions]
        used = set(guarded)
        guarded.extend(int(row) for row in ordered if int(row) not in used)
        return np.asarray(guarded, dtype=np.int32)

    def rank_context(
        self,
        context: QueryContext,
        method: str,
        n: int = 100,
    ) -> List[int]:
        if method == "audio_only":
            return self._cap(context.query_row, context.audio_rows, n)
        if method == "music4all_sparse":
            return self._cap(context.query_row, context.sparse_rows, n)
        if method == "catalog_graph":
            return self._cap(context.query_row, context.catalog_rows, n)
        if method == "production":
            return self.production.rank(
                context.query_row, "production_baseline", n=n
            )
        if method != "hybrid":
            raise ValueError(f"Unknown method: {method}")
        if self.scorer is None:
            raise ValueError("Hybrid ranking requires a DEV-selected scorer")
        scores = self.scorer.score(context.features)
        ordered = context.union_rows[np.argsort(scores)[::-1]]
        ordered = self._scene_guard(context, ordered)
        return self._cap(context.query_row, ordered, n)


def _serialise(rec: Any, rows: Sequence[int]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": rank,
            "row": int(row),
            "track_id": int(rec.track_ids[int(row)]),
            "title": str(rec.titles[int(row)]),
            "artist": str(rec.artists[int(row)]),
        }
        for rank, row in enumerate(rows, 1)
    ]


def _candidate_recall(
    contexts: Sequence[
        Tuple[Mapping[str, Any], QueryContext, Dict[int, Tuple[str, int]]]
    ],
    field: str,
) -> Dict[str, float]:
    result = {}
    for cutoff in (100, 500, 1000):
        values = []
        for _, context, relevance in contexts:
            groups = {group for group, _ in relevance.values()}
            found = {
                relevance[int(row)][0]
                for row in getattr(context, field)[:cutoff]
                if int(row) in relevance
            }
            values.append(len(found) / max(len(groups), 1))
        result[f"recall_at_{cutoff}"] = float(np.mean(values))
    return result


def _scorers() -> Iterable[HybridScorer]:
    """Predeclared interpretable grid; static popularity is always zero."""
    for audio in (0.15, 0.30, 0.50):
        for graph_strength in (0.50, 1.00, 1.50):
            for graph_rank in (0.50, 1.00, 2.00):
                for sparse in (0.20, 0.50, 1.00, 2.00):
                    for sparse_rank in (0.50, 1.50, 4.00):
                        for scene in (0.20, 0.50, 0.80):
                            values = np.zeros(
                                len(FEATURE_NAMES), dtype=np.float32
                            )
                            values[0] = audio
                            values[4] = graph_strength
                            values[5] = graph_rank
                            values[6] = sparse
                            values[7] = sparse_rank
                            values[8] = 0.25
                            values[9] = scene
                            yield HybridScorer(
                                tuple(map(float, values)), True
                            )


def _split(record_id: str) -> str:
    return (
        "fit"
        if hashlib.sha256(record_id.encode("utf-8")).digest()[0] < 180
        else "selection"
    )


def _ranking_metrics(
    ranker: CatalogHybridRanker,
    contexts: Sequence[
        Tuple[Mapping[str, Any], QueryContext, Dict[int, Tuple[str, int]]]
    ],
    method: str,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    values = []
    for _, context, grades in contexts:
        ranking = _serialise(
            ranker.rec, ranker.rank_context(context, method, n=100)
        )
        values.append(_per_seed(ranking, grades))
    return _aggregate(values), values


def train_and_evaluate_dev(
    index_path: Path,
    benchmark_path: Path,
    sparse_path: Path,
    catalog_path: Path,
    scorer_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Gate candidate recall first, then select the reranker only on DEV."""
    from webapp.api._reco import WebRecommender

    started = time.perf_counter()
    rec = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(rec.titles, rec.artists)
    sparse = CollaborativeIndex(sparse_path, len(rec))
    catalog = CatalogArtistGraph(catalog_path)
    ranker = CatalogHybridRanker(rec, sparse, catalog)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in benchmark["records"]
        if record["split"] == "development"
    ]
    contexts = []
    for record in records:
        query_row = resolver.query_row(record["query"])
        grades = _graded_rows(resolver, record)
        if query_row is None or not grades:
            continue
        contexts.append((record, ranker.context(query_row), grades))
    candidate_recall = {
        "audio_only": _candidate_recall(contexts, "audio_rows"),
        "music4all_sparse": _candidate_recall(contexts, "sparse_rows"),
        "catalog_wide_graph": _candidate_recall(contexts, "catalog_rows"),
        "hybrid_union": _candidate_recall(contexts, "union_rows"),
    }
    audio_at_1000 = candidate_recall["audio_only"]["recall_at_1000"]
    hybrid_at_1000 = candidate_recall["hybrid_union"]["recall_at_1000"]
    candidate_gate = {
        "minimum_absolute_lift_at_1000": 0.10,
        "absolute_lift_at_1000": hybrid_at_1000 - audio_at_1000,
        "passes": hybrid_at_1000 - audio_at_1000 >= 0.10,
    }
    if not candidate_gate["passes"]:
        raise RuntimeError(
            "Candidate recall gate failed; reranker selection is prohibited"
        )
    selection = [item for item in contexts if _split(item[0]["id"]) == "selection"]
    candidates = []
    for scorer in _scorers():
        ranker.scorer = scorer
        metrics, _ = _ranking_metrics(ranker, selection, "hybrid")
        candidates.append((metrics, scorer))
    chosen_metrics, chosen = max(
        candidates,
        key=lambda item: (
            item[0]["primary"],
            item[0]["mrr_at_10"],
            item[0]["recall_at_10"],
        ),
    )
    ranker.scorer = chosen
    methods = (
        "production",
        "audio_only",
        "music4all_sparse",
        "catalog_graph",
        "hybrid",
    )
    method_metrics = {}
    method_values = {}
    for method in methods:
        method_metrics[method], method_values[method] = _ranking_metrics(
            ranker, contexts, method
        )
    comparisons = {
        method: _bootstrap(
            [value["ndcg_at_10"] for value in method_values["production"]],
            [value["ndcg_at_10"] for value in method_values[method]],
        )
        for method in methods
        if method != "production"
    }
    query_modes: Dict[str, int] = {}
    for _, context, _ in contexts:
        query_modes[context.query_mode] = query_modes.get(
            context.query_mode, 0
        ) + 1
    scorer_doc = {
        "schema_version": 1,
        "method": "dev-selected-catalog-hybrid",
        "created_at": time.time(),
        "scorer": chosen.to_dict(),
        "candidate_configuration": asdict(ranker.config),
        "selection": {
            "records": len(selection),
            "grid_candidates": len(candidates),
            "metric": "graded_nDCG@10",
            "selected_metrics": chosen_metrics,
            "final_labels_compared": False,
        },
    }
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(
        json.dumps(scorer_doc, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "split": "development-only",
        "final_labels_compared": False,
        "benchmark_sha256": file_sha256(benchmark_path),
        "index_sha256": file_sha256(index_path),
        "records": len(contexts),
        "candidate_recall": candidate_recall,
        "candidate_recall_gate": candidate_gate,
        "metrics": method_metrics,
        "comparisons_to_production": comparisons,
        "ablations": {
            "graph_only": method_metrics["catalog_graph"],
            "audio_only": method_metrics["audio_only"],
            "hybrid": method_metrics["hybrid"],
            "music4all_sparse": method_metrics["music4all_sparse"],
            "static_popularity_weight": 0.0,
        },
        "query_modes": query_modes,
        "selected_scorer": chosen.to_dict(),
        "scene_guard": {
            "positions": 3,
            "rule": "candidates below DEV-fixed scene-consistency quantile 0.35 "
            "cannot occupy positions 1-3",
        },
        "wall_seconds": time.perf_counter() - started,
    }
    report["content_sha256"] = content_sha256(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_locked_rankings(
    index_path: Path,
    protocol_manifest_path: Path,
    method_manifest_path: Path,
    sparse_path: Path,
    catalog_path: Path,
    scorer_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate FINAL rankings without resolving or comparing positives."""
    from webapp.api._reco import WebRecommender

    rec = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(rec.titles, rec.artists)
    scorer_doc = json.loads(scorer_path.read_text(encoding="utf-8"))
    ranker = CatalogHybridRanker(
        rec,
        CollaborativeIndex(sparse_path, len(rec)),
        CatalogArtistGraph(catalog_path),
        scorer=HybridScorer.from_dict(scorer_doc["scorer"]),
    )
    manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["records"]:
        query_row = resolver.query_row(record["query"])
        if query_row is None:
            raise RuntimeError(f"Missing FINAL query: {record['id']}")
        contexts = {
            variant: ranker.context(query_row, variant=variant)
            for variant in ("full", "direct", "twohop")
        }
        deciding = contexts["twohop"]
        controls = {
            "audio_only": ranker.rank_context(
                deciding, "audio_only", n=100
            ),
            "music4all_sparse": ranker.rank_context(
                deciding, "music4all_sparse", n=100
            ),
            "catalog_graph_full": ranker.rank_context(
                contexts["full"], "catalog_graph", n=100
            ),
            "catalog_graph_direct_masked": ranker.rank_context(
                contexts["direct"], "catalog_graph", n=100
            ),
            "catalog_graph_twohop_masked": ranker.rank_context(
                deciding, "catalog_graph", n=100
            ),
            "hybrid_full_graph_ablation": ranker.rank_context(
                contexts["full"], "hybrid", n=100
            ),
            "hybrid_direct_masked_ablation": ranker.rank_context(
                contexts["direct"], "hybrid", n=100
            ),
            "hybrid_twohop_masked": ranker.rank_context(
                deciding, "hybrid", n=100
            ),
        }
        candidate_sets = {
            "audio_only": deciding.audio_rows[:1000].tolist(),
            "music4all_sparse": deciding.sparse_rows[:1000].tolist(),
            "catalog_graph_full": contexts["full"].catalog_rows[:1000].tolist(),
            "catalog_graph_direct_masked": (
                contexts["direct"].catalog_rows[:1000].tolist()
            ),
            "catalog_graph_twohop_masked": (
                deciding.catalog_rows[:1000].tolist()
            ),
            "hybrid_union_twohop_masked": deciding.union_rows[:1000].tolist(),
        }
        records.append(
            {
                "record_id": record["id"],
                "query": dict(record["query"]),
                "query_row": int(query_row),
                "query_mode": deciding.query_mode,
                "ranking": _serialise(
                    rec, controls["hybrid_twohop_masked"]
                ),
                "diagnostic_rankings": {
                    method: _serialise(rec, rows)
                    for method, rows in controls.items()
                },
                "candidate_sets": candidate_sets,
            }
        )
    result = {
        "schema_version": 7,
        "created_at": time.time(),
        "target_labels_compared": False,
        "method_manifest_sha256": file_sha256(method_manifest_path),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("dev")
    for command in (dev,):
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--benchmark", type=Path, required=True)
        command.add_argument("--sparse", type=Path, required=True)
        command.add_argument("--catalog", type=Path, required=True)
    dev.add_argument("--scorer", type=Path, required=True)
    dev.add_argument("--report", type=Path, required=True)
    locked = commands.add_parser("locked-rankings")
    locked.add_argument("--index", type=Path, required=True)
    locked.add_argument("--manifest", type=Path, required=True)
    locked.add_argument("--method-manifest", type=Path, required=True)
    locked.add_argument("--sparse", type=Path, required=True)
    locked.add_argument("--catalog", type=Path, required=True)
    locked.add_argument("--scorer", type=Path, required=True)
    locked.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "dev":
        result = train_and_evaluate_dev(
            args.index,
            args.benchmark,
            args.sparse,
            args.catalog,
            args.scorer,
            args.report,
        )
        print(
            json.dumps(
                {
                    "candidate_recall": result["candidate_recall"],
                    "candidate_gate": result["candidate_recall_gate"],
                    "metrics": result["metrics"],
                },
                indent=2,
            )
        )
    else:
        result = build_locked_rankings(
            args.index,
            args.manifest,
            args.method_manifest,
            args.sparse,
            args.catalog,
            args.scorer,
            args.output,
        )
        print(
            json.dumps(
                {
                    "records": len(result["records"]),
                    "target_labels_compared": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
