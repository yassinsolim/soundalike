"""DEV-only training and evaluation for collaborative candidate retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .collaborative import CollaborativeIndex
from .final_protocol import _bootstrap, _metrics, content_sha256
from .quality_filter import TitleQualityFilter
from .real_benchmark import PairResolver, ProductionRanker, normalize_text

FEATURE_NAMES = (
    "audio_blend",
    "efficientnet_cosine",
    "clap_cosine",
    "collaborative_cosine",
    "production_reciprocal_rank",
    "audio_reciprocal_rank",
    "collaborative_reciprocal_rank",
    "production_audio_blend",
    "audio_x_collaborative",
    "global_notability_zero",
)


@dataclass(frozen=True)
class CandidateConfig:
    audio_candidates: int = 1000
    collaborative_candidates: int = 1000
    production_candidates: int = 250
    union_candidates: int = 1500
    max_candidates_per_artist: int = 3
    final_max_per_artist: int = 1
    bridge_size: int = 24


@dataclass(frozen=True)
class LinearScorer:
    feature_names: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    intercept: float
    c: float

    def score(self, features: np.ndarray) -> np.ndarray:
        return features @ np.asarray(self.coefficients, dtype=np.float32) + self.intercept

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "c": self.c,
            "global_notability_weight": 0.0,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinearScorer":
        return cls(
            feature_names=tuple(value["feature_names"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            c=float(value.get("c", 1.0)),
        )


@dataclass
class QueryContext:
    query_row: int
    query_mode: str
    audio_rows: np.ndarray
    collaborative_rows: np.ndarray
    production_rows: np.ndarray
    union_rows: np.ndarray
    features: np.ndarray


class CollaborativeHybridRanker:
    """Union audio, collaborative, and production candidates, then rerank."""

    def __init__(
        self,
        recommender: Any,
        collaborative: CollaborativeIndex,
        scorer: LinearScorer | None = None,
        config: CandidateConfig | None = None,
    ):
        self.rec = recommender
        self.collaborative = collaborative
        self.scorer = scorer
        self.config = config or CandidateConfig()
        self.production = ProductionRanker(recommender, heldout=set())
        self.quality = TitleQualityFilter()
        self.quality_mask = self.quality.keep_mask(
            recommender.titles, recommender.artists
        )

    @staticmethod
    def _z(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - float(values.mean())) / (float(values.std()) + 1e-8)

    def _audio_scores(
        self, row: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        efficientnet = self.rec._compact_cosine(
            self.rec._sonic, np.asarray(self.rec._sonic[row], dtype=np.float32)
        )
        clap = self.rec._compact_cosine(
            self.rec._clap, np.asarray(self.rec._clap[row], dtype=np.float32)
        )
        audio = 0.25 * self._z(efficientnet) + 0.75 * self._z(clap)
        return audio, efficientnet, clap

    def _filter_rows(
        self,
        query_row: int,
        ordered: Iterable[int],
        limit: int,
        max_per_artist: int,
    ) -> np.ndarray:
        seed_artist = normalize_text(str(self.rec.artists[query_row]))
        seed_title = str(self.rec.titles[query_row])
        artist_counts: Dict[str, int] = {}
        tracks: Set[Tuple[str, str]] = set()
        selected = []
        for raw in ordered:
            row = int(raw)
            if row == query_row or not bool(self.quality_mask[row]):
                continue
            artist = normalize_text(str(self.rec.artists[row]))
            title = str(self.rec.titles[row])
            if artist == seed_artist:
                continue
            if self.quality.seed_title_in_result(seed_title, title):
                continue
            track = (normalize_text(title), artist)
            if track in tracks or artist_counts.get(artist, 0) >= max_per_artist:
                continue
            tracks.add(track)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            selected.append(row)
            if len(selected) >= limit:
                break
        return np.asarray(selected, dtype=np.int32)

    @staticmethod
    def _round_robin(
        pools: Sequence[np.ndarray],
        limit: int,
    ) -> np.ndarray:
        selected: List[int] = []
        seen: Set[int] = set()
        depth = 0
        while len(selected) < limit:
            added = False
            for pool in pools:
                if depth >= len(pool):
                    continue
                row = int(pool[depth])
                if row not in seen:
                    selected.append(row)
                    seen.add(row)
                    added = True
                    if len(selected) >= limit:
                        break
            if not added and all(depth >= len(pool) for pool in pools):
                break
            depth += 1
        return np.asarray(selected, dtype=np.int32)

    @staticmethod
    def _rank_lookup(rows: np.ndarray) -> Dict[int, int]:
        return {int(row): rank for rank, row in enumerate(rows, 1)}

    def context(self, row: int) -> QueryContext:
        audio_score, efficientnet, clap = self._audio_scores(row)
        raw_audio = np.argsort(audio_score)[::-1]
        audio_rows = self._filter_rows(
            row,
            raw_audio,
            self.config.audio_candidates,
            self.config.max_candidates_per_artist,
        )
        collab_raw, _, query_mode = self.collaborative.candidates(
            row,
            str(self.rec.artists[row]),
            audio_scores=audio_score,
            n=self.config.collaborative_candidates * 2,
        )
        collaborative_rows = self._filter_rows(
            row,
            collab_raw,
            self.config.collaborative_candidates,
            self.config.max_candidates_per_artist,
        )
        production_raw = self.production.rank(
            row, "dual_sonic", n=self.config.production_candidates
        )
        production_rows = self._filter_rows(
            row,
            production_raw,
            self.config.production_candidates,
            self.config.max_candidates_per_artist,
        )
        union_rows = self._round_robin(
            (collaborative_rows, audio_rows, production_rows),
            self.config.union_candidates,
        )
        production_score = self.production.scores(row, "production_baseline")
        collab_vector, _ = self.collaborative.query_vector(
            row,
            str(self.rec.artists[row]),
            audio_scores=audio_score,
            bridge_size=self.config.bridge_size,
        )
        collaborative_score = np.zeros(len(self.rec), dtype=np.float32)
        if collab_vector is not None:
            collaborative_score[self.collaborative.rows] = (
                self.collaborative.vectors @ collab_vector
            )
        audio_rank = self._rank_lookup(audio_rows)
        collab_rank = self._rank_lookup(collaborative_rows)
        production_rank = self._rank_lookup(production_rows)
        features = np.empty((len(union_rows), len(FEATURE_NAMES)), dtype=np.float32)
        for position, candidate in enumerate(union_rows):
            candidate = int(candidate)
            arank = audio_rank.get(candidate, 0)
            crank = collab_rank.get(candidate, 0)
            prank = production_rank.get(candidate, 0)
            collab_value = float(collaborative_score[candidate])
            audio_value = float(audio_score[candidate])
            features[position] = (
                audio_value,
                float(efficientnet[candidate]),
                float(clap[candidate]),
                collab_value,
                1.0 / prank if prank else 0.0,
                1.0 / arank if arank else 0.0,
                1.0 / crank if crank else 0.0,
                float(production_score[candidate]),
                audio_value * collab_value,
                0.0,
            )
        return QueryContext(
            query_row=row,
            query_mode=query_mode,
            audio_rows=audio_rows,
            collaborative_rows=collaborative_rows,
            production_rows=production_rows,
            union_rows=union_rows,
            features=features,
        )

    def _final_cap(self, query_row: int, rows: Iterable[int], n: int) -> List[int]:
        return self._filter_rows(
            query_row,
            rows,
            n,
            self.config.final_max_per_artist,
        ).tolist()

    def rank_context(
        self,
        context: QueryContext,
        method: str,
        n: int = 1000,
    ) -> List[int]:
        if method == "audio_only":
            return self._final_cap(context.query_row, context.audio_rows, n)
        if method == "collaborative_only":
            return self._final_cap(
                context.query_row, context.collaborative_rows, n
            )
        if method == "production":
            return self._final_cap(context.query_row, context.production_rows, n)
        if method != "hybrid":
            raise ValueError(f"Unknown candidate method: {method}")
        if self.scorer is None:
            raise ValueError("Hybrid ranking requires a learned scorer")
        score = self.scorer.score(context.features)
        order = context.union_rows[np.argsort(score)[::-1]]
        return self._final_cap(context.query_row, order, n)

    def rank(self, row: int, method: str = "hybrid", n: int = 1000) -> List[int]:
        return self.rank_context(self.context(row), method, n=n)


def _target_rank(rows: Sequence[int], targets: Set[int]) -> int:
    for rank, row in enumerate(rows, 1):
        if int(row) in targets:
            return rank
    return 0


def _candidate_recall(
    contexts: Sequence[Tuple[Mapping[str, Any], QueryContext, Set[int]]],
    pool_name: str,
) -> Dict[str, float]:
    result = {}
    for cutoff in (100, 500, 1000):
        found = 0
        for _, context, targets in contexts:
            rows = getattr(context, pool_name)[:cutoff]
            found += bool(set(map(int, rows)) & targets)
        result[f"recall_at_{cutoff}"] = found / max(len(contexts), 1)
    return result


def _fit_pairwise(
    contexts: Sequence[Tuple[Mapping[str, Any], QueryContext, Set[int]]],
    c: float,
) -> Tuple[LinearScorer, Dict[str, int]]:
    differences = []
    labels = []
    positives_in_union = 0
    for pair, context, targets in contexts:
        positions = {
            int(row): position for position, row in enumerate(context.union_rows)
        }
        positive_position = next(
            (positions[target] for target in targets if target in positions), None
        )
        if positive_position is None:
            continue
        positives_in_union += 1
        positive = context.features[positive_position]
        negative_positions = np.linspace(
            0, min(len(context.union_rows), 250) - 1, 24, dtype=int
        )
        for negative_position in negative_positions:
            if negative_position == positive_position:
                continue
            negative = context.features[int(negative_position)]
            differences.append(positive - negative)
            labels.append(1)
            differences.append(negative - positive)
            labels.append(0)
    if not differences:
        raise RuntimeError("No DEV targets entered the hybrid candidate union")
    model = LogisticRegression(
        C=c,
        fit_intercept=False,
        max_iter=2000,
        solver="lbfgs",
        random_state=20260712,
    )
    model.fit(np.asarray(differences), np.asarray(labels))
    scorer = LinearScorer(
        feature_names=FEATURE_NAMES,
        coefficients=tuple(float(value) for value in model.coef_[0]),
        intercept=0.0,
        c=c,
    )
    return scorer, {
        "training_pairs": len(contexts),
        "positives_in_union": positives_in_union,
        "pairwise_examples": len(differences),
    }


def _listwise_grid_scorers() -> Iterable[LinearScorer]:
    """Yield interpretable linear blends for DEV listwise selection."""
    for audio_weight in (0.01, 0.05, 0.15, 0.30):
        for collaborative_weight in (0.5, 1.0, 2.0):
            for collaborative_rank_weight in (0.5, 1.5, 4.0, 8.0):
                for production_rank_weight in (0.0, 0.15, 0.5):
                    coefficients = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
                    coefficients[0] = audio_weight
                    coefficients[3] = collaborative_weight
                    coefficients[4] = production_rank_weight
                    coefficients[6] = collaborative_rank_weight
                    yield LinearScorer(
                        feature_names=FEATURE_NAMES,
                        coefficients=tuple(float(value) for value in coefficients),
                        intercept=0.0,
                        c=0.0,
                    )


def _pair_split(pair_id: str) -> str:
    value = hashlib.sha256(pair_id.encode("utf-8")).digest()[0]
    return "fit" if value < 180 else "selection"


def _evaluate(
    ranker: CollaborativeHybridRanker,
    contexts: Sequence[Tuple[Mapping[str, Any], QueryContext, Set[int]]],
) -> Dict[str, Any]:
    methods = ("production", "audio_only", "collaborative_only", "hybrid")
    ranks = {method: [] for method in methods}
    rows = []
    for pair, context, targets in contexts:
        pair_ranks = {}
        for method in methods:
            ranking = ranker.rank_context(context, method, n=50)
            rank = _target_rank(ranking, targets)
            ranks[method].append(rank)
            pair_ranks[method] = rank
        rows.append({
            "pair_id": pair["id"],
            "scene": pair["scene"],
            "query_mode": context.query_mode,
            "ranks": pair_ranks,
        })
    metrics = {method: _metrics(values) for method, values in ranks.items()}
    comparisons = {
        method: _bootstrap(ranks["production"], ranks[method], seed=20260712)
        for method in methods if method != "production"
    }
    return {"metrics": metrics, "comparisons": comparisons, "pairs": rows}


def _selection_metric(
    ranker: CollaborativeHybridRanker,
    contexts: Sequence[Tuple[Mapping[str, Any], QueryContext, Set[int]]],
) -> Dict[str, Any]:
    ranks = []
    for _, context, targets in contexts:
        ranking = ranker.rank_context(context, "hybrid", n=50)
        ranks.append(_target_rank(ranking, targets))
    return _metrics(ranks)


def train_and_evaluate_dev(
    index_path: Path,
    benchmark_path: Path,
    masked_asset_path: Path,
    full_asset_path: Path,
    scorer_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Tune only on opened DEV, with source-family-aware fit/selection splits."""
    from webapp.api._reco import WebRecommender

    started = time.perf_counter()
    recommender = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(recommender.titles, recommender.artists)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    dev_pairs = [pair for pair in benchmark["pairs"] if pair["split"] == "development"]
    masked_index = CollaborativeIndex(masked_asset_path, len(recommender))
    full_index = CollaborativeIndex(full_asset_path, len(recommender))
    masked_ranker = CollaborativeHybridRanker(recommender, masked_index)
    contexts = []
    for pair in dev_pairs:
        query_row = resolver.query_row(pair["query"])
        targets = set(resolver.target_rows(pair["target"]))
        if query_row is None or not targets:
            continue
        contexts.append((pair, masked_ranker.context(query_row), targets))
    fit_contexts = [
        item for item in contexts if _pair_split(item[0]["id"]) == "fit"
    ]
    selection_contexts = [
        item for item in contexts if _pair_split(item[0]["id"]) == "selection"
    ]
    candidates = []
    for c in (0.01, 0.1, 1.0, 10.0):
        scorer, fit_stats = _fit_pairwise(fit_contexts, c)
        masked_ranker.scorer = scorer
        metric = _selection_metric(masked_ranker, selection_contexts)
        candidates.append({
            "model_type": "pairwise_logistic",
            "c": c,
            "scorer": scorer,
            "fit_stats": fit_stats,
            "selection_primary": metric["primary"],
            "selection_recall_at_10": metric["recall_at_10"],
            "selection_mrr": metric["mrr"],
        })
    for scorer in _listwise_grid_scorers():
        masked_ranker.scorer = scorer
        metric = _selection_metric(masked_ranker, selection_contexts)
        candidates.append({
            "model_type": "listwise_linear_grid",
            "c": 0.0,
            "scorer": scorer,
            "fit_stats": {
                "training_pairs": len(fit_contexts),
                "objective": "selection ranked-list primary",
            },
            "selection_primary": metric["primary"],
            "selection_recall_at_10": metric["recall_at_10"],
            "selection_mrr": metric["mrr"],
        })
    chosen = max(
        candidates,
        key=lambda value: (
            value["selection_primary"],
            value["selection_recall_at_10"],
            value["selection_mrr"],
            -value["c"],
        ),
    )
    final_scorer = chosen["scorer"]
    final_fit_stats = chosen["fit_stats"]
    masked_ranker.scorer = final_scorer
    masked_all = _evaluate(masked_ranker, contexts)

    full_ranker = CollaborativeHybridRanker(
        recommender, full_index, scorer=final_scorer
    )
    full_contexts = []
    for pair, _, targets in contexts:
        query_row = resolver.query_row(pair["query"])
        full_contexts.append((pair, full_ranker.context(query_row), targets))
    full_all = _evaluate(full_ranker, full_contexts)
    candidate_recall = {
        "audio_only": _candidate_recall(contexts, "audio_rows"),
        "collaborative_edge_masked": _candidate_recall(
            contexts, "collaborative_rows"
        ),
        "hybrid_union_edge_masked": _candidate_recall(contexts, "union_rows"),
        "collaborative_unmasked": _candidate_recall(
            full_contexts, "collaborative_rows"
        ),
        "hybrid_union_unmasked": _candidate_recall(
            full_contexts, "union_rows"
        ),
    }
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_document = {
        "schema_version": 1,
        "method": "dev-tuned-linear-hybrid-reranker",
        "created_at": datetime_now(),
        "scorer": final_scorer.to_dict(),
        "candidate_configuration": asdict(masked_ranker.config),
        "selection": {
            "rule": "maximum selection primary, then R@10, MRR, simpler regularization",
            "chosen_model_type": chosen["model_type"],
            "chosen_c": chosen["c"],
            "candidates": [
                {key: value for key, value in item.items() if key != "scorer"}
                for item in candidates
            ],
        },
    }
    scorer_path.write_text(
        json.dumps(scorer_document, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "created_at": datetime_now(),
        "split": "development-only",
        "final_labels_compared": False,
        "benchmark_sha256": file_digest(benchmark_path),
        "index_sha256": file_digest(index_path),
        "pairs": {
            "resolved": len(contexts),
            "fit": len(fit_contexts),
            "selection": len(selection_contexts),
        },
        "fit": final_fit_stats,
        "candidate_recall": candidate_recall,
        "edge_masked": masked_all,
        "unmasked_ablation": full_all,
        "selected_scorer": final_scorer.to_dict(),
        "wall_seconds": time.perf_counter() - started,
    }
    report["content_sha256"] = content_sha256({
        key: value for key, value in report.items() if key != "content_sha256"
    })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def datetime_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_locked_rankings(
    index_path: Path,
    protocol_manifest_path: Path,
    method_manifest_path: Path,
    masked_asset_path: Path,
    full_asset_path: Path,
    scorer_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate all target-agnostic control and candidate rankings."""
    from webapp.api._reco import WebRecommender

    recommender = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(recommender.titles, recommender.artists)
    scorer_doc = json.loads(scorer_path.read_text(encoding="utf-8"))
    scorer = LinearScorer.from_dict(scorer_doc["scorer"])
    masked = CollaborativeHybridRanker(
        recommender,
        CollaborativeIndex(masked_asset_path, len(recommender)),
        scorer=scorer,
    )
    full = CollaborativeHybridRanker(
        recommender,
        CollaborativeIndex(full_asset_path, len(recommender)),
        scorer=scorer,
    )
    manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    records = []
    for pair in manifest["pairs"]:
        query_row = resolver.query_row(pair["query"])
        if query_row is None:
            raise RuntimeError(f"Missing frozen query: {pair['id']}")
        masked_context = masked.context(query_row)
        full_context = full.context(query_row)
        controls = {
            "audio_only": masked.rank_context(
                masked_context, "audio_only", n=50
            ),
            "collaborative_only": masked.rank_context(
                masked_context, "collaborative_only", n=50
            ),
            "hybrid_edge_masked": masked.rank_context(
                masked_context, "hybrid", n=50
            ),
            "hybrid_unmasked_ablation": full.rank_context(
                full_context, "hybrid", n=50
            ),
        }
        candidate_sets = {
            "audio_only": masked_context.audio_rows[:1000].tolist(),
            "collaborative_edge_masked": (
                masked_context.collaborative_rows[:1000].tolist()
            ),
            "production": masked_context.production_rows[:1000].tolist(),
            "hybrid_union_edge_masked": (
                masked_context.union_rows[:1000].tolist()
            ),
            "collaborative_unmasked": (
                full_context.collaborative_rows[:1000].tolist()
            ),
            "hybrid_union_unmasked": full_context.union_rows[:1000].tolist(),
        }
        records.append({
            "pair_id": pair["id"],
            "query": pair["query"],
            "query_row": int(query_row),
            "query_mode": masked_context.query_mode,
            "ranking": serialise_rows(
                recommender, controls["hybrid_edge_masked"]
            ),
            "diagnostic_rankings": {
                method: serialise_rows(recommender, rows)
                for method, rows in controls.items()
            },
            "candidate_sets": candidate_sets,
        })
    result = {
        "schema_version": 2,
        "created_at": datetime_now(),
        "target_labels_compared": False,
        "method_manifest_sha256": file_digest(method_manifest_path),
        "records": records,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def serialise_rows(recommender: Any, rows: Sequence[int]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": rank,
            "row": int(row),
            "track_id": int(recommender.track_ids[int(row)]),
            "title": str(recommender.titles[int(row)]),
            "artist": str(recommender.artists[int(row)]),
        }
        for rank, row in enumerate(rows, 1)
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("dev")
    for command in (dev,):
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--benchmark", type=Path, required=True)
        command.add_argument("--masked-asset", type=Path, required=True)
        command.add_argument("--full-asset", type=Path, required=True)
    dev.add_argument("--scorer", type=Path, required=True)
    dev.add_argument("--report", type=Path, required=True)
    locked = sub.add_parser("locked-rankings")
    locked.add_argument("--index", type=Path, required=True)
    locked.add_argument("--manifest", type=Path, required=True)
    locked.add_argument("--method-manifest", type=Path, required=True)
    locked.add_argument("--masked-asset", type=Path, required=True)
    locked.add_argument("--full-asset", type=Path, required=True)
    locked.add_argument("--scorer", type=Path, required=True)
    locked.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "dev":
        result = train_and_evaluate_dev(
            args.index,
            args.benchmark,
            args.masked_asset,
            args.full_asset,
            args.scorer,
            args.report,
        )
    else:
        result = build_locked_rankings(
            args.index,
            args.manifest,
            args.method_manifest,
            args.masked_asset,
            args.full_asset,
            args.scorer,
            args.output,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
