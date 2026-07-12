"""Tamper-evident, one-open protocol for the final retrieval benchmark.

The freeze phase stores target-agnostic ranked outputs for every baseline.  It
does not calculate target ranks or metrics.  A selected method can be locked
only after the freeze, and the final labels can be scored only once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .quality_filter import TitleQualityFilter
from .real_benchmark import (
    PairResolver,
    ProductionRanker,
    credited_artists,
    normalize_text,
)


BASELINE_METHODS = (
    "production_baseline",
    "iteration3_deployed",
    "raw_encoder",
    "audio_priors_zero",
)
RECALL_CUTOFFS = (1, 5, 10, 20, 50)


class ProtocolError(RuntimeError):
    """Raised when an operation would violate the frozen protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible content."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_state_signature(
    state: Mapping[str, Any],
    state_path: Path | None = None,
) -> None:
    """Reject state corruption and verify an optional detached final seal."""
    expected = state.get("integrity_signature")
    if not expected:
        raise ProtocolError("Protocol state has no integrity signature")
    unsigned = dict(state)
    unsigned.pop("integrity_signature", None)
    unsigned.pop("signature_algorithm", None)
    actual = content_sha256(unsigned)
    if actual != expected:
        raise ProtocolError("Protocol state integrity signature mismatch")
    if state_path is None:
        return
    protocol_dir = state_path.parent
    signature = protocol_dir / "state.sig"
    allowed = protocol_dir / "allowed_signers"
    if not signature.exists() and not allowed.exists():
        return
    if not signature.exists() or not allowed.exists():
        raise ProtocolError("Detached protocol signature files are incomplete")
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise ProtocolError("ssh-keygen is required to verify protocol signature")
    verified = subprocess.run(
        [
            executable, "-Y", "verify", "-f", str(allowed),
            "-I", "soundalike-protocol", "-n", "soundalike-protocol",
            "-s", str(signature),
        ],
        input=state_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode != 0:
        raise ProtocolError("Detached Ed25519 protocol signature mismatch")


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise ProtocolError(f"Frozen {label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ProtocolError(
            f"Frozen {label} hash mismatch: expected {expected}, got {actual}"
        )


def _verify_frozen_inputs(state: Mapping[str, Any], locked: bool = False) -> None:
    """Verify every immutable file referenced by protocol state."""
    _verify_file(
        Path(state["benchmark_path"]), state["benchmark_sha256"], "benchmark"
    )
    _verify_file(Path(state["index_path"]), state["index_sha256"], "index")
    _verify_file(
        Path(state["manifest_path"]), state["manifest_sha256"], "FINAL manifest"
    )
    _verify_file(
        Path(state["baseline_path"]), state["baseline_sha256"],
        "baseline rankings",
    )
    if locked:
        _verify_file(
            Path(state["method_manifest_path"]),
            state["method_manifest_sha256"], "method manifest",
        )
        _verify_file(
            Path(state["dev_report_path"]), state["dev_report_sha256"],
            "development report",
        )
        for raw_path, digest in state.get("asset_hashes", {}).items():
            _verify_file(Path(raw_path), digest, f"method asset {raw_path}")


def _pair_artists(pair: Mapping[str, Any]) -> Set[str]:
    return (
        credited_artists(pair["query"]["artist"])
        | credited_artists(pair["target"]["artist"])
    )


def validate_benchmark(benchmark: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate scale, provenance, deduplication, and component isolation."""
    pairs = [
        pair for pair in benchmark.get("pairs", [])
        if pair.get("deciding_primary")
        and pair.get("evidence_category") == "category_a_sonic"
    ]
    development = [pair for pair in pairs if pair.get("split") == "development"]
    final = [pair for pair in pairs if pair.get("split") == "final"]
    if len(pairs) < 100:
        raise ProtocolError("Category-A benchmark must contain at least 100 pairs")
    if len(final) < 30:
        raise ProtocolError("FINAL must contain at least 30 deciding pairs")

    tracks: Set[Tuple[str, str]] = set()
    artists: Set[str] = set()
    for pair in pairs:
        if not pair.get("sources"):
            raise ProtocolError(f"{pair['id']} has no provenance")
        for source in pair["sources"]:
            for field in ("url", "accessed_at", "source_class", "excerpt"):
                if not source.get(field):
                    raise ProtocolError(f"{pair['id']} source lacks {field}")
        pair_artist_set = _pair_artists(pair)
        if len(pair_artist_set) < 2:
            raise ProtocolError(f"{pair['id']} is not cross-artist")
        if artists & pair_artist_set:
            raise ProtocolError(f"{pair['id']} repeats a benchmark artist")
        artists |= pair_artist_set
        for side in ("query", "target"):
            song = pair[side]
            key = (
                normalize_text(song["title"]),
                normalize_text(song["artist"]),
            )
            if key in tracks:
                raise ProtocolError(f"{pair['id']} repeats a benchmark track")
            tracks.add(key)

    dev_artists = set().union(*(_pair_artists(pair) for pair in development))
    final_artists = set().union(*(_pair_artists(pair) for pair in final))
    overlap = sorted(dev_artists & final_artists)
    if overlap:
        raise ProtocolError(f"Development/FINAL artist overlap: {overlap}")

    scenes = {pair["scene"] for pair in pairs}
    tiers = {pair.get("catalog_tier") for pair in pairs}
    if len(scenes) < 15:
        raise ProtocolError("Benchmark must span at least 15 scenes")
    if not {"popular", "deep_cut", "niche"} <= tiers:
        raise ProtocolError("Benchmark must include popular, deep-cut, and niche music")
    return {
        "category_a_pairs": len(pairs),
        "development_pairs": len(development),
        "final_pairs": len(final),
        "scenes": len(scenes),
        "artists": len(artists),
        "artist_overlap": overlap,
    }


def _rank_audio_priors_zero(
    recommender: Any,
    row: int,
    n: int = 50,
) -> List[int]:
    """Rank only the two cached audio representations; use no global priors."""
    if recommender._sonic is None or recommender._clap is None:
        raise ProtocolError("Audio ablation requires sonic and CLAP arrays")
    sonic = recommender._compact_cosine(
        recommender._sonic,
        np.asarray(recommender._sonic[row], dtype=np.float32),
    )
    clap = recommender._compact_cosine(
        recommender._clap,
        np.asarray(recommender._clap[row], dtype=np.float32),
    )
    score = 0.25 * recommender._z(sonic) + 0.75 * recommender._z(clap)
    quality = TitleQualityFilter()
    quality_mask = quality.keep_mask(recommender.titles, recommender.artists)
    seed_artist = normalize_text(str(recommender.artists[row]))
    seed_title = str(recommender.titles[row])
    seen_artists: Set[str] = set()
    seen_tracks: Set[Tuple[str, str]] = set()
    ranked: List[int] = []
    for raw in np.argsort(score)[::-1]:
        candidate = int(raw)
        artist = normalize_text(str(recommender.artists[candidate]))
        title = str(recommender.titles[candidate])
        track = (normalize_text(title), artist)
        if candidate == row or artist == seed_artist:
            continue
        if not quality_mask[candidate] or quality.seed_title_in_result(
            seed_title, title
        ):
            continue
        if artist in seen_artists or track in seen_tracks:
            continue
        ranked.append(candidate)
        seen_artists.add(artist)
        seen_tracks.add(track)
        if len(ranked) == n:
            break
    return ranked


def _serialise_ranking(
    recommender: Any,
    rows: Iterable[int],
) -> List[Dict[str, Any]]:
    return [
        {
            "rank": rank,
            "row": int(row),
            "track_id": int(recommender.track_ids[row]),
            "title": str(recommender.titles[row]),
            "artist": str(recommender.artists[row]),
        }
        for rank, row in enumerate(rows, 1)
    ]


def freeze_protocol(
    benchmark_path: Path,
    index_path: Path,
    protocol_dir: Path,
) -> Dict[str, Any]:
    """Freeze FINAL and target-agnostic baseline rankings before model work."""
    state_path = protocol_dir / "state.json"
    if state_path.exists():
        raise ProtocolError("Protocol state already exists; freeze is immutable")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit = validate_benchmark(benchmark)
    final_pairs = [
        pair for pair in benchmark["pairs"]
        if pair.get("split") == "final"
        and pair.get("deciding_primary")
        and pair.get("evidence_category") == "category_a_sonic"
    ]
    manifest = {
        "schema_version": 1,
        "benchmark_id": benchmark["benchmark_id"],
        "created_at": _now(),
        "pairs": final_pairs,
    }
    manifest["content_sha256"] = content_sha256(manifest["pairs"])

    # Import lazily so protocol unit tests do not load the 299 MB release index.
    from webapp.api._reco import WebRecommender

    recommender = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(recommender.titles, recommender.artists)
    ranker = ProductionRanker(recommender, heldout=set())
    records: List[Dict[str, Any]] = []
    for pair in final_pairs:
        row = recommender.find_row(
            pair["query"]["title"], pair["query"]["artist"]
        )
        if row is None:
            raise ProtocolError(f"FINAL query is absent: {pair['id']}")
        resolved = resolver.target_rows(pair["target"])
        if not resolved:
            raise ProtocolError(f"FINAL target is absent: {pair['id']}")
        method_rows = {
            "production_baseline": ranker.rank(row, "production_baseline"),
            "iteration3_deployed": ranker.rank(row, "dual_sonic"),
            "raw_encoder": ranker.rank(row, "raw_encoder"),
            "audio_priors_zero": _rank_audio_priors_zero(recommender, row),
        }
        records.append({
            "pair_id": pair["id"],
            "query": dict(pair["query"]),
            "query_row": int(row),
            "rankings": {
                method: _serialise_ranking(recommender, rows)
                for method, rows in method_rows.items()
            },
        })
    baseline_outputs = {
        "schema_version": 1,
        "created_at": _now(),
        "target_labels_compared": False,
        "methods": list(BASELINE_METHODS),
        "records": records,
    }
    baseline_outputs["content_sha256"] = content_sha256(records)

    manifest_path = protocol_dir / "final-test-manifest.json"
    baseline_path = protocol_dir / "frozen-baseline-rankings.json"
    _write_json(manifest_path, manifest)
    _write_json(baseline_path, baseline_outputs)
    state = {
        "schema_version": 1,
        "status": "FROZEN",
        "created_at": _now(),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": file_sha256(benchmark_path),
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "baseline_path": str(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "final_open_count": 0,
        "audit": audit,
    }
    unsigned = dict(state)
    state["integrity_signature"] = content_sha256(unsigned)
    state["signature_algorithm"] = "SHA-256 canonical JSON"
    _write_json(state_path, state)
    return state


def lock_method(
    protocol_dir: Path,
    method_manifest_path: Path,
    dev_report_path: Path,
) -> Dict[str, Any]:
    """Lock a model, configuration, and DEV report before FINAL evaluation."""
    state_path = protocol_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state_signature(state, state_path)
    _verify_frozen_inputs(state)
    if state.get("status") != "FROZEN" or state.get("final_open_count") != 0:
        raise ProtocolError("Method can be locked only from unopened FROZEN state")
    method = json.loads(method_manifest_path.read_text(encoding="utf-8"))
    for field in ("method_id", "configuration", "assets"):
        if not method.get(field):
            raise ProtocolError(f"Method manifest lacks {field}")
    asset_hashes = {}
    for asset in method["assets"]:
        path = Path(asset["path"])
        digest = file_sha256(path)
        if asset.get("sha256") and asset["sha256"] != digest:
            raise ProtocolError(f"Asset hash mismatch: {path}")
        asset_hashes[str(path)] = digest
    state.update({
        "status": "METHOD_LOCKED",
        "locked_at": _now(),
        "method_id": method["method_id"],
        "method_manifest_path": str(method_manifest_path),
        "method_manifest_sha256": file_sha256(method_manifest_path),
        "dev_report_path": str(dev_report_path),
        "dev_report_sha256": file_sha256(dev_report_path),
        "asset_hashes": asset_hashes,
    })
    state.pop("integrity_signature", None)
    state.pop("signature_algorithm", None)
    state["integrity_signature"] = content_sha256(state)
    state["signature_algorithm"] = "SHA-256 canonical JSON"
    _write_json(state_path, state)
    return state


def commit_rankings(
    protocol_dir: Path,
    winner_rankings_path: Path,
) -> Dict[str, Any]:
    """Commit target-agnostic winner rankings before labels may be scored."""
    state_path = protocol_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state_signature(state, state_path)
    if state.get("final_open_count") != 0:
        raise ProtocolError("FINAL has already been opened")
    if state.get("status") != "METHOD_LOCKED":
        raise ProtocolError("Rankings can be committed only after method lock")
    _verify_frozen_inputs(state, locked=True)
    rankings = json.loads(winner_rankings_path.read_text(encoding="utf-8"))
    if rankings.get("target_labels_compared") is not False:
        raise ProtocolError("Winner rankings must be target-agnostic")
    if rankings.get("method_manifest_sha256") != state["method_manifest_sha256"]:
        raise ProtocolError("Winner rankings do not match the locked method")
    manifest = json.loads(Path(state["manifest_path"]).read_text(encoding="utf-8"))
    expected_ids = {pair["id"] for pair in manifest["pairs"]}
    ranking_ids = {record["pair_id"] for record in rankings.get("records", [])}
    if ranking_ids != expected_ids:
        raise ProtocolError("Winner rankings and FINAL manifest differ")
    state.update({
        "status": "RANKINGS_LOCKED",
        "winner_rankings_path": str(winner_rankings_path),
        "winner_rankings_sha256": file_sha256(winner_rankings_path),
        "winner_rankings_content_sha256": content_sha256(rankings["records"]),
        "rankings_locked_at": _now(),
    })
    state.pop("integrity_signature", None)
    state.pop("signature_algorithm", None)
    state["integrity_signature"] = content_sha256(state)
    state["signature_algorithm"] = "SHA-256 canonical JSON"
    _write_json(state_path, state)
    return state


def _rank_for_rows(ranking: Sequence[Mapping[str, Any]], targets: Set[int]) -> int:
    for item in ranking:
        if int(item["row"]) in targets:
            return int(item["rank"])
    return 0


def _metrics(ranks: Sequence[int]) -> Dict[str, Any]:
    values = np.asarray(ranks, dtype=np.int64)
    rr = np.where(values > 0, 1.0 / np.maximum(values, 1), 0.0)
    result = {
        f"recall_at_{cutoff}": float(np.mean((values > 0) & (values <= cutoff)))
        for cutoff in RECALL_CUTOFFS
    }
    result["mrr"] = float(np.mean(rr))
    for cutoff in (10, 50):
        result[f"ndcg_at_{cutoff}"] = float(np.mean([
            1.0 / math.log2(rank + 1) if 0 < rank <= cutoff else 0.0
            for rank in values
        ]))
    result["primary"] = float(np.mean([
        result["ndcg_at_10"], result["mrr"], result["recall_at_10"]
    ]))
    return result


def _contribution(rank: int) -> float:
    rr = 1.0 / rank if rank else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if 0 < rank <= 10 else 0.0
    recall = float(0 < rank <= 10)
    return (ndcg + rr + recall) / 3.0


def _bootstrap(
    baseline: Sequence[int],
    winner: Sequence[int],
    iterations: int = 20_000,
    seed: int = 20260711,
) -> Dict[str, float]:
    base = np.asarray([_contribution(rank) for rank in baseline])
    test = np.asarray([_contribution(rank) for rank in winner])
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = rng.integers(0, len(base), len(base))
        deltas[iteration] = np.mean(test[sample] - base[sample])
    absolute = float(np.mean(test - base))
    return {
        "absolute_delta": absolute,
        "relative_gain": absolute / (float(np.mean(base)) + 1e-12),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "probability_positive": float(np.mean(deltas > 0)),
    }


def open_final_once(
    protocol_dir: Path,
    winner_rankings_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Score the locked method and frozen baselines, atomically, exactly once."""
    state_path = protocol_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state_signature(state, state_path)
    if state.get("final_open_count") != 0:
        raise ProtocolError("FINAL has already been opened")
    if state.get("status") != "RANKINGS_LOCKED":
        raise ProtocolError("FINAL can be opened only after rankings lock")
    _verify_frozen_inputs(state, locked=True)
    if str(winner_rankings_path) != state.get("winner_rankings_path"):
        raise ProtocolError("Winner rankings path differs from the locked path")
    _verify_file(
        winner_rankings_path, state["winner_rankings_sha256"],
        "winner rankings",
    )
    winner_doc = json.loads(winner_rankings_path.read_text(encoding="utf-8"))
    if content_sha256(winner_doc.get("records", [])) != state.get(
        "winner_rankings_content_sha256"
    ):
        raise ProtocolError("Winner rankings content hash mismatch")
    if winner_doc.get("method_manifest_sha256") != state["method_manifest_sha256"]:
        raise ProtocolError("Winner rankings do not match the locked method")
    manifest = json.loads(
        (protocol_dir / "final-test-manifest.json").read_text(encoding="utf-8")
    )
    frozen = json.loads(
        (protocol_dir / "frozen-baseline-rankings.json").read_text(
            encoding="utf-8"
        )
    )
    if content_sha256(manifest["pairs"]) != manifest.get("content_sha256"):
        raise ProtocolError("FINAL manifest content hash mismatch")
    if content_sha256(frozen["records"]) != frozen.get("content_sha256"):
        raise ProtocolError("Frozen baseline content hash mismatch")
    pair_by_id = {pair["id"]: pair for pair in manifest["pairs"]}
    frozen_by_id = {record["pair_id"]: record for record in frozen["records"]}
    winner_by_id = {
        record["pair_id"]: record for record in winner_doc["records"]
    }
    if set(pair_by_id) != set(frozen_by_id) or set(pair_by_id) != set(winner_by_id):
        raise ProtocolError("Winner, baselines, and FINAL manifest differ")

    index_path = Path(state["index_path"])
    with np.load(index_path, allow_pickle=False) as index:
        resolver = PairResolver(index["titles"], index["artists"])
        ranks: Dict[str, List[int]] = {
            method: [] for method in (*BASELINE_METHODS, "winner")
        }
        rows = []
        for pair_id in pair_by_id:
            pair = pair_by_id[pair_id]
            targets = set(resolver.target_rows(pair["target"]))
            record = frozen_by_id[pair_id]
            pair_ranks = {
                method: _rank_for_rows(record["rankings"][method], targets)
                for method in BASELINE_METHODS
            }
            pair_ranks["winner"] = _rank_for_rows(
                winner_by_id[pair_id]["ranking"], targets
            )
            for method, rank in pair_ranks.items():
                ranks[method].append(rank)
            rows.append({
                "pair_id": pair_id,
                "scene": pair["scene"],
                "query": pair["query"],
                "target": pair["target"],
                "ranks": pair_ranks,
            })
    metrics = {method: _metrics(values) for method, values in ranks.items()}
    comparison = _bootstrap(ranks["production_baseline"], ranks["winner"])
    base_contrib = np.asarray([
        _contribution(rank) for rank in ranks["production_baseline"]
    ])
    win_contrib = np.asarray([_contribution(rank) for rank in ranks["winner"]])
    comparison.update({
        "improved_pairs": int(np.sum(win_contrib > base_contrib)),
        "worsened_pairs": int(np.sum(win_contrib < base_contrib)),
        "unchanged_pairs": int(np.sum(win_contrib == base_contrib)),
    })
    scene_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"baseline": [], "winner": []}
    )
    for record in rows:
        scene_values[record["scene"]]["baseline"].append(
            _contribution(record["ranks"]["production_baseline"])
        )
        scene_values[record["scene"]]["winner"].append(
            _contribution(record["ranks"]["winner"])
        )
    per_scene = {}
    for scene, values in sorted(scene_values.items()):
        baseline_mean = float(np.mean(values["baseline"]))
        winner_mean = float(np.mean(values["winner"]))
        relative_delta = (
            (winner_mean - baseline_mean) / baseline_mean
            if baseline_mean > 0 else 0.0
        )
        per_scene[scene] = {
            "baseline_primary": baseline_mean,
            "winner_primary": winner_mean,
            "relative_delta": relative_delta,
        }
    comparison["per_scene"] = per_scene
    policy = json.loads(Path(state["benchmark_path"]).read_text(
        encoding="utf-8"
    ))["metric_policy"]["success"]
    comparison["passes"] = {
        "relative_gain": comparison["relative_gain"]
        >= policy["minimum_relative_primary_gain"],
        "positive_absolute": comparison["absolute_delta"] > 0,
        "recall_at_10_non_regression": metrics["winner"]["recall_at_10"]
        >= metrics["production_baseline"]["recall_at_10"],
        "mrr_non_regression": metrics["winner"]["mrr"]
        >= metrics["production_baseline"]["mrr"],
        "ci_excludes_zero": comparison["ci95_low"]
        > policy["paired_bootstrap_ci95_low_must_exceed"],
        "meaningful_count": comparison["improved_pairs"]
        >= policy["minimum_improved_pairs"],
        "scene_no_regression": all(
            values["relative_delta"]
            >= policy.get("maximum_scene_relative_regression", -0.10)
            for values in per_scene.values()
        ),
    }
    comparison["final_pass"] = all(comparison["passes"].values())
    report = {
        "schema_version": 1,
        "opened_at": _now(),
        "open_number": 1,
        "method_id": state["method_id"],
        "metrics": metrics,
        "comparison_to_production_baseline": comparison,
        "pairs": rows,
    }
    _write_json(report_path, report)
    state.update({
        "status": "FINALIZED",
        "final_open_count": 1,
        "final_opened_at": report["opened_at"],
        "final_report_path": str(report_path),
        "final_report_sha256": file_sha256(report_path),
        "final_pass": comparison["final_pass"],
    })
    state.pop("integrity_signature", None)
    state.pop("signature_algorithm", None)
    state["integrity_signature"] = content_sha256(state)
    state["signature_algorithm"] = "SHA-256 canonical JSON"
    _write_json(state_path, state)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--benchmark", type=Path, required=True)
    freeze.add_argument("--index", type=Path, required=True)
    freeze.add_argument("--protocol-dir", type=Path, required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--protocol-dir", type=Path, required=True)
    lock.add_argument("--method-manifest", type=Path, required=True)
    lock.add_argument("--dev-report", type=Path, required=True)
    rankings = sub.add_parser("commit-rankings")
    rankings.add_argument("--protocol-dir", type=Path, required=True)
    rankings.add_argument("--winner-rankings", type=Path, required=True)
    final = sub.add_parser("open-final")
    final.add_argument("--protocol-dir", type=Path, required=True)
    final.add_argument("--winner-rankings", type=Path, required=True)
    final.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_protocol(args.benchmark, args.index, args.protocol_dir)
    elif args.command == "lock":
        result = lock_method(
            args.protocol_dir, args.method_manifest, args.dev_report
        )
    elif args.command == "commit-rankings":
        result = commit_rankings(args.protocol_dir, args.winner_rankings)
    else:
        result = open_final_once(
            args.protocol_dir, args.winner_rankings, args.report
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
