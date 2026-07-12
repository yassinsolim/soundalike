"""Subprocess resource measurement for the compact graph-first catalogue ranker."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


GIB = 1024 ** 3
PEAK_LIMIT_BYTES = int(1.5 * GIB)
RESIDENT_TARGET_BYTES = int(1.1 * GIB)
CONSERVATIVE_PLATFORM_LIMIT_BYTES = 2 * GIB
MINIMUM_WARM_QUERIES = 16
VERCEL_DOCUMENTATION_URL = "https://vercel.com/docs/functions/limitations"
VERCEL_DOCUMENTATION_AS_OF = "2026-07-01"


class ResourceMeasurementError(ValueError):
    """Raised for an invalid or incomplete resource measurement."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ResourceMeasurementError("latency samples must not be empty")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_statistics(seconds: Sequence[float]) -> Dict[str, Any]:
    """Return stable millisecond summary statistics."""
    values = [float(value) * 1000.0 for value in seconds]
    if not values:
        raise ResourceMeasurementError("latency samples must not be empty")
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _asset_details(path: Path) -> Dict[str, Any]:
    import numpy as np

    arrays: Dict[str, Dict[str, Any]] = {}
    with np.load(str(path), allow_pickle=False) as asset:
        for name in asset.files:
            value = asset[name]
            arrays[name] = {
                "dtype": str(value.dtype),
                "shape": [int(size) for size in value.shape],
                "uncompressed_bytes": int(value.nbytes),
            }
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "arrays": arrays,
        "dtypes": {name: value["dtype"] for name, value in arrays.items()},
    }


def _json_asset_details(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "format": "json",
    }


def _graph_contract(graph: Any, details: Mapping[str, Any]) -> Dict[str, Any]:
    variants = sorted(str(name) for name in graph.variants)
    arrays = details["arrays"]
    masked = [
        name
        for name in ("direct", "twohop")
        if name + "_indices" in arrays or name + "_weights" in arrays
    ]
    aliases: List[str] = []
    if masked:
        import numpy as np

        full_indices, full_weights = graph.variants["full"]
        for name in masked:
            pair = graph.variants.get(name)
            if pair is not None and np.array_equal(pair[0], full_indices) and np.array_equal(
                pair[1], full_weights
            ):
                aliases.append(name)
    metadata = graph.metadata if isinstance(graph.metadata, Mapping) else {}
    only_full = variants == ["full"]
    return {
        "variants": variants,
        "only_full": only_full,
        "masked_variants_present": masked,
        "masked_variants_alias_full": aliases,
        "silent_fallback_declared": bool(metadata.get("silent_fallback", True)),
        "passed": (
            only_full
            and not masked
            and not aliases
            and metadata.get("silent_fallback") is False
        ),
    }


def _load_rows(path: Path) -> List[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        for key in ("query_rows", "rows"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or not value:
        raise ResourceMeasurementError(
            "pre-resolved row artifact must contain a non-empty JSON list"
        )
    rows: List[int] = []
    for item in value:
        if isinstance(item, Mapping):
            if set(item) - {"row", "id", "scene"} or "row" not in item:
                raise ResourceMeasurementError(
                    "row records must be pre-resolved and contain a row"
                )
            item = item["row"]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ResourceMeasurementError("query rows must be non-negative integers")
        rows.append(int(item))
    return rows


def _load_policy(path: Path) -> Tuple[Any, Dict[str, float]]:
    from .catalog_policy import CatalogPolicy

    value: Any = json.loads(path.read_text(encoding="utf-8"))
    fields = ("audio_weight", "style_weight", "style_guard_min")
    if isinstance(value, Mapping):
        for keys in (
            ("exact_policy",),
            ("policy",),
            ("selected_policy",),
            ("selected_policy_evaluation", "exact_policy"),
            ("nested_5fold", "final_policy"),
        ):
            candidate: Any = value
            for key in keys:
                candidate = candidate.get(key) if isinstance(candidate, Mapping) else None
            if isinstance(candidate, Mapping):
                value = candidate
                break
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ResourceMeasurementError(
            "policy artifact must contain exactly the three policy parameters"
        )
    exact = {field: float(value[field]) for field in fields}
    return CatalogPolicy(**exact), exact


def _rss_bytes(process: Any) -> int:
    """Return aggregate RSS for a process tree, tolerating normal exit races."""
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except Exception:
        pass
    total = 0
    seen = set()
    for child in processes:
        pid = getattr(child, "pid", id(child))
        if pid in seen:
            continue
        seen.add(pid)
        try:
            total += int(child.memory_info().rss)
        except Exception:
            continue
    return total


def poll_peak_rss(
    child: Any,
    psutil_module: Any,
    interval_seconds: float = 0.01,
) -> Tuple[int, int]:
    """Poll child and descendant RSS until exit and return peak and sample count."""
    if interval_seconds <= 0:
        raise ValueError("poll interval must be positive")
    process = psutil_module.Process(child.pid)
    peak, samples = 0, 0
    while True:
        peak = max(peak, _rss_bytes(process))
        samples += 1
        if child.poll() is not None:
            peak = max(peak, _rss_bytes(process))
            samples += 1
            break
        time.sleep(interval_seconds)
    return peak, samples


def platform_limit_provenance() -> Dict[str, Any]:
    """Return conservative, credential-honest Vercel memory-limit provenance."""
    return {
        "provider": "Vercel Functions",
        "documentation_url": VERCEL_DOCUMENTATION_URL,
        "documentation_as_of": VERCEL_DOCUMENTATION_AS_OF,
        "documented_maximums_bytes": {
            "Hobby": 2 * GIB,
            "Pro": 4 * GIB,
            "Enterprise": 4 * GIB,
        },
        "project_tier": "unknown",
        "project_tier_credential_verification": {
            "available": False,
            "reason": "Vercel CLI credentials unavailable",
        },
        "limit_used_bytes": CONSERVATIVE_PLATFORM_LIMIT_BYTES,
        "limit_basis": "conservative Hobby maximum; no project plan claimed",
    }


def apply_resource_gates(report: Dict[str, Any]) -> Dict[str, Any]:
    """Add byte-exact resource and correctness gates to a worker report."""
    peak = int(report.get("peak_rss_bytes", 0))
    resident = int(report.get("post_gc_resident_rss_bytes", 0))
    core = int(report.get("core_index_post_gc_rss_bytes", 0))
    provenance = platform_limit_provenance()
    platform_limit = int(provenance["limit_used_bytes"])
    resident_met = resident <= RESIDENT_TARGET_BYTES
    core_exceeds_target = core > RESIDENT_TARGET_BYTES
    graph_passed = bool(report.get("graph_contract", {}).get("passed", False))
    style_present = bool(report.get("style", {}).get("present", False))
    zero_fallbacks = int(report.get("fallback_count", 0)) == 0
    zero_errors = not report.get("errors")
    deterministic = bool(report.get("determinism", {}).get("passed", False))
    peak_passed = 0 < peak <= PEAK_LIMIT_BYTES
    headroom = platform_limit - peak
    gates = {
        "peak": {
            "limit_bytes": PEAK_LIMIT_BYTES,
            "measured_bytes": peak,
            "passed": peak_passed,
        },
        "resident_target": {
            "target_bytes": RESIDENT_TARGET_BYTES,
            "measured_bytes": resident,
            "target_met": resident_met,
            "blocking": not core_exceeds_target,
            "infeasible_nonblocking": bool(not resident_met and core_exceeds_target),
            "reason": (
                "core production index alone exceeds resident target"
                if not resident_met and core_exceeds_target
                else ""
            ),
        },
        "platform": {
            "limit_bytes": platform_limit,
            "headroom_bytes": headroom,
            "passed": headroom > 0,
        },
        "compact_graph_only_full": graph_passed,
        "style_present": style_present,
        "zero_fallbacks": zero_fallbacks,
        "zero_errors": zero_errors,
        "deterministic_output": deterministic,
    }
    required = (
        peak_passed,
        headroom > 0,
        graph_passed,
        style_present,
        zero_fallbacks,
        zero_errors,
        deterministic,
        resident_met or core_exceeds_target,
    )
    report["platform_limit"] = provenance
    report["headroom_bytes"] = headroom
    report.setdefault(
        "units",
        {"memory": "bytes", "latency": "milliseconds", "duration": "seconds"},
    )
    report["gates"] = gates
    report["passed"] = all(required)
    return report


def _process_rss() -> int:
    import psutil

    return int(psutil.Process(os.getpid()).memory_info().rss)


def _worker_measure(arguments: argparse.Namespace) -> Dict[str, Any]:
    start_rss = _process_rss()
    started = time.perf_counter()
    errors: List[Dict[str, str]] = []

    from webapp.api._reco import WebRecommender
    from .catalog_graph import CatalogArtistGraph
    from .catalog_policy import CatalogPolicyRanker
    from .catalog_style import CatalogStyleIndex

    # Preserve caller-relative paths in reproducible artifacts; the worker runs
    # from the repository root and does not need to disclose a local user path.
    paths = {
        "index": Path(arguments.index),
        "graph": Path(arguments.graph),
        "style": Path(arguments.style),
        "rows": Path(arguments.rows),
        "policy": Path(arguments.policy),
    }
    assets = {
        name: _asset_details(path)
        for name, path in paths.items()
        if name in ("index", "graph", "style")
    }
    assets["rows"] = _json_asset_details(paths["rows"])
    assets["policy"] = _json_asset_details(paths["policy"])
    rows = _load_rows(paths["rows"])
    policy, exact_policy = _load_policy(paths["policy"])

    recommender = WebRecommender(str(paths["index"]), enhance=False)
    gc.collect()
    core_rss = _process_rss()
    graph = CatalogArtistGraph(paths["graph"])
    graph_contract = _graph_contract(graph, assets["graph"])
    styles = CatalogStyleIndex(paths["style"])
    style_present = bool(
        len(styles.scene_names)
        and len(styles.vectors)
        and styles.vectors.shape[1] == len(styles.scene_names)
    )
    ranker = CatalogPolicyRanker(recommender, graph, styles, policy)
    gc.collect()
    resident_rss = _process_rss()
    load_seconds = time.perf_counter() - started

    for row in rows:
        if row >= len(recommender.titles):
            raise ResourceMeasurementError("query row is outside the catalogue")
    cold_started = time.perf_counter()
    first = ranker.recommend(rows[0])
    cold_seconds = time.perf_counter() - cold_started
    outputs = [first]
    warm_seconds: List[float] = []
    warm_count = max(int(arguments.warm_queries), MINIMUM_WARM_QUERIES)
    for position in range(warm_count):
        row = rows[(position + 1) % len(rows)]
        query_started = time.perf_counter()
        outputs.append(ranker.recommend(row))
        warm_seconds.append(time.perf_counter() - query_started)

    repeated = ranker.recommend(rows[0])
    first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8")
    repeated_bytes = json.dumps(
        repeated, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    modes: Dict[str, int] = {}
    fallback_count = 0
    for output in outputs + [repeated]:
        mode = str(output.get("query_mode", "missing"))
        modes[mode] = modes.get(mode, 0) + 1
        for result in output.get("results", []):
            source = str(result.get("rationale", {}).get("source", ""))
            if "fallback" in source.casefold():
                fallback_count += 1

    return {
        "schema_version": 1,
        "candidate": "compact_graph_first",
        "runtime_components": [
            "WebRecommender(enhance=False)",
            "CatalogArtistGraph",
            "CatalogStyleIndex",
            "CatalogPolicy",
            "CatalogPolicyRanker",
        ],
        "assets": assets,
        "policy": exact_policy,
        "process_start_rss_bytes": start_rss,
        "core_index_post_gc_rss_bytes": core_rss,
        "post_gc_resident_rss_bytes": resident_rss,
        "load_seconds": load_seconds,
        "first_recommendation_latency_ms": cold_seconds * 1000.0,
        "cold_recommendation_latency_ms": cold_seconds * 1000.0,
        "warm_latency": latency_statistics(warm_seconds),
        "query_modes": modes,
        "fallback_count": fallback_count,
        "errors": errors,
        "graph_contract": graph_contract,
        "style": {
            "present": style_present,
            "scene_count": len(styles.scene_names),
            "artist_count": len(styles.vectors),
        },
        "determinism": {
            "repetitions": 2,
            "passed": first_bytes == repeated_bytes,
            "output_sha256": hashlib.sha256(first_bytes).hexdigest(),
        },
    }


def run_measurement(
    index: Any,
    graph: Any,
    style: Any,
    rows: Any,
    policy: Any,
    *,
    warm_queries: int = MINIMUM_WARM_QUERIES,
    poll_interval_seconds: float = 0.01,
    python_executable: Optional[str] = None,
    popen_factory: Any = subprocess.Popen,
    psutil_module: Any = None,
) -> Dict[str, Any]:
    """Run the isolated worker, poll its process tree, and apply all gates."""
    if psutil_module is None:
        import psutil as psutil_module

    command = [
        python_executable or sys.executable,
        "-m",
        "soundalike.ml.catalog_resources_v8",
        "_worker",
        "--index",
        str(index),
        "--graph",
        str(graph),
        "--style",
        str(style),
        "--rows",
        str(rows),
        "--policy",
        str(policy),
        "--warm-queries",
        str(max(int(warm_queries), MINIMUM_WARM_QUERIES)),
    ]
    child = popen_factory(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Drain pipes concurrently while polling.  Waiting for process exit before
    # reading can deadlock on Windows when the JSON report exceeds pipe capacity.
    captured: Dict[str, Tuple[str, str]] = {}

    def drain() -> None:
        captured["streams"] = child.communicate()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    peak, samples = poll_peak_rss(child, psutil_module, poll_interval_seconds)
    reader.join()
    stdout, stderr = captured.get("streams", ("", ""))
    try:
        report = json.loads(stdout)
        if not isinstance(report, dict):
            raise TypeError
    except (json.JSONDecodeError, TypeError):
        report = {
            "errors": [
                {
                    "type": "WorkerProtocolError",
                    "message": "worker did not return one JSON object",
                }
            ]
        }
    if child.returncode:
        report.setdefault("errors", []).append(
            {
                "type": "WorkerExitError",
                "message": "worker exited with status %s" % child.returncode,
                "stderr": stderr[-2000:],
            }
        )
    report["peak_rss_bytes"] = int(peak)
    report["peak_poll_samples"] = int(samples)
    report["peak_poll_interval_seconds"] = float(poll_interval_seconds)
    return apply_resource_gates(report)


measure_resources = run_measurement


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("measure", "_worker"):
        child = subparsers.add_parser(command)
        child.add_argument("--index", required=True)
        child.add_argument("--graph", required=True)
        child.add_argument("--style", required=True)
        child.add_argument("--rows", required=True)
        child.add_argument("--policy", required=True)
        child.add_argument("--warm-queries", type=int, default=MINIMUM_WARM_QUERIES)
        if command == "measure":
            child.add_argument("--output", required=True)
            child.add_argument("--poll-interval", type=float, default=0.01)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "_worker":
        try:
            result = _worker_measure(arguments)
            code = 0
        except Exception as error:
            result = {
                "errors": [
                    {"type": type(error).__name__, "message": str(error)}
                ]
            }
            code = 1
        print(json.dumps(result, sort_keys=True))
        return code

    report = run_measurement(
        arguments.index,
        arguments.graph,
        arguments.style,
        arguments.rows,
        arguments.policy,
        warm_queries=arguments.warm_queries,
        poll_interval_seconds=arguments.poll_interval,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
