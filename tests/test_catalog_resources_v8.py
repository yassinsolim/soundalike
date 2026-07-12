from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.catalog_resources_v8 import (
    CONSERVATIVE_PLATFORM_LIMIT_BYTES,
    GIB,
    PEAK_LIMIT_BYTES,
    RESIDENT_TARGET_BYTES,
    _graph_contract,
    apply_resource_gates,
    latency_statistics,
    platform_limit_provenance,
    poll_peak_rss,
)


class _Memory:
    def __init__(self, rss):
        self.rss = rss


class _Process:
    def __init__(self, samples, descendants=(), pid=10):
        self.pid = pid
        self.samples = iter(samples)
        self.value = 0
        self.descendants = descendants

    def memory_info(self):
        self.value = next(self.samples, self.value)
        return _Memory(self.value)

    def children(self, recursive=True):
        assert recursive
        return list(self.descendants)


class _Child:
    pid = 10

    def __init__(self):
        self.polls = iter((None, None, 0))

    def poll(self):
        return next(self.polls, 0)


def test_peak_polling_includes_descendants(monkeypatch):
    parent = _Process([100, 200, 150, 0])
    descendant = _Process([50, 400, 0, 0], pid=11)

    class Psutil:
        @staticmethod
        def Process(pid):
            assert pid == 10
            parent.descendants = (descendant,)
            return parent

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    peak, samples = poll_peak_rss(_Child(), Psutil, 0.001)
    assert peak == 600
    assert samples >= 4


def _passing_report():
    return {
        "peak_rss_bytes": GIB,
        "post_gc_resident_rss_bytes": GIB,
        "core_index_post_gc_rss_bytes": GIB,
        "fallback_count": 0,
        "errors": [],
        "graph_contract": {"passed": True},
        "style": {"present": True},
        "determinism": {"passed": True},
    }


def test_byte_exact_gates_and_headroom():
    report = apply_resource_gates(_passing_report())
    assert report["passed"]
    assert report["headroom_bytes"] == GIB
    assert report["gates"]["peak"]["limit_bytes"] == int(1.5 * 1024 ** 3)
    assert report["gates"]["resident_target"]["target_bytes"] == int(1.1 * 1024 ** 3)

    over = _passing_report()
    over["peak_rss_bytes"] = PEAK_LIMIT_BYTES + 1
    assert not apply_resource_gates(over)["passed"]


def test_resident_target_is_nonblocking_only_when_core_alone_exceeds_it():
    report = _passing_report()
    report["post_gc_resident_rss_bytes"] = RESIDENT_TARGET_BYTES + 1
    report["core_index_post_gc_rss_bytes"] = RESIDENT_TARGET_BYTES + 1
    gated = apply_resource_gates(report)
    assert gated["passed"]
    assert gated["gates"]["resident_target"]["infeasible_nonblocking"]

    report = _passing_report()
    report["post_gc_resident_rss_bytes"] = RESIDENT_TARGET_BYTES + 1
    assert not apply_resource_gates(report)["passed"]


def test_fallback_style_graph_and_determinism_are_hard_failures():
    for mutation in (
        lambda value: value.update(fallback_count=1),
        lambda value: value["style"].update(present=False),
        lambda value: value["graph_contract"].update(passed=False),
        lambda value: value["determinism"].update(passed=False),
    ):
        report = _passing_report()
        mutation(report)
        assert not apply_resource_gates(report)["passed"]


def test_masked_graph_alias_is_detected_not_silently_accepted():
    indices = np.array([[1], [0]], dtype=np.int16)
    weights = np.ones((2, 1), dtype=np.float16)

    class Graph:
        variants = {
            "full": (indices, weights),
            "twohop": (indices.copy(), weights.copy()),
        }
        metadata = {"silent_fallback": False}

    details = {
        "arrays": {
            "full_indices": {},
            "full_weights": {},
            "twohop_indices": {},
            "twohop_weights": {},
        }
    }
    contract = _graph_contract(Graph(), details)
    assert contract["masked_variants_alias_full"] == ["twohop"]
    assert not contract["passed"]


def test_latency_stats_require_and_report_warm_samples():
    stats = latency_statistics([value / 1000 for value in range(1, 17)])
    assert stats["count"] == 16
    assert stats["mean_ms"] == pytest.approx(8.5)
    assert stats["p50_ms"] == pytest.approx(8.5)
    assert stats["p95_ms"] == pytest.approx(15.25)
    with pytest.raises(ValueError):
        latency_statistics([])


def test_limit_provenance_is_honest_and_conservative():
    value = platform_limit_provenance()
    assert value["documented_maximums_bytes"] == {
        "Hobby": 2 * GIB,
        "Pro": 4 * GIB,
        "Enterprise": 4 * GIB,
    }
    assert value["limit_used_bytes"] == CONSERVATIVE_PLATFORM_LIMIT_BYTES
    assert value["project_tier"] == "unknown"
    assert not value["project_tier_credential_verification"]["available"]
    assert value["documentation_as_of"] == "2026-07-01"


def test_runtime_module_has_no_forbidden_runtime_dependencies():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "soundalike"
        / "ml"
        / "catalog_resources_v8.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "Production" + "Ranker",
        "Pair" + "Resolver",
        "Music" + "4All",
    )
    assert all(name not in source for name in forbidden)
    assert "enhance=False" in source
