import json
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.catalog_resources_v8 import (
    GIB,
    PEAK_LIMIT_BYTES,
    RESIDENT_TARGET_BYTES,
    _graph_contract,
    _load_policy,
    _summarize_query_outputs,
    apply_resource_gates,
    latency_statistics,
    platform_limit_provenance,
    poll_peak_rss,
)

LINKED_PROJECT = {
    "projectId": "prj_dIRX8P1uu7SVtYJqkbImGLADIgBa",
    "orgId": "team_7KDYLvkpFCn35sRqOT8Ap96z",
    "projectName": "soundalike",
}


def _evidence(tier="Hobby", limit=2 * GIB, **changes):
    value = {
        **LINKED_PROJECT,
        "cli_attempts": [{"command": "vercel project inspect", "status": "ok"}],
        "api_attempts": [{"endpoint": "/v9/projects", "status": 200}],
        "tier_verified": True,
        "project_tier": tier,
        "actual_memory_limit_bytes": limit,
    }
    value.update(changes)
    return value


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
        "graph_contract": {
            "passed": True,
            "silent_fallback_declared": False,
        },
        "style": {"present": True},
        "determinism": {"passed": True},
    }


def test_byte_exact_gates_and_headroom():
    report = apply_resource_gates(_passing_report(), _evidence())
    assert report["passed"]
    assert report["headroom_bytes"] == GIB
    assert report["gates"]["peak"]["limit_bytes"] == int(1.5 * 1024 ** 3)
    assert report["gates"]["resident_target"]["target_bytes"] == int(1.1 * 1024 ** 3)

    over = _passing_report()
    over["peak_rss_bytes"] = PEAK_LIMIT_BYTES + 1
    assert not apply_resource_gates(over, _evidence())["passed"]


def test_resident_target_is_nonblocking_only_when_core_alone_exceeds_it():
    report = _passing_report()
    report["post_gc_resident_rss_bytes"] = RESIDENT_TARGET_BYTES + 1
    report["core_index_post_gc_rss_bytes"] = RESIDENT_TARGET_BYTES + 1
    gated = apply_resource_gates(report, _evidence())
    assert gated["passed"]
    assert gated["gates"]["resident_target"]["infeasible_nonblocking"]

    report = _passing_report()
    report["post_gc_resident_rss_bytes"] = RESIDENT_TARGET_BYTES + 1
    assert not apply_resource_gates(report, _evidence())["passed"]


def test_fallback_style_graph_and_determinism_are_hard_failures():
    for mutation in (
        lambda value: value.update(fallback_count=1),
        lambda value: value["style"].update(present=False),
        lambda value: value["graph_contract"].update(passed=False),
        lambda value: value["graph_contract"].update(
            silent_fallback_declared=True
        ),
        lambda value: value["determinism"].update(passed=False),
    ):
        report = _passing_report()
        mutation(report)
        assert not apply_resource_gates(report, _evidence())["passed"]


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


def test_full_only_graph_requires_complete_aligned_dual_source_metadata():
    indices = np.array([[1], [0]], dtype=np.int16)
    weights = np.ones((2, 1), dtype=np.float16)

    class Graph:
        artist_names = np.array(["a", "b"])
        variants = {"full": (indices, weights)}
        music4all_query_artist_ids = np.array([0, 1], dtype=np.int16)
        music4all_indices = indices
        music4all_weights = weights
        metadata = {
            "asset_type": "catalog_artist_graph_dual_source_runtime",
            "music4all_aligned_artists": 2,
            "runtime_contains_raw_vectors": False,
            "silent_fallback": False,
        }

    details = {
        "arrays": {
            "full_indices": {},
            "full_weights": {},
            "music4all_query_artist_ids": {},
            "music4all_indices": {},
            "music4all_weights": {},
        }
    }
    assert _graph_contract(Graph(), details)["passed"]

    del details["arrays"]["music4all_weights"]
    partial = _graph_contract(Graph(), details)
    assert not partial["music4all_dual_arrays_complete"]
    assert not partial["passed"]


def test_policy_loader_accepts_only_tau_sigma_audio_weight(tmp_path):
    exact = {"tau": 0.7, "sigma": 0.8, "audio_weight": 0.25}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"exact_policy": exact}), encoding="utf-8")
    policy, loaded = _load_policy(path)
    assert loaded == exact
    assert (policy.tau, policy.sigma, policy.audio_weight) == (0.7, 0.8, 0.25)

    path.write_text(
        json.dumps({"tau": 0.7, "sigma": 0.8, "style_weight": 0.25}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _load_policy(path)


def test_latency_stats_require_and_report_warm_samples():
    stats = latency_statistics([value / 1000 for value in range(1, 17)])
    assert stats["count"] == 16
    assert stats["mean_ms"] == pytest.approx(8.5)
    assert stats["p50_ms"] == pytest.approx(8.5)
    assert stats["p95_ms"] == pytest.approx(15.25)
    with pytest.raises(ValueError):
        latency_statistics([])


def test_unverified_tier_fails_closed_with_null_limit_and_headroom():
    value = platform_limit_provenance()
    assert value["documented_maximums_bytes"] == {
        "Hobby": 2 * GIB,
        "Pro": 4 * GIB,
        "Enterprise": 4 * GIB,
    }
    assert value["limit_used_bytes"] is None
    assert value["project_tier"] == "unknown"
    assert not value["project_tier_credential_verification"]["available"]
    assert value["documentation_as_of"] == "2026-07-01"
    report = apply_resource_gates(_passing_report())
    assert not report["passed"]
    assert report["headroom_bytes"] is None
    assert report["gates"]["platform"]["passed"] is False
    assert "verified" in report["gates"]["platform"]["reason"]


@pytest.mark.parametrize("tier,limit", [("Hobby", 2 * GIB), ("Pro", 4 * GIB)])
def test_verified_exact_matching_tier_passes_with_exact_headroom(tier, limit):
    report = apply_resource_gates(_passing_report(), _evidence(tier, limit))
    assert report["passed"]
    assert report["headroom_bytes"] == limit - GIB
    assert report["gates"]["platform"]["limit_bytes"] == limit
    assert report["platform_limit"]["linked_project"] == LINKED_PROJECT


def test_mismatched_linked_project_fails_even_when_rss_fits_hobby():
    evidence = _evidence(projectId="prj_wrong")
    provenance = platform_limit_provenance(evidence)
    assert not provenance["project_tier_credential_verification"]["available"]
    assert provenance["limit_used_bytes"] is None
    report = apply_resource_gates(_passing_report(), evidence)
    assert not report["passed"]
    assert report["headroom_bytes"] is None
    assert "does not match" in report["gates"]["platform"]["reason"]


def test_evidence_json_path_and_prebuilt_provenance_are_supported(tmp_path):
    path = tmp_path / "platform.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")
    provenance = platform_limit_provenance(path)
    report = apply_resource_gates(_passing_report(), provenance=provenance)
    assert report["passed"]
    assert report["headroom_bytes"] == GIB


def test_production_abstention_is_reported_but_not_counted_as_fallback():
    abstention = {
        "query": {"row": 7},
        "query_mode": "production_abstention",
        "gate": {"fired": False, "reason": "agreement_below_tau"},
        "results": [
            {"rationale": {"source": "production_abstention"}},
        ],
    }
    modes, fallback_count, gates = _summarize_query_outputs(
        [abstention], abstention
    )
    assert modes == {"production_abstention": 1}
    assert fallback_count == 0
    assert gates["production_abstention_rate"] == 1.0
    assert gates["gate_firing_rate"] == 0.0
    assert gates["reasons"] == {"agreement_below_tau": 1}


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
