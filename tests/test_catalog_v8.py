import json
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.catalog_policy import CatalogPolicy
from soundalike.ml.catalog_v8 import (
    DevelopmentProtocolError,
    _parser,
    audit_source_independence,
    run_development_cv,
    write_signed_development_protocol,
)


def _write_audit_inputs(tmp_path):
    records = {
        "source_policy": {"automated_evaluation": "ListenBrainz session-based"},
        "records": [
            {
                "id": "one",
                "query": {"artist": "Alpha", "deezer_track_id": 10},
                "source": {"publisher": "Deezer"},
                "sources": [
                    {"publisher": "Deezer"},
                    {"publisher": "ListenBrainz Labs"},
                ],
                "positives": [
                    {
                        "artist": "Beta",
                        "source_related_artist": "Beta",
                        "source_artist_id": 20,
                        "source_provider": "Deezer related artists",
                    },
                    {"artist": "Beta", "source_related_artist": "Beta"},
                    {"artist": "Gamma", "source_related_artist": "Gamma"},
                ],
            }
        ],
    }
    benchmark = tmp_path / "v7.json"
    benchmark.write_text(json.dumps(records), encoding="utf-8")
    np.savez(
        tmp_path / "graph.npz",
        artist_names=np.array(["alpha", "beta", "gamma"]),
        full_indices=np.array([[1, -1], [0, -1], [0, -1]], np.int32),
    )
    np.savez(
        tmp_path / "music.npz",
        artist_names=np.array(["alpha", "beta", "gamma"]),
        artist_vectors=np.array([[1, 0], [0.9, 0.1], [0, 1]], np.float32),
        catalog_rows=np.array([4, 5, 6]),
    )
    return benchmark, tmp_path / "graph.npz", tmp_path / "music.npz"


def test_audit_corrects_erratum_and_counts_independent_overlap(tmp_path):
    paths = _write_audit_inputs(tmp_path)
    result = audit_source_independence(*paths)
    assert result["signed_v7_erratum"]["records_with_deezer_primary"] == 1
    assert result["signed_v7_erratum"]["records_with_listenbrainz_secondary"] == 1
    assert result["deezer_directed_edges"]["unique_query_positive_artist_edges"] == 2
    assert result["lastfm_360k_vs_deezer"]["full_top_neighbor_edge_overlap"] == 1
    music = result["music4all_onion_vs_deezer"]
    assert music["learned_top96_artist_neighborhood_overlap"] == 2
    assert "not raw cooccurrence" in music["overlap_kind"]
    assert result["id_isolation"]["passed"]
    assert result["decision"]["unmasked_lastfm_full_direct_edges_allowed"]
    assert "diagnostics only" in result["decision"]["mask_policy"]


def test_audit_rejects_non_deezer_primary(tmp_path):
    benchmark, graph, music = _write_audit_inputs(tmp_path)
    value = json.loads(benchmark.read_text())
    value["records"][0]["source"]["publisher"] = "ListenBrainz"
    value["records"][0]["sources"][0]["publisher"] = "ListenBrainz"
    benchmark.write_text(json.dumps(value))
    with pytest.raises(DevelopmentProtocolError, match="not Deezer"):
        audit_source_independence(benchmark, graph, music)


def _fake_sign(directory: Path, state_path: Path):
    (directory / "signer.pub").write_text("ssh-ed25519 toy\n")
    (directory / "allowed_signers").write_text(
        "soundalike-protocol ssh-ed25519 toy\n"
    )
    (directory / "state.sig").write_bytes(b"signature")
    return {"algorithm": "test-Ed25519"}


def test_development_lock_hashes_inputs_and_never_opens_final(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    result = write_signed_development_protocol(
        tmp_path / "protocol-v8",
        {"source_path": str(source), "decision": {"unmasked": True}},
        {"source": {"cache": str(source)}, "asset_path": str(source)},
        policy_grid=[CatalogPolicy(0.1, 0.2, 0.3)],
        signing_helper=_fake_sign,
    )
    state = result["state"]
    protocol = result["protocol"]
    assert state["phase"] == "DEVELOPMENT_LOCKED"
    assert state["final_open_count"] == 0
    assert state["fresh_final_blocked"] and state["deployment_blocked"]
    assert protocol["policy"]["numeric_parameter_count"] == 3
    assert protocol["development_input_sha256"][str(source)]
    assert protocol["development_primary"]["formula"].startswith("0.80*nDCG")
    assert "16/20" in protocol["gates"]["direct_review_prerequisite"]
    assert (tmp_path / "protocol-v8" / "state.sig").is_file()


def test_signing_failure_removes_partial_protocol(tmp_path):
    def fail(_directory, _state):
        raise DevelopmentProtocolError("no ssh-keygen")

    directory = tmp_path / "protocol-v8"
    with pytest.raises(DevelopmentProtocolError, match="ssh-keygen"):
        write_signed_development_protocol(
            directory, {}, {}, signing_helper=fail
        )
    assert not directory.exists()


def test_cli_has_no_final_operation():
    parser = _parser()
    action = next(action for action in parser._actions if action.dest == "command")
    assert set(action.choices) == {"audit", "lock-development", "dev-cv"}
    with pytest.raises(SystemExit):
        parser.parse_args(["final"])


def test_dev_cv_precomputes_unique_query_and_reports_slices(tmp_path):
    query = {"title": "Query", "artist": "Alpha"}
    v6 = {
        "pairs": [
            {
                "id": str(number),
                "evidence_category": "category_a_human_songs_like",
                "split": "final",
                "scene": "rock" if number < 3 else "pop",
                "query": query,
                "target": {"title": "Target", "artist": "Beta"},
            }
            for number in range(5)
        ]
    }
    v7 = {
        "records": [
            {
                "id": "six",
                "split": "final",
                "scene": "jazz",
                "query": query,
                "source": {"publisher": "Deezer"},
                "positives": [
                    {"title": "Target", "artist": "Beta", "grade": 3}
                ],
            }
        ]
    }
    v6_path, v7_path = tmp_path / "v6.json", tmp_path / "v7.json"
    v6_path.write_text(json.dumps(v6))
    v7_path.write_text(json.dumps(v7))
    state = {
        "final_open_count": 1,
        "benchmark_sha256": __import__("hashlib").sha256(
            v7_path.read_bytes()
        ).hexdigest(),
    }
    for name in ("index.npz", "graph.npz", "style.npz"):
        (tmp_path / name).write_bytes(b"toy")

    class Recommender:
        titles = np.array(["Query", "Target", "Other"])
        artists = np.array(["Alpha", "Beta", "Gamma"])
        track_ids = np.array([1, 2, 3])

    class Style:
        @staticmethod
        def style_overlap(_left, right):
            return 1.0 if right == "Beta" else 0.0

    calls = []

    def precompute(_ranker, _production, row, candidate_limit):
        calls.append((row, candidate_limit))
        return {
            "production_rows": [2, 1],
            "graph_union_rows": [1, 2],
            "components": [
                {"row": 1, "G": 1.0, "A": 1.0, "S": 1.0, "source": "graph"},
                {"row": 2, "G": 0.0, "A": 0.0, "S": 0.0, "source": "audio_fallback"},
            ],
        }

    def report_builder(v6_doc, v7_doc, evaluator, policies):
        from soundalike.ml.catalog_cv import normalize_opened_benchmarks

        records = normalize_opened_benchmarks(v6_doc, v7_doc)["records"]
        policy = policies[0]
        return {
            "nested_5fold": {"final_policy": {
                "audio_weight": policy.audio_weight,
                "style_weight": policy.style_weight,
                "style_guard_min": policy.style_guard_min,
            }},
            "probe": evaluator(policy, records, records),
        }

    report = run_development_cv(
        v6_path,
        v7_path,
        state,
        tmp_path / "index.npz",
        tmp_path / "graph.npz",
        tmp_path / "style.npz",
        tmp_path / "report.json",
        policies=[CatalogPolicy(0.1, 0.1, 0.0)],
        recommender_factory=lambda _path: Recommender(),
        graph_factory=lambda _path: object(),
        style_factory=lambda _path: Style(),
        production_factory=lambda *_args: object(),
        ranker_factory=lambda *_args: object(),
        component_precomputer=precompute,
        report_builder=report_builder,
    )
    assert calls == [(0, 1000)]
    assert set(report["probe"]["per_scene"]) == {"jazz", "pop", "rock"}
    assert set(report["probe"]["per_axis"]) == {"taste_affinity"}
    selected = report["selected_policy_evaluation"]["aggregate_and_slices"]
    assert {"baseline", "challenger", "improved", "worsened"} <= set(selected)
    assert report["execution"]["unique_queries"] == 1
    assert json.loads((tmp_path / "report.json").read_text())["execution"]
