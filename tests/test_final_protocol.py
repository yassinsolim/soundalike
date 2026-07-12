import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.final_protocol import (
    ProtocolError,
    content_sha256,
    file_sha256,
    commit_rankings,
    freeze_protocol,
    open_final_once,
    validate_benchmark,
    _verify_state_signature,
)


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_v5_benchmark_passes_protocol_audit():
    benchmark = json.loads(
        Path("benchmarks/soundalike_pairs.v5.json").read_text(encoding="utf-8")
    )
    audit = validate_benchmark(benchmark)
    assert audit["category_a_pairs"] >= 100
    assert audit["final_pairs"] == 40
    assert audit["scenes"] >= 15
    assert audit["artist_overlap"] == []


def test_freeze_is_immutable_when_state_exists(tmp_path):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    _write(protocol / "state.json", {"status": "FROZEN"})
    with pytest.raises(ProtocolError, match="already exists"):
        freeze_protocol(Path("missing.json"), Path("missing.npz"), protocol)


def test_rankings_cannot_commit_before_method_lock(tmp_path):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    state = {"status": "FROZEN", "final_open_count": 0}
    state["integrity_signature"] = content_sha256(state)
    state["signature_algorithm"] = "SHA-256 canonical JSON"
    _write(protocol / "state.json", state)
    winner = tmp_path / "winner.json"
    _write(winner, {"target_labels_compared": False, "records": []})
    with pytest.raises(ProtocolError, match="only after method lock"):
        commit_rankings(protocol, winner)


def test_final_can_be_opened_exactly_once(tmp_path):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    index_path = tmp_path / "index.npz"
    np.savez(
        index_path,
        titles=np.asarray(["Seed", "Target", "Other"]),
        artists=np.asarray(["A", "B", "C"]),
    )
    pair = {
        "id": "F1",
        "scene": "test",
        "query": {"title": "Seed", "artist": "A"},
        "target": {"title": "Target", "artist": "B"},
    }
    final_manifest = {"pairs": [pair]}
    final_manifest["content_sha256"] = content_sha256(final_manifest["pairs"])
    _write(protocol / "final-test-manifest.json", final_manifest)
    ranking = [
        {"rank": 1, "row": 1, "title": "Target", "artist": "B"},
        {"rank": 2, "row": 2, "title": "Other", "artist": "C"},
    ]
    baseline_records = [{
        "pair_id": "F1",
        "rankings": {
            "production_baseline": ranking[::-1],
            "iteration3_deployed": ranking[::-1],
            "raw_encoder": ranking[::-1],
            "audio_priors_zero": ranking[::-1],
        },
    }]
    _write(
        protocol / "frozen-baseline-rankings.json",
        {
            "records": baseline_records,
            "content_sha256": content_sha256(baseline_records),
        },
    )
    method = tmp_path / "method.json"
    _write(method, {"method_id": "locked"})
    method_hash = hashlib.sha256(method.read_bytes()).hexdigest()
    winner = tmp_path / "winner.json"
    _write(
        winner,
        {
            "target_labels_compared": False,
            "method_manifest_sha256": method_hash,
            "records": [{"pair_id": "F1", "ranking": ranking}],
        },
    )
    benchmark = tmp_path / "benchmark.json"
    _write(
        benchmark,
        {
            "metric_policy": {
                "success": {
                    "minimum_relative_primary_gain": 0.2,
                    "paired_bootstrap_ci95_low_must_exceed": 0.0,
                    "minimum_improved_pairs": 1,
                }
            }
        },
    )
    dev_report = tmp_path / "dev.json"
    _write(dev_report, {"split": "development"})
    manifest_path = protocol / "final-test-manifest.json"
    baseline_path = protocol / "frozen-baseline-rankings.json"
    state = {
        "status": "METHOD_LOCKED",
        "final_open_count": 0,
        "method_id": "locked",
        "method_manifest_path": str(method),
        "method_manifest_sha256": method_hash,
        "dev_report_path": str(dev_report),
        "dev_report_sha256": file_sha256(dev_report),
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "benchmark_path": str(benchmark),
        "benchmark_sha256": file_sha256(benchmark),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "baseline_path": str(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "asset_hashes": {},
    }
    state["integrity_signature"] = content_sha256(state)
    state["signature_algorithm"] = "SHA-256 canonical JSON"
    _write(protocol / "state.json", state)
    committed = commit_rankings(protocol, winner)
    assert committed["status"] == "RANKINGS_LOCKED"
    winner_document = json.loads(winner.read_text(encoding="utf-8"))
    winner_document["records"][0]["ranking"][0]["rank"] = 99
    _write(winner, winner_document)
    with pytest.raises(ProtocolError, match="winner rankings.*hash mismatch"):
        open_final_once(protocol, winner, tmp_path / "tampered.json")
    winner_document["records"][0]["ranking"][0]["rank"] = 1
    _write(winner, winner_document)

    report = open_final_once(protocol, winner, tmp_path / "report.json")
    state = json.loads((protocol / "state.json").read_text(encoding="utf-8"))
    assert report["open_number"] == 1
    assert report["comparison_to_production_baseline"]["passes"][
        "scene_no_regression"
    ]
    assert state["status"] == "FINALIZED"
    assert state["final_open_count"] == 1

    with pytest.raises(ProtocolError, match="already been opened"):
        open_final_once(protocol, winner, tmp_path / "second.json")


def test_state_signature_rejects_tampering():
    state = {"status": "FROZEN", "final_open_count": 0}
    state["integrity_signature"] = content_sha256(state)
    state["signature_algorithm"] = "SHA-256 canonical JSON"
    _verify_state_signature(state)
    state["final_open_count"] = 1
    with pytest.raises(ProtocolError, match="signature mismatch"):
        _verify_state_signature(state)


def test_committed_final_state_has_valid_ed25519_seal():
    if shutil.which("ssh-keygen") is None:
        pytest.skip("ssh-keygen is unavailable")
    state_path = Path(
        ".goals/human-quality-recommendations/protocol-v5/state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state_signature(state, state_path)
