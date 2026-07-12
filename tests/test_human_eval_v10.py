"""Focused tests for the iteration-10 blinded served-list evaluator."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from soundalike.ml.human_aggregate_v10 import (
    AggregateError,
    aggregate,
    main as aggregate_main,
    sign_export,
)
from soundalike.ml.human_eval_v10 import approve_export, content_hash, freeze_pack


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _synthetic_pack(tmp_path: Path):
    seed_id = "SYNTH-1"
    common = {"position": 1, "track_id": 99, "title": "Shared", "artist": "Other"}
    lists = []
    for alias, offset in (("A", 100), ("B", 200)):
        results = [dict(common)]
        results.extend({
            "position": position,
            "track_id": offset + position,
            "title": f"{alias} track {position}",
            "artist": f"Artist {position}",
            "same_artist": False,
            "preview_url": "https://preview.example/audio.mp3" if position == 2 else None,
        } for position in range(2, 6))
        # This source row must be filtered before selecting the five served results.
        results.extend([{
            "position": 6, "track_id": offset + 6, "title": "Seed reprise",
            "artist": "Seed Artist", "same_artist": True,
        }])
        lists.append({"alias": alias, "results": results})
    source = {
        "schema_version": 9,
        "protocol": "synthetic",
        "records": [{
            "id": seed_id, "scene": "test_scene",
            "query": {"title": "Seed", "artist": "Seed Artist", "track_id": 1},
            "lists": lists,
        }],
    }
    source["content_sha256"] = content_hash(source)
    key = {
        "schema_version": 9,
        "blind_lists_sha256": source["content_sha256"],
        "records": [
            {"id": seed_id, "alias": "A", "method_role": "production_baseline"},
            {"id": seed_id, "alias": "B", "method_role": "challenger"},
        ],
    }
    source_path = _write(tmp_path / "source.json", source)
    key_path = _write(tmp_path / "source-key.json", key)
    paths = freeze_pack(source_path, key_path, tmp_path / "frozen",
                        enforce_real_suite=False)
    return paths


def _export(paths, tmp_path: Path, *, provider="standalone_local_evaluator",
            ratings=True, rater="anon-abcdefghijklmnop", suffix="a") -> Path:
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    lists = json.loads(paths["lists"].read_text(encoding="utf-8"))
    started = datetime(2026, 7, 12, 18, 0, tzinfo=timezone.utc)
    rated = (started + timedelta(seconds=30)).isoformat()
    document = {
        "schema_version": 10,
        "source_kind": "human_listener",
        "provider": provider,
        "anonymous_rater_id": rater,
        "session_id": f"session-{suffix}-12345678",
        "protocol_sha256": protocol["content_sha256"],
        "served_lists_sha256": lists["content_sha256"],
        "local_session_key": "0123456789abcdef" * 4,
        "started_at": started.isoformat(),
        "exported_at": (started + timedelta(minutes=2)).isoformat(),
        "duration_ms": 120_000,
        "result_ratings": {},
        "list_ratings": {},
    }
    if ratings:
        for result in lists["seeds"][0]["results"]:
            document["result_ratings"][result["result_id"]] = {
                "similarity": "very_similar",
                "score_0_10": 9,
                "junk_or_version": False,
                "rated_at": rated,
                "interaction_ms": 1000,
            }
        for served_list in lists["seeds"][0]["lists"]:
            document["list_ratings"][served_list["list_id"]] = {
                "whole_list_coherence": "very_coherent",
                "unrelated_positions_1_to_3": 0,
                "rated_at": rated,
                "interaction_ms": 1000,
            }
    document["integrity_hmac_sha256"] = sign_export(
        document, document["local_session_key"]
    )
    output = _write(tmp_path / f"ratings-{suffix}.json", document)
    approve_export(output, paths["collector_private"])
    return output


def test_freeze_is_locked_blind_exact_and_shared(tmp_path):
    paths = _synthetic_pack(tmp_path)
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    public = json.loads(paths["lists"].read_text(encoding="utf-8"))
    private = json.loads(paths["key"].read_text(encoding="utf-8"))

    assert protocol["rankings_state"] == public["rankings_state"] == "RANKINGS_LOCKED"
    assert protocol["ratings_count_at_freeze"] == public["ratings_count_at_freeze"] == 0
    assert protocol["served_lists_sha256"] == public["content_sha256"]
    assert protocol["private_key_sha256"] == private["content_sha256"]
    assert protocol["collector_allowed_signers_sha256"] == hashlib.sha256(
        paths["collector_allowed_signers"].read_bytes()
    ).hexdigest()
    assert all(len(item["ranking"]) == 5 for item in public["seeds"][0]["lists"])
    assert len(public["seeds"][0]["results"]) == 9  # shared track represented once
    assert sum(
        row["result_id"] == public["seeds"][0]["lists"][0]["ranking"][0]["result_id"]
        for item in public["seeds"][0]["lists"] for row in item["ranking"]
    ) == 2
    public_text = paths["lists"].read_text(encoding="utf-8")
    protocol_text = paths["protocol"].read_text(encoding="utf-8")
    for secret in ("production_baseline", "challenger", '"method_role"'):
        assert secret not in public_text
        assert secret not in protocol_text


def test_public_html_has_no_role_disclosure_and_is_standalone():
    html_path = Path(__file__).parents[1] / "benchmarks" / "human_eval_v10.html"
    html = html_path.read_text(encoding="utf-8")
    assert "production_baseline" not in html
    assert "challenger" not in html
    assert "localStorage" in html and "integrity_hmac_sha256" in html
    assert "https://cdn" not in html and "<script src=" not in html
    assert "unrelated_positions_1_to_3" in html
    assert "junk_or_version" in html


def test_rejects_proxy_and_no_rating_exports(tmp_path):
    paths = _synthetic_pack(tmp_path)
    proxy = _export(paths, tmp_path, provider="model_proxy", suffix="proxy")
    with pytest.raises(AggregateError, match="forbidden"):
        aggregate(paths["protocol"], paths["lists"], paths["key"], [proxy])

    hidden_proxy = _export(paths, tmp_path, suffix="hidden-proxy")
    payload = json.loads(hidden_proxy.read_text(encoding="utf-8"))
    next(iter(payload["result_ratings"].values()))["source_dataset"] = "Music4All"
    payload.pop("integrity_hmac_sha256")
    payload["integrity_hmac_sha256"] = sign_export(
        payload, payload["local_session_key"]
    )
    _write(hidden_proxy, payload)
    approve_export(hidden_proxy, paths["collector_private"])
    with pytest.raises(AggregateError, match="forbidden"):
        aggregate(
            paths["protocol"], paths["lists"], paths["key"], [hidden_proxy]
        )

    unsigned = _export(paths, tmp_path, suffix="unsigned")
    Path(str(unsigned) + ".sig").unlink()
    with pytest.raises(AggregateError, match="collector approval"):
        aggregate(paths["protocol"], paths["lists"], paths["key"], [unsigned])

    empty = _export(paths, tmp_path, ratings=False, suffix="empty")
    with pytest.raises(AggregateError, match="no ratings"):
        aggregate(paths["protocol"], paths["lists"], paths["key"], [empty])
    with pytest.raises(AggregateError, match="no rater exports"):
        aggregate(paths["protocol"], paths["lists"], paths["key"], [])

    output = tmp_path / "sonic_human.json"
    output.write_text("stale", encoding="utf-8")
    assert aggregate_main([
        "--protocol", str(paths["protocol"]), "--lists", str(paths["lists"]),
        "--key", str(paths["key"]), "--output", str(output),
    ]) == 2
    assert not output.exists()
    output.write_text("stale", encoding="utf-8")
    assert aggregate_main([
        "--protocol", str(paths["protocol"]), "--lists", str(paths["lists"]),
        "--key", str(paths["key"]), "--exports", str(empty), "--output", str(output),
    ]) == 2
    assert not output.exists()


def test_aggregate_is_deterministic_and_dedupes_rater_seed(tmp_path):
    paths = _synthetic_pack(tmp_path)
    first = _export(paths, tmp_path, suffix="first")
    duplicate = _export(paths, tmp_path, suffix="second")
    one = aggregate(paths["protocol"], paths["lists"], paths["key"],
                    [duplicate, first])
    two = aggregate(paths["protocol"], paths["lists"], paths["key"],
                    [first, duplicate])
    assert one == two
    assert one["valid_export_count"] == 2
    assert one["deduplicated_rater_seed_count"] == 1
    assert one["partial_rater_seed_count"] == 0
    assert one["paired_bootstrap_challenger_minus_baseline_ndcg_at_5"][
        "n_pairs"
    ] == 1
    assert len(one["per_seed"]) == 1


def test_paired_bootstrap_never_pairs_different_raters(tmp_path):
    paths = _synthetic_pack(tmp_path)
    public = json.loads(paths["lists"].read_text(encoding="utf-8"))
    private = json.loads(paths["key"].read_text(encoding="utf-8"))
    role_by_list = {
        item["method_role"]: item["list_id"] for item in private["records"]
    }
    rankings = {
        item["list_id"]: [row["result_id"] for row in item["ranking"]]
        for item in public["seeds"][0]["lists"]
    }
    exports = []
    for suffix, rater, role in (
        ("baseline-only", "anon-baseline-abcdefghijkl", "production_baseline"),
        ("challenger-only", "anon-challenger-abcdefghijk", "challenger"),
    ):
        path = _export(paths, tmp_path, suffix=suffix, rater=rater)
        document = json.loads(path.read_text(encoding="utf-8"))
        list_id = role_by_list[role]
        allowed = set(rankings[list_id])
        document["result_ratings"] = {
            key: value for key, value in document["result_ratings"].items()
            if key in allowed
        }
        document["list_ratings"] = {
            list_id: document["list_ratings"][list_id]
        }
        document.pop("integrity_hmac_sha256")
        document["integrity_hmac_sha256"] = sign_export(
            document, document["local_session_key"]
        )
        _write(path, document)
        approve_export(path, paths["collector_private"])
        exports.append(path)
    report = aggregate(
        paths["protocol"], paths["lists"], paths["key"], exports
    )
    assert report[
        "paired_bootstrap_challenger_minus_baseline_ndcg_at_5"
    ]["n_pairs"] == 0
    assert all(not row["paired_complete"] for row in report["per_rater_seed"])


def test_real_v9_suite_freezes_60_seeds_13_scenes_when_present(tmp_path):
    root = Path(__file__).parents[1]
    artifact_dir = root / ".goals" / "human-quality-recommendations" / "artifacts"
    source = artifact_dir / "catalog-powered-blind-lists-v9.json"
    key = artifact_dir / "catalog-powered-blind-key-v9.json"
    if not source.exists() or not key.exists():
        pytest.skip("real v9 artifacts are not present in this checkout")
    paths = freeze_pack(source, key, tmp_path / "real")
    public = json.loads(paths["lists"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    assert public["seed_count"] == protocol["seed_count"] == 60
    assert public["scene_count"] == protocol["scene_count"] == 13
    assert public["content_sha256"] == protocol["served_lists_sha256"]
    assert all(len(item["ranking"]) == 5
               for seed in public["seeds"] for item in seed["lists"])
    assert source.read_bytes()  # the immutable source was consumed, never rewritten


def test_committed_protocol_is_locked_signed_and_keeps_role_key_private():
    root = Path(__file__).parents[1]
    directory = (
        root / ".goals" / "human-quality-recommendations"
        / "protocol-v10-human-development"
    )
    protocol = json.loads((directory / "protocol-v10.json").read_text(encoding="utf-8"))
    public = json.loads((directory / "served-lists-v10.json").read_text(encoding="utf-8"))
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    assert protocol["served_lists_sha256"] == public["content_sha256"]
    assert state["phase"] == "RANKINGS_LOCKED"
    assert state["ratings_count_at_freeze"] == 0
    assert state["human_rater_exports_ingested"] == 0
    assert state["sonic_human_report_exists"] is False
    assert state["production_deployment_blocked"] is True
    assert state["evaluator_sha256"] == hashlib.sha256(
        (root / "benchmarks" / "human_eval_v10.html").read_bytes()
    ).hexdigest()
    assert not (directory / "method-key-v10.json").exists()
    assert not (directory / "collector_signer").exists()
    assert (directory / "collector_signer.pub").is_file()
    assert (directory / "collector_allowed_signers").is_file()
    assert "production_baseline" not in (directory / "served-lists-v10.json").read_text(
        encoding="utf-8"
    )
    executable = shutil.which("ssh-keygen")
    if executable is None:
        pytest.skip("ssh-keygen is unavailable")
    verified = subprocess.run(
        [
            executable, "-Y", "verify",
            "-f", str(directory / "allowed_signers"),
            "-I", "soundalike-human-eval",
            "-n", "soundalike-human-eval",
            "-s", str(directory / "state.sig"),
        ],
        input=(directory / "state.json").read_bytes(),
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr.decode(errors="replace")
