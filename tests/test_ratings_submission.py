"""Audit, parity, privacy, and migration tests for the signed v17 successor."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / ".goals" / "human-quality-recommendations"
V16 = GOAL / "protocol-v16-hosted-human-development"
V17 = GOAL / "protocol-v17-submission-human-development"
DEPLOY = ROOT / "webapp" / "evaluate"
EVALUATOR = DEPLOY / "index.html"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(document: dict) -> str:
    return hashlib.sha256(
        _canonical({key: value for key, value in document.items() if key != "content_sha256"})
    ).hexdigest()


def _served_payload(document: dict) -> dict:
    return {
        key: document[key]
        for key in (
            "rankings_state",
            "ratings_count_at_freeze",
            "seed_count",
            "scene_count",
            "results_per_method",
            "same_artist_filtered",
            "shared_results_rated_once",
            "stable_id_field",
            "preview_urls_resolved_at_freeze",
            "audio_access",
            "seeds",
        )
    }


def test_v17_hashes_bind_evaluator_protocol_lists_and_zero_rating_freeze():
    protocol = _load(V17 / "protocol-v17.json")
    lists = _load(V17 / "served-lists-v17.json")
    state = _load(V17 / "state.json")

    for document in (protocol, lists, state):
        assert document["schema_version"] == 17
        assert document["rankings_state"] == "RANKINGS_LOCKED"
        assert _content_hash(document) == document["content_sha256"]
    assert protocol["served_lists_sha256"] == lists["content_sha256"]
    assert state["served_lists_sha256"] == lists["content_sha256"]
    assert state["protocol_sha256"] == protocol["content_sha256"]
    assert protocol["evaluator_sha256"] == state["evaluator_sha256"]
    assert protocol["evaluator_sha256"] == _file_hash(EVALUATOR)
    assert protocol["ratings_count_at_freeze"] == 0
    assert lists["ratings_count_at_freeze"] == 0
    assert state["ratings_count_at_freeze"] == 0
    assert state["human_rater_exports_ingested"] == 0


def test_v16_to_v17_preserves_exact_public_payload_and_blind_commitments():
    old_protocol = _load(V16 / "protocol-v16.json")
    old_lists = _load(V16 / "served-lists-v16.json")
    old_state = _load(V16 / "state.json")
    protocol = _load(V17 / "protocol-v17.json")
    lists = _load(V17 / "served-lists-v17.json")
    state = _load(V17 / "state.json")
    successor = protocol["supersedes_v16"]

    assert successor == state["supersedes_v16"]
    assert successor["old_protocol_sha256"] == old_protocol["content_sha256"]
    assert successor["old_lists_sha256"] == old_lists["content_sha256"]
    assert successor["old_state_sha256"] == old_state["content_sha256"]
    assert _served_payload(lists) == _served_payload(old_lists)
    assert lists["seeds"] == old_lists["seeds"]
    assert lists["semantic_order_sha256"] == old_lists["semantic_order_sha256"]
    assert state["method_assignment_sha256"] == old_state["method_assignment_sha256"]
    assert state["blinding_salt_sha256"] == old_state["blinding_salt_sha256"]
    assert (
        state["collector_public_key_sha256"]
        == old_state["collector_public_key_sha256"]
    )
    assert (V17 / "collector_signer.pub").read_bytes() == (
        V16 / "collector_signer.pub"
    ).read_bytes()
    assert (V17 / "collector_allowed_signers").read_bytes() == (
        V16 / "collector_allowed_signers"
    ).read_bytes()
    for flag in (
        "ranking_order_parity",
        "served_payload_semantically_identical",
        "method_assignments_identical",
        "blinding_salt_identical",
        "opaque_identifiers_retained",
        "candidate_pack_semantics_identical",
        "collector_public_trust_identical",
    ):
        assert successor[flag] is True
    assert successor["recommendation_behavior_changed"] is False


def test_known_predecessor_rating_provenance_is_honest_and_not_in_freeze_count():
    protocol = _load(V17 / "protocol-v17.json")
    state = _load(V17 / "state.json")
    expected = {
        "known_external_exports_observed": 1,
        "known_predecessor_result_ratings": 5,
        "ratings_discarded": 0,
        "ratings_count_at_ranking_freeze": 0,
        "exports_ingested_at_ranking_freeze": 0,
    }
    for document in (protocol, state):
        provenance = document["known_rating_provenance"]
        assert {key: provenance[key] for key in expected} == expected
        assert "observed after the rankings were locked" in provenance["note"]
        assert "not committed or counted as ingested evidence" in provenance["note"]
    successor = state["supersedes_v16"]
    assert successor["known_external_exports_observed"] == 1
    assert successor["known_predecessor_result_ratings"] == 5
    assert successor["predecessor_result_ratings_preserved_by_migration"] == 5
    assert successor["ratings_discarded"] == 0
    assert successor["browser_autosave_migration_supported"] is True


def test_v17_deploy_contains_only_public_evaluator_payload():
    assert set(path.name for path in DEPLOY.iterdir()) == {
        "index.html",
        "protocol.json",
        "served-lists.json",
    }
    assert (DEPLOY / "protocol.json").read_bytes() == (
        V17 / "protocol-v17.json"
    ).read_bytes()
    assert (DEPLOY / "served-lists.json").read_bytes() == (
        V17 / "served-lists-v17.json"
    ).read_bytes()
    forbidden_keys = {
        "method_role",
        "method_identity",
        "unblinding_map",
        "signing_private_key",
        "private_method_key",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key.lower() not in forbidden_keys
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(_load(DEPLOY / "served-lists.json"))
    for path in (*V17.iterdir(), *DEPLOY.iterdir()):
        if path.is_file():
            assert b"BEGIN OPENSSH PRIVATE KEY" not in path.read_bytes()
            assert b"BEGIN PRIVATE KEY" not in path.read_bytes()
    assert not any("private" in path.name and "metadata" not in path.name for path in V17.iterdir())


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen required")
def test_v17_one_time_ed25519_signature_verifies_and_metadata_is_bound():
    metadata = _load(V17 / "signature-metadata.json")
    assert metadata["state_sha256"] == _file_hash(V17 / "state.json")
    assert metadata["state_content_sha256"] == _load(V17 / "state.json")[
        "content_sha256"
    ]
    assert metadata["signer_public_key_sha256"] == _file_hash(V17 / "signer.pub")
    assert metadata["allowed_signers_sha256"] == _file_hash(V17 / "allowed_signers")
    assert metadata["signature_sha256"] == _file_hash(V17 / "state.sig")
    result = subprocess.run(
        [
            shutil.which("ssh-keygen"),
            "-Y",
            "verify",
            "-f",
            str(V17 / "allowed_signers"),
            "-I",
            "soundalike-human-eval-v17",
            "-n",
            "soundalike-human-eval-v17",
            "-s",
            str(V17 / "state.sig"),
        ],
        input=(V17 / "state.json").read_bytes(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_v16_autosave_migration_is_exact_hash_and_id_gated():
    html = EVALUATOR.read_text(encoding="utf-8")
    assert "soundalike-human-v16-current:${V16_PROTOCOL_SHA256}" in html
    assert "saved.protocol_sha256!==protocolHash" in html
    assert "saved.served_lists_sha256!==listsHash" in html
    assert "validSessionShape(old,16))return migrateV16(old)" in html
    assert "from_schema_version:16" in html
    assert 'from_provider:"hosted_client_only_evaluator"' in html
    assert 'schema_version:17,provider:"hosted_private_submission_evaluator"' in html
    assert "validAutosaveIds(saved)" in html
    assert "delete" not in html[
        html.index("function restoreAutosave()") : html.index("function previewOrigin()")
    ]


def test_submission_is_manual_consent_gated_and_privacy_bounded():
    html = EVALUATOR.read_text(encoding="utf-8")
    assert '<input id="consent" type="checkbox">' in html
    assert '<button id="submit" type="button" disabled>' in html
    assert "complete>0&&$(\"consent\").checked" in html
    assert 'body:JSON.stringify({consent:true,ratings:submissionPayload})' in html
    assert 'method:"POST",credentials:"omit",cache:"no-store",referrerPolicy:"no-referrer"' in html
    assert "DOMContentLoaded\",submitRatings" not in html
    assert "sendBeacon" not in html
    assert "Export JSON as a manual fallback" in html
    for disclosure in (
        "IP address",
        "browser/user-agent",
        "Spotify identity",
        "email",
        "cookies",
    ):
        assert disclosure in html


def test_blob_dependency_route_and_firewall_guidance_are_present():
    package = _load(ROOT / "webapp" / "package.json")
    lock = _load(ROOT / "webapp" / "package-lock.json")
    assert package["dependencies"] == {"@vercel/blob": "2.6.1"}
    assert lock["packages"]["node_modules/@vercel/blob"]["version"] == "2.6.1"
    config = _load(ROOT / "webapp" / "vercel.json")
    assert config["functions"]["api/ratings.js"]["maxDuration"] == 15
    deploy_docs = (ROOT / "webapp" / "DEPLOY.md").read_text(encoding="utf-8")
    assert "private Vercel Blob store" in deploy_docs
    assert "`BLOB_STORE_ID`" in deploy_docs
    assert "`VERCEL_OIDC_TOKEN`" in deploy_docs
    assert "`BLOB_READ_WRITE_TOKEN`" in deploy_docs
    assert "Vercel Firewall rate-limit" in deploy_docs
    assert "not authentication" in deploy_docs
    assert "There is no public GET" in deploy_docs
