"""Integrity and compatibility tests for the mobile-only v15 successor."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

from soundalike.ml.human_aggregate_v10 import _load_bound
from soundalike.ml.human_eval_v10 import content_hash, file_hash
from soundalike.ml.human_eval_v11 import evaluator_handler
from soundalike.ml.human_eval_v15 import (
    TRUSTED_V15_FILES,
    TRUSTED_V15_LISTS,
    TRUSTED_V15_PROTOCOL,
    TRUSTED_V15_STATE,
    method_assignment_hash,
    semantic_order_hash,
    served_payload_hash,
    verify_pack,
)

ROOT = Path(__file__).parents[1]
GOAL = ROOT / ".goals" / "human-quality-recommendations"
V14 = GOAL / "protocol-v14-clap-human-development"
V15 = GOAL / "protocol-v15-clap-human-development"
V14_KEY = ROOT / "ml_data" / "clap_v14" / "human_eval" / "method-key-v14.json"
V15_KEY = ROOT / "ml_data" / "clap_v15" / "human_eval" / "method-key-v15.json"
V14_EVALUATOR = ROOT / "benchmarks" / "human_eval_v14.html"
V15_EVALUATOR = ROOT / "benchmarks" / "human_eval_v15.html"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v14_history_is_still_byte_exact():
    protocol = _load(V14 / "protocol-v14.json")
    assert file_hash(V14_EVALUATOR) == protocol["evaluator_sha256"]
    assert protocol["evaluator_sha256"] == (
        "b9cf6b127f0002b490c2a307d64324bd573d503b4c485d261ecc3c6517188689"
    )


@pytest.mark.skipif(
    shutil.which("ssh-keygen") is None, reason="ssh-keygen is required"
)
def test_committed_v15_pack_signature_hashes_and_zero_ratings():
    verified = verify_pack(
        V15,
        private_key=V15_KEY if V14_KEY.is_file() and V15_KEY.is_file() else None,
        evaluator=V15_EVALUATOR,
        require_trusted=True,
    )
    protocol, lists, state = (
        verified["protocol"],
        verified["lists"],
        verified["state"],
    )
    assert protocol["content_sha256"] == TRUSTED_V15_PROTOCOL
    assert lists["content_sha256"] == TRUSTED_V15_LISTS
    assert state["content_sha256"] == TRUSTED_V15_STATE
    assert protocol["rankings_state"] == lists["rankings_state"] == "RANKINGS_LOCKED"
    assert (
        protocol["ratings_count_at_freeze"]
        == lists["ratings_count_at_freeze"]
        == state["ratings_count_at_freeze"]
        == state["human_rater_exports_ingested"]
        == 0
    )
    assert all(file_hash(V15 / name) == digest for name, digest in TRUSTED_V15_FILES.items())


def test_v14_v15_served_order_content_blinding_and_roles_are_identical():
    if not V14_KEY.is_file() or not V15_KEY.is_file():
        pytest.skip("local gitignored method keys are required for private-role parity")
    old_lists = _load(V14 / "served-lists-v14.json")
    new_lists = _load(V15 / "served-lists-v15.json")
    old_key = _load(V14_KEY)
    new_key = _load(V15_KEY)
    supersession = _load(V15 / "state.json")["supersedes_v14"]

    assert old_lists["seeds"] == new_lists["seeds"]
    assert semantic_order_hash(old_lists) == semantic_order_hash(new_lists)
    assert served_payload_hash(old_lists) == served_payload_hash(new_lists)
    assert old_key["records"] == new_key["records"]
    assert old_key["blinding_salt_sha256"] == new_key["blinding_salt_sha256"]
    assert method_assignment_hash(old_key) == method_assignment_hash(new_key)
    assert supersession["ratings_discarded"] == supersession["ratings_migrated"] == 0
    assert supersession["ranking_order_parity"] is True
    assert supersession["served_payload_semantically_identical"] is True
    assert supersession["method_assignments_identical"] is True
    assert supersession["blinding_identifiers_retained"] is True
    assert supersession["candidate_pack_semantics_identical"] is True


def test_v15_protocol_records_predecessor_and_mobile_only_rationale():
    old_protocol = _load(V14 / "protocol-v14.json")
    old_lists = _load(V14 / "served-lists-v14.json")
    old_state = _load(V14 / "state.json")
    protocol = _load(V15 / "protocol-v15.json")
    supersession = protocol["supersedes_v14"]

    assert supersession["old_protocol_sha256"] == old_protocol["content_sha256"]
    assert supersession["old_lists_sha256"] == old_lists["content_sha256"]
    assert supersession["old_state_sha256"] == old_state["content_sha256"]
    assert supersession["old_evaluator_sha256"] == old_protocol["evaluator_sha256"]
    assert supersession["reason"] == "mobile-only evaluator layout and accessibility polish"
    assert protocol["evaluator_sha256"] == file_hash(V15_EVALUATOR)
    assert protocol["evaluator_sha256"] != old_protocol["evaluator_sha256"]


def test_v15_preview_metadata_preserves_committed_600_of_600_evidence():
    protocol = _load(V15 / "protocol-v15.json")
    audit_path = GOAL / "artifacts" / "human-eval-preview-audit-v14.json"
    audit = _load(audit_path)
    assert content_hash(audit) == protocol["preview_audit_content_sha256"]
    assert file_hash(audit_path) == protocol["preview_audit_file_sha256"]
    assert protocol["preview_ranked_positions_available"] == 600
    assert protocol["preview_ranked_positions_total"] == 600
    assert audit["ranked_positions"]["available"] == audit["ranked_positions"]["total"] == 600


def test_v15_evaluator_is_mobile_polished_without_reblinding():
    html = V15_EVALUATOR.read_text(encoding="utf-8")
    assert "schema_version:15" in html
    assert "schema_version!==15" in html
    assert "soundalike-human-v15" in html
    assert "human-ratings-v15-" in html
    assert "[LT]14-" in html
    assert "viewport-fit=cover" in html
    assert "@media(max-width:600px)" in html
    assert "prefers-reduced-motion:reduce" in html
    assert "production_baseline" not in html
    assert '"challenger"' not in html


def test_bundled_routes_serve_v15_locked_files():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        evaluator_handler(
            V15_EVALUATOR,
            V15 / "protocol-v15.json",
            V15 / "served-lists-v15.json",
            resolver=lambda _track_id: None,
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/", timeout=5) as response:
            assert b"Load bundled locked study" in response.read()
        with urlopen(base + "/protocol.json", timeout=5) as response:
            assert json.load(response)["schema_version"] == 15
        with urlopen(base + "/served-lists.json", timeout=5) as response:
            lists = json.load(response)
            assert lists["schema_version"] == 15
            assert lists["rankings_state"] == "RANKINGS_LOCKED"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(
    shutil.which("ssh-keygen") is None, reason="ssh-keygen is required"
)
def test_aggregator_accepts_v15_without_changing_method_roles():
    if not V14_KEY.is_file() or not V15_KEY.is_file():
        pytest.skip("local gitignored method keys are required for aggregation")
    protocol, lists, key, roles, collector = _load_bound(
        V15 / "protocol-v15.json",
        V15 / "served-lists-v15.json",
        V15_KEY,
    )
    old_key = _load(V14_KEY)
    assert protocol["schema_version"] == lists["schema_version"] == key["schema_version"] == 15
    assert roles == {row["list_id"]: row["method_role"] for row in old_key["records"]}
    assert collector.read_bytes() == (V14 / "collector_allowed_signers").read_bytes()


def test_no_private_key_material_in_v15_public_files():
    for path in V15.iterdir():
        assert "PRIVATE KEY" not in path.read_text(encoding="utf-8", errors="ignore")
    assert not any(path.name == "collector_signer" for path in V15.iterdir())
    assert hashlib.sha256((V15 / "collector_signer.pub").read_bytes()).hexdigest() == (
        "514d96fd9262e5e5f24ebdf5cc0287573f9e85b2f101cbdf5f48c12b598550f8"
    )
