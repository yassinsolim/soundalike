"""Deployment-boundary tests for the full-track v2 pilot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
V1 = WEB / "evaluate-v1"
V2 = WEB / "evaluate-v2"
PACK_SHA = "1980da60810959e7cdd24f39bd7142c8e34c76dab633c705976b85e49b297023"
PROTOCOL_SHA = "1f7a3cc48ecb62f85d3c3d65fa0f0c0fa5cd73eeccd67049d7bc5e84d1dcc227"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _content_hash(document: dict) -> str:
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _file_hash(path: Path) -> str:
    # The reviewed archive uses canonical Git bytes, independent of checkout
    # line endings. Content changes still require an explicit pin update.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()




def test_v2_archive_deploys_the_validated_public_pack_and_protocol():
    source = (
        ROOT
        / ".goals"
        / "fulltrack-v2-pilot"
        / "artifacts"
        / "public-pack.json"
    )
    deployed = V2 / "pilot-pack.json"
    protocol = _load(V2 / "protocol-v2.json")
    pack = _load(deployed)

    assert deployed.read_bytes().replace(b"\r\n", b"\n") == source.read_bytes().replace(
        b"\r\n", b"\n"
    )
    assert _content_hash(pack) == pack["content_sha256"] == PACK_SHA
    assert _content_hash(protocol) == protocol["content_sha256"] == PROTOCOL_SHA
    assert protocol["pilot_pack_sha256"] == PACK_SHA
    assert protocol["submission_endpoint"] == "/api/ratings-v2"
    assert protocol["private_blob_prefix"] == "human-ratings/fulltrack-v2/"
    assert protocol["local_storage_namespace"] == "soundalike-fulltrack-v2"
    assert protocol["production_recommendation_changed"] is False
    assert pack["research_only"] is True
    assert pack["promotion_allowed"] is False


def test_v17_evaluator_is_byte_preserved_at_the_v1_route():
    assert {
        path.name: _file_hash(path) for path in V1.iterdir() if path.is_file()
    } == {
        "index.html": "0421f2dd9207b476a4fb44e71309ef2d3ea9aab93465587e1fb2d8bdf61b6c68",
        "protocol.json": "02fb2baa60d3a7bc2ae67f198ea470f5cd1837ff6c9704526f4c41b3281975a1",
        "served-lists.json": "1253cfd0501f320bf6cda4d451509d7b2fa552a1ecbe5636a9e3477137850f20",
    }
    html = (V1 / "index.html").read_text(encoding="utf-8")
    assert "soundalike-human-v17" in html
    assert 'fetch("/api/ratings"' in html


def test_v1_v2_routes_state_submission_and_blob_namespaces_are_isolated():
    config = _load(WEB / "vercel.json")
    rewrites = {item["source"]: item["destination"] for item in config["rewrites"]}
    assert rewrites["/evaluate-v2"] == "/evaluate-v2/index.html"
    assert rewrites["/evaluate-v1"] == "/evaluate-v1/index.html"
    assert config["functions"]["api/ratings.js"]["maxDuration"] == 15
    assert config["functions"]["api/ratings-v2.js"]["maxDuration"] == 15

    v1_html = (V1 / "index.html").read_text(encoding="utf-8")
    v2_html = (V2 / "index.html").read_text(encoding="utf-8")
    v1_api = (WEB / "api" / "ratings.js").read_text(encoding="utf-8")
    v2_api = (WEB / "api" / "ratings-v2.js").read_text(encoding="utf-8")
    assert "soundalike-human-v17" in v1_html
    assert "soundalike-fulltrack-v2" not in v1_html
    assert "soundalike-fulltrack-v2" in v2_html
    assert "soundalike-human-v17" not in v2_html
    assert "human-ratings/v17/" in v1_api
    assert "human-ratings/fulltrack-v2/" not in v1_api
    assert "human-ratings/fulltrack-v2/" in v2_api
    assert "human-ratings/v17/" not in v2_api
    assert "blobList" not in v2_api
    assert "blobGet" not in v2_api


def test_v2_public_assets_are_blinded_and_audio_is_not_committed():
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (V2 / "index.html", V2 / "protocol-v2.json", V2 / "pilot-pack.json")
    )
    for marker in (
        "nonnegative_linear",
        "monotonic_network",
        "channel_gated_embedding",
        "frozen_hybrid",
        "BEGIN PRIVATE KEY",
    ):
        assert marker not in public_text
    assert not any(
        path.suffix.casefold() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        for path in V2.rglob("*")
    )


def test_production_recommendation_implementation_hash_is_reviewed(
    tmp_path, monkeypatch,
):
    # These literal pins are the explicit review checkpoint. Do not derive
    # them from the generated runtime manifest, which may be refreshed alone.
    reviewed = {
        "webapp/api/_reco.py": "5ebd5c630b8527387cbfe82930cb812b0039851547129d6ea6360c93983ec371",
        "webapp/api/recommend.py": "dbf66c9bc32c6438b05c67c23784d91dab6c196d3edaf79107a9b89c231d7b1e",
    }
    spec = importlib.util.spec_from_file_location(
        "homelab_update_review", ROOT / "deploy/homelab/update.py",
    )
    updater = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, updater)
    spec.loader.exec_module(updater)

    manifest = _load(ROOT / updater.MANIFEST_PATH)
    manifest["runtime_files"].update(reviewed)
    for relative in (*manifest["runtime_files"], manifest["requirements"]["path"]):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        )
    manifest_path = tmp_path / updater.MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    commit = "a" * 40
    updater.verify_release(tmp_path, commit, lambda _command: commit)

    # The real release verifier must reject an unreviewed change to either
    # serving file even though the other reviewed file remains unchanged.
    for relative in reviewed:
        path = tmp_path / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\n# unreviewed change\n")
        with pytest.raises(updater.UpdateError, match="runtime checksum"):
            updater.verify_release(tmp_path, commit, lambda _command: commit)
        path.write_bytes(original)


