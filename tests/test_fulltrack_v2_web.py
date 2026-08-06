"""Deployment-boundary tests for the full-track v2 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "index.html": "b6445a1400e0b92a7187e895ec22e8301e53abcc73f9974ceb13436fecc9f537",
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


def test_production_recommendation_assets_remain_byte_compatible():
    assert {
        str(path.relative_to(ROOT)).replace("\\", "/"): _file_hash(path)
        for path in (
            WEB / "api" / "_reco.py",
            WEB / "api" / "recommend.py",
            WEB / "index.html",
        )
    } == {
        "webapp/api/_reco.py": "01d4d648c780a9d2e33a175c00e606bead54c39738ee7d1da36cf6e41294e311",
        "webapp/api/recommend.py": "6e455a325a8c0bf79d06c23dac5d68b88fda7b074d87567dc0a1a40923ca63fc",
        "webapp/index.html": "9f8f2b9e03b798400e4adcb28baf9f851414612d2eb3823309e41015644b6ed8",
    }
