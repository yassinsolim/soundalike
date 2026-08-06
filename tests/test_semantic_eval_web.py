"""Deployment-boundary tests for the semantic listening study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
SEMANTIC = WEB / "evaluate"
V2 = WEB / "evaluate-v2"
PACK_SHA = "4f3c34250d5c5fca35dcc671dae1c256f0d56d8ce404d7a758bbbf62a2e5b48a"
PROTOCOL_SHA = "662fbab57f5264329bfc8d75398bd5998d5e7b01ff049a548778bc834655ddd2"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _content_hash(document: dict) -> str:
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_route_deploys_locked_semantic_protocol_and_pack():
    protocol = _load(SEMANTIC / "protocol-semantic-v1.json")
    pack = _load(SEMANTIC / "semantic-pack.json")
    v2_pack = _load(V2 / "pilot-pack.json")

    assert _content_hash(protocol) == protocol["content_sha256"] == PROTOCOL_SHA
    assert _content_hash(pack) == pack["content_sha256"] == PACK_SHA
    assert protocol["pilot_pack_sha256"] == PACK_SHA
    assert protocol["local_storage_namespace"] == "soundalike-semantic-v1"
    assert protocol["submission_endpoint"] == "/api/ratings-semantic-v1"
    assert protocol["private_blob_prefix"] == "human-ratings/semantic-v1/"
    assert protocol["list_count"] == 40
    assert pack["method_count"] == 2
    assert pack["language_policy"]["evaluated_here"] is False
    assert pack["section_coverage"]["uniform_window_budget"] == 32
    assert pack["section_coverage"]["repeated_section_budget"] == 32
    assert pack["section_coverage"]["salient_section_budget"] == 32
    assert all("source_v2_seed_id" not in seed for seed in pack["seeds"])

    v2_by_seed = {seed["seed_track_id"]: seed for seed in v2_pack["seeds"]}
    for seed in pack["seeds"]:
        assert len(seed["lists"]) == 2
        published_v2_rankings = {
            tuple(row["track_id"] for row in candidate["ranking"])
            for candidate in v2_by_seed[seed["seed_track_id"]]["lists"]
        }
        published_v2_track_ids = {
            track_id
            for ranking in published_v2_rankings
            for track_id in ranking
        }
        for candidate in seed["lists"]:
            assert len(candidate["ranking"]) == 5
            ranking = tuple(row["track_id"] for row in candidate["ranking"])
            assert ranking not in published_v2_rankings
            assert set(ranking).isdisjoint(published_v2_track_ids)
            artist_ids = {
                pack["tracks"][str(track_id)]["source_identity"]["artist_id"]
                for track_id in ranking
            }
            assert len(artist_ids) == 5


def test_v2_is_byte_preserved_at_its_versioned_route():
    assert {path.name: _file_hash(path) for path in V2.iterdir()} == {
        "index.html": "b245ba0cbdc1be2821e5a7722b946c3e4330b508d848bdc74c593ff68fb628c6",
        "pilot-pack.json": "d23d66768f15fd5e37e01ad2a8905d181b4ff278c85674386edcd7dc50b267d3",
        "protocol-v2.json": "a88108894e3875159a9ae5b3fae61b01522e9c22647d9ff32748d53d0a5c981c",
    }


def test_routes_state_and_private_inboxes_are_version_isolated():
    config = _load(WEB / "vercel.json")
    rewrites = {item["source"]: item["destination"] for item in config["rewrites"]}
    assert rewrites["/evaluate"] == "/evaluate/index.html"
    assert rewrites["/evaluate-v2"] == "/evaluate-v2/index.html"
    assert rewrites["/evaluate-v1"] == "/evaluate-v1/index.html"
    assert config["functions"]["api/ratings-semantic-v1.js"]["maxDuration"] == 15

    semantic_html = (SEMANTIC / "index.html").read_text(encoding="utf-8")
    v2_html = (V2 / "index.html").read_text(encoding="utf-8")
    semantic_api = (WEB / "api" / "ratings-semantic-v1.js").read_text(
        encoding="utf-8"
    )
    assert "soundalike-semantic-v1" in semantic_html
    assert "soundalike-fulltrack-v2" not in semantic_html
    assert "soundalike-fulltrack-v2" in v2_html
    assert "human-ratings/semantic-v1/" in semantic_api
    assert "human-ratings/fulltrack-v2/" not in semantic_api
    assert "blobList" not in semantic_api
    assert "blobGet" not in semantic_api


def test_public_semantic_assets_are_blinded_and_audio_is_not_committed():
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            SEMANTIC / "index.html",
            SEMANTIC / "protocol-semantic-v1.json",
            SEMANTIC / "semantic-pack.json",
        )
    )
    for marker in (
        "fulltrack_audio_control_v1",
        "semantic_fulltrack_v1",
        "BEGIN PRIVATE KEY",
    ):
        assert marker not in public_text
    assert not any(
        path.suffix.casefold() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        for path in SEMANTIC.rglob("*")
    )
