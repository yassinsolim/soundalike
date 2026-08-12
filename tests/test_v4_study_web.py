"""Deployment-boundary tests for the V4 active listening study."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
ACTIVE = WEB / "evaluate"
ARCHIVE = WEB / "evaluate-pacing-v3"
PACK_SHA = "899197ad4eed5b84d69e2f37ee2a4fc04f36f73ed1bf0a8421dd1ef5654b1384"
PACK_FILE_SHA = "0b3b6875b2f19394f6d1a9ac1bcf2fdae4fb90d0be03c80d8f42b02e32c96f01"
PROTOCOL_SHA = "c0a1ff45bd5c57099aff352553b84579df8d3b4fc806bcb74e731b0dd5581966"


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


def test_active_route_deploys_exact_blinded_v4_pack_and_protocol():
    pack = _load(ACTIVE / "active-pack.json")
    protocol = _load(ACTIVE / "protocol-v4.json")
    assert _content_hash(pack) == pack["content_sha256"] == PACK_SHA
    assert _file_hash(ACTIVE / "active-pack.json") == PACK_FILE_SHA
    assert _content_hash(protocol) == protocol["content_sha256"] == PROTOCOL_SHA
    assert protocol["pilot_pack_sha256"] == PACK_SHA
    assert protocol["schema_version"] == 2
    assert protocol["local_storage_namespace"] == "soundalike-active-v4-ranking-v2"
    assert protocol["submission_endpoint"] == "/api/ratings-v4"
    assert protocol["private_blob_prefix"] == "human-ratings/active-v4-ranking-v2/"
    assert protocol["adaptive_stop_after_unique_tasks"] == 12
    assert protocol["repeated_anchor_count"] == 2
    assert protocol["language_evaluated"] is True
    assert protocol["unknown_language_allowed"] is False
    assert protocol["transcription_saved"] is False

    assert len(pack["tasks"]) == 18
    assert [task["priority_rank"] for task in pack["tasks"]] == list(range(1, 19))
    signatures = [
        (
            task["seed_track_id"],
            tuple(sorted(row["track_id"] for row in task["candidates"])),
        )
        for task in pack["tasks"]
    ]
    counts = Counter(signatures)
    assert len(counts) == 16
    assert sorted(counts.values()) == [1] * 14 + [2, 2]
    assert [
        index
        for index, signature in enumerate(signatures, 1)
        if signatures.index(signature) + 1 != index
    ] == [7, 14]
    assert pack["provenance"]["listener_ratings_used_for_ranking"] is False
    assert pack["provenance"]["learned_preference_head_used"] is False
    assert (
        pack["provenance"]["detector_gate_sha256"]
        == "504ce2b35bce262893861435257dd7d56800877f1d2735ca5e84ae3c2c9c5276"
    )


def test_v4_tasks_are_artist_diverse_and_repeated_excerpt_bounded():
    pack = _load(ACTIVE / "active-pack.json")
    used = set()
    choices = set()
    for task in pack["tasks"]:
        candidate_ids = [row["track_id"] for row in task["candidates"]]
        choice_ids = [row["choice_id"] for row in task["candidates"]]
        assert len(set(candidate_ids)) == len(set(choice_ids)) == 4
        assert not choices.intersection(choice_ids)
        choices.update(choice_ids)
        ids = [task["seed_track_id"], *candidate_ids]
        artists = {
            pack["tracks"][str(track_id)]["source_identity"]["artist_id"]
            for track_id in ids
        }
        assert len(artists) == 5
        used.update(ids)
    assert {str(track_id) for track_id in used} == set(pack["tracks"])
    for track in pack["tracks"].values():
        excerpt = track["audio"]["excerpt"]
        assert 0 < excerpt["end_seconds"] - excerpt["start_seconds"] <= 20
        assert excerpt["kind"] == "strongest_nonlocal_recurrence"


def test_public_v4_assets_are_method_blind_and_audio_free():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ACTIVE / "index.html",
            ACTIVE / "protocol-v4.json",
            ACTIVE / "active-pack.json",
        )
    )
    for marker in ('"method_bindings"', '"control"', '"challenger"', "blinding_key"):
        assert marker not in text
    assert "textarea" not in text
    assert not any(
        path.suffix.casefold() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        for path in ACTIVE.rglob("*")
    )


def test_pacing_v3_is_byte_preserved_at_versioned_route():
    assert {path.name: _file_hash(path) for path in ARCHIVE.iterdir()} == {
        "index.html": "575bf10c941ddc82ff31c2f196cedc204f4d15802a53dba30d18bcc6a86cd184",
        "pacing-pack.json": "3745fa4fa2df78e4f7feda4ccec924fac221ce91b5bf9d2c3316658e7a4e7525",
        "protocol-pacing-v3.json": "ba1db6bc3ad447c5eb2d1e2959d280bf6789a141c04182e0cf35976e9192bf02",
    }
    assert "webapp/evaluate-pacing-v3/* -text -whitespace" in (
        ROOT / ".gitattributes"
    ).read_text(encoding="utf-8")


def test_routes_and_private_inboxes_are_version_isolated():
    config = _load(WEB / "vercel.json")
    rewrites = {item["source"]: item["destination"] for item in config["rewrites"]}
    assert rewrites["/evaluate"] == "/evaluate/index.html"
    assert rewrites["/evaluate-pacing-v3"] == "/evaluate-pacing-v3/index.html"
    assert config["functions"]["api/ratings-v4.js"]["maxDuration"] == 15
    active_html = (ACTIVE / "index.html").read_text(encoding="utf-8")
    active_api = (WEB / "api" / "ratings-v4.js").read_text(encoding="utf-8")
    pacing_api = (WEB / "api" / "ratings-pacing-v3.js").read_text(encoding="utf-8")
    assert "soundalike-active-v4-ranking-v2" in active_html
    assert "soundalike-pacing-v3" not in active_html
    assert "human-ratings/active-v4-ranking-v2/" in active_api
    assert "human-ratings/pacing-v3/" not in active_api
    assert "../evaluate-pacing-v3/pacing-pack.json" in pacing_api
    assert "blobList" not in active_api
    assert "blobGet" not in active_api
