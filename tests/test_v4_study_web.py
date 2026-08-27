"""Deployment-boundary tests for active V6 and archived V5/V4 studies."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
ACTIVE = WEB / "evaluate"
V4_ARCHIVE = WEB / "evaluate-v4"
V5_ARCHIVE = WEB / "evaluate-v5"
PACING_ARCHIVE = WEB / "evaluate-pacing-v3"
PACK_SHA = "38cf7b0a4c035b27237288c9e4022a2b44d73ad82a0f3bd9085a2f862bea9637"
PACK_FILE_SHA = "e8301eb6452101ba56fc326e8c59feb27356532f094d4fe94b91bc34913eec04"
PROTOCOL_SHA = "084c25271bd8630949dacf50bfa8670328afcbab197303a7c79af7f95801d0f1"


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


def test_active_route_deploys_exact_blinded_v6_pack_and_protocol():
    pack = _load(ACTIVE / "active-pack-v6.json")
    protocol = _load(ACTIVE / "protocol-v6.json")
    assert _content_hash(pack) == pack["content_sha256"] == PACK_SHA
    assert _file_hash(ACTIVE / "active-pack-v6.json") == PACK_FILE_SHA
    assert _content_hash(protocol) == protocol["content_sha256"] == PROTOCOL_SHA
    assert protocol["pilot_pack_sha256"] == PACK_SHA
    assert protocol["schema_version"] == 1
    assert protocol["local_storage_namespace"] == "soundalike-development-v6-ranking-v1"
    assert protocol["submission_endpoint"] == "/api/ratings-v6"
    assert (
        protocol["private_blob_prefix"]
        == "human-ratings/development-v6-ranking-v1/"
    )
    assert protocol["adaptive_stop_after_unique_tasks"] == 12
    assert protocol["repeated_anchor_count"] == 2
    assert protocol["language_evaluated"] is True
    assert protocol["language_segments_per_track"] == 3
    assert protocol["unknown_language_allowed"] is False
    assert protocol["transcription_saved"] is False
    assert protocol["ranking_slots"] == [
        "most_similar",
        "next_most_similar",
        "second_least_similar",
        "least_similar",
    ]
    assert protocol["development_evidence"] is True
    assert protocol["independent_holdout"] is False
    assert protocol["promotion_allowed"] is False

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
    assert pack["provenance"]["listener_ratings_used_for_pack_selection"] is False
    assert pack["provenance"]["frozen_method_count"] == 3
    assert pack["provenance"]["method_identity_public"] is False
    assert pack["provenance"]["includes_v5_exposure"] is True
    assert (
        pack["provenance"]["excludes_all_prior_exposed_tracks_and_artists"] is True
    )
    assert len(pack["provenance"]["prior_exposure_pack_sha256s"]) == 7
    assert (
        pack["provenance"]["detector_gate_sha256"]
        == "7884f97795dc91cec436758f692a011816cbb35002d94718b8053c9e9b52a31e"
    )


def test_v6_tasks_are_artist_diverse_and_repeated_excerpt_bounded():
    pack = _load(ACTIVE / "active-pack-v6.json")
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
    assert (
        len(
            {
                track["source_identity"]["artist_id"]
                for track in pack["tracks"].values()
            }
        )
        == len(pack["tracks"])
        == 80
    )
    for track in pack["tracks"].values():
        excerpt = track["audio"]["excerpt"]
        assert 0 < excerpt["end_seconds"] - excerpt["start_seconds"] <= 20
        assert excerpt["kind"] == "strongest_nonlocal_recurrence"


def test_public_v6_assets_are_method_blind_and_audio_free():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ACTIVE / "index.html",
            ACTIVE / "protocol-v6.json",
            ACTIVE / "active-pack-v6.json",
        )
    )
    for marker in (
        '"method_bindings"',
        '"method_orders"',
        '"candidate_selection_sources"',
        "acoustic_control",
        "fixed_v4",
        "frozen_preference_v1",
        "blinding_key",
    ):
        assert marker not in text
    assert "textarea" not in text
    assert not any(
        path.suffix.casefold() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        for path in ACTIVE.rglob("*")
    )


def test_v5_archive_is_byte_locked_and_compatibility_assets_remain_available():
    assert {path.name: _file_hash(path) for path in V5_ARCHIVE.iterdir()} == {
        "active-pack.json": "4d138e12604e8119f1a3ee76e2fb040d301ef8bc478c7211c2977bac48612c18",
        "index.html": "a4c0108622bb856c3c9c72b28ae01dc36ea33e5c573465dc01cbe0f049043ccd",
        "protocol-v5.json": "fd077b80dfcbe873ab413e21c87b9150f668030343bea3983f3fa1ba9c1d552e",
    }
    assert (ACTIVE / "active-pack.json").read_bytes() == (
        V5_ARCHIVE / "active-pack.json"
    ).read_bytes()
    assert (ACTIVE / "protocol-v5.json").read_bytes() == (
        V5_ARCHIVE / "protocol-v5.json"
    ).read_bytes()
    assert "webapp/evaluate-v5/* -text -whitespace" in (
        ROOT / ".gitattributes"
    ).read_text(encoding="utf-8")


def test_pacing_v3_is_byte_preserved_at_versioned_route():
    assert {path.name: _file_hash(path) for path in PACING_ARCHIVE.iterdir()} == {
        "index.html": "575bf10c941ddc82ff31c2f196cedc204f4d15802a53dba30d18bcc6a86cd184",
        "pacing-pack.json": "3745fa4fa2df78e4f7feda4ccec924fac221ce91b5bf9d2c3316658e7a4e7525",
        "protocol-pacing-v3.json": "ba1db6bc3ad447c5eb2d1e2959d280bf6789a141c04182e0cf35976e9192bf02",
    }
    assert "webapp/evaluate-pacing-v3/* -text -whitespace" in (
        ROOT / ".gitattributes"
    ).read_text(encoding="utf-8")


def test_v4_archive_is_byte_locked_at_versioned_route():
    assert {path.name: _file_hash(path) for path in V4_ARCHIVE.iterdir()} == {
        "index.html": "a3f72c100bf7c5c7749cd1f38610a30d72769cb504c9fb2c79f220fc14f4a7dd",
        "active-pack.json": "0b3b6875b2f19394f6d1a9ac1bcf2fdae4fb90d0be03c80d8f42b02e32c96f01",
        "protocol-v4.json": "4fc10ebb01ca977072562288f4a005bcde1c92ee4f1f79af86532efbed85b2e7",
    }
    assert "webapp/evaluate-v4/* -text -whitespace" in (
        ROOT / ".gitattributes"
    ).read_text(encoding="utf-8")


def test_routes_and_private_inboxes_are_version_isolated():
    config = _load(WEB / "vercel.json")
    rewrites = {item["source"]: item["destination"] for item in config["rewrites"]}
    assert rewrites["/evaluate"] == "/evaluate/index.html"
    assert rewrites["/evaluate-v5"] == "/evaluate-v5/index.html"
    assert rewrites["/evaluate-v4"] == "/evaluate-v4/index.html"
    assert rewrites["/evaluate-pacing-v3"] == "/evaluate-pacing-v3/index.html"
    headers = {
        item["source"]: {row["key"]: row["value"] for row in item["headers"]}
        for item in config["headers"]
    }
    assert headers["/evaluate-v5"] == headers["/evaluate-v5/(.*)"]
    assert headers["/evaluate-v5"]["Cache-Control"] == "no-store, max-age=0"
    assert "media-src https://prod-1.storage.jamendo.com" in (
        headers["/evaluate-v5"]["Content-Security-Policy"]
    )
    assert config["functions"]["api/ratings-v4.js"]["maxDuration"] == 15
    assert config["functions"]["api/ratings-v5.js"]["maxDuration"] == 15
    assert len(config["functions"]) == 12
    assert "api/ratings-v6.js" not in config["functions"]
    assert rewrites["/api/ratings-v6"] == (
        "/api/ratings?__soundalike_handler=ratings-v6"
    )
    assert rewrites["/api/spicetify-feedback"] == (
        "/api/ratings?__soundalike_handler=spicetify-feedback"
    )
    active_html = (ACTIVE / "index.html").read_text(encoding="utf-8")
    active_api = (WEB / "server" / "ratings-v6.js").read_text(encoding="utf-8")
    v5_api = (WEB / "api" / "ratings-v5.js").read_text(encoding="utf-8")
    archived_api = (WEB / "api" / "ratings-v4.js").read_text(encoding="utf-8")
    archived_analysis = (WEB / "tools" / "ratings-v4-analysis.js").read_text(
        encoding="utf-8"
    )
    pacing_api = (WEB / "api" / "ratings-pacing-v3.js").read_text(encoding="utf-8")
    assert "soundalike-development-v6-ranking-v1" in active_html
    assert "soundalike-strict-v5-ranking-v1" not in active_html
    assert "soundalike-active-v4-ranking-v2" not in active_html
    assert "soundalike-pacing-v3" not in active_html
    assert "human-ratings/development-v6-ranking-v1/" in active_api
    assert "human-ratings/strict-v5-ranking-v1/" not in active_api
    assert "human-ratings/strict-v5-ranking-v1/" in v5_api
    assert "../evaluate-v5/active-pack.json" in v5_api
    assert "human-ratings/active-v4-ranking-v2/" not in active_api
    assert "../evaluate-v4/active-pack.json" in archived_api
    assert "../evaluate-v4/active-pack.json" in archived_analysis
    assert "human-ratings/pacing-v3/" not in active_api
    assert "../evaluate-pacing-v3/pacing-pack.json" in pacing_api
    assert "blobList" not in active_api
    assert "blobGet" not in active_api
