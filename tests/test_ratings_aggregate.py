"""Tests for private v17 receipt aggregation and v16 client-export migration."""

from __future__ import annotations

import copy
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "aggregate_ratings.py"


def _module():
    spec = importlib.util.spec_from_file_location("ratings_aggregate", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _study(version: int) -> tuple[dict, dict]:
    directory_name = (
        f"protocol-v{version}-"
        + ("hosted-human-development" if version == 16 else "submission-human-development")
    )
    directory = (
        ROOT
        / ".goals"
        / "human-quality-recommendations"
        / directory_name
    )
    return (
        json.loads((directory / f"protocol-v{version}.json").read_text(encoding="utf-8")),
        json.loads(
            (directory / f"served-lists-v{version}.json").read_text(encoding="utf-8")
        ),
    )


def _sign(document: dict, key: str) -> dict:
    payload = dict(document)
    payload.pop("integrity_hmac_sha256", None)
    document["integrity_hmac_sha256"] = hmac.new(
        key.encode("utf-8"),
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return document


def _v16_export() -> dict:
    protocol, lists = _study(16)
    ids = [item["result_id"] for item in lists["seeds"][0]["results"][:5]]
    key = "a" * 64
    ratings = {
        result_id: {
            "similarity": "very_similar",
            "score_0_10": 8,
            "junk_or_version": False,
            "rated_at": f"2026-07-14T00:00:0{index}.000Z",
            "interaction_ms": 1000,
        }
        for index, result_id in enumerate(ids, start=1)
    }
    return _sign(
        {
            "schema_version": 16,
            "source_kind": "human_listener",
            "provider": "hosted_client_only_evaluator",
            "anonymous_rater_id": "anon-" + "1" * 24,
            "session_id": "session-" + "2" * 24,
            "protocol_sha256": protocol["content_sha256"],
            "served_lists_sha256": lists["content_sha256"],
            "local_session_key": key,
            "started_at": "2026-07-14T00:00:00.000Z",
            "last_activity_at": "2026-07-14T00:00:05.000Z",
            "result_ratings": ratings,
            "list_ratings": {},
            "exported_at": "2026-07-14T00:00:06.000Z",
            "duration_ms": 6000,
            "integrity_notice": (
                "Local-key HMAC provides integrity, not identity or authenticity; "
                "the key is included in this export."
            ),
        },
        key,
    )


def _v17_server_record(source: dict, *, add_result: bool = False) -> dict:
    protocol, lists = _study(17)
    migrated = {
        **copy.deepcopy(source),
        "schema_version": 17,
        "provider": "hosted_private_submission_evaluator",
        "protocol_sha256": protocol["content_sha256"],
        "served_lists_sha256": lists["content_sha256"],
        "migration": {
            "from_schema_version": 16,
            "from_provider": "hosted_client_only_evaluator",
            "from_protocol_sha256": source["protocol_sha256"],
            "from_served_lists_sha256": source["served_lists_sha256"],
            "migrated_at": "2026-07-14T00:00:06.500Z",
        },
        "exported_at": "2026-07-14T00:00:07.000Z",
        "duration_ms": 7000,
    }
    if add_result:
        known = set(migrated["result_ratings"])
        result_id = next(
            item["result_id"]
            for seed in lists["seeds"]
            for item in seed["results"]
            if item["result_id"] not in known
        )
        migrated["result_ratings"] = {
            **migrated["result_ratings"],
            result_id: {
                "similarity": "somewhat_similar",
                "score_0_10": None,
                "junk_or_version": False,
                "rated_at": "2026-07-14T00:00:06.000Z",
                "interaction_ms": 1000,
            },
        }
        migrated["last_activity_at"] = "2026-07-14T00:00:06.000Z"
    sanitized = {
        key: value
        for key, value in migrated.items()
        if key
        not in {"local_session_key", "integrity_hmac_sha256", "integrity_notice"}
    }
    canonical = json.dumps(
        sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **sanitized,
        "received_at": "2026-07-14T00:00:08.000Z",
        "canonical_payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "counts": {
            "complete_result_ratings": len(sanitized["result_ratings"]),
            "complete_list_ratings": len(sanitized["list_ratings"]),
        },
    }


def test_v16_five_rating_export_and_sanitized_v17_receipt_merge_exactly():
    aggregate = _module()
    old = _v16_export()
    migrated = _v17_server_record(old)

    result = aggregate.merge_snapshots([old, migrated, migrated])

    assert result["session_count"] == 1
    assert result["complete_result_ratings"] == 5
    assert result["complete_list_ratings"] == 0
    session = result["sessions"][0]
    assert session["result_ratings"] == old["result_ratings"]
    assert session["snapshot_count"] == 2
    assert session["source_schemas"] == [16, 17]


def test_later_non_conflicting_partial_snapshot_adds_ratings():
    aggregate = _module()
    old = _v16_export()
    first = _v17_server_record(old)
    later = _v17_server_record(old, add_result=True)
    later["received_at"] = "2026-07-14T00:00:09.000Z"

    result = aggregate.merge_snapshots([first, later])

    assert result["complete_result_ratings"] == 6
    assert result["sessions"][0]["snapshot_count"] == 2


def test_conflicting_later_rating_is_never_accepted_silently():
    aggregate = _module()
    old = _v16_export()
    first = _v17_server_record(old)
    conflict = _v17_server_record(old)
    result_id = next(iter(conflict["result_ratings"]))
    conflict["result_ratings"][result_id]["similarity"] = "not_similar"
    sanitized = {
        key: value
        for key, value in conflict.items()
        if key not in {"received_at", "canonical_payload_sha256", "counts"}
    }
    conflict["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(
            sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    conflict["received_at"] = "2026-07-14T00:00:09.000Z"

    with pytest.raises(ValueError, match="conflicting result_ratings"):
        aggregate.merge_snapshots([first, conflict])


def test_tampered_hmac_invented_id_and_false_server_count_are_rejected():
    aggregate = _module()
    tampered = _v16_export()
    tampered["result_ratings"][next(iter(tampered["result_ratings"]))][
        "similarity"
    ] = "not_similar"
    with pytest.raises(ValueError, match="HMAC"):
        aggregate.validate_snapshot(tampered)

    invented = _v16_export()
    rating = invented["result_ratings"].pop(next(iter(invented["result_ratings"])))
    invented["result_ratings"]["T14-" + "f" * 24] = rating
    _sign(invented, invented["local_session_key"])
    with pytest.raises(ValueError, match="outside the committed"):
        aggregate.validate_snapshot(invented)

    receipt = _v17_server_record(_v16_export())
    receipt["counts"]["complete_result_ratings"] = 999
    with pytest.raises(ValueError, match="count mismatch"):
        aggregate.validate_snapshot(receipt)
