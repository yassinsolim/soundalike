"""Verify and merge blinded ratings snapshots into deterministic analysis input."""

from __future__ import annotations

import argparse
import functools
import hashlib
import hmac
import json
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / ".goals" / "human-quality-recommendations"
V16 = GOAL / "protocol-v16-hosted-human-development"
V17 = GOAL / "protocol-v17-submission-human-development"
V16_PROVIDER = "hosted_client_only_evaluator"
V17_PROVIDER = "hosted_private_submission_evaluator"
NOTICE = (
    "Local-key HMAC provides integrity, not identity or authenticity; "
    "the key is included in this export."
)
MAX_FILE_BYTES = 600 * 1024
MAX_DURATION_MS = 366 * 24 * 60 * 60 * 1000
ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
RESULT_ID = re.compile(r"^T14-[a-f0-9]{24}$")
LIST_ID = re.compile(r"^L14-[a-f0-9]{24}$")
RATER_ID = re.compile(r"^anon-[a-f0-9]{24}$")
SESSION_ID = re.compile(r"^session-[a-f0-9]{24}$")
HEX_64 = re.compile(r"^[a-f0-9]{64}$")
FORBIDDEN_KEYS = {"__proto__", "constructor", "prototype"}

TRUSTED_STUDIES = {
    16: {
        "protocol": "c94ce615c68cde595b4e48ac5010297d76bedbed52948b10d315a39286117727",
        "lists": "809b98ae4314b396ffb33f7349fee72c94e1a80a33d84b1661ab83166a52b9e9",
        "allowed_signers": (
            "2ed0c27a387d63f9aa6eb3d56f9217eeef7a85d536ae9238c076032c602c7410"
        ),
    },
    17: {
        "protocol": "5b20dc6a1399959b3afe246743b2c76c20cb652078c9938c80a6316377a32eb5",
        "lists": "2311a7f3dc3b84452060e7ba1c42ed33cd886d602caeb2511363dd8cb90e2eeb",
        "allowed_signers": (
            "f85e4f151318b9484792b6fa932afc3ffb10456bc84cc9385fc62dd764dce71c"
        ),
    },
}

RESULT_KEYS = {
    "similarity",
    "score_0_10",
    "junk_or_version",
    "rated_at",
    "interaction_ms",
}
LIST_KEYS = {
    "whole_list_coherence",
    "unrelated_positions_1_to_3",
    "rated_at",
    "interaction_ms",
}
MIGRATION_KEYS = {
    "from_schema_version",
    "from_provider",
    "from_protocol_sha256",
    "from_served_lists_sha256",
    "migrated_at",
}
CLIENT_COMMON_KEYS = {
    "schema_version",
    "source_kind",
    "provider",
    "anonymous_rater_id",
    "session_id",
    "protocol_sha256",
    "served_lists_sha256",
    "local_session_key",
    "started_at",
    "last_activity_at",
    "result_ratings",
    "list_ratings",
    "exported_at",
    "duration_ms",
    "integrity_notice",
    "integrity_hmac_sha256",
}
V16_CLIENT_KEYS = CLIENT_COMMON_KEYS
V17_CLIENT_KEYS = CLIENT_COMMON_KEYS | {"migration"}
V17_SANITIZED_KEYS = V17_CLIENT_KEYS - {
    "local_session_key",
    "integrity_notice",
    "integrity_hmac_sha256",
}
SERVER_KEYS = V17_SANITIZED_KEYS | {
    "received_at",
    "canonical_payload_sha256",
    "counts",
}
COUNT_KEYS = {"complete_result_ratings", "complete_list_ratings"}


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        if key in FORBIDDEN_KEYS:
            raise ValueError("forbidden JSON key")
        result[key] = value
    return result


def strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_object_without_duplicates,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            ValueError("non-finite JSON number")
        ),
    )


def _load_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("ratings or audit JSON exceeds the size limit")
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _content_hash(document: dict[str, Any]) -> str:
    return sha256(canonical({k: v for k, v in document.items() if k != "content_sha256"}))


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes())


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not ISO_TIMESTAMP.fullmatch(value):
        raise ValueError("timestamp is not canonical UTC milliseconds")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp is invalid") from error
    if parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z") != value:
        raise ValueError("timestamp is not canonical UTC milliseconds")
    return parsed


def _verify_signature(directory: Path, version: int, metadata: dict[str, Any]) -> None:
    expected_metadata = {
        "algorithm",
        "namespace",
        "identity",
        "state_sha256",
        "state_content_sha256",
        "signer_public_key_sha256",
        "allowed_signers_sha256",
        "signature_sha256",
    }
    if set(metadata) != expected_metadata:
        raise ValueError("study signature metadata schema is invalid")
    namespace = f"soundalike-human-eval-v{version}"
    if (
        metadata["algorithm"] != "Ed25519 detached SSH signature"
        or metadata["namespace"] != namespace
        or metadata["identity"] != namespace
        or metadata["state_sha256"] != _file_hash(directory / "state.json")
        or metadata["signer_public_key_sha256"] != _file_hash(directory / "signer.pub")
        or metadata["allowed_signers_sha256"]
        != _file_hash(directory / "allowed_signers")
        or metadata["signature_sha256"] != _file_hash(directory / "state.sig")
        or metadata["allowed_signers_sha256"]
        != TRUSTED_STUDIES[version]["allowed_signers"]
    ):
        raise ValueError("study signature provenance is inconsistent")
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise ValueError("ssh-keygen is required to verify ratings study integrity")
    result = subprocess.run(
        [
            executable,
            "-Y",
            "verify",
            "-f",
            str(directory / "allowed_signers"),
            "-I",
            namespace,
            "-n",
            namespace,
            "-s",
            str(directory / "state.sig"),
        ],
        input=(directory / "state.json").read_bytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("ratings study signature verification failed")


@functools.lru_cache(maxsize=2)
def _study(schema_version: int) -> tuple[str, str, frozenset[str], frozenset[str]]:
    if schema_version not in TRUSTED_STUDIES:
        raise ValueError("unsupported ratings schema")
    directory = V16 if schema_version == 16 else V17
    protocol = _load_json(directory / f"protocol-v{schema_version}.json")
    lists = _load_json(directory / f"served-lists-v{schema_version}.json")
    state = _load_json(directory / "state.json")
    metadata = _load_json(directory / "signature-metadata.json")
    trusted = TRUSTED_STUDIES[schema_version]
    for document in (protocol, lists, state):
        if (
            document.get("schema_version") != schema_version
            or document.get("rankings_state") != "RANKINGS_LOCKED"
            or _content_hash(document) != document.get("content_sha256")
        ):
            raise ValueError("ratings study content integrity failed")
    if (
        protocol.get("content_sha256") != trusted["protocol"]
        or lists.get("content_sha256") != trusted["lists"]
        or protocol.get("served_lists_sha256") != trusted["lists"]
        or state.get("protocol_sha256") != trusted["protocol"]
        or state.get("served_lists_sha256") != trusted["lists"]
        or metadata.get("state_content_sha256") != state.get("content_sha256")
    ):
        raise ValueError("ratings study hashes are not trusted")
    _verify_signature(directory, schema_version, metadata)

    result_ids: set[str] = set()
    list_ids: set[str] = set()
    seeds = lists.get("seeds")
    if not isinstance(seeds, list) or lists.get("seed_count") != len(seeds):
        raise ValueError("ratings study seed identity is invalid")
    for seed in seeds:
        if not isinstance(seed, dict):
            raise ValueError("ratings study seed is invalid")
        seed_result_ids: set[str] = set()
        for result in seed.get("results", []):
            result_id = result.get("result_id") if isinstance(result, dict) else None
            if (
                not isinstance(result_id, str)
                or not RESULT_ID.fullmatch(result_id)
                or result_id in seed_result_ids
                or result_id in result_ids
            ):
                raise ValueError("ratings study result identity is invalid")
            seed_result_ids.add(result_id)
            result_ids.add(result_id)
        for item in seed.get("lists", []):
            list_id = item.get("list_id") if isinstance(item, dict) else None
            ranking = item.get("ranking") if isinstance(item, dict) else None
            if (
                not isinstance(list_id, str)
                or not LIST_ID.fullmatch(list_id)
                or list_id in list_ids
                or not isinstance(ranking, list)
                or len(ranking) != 5
            ):
                raise ValueError("ratings study list identity is invalid")
            list_ids.add(list_id)
            ranked_ids: set[str] = set()
            for index, row in enumerate(ranking, start=1):
                result_id = row.get("result_id") if isinstance(row, dict) else None
                if (
                    not isinstance(row, dict)
                    or row.get("position") != index
                    or result_id not in seed_result_ids
                    or result_id in ranked_ids
                ):
                    raise ValueError("ratings study candidate membership is invalid")
                ranked_ids.add(result_id)
    return (
        trusted["protocol"],
        trusted["lists"],
        frozenset(result_ids),
        frozenset(list_ids),
    )


def _validate_migration(value: Any, started_at: datetime, exported_at: datetime) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != MIGRATION_KEYS:
        raise ValueError("v17 migration schema is invalid")
    migrated_at = _timestamp(value["migrated_at"])
    if (
        value["from_schema_version"] != 16
        or value["from_provider"] != V16_PROVIDER
        or value["from_protocol_sha256"] != TRUSTED_STUDIES[16]["protocol"]
        or value["from_served_lists_sha256"] != TRUSTED_STUDIES[16]["lists"]
        or not started_at <= migrated_at <= exported_at
    ):
        raise ValueError("v17 migration provenance is invalid")


def _complete_counts(
    document: dict[str, Any],
    result_ids: frozenset[str],
    list_ids: frozenset[str],
    started_at: datetime,
    exported_at: datetime,
    duration_ms: int,
) -> dict[str, int]:
    results = document.get("result_ratings")
    lists = document.get("list_ratings")
    if not isinstance(results, dict) or not isinstance(lists, dict):
        raise ValueError("ratings maps are required")
    if not set(results).issubset(result_ids) or not set(lists).issubset(list_ids):
        raise ValueError("snapshot contains an ID outside the committed served lists")

    for result_id, rating in results.items():
        if (
            not RESULT_ID.fullmatch(result_id)
            or not isinstance(rating, dict)
            or set(rating) != RESULT_KEYS
        ):
            raise ValueError(f"malformed result rating {result_id}")
        score = rating["score_0_10"]
        interaction = rating["interaction_ms"]
        if (
            rating["similarity"]
            not in {"not_similar", "somewhat_similar", "very_similar"}
            or (
                score is not None
                and (
                    isinstance(score, bool)
                    or not isinstance(score, int)
                    or not 0 <= score <= 10
                )
            )
            or not isinstance(rating["junk_or_version"], bool)
            or isinstance(interaction, bool)
            or not isinstance(interaction, int)
            or not 1 <= interaction <= duration_ms
            or not started_at <= _timestamp(rating["rated_at"]) <= exported_at
        ):
            raise ValueError(f"malformed result rating {result_id}")

    for list_id, rating in lists.items():
        if (
            not LIST_ID.fullmatch(list_id)
            or not isinstance(rating, dict)
            or set(rating) != LIST_KEYS
        ):
            raise ValueError(f"malformed list rating {list_id}")
        unrelated = rating["unrelated_positions_1_to_3"]
        interaction = rating["interaction_ms"]
        if (
            rating["whole_list_coherence"]
            not in {"not_coherent", "somewhat_coherent", "very_coherent"}
            or isinstance(unrelated, bool)
            or not isinstance(unrelated, int)
            or not 0 <= unrelated <= 3
            or isinstance(interaction, bool)
            or not isinstance(interaction, int)
            or not 1 <= interaction <= duration_ms
            or not started_at <= _timestamp(rating["rated_at"]) <= exported_at
        ):
            raise ValueError(f"malformed list rating {list_id}")
    if not results and not lists:
        raise ValueError("snapshot contains no complete ratings")
    return {
        "complete_result_ratings": len(results),
        "complete_list_ratings": len(lists),
    }


def _validate_common(document: dict[str, Any], schema_version: int) -> dict[str, int]:
    protocol_hash, lists_hash, result_ids, list_ids = _study(schema_version)
    provider = V16_PROVIDER if schema_version == 16 else V17_PROVIDER
    if (
        document.get("schema_version") != schema_version
        or document.get("source_kind") != "human_listener"
        or document.get("provider") != provider
        or document.get("protocol_sha256") != protocol_hash
        or document.get("served_lists_sha256") != lists_hash
        or not isinstance(document.get("anonymous_rater_id"), str)
        or not RATER_ID.fullmatch(document["anonymous_rater_id"])
        or not isinstance(document.get("session_id"), str)
        or not SESSION_ID.fullmatch(document["session_id"])
    ):
        raise ValueError("snapshot does not match a committed ratings protocol")
    started_at = _timestamp(document.get("started_at"))
    last_activity_at = _timestamp(document.get("last_activity_at"))
    exported_at = _timestamp(document.get("exported_at"))
    duration_ms = document.get("duration_ms")
    if (
        not started_at <= last_activity_at <= exported_at
        or isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or not 1 <= duration_ms <= MAX_DURATION_MS
        or abs((exported_at - started_at).total_seconds() * 1000 - duration_ms) > 1000
    ):
        raise ValueError("snapshot timing is invalid")
    if schema_version == 17:
        _validate_migration(document.get("migration"), started_at, exported_at)
    return _complete_counts(
        document, result_ids, list_ids, started_at, exported_at, duration_ms
    )


def validate_snapshot(document: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized snapshot from a strict client export or server record."""
    if not isinstance(document, dict):
        raise ValueError("snapshot must be a JSON object")
    schema_version = document.get("schema_version")
    if schema_version not in (16, 17):
        raise ValueError("unsupported ratings schema")
    is_server_record = "canonical_payload_sha256" in document

    if is_server_record:
        if schema_version != 17 or set(document) != SERVER_KEYS:
            raise ValueError("server record schema is invalid")
        sanitized = {
            key: value
            for key, value in document.items()
            if key not in {"received_at", "canonical_payload_sha256", "counts"}
        }
        expected_digest = sha256(canonical(sanitized))
        if document["canonical_payload_sha256"] != expected_digest:
            raise ValueError("server record canonical payload hash mismatch")
        counts = _validate_common(sanitized, 17)
        if (
            not isinstance(document["counts"], dict)
            or set(document["counts"]) != COUNT_KEYS
            or document["counts"] != counts
        ):
            raise ValueError("server record count mismatch")
        _timestamp(document["received_at"])
        normalized = dict(sanitized)
        normalized["_snapshot_sha256"] = expected_digest
        normalized["_snapshot_at"] = document["received_at"]
        normalized["_source"] = "private_server_record_v17"
        return normalized

    expected_keys = V16_CLIENT_KEYS if schema_version == 16 else V17_CLIENT_KEYS
    if set(document) != expected_keys:
        raise ValueError("client export schema is invalid")
    key = document["local_session_key"]
    signature = document["integrity_hmac_sha256"]
    if (
        not isinstance(key, str)
        or not HEX_64.fullmatch(key)
        or not isinstance(signature, str)
        or not HEX_64.fullmatch(signature)
        or document["integrity_notice"] != NOTICE
    ):
        raise ValueError("client export integrity fields are malformed")
    payload = dict(document)
    del payload["integrity_hmac_sha256"]
    expected = hmac.new(key.encode("utf-8"), canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("client export HMAC mismatch")
    _validate_common(document, schema_version)

    sanitized = {
        key_name: value
        for key_name, value in document.items()
        if key_name
        not in {"local_session_key", "integrity_hmac_sha256", "integrity_notice"}
    }
    normalized = dict(sanitized)
    normalized["_snapshot_sha256"] = sha256(canonical(sanitized))
    normalized["_snapshot_at"] = document["exported_at"]
    normalized["_source"] = f"signed_client_export_v{schema_version}"
    return normalized


def merge_snapshots(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Deduplicate snapshots and merge additions; conflicting ratings raise."""
    unique: dict[str, dict[str, Any]] = {}
    for document in documents:
        normalized = validate_snapshot(document)
        unique.setdefault(normalized["_snapshot_sha256"], normalized)

    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in unique.values():
        sessions[snapshot["session_id"]].append(snapshot)

    merged_sessions = []
    for analysis_index, (session_id, snapshots) in enumerate(
        sorted(sessions.items()), start=1
    ):
        snapshots.sort(key=lambda item: (item["_snapshot_at"], item["_snapshot_sha256"]))
        rater_ids = {item["anonymous_rater_id"] for item in snapshots}
        if len(rater_ids) != 1:
            raise ValueError("session has conflicting anonymous rater IDs")
        schemas = {int(item["schema_version"]) for item in snapshots}
        if schemas == {16, 17} and any(
            item["schema_version"] == 17 and item.get("migration") is None
            for item in snapshots
        ):
            raise ValueError("v16/v17 session transition lacks migration provenance")
        merged_results: dict[str, dict[str, Any]] = {}
        merged_lists: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            for kind, target in (
                ("result_ratings", merged_results),
                ("list_ratings", merged_lists),
            ):
                for rating_id, rating in snapshot[kind].items():
                    if rating_id in target and target[rating_id] != rating:
                        raise ValueError(f"conflicting {kind} value")
                    target[rating_id] = rating
        merged_sessions.append(
            {
                "analysis_session_id": f"S{analysis_index:06d}",
                "result_ratings": dict(sorted(merged_results.items())),
                "list_ratings": dict(sorted(merged_lists.items())),
                "snapshot_count": len(snapshots),
                "source_schemas": sorted(schemas),
            }
        )
    return {
        "schema_version": 1,
        "aggregate_kind": "blinded_human_ratings_analysis",
        "session_count": len(merged_sessions),
        "complete_result_ratings": sum(
            len(item["result_ratings"]) for item in merged_sessions
        ),
        "complete_list_ratings": sum(
            len(item["list_ratings"]) for item in merged_sessions
        ),
        "sessions": merged_sessions,
    }


def _input_files(paths: Iterable[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError("ratings input does not exist")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and merge authorized local ratings inbox files."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = _input_files(args.inputs)
    result = merge_snapshots(_load_json(path) for path in files)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.resolve() in {path.resolve() for path in files}:
            raise ValueError("output must not overwrite a private input")
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"Wrote {result['session_count']} merged sessions.")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
