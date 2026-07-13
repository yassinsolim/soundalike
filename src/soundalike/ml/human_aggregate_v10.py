"""Validate and aggregate anonymous iteration-10 human-listener exports.

Usage:
  python -m soundalike.ml.human_aggregate_v10 --protocol human-eval-v10/protocol-v10.json --lists human-eval-v10/served-lists-v10.json --key human-eval-v10/method-key-v10.json --exports ratings/rater-1.json ratings/rater-2.json --output sonic_human-v10.json

With no exports (or no actual ratings), the command exits non-zero and removes
the requested output. Method roles are revealed only in a valid aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

from .human_eval_v10 import canonical_bytes, content_hash, file_hash
from .human_eval_v11 import (
    ERRATUM_IDENTITY,
    ERRATUM_NAMESPACE,
    TRUSTED_ERRATUM_ALLOWED_SIGNERS_FILE,
    TRUSTED_V10_FILES,
    TRUSTED_V10_KEY,
    TRUSTED_V10_LISTS,
    TRUSTED_V10_PROTOCOL,
    TRUSTED_V11_ERRATUM,
    TRUSTED_V11_LISTS,
    TRUSTED_V11_PROTOCOL,
    ranking_order_hash,
)
from .human_eval_v13 import (
    STATE_IDENTITY as V13_STATE_IDENTITY,
    STATE_NAMESPACE as V13_STATE_NAMESPACE,
    TRUSTED_V13_FILES,
    TRUSTED_V13_LISTS,
    TRUSTED_V13_PROTOCOL,
    TRUSTED_V13_STATE,
    semantic_order_hash as v13_semantic_order_hash,
)

CLASSES = {"not_similar": 0, "somewhat_similar": 1, "very_similar": 2}
COHERENCE = {"not_coherent": 0, "somewhat_coherent": 1, "very_coherent": 2}
FORBIDDEN = ("last.fm", "lastfm", "deezer", "music4all", "gnod", "model",
             "proxy", "editorial")


class AggregateError(ValueError):
    """Input is not valid actual interactive human-listener evidence."""


def _verify_v13_state(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    lists: Mapping[str, Any],
    key: Mapping[str, Any],
) -> None:
    """Verify the new study's signed state and semantic track-order binding."""
    directory = protocol_path.parent
    state_path = directory / "state.json"
    signature = directory / "state.sig"
    allowed = directory / "allowed_signers"
    if not all(path.is_file() for path in (state_path, signature, allowed)):
        raise AggregateError("signed v13 study state is incomplete")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    semantic = v13_semantic_order_hash(lists)
    if (
        protocol.get("content_sha256") != TRUSTED_V13_PROTOCOL
        or lists.get("content_sha256") != TRUSTED_V13_LISTS
        or state.get("content_sha256") != TRUSTED_V13_STATE
        or any(
            not (directory / name).is_file()
            or file_hash(directory / name) != digest
            for name, digest in TRUSTED_V13_FILES.items()
        )
        or
        state.get("schema_version") != 13
        or content_hash(state) != state.get("content_sha256")
        or state.get("rankings_state") != "RANKINGS_LOCKED"
        or state.get("ratings_count_at_freeze") != 0
        or protocol.get("ratings_count_at_freeze") != 0
        or lists.get("ratings_count_at_freeze") != 0
        or not (
            protocol.get("served_lists_sha256")
            == lists.get("content_sha256")
            == state.get("served_lists_sha256")
        )
        or protocol.get("content_sha256") != state.get("protocol_sha256")
        or not (
            protocol.get("private_key_sha256")
            == key.get("content_sha256")
            == state.get("private_method_key_sha256")
        )
        or key.get("served_lists_sha256") != lists.get("content_sha256")
        or not (
            protocol.get("semantic_order_sha256")
            == lists.get("semantic_order_sha256")
            == key.get("semantic_order_sha256")
            == state.get("semantic_order_sha256")
            == semantic
        )
    ):
        raise AggregateError("v13 signed state binding mismatch")
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise AggregateError("ssh-keygen is required to verify v13 study state")
    verified = subprocess.run(
        [
            executable,
            "-Y",
            "verify",
            "-f",
            str(allowed),
            "-I",
            V13_STATE_IDENTITY,
            "-n",
            V13_STATE_NAMESPACE,
            "-s",
            str(signature),
        ],
        input=state_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode:
        raise AggregateError("v13 study state signature is invalid")


def _verify_audio_access_erratum(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    lists: Mapping[str, Any],
    key: Mapping[str, Any],
) -> None:
    """Allow only the signed metadata-only v11 successor of the v10 pack."""
    if protocol.get("audio_access_erratum_file") != "audio-access-erratum-v11.json":
        raise AggregateError("key/list hash mismatch")
    directory = protocol_path.parent
    artifact = directory / "audio-access-erratum-v11.json"
    signature = directory / "audio-access-erratum-v11.sig"
    allowed = directory / "erratum-allowed-signers"
    if not all(path.is_file() for path in (artifact, signature, allowed)):
        raise AggregateError("signed audio-access erratum is incomplete")
    erratum = json.loads(artifact.read_text(encoding="utf-8"))
    if (
        erratum.get("schema_version") != 11
        or content_hash(erratum) != erratum.get("content_sha256")
        or erratum.get("content_sha256") != TRUSTED_V11_ERRATUM
    ):
        raise AggregateError("audio-access erratum content hash mismatch")
    if (
        protocol.get("content_sha256") != TRUSTED_V11_PROTOCOL
        or lists.get("content_sha256") != TRUSTED_V11_LISTS
        or key.get("content_sha256") != TRUSTED_V10_KEY
        or file_hash(allowed) != TRUSTED_ERRATUM_ALLOWED_SIGNERS_FILE
    ):
        raise AggregateError("audio-access erratum is not rooted in the trusted study")

    predecessor = protocol_path.parent.parent / "protocol-v10-human-development"
    if any(
        not (predecessor / name).is_file()
        or file_hash(predecessor / name) != digest
        for name, digest in TRUSTED_V10_FILES.items()
    ):
        raise AggregateError("trusted v10 predecessor files are missing or changed")
    old_protocol = json.loads(
        (predecessor / "protocol-v10.json").read_text(encoding="utf-8")
    )
    old_lists = json.loads(
        (predecessor / "served-lists-v10.json").read_text(encoding="utf-8")
    )
    old_state = json.loads((predecessor / "state.json").read_text(encoding="utf-8"))
    if not (
        content_hash(old_protocol) == old_protocol.get("content_sha256")
        == TRUSTED_V10_PROTOCOL
        and content_hash(old_lists) == old_lists.get("content_sha256")
        == TRUSTED_V10_LISTS
        and old_protocol.get("served_lists_sha256") == TRUSTED_V10_LISTS
        and old_protocol.get("private_key_sha256") == TRUSTED_V10_KEY
        and old_state.get("served_lists_sha256") == TRUSTED_V10_LISTS
        and old_state.get("protocol_sha256") == TRUSTED_V10_PROTOCOL
    ):
        raise AggregateError("trusted v10 predecessor binding mismatch")

    # The metadata successor must retain every displayed identity, not merely
    # opaque list IDs.  This prevents swapping Deezer IDs under a valid order.
    if len(old_lists["seeds"]) != len(lists["seeds"]):
        raise AggregateError("audio-access erratum changed seed identities")
    for old_seed, new_seed in zip(old_lists["seeds"], lists["seeds"]):
        if (
            old_seed["seed_id"] != new_seed["seed_id"]
            or old_seed["scene"] != new_seed["scene"]
            or any(
                old_seed["query"][field] != new_seed["query"][field]
                for field in ("title", "artist", "track_id")
            )
            or new_seed["query"].get("deezer_track_id")
            != old_seed["query"]["track_id"]
        ):
            raise AggregateError("audio-access erratum changed a seed identity")
        old_results = {row["result_id"]: row for row in old_seed["results"]}
        new_results = {row["result_id"]: row for row in new_seed["results"]}
        if set(old_results) != set(new_results):
            raise AggregateError("audio-access erratum changed result identities")
        for result_id, old_result in old_results.items():
            new_result = new_results[result_id]
            if (
                any(
                    old_result[field] != new_result[field]
                    for field in ("track_id", "title", "artist")
                )
                or new_result.get("deezer_track_id") != old_result["track_id"]
            ):
                raise AggregateError("audio-access erratum changed a track identity")

    new_order = ranking_order_hash(lists)
    if not (
        protocol.get("predecessor_served_lists_sha256")
        == key.get("served_lists_sha256")
        == erratum.get("old_served_lists_sha256")
        and protocol.get("served_lists_sha256")
        == lists.get("content_sha256")
        == erratum.get("new_served_lists_sha256")
        and protocol.get("content_sha256") == erratum.get("new_protocol_sha256")
        and protocol.get("private_key_sha256")
        == key.get("content_sha256")
        == erratum.get("private_method_key_sha256")
        and protocol.get("ranking_order_sha256")
        == new_order
        == erratum.get("old_list_order_sha256")
        == erratum.get("new_list_order_sha256")
        and erratum.get("list_order_semantically_identical") is True
    ):
        raise AggregateError("audio-access erratum binding/parity mismatch")
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise AggregateError("ssh-keygen is required to verify audio-access erratum")
    verified = subprocess.run(
        [
            executable, "-Y", "verify", "-f", str(allowed),
            "-I", ERRATUM_IDENTITY, "-n", ERRATUM_NAMESPACE,
            "-s", str(signature),
        ],
        input=artifact.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode:
        raise AggregateError("audio-access erratum signature is invalid")
    predecessor_verified = subprocess.run(
        [
            executable, "-Y", "verify",
            "-f", str(predecessor / "allowed_signers"),
            "-I", "soundalike-human-eval",
            "-n", "soundalike-human-eval",
            "-s", str(predecessor / "state.sig"),
        ],
        input=(predecessor / "state.json").read_bytes(),
        capture_output=True,
        check=False,
    )
    if predecessor_verified.returncode:
        raise AggregateError("trusted v10 predecessor signature is invalid")


def sign_export(payload: Mapping[str, Any], local_session_key: str) -> str:
    """Return the portable HMAC used by the browser (integrity, not authenticity)."""
    return hmac.new(
        local_session_key.encode("utf-8"), canonical_bytes(payload), hashlib.sha256
    ).hexdigest()


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise AggregateError("interactive timestamps must be ISO-8601 strings")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AggregateError("invalid interactive timestamp") from exc


def _load_bound(protocol_path: Path, lists_path: Path, key_path: Path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lists = json.loads(lists_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    schemas = {
        protocol.get("schema_version"),
        lists.get("schema_version"),
        key.get("schema_version"),
    }
    if len(schemas) != 1 or next(iter(schemas)) not in {10, 13}:
        raise AggregateError("protocol/list/key schema versions are incompatible")
    schema = int(next(iter(schemas)))
    for name, doc in (("protocol", protocol), ("lists", lists), ("key", key)):
        if content_hash(doc) != doc.get("content_sha256"):
            raise AggregateError(f"{name} schema/content hash mismatch")
        if doc.get("rankings_state") != "RANKINGS_LOCKED":
            raise AggregateError(f"{name} rankings are not locked")
    if protocol.get("served_lists_sha256") != lists["content_sha256"]:
        raise AggregateError("protocol/list hash mismatch")
    if protocol.get("private_key_sha256") != key["content_sha256"]:
        raise AggregateError("protocol/key hash mismatch")
    if schema == 13:
        _verify_v13_state(protocol_path, protocol, lists, key)
    elif key.get("served_lists_sha256") != lists["content_sha256"]:
        _verify_audio_access_erratum(protocol_path, protocol, lists, key)
    collector_allowed = protocol_path.parent / "collector_allowed_signers"
    if not collector_allowed.is_file():
        raise AggregateError("trusted collector allowed-signers file is missing")
    if hashlib.sha256(collector_allowed.read_bytes()).hexdigest() != protocol.get(
        "collector_allowed_signers_sha256"
    ):
        raise AggregateError("collector allowed-signers hash mismatch")
    roles = {row["list_id"]: row["method_role"] for row in key.get("records", [])}
    valid_lists = {item["list_id"]
                   for seed in lists.get("seeds", []) for item in seed["lists"]}
    if set(roles) != valid_lists or set(roles.values()) != {
        "production_baseline", "challenger"
    }:
        raise AggregateError("private method key is incomplete")
    return protocol, lists, key, roles, collector_allowed


def _verify_collector_approval(export_path: Path, allowed_signers: Path) -> None:
    executable = shutil.which("ssh-keygen")
    signature = Path(str(export_path) + ".sig")
    if executable is None or not signature.is_file():
        raise AggregateError("trusted collector approval signature is required")
    verified = subprocess.run(
        [
            executable, "-Y", "verify", "-f", str(allowed_signers),
            "-I", "soundalike-human-rater", "-n", "soundalike-human-rater",
            "-s", str(signature),
        ],
        input=export_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode:
        raise AggregateError("trusted collector approval signature is invalid")


def _validate_export(
    document: Mapping[str, Any], protocol: Mapping[str, Any],
    lists: Mapping[str, Any],
) -> Dict[str, Any]:
    if document.get("schema_version") != protocol.get("schema_version"):
        raise AggregateError("export schema_version must match the study protocol")
    source = str(document.get("source_kind", "")).casefold()
    provider = str(document.get("provider", "")).casefold()
    if source != "human_listener":
        raise AggregateError("source_kind must be 'human_listener'")
    if any(term in source or term in provider for term in FORBIDDEN):
        raise AggregateError("proxy/model/provider evidence is forbidden")
    serialized = json.dumps(document, ensure_ascii=False, sort_keys=True).casefold()
    if any(term in serialized for term in FORBIDDEN):
        raise AggregateError("forbidden non-human evidence appears in export rows")
    if provider != "standalone_local_evaluator":
        raise AggregateError("unknown human-listener collection provider")
    if document.get("protocol_sha256") != protocol["content_sha256"]:
        raise AggregateError("export protocol hash mismatch")
    if document.get("served_lists_sha256") != lists["content_sha256"]:
        raise AggregateError("export served-list hash mismatch")
    key = document.get("local_session_key")
    signature = document.get("integrity_hmac_sha256")
    if not isinstance(key, str) or len(key) < 32 or not isinstance(signature, str):
        raise AggregateError("missing local integrity key/signature")
    payload = dict(document)
    payload.pop("integrity_hmac_sha256", None)
    if not hmac.compare_digest(sign_export(payload, key), signature):
        raise AggregateError("export HMAC mismatch")
    rater = document.get("anonymous_rater_id")
    session = document.get("session_id")
    if not isinstance(rater, str) or len(rater) < 12:
        raise AggregateError("invalid anonymous rater ID")
    if not isinstance(session, str) or len(session) < 8:
        raise AggregateError("invalid session ID")
    started = _parse_time(document.get("started_at"))
    exported = _parse_time(document.get("exported_at"))
    duration = document.get("duration_ms")
    if not isinstance(duration, int) or duration <= 0 or exported <= started:
        raise AggregateError("an actual positive interactive duration is required")
    wall_ms = (exported - started).total_seconds() * 1000
    if duration > wall_ms + 5000:
        raise AggregateError("interactive duration exceeds timestamp interval")

    valid_results = {row["result_id"]
                     for seed in lists["seeds"] for row in seed["results"]}
    valid_lists = {row["list_id"]
                   for seed in lists["seeds"] for row in seed["lists"]}
    result_ratings, list_ratings = {}, {}
    interaction_total = 0
    for result_id, rating in document.get("result_ratings", {}).items():
        if result_id not in valid_results or rating.get("similarity") not in CLASSES:
            raise AggregateError("invalid result rating")
        score = rating.get("score_0_10")
        if score is not None and (not isinstance(score, int) or not 0 <= score <= 10):
            raise AggregateError("optional score must be an integer from 0 to 10")
        if not isinstance(rating.get("junk_or_version"), bool):
            raise AggregateError("junk/version flag is required")
        rated_at = _parse_time(rating.get("rated_at"))
        if not started <= rated_at <= exported:
            raise AggregateError("result rating timestamp is outside the session")
        if not isinstance(rating.get("interaction_ms"), int) or rating["interaction_ms"] <= 0:
            raise AggregateError("result rating lacks actual interaction duration")
        interaction_total += rating["interaction_ms"]
        result_ratings[result_id] = dict(rating)
    for list_id, rating in document.get("list_ratings", {}).items():
        if list_id not in valid_lists or rating.get("whole_list_coherence") not in COHERENCE:
            raise AggregateError("invalid list rating")
        unrelated = rating.get("unrelated_positions_1_to_3")
        if not isinstance(unrelated, int) or not 0 <= unrelated <= 3:
            raise AggregateError("top-3 unrelated count must be 0..3")
        rated_at = _parse_time(rating.get("rated_at"))
        if not started <= rated_at <= exported:
            raise AggregateError("list rating timestamp is outside the session")
        if not isinstance(rating.get("interaction_ms"), int) or rating["interaction_ms"] <= 0:
            raise AggregateError("list rating lacks actual interaction duration")
        interaction_total += rating["interaction_ms"]
        list_ratings[list_id] = dict(rating)
    if not result_ratings and not list_ratings:
        raise AggregateError("export contains no ratings")
    if interaction_total > duration + 5000:
        raise AggregateError("rating interaction durations exceed session duration")
    clean = dict(document)
    clean["result_ratings"], clean["list_ratings"] = result_ratings, list_ratings
    return clean


def _ndcg(grades: Sequence[Optional[int]]) -> Optional[float]:
    observed = [(index, grade) for index, grade in enumerate(grades, 1)
                if grade is not None]
    if not observed:
        return None
    dcg = sum((2 ** int(grade) - 1) / math.log2(index + 1)
              for index, grade in observed)
    ideal = sorted((int(grade) for _, grade in observed), reverse=True)
    idcg = sum((2 ** grade - 1) / math.log2(index + 1)
               for index, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _agreement(values_by_item: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    agree = total = 0
    marginals: Counter[int] = Counter()
    for values in values_by_item.values():
        marginals.update(values)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total += 1
                agree += values[i] == values[j]
    if not total:
        return {"pair_count": 0, "exact_agreement": None, "kappa": None}
    observed = agree / total
    count = sum(marginals.values())
    expected = sum((n / count) ** 2 for n in marginals.values())
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"pair_count": total, "exact_agreement": observed, "kappa": kappa}


def _bootstrap(differences: Sequence[float], *, samples: int = 10000) -> Dict[str, Any]:
    if not differences:
        return {"n_pairs": 0, "mean_difference": None, "ci95": None}
    rng = random.Random(1010)
    n = len(differences)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return {
        "n_pairs": n,
        "mean_difference": sum(differences) / n,
        "ci95": [means[int(.025 * (samples - 1))],
                 means[int(.975 * (samples - 1))]],
        "bootstrap_samples": samples,
        "random_seed": 1010,
    }


def aggregate(
    protocol_path: Union[Path, str], lists_path: Union[Path, str],
    key_path: Union[Path, str], export_paths: Iterable[Union[Path, str]],
) -> Dict[str, Any]:
    """Return a deterministic aggregate; partial sessions remain explicitly partial."""
    protocol, lists, key, roles, collector_allowed = _load_bound(
        Path(protocol_path), Path(lists_path), Path(key_path)
    )
    paths = sorted({Path(path) for path in export_paths}, key=lambda p: str(p))
    if not paths:
        raise AggregateError("no rater exports supplied; refusing sonic_human report")
    exports = []
    for path in paths:
        _verify_collector_approval(path, collector_allowed)
        exports.append(_validate_export(
            json.loads(path.read_text(encoding="utf-8")), protocol, lists
        ))

    seed_by_result, seed_by_list, rankings = {}, {}, {}
    for seed in lists["seeds"]:
        for result in seed["results"]:
            seed_by_result[result["result_id"]] = seed["seed_id"]
        for ranked in seed["lists"]:
            seed_by_list[ranked["list_id"]] = seed["seed_id"]
            rankings[ranked["list_id"]] = [x["result_id"] for x in ranked["ranking"]]

    # Merge repeated exports from one anonymous rater deterministically by latest event.
    merged: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: {"results": {}, "lists": {}}
    )
    for export in exports:
        rater = export["anonymous_rater_id"]
        for result_id, rating in export["result_ratings"].items():
            bucket = merged[(rater, seed_by_result[result_id])]["results"]
            prior = bucket.get(result_id)
            if prior is None or (rating["rated_at"], export["session_id"]) > (
                prior["rated_at"], prior["_session"]
            ):
                bucket[result_id] = {**rating, "_session": export["session_id"]}
        for list_id, rating in export["list_ratings"].items():
            bucket = merged[(rater, seed_by_list[list_id])]["lists"]
            prior = bucket.get(list_id)
            if prior is None or (rating["rated_at"], export["session_id"]) > (
                prior["rated_at"], prior["_session"]
            ):
                bucket[list_id] = {**rating, "_session": export["session_id"]}
    if not merged:
        raise AggregateError("no actual ratings after validation")

    per_seed = []
    similarity_agreement: Dict[str, list[int]] = defaultdict(list)
    coherence_agreement: Dict[str, list[int]] = defaultdict(list)
    for (rater, seed_id), ratings in sorted(merged.items()):
        del rater  # one value per deduplicated anonymous rater/item
        for result_id, rating in ratings["results"].items():
            similarity_agreement[f"{seed_id}:{result_id}"].append(
                CLASSES[rating["similarity"]]
            )
        for list_id, rating in ratings["lists"].items():
            coherence_agreement[f"{seed_id}:{list_id}"].append(
                COHERENCE[rating["whole_list_coherence"]]
            )
    for (rater, seed_id), ratings in sorted(merged.items()):
        row: Dict[str, Any] = {
            "anonymous_rater_id": rater, "seed_id": seed_id,
            "partial": True, "methods": {},
        }
        ndcgs = {}
        seed_lists = [lid for lid, sid in seed_by_list.items() if sid == seed_id]
        for list_id in sorted(seed_lists):
            role = roles[list_id]
            grades = []
            for result_id in rankings[list_id]:
                rating = ratings["results"].get(result_id)
                grade = CLASSES[rating["similarity"]] if rating else None
                grades.append(grade)
            list_rating = ratings["lists"].get(list_id)
            metric = {
                "rated_results": sum(value is not None for value in grades),
                "ndcg_at_5": _ndcg(grades),
                "similarity_class_counts": dict(sorted(Counter(
                    ratings["results"][rid]["similarity"]
                    for rid in rankings[list_id] if rid in ratings["results"]
                ).items())),
                "mean_optional_score_0_10": (
                    lambda values: sum(values) / len(values) if values else None
                )([
                    ratings["results"][rid]["score_0_10"]
                    for rid in rankings[list_id]
                    if rid in ratings["results"]
                    and ratings["results"][rid].get("score_0_10") is not None
                ]),
                "whole_list_coherence": (
                    list_rating["whole_list_coherence"] if list_rating else None
                ),
                "coherence_score": (
                    COHERENCE[list_rating["whole_list_coherence"]] / 2
                    if list_rating else None
                ),
                "unrelated_positions_1_to_3": (
                    list_rating["unrelated_positions_1_to_3"] if list_rating else None
                ),
                "junk_or_version_count": sum(
                    bool(ratings["results"][rid]["junk_or_version"])
                    for rid in rankings[list_id] if rid in ratings["results"]
                ),
            }
            row["methods"][role] = metric
            ndcgs[role] = metric["ndcg_at_5"]
        row["partial"] = any(
            metric["rated_results"] < 5 or metric["whole_list_coherence"] is None
            for metric in row["methods"].values()
        )
        per_seed.append(row)

    seed_summary = []
    for seed_id in sorted({row["seed_id"] for row in per_seed}):
        rows = [row for row in per_seed if row["seed_id"] == seed_id]
        methods = {}
        for role in ("production_baseline", "challenger"):
            method_rows = [row["methods"][role] for row in rows]
            methods[role] = {}
            for field in ("ndcg_at_5", "coherence_score",
                          "unrelated_positions_1_to_3", "junk_or_version_count",
                          "mean_optional_score_0_10"):
                values = [item[field] for item in method_rows if item[field] is not None]
                methods[role][field] = sum(values) / len(values) if values else None
            methods[role]["rated_raters"] = sum(
                item["rated_results"] > 0 for item in method_rows
            )
        seed_summary.append({
            "seed_id": seed_id, "anonymous_rater_count": len(rows), "methods": methods
        })
    # Inferential comparisons require both methods to be complete for the same
    # anonymous rater and seed.  Then average those within-rater deltas per seed
    # before bootstrapping seeds, avoiding both rater confounding and
    # pseudo-replication.
    paired_by_seed: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: {"ndcg": [], "coherence": []}
    )
    for row in per_seed:
        baseline = row["methods"]["production_baseline"]
        challenger = row["methods"]["challenger"]
        complete = all(
            method["rated_results"] == 5
            and method["whole_list_coherence"] is not None
            for method in (baseline, challenger)
        )
        row["paired_complete"] = complete
        if not complete:
            continue
        paired_by_seed[row["seed_id"]]["ndcg"].append(
            challenger["ndcg_at_5"] - baseline["ndcg_at_5"]
        )
        paired_by_seed[row["seed_id"]]["coherence"].append(
            challenger["coherence_score"] - baseline["coherence_score"]
        )
    paired_ndcg = [
        sum(values["ndcg"]) / len(values["ndcg"])
        for _, values in sorted(paired_by_seed.items()) if values["ndcg"]
    ]
    paired_coherence = [
        sum(values["coherence"]) / len(values["coherence"])
        for _, values in sorted(paired_by_seed.items()) if values["coherence"]
    ]

    method_summary = {}
    for role in ("production_baseline", "challenger"):
        rows = [row["methods"][role] for row in per_seed]
        method_summary[role] = {}
        for field in ("ndcg_at_5", "coherence_score",
                      "unrelated_positions_1_to_3", "junk_or_version_count",
                      "mean_optional_score_0_10"):
            values = [row[field] for row in rows if row[field] is not None]
            method_summary[role][field] = (
                sum(values) / len(values) if values else None
            )
        method_summary[role]["rated_rater_seed_count"] = sum(
            row["rated_results"] > 0 for row in rows
        )

    return {
        "schema_version": int(protocol["schema_version"]),
        "report_kind": "sonic_human",
        "source_kind": "human_listener",
        "protocol_sha256": protocol["content_sha256"],
        "served_lists_sha256": lists["content_sha256"],
        "valid_export_count": len(exports),
        "deduplicated_rater_seed_count": len(merged),
        "partial_rater_seed_count": sum(row["partial"] for row in per_seed),
        "inter_rater_agreement": {
            "result_similarity": _agreement(similarity_agreement),
            "whole_list_coherence": _agreement(coherence_agreement),
        },
        "paired_bootstrap_challenger_minus_baseline_ndcg_at_5":
            _bootstrap(paired_ndcg),
        "paired_bootstrap_challenger_minus_baseline_coherence":
            _bootstrap(paired_coherence),
        "method_summary": method_summary,
        "per_seed": seed_summary,
        "per_rater_seed": per_seed,
        "integrity_notice": (
            "Validated local HMACs provide integrity only, not rater identity "
            "or authenticity, because each key travels in its export."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate actual anonymous human-listener v10 exports.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--lists", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--exports", nargs="*", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = aggregate(args.protocol, args.lists, args.key, args.exports)
    except (AggregateError, OSError, json.JSONDecodeError) as exc:
        args.output.unlink(missing_ok=True)
        print(f"ERROR: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"valid sonic_human report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
