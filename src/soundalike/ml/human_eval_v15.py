"""Freeze, verify, and serve the mobile-only v15 human-evaluation successor.

v15 supersedes the zero-rating v14 pack only because the evaluator HTML changed.
The signed v14 files remain immutable.  Seed identities, opaque IDs, candidate
content, ranked order, method assignments, blinding salt, and collector trust
root are copied exactly into the new pack.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .human_eval_v10 import canonical_bytes, content_hash, file_hash
from .human_eval_v11 import evaluator_handler
from .human_eval_v14 import (
    FreshV14PreviewResolver,
    TRUSTED_V14_FILES,
    TRUSTED_V14_LISTS,
    TRUSTED_V14_PROTOCOL,
    TRUSTED_V14_STATE,
    verify_pack as verify_v14_pack,
)

SCHEMA_VERSION = 15
STATE_IDENTITY = "soundalike-human-eval-v15"
STATE_NAMESPACE = "soundalike-human-eval-v15"

# Byte-exact trust anchors for the immutable zero-rating v15 pack.
TRUSTED_V15_PROTOCOL = "4ee45316350ed1b4b49ffa2758d09f3479231d832cab81f8ae5c62985c852140"
TRUSTED_V15_LISTS = "5218bcec24cfb05776ee76af46c86059a89244f15109aded39fb90f5d279b1d4"
TRUSTED_V15_STATE = "ac69cced7cea708f9192de03f9d44cfa46867dda88ab4aba597156255ed1694f"
TRUSTED_V15_FILES: Dict[str, str] = {
    "allowed_signers": "190ed1c5fb73488d4dddaf3abf32000febf73c433d074be7585554076446dbd4",
    "collector_allowed_signers": "af787b33d44f2db435dd620fe1fe97c02473d7ff3cf7f33da1f896e979bd7101",
    "collector_signer.pub": "514d96fd9262e5e5f24ebdf5cc0287573f9e85b2f101cbdf5f48c12b598550f8",
    "protocol-v15.json": "c8e1cb92a0a81866b853087441be331c3f4d07142331be5a9c7d6596f367156b",
    "served-lists-v15.json": "4c0b5cd2e19fcb8233f75bb7a194a14a841800ab42955d6c6ed3fabc2ccd4a85",
    "signature-metadata.json": "0f3d997e9736153c5c76a175410c00c160913bd12a252243b23b74e0ac565e36",
    "signer.pub": "a6b587fb2a863e4f5d68df9413bea85206d133cacd16ccdcf2d61eb2a67efa27",
    "state.json": "69e62f3613a052244235fe56c1ca95652550227bb2cdd1f66b239b94129e0cbf",
    "state.sig": "2a09eae2af71d4fb8a82dcc879430ed7442e2f9039d08c13ccbb65ac5410927a",
}


class HumanV15Error(ValueError):
    """The evaluator-only v15 successor is unsafe or malformed."""


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def semantic_order_hash(document: Mapping[str, Any]) -> str:
    """Hash the exact displayed IDs, track IDs, positions, and list order."""
    order = []
    for seed in document["seeds"]:
        result_ids = {
            str(row["result_id"]): int(row["deezer_track_id"])
            for row in seed["results"]
        }
        order.append(
            {
                "seed_id": str(seed["seed_id"]),
                "seed_deezer_track_id": int(seed["query"]["deezer_track_id"]),
                "lists": [
                    {
                        "list_id": str(item["list_id"]),
                        "ranking": [
                            {
                                "position": int(row["position"]),
                                "result_id": str(row["result_id"]),
                                "deezer_track_id": result_ids[str(row["result_id"])],
                            }
                            for row in item["ranking"]
                        ],
                    }
                    for item in seed["lists"]
                ],
            }
        )
    return hashlib.sha256(canonical_bytes(order)).hexdigest()


def served_payload_hash(document: Mapping[str, Any]) -> str:
    """Hash every evaluation-bearing served-list field, excluding version metadata."""
    payload = {
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
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def method_assignment_hash(document: Mapping[str, Any]) -> str:
    """Hash the private role assignments without version/binding metadata."""
    payload = {
        "blinding_salt_sha256": document["blinding_salt_sha256"],
        "records": document["records"],
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _sign_state(directory: Path, state_path: Path) -> None:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV15Error("ssh-keygen is required; v15 signing fails closed")
    with tempfile.TemporaryDirectory(prefix="soundalike-human-v15-state-") as temp:
        private = Path(temp) / "signer"
        generated = subprocess.run(
            [
                executable,
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                STATE_IDENTITY,
                "-f",
                str(private),
            ],
            capture_output=True,
            check=False,
        )
        if generated.returncode:
            raise HumanV15Error("Ed25519 key generation failed")
        public_text = private.with_suffix(".pub").read_text(encoding="ascii").strip()
        fields = public_text.split()
        (directory / "signer.pub").write_text(public_text + "\n", encoding="ascii")
        (directory / "allowed_signers").write_text(
            f"{STATE_IDENTITY} {fields[0]} {fields[1]}\n", encoding="ascii"
        )
        signed = subprocess.run(
            [
                executable,
                "-Y",
                "sign",
                "-f",
                str(private),
                "-n",
                STATE_NAMESPACE,
                str(state_path),
            ],
            capture_output=True,
            check=False,
        )
        generated_signature = Path(str(state_path) + ".sig")
        if signed.returncode or not generated_signature.is_file():
            raise HumanV15Error("v15 state signing failed")
        os.replace(generated_signature, directory / "state.sig")

    metadata = {
        "algorithm": "Ed25519 detached SSH signature",
        "namespace": STATE_NAMESPACE,
        "identity": STATE_IDENTITY,
        "state_sha256": file_hash(state_path),
        "signer_public_key_sha256": file_hash(directory / "signer.pub"),
        "allowed_signers_sha256": file_hash(directory / "allowed_signers"),
        "signature_sha256": file_hash(directory / "state.sig"),
    }
    _write(directory / "signature-metadata.json", metadata)


def _verify_collector_private(private: Path, public: Path) -> None:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV15Error("ssh-keygen is required to verify the collector key")
    derived = subprocess.run(
        [executable, "-y", "-f", str(private)],
        capture_output=True,
        check=False,
        text=True,
    )
    expected = " ".join(public.read_text(encoding="ascii").split()[:2])
    actual = " ".join(derived.stdout.split()[:2])
    if derived.returncode or actual != expected:
        raise HumanV15Error("local v14 collector private key does not match trust root")


def freeze_pack(
    v14_pack_dir: Path,
    v14_private_dir: Path,
    evaluator_path: Path,
    preview_audit_path: Path,
    public_dir: Path,
    private_dir: Path,
) -> Dict[str, Path]:
    """Create an immutable, evaluator-only v15 successor from trusted v14."""
    if public_dir.exists() and any(public_dir.iterdir()):
        raise HumanV15Error("v15 public output must be empty; packs are immutable")
    if private_dir.exists() and any(private_dir.iterdir()):
        raise HumanV15Error("v15 private output must be empty; packs are immutable")
    if not evaluator_path.is_file():
        raise HumanV15Error("mobile-polished v15 evaluator is missing")

    v14_key_path = v14_private_dir / "method-key-v14.json"
    v14_collector_private = v14_private_dir / "collector_signer"
    if not v14_key_path.is_file() or not v14_collector_private.is_file():
        raise HumanV15Error("local v14 method/collector keys are required for supersession")

    predecessor = verify_v14_pack(
        v14_pack_dir, private_key=v14_key_path, require_trusted=True
    )
    old_protocol = predecessor["protocol"]
    old_lists = predecessor["lists"]
    old_state = predecessor["state"]
    old_key = json.loads(v14_key_path.read_text(encoding="utf-8"))

    if (
        old_protocol.get("ratings_count_at_freeze") != 0
        or old_lists.get("ratings_count_at_freeze") != 0
        or old_state.get("ratings_count_at_freeze") != 0
        or old_state.get("human_rater_exports_ingested") != 0
        or old_state.get("rankings_state") != "RANKINGS_LOCKED"
    ):
        raise HumanV15Error("v14 must be locked with zero ratings/exports")

    preview_audit = json.loads(preview_audit_path.read_text(encoding="utf-8"))
    ranked = preview_audit.get("ranked_positions", {})
    if (
        content_hash(preview_audit) != preview_audit.get("content_sha256")
        or preview_audit.get("served_lists_sha256") != old_lists["content_sha256"]
        or ranked.get("available") != 600
        or ranked.get("total") != 600
        or ranked.get("errors") != 0
    ):
        raise HumanV15Error("v14 600/600 preview audit is not valid")

    new_evaluator_hash = file_hash(evaluator_path)
    if new_evaluator_hash == old_protocol.get("evaluator_sha256"):
        raise HumanV15Error("evaluator is byte-identical; supersession is unwarranted")

    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    lists = copy.deepcopy(old_lists)
    lists["schema_version"] = SCHEMA_VERSION
    lists["pack_kind"] = "blinded_actual_served_lists_clap_v15_mobile_supersession"
    lists["semantic_order_sha256"] = semantic_order_hash(lists)
    lists["content_sha256"] = content_hash(lists)

    old_payload_sha = served_payload_hash(old_lists)
    new_payload_sha = served_payload_hash(lists)
    if (
        old_lists["seeds"] != lists["seeds"]
        or old_payload_sha != new_payload_sha
        or old_lists["semantic_order_sha256"] != lists["semantic_order_sha256"]
    ):
        raise HumanV15Error("v15 changed served content, IDs, rankings, or order")

    key = copy.deepcopy(old_key)
    key["schema_version"] = SCHEMA_VERSION
    key["served_lists_sha256"] = lists["content_sha256"]
    key["semantic_order_sha256"] = lists["semantic_order_sha256"]
    key["content_sha256"] = content_hash(key)
    if method_assignment_hash(old_key) != method_assignment_hash(key):
        raise HumanV15Error("v15 changed method assignments or blinding salt")

    method_key_path = private_dir / "method-key-v15.json"
    _write(method_key_path, key)
    shutil.copy2(v14_collector_private, private_dir / "collector_signer")
    for name in ("collector_signer.pub", "collector_allowed_signers"):
        shutil.copy2(v14_pack_dir / name, public_dir / name)
    _verify_collector_private(
        private_dir / "collector_signer", public_dir / "collector_signer.pub"
    )

    supersedes_v14 = {
        "old_protocol_sha256": old_protocol["content_sha256"],
        "old_lists_sha256": old_lists["content_sha256"],
        "old_state_sha256": old_state["content_sha256"],
        "old_protocol_file_sha256": file_hash(v14_pack_dir / "protocol-v14.json"),
        "old_lists_file_sha256": file_hash(v14_pack_dir / "served-lists-v14.json"),
        "old_state_file_sha256": file_hash(v14_pack_dir / "state.json"),
        "old_evaluator_sha256": old_protocol["evaluator_sha256"],
        "old_semantic_order_sha256": old_lists["semantic_order_sha256"],
        "new_semantic_order_sha256": lists["semantic_order_sha256"],
        "old_served_payload_sha256": old_payload_sha,
        "new_served_payload_sha256": new_payload_sha,
        "old_method_assignment_sha256": method_assignment_hash(old_key),
        "new_method_assignment_sha256": method_assignment_hash(key),
        "ratings_discarded": 0,
        "ratings_migrated": 0,
        "ranking_order_parity": True,
        "served_payload_semantically_identical": True,
        "method_assignments_identical": True,
        "blinding_identifiers_retained": True,
        "candidate_pack_semantics_identical": True,
        "reason": "mobile-only evaluator layout and accessibility polish",
    }

    protocol = copy.deepcopy(old_protocol)
    protocol.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_kind": "blinded_served_list_human_listener_clap_v15_mobile_supersession",
            "served_lists_sha256": lists["content_sha256"],
            "semantic_order_sha256": lists["semantic_order_sha256"],
            "private_key_sha256": key["content_sha256"],
            "evaluator_sha256": new_evaluator_hash,
            "preview_audit_content_sha256": preview_audit["content_sha256"],
            "preview_audit_file_sha256": file_hash(preview_audit_path),
            "preview_ranked_positions_available": 600,
            "preview_ranked_positions_total": 600,
            "supersedes_v14": supersedes_v14,
        }
    )
    protocol["content_sha256"] = content_hash(protocol)

    state = copy.deepcopy(old_state)
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "served_lists_sha256": lists["content_sha256"],
            "semantic_order_sha256": lists["semantic_order_sha256"],
            "protocol_sha256": protocol["content_sha256"],
            "private_method_key_sha256": key["content_sha256"],
            "evaluator_sha256": new_evaluator_hash,
            "preview_audit_content_sha256": preview_audit["content_sha256"],
            "preview_audit_file_sha256": file_hash(preview_audit_path),
            "preview_ranked_positions_available": 600,
            "preview_ranked_positions_total": 600,
            "supersedes_v14": supersedes_v14,
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    state["content_sha256"] = content_hash(state)

    lists_path = public_dir / "served-lists-v15.json"
    protocol_path = public_dir / "protocol-v15.json"
    state_path = public_dir / "state.json"
    _write(lists_path, lists)
    _write(protocol_path, protocol)
    _write(state_path, state)
    _sign_state(public_dir, state_path)

    return {
        "protocol": protocol_path,
        "lists": lists_path,
        "state": state_path,
        "signature": public_dir / "state.sig",
        "method_key": method_key_path,
        "collector_private": private_dir / "collector_signer",
        "collector_public": public_dir / "collector_signer.pub",
        "collector_allowed_signers": public_dir / "collector_allowed_signers",
    }


def _verify_signature(directory: Path, state_path: Path) -> None:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV15Error("ssh-keygen is required to verify the v15 pack")
    verified = subprocess.run(
        [
            executable,
            "-Y",
            "verify",
            "-f",
            str(directory / "allowed_signers"),
            "-I",
            STATE_IDENTITY,
            "-n",
            STATE_NAMESPACE,
            "-s",
            str(directory / "state.sig"),
        ],
        input=state_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode:
        raise HumanV15Error("v15 state signature is invalid")


def verify_pack(
    directory: Path,
    *,
    private_key: Optional[Path] = None,
    evaluator: Optional[Path] = None,
    require_trusted: bool = False,
) -> Dict[str, Any]:
    """Verify v15, its v14 predecessor, parity, hashes, and Ed25519 seal."""
    protocol_path = directory / "protocol-v15.json"
    lists_path = directory / "served-lists-v15.json"
    state_path = directory / "state.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lists = json.loads(lists_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))

    if require_trusted:
        if not TRUSTED_V15_FILES:
            raise HumanV15Error("v15 trust anchors are not populated")
        if (
            protocol.get("content_sha256") != TRUSTED_V15_PROTOCOL
            or lists.get("content_sha256") != TRUSTED_V15_LISTS
            or state.get("content_sha256") != TRUSTED_V15_STATE
            or any(
                not (directory / name).is_file()
                or file_hash(directory / name) != digest
                for name, digest in TRUSTED_V15_FILES.items()
            )
        ):
            raise HumanV15Error("v15 pack differs from committed trust anchors")

    for name, document in (("protocol", protocol), ("lists", lists), ("state", state)):
        if (
            document.get("schema_version") != SCHEMA_VERSION
            or content_hash(document) != document.get("content_sha256")
            or document.get("rankings_state") != "RANKINGS_LOCKED"
        ):
            raise HumanV15Error(f"v15 {name} hash/schema/lock mismatch")

    semantic = semantic_order_hash(lists)
    if not (
        protocol["served_lists_sha256"]
        == lists["content_sha256"]
        == state["served_lists_sha256"]
        and protocol["content_sha256"] == state["protocol_sha256"]
        and protocol["semantic_order_sha256"]
        == lists["semantic_order_sha256"]
        == state["semantic_order_sha256"]
        == semantic
        and protocol["ratings_count_at_freeze"]
        == lists["ratings_count_at_freeze"]
        == state["ratings_count_at_freeze"]
        == 0
        and state.get("human_rater_exports_ingested") == 0
    ):
        raise HumanV15Error("v15 pack hash/order/rating binding mismatch")

    predecessor_dir = directory.parent / "protocol-v14-clap-human-development"
    predecessor = verify_v14_pack(predecessor_dir, require_trusted=True)
    old_protocol = predecessor["protocol"]
    old_lists = predecessor["lists"]
    old_state = predecessor["state"]
    supersession = state.get("supersedes_v14")
    if (
        not isinstance(supersession, dict)
        or protocol.get("supersedes_v14") != supersession
        or supersession.get("old_protocol_sha256") != old_protocol["content_sha256"]
        or supersession.get("old_lists_sha256") != old_lists["content_sha256"]
        or supersession.get("old_state_sha256") != old_state["content_sha256"]
        or supersession.get("old_protocol_file_sha256")
        != file_hash(predecessor_dir / "protocol-v14.json")
        or supersession.get("old_lists_file_sha256")
        != file_hash(predecessor_dir / "served-lists-v14.json")
        or supersession.get("old_state_file_sha256")
        != file_hash(predecessor_dir / "state.json")
        or supersession.get("old_evaluator_sha256") != old_protocol["evaluator_sha256"]
        or supersession.get("ratings_discarded") != 0
        or supersession.get("ratings_migrated") != 0
        or supersession.get("ranking_order_parity") is not True
        or supersession.get("served_payload_semantically_identical") is not True
        or supersession.get("method_assignments_identical") is not True
        or supersession.get("blinding_identifiers_retained") is not True
        or supersession.get("candidate_pack_semantics_identical") is not True
        or old_lists["seeds"] != lists["seeds"]
        or supersession.get("old_semantic_order_sha256")
        != old_lists["semantic_order_sha256"]
        or supersession.get("new_semantic_order_sha256") != semantic
        or old_lists["semantic_order_sha256"] != semantic
        or supersession.get("old_served_payload_sha256")
        != served_payload_hash(old_lists)
        or supersession.get("new_served_payload_sha256") != served_payload_hash(lists)
        or served_payload_hash(old_lists) != served_payload_hash(lists)
    ):
        raise HumanV15Error("v14/v15 supersession or ranking parity mismatch")

    preview_path = directory.parent / "artifacts" / "human-eval-preview-audit-v14.json"
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    if (
        content_hash(preview) != protocol.get("preview_audit_content_sha256")
        or file_hash(preview_path) != protocol.get("preview_audit_file_sha256")
        or protocol.get("preview_audit_content_sha256")
        != state.get("preview_audit_content_sha256")
        or protocol.get("preview_ranked_positions_available")
        != state.get("preview_ranked_positions_available")
        != 600
        or protocol.get("preview_ranked_positions_total")
        != state.get("preview_ranked_positions_total")
        != 600
    ):
        raise HumanV15Error("v15 600/600 preview metadata binding mismatch")

    for name in ("collector_signer.pub", "collector_allowed_signers"):
        if (directory / name).read_bytes() != (predecessor_dir / name).read_bytes():
            raise HumanV15Error("v15 collector trust root differs from v14")
    if not (
        file_hash(directory / "collector_signer.pub")
        == protocol.get("collector_public_key_sha256")
        == state.get("collector_public_key_sha256")
        and file_hash(directory / "collector_allowed_signers")
        == protocol.get("collector_allowed_signers_sha256")
        == state.get("collector_allowed_signers_sha256")
    ):
        raise HumanV15Error("v15 collector trust-root hash mismatch")

    if evaluator is not None and not (
        file_hash(evaluator)
        == protocol.get("evaluator_sha256")
        == state.get("evaluator_sha256")
    ):
        raise HumanV15Error("evaluator differs from the signed v15 state")

    _verify_signature(directory, state_path)

    if private_key is not None:
        key = json.loads(private_key.read_text(encoding="utf-8"))
        old_key_path = directory.parents[2] / "ml_data" / "clap_v14" / "human_eval" / "method-key-v14.json"
        old_key = json.loads(old_key_path.read_text(encoding="utf-8"))
        public_ids = {item["list_id"] for seed in lists["seeds"] for item in seed["lists"]}
        private_ids = {item["list_id"] for item in key.get("records", [])}
        if not (
            key.get("schema_version") == SCHEMA_VERSION
            and content_hash(key) == key.get("content_sha256")
            and key.get("content_sha256")
            == protocol["private_key_sha256"]
            == state["private_method_key_sha256"]
            and key.get("served_lists_sha256") == lists["content_sha256"]
            and key.get("semantic_order_sha256") == semantic
            and public_ids == private_ids
            and method_assignment_hash(key) == method_assignment_hash(old_key)
            and supersession.get("old_method_assignment_sha256")
            == method_assignment_hash(old_key)
            and supersession.get("new_method_assignment_sha256")
            == method_assignment_hash(key)
        ):
            raise HumanV15Error("v15 private method key/parity binding mismatch")

    return {"protocol": protocol, "lists": lists, "state": state}


def serve(
    directory: Path,
    evaluator: Path,
    *,
    port: int = 8000,
) -> None:
    """Serve the trusted v15 evaluator and bundled locked files on loopback."""
    from http.server import ThreadingHTTPServer

    verify_pack(directory, evaluator=evaluator, require_trusted=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", int(port)),
        evaluator_handler(
            evaluator,
            directory / "protocol-v15.json",
            directory / "served-lists-v15.json",
            resolver=FreshV14PreviewResolver(),
        ),
    )
    print(f"Private blinded v15 evaluator: http://127.0.0.1:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    goal = root / ".goals" / "human-quality-recommendations"
    public = goal / "protocol-v15-clap-human-development"
    private = root / "ml_data" / "clap_v15" / "human_eval"
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument(
        "--v14-pack-dir",
        type=Path,
        default=goal / "protocol-v14-clap-human-development",
    )
    freeze.add_argument(
        "--v14-private-dir",
        type=Path,
        default=root / "ml_data" / "clap_v14" / "human_eval",
    )
    freeze.add_argument(
        "--evaluator", type=Path, default=root / "benchmarks" / "human_eval_v15.html"
    )
    freeze.add_argument(
        "--preview-audit",
        type=Path,
        default=goal / "artifacts" / "human-eval-preview-audit-v14.json",
    )
    freeze.add_argument("--public-dir", type=Path, default=public)
    freeze.add_argument("--private-dir", type=Path, default=private)

    verify = commands.add_parser("verify")
    verify.add_argument("--directory", type=Path, default=public)
    verify.add_argument("--private-key", type=Path)
    verify.add_argument(
        "--evaluator", type=Path, default=root / "benchmarks" / "human_eval_v15.html"
    )

    local = commands.add_parser("serve")
    local.add_argument("--directory", type=Path, default=public)
    local.add_argument(
        "--evaluator", type=Path, default=root / "benchmarks" / "human_eval_v15.html"
    )
    local.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        paths = freeze_pack(
            args.v14_pack_dir,
            args.v14_private_dir,
            args.evaluator,
            args.preview_audit,
            args.public_dir,
            args.private_dir,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "verify":
        value = verify_pack(
            args.directory,
            private_key=args.private_key,
            evaluator=args.evaluator,
            require_trusted=True,
        )
        print(
            f"RANKINGS_LOCKED: {value['lists']['content_sha256']} "
            "(ratings_count_at_freeze=0; v14/v15 ranking parity verified)"
        )
    else:
        serve(args.directory, args.evaluator, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
