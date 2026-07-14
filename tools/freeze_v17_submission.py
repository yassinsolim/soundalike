"""Build the unsigned v17 submission successor from the byte-locked v16 audit."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / ".goals" / "human-quality-recommendations"
V16 = GOAL / "protocol-v16-hosted-human-development"
V17 = GOAL / "protocol-v17-submission-human-development"
DEPLOY = ROOT / "webapp" / "evaluate"
LOCKED_AT = "2026-07-14T04:20:17.197000+00:00"
EXPECTED_V16_FILES = {
    "protocol-v16.json": "2936b184c86e83f0080f6a7c3956860d57d93f802cc066baebde4444041ffdcd",
    "served-lists-v16.json": "d61aee560308dfaee1faee5f5282960a454a1fe45462680951727557d0992918",
    "state.json": "0702d6c0cbac77872a84f891f0ee62bf3cb645a217167f83beecadf112865002",
    "state.sig": "66abd03472c76f5bc2efd486f328bbca682e04a39d20fc232670fdc9e4d52701",
}
EXPECTED_V16_EVALUATOR = (
    "f49a190692d3d8e169a594644d1b646ed6f81da084fc55062e5ab554501bc884"
)
SIGNING_IDENTITY = "soundalike-human-eval-v17"
SIGNING_NAMESPACE = "soundalike-human-eval-v17"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def content_digest(document: dict) -> str:
    return digest_bytes(canonical({k: v for k, v in document.items() if k != "content_sha256"}))


def write_document(path: Path, document: dict) -> None:
    document["content_sha256"] = content_digest(document)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def served_payload_digest(document: dict) -> str:
    keys = (
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
    return digest_bytes(canonical({key: document[key] for key in keys}))


def sign_state() -> None:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        raise RuntimeError("ssh-keygen is required to freeze the v17 audit")
    state_path = V17 / "state.json"
    signature_path = V17 / "state.sig"
    metadata_path = V17 / "signature-metadata.json"
    public_path = V17 / "signer.pub"
    allowed_path = V17 / "allowed_signers"
    existing = (signature_path, metadata_path, public_path, allowed_path)
    if all(path.is_file() for path in existing):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_matches = (
                metadata["algorithm"] == "Ed25519 detached SSH signature"
                and metadata["namespace"] == SIGNING_NAMESPACE
                and metadata["identity"] == SIGNING_IDENTITY
                and metadata["state_sha256"] == file_digest(state_path)
                and metadata["state_content_sha256"]
                == json.loads(state_path.read_text(encoding="utf-8"))["content_sha256"]
                and metadata["signer_public_key_sha256"] == file_digest(public_path)
                and metadata["allowed_signers_sha256"] == file_digest(allowed_path)
                and metadata["signature_sha256"] == file_digest(signature_path)
            )
        except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            metadata_matches = False
        if metadata_matches:
            verified = subprocess.run(
                [
                    ssh_keygen,
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed_path),
                    "-I",
                    SIGNING_IDENTITY,
                    "-n",
                    SIGNING_NAMESPACE,
                    "-s",
                    str(signature_path),
                ],
                input=state_path.read_bytes(),
                capture_output=True,
                check=False,
            )
            if verified.returncode == 0:
                return

    with tempfile.TemporaryDirectory(prefix="soundalike-v17-sign-") as temporary:
        private_key = Path(temporary) / "signer"
        generated = subprocess.run(
            [
                ssh_keygen,
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "soundalike-human-eval-v17-one-time",
                "-f",
                str(private_key),
            ],
            capture_output=True,
            check=False,
        )
        if generated.returncode != 0:
            raise RuntimeError(
                "failed to generate the one-time v17 signing key: "
                + generated.stderr.decode(errors="replace")
            )
        public_key = Path(f"{private_key}.pub").read_text(encoding="utf-8").strip()
        key_fields = public_key.split()
        if len(key_fields) < 2:
            raise RuntimeError("generated v17 public key is malformed")
        public_path.write_text(public_key + "\n", encoding="utf-8", newline="\n")
        allowed_path.write_text(
            f'{SIGNING_IDENTITY} namespaces="{SIGNING_NAMESPACE}" '
            f"{key_fields[0]} {key_fields[1]}\n",
            encoding="utf-8",
            newline="\n",
        )
        generated_signature = Path(f"{state_path}.sig")
        generated_signature.unlink(missing_ok=True)
        signed = subprocess.run(
            [
                ssh_keygen,
                "-Y",
                "sign",
                "-f",
                str(private_key),
                "-n",
                SIGNING_NAMESPACE,
                str(state_path),
            ],
            capture_output=True,
            check=False,
        )
        if signed.returncode != 0 or not generated_signature.is_file():
            raise RuntimeError(
                "failed to sign the frozen v17 state: "
                + signed.stderr.decode(errors="replace")
            )
        signature_path.unlink(missing_ok=True)
        generated_signature.replace(signature_path)

    metadata = {
        "algorithm": "Ed25519 detached SSH signature",
        "namespace": SIGNING_NAMESPACE,
        "identity": SIGNING_IDENTITY,
        "state_sha256": file_digest(state_path),
        "state_content_sha256": json.loads(state_path.read_text(encoding="utf-8"))[
            "content_sha256"
        ],
        "signer_public_key_sha256": file_digest(public_path),
        "allowed_signers_sha256": file_digest(allowed_path),
        "signature_sha256": file_digest(signature_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build() -> None:
    for name, expected in EXPECTED_V16_FILES.items():
        actual = file_digest(V16 / name)
        if actual != expected:
            raise RuntimeError(f"v16 audit changed: {name} is {actual}")
    v16_evaluator = (
        ROOT / "webapp" / "evaluate" / "index.html"
    )  # v17 now occupies this path.
    predecessor_evaluator = json.loads(
        (V16 / "state.json").read_text(encoding="utf-8")
    )["evaluator_sha256"]
    if predecessor_evaluator != EXPECTED_V16_EVALUATOR:
        raise RuntimeError("v16 evaluator commitment changed")

    old_protocol = json.loads((V16 / "protocol-v16.json").read_text(encoding="utf-8"))
    old_lists = json.loads((V16 / "served-lists-v16.json").read_text(encoding="utf-8"))
    old_state = json.loads((V16 / "state.json").read_text(encoding="utf-8"))
    evaluator_sha = file_digest(v16_evaluator)

    V17.mkdir(parents=True, exist_ok=True)
    for name in ("collector_allowed_signers", "collector_signer.pub"):
        shutil.copyfile(V16 / name, V17 / name)

    lists = copy.deepcopy(old_lists)
    lists["schema_version"] = 17
    lists["pack_kind"] = (
        "blinded_actual_served_lists_v17_private_submission_supersession"
    )
    write_document(V17 / "served-lists-v17.json", lists)

    old_served_payload = old_protocol["supersedes_v15"]["new_served_payload_sha256"]
    if served_payload_digest(lists) != old_served_payload:
        raise RuntimeError("v17 public served payload changed")

    provenance = {
        "known_external_exports_observed": 1,
        "known_predecessor_result_ratings": 5,
        "ratings_discarded": 0,
        "ratings_count_at_ranking_freeze": 0,
        "exports_ingested_at_ranking_freeze": 0,
        "note": (
            "One integrity-valid external v16 export with five result ratings was "
            "observed after the rankings were locked. It is not committed or counted "
            "as ingested evidence. The v17 browser migration preserves the matching "
            "trusted v16 autosave; no rating changed the frozen rankings."
        ),
    }
    supersession = {
        "predecessor_schema_version": 16,
        "old_protocol_sha256": old_protocol["content_sha256"],
        "old_lists_sha256": old_lists["content_sha256"],
        "old_state_sha256": old_state["content_sha256"],
        "old_protocol_file_sha256": file_digest(V16 / "protocol-v16.json"),
        "old_lists_file_sha256": file_digest(V16 / "served-lists-v16.json"),
        "old_state_file_sha256": file_digest(V16 / "state.json"),
        "old_signature_file_sha256": file_digest(V16 / "state.sig"),
        "old_evaluator_sha256": EXPECTED_V16_EVALUATOR,
        "old_semantic_order_sha256": old_lists["semantic_order_sha256"],
        "new_semantic_order_sha256": lists["semantic_order_sha256"],
        "old_served_payload_sha256": old_served_payload,
        "new_served_payload_sha256": served_payload_digest(lists),
        "old_method_assignment_sha256": old_state["method_assignment_sha256"],
        "new_method_assignment_sha256": old_state["method_assignment_sha256"],
        "old_blinding_salt_sha256": old_state["blinding_salt_sha256"],
        "new_blinding_salt_sha256": old_state["blinding_salt_sha256"],
        **provenance,
        "browser_autosave_migration_supported": True,
        "predecessor_result_ratings_preserved_by_migration": 5,
        "ranking_order_parity": True,
        "served_payload_semantically_identical": True,
        "method_assignments_identical": True,
        "blinding_salt_identical": True,
        "opaque_identifiers_retained": True,
        "candidate_pack_semantics_identical": True,
        "collector_public_trust_identical": True,
        "recommendation_behavior_changed": False,
        "reason": (
            "manual consent-based private ratings submission and exact trusted v16 "
            "browser-autosave migration; frozen recommendations are unchanged"
        ),
    }

    protocol = copy.deepcopy(old_protocol)
    protocol["schema_version"] = 17
    protocol["protocol_kind"] = (
        "blinded_served_list_human_listener_v17_private_submission_supersession"
    )
    protocol["served_lists_sha256"] = lists["content_sha256"]
    protocol["evaluator_sha256"] = evaluator_sha
    protocol["supersedes_v16"] = supersession
    protocol["known_rating_provenance"] = provenance
    protocol["ratings_submission"] = {
        "automatic_submission": False,
        "explicit_consent_required": True,
        "storage_access": "private",
        "client_export_fallback_retained": True,
        "stored_identifiers": "random anonymous rater and session IDs only",
        "excluded_from_application_record": [
            "local_session_key",
            "integrity_hmac_sha256",
            "IP address",
            "Origin",
            "user agent",
            "cookies",
            "Spotify identity",
            "email",
            "raw headers",
        ],
    }
    protocol["deployment_scope"] = (
        "public hosted evaluator plus private ratings submission"
    )
    protocol["hosted_only_rationale"] = (
        "v17 adds manual private submission and trusted v16 autosave migration; "
        "the frozen recommendation payload remains unchanged"
    )
    protocol["recommendation_behavior_changed"] = False
    write_document(V17 / "protocol-v17.json", protocol)

    state = copy.deepcopy(old_state)
    state["schema_version"] = 17
    state["served_lists_sha256"] = lists["content_sha256"]
    state["protocol_sha256"] = protocol["content_sha256"]
    state["evaluator_sha256"] = evaluator_sha
    state["locked_at"] = LOCKED_AT
    state["supersedes_v16"] = supersession
    state["known_rating_provenance"] = provenance
    state["deployment_scope"] = (
        "public hosted evaluator plus private ratings submission"
    )
    state["hosted_only_rationale"] = (
        "v17 adds manual private submission and trusted v16 autosave migration; "
        "the frozen recommendation payload remains unchanged"
    )
    state["recommendation_behavior_changed"] = False
    state["deployed"] = False
    state["human_rater_exports_ingested"] = 0
    state["ratings_count_at_freeze"] = 0
    write_document(V17 / "state.json", state)
    sign_state()

    shutil.copyfile(V17 / "protocol-v17.json", DEPLOY / "protocol.json")
    shutil.copyfile(V17 / "served-lists-v17.json", DEPLOY / "served-lists.json")
    print(f"protocol_sha256={protocol['content_sha256']}")
    print(f"served_lists_sha256={lists['content_sha256']}")
    print(f"state_content_sha256={state['content_sha256']}")
    print(f"evaluator_sha256={evaluator_sha}")


if __name__ == "__main__":
    build()
