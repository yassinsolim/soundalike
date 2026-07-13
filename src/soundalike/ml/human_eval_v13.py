"""Freeze and serve the new CLAP-vs-production blind development study.

The v10/v11 studies remain immutable.  V13 creates new opaque identities,
fresh stable-Deezer-ID lists, a new collector trust root, and a detached
Ed25519 signature over a state that binds every public/private artifact hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .clap_catalog_v13 import (
    BACKOFF,
    EXPECTED_ROWS,
    NETWORK_ATTEMPTS,
    PREREGISTRATION_SHA256,
    RateLimiter,
    SCHEMA_VERSION,
    TRACK_IDS_SHA256,
    ClapCatalogError,
    _network_session,
    _preview_url,
)
from .human_eval_v10 import canonical_bytes, content_hash, file_hash
from .human_eval_v11 import (
    PRODUCTION_PREVIEW_ENDPOINT,
    audit_preview_coverage,
    evaluator_handler,
)


EXPECTED_SEEDS = 60
EXPECTED_SCENES = 13
RESULTS_PER_METHOD = 5
STATE_IDENTITY = "soundalike-human-eval-v13"
STATE_NAMESPACE = "soundalike-human-eval-v13"
COLLECTOR_IDENTITY = "soundalike-human-rater"
COLLECTOR_NAMESPACE = "soundalike-human-rater"
TRUSTED_V13_PROTOCOL = "35c106be3cb90ff5c5ec2be159007f68472b3c508315ebdf83cc77690ef93d3e"
TRUSTED_V13_LISTS = "8c09b31e55efdbc7399a4e26f8291e0b408d07395b61cda6c873e6eb46eaa370"
TRUSTED_V13_STATE = "4f8af084e9f708fef16ae9909e7687cf46b3ad5c7248df5171485813f2e69c8e"
TRUSTED_V13_FILES = {
    "protocol-v13.json": "b753464522619583936ce730947b8da4c5df02808efab91a1f03014ae22246f9",
    "served-lists-v13.json": "af227746dfce527466ed2f7ba15401efc78bf256dde289341cae278f3980456f",
    "state.json": "9bd3219dd8f83adfc8bcb89858e6f511195b7ba34608bf21066f306154fa587a",
    "state.sig": "e669192febe922c0de456231bc2c9e1fce60f1b31e1ddb3ad9fa13054caee044",
    "allowed_signers": "36cd3775aa7fa3dc4197da48db621e9d1578df90015c95172f40aecc782865d9",
    "signer.pub": "c985728eb2b7341dfd79334e4459aacf0812c7a731a6bbc636e66bbd60d5b40a",
    "collector_allowed_signers":
        "e12112c7ebb347087d4049e770c2035a46ef5ef44a971ea1133084397c397769",
    "collector_signer.pub":
        "5b582a584c3de89f9e8e5e5c03eac82607f3d7ab2f18bb8f4a6081714c07a7e5",
}


class HumanV13Error(ClapCatalogError):
    """The CLAP human-development pack is unsafe or malformed."""


class FreshV13PreviewResolver:
    """Provider-aware fresh resolver for the loopback study server."""

    def __init__(self) -> None:
        self.limiter = RateLimiter(10.0)

    def __call__(self, track_id: int) -> Optional[str]:
        error: Optional[Exception] = None
        for attempt in range(NETWORK_ATTEMPTS):
            try:
                return _preview_url(
                    int(track_id), self.limiter, _network_session()
                )
            except Exception as current:
                error = current
                if attempt + 1 < NETWORK_ATTEMPTS:
                    import time

                    time.sleep(BACKOFF[attempt])
        if error is not None:
            raise error
        return None


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _opaque(prefix: str, salt: str, *parts: object) -> str:
    value = "\0".join([salt, *(str(part) for part in parts)]).encode("utf-8")
    return prefix + hashlib.sha256(value).hexdigest()[:24]


def semantic_order_hash(document: Mapping[str, Any]) -> str:
    """Hash displayed seed/list positions and stable track IDs, not metadata."""
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


def _generate_key(
    private: Path, *, comment: str, public: Path, allowed: Path, identity: str
) -> None:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV13Error("ssh-keygen is required; v13 signing fails closed")
    generated = subprocess.run(
        [
            executable,
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            comment,
            "-f",
            str(private),
        ],
        capture_output=True,
        check=False,
    )
    if generated.returncode:
        raise HumanV13Error("Ed25519 key generation failed")
    text = private.with_suffix(".pub").read_text(encoding="utf-8").strip()
    fields = text.split()
    public.write_text(text + "\n", encoding="ascii")
    allowed.write_text(
        f"{identity} {fields[0]} {fields[1]}\n", encoding="ascii"
    )


def _sign_state(directory: Path, state: Path) -> Dict[str, Any]:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV13Error("ssh-keygen is required; v13 signing fails closed")
    with tempfile.TemporaryDirectory(prefix="soundalike-human-v13-state-") as temp:
        private = Path(temp) / "signer"
        public = directory / "signer.pub"
        allowed = directory / "allowed_signers"
        _generate_key(
            private,
            comment="soundalike-human-eval-v13",
            public=public,
            allowed=allowed,
            identity=STATE_IDENTITY,
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
                str(state),
            ],
            capture_output=True,
            check=False,
        )
        generated = Path(str(state) + ".sig")
        if signed.returncode or not generated.is_file():
            raise HumanV13Error("v13 state signing failed")
        os.replace(generated, directory / "state.sig")
    metadata = {
        "algorithm": "Ed25519 detached SSH signature",
        "namespace": STATE_NAMESPACE,
        "identity": STATE_IDENTITY,
        "state_sha256": file_hash(state),
        "signer_public_key_sha256": file_hash(directory / "signer.pub"),
        "allowed_signers_sha256": file_hash(directory / "allowed_signers"),
        "signature_sha256": file_hash(directory / "state.sig"),
    }
    _write(directory / "signature-metadata.json", metadata)
    return metadata


def _collector(public_dir: Path, private_dir: Path) -> Dict[str, Path]:
    private = private_dir / "collector_signer"
    public = public_dir / "collector_signer.pub"
    allowed = public_dir / "collector_allowed_signers"
    _generate_key(
        private,
        comment="soundalike-human-rater-v13",
        public=public,
        allowed=allowed,
        identity=COLLECTOR_IDENTITY,
    )
    # _generate_key writes a second public copy beside the private key; it is
    # redundant and removed so gitignored private state has one trust root.
    private.with_suffix(".pub").unlink(missing_ok=True)
    return {"private": private, "public": public, "allowed": allowed}


def freeze_pack(
    diagnostics_path: Path,
    index_path: Path,
    compact_report_path: Path,
    preregistration_path: Path,
    evaluator_path: Path,
    public_dir: Path,
    private_dir: Path,
) -> Dict[str, Path]:
    """Freeze a new signed 60-seed baseline-vs-CLAP pack before ratings."""
    if public_dir.exists() and any(public_dir.iterdir()):
        raise HumanV13Error("v13 public output must be empty; frozen packs are immutable")
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    if not evaluator_path.is_file():
        raise HumanV13Error("v13 evaluator is missing")

    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    compact = json.loads(compact_report_path.read_text(encoding="utf-8"))
    for name, document in (
        ("preregistration", preregistration),
        ("diagnostics", diagnostics),
        ("compact report", compact),
    ):
        if content_hash(document) != document.get("content_sha256"):
            raise HumanV13Error(f"{name} content hash mismatch")
    if (
        preregistration.get("content_sha256") != PREREGISTRATION_SHA256
        or diagnostics.get("commercial_human_ratings_used") != 0
        or diagnostics.get("proxy_evidence_is_deciding") is not False
        or diagnostics.get("safety", {}).get("production_changed") is not False
        or diagnostics.get("preregistration_content_sha256")
        != PREREGISTRATION_SHA256
        or compact.get("preregistration_content_sha256")
        != PREREGISTRATION_SHA256
        or diagnostics.get("compact_asset_sha256")
        != compact.get("asset", {}).get("sha256")
        or compact.get("coverage", {}).get("pending") != 0
        or compact.get("coverage", {}).get("error") != 0
        or compact.get("asset", {}).get("bytes", 70_000_001) > 70_000_000
        or compact.get("float16_reload_metrics", {}).get(
            "mean_top50_overlap", 0.0
        )
        < 0.75
    ):
        raise HumanV13Error("v13 prerequisite isolation state is invalid")
    selected = str(diagnostics.get("selected_challenger"))
    variant = diagnostics.get("variants", {}).get(selected)
    if not isinstance(variant, Mapping) or not variant.get("metrics", {}).get(
        "passes_proxy_safety"
    ):
        raise HumanV13Error("selected CLAP challenger did not pass proxy safety")
    baseline_records = diagnostics["production_baseline"]["records"]
    challenger_records = variant["records"]
    if len(baseline_records) != EXPECTED_SEEDS or len(challenger_records) != EXPECTED_SEEDS:
        raise HumanV13Error("v13 requires exactly 60 paired seed records")
    challenger_by_seed = {
        str(item["seed_id"]): item for item in challenger_records
    }

    import numpy as np

    with np.load(index_path, allow_pickle=False) as index:
        ids = np.asarray(index["track_ids"], dtype=np.int64)
        titles = np.asarray(index["titles"])
        artists = np.asarray(index["artists"])
    if (
        len(ids) != EXPECTED_ROWS
        or hashlib.sha256(ids.tobytes()).hexdigest() != TRACK_IDS_SHA256
    ):
        raise HumanV13Error("v13 source index row identity mismatch")
    salt = secrets.token_hex(32)
    public_seeds = []
    role_records = []
    scenes = set()
    for baseline in baseline_records:
        seed_id = str(baseline["seed_id"])
        challenger = challenger_by_seed.get(seed_id)
        if challenger is None:
            raise HumanV13Error(f"missing challenger rows for {seed_id}")
        query_row = int(baseline["query_row"])
        if not 0 <= query_row < len(ids):
            raise HumanV13Error(f"query row is out of range for {seed_id}")
        if int(challenger["query_row"]) != query_row:
            raise HumanV13Error(f"paired query row mismatch for {seed_id}")
        scene = str(baseline["scene"])
        scenes.add(scene)
        result_catalog: Dict[int, Dict[str, Any]] = {}
        lists = []
        for role, record in (
            ("production_baseline", baseline),
            ("challenger", challenger),
        ):
            rows = list(map(int, record["rows"]))
            if len(rows) != RESULTS_PER_METHOD or len(set(rows)) != RESULTS_PER_METHOD:
                raise HumanV13Error(f"{seed_id}/{role} is not a distinct top five")
            if any(row < 0 or row >= len(ids) for row in rows):
                raise HumanV13Error(f"{seed_id}/{role} contains an out-of-range row")
            list_id = _opaque("L13-", salt, seed_id, role)
            ranking = []
            for position, row in enumerate(rows, start=1):
                track_id = int(ids[row])
                result_id = _opaque("T13-", salt, seed_id, track_id)
                result_catalog.setdefault(
                    track_id,
                    {
                        "result_id": result_id,
                        "track_id": track_id,
                        "deezer_track_id": track_id,
                        "title": str(titles[row]),
                        "artist": str(artists[row]),
                    },
                )
                ranking.append({"position": position, "result_id": result_id})
            lists.append({"list_id": list_id, "ranking": ranking})
            role_records.append(
                {
                    "seed_id": seed_id,
                    "list_id": list_id,
                    "method_role": role,
                    "method_name": (
                        "current_production_dual_sonic"
                        if role == "production_baseline"
                        else selected
                    ),
                }
            )
        public_seeds.append(
            {
                "seed_id": _opaque("S13-", salt, seed_id),
                "source_seed_id": seed_id,
                "scene": scene,
                "query": {
                    "track_id": int(ids[query_row]),
                    "deezer_track_id": int(ids[query_row]),
                    "title": str(titles[query_row]),
                    "artist": str(artists[query_row]),
                },
                "results": list(result_catalog.values()),
                "lists": lists,
            }
        )
    if len(public_seeds) != EXPECTED_SEEDS or len(scenes) != EXPECTED_SCENES:
        raise HumanV13Error("v13 pack must preserve 60 seeds and 13 scenes")

    public: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pack_kind": "blinded_actual_served_lists_clap_development",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "seed_count": EXPECTED_SEEDS,
        "scene_count": EXPECTED_SCENES,
        "results_per_method": RESULTS_PER_METHOD,
        "same_artist_filtered": True,
        "shared_results_rated_once": True,
        "stable_id_field": "deezer_track_id",
        "preview_urls_resolved_at_freeze": False,
        "audio_access": {
            "provider": "Deezer public 30-second previews",
            "resolution": "fresh on demand through /api/preview?id=<numeric_deezer_id>",
            "signed_preview_urls_persisted": False,
            "browser_cache_scope": "memory only; current page session",
            "refresh_on_playback_failure": True,
            "external_request_disclosure": (
                "Only the numeric Deezer track ID is requested; no rating value, "
                "anonymous rater ID, session ID, or localStorage data is transmitted."
            ),
            "fallbacks": ["Deezer track page", "Spotify title/artist search"],
        },
        "seeds": public_seeds,
    }
    public["semantic_order_sha256"] = semantic_order_hash(public)
    public["content_sha256"] = content_hash(public)

    private: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key_kind": "private_method_role_key",
        "rankings_state": "RANKINGS_LOCKED",
        "served_lists_sha256": public["content_sha256"],
        "semantic_order_sha256": public["semantic_order_sha256"],
        "blinding_salt_sha256": hashlib.sha256(salt.encode("ascii")).hexdigest(),
        "records": role_records,
    }
    private["content_sha256"] = content_hash(private)
    method_key = private_dir / "method-key-v13.json"
    _write(method_key, private)

    collector = _collector(public_dir, private_dir)
    protocol: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_kind": "blinded_served_list_human_listener_clap_development",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "served_lists_sha256": public["content_sha256"],
        "semantic_order_sha256": public["semantic_order_sha256"],
        "private_key_sha256": private["content_sha256"],
        "seed_count": EXPECTED_SEEDS,
        "scene_count": EXPECTED_SCENES,
        "results_per_method": RESULTS_PER_METHOD,
        "assignment": "per-session randomized seed, list, and display order",
        "rating_scale": (
            "MIREX three-class result similarity plus optional integer 0-10; "
            "three-class list coherence, unrelated top-3 count, and junk/version flag"
        ),
        "collector_public_key_sha256": file_hash(collector["public"]),
        "collector_allowed_signers_sha256": file_hash(collector["allowed"]),
        "human_evidence_gate": (
            "Each browser export needs detached Ed25519 collector approval; "
            "at least three independent raters are required before AC#3 can be tested."
        ),
        "preregistration_content_sha256": PREREGISTRATION_SHA256,
        "diagnostics_content_sha256": diagnostics["content_sha256"],
        "compact_asset_sha256": compact["asset"]["sha256"],
        "evaluator_sha256": file_hash(evaluator_path),
        "production_changed": False,
        "deployed": False,
        "commercial_final_opened": False,
        "ac3_claimed": False,
    }
    protocol["content_sha256"] = content_hash(protocol)

    lists_path = public_dir / "served-lists-v13.json"
    protocol_path = public_dir / "protocol-v13.json"
    state_path = public_dir / "state.json"
    _write(lists_path, public)
    _write(protocol_path, protocol)
    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "RANKINGS_LOCKED",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "human_rater_exports_ingested": 0,
        "sonic_human_report_exists": False,
        "production_deployment_blocked": True,
        "served_lists_sha256": public["content_sha256"],
        "semantic_order_sha256": public["semantic_order_sha256"],
        "protocol_sha256": protocol["content_sha256"],
        "private_method_key_sha256": private["content_sha256"],
        "collector_public_key_sha256": file_hash(collector["public"]),
        "collector_allowed_signers_sha256": file_hash(collector["allowed"]),
        "evaluator_sha256": file_hash(evaluator_path),
        "diagnostics_content_sha256": diagnostics["content_sha256"],
        "compact_asset_sha256": compact["asset"]["sha256"],
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "production_changed": False,
        "deployed": False,
        "commercial_final_opened": False,
        "ac3_claimed": False,
    }
    state["content_sha256"] = content_hash(state)
    _write(state_path, state)
    _sign_state(public_dir, state_path)
    return {
        "protocol": protocol_path,
        "lists": lists_path,
        "state": state_path,
        "signature": public_dir / "state.sig",
        "method_key": method_key,
        "collector_private": collector["private"],
        "collector_public": collector["public"],
        "collector_allowed_signers": collector["allowed"],
    }


def verify_pack(
    directory: Path,
    *,
    private_key: Optional[Path] = None,
    require_trusted: bool = False,
) -> Dict[str, Any]:
    """Verify hashes, semantic order, detached state signature, and optional role key."""
    protocol_path = directory / "protocol-v13.json"
    lists_path = directory / "served-lists-v13.json"
    state_path = directory / "state.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lists = json.loads(lists_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if require_trusted and (
        protocol.get("content_sha256") != TRUSTED_V13_PROTOCOL
        or lists.get("content_sha256") != TRUSTED_V13_LISTS
        or state.get("content_sha256") != TRUSTED_V13_STATE
        or any(
            not (directory / name).is_file()
            or file_hash(directory / name) != digest
            for name, digest in TRUSTED_V13_FILES.items()
        )
    ):
        raise HumanV13Error("v13 pack differs from the committed trust anchors")
    for name, document in (
        ("protocol", protocol),
        ("lists", lists),
        ("state", state),
    ):
        if (
            document.get("schema_version") != SCHEMA_VERSION
            or content_hash(document) != document.get("content_sha256")
            or document.get("rankings_state") != "RANKINGS_LOCKED"
        ):
            raise HumanV13Error(f"v13 {name} hash/schema/lock mismatch")
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
    ):
        raise HumanV13Error("v13 pack hash/order/rating binding mismatch")
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV13Error("ssh-keygen is required to verify the v13 pack")
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
        raise HumanV13Error("v13 state signature is invalid")
    if private_key is not None:
        key = json.loads(private_key.read_text(encoding="utf-8"))
        if (
            content_hash(key) != key.get("content_sha256")
            or key.get("content_sha256") != protocol["private_key_sha256"]
            or key.get("served_lists_sha256") != lists["content_sha256"]
            or key.get("semantic_order_sha256") != semantic
        ):
            raise HumanV13Error("v13 private method key binding mismatch")
        public_ids = {
            item["list_id"] for seed in lists["seeds"] for item in seed["lists"]
        }
        private_ids = {item["list_id"] for item in key["records"]}
        if public_ids != private_ids:
            raise HumanV13Error("v13 private method key is incomplete")
    return {"protocol": protocol, "lists": lists, "state": state}


def audit_v13_preview_coverage(
    lists_path: Path, endpoint: str, *, workers: int = 10
) -> Dict[str, Any]:
    report = audit_preview_coverage(lists_path, endpoint, workers=workers)
    report.pop("content_sha256", None)
    report["schema_version"] = SCHEMA_VERSION
    report["artifact_kind"] = "v13_live_preview_resolution_audit"
    report["human_ratings_used"] = 0
    report["production_changed"] = False
    report["content_sha256"] = content_hash(report)
    return report


def serve(
    directory: Path,
    evaluator: Path,
    *,
    port: int = 8000,
) -> None:
    from http.server import ThreadingHTTPServer

    verified = verify_pack(directory, require_trusted=True)
    evaluator_sha256 = file_hash(evaluator)
    if not (
        evaluator_sha256
        == verified["protocol"].get("evaluator_sha256")
        == verified["state"].get("evaluator_sha256")
    ):
        raise HumanV13Error("served evaluator hash differs from the signed v13 state")
    server = ThreadingHTTPServer(
        ("127.0.0.1", int(port)),
        evaluator_handler(
            evaluator,
            directory / "protocol-v13.json",
            directory / "served-lists-v13.json",
            resolver=FreshV13PreviewResolver(),
        ),
    )
    print(f"Private blinded v13 evaluator: http://127.0.0.1:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    preregistration = (
        root
        / ".goals/human-quality-recommendations/"
        "protocol-v13-clap-development"
    )
    public = (
        root
        / ".goals/human-quality-recommendations/"
        "protocol-v13-clap-human-development"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--diagnostics", type=Path, required=True)
    freeze.add_argument("--compact-report", type=Path, required=True)
    freeze.add_argument("--index", type=Path, default=root / "ml_data/deepvibe_index_v5.npz")
    freeze.add_argument(
        "--preregistration",
        type=Path,
        default=preregistration / "preregistration-v13-r3.json",
    )
    freeze.add_argument(
        "--evaluator",
        type=Path,
        default=root / "benchmarks/human_eval_v13.html",
    )
    freeze.add_argument("--public-dir", type=Path, default=public)
    freeze.add_argument(
        "--private-dir", type=Path, default=root / "ml_data/clap_v13/human_eval"
    )
    verify = sub.add_parser("verify")
    verify.add_argument("--directory", type=Path, default=public)
    verify.add_argument("--private-key", type=Path)
    local = sub.add_parser("serve")
    local.add_argument("--directory", type=Path, default=public)
    local.add_argument(
        "--evaluator",
        type=Path,
        default=root / "benchmarks/human_eval_v13.html",
    )
    local.add_argument("--port", type=int, default=8000)
    audit = sub.add_parser("audit")
    audit.add_argument("--lists", type=Path, default=public / "served-lists-v13.json")
    audit.add_argument("--endpoint", default=PRODUCTION_PREVIEW_ENDPOINT)
    audit.add_argument("--workers", type=int, default=10)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        paths = freeze_pack(
            args.diagnostics,
            args.index,
            args.compact_report,
            args.preregistration,
            args.evaluator,
            args.public_dir,
            args.private_dir,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "verify":
        value = verify_pack(
            args.directory, private_key=args.private_key, require_trusted=True
        )
        print(
            f"RANKINGS_LOCKED: {value['lists']['content_sha256']} "
            "(ratings_count_at_freeze=0)"
        )
    elif args.command == "serve":
        serve(args.directory, args.evaluator, port=args.port)
    else:
        report = audit_v13_preview_coverage(
            args.lists, args.endpoint, workers=args.workers
        )
        _write(args.output, report)
        print(
            f"{report['ranked_positions']['available']}/"
            f"{report['ranked_positions']['total']} ranked positions resolve"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
