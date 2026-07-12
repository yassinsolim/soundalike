"""Freeze iteration-9 served lists for an anonymous, blinded human evaluation.

Usage:
  python -m soundalike.ml.human_eval_v10 freeze --v9-lists .goals/human-quality-recommendations/artifacts/catalog-powered-blind-lists-v9.json --v9-key .goals/human-quality-recommendations/artifacts/catalog-powered-blind-key-v9.json --out-dir human-eval-v10

Open ``benchmarks/human_eval_v10.html`` locally and import the emitted protocol
and served-list files. Keep the method-key file away from raters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union
from urllib.request import urlopen

SCHEMA_VERSION = 10
EXPECTED_SEEDS = 60
EXPECTED_SCENES = 13


class FreezeError(ValueError):
    """The source rankings cannot safely be frozen."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def content_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _preview_for_track(track_id: object) -> str | None:
    try:
        numeric = int(track_id)
        with urlopen(f"https://api.deezer.com/track/{numeric}", timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        preview = payload.get("preview")
        return preview if isinstance(preview, str) and preview.startswith("https://") else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _sign_state(directory: Path, state_path: Path) -> Dict[str, Any]:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise FreezeError("ssh-keygen is required; protocol signing fails closed")
    with tempfile.TemporaryDirectory(prefix="soundalike-human-v10-") as temporary:
        private = Path(temporary) / "signer"
        generated = subprocess.run(
            [
                executable, "-q", "-t", "ed25519", "-N", "",
                "-C", "soundalike-human-eval-v10", "-f", str(private),
            ],
            capture_output=True, check=False,
        )
        if generated.returncode:
            raise FreezeError("Ed25519 key generation failed")
        public = private.with_suffix(".pub").read_text(encoding="utf-8").strip()
        fields = public.split()
        (directory / "signer.pub").write_text(public + "\n", encoding="utf-8")
        (directory / "allowed_signers").write_text(
            f"soundalike-human-eval {fields[0]} {fields[1]}\n", encoding="utf-8"
        )
        signed = subprocess.run(
            [
                executable, "-Y", "sign", "-f", str(private),
                "-n", "soundalike-human-eval", str(state_path),
            ],
            capture_output=True, check=False,
        )
        generated_signature = Path(str(state_path) + ".sig")
        if signed.returncode or not generated_signature.is_file():
            raise FreezeError("detached Ed25519 state signing failed")
        os.replace(generated_signature, directory / "state.sig")
    metadata = {
        "algorithm": "Ed25519 detached SSH signature",
        "namespace": "soundalike-human-eval",
        "identity": "soundalike-human-eval",
        "state_sha256": file_hash(state_path),
        "signer_public_key_sha256": file_hash(directory / "signer.pub"),
        "allowed_signers_sha256": file_hash(directory / "allowed_signers"),
        "signature_sha256": file_hash(directory / "state.sig"),
    }
    _write(directory / "signature-metadata.json", metadata)
    return metadata


def _create_collector_key(directory: Path) -> Dict[str, Any]:
    """Create the local operator key used to approve genuine rater exports."""
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise FreezeError("ssh-keygen is required; collector key creation fails closed")
    private = directory / "collector_signer"
    generated = subprocess.run(
        [
            executable, "-q", "-t", "ed25519", "-N", "",
            "-C", "soundalike-human-rater-v10", "-f", str(private),
        ],
        capture_output=True, check=False,
    )
    if generated.returncode:
        raise FreezeError("collector Ed25519 key generation failed")
    public = private.with_suffix(".pub")
    fields = public.read_text(encoding="utf-8").strip().split()
    allowed = directory / "collector_allowed_signers"
    allowed.write_text(
        f"soundalike-human-rater {fields[0]} {fields[1]}\n", encoding="utf-8"
    )
    return {
        "private": private,
        "public": public,
        "allowed_signers": allowed,
        "public_key_sha256": file_hash(public),
        "allowed_signers_sha256": file_hash(allowed),
    }


def approve_export(
    export_path: Union[Path, str],
    collector_private_key: Union[Path, str],
) -> Path:
    """Operator-attest one reviewed actual-listener export.

    The browser HMAC only detects accidental edits.  This detached Ed25519
    approval is the trust boundary that prevents self-attested proxy rows from
    entering a ``sonic_human`` report.
    """
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise FreezeError("ssh-keygen is required to approve a rater export")
    export = Path(export_path)
    private = Path(collector_private_key)
    if not export.is_file() or not private.is_file():
        raise FreezeError("export and collector private key must exist")
    signature = Path(str(export) + ".sig")
    signature.unlink(missing_ok=True)
    signed = subprocess.run(
        [
            executable, "-Y", "sign", "-f", str(private),
            "-n", "soundalike-human-rater", str(export),
        ],
        capture_output=True, check=False,
    )
    if signed.returncode or not signature.is_file():
        raise FreezeError("collector approval signature failed")
    return signature


def _artist_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(feat|featuring|ft)\b.*$", "", text, flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("&", " and ")))


def _opaque(prefix: str, *parts: object) -> str:
    raw = "\0".join(str(part) for part in parts).encode()
    return prefix + hashlib.sha256(raw).hexdigest()[:20]


def _validate_v9(document: Mapping[str, Any], key: Mapping[str, Any]) -> None:
    if document.get("schema_version") != 9 or key.get("schema_version") != 9:
        raise FreezeError("both inputs must be iteration-9 artifacts")
    declared = document.get("content_sha256")
    if not isinstance(declared, str) or content_hash(document) != declared:
        raise FreezeError("v9 served-list content hash mismatch")
    if key.get("blind_lists_sha256") != declared:
        raise FreezeError("v9 key is not bound to the served lists")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise FreezeError("v9 served lists contain no records")
    mappings = key.get("records")
    if not isinstance(mappings, list):
        raise FreezeError("v9 method key has no records")
    role_map = {(row.get("id"), row.get("alias")): row.get("method_role")
                for row in mappings}
    expected = {"production_baseline", "challenger"}
    for record in records:
        lists = record.get("lists", [])
        roles = {role_map.get((record.get("id"), item.get("alias")))
                 for item in lists}
        if len(lists) != 2 or roles != expected:
            raise FreezeError(f"{record.get('id')}: expected exactly two mapped methods")


def freeze_pack(
    v9_lists_path: Union[Path, str],
    v9_key_path: Union[Path, str],
    out_dir: Union[Path, str],
    *,
    enforce_real_suite: bool = True,
    resolve_previews: bool = False,
    evaluator_path: Union[Path, str] = "benchmarks/human_eval_v10.html",
) -> Dict[str, Path]:
    """Freeze top-5 actual served rankings without exposing method identity."""
    source_path, source_key_path = Path(v9_lists_path), Path(v9_key_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_key = json.loads(source_key_path.read_text(encoding="utf-8"))
    _validate_v9(source, source_key)

    records = source["records"]
    scenes = {str(row.get("scene")) for row in records}
    if enforce_real_suite and (
        len(records) != EXPECTED_SEEDS or len(scenes) != EXPECTED_SCENES
    ):
        raise FreezeError("real v9 suite must contain exactly 60 seeds / 13 scenes")

    roles = {(row["id"], row["alias"]): row["method_role"]
             for row in source_key["records"]}
    blinding_salt = os.urandom(32).hex()
    preview_ids = {
        int(row["track_id"])
        for seed in records
        for old_list in seed["lists"]
        for row in old_list.get("results", [])[:10]
        if row.get("track_id") is not None and not row.get("preview_url")
    }
    preview_ids.update(
        int(seed["query"]["track_id"])
        for seed in records
        if seed.get("query", {}).get("track_id") is not None
        and not seed["query"].get("preview_url")
    )
    resolved_previews: Dict[int, str | None] = {}
    if resolve_previews:
        ordered_ids = sorted(preview_ids)
        with ThreadPoolExecutor(max_workers=16) as executor:
            resolved_previews = dict(zip(
                ordered_ids, executor.map(_preview_for_track, ordered_ids)
            ))
    public_seeds = []
    private_records = []
    for seed in records:
        seed_id = str(seed["id"])
        seed_artist = _artist_key(str(seed["query"]["artist"]))
        result_catalog: Dict[str, Dict[str, Any]] = {}
        public_lists = []
        for old_list in seed["lists"]:
            old_alias = str(old_list["alias"])
            list_id = _opaque("L-", blinding_salt, seed_id, old_alias)
            eligible = [
                row for row in old_list.get("results", [])
                if not bool(row.get("same_artist"))
                and _artist_key(str(row.get("artist", ""))) != seed_artist
            ][:5]
            if len(eligible) != 5:
                raise FreezeError(f"{seed_id}/{old_alias}: fewer than five filtered results")
            ranking = []
            for position, row in enumerate(eligible, 1):
                identity = str(row.get("track_id") or (
                    _artist_key(str(row.get("artist", ""))) + "\0" +
                    str(row.get("title", "")).casefold()
                ))
                result_id = _opaque("T-", blinding_salt, seed_id, identity)
                availability = row.get("preview_availability")
                preview_url = row.get("preview_url")
                if not isinstance(preview_url, str) or not preview_url.startswith(
                    ("https://", "http://")
                ):
                    preview_url = resolved_previews.get(int(row["track_id"])) \
                        if row.get("track_id") is not None else None
                result_catalog.setdefault(result_id, {
                    "result_id": result_id,
                    "track_id": row.get("track_id"),
                    "title": str(row.get("title", "")),
                    "artist": str(row.get("artist", "")),
                    "preview_url": preview_url,
                    "preview_available": bool(
                        preview_url or isinstance(availability, Mapping)
                        and availability.get("status") == "available"
                    ),
                })
                ranking.append({"position": position, "result_id": result_id})
            public_lists.append({"list_id": list_id, "ranking": ranking})
            private_records.append({
                "seed_id": seed_id,
                "list_id": list_id,
                "method_role": roles[(seed_id, old_alias)],
            })
        public_seeds.append({
            "seed_id": seed_id,
            "scene": str(seed["scene"]),
            "query": {
                "title": str(seed["query"]["title"]),
                "artist": str(seed["query"]["artist"]),
                "track_id": seed["query"].get("track_id"),
                "preview_url": (
                    seed["query"].get("preview_url")
                    or resolved_previews.get(int(seed["query"]["track_id"]))
                    if seed["query"].get("track_id") is not None else None
                ),
            },
            "results": list(result_catalog.values()),
            "lists": public_lists,
        })

    public: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pack_kind": "blinded_actual_served_lists",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "seed_count": len(public_seeds),
        "scene_count": len(scenes),
        "results_per_method": 5,
        "same_artist_filtered": True,
        "shared_results_rated_once": True,
        "preview_urls_resolved_at_freeze": bool(resolve_previews),
        "preview_urls_are_supporting_playback_only": True,
        "seeds": public_seeds,
    }
    public["content_sha256"] = content_hash(public)

    private: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key_kind": "private_method_role_key",
        "rankings_state": "RANKINGS_LOCKED",
        "served_lists_sha256": public["content_sha256"],
        "source_v9_lists_file_sha256": file_hash(source_path),
        "source_v9_key_file_sha256": file_hash(source_key_path),
        "blinding_salt_sha256": hashlib.sha256(
            blinding_salt.encode("ascii")
        ).hexdigest(),
        "records": private_records,
    }
    private["content_sha256"] = content_hash(private)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    collector = _create_collector_key(output)
    protocol: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_kind": "blinded_served_list_human_listener",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "served_lists_sha256": public["content_sha256"],
        "private_key_sha256": private["content_sha256"],
        "seed_count": len(public_seeds),
        "scene_count": len(scenes),
        "results_per_method": 5,
        "assignment": "per-session randomized seed, list, and display order",
        "duplicate_accounting": (
            "a shared track is rated once per seed and the same human grade is "
            "credited to each method position containing it"
        ),
        "estimated_workload": (
            "60 seeds x up to 9-10 unique results; about 90-150 minutes for a "
            "complete rater, with partial export/resume supported"
        ),
        "human_evidence_gate": (
            "Each browser export must also have a detached Ed25519 approval "
            "from the local study collector before aggregation."
        ),
        "collector_public_key_sha256": collector["public_key_sha256"],
        "collector_allowed_signers_sha256": collector["allowed_signers_sha256"],
        "integrity_notice": (
            "Exports use a local per-session HMAC key carried in the export. "
            "This detects accidental changes; it proves integrity, not identity or authenticity."
        ),
    }
    protocol["content_sha256"] = content_hash(protocol)

    evaluator = Path(evaluator_path)
    if not evaluator.is_file():
        raise FreezeError(f"standalone evaluator is missing: {evaluator}")
    paths = {
        "protocol": output / "protocol-v10.json",
        "lists": output / "served-lists-v10.json",
        "key": output / "method-key-v10.json",
        "state": output / "state.json",
        "signature": output / "state.sig",
        "signer": output / "signer.pub",
        "allowed_signers": output / "allowed_signers",
        "signature_metadata": output / "signature-metadata.json",
        "collector_private": collector["private"],
        "collector_public": collector["public"],
        "collector_allowed_signers": collector["allowed_signers"],
    }
    _write(paths["lists"], public)
    _write(paths["key"], private)
    _write(paths["protocol"], protocol)
    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "RANKINGS_LOCKED",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "human_rater_exports_ingested": 0,
        "sonic_human_report_exists": False,
        "production_deployment_blocked": True,
        "served_lists_sha256": public["content_sha256"],
        "protocol_sha256": protocol["content_sha256"],
        "private_method_key_sha256": private["content_sha256"],
        "collector_public_key_sha256": collector["public_key_sha256"],
        "collector_allowed_signers_sha256": collector["allowed_signers_sha256"],
        "evaluator_sha256": file_hash(evaluator),
        "source_v9_lists_file_sha256": file_hash(source_path),
        "source_v9_key_file_sha256": file_hash(source_key_path),
        "locked_at": datetime.now(timezone.utc).isoformat(),
    }
    state["content_sha256"] = content_hash(state)
    _write(paths["state"], state)
    _sign_state(output, paths["state"])
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze exact v9 actual served lists for blinded human rating.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--v9-lists", required=True, type=Path)
    freeze.add_argument("--v9-key", required=True, type=Path)
    freeze.add_argument("--out-dir", required=True, type=Path)
    freeze.add_argument(
        "--resolve-previews", action="store_true",
        help="resolve public Deezer preview URLs into the locked local pack",
    )
    freeze.add_argument(
        "--evaluator", type=Path, default=Path("benchmarks/human_eval_v10.html")
    )
    approve = sub.add_parser(
        "approve",
        help="operator-approve a reviewed actual human export with Ed25519",
    )
    approve.add_argument("--export", required=True, type=Path)
    approve.add_argument("--collector-key", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "approve":
        print(approve_export(args.export, args.collector_key))
    else:
        paths = freeze_pack(
            args.v9_lists, args.v9_key, args.out_dir,
            resolve_previews=args.resolve_previews,
            evaluator_path=args.evaluator,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        print("RANKINGS_LOCKED; no ratings were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
