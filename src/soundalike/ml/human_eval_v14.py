"""Freeze and serve the v14 identity-corrected CLAP blind development study.

v13 is superseded because the final v14 diagnostics changed the selected
conservative-challenger lists for 2 seeds via artist-identity collision
correction.  v14 is a NEW immutable pack — v13 is not modified.

Key invariants (identical to v13 except where noted)
-----------------------------------------------------
* 60 seeds, 13 scenes, 2 lists × 5 results, RANKINGS_LOCKED, ratings = 0.
* Per-session randomised seed/list/display order.
* Opaque S14-/L14-/T14- IDs; fresh uncommitted blinding salt.
* Ed25519 detached signature over state (same ssh-keygen pattern).
* Only .pub / allowed_signers / .sig files committed; private key stays local.
* Three-class result similarity + optional 0–10 score; three-class list
  coherence, unrelated-top-3 count, junk/version flag.
* ``no-store`` privacy; only numeric Deezer ID leaves the browser.

Supersession provenance
-----------------------
``freeze_pack`` records ``supersedes_v13`` with the v13 committed-pack content
hashes, ratings state, semantic-order hash, the concrete semantic diff
(changed_seed_count, per-seed positions), and the v13/v14 diagnostics hashes.
Fails if the v14 diagnostics produce no list changes relative to v13.
``ratings_discarded`` and ``ratings_migrated`` are both 0 (v13 had 0 ratings).

Collector trust root
--------------------
A fresh collector key pair is generated per-freeze.  The private key lives
only in ``private_dir``; the public key and allowed_signers are committed.

Trust anchors
-------------
``TRUSTED_V14_PROTOCOL``, ``TRUSTED_V14_LISTS``, ``TRUSTED_V14_STATE``, and
``TRUSTED_V14_FILES`` are empty-string sentinels until the orchestrator patches
them after freeze.  ``verify_pack(require_trusted=True)`` fails closed if the
constants are not yet populated; ``require_trusted=False`` skips that check and
is used during generation.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .clap_catalog_v13 import (
    BACKOFF,
    EXPECTED_ROWS,
    NETWORK_ATTEMPTS,
    PREREGISTRATION_SHA256,
    RateLimiter,
    SCHEMA_VERSION as _SCHEMA_V13,
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
from .human_eval_v13 import (
    TRUSTED_V13_LISTS,
    TRUSTED_V13_PROTOCOL,
    TRUSTED_V13_STATE,
    semantic_order_hash as v13_semantic_order_hash,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 14
EXPECTED_SEEDS = 60
EXPECTED_SCENES = 13
RESULTS_PER_METHOD = 5

STATE_IDENTITY = "soundalike-human-eval-v14"
STATE_NAMESPACE = "soundalike-human-eval-v14"
COLLECTOR_IDENTITY = "soundalike-human-rater"
COLLECTOR_NAMESPACE = "soundalike-human-rater"

# Byte-exact trust anchors for the immutable zero-rating v14 pack.
TRUSTED_V14_PROTOCOL = "38fdb950fba1673de7d3990534e9a7801f6aa29df27e53f53fb698df8bebc2e2"
TRUSTED_V14_LISTS = "5e7d852ea7ea5d1e9fe04f1b77c156f1c9a0894cd2d913f16e08d6ec304358b2"
TRUSTED_V14_STATE = "823fd692239803629bb262cc05b2d6bfaf5b2ef6b0fd28620cd223e2caf2d33f"
TRUSTED_V14_FILES: Dict[str, str] = {
    "allowed_signers": "6f057c3088062c75ca85c2ffcc6c5f67e6bfea772c78c36010dcbc3a36be2e32",
    "collector_allowed_signers": "af787b33d44f2db435dd620fe1fe97c02473d7ff3cf7f33da1f896e979bd7101",
    "collector_signer.pub": "514d96fd9262e5e5f24ebdf5cc0287573f9e85b2f101cbdf5f48c12b598550f8",
    "protocol-v14.json": "5cb046b463aedd0e36ce554b5734501fe6be5f0b8d3c89170657f9dfacc02fd3",
    "served-lists-v14.json": "974c4490f8961352fb74974ab5fe445fbc53c762db2f855906a4c163dafbd8ec",
    "signature-metadata.json": "0a86b47cb6aefd243baec5e689984a8c7a22e0a67ac46031e66fbd85b8a123c9",
    "signer.pub": "4c9269f9f63de05ec932fc52ea48a3262f8beb31e4ed09ae654c6c20f0cd759d",
    "state.json": "72d987d5edb215e6ea31242c1d25815155142629bb0695e807fcecf46f1a617b",
    "state.sig": "3d42de54608511a8dff059921a8817c67c9f57fa0df4ccad347cf2fe690ed121",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HumanV14Error(ClapCatalogError):
    """The v14 human-development pack is unsafe or malformed."""


# ---------------------------------------------------------------------------
# Preview resolver (same pattern as v13)
# ---------------------------------------------------------------------------


class FreshV14PreviewResolver:
    """Provider-aware fresh resolver for the loopback study server."""

    def __init__(self) -> None:
        self.limiter = RateLimiter(10.0)

    def __call__(self, track_id: int) -> Optional[str]:
        error: Optional[Exception] = None
        for attempt in range(NETWORK_ATTEMPTS):
            try:
                return _preview_url(int(track_id), self.limiter, _network_session())
            except Exception as current:
                error = current
                if attempt + 1 < NETWORK_ATTEMPTS:
                    import time

                    time.sleep(BACKOFF[attempt])
        if error is not None:
            raise error
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Key generation / signing
# ---------------------------------------------------------------------------


def _generate_key(
    private: Path,
    *,
    comment: str,
    public: Path,
    allowed: Path,
    identity: str,
) -> None:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV14Error("ssh-keygen is required; v14 signing fails closed")
    generated = subprocess.run(
        [executable, "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", str(private)],
        capture_output=True,
        check=False,
    )
    if generated.returncode:
        raise HumanV14Error("Ed25519 key generation failed")
    text = private.with_suffix(".pub").read_text(encoding="utf-8").strip()
    fields = text.split()
    public.write_text(text + "\n", encoding="ascii")
    allowed.write_text(f"{identity} {fields[0]} {fields[1]}\n", encoding="ascii")


def _sign_state(directory: Path, state: Path) -> Dict[str, Any]:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV14Error("ssh-keygen is required; v14 signing fails closed")
    with tempfile.TemporaryDirectory(prefix="soundalike-human-v14-state-") as temp:
        private = Path(temp) / "signer"
        public = directory / "signer.pub"
        allowed = directory / "allowed_signers"
        _generate_key(
            private,
            comment=STATE_IDENTITY,
            public=public,
            allowed=allowed,
            identity=STATE_IDENTITY,
        )
        signed = subprocess.run(
            [executable, "-Y", "sign", "-f", str(private), "-n", STATE_NAMESPACE, str(state)],
            capture_output=True,
            check=False,
        )
        generated = Path(str(state) + ".sig")
        if signed.returncode or not generated.is_file():
            raise HumanV14Error("v14 state signing failed")
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
        comment="soundalike-human-rater-v14",
        public=public,
        allowed=allowed,
        identity=COLLECTOR_IDENTITY,
    )
    private.with_suffix(".pub").unlink(missing_ok=True)
    return {"private": private, "public": public, "allowed": allowed}


# ---------------------------------------------------------------------------
# Semantic diff helpers
# ---------------------------------------------------------------------------


def _compute_semantic_diff(
    v13_records: Sequence[Mapping[str, Any]],
    v14_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return the diff of challenger row selections between v13 and v14."""
    v13_by_seed = {str(r["seed_id"]): list(map(int, r["rows"])) for r in v13_records}
    v14_by_seed = {str(r["seed_id"]): list(map(int, r["rows"])) for r in v14_records}
    changed: List[Dict[str, Any]] = []
    for seed_id, v13_rows in sorted(v13_by_seed.items()):
        v14_rows = v14_by_seed.get(seed_id, [])
        if v13_rows != v14_rows:
            changed.append(
                {
                    "seed_id": seed_id,
                    "v13_rows": v13_rows,
                    "v14_rows": v14_rows,
                }
            )
    return {
        "changed_seed_count": len(changed),
        "changed_positions": changed,
    }


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def freeze_pack(
    v14_diagnostics_path: Path,
    v13_diagnostics_path: Path,
    index_path: Path,
    compact_report_path: Path,
    preregistration_path: Path,
    evaluator_path: Path,
    v13_pack_dir: Path,
    identity_audit_path: Path,
    public_dir: Path,
    private_dir: Path,
) -> Dict[str, Path]:
    """Freeze a new signed v14 pack superseding v13 after collision correction.

    Parameters
    ----------
    v14_diagnostics_path:
        v14 variant-diagnostics JSON (from ``clap_catalog_v14``).
    v13_diagnostics_path:
        v13 variant-diagnostics JSON (for semantic diff computation).
    index_path:
        Production catalog index NPZ (``deepvibe_index_v5.npz``).
    compact_report_path:
        v13 compact-geometry report JSON (shared physical asset).
    preregistration_path:
        v13-r3 preregistration JSON.
    evaluator_path:
        v14 evaluator HTML (``benchmarks/human_eval_v14.html``).
    v13_pack_dir:
        Committed v13 public-pack directory (for supersession hashes).
    identity_audit_path:
        v14 artist-identity audit JSON.
    public_dir:
        Destination for committed public files.
    private_dir:
        Destination for local-only private files (gitignored).
    """
    if public_dir.exists() and any(public_dir.iterdir()):
        raise HumanV14Error(
            "v14 public output must be empty; frozen packs are immutable"
        )
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    if not evaluator_path.is_file():
        raise HumanV14Error("v14 evaluator is missing")

    # ------------------------------------------------------------------
    # Load and verify inputs
    # ------------------------------------------------------------------
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    v14_diagnostics = json.loads(v14_diagnostics_path.read_text(encoding="utf-8"))
    v13_diagnostics = json.loads(v13_diagnostics_path.read_text(encoding="utf-8"))
    compact = json.loads(compact_report_path.read_text(encoding="utf-8"))
    identity_audit = json.loads(identity_audit_path.read_text(encoding="utf-8"))

    for name, document in (
        ("preregistration", preregistration),
        ("v14 diagnostics", v14_diagnostics),
        ("v13 diagnostics", v13_diagnostics),
        ("compact report", compact),
        ("identity audit", identity_audit),
    ):
        if content_hash(document) != document.get("content_sha256"):
            raise HumanV14Error(f"{name} content hash mismatch")

    # Preregistration chain
    if (
        preregistration.get("content_sha256") != PREREGISTRATION_SHA256
        or v14_diagnostics.get("commercial_human_ratings_used") != 0
        or v14_diagnostics.get("proxy_evidence_is_deciding") is not False
        or v14_diagnostics.get("safety", {}).get("production_changed") is not False
        or v14_diagnostics.get("preregistration_content_sha256") != PREREGISTRATION_SHA256
        or compact.get("preregistration_content_sha256") != PREREGISTRATION_SHA256
        or compact.get("coverage", {}).get("pending") != 0
        or compact.get("coverage", {}).get("error") != 0
        or compact.get("asset", {}).get("bytes", 70_000_001) > 70_000_000
        or compact.get("float16_reload_metrics", {}).get("mean_top50_overlap", 0.0) < 0.75
    ):
        raise HumanV14Error("v14 prerequisite isolation state is invalid")

    # Schema version checks
    if v14_diagnostics.get("schema_version") != SCHEMA_VERSION:
        raise HumanV14Error("v14 diagnostics schema_version must be 14")
    if v13_diagnostics.get("schema_version") != _SCHEMA_V13:
        raise HumanV14Error("v13 diagnostics schema_version must be 13")

    # Selected challenger
    v14_selected = str(v14_diagnostics.get("selected_challenger", ""))
    v14_variant = v14_diagnostics.get("variants", {}).get(v14_selected)
    if not isinstance(v14_variant, Mapping) or not v14_variant.get("metrics", {}).get(
        "passes_proxy_safety"
    ):
        raise HumanV14Error("v14 selected CLAP challenger did not pass proxy safety")

    v13_selected = str(v13_diagnostics.get("selected_challenger", ""))
    v13_variant = v13_diagnostics.get("variants", {}).get(v13_selected)
    if not isinstance(v13_variant, Mapping):
        raise HumanV14Error("v13 selected CLAP challenger missing from diagnostics")

    # Semantic diff — must have at least one change
    v13_challenger_records = list(v13_variant.get("records", []))
    v14_challenger_records = list(v14_variant.get("records", []))
    semantic_diff = _compute_semantic_diff(v13_challenger_records, v14_challenger_records)
    if semantic_diff["changed_seed_count"] == 0:
        raise HumanV14Error(
            "v14 diagnostics produced no list changes relative to v13; "
            "a new pack is not warranted"
        )

    # ------------------------------------------------------------------
    # Load v13 committed pack for supersession hashes
    # ------------------------------------------------------------------
    v13_protocol_path = v13_pack_dir / "protocol-v13.json"
    v13_lists_path = v13_pack_dir / "served-lists-v13.json"
    v13_state_path = v13_pack_dir / "state.json"
    for path in (v13_protocol_path, v13_lists_path, v13_state_path):
        if not path.is_file():
            raise HumanV14Error(f"v13 committed pack file missing: {path.name}")
    v13_protocol = json.loads(v13_protocol_path.read_text(encoding="utf-8"))
    v13_lists = json.loads(v13_lists_path.read_text(encoding="utf-8"))
    v13_state = json.loads(v13_state_path.read_text(encoding="utf-8"))

    # Verify v13 pack hashes
    for name, doc, expected in (
        ("v13 protocol", v13_protocol, TRUSTED_V13_PROTOCOL),
        ("v13 lists", v13_lists, TRUSTED_V13_LISTS),
        ("v13 state", v13_state, TRUSTED_V13_STATE),
    ):
        if content_hash(doc) != doc.get("content_sha256"):
            raise HumanV14Error(f"{name} content hash mismatch")
        if expected and doc.get("content_sha256") != expected:
            raise HumanV14Error(f"{name} does not match committed trust anchor")

    # Verify v13 pack has no ratings
    if (
        v13_state.get("ratings_count_at_freeze") != 0
        or v13_protocol.get("ratings_count_at_freeze") != 0
        or v13_lists.get("ratings_count_at_freeze") != 0
    ):
        raise HumanV14Error("v13 pack must have ratings_count=0 before supersession")
    if v13_state.get("rankings_state") != "RANKINGS_LOCKED":
        raise HumanV14Error("v13 pack must be RANKINGS_LOCKED for supersession")

    old_semantic_order = v13_semantic_order_hash(v13_lists)
    if v13_state.get("semantic_order_sha256") != old_semantic_order:
        raise HumanV14Error("v13 state semantic order hash mismatch")

    supersedes_v13: Dict[str, Any] = {
        "old_protocol_sha256": v13_protocol.get("content_sha256"),
        "old_lists_sha256": v13_lists.get("content_sha256"),
        "old_state_sha256": v13_state.get("content_sha256"),
        "old_semantic_order_sha256": old_semantic_order,
        "ratings_discarded": 0,
        "ratings_migrated": 0,
        "reason": "artist-identity collision correction",
        "v13_diagnostics_sha256": v13_diagnostics.get("content_sha256"),
        "v14_diagnostics_sha256": v14_diagnostics.get("content_sha256"),
        "semantic_diff": semantic_diff,
    }

    # ------------------------------------------------------------------
    # Load index
    # ------------------------------------------------------------------
    import numpy as np

    with np.load(index_path, allow_pickle=False) as index:
        ids = np.asarray(index["track_ids"], dtype=np.int64)
        titles = np.asarray(index["titles"])
        artists = np.asarray(index["artists"])
    if (
        len(ids) != EXPECTED_ROWS
        or hashlib.sha256(ids.tobytes()).hexdigest() != TRACK_IDS_SHA256
    ):
        raise HumanV14Error("v14 source index row identity mismatch")

    # v14 reuses the hash-bound v13 compact asset; mismatches fail closed.
    compact_asset_sha = compact.get("asset", {}).get("sha256", "")
    if compact_asset_sha != v14_diagnostics.get("compact_asset_sha256"):
        raise HumanV14Error("v14 diagnostics compact asset hash mismatch")

    # ------------------------------------------------------------------
    # Build blinded served lists with fresh salt
    # ------------------------------------------------------------------
    salt = secrets.token_hex(32)
    selected = v14_selected
    baseline_records = v14_diagnostics["production_baseline"]["records"]
    challenger_records = v14_variant["records"]
    if (
        len(baseline_records) != EXPECTED_SEEDS
        or len(challenger_records) != EXPECTED_SEEDS
    ):
        raise HumanV14Error("v14 requires exactly 60 paired seed records")
    challenger_by_seed = {str(item["seed_id"]): item for item in challenger_records}

    public_seeds = []
    role_records = []
    scenes = set()
    for baseline in baseline_records:
        seed_id = str(baseline["seed_id"])
        challenger = challenger_by_seed.get(seed_id)
        if challenger is None:
            raise HumanV14Error(f"missing challenger rows for {seed_id}")
        query_row = int(baseline["query_row"])
        if not 0 <= query_row < len(ids):
            raise HumanV14Error(f"query row is out of range for {seed_id}")
        if int(challenger["query_row"]) != query_row:
            raise HumanV14Error(f"paired query row mismatch for {seed_id}")
        scene = str(baseline["scene"])
        scenes.add(scene)

        result_catalog: Dict[int, Dict[str, Any]] = {}
        lists: List[Dict[str, Any]] = []
        for role, record in (
            ("production_baseline", baseline),
            ("challenger", challenger),
        ):
            rows = list(map(int, record["rows"]))
            if len(rows) != RESULTS_PER_METHOD or len(set(rows)) != RESULTS_PER_METHOD:
                raise HumanV14Error(f"{seed_id}/{role} is not a distinct top five")
            if any(row < 0 or row >= len(ids) for row in rows):
                raise HumanV14Error(f"{seed_id}/{role} contains an out-of-range row")
            list_id = _opaque("L14-", salt, seed_id, role)
            ranking = []
            for position, row in enumerate(rows, start=1):
                track_id = int(ids[row])
                result_id = _opaque("T14-", salt, seed_id, track_id)
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
                "seed_id": _opaque("S14-", salt, seed_id),
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
        raise HumanV14Error("v14 pack must preserve 60 seeds and 13 scenes")

    # ------------------------------------------------------------------
    # Build public served-lists document
    # ------------------------------------------------------------------
    public: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pack_kind": "blinded_actual_served_lists_clap_v14_identity_corrected",
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

    # ------------------------------------------------------------------
    # Build private method key
    # ------------------------------------------------------------------
    private_key: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key_kind": "private_method_role_key",
        "rankings_state": "RANKINGS_LOCKED",
        "served_lists_sha256": public["content_sha256"],
        "semantic_order_sha256": public["semantic_order_sha256"],
        "blinding_salt_sha256": hashlib.sha256(salt.encode("ascii")).hexdigest(),
        "records": role_records,
    }
    private_key["content_sha256"] = content_hash(private_key)
    method_key = private_dir / "method-key-v14.json"
    _write(method_key, private_key)

    # ------------------------------------------------------------------
    # Collector key
    # ------------------------------------------------------------------
    collector = _collector(public_dir, private_dir)

    # ------------------------------------------------------------------
    # Build protocol
    # ------------------------------------------------------------------
    protocol: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_kind": "blinded_served_list_human_listener_clap_v14_identity_corrected",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "served_lists_sha256": public["content_sha256"],
        "semantic_order_sha256": public["semantic_order_sha256"],
        "private_key_sha256": private_key["content_sha256"],
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
        "diagnostics_content_sha256": v14_diagnostics["content_sha256"],
        "compact_asset_sha256": compact_asset_sha,
        "identity_audit_sha256": identity_audit["content_sha256"],
        "evaluator_sha256": file_hash(evaluator_path),
        "supersedes_v13": supersedes_v13,
        "production_changed": False,
        "deployed": False,
        "commercial_final_opened": False,
        "ac3_claimed": False,
    }
    protocol["content_sha256"] = content_hash(protocol)

    # ------------------------------------------------------------------
    # Build state
    # ------------------------------------------------------------------
    lists_path = public_dir / "served-lists-v14.json"
    protocol_path = public_dir / "protocol-v14.json"
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
        "private_method_key_sha256": private_key["content_sha256"],
        "collector_public_key_sha256": file_hash(collector["public"]),
        "collector_allowed_signers_sha256": file_hash(collector["allowed"]),
        "evaluator_sha256": file_hash(evaluator_path),
        "diagnostics_content_sha256": v14_diagnostics["content_sha256"],
        "compact_asset_sha256": compact_asset_sha,
        "identity_audit_sha256": identity_audit["content_sha256"],
        "supersedes_v13": supersedes_v13,
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


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify_pack(
    directory: Path,
    *,
    private_key: Optional[Path] = None,
    require_trusted: bool = False,
) -> Dict[str, Any]:
    """Verify hashes, semantic order, detached state signature, and optional key.

    Parameters
    ----------
    directory:
        Public v14 pack directory containing protocol-v14.json,
        served-lists-v14.json, state.json, state.sig, allowed_signers.
    private_key:
        Optional path to the local private method-key-v14.json.
    require_trusted:
        When ``True`` and ``TRUSTED_V14_FILES`` is non-empty, verify all file
        hashes against the committed trust anchors; fails closed if the
        constants are still empty sentinels.
    """
    protocol_path = directory / "protocol-v14.json"
    lists_path = directory / "served-lists-v14.json"
    state_path = directory / "state.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lists = json.loads(lists_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))

    if require_trusted:
        if not TRUSTED_V14_FILES:
            raise HumanV14Error(
                "v14 trust anchors are not yet populated; "
                "run with require_trusted=False during pack generation"
            )
        if (
            protocol.get("content_sha256") != TRUSTED_V14_PROTOCOL
            or lists.get("content_sha256") != TRUSTED_V14_LISTS
            or state.get("content_sha256") != TRUSTED_V14_STATE
            or any(
                not (directory / name).is_file()
                or file_hash(directory / name) != digest
                for name, digest in TRUSTED_V14_FILES.items()
            )
        ):
            raise HumanV14Error("v14 pack differs from the committed trust anchors")

    # Structural integrity
    for name, document in (("protocol", protocol), ("lists", lists), ("state", state)):
        if (
            document.get("schema_version") != SCHEMA_VERSION
            or content_hash(document) != document.get("content_sha256")
            or document.get("rankings_state") != "RANKINGS_LOCKED"
        ):
            raise HumanV14Error(f"v14 {name} hash/schema/lock mismatch")

    # Cross-document binding
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
        raise HumanV14Error("v14 pack hash/order/rating binding mismatch")

    # Supersedes_v13 must be present and internally consistent
    sup = state.get("supersedes_v13", {})
    if not isinstance(sup, dict) or not sup.get("old_protocol_sha256"):
        raise HumanV14Error("v14 state missing supersedes_v13 provenance")
    if sup.get("ratings_discarded") != 0 or sup.get("ratings_migrated") != 0:
        raise HumanV14Error("v14 supersedes_v13 must record zero discarded/migrated ratings")
    if int(sup.get("semantic_diff", {}).get("changed_seed_count", 0)) == 0:
        raise HumanV14Error("v14 supersedes_v13 semantic diff must have at least one change")

    # Ed25519 signature
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise HumanV14Error("ssh-keygen is required to verify the v14 pack")
    verified = subprocess.run(
        [
            executable,
            "-Y", "verify",
            "-f", str(directory / "allowed_signers"),
            "-I", STATE_IDENTITY,
            "-n", STATE_NAMESPACE,
            "-s", str(directory / "state.sig"),
        ],
        input=state_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode:
        raise HumanV14Error("v14 state signature is invalid")

    if private_key is not None:
        key = json.loads(private_key.read_text(encoding="utf-8"))
        if (
            content_hash(key) != key.get("content_sha256")
            or key.get("content_sha256") != protocol["private_key_sha256"]
            or key.get("served_lists_sha256") != lists["content_sha256"]
            or key.get("semantic_order_sha256") != semantic
        ):
            raise HumanV14Error("v14 private method key binding mismatch")
        public_ids = {item["list_id"] for seed in lists["seeds"] for item in seed["lists"]}
        private_ids = {item["list_id"] for item in key["records"]}
        if public_ids != private_ids:
            raise HumanV14Error("v14 private method key is incomplete")

    return {"protocol": protocol, "lists": lists, "state": state}


# ---------------------------------------------------------------------------
# Preview coverage audit
# ---------------------------------------------------------------------------


def audit_v14_preview_coverage(
    lists_path: Path, endpoint: str, *, workers: int = 10
) -> Dict[str, Any]:
    report = audit_preview_coverage(lists_path, endpoint, workers=workers)
    report.pop("content_sha256", None)
    report["schema_version"] = SCHEMA_VERSION
    report["artifact_kind"] = "v14_live_preview_resolution_audit"
    report["human_ratings_used"] = 0
    report["production_changed"] = False
    report["content_sha256"] = content_hash(report)
    return report


# ---------------------------------------------------------------------------
# Loopback server
# ---------------------------------------------------------------------------


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
        raise HumanV14Error("served evaluator hash differs from the signed v14 state")
    server = ThreadingHTTPServer(
        ("127.0.0.1", int(port)),
        evaluator_handler(
            evaluator,
            directory / "protocol-v14.json",
            directory / "served-lists-v14.json",
            resolver=FreshV14PreviewResolver(),
        ),
    )
    print(f"Private blinded v14 evaluator: http://127.0.0.1:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    v14_dev = root / ".goals/human-quality-recommendations/protocol-v13-clap-development"
    v13_dev = root / ".goals/human-quality-recommendations/artifacts"
    public = root / ".goals/human-quality-recommendations/protocol-v14-clap-human-development"

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze")
    freeze.add_argument(
        "--v14-diagnostics",
        type=Path,
        default=root / "ml_data/clap_v14/diagnostics-report-v14.json",
    )
    freeze.add_argument(
        "--v13-diagnostics",
        type=Path,
        default=v13_dev / "clap-variant-diagnostics-v13.json",
    )
    freeze.add_argument(
        "--index", type=Path, default=root / "ml_data/deepvibe_index_v5.npz"
    )
    freeze.add_argument(
        "--compact-report",
        type=Path,
        default=v13_dev / "clap-compact-geometry-v13.json",
    )
    freeze.add_argument(
        "--preregistration",
        type=Path,
        default=v14_dev / "preregistration-v13-r3.json",
    )
    freeze.add_argument(
        "--evaluator", type=Path, default=root / "benchmarks/human_eval_v14.html"
    )
    freeze.add_argument(
        "--v13-pack-dir",
        type=Path,
        default=root / ".goals/human-quality-recommendations/protocol-v13-clap-human-development",
    )
    freeze.add_argument(
        "--identity-audit",
        type=Path,
        default=v13_dev / "artist-identity-collision-audit-v14.json",
    )
    freeze.add_argument("--public-dir", type=Path, default=public)
    freeze.add_argument(
        "--private-dir", type=Path, default=root / "ml_data/clap_v14/human_eval"
    )

    verify = sub.add_parser("verify")
    verify.add_argument("--directory", type=Path, default=public)
    verify.add_argument("--private-key", type=Path)

    local = sub.add_parser("serve")
    local.add_argument("--directory", type=Path, default=public)
    local.add_argument(
        "--evaluator", type=Path, default=root / "benchmarks/human_eval_v14.html"
    )
    local.add_argument("--port", type=int, default=8000)

    audit = sub.add_parser("audit")
    audit.add_argument(
        "--lists", type=Path, default=public / "served-lists-v14.json"
    )
    audit.add_argument("--endpoint", default=PRODUCTION_PREVIEW_ENDPOINT)
    audit.add_argument("--workers", type=int, default=10)
    audit.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        paths = freeze_pack(
            args.v14_diagnostics,
            args.v13_diagnostics,
            args.index,
            args.compact_report,
            args.preregistration,
            args.evaluator,
            args.v13_pack_dir,
            args.identity_audit,
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
        report = audit_v14_preview_coverage(
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
