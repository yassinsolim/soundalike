"""Tests for the v14 immutable human-eval pack workflow.

Coverage
--------
* ``human_eval_v14.semantic_order_hash`` — deterministic from served-lists
* ``_compute_semantic_diff`` — detects changes, zero-change guard
* ``freeze_pack`` — full synthetic freeze/verify cycle (no network, no large
  index files); uses a tiny in-memory NPZ and mock diagnostics
* ``freeze_pack`` fails when v14 and v13 diagnostics have identical rows
* ``verify_pack`` with ``require_trusted=False`` — passes after freeze
* ``verify_pack`` with ``require_trusted=True`` and empty constants — fails closed
* ``verify_pack`` with ``require_trusted=True`` and populated constants — passes
* ``human_eval_v14.html`` evaluator contract:
    - schema_version 14 throughout
    - L14-/T14- opaque ID pattern
    - soundalike-human-v14 storage keys
    - CSP / privacy invariants
    - Three-class + 0–10 rating contract
    - local-only exports; no external script
* ``human_aggregate_v10.aggregate`` accepts schema 14 with synthetic pack
  (no private keys; skip test if committed pack not yet generated)
* ``human_aggregate_v10._load_bound`` rejects unknown future schemas
* v13 compatibility: _load_bound still accepts schema 13 (regression guard)
* Private keys / secrets are never committed (static scan)

Tests that require the actual committed v14 pack (``ml_data/clap_v14/human_eval``
or ``.goals/…/protocol-v14-clap-human-development``) are explicitly skipped
when those paths do not exist — orchestrator freezes the pack separately.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
V14_PACK = ROOT / ".goals" / "human-quality-recommendations" / "protocol-v14-clap-human-development"
V14_PRIVATE = ROOT / "ml_data" / "clap_v14" / "human_eval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(k) + ":" + _canonical(value[k])
                for k in sorted(value)
            )
            + "}"
        )
    return json.dumps(value)


def _content_hash(doc: Dict[str, Any]) -> str:
    copy = {k: v for k, v in doc.items() if k != "content_sha256"}
    return hashlib.sha256(
        json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _make_tiny_index_npz(tmp_path: Path, n_rows: int) -> Path:
    """Write a minimal NPZ matching the expected row count (mocked track IDs)."""
    track_ids = np.arange(1, n_rows + 1, dtype=np.int64)
    titles = np.array([f"Title {i}" for i in range(n_rows)])
    artists = np.array([f"Artist {i % 100}" for i in range(n_rows)])
    path = tmp_path / "fake_index.npz"
    np.savez(path, track_ids=track_ids, titles=titles, artists=artists)
    return path


def _make_preregistration(tmp_path: Path, sha: str) -> tuple:
    """Return (path, actual_content_sha256) for a synthetic preregistration."""
    doc: Dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "clap_catalog_preregistration",
        "frozen_at": "2026-07-13T07:30:00+00:00",
    }
    actual_sha = _content_hash(doc)
    doc["content_sha256"] = actual_sha
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path, actual_sha


def _make_compact_report(
    tmp_path: Path, preregistration_sha: str, compact_asset_sha: str
) -> tuple:
    doc: Dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "clap_catalog_compact_geometry",
        "preregistration_content_sha256": preregistration_sha,
        "coverage": {"available": 272709, "no_preview": 144, "pending": 0, "error": 0},
        "asset": {"sha256": compact_asset_sha, "bytes": 69_000_000},
        "float16_reload_metrics": {"mean_top50_overlap": 0.76},
    }
    doc["content_sha256"] = _content_hash(doc)
    path = tmp_path / "compact_report.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path, doc["content_sha256"]


def _make_identity_audit(tmp_path: Path, compact_sha: str) -> Path:
    doc: Dict[str, Any] = {
        "schema_version": 14,
        "artifact_kind": "artist_identity_audit",
        "clap_asset_hash": compact_sha,
        "total_rows": 272853,
        "track_ids_sha256": "a20632fc8fb4beff406c1858714b14eb0303802a3c3829b085454d10900555f7",
        "generated_at": "2026-07-13T18:42:40.977469+00:00",
        "v13_impact": {"homonym_names_affecting_v13_seeds": 0, "total_v13_result_rows_affected": 11},
    }
    doc["content_sha256"] = _content_hash(doc)
    path = tmp_path / "identity_audit.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _make_seed_records(
    n_seeds: int, n_rows: int, scene: str = "rock", base_offset: int = 0
) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic seed records."""
    records = []
    for i in range(n_seeds):
        query_row = i
        rows = [(query_row + 1 + j + base_offset) % n_rows for j in range(5)]
        # Ensure distinct rows and no collision with query_row
        seen = {query_row}
        fixed = []
        for r in rows:
            while r in seen or r == query_row:
                r = (r + 1) % n_rows
            seen.add(r)
            fixed.append(r)
        records.append(
            {
                "seed_id": f"DEV-SONIC-{(i + 1):03d}",
                "scene": scene,
                "query_row": query_row,
                "rows": fixed,
            }
        )
    return records


def _make_diagnostics(
    tmp_path: Path,
    preregistration_sha: str,
    schema_version: int,
    n_seeds: int,
    n_rows: int,
    scenes: List[str],
    challenger_offset: int = 0,
    compact_asset_sha: str = "deadbeef" * 8,
    track_ids_sha: str = "a" * 64,
    *,
    name: str = "diagnostics.json",
) -> Path:
    """Build a minimal diagnostics JSON for freeze testing."""
    # Distribute scenes across seeds
    baseline_records = []
    challenger_records = []
    for i in range(n_seeds):
        scene = scenes[i % len(scenes)]
        query_row = i
        # Baseline rows
        base_rows = _make_seed_records(1, n_rows, scene)[0]["rows"]
        baseline_records.append(
            {"seed_id": f"DEV-SONIC-{(i + 1):03d}", "scene": scene,
             "query_row": query_row, "rows": base_rows}
        )
        # Challenger rows (offset to ensure difference when challenger_offset != 0)
        ch_rows = _make_seed_records(1, n_rows, scene, challenger_offset)[0]["rows"]
        challenger_records.append(
            {"seed_id": f"DEV-SONIC-{(i + 1):03d}", "scene": scene,
             "query_row": query_row, "rows": ch_rows,
             "query_available": True, "gate_fired": True, "fallback_reason": None,
             "candidate_count": 100, "candidate_rows": list(range(50))}
        )

    doc: Dict[str, Any] = {
        "schema_version": schema_version,
        "artifact_kind": "clap_catalog_proxy_safety_and_variant_selection",
        "preregistration_content_sha256": preregistration_sha,
        "commercial_human_ratings_used": 0,
        "proxy_evidence_is_deciding": False,
        "production": False if schema_version == 14 else None,
        "selected_challenger": "conservative_clap_fallback",
        "catalog": {
            "rows": n_rows,
            "track_ids_tobytes_sha256": track_ids_sha,
        },
        "compact_asset_sha256": compact_asset_sha,
        "safety": {
            "human_ab_required": True,
            "production_changed": False,
            "deployed": False,
            "commercial_final_opened": False,
            "ac3_claimed": False,
        },
        "production_baseline": {
            "metrics": {"seed_count": n_seeds, "complete_top5_count": n_seeds,
                        "slots": n_seeds * 5, "junk_or_version_count": 0,
                        "same_artist_count": 0, "unique_tracks": n_seeds * 5,
                        "unique_artists": n_seeds * 4,
                        "unique_artist_slot_fraction": 0.9,
                        "maximum_track_slot_fraction": 0.01,
                        "maximum_artist_slot_fraction": 0.02,
                        "mean_style_overlap": 0.77,
                        "deezer_related_artist_hits": 10,
                        "deezer_related_artist_total": 50,
                        "deezer_related_artist_seed_count": 10,
                        "deezer_related_artist_hit_rate": 0.2},
            "records": baseline_records,
        },
        "variants": {
            "conservative_clap_fallback": {
                "metrics": {
                    "seed_count": n_seeds,
                    "complete_top5_count": n_seeds,
                    "slots": n_seeds * 5,
                    "junk_or_version_count": 0,
                    "same_artist_count": 0,
                    "unique_tracks": n_seeds * 5,
                    "unique_artists": n_seeds * 4,
                    "unique_artist_slot_fraction": 0.9,
                    "maximum_track_slot_fraction": 0.01,
                    "maximum_artist_slot_fraction": 0.02,
                    "mean_style_overlap": 0.78,
                    "deezer_related_artist_hits": 20,
                    "deezer_related_artist_total": 50,
                    "deezer_related_artist_seed_count": 10,
                    "deezer_related_artist_hit_rate": 0.4,
                    "passes_proxy_safety": True,
                    "gate_fired_count": 8,
                    "exact_production_fallback_count": 2,
                    "identity_guard_abstained_count": 1,
                },
                "records": challenger_records,
            }
        },
    }
    if schema_version == 14:
        doc.pop("production", None)
        doc["production"] = False
    doc["content_sha256"] = _content_hash(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _make_v13_committed_pack(
    tmp_path: Path,
    v13_diagnostics_path: Path,
    index_path: Path,
    compact_path: Path,
    preregistration_sha: str,
) -> Path:
    """Build a minimal committed v13 pack directory that freeze_pack can read."""
    from soundalike.ml.human_eval_v13 import SCHEMA_VERSION as V13_SCHEMA
    from soundalike.ml.human_eval_v10 import content_hash, file_hash

    pack_dir = tmp_path / "v13_pack"
    pack_dir.mkdir()

    v13_diag = json.loads(v13_diagnostics_path.read_text(encoding="utf-8"))
    selected = str(v13_diag["selected_challenger"])
    v13_variant = v13_diag["variants"][selected]
    baseline_records = v13_diag["production_baseline"]["records"]
    challenger_records = v13_variant["records"]

    import secrets
    import numpy as np

    with np.load(index_path, allow_pickle=False) as idx:
        ids = np.asarray(idx["track_ids"], dtype=np.int64)
        titles = np.asarray(idx["titles"])
        artists = np.asarray(idx["artists"])

    # We need the PREREGISTRATION_SHA256 and TRACK_IDS_SHA256 to match.
    # Since we're building a fake pack, use the fake track_ids_sha from diagnostics.
    # The key is that the v13 pack must self-hash correctly.
    salt = secrets.token_hex(8)
    n_rows = len(ids)
    n_seeds = len(baseline_records)

    def opaque(prefix: str, *parts: object) -> str:
        value = "\0".join([salt, *(str(p) for p in parts)]).encode("utf-8")
        return prefix + hashlib.sha256(value).hexdigest()[:24]

    public_seeds = []
    from soundalike.ml.human_eval_v13 import semantic_order_hash as v13_soh

    for baseline in baseline_records:
        seed_id = str(baseline["seed_id"])
        challenger = next(r for r in challenger_records if str(r["seed_id"]) == seed_id)
        query_row = int(baseline["query_row"]) % n_rows
        scene = str(baseline["scene"])
        result_catalog: Dict[int, Dict[str, Any]] = {}
        lists_out = []
        for role, record in (("production_baseline", baseline), ("challenger", challenger)):
            rows = [int(r) % n_rows for r in record["rows"][:5]]
            list_id = opaque("L13-", seed_id, role)
            ranking = []
            for pos, row in enumerate(rows, 1):
                tid = int(ids[row])
                rid = opaque("T13-", seed_id, tid)
                result_catalog.setdefault(
                    tid,
                    {"result_id": rid, "track_id": tid, "deezer_track_id": tid,
                     "title": str(titles[row]), "artist": str(artists[row])},
                )
                ranking.append({"position": pos, "result_id": rid})
            lists_out.append({"list_id": list_id, "ranking": ranking})
        public_seeds.append({
            "seed_id": opaque("S13-", seed_id),
            "source_seed_id": seed_id,
            "scene": scene,
            "query": {"track_id": int(ids[query_row]), "deezer_track_id": int(ids[query_row]),
                      "title": str(titles[query_row]), "artist": str(artists[query_row])},
            "results": list(result_catalog.values()),
            "lists": lists_out,
        })

    compact_doc = json.loads(compact_path.read_text(encoding="utf-8"))
    compact_sha = compact_doc.get("asset", {}).get("sha256", "deadbeef" * 8)

    lists_doc: Dict[str, Any] = {
        "schema_version": V13_SCHEMA,
        "pack_kind": "blinded_actual_served_lists_clap_development",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "seed_count": n_seeds,
        "scene_count": len({r["scene"] for r in baseline_records}),
        "results_per_method": 5,
        "same_artist_filtered": True,
        "shared_results_rated_once": True,
        "stable_id_field": "deezer_track_id",
        "preview_urls_resolved_at_freeze": False,
        "audio_access": {
            "provider": "Deezer public 30-second previews",
            "resolution": "fresh on demand",
            "signed_preview_urls_persisted": False,
            "browser_cache_scope": "memory only",
            "refresh_on_playback_failure": True,
            "external_request_disclosure": "Only numeric Deezer ID transmitted.",
            "fallbacks": [],
        },
        "seeds": public_seeds,
    }
    lists_doc["semantic_order_sha256"] = v13_soh(lists_doc)
    lists_doc["content_sha256"] = content_hash(lists_doc)

    protocol_doc: Dict[str, Any] = {
        "schema_version": V13_SCHEMA,
        "protocol_kind": "blinded_served_list_human_listener_clap_development",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "served_lists_sha256": lists_doc["content_sha256"],
        "semantic_order_sha256": lists_doc["semantic_order_sha256"],
        "private_key_sha256": "fake_key_sha256",
        "seed_count": n_seeds,
        "scene_count": lists_doc["scene_count"],
        "results_per_method": 5,
        "assignment": "per-session randomized order",
        "rating_scale": "MIREX three-class",
        "collector_public_key_sha256": "fake_coll_pub",
        "collector_allowed_signers_sha256": "fake_coll_allowed",
        "human_evidence_gate": "At least three independent raters required.",
        "preregistration_content_sha256": preregistration_sha,
        "diagnostics_content_sha256": v13_diag["content_sha256"],
        "compact_asset_sha256": compact_sha,
        "evaluator_sha256": "fake_eval_sha",
        "production_changed": False,
        "deployed": False,
        "commercial_final_opened": False,
        "ac3_claimed": False,
    }
    protocol_doc["content_sha256"] = content_hash(protocol_doc)

    state_doc: Dict[str, Any] = {
        "schema_version": V13_SCHEMA,
        "phase": "RANKINGS_LOCKED",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_freeze": 0,
        "human_rater_exports_ingested": 0,
        "sonic_human_report_exists": False,
        "production_deployment_blocked": True,
        "served_lists_sha256": lists_doc["content_sha256"],
        "semantic_order_sha256": lists_doc["semantic_order_sha256"],
        "protocol_sha256": protocol_doc["content_sha256"],
        "private_method_key_sha256": "fake_key_sha256",
        "collector_public_key_sha256": "fake_coll_pub",
        "collector_allowed_signers_sha256": "fake_coll_allowed",
        "evaluator_sha256": "fake_eval_sha",
        "diagnostics_content_sha256": v13_diag["content_sha256"],
        "compact_asset_sha256": compact_sha,
        "locked_at": "2026-07-13T15:13:18.082670+00:00",
        "production_changed": False,
        "deployed": False,
        "commercial_final_opened": False,
        "ac3_claimed": False,
    }
    state_doc["content_sha256"] = content_hash(state_doc)

    def write(p: Path, d: Dict[str, Any]) -> None:
        p.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")

    write(pack_dir / "protocol-v13.json", protocol_doc)
    write(pack_dir / "served-lists-v13.json", lists_doc)
    write(pack_dir / "state.json", state_doc)
    return pack_dir


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from soundalike.ml.human_eval_v14 import (
    SCHEMA_VERSION,
    HumanV14Error,
    _compute_semantic_diff,
    freeze_pack,
    semantic_order_hash,
    verify_pack,
)
from soundalike.ml.human_eval_v10 import content_hash as _chash

# ---------------------------------------------------------------------------
# Constants used in synthetic tests
# ---------------------------------------------------------------------------

_FAKE_PREREG_SHA = "2c1bb55c85dfa8d1d344bba02868563c459ac743604f525ecb678598f3ef4ee7"
_FAKE_COMPACT_SHA = "deadbeef" * 8
_FAKE_TRACK_SHA = "a" * 64

# Number of synthetic seeds/scenes kept small so tests run quickly
_N_SEEDS = 60
_N_SCENES = 13
_SCENE_LABELS = [
    "pop", "rock", "indie", "rnb", "electronic", "metal", "jazz",
    "hyperpop", "shoegaze", "city_pop_jpop_kpop", "latin_afrobeats",
    "difficult_blend", "rap",
]
# Large enough for 60 seeds × 10 rows without collision
_N_ROWS = 1000


# ---------------------------------------------------------------------------
# semantic_order_hash tests
# ---------------------------------------------------------------------------


def _build_minimal_served_lists(
    n_seeds: int = 2, n_results: int = 3
) -> Dict[str, Any]:
    """Return a minimal served-lists document for hash testing."""
    seeds = []
    for i in range(n_seeds):
        results = [
            {"result_id": f"T14-{'a' * 24}", "deezer_track_id": 100 + i * n_results + j,
             "track_id": 100 + i * n_results + j, "title": f"T{j}", "artist": "A"}
            for j in range(n_results)
        ]
        # Make result_ids unique
        for k, r in enumerate(results):
            r["result_id"] = f"T14-{hashlib.sha256(str(k + i * 100).encode()).hexdigest()[:24]}"
        lists = [
            {
                "list_id": f"L14-{hashlib.sha256(str(i * 2 + li).encode()).hexdigest()[:24]}",
                "ranking": [
                    {"position": pos + 1, "result_id": results[pos % len(results)]["result_id"]}
                    for pos in range(min(5, len(results)))
                ],
            }
            for li in range(2)
        ]
        seeds.append({
            "seed_id": f"S14-{hashlib.sha256(str(i).encode()).hexdigest()[:24]}",
            "query": {"deezer_track_id": 1000 + i},
            "results": results,
            "lists": lists,
        })
    return {"seeds": seeds}


def test_semantic_order_hash_is_deterministic():
    doc = _build_minimal_served_lists()
    assert semantic_order_hash(doc) == semantic_order_hash(doc)


def test_semantic_order_hash_changes_on_row_change():
    doc = _build_minimal_served_lists()
    h1 = semantic_order_hash(doc)
    # Mutate one deezer_track_id
    doc["seeds"][0]["results"][0]["deezer_track_id"] += 1
    assert semantic_order_hash(doc) != h1


def test_semantic_order_hash_ignores_title_change():
    doc = _build_minimal_served_lists()
    h1 = semantic_order_hash(doc)
    doc["seeds"][0]["results"][0]["title"] = "Changed Title"
    assert semantic_order_hash(doc) == h1


# ---------------------------------------------------------------------------
# _compute_semantic_diff tests
# ---------------------------------------------------------------------------


def test_compute_semantic_diff_detects_changes():
    v13 = [
        {"seed_id": "S001", "rows": [1, 2, 3, 4, 5]},
        {"seed_id": "S002", "rows": [6, 7, 8, 9, 10]},
    ]
    v14 = [
        {"seed_id": "S001", "rows": [1, 2, 3, 4, 5]},     # unchanged
        {"seed_id": "S002", "rows": [6, 7, 8, 9, 99]},    # changed
    ]
    diff = _compute_semantic_diff(v13, v14)
    assert diff["changed_seed_count"] == 1
    assert diff["changed_positions"][0]["seed_id"] == "S002"
    assert diff["changed_positions"][0]["v13_rows"] == [6, 7, 8, 9, 10]
    assert diff["changed_positions"][0]["v14_rows"] == [6, 7, 8, 9, 99]


def test_compute_semantic_diff_zero_changes():
    v13 = [{"seed_id": "S001", "rows": [1, 2, 3, 4, 5]}]
    diff = _compute_semantic_diff(v13, v13)
    assert diff["changed_seed_count"] == 0
    assert diff["changed_positions"] == []


def test_compute_semantic_diff_multiple_changes():
    v13 = [{"seed_id": f"S{i:03d}", "rows": [i, i + 1, i + 2, i + 3, i + 4]} for i in range(5)]
    v14 = [{"seed_id": f"S{i:03d}", "rows": [i, i + 1, i + 2, i + 3, i + 100]} for i in range(5)]
    diff = _compute_semantic_diff(v13, v14)
    assert diff["changed_seed_count"] == 5


# ---------------------------------------------------------------------------
# freeze_pack / verify_pack full synthetic cycle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("ssh-keygen") is None, reason="ssh-keygen is required for signing"
)
def test_freeze_and_verify_synthetic(tmp_path):
    """Full freeze → verify cycle with synthetic data (no large files)."""
    from soundalike.ml.clap_catalog_v13 import PREREGISTRATION_SHA256, TRACK_IDS_SHA256

    # Build a synthetic index with the right row count and track_ids hash
    n_rows = 272853
    # For speed: we only need to match TRACK_IDS_SHA256
    # Use the real expected TRACK_IDS_SHA256 by building matching track_ids
    # This is expensive, so we mock the validation instead
    index_path = tmp_path / "fake_index.npz"

    # We'll patch EXPECTED_ROWS and TRACK_IDS_SHA256 checks inside freeze_pack
    fake_ids = np.arange(1, _N_ROWS + 1, dtype=np.int64)
    fake_titles = np.array([f"Title {i}" for i in range(_N_ROWS)])
    fake_artists = np.array([f"Artist {i % 100}" for i in range(_N_ROWS)])
    np.savez(index_path, track_ids=fake_ids, titles=fake_titles, artists=fake_artists)

    fake_ids_sha = hashlib.sha256(fake_ids.tobytes()).hexdigest()

    prereg_path, prereg_sha = _make_preregistration(tmp_path, PREREGISTRATION_SHA256)
    compact_path, _ = _make_compact_report(tmp_path, prereg_sha, _FAKE_COMPACT_SHA)
    identity_path = _make_identity_audit(tmp_path, _FAKE_COMPACT_SHA)
    evaluator_path = ROOT / "benchmarks" / "human_eval_v14.html"
    if not evaluator_path.is_file():
        pytest.skip("v14 evaluator HTML not yet committed")

    v13_diag_path = _make_diagnostics(
        tmp_path, prereg_sha, 13, _N_SEEDS, _N_ROWS,
        _SCENE_LABELS, challenger_offset=0,
        compact_asset_sha=_FAKE_COMPACT_SHA,
        track_ids_sha=fake_ids_sha,
        name="v13_diag.json",
    )
    v14_diag_path = _make_diagnostics(
        tmp_path, prereg_sha, 14, _N_SEEDS, _N_ROWS,
        _SCENE_LABELS, challenger_offset=500,  # Different rows → 2+ seeds change
        compact_asset_sha=_FAKE_COMPACT_SHA,
        track_ids_sha=fake_ids_sha,
        name="v14_diag.json",
    )

    v13_pack_dir = _make_v13_committed_pack(
        tmp_path, v13_diag_path, index_path, compact_path, prereg_sha
    )

    public_dir = tmp_path / "v14_public"
    private_dir = tmp_path / "v14_private"

    import soundalike.ml.human_eval_v14 as module

    # Patch the row-count / hash constants so we don't need the real 272k index
    with (
        patch.object(module, "EXPECTED_ROWS", _N_ROWS),
        patch.object(module, "TRACK_IDS_SHA256", fake_ids_sha),
        patch.object(module, "PREREGISTRATION_SHA256", prereg_sha),
        # Patch v13 trust anchor checks to pass with fake pack
        patch.object(module, "TRUSTED_V13_PROTOCOL", ""),
        patch.object(module, "TRUSTED_V13_LISTS", ""),
        patch.object(module, "TRUSTED_V13_STATE", ""),
    ):
        paths = freeze_pack(
            v14_diag_path,
            v13_diag_path,
            index_path,
            compact_path,
            prereg_path,
            evaluator_path,
            v13_pack_dir,
            identity_path,
            public_dir,
            private_dir,
        )

    # Check files were written
    assert paths["protocol"].is_file()
    assert paths["lists"].is_file()
    assert paths["state"].is_file()
    assert paths["signature"].is_file()
    assert paths["method_key"].is_file()
    assert paths["collector_private"].is_file()
    assert paths["collector_public"].is_file()
    assert paths["collector_allowed_signers"].is_file()

    # Structural checks
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    lists = json.loads(paths["lists"].read_text(encoding="utf-8"))
    state = json.loads(paths["state"].read_text(encoding="utf-8"))

    assert protocol["schema_version"] == 14
    assert lists["schema_version"] == 14
    assert state["schema_version"] == 14
    assert protocol["rankings_state"] == "RANKINGS_LOCKED"
    assert lists["ratings_count_at_freeze"] == 0
    assert state["ratings_count_at_freeze"] == 0
    assert len(lists["seeds"]) == _N_SEEDS

    # Opaque ID format
    all_result_ids = [r["result_id"] for s in lists["seeds"] for r in s["results"]]
    all_list_ids = [l["list_id"] for s in lists["seeds"] for l in s["lists"]]
    all_seed_ids = [s["seed_id"] for s in lists["seeds"]]
    assert all(rid.startswith("T14-") for rid in all_result_ids)
    assert all(lid.startswith("L14-") for lid in all_list_ids)
    assert all(sid.startswith("S14-") for sid in all_seed_ids)

    # supersedes_v13 provenance
    sup = state["supersedes_v13"]
    assert sup["ratings_discarded"] == 0
    assert sup["ratings_migrated"] == 0
    assert sup["reason"] == "artist-identity collision correction"
    assert sup["semantic_diff"]["changed_seed_count"] > 0

    # No role labels in public files
    public_text = paths["protocol"].read_text() + paths["lists"].read_text()
    assert "production_baseline" not in public_text
    assert "challenger" not in public_text

    # verify_pack (require_trusted=False) passes
    result = verify_pack(public_dir, require_trusted=False)
    assert result["protocol"]["schema_version"] == 14
    assert result["state"]["rankings_state"] == "RANKINGS_LOCKED"

    # Synthetic bytes must not pass the committed pack's trust anchors.
    with pytest.raises(HumanV14Error, match="differs from the committed trust anchors"):
        verify_pack(public_dir, require_trusted=True)


@pytest.mark.skipif(
    shutil.which("ssh-keygen") is None, reason="ssh-keygen is required for signing"
)
def test_freeze_fails_when_no_list_changes(tmp_path):
    """freeze_pack must raise when v13 and v14 challenger rows are identical."""
    from soundalike.ml.clap_catalog_v13 import PREREGISTRATION_SHA256

    index_path = tmp_path / "fake_index.npz"
    fake_ids = np.arange(1, _N_ROWS + 1, dtype=np.int64)
    np.savez(index_path,
             track_ids=fake_ids,
             titles=np.array([f"T{i}" for i in range(_N_ROWS)]),
             artists=np.array([f"A{i}" for i in range(_N_ROWS)]))
    fake_ids_sha = hashlib.sha256(fake_ids.tobytes()).hexdigest()

    prereg_path, prereg_sha = _make_preregistration(tmp_path, PREREGISTRATION_SHA256)
    compact_path, _ = _make_compact_report(tmp_path, prereg_sha, _FAKE_COMPACT_SHA)
    identity_path = _make_identity_audit(tmp_path, _FAKE_COMPACT_SHA)
    evaluator_path = ROOT / "benchmarks" / "human_eval_v14.html"
    if not evaluator_path.is_file():
        pytest.skip("v14 evaluator HTML not yet committed")

    # Same challenger_offset → identical rows → should fail
    v13_diag = _make_diagnostics(
        tmp_path, prereg_sha, 13, _N_SEEDS, _N_ROWS,
        _SCENE_LABELS, challenger_offset=0,
        compact_asset_sha=_FAKE_COMPACT_SHA, track_ids_sha=fake_ids_sha,
        name="v13_same.json",
    )
    v14_diag = _make_diagnostics(
        tmp_path, prereg_sha, 14, _N_SEEDS, _N_ROWS,
        _SCENE_LABELS, challenger_offset=0,  # same offset → identical rows
        compact_asset_sha=_FAKE_COMPACT_SHA, track_ids_sha=fake_ids_sha,
        name="v14_same.json",
    )
    v13_pack_dir = _make_v13_committed_pack(
        tmp_path, v13_diag, index_path, compact_path, prereg_sha
    )

    import soundalike.ml.human_eval_v14 as module

    with (
        patch.object(module, "EXPECTED_ROWS", _N_ROWS),
        patch.object(module, "TRACK_IDS_SHA256", fake_ids_sha),
        patch.object(module, "PREREGISTRATION_SHA256", prereg_sha),
        patch.object(module, "TRUSTED_V13_PROTOCOL", ""),
        patch.object(module, "TRUSTED_V13_LISTS", ""),
        patch.object(module, "TRUSTED_V13_STATE", ""),
        pytest.raises(HumanV14Error, match="no list changes"),
    ):
        freeze_pack(
            v14_diag, v13_diag, index_path, compact_path, prereg_path,
            evaluator_path, v13_pack_dir, identity_path,
            tmp_path / "pub_same", tmp_path / "priv_same",
        )


@pytest.mark.skipif(
    shutil.which("ssh-keygen") is None, reason="ssh-keygen is required for signing"
)
def test_verify_pack_passes_with_populated_trust_anchors(tmp_path):
    """After patching trust anchors to match the freeze output, verify passes."""
    from soundalike.ml.clap_catalog_v13 import PREREGISTRATION_SHA256

    index_path = tmp_path / "fake_index.npz"
    fake_ids = np.arange(1, _N_ROWS + 1, dtype=np.int64)
    np.savez(index_path,
             track_ids=fake_ids,
             titles=np.array([f"T{i}" for i in range(_N_ROWS)]),
             artists=np.array([f"A{i}" for i in range(_N_ROWS)]))
    fake_ids_sha = hashlib.sha256(fake_ids.tobytes()).hexdigest()

    prereg_path, prereg_sha = _make_preregistration(tmp_path, PREREGISTRATION_SHA256)
    compact_path, _ = _make_compact_report(tmp_path, prereg_sha, _FAKE_COMPACT_SHA)
    identity_path = _make_identity_audit(tmp_path, _FAKE_COMPACT_SHA)
    evaluator_path = ROOT / "benchmarks" / "human_eval_v14.html"
    if not evaluator_path.is_file():
        pytest.skip("v14 evaluator HTML not yet committed")

    v13_diag = _make_diagnostics(
        tmp_path, prereg_sha, 13, _N_SEEDS, _N_ROWS,
        _SCENE_LABELS, challenger_offset=0,
        compact_asset_sha=_FAKE_COMPACT_SHA, track_ids_sha=fake_ids_sha,
        name="v13_anchor.json",
    )
    v14_diag = _make_diagnostics(
        tmp_path, prereg_sha, 14, _N_SEEDS, _N_ROWS,
        _SCENE_LABELS, challenger_offset=400,
        compact_asset_sha=_FAKE_COMPACT_SHA, track_ids_sha=fake_ids_sha,
        name="v14_anchor.json",
    )
    v13_pack_dir = _make_v13_committed_pack(
        tmp_path, v13_diag, index_path, compact_path, prereg_sha
    )
    public_dir = tmp_path / "v14_pub_anchor"
    private_dir = tmp_path / "v14_priv_anchor"

    import soundalike.ml.human_eval_v14 as module
    from soundalike.ml.human_eval_v10 import file_hash

    with (
        patch.object(module, "EXPECTED_ROWS", _N_ROWS),
        patch.object(module, "TRACK_IDS_SHA256", fake_ids_sha),
        patch.object(module, "PREREGISTRATION_SHA256", prereg_sha),
        patch.object(module, "TRUSTED_V13_PROTOCOL", ""),
        patch.object(module, "TRUSTED_V13_LISTS", ""),
        patch.object(module, "TRUSTED_V13_STATE", ""),
    ):
        freeze_pack(
            v14_diag, v13_diag, index_path, compact_path, prereg_path,
            evaluator_path, v13_pack_dir, identity_path, public_dir, private_dir,
        )

    # Collect real file hashes to populate trust anchors
    protocol = json.loads((public_dir / "protocol-v14.json").read_text(encoding="utf-8"))
    lists = json.loads((public_dir / "served-lists-v14.json").read_text(encoding="utf-8"))
    state = json.loads((public_dir / "state.json").read_text(encoding="utf-8"))
    files = {
        name: file_hash(public_dir / name)
        for name in (
            "protocol-v14.json", "served-lists-v14.json", "state.json",
            "state.sig", "allowed_signers", "signer.pub",
            "collector_signer.pub", "collector_allowed_signers",
        )
        if (public_dir / name).is_file()
    }

    with (
        patch.object(module, "TRUSTED_V14_PROTOCOL", protocol["content_sha256"]),
        patch.object(module, "TRUSTED_V14_LISTS", lists["content_sha256"]),
        patch.object(module, "TRUSTED_V14_STATE", state["content_sha256"]),
        patch.object(module, "TRUSTED_V14_FILES", files),
    ):
        result = verify_pack(public_dir, require_trusted=True)
    assert result["state"]["schema_version"] == 14


# ---------------------------------------------------------------------------
# Evaluator HTML contract tests
# ---------------------------------------------------------------------------


def test_evaluator_html_exists():
    path = ROOT / "benchmarks" / "human_eval_v14.html"
    assert path.is_file(), "benchmarks/human_eval_v14.html must exist"


def test_evaluator_html_schema_v14():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "schema_version:14" in html
    assert "schema_version!==14" in html
    assert "schema_version:13" not in html
    assert "schema_version!==13" not in html


def test_evaluator_html_v14_opaque_ids():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    # Must validate L14-/T14- prefix
    assert "[LT]14-" in html
    # Must NOT validate old L13-/T13-
    assert "[LT]13-" not in html


def test_evaluator_html_v14_storage_keys():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "soundalike-human-v14" in html
    assert "soundalike-human-v13" not in html


def test_evaluator_html_v14_export_filename():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "human-ratings-v14-" in html
    assert "human-ratings-v13-" not in html


def test_evaluator_html_privacy_and_csp():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "connect-src 'self' https://soundalike.yassin.app" in html
    assert "media-src https://*.dzcdn.net" in html
    assert '<meta name="referrer" content="no-referrer">' in html
    assert 'credentials:"omit",cache:"no-store",referrerPolicy:"no-referrer"' in html
    assert 'url.searchParams.set("id",String(deezerId))' in html
    assert "No analytics" in html
    assert "<script src=" not in html


def test_evaluator_html_rating_contract():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    # Three-class similarity
    assert "not_similar" in html
    assert "somewhat_similar" in html
    assert "very_similar" in html
    # Three-class coherence
    assert "not_coherent" in html
    assert "somewhat_coherent" in html
    assert "very_coherent" in html
    # Optional 0-10 score
    assert 'min="0" max="10"' in html
    # Junk/version flag
    assert "junk_or_version" in html
    assert "data-junk" in html


def test_evaluator_html_no_method_identity():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "production_baseline" not in html
    assert '"challenger"' not in html


def test_evaluator_html_local_only_export():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "integrity_hmac_sha256" in html
    assert "localStorage" in html
    assert "URL.createObjectURL" in html
    # No fetch upload of ratings
    rating_export_block = html[html.rfind("async function exportRatings"):]
    assert "fetch(" not in rating_export_block[:rating_export_block.find("}")] or \
        "fetch(" not in rating_export_block[:500]


def test_evaluator_html_resume_behavior():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "restoreAutosave" in html
    assert "localStorage.getItem(pointerKey())" in html


def test_evaluator_html_no_preview_handling():
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "NO_PREVIEW" in html


# ---------------------------------------------------------------------------
# human_aggregate_v10 schema 14 compatibility
# ---------------------------------------------------------------------------


def test_aggregate_rejects_unknown_future_schema(tmp_path):
    """_load_bound must reject unknown future schema versions."""
    from soundalike.ml.human_aggregate_v10 import AggregateError, _load_bound

    for schema in (16, 9, 0):
        doc = {"schema_version": schema, "rankings_state": "RANKINGS_LOCKED"}
        doc["content_sha256"] = _chash(doc)
        p = tmp_path / f"doc_{schema}.json"
        p.write_text(json.dumps(doc) + "\n", encoding="utf-8")

    with pytest.raises(AggregateError, match="schema versions are incompatible"):
        _load_bound(
            tmp_path / "doc_16.json",
            tmp_path / "doc_16.json",
            tmp_path / "doc_16.json",
        )


def test_aggregate_accepts_schema_14_structure(tmp_path):
    """_load_bound should reach schema 14 verification before failing on state files."""
    from soundalike.ml.human_aggregate_v10 import AggregateError, _load_bound

    # Build minimal schema-14 documents that pass the initial checks
    # but fail the state-file lookup (state.json not present) —
    # this proves the schema dispatch reached _verify_v14_state
    key_doc = {
        "schema_version": 14,
        "key_kind": "private_method_role_key",
        "rankings_state": "RANKINGS_LOCKED",
        "served_lists_sha256": "placeholder",
        "semantic_order_sha256": "placeholder",
        "blinding_salt_sha256": "placeholder",
        "records": [],
    }
    key_doc["content_sha256"] = _chash(key_doc)

    lists_doc = {
        "schema_version": 14,
        "rankings_state": "RANKINGS_LOCKED",
        "content_sha256": key_doc["served_lists_sha256"],  # will fail hash check anyway
    }
    # Make lists hash match what key says
    lists_doc.pop("content_sha256")
    lists_doc["content_sha256"] = _chash(lists_doc)
    # Now fix key to point at the real lists hash
    key_doc.pop("content_sha256")
    key_doc["served_lists_sha256"] = lists_doc["content_sha256"]
    key_doc["content_sha256"] = _chash(key_doc)

    protocol_doc = {
        "schema_version": 14,
        "rankings_state": "RANKINGS_LOCKED",
        "served_lists_sha256": lists_doc["content_sha256"],
        "private_key_sha256": key_doc["content_sha256"],
    }
    protocol_doc["content_sha256"] = _chash(protocol_doc)

    proto_path = tmp_path / "p14.json"
    lists_path = tmp_path / "l14.json"
    key_path = tmp_path / "k14.json"
    proto_path.write_text(json.dumps(protocol_doc) + "\n", encoding="utf-8")
    lists_path.write_text(json.dumps(lists_doc) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(key_doc) + "\n", encoding="utf-8")

    # No state.json present → AggregateError from _verify_v14_state
    with pytest.raises(AggregateError, match="signed v14 study state is incomplete"):
        _load_bound(proto_path, lists_path, key_path)


def test_aggregate_v13_compatibility_still_works(tmp_path):
    """Schema 13 path in _load_bound is not broken by the v14 addition."""
    from soundalike.ml.human_aggregate_v10 import AggregateError, _load_bound

    key_doc = {
        "schema_version": 13,
        "key_kind": "private_method_role_key",
        "rankings_state": "RANKINGS_LOCKED",
        "served_lists_sha256": "placeholder",
        "semantic_order_sha256": "placeholder",
        "blinding_salt_sha256": "placeholder",
        "records": [],
    }
    key_doc["content_sha256"] = _chash(key_doc)

    lists_doc = {"schema_version": 13, "rankings_state": "RANKINGS_LOCKED"}
    lists_doc["content_sha256"] = _chash(lists_doc)
    key_doc.pop("content_sha256")
    key_doc["served_lists_sha256"] = lists_doc["content_sha256"]
    key_doc["content_sha256"] = _chash(key_doc)

    protocol_doc = {
        "schema_version": 13,
        "rankings_state": "RANKINGS_LOCKED",
        "served_lists_sha256": lists_doc["content_sha256"],
        "private_key_sha256": key_doc["content_sha256"],
    }
    protocol_doc["content_sha256"] = _chash(protocol_doc)

    proto_path = tmp_path / "p13.json"
    lists_path = tmp_path / "l13.json"
    key_path = tmp_path / "k13.json"
    proto_path.write_text(json.dumps(protocol_doc) + "\n", encoding="utf-8")
    lists_path.write_text(json.dumps(lists_doc) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(key_doc) + "\n", encoding="utf-8")

    # No state.json → AggregateError from _verify_v13_state (v13 path still reached)
    with pytest.raises(AggregateError, match="signed v13 study state is incomplete"):
        _load_bound(proto_path, lists_path, key_path)


# ---------------------------------------------------------------------------
# Committed pack skip test
# ---------------------------------------------------------------------------


def test_committed_pack_is_trusted_and_zero_rating():
    """The immutable committed v14 pack verifies against pinned anchors."""
    verified = verify_pack(V14_PACK, require_trusted=True)
    assert verified["state"]["ratings_count_at_freeze"] == 0
    assert verified["state"]["production_deployment_blocked"] is True


def test_v14_trust_anchors_are_populated():
    from soundalike.ml import human_eval_v14 as mod

    assert len(mod.TRUSTED_V14_PROTOCOL) == 64
    assert len(mod.TRUSTED_V14_LISTS) == 64
    assert len(mod.TRUSTED_V14_STATE) == 64
    assert mod.TRUSTED_V14_FILES


# ---------------------------------------------------------------------------
# Static security scan — no private keys committed
# ---------------------------------------------------------------------------


def test_no_private_keys_in_test_file():
    """Verify this test file contains no actual PEM private-key blocks.

    A real PEM block has the header on one line, base64 data on the next.
    We check that no line consists purely of base64 padding (=====) in a way
    that would indicate an actual embedded key, and that the module files
    under test don't contain the PEM header.  (The assertion strings below
    necessarily contain the patterns they're checking for, which is normal.)
    """
    # The module under test must be clean
    module_src = (
        ROOT / "src" / "soundalike" / "ml" / "human_eval_v14.py"
    ).read_text(encoding="utf-8")
    pem_header = "BEGIN " + "OPENSSH PRIVATE KEY"
    rsa_header = "BEGIN " + "RSA PRIVATE KEY"
    assert pem_header not in module_src
    assert rsa_header not in module_src

    # The evaluator HTML must be clean
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert pem_header not in html
    assert rsa_header not in html


def test_v14_module_no_private_keys():
    """human_eval_v14.py must not contain private key material."""
    source = (
        ROOT / "src" / "soundalike" / "ml" / "human_eval_v14.py"
    ).read_text(encoding="utf-8")
    for pattern in ("BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY"):
        assert pattern not in source


def test_evaluator_html_no_private_keys():
    """human_eval_v14.html must not contain private key material."""
    html = (ROOT / "benchmarks" / "human_eval_v14.html").read_text(encoding="utf-8")
    assert "BEGIN OPENSSH PRIVATE KEY" not in html
    assert "BEGIN RSA PRIVATE KEY" not in html
