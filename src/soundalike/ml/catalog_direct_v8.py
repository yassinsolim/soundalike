"""Locked, target-blind direct-list inspection for the catalogue v8 policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from soundalike.audio.previews import DeezerClient

from .catalog_graph import CatalogArtistGraph
from .catalog_policy import CatalogPolicy, CatalogPolicyRanker
from .catalog_style import CatalogStyleIndex
from .quality_filter import TitleQualityFilter
from .real_benchmark import PairResolver, ProductionRanker


class DirectListError(ValueError):
    """Raised when a locked direct-list input or judgment is invalid."""


# This is intentionally separate from every earlier review set.  Spellings are
# preserved here; catalogue spellings are accepted only when PairResolver finds
# the locked identity.
LOCKED_SEEDS: Tuple[Mapping[str, str], ...] = (
    {"artist": "Pixies", "title": "Where Is My Mind?", "scene": "alternative_rock", "failure_class": "pixies_to_trip_hop"},
    {"artist": "Anri", "title": "Last Summer Whisper", "scene": "city_pop", "failure_class": "city_pop"},
    {"artist": "Miki Matsubara", "title": "Mayonaka no Door / Stay With Me", "scene": "city_pop", "failure_class": "city_pop"},
    {"artist": "Kali Uchis", "title": "telepatía", "scene": "latin_alt_pop", "failure_class": "latin"},
    {"artist": "Bad Bunny", "title": "Tití Me Preguntó", "scene": "latin_reggaeton", "failure_class": "latin"},
    {"artist": "100 gecs", "title": "money machine", "scene": "hyperpop", "failure_class": "hyperpop_digicore"},
    {"artist": "brakence", "title": "rosier/punk2", "scene": "digicore", "failure_class": "hyperpop_digicore"},
    {"artist": "glaive", "title": "astrid", "scene": "digicore", "failure_class": "hyperpop_digicore"},
    {"artist": "Daft Punk", "title": "Digital Love", "scene": "electronic", "failure_class": "daft_punk"},
    {"artist": "Gorillaz", "title": "Clint Eastwood", "scene": "art_pop", "failure_class": "gorillaz"},
    {"artist": "Massive Attack", "title": "Teardrop", "scene": "trip_hop", "failure_class": "pixies_to_trip_hop"},
    {"artist": "my bloody valentine", "title": "Sometimes", "scene": "shoegaze", "failure_class": "shoegaze"},
    {"artist": "Deftones", "title": "Be Quiet and Drive (Far Away)", "scene": "alternative_metal", "failure_class": "metal"},
    {"artist": "A Tribe Called Quest", "title": "Electric Relaxation", "scene": "jazz_rap", "failure_class": "rap"},
    {"artist": "Frank Ocean", "title": "Nights", "scene": "alternative_rnb", "failure_class": "rnb"},
    {"artist": "Metallica", "title": "Orion (Remastered)", "scene": "thrash_metal", "failure_class": "metal"},
    {"artist": "Miles Davis", "title": "So What (Album Version)", "scene": "modal_jazz", "failure_class": "jazz"},
    {"artist": "Burna Boy", "title": "Ye", "scene": "afrobeats", "failure_class": "afrobeats"},
    {"artist": "NewJeans", "title": "Super Shy", "scene": "k_pop", "failure_class": "k_pop"},
    {"artist": "FKA twigs", "title": "cellophane", "scene": "art_pop", "failure_class": "art_pop"},
)

_POLICY_FIELDS = ("tau", "sigma", "audio_weight")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _verify_content_hash(document: Mapping[str, Any], label: str) -> None:
    expected = document.get("content_sha256")
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    if not expected or expected != _content_hash(unsigned):
        raise DirectListError("%s content hash mismatch" % label)


def write_locked_seed_manifest(
    output: Any, policy_manifest_hash: str
) -> Dict[str, Any]:
    """Write the immutable identities and predeclared human-inspection rules."""
    digest = str(policy_manifest_hash).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DirectListError("policy_manifest_hash must be a SHA-256 hex digest")
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "set_id": "catalog-v8-direct-dev-adjacent-difficult-20",
        "set_status": "new locked DEV-adjacent difficult set",
        "output_status": "fresh list outputs; not FINAL labels",
        "seed_count": 20,
        "seeds": [dict(seed, id="direct-%02d" % (i + 1)) for i, seed in enumerate(LOCKED_SEEDS)],
        "policy_manifest_sha256": digest,
        "inspection_rules": {
            "required_seed_passes": 16,
            "total_seeds": 20,
            "positions_inspected_per_list": 5,
            "positions_1_to_3": "no unrelated result",
            "coherent_results_required_per_list": 4,
            "automatic_seed_failure": [
                "any junk result",
                "any duplicate result",
                "any seed-title variant",
            ],
        },
        "results_inspected": False,
        "result_outputs_at_lock": "unreviewed",
        "target_labels_included": False,
        "fresh_final_identities_included": False,
        "target_blind": True,
    }
    manifest["content_sha256"] = _content_hash(manifest)
    _write_json(Path(output), manifest)
    return manifest


def _policy_document(value: Any) -> Tuple[Mapping[str, Any], str, Optional[Path]]:
    if isinstance(value, Mapping):
        document = value
        return document, _content_hash(document), None
    path = Path(value)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise DirectListError("policy manifest must be a JSON object")
    return document, _file_hash(path), path


def _locked_policy(document: Mapping[str, Any]) -> CatalogPolicy:
    value: Any = document
    for keys in (
        ("policy",),
        ("exact_policy",),
        ("selected_policy",),
        ("selected_policy_evaluation", "exact_policy"),
        ("nested_5fold", "final_policy"),
    ):
        candidate: Any = document
        for key in keys:
            candidate = candidate.get(key) if isinstance(candidate, Mapping) else None
        if isinstance(candidate, Mapping):
            value = candidate
            break
    if not isinstance(value, Mapping) or set(value) != set(_POLICY_FIELDS):
        raise DirectListError("locked policy must contain exactly the three policy parameters")
    try:
        return CatalogPolicy(*(float(value[field]) for field in _POLICY_FIELDS))
    except (TypeError, ValueError, KeyError) as error:
        raise DirectListError("invalid locked policy") from error


def _default_preview_lookup(
    track_id: Any, title: str, artist: str, client: Optional[DeezerClient] = None
) -> Mapping[str, Any]:
    """Credential-free Deezer lookup; catalogue names remain authoritative."""
    deezer = client or DeezerClient()
    try:
        item = deezer._get("/track/%s" % int(track_id)) if track_id is not None else {}
        if not item or item.get("error"):
            found = deezer.search_track(title, artist)
            url = found.preview_url if found is not None else ""
        else:
            url = str(item.get("preview") or "")
        return {"url": url, "status": "available" if url else "missing"}
    except Exception as error:  # Network/API failure is data, not a fabricated preview.
        return {"url": "", "status": "error", "error": type(error).__name__}


def _preview(
    lookup: Callable[..., Any], track_id: Any, title: str, artist: str
) -> Dict[str, Any]:
    try:
        value = lookup(track_id, title, artist)
        if isinstance(value, Mapping):
            url = str(value.get("url", value.get("preview_url", "")) or "")
            status = str(value.get("status") or ("available" if url else "missing"))
            result = {"preview_url": url, "preview_status": status}
            if value.get("error"):
                result["preview_error"] = str(value["error"])
            return result
        url = str(value or "")
        return {"preview_url": url, "preview_status": "available" if url else "missing"}
    except Exception as error:
        return {
            "preview_url": "",
            "preview_status": "error",
            "preview_error": type(error).__name__,
        }


def _style_labels(styles: Any, artist: str, limit: int = 4) -> Dict[str, Any]:
    row = styles.artist_id(artist) if hasattr(styles, "artist_id") else None
    vector = styles.artist_vector(artist) if hasattr(styles, "artist_vector") else np.array([])
    names = tuple(getattr(styles, "scene_names", ()))
    order = np.argsort(-np.asarray(vector, dtype=np.float32), kind="stable")
    labels = [
        {"label": str(names[int(i)]), "weight": float(vector[int(i)])}
        for i in order[:limit]
        if int(i) < len(names) and float(vector[int(i)]) > 0.0
    ]
    direct = bool(styles.direct_mask[row]) if row is not None and hasattr(styles, "direct_mask") else False
    confidence = (
        float(styles.confidence[row])
        if row is not None and hasattr(styles, "confidence")
        else 0.0
    )
    return {
        "labels": labels,
        "source": "MusicBrainz_direct" if direct else "audio_propagated",
        "confidence": confidence,
    }


def _track_id(rec: Any, row: int) -> Any:
    value = rec.track_ids[row]
    return value.item() if isinstance(value, np.generic) else value


def _flags(
    rows: List[Dict[str, Any]], seed_title: str, seed_artist: str, quality: TitleQualityFilter
) -> None:
    seen_rows, seen_tracks = set(), set()
    for item in rows:
        row, track = int(item["row"]), item.get("track_id")
        duplicate = row in seen_rows or (track is not None and track in seen_tracks)
        seen_rows.add(row)
        if track is not None:
            seen_tracks.add(track)
        item["flags"] = {
            "junk": bool(quality.is_junk(item["title"], item["artist"])),
            "duplicate": bool(duplicate),
            "seed_variant": bool(quality.seed_title_in_result(seed_title, item["title"])),
            "same_artist": bool(PairResolver._artist_match(seed_artist, item["artist"])),
        }


def _serialize_candidate(
    item: Mapping[str, Any], styles: Any, preview_lookup: Callable[..., Any]
) -> Dict[str, Any]:
    rationale = item["rationale"]
    result = {
        "position": int(item["position"]),
        "title": str(item["title"]),
        "artist": str(item["artist"]),
        "row": int(item["row"]),
        "track_id": item.get("track_id"),
        "rationale": {
            "G": float(rationale["G"]),
            "A": float(rationale["A"]),
            "S": float(rationale["S"]),
            "lastfm_G": float(rationale.get("lastfm_G", 0.0)),
            "music4all_G": float(rationale.get("music4all_G", 0.0)),
            "A_definition": "audio-derived sonic/CLAP/vibe similarity",
            "source": str(rationale["source"]),
            "query_mode": str(rationale["query_mode"]),
        },
        "style": _style_labels(styles, str(item["artist"])),
    }
    result.update(_preview(preview_lookup, result["track_id"], result["title"], result["artist"]))
    return result


def run_direct_lists(
    manifest_path: Any,
    manifest_hash: str,
    policy_manifest: Any,
    index_path: Any,
    graph_path: Any,
    style_path: Any,
    output: Any = None,
    *,
    recommender_factory: Optional[Callable[[Path], Any]] = None,
    graph_factory: Callable[[Any], Any] = CatalogArtistGraph,
    style_factory: Callable[[Any], Any] = CatalogStyleIndex,
    resolver_factory: Callable[[Sequence[str], Sequence[str]], Any] = PairResolver,
    ranker_factory: Callable[..., Any] = CatalogPolicyRanker,
    production_factory: Callable[..., Any] = ProductionRanker,
    preview_lookup: Callable[..., Any] = _default_preview_lookup,
) -> Dict[str, Any]:
    """Generate challenger and current-production top fives without judgments."""
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise DirectListError("lock-seeds must be run before lists")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _verify_content_hash(manifest, "seed manifest")
    supplied_manifest_hash = str(manifest_hash).lower()
    manifest_file_hash = _file_hash(manifest_file)
    if supplied_manifest_hash not in {
        manifest_file_hash,
        str(manifest.get("content_sha256", "")).lower(),
    }:
        raise DirectListError("seed manifest hash mismatch")
    if manifest.get("results_inspected") is not False or manifest.get("seed_count") != 20:
        raise DirectListError("seed manifest is not the untouched locked 20-seed set")
    expected_seeds = [dict(seed, id="direct-%02d" % (i + 1)) for i, seed in enumerate(LOCKED_SEEDS)]
    if manifest.get("seeds") != expected_seeds:
        raise DirectListError("seed identities or scenes differ from the locked set")

    policy_doc, policy_hash, policy_file = _policy_document(policy_manifest)
    if policy_hash != manifest.get("policy_manifest_sha256"):
        raise DirectListError("policy manifest hash does not match the seed lock")
    policy = _locked_policy(policy_doc)

    if recommender_factory is None:
        from webapp.api._reco import WebRecommender
        recommender_factory = lambda path: WebRecommender(str(path))
    rec = recommender_factory(Path(index_path))
    graph = graph_factory(graph_path)
    styles = style_factory(style_path)
    resolver = resolver_factory(rec.titles, rec.artists)
    quality = TitleQualityFilter()
    ranker = ranker_factory(rec, graph, styles, policy, quality)
    production = production_factory(rec, set())

    records: List[Dict[str, Any]] = []
    for seed in manifest["seeds"]:
        query_row = resolver.query_row(seed)
        record: Dict[str, Any] = {"id": seed["id"], "seed": dict(seed)}
        if query_row is None:
            record["resolution"] = {"status": "failed", "reason": "catalogue identity unresolved"}
            record["lists"] = {}
            records.append(record)
            continue
        query_row = int(query_row)
        record["resolution"] = {
            "status": "resolved",
            "row": query_row,
            "track_id": _track_id(rec, query_row),
            "catalogue_title": str(rec.titles[query_row]),
            "catalogue_artist": str(rec.artists[query_row]),
            "resolver": "PairResolver",
        }
        challenger_payload = ranker.recommend(query_row, n=5)
        challenger = [
            _serialize_candidate(item, styles, preview_lookup)
            for item in challenger_payload["results"]
        ]
        production_rows = [int(row) for row in production.rank(query_row, "dual_sonic", n=5)]
        if len(challenger) != 5 or len(production_rows) != 5:
            raise DirectListError("%s requires two complete top-five lists" % seed["id"])
        audio = ranker.audio_scores(query_row)
        production_list: List[Dict[str, Any]] = []
        for position, row in enumerate(production_rows, start=1):
            title, artist = str(rec.titles[row]), str(rec.artists[row])
            item = {
                "position": position,
                "title": title,
                "artist": artist,
                "row": row,
                "track_id": _track_id(rec, row),
                "rationale": {
                    "G": 0.0,
                    "A": float(audio[row]),
                    "S": float(styles.style_overlap(str(rec.artists[query_row]), artist)),
                    "lastfm_G": 0.0,
                    "music4all_G": 0.0,
                    "A_definition": "audio-derived sonic/CLAP/vibe similarity",
                    "source": "current_production_dual_sonic",
                    "query_mode": str(getattr(rec, "last_retrieval_mode", "dual_sonic")),
                },
                "style": _style_labels(styles, artist),
            }
            item.update(_preview(preview_lookup, item["track_id"], title, artist))
            production_list.append(item)
        for values in (challenger, production_list):
            _flags(values, str(rec.titles[query_row]), str(rec.artists[query_row]), quality)
        raw_gate = challenger_payload.get("gate")
        if not isinstance(raw_gate, Mapping):
            raise DirectListError("%s challenger omitted gate metadata" % seed["id"])
        fired = bool(raw_gate.get("fired"))
        record["gate"] = {
            "fired": fired,
            "abstained": not fired,
            "reason": str(raw_gate.get("reason", "")),
            "agreement": float(raw_gate.get("agreement", 0.0)),
            "consistency": float(raw_gate.get("consistency", 0.0)),
            "thresholds": {
                "tau": float(raw_gate.get("thresholds", {}).get("tau", policy.tau)),
                "sigma": float(raw_gate.get("thresholds", {}).get("sigma", policy.sigma)),
            },
            "shared_count": int(raw_gate.get("shared_count", 0)),
            "source_coverage": dict(raw_gate.get("source_coverage", {})),
        }
        if not fired and [item["row"] for item in challenger] != production_rows:
            raise DirectListError(
                "%s abstention does not preserve exact production ordering" % seed["id"]
            )
        record["lists"] = {
            "catalog_policy": challenger,
            "current_production_dual_sonic": production_list,
        }
        records.append(record)

    report: Dict[str, Any] = {
        "schema_version": 1,
        "seed_manifest_sha256": manifest_file_hash,
        "verified_seed_manifest_hash_input": supplied_manifest_hash,
        "seed_manifest_content_sha256": manifest["content_sha256"],
        "policy_manifest_sha256": policy_hash,
        "policy": asdict(policy),
        "method": {
            "challenger": "CatalogPolicyRanker over CatalogArtistGraph and CatalogStyleIndex",
            "baseline": "current production WebRecommender dual_sonic",
            "formula": "Production-default abstention; fire only when independent-source agreement >= tau and consistency >= sigma. G = 0.5*lastfm_G + 0.5*music4all_G is the frozen equal source-graph blend, with G + audio_weight*A as the single audio tie-break.",
            "parameters": list(_POLICY_FIELDS),
            "A": "audio-derived equal sonic, CLAP, and vibe similarity blend",
            "preview": "credential-free Deezer public API availability lookup; names are never replaced",
        },
        "provenance": {
            "policy": "Production remains the default on abstention. Both the independent-source agreement threshold tau and consistency threshold sigma must pass; the source graph blend is frozen equal and audio_weight supplies the single audio tie-break.",
            "seed_manifest": str(manifest_file),
            "policy_manifest": str(policy_file) if policy_file else "inline canonical JSON",
            "assets": {
                "index": {"path": str(index_path), "sha256": _file_hash(Path(index_path))},
                "graph": {"path": str(graph_path), "sha256": _file_hash(Path(graph_path))},
                "style": {"path": str(style_path), "sha256": _file_hash(Path(style_path))},
            },
        },
        "target_blind_disclosure": {
            "target_labels_used": False,
            "output_based_replacement": False,
            "identity_resolution": "locked title/artist through PairResolver only",
            "human_judgments_included": False,
        },
        "records": records,
    }
    by_failure_class: Dict[str, Dict[str, int]] = {}
    for record in records:
        name = record["seed"]["failure_class"]
        counts = by_failure_class.setdefault(name, {"fired": 0, "abstained": 0})
        if "gate" in record:
            counts["fired" if record["gate"]["fired"] else "abstained"] += 1
    report["gate_summary_by_failure_class"] = by_failure_class
    report["content_sha256"] = _content_hash(report)
    if output is not None:
        _write_json(Path(output), report)
    return report


def _judgment_positions(judgment: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    value = judgment.get("positions")
    if not isinstance(value, list) or len(value) != 10:
        raise DirectListError("each judgment must inspect exactly ten positions (two top-five lists)")
    return value


def validate_judgments(lists: Any, judgments: Any) -> Dict[str, Any]:
    """Validate human records and compute the 16/20 gate without inferring coherence."""
    lists_doc = (
        json.loads(Path(lists).read_text(encoding="utf-8"))
        if not isinstance(lists, Mapping) else lists
    )
    judgment_doc = (
        json.loads(Path(judgments).read_text(encoding="utf-8"))
        if not isinstance(judgments, Mapping) else judgments
    )
    _verify_content_hash(lists_doc, "lists")
    supplied_hash = judgment_doc.get("lists_sha256")
    if supplied_hash != lists_doc["content_sha256"]:
        raise DirectListError("judgments are bound to a different list hash")
    values = judgment_doc.get("judgments")
    if not isinstance(values, list) or len(values) != 20:
        raise DirectListError("exactly 20 seed judgments are required")
    records = {record["id"]: record for record in lists_doc.get("records", [])}
    if len(records) != 20 or {item.get("id") for item in values} != set(records):
        raise DirectListError("judgments must cover each locked seed exactly once")

    effective: List[Dict[str, Any]] = []
    for judgment in values:
        seed_id = judgment["id"]
        for field in ("challenger_pass", "production_pass"):
            if type(judgment.get(field)) is not bool:
                raise DirectListError("%s requires an explicit %s bool" % (seed_id, field))
        generated_lists = records[seed_id].get("lists", {})
        if set(generated_lists) != {
            "catalog_policy", "current_production_dual_sonic"
        } or any(len(rows) != 5 for rows in generated_lists.values()):
            raise DirectListError("%s does not have two complete generated top fives" % seed_id)
        expected = []
        for list_name, rows in generated_lists.items():
            for item in rows:
                expected.append((list_name, int(item["position"]), item))
        positions = _judgment_positions(judgment)
        actual = {}
        inspected_junk = {
            "catalog_policy": False,
            "current_production_dual_sonic": False,
        }
        for position in positions:
            required = (
                "list", "position", "title", "artist", "rationale", "junk",
                "junk_evidence",
            )
            if any(field not in position for field in required):
                raise DirectListError("%s position judgment is incomplete" % seed_id)
            if (
                type(position["junk"]) is not bool
                or not str(position["rationale"]).strip()
                or not str(position["junk_evidence"]).strip()
            ):
                raise DirectListError(
                    "%s needs junk bool, rationale, and junk evidence" % seed_id
                )
            key = (str(position["list"]), int(position["position"]))
            if key in actual:
                raise DirectListError("%s repeats an inspected position" % seed_id)
            actual[key] = position
            if key[0] in inspected_junk:
                inspected_junk[key[0]] = (
                    inspected_junk[key[0]] or bool(position["junk"])
                )
        automatic = {
            "catalog_policy": False,
            "current_production_dual_sonic": False,
        }
        for list_name, number, generated in expected:
            inspected = actual.get((list_name, number))
            if inspected is None:
                raise DirectListError("%s omits an inspected position" % seed_id)
            if (
                str(inspected["title"]) != generated["title"]
                or str(inspected["artist"]) != generated["artist"]
            ):
                raise DirectListError("%s inspected names do not match locked output" % seed_id)
            flags = generated["flags"]
            automatic[list_name] = automatic[list_name] or any(
                bool(flags[name]) for name in ("junk", "duplicate", "seed_variant")
            )
        challenger_automatic = (
            automatic["catalog_policy"] or inspected_junk["catalog_policy"]
        )
        production_automatic = (
            automatic["current_production_dual_sonic"]
            or inspected_junk["current_production_dual_sonic"]
        )
        effective.append(
            {
                "id": seed_id,
                "failure_class": records[seed_id]["seed"]["failure_class"],
                "gate": dict(records[seed_id]["gate"]),
                "human_pass": {
                    "challenger": judgment["challenger_pass"],
                    "production": judgment["production_pass"],
                },
                "automatic_failure": {
                    "challenger": challenger_automatic,
                    "production": production_automatic,
                },
                "challenger_effective_pass": (
                    judgment["challenger_pass"] and not challenger_automatic
                ),
                "production_effective_pass": (
                    judgment["production_pass"] and not production_automatic
                ),
            }
        )
    challenger_passed = sum(item["challenger_effective_pass"] for item in effective)
    production_passed = sum(item["production_effective_pass"] for item in effective)
    gate_by_failure_class: Dict[str, Dict[str, int]] = {}
    for item in effective:
        counts = gate_by_failure_class.setdefault(
            item["failure_class"], {"fired": 0, "abstained": 0}
        )
        counts["fired" if item["gate"]["fired"] else "abstained"] += 1
    return {
        "schema_version": 1,
        "lists_sha256": lists_doc["content_sha256"],
        "judgments": 20,
        "challenger_effective_passes": challenger_passed,
        "production_effective_passes": production_passed,
        "required_passes": 16,
        "gate_met": challenger_passed >= 16,
        "coherence_inferred": False,
        "gate_summary_by_failure_class": gate_by_failure_class,
        "review_evidence_disclosure": {
            "explicit_method_pass_booleans": True,
            "all_top5_positions_reviewed_for_both_methods": True,
            "position_rationales_included": True,
            "position_junk_evidence_included": True,
            "coherence_inferred": False,
        },
        "per_seed": effective,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock-seeds")
    lock.add_argument("output", type=Path)
    lock.add_argument("policy_manifest_hash")
    lists = commands.add_parser("lists")
    lists.add_argument("manifest", type=Path)
    lists.add_argument("manifest_hash")
    lists.add_argument("policy_manifest", type=Path)
    lists.add_argument("index", type=Path)
    lists.add_argument("graph", type=Path)
    lists.add_argument("style", type=Path)
    lists.add_argument("output", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("lists", type=Path)
    validate.add_argument("judgments", type=Path)
    validate.add_argument("output", type=Path, nargs="?")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "lock-seeds":
        write_locked_seed_manifest(args.output, args.policy_manifest_hash)
    elif args.command == "lists":
        run_direct_lists(
            args.manifest, args.manifest_hash, args.policy_manifest,
            args.index, args.graph, args.style, args.output,
        )
    else:
        result = validate_judgments(args.lists, args.judgments)
        if args.output:
            _write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
