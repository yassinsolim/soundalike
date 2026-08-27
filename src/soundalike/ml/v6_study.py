"""Build an exposure-disjoint V6 development listening study.

V6 is model-improvement evidence. It is deliberately not an independent
promotion holdout, and its public artifacts keep method identity blinded.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from . import v5_study
from .v4_study import _content_sha256, _nested_keys, _write


SCHEMA_VERSION = 1
PACK_KIND = "soundalike_v6_development_full_ranking"
PRIVATE_KIND = "soundalike_v6_development_full_ranking_private"
PLAN_KIND = "soundalike_v6_development_study_plan"
PACK_ID = "v6-development-full-ranking-1"
PROTOCOL_KIND = "development_complete_ranking_v6_private_submission"
SUBMISSION_SCHEMA = "v6_development_listener_submission_v1"
LOCAL_STORAGE_NAMESPACE = "soundalike-development-v6-ranking-v1"
SUBMISSION_ENDPOINT = "/api/ratings-v6"
PRIVATE_BLOB_PREFIX = "human-ratings/development-v6-ranking-v1/"
EVIDENCE_ROLE = "development_model_improvement"
RANKING_SLOTS = (
    "most_similar",
    "next_most_similar",
    "second_least_similar",
    "least_similar",
)
REQUIRED_EXPOSURE_KINDS = frozenset(
    {
        "blinded_actual_served_lists_v17_private_submission_supersession",
        "fulltrack_jamendo_blind_pilot_v2",
        "fulltrack_semantic_blind_pilot_v1",
        "fulltrack_semantic_repeated_excerpt_pilot_v2",
        "blinded_repeated_excerpt_comparison_v3",
        "soundalike_v4_active_full_ranking",
        "soundalike_v5_strict_three_method_ranking",
    }
)
SAME_CORPUS_EXPOSURE_KINDS = REQUIRED_EXPOSURE_KINDS - {
    "blinded_actual_served_lists_v17_private_submission_supersession"
}
_TASK_ID = re.compile(r"^v6-(task|anchor)-[a-f0-9]{24}$")
_CHOICE_ID = re.compile(r"^v6-choice-[a-f0-9]{24}$")
_HEX_64 = re.compile(r"^[a-f0-9]{64}$")


class V6StudyError(RuntimeError):
    """The V6 study inputs or generated artifacts are invalid."""


@dataclass(frozen=True)
class ExposureInventory:
    """Validated earlier public exposure needed by the V6 builder."""

    hashes: Mapping[str, str]
    same_corpus_paths: tuple[Path, ...]
    same_corpus_labels: tuple[str, ...]
    external_artist_names: tuple[str, ...]


def _artist_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"artist", "artist_name"} and isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    names.add(cleaned)
            else:
                names.update(_artist_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_artist_names(item))
    return names


def load_v6_exposures(paths: Sequence[Path]) -> ExposureInventory:
    """Validate complete prior exposure and partition cross-corpus evidence."""
    resolved = tuple(path.resolve() for path in paths)
    if len(resolved) != len(set(resolved)):
        raise V6StudyError("V6 exposure paths must be distinct")
    documents: dict[str, tuple[Path, Mapping[str, object]]] = {}
    for path in resolved:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise V6StudyError(f"cannot read V6 exposure pack: {path}") from error
        if not isinstance(document, Mapping):
            raise V6StudyError(f"V6 exposure pack is not an object: {path.name}")
        kind = document.get("pack_kind")
        if not isinstance(kind, str) or kind not in REQUIRED_EXPOSURE_KINDS:
            raise V6StudyError(f"unexpected V6 exposure kind: {path.name}")
        if kind in documents:
            raise V6StudyError(f"duplicate V6 exposure kind: {kind}")
        digest = document.get("content_sha256")
        if not isinstance(digest, str) or digest != _content_sha256(document):
            raise V6StudyError(f"V6 exposure binding failed: {path.name}")
        documents[kind] = (path, document)
    missing = REQUIRED_EXPOSURE_KINDS - set(documents)
    extra = set(documents) - REQUIRED_EXPOSURE_KINDS
    if missing or extra:
        raise V6StudyError(
            "V6 requires every earlier evaluator exposure kind; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    same_corpus_paths = []
    same_corpus_labels = []
    external_artist_names: set[str] = set()
    hashes = {}
    for kind in sorted(documents):
        path, document = documents[kind]
        hashes[kind] = str(document["content_sha256"])
        if kind in SAME_CORPUS_EXPOSURE_KINDS:
            if not v5_study._collect_track_ids(document):
                raise V6StudyError(f"V6 exposure contains no tracks: {kind}")
            same_corpus_paths.append(path)
            same_corpus_labels.append(kind)
        else:
            external_artist_names.update(_artist_names(document))
    if not external_artist_names:
        raise V6StudyError("cross-corpus V6 exposure contains no artist names")
    return ExposureInventory(
        hashes=dict(sorted(hashes.items())),
        same_corpus_paths=tuple(same_corpus_paths),
        same_corpus_labels=tuple(same_corpus_labels),
        external_artist_names=tuple(sorted(external_artist_names, key=str.casefold)),
    )


def _require_inputs(
    *,
    directories: Mapping[str, Path],
    files: Mapping[str, Path],
) -> None:
    failures = []
    for name, path in directories.items():
        if not path.is_dir():
            failures.append(f"{name} is not a directory: {path}")
    for name, path in files.items():
        if not path.is_file():
            failures.append(f"{name} is not a file: {path}")
    if failures:
        raise V6StudyError("V6 private input preflight failed: " + "; ".join(failures))


def _rename_id(value: object) -> str:
    if not isinstance(value, str):
        raise V6StudyError("V5 source artifact has an invalid opaque ID")
    match = re.fullmatch(r"v5-(task|anchor|choice)-([a-f0-9]{24})", value)
    if not match:
        raise V6StudyError("V5 source artifact has an unexpected opaque ID")
    return f"v6-{match.group(1)}-{match.group(2)}"


def convert_v5_artifacts(
    public_v5: Mapping[str, object],
    private_v5: Mapping[str, object],
    plan_v5: Mapping[str, object],
    exposure: ExposureInventory,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    """Create a separately named, bound V6 development artifact family."""
    public = copy.deepcopy(dict(public_v5))
    private = copy.deepcopy(dict(private_v5))
    plan = copy.deepcopy(dict(plan_v5))
    tasks = public.get("tasks")
    private_tasks = private.get("tasks")
    if not isinstance(tasks, list) or not isinstance(private_tasks, list):
        raise V6StudyError("V5 source artifact task collections are invalid")

    task_ids = {
        task["task_id"]: _rename_id(task.get("task_id"))
        for task in tasks
        if isinstance(task, Mapping)
    }
    if len(task_ids) != len(tasks):
        raise V6StudyError("V5 source artifact task IDs are invalid")
    for task in tasks:
        task["task_id"] = task_ids[task["task_id"]]
        candidates = task.get("candidates")
        if not isinstance(candidates, list):
            raise V6StudyError("V5 source artifact candidates are invalid")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise V6StudyError("V5 source artifact candidate is invalid")
            candidate["choice_id"] = _rename_id(candidate.get("choice_id"))
    for task in private_tasks:
        if not isinstance(task, dict) or task.get("task_id") not in task_ids:
            raise V6StudyError("V5 private source task IDs are invalid")
        task["task_id"] = task_ids[task["task_id"]]
        if "anchor_of" in task:
            if task["anchor_of"] not in task_ids:
                raise V6StudyError("V5 private anchor source is invalid")
            task["anchor_of"] = task_ids[task["anchor_of"]]

    public.update(
        {
            "schema_version": SCHEMA_VERSION,
            "pack_kind": PACK_KIND,
            "pack_id": PACK_ID,
            "research_only": True,
            "promotion_allowed": False,
            "production_recommendation_changed": False,
            "development_evidence": True,
            "independent_holdout": False,
            "evidence_role": EVIDENCE_ROLE,
        }
    )
    task_format = dict(public.get("task_format", {}))
    task_format.update(
        {
            "candidates": 4,
            "questions": ["full_similarity_ranking", "worst_primary_reason"],
            "ranking_slots": list(RANKING_SLOTS),
        }
    )
    public["task_format"] = task_format
    provenance = dict(public.get("provenance", {}))
    provenance.update(
        {
            "prior_exposure_pack_sha256s": dict(exposure.hashes),
            "excludes_all_prior_exposed_tracks_and_artists": True,
            "includes_v5_exposure": True,
            "selection_objective": (
                "high-value frozen-method disagreement for model improvement"
            ),
            "listener_ratings_used_for_pack_selection": False,
        }
    )
    public["provenance"] = provenance

    private.update(
        {
            "schema_version": SCHEMA_VERSION,
            "private_kind": PRIVATE_KIND,
            "pack_id": PACK_ID,
            "development_evidence": True,
            "independent_holdout": False,
            "evidence_role": EVIDENCE_ROLE,
        }
    )
    plan.update(
        {
            "schema_version": SCHEMA_VERSION,
            "plan_kind": PLAN_KIND,
            "pack_id": PACK_ID,
            "development_evidence": True,
            "independent_holdout": False,
            "evidence_role": EVIDENCE_ROLE,
            "exposure_pack_sha256s": dict(exposure.hashes),
        }
    )
    private.pop("content_sha256", None)
    private["content_sha256"] = _content_sha256(private)
    public["private_unblinding_sha256"] = private["content_sha256"]
    public.pop("content_sha256", None)
    public["content_sha256"] = _content_sha256(public)
    plan["public_pack_sha256"] = public["content_sha256"]
    plan["private_unblinding_sha256"] = private["content_sha256"]
    plan.pop("content_sha256", None)
    plan["content_sha256"] = _content_sha256(plan)
    validate_v6_artifacts(public, private, plan, exposure=exposure)
    return public, private, plan


def _track_artist(tracks: Mapping[str, object], track_id: int) -> int:
    track = tracks.get(str(track_id))
    identity = track.get("source_identity") if isinstance(track, Mapping) else None
    artist_id = identity.get("artist_id") if isinstance(identity, Mapping) else None
    if isinstance(artist_id, bool) or not isinstance(artist_id, int) or artist_id <= 0:
        raise V6StudyError("V6 track artist identity is invalid")
    return artist_id


def _validate_track(track_id: int, value: object) -> None:
    if not isinstance(value, Mapping) or value.get("track_id") != track_id:
        raise V6StudyError("V6 track identity is invalid")
    if not all(
        isinstance(value.get(field), str) and str(value[field]).strip()
        for field in ("title", "artist")
    ):
        raise V6StudyError("V6 track labels are invalid")
    identity = value.get("source_identity")
    audio = value.get("audio")
    excerpt = audio.get("excerpt") if isinstance(audio, Mapping) else None
    attribution = value.get("attribution")
    start = excerpt.get("start_seconds") if isinstance(excerpt, Mapping) else None
    end = excerpt.get("end_seconds") if isinstance(excerpt, Mapping) else None
    if (
        not isinstance(identity, Mapping)
        or not _HEX_64.fullmatch(str(identity.get("source_audio_sha256", "")))
        or isinstance(identity.get("source_audio_bytes"), bool)
        or not isinstance(identity.get("source_audio_bytes"), int)
        or identity["source_audio_bytes"] < 100_000
        or not isinstance(audio, Mapping)
        or audio.get("url")
        != f"https://prod-1.storage.jamendo.com/?trackid={track_id}&format=mp31"
        or not isinstance(start, (int, float))
        or isinstance(start, bool)
        or not isinstance(end, (int, float))
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end - start > 20
        or not isinstance(attribution, Mapping)
        or not all(
            isinstance(attribution.get(field), str) and attribution[field].strip()
            for field in ("credit", "license_name", "license_url", "track_url")
        )
    ):
        raise V6StudyError("V6 lawful playback evidence is invalid")


def validate_v6_artifacts(
    public: Mapping[str, object],
    private: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    exposure: ExposureInventory,
) -> None:
    """Fail closed on V6 naming, role, blinding, playback, and diversity."""
    provenance = public.get("provenance")
    task_format = public.get("task_format")
    if (
        public.get("schema_version") != SCHEMA_VERSION
        or public.get("pack_kind") != PACK_KIND
        or public.get("pack_id") != PACK_ID
        or public.get("content_sha256") != _content_sha256(public)
        or public.get("research_only") is not True
        or public.get("promotion_allowed") is not False
        or public.get("production_recommendation_changed") is not False
        or public.get("development_evidence") is not True
        or public.get("independent_holdout") is not False
        or public.get("evidence_role") != EVIDENCE_ROLE
        or private.get("private_kind") != PRIVATE_KIND
        or private.get("pack_id") != PACK_ID
        or private.get("content_sha256") != _content_sha256(private)
        or private.get("development_evidence") is not True
        or private.get("independent_holdout") is not False
        or plan.get("plan_kind") != PLAN_KIND
        or plan.get("pack_id") != PACK_ID
        or plan.get("content_sha256") != _content_sha256(plan)
        or plan.get("development_evidence") is not True
        or plan.get("independent_holdout") is not False
        or public.get("private_unblinding_sha256")
        != private.get("content_sha256")
        or plan.get("public_pack_sha256") != public.get("content_sha256")
        or plan.get("private_unblinding_sha256")
        != private.get("content_sha256")
        or plan.get("exposure_pack_sha256s") != exposure.hashes
        or not isinstance(provenance, Mapping)
        or provenance.get("prior_exposure_pack_sha256s") != exposure.hashes
        or provenance.get("excludes_all_prior_exposed_tracks_and_artists") is not True
        or provenance.get("includes_v5_exposure") is not True
        or not isinstance(task_format, Mapping)
        or task_format.get("ranking_slots") != list(RANKING_SLOTS)
    ):
        raise V6StudyError("V6 study artifact binding failed")
    forbidden = set(v5_study.METHODS) | {
        "candidate_selection_sources",
        "method_bindings",
        "method_orders",
        "method_rankings",
    }
    if forbidden & _nested_keys(public):
        raise V6StudyError("V6 public artifact leaks method identity")

    tasks = public.get("tasks")
    private_tasks = private.get("tasks")
    tracks = public.get("tracks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != v5_study.UNIQUE_TASKS + v5_study.ANCHOR_TASKS
        or not isinstance(private_tasks, list)
        or len(private_tasks) != len(tasks)
        or not isinstance(tracks, Mapping)
    ):
        raise V6StudyError("V6 task coverage is invalid")
    private_by_id = {
        task.get("task_id"): task
        for task in private_tasks
        if isinstance(task, Mapping)
    }
    if len(private_by_id) != len(private_tasks):
        raise V6StudyError("V6 private task identities are invalid")
    used_tracks: set[int] = set()
    used_artists: set[int] = set()
    used_candidates: set[int] = set()
    choice_ids: set[str] = set()
    public_by_id = {}
    unique_count = 0
    anchor_count = 0
    for priority, task in enumerate(tasks, 1):
        if not isinstance(task, Mapping):
            raise V6StudyError("V6 task row is invalid")
        task_id = task.get("task_id")
        seed_id = task.get("seed_track_id")
        candidates = task.get("candidates")
        private_task = private_by_id.get(task_id)
        if (
            not isinstance(task_id, str)
            or not _TASK_ID.fullmatch(task_id)
            or task_id in public_by_id
            or task.get("priority_rank") != priority
            or isinstance(seed_id, bool)
            or not isinstance(seed_id, int)
            or seed_id <= 0
            or not isinstance(candidates, list)
            or len(candidates) != 4
            or not isinstance(private_task, Mapping)
            or private_task.get("seed_track_id") != seed_id
        ):
            raise V6StudyError("V6 task identity is invalid")
        public_by_id[task_id] = task
        candidate_ids = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise V6StudyError("V6 candidate row is invalid")
            choice_id = candidate.get("choice_id")
            track_id = candidate.get("track_id")
            if (
                not isinstance(choice_id, str)
                or not _CHOICE_ID.fullmatch(choice_id)
                or choice_id in choice_ids
                or isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id <= 0
                or track_id == seed_id
            ):
                raise V6StudyError("V6 candidate identity is invalid")
            choice_ids.add(choice_id)
            candidate_ids.append(track_id)
        if len(set(candidate_ids)) != 4:
            raise V6StudyError("V6 candidate identity is invalid")
        used_tracks.update([seed_id, *candidate_ids])
        if "anchor_of" in private_task:
            anchor_count += 1
            source = public_by_id.get(private_task.get("anchor_of"))
            if (
                not isinstance(source, Mapping)
                or source.get("seed_track_id") != seed_id
                or candidate_ids
                != [
                    row.get("track_id")
                    for row in reversed(source.get("candidates", []))
                    if isinstance(row, Mapping)
                ]
            ):
                raise V6StudyError("V6 repeated anchor binding is invalid")
            continue
        unique_count += 1
        artists = {_track_artist(tracks, seed_id)}
        artists.update(_track_artist(tracks, track_id) for track_id in candidate_ids)
        if (
            len(artists) != 5
            or artists & used_artists
            or set(candidate_ids) & used_candidates
        ):
            raise V6StudyError("V6 artist or candidate diversity is invalid")
        used_artists.update(artists)
        used_candidates.update(candidate_ids)
        orders = private_task.get("method_orders")
        rankings = private_task.get("method_rankings")
        if (
            not isinstance(orders, Mapping)
            or set(orders) != set(v5_study.METHODS)
            or not isinstance(rankings, Mapping)
            or set(rankings) != set(v5_study.METHODS)
            or any(
                not isinstance(orders[method], list)
                or set(orders[method]) != set(candidate_ids)
                or len(orders[method]) != 4
                or not isinstance(rankings[method], list)
                or len(rankings[method]) != v5_study.RANKING_DEPTH
                or len(set(rankings[method])) != v5_study.RANKING_DEPTH
                for method in v5_study.METHODS
            )
        ):
            raise V6StudyError("V6 private method rankings are invalid")
    if (
        unique_count != v5_study.UNIQUE_TASKS
        or anchor_count != v5_study.ANCHOR_TASKS
        or set(tracks) != {str(track_id) for track_id in used_tracks}
    ):
        raise V6StudyError("V6 task coverage or track table drifted")
    for track_id in sorted(used_tracks):
        _validate_track(track_id, tracks[str(track_id)])


def build_v6_protocol(pack_sha256: str) -> Mapping[str, object]:
    if not _HEX_64.fullmatch(pack_sha256):
        raise V6StudyError("V6 pack hash is invalid")
    protocol: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_kind": PROTOCOL_KIND,
        "submission_schema": SUBMISSION_SCHEMA,
        "pilot_pack_sha256": pack_sha256,
        "local_storage_namespace": LOCAL_STORAGE_NAMESPACE,
        "submission_endpoint": SUBMISSION_ENDPOINT,
        "private_blob_prefix": PRIVATE_BLOB_PREFIX,
        "task_count": v5_study.UNIQUE_TASKS + v5_study.ANCHOR_TASKS,
        "unique_task_count": v5_study.UNIQUE_TASKS,
        "repeated_anchor_count": v5_study.ANCHOR_TASKS,
        "adaptive_stop_after_unique_tasks": 12,
        "candidates_per_task": 4,
        "pairwise_predictions_per_rated_task": 6,
        "ranking_slots": list(RANKING_SLOTS),
        "worst_item_reason_required": True,
        "explicit_consent_required": True,
        "automatic_submission": False,
        "partial_submission_allowed": True,
        "skip_allowed": True,
        "language_evaluated": True,
        "language_segments_per_track": 3,
        "unknown_language_allowed": False,
        "transcription_saved": False,
        "research_only": True,
        "development_evidence": True,
        "independent_holdout": False,
        "promotion_allowed": False,
        "production_recommendation_changed": False,
        "evidence_role": EVIDENCE_ROLE,
    }
    protocol["content_sha256"] = _content_sha256(protocol)
    return protocol


def validate_v6_protocol(
    protocol: Mapping[str, object], *, pack_sha256: str
) -> None:
    expected = build_v6_protocol(pack_sha256)
    if protocol != expected or protocol.get("content_sha256") != _content_sha256(
        protocol
    ):
        raise V6StudyError("V6 protocol schema or binding is invalid")


def build_study(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    population_path: Path,
    store_root: Path,
    predictor_model: Path,
    predictor_metadata: Path,
    semantic_cache_path: Path,
    semantic_metadata_path: Path,
    vibe_cache_path: Path,
    preference_model_path: Path,
    blinding_key_path: Path,
    gate_cache_path: Path,
    exposure_pack_paths: Sequence[Path],
    workers: int,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    """Build V6 only when every private input and exposure is present."""
    _require_inputs(
        directories={
            "metadata_root": metadata_root,
            "audio_root": audio_root,
            "state_root": state_root,
            "store_root": store_root,
        },
        files={
            "population": population_path,
            "predictor_model": predictor_model,
            "predictor_metadata": predictor_metadata,
            "semantic_cache": semantic_cache_path,
            "semantic_metadata": semantic_metadata_path,
            "vibe_cache": vibe_cache_path,
            "preference_model": preference_model_path,
            "blinding_key": blinding_key_path,
            "gate_cache": gate_cache_path,
        },
    )
    exposure = load_v6_exposures(exposure_pack_paths)
    public_v5, private_v5, plan_v5 = v5_study.build_study(
        metadata_root=metadata_root,
        audio_root=audio_root,
        state_root=state_root,
        population_path=population_path,
        store_root=store_root,
        predictor_model=predictor_model,
        predictor_metadata=predictor_metadata,
        semantic_cache_path=semantic_cache_path,
        semantic_metadata_path=semantic_metadata_path,
        vibe_cache_path=vibe_cache_path,
        preference_model_path=preference_model_path,
        blinding_key_path=blinding_key_path,
        gate_cache_path=gate_cache_path,
        exposure_pack_paths=exposure.same_corpus_paths,
        exposure_pack_labels=exposure.same_corpus_labels,
        additional_exposed_artist_names=exposure.external_artist_names,
        workers=workers,
    )
    public, private, plan = convert_v5_artifacts(
        public_v5, private_v5, plan_v5, exposure
    )
    protocol = build_v6_protocol(str(public["content_sha256"]))
    validate_v6_protocol(protocol, pack_sha256=str(public["content_sha256"]))
    return public, private, plan, protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exposure-disjoint V6 development/model-improvement study."
        )
    )
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "population",
        "store_root",
        "predictor_model",
        "predictor_metadata",
        "semantic_cache",
        "semantic_metadata",
        "vibe_cache",
        "preference_model",
        "blinding_key",
        "gate_cache",
        "public_output",
        "private_output",
        "plan_output",
        "protocol_output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument(
        "--exposure-pack", action="append", type=Path, required=True
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    public, private, plan, protocol = build_study(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        population_path=args.population,
        store_root=args.store_root,
        predictor_model=args.predictor_model,
        predictor_metadata=args.predictor_metadata,
        semantic_cache_path=args.semantic_cache,
        semantic_metadata_path=args.semantic_metadata,
        vibe_cache_path=args.vibe_cache,
        preference_model_path=args.preference_model,
        blinding_key_path=args.blinding_key,
        gate_cache_path=args.gate_cache,
        exposure_pack_paths=args.exposure_pack,
        workers=args.workers,
    )
    _write(args.public_output, public)
    _write(args.private_output, private)
    _write(args.plan_output, plan)
    _write(args.protocol_output, protocol)
    print(
        json.dumps(
            {
                "pack_id": public["pack_id"],
                "content_sha256": public["content_sha256"],
                "protocol_sha256": protocol["content_sha256"],
                "evidence_role": EVIDENCE_ROLE,
                "independent_holdout": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
