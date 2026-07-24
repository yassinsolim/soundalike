"""Deterministic model-selection report for full-track trained candidates.

Rules
-----
* Automated metrics alone MUST NEVER promote.  `promotion_allowed` is always
  `False` unless strictly verified independent human evidence is supplied via
  `trusted_ratings_path` and every gate passes without exception.
* Jamendo tag retrieval metrics (per_scene / per_tag) are explicitly
  descriptive and non-deciding: no automated gate reads them.
* No current-time output; the report is fully deterministic given the inputs.
* No network or credential access.
* Symlinks, junctions, and reparse points are rejected in every existing path
  component.
* JSON reads are bounded, reject duplicate keys, and reject non-finite floats.
* Report output is written atomically with a canonical SHA-256 checksum.
* Both path-loaded and Mapping training/evaluation reports undergo identical
  schema/type/range/checksum validation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, secrets, stat, sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SELECTION_SCHEMA_VERSION: int = 1
CANDIDATE_LIST_SCHEMA_VERSION: int = 3
CANDIDATE_EVALUATION_SCHEMA_VERSION: int = 3
HUMAN_EVIDENCE_SCHEMA_VERSION: int = 1
OFFICIAL_FOLDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_SEEDS_PER_CANDIDATE_FOLD: int = 3
DEFAULT_CROSS_SEED_STABILITY_THRESHOLD: float = 0.05
_MIN_PRIMARY_GAIN_REL: float = 0.20
_MAX_SCENE_REGRESSION_ABS: float = 0.10
_MIN_COHERENT_TOP5_FRAC: float = 0.80
_MIN_INDEPENDENT_RATERS: int = 3
_MIN_DIFFICULT_SEEDS: int = 20
_MIN_RATER_SEED_COVERAGE: float = 0.80
_MAX_JSON_BYTES: int = 16 * 1024 * 1024

REASON_CODE_NOT_SUPPLIED: str = "not_supplied"
REASON_CODE_REJECTED: str = "rejected"
REASON_CODE_AUTOMATED_GATES_FAILED: str = "automated_gates_failed"
REASON_CODE_ACCEPTED: str = "accepted"

JAMENDO_TAG_DESCRIPTIVE_NOTICE: str = (
    "Jamendo tag-based retrieval metrics (per_scene, per_tag) are descriptive "
    "auxiliary information produced by the evaluation harness.  They are "
    "non-deciding: no automated gate or promotion decision reads these tag groupings."
)
AUTOMATED_PROMOTION_PROHIBITED_NOTICE: str = (
    "Automated metrics alone must never promote a model.  promotion_allowed is "
    "always False unless strictly verified independent human evidence is supplied and every gate passes."
)


class FullTrackSelectionError(RuntimeError):
    """Invalid input, unsafe path, tampered artifact, or gate failure."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(obj: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(obj))


def _validate_sha256_str(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FullTrackSelectionError(f"{label} must be a 64-character SHA-256 hex string")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise FullTrackSelectionError(f"{label} must be lowercase SHA-256 hex")
    return value


def _is_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        st = os.lstat(str(path))
    except OSError:
        return False
    attr = getattr(st, "st_file_attributes", 0)
    rp_bit = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attr & rp_bit)


def _reject_link_or_reparse(path: Path, label: str) -> None:
    try:
        if path.is_symlink():
            raise FullTrackSelectionError(f"{label} may not be a symlink/junction: {path}")
    except OSError as exc:
        raise FullTrackSelectionError(f"cannot inspect {label}: {path}") from exc
    if _is_reparse_point(path):
        raise FullTrackSelectionError(f"{label} may not be a reparse point/junction: {path}")


def _check_path_safety(raw: Union[str, Path], label: str) -> Path:
    path = Path(raw)
    for candidate in [path] + list(path.parents):
        if candidate.exists() or candidate.is_symlink():
            _reject_link_or_reparse(candidate, f"{label} component")
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise FullTrackSelectionError(f"cannot resolve {label}: {path}") from exc


def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise FullTrackSelectionError(f"JSON contains duplicate key: {key!r}")
        seen[key] = value
    return seen


def _reject_nonfinite(obj: object, label: str) -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise FullTrackSelectionError(f"{label} contains a non-finite float")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _reject_nonfinite(v, f"{label}.{k}")
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            _reject_nonfinite(v, f"{label}[{i}]")


def _safe_read_bounded(path: Path, label: str, max_bytes: int) -> bytes:
    resolved = _check_path_safety(path, label)
    _reject_link_or_reparse(resolved, label)
    if not resolved.is_file():
        raise FullTrackSelectionError(f"{label} is missing: {resolved}")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(str(resolved), flags)
    except OSError as exc:
        raise FullTrackSelectionError(f"cannot open {label}: {resolved}") from exc
    try:
        before = os.fstat(fd)
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise FullTrackSelectionError(f"{label} has invalid size: {before.st_size}")
        raw = bytearray()
        while True:
            block = os.read(fd, 65536)
            if not block:
                break
            raw.extend(block)
            if len(raw) > max_bytes:
                raise FullTrackSelectionError(f"{label} exceeds {max_bytes}-byte bound")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise FullTrackSelectionError(f"{label} changed during read")
    return bytes(raw)


def _parse_json_bytes_strict(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrackSelectionError(f"{label} is not valid JSON: {exc}") from exc
    except FullTrackSelectionError:
        raise
    if not isinstance(parsed, dict):
        raise FullTrackSelectionError(f"{label} must be a JSON object")
    _reject_nonfinite(parsed, label)
    return parsed


def _read_json_strict(path: Union[str, Path], label: str) -> Dict[str, Any]:
    resolved = _check_path_safety(path, label)
    raw = _safe_read_bounded(resolved, label, _MAX_JSON_BYTES)
    return _parse_json_bytes_strict(raw, label)


def _atomic_write_canonical_document_unchecked(
    path: Union[str, Path], value: Mapping[str, Any], label: str
) -> str:
    target = _check_path_safety(path, label)
    parent = _check_path_safety(target.parent, f"{label} parent")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_link_or_reparse(parent, f"{label} parent")
    if target.exists():
        _reject_link_or_reparse(target, label)
    raw = _canonical_json_bytes(dict(value))
    file_sha = _sha256_bytes(raw)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("xb") as fh:
            fh.write(raw)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        _reject_link_or_reparse(tmp, f"{label} temp")
        os.replace(str(tmp), str(target))
        try:
            parent_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
            finally:
                os.close(parent_fd)
        except OSError:
            pass
        _reject_link_or_reparse(target, label)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return file_sha


def _atomic_write_canonical_document(
    path: Union[str, Path], value: Mapping[str, Any], label: str
) -> str:
    try:
        return _atomic_write_canonical_document_unchecked(path, value, label)
    except OSError as exc:
        raise FullTrackSelectionError(f"cannot write {label}: {exc}") from exc

# ---------------------------------------------------------------------------
# aggregate_ratings.py output schema detection (tools/aggregate_ratings.py)
# ---------------------------------------------------------------------------

_AGGREGATE_RATINGS_OUTPUT_KEYS: frozenset = frozenset({
    "schema_version", "aggregate_kind", "session_count",
    "complete_result_ratings", "complete_list_ratings", "sessions",
})
_AGGREGATE_RATINGS_BINDING_REASON: str = (
    "current blinded aggregate (aggregate_ratings.py output, "
    "aggregate_kind='blinded_human_ratings_analysis') lacks full-track "
    "model/evaluation/candidate-list bindings and cannot authorize promotion"
)
_AGGREGATE_RATINGS_ZERO_REASON: str = (
    "blinded aggregate reports zero sessions and no rating evidence; "
    "cannot authorize promotion"
)

# ---------------------------------------------------------------------------
# Training report schema
# ---------------------------------------------------------------------------

_TRAIN_SCHEMA_VERSION: int = 1
_TRAIN_ARTIFACT_KIND: str = "fulltrack_train_report"
_TRAIN_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "artifact_kind", "job_status", "job_id",
    "fold", "seed", "candidate_kind", "created_at", "source_fingerprint",
    "store_binding", "store_binding_sha256", "store_manifest_sha256",
    "training_config", "training_config_sha256", "job_config_sha256",
    "dataset_hashes", "ranking_hashes", "view_hashes", "view_stats",
    "negative_mining", "metrics", "history", "resources", "model",
    "checkpoint", "notices", "report_sha256",
})
# store_binding exact fields: StoreBinding.as_dict() + sealed_manifest_sha256
_STORE_BINDING_FIELDS: frozenset = frozenset({
    "schema_version", "source_fingerprint", "config_sha256", "model_sha256",
    "model_id", "embedding_dim", "track_count", "shard_tracks",
    "repetition_sections", "salient_sections", "track_plan_sha256",
    "sealed_manifest_sha256",
})
_STORE_BINDING_POS_INT_FIELDS: Tuple[str, ...] = (
    "embedding_dim", "track_count", "shard_tracks",
    "repetition_sections", "salient_sections",
)
# training_config exact fields: TrainingConfig.as_dict()
_TRAINING_CONFIG_FIELDS: frozenset = frozenset({
    "max_epochs", "patience", "min_delta", "learning_rate", "weight_decay",
    "margin", "temperature", "gradient_clip_norm", "hard_negatives",
    "random_negatives", "maxsim_budget", "top_k", "coverage_threshold",
    "monotonic_hidden_dims", "min_train_tracks", "min_validation_tracks",
    "max_train_tracks", "max_validation_tracks", "device", "non_production",
})
_TRAINING_CONFIG_INT_FIELDS: Tuple[str, ...] = (
    "max_epochs", "patience", "hard_negatives", "random_negatives",
    "maxsim_budget", "top_k", "min_train_tracks", "min_validation_tracks",
)
_TRAINING_CONFIG_FINITE_FLOAT_FIELDS: Tuple[str, ...] = (
    "min_delta", "learning_rate", "weight_decay", "margin",
    "temperature", "gradient_clip_norm", "coverage_threshold",
)
_TRAIN_VAL_KEYS: frozenset = frozenset({"train", "validation"})
_TRAIN_RESOURCES_FIELDS: frozenset = frozenset({
    "wall_time_seconds", "cpu_rss_peak_bytes", "cuda_peak_bytes", "device",
})
_TRAIN_RESOURCE_FINITE_FIELDS: Tuple[str, ...] = (
    "wall_time_seconds", "cpu_rss_peak_bytes", "cuda_peak_bytes"
)
_METRICS_FIELDS: frozenset = frozenset({
    "train_loss", "validation_loss", "train_ranking_accuracy",
    "validation_ranking_accuracy", "train_pairwise_auc", "validation_pairwise_auc",
    "early_stopping_metric", "epochs_ran", "best_epoch",
})
_METRICS_FINITE_FLOAT_FIELDS: Tuple[str, ...] = (
    "train_loss", "validation_loss", "train_ranking_accuracy",
    "validation_ranking_accuracy", "train_pairwise_auc", "validation_pairwise_auc",
)
_METRICS_STRICT_INT_FIELDS: Tuple[str, ...] = ("epochs_ran", "best_epoch")
_MODEL_FIELDS: frozenset = frozenset({
    "model_json_sha256", "weights_npz_sha256", "artifact_sha256",
    "fusion_metadata", "parameter_count", "model_bytes", "runtime_parity_abs_diff",
})
_TRAIN_MODEL_INT_FIELDS: Tuple[str, ...] = ("parameter_count", "model_bytes")
_TRAIN_MODEL_FLOAT_FIELDS: Tuple[str, ...] = ("runtime_parity_abs_diff",)
_CHECKPOINT_FIELDS: frozenset = frozenset({
    "relative_dir", "checkpoint_sha256", "arrays_npz_sha256",
})
# Eval report resource fields (exact)
_EVAL_RESOURCES_FIELDS: frozenset = frozenset({"wall_seconds", "rss_bytes"})


def _validate_training_report_dict(report: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Full schema/type/range/checksum validation; identical for path-loaded and Mapping inputs."""
    if frozenset(report.keys()) != _TRAIN_REQUIRED_FIELDS:
        missing = _TRAIN_REQUIRED_FIELDS - frozenset(report.keys())
        extra = frozenset(report.keys()) - _TRAIN_REQUIRED_FIELDS
        raise FullTrackSelectionError(
            f"{label} schema fields differ (missing={sorted(missing)!r}, extra={sorted(extra)!r})"
        )
    if report.get("schema_version") != _TRAIN_SCHEMA_VERSION:
        raise FullTrackSelectionError(f"{label} schema_version must be {_TRAIN_SCHEMA_VERSION}")
    if report.get("artifact_kind") != _TRAIN_ARTIFACT_KIND:
        raise FullTrackSelectionError(f"{label} artifact_kind must be {_TRAIN_ARTIFACT_KIND!r}")
    if report.get("job_status") != "complete":
        raise FullTrackSelectionError(f"{label} job_status must be complete")
    payload = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("report_sha256") != _canonical_sha256(payload):
        raise FullTrackSelectionError(f"{label} report_sha256 mismatch")
    fold = report.get("fold")
    seed = report.get("seed")
    candidate_kind = report.get("candidate_kind")
    if isinstance(fold, bool) or not isinstance(fold, int):
        raise FullTrackSelectionError(f"{label} fold must be int")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FullTrackSelectionError(f"{label} seed must be non-negative int")
    if not isinstance(candidate_kind, str) or not candidate_kind:
        raise FullTrackSelectionError(f"{label} candidate_kind invalid")
    # Validate top-level SHA-256 fields
    for sha_field in ("source_fingerprint", "store_binding_sha256", "store_manifest_sha256",
                      "training_config_sha256", "job_config_sha256"):
        _validate_sha256_str(report.get(sha_field), f"{label} {sha_field}")
    # --- store_binding: exact fields + nested integrity ---
    sb = report.get("store_binding")
    if not isinstance(sb, dict) or frozenset(sb.keys()) != _STORE_BINDING_FIELDS:
        raise FullTrackSelectionError(f"{label} store_binding must have exact fields "
                                      f"{sorted(_STORE_BINDING_FIELDS)}")
    if sb.get("schema_version") != 2:
        raise FullTrackSelectionError(f"{label} store_binding.schema_version must be 2")
    for sha_f in ("source_fingerprint", "config_sha256", "model_sha256",
                  "track_plan_sha256", "sealed_manifest_sha256"):
        _validate_sha256_str(sb.get(sha_f), f"{label} store_binding.{sha_f}")
    if not isinstance(sb.get("model_id"), str) or not sb["model_id"]:
        raise FullTrackSelectionError(f"{label} store_binding.model_id must be nonempty string")
    for int_f in _STORE_BINDING_POS_INT_FIELDS:
        v = sb.get(int_f)
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise FullTrackSelectionError(f"{label} store_binding.{int_f} must be positive int")
    # Cross-check top-level fields against nested store_binding
    computed_sbsha = _canonical_sha256(sb)
    if str(report.get("store_binding_sha256", "")) != computed_sbsha:
        raise FullTrackSelectionError(
            f"{label} store_binding_sha256 does not match canonical SHA-256 of store_binding"
        )
    if str(report.get("source_fingerprint", "")) != str(sb.get("source_fingerprint", "")):
        raise FullTrackSelectionError(
            f"{label} top-level source_fingerprint does not match store_binding.source_fingerprint"
        )
    if str(report.get("store_manifest_sha256", "")) != str(sb.get("sealed_manifest_sha256", "")):
        raise FullTrackSelectionError(
            f"{label} store_manifest_sha256 does not match store_binding.sealed_manifest_sha256"
        )
    # --- training_config: exact fields + sha256 ---
    tc = report.get("training_config")
    if not isinstance(tc, dict) or frozenset(tc.keys()) != _TRAINING_CONFIG_FIELDS:
        raise FullTrackSelectionError(f"{label} training_config must have exact fields "
                                      f"{sorted(_TRAINING_CONFIG_FIELDS)}")
    for int_f in _TRAINING_CONFIG_INT_FIELDS:
        v = tc.get(int_f)
        if isinstance(v, bool) or not isinstance(v, int):
            raise FullTrackSelectionError(f"{label} training_config.{int_f} must be int")
    for flt_f in _TRAINING_CONFIG_FINITE_FLOAT_FIELDS:
        v = tc.get(flt_f)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise FullTrackSelectionError(f"{label} training_config.{flt_f} must be finite float")
    mhd = tc.get("monotonic_hidden_dims")
    if not isinstance(mhd, list) or not mhd:
        raise FullTrackSelectionError(f"{label} training_config.monotonic_hidden_dims must be nonempty list")
    for dim in mhd:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise FullTrackSelectionError(
                f"{label} training_config.monotonic_hidden_dims entries must be positive ints")
    for opt_f in ("max_train_tracks", "max_validation_tracks"):
        v = tc.get(opt_f)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 2):
            raise FullTrackSelectionError(f"{label} training_config.{opt_f} must be null or int >= 2")
    if not isinstance(tc.get("device"), str):
        raise FullTrackSelectionError(f"{label} training_config.device must be string")
    if not isinstance(tc.get("non_production"), bool):
        raise FullTrackSelectionError(f"{label} training_config.non_production must be bool")
    computed_tc_sha = _canonical_sha256(tc)
    if str(report.get("training_config_sha256", "")) != computed_tc_sha:
        raise FullTrackSelectionError(
            f"{label} training_config_sha256 does not match canonical SHA-256 of training_config"
        )
    # --- dataset_hashes, ranking_hashes: exact {train, validation} + SHA-256 values ---
    for hash_section in ("dataset_hashes", "ranking_hashes"):
        hs = report.get(hash_section)
        if not isinstance(hs, dict) or frozenset(hs.keys()) != _TRAIN_VAL_KEYS:
            raise FullTrackSelectionError(f"{label} {hash_section} must have exactly train/validation")
        for k in _TRAIN_VAL_KEYS:
            _validate_sha256_str(hs.get(k), f"{label} {hash_section}.{k}")
    # --- view_hashes: exact outer keys, lists of SHA-256 strings ---
    vh = report.get("view_hashes")
    if not isinstance(vh, dict) or frozenset(vh.keys()) != _TRAIN_VAL_KEYS:
        raise FullTrackSelectionError(f"{label} view_hashes must have exactly train/validation")
    for k in _TRAIN_VAL_KEYS:
        vhl = vh.get(k)
        if not isinstance(vhl, list) or not vhl:
            raise FullTrackSelectionError(f"{label} view_hashes.{k} must be nonempty list")
        for i, h in enumerate(vhl):
            _validate_sha256_str(h, f"{label} view_hashes.{k}[{i}]")
    # --- view_stats, negative_mining: exact outer {train, validation} dicts ---
    for outer_sec in ("view_stats", "negative_mining"):
        os_ = report.get(outer_sec)
        if not isinstance(os_, dict) or frozenset(os_.keys()) != _TRAIN_VAL_KEYS:
            raise FullTrackSelectionError(f"{label} {outer_sec} must have exactly train/validation")
        for k in _TRAIN_VAL_KEYS:
            if not isinstance(os_.get(k), dict):
                raise FullTrackSelectionError(f"{label} {outer_sec}.{k} must be object")
    # --- metrics: exact fields, finite floats, strict int epochs ---
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or frozenset(metrics.keys()) != _METRICS_FIELDS:
        raise FullTrackSelectionError(f"{label} metrics must have exact fields {sorted(_METRICS_FIELDS)}")
    for fld in _METRICS_FINITE_FLOAT_FIELDS:
        v = metrics.get(fld)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise FullTrackSelectionError(f"{label} metrics.{fld} must be finite numeric")
    for fld in _METRICS_STRICT_INT_FIELDS:
        v = metrics.get(fld)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise FullTrackSelectionError(f"{label} metrics.{fld} must be non-negative int")
    if not isinstance(metrics.get("early_stopping_metric"), str):
        raise FullTrackSelectionError(f"{label} metrics.early_stopping_metric must be string")
    # --- history: list of dicts with finite JSON ---
    history = report.get("history")
    if not isinstance(history, list):
        raise FullTrackSelectionError(f"{label} history must be list")
    for i, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise FullTrackSelectionError(f"{label} history[{i}] must be object")
        _reject_nonfinite(entry, f"{label} history[{i}]")
    # --- resources: exact fields + types ---
    resources = report.get("resources")
    if not isinstance(resources, dict) or frozenset(resources.keys()) != _TRAIN_RESOURCES_FIELDS:
        raise FullTrackSelectionError(f"{label} resources must have exact fields "
                                      f"{sorted(_TRAIN_RESOURCES_FIELDS)}")
    for field in _TRAIN_RESOURCE_FINITE_FIELDS:
        val = resources.get(field)
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(float(val))
            or float(val) < 0.0
        ):
            raise FullTrackSelectionError(
                f"{label} resources.{field} must be finite non-negative"
            )
    if not isinstance(resources.get("device"), str):
        raise FullTrackSelectionError(f"{label} resources.device must be string")
    # --- model: exact fields + types ---
    model = report.get("model")
    if not isinstance(model, dict) or frozenset(model.keys()) != _MODEL_FIELDS:
        raise FullTrackSelectionError(f"{label} model must have exact fields {sorted(_MODEL_FIELDS)}")
    for sha_f in ("model_json_sha256", "weights_npz_sha256", "artifact_sha256"):
        _validate_sha256_str(model.get(sha_f), f"{label} model.{sha_f}")
    if not isinstance(model.get("fusion_metadata"), dict):
        raise FullTrackSelectionError(f"{label} model.fusion_metadata must be object")
    for field in _TRAIN_MODEL_INT_FIELDS:
        val = model.get(field)
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise FullTrackSelectionError(f"{label} model.{field} must be non-negative int")
    for field in _TRAIN_MODEL_FLOAT_FIELDS:
        val = model.get(field)
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(float(val))
            or float(val) < 0.0
        ):
            raise FullTrackSelectionError(f"{label} model.{field} must be finite non-negative")
    # --- checkpoint: exact fields + hashes ---
    ckpt = report.get("checkpoint")
    if not isinstance(ckpt, dict) or frozenset(ckpt.keys()) != _CHECKPOINT_FIELDS:
        raise FullTrackSelectionError(f"{label} checkpoint must have exact fields "
                                      f"{sorted(_CHECKPOINT_FIELDS)}")
    if not isinstance(ckpt.get("relative_dir"), str) or not ckpt["relative_dir"]:
        raise FullTrackSelectionError(f"{label} checkpoint.relative_dir must be nonempty string")
    for sha_f in ("checkpoint_sha256", "arrays_npz_sha256"):
        _validate_sha256_str(ckpt.get(sha_f), f"{label} checkpoint.{sha_f}")
    # --- notices: nonempty list of strings ---
    notices = report.get("notices")
    if not isinstance(notices, list) or not notices:
        raise FullTrackSelectionError(f"{label} notices must be nonempty list")
    for i, n in enumerate(notices):
        if not isinstance(n, str) or not n:
            raise FullTrackSelectionError(f"{label} notices[{i}] must be nonempty string")
    return report


def _load_training_report(path: Union[str, Path]) -> Dict[str, Any]:
    label = f"training report {path}"
    report = _read_json_strict(path, label)
    return _validate_training_report_dict(report, label)

# ---------------------------------------------------------------------------
# Candidate list schema
# ---------------------------------------------------------------------------

_CLIST_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "artifact_kind", "list_id", "evaluation_identity",
    "candidates", "cross_seed_stability_threshold", "deciding_budget",
    "primary_metric", "content_sha256",
})
_EVAL_IDENTITY_REQUIRED_FIELDS: frozenset = frozenset(
    {"source_fingerprint", "store_binding_sha256"}
)


def _validate_candidate_list_dict(data: Dict[str, Any], label: str) -> Dict[str, Any]:
    if frozenset(data.keys()) != _CLIST_REQUIRED_FIELDS:
        raise FullTrackSelectionError(f"{label} schema fields differ")
    if data.get("schema_version") != CANDIDATE_LIST_SCHEMA_VERSION:
        raise FullTrackSelectionError(
            f"{label} schema_version must be {CANDIDATE_LIST_SCHEMA_VERSION}"
        )
    if data.get("artifact_kind") != "fulltrack_selection_candidate_list":
        raise FullTrackSelectionError(f"{label} artifact_kind mismatch")
    if not isinstance(data.get("list_id"), str) or not data["list_id"]:
        raise FullTrackSelectionError(f"{label} list_id must be non-empty str")
    payload = {k: v for k, v in data.items() if k != "content_sha256"}
    if data.get("content_sha256") != _canonical_sha256(payload):
        raise FullTrackSelectionError(f"{label} content_sha256 mismatch (tampered?)")
    eid = data.get("evaluation_identity")
    if not isinstance(eid, dict) or frozenset(eid.keys()) != _EVAL_IDENTITY_REQUIRED_FIELDS:
        raise FullTrackSelectionError(f"{label} evaluation_identity has wrong fields")
    _validate_sha256_str(
        eid.get("source_fingerprint"), f"{label} evaluation_identity.source_fingerprint"
    )
    _validate_sha256_str(
        eid.get("store_binding_sha256"), f"{label} evaluation_identity.store_binding_sha256"
    )
    deciding_budget = data.get("deciding_budget")
    if (
        isinstance(deciding_budget, bool)
        or not isinstance(deciding_budget, int)
        or deciding_budget not in (8, 16, 32)
    ):
        raise FullTrackSelectionError(
            f"{label} deciding_budget must be one of [8, 16, 32]"
        )
    if data.get("primary_metric") not in _METRIC_NAMES:
        raise FullTrackSelectionError(
            f"{label} primary_metric must be in {sorted(_METRIC_NAMES)}"
        )
    threshold = data.get("cross_seed_stability_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0.0
        or float(threshold) > DEFAULT_CROSS_SEED_STABILITY_THRESHOLD
    ):
        raise FullTrackSelectionError(
            f"{label} cross_seed_stability_threshold must be positive finite float "
            f"in (0, {DEFAULT_CROSS_SEED_STABILITY_THRESHOLD}] "
            "(artifact may be stricter, not looser)"
        )
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        raise FullTrackSelectionError(f"{label} candidates must be non-empty list")
    seen_kinds: set = set()
    for i, cand in enumerate(candidates):
        if not isinstance(cand, dict) or frozenset(cand.keys()) != frozenset(
            {"candidate_kind", "model_bundle_sha256"}
        ):
            raise FullTrackSelectionError(f"{label} candidates[{i}] has wrong fields")
        ck = cand.get("candidate_kind")
        if not isinstance(ck, str) or not ck:
            raise FullTrackSelectionError(f"{label} candidates[{i}].candidate_kind invalid")
        if ck in seen_kinds:
            raise FullTrackSelectionError(f"{label} duplicate candidate_kind: {ck!r}")
        seen_kinds.add(ck)
        _validate_sha256_str(
            cand.get("model_bundle_sha256"), f"{label} candidates[{i}].model_bundle_sha256"
        )
    return data


def _load_candidate_list(path: Union[str, Path]) -> Dict[str, Any]:
    label = f"candidate list {path}"
    data = _read_json_strict(path, label)
    return _validate_candidate_list_dict(data, label)


# ---------------------------------------------------------------------------
# Metric / paired summary validation (strict ranges)
# ---------------------------------------------------------------------------

_METRIC_NAMES: frozenset = frozenset({"recall_at_k", "mrr", "graded_ndcg_at_k"})
_METRIC_GROUPS: frozenset = frozenset({"candidate", "global", "frozen_hybrid"})
_EVAL_REPORT_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "artifact_kind", "candidate_kind", "fold", "seed",
    "model_artifact_sha256", "model_json_sha256", "weights_npz_sha256",
    "training_report_sha256", "job_config_sha256", "evaluation_identity",
    "evaluation_identity_sha256",
    "fold_query_sha256", "candidate_list_sha256", "benchmark_budget",
    "primary_metric",
    "metrics", "paired_candidate_minus_global",
    "paired_candidate_minus_frozen_hybrid", "resources", "content_sha256",
})
_PAIRED_REQUIRED: frozenset = frozenset({
    "mean_delta", "paired_bootstrap_ci95", "bootstrap_probability_delta_gt_zero"
})


def _validate_metric_group(obj: object, label: str) -> None:
    if not isinstance(obj, dict) or frozenset(obj.keys()) != _METRIC_NAMES:
        raise FullTrackSelectionError(f"{label} must have exactly {sorted(_METRIC_NAMES)}")
    for name in _METRIC_NAMES:
        val = obj[name]
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(float(val))
        ):
            raise FullTrackSelectionError(f"{label}.{name} must be finite float")
        if not (0.0 <= float(val) <= 1.0):
            raise FullTrackSelectionError(f"{label}.{name} must be in [0, 1]")


def _validate_paired_summary(obj: object, label: str) -> None:
    if not isinstance(obj, dict):
        raise FullTrackSelectionError(f"{label} must be object")
    if frozenset(obj.keys()) != _PAIRED_REQUIRED:
        missing = _PAIRED_REQUIRED - frozenset(obj.keys())
        extra = frozenset(obj.keys()) - _PAIRED_REQUIRED
        raise FullTrackSelectionError(
            f"{label} must have exact fields {sorted(_PAIRED_REQUIRED)} "
            f"(missing={sorted(missing)!r}, extra={sorted(extra)!r})"
        )
    mean_d = obj.get("mean_delta")
    if (
        isinstance(mean_d, bool)
        or not isinstance(mean_d, (int, float))
        or not math.isfinite(float(mean_d))
    ):
        raise FullTrackSelectionError(f"{label}.mean_delta must be finite")
    if not (-1.0 <= float(mean_d) <= 1.0):
        raise FullTrackSelectionError(f"{label}.mean_delta must be in [-1, 1]")
    ci = obj.get("paired_bootstrap_ci95")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v))
            for v in ci
        )
    ):
        raise FullTrackSelectionError(
            f"{label}.paired_bootstrap_ci95 must be [lo, hi] finite"
        )
    lo, hi = float(ci[0]), float(ci[1])
    if not (-1.0 <= lo <= hi <= 1.0):
        raise FullTrackSelectionError(
            f"{label}.paired_bootstrap_ci95 must be ordered lo<=hi in [-1, 1]"
        )
    prob = obj.get("bootstrap_probability_delta_gt_zero")
    if (
        isinstance(prob, bool)
        or not isinstance(prob, (int, float))
        or not math.isfinite(float(prob))
    ):
        raise FullTrackSelectionError(
            f"{label}.bootstrap_probability_delta_gt_zero must be finite"
        )
    if not (0.0 <= float(prob) <= 1.0):
        raise FullTrackSelectionError(
            f"{label}.bootstrap_probability_delta_gt_zero must be in [0, 1]"
        )

# ---------------------------------------------------------------------------
# Evaluation report schema + bundle helpers
# ---------------------------------------------------------------------------

def _validate_evaluation_report_dict(
    data: Dict[str, Any],
    label: str,
    *,
    expected_candidate_list_sha256: str,
    expected_deciding_budget: Optional[int] = None,
    expected_primary_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Full schema/type/range/checksum validation; identical for path-loaded and Mapping inputs."""
    if frozenset(data.keys()) != _EVAL_REPORT_REQUIRED_FIELDS:
        raise FullTrackSelectionError(f"{label} schema fields differ")
    if data.get("schema_version") != CANDIDATE_EVALUATION_SCHEMA_VERSION:
        raise FullTrackSelectionError(f"{label} schema_version mismatch")
    if data.get("artifact_kind") != "fulltrack_trained_candidate_evaluation":
        raise FullTrackSelectionError(f"{label} artifact_kind mismatch")
    cl_sha = data.get("candidate_list_sha256")
    _validate_sha256_str(cl_sha, f"{label} candidate_list_sha256")
    if cl_sha != expected_candidate_list_sha256:
        raise FullTrackSelectionError(
            f"{label} candidate_list_sha256 does not match provided candidate list "
            "(unrelated evaluation)"
        )
    fold = data.get("fold")
    seed = data.get("seed")
    candidate_kind = data.get("candidate_kind")
    if isinstance(fold, bool) or not isinstance(fold, int) or int(fold) not in OFFICIAL_FOLDS:
        raise FullTrackSelectionError(f"{label} fold must be in {OFFICIAL_FOLDS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FullTrackSelectionError(f"{label} seed must be non-negative int")
    if not isinstance(candidate_kind, str) or not candidate_kind:
        raise FullTrackSelectionError(f"{label} candidate_kind invalid")
    for field in (
        "model_artifact_sha256",
        "model_json_sha256",
        "weights_npz_sha256",
        "training_report_sha256",
        "job_config_sha256",
    ):
        _validate_sha256_str(data.get(field), f"{label} {field}")
    eid = data.get("evaluation_identity")
    if not isinstance(eid, dict):
        raise FullTrackSelectionError(f"{label} evaluation_identity missing")
    if frozenset(eid.keys()) != _EVAL_IDENTITY_REQUIRED_FIELDS:
        raise FullTrackSelectionError(
            f"{label} evaluation_identity must have exact fields "
            f"{sorted(_EVAL_IDENTITY_REQUIRED_FIELDS)}"
        )
    eid_sha = data.get("evaluation_identity_sha256")
    _validate_sha256_str(eid_sha, f"{label} evaluation_identity_sha256")
    if _canonical_sha256(eid) != eid_sha:
        raise FullTrackSelectionError(f"{label} evaluation_identity_sha256 mismatch")
    _validate_sha256_str(data.get("fold_query_sha256"), f"{label} fold_query_sha256")
    benchmark_budget = data.get("benchmark_budget")
    if (
        isinstance(benchmark_budget, bool)
        or not isinstance(benchmark_budget, int)
        or benchmark_budget not in (8, 16, 32)
    ):
        raise FullTrackSelectionError(
            f"{label} benchmark_budget must be one of [8, 16, 32]"
        )
    if (
        expected_deciding_budget is not None
        and benchmark_budget != expected_deciding_budget
    ):
        raise FullTrackSelectionError(
            f"{label} benchmark_budget does not match candidate list deciding_budget"
        )
    pm = data.get("primary_metric")
    if pm not in _METRIC_NAMES:
        raise FullTrackSelectionError(
            f"{label} primary_metric must be in {sorted(_METRIC_NAMES)}"
        )
    if expected_primary_metric is not None and pm != expected_primary_metric:
        raise FullTrackSelectionError(
            f"{label} primary_metric does not match candidate list primary_metric"
        )
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or frozenset(metrics.keys()) != _METRIC_GROUPS:
        raise FullTrackSelectionError(
            f"{label} metrics must have groups {sorted(_METRIC_GROUPS)}"
        )
    for group in _METRIC_GROUPS:
        _validate_metric_group(metrics[group], f"{label} metrics.{group}")
    _validate_paired_summary(
        data.get("paired_candidate_minus_global"),
        f"{label} paired_candidate_minus_global",
    )
    _validate_paired_summary(
        data.get("paired_candidate_minus_frozen_hybrid"),
        f"{label} paired_candidate_minus_frozen_hybrid",
    )
    resources = data.get("resources")
    if not isinstance(resources, dict) or frozenset(resources.keys()) != _EVAL_RESOURCES_FIELDS:
        raise FullTrackSelectionError(
            f"{label} resources must have exact fields {sorted(_EVAL_RESOURCES_FIELDS)}"
        )
    for field in ("wall_seconds", "rss_bytes"):
        val = resources.get(field)
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(float(val))
            or float(val) < 0.0
        ):
            raise FullTrackSelectionError(
                f"{label} resources.{field} must be non-negative finite"
            )
    payload = {k: v for k, v in data.items() if k != "content_sha256"}
    if data.get("content_sha256") != _canonical_sha256(payload):
        raise FullTrackSelectionError(f"{label} content_sha256 mismatch (tampered?)")
    return data


def _load_evaluation_report(
    path: Union[str, Path], *, expected_candidate_list_sha256: str,
    expected_deciding_budget: Optional[int] = None,
    expected_primary_metric: Optional[str] = None,
) -> Dict[str, Any]:
    label = f"evaluation report {path}"
    data = _read_json_strict(path, label)
    return _validate_evaluation_report_dict(
        data,
        label,
        expected_candidate_list_sha256=expected_candidate_list_sha256,
        expected_deciding_budget=expected_deciding_budget,
        expected_primary_metric=expected_primary_metric,
    )


def _compute_model_bundle_sha256(
    candidate_kind: str, training_reports: Sequence[Mapping[str, Any]]
) -> str:
    entries = sorted(
        (
            {
                "fold": int(r["fold"]),
                "seed": int(r["seed"]),
                "artifact_sha256": str(r["model"]["artifact_sha256"]),
                "model_json_sha256": str(r["model"]["model_json_sha256"]),
                "weights_npz_sha256": str(r["model"]["weights_npz_sha256"]),
                "training_report_sha256": str(r["report_sha256"]),
                "job_config_sha256": str(r["job_config_sha256"]),
            }
            for r in training_reports
        ),
        key=lambda e: (e["fold"], e["seed"]),
    )
    return _canonical_sha256(
        {"candidate_kind": candidate_kind, "model_identity_by_fold_seed": entries}
    )


def _compute_evaluation_bundle_sha256(
    candidate_kind: str, eval_reports: Sequence[Mapping[str, Any]]
) -> str:
    entries = sorted(
        (
            {
                "fold": int(r["fold"]),
                "seed": int(r["seed"]),
                "content_sha256": str(r["content_sha256"]),
            }
            for r in eval_reports
        ),
        key=lambda e: (e["fold"], e["seed"]),
    )
    return _canonical_sha256(
        {"candidate_kind": candidate_kind, "content_sha256_by_fold_seed": entries}
    )


def _selector_paired_summary(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, Mapping):
        raise FullTrackSelectionError(f"{label} must be an object")
    try:
        summary = {field: value[field] for field in _PAIRED_REQUIRED}
    except KeyError as exc:
        raise FullTrackSelectionError(f"{label} is incomplete") from exc
    _validate_paired_summary(summary, label)
    return summary


def build_selection_inputs(
    benchmark_reports: Sequence[Mapping[str, Any]],
    *,
    deciding_budget: int,
    primary_metric: str,
    list_id: str,
    stability_threshold: float = DEFAULT_CROSS_SEED_STABILITY_THRESHOLD,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert one validated trained benchmark matrix into selector inputs.

    The deciding budget and primary metric are bound into the candidate-list
    checksum before any per-candidate evaluation artifact is created. Jamendo
    tag metrics remain descriptive and cannot authorize promotion.
    """
    from .fulltrack_eval import (
        OFFICIAL_BUDGETS,
        FullTrackEvaluationError,
        aggregate_all_fold_results,
    )
    from .fulltrack_store import stable_json_sha256

    if (
        isinstance(deciding_budget, bool)
        or not isinstance(deciding_budget, int)
        or deciding_budget not in OFFICIAL_BUDGETS
    ):
        raise FullTrackSelectionError(
            f"deciding_budget must be one of {list(OFFICIAL_BUDGETS)}"
        )
    if primary_metric not in _METRIC_NAMES:
        raise FullTrackSelectionError(
            f"primary_metric must be in {sorted(_METRIC_NAMES)}"
        )
    if not isinstance(list_id, str) or not list_id:
        raise FullTrackSelectionError("list_id must be a non-empty string")
    if (
        isinstance(stability_threshold, bool)
        or not isinstance(stability_threshold, (int, float))
        or not math.isfinite(float(stability_threshold))
        or not 0.0 < float(stability_threshold) <= DEFAULT_CROSS_SEED_STABILITY_THRESHOLD
    ):
        raise FullTrackSelectionError(
            "stability_threshold must be positive and no greater than the default"
        )
    if not isinstance(benchmark_reports, Sequence) or not benchmark_reports:
        raise FullTrackSelectionError("benchmark_reports must be a non-empty sequence")
    first = benchmark_reports[0]
    if not isinstance(first, Mapping) or not isinstance(first.get("protocol"), Mapping):
        raise FullTrackSelectionError("benchmark report protocol is missing")
    bootstrap_iterations = first["protocol"].get("bootstrap_iterations")
    bootstrap_seed = first["protocol"].get("bootstrap_seed")
    if (
        isinstance(bootstrap_iterations, bool)
        or not isinstance(bootstrap_iterations, int)
        or bootstrap_iterations <= 0
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise FullTrackSelectionError("benchmark bootstrap protocol is invalid")
    try:
        aggregate_all_fold_results(
            benchmark_reports,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
    except FullTrackEvaluationError as exc:
        raise FullTrackSelectionError(
            f"benchmark matrix validation failed: {exc}"
        ) from exc

    selected_reports = [
        report
        for report in benchmark_reports
        if report["protocol"]["maxsim_budget"] == deciding_budget
    ]
    if len(selected_reports) != len(OFFICIAL_FOLDS):
        raise FullTrackSelectionError(
            "benchmark matrix does not contain exactly one deciding-budget report per fold"
        )
    first_selected = selected_reports[0]
    store = first_selected.get("store")
    if not isinstance(store, Mapping):
        raise FullTrackSelectionError("benchmark store binding is missing")
    evaluation_identity = {
        "source_fingerprint": first_selected["source_fingerprint"],
        "store_binding_sha256": stable_json_sha256(store),
    }
    evaluation_identity_sha256 = _canonical_sha256(evaluation_identity)

    by_candidate: Dict[str, Dict[Tuple[int, int], Mapping[str, object]]] = {}
    source_by_tuple: Dict[
        Tuple[str, int, int], Tuple[Mapping[str, Any], str, Mapping[str, object]]
    ] = {}
    for report in selected_reports:
        fold = int(report["protocol"]["fold_index"])
        model_bindings = report.get("trained_model_bindings")
        if not isinstance(model_bindings, Mapping):
            raise FullTrackSelectionError("deciding-budget report has no trained models")
        for method in report.get("trained_methods", []):
            details = model_bindings.get(method)
            if not isinstance(details, Mapping) or details.get("ablation") != "none":
                continue
            candidate_kind = str(details["candidate_kind"])
            seed = int(details["seed"])
            key = (fold, seed)
            candidate_models = by_candidate.setdefault(candidate_kind, {})
            if key in candidate_models:
                raise FullTrackSelectionError(
                    f"duplicate trained candidate tuple {candidate_kind}/{fold}/{seed}"
                )
            candidate_models[key] = details
            source_by_tuple[(candidate_kind, fold, seed)] = (report, method, details)
    if not by_candidate:
        raise FullTrackSelectionError("benchmark matrix has no non-ablation trained methods")

    candidates: List[Dict[str, object]] = []
    for candidate_kind in sorted(by_candidate):
        entries = by_candidate[candidate_kind]
        seeds = sorted({seed for _, seed in entries})
        expected_tuples = {
            (fold, seed) for fold in OFFICIAL_FOLDS for seed in seeds
        }
        if len(seeds) < MIN_SEEDS_PER_CANDIDATE_FOLD or set(entries) != expected_tuples:
            raise FullTrackSelectionError(
                f"candidate {candidate_kind} does not cover every fold/seed tuple"
            )
        bundle_entries = [
            {
                "fold": fold,
                "seed": seed,
                "artifact_sha256": str(entries[(fold, seed)]["model_artifact_sha256"]),
                "model_json_sha256": str(entries[(fold, seed)]["model_json_sha256"]),
                "weights_npz_sha256": str(entries[(fold, seed)]["weights_npz_sha256"]),
                "training_report_sha256": str(entries[(fold, seed)]["report_sha256"]),
                "job_config_sha256": str(entries[(fold, seed)]["job_config_sha256"]),
            }
            for fold, seed in sorted(entries)
        ]
        candidates.append(
            {
                "candidate_kind": candidate_kind,
                "model_bundle_sha256": _canonical_sha256(
                    {
                        "candidate_kind": candidate_kind,
                        "model_identity_by_fold_seed": bundle_entries,
                    }
                ),
            }
        )

    candidate_list: Dict[str, Any] = {
        "schema_version": CANDIDATE_LIST_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_selection_candidate_list",
        "list_id": list_id,
        "evaluation_identity": evaluation_identity,
        "candidates": candidates,
        "cross_seed_stability_threshold": float(stability_threshold),
        "deciding_budget": deciding_budget,
        "primary_metric": primary_metric,
    }
    candidate_list["content_sha256"] = _canonical_sha256(candidate_list)
    _validate_candidate_list_dict(candidate_list, "generated candidate list")

    evaluations: List[Dict[str, Any]] = []
    for candidate_kind, fold, seed in sorted(source_by_tuple):
        report, method, details = source_by_tuple[(candidate_kind, fold, seed)]
        aggregate = report.get("aggregate")
        paired = report.get("trained_paired_deltas")
        resources = report.get("resources")
        if (
            not isinstance(aggregate, Mapping)
            or not isinstance(paired, Mapping)
            or not isinstance(resources, Mapping)
        ):
            raise FullTrackSelectionError("trained benchmark result is incomplete")
        method_paired = paired.get(method)
        if not isinstance(method_paired, Mapping):
            raise FullTrackSelectionError(
                f"trained paired result is missing for {method}"
            )
        evaluation: Dict[str, Any] = {
            "schema_version": CANDIDATE_EVALUATION_SCHEMA_VERSION,
            "artifact_kind": "fulltrack_trained_candidate_evaluation",
            "candidate_kind": candidate_kind,
            "fold": fold,
            "seed": seed,
            "model_artifact_sha256": details["model_artifact_sha256"],
            "model_json_sha256": details["model_json_sha256"],
            "weights_npz_sha256": details["weights_npz_sha256"],
            "training_report_sha256": details["report_sha256"],
            "job_config_sha256": details["job_config_sha256"],
            "evaluation_identity": dict(evaluation_identity),
            "evaluation_identity_sha256": evaluation_identity_sha256,
            "fold_query_sha256": report["protocol"]["query_descriptor_sha256"],
            "candidate_list_sha256": candidate_list["content_sha256"],
            "benchmark_budget": deciding_budget,
            "primary_metric": primary_metric,
            "metrics": {
                "candidate": dict(aggregate[method]["metrics"]),
                "global": dict(aggregate["global_cosine"]["metrics"]),
                "frozen_hybrid": dict(aggregate["hybrid"]["metrics"]),
            },
            "paired_candidate_minus_global": _selector_paired_summary(
                method_paired["paired_candidate_minus_global"][primary_metric],
                f"{method} paired_candidate_minus_global",
            ),
            "paired_candidate_minus_frozen_hybrid": _selector_paired_summary(
                method_paired["paired_candidate_minus_frozen_hybrid"][primary_metric],
                f"{method} paired_candidate_minus_frozen_hybrid",
            ),
            "resources": {
                "wall_seconds": resources["wall_seconds"],
                "rss_bytes": resources["rss_observed_peak_bytes"],
            },
        }
        evaluation["content_sha256"] = _canonical_sha256(evaluation)
        _validate_evaluation_report_dict(
            evaluation,
            f"generated evaluation {candidate_kind}/{fold}/{seed}",
            expected_candidate_list_sha256=candidate_list["content_sha256"],
            expected_deciding_budget=deciding_budget,
            expected_primary_metric=primary_metric,
        )
        evaluations.append(evaluation)
    return candidate_list, evaluations


def write_selection_inputs(
    output_dir: Union[str, Path],
    candidate_list: Mapping[str, Any],
    evaluation_reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Atomically write selector inputs and a manifest, returning the manifest."""
    candidate = dict(candidate_list)
    _reject_nonfinite(candidate, "candidate list")
    _validate_candidate_list_dict(candidate, "candidate list")
    candidate_sha = str(candidate["content_sha256"])
    deciding_budget = int(candidate["deciding_budget"])
    primary_metric = str(candidate["primary_metric"])
    candidate_kinds = {
        str(item["candidate_kind"]) for item in candidate["candidates"]
    }

    requested_dir = Path(output_dir)
    if requested_dir.is_symlink():
        raise FullTrackSelectionError("selection input directory may not be a symlink")
    resolved_dir = _check_path_safety(requested_dir, "selection input directory")
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _reject_link_or_reparse(resolved_dir, "selection input directory")

    candidate_file = "candidate-list.json"
    candidate_file_sha = _atomic_write_canonical_document(
        resolved_dir / candidate_file, candidate, "candidate list"
    )
    entries: List[Dict[str, Any]] = []
    seen: set = set()
    for raw_report in evaluation_reports:
        report = dict(raw_report)
        _reject_nonfinite(report, "evaluation report")
        _validate_evaluation_report_dict(
            report,
            "evaluation report",
            expected_candidate_list_sha256=candidate_sha,
            expected_deciding_budget=deciding_budget,
            expected_primary_metric=primary_metric,
        )
        candidate_kind = str(report["candidate_kind"])
        fold = int(report["fold"])
        seed = int(report["seed"])
        key = (candidate_kind, fold, seed)
        if candidate_kind not in candidate_kinds:
            raise FullTrackSelectionError(
                f"evaluation candidate {candidate_kind!r} is not in candidate list"
            )
        if key in seen:
            raise FullTrackSelectionError(
                f"duplicate evaluation tuple {candidate_kind}/{fold}/{seed}"
            )
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in candidate_kind
        ):
            raise FullTrackSelectionError(
                f"candidate kind is not filename-safe: {candidate_kind!r}"
            )
        seen.add(key)
        file_name = (
            f"evaluation-{candidate_kind}-fold-{fold}-seed-{seed}.json"
        )
        file_sha = _atomic_write_canonical_document(
            resolved_dir / file_name, report, "evaluation report"
        )
        entries.append(
            {
                "candidate_kind": candidate_kind,
                "fold": fold,
                "seed": seed,
                "file": file_name,
                "file_sha256": file_sha,
                "content_sha256": report["content_sha256"],
            }
        )
    if not entries:
        raise FullTrackSelectionError("at least one evaluation report is required")
    entries.sort(key=lambda item: (item["candidate_kind"], item["fold"], item["seed"]))
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "fulltrack_selection_inputs",
        "candidate_list": {
            "file": candidate_file,
            "file_sha256": candidate_file_sha,
            "content_sha256": candidate_sha,
        },
        "evaluation_identity_sha256": _canonical_sha256(
            candidate["evaluation_identity"]
        ),
        "deciding_budget": deciding_budget,
        "primary_metric": primary_metric,
        "evaluation_reports": entries,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _atomic_write_canonical_document(
        resolved_dir / "selection-inputs.json",
        manifest,
        "selection input manifest",
    )
    return manifest


def _manifest_child(base: Path, name: object, label: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or Path(name).is_absolute()
    ):
        raise FullTrackSelectionError(f"{label} must be a plain relative filename")
    path = _check_path_safety(base / name, label)
    _reject_link_or_reparse(path, label)
    return path


def _resolve_selection_manifest(
    training_root: Union[str, Path], manifest_path: Union[str, Path]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    manifest_file = _check_path_safety(manifest_path, "selection input manifest")
    manifest = _read_json_strict(manifest_file, "selection input manifest")
    fields = {
        "schema_version",
        "artifact_kind",
        "candidate_list",
        "evaluation_identity_sha256",
        "deciding_budget",
        "primary_metric",
        "evaluation_reports",
        "manifest_sha256",
    }
    if set(manifest) != fields:
        raise FullTrackSelectionError("selection input manifest schema fields differ")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != "fulltrack_selection_inputs"
    ):
        raise FullTrackSelectionError("selection input manifest schema/version drift")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _canonical_sha256(manifest_payload):
        raise FullTrackSelectionError("selection input manifest checksum mismatch")
    _validate_sha256_str(
        manifest.get("evaluation_identity_sha256"),
        "selection input manifest evaluation_identity_sha256",
    )
    deciding_budget = manifest.get("deciding_budget")
    primary_metric = manifest.get("primary_metric")
    if (
        isinstance(deciding_budget, bool)
        or not isinstance(deciding_budget, int)
        or deciding_budget not in (8, 16, 32)
        or primary_metric not in _METRIC_NAMES
    ):
        raise FullTrackSelectionError(
            "selection input manifest budget/primary metric is invalid"
        )

    base = manifest_file.parent
    candidate_entry = manifest.get("candidate_list")
    candidate_fields = {"file", "file_sha256", "content_sha256"}
    if not isinstance(candidate_entry, dict) or set(candidate_entry) != candidate_fields:
        raise FullTrackSelectionError("selection input candidate-list entry is invalid")
    for field in ("file_sha256", "content_sha256"):
        _validate_sha256_str(
            candidate_entry.get(field), f"selection input candidate list {field}"
        )
    candidate_path = _manifest_child(
        base, candidate_entry.get("file"), "selection input candidate list"
    )
    candidate_raw = _safe_read_bounded(
        candidate_path, "selection input candidate list", _MAX_JSON_BYTES
    )
    if _sha256_bytes(candidate_raw) != candidate_entry["file_sha256"]:
        raise FullTrackSelectionError("candidate list file SHA-256 mismatch")
    candidate_list = _validate_candidate_list_dict(
        _parse_json_bytes_strict(
            candidate_raw, "selection input candidate list"
        ),
        "selection input candidate list",
    )
    if (
        candidate_list["content_sha256"] != candidate_entry["content_sha256"]
        or candidate_list["deciding_budget"] != deciding_budget
        or candidate_list["primary_metric"] != primary_metric
        or _canonical_sha256(candidate_list["evaluation_identity"])
        != manifest["evaluation_identity_sha256"]
    ):
        raise FullTrackSelectionError(
            "candidate list does not match selection input manifest"
        )

    raw_entries = manifest.get("evaluation_reports")
    evaluation_fields = {
        "candidate_kind",
        "fold",
        "seed",
        "file",
        "file_sha256",
        "content_sha256",
    }
    if not isinstance(raw_entries, list) or not raw_entries:
        raise FullTrackSelectionError("selection input evaluations are missing")
    evaluations: List[Dict[str, Any]] = []
    training_reports: List[Dict[str, Any]] = []
    seen_tuples: set = set()
    seen_files: set = set()

    root = _check_path_safety(training_root, "training root")
    _reject_link_or_reparse(root, "training root")
    if not root.is_dir():
        raise FullTrackSelectionError(f"training root is missing: {root}")
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict) or set(entry) != evaluation_fields:
            raise FullTrackSelectionError(
                f"selection input evaluation entry {index} is invalid"
            )
        candidate_kind = entry.get("candidate_kind")
        fold = entry.get("fold")
        seed = entry.get("seed")
        if (
            not isinstance(candidate_kind, str)
            or not candidate_kind
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in candidate_kind
            )
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in OFFICIAL_FOLDS
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise FullTrackSelectionError(
                f"selection input evaluation identity {index} is invalid"
            )
        key = (candidate_kind, fold, seed)
        if key in seen_tuples:
            raise FullTrackSelectionError(
                f"duplicate selection input evaluation tuple: {key}"
            )
        seen_tuples.add(key)
        for field in ("file_sha256", "content_sha256"):
            _validate_sha256_str(
                entry.get(field), f"selection input evaluation {index} {field}"
            )
        evaluation_path = _manifest_child(
            base, entry.get("file"), f"selection input evaluation {index}"
        )
        if evaluation_path in seen_files:
            raise FullTrackSelectionError("duplicate selection input evaluation file")
        seen_files.add(evaluation_path)
        raw = _safe_read_bounded(
            evaluation_path,
            f"selection input evaluation {index}",
            _MAX_JSON_BYTES,
        )
        if _sha256_bytes(raw) != entry["file_sha256"]:
            raise FullTrackSelectionError("evaluation report file SHA-256 mismatch")
        evaluation = _validate_evaluation_report_dict(
            _parse_json_bytes_strict(
                raw, f"selection input evaluation {index}"
            ),
            f"selection input evaluation {index}",
            expected_candidate_list_sha256=str(candidate_list["content_sha256"]),
            expected_deciding_budget=int(deciding_budget),
            expected_primary_metric=str(primary_metric),
        )
        if (
            evaluation["candidate_kind"] != candidate_kind
            or evaluation["fold"] != fold
            or evaluation["seed"] != seed
            or evaluation["content_sha256"] != entry["content_sha256"]
        ):
            raise FullTrackSelectionError(
                f"selection input evaluation {index} disagrees with manifest"
            )
        training_path = _check_path_safety(
            root
            / f"fold-{fold}"
            / candidate_kind
            / f"seed-{seed}"
            / "report.json",
            f"training report {candidate_kind}/{fold}/{seed}",
        )
        _reject_link_or_reparse(
            training_path, f"training report {candidate_kind}/{fold}/{seed}"
        )
        if not training_path.is_file():
            raise FullTrackSelectionError(
                f"training report is missing: {candidate_kind}/{fold}/{seed}"
            )
        training_raw = _safe_read_bounded(
            training_path,
            f"training report {candidate_kind}/{fold}/{seed}",
            _MAX_JSON_BYTES,
        )
        training = _validate_training_report_dict(
            _parse_json_bytes_strict(
                training_raw,
                f"training report {candidate_kind}/{fold}/{seed}",
            ),
            f"training report {candidate_kind}/{fold}/{seed}",
        )
        if (
            training["candidate_kind"] != candidate_kind
            or training["fold"] != fold
            or training["seed"] != seed
        ):
            raise FullTrackSelectionError(
                f"training report identity disagrees with manifest: {candidate_kind}/{fold}/{seed}"
            )
        expected_training_identity = {
            "model_artifact_sha256": training["model"]["artifact_sha256"],
            "model_json_sha256": training["model"]["model_json_sha256"],
            "weights_npz_sha256": training["model"]["weights_npz_sha256"],
            "training_report_sha256": training["report_sha256"],
            "job_config_sha256": training["job_config_sha256"],
        }
        if any(
            evaluation[field] != expected
            for field, expected in expected_training_identity.items()
        ):
            raise FullTrackSelectionError(
                f"training report identity differs from evaluated model: {candidate_kind}/{fold}/{seed}"
            )
        evaluations.append(evaluation)
        training_reports.append(training)
    return training_reports, evaluations, candidate_list

# ---------------------------------------------------------------------------
# Automated gates
# ---------------------------------------------------------------------------

def _run_automated_gates(
    candidate_kind: str,
    training_reports: List[Mapping[str, Any]],
    eval_reports: List[Mapping[str, Any]],
    *,
    stability_threshold: float,
    declared_model_bundle_sha256: str,
    evaluation_identity: Dict[str, Any],
    evaluation_identity_sha256: str,
    global_fq_conflict_folds: Optional[set] = None,
    global_pm_consistent: Optional[bool] = None,
) -> Dict[str, Any]:
    gates: Dict[str, Dict[str, Any]] = {}
    train_folds = {int(r["fold"]) for r in training_reports}
    gate_tf = train_folds == set(OFFICIAL_FOLDS)
    gates["training_folds_complete"] = {
        "passed": gate_tf,
        "reason": (
            "all official folds 0..4 present in training"
            if gate_tf
            else f"training fold set {sorted(train_folds)} != official {sorted(OFFICIAL_FOLDS)}"
        ),
    }
    eval_folds = {int(r["fold"]) for r in eval_reports}
    gate_a = eval_folds == set(OFFICIAL_FOLDS)
    gates["all_official_folds_present"] = {
        "passed": gate_a,
        "reason": (
            "all official folds 0..4 present"
            if gate_a
            else f"fold set {sorted(eval_folds)} != official {sorted(OFFICIAL_FOLDS)}"
        ),
    }
    seeds_by_fold: Dict[int, set] = {}
    for r in eval_reports:
        seeds_by_fold.setdefault(int(r["fold"]), set()).add(int(r["seed"]))
    insufficient = [
        f for f in sorted(seeds_by_fold)
        if len(seeds_by_fold[f]) < MIN_SEEDS_PER_CANDIDATE_FOLD
    ]
    gate_b = not insufficient
    gates["min_seeds_per_fold"] = {
        "passed": gate_b,
        "reason": (
            f">={MIN_SEEDS_PER_CANDIDATE_FOLD} unique seeds per fold"
            if gate_b
            else f"folds with < {MIN_SEEDS_PER_CANDIDATE_FOLD} seeds: {insufficient}"
        ),
        "seeds_by_fold": {str(f): len(s) for f, s in sorted(seeds_by_fold.items())},
    }
    train_tuples = {(int(r["fold"]), int(r["seed"])) for r in training_reports}
    eval_tuples = {(int(r["fold"]), int(r["seed"])) for r in eval_reports}
    gate_ta = train_tuples == eval_tuples
    gates["fold_seed_tuple_alignment"] = {
        "passed": gate_ta,
        "reason": (
            "training and evaluation (fold, seed) tuple sets match"
            if gate_ta
            else "training/evaluation (fold, seed) sets differ"
        ),
    }
    train_dupe = len(train_tuples) != len(training_reports)
    eval_dupe = len(eval_tuples) != len(eval_reports)
    gate_nd = not train_dupe and not eval_dupe
    _ds = (
        ("training" if train_dupe else "")
        + ("/" if train_dupe and eval_dupe else "")
        + ("evaluation" if eval_dupe else "")
    )
    gates["no_duplicate_tuples"] = {
        "passed": gate_nd,
        "reason": (
            "no duplicate (fold, seed) tuples"
            if gate_nd
            else f"duplicate (fold, seed) tuples in {_ds}"
        ),
    }
    train_map: Dict[Tuple[int, int], Dict[str, str]] = {
        (int(r["fold"]), int(r["seed"])): {
            "model_artifact_sha256": str(r["model"]["artifact_sha256"]),
            "model_json_sha256": str(r["model"]["model_json_sha256"]),
            "weights_npz_sha256": str(r["model"]["weights_npz_sha256"]),
            "training_report_sha256": str(r["report_sha256"]),
            "job_config_sha256": str(r["job_config_sha256"]),
        }
        for r in training_reports
    }
    mismatches: List[str] = []
    for r in eval_reports:
        key = (int(r["fold"]), int(r["seed"]))
        expected_identity = train_map.get(key)
        if expected_identity is None:
            mismatches.append(f"fold={key[0]},seed={key[1]}: no matching training report")
            continue
        for field, expected in expected_identity.items():
            actual = str(r.get(field, ""))
            if actual != expected:
                mismatches.append(
                    f"fold={key[0]},seed={key[1]}: {field} "
                    f"train={expected[:10]} eval={actual[:10]}"
                )
    gate_c = not mismatches
    gates["model_hash_match"] = {
        "passed": gate_c,
        "reason": (
            "training report, job config, and model file identities match"
            if gate_c
            else f"identity mismatches: {mismatches[:5]}"
        ),
    }
    cl_eid_sha = evaluation_identity_sha256
    eid_bad = [
        f"fold={r['fold']},seed={r['seed']}"
        for r in eval_reports
        if r.get("evaluation_identity_sha256") != cl_eid_sha
    ]
    gate_eim = not eid_bad
    gates["eval_identity_matches_candidate_list"] = {
        "passed": gate_eim,
        "reason": (
            "all eval reports share the candidate list evaluation_identity"
            if gate_eim
            else f"eval reports with mismatched identity: {eid_bad[:5]}"
        ),
    }
    eid_shas = {str(r.get("evaluation_identity_sha256", "")) for r in eval_reports}
    gate_d = len(eid_shas) == 1
    gates["evaluation_identity_aligned"] = {
        "passed": gate_d,
        "reason": (
            "all evaluation reports share the same evaluation_identity"
            if gate_d
            else f"{len(eid_shas)} distinct evaluation_identity values"
        ),
    }
    eid_sfp = str(evaluation_identity.get("source_fingerprint", ""))
    eid_sbsha = str(evaluation_identity.get("store_binding_sha256", ""))
    train_id_bad: List[str] = []
    for r in training_reports:
        if str(r.get("source_fingerprint", "")) != eid_sfp:
            train_id_bad.append(
                f"fold={r['fold']},seed={r['seed']}: source_fingerprint mismatch"
            )
        if str(r.get("store_binding_sha256", "")) != eid_sbsha:
            train_id_bad.append(
                f"fold={r['fold']},seed={r['seed']}: store_binding_sha256 mismatch"
            )
    gate_tsa = not train_id_bad
    gates["training_source_aligned"] = {
        "passed": gate_tsa,
        "reason": (
            "training source_fingerprint/store_binding_sha256 aligned to evaluation identity"
            if gate_tsa
            else f"training identity mismatches: {train_id_bad[:5]}"
        ),
    }
    fq_by_fold: Dict[int, set] = {}
    for r in eval_reports:
        fq_by_fold.setdefault(int(r["fold"]), set()).add(str(r.get("fold_query_sha256", "")))
    fq_drift = [f for f, qs in fq_by_fold.items() if len(qs) > 1]
    gate_fq = not fq_drift
    gates["fold_query_sha256_aligned"] = {
        "passed": gate_fq,
        "reason": (
            "fold_query_sha256 consistent within each fold"
            if gate_fq
            else f"fold_query_sha256 drift in folds: {fq_drift[:5]}"
        ),
    }
    pm_vals = {str(r.get("primary_metric", "")) for r in eval_reports}
    gate_pm = len(pm_vals) == 1
    gates["primary_metric_aligned"] = {
        "passed": gate_pm,
        "reason": (
            "primary_metric consistent across all evaluation reports"
            if gate_pm
            else f"primary_metric values differ: {sorted(pm_vals)}"
        ),
    }
    missing_pairs: List[str] = []
    for r in eval_reports:
        for field in (
            "paired_candidate_minus_global",
            "paired_candidate_minus_frozen_hybrid",
        ):
            if not isinstance(r.get(field), dict):
                missing_pairs.append(f"fold={r['fold']},seed={r['seed']}: {field}")
    gate_e = not missing_pairs
    gates["both_paired_summaries_present"] = {
        "passed": gate_e,
        "reason": (
            "both paired summaries present"
            if gate_e
            else f"missing: {missing_pairs[:5]}"
        ),
    }
    primary_metric = (
        str(eval_reports[0].get("primary_metric", "recall_at_k"))
        if eval_reports
        else "recall_at_k"
    )
    # F: Check ALL candidate metrics + both paired mean_deltas for cross-seed stability.
    _stab_metrics = sorted(_METRIC_NAMES)  # graded_ndcg_at_k, mrr, recall_at_k
    _stab_paired = [
        "paired_candidate_minus_global",
        "paired_candidate_minus_frozen_hybrid",
    ]
    unstable: List[str] = []
    for metric_name in _stab_metrics:
        _bfv: Dict[int, List[float]] = {}
        for r in eval_reports:
            v = r.get("metrics", {}).get("candidate", {}).get(metric_name)
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        _bfv.setdefault(int(r["fold"]), []).append(fv)
                except (TypeError, ValueError):
                    pass
        for fold, vals in sorted(_bfv.items()):
            if len(vals) < 2:
                continue
            mv = sum(vals) / len(vals)
            sv = (sum((v - mv) ** 2 for v in vals) / len(vals)) ** 0.5
            if sv > stability_threshold:
                unstable.append(
                    f"fold={fold} {metric_name}: std={sv:.5f} > {stability_threshold:.5f}"
                )
    for pf in _stab_paired:
        _bfd: Dict[int, List[float]] = {}
        for r in eval_reports:
            paired = r.get(pf)
            if isinstance(paired, dict):
                delta = paired.get("mean_delta")
                if delta is not None:
                    try:
                        fd = float(delta)
                        if math.isfinite(fd):
                            _bfd.setdefault(int(r["fold"]), []).append(fd)
                    except (TypeError, ValueError):
                        pass
        for fold, vals in sorted(_bfd.items()):
            if len(vals) < 2:
                continue
            mv = sum(vals) / len(vals)
            sv = (sum((v - mv) ** 2 for v in vals) / len(vals)) ** 0.5
            if sv > stability_threshold:
                unstable.append(
                    f"fold={fold} {pf}.mean_delta: std={sv:.5f} > {stability_threshold:.5f}"
                )
    gate_f = not unstable
    _checked = _stab_metrics + [f"{p}.mean_delta" for p in _stab_paired]
    gates["cross_seed_stability"] = {
        "passed": gate_f,
        "reason": (
            f"cross-seed stability within {stability_threshold} for all metrics and paired deltas"
            if gate_f
            else f"unstable metrics/deltas: {unstable[:5]}"
        ),
        "primary_metric": primary_metric,
        "stability_threshold": stability_threshold,
        "checked_metrics": _checked,
    }
    # E: Cross-candidate fold_query_sha256 alignment gate
    if global_fq_conflict_folds is not None:
        local_folds = {int(r["fold"]) for r in eval_reports}
        cross_conflicts = sorted(local_folds & global_fq_conflict_folds)
        gate_fqcc = not cross_conflicts
        gates["fold_query_sha256_cross_candidate_aligned"] = {
            "passed": gate_fqcc,
            "reason": (
                "fold_query_sha256 consistent across all candidates per fold"
                if gate_fqcc
                else f"fold_query_sha256 conflicts across candidates in folds: {cross_conflicts}"
            ),
        }
    # E: Cross-candidate primary_metric alignment gate
    if global_pm_consistent is not None:
        gate_pmcc = bool(global_pm_consistent)
        gates["primary_metric_cross_candidate_aligned"] = {
            "passed": gate_pmcc,
            "reason": (
                "primary_metric identical across all candidates"
                if gate_pmcc
                else "primary_metric differs across candidates"
            ),
        }
    bad: List[str] = []
    for r in training_reports:
        res = r.get("resources") or {}
        for field in _TRAIN_RESOURCE_FINITE_FIELDS:
            val = res.get(field)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))
                or float(val) < 0.0
            ):
                bad.append(f"fold={r['fold']},seed={r['seed']}: resources.{field}")
        mdl = r.get("model") or {}
        for field in _TRAIN_MODEL_INT_FIELDS:
            val = mdl.get(field)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                bad.append(f"fold={r['fold']},seed={r['seed']}: model.{field}")
        for field in _TRAIN_MODEL_FLOAT_FIELDS:
            val = mdl.get(field)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))
                or float(val) < 0.0
            ):
                bad.append(f"fold={r['fold']},seed={r['seed']}: model.{field}")
    gate_g = not bad
    gates["finite_resource_and_model_stats"] = {
        "passed": gate_g,
        "reason": (
            "all resource/model stats finite non-negative"
            if gate_g
            else f"invalid: {bad[:5]}"
        ),
    }
    computed_bundle = _compute_model_bundle_sha256(candidate_kind, training_reports)
    gate_h = computed_bundle == declared_model_bundle_sha256
    gates["model_bundle_hash_match"] = {
        "passed": gate_h,
        "reason": (
            "computed model bundle hash matches declaration"
            if gate_h
            else (
                f"bundle mismatch: computed={computed_bundle[:16]}... "
                f"declared={declared_model_bundle_sha256[:16]}..."
            )
        ),
    }
    all_passed = all(g["passed"] for g in gates.values())
    return {"passed": all_passed, "gates": gates}

# ---------------------------------------------------------------------------
# Protocol threshold validation
# ---------------------------------------------------------------------------

def _validate_protocol_thresholds(protocol: Dict[str, Any], label: str) -> None:
    min_gain = protocol.get("min_primary_gain_rel")
    if (
        isinstance(min_gain, bool)
        or not isinstance(min_gain, (int, float))
        or not math.isfinite(float(min_gain))
        or float(min_gain) < _MIN_PRIMARY_GAIN_REL
        or float(min_gain) > 1.0
    ):
        raise FullTrackSelectionError(
            f"{label} protocol.min_primary_gain_rel must be finite in "
            f"[{_MIN_PRIMARY_GAIN_REL}, 1.0]"
        )
    max_reg = protocol.get("max_scene_regression_abs")
    if (
        isinstance(max_reg, bool)
        or not isinstance(max_reg, (int, float))
        or not math.isfinite(float(max_reg))
        or float(max_reg) < 0.0
        or float(max_reg) > _MAX_SCENE_REGRESSION_ABS
    ):
        raise FullTrackSelectionError(
            f"{label} protocol.max_scene_regression_abs must be finite in "
            f"[0.0, {_MAX_SCENE_REGRESSION_ABS}]"
        )
    min_top5 = protocol.get("min_coherent_top5_frac")
    if (
        isinstance(min_top5, bool)
        or not isinstance(min_top5, (int, float))
        or not math.isfinite(float(min_top5))
        or float(min_top5) < _MIN_COHERENT_TOP5_FRAC
        or float(min_top5) > 1.0
    ):
        raise FullTrackSelectionError(
            f"{label} protocol.min_coherent_top5_frac must be finite in "
            f"[{_MIN_COHERENT_TOP5_FRAC}, 1.0]"
        )
    for field, expected in (
        ("zero_unrelated_in_top3", True),
        ("blinded_randomized_labels", True),
    ):
        if protocol.get(field) is not expected:
            raise FullTrackSelectionError(f"{label} protocol.{field} must be {expected}")
    min_raters = protocol.get("min_independent_raters")
    if (
        isinstance(min_raters, bool)
        or not isinstance(min_raters, int)
        or int(min_raters) < _MIN_INDEPENDENT_RATERS
    ):
        raise FullTrackSelectionError(
            f"{label} protocol.min_independent_raters must be int >= {_MIN_INDEPENDENT_RATERS}"
        )
    min_seeds_h = protocol.get("min_difficult_seeds")
    if (
        isinstance(min_seeds_h, bool)
        or not isinstance(min_seeds_h, int)
        or int(min_seeds_h) < _MIN_DIFFICULT_SEEDS
    ):
        raise FullTrackSelectionError(
            f"{label} protocol.min_difficult_seeds must be int >= {_MIN_DIFFICULT_SEEDS}"
        )
    min_cov = protocol.get("min_rater_seed_coverage")
    if (
        isinstance(min_cov, bool)
        or not isinstance(min_cov, (int, float))
        or not math.isfinite(float(min_cov))
        or float(min_cov) < _MIN_RATER_SEED_COVERAGE
        or float(min_cov) > 1.0
    ):
        raise FullTrackSelectionError(
            f"{label} protocol.min_rater_seed_coverage must be finite in "
            f"[{_MIN_RATER_SEED_COVERAGE}, 1.0]"
        )


# ---------------------------------------------------------------------------
# Human evidence loading
# ---------------------------------------------------------------------------

_HE_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "artifact_kind", "aggregate_kind",
    "bindings", "protocol", "rater_ids",
    "independent_rater_declarations", "difficult_seeds",
    "raw_ratings", "content_sha256",
})
_HE_BINDINGS_FIELDS: frozenset = frozenset({
    "selected_candidate", "model_bundle_sha256",
    "evaluation_bundle_sha256", "candidate_list_sha256",
    "evaluation_identity_sha256",
})
_HE_PROTOCOL_FIELDS: frozenset = frozenset({
    "min_primary_gain_rel", "max_scene_regression_abs",
    "min_coherent_top5_frac", "zero_unrelated_in_top3",
    "blinded_randomized_labels", "min_independent_raters",
    "min_difficult_seeds", "min_rater_seed_coverage",
})
_HE_RATING_FIELDS: frozenset = frozenset({
    "rater_id", "seed_id",
    "primary_gain_relative", "scene_regression_max_abs",
    "coherent_top5_frac", "unrelated_in_top3",
})
_SEED_REQUIRED_FIELDS: frozenset = frozenset(
    {"seed_id", "blinded_label", "presentation_order"}
)
_IRD_REQUIRED_FIELDS: frozenset = frozenset(
    {"is_independent", "not_self_authored", "affiliation_declared"}
)
_RECOGNIZED_AGGREGATE_KINDS: frozenset = frozenset({None, "blinded_human_ratings_analysis"})


def load_trusted_human_evidence(
    path: Union[str, Path], *, expected_bindings: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:
    """Load and strictly validate a trusted_fulltrack_human_evidence artifact.

    * aggregate_kind=blinded_human_ratings_analysis is recognized safely.
    * Other aggregate_kind values (aggregate_ratings.py outputs, etc.) are rejected.
    * Raw ratings are never copied or synthesized into report output.
    """
    label = f"human evidence {path}"
    data = _read_json_strict(path, label)
    # A: Detect tools/aggregate_ratings.py output schema BEFORE full HE field validation.
    if (
        frozenset(data.keys()) == _AGGREGATE_RATINGS_OUTPUT_KEYS
        and data.get("aggregate_kind") == "blinded_human_ratings_analysis"
    ):
        sc = data.get("session_count", 0)
        if isinstance(sc, bool) or not isinstance(sc, int) or sc <= 0:
            raise FullTrackSelectionError(_AGGREGATE_RATINGS_ZERO_REASON)
        raise FullTrackSelectionError(_AGGREGATE_RATINGS_BINDING_REASON)
    if frozenset(data.keys()) != _HE_REQUIRED_FIELDS:
        raise FullTrackSelectionError(f"{label} schema fields differ")
    if data.get("schema_version") != HUMAN_EVIDENCE_SCHEMA_VERSION:
        raise FullTrackSelectionError(
            f"{label} schema_version must be {HUMAN_EVIDENCE_SCHEMA_VERSION}"
        )
    if data.get("artifact_kind") != "trusted_fulltrack_human_evidence":
        raise FullTrackSelectionError(f"{label} artifact_kind mismatch")
    agg = data.get("aggregate_kind")
    if agg not in _RECOGNIZED_AGGREGATE_KINDS:
        raise FullTrackSelectionError(
            f"{label} aggregate_kind {agg!r} is not recognized; "
            "v16/v17 aggregate_ratings.py outputs and aggregates for other served lists are rejected"
        )
    payload = {k: v for k, v in data.items() if k != "content_sha256"}
    if data.get("content_sha256") != _canonical_sha256(payload):
        raise FullTrackSelectionError(f"{label} content_sha256 mismatch (tampered?)")
    bindings = data.get("bindings")
    if not isinstance(bindings, dict) or frozenset(bindings.keys()) != _HE_BINDINGS_FIELDS:
        raise FullTrackSelectionError(f"{label} bindings object has wrong fields")
    selected = bindings.get("selected_candidate")
    if not isinstance(selected, str) or not selected:
        raise FullTrackSelectionError(f"{label} bindings.selected_candidate invalid")
    for field in (
        "model_bundle_sha256",
        "evaluation_bundle_sha256",
        "candidate_list_sha256",
        "evaluation_identity_sha256",
    ):
        _validate_sha256_str(bindings.get(field), f"{label} bindings.{field}")
    if expected_bindings is not None:
        for key, expected_val in expected_bindings.items():
            if bindings.get(key) != expected_val:
                raise FullTrackSelectionError(
                    f"{label} binding {key!r} mismatch: expected {expected_val!r}, "
                    f"got {bindings.get(key)!r}"
                )
    protocol = data.get("protocol")
    if not isinstance(protocol, dict) or frozenset(protocol.keys()) != _HE_PROTOCOL_FIELDS:
        raise FullTrackSelectionError(f"{label} protocol has wrong fields")
    _validate_protocol_thresholds(protocol, label)
    rater_ids = data.get("rater_ids")
    if not isinstance(rater_ids, list) or len(rater_ids) == 0:
        raise FullTrackSelectionError(f"{label} rater_ids must be non-empty list")
    if any(not isinstance(r, str) or not r for r in rater_ids):
        raise FullTrackSelectionError(f"{label} rater_ids must be non-empty strings")
    if len(set(rater_ids)) != len(rater_ids):
        raise FullTrackSelectionError(f"{label} rater_ids must be unique")
    irds = data.get("independent_rater_declarations")
    if not isinstance(irds, dict) or set(irds.keys()) != set(rater_ids):
        raise FullTrackSelectionError(
            f"{label} independent_rater_declarations keys must match rater_ids"
        )
    for rater_id, decl in irds.items():
        if not isinstance(decl, dict) or frozenset(decl.keys()) != _IRD_REQUIRED_FIELDS:
            raise FullTrackSelectionError(
                f"{label} declaration for {rater_id!r} must have exact fields "
                f"{sorted(_IRD_REQUIRED_FIELDS)}"
            )
        for field in ("is_independent", "not_self_authored"):
            if decl.get(field) is not True:
                raise FullTrackSelectionError(
                    f"{label} rater {rater_id!r}: {field} must be true "
                    "(self-authored/non-independent rejected)"
                )
    seeds_list = data.get("difficult_seeds")
    if not isinstance(seeds_list, list):
        raise FullTrackSelectionError(f"{label} difficult_seeds must be list")
    seed_ids: List[str] = []
    all_presentation_orders: List[Tuple] = []
    for i, s in enumerate(seeds_list):
        if not isinstance(s, dict) or frozenset(s.keys()) != _SEED_REQUIRED_FIELDS:
            raise FullTrackSelectionError(
                f"{label} difficult_seeds[{i}] must have exact fields "
                f"{sorted(_SEED_REQUIRED_FIELDS)}"
            )
        sid = s.get("seed_id")
        if not isinstance(sid, str) or not sid:
            raise FullTrackSelectionError(f"{label} difficult_seeds[{i}].seed_id invalid")
        bl = s.get("blinded_label")
        if not isinstance(bl, str) or not bl:
            raise FullTrackSelectionError(
                f"{label} difficult_seeds[{i}].blinded_label invalid"
            )
        po = s.get("presentation_order")
        if not isinstance(po, list) or not po:
            raise FullTrackSelectionError(
                f"{label} difficult_seeds[{i}].presentation_order must be non-empty list"
            )
        for j, item in enumerate(po):
            if not isinstance(item, str) or not item:
                raise FullTrackSelectionError(
                    f"{label} difficult_seeds[{i}].presentation_order[{j}] "
                    "must be non-empty string"
                )
        if len(set(po)) != len(po):
            raise FullTrackSelectionError(
                f"{label} difficult_seeds[{i}].presentation_order items must be unique"
            )
        # G: blinded_label must appear in presentation_order
        if bl not in po:
            raise FullTrackSelectionError(
                f"{label} difficult_seeds[{i}].blinded_label {bl!r} "
                "not found in presentation_order"
            )
        seed_ids.append(sid)
        all_presentation_orders.append(tuple(po))
    if len(set(seed_ids)) != len(seed_ids):
        raise FullTrackSelectionError(f"{label} difficult_seeds must have unique seed_ids")
    if (
        protocol.get("blinded_randomized_labels") is True
        and len(seeds_list) > 1
        and len(set(all_presentation_orders)) < 2
    ):
        raise FullTrackSelectionError(
            f"{label} blinded_randomized_labels requires >= 2 distinct presentation_orders "
            "when multiple seeds are present"
        )
    ratings = data.get("raw_ratings")
    if not isinstance(ratings, list):
        raise FullTrackSelectionError(f"{label} raw_ratings must be list")
    seed_id_set = set(seed_ids)
    rater_set = set(rater_ids)
    seen_pairs: set = set()
    for i, rating in enumerate(ratings):
        if not isinstance(rating, dict) or frozenset(rating.keys()) != _HE_RATING_FIELDS:
            raise FullTrackSelectionError(f"{label} raw_ratings[{i}] has wrong fields")
        rid = rating.get("rater_id")
        if rid not in rater_set:
            raise FullTrackSelectionError(
                f"{label} raw_ratings[{i}].rater_id {rid!r} not in rater_ids"
            )
        sid = rating.get("seed_id")
        if sid not in seed_id_set:
            raise FullTrackSelectionError(
                f"{label} raw_ratings[{i}].seed_id {sid!r} not in difficult_seeds"
            )
        pair = (rid, sid)
        if pair in seen_pairs:
            raise FullTrackSelectionError(
                f"{label} duplicate rating rater={rid!r} seed={sid!r}"
            )
        seen_pairs.add(pair)
        pg_val = rating.get("primary_gain_relative")
        if (
            isinstance(pg_val, bool)
            or not isinstance(pg_val, (int, float))
            or not math.isfinite(float(pg_val))
            or float(pg_val) < -1.0
        ):
            raise FullTrackSelectionError(
                f"{label} raw_ratings[{i}].primary_gain_relative must be finite >= -1.0"
            )
        sr_val = rating.get("scene_regression_max_abs")
        if (
            isinstance(sr_val, bool)
            or not isinstance(sr_val, (int, float))
            or not math.isfinite(float(sr_val))
            or not (0.0 <= float(sr_val) <= 1.0)
        ):
            raise FullTrackSelectionError(
                f"{label} raw_ratings[{i}].scene_regression_max_abs must be in [0, 1]"
            )
        ct_val = rating.get("coherent_top5_frac")
        if (
            isinstance(ct_val, bool)
            or not isinstance(ct_val, (int, float))
            or not math.isfinite(float(ct_val))
            or not (0.0 <= float(ct_val) <= 1.0)
        ):
            raise FullTrackSelectionError(
                f"{label} raw_ratings[{i}].coherent_top5_frac must be in [0, 1]"
            )
        if not isinstance(rating.get("unrelated_in_top3"), bool):
            raise FullTrackSelectionError(
                f"{label} raw_ratings[{i}].unrelated_in_top3 must be bool"
            )
    return data

# ---------------------------------------------------------------------------
# Human evidence evaluation gates
# ---------------------------------------------------------------------------

def _evaluate_human_gates(
    evidence: Dict[str, Any],
    *,
    actual_model_bundle_sha256: str,
    actual_evaluation_bundle_sha256: str,
    candidate_list_sha256: str,
    evaluation_identity_sha256: str,
    candidate_kind: str,
) -> Dict[str, Any]:
    gates: Dict[str, Dict[str, Any]] = {}
    bindings = evidence["bindings"]
    protocol = evidence["protocol"]
    rater_ids: List[str] = evidence["rater_ids"]
    seeds_list: List[Dict[str, Any]] = evidence["difficult_seeds"]
    ratings: List[Dict[str, Any]] = evidence["raw_ratings"]
    for gate_name, binding_key, expected_val in (
        ("selected_candidate_match", "selected_candidate", candidate_kind),
        ("model_bundle_sha256_binding", "model_bundle_sha256", actual_model_bundle_sha256),
        (
            "evaluation_bundle_sha256_binding",
            "evaluation_bundle_sha256",
            actual_evaluation_bundle_sha256,
        ),
        ("candidate_list_sha256_binding", "candidate_list_sha256", candidate_list_sha256),
        (
            "evaluation_identity_sha256_binding",
            "evaluation_identity_sha256",
            evaluation_identity_sha256,
        ),
    ):
        actual = bindings.get(binding_key)
        ok = actual == expected_val
        gates[gate_name] = {
            "passed": ok,
            "reason": (
                f"{binding_key} binding verified"
                if ok
                else f"{binding_key} mismatch (stale or unrelated artifact)"
            ),
        }
    proto_min_raters = int(protocol["min_independent_raters"])
    proto_min_seeds = int(protocol["min_difficult_seeds"])
    proto_min_cov = float(protocol["min_rater_seed_coverage"])
    n_raters = len(rater_ids)
    n_seeds = len(seeds_list)
    gate_raters = n_raters >= proto_min_raters
    gates["min_independent_raters"] = {
        "passed": gate_raters,
        "reason": (
            f"{n_raters} raters >= {proto_min_raters}"
            if gate_raters
            else f"only {n_raters} raters, need >= {proto_min_raters}"
        ),
    }
    gate_seeds = n_seeds >= proto_min_seeds
    gates["min_difficult_seeds"] = {
        "passed": gate_seeds,
        "reason": (
            f"{n_seeds} seeds >= {proto_min_seeds}"
            if gate_seeds
            else f"only {n_seeds} seeds, need >= {proto_min_seeds}"
        ),
    }
    # Initialize ALL declared seeds; seeds with zero ratings fail the per-seed gate
    seed_id_order = [str(s["seed_id"]) for s in seeds_list]
    raters_per_seed: Dict[str, set] = {sid: set() for sid in seed_id_order}
    for rating in ratings:
        raters_per_seed[str(rating["seed_id"])].add(str(rating["rater_id"]))
    under = [sid for sid in seed_id_order if len(raters_per_seed[sid]) < proto_min_raters]
    gate_per_seed = not under
    gates["min_raters_per_seed"] = {
        "passed": gate_per_seed,
        "reason": (
            f"every seed has >= {proto_min_raters} raters"
            if gate_per_seed
            else (
                f"seeds with < {proto_min_raters} raters (including zero-rated): "
                f"{under[:5]}"
            )
        ),
    }
    n_rated = len(ratings)
    expected_pairs = n_raters * n_seeds
    agg_coverage = n_rated / expected_pairs if expected_pairs > 0 else 0.0
    gate_cov = agg_coverage >= proto_min_cov
    gates["rater_seed_coverage"] = {
        "passed": gate_cov,
        "reason": (
            f"aggregate coverage {agg_coverage:.4f} >= {proto_min_cov}"
            if gate_cov
            else f"aggregate coverage {agg_coverage:.4f} < {proto_min_cov}"
        ),
        "rated_pairs": n_rated,
        "expected_pairs": expected_pairs,
    }
    # Per-rater coverage: every counted independent rater must meet the threshold
    seeds_per_rater: Dict[str, set] = {rid: set() for rid in rater_ids}
    for rating in ratings:
        seeds_per_rater[str(rating["rater_id"])].add(str(rating["seed_id"]))
    rater_coverages = {
        rid: (len(ss) / n_seeds if n_seeds > 0 else 0.0)
        for rid, ss in seeds_per_rater.items()
    }
    n_under_raters = sum(1 for cov in rater_coverages.values() if cov < proto_min_cov)
    gate_per_rater = n_under_raters == 0
    gates["per_rater_coverage"] = {
        "passed": gate_per_rater,
        "reason": (
            f"all {n_raters} raters meet {proto_min_cov:.0%} seed coverage"
            if gate_per_rater
            else f"{n_under_raters} of {n_raters} raters below {proto_min_cov:.0%} per-rater coverage"
        ),
        "raters_below_threshold": n_under_raters,
        "raters_total": n_raters,
    }
    if ratings:
        pg = [float(r["primary_gain_relative"]) for r in ratings]
        sr = [float(r["scene_regression_max_abs"]) for r in ratings]
        ct = [float(r["coherent_top5_frac"]) for r in ratings]
        utr = [bool(r["unrelated_in_top3"]) for r in ratings]
        mean_pg = sum(pg) / len(pg)
        max_sr = max(sr)
        mean_ct = sum(ct) / len(ct)
        any_utr = any(utr)
    else:
        mean_pg, max_sr, mean_ct, any_utr = 0.0, float("inf"), 0.0, True
    min_gain = float(protocol["min_primary_gain_rel"])
    gate_pg = mean_pg >= min_gain
    gates["primary_gain_gate"] = {
        "passed": gate_pg,
        "reason": (
            f"mean primary_gain_relative {mean_pg:.4f} >= {min_gain}"
            if gate_pg
            else f"mean primary_gain_relative {mean_pg:.4f} < {min_gain}"
        ),
        "computed_from_raw_ratings": True,
    }
    max_reg = float(protocol["max_scene_regression_abs"])
    gate_sr = max_sr <= max_reg
    gates["scene_regression_gate"] = {
        "passed": gate_sr,
        "reason": (
            f"max scene_regression {max_sr:.4f} <= {max_reg}"
            if gate_sr
            else f"max scene_regression {max_sr:.4f} > {max_reg}"
        ),
        "computed_from_raw_ratings": True,
    }
    min_top5 = float(protocol["min_coherent_top5_frac"])
    gate_ct = mean_ct >= min_top5
    gates["coherent_top5_gate"] = {
        "passed": gate_ct,
        "reason": (
            f"mean coherent_top5_frac {mean_ct:.4f} >= {min_top5}"
            if gate_ct
            else f"mean coherent_top5_frac {mean_ct:.4f} < {min_top5}"
        ),
        "computed_from_raw_ratings": True,
    }
    gate_utr = not any_utr
    gates["zero_unrelated_in_top3"] = {
        "passed": gate_utr,
        "reason": (
            "zero unrelated scenes in ranks 1-3"
            if gate_utr
            else "unrelated scenes found in ranks 1-3"
        ),
        "computed_from_raw_ratings": True,
    }
    agg = evidence.get("aggregate_kind")
    if agg == "blinded_human_ratings_analysis":
        gates["aggregate_kind_note"] = {
            "passed": True,
            "reason": (
                "aggregate_kind=blinded_human_ratings_analysis recognized; "
                "authorization still requires all binding and raw-rating gates"
            ),
        }
    all_passed = all(g["passed"] for g in gates.values())
    failed = [name for name, g in gates.items() if not g["passed"]]
    reason = (
        f"human evidence gates failed: {failed}"
        if not all_passed
        else "all human evidence gates passed"
    )
    return {"authorized": all_passed, "reason": reason, "gate_details": gates}


# ---------------------------------------------------------------------------
# Selection report schema (for write validation)
# ---------------------------------------------------------------------------

_SELECTION_REPORT_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "artifact_kind", "candidate_list_sha256",
    "evaluation_identity", "evaluation_identity_sha256",
    "cross_seed_stability_threshold", "training_report_count",
    "evaluation_report_count", "candidates", "candidate_gate_details",
    "human_decision", "promotion_allowed", "notices", "report_sha256",
})

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_selection_report(
    training_reports: Sequence[Union[str, Path, Mapping[str, Any]]],
    evaluation_reports: Sequence[Union[str, Path, Mapping[str, Any]]],
    candidate_list_path: Union[str, Path, Mapping[str, Any]],
    *,
    trusted_ratings_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build a deterministic model-selection report.

    Both path-loaded and Mapping training/evaluation inputs undergo identical
    schema/type/range/checksum validation.  promotion_allowed is always False
    unless all automated gates pass AND human evidence gates pass.
    """
    if isinstance(candidate_list_path, Mapping):
        candidate_list = dict(candidate_list_path)
        _reject_nonfinite(candidate_list, "candidate list <Mapping>")
        candidate_list = _validate_candidate_list_dict(
            candidate_list, "candidate list <Mapping>"
        )
    else:
        candidate_list = _load_candidate_list(Path(candidate_list_path))
    candidate_list_sha256 = str(candidate_list["content_sha256"])
    evaluation_identity: Dict[str, Any] = dict(candidate_list["evaluation_identity"])
    evaluation_identity_sha256 = _canonical_sha256(evaluation_identity)
    stability_threshold = float(candidate_list["cross_seed_stability_threshold"])
    deciding_budget = int(candidate_list["deciding_budget"])
    primary_metric = str(candidate_list["primary_metric"])
    candidates_by_kind: Dict[str, Dict[str, Any]] = {
        c["candidate_kind"]: c for c in candidate_list["candidates"]
    }
    known_kinds = set(candidates_by_kind.keys())

    loaded_training: List[Dict[str, Any]] = []
    for report in training_reports:
        if isinstance(report, (str, Path)):
            loaded_training.append(_load_training_report(report))
        else:
            _lbl = "training report <Mapping>"
            _d = dict(report)
            _reject_nonfinite(_d, _lbl)
            loaded_training.append(_validate_training_report_dict(_d, _lbl))
    if not loaded_training:
        raise FullTrackSelectionError("at least one training report is required")
    from .fulltrack_train import (
        FullTrackTrainingError,
        TrainJobSpec,
        validate_training_report_bindings,
    )

    for r in loaded_training:
        if str(r["candidate_kind"]) not in known_kinds:
            raise FullTrackSelectionError(
                f"training report candidate_kind {r['candidate_kind']!r} not in candidate list"
            )
        candidate_kind = str(r["candidate_kind"])
        fold = int(r["fold"])
        seed = int(r["seed"])
        spec = TrainJobSpec(
            fold_index=fold,
            candidate_kind=candidate_kind,
            seed=seed,
            job_id=f"fold-{fold}__{candidate_kind}__seed-{seed}",
            relative_dir=f"fold-{fold}/{candidate_kind}/seed-{seed}",
        )
        store_binding = r["store_binding"]
        try:
            validate_training_report_bindings(
                r,
                spec=spec,
                source_fingerprint=str(r["source_fingerprint"]),
                store_binding_hash=str(r["store_binding_sha256"]),
                store_manifest_sha256=str(
                    store_binding["sealed_manifest_sha256"]
                ),
            )
        except FullTrackTrainingError as exc:
            raise FullTrackSelectionError(
                f"training report job binding invalid: {exc}"
            ) from exc

    loaded_eval: List[Dict[str, Any]] = []
    for report in evaluation_reports:
        if isinstance(report, (str, Path)):
            loaded_eval.append(
                _load_evaluation_report(
                    report,
                    expected_candidate_list_sha256=candidate_list_sha256,
                    expected_deciding_budget=deciding_budget,
                    expected_primary_metric=primary_metric,
                )
            )
        else:
            _lbl = "evaluation report <Mapping>"
            _d = dict(report)
            _reject_nonfinite(_d, _lbl)
            loaded_eval.append(
                _validate_evaluation_report_dict(
                    _d, _lbl,
                    expected_candidate_list_sha256=candidate_list_sha256,
                    expected_deciding_budget=deciding_budget,
                    expected_primary_metric=primary_metric,
                )
            )
    if not loaded_eval:
        raise FullTrackSelectionError("at least one evaluation report is required")
    for r in loaded_eval:
        if str(r["candidate_kind"]) not in known_kinds:
            raise FullTrackSelectionError(
                f"evaluation report candidate_kind {r['candidate_kind']!r} not in candidate list"
            )

    train_by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in loaded_training:
        train_by_kind.setdefault(str(r["candidate_kind"]), []).append(r)
    eval_by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in loaded_eval:
        eval_by_kind.setdefault(str(r["candidate_kind"]), []).append(r)

    # E: Compute global fold_query_sha256 conflicts and primary_metric consistency
    # across ALL evaluation reports from ALL candidates combined.
    _all_eval_flat: List[Dict[str, Any]] = [
        r for evlist in eval_by_kind.values() for r in evlist
    ]
    _global_fq_by_fold: Dict[int, set] = {}
    for _r in _all_eval_flat:
        _global_fq_by_fold.setdefault(int(_r["fold"]), set()).add(
            str(_r.get("fold_query_sha256", ""))
        )
    _global_fq_conflict_folds: set = {
        f for f, qs in _global_fq_by_fold.items() if len(qs) > 1
    }
    _all_pms = {str(_r.get("primary_metric", "")) for _r in _all_eval_flat}
    _global_pm_consistent: bool = len(_all_pms) == 1

    candidate_details: Dict[str, Any] = {}
    for ck, cinfo in candidates_by_kind.items():
        declared_bundle = str(cinfo["model_bundle_sha256"])
        tr = train_by_kind.get(ck, [])
        ev = eval_by_kind.get(ck, [])
        if not tr:
            candidate_details[ck] = {
                "passed": False,
                "reason": f"no training reports for {ck!r}",
                "declared_model_bundle_sha256": declared_bundle,
            }
            continue
        if not ev:
            candidate_details[ck] = {
                "passed": False,
                "reason": f"no evaluation reports for {ck!r}",
                "declared_model_bundle_sha256": declared_bundle,
            }
            continue
        gate_result = _run_automated_gates(
            ck, tr, ev,
            stability_threshold=stability_threshold,
            declared_model_bundle_sha256=declared_bundle,
            evaluation_identity=evaluation_identity,
            evaluation_identity_sha256=evaluation_identity_sha256,
            global_fq_conflict_folds=_global_fq_conflict_folds,
            global_pm_consistent=_global_pm_consistent,
        )
        computed_model = _compute_model_bundle_sha256(ck, tr)
        computed_eval = _compute_evaluation_bundle_sha256(ck, ev)
        candidate_details[ck] = {
            **gate_result,
            "declared_model_bundle_sha256": declared_bundle,
            "computed_model_bundle_sha256": computed_model,
            "computed_evaluation_bundle_sha256": computed_eval,
            "evaluation_identity_sha256": evaluation_identity_sha256,
        }

    human_decision: Dict[str, Any] = {
        "provided": trusted_ratings_path is not None,
        "promotion_allowed": False,
        "reason": AUTOMATED_PROMOTION_PROHIBITED_NOTICE,
        "reason_code": REASON_CODE_NOT_SUPPLIED,
    }
    if trusted_ratings_path is not None:
        try:
            evidence = load_trusted_human_evidence(trusted_ratings_path)
            sel_cand = str(evidence["bindings"]["selected_candidate"])
            cdet = candidate_details.get(sel_cand)
            if cdet is None:
                human_decision = {
                    "provided": True,
                    "promotion_allowed": False,
                    "reason": (
                        f"human evidence selected_candidate {sel_cand!r} not in candidate list"
                    ),
                    "reason_code": REASON_CODE_REJECTED,
                }
            elif not cdet.get("passed", False):
                human_decision = {
                    "provided": True,
                    "promotion_allowed": False,
                    "reason": (
                        f"automated gates failed for {sel_cand!r}; "
                        "human review insufficient"
                    ),
                    "reason_code": REASON_CODE_AUTOMATED_GATES_FAILED,
                }
            else:
                he = _evaluate_human_gates(
                    evidence,
                    actual_model_bundle_sha256=cdet["computed_model_bundle_sha256"],
                    actual_evaluation_bundle_sha256=cdet[
                        "computed_evaluation_bundle_sha256"
                    ],
                    candidate_list_sha256=candidate_list_sha256,
                    evaluation_identity_sha256=evaluation_identity_sha256,
                    candidate_kind=sel_cand,
                )
                human_decision = {
                    "provided": True,
                    "promotion_allowed": bool(he["authorized"]),
                    "reason": he["reason"],
                    "reason_code": (
                        REASON_CODE_ACCEPTED if he["authorized"] else REASON_CODE_REJECTED
                    ),
                    "selected_candidate": sel_cand,
                    "human_gate_details": he["gate_details"],
                    "aggregate_kind": evidence.get("aggregate_kind"),
                    "jamendo_tag_notice": JAMENDO_TAG_DESCRIPTIVE_NOTICE,
                }
        except FullTrackSelectionError as exc:
            human_decision = {
                "provided": True,
                "promotion_allowed": False,
                "reason": f"human evidence rejected: {exc}",
                "reason_code": REASON_CODE_REJECTED,
            }

    report_payload: Dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_selection_report",
        "candidate_list_sha256": candidate_list_sha256,
        "evaluation_identity": evaluation_identity,
        "evaluation_identity_sha256": evaluation_identity_sha256,
        "cross_seed_stability_threshold": stability_threshold,
        "training_report_count": len(loaded_training),
        "evaluation_report_count": len(loaded_eval),
        "candidates": sorted(candidates_by_kind.keys()),
        "candidate_gate_details": candidate_details,
        "human_decision": human_decision,
        "promotion_allowed": bool(human_decision.get("promotion_allowed", False)),
        "notices": [JAMENDO_TAG_DESCRIPTIVE_NOTICE, AUTOMATED_PROMOTION_PROHIBITED_NOTICE],
    }
    report_payload["report_sha256"] = _canonical_sha256(
        {k: v for k, v in report_payload.items() if k != "report_sha256"}
    )
    return report_payload


def build_selection_report_from_manifest(
    training_root: Union[str, Path],
    manifest_path: Union[str, Path],
    *,
    trusted_ratings_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build a selection report from a verified benchmark selection manifest."""
    training_reports, evaluation_reports, candidate_list = (
        _resolve_selection_manifest(training_root, manifest_path)
    )
    return build_selection_report(
        training_reports,
        evaluation_reports,
        candidate_list,
        trusted_ratings_path=trusted_ratings_path,
    )


def _write_selection_report_document(
    path: Union[str, Path],
    report: Mapping[str, Any],
    *,
    allow_promoted: bool,
) -> str:
    d = dict(report)
    if frozenset(d.keys()) != _SELECTION_REPORT_REQUIRED_FIELDS:
        missing = _SELECTION_REPORT_REQUIRED_FIELDS - frozenset(d.keys())
        extra = frozenset(d.keys()) - _SELECTION_REPORT_REQUIRED_FIELDS
        raise FullTrackSelectionError(
            f"selection report has wrong fields "
            f"(missing={sorted(missing)!r}, extra={sorted(extra)!r})"
        )
    if d.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise FullTrackSelectionError("selection report schema_version mismatch")
    if d.get("artifact_kind") != "fulltrack_selection_report":
        raise FullTrackSelectionError(
            "selection report artifact_kind must be 'fulltrack_selection_report'"
        )
    if not isinstance(d.get("promotion_allowed"), bool):
        raise FullTrackSelectionError("selection report promotion_allowed must be bool")
    if d["promotion_allowed"] and not allow_promoted:
        raise FullTrackSelectionError(
            "promoted reports must be rebuilt from source evidence while writing"
        )
    hd = d.get("human_decision")
    if not isinstance(hd, dict):
        raise FullTrackSelectionError("selection report human_decision must be object")
    if bool(d["promotion_allowed"]) != bool(hd.get("promotion_allowed", False)):
        raise FullTrackSelectionError(
            "selection report promotion_allowed inconsistent with "
            "human_decision.promotion_allowed"
        )
    if d["promotion_allowed"]:
        selected = hd.get("selected_candidate")
        candidate_details = d.get("candidate_gate_details")
        human_gates = hd.get("human_gate_details")
        if (
            hd.get("provided") is not True
            or hd.get("reason_code") != REASON_CODE_ACCEPTED
            or not isinstance(selected, str)
            or selected not in d.get("candidates", [])
            or not isinstance(candidate_details, dict)
            or not isinstance(candidate_details.get(selected), dict)
            or candidate_details[selected].get("passed") is not True
            or not isinstance(candidate_details[selected].get("gates"), dict)
            or not candidate_details[selected]["gates"]
            or any(
                not isinstance(gate, dict) or gate.get("passed") is not True
                for gate in candidate_details[selected]["gates"].values()
            )
            or not isinstance(human_gates, dict)
            or not human_gates
            or any(
                not isinstance(gate, dict) or gate.get("passed") is not True
                for gate in human_gates.values()
            )
        ):
            raise FullTrackSelectionError(
                "promoted selection report lacks verified automated and human gates"
            )
    _reject_nonfinite(d, "selection report")
    payload = {k: v for k, v in d.items() if k != "report_sha256"}
    if d.get("report_sha256") != _canonical_sha256(payload):
        raise FullTrackSelectionError(
            "selection report report_sha256 mismatch (tampered?)"
        )
    return _atomic_write_canonical_document(path, d, "selection report")


def write_selection_report(path: Union[str, Path], report: Mapping[str, Any]) -> str:
    """Atomically write a non-promoted report.

    Promotion can only be persisted through a source-aware build-and-write API,
    which revalidates the original training, evaluation, candidate-list, and
    trusted-rating artifacts immediately before writing.
    """
    return _write_selection_report_document(path, report, allow_promoted=False)


def build_and_write_selection_report(
    path: Union[str, Path],
    training_reports: Sequence[Union[str, Path]],
    evaluation_reports: Sequence[Union[str, Path]],
    candidate_list_path: Union[str, Path],
    *,
    trusted_ratings_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], str]:
    if (
        not training_reports
        or not evaluation_reports
        or any(not isinstance(item, (str, Path)) for item in training_reports)
        or any(not isinstance(item, (str, Path)) for item in evaluation_reports)
        or not isinstance(candidate_list_path, (str, Path))
        or (
            trusted_ratings_path is not None
            and not isinstance(trusted_ratings_path, (str, Path))
        )
    ):
        raise FullTrackSelectionError(
            "source-aware promoted writes require filesystem paths for every source artifact"
        )
    report = build_selection_report(
        training_reports,
        evaluation_reports,
        candidate_list_path,
        trusted_ratings_path=trusted_ratings_path,
    )
    file_sha = _write_selection_report_document(
        path, report, allow_promoted=True
    )
    return report, file_sha


def build_and_write_selection_report_from_manifest(
    path: Union[str, Path],
    training_root: Union[str, Path],
    manifest_path: Union[str, Path],
    *,
    trusted_ratings_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], str]:
    report = build_selection_report_from_manifest(
        training_root,
        manifest_path,
        trusted_ratings_path=trusted_ratings_path,
    )
    file_sha = _write_selection_report_document(
        path, report, allow_promoted=True
    )
    return report, file_sha


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m soundalike.ml.fulltrack_selection"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("report", help="build a model-selection report")
    rp.add_argument(
        "--training-report", type=Path, action="append",
        dest="training_reports", required=True, metavar="PATH",
    )
    rp.add_argument(
        "--evaluation-report", type=Path, action="append",
        dest="evaluation_reports", required=True, metavar="PATH",
    )
    rp.add_argument("--candidate-list", type=Path, required=True, metavar="PATH")
    rp.add_argument("--trusted-ratings", type=Path, default=None, metavar="PATH")
    rp.add_argument("--output", type=Path, required=True, metavar="PATH")
    mp = sub.add_parser(
        "report-from-manifest",
        help="build a model-selection report from benchmark-all selector inputs",
    )
    mp.add_argument("--training-root", type=Path, required=True, metavar="PATH")
    mp.add_argument("--manifest", type=Path, required=True, metavar="PATH")
    mp.add_argument("--trusted-ratings", type=Path, default=None, metavar="PATH")
    mp.add_argument("--output", type=Path, required=True, metavar="PATH")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        try:
            report, file_sha = build_and_write_selection_report(
                args.output,
                args.training_reports,
                args.evaluation_reports,
                args.candidate_list,
                trusted_ratings_path=args.trusted_ratings,
            )
            print(json.dumps({
                "output": str(args.output),
                "report_sha256": report.get("report_sha256"),
                "file_sha256": file_sha,
                "promotion_allowed": report.get("promotion_allowed"),
            }, indent=2))
        except FullTrackSelectionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "report-from-manifest":
        try:
            report, file_sha = build_and_write_selection_report_from_manifest(
                args.output,
                args.training_root,
                args.manifest,
                trusted_ratings_path=args.trusted_ratings,
            )
            print(json.dumps({
                "output": str(args.output),
                "report_sha256": report.get("report_sha256"),
                "file_sha256": file_sha,
                "promotion_allowed": report.get("promotion_allowed"),
            }, indent=2))
        except FullTrackSelectionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()