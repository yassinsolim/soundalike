"""Pinned MusicFM-FMA canary extraction for the lawful full-track corpus."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np

from .fulltrack_extract import (
    ExtractionConfig,
    FullTrackExtractionError,
    PyAVAudioDecoder,
    _offline_model_environment,
    extract_context,
    normalize_rows,
)
from .fulltrack_store import FullTrackStore, stable_json_sha256
from .jamendo_fulltrack import (
    JamendoValidationError,
    load_jamendo_context,
    sha256_file,
)


MODEL_ID = "musicfm_fma_b83ebed_layer7"
SOURCE_COMMIT = "b83ebedb401bcef639b26b05c0c8bee1dc2dfe71"
MODEL_REVISION = "4513b38bc25ad1d227b1980819b9691ba97f4d87"
CHECKPOINT_SHA256 = "68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96"
CHECKPOINT_BYTES = 1_316_802_154
STATS_SHA256 = "5416e468018bae68c6231d4cbb2b11f0d11c04e6437881505ae427a3f8344904"
CONFORMER_CONFIG_SHA256 = (
    "7a63cb5706c9a37483f1973a3c226d54eb504ce15cf62cb52637019540c8a75d"
)
SOURCE_LICENSE_SHA256 = (
    "0f0b1e4988266f51fbaa325e80979d37c8cf3f5723efb82fcfa19feef096176a"
)
DEFAULT_ASSET_ROOT = Path(r"C:\soundalike-data\model-assets\musicfm-fma")
EMBEDDING_DIM = 1024
SAMPLE_RATE = 24_000
WINDOW_SECONDS = 30.0
HOP_SECONDS = 15.0
LAYER_INDEX = 7
MAX_BATCH_SIZE = 2


@dataclass(frozen=True)
class MusicFMAssets:
    root: Path
    source_root: Path
    checkpoint: Path
    stats: Path
    conformer_config: Path

    @classmethod
    def from_root(cls, root: Path) -> "MusicFMAssets":
        resolved = Path(root).resolve()
        return cls(
            root=resolved,
            source_root=resolved / "musicfm",
            checkpoint=resolved / "pretrained_fma.pt",
            stats=resolved / "fma_stats.json",
            conformer_config=resolved / "wav2vec2-conformer-config.json",
        )


@dataclass(frozen=True)
class MusicFMCapability:
    available: bool
    model_id: str
    asset_root: str
    source_commit: Optional[str]
    checkpoint_sha256: Optional[str]
    checkpoint_bytes: Optional[int]
    stats_sha256: Optional[str]
    conformer_config_sha256: Optional[str]
    source_license_sha256: Optional[str]
    package_versions: Dict[str, str]
    cuda_device: Optional[str]
    license: str
    reasons: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def binding(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "source_commit": self.source_commit,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "stats_sha256": self.stats_sha256,
            "conformer_config_sha256": self.conformer_config_sha256,
            "source_license_sha256": self.source_license_sha256,
            "package_versions": dict(sorted(self.package_versions.items())),
            "layer_index": LAYER_INDEX,
            "embedding_dim": EMBEDDING_DIM,
            "sample_rate": SAMPLE_RATE,
            "license": self.license,
        }


def _git_source_state(source_root: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if head.returncode or status.returncode:
        return None, None
    return head.stdout.strip().lower(), status.stdout.strip()


def inspect_capability(asset_root: Path = DEFAULT_ASSET_ROOT) -> MusicFMCapability:
    assets = MusicFMAssets.from_root(asset_root)
    reasons = []
    versions: Dict[str, str] = {}
    cuda_device: Optional[str] = None
    source_commit, source_status = _git_source_state(assets.source_root)
    if source_commit != SOURCE_COMMIT:
        reasons.append("MusicFM source checkout is missing or not at the approved commit")
    if source_status:
        reasons.append("MusicFM source checkout has tracked modifications")

    def verify(path: Path, expected: str, label: str) -> Optional[str]:
        if not path.is_file() or path.is_symlink():
            reasons.append(f"{label} is missing or not a concrete file")
            return None
        actual = sha256_file(path)
        if actual != expected:
            reasons.append(f"{label} SHA-256 is not approved")
        return actual

    checkpoint_hash = verify(assets.checkpoint, CHECKPOINT_SHA256, "FMA checkpoint")
    checkpoint_bytes = assets.checkpoint.stat().st_size if checkpoint_hash is not None else None
    if checkpoint_bytes is not None and checkpoint_bytes != CHECKPOINT_BYTES:
        reasons.append("FMA checkpoint size is not approved")
    stats_hash = verify(assets.stats, STATS_SHA256, "FMA normalization stats")
    config_hash = verify(
        assets.conformer_config,
        CONFORMER_CONFIG_SHA256,
        "Wav2Vec2 conformer config",
    )
    license_hash = verify(
        assets.source_root / "LICENSE",
        SOURCE_LICENSE_SHA256,
        "MusicFM source license",
    )

    for distribution in ("torch", "torchaudio", "transformers", "einops"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            reasons.append(f"{distribution} is not installed")
    try:
        import torch
    except ImportError:
        pass
    else:
        if not torch.cuda.is_available():
            reasons.append("CUDA is unavailable")
        else:
            cuda_device = torch.cuda.get_device_name(0)
    return MusicFMCapability(
        available=not reasons,
        model_id=MODEL_ID,
        asset_root=str(assets.root),
        source_commit=source_commit,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_bytes=checkpoint_bytes,
        stats_sha256=stats_hash,
        conformer_config_sha256=config_hash,
        source_license_sha256=license_hash,
        package_versions=versions,
        cuda_device=cuda_device,
        license="MIT model metadata; MusicFM source MIT/Apache-2.0",
        reasons=tuple(reasons),
    )


@contextmanager
def _musicfm_source_import(source_root: Path) -> Iterator[None]:
    parent = str(source_root.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def _pool_hidden_states(hidden_states) -> np.ndarray:
    values = hidden_states.float()
    if values.ndim != 3 or values.shape[2] != EMBEDDING_DIM:
        raise FullTrackExtractionError(
            f"MusicFM returned unexpected hidden-state shape {tuple(values.shape)}"
        )
    return normalize_rows(values.mean(dim=1).cpu().numpy())


class FrozenMusicFMAdapter:
    """Frozen layer-7 MusicFM-FMA adapter with local-only, hash-pinned assets."""

    def __init__(self, asset_root: Path = DEFAULT_ASSET_ROOT) -> None:
        capability = inspect_capability(asset_root)
        if not capability.available:
            raise FullTrackExtractionError(
                "MusicFM capability gate failed: " + "; ".join(capability.reasons)
            )
        assets = MusicFMAssets.from_root(asset_root)
        with _offline_model_environment(), _musicfm_source_import(assets.source_root):
            import torch
            from transformers import Wav2Vec2ConformerConfig

            try:
                local_config = Wav2Vec2ConformerConfig.from_json_file(
                    str(assets.conformer_config)
                )
                module = importlib.import_module("musicfm.model.musicfm_25hz")
                with patch.object(
                    Wav2Vec2ConformerConfig,
                    "from_pretrained",
                    return_value=local_config,
                ):
                    model = module.MusicFM25Hz(
                        is_flash=False,
                        stat_path=str(assets.stats),
                        model_path=None,
                    )
                payload = torch.load(
                    str(assets.checkpoint),
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise FullTrackExtractionError(
                    f"MusicFM model loading failed: {exc}"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
                raise FullTrackExtractionError("MusicFM checkpoint has no state_dict")
            state = payload["state_dict"]
            if not state or any(
                not isinstance(key, str) or not key.startswith("model.") for key in state
            ):
                raise FullTrackExtractionError("MusicFM checkpoint state keys are invalid")
            try:
                model.load_state_dict(
                    {key[6:]: value for key, value in state.items()}, strict=True
                )
            except RuntimeError as exc:
                raise FullTrackExtractionError(
                    f"MusicFM checkpoint is incompatible: {exc}"
                ) from exc
        self._torch = torch
        try:
            self._model = model.cuda().eval()
        except RuntimeError as exc:
            raise FullTrackExtractionError(
                f"MusicFM CUDA initialization failed: {exc}"
            ) from exc
        self._capability = capability
        self._binding = capability.binding()

    @property
    def model_id(self) -> str:
        return MODEL_ID

    @property
    def checkpoint_sha256(self) -> str:
        return CHECKPOINT_SHA256

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def max_batch_size(self) -> int:
        return MAX_BATCH_SIZE

    @property
    def binding(self) -> Dict[str, object]:
        return dict(self._binding)

    @property
    def capability(self) -> MusicFMCapability:
        return self._capability

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        waveforms = np.asarray(windows, dtype=np.float32)
        expected_samples = int(SAMPLE_RATE * WINDOW_SECONDS)
        if waveforms.ndim != 2 or waveforms.shape[1] != expected_samples:
            raise FullTrackExtractionError(
                "MusicFM requires exact 30-second/24 kHz windows"
            )
        if not 0 < len(waveforms) <= self.max_batch_size:
            raise FullTrackExtractionError("MusicFM batch exceeds the capability bound")
        tensor = self._torch.from_numpy(np.ascontiguousarray(waveforms)).cuda()
        with self._torch.inference_mode():
            hidden_states = self._model.get_latent(tensor, layer_ix=LAYER_INDEX)
        return _pool_hidden_states(hidden_states)


def _extraction_binding(
    config: ExtractionConfig, capability: MusicFMCapability
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "musicfm_fma_frozen_representation_canary",
        "extraction": config.as_dict(),
        "model": capability.binding(),
        "promotion_allowed": False,
    }


def _capability_command(args: argparse.Namespace) -> int:
    capability = inspect_capability(Path(args.assets))
    output = capability.as_dict()
    try:
        PyAVAudioDecoder()
    except FullTrackExtractionError as exc:
        output["pyav_error"] = str(exc)
    else:
        output["pyav_error"] = None
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if capability.available and output["pyav_error"] is None else 2


def _extract_command(args: argparse.Namespace) -> int:
    context = load_jamendo_context(
        Path(args.metadata_root),
        Path(args.audio_root),
        Path(args.state_root),
        production=True,
    )
    config = ExtractionConfig(
        sample_rate=SAMPLE_RATE,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=HOP_SECONDS,
        model_batch_size=args.batch_size,
        repetition_sections=args.repetition_sections,
        salient_sections=args.salient_sections,
        shard_tracks=args.shard_tracks,
    )
    config.validate()
    encoder = FrozenMusicFMAdapter(Path(args.assets))
    binding = _extraction_binding(config, encoder.capability)
    with FullTrackStore(
        Path(args.output),
        track_ids=[track.track_id for track in context.tracks],
        source_hashes=[track.expected_audio_sha256 for track in context.tracks],
        source_fingerprint=context.source_fingerprint,
        config_sha256=stable_json_sha256(binding),
        model_sha256=encoder.checkpoint_sha256,
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        shard_tracks=config.shard_tracks,
        repetition_sections=config.repetition_sections,
        salient_sections=config.salient_sections,
    ) as store:
        summary = extract_context(
            context,
            store,
            decoder=PyAVAudioDecoder(),
            encoder=encoder,
            config=config,
            max_tracks=args.max_tracks,
        )
    print(
        json.dumps(
            {"binding": binding, "summary": asdict(summary)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capability = subparsers.add_parser(
        "capability", help="verify all local MusicFM-FMA assets without loading weights"
    )
    capability.add_argument("--assets", default=str(DEFAULT_ASSET_ROOT))
    capability.set_defaults(handler=_capability_command)
    extract = subparsers.add_parser(
        "extract", help="run isolated MusicFM-FMA full-track extraction"
    )
    extract.add_argument("--metadata-root", required=True)
    extract.add_argument("--audio-root", required=True)
    extract.add_argument("--state-root", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--assets", default=str(DEFAULT_ASSET_ROOT))
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--repetition-sections", type=int, default=32)
    extract.add_argument("--salient-sections", type=int, default=32)
    extract.add_argument("--shard-tracks", type=int, default=64)
    extract.add_argument("--max-tracks", type=int)
    extract.set_defaults(handler=_extract_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FullTrackExtractionError, JamendoValidationError) as exc:
        raise SystemExit(f"MusicFM canary blocked: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
