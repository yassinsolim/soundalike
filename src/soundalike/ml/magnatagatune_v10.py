"""Human audio calibration with MagnaTagATune relative-similarity votes.

This module deliberately keeps the MagnaTagATune CSVs, audio and derived
embeddings under ``ml_data/``.  The repository stores only code, hashes and
aggregate results: the audio is copyrighted music and is never re-hosted.

``comparisons_final.csv`` contains triadic odd-one-out votes.  A vote in
``clipN_numvotes`` means clip N was judged the odd item, so the other two clips
form the human-similar pair.  Tied winners are ambiguous and are never converted
to a constraint.  The primary metric predicts the odd clip by finding the
closest of the three representation-space pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.request import urlopen

import numpy as np


MTAT_BASE_URL = "https://mirg.city.ac.uk/datasets/magnatagatune"
MTAT_CSV_HASHES = {
    "comparisons_final.csv":
        "cf210e087ed5b3f3f8b164626e1d2857cf0ba9ae66bd9229bafe042889107a98",
    "clip_info_final.csv":
        "cb6108a10d3a91f0bfd7d2fbec2382559d15f20c8d28093f14e12162a47a3e78",
}
MTAT_AUDIO_PART_HASHES = {
    "mp3.zip.001":
        "f857fe185968773058cc71662c2ef5d8f2d4b7338e3c122cfd52f82dcb9760b9",
    "mp3.zip.002":
        "fc2e1ec441755556ed1398b1808f1b08b6034372f8bc27394510c0c58cdb52ce",
    "mp3.zip.003":
        "83a689824c17e82f6eb81cdbc4e4ca239a4cfc1fb41f1a5c80b861caec90450f",
}
COMPARISON_COLUMNS = (
    "clip1_id", "clip2_id", "clip3_id",
    "clip1_numvotes", "clip2_numvotes", "clip3_numvotes",
    "clip1_mp3_path", "clip2_mp3_path", "clip3_mp3_path",
)
CLIP_COLUMNS = (
    "clip_id", "track_number", "title", "artist", "album", "url",
    "segmentStart", "segmentEnd", "original_url", "mp3_path",
)
SPLIT_SEED = 20260712
MIN_TOTAL_VOTES = 3
LOUVAIN_RESOLUTION = 1.0
SPLIT_RATIOS = (0.60, 0.20, 0.20)
SPLIT_NAMES = ("train", "dev", "test")
MATERIAL_WIN = 0.05


class MTATError(RuntimeError):
    """Raised when real MTAT inputs or test-open discipline are invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def download_metadata(root: str | Path) -> Dict[str, Any]:
    """Download the two public CSVs and verify their established SHA-256s."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Any] = {}
    for name, expected in MTAT_CSV_HASHES.items():
        path = directory / name
        if not path.exists():
            with urlopen(f"{MTAT_BASE_URL}/{name}", timeout=120) as response:
                path.write_bytes(response.read())
        actual = sha256_path(path)
        if actual != expected:
            raise MTATError(f"{name} SHA-256 mismatch: {actual}")
        files[name] = {
            "url": f"{MTAT_BASE_URL}/{name}",
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return files


def _read_tsv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise MTATError(
                f"{path.name} schema mismatch: {reader.fieldnames!r}"
            )
        rows = list(reader)
    if not rows:
        raise MTATError(f"{path.name} is empty")
    return rows


@dataclass(frozen=True)
class HumanConstraint:
    source_row: int
    clip_ids: tuple[int, int, int]
    similar_clip_ids: tuple[int, int]
    odd_clip_id: int
    votes: tuple[int, int, int]
    total_votes: int
    winner_votes: int
    runner_up_votes: int
    confidence: float
    winner_share: float
    artists: tuple[str, str, str]
    split: str = ""


def load_inputs(
    root: str | Path,
) -> tuple[list[dict[str, str]], dict[int, dict[str, str]], Dict[str, Any]]:
    directory = Path(root)
    files = download_metadata(directory)
    comparisons = _read_tsv(directory / "comparisons_final.csv", COMPARISON_COLUMNS)
    clip_rows = _read_tsv(directory / "clip_info_final.csv", CLIP_COLUMNS)
    clips = {int(row["clip_id"]): row for row in clip_rows}
    if len(clips) != len(clip_rows):
        raise MTATError("clip_info_final.csv contains duplicate clip IDs")
    referenced = {
        int(row[f"clip{slot}_id"])
        for row in comparisons
        for slot in (1, 2, 3)
    }
    missing = sorted(referenced - set(clips))
    if missing:
        raise MTATError(f"{len(missing)} comparison clip IDs lack metadata")
    audit = {
        "files": files,
        "comparison_rows": len(comparisons),
        "clip_rows": len(clip_rows),
        "unique_compared_clips": len(referenced),
        "unique_artists_all_metadata": len({row["artist"] for row in clip_rows}),
        "unique_artists_compared": len({clips[item]["artist"] for item in referenced}),
    }
    return comparisons, clips, audit


def parse_constraints(
    comparisons: Sequence[Mapping[str, str]],
    clips: Mapping[int, Mapping[str, str]],
    min_total_votes: int = MIN_TOTAL_VOTES,
) -> tuple[list[HumanConstraint], Dict[str, int]]:
    """Convert unique odd-one-out vote winners into triadic constraints."""
    accepted: list[HumanConstraint] = []
    rejected = Counter()
    for source_row, row in enumerate(comparisons, start=1):
        clip_ids = tuple(int(row[f"clip{slot}_id"]) for slot in (1, 2, 3))
        votes = tuple(int(row[f"clip{slot}_numvotes"]) for slot in (1, 2, 3))
        total = sum(votes)
        top = max(votes)
        if total < min_total_votes:
            rejected["too_few_votes"] += 1
            continue
        if votes.count(top) != 1:
            rejected["tied_winner"] += 1
            continue
        odd_index = votes.index(top)
        similar = tuple(
            clip_ids[index] for index in range(3) if index != odd_index
        )
        runner_up = sorted(votes, reverse=True)[1]
        artists = tuple(str(clips[item]["artist"]).strip() for item in clip_ids)
        accepted.append(HumanConstraint(
            source_row=source_row,
            clip_ids=clip_ids,
            similar_clip_ids=(similar[0], similar[1]),
            odd_clip_id=clip_ids[odd_index],
            votes=votes,
            total_votes=total,
            winner_votes=top,
            runner_up_votes=runner_up,
            confidence=float((top - runner_up) / total),
            winner_share=float(top / total),
            artists=artists,
        ))
    rejected["accepted"] = len(accepted)
    return accepted, dict(rejected)


def artist_disjoint_split(
    constraints: Sequence[HumanConstraint],
    seed: int = SPLIT_SEED,
) -> tuple[list[HumanConstraint], Dict[str, Any]]:
    """Partition artist communities and discard cross-community constraints.

    Artist co-occurrence forms one giant connected component, so connected
    components cannot create a useful held-out split.  We use deterministic
    weighted Louvain communities, assign whole communities to train/dev/test,
    and retain only constraints wholly inside one community.  No artist can
    therefore occur in more than one retained split.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise MTATError("networkx is required for the artist-disjoint split") from exc

    graph = nx.Graph()
    for constraint in constraints:
        artists = tuple(artist.casefold() for artist in constraint.artists)
        graph.add_nodes_from(artists)
        for left, right in itertools.combinations(artists, 2):
            if left == right:
                continue
            prior = graph.get_edge_data(left, right, {}).get("weight", 0)
            graph.add_edge(left, right, weight=prior + 1)
    communities = list(nx.community.louvain_communities(
        graph, weight="weight", resolution=LOUVAIN_RESOLUTION, seed=seed
    ))
    communities.sort(key=lambda values: (min(values), len(values)))
    community_of = {
        artist: index
        for index, values in enumerate(communities)
        for artist in values
    }
    internal: list[tuple[HumanConstraint, int]] = []
    cross = 0
    for constraint in constraints:
        memberships = {
            community_of[artist.casefold()] for artist in constraint.artists
        }
        if len(memberships) == 1:
            internal.append((constraint, memberships.pop()))
        else:
            cross += 1
    weights = Counter(index for _, index in internal)
    total = len(internal)
    targets = tuple(ratio * total for ratio in SPLIT_RATIOS)

    # Exact three-bin dynamic programming.  Community count is small (11-12 for
    # the released file); the DP makes the 60/20/20 row balance reproducible.
    states: Dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for index in range(len(communities)):
        weight = int(weights[index])
        updated: Dict[tuple[int, int], tuple[int, ...]] = {}
        for (dev_count, test_count), assignment in states.items():
            for split_index in range(3):
                key = (
                    dev_count + (weight if split_index == 1 else 0),
                    test_count + (weight if split_index == 2 else 0),
                )
                candidate = assignment + (split_index,)
                if key not in updated or candidate < updated[key]:
                    updated[key] = candidate
        states = updated

    def objective(item: tuple[tuple[int, int], tuple[int, ...]]) -> tuple[Any, ...]:
        (dev_count, test_count), assignment = item
        counts = (total - dev_count - test_count, dev_count, test_count)
        error = sum((counts[i] - targets[i]) ** 2 for i in range(3))
        return error, max(abs(counts[i] - targets[i]) for i in range(3)), assignment

    (dev_count, test_count), assignment = min(states.items(), key=objective)
    split_counts = (total - dev_count - test_count, dev_count, test_count)
    output = [
        HumanConstraint(**{
            **asdict(constraint),
            "split": SPLIT_NAMES[assignment[community_index]],
        })
        for constraint, community_index in internal
    ]
    artist_sets = {
        name: {
            artist.casefold()
            for constraint in output if constraint.split == name
            for artist in constraint.artists
        }
        for name in SPLIT_NAMES
    }
    overlaps = {
        f"{left}_{right}": sorted(artist_sets[left] & artist_sets[right])
        for left, right in itertools.combinations(SPLIT_NAMES, 2)
    }
    if any(overlaps.values()):
        raise MTATError("artist-disjoint split invariant failed")
    report = {
        "algorithm": "networkx weighted Louvain communities + exact whole-community bin DP",
        "seed": seed,
        "louvain_resolution": LOUVAIN_RESOLUTION,
        "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
        "communities": len(communities),
        "accepted_before_partition": len(constraints),
        "retained_constraints": len(output),
        "discarded_cross_community": cross,
        "constraint_counts": dict(zip(SPLIT_NAMES, split_counts)),
        "artist_counts": {name: len(values) for name, values in artist_sets.items()},
        "artist_overlap": overlaps,
        "clip_counts": {
            name: len({
                clip_id
                for constraint in output if constraint.split == name
                for clip_id in constraint.clip_ids
            })
            for name in SPLIT_NAMES
        },
    }
    return output, report


def build_benchmark(root: str | Path, output: str | Path) -> Dict[str, Any]:
    comparisons, clips, audit = load_inputs(root)
    constraints, rejected = parse_constraints(comparisons, clips)
    split, split_report = artist_disjoint_split(constraints)
    document: Dict[str, Any] = {
        "schema_version": 10,
        "benchmark_id": "magnatagatune-human-odd-one-out-v10",
        "human_judgment_semantics": (
            "clipN_numvotes counts listeners selecting clip N as the odd one out; "
            "the other two clips are the relative-similarity pair"
        ),
        "primary_metric": (
            "odd-one-out accuracy: representation predicts the clip opposite "
            "the closest of the three pairwise distances"
        ),
        "minimum_total_votes": MIN_TOTAL_VOTES,
        "ties": "excluded; never broken algorithmically",
        "source": {
            "dataset_page": (
                "https://web.archive.org/web/20231211051442id_/"
                "https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset"
            ),
            "paper": (
                "Law, West, Mandel, Bay & Downie (2009), Evaluation of "
                "Algorithms Using Games: The Case of Music Tagging"
            ),
            "csv_base_url": MTAT_BASE_URL,
            "license_distinction": (
                "The City page grants download access and requests citation but "
                "states no dataset-wide audio redistribution license. CSVs and "
                "aggregate hashes are reproducible; copyrighted audio remains "
                "local under ml_data and is not committed or re-hosted."
            ),
        },
        "input_audit": audit,
        "vote_audit": rejected,
        "split": split_report,
        "constraints": [asdict(item) for item in split],
        "created_at": _now(),
    }
    document["content_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    _write_json(output, document)
    return document


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-9, None)


def odd_predictions(
    embeddings: np.ndarray,
    clip_to_row: Mapping[int, int],
    constraints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    values = _normalise_rows(embeddings)
    predictions = []
    for constraint in constraints:
        ids = tuple(map(int, constraint["clip_ids"]))
        rows = [clip_to_row[item] for item in ids]
        pair_distances = [
            (1.0 - float(values[rows[left]] @ values[rows[right]]), left, right)
            for left, right in ((0, 1), (0, 2), (1, 2))
        ]
        _, left, right = min(pair_distances, key=lambda item: (item[0], item[1], item[2]))
        predictions.append(next(index for index in range(3) if index not in (left, right)))
    return np.asarray(predictions, dtype=np.int8)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def score_representation(
    embeddings: np.ndarray,
    clip_to_row: Mapping[int, int],
    constraints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    predicted = odd_predictions(embeddings, clip_to_row, constraints)
    actual = np.asarray([
        tuple(map(int, item["clip_ids"])).index(int(item["odd_clip_id"]))
        for item in constraints
    ], dtype=np.int8)
    correct = predicted == actual
    low, high = _wilson(int(correct.sum()), len(correct))
    confidence = np.asarray([float(item["confidence"]) for item in constraints])
    return {
        "correct": int(correct.sum()),
        "constraints": len(correct),
        "accuracy": float(correct.mean()) if len(correct) else 0.0,
        "wilson_ci95": [float(low), float(high)],
        "confidence_weighted_accuracy": float(np.average(correct, weights=confidence))
        if len(correct) and float(confidence.sum()) > 0 else 0.0,
        "predicted_odd_indices": predicted.tolist(),
        "correct_vector": correct.astype(int).tolist(),
    }


def paired_bootstrap_delta(
    challenger: Sequence[int],
    baseline: Sequence[int],
    *,
    iterations: int = 50_000,
    seed: int = 20260712,
) -> Dict[str, Any]:
    left = np.asarray(challenger, dtype=np.float32)
    right = np.asarray(baseline, dtype=np.float32)
    if len(left) != len(right) or not len(left):
        raise ValueError("paired vectors must be non-empty and equally sized")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float32)
    for start in range(0, iterations, 2000):
        count = min(2000, iterations - start)
        sampled = rng.integers(0, len(left), size=(count, len(left)))
        draws[start:start + count] = (left[sampled] - right[sampled]).mean(axis=1)
    return {
        "delta": float((left - right).mean()),
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "probability_positive": float(np.mean(draws > 0)),
        "iterations": iterations,
        "seed": seed,
    }


def prepare_audio_features(
    benchmark_path: str | Path,
    metadata_root: str | Path,
    audio_root: str | Path,
    output: str | Path,
    workers: int = 12,
) -> Dict[str, Any]:
    """Decode only benchmark clips and cache production-compatible mels + DSP."""
    from soundalike.audio.vibe import vibe_from_signal
    from .spectrogram import SpectrogramConfig, _fit_frames, load_audio, log_mel_full

    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    _, clips, _ = load_inputs(metadata_root)
    clip_ids = sorted({
        int(clip_id)
        for item in benchmark["constraints"]
        for clip_id in item["clip_ids"]
    })
    config = SpectrogramConfig()

    def process(clip_id: int) -> tuple[int, np.ndarray, np.ndarray]:
        path = Path(audio_root) / str(clips[clip_id]["mp3_path"])
        if not path.is_file():
            raise MTATError(f"missing local MTAT audio: {path}")
        signal = load_audio(path, config.sample_rate)
        mel = _fit_frames(log_mel_full(signal, config), config.target_frames)
        vibe = vibe_from_signal(signal, config.sample_rate).vector()
        return clip_id, mel.astype(np.float16), vibe.astype(np.float32)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        rows = list(executor.map(process, clip_ids))
    rows.sort(key=lambda item: item[0])
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        clip_ids=np.asarray([item[0] for item in rows], dtype=np.int64),
        mels=np.stack([item[1] for item in rows]),
        vibe=np.stack([item[2] for item in rows]),
    )
    return {
        "clips": len(rows),
        "seconds": time.perf_counter() - started,
        "output": str(target),
        "sha256": sha256_path(target),
    }


def _load_encoder(checkpoint: str | Path, *, pool_type: str | None = None) -> Any:
    import torch
    from .model import ResNetAudioEncoder

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    width = int(payload.get("width", state["stem.0.weight"].shape[0]))
    embedding_dim = int(payload["embedding_dim"])
    inferred_pool = "gem" if "pool.p" in state else "avg"
    model = ResNetAudioEncoder(
        embedding_dim=embedding_dim,
        width=width,
        pool_type=pool_type or payload.get("pool_type", inferred_pool),
    )
    model.load_state_dict({key: value.float() for key, value in state.items()})
    return model.eval()


def embed_mels(
    mels: np.ndarray,
    checkpoint: str | Path,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    import torch

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_encoder(checkpoint).to(selected_device)
    started = time.perf_counter()
    output = []
    with torch.inference_mode():
        for start in range(0, len(mels), batch_size):
            batch = torch.from_numpy(
                np.asarray(mels[start:start + batch_size], dtype=np.float32)
            ).unsqueeze(1).to(selected_device)
            output.append(model(batch).float().cpu().numpy())
    values = np.concatenate(output)
    return values, {
        "seconds": time.perf_counter() - started,
        "device": selected_device,
        "embedding_dim": values.shape[1],
        "checkpoint_sha256": sha256_path(checkpoint),
    }


def extract_clap(
    file_paths: Sequence[str | Path],
    *,
    batch_size: int = 8,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Extract LAION CLAP music-audio embeddings; import remains optional."""
    try:
        import laion_clap
        import torch
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise MTATError("laion-clap is required for the pretrained baseline") from exc
    if not torch.cuda.is_available():
        raise MTATError("CLAP extraction is predeclared to require CUDA")
    torch.cuda.empty_cache()
    started = time.perf_counter()
    model = laion_clap.CLAP_Module(enable_fusion=False, device="cuda")
    model.load_ckpt(model_id=1, verbose=False)
    checkpoint = Path(laion_clap.__file__).resolve().parent / "630k-audioset-best.pt"
    if not checkpoint.is_file():
        raise MTATError("LAION-CLAP model_id=1 checkpoint was not materialized")
    batches = []
    paths = [str(path) for path in file_paths]
    for start in range(0, len(paths), batch_size):
        batches.append(model.get_audio_embedding_from_filelist(
            x=paths[start:start + batch_size], use_tensor=False
        ))
    values = np.concatenate(batches)
    return _normalise_rows(values), {
        "seconds": time.perf_counter() - started,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "model": "LAION-CLAP HTSAT-tiny 630k+AudioSet non-fusion",
        "model_id": 1,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_path(checkpoint),
        "embedding_dim": int(values.shape[1]),
    }


def _fit_vibe(
    vibe: np.ndarray,
    clip_ids: Sequence[int],
    constraints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    from soundalike.audio.vibe import DEFAULT_WEIGHTS, FEATURE_NAMES, weight_vector

    row = {int(clip_id): index for index, clip_id in enumerate(clip_ids)}
    train_ids = sorted({
        int(clip_id)
        for item in constraints if item["split"] == "train"
        for clip_id in item["clip_ids"]
    })
    train = np.asarray([vibe[row[item]] for item in train_ids], dtype=np.float32)
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-6
    weights = np.sqrt(weight_vector(DEFAULT_WEIGHTS)).astype(np.float32)
    if len(weights) != len(FEATURE_NAMES) or len(weights) != vibe.shape[1]:
        raise MTATError("DSP feature layout does not match production vibe layout")
    return _normalise_rows(((vibe - mean) / std) * weights)


def _fma_regularizer_embeddings(
    fma_path: str | Path,
    encoder_checkpoint: str | Path,
    *,
    sample_count: int = 512,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, Dict[str, Any]]:
    with np.load(fma_path, allow_pickle=False) as packed:
        total = len(packed["X"])
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(total, size=min(sample_count, total), replace=False))
        mels = packed["X"][selected].astype(np.float32)
    embeddings, resources = embed_mels(mels, encoder_checkpoint)
    resources.update({
        "source": "independent FMA packed mel cache",
        "source_sha256": sha256_path(fma_path),
        "samples": len(embeddings),
        "selection_seed": seed,
    })
    return embeddings, resources


def train_triplet_projection(
    base_embeddings: np.ndarray,
    clip_ids: Sequence[int],
    constraints: Sequence[Mapping[str, Any]],
    fma_embeddings: np.ndarray,
    *,
    output_checkpoint: str | Path,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Select a compact projection on DEV; TEST is not read here."""
    import torch
    import torch.nn.functional as functional

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise MTATError("triplet projection training is predeclared to require CUDA")
    base = _normalise_rows(base_embeddings)
    fma = _normalise_rows(fma_embeddings)
    row = {int(clip_id): index for index, clip_id in enumerate(clip_ids)}
    train = [item for item in constraints if item["split"] == "train"]
    dev = [item for item in constraints if item["split"] == "dev"]
    train_indices = np.asarray([
        [row[int(item["similar_clip_ids"][0])],
         row[int(item["similar_clip_ids"][1])],
         row[int(item["odd_clip_id"])]]
        for item in train
    ], dtype=np.int64)
    train_weights = np.asarray([float(item["confidence"]) for item in train], np.float32)
    x = torch.from_numpy(base).to(device)
    fma_x = torch.from_numpy(fma).to(device)
    indices = torch.from_numpy(train_indices).to(device)
    weights = torch.from_numpy(train_weights).to(device)
    grid = [
        {"dim": dim, "margin": margin, "lr": lr, "fma_lambda": regularizer}
        for dim in (32, 64, 128)
        for margin in (0.10, 0.20)
        for lr in (3e-4, 1e-3)
        for regularizer in (0.05, 0.20)
    ]
    candidates = []
    best: tuple[Any, ...] | None = None
    best_state = None
    started = time.perf_counter()
    for candidate_index, config in enumerate(grid):
        torch.manual_seed(SPLIT_SEED + candidate_index)
        layer = torch.nn.Linear(base.shape[1], config["dim"], bias=False).to(device)
        torch.nn.init.orthogonal_(layer.weight)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=config["lr"], weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(SPLIT_SEED + candidate_index)
        for _ in range(250):
            projected = functional.normalize(layer(x), dim=1)
            anchor = projected[indices[:, 0]]
            positive = projected[indices[:, 1]]
            negative = projected[indices[:, 2]]
            triplet = functional.relu(
                (1 - (anchor * positive).sum(1))
                - (1 - (anchor * negative).sum(1))
                + config["margin"]
            )
            # Independent FMA geometry-preservation regularizer.
            sampled = torch.randint(
                0, len(fma_x), (256, 2), generator=generator, device=device
            )
            original = (fma_x[sampled[:, 0]] * fma_x[sampled[:, 1]]).sum(1)
            projected_fma = functional.normalize(layer(fma_x), dim=1)
            compressed = (
                projected_fma[sampled[:, 0]] * projected_fma[sampled[:, 1]]
            ).sum(1)
            loss = (
                (triplet * weights).sum() / weights.sum().clamp_min(1e-6)
                + config["fma_lambda"] * functional.mse_loss(compressed, original)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            embedding = functional.normalize(layer(x), dim=1).cpu().numpy()
        dev_score = score_representation(embedding, row, dev)
        candidate = {
            **config,
            "dev_accuracy": dev_score["accuracy"],
            "dev_confidence_weighted_accuracy":
                dev_score["confidence_weighted_accuracy"],
        }
        candidates.append(candidate)
        rank = (
            candidate["dev_accuracy"],
            candidate["dev_confidence_weighted_accuracy"],
            -candidate["dim"],
            -candidate["fma_lambda"],
            -candidate_index,
        )
        if best is None or rank > best:
            best = rank
            best_state = {
                "config": config,
                "state_dict": {
                    key: value.detach().cpu() for key, value in layer.state_dict().items()
                },
            }
    assert best_state is not None
    checkpoint = Path(output_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)
    selected = torch.nn.Linear(
        base.shape[1], int(best_state["config"]["dim"]), bias=False
    ).to(device)
    selected.load_state_dict(best_state["state_dict"])
    with torch.inference_mode():
        embedding = functional.normalize(selected(x), dim=1).cpu().numpy()
    return embedding, {
        "selected_on": "MTAT DEV only",
        "training_labels": "MTAT train odd-one-out constraints only",
        "regularization": "independent FMA pairwise-geometry preservation",
        "commercial_benchmark_rows_used": 0,
        "grid": candidates,
        "selected": best_state["config"],
        "training_seconds": time.perf_counter() - started,
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
    }


def run_calibration(
    benchmark_path: str | Path,
    feature_cache: str | Path,
    metadata_root: str | Path,
    audio_root: str | Path,
    artist_checkpoint: str | Path,
    fma_supcon_checkpoint: str | Path,
    fma_path: str | Path,
    work_dir: str | Path,
    report_path: str | Path,
) -> Dict[str, Any]:
    """Run DEV selection, lock it, then open MTAT TEST exactly once."""
    import torch

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    constraints = list(benchmark["constraints"])
    split = {name: [item for item in constraints if item["split"] == name]
             for name in SPLIT_NAMES}
    if not all(split.values()):
        raise MTATError("benchmark must contain non-empty train/dev/test splits")
    cache = np.load(feature_cache, allow_pickle=False)
    clip_ids = cache["clip_ids"].astype(np.int64)
    mels = cache["mels"].astype(np.float32)
    vibe = cache["vibe"].astype(np.float32)
    clip_to_row = {int(item): index for index, item in enumerate(clip_ids)}

    state_path = work / "state.json"
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if int(existing.get("test_open_count", 0)) != 0:
            raise MTATError("MTAT TEST has already been opened for this work directory")
    state = {
        "schema_version": 10,
        "phase": "REPRESENTATIONS_LOCKED",
        "test_open_count": 0,
        "benchmark_sha256": sha256_path(benchmark_path),
        "feature_cache_sha256": sha256_path(feature_cache),
        "representation_set": [
            "artist_supcon_production_encoder",
            "fma_cross_artist_supcon",
            "vibe_dsp29",
            "laion_clap_pretrained_music_audio",
            "mtat_triplet_projection_fma_regularized",
        ],
        "material_win_rule": {
            "minimum_absolute_accuracy_points": MATERIAL_WIN,
            "paired_bootstrap_ci95_low_must_exceed": 0.0,
            "comparison": "learned compact projection vs artist SupCon incumbent",
        },
        "created_at": _now(),
    }
    state["integrity_sha256"] = hashlib.sha256(_canonical(state)).hexdigest()
    _write_json(state_path, state)

    resources: Dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    artist, resources["artist_supcon"] = embed_mels(mels, artist_checkpoint)
    fma_supcon, resources["fma_cross_artist_supcon"] = embed_mels(
        mels, fma_supcon_checkpoint
    )
    vibe_dsp = _fit_vibe(vibe, clip_ids, constraints)
    _, clips, _ = load_inputs(metadata_root)
    file_paths = [
        Path(audio_root) / str(clips[int(clip_id)]["mp3_path"])
        for clip_id in clip_ids
    ]
    clap, resources["laion_clap"] = extract_clap(file_paths)
    fma_regularizer, resources["fma_regularizer"] = _fma_regularizer_embeddings(
        fma_path, artist_checkpoint
    )
    learned, training = train_triplet_projection(
        artist, clip_ids, constraints, fma_regularizer,
        output_checkpoint=work / "mtat-triplet-projection.pt",
    )
    representations = {
        "artist_supcon_production_encoder": artist,
        "fma_cross_artist_supcon": fma_supcon,
        "vibe_dsp29": vibe_dsp,
        "laion_clap_pretrained_music_audio": clap,
        "mtat_triplet_projection_fma_regularized": learned,
    }
    dev_scores = {
        name: score_representation(values, clip_to_row, split["dev"])
        for name, values in representations.items()
    }
    lock = {
        "phase": "TEST_METHOD_LOCKED",
        "selected_learned_config": training["selected"],
        "representation_names": list(representations),
        "dev_scores": {
            name: {
                key: value for key, value in score.items()
                if key not in ("predicted_odd_indices", "correct_vector")
            }
            for name, score in dev_scores.items()
        },
        "test_labels_compared": False,
        "locked_at": _now(),
    }
    lock["content_sha256"] = hashlib.sha256(_canonical(lock)).hexdigest()
    _write_json(work / "test-method-lock.json", lock)

    # The only test-label access in the pipeline occurs below, after the method
    # and representation set have been written and hash-locked.
    test_scores = {
        name: score_representation(values, clip_to_row, split["test"])
        for name, values in representations.items()
    }
    comparison = paired_bootstrap_delta(
        test_scores["mtat_triplet_projection_fma_regularized"]["correct_vector"],
        test_scores["artist_supcon_production_encoder"]["correct_vector"],
    )
    material_win = bool(
        comparison["delta"] >= MATERIAL_WIN and comparison["ci95"][0] > 0
    )
    opened = {
        **state,
        "phase": "TEST_OPENED_ONCE",
        "test_open_count": 1,
        "test_method_lock_sha256": sha256_path(work / "test-method-lock.json"),
        "test_opened_at": _now(),
    }
    opened["integrity_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in opened.items()
                    if key != "integrity_sha256"})
    ).hexdigest()
    _write_json(state_path, opened)
    report: Dict[str, Any] = {
        "schema_version": 10,
        "kind": "magnatagatune-human-audio-calibration",
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": sha256_path(benchmark_path),
            "split": benchmark["split"],
            "vote_audit": benchmark["vote_audit"],
        },
        "training": training,
        "development": dev_scores,
        "test_once": {
            "open_count": 1,
            "method_lock_sha256": sha256_path(work / "test-method-lock.json"),
            "scores": test_scores,
            "learned_vs_incumbent": comparison,
        },
        "compact_material_win": material_win,
        "catalog_reembedding_permitted": material_win,
        "catalog_reembedded": False,
        "catalog_reembedding_reason": (
            "not executed: material-win and CI rule failed"
            if not material_win else
            "eligible in principle; a separate full-catalog job is required"
        ),
        "production_changed": False,
        "commercial_benchmark_leakage": False,
        "resources": resources,
        "created_at": _now(),
    }
    report["content_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    _write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human audio calibration on MagnaTagATune odd-one-out votes"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download and hash public CSVs")
    download.add_argument("--root", default="ml_data/magnatagatune")
    build = commands.add_parser("build", help="build artist-disjoint benchmark")
    build.add_argument("--root", default="ml_data/magnatagatune")
    build.add_argument(
        "--output", default="ml_data/magnatagatune/benchmark-v10.json"
    )
    prepare = commands.add_parser("prepare", help="cache mels and DSP for retained clips")
    prepare.add_argument(
        "--benchmark", default="ml_data/magnatagatune/benchmark-v10.json"
    )
    prepare.add_argument("--metadata-root", default="ml_data/magnatagatune")
    prepare.add_argument("--audio-root", default="ml_data/magnatagatune/audio")
    prepare.add_argument(
        "--output", default="ml_data/magnatagatune/features-v10.npz"
    )
    prepare.add_argument("--workers", type=int, default=12)
    run = commands.add_parser("run", help="DEV-select and open MTAT TEST once")
    run.add_argument(
        "--benchmark", default="ml_data/magnatagatune/benchmark-v10.json"
    )
    run.add_argument(
        "--features", default="ml_data/magnatagatune/features-v10.npz"
    )
    run.add_argument("--metadata-root", default="ml_data/magnatagatune")
    run.add_argument("--audio-root", default="ml_data/magnatagatune/audio")
    run.add_argument(
        "--artist-checkpoint", default="ml_data/model_artist384/encoder_best.pt"
    )
    run.add_argument(
        "--fma-supcon-checkpoint",
        default="ml_data/iteration4/supcon/supcon_encoder.pt",
    )
    run.add_argument("--fma", default="ml_data/fma_packed.npz")
    run.add_argument("--work-dir", default="ml_data/magnatagatune/v10-run")
    run.add_argument(
        "--report",
        default=".goals/human-quality-recommendations/artifacts/"
                "magnatagatune-human-calibration-v10.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "download":
        print(json.dumps(download_metadata(args.root), indent=2))
    elif args.command == "build":
        result = build_benchmark(args.root, args.output)
        print(json.dumps({
            "output": args.output,
            "content_sha256": result["content_sha256"],
            "split": result["split"],
        }, indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare_audio_features(
            args.benchmark, args.metadata_root, args.audio_root,
            args.output, args.workers,
        ), indent=2))
    else:
        report = run_calibration(
            args.benchmark, args.features, args.metadata_root, args.audio_root,
            args.artist_checkpoint, args.fma_supcon_checkpoint, args.fma,
            args.work_dir, args.report,
        )
        print(json.dumps({
            "report": args.report,
            "test_scores": {
                name: {
                    "accuracy": score["accuracy"],
                    "wilson_ci95": score["wilson_ci95"],
                }
                for name, score in report["test_once"]["scores"].items()
            },
            "compact_material_win": report["compact_material_win"],
        }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
