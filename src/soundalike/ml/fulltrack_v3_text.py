"""Build and verify the frozen CLAP text-tag vectors used by V3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_extract import FrozenClapAdapter, normalize_rows
from .fulltrack_store import sha256_path, stable_json_sha256
from .fulltrack_v3_protocol import load_protocol
from .fulltrack_v3_ranker import _write_npz_exclusive
from .fulltrack_v3_semantic import (
    SCALE_PROTOCOL_PAYLOAD_SHA256,
    _protocol_entries,
    build_label_targets,
    load_train_development_tags,
)


TEXT_SCHEMA_VERSION = 1
TEXT_KIND = "v3_clap_tag_text_embeddings"
TAG_COUNT = 183
PROMPTS_PER_TAG = 3
EMBEDDING_DIMENSION = 512
MODEL_INITIALIZATION_SEED = 20260807
VOCABULARY_SHA256 = (
    "f2439dcaef8e77f5a5158e31376bca598e22b49a1198ba2183e6085b50c16734"
)
PROMPTS_SHA256 = (
    "e2fb49128782948372daeed98a8ad547d95bb82b8097e7cefd5c30ff0d74f32e"
)
EMBEDDINGS_BYTES_SHA256 = (
    "d4e1f5cc419ac395d48425d29131ed2e2a6804fc5baa4b8861e73e30f45fafb8"
)


class V3TextError(RuntimeError):
    """Invalid prompt, vocabulary, text embedding, or frozen text artifact."""


def prompts_for_tag(tag: str) -> Tuple[str, str, str]:
    if not isinstance(tag, str) or "---" not in tag:
        raise V3TextError("tag is malformed")
    domain, value = tag.split("---", 1)
    label = value.replace("-", " ")
    generic = f"A music track with {label}."
    if domain == "genre":
        return (
            generic,
            f"A {label} music track.",
            f"Music in the {label} genre.",
        )
    if domain == "mood/theme":
        return (
            generic,
            f"A music track with a {label} mood.",
            f"Music that feels {label}.",
        )
    if domain == "instrument":
        return (
            generic,
            f"A music track featuring {label}.",
            f"Music played with {label}.",
        )
    raise V3TextError(f"unsupported tag domain: {domain}")


def _embedding_bytes_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_content(
    vocabulary: Sequence[str],
    prompts: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    vocabulary_tuple = tuple(vocabulary)
    prompts_tuple = tuple(prompts)
    values = np.asarray(embeddings, dtype=np.float32)
    if (
        len(vocabulary_tuple) != TAG_COUNT
        or len(prompts_tuple) != TAG_COUNT * PROMPTS_PER_TAG
        or values.shape != (TAG_COUNT, EMBEDDING_DIMENSION)
        or stable_json_sha256(vocabulary_tuple) != VOCABULARY_SHA256
        or stable_json_sha256(prompts_tuple) != PROMPTS_SHA256
        or not np.all(np.isfinite(values))
        or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)
    ):
        raise V3TextError("CLAP text artifact content drift")
    embedding_hash = _embedding_bytes_sha256(values)
    if embedding_hash != EMBEDDINGS_BYTES_SHA256:
        raise V3TextError(
            "CLAP text artifact content drift: "
            f"expected {EMBEDDINGS_BYTES_SHA256}, got {embedding_hash}"
        )


def load_text_artifact(
    path: Path,
    *,
    expected_vocabulary: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...], Mapping[str, object]]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise V3TextError("CLAP text artifact may not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > 4 * 1024 * 1024:
        raise V3TextError("CLAP text artifact is missing or too large")
    with np.load(resolved, allow_pickle=False) as archive:
        if set(archive.files) != {"vocabulary", "prompts", "embeddings"}:
            raise V3TextError("CLAP text artifact member drift")
        vocabulary = tuple(str(value) for value in archive["vocabulary"])
        prompts = tuple(str(value) for value in archive["prompts"])
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32).copy()
    if vocabulary != tuple(expected_vocabulary):
        raise V3TextError("CLAP text vocabulary differs from the train vocabulary")
    _validate_content(vocabulary, prompts, embeddings)
    return (
        embeddings,
        prompts,
        {
            "file_sha256": sha256_path(resolved),
            "vocabulary_sha256": VOCABULARY_SHA256,
            "prompts_sha256": PROMPTS_SHA256,
            "embeddings_bytes_sha256": EMBEDDINGS_BYTES_SHA256,
        },
    )


def clap_text_profiles(
    clap_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> np.ndarray:
    audio = normalize_rows(np.asarray(clap_embeddings, dtype=np.float64))
    text = normalize_rows(np.asarray(text_embeddings, dtype=np.float64))
    if (
        audio.ndim != 2
        or text.shape != (TAG_COUNT, EMBEDDING_DIMENSION)
        or audio.shape[1] != EMBEDDING_DIMENSION
    ):
        raise V3TextError("CLAP audio/text embedding shape drift")
    logits = audio @ text.T
    zscores = (
        logits - np.mean(logits, axis=1, keepdims=True)
    ) / np.maximum(np.std(logits, axis=1, keepdims=True), 1e-8)
    zscores -= np.max(zscores, axis=1, keepdims=True)
    probabilities = np.exp(zscores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return normalize_rows(probabilities)


def build_text_artifact(
    *,
    metadata_root: Path,
    protocol_path: Path,
    output: Path,
) -> Mapping[str, object]:
    if Path(output).exists():
        raise V3TextError("CLAP text output already exists; refusing overwrite")
    protocol = load_protocol(Path(protocol_path))
    if protocol.get("payload_sha256") != SCALE_PROTOCOL_PAYLOAD_SHA256:
        raise V3TextError("scale protocol binding drift")
    labels = load_train_development_tags(Path(metadata_root), protocol)
    train_entries = _protocol_entries(protocol, "train")
    vocabulary, _ = build_label_targets(train_entries, labels)
    prompt_groups = tuple(prompts_for_tag(tag) for tag in vocabulary)
    prompts = tuple(prompt for group in prompt_groups for prompt in group)
    if (
        stable_json_sha256(vocabulary) != VOCABULARY_SHA256
        or stable_json_sha256(prompts) != PROMPTS_SHA256
    ):
        raise V3TextError("frozen vocabulary or prompt drift")
    try:
        import torch
    except ImportError as exc:
        raise V3TextError("PyTorch is required to build CLAP text vectors") from exc
    torch.manual_seed(MODEL_INITIALIZATION_SEED)
    adapter = FrozenClapAdapter()
    raw = adapter.embed_texts(prompts)
    embeddings = normalize_rows(
        raw.reshape(TAG_COUNT, PROMPTS_PER_TAG, EMBEDDING_DIMENSION).mean(axis=1)
    ).astype(np.float32)
    _validate_content(vocabulary, prompts, embeddings)
    _write_npz_exclusive(
        Path(output),
        {
            "vocabulary": np.asarray(vocabulary, dtype=np.str_),
            "prompts": np.asarray(prompts, dtype=np.str_),
            "embeddings": embeddings,
        },
    )
    return {
        "schema_version": TEXT_SCHEMA_VERSION,
        "artifact_kind": TEXT_KIND,
        "protocol_payload_sha256": SCALE_PROTOCOL_PAYLOAD_SHA256,
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "model_initialization_seed": MODEL_INITIALIZATION_SEED,
        "output_file_sha256": sha256_path(Path(output)),
        "vocabulary_sha256": VOCABULARY_SHA256,
        "prompts_sha256": PROMPTS_SHA256,
        "embeddings_bytes_sha256": EMBEDDINGS_BYTES_SHA256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_text_artifact(
            metadata_root=Path(args.metadata_root),
            protocol_path=Path(args.protocol),
            output=Path(args.output),
        )
    except (OSError, ValueError, V3TextError) as exc:
        raise SystemExit(f"V3 CLAP text build failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
