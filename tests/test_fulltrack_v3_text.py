from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soundalike.ml import fulltrack_v3_text as v3_text
from soundalike.ml.fulltrack_extract import (
    FrozenClapAdapter,
    FullTrackExtractionError,
)
from soundalike.ml.fulltrack_store import stable_json_sha256
from soundalike.ml.fulltrack_v3_text import (
    EMBEDDING_DIMENSION,
    TAG_COUNT,
    V3TextError,
    clap_text_profiles,
    prompts_for_tag,
)


class _FakeTextModel:
    def get_text_embedding(self, texts):
        output = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for row in range(len(texts)):
            output[row, row % EMBEDDING_DIMENSION] = 1.0
        return output


def test_frozen_clap_adapter_validates_and_normalizes_texts():
    adapter = object.__new__(FrozenClapAdapter)
    adapter._model = _FakeTextModel()
    result = adapter.embed_texts(["first prompt", "second prompt"])
    assert result.shape == (2, EMBEDDING_DIMENSION)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)
    with pytest.raises(FullTrackExtractionError, match="text batch"):
        adapter.embed_texts([])


def test_prompt_templates_are_domain_specific_and_deterministic():
    assert prompts_for_tag("genre---post-rock") == (
        "A music track with post rock.",
        "A post rock music track.",
        "Music in the post rock genre.",
    )
    assert "mood" in prompts_for_tag("mood/theme---happy")[1]
    assert "featuring" in prompts_for_tag("instrument---piano")[1]
    with pytest.raises(V3TextError, match="domain"):
        prompts_for_tag("unknown---value")


def test_clap_text_profiles_are_finite_and_normalized():
    rng = np.random.default_rng(8)
    audio = rng.normal(size=(4, EMBEDDING_DIMENSION))
    text = rng.normal(size=(TAG_COUNT, EMBEDDING_DIMENSION))
    profiles = clap_text_profiles(audio, text)
    assert profiles.shape == (4, TAG_COUNT)
    assert np.all(np.isfinite(profiles))
    np.testing.assert_allclose(np.linalg.norm(profiles, axis=1), 1.0)


def test_clap_text_profiles_reject_shape_drift():
    with pytest.raises(V3TextError, match="shape drift"):
        clap_text_profiles(
            np.ones((2, EMBEDDING_DIMENSION)),
            np.ones((TAG_COUNT - 1, EMBEDDING_DIMENSION)),
        )


def test_build_text_artifact_seeds_and_writes_verified_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocabulary = tuple(f"genre---tag-{index}" for index in range(TAG_COUNT))
    prompts = tuple(
        prompt for tag in vocabulary for prompt in prompts_for_tag(tag)
    )
    raw = np.zeros(
        (TAG_COUNT * v3_text.PROMPTS_PER_TAG, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    for tag_index in range(TAG_COUNT):
        raw[
            tag_index * v3_text.PROMPTS_PER_TAG : (tag_index + 1)
            * v3_text.PROMPTS_PER_TAG,
            tag_index,
        ] = 1.0
    embeddings = raw.reshape(
        TAG_COUNT, v3_text.PROMPTS_PER_TAG, EMBEDDING_DIMENSION
    ).mean(axis=1)
    seeded = []

    class _FakeAdapter:
        checkpoint_sha256 = "checkpoint"

        def embed_texts(self, values):
            assert tuple(values) == prompts
            return raw

    monkeypatch.setattr(
        v3_text,
        "load_protocol",
        lambda _path: {"payload_sha256": v3_text.SCALE_PROTOCOL_PAYLOAD_SHA256},
    )
    monkeypatch.setattr(v3_text, "load_train_development_tags", lambda *_: {})
    monkeypatch.setattr(v3_text, "_protocol_entries", lambda *_: ())
    monkeypatch.setattr(
        v3_text, "build_label_targets", lambda *_: (vocabulary, np.empty((0, 0)))
    )
    monkeypatch.setattr(v3_text, "FrozenClapAdapter", _FakeAdapter)
    monkeypatch.setattr(
        v3_text, "VOCABULARY_SHA256", stable_json_sha256(vocabulary)
    )
    monkeypatch.setattr(v3_text, "PROMPTS_SHA256", stable_json_sha256(prompts))
    monkeypatch.setattr(
        v3_text,
        "EMBEDDINGS_BYTES_SHA256",
        v3_text._embedding_bytes_sha256(embeddings),
    )
    monkeypatch.setattr("torch.manual_seed", seeded.append)

    output = tmp_path / "text.npz"
    metadata = v3_text.build_text_artifact(
        metadata_root=tmp_path,
        protocol_path=tmp_path / "protocol.json",
        output=output,
    )
    loaded, loaded_prompts, _ = v3_text.load_text_artifact(
        output, expected_vocabulary=vocabulary
    )

    assert seeded == [v3_text.MODEL_INITIALIZATION_SEED]
    assert metadata["model_initialization_seed"] == v3_text.MODEL_INITIALIZATION_SEED
    assert loaded_prompts == prompts
    np.testing.assert_array_equal(loaded, embeddings)
