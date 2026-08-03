import csv
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.fulltrack_v3_ranker import _write_npz_exclusive
from soundalike.ml.fulltrack_v3_reserve_candidate import (
    CLAP_HEAD_NAMES,
    ReserveCandidateModel,
    V3ReserveCandidateError,
    _model_arrays,
    audit_frozen_shadow,
    calibrated_candidate_scores,
    development_gate,
    load_candidate_model,
    load_protocol_tags,
    set_aware_similarity,
    shadow_gate,
)
from soundalike.ml.fulltrack_v3_semantic import LABEL_HEADER, SemanticHead


def _head(representation: str, *, dimension: int = 3) -> SemanticHead:
    vocabulary = ("genre---rock", "instrument---guitar")
    return SemanticHead(
        representation=representation,
        ridge=10.0,
        vocabulary=vocabulary,
        input_mean=np.arange(dimension, dtype=np.float64),
        input_scale=np.arange(1, dimension + 1, dtype=np.float64),
        coefficients=np.arange(
            dimension * len(vocabulary), dtype=np.float64
        ).reshape(dimension, len(vocabulary))
        / 10.0,
        prior=np.asarray([0.2, 0.4], dtype=np.float64),
        idf=np.asarray([1.5, 2.0], dtype=np.float64),
    )


def _model() -> ReserveCandidateModel:
    vocabulary = ("genre---rock", "instrument---guitar")
    return ReserveCandidateModel(
        vocabulary=vocabulary,
        clap_heads={name: _head("clap") for name in CLAP_HEAD_NAMES},
        musicfm_head=_head("musicfm", dimension=4),
    )


def _evaluation(recall: float = 0.2):
    return {
        "relative_delta": {
            "recall_at_k": recall,
            "mrr": 0.01,
            "graded_ndcg_at_k": 0.02,
        },
        "paired_delta": {
            "recall_at_k": {"paired_bootstrap_ci95": [0.001, 0.01]}
        },
        "positive_folds": {"recall_at_k": 4},
        "worst_fold_relative_delta": {"recall_at_k": -0.04},
    }


def test_candidate_model_round_trips_without_pickle(tmp_path: Path):
    model = _model()
    output = tmp_path / "candidate.npz"

    _write_npz_exclusive(output, _model_arrays(model))
    loaded = load_candidate_model(output)

    assert loaded.vocabulary == model.vocabulary
    for name in CLAP_HEAD_NAMES:
        np.testing.assert_array_equal(
            loaded.clap_heads[name].coefficients,
            model.clap_heads[name].coefficients,
        )
    np.testing.assert_array_equal(
        loaded.musicfm_head.input_mean,
        model.musicfm_head.input_mean,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["vocabulary"].dtype.kind == "U"


def test_calibrated_scores_apply_frozen_channel_weights():
    values = np.asarray([-2.0, -0.5, 0.25, 3.0])
    reverse = values[::-1]

    scores = calibrated_candidate_scores(values, reverse, values, reverse)

    standard = (values - values.mean()) / values.std()
    reverse_standard = (reverse - reverse.mean()) / reverse.std()
    audio = 0.925 * standard + 0.075 * reverse_standard
    semantic = 0.9 * standard + 0.1 * reverse_standard
    np.testing.assert_allclose(scores, 0.6 * audio + 0.4 * semantic)


def test_set_aware_similarity_is_symmetric_and_finite():
    predictions = np.asarray(
        [
            [0.9, 0.8, 0.7, 0.2, 0.1],
            [0.8, 0.7, 0.6, 0.3, 0.2],
            [0.1, 0.2, 0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )

    similarity = set_aware_similarity(
        predictions, np.ones(5, dtype=np.float32)
    )

    np.testing.assert_allclose(similarity, similarity.T)
    assert np.all(np.isfinite(similarity))
    assert similarity[0, 1] > similarity[0, 2]


def test_development_and_shadow_gates_keep_promotion_blocked():
    assert development_gate(_evaluation())["passed"] is True
    assert development_gate(_evaluation(0.199))["passed"] is False

    shadow = shadow_gate(_evaluation())

    assert shadow["automated_passed"] is True
    assert shadow["human_pilot_required"] is True
    assert shadow["promotion_allowed"] is False


def test_label_loader_does_not_open_unselected_shadow(
    tmp_path: Path, monkeypatch
):
    entries = {
        "train": ({"track_id": 1, "artist_id": 10},),
        "development": ({"track_id": 2, "artist_id": 20},),
        "shadow": ({"track_id": 3, "artist_id": 30},),
    }
    monkeypatch.setattr(
        "soundalike.ml.fulltrack_v3_reserve_candidate._protocol_entries",
        lambda _protocol, split: entries[split],
    )
    split_root = tmp_path / "data" / "splits" / "split-0"
    split_root.mkdir(parents=True)
    with (split_root / "autotagging-train.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(LABEL_HEADER)
        writer.writerow(
            [
                "track_1",
                "artist_10",
                "album_100",
                "1.mp3",
                "30",
                "genre---rock",
            ]
        )

    labels = load_protocol_tags(tmp_path, {}, ("train",))

    assert labels == {1: ("genre---rock",)}
    assert not (split_root / "autotagging-test.tsv").exists()


def test_shadow_audit_refuses_existing_state_before_reading_inputs(
    tmp_path: Path,
):
    state = tmp_path / "shadow-state.json"
    state.write_text("{}", encoding="utf-8")

    with pytest.raises(V3ReserveCandidateError, match="refusing reopen"):
        audit_frozen_shadow(
            metadata_root=tmp_path / "missing",
            protocol_path=tmp_path / "missing-protocol.json",
            clap_store=tmp_path / "missing-clap",
            musicfm_shadow_store=tmp_path / "missing-musicfm",
            model_path=tmp_path / "missing-model.npz",
            metadata_path=tmp_path / "missing-metadata.json",
            development_report_path=tmp_path / "missing-development.json",
            shadow_extraction_plan_path=tmp_path / "missing-plan.json",
            freeze_path=tmp_path / "missing-freeze.json",
            output=tmp_path / "shadow-report.json",
            audit_state_path=state,
        )
