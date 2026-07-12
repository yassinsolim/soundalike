import hashlib
import json
from pathlib import Path

import numpy as np

from soundalike.ml import magnatagatune_v10 as v10

from soundalike.ml.magnatagatune_v12 import (
    FAMILY_ORDER,
    _save_checkpoint,
    artist_graph_component_count,
    artist_cluster_bootstrap,
    build_purged_folds,
    confidence_stratum,
    evaluate_win_gate,
    exact_mcnemar,
    materialize_fold,
    paired_sign_flip_test,
    select_config_nested,
    summarize_predictions,
)


def _rows(count=30):
    rows = []
    # One connected raw artist graph, with overlapping boundary artists.
    for index in range(count):
        clips = (index * 3 + 1, index * 3 + 2, index * 3 + 3)
        artists = (
            f"artist-{index}",
            f"artist-{(index + 1) % count}",
            f"extra-{index}",
        )
        confidence = (0.2, 0.35, 0.6)[index % 3]
        rows.append({
            "source_row": 100 + index,
            "clip_ids": clips,
            "similar_clip_ids": clips[:2],
            "odd_clip_id": clips[2],
            "artists": artists,
            "confidence": confidence,
        })
    return rows


def test_connected_graph_keeps_every_row_oof_once_and_purges_leakage():
    rows = _rows()
    assert artist_graph_component_count(rows) == 1
    folds = build_purged_folds(rows)
    assert len(folds) == 5
    tested = [row for fold in folds for row in fold["test_source_rows"]]
    assert sorted(tested) == list(range(100, 130))
    assert len(tested) == len(set(tested))
    for fold in folds:
        train, test, purged = materialize_fold(rows, fold)
        train_clips = {clip for row in train for clip in row["clip_ids"]}
        test_clips = {clip for row in test for clip in row["clip_ids"]}
        train_artists = {artist.casefold() for row in train for artist in row["artists"]}
        test_artists = {artist.casefold() for row in test for artist in row["artists"]}
        assert not train_clips & test_clips
        assert not train_artists & test_artists
        assert len(train) + len(test) + len(purged) == len(rows)
        assert "not GroupKFold" in fold["method"]


def test_folds_and_predeclared_strata_are_deterministic():
    rows = _rows()
    assert build_purged_folds(rows) == build_purged_folds(rows)
    assert [confidence_stratum(x) for x in (0.0, 0.249, 0.25, 0.499, 0.5)] == [
        "low", "low", "medium", "medium", "high"
    ]


def test_nested_tuner_evaluator_never_sees_outer_rows():
    outer_train = _rows(24)
    forbidden = set(range(124, 130))
    seen = []

    def evaluator(config, train, validation, seed):
        supplied = {row["source_row"] for row in (*train, *validation)}
        assert not supplied & forbidden
        seen.append((supplied, seed))
        return [2] * len(validation)

    result = select_config_nested(
        outer_train,
        [{"dim": 32}, {"dim": 64}],
        evaluator,
        input_dim=8,
        family="linear_triplet",
        seed=42,
    )
    assert seen
    assert result["selected"]["config"]["dim"] == 32  # fewer params tie-break
    assert sorted(result["selected"]["inner_oof_source_rows"]) == list(range(100, 124))


def test_statistics_prediction_schema_and_gate_conditions():
    predictions = [
        {
            "source_row": index,
            "fold": index % 5,
            "predicted_odd_index": 2,
            "actual_odd_index": 2 if index % 2 else 1,
            "correct": index % 2,
            "confidence": (0.2, 0.4, 0.7)[index % 3],
            "stratum": confidence_stratum((0.2, 0.4, 0.7)[index % 3]),
            "artists": [f"a{index % 4}", f"a{(index + 1) % 4}"],
        }
        for index in range(20)
    ]
    summary = summarize_predictions(predictions)
    assert summary["count"] == 20
    assert len(summary["source_rows"]) == 20
    assert set(summary["calibration_by_vote_strength"]) == {"low", "medium", "high"}
    challenger = [1] * 20
    baseline = [0] * 15 + [1] * 5
    cluster = artist_cluster_bootstrap(
        challenger, baseline, [row["artists"] for row in predictions],
        iterations=200, seed=7,
    )
    assert cluster["ci95"][0] > 0
    assert paired_sign_flip_test(
        challenger, baseline, iterations=500, seed=7
    )["p_value_two_sided"] < 0.05
    assert exact_mcnemar(challenger, baseline)["discordant"] == 15
    assert evaluate_win_gate(0.06, (0.01, 0.1), (0.005, 0.1),
                             (0.1, 0.1, 0.1, 0.0, -0.1))["passed"]
    failures = evaluate_win_gate(0.049, (0.0, 0.1), (-0.1, 0.1),
                                 (0.1, 0.0, 0.0, 0.0, 0.0))
    assert not failures["passed"]
    assert len(failures["reasons"]) == 4
    assert not evaluate_win_gate(
        -0.06, (0.01, 0.1), (0.005, 0.1), (0.1, 0.1, 0.1, 0.0, -0.1)
    )["passed"]


def test_compact_checkpoint_contains_auditable_schema(tmp_path):
    path = tmp_path / "model.npz"
    logged = _save_checkpoint(
        path, {"weight": np.eye(2, dtype=np.float32)},
        family=FAMILY_ORDER[0], config_id="linear_triplet-00",
        config={"dim": 32}, seed=123, loss=0.25,
    )
    assert len(logged["sha256"]) == 64
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"]))
        assert metadata["schema_version"] == 12
        assert metadata["seed"] == 123
        assert metadata["final_training_loss"] == 0.25



def test_committed_report_and_fold_checkpoints_are_fail_closed():
    root = Path(__file__).resolve().parents[1]
    report_path = (
        root / ".goals" / "human-quality-recommendations" / "artifacts"
        / "magnatagatune-human-nested-cv-v12.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    content = {key: value for key, value in report.items() if key != "content_sha256"}
    assert hashlib.sha256(v10._canonical(content)).hexdigest() == report["content_sha256"]
    assert report["benchmark"]["constraints"] == 307
    assert not report["win_gate"]["passed"]
    assert not report["final_all_data_projection_trained"]
    assert not report["catalog_embeddings_changed"]
    assert not report["commercial_evaluator_changed"]
    assert not report["commercial_final_opened"]
    assert not report["production_ranking_changed"]
    expected_sources = report["scores"]["artist_supcon"]["source_rows"]
    assert len(expected_sources) == len(set(expected_sources)) == 307
    for score in report["scores"].values():
        assert score["source_rows"] == expected_sources
    checkpoint_root = root / "benchmarks" / "evidence" / "v12" / "mtat-fold-checkpoints"
    checked = 0
    for fold in report["fold_logs"]:
        for family, metadata in fold["checkpoints"].items():
            checkpoint = checkpoint_root / f"fold-{fold['fold']}" / f"{family}.npz"
            assert checkpoint.is_file()
            assert v10.sha256_path(checkpoint) == metadata["sha256"]
            with np.load(checkpoint, allow_pickle=False) as cache:
                stored = json.loads(str(cache["metadata_json"]))
            assert stored["config_id"] == metadata["config_id"]
            assert stored["seed"] == metadata["seed"]
            checked += 1
    assert checked == report["resources"]["selected_fold_checkpoint_count"] == 25

    preview_path = (
        root / ".goals" / "human-quality-recommendations" / "artifacts"
        / "human-eval-preview-reaudit-v12.json"
    )
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    preview_content = {
        key: value for key, value in preview.items() if key != "content_sha256"
    }
    assert hashlib.sha256(v10._canonical(preview_content)).hexdigest() == preview["content_sha256"]
    assert preview["unique_results"]["id_covered"] == 480
    assert preview["seeds"]["id_covered"] == 60
    assert preview["ranked_positions"]["resolvable_fraction"] >= 0.9
    assert not preview["errors"]
