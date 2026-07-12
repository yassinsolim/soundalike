import json
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.magnatagatune_v10 import (
    MATERIAL_WIN,
    MTAT_CSV_HASHES,
    HumanConstraint,
    artist_disjoint_split,
    odd_predictions,
    paired_bootstrap_delta,
    parse_constraints,
    score_representation,
)


def _clips():
    return {
        index: {"artist": f"Artist {index // 3}"}
        for index in range(1, 40)
    }


def test_votes_are_parsed_as_odd_one_out_without_breaking_ties():
    rows = [
        {
            "clip1_id": "1", "clip2_id": "2", "clip3_id": "3",
            "clip1_numvotes": "1", "clip2_numvotes": "1",
            "clip3_numvotes": "5",
        },
        {
            "clip1_id": "4", "clip2_id": "5", "clip3_id": "6",
            "clip1_numvotes": "2", "clip2_numvotes": "2",
            "clip3_numvotes": "0",
        },
        {
            "clip1_id": "7", "clip2_id": "8", "clip3_id": "9",
            "clip1_numvotes": "1", "clip2_numvotes": "0",
            "clip3_numvotes": "0",
        },
    ]
    constraints, audit = parse_constraints(rows, _clips(), min_total_votes=3)
    assert len(constraints) == 1
    assert constraints[0].odd_clip_id == 3
    assert constraints[0].similar_clip_ids == (1, 2)
    assert constraints[0].confidence == pytest.approx(4 / 7)
    assert audit == {"tied_winner": 1, "too_few_votes": 1, "accepted": 1}


def test_artist_disjoint_split_has_no_artist_or_clip_overlap():
    constraints = []
    source = 1
    # Three dense artist communities connected only by discarded bridge rows.
    for group in range(3):
        base = group * 9 + 1
        artists = tuple(f"group-{group}-artist-{i}" for i in range(3))
        for offset in range(5):
            ids = (base + offset, base + 3 + offset, base + 6 + offset)
            constraints.append(HumanConstraint(
                source_row=source,
                clip_ids=ids,
                similar_clip_ids=ids[:2],
                odd_clip_id=ids[2],
                votes=(4, 1, 0),
                total_votes=5,
                winner_votes=4,
                runner_up_votes=1,
                confidence=0.6,
                winner_share=0.8,
                artists=artists,
            ))
            source += 1
    split, report = artist_disjoint_split(constraints)
    by_split = {
        name: {
            artist.casefold()
            for item in split if item.split == name
            for artist in item.artists
        }
        for name in ("train", "dev", "test")
    }
    assert not (by_split["train"] & by_split["dev"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["dev"] & by_split["test"])
    assert report["artist_overlap"] == {
        "train_dev": [], "train_test": [], "dev_test": []
    }


def test_odd_one_out_metric_uses_closest_pair():
    # Clips 1/2 are identical; clip 3 is orthogonal.
    embeddings = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    constraints = [{
        "clip_ids": [1, 2, 3],
        "odd_clip_id": 3,
        "confidence": 1.0,
    }]
    assert odd_predictions(embeddings, {1: 0, 2: 1, 3: 2}, constraints).tolist() == [2]
    score = score_representation(
        embeddings, {1: 0, 2: 1, 3: 2}, constraints
    )
    assert score["accuracy"] == 1.0
    assert score["correct_vector"] == [1]


def test_material_win_requires_both_five_points_and_positive_ci():
    result = paired_bootstrap_delta(
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        iterations=2000,
    )
    assert result["delta"] >= MATERIAL_WIN
    assert result["ci95"][0] > 0
    inconclusive = paired_bootstrap_delta(
        [1, 0, 1, 0], [0, 1, 0, 1], iterations=2000
    )
    assert inconclusive["ci95"][0] <= 0


def test_real_mtat_report_is_once_opened_and_blocks_catalog_reembedding():
    path = Path(
        ".goals/human-quality-recommendations/artifacts/"
        "magnatagatune-human-calibration-v10.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["benchmark"]["split"]["constraint_counts"] == {
        "dev": 28, "test": 29, "train": 86
    }
    assert report["benchmark"]["split"]["artist_overlap"] == {
        "dev_test": [], "train_dev": [], "train_test": []
    }
    assert report["test_once"]["open_count"] == 1
    assert len(report["test_once"]["scores"]) == 5
    assert report["test_once"]["scores"][
        "mtat_triplet_projection_fma_regularized"
    ]["accuracy"] == pytest.approx(19 / 29)
    assert report["test_once"]["learned_vs_incumbent"]["ci95"][0] < 0
    assert report["compact_material_win"] is False
    assert report["catalog_reembedding_permitted"] is False
    assert report["catalog_reembedded"] is False
    assert report["production_changed"] is False
    assert report["commercial_benchmark_leakage"] is False
    assert report["resources"]["gpu"] == "NVIDIA GeForce RTX 5080"
    assert report["resources"]["laion_clap"]["checkpoint_sha256"] == (
        "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
    )
    assert MTAT_CSV_HASHES == {
        "comparisons_final.csv":
            "cf210e087ed5b3f3f8b164626e1d2857cf0ba9ae66bd9229bafe042889107a98",
        "clip_info_final.csv":
            "cb6108a10d3a91f0bfd7d2fbec2382559d15f20c8d28093f14e12162a47a3e78",
    }
    provenance = json.loads(Path(
        "benchmarks/evidence/v10/magnatagatune-provenance.json"
    ).read_text(encoding="utf-8"))
    assert provenance["actual_counts"]["comparison_rows"] == 533
    assert provenance["actual_counts"]["metadata_clip_rows"] == 31382
    assert provenance["actual_counts"]["all_vote_events"] == 7650
    assert provenance["license_and_terms"][
        "explicit_dataset_wide_audio_license_found"
    ] is False
