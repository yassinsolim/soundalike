from __future__ import annotations

import pytest

from soundalike.ml.fulltrack_v3_reserve_protocol import (
    V3ReserveProtocolError,
    reserve_split,
)


def test_reserve_split_keeps_base_training_artists_in_train():
    assert reserve_split("train", 11, {11}, {22}) == "train"
    assert reserve_split("train", 33, {11}, {22}) == "train"


def test_reserve_split_uses_natural_parts_for_untouched_artists():
    assert reserve_split("validation", 33, {11}, {22}) == "development"
    assert reserve_split("test", 33, {11}, {22}) == "shadow"


def test_reserve_split_excludes_historical_and_unassigned_artists():
    assert reserve_split("validation", 22, {11}, {22}) is None
    assert reserve_split("test", 11, {11}, {22}) is None
    assert reserve_split(None, 33, {11}, {22}) is None


def test_reserve_split_rejects_invalid_artist_ids():
    with pytest.raises(V3ReserveProtocolError, match="positive integers"):
        reserve_split("validation", 0, set(), set())
    with pytest.raises(V3ReserveProtocolError, match="positive integers"):
        reserve_split("test", True, set(), set())
