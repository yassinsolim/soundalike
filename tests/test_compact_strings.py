"""UTF-8 boundaries and empty rows must preserve catalog row identity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp" / "api"))

from _compact import CompactStrings  # noqa: E402


def test_multibyte_metadata_preserves_rows_and_sequence_access():
    values = ["Simple", "", "Caf\u00e9", "\u6771\u4eac", "\U00010400", "  spaced  ", "a" * 300]
    strings = CompactStrings.from_strings(iter(values))
    assert list(strings) == values
    assert [strings[i] for i in range(len(values))] == values
    assert strings[np.int64(-1)] == values[-1]
    assert strings[2:5] == values[2:5]
    assert strings[::-1] == values[::-1]
    with pytest.raises(IndexError):
        strings[len(values)]
    with pytest.raises(IndexError):
        strings[-len(values) - 1]


def test_zero_byte_rows_do_not_collapse_catalog_positions():
    strings = CompactStrings.from_strings(["", "", ""])
    assert list(strings) == ["", "", ""]
    assert strings[2] == ""
    with pytest.raises(IndexError):
        strings[3]


def test_empty_catalog_has_no_indexable_rows():
    strings = CompactStrings.from_strings([])
    assert list(strings) == []
    assert strings[:] == []
    with pytest.raises(IndexError):
        strings[-1]
