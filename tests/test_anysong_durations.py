import json

import numpy as np
import pytest

from soundalike.anysong import durations


def test_table_keeps_lengths_by_id_and_leaves_out_what_is_unknown(tmp_path):
    path = tmp_path / "durations.npz"
    assert durations.save(path, {30: 200, 10: 89.6, 20: 0, 40: -5, 50: 70_000, 2**40: 240}) == 4
    table = durations.DurationTable.load(path)
    assert len(table) == 4 and table.ids.dtype == np.int64 and table.values.dtype == np.uint16
    assert table.seconds(30) == 200 and table.seconds(10) == 90 and table.seconds(2**40) == 240
    assert table.seconds(50) == durations.MAX_SECONDS  # an 18-hour track still counts as long
    for unknown in (20, 40, 5, 31, 2**41):
        assert table.seconds(unknown) is None
    assert durations.DurationTable(np.array([], np.int64), np.array([], np.uint16)).seconds(1) is None
    with pytest.raises(ValueError, match="sorted"):
        durations.DurationTable(np.array([2, 1]), np.array([5, 5]))
    with pytest.raises(ValueError, match="differ"):
        durations.DurationTable(np.array([1, 2]), np.array([5]))


def test_build_from_track_files_earlier_files_win_and_only_indexed_songs_stay(tmp_path, capsys):
    crawl, previews = tmp_path / "deezer_tracks.jsonl", tmp_path / "tracks.jsonl"
    crawl.write_text("\n".join(json.dumps(row) for row in (
        {"track_id": 1, "duration": 200}, {"track_id": 2, "duration": 0}, {"track_id": 9, "duration": 300})) + "\n\n")
    previews.write_text("\n".join(json.dumps(row) for row in (
        {"id": 1, "duration": 999}, {"id": 2, "duration": 61.4}, {"id": 3, "duration": None}, {"id": 4},
        {"id": "5", "duration": 100}, {"id": 6, "duration": True}, {"title": "no id", "duration": 100})))
    np.save(tmp_path / "ids.npy", np.array([1, 2, 3, 4, 5, 6]))
    out = tmp_path / "durations.npz"
    durations.main(["--tracks", str(crawl), str(previews), "--ids", str(tmp_path / "ids.npy"), "--out", str(out)])
    table = durations.DurationTable.load(out)
    assert [(int(t), table.seconds(int(t))) for t in table.ids] == [(1, 200), (2, 61)]
    report = json.loads(capsys.readouterr().out)
    assert report["songs"] == 2 and report["indexed"] == 6 and report["under_90_s"] == 1
    durations.main(["--tracks", str(crawl), "--out", str(out)])  # without --ids: every song of the files
    assert len(durations.DurationTable.load(out)) == 2 and durations.DurationTable.load(out).seconds(9) == 300
