import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest

pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

from soundalike.anysong import audio, bench_build, embed_run


@pytest.fixture
def runner(tmp_path, monkeypatch):
    ids = list(range(1, 8))
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("\n".join(map(str, ids)))
    previews = tmp_path / "previews"
    for tid in ids:
        path = embed_run.preview_path(previews, tid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    decoded = []

    def decode(path, sr):
        decoded.append(int(path.stem))
        return np.full(8, int(path.stem), dtype=np.float32)

    class FakeEmbedder:
        sr = 16_000
        calls = 0

        def load(self):
            pass

        def embed(self, clips):
            self.calls += 1
            return {"track": clips[:, :2], "win": clips[:, None, :2]}

    monkeypatch.setitem(embed_run.REGISTRY, "fake", FakeEmbedder)
    monkeypatch.setattr(embed_run, "load_clip", decode)
    args = ["--model", "fake", "--ids", str(ids_path), "--previews", str(previews),
            "--out", str(tmp_path / "emb"), "--batch", "2", "--workers", "2", "--shard", "2"]
    return args, tmp_path / "emb" / "fake", decoded, FakeEmbedder


def test_bounded_decoding_and_resume(runner, monkeypatch):
    args, model_dir, decoded, model = runner
    original = model.embed
    calls = []

    def embed(self, clips):
        calls.append(len(clips))
        assert sum(calls) <= len(decoded) <= sum(calls) + 2
        return original(self, clips)

    monkeypatch.setattr(model, "embed", embed)
    embed_run.main(args)
    assert calls == [2, 2, 2, 1]
    np.testing.assert_array_equal(embed_run.load_store(model_dir)["ids"], range(1, 8))
    assert not list(model_dir.glob("*.tmp"))
    embed_run.main(args)
    assert len(decoded) == 7
    last_shard = (model_dir / "part-0003.npz").read_bytes()
    (model_dir / "part-0001.npz").unlink()
    embed_run.main(args)
    assert (model_dir / "part-0003.npz").read_bytes() == last_shard
    assert (model_dir / "part-0004.npz").exists()
    assert embed_run.done_ids(model_dir) == set(range(1, 8))


def test_next_batch_is_decoded_during_inference(runner, monkeypatch):
    args, _, _, model = runner
    original_decode, original_embed = embed_run.load_clip, model.embed
    next_ready = Event()

    def decode(path, sr):
        value = original_decode(path, sr)
        if path.stem == "3":
            next_ready.set()
        return value

    def embed(self, clips):
        assert next_ready.wait(timeout=5), "Next batch was not prefetched"
        return original_embed(self, clips)

    monkeypatch.setattr(embed_run, "load_clip", decode)
    monkeypatch.setattr(model, "embed", embed)
    embed_run.main(args)


def test_prefetch_is_bounded_and_preserves_order():
    decoded = []

    def decode(tid):
        decoded.append(tid)
        return tid

    with ThreadPoolExecutor(4) as executor:
        batches = embed_run.decoded_batches(executor, decode, list(range(101)), 4)
        consumed = []
        for batch in batches:
            consumed.extend(batch)
            assert len(decoded) <= len(consumed) + 4
    assert consumed == list(range(101))


def test_memory_error_is_fatal(runner, monkeypatch):
    args, model_dir, _, _ = runner

    def oom(*args):
        raise MemoryError("test allocation")

    monkeypatch.setattr(embed_run, "load_clip", oom)
    with pytest.raises(MemoryError, match="test allocation"):
        embed_run.main(args)
    assert not list(model_dir.glob("part-*.npz"))


def test_bad_audio_is_reported_but_valid_shards_survive(runner, monkeypatch):
    args, model_dir, _, _ = runner
    original = embed_run.load_clip

    def decode(path, sr):
        if path.stem == "3":
            raise ValueError("bad audio")
        return original(path, sr)

    monkeypatch.setattr(embed_run, "load_clip", decode)
    with pytest.raises(RuntimeError, match="1 previews failed decoding"):
        embed_run.main(args)
    assert embed_run.done_ids(model_dir) == {1, 2, 4, 5, 6, 7}


@pytest.mark.parametrize("output", [
    {},
    {"track": np.full((2, 2), np.nan)},
    {"track": np.ones((1, 2))},
    {"track": np.ones((2, 0))},
    {"ids": np.ones((2, 2))},
])
def test_invalid_embeddings_are_not_saved(runner, monkeypatch, output):
    args, model_dir, _, model = runner
    monkeypatch.setattr(model, "embed", lambda self, clips: output)
    with pytest.raises(ValueError):
        embed_run.main(args)
    assert not list(model_dir.glob("part-*.npz"))


def test_shard_write_failure_is_not_resumed(runner, monkeypatch):
    args, model_dir, _, _ = runner

    def fail(stream, **arrays):
        stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="disk full"):
        embed_run.main(args)
    assert not list(model_dir.iterdir())


@pytest.mark.parametrize("option", ["--batch", "--workers", "--shard"])
def test_invalid_resource_limits_fail_before_loading(runner, option):
    args, model_dir, _, _ = runner
    with pytest.raises(SystemExit):
        embed_run.main(args + [option, "0"])
    assert not model_dir.exists()


def test_empty_audio_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.empty(0, dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="Empty audio"):
        audio.load_clip(path, 16_000)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "mac_roman"])
def test_samplepairs_preserves_smart_quotes(tmp_path, monkeypatch, encoding):
    path = tmp_path / "pairs.tsv"
    title = "Don't stop \u2019til morning - Lu\u00edsa / Dan\u00e7a"
    row = ["1", "001", "https://example.test/1", "Artist", title, "2024", "0:20",
           "002", "https://example.test/2", "Original artist", "Original", "1999", "0:15"]
    path.write_text("\t".join(["header"] * 13) + "\n" + "\t".join(row), encoding=encoding)
    monkeypatch.setattr(bench_build, "_catalog_title_index", lambda _: {})
    monkeypatch.setattr(bench_build, "_resolve_title_artist", lambda *args: (123, "test"))

    class Deezer:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_preview(self, tid):
            return Path("preview.mp3")

    monkeypatch.setattr(bench_build, "Deezer", Deezer)
    args = argparse.Namespace(input=str(path), out=str(tmp_path / "out.json"),
                              catalog=None, previews=str(tmp_path), rate=1)
    if encoding == "mac_roman":
        with pytest.warns(RuntimeWarning, match="MacRoman"):
            bench_build.build_title_pairs(args)
    else:
        bench_build.build_title_pairs(args)
    result = json.loads(Path(args.out).read_text(encoding="utf-8"))["pairs"][0]
    assert result["query"]["title"] == title
    assert result["split"] == "held_out"
    assert result["source"]["query_start"] == "0:20"


def test_malformed_samplepairs_is_not_silently_skipped(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("header\n1\tshort\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 13 SamplePairs columns"):
        bench_build.build_title_pairs(argparse.Namespace(input=str(path)))
