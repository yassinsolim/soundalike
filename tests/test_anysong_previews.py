import errno
import io
import json
from http.client import IncompleteRead
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from urllib.error import HTTPError, URLError

import pytest

from soundalike.anysong import deezer, previews


@pytest.fixture
def client(tmp_path, monkeypatch):
    dz = deezer.Deezer(tmp_path)
    monkeypatch.setattr(dz, "track", lambda _: {"preview": "https://example.test/preview.mp3"})
    monkeypatch.setattr(deezer.time, "sleep", lambda _: None)
    return dz


def test_network_failure_is_not_reported_as_missing_preview(client, monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("temporary outage")

    monkeypatch.setattr(deezer.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Preview download failed"):
        client.fetch_preview(123)
    assert not client.preview_path(123).with_suffix(".part").exists()


def test_partial_download_is_not_cached_as_unavailable(client, monkeypatch):
    monkeypatch.setattr(deezer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"short"))
    with pytest.raises(RuntimeError, match="too small"):
        client.fetch_preview(123)
    assert not client.preview_path(123).exists()
    assert not client.preview_path(123).with_suffix(".part").exists()


class TruncatedResponse(io.BytesIO):
    def read(self, *args):
        raise IncompleteRead(b"partial", 445872)


def test_incomplete_http_download_is_retried_atomically(client, monkeypatch):
    replies = iter([TruncatedResponse(), io.BytesIO(b"x" * 10001)])
    monkeypatch.setattr(deezer.urllib.request, "urlopen", lambda *a, **k: next(replies))
    assert client.fetch_preview(123).read_bytes() == b"x" * 10001
    assert not client.preview_path(123).with_suffix(".part").exists()


def test_persistent_incomplete_http_is_retryable_failure_not_unavailable(client, monkeypatch):
    attempts = []

    def download(*args, **kwargs):
        attempts.append(1)
        return TruncatedResponse()

    monkeypatch.setattr(deezer.urllib.request, "urlopen", download)
    with pytest.raises(RuntimeError, match="Preview download failed"):
        client.fetch_preview(123)
    assert len(attempts) == 3
    assert not client.preview_path(123).exists()
    assert not client.preview_path(123).with_suffix(".part").exists()


def test_incomplete_metadata_response_is_retried(client, monkeypatch):
    replies = iter([TruncatedResponse(), io.BytesIO(b'{"id": 123}')])
    monkeypatch.setattr(deezer.urllib.request, "urlopen", lambda *a, **k: next(replies))
    monkeypatch.setattr(client.limiter, "wait", lambda: None)
    assert client._get("track/123") == {"id": 123}


def test_cooldown_can_extend_an_already_waiting_worker(monkeypatch):
    clock = [0.0]
    sleeps = []
    limiter = deezer.RateLimiter(1)
    monkeypatch.setattr(deezer.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        assert limiter._lock.acquire(blocking=False), "Pacer must not hold its lock while sleeping"
        limiter._lock.release()
        sleeps.append(seconds)
        if len(sleeps) == 1:
            limiter.cooldown(10)
        clock[0] += seconds

    monkeypatch.setattr(deezer.time, "sleep", sleep)
    limiter.wait()
    limiter.wait()
    assert sleeps == [1, 9]
    assert clock[0] == 10
    limiter.wait()
    assert clock[0] == 11


@pytest.mark.parametrize("value,expected", [
    ("30", 30), ("Thu, 01 Jan 1970 00:00:20 GMT", 20),
    ("-2", 5), ("NaN", 5), ("invalid", 5),
])
def test_retry_after_parsing(value, expected, monkeypatch):
    monkeypatch.setattr(deezer.time, "time", lambda: 0)
    error = HTTPError("https://example.test", 429, "limited", {"Retry-After": value}, None)
    assert deezer._retry_delay(error, 5) == expected


@pytest.mark.parametrize("metadata", [False, True])
def test_429_defers_shared_workers_until_retry_after(client, monkeypatch, metadata):
    clock = [0.0]
    attempts = []
    monkeypatch.setattr(deezer.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(deezer.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    def download(*args, **kwargs):
        attempts.append(clock[0])
        if len(attempts) == 1:
            raise HTTPError("https://example.test", 429, "limited", {"Retry-After": "30"}, None)
        return io.BytesIO(b'{"id":123}' if metadata else b"x" * 10001)

    monkeypatch.setattr(deezer.urllib.request, "urlopen", download)
    with pytest.warns(RuntimeWarning, match="rate limited"):
        if metadata:
            assert client._get("track/123") == {"id": 123}
        else:
            assert client.fetch_preview(123).stat().st_size == 10001
    assert len(attempts) == 2
    assert attempts[1] - attempts[0] >= 30


def test_api_json_quota_response_pauses_all_metadata_workers(client, monkeypatch):
    clock = [0.0]
    replies = iter([io.BytesIO(b'{"error":{"code":4}}'), io.BytesIO(b'{"id":123}')])
    monkeypatch.setattr(deezer.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(deezer.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    monkeypatch.setattr(deezer.urllib.request, "urlopen", lambda *a, **k: next(replies))
    with pytest.warns(RuntimeWarning, match="quota exceeded"):
        assert client._get("track/123") == {"id": 123}
    assert clock[0] >= 5


def test_download_is_atomic_and_reused(client, monkeypatch):
    calls = []

    def download(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b"x" * 10001)

    monkeypatch.setattr(deezer.urllib.request, "urlopen", download)
    assert client.fetch_preview(123).read_bytes() == b"x" * 10001
    assert client.fetch_preview(123).stat().st_size == 10001
    assert len(calls) == 1


def test_disk_full_fails_without_retries(client, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(deezer.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="disk full"):
        client.fetch_preview(123)
    assert len(calls) == 1


def test_manifest_retries_errors_and_missing_or_truncated_cached_files(tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("1 2 3 4 5")
    cache = tmp_path / "previews"
    cache.mkdir()
    records = [{"id": 1, "ok": True}, {"id": 2, "ok": True},
               {"id": 3, "ok": False, "reason": "temporary outage"},
               {"id": 4, "ok": False, "reason": "no_preview"}, {"id": 5, "ok": True}]
    (cache / "manifest.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    calls = []

    class FakeDeezer:
        def __init__(self, path, **kwargs):
            self.path = Path(path)

        def preview_path(self, tid):
            return self.path / f"{tid}.mp3"

        def fetch_preview(self, tid):
            calls.append(tid)
            self.preview_path(tid).write_bytes(b"x" * 10001)
            return self.preview_path(tid)

    (cache / "1.mp3").write_bytes(b"x" * 10001)
    (cache / "5.mp3").write_bytes(b"partial")
    monkeypatch.setattr(previews, "Deezer", FakeDeezer)
    previews.main(["--ids", str(ids), "--out", str(cache), "--workers", "2"])
    assert sorted(calls) == [2, 3, 5]


@pytest.mark.parametrize("error", [RuntimeError("outage"), MemoryError("allocation")])
def test_bulk_errors_never_report_success(tmp_path, monkeypatch, error):
    ids = tmp_path / "ids.txt"
    ids.write_text("123")

    class FakeDeezer:
        def __init__(self, path, **kwargs):
            self.path = Path(path)
            self.path.mkdir()

        def preview_path(self, tid):
            return self.path / f"{tid}.mp3"

        def fetch_preview(self, tid):
            raise error

    monkeypatch.setattr(previews, "Deezer", FakeDeezer)
    with pytest.raises(type(error)):
        previews.main(["--ids", str(ids), "--out", str(tmp_path / "cache")])


def test_artist_batches_keep_exact_ids_and_fall_back_without_losing_tracks(tmp_path, monkeypatch, capsys):
    ids = tmp_path / "ids.txt"
    ids.write_text("1 2 3 4 5 6 7 8 9")
    mapping = tmp_path / "artists.json"
    mapping.write_text(json.dumps({1: 10, 2: 10, 3: 10, 4: 20, 5: 20, 6: 20, 8: 30, 9: 30}))
    batches, calls = [], []

    class FakeDeezer:
        def __init__(self, path, **kwargs):
            self.path = Path(path)
            self.path.mkdir()

        def preview_path(self, tid):
            return self.path / f"{tid}.mp3"

        def artist_top_tracks(self, aid):
            batches.append(aid)
            if aid == 20:
                raise RuntimeError("metadata outage")
            return [{"id": 1, "preview": "https://example.test/1.mp3"},
                    {"id": 1, "preview": "https://example.test/1.mp3"},
                    {"id": 2, "preview": ""}, {"id": 3, "preview": 123},
                    {"id": 999, "preview": "https://example.test/unrequested.mp3"}]

        def fetch_preview(self, tid, meta=None):
            calls.append((tid, meta is not None))
            self.preview_path(tid).write_bytes(b"x" * 10001)
            return self.preview_path(tid)

    monkeypatch.setattr(previews, "Deezer", FakeDeezer)
    previews.main(["--ids", str(ids), "--out", str(tmp_path / "cache"),
                   "--artist-map", str(mapping), "--workers", "2"])
    assert sorted(batches) == [10, 20]
    assert sorted(calls) == [(i, i == 1) for i in range(1, 10)]
    assert "artist metadata failed 20; using individual lookups" in capsys.readouterr().out
    records = [json.loads(line) for line in (tmp_path / "cache" / "manifest.jsonl").read_text().splitlines()]
    assert {r["id"] for r in records} == set(range(1, 10))
    assert all(r["ok"] for r in records)


def test_artist_groups_only_include_pending_ids_and_worthwhile_batches(tmp_path):
    mapping = tmp_path / "artists.json"
    mapping.write_text(json.dumps({1: 10, 2: 10, 3: 10, 4: 10, 5: 20, 6: 20}))
    assert previews.artist_groups(mapping, [2, 3, 4, 5, 6]) == [(10, {2, 3, 4})]


@pytest.mark.parametrize("mapping", [[], {"1": 0}, {"-1": 3}, {"1": True}, {"1": "3"}, {"01": 3}])
def test_artist_map_rejects_invalid_mappings(tmp_path, mapping):
    path = tmp_path / "artists.json"
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="artist|Artist"):
        previews.artist_groups(path, [1])


@pytest.mark.parametrize("payload", [[], {"data": {}}, {"data": [None]}, {"data": [{"id": True}]}])
def test_artist_metadata_rejects_invalid_api_shape(client, monkeypatch, payload):
    monkeypatch.setattr(client, "_get", lambda *a, **k: payload)
    with pytest.raises(ValueError, match="Invalid track metadata"):
        client.artist_top_tracks(10)


def test_artist_metadata_uses_existing_rate_limited_client(client, monkeypatch):
    calls = []

    def get(path, params):
        calls.append((path, params))
        return {"data": [{"id": 123, "preview": "https://example.test/123.mp3"}]}

    monkeypatch.setattr(client, "_get", get)
    assert client.artist_top_tracks(10)[0]["id"] == 123
    assert calls == [("artist/10/top", {"limit": 100})]


def test_batch_metadata_download_and_expired_url_refresh(client, monkeypatch):
    urls = []
    refreshes = []

    def download(request, **kwargs):
        urls.append(request.full_url)
        if request.full_url.endswith("expired.mp3"):
            raise URLError("expired URL")
        return io.BytesIO(b"x" * 10001)

    def track(tid):
        refreshes.append(tid)
        return {"preview": "https://example.test/fresh.mp3"}

    monkeypatch.setattr(deezer.urllib.request, "urlopen", download)
    monkeypatch.setattr(client, "track", track)
    client.fetch_preview(123, {"preview": "https://example.test/valid.mp3"})
    assert not refreshes
    client.fetch_preview(456, {"preview": "https://example.test/expired.mp3"})
    assert refreshes == [456]
    assert urls[-1] == "https://example.test/fresh.mp3"
    assert client.preview_path(456).stat().st_size == 10001


def test_bounded_results_refill_while_an_earlier_job_is_still_running():
    refilled = Event()

    def work(i):
        if i == 0:
            assert refilled.wait(5), "Workers were blocked behind the first slow job"
        if i == 3:
            refilled.set()
        return i

    with ThreadPoolExecutor(2) as pool:
        assert sorted(previews.bounded_results(pool, work, range(6), 2)) == list(range(6))


def test_bounded_results_does_not_eagerly_submit_or_swallow_fatal_errors():
    produced = []

    def items():
        for i in range(10000):
            produced.append(i)
            yield i

    def work(i):
        raise MemoryError("allocation")

    with ThreadPoolExecutor(2) as pool:
        results = previews.bounded_results(pool, work, items(), 2)
        assert produced == []
        with pytest.raises(MemoryError):
            list(results)
    assert len(produced) == 2
