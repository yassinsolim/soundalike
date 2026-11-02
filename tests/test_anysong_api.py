import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from soundalike.anysong import api

META = {
    1: ("Daft Punk", "One More Time"),
    2: ("Daft Punk", "One More Time (Radio Edit)"),
    3: ("Some Choir", "One More Time"),
    4: ("Daft Punk feat. Romanthony", "One More Time - 2021 Remaster"),
    5: ("Other Artist", "Different Song"),
    6: ("Other Artist", "Different Song [Live]"),
    7: ("Third", "Third Song"),
}


class FakeIndex:
    def __init__(self, ids):
        self.pos = {int(t): i for i, t in enumerate(ids)}

    def __len__(self):
        return len(self.pos)


class FakeEngine:
    device = "cpu"

    def __init__(self, ids=(1, 2, 3, 4, 5, 6, 7), rows=None, gate=None):
        self.indexes = {"melody": FakeIndex(ids), "samples": FakeIndex(ids)}
        self.rows = rows or [(2, 0.99), (3, 0.9), (4, 0.8), (5, 0.7), (6, 0.6), (7, 0.5)]
        self.gate = gate
        self.calls = []
        self.decoded = []

    def indexed(self, tid):
        return all(tid in ix.pos for ix in self.indexes.values())

    def _ranked(self, exclude):
        if self.gate is not None:
            self.gate.wait(5)
        return {ch: [r for r in self.rows if r[0] not in exclude] for ch in self.indexes}, {"search_melody": 1.0}

    def search_stored(self, tid, k):
        self.calls.append(("stored", tid, k))
        return self._ranked({tid})

    def decode(self, path):
        self.decoded.append(path)
        if path.read_bytes() == b"bad":
            raise RuntimeError("corrupt mp3")
        return {16000: np.zeros(4, np.float32)}, {"decode": 2.0}

    def embed_clips(self, clips):
        self.calls.append(("embed",))
        return {"melody": np.zeros((3, 4))}, {"embed_clews_dvi": 3.0}

    def search_windows(self, wins, k, exclude=()):
        self.calls.append(("windows", tuple(exclude), k))
        return self._ranked(set(exclude))


class FakeDeezer:
    def __init__(self, tmp_path, tracks=None, preview=b"mp3"):
        self.tmp_path, self.preview = tmp_path, preview
        self.tracks = tracks if tracks is not None else {
            99: {"id": 99, "title": "One More Time (Live)", "artist": {"name": "Daft Punk"},
                 "album": {"title": "Alive", "cover_medium": "http://c/99.jpg"}, "duration": 30, "preview": "u"},
        }
        self.fail = False
        self.searches = 0
        self.lookups = 0

    def track(self, tid):
        self.lookups += 1
        if self.fail:
            raise RuntimeError("down")
        return self.tracks.get(tid, {})

    def fetch_preview(self, tid, meta=None):
        if not (meta or {}).get("preview"):
            return None
        path = self.tmp_path / f"{tid}.mp3"
        path.write_bytes(self.preview)
        return path

    def search(self, q, limit):
        self.searches += 1
        if self.fail:
            raise RuntimeError("down")
        return [self.tracks[99], {"id": 5, "title": "Different Song", "artist": {"name": "Other Artist"},
                                  "album": {}, "preview": ""}, {"no": "id"}][:limit]


@pytest.fixture
def svc(tmp_path):
    return api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META)


@pytest.mark.parametrize("artist,title,expected", [
    ("Daft Punk", "One More Time (Radio Edit)", ("daft punk", "one more time")),
    ("Daft Punk feat. Romanthony", "One More Time - 2021 Remaster", ("daft punk", "one more time")),
    ("Lil Freaky", "Dripset (feat. Future)", ("lil freaky", "dripset")),
    ("Crosby, Stills, Nash & Young", "Looking Forward", ("crosby", "looking forward")),
    ("A", "(Intro)", ("a", "intro")),
])
def test_song_key_drops_version_tags(artist, title, expected):
    assert api.song_key(artist, title) == expected


def test_catalog_track_uses_stored_windows_and_hides_same_song_versions(svc):
    r = svc.similar(1, k=10)
    assert svc.engine.calls == [("stored", 1, api.FETCH)]
    assert r["query"] == {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog"}
    # 2 and 4 are other releases of the query; 6 is a live version of 5. The cover by another artist stays.
    assert [x["id"] for x in r["results"]["melody"]] == [3, 5, 7]
    assert r["results"]["melody"][0] == {"id": 3, "score": 0.9, "artist": "Some Choir", "title": "One More Time"}
    assert r["cached"] is False and "total" in r["timings_ms"] and "queue" in r["timings_ms"]


def test_dedupe_can_be_disabled_and_k_truncates(svc):
    assert [x["id"] for x in svc.similar(1, k=3, dedupe=False)["results"]["samples"]] == [2, 3, 4]


def test_results_are_cached_across_k_and_dedupe(svc):
    svc.similar(1, k=2)
    r = svc.similar(1, k=5, dedupe=False)
    assert len(svc.engine.calls) == 1
    assert r["cached"] is True and len(r["results"]["melody"]) == 5
    assert svc.health()["served"] == {"catalog": 1, "live": 0, "cached": 1}


def test_fuse_sums_weighted_reciprocal_ranks():
    fused = api.fuse({"melody": ["a", "b", "c"], "samples": ["c", "d"], "sonic": ["e"]},
                     {"melody": 1.0, "samples": 1.0}, k=1)
    # c: 1/4 + 1/2; a: 1/2; b and d tie at 1/3 with the same best rank, so first appearance wins
    assert [key for key, _, _ in fused] == ["c", "a", "b", "d"]
    assert fused[0][1] == pytest.approx(0.75) and fused[0][2] == {"melody": 3, "samples": 1}
    assert api.fuse({"melody": ["a"], "samples": ["b"]}, {"melody": 1.0, "samples": 2.0}, k=1)[0][0] == "b"
    assert api.fuse({"sonic": ["e"]}) == []


def test_best_list_fuses_the_song_views(svc):
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog"},
           "raw": {"melody": [(2, 0.99), (3, 0.9), (5, 0.8), (7, 0.7)],
                   "samples": [(7, 0.5), (6, 0.4), (3, 0.3)]},
           "timings_ms": {}}
    r = svc._view(res, k=10, dedupe=True, cached=False)
    assert list(r["results"]) == ["best", "melody", "samples"]
    best = r["results"]["best"]
    # 3 and 7 tie (1st and 3rd in each list) and beat 2nd + 2nd for any k; 6 is a live version of 5, so song 5
    # is 2nd in both lists
    assert [x["id"] for x in best] == [3, 7, 5]
    k = api.FUSE_K
    assert best[1] == {"id": 7, "score": round(1 / (k + 1) + 1 / (k + 3), 5), "artist": "Third",
                       "title": "Third Song", "ranks": {"melody": 3, "samples": 1}}
    assert best[2]["ranks"] == {"melody": 2, "samples": 2} and best[2]["title"] == "Different Song"
    assert [x["id"] for x in svc._view(res, k=2, dedupe=True, cached=False)["results"]["best"]] == [3, 7]
    assert [x["id"] for x in r["results"]["samples"]] == [7, 6, 3]


def test_no_best_list_without_fused_channels(svc):
    res = {"query": {"id": 1, "artist": "A", "title": "B", "source": "catalog"}, "raw": {"sonic": [(3, 0.9)]},
           "timings_ms": {}}
    assert list(svc._view(res, k=5, dedupe=True, cached=False)["results"]) == ["sonic"]


def test_unseen_track_is_fetched_embedded_and_excluded(svc, tmp_path):
    r = svc.similar(99, k=10)
    assert r["query"]["source"] == "live" and r["query"]["cover"] == "http://c/99.jpg"
    assert svc.engine.calls == [("embed",), ("windows", (99,), api.FETCH)]
    assert not (tmp_path / "99.mp3").exists()  # scratch preview removed after decoding
    assert [x["id"] for x in r["results"]["melody"]] == [3, 5, 7]  # query is a Daft Punk version too
    assert {"lookup", "fetch", "decode", "embed_clews_dvi", "search_melody", "total"} <= set(r["timings_ms"])
    assert svc.deezer.lookups == 1


def test_search_results_skip_the_track_lookup(svc):
    svc.search("one more time")
    r = svc.similar(99)
    assert r["query"]["source"] == "live" and svc.deezer.lookups == 0 and "lookup" not in r["timings_ms"]


def test_known_tracks_expire(tmp_path):
    now = [0.0]
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, known_ttl=60)
    svc.known.clock = lambda: now[0]
    svc.search("one more time")
    now[0] = 61
    svc.similar(99)
    assert svc.deezer.lookups == 1


def test_keep_previews_option(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, keep_previews=True)
    svc.similar(99)
    assert (tmp_path / "99.mp3").exists()


@pytest.mark.parametrize("setup,status", [
    (lambda s: None, 404),  # unknown track id
    (lambda s: s.deezer.tracks.__setitem__(42, {"id": 42, "title": "x", "preview": ""}), 404),
    (lambda s: setattr(s.deezer, "fail", True), 502),
    (lambda s: (s.deezer.tracks.__setitem__(42, {"id": 42, "title": "x", "preview": "u"}),
                setattr(s.deezer, "preview", b"bad")), 422),
])
def test_live_errors(svc, setup, status):
    setup(svc)
    with pytest.raises(api.ApiError) as err:
        svc.similar(42)
    assert err.value.status == status
    assert svc.results.get(42) is None and not svc._flight


def test_busy_when_pending_slots_are_full(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, max_pending=1)
    svc.pending.acquire()
    with pytest.raises(api.ApiError) as err:
        svc.similar(1)
    assert err.value.status == 503
    svc.pending.release()
    assert svc.similar(1)["ok"]


def test_concurrent_requests_for_one_track_compute_once(tmp_path):
    gate = threading.Event()
    svc = api.AnySong(FakeEngine(gate=gate), FakeDeezer(tmp_path), META)
    out = []
    threads = [threading.Thread(target=lambda: out.append(svc.similar(1, k=3))) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(100):
        if svc._flight:
            break
        threading.Event().wait(0.01)
    gate.set()
    for t in threads:
        t.join(5)
    assert len(out) == 4 and len(svc.engine.calls) == 1
    assert sorted(r["cached"] for r in out) == [False, True, True, True]


def test_search_cards_flag_catalog_tracks_and_cache(svc):
    r = svc.search("  one   more time ", limit=5)
    assert r["query"] == "one more time"
    assert r["tracks"] == [
        {"id": 99, "title": "One More Time (Live)", "artist": "Daft Punk", "album": "Alive",
         "cover": "http://c/99.jpg", "duration": 30, "has_preview": True, "in_catalog": False},
        {"id": 5, "title": "Different Song", "artist": "Other Artist", "album": "", "cover": "",
         "duration": None, "has_preview": False, "in_catalog": True},
    ]
    svc.search("ONE MORE TIME", limit=5)
    assert svc.deezer.searches == 1


@pytest.mark.parametrize("q", ["", "   ", "x" * 201])
def test_search_rejects_bad_queries(svc, q):
    with pytest.raises(api.ApiError) as err:
        svc.search(q)
    assert err.value.status == 400


def test_search_upstream_failure(svc):
    svc.deezer.fail = True
    with pytest.raises(api.ApiError) as err:
        svc.search("abc")
    assert err.value.status == 502


def test_lru_evicts_and_expires():
    now = [0.0]
    c = api.LRU(2, ttl=10, clock=lambda: now[0])
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1
    now[0] = 11
    assert c.get("a") is None and len(c) == 1


def test_token_bucket_bursts_refills_and_prunes():
    now = [0.0]
    b = api.TokenBucket(per_min=60, burst=2, clock=lambda: now[0], max_clients=3)
    assert [b.allow("x") for _ in range(3)] == [True, True, False]
    now[0] = 1.0
    assert b.allow("x") and not b.allow("x")
    assert b.allow("y") and b.allow("z")
    now[0] = 100.0
    assert b.allow("w")  # table was full: idle clients are forgotten
    assert set(b._buckets) == {"w"}


@pytest.mark.parametrize("raw,key", [
    ("203.0.113.5", "203.0.113.5"),
    (" 203.0.113.5 ", "203.0.113.5"),
    ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
    ("2001:db8:1:2:ffff:ffff:ffff:ffff", "2001:db8:1:2::/64"),
    ("::ffff:198.51.100.7", "198.51.100.7"),
    ("", None),
    ("not-an-ip", None),
    ("203.0.113.5, 10.0.0.1", None),
])
def test_client_key_normalises_addresses(raw, key):
    assert api.client_key(raw) == key


@pytest.fixture
def server(tmp_path):
    srv = api.Server(("127.0.0.1", 0), api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META),
                     search_per_min=600, similar_per_min=8, cors_origin="https://yassin.app",
                     client_header="CF-Connecting-IP")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def get(srv, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), json.loads(err.read())


def test_http_routes(server):
    code, headers, body = get(server, "/healthz")
    assert code == 200 and body["channels"] == {"melody": 7, "samples": 7}
    assert headers["Cache-Control"] == "no-store" and headers["Access-Control-Allow-Origin"] == "https://yassin.app"

    code, headers, body = get(server, "/api/anysong/similar?id=1&k=2")
    assert code == 200 and [x["id"] for x in body["results"]["melody"]] == [3, 5]
    assert headers["Cache-Control"] == "public, max-age=3600"
    assert headers["Content-Type"] == "application/json; charset=utf-8"

    code, _, body = get(server, "/api/anysong/search?q=one%20more%20time&limit=1")
    assert code == 200 and [t["id"] for t in body["tracks"]] == [99]


@pytest.mark.parametrize("path,status", [
    ("/nope", 404),
    ("/api/anysong/similar", 400),
    ("/api/anysong/similar?id=abc", 400),
    ("/api/anysong/similar?id=0", 400),
    ("/api/anysong/similar?id=1&k=51", 400),
    ("/api/anysong/similar?id=1&dedupe=2", 400),
    ("/api/anysong/similar?id=12345", 404),
    ("/api/anysong/search?q=", 400),
    ("/api/anysong/search?q=a&limit=26", 400),
    ("/api/anysong/similar?" + "&".join(f"a{i}=1" for i in range(9)), 400),
])
def test_http_errors(server, path, status):
    code, headers, body = get(server, path)
    assert code == status and body["ok"] is False and body["error"]
    assert headers["Cache-Control"] == "no-store"


def test_http_rate_limit_is_per_client_and_skips_cached_results(server):
    ip = {"CF-Connecting-IP": "203.0.113.5"}
    codes = [get(server, f"/api/anysong/similar?id={i}", ip)[0] for i in (1, 3, 5)]
    assert codes == [200, 200, 429]  # similar_per_min=8 -> burst of 2
    assert [get(server, "/api/anysong/similar?id=1&k=3", ip)[0] for _ in range(4)] == [200] * 4
    assert get(server, "/api/anysong/similar?id=5", {"CF-Connecting-IP": "203.0.113.6"})[0] == 200


def test_http_rate_limit_groups_ipv6_by_64_and_falls_back_to_peer(server):
    codes = [get(server, f"/api/anysong/similar?id={i}", {"CF-Connecting-IP": f"2001:db8:7:7::{i}"})[0]
             for i in (1, 3, 5)]
    assert codes == [200, 200, 429]  # rotating addresses inside one /64 shares a bucket
    assert get(server, "/api/anysong/similar?id=5", {"CF-Connecting-IP": "2001:db8:7:8::1"})[0] == 200
    codes = [get(server, f"/api/anysong/similar?id={i}", {"CF-Connecting-IP": "garbage"})[0] for i in (2, 4, 6)]
    assert codes[-1] == 429  # unparseable header -> keyed by the peer address, not a fresh bucket per value


def test_http_internal_errors_are_generic(server, capsys):
    server.service.engine.search_stored = lambda tid, k: 1 / 0
    code, _, body = get(server, "/api/anysong/similar?id=7")
    assert code == 500 and body == {"ok": False, "error": "internal error"}
    assert "ZeroDivisionError" in capsys.readouterr().err


def test_engine_search_stored_matches_live_search_on_same_windows():
    torch = pytest.importorskip("torch")
    from soundalike.anysong import serve

    rng = np.random.default_rng(3)
    ids = np.arange(500, 560, dtype=np.int64)
    eng = serve.Engine.__new__(serve.Engine)
    eng.device, eng.indexes, eng.channels = "cpu", {}, []
    raw = {}
    for name in ("melody", "samples", "sonic"):
        ch = serve.CHANNELS[name]
        raw[name] = rng.standard_normal((len(ids), 3, 16)).astype(np.float16)
        eng.channels.append(ch)
        eng.indexes[name] = serve.ChannelIndex.from_windows(ch, ids, raw[name])
    tid = int(ids[11])
    assert eng.indexed(tid) and not eng.indexed(1)
    stored, timings = eng.search_stored(tid, 10)
    live, _ = eng.search_windows({n: w[11] for n, w in raw.items()}, 10, [tid])
    assert set(timings) == {"search_melody", "search_samples", "search_sonic"}
    for name in raw:
        assert tid not in [t for t, _ in stored[name]]
        assert [t for t, _ in stored[name]] == [t for t, _ in live[name]]
        np.testing.assert_allclose([s for _, s in stored[name]], [s for _, s in live[name]], atol=2e-3)
    assert torch.is_tensor(eng.indexes["melody"].stored_query(tid))
