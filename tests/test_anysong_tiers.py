import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import api
from soundalike.anysong.scene import build as build_scene
from soundalike.anysong.tiers import TierGate

# Core catalog: 1-3. Added by the crawl: 5 and 6 by Other Artist (200), 7 by Third (300), 8 by Daft Punk (100).
META = {1: ("Daft Punk", "One More Time"), 2: ("Some Choir", "Choir Song"), 3: ("Third", "Third Song"),
        5: ("Other Artist", "Different Song"), 6: ("Other Artist", "Another Song"), 7: ("Third", "New Third Song"),
        8: ("Daft Punk", "New Daft Song")}
TRACKS = {1: 100, 2: 400, 3: 300, 5: 200, 6: 200, 7: 300, 8: 100}
CORE = [1, 2, 3]


def scene_table():
    # Daft Punk lists Third and 500; Other Artist lists 600; nobody's list names Other Artist except via 999 live.
    return build_scene(TRACKS, {100: [300, 500], 300: [100], 200: [600]}, {}, {})


class Index:
    def __init__(self, ids):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.pos = {int(t): i for i, t in enumerate(self.ids)}
        self.id_tensor = torch.as_tensor(self.ids)

    def __len__(self):
        return len(self.ids)


class Engine:
    """The API's engine interface over fixed rows, ranking only rows inside the ``allow`` masks."""

    device = "cpu"

    def __init__(self):
        self.indexes = {"melody": Index(sorted(META)), "samples": Index(sorted(META))}
        self.rows = [(2, 0.9), (5, 0.8), (6, 0.7), (7, 0.6), (8, 0.5), (3, 0.4), (1, 0.3)]
        self.allow = []

    def indexed(self, tid):
        return tid in self.indexes["melody"].pos

    def _ranked(self, exclude, allow):
        self.allow.append(allow)
        out = {}
        for ch, ix in self.indexes.items():
            ok = None if allow is None else allow[ch]
            out[ch] = [(t, s) for t, s in self.rows if t not in exclude and (ok is None or bool(ok[ix.pos[t]]))]
        return out, {}

    def search_stored(self, tid, k, allow=None):
        return self._ranked({tid}, allow)

    def decode(self, path):
        return {16000: np.zeros(4, np.float32)}, {"decode": 1.0}

    def embed_clips(self, clips):
        return {"melody": np.zeros((3, 4))}, {}

    def search_windows(self, wins, k, exclude=(), allow=None):
        return self._ranked(set(exclude), allow)


class Deezer:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.related = {999: [200]}
        self.fail_related = False

    def track(self, tid):
        return {"id": tid, "title": "Live Song", "artist": {"id": 999, "name": "New Artist"}, "album": {},
                "preview": "u"}

    def related_artists(self, artist_id):
        if self.fail_related:
            raise RuntimeError("down")
        return self.related.get(artist_id, [])

    def fetch_preview(self, tid, meta=None):
        path = self.tmp_path / f"{tid}.mp3"
        path.write_bytes(b"mp3")
        return path


def service(tmp_path, tiers=True):
    engine, scene = Engine(), scene_table()
    gate = TierGate(np.array(CORE), scene, engine.indexes) if tiers else None
    return api.AnySong(engine, Deezer(tmp_path), META, scene=scene, tiers=gate)


def ids(result, channel="melody"):
    return [x["id"] for x in result["results"][channel]]


def test_masks_cover_core_rows_and_the_seed_artists_scene():
    scene = scene_table()
    gate = TierGate(np.array(CORE), scene, {"melody": Index(sorted(META))})
    assert gate.sizes == {"melody": {"core": 3, "discovered": 4}}
    assert gate.artists(100) == [100, 300, 500]  # the table's own list for Daft Punk
    assert gate.artists(400, live_related=[200]) == [200, 400]  # no list in the table: the live one
    assert gate.artists(300, live_related=[200]) == [100, 300]  # the table's list wins
    assert gate.artists(None, live_related=[200]) == [] and gate.artists(600) == [600]
    order = sorted(META)
    allowed = lambda artists: [t for t, ok in zip(order, gate.mask("melody", artists).tolist()) if ok]  # noqa: E731
    assert allowed(gate.artists(100)) == [1, 2, 3, 7, 8]
    assert allowed([]) == CORE
    assert allowed([200]) == [1, 2, 3, 5, 6]
    bare = scene_table()
    bare.rel_ptr = bare.rel_idx = None  # a table built before related lists were stored
    with pytest.raises(ValueError):
        TierGate(np.array(CORE), bare, {"melody": Index(sorted(META))})
    # An index of core songs only has nothing to gate, so such a table serves it (the same unit file can roll back).
    everything = TierGate(np.array(sorted(META)), bare, {"melody": Index(sorted(META))})
    assert everything.sizes == {"melody": {"core": 7, "discovered": 0}}
    assert bool(everything.mask("melody", everything.artists(400)).all())


def test_catalog_seed_gets_core_songs_and_its_related_artists_new_songs(tmp_path):
    svc = service(tmp_path)
    r = svc.similar(1, k=10)
    assert ids(r) == [2, 7, 8, 3]  # Other Artist's new songs (5, 6) are not in Daft Punk's scene
    assert ids(r, "samples") == [2, 7, 8, 3] and {x["id"] for x in r["results"]["best"]} == {2, 3, 7, 8}
    assert ids(svc.similar(1, k=10, scene=False)) == [2, 7, 8, 3]  # gated before ranking, scene layer or not
    assert svc.health()["tiers"] == {"melody": {"core": 3, "discovered": 4}, "samples": {"core": 3, "discovered": 4}}
    # Without the gate every indexed song is a candidate.
    assert ids(service(tmp_path, tiers=False).similar(1, k=10)) == [2, 5, 6, 7, 8, 3]


def test_live_seed_uses_its_live_related_list_and_fails_closed_for_new_songs(tmp_path):
    svc = service(tmp_path)
    r = svc.similar(99, k=10)
    assert r["query"]["source"] == "live" and r["query"]["artist_id"] == 999
    assert ids(r) == [2, 5, 6, 3, 1]  # New Artist lists Other Artist; Third's and Daft Punk's new songs are out
    svc = service(tmp_path)
    svc.deezer.fail_related = True
    assert ids(svc.similar(99, k=10)) == [2, 3, 1]  # no related list: core songs only


def test_live_seed_embeds_while_its_related_list_is_fetched(tmp_path):
    import threading

    svc = service(tmp_path)
    embedded = threading.Event()
    embed, related = svc.engine.embed_clips, svc.deezer.related_artists

    def embed_clips(clips):
        embedded.set()
        return embed(clips)

    def related_artists(artist_id):  # Deezer answers only after the GPU has embedded the preview
        if not embedded.wait(5):
            raise RuntimeError("the related list was awaited before embedding")
        return related(artist_id)

    svc.engine.embed_clips, svc.deezer.related_artists = embed_clips, related_artists
    r = svc.similar(99, k=10)
    assert ids(r) == [2, 5, 6, 3, 1] and r["timings_ms"]["related"] >= 0
