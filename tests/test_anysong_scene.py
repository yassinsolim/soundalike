import numpy as np
import pytest
from dataclasses import replace

from soundalike.anysong import api
from soundalike.anysong.scene import (SceneParams, SceneTable, affinity, build, components, inferred,
                                      scene_fuse, standout)

OFF = SceneParams(boost=0.0, sonic=0.0, pattern="", max_same=10**6, max_other=10**6)


@pytest.fixture()
def table():
    # 10, 11 and 13 form a UK drill scene; 12 is a US trap artist; 13 has no stated country or genre.
    tracks = {1: 10, 2: 11, 3: 12, 4: 13, 5: 10, 6: 10, 7: 10, 8: 11, 9: 11}
    related = {10: [11, 13, 20], 11: [10, 13, 20], 12: [30, 31], 13: [10, 11]}
    countries = {10: ["GB"], 11: ["GB"], 12: ["US"], 20: ["GB"]}
    genres = {10: [("drill", 1.0), ("hiphop", 0.5)], 11: [("drill", 1.0), ("hiphop", 0.5)],
              12: [("trap", 1.0), ("hiphop", 0.5)], 20: [("grime", 1.0)]}
    return build(tracks, related, countries, genres)


def test_without_metadata_matches_api_fuse():
    rng = np.random.default_rng(0)
    for _ in range(50):
        keys = [f"s{i}" for i in range(40)]
        orders = {ch: list(rng.permutation(keys)[:rng.integers(5, 30)]) for ch in ("melody", "samples", "sonic")}
        expected = api.fuse({"melody": orders["melody"], "samples": orders["samples"]})
        assert scene_fuse(orders, {}, None, None, SceneParams()) == expected
        assert scene_fuse(orders, {k: 7 for k in keys}, None, None, OFF) == expected


def test_graph_region_and_genre_rank_scene_neighbours_higher(table):
    params = SceneParams()
    q = table.profile(10)
    uk, us = affinity(q, table.stored(11), 11, params), affinity(q, table.stored(12), 12, params)
    assert uk > us >= 0
    assert affinity(q, table.stored(10), 10, params) == params.same
    assert affinity(q, None, 99, params) == 0.0
    assert affinity(None, table.stored(11), 11, params) == 0.0
    graph, region, genre = components(q, table.stored(11))
    assert graph == pytest.approx(1.0) and region == pytest.approx(1.0)  # both: {10, 11, 13, 20}
    assert genre == pytest.approx(1.0, abs=1e-6)
    graph, region, genre = components(q, table.stored(12))
    assert graph == 0.0 and region == 0.0 and 0 < genre < 0.5  # shared parent genre only


def test_missing_country_and_genre_are_inferred_from_neighbours(table):
    p = table.stored(13)
    assert p.country == {"GB": pytest.approx(0.8)}
    assert max(p.genre, key=p.genre.get) == "drill"
    assert inferred([table.stored(10)]) == ({}, {})  # one neighbour is not enough support


def test_live_profile_for_an_artist_outside_the_table(table):
    p = table.profile(999, related=[10, 11, 999])
    assert p.artist_id == 999 and {10, 11, 999} <= p.neighbours
    assert p.country == {"GB": pytest.approx(0.8)}
    assert table.profile(None) is None
    assert table.profile(10, related=[12]).neighbours == table.stored(10).neighbours | {12}


def test_scene_boost_lifts_a_neighbour_and_sonic_needs_affinity(table):
    artists = {"us": 12, "uk": 11, "uk_sonic": 11, "unrelated_sonic": 12}
    orders = {"melody": ["us", "uk"], "samples": [], "sonic": ["unrelated_sonic", "uk_sonic"]}
    q = table.profile(10)
    plain = [k for k, _, _ in scene_fuse(orders, artists, q, table, OFF)]
    assert plain == ["us", "uk"]
    ranked = scene_fuse(orders, artists, q, table, SceneParams(region=0.0, genre=0.0, graph=1.0, sonic=1.0,
                                                               pattern=""))
    keys = [k for k, _, _ in ranked]
    assert keys[0] == "uk" and "uk_sonic" in keys and "unrelated_sonic" not in keys
    assert dict((k, r) for k, _, r in ranked)["uk_sonic"] == {"sonic": 2}


def test_standout_protects_a_likely_cover_from_scene_neighbours(table):
    rest = [f"x{i}" for i in range(18)]
    orders = {"melody": ["cover", *rest, "uk"], "samples": [], "sonic": []}
    scores = {"melody": [5.0] + [1.0 + 0.01 * (i % 5) for i in range(19)], "samples": []}
    artists = {"cover": 12, "uk": 11, **{x: 0 for x in rest}}
    q = table.profile(10)
    params = SceneParams(boost=10.0, sonic=0.0, region=0.0, genre=0.0, protect=0.0, pattern="")
    assert [k for k, _, _ in scene_fuse(orders, artists, q, table, params, scores=scores)][:2] == ["uk", "cover"]
    protected = SceneParams(boost=10.0, sonic=0.0, region=0.0, genre=0.0, protect=3.0, pattern="")
    assert [k for k, _, _ in scene_fuse(orders, artists, q, table, protected, scores=scores)][:2] == ["cover", "uk"]
    # protection needs a scene query, and scores aligned with the list
    plain = [k for k, _, _ in api.fuse({"melody": orders["melody"]})]
    assert [k for k, _, _ in scene_fuse(orders, artists, None, table, protected, scores=scores)] == plain
    assert standout({"melody": ["a", "b"]}, {"melody": [1.0]}) == {}
    z = standout({"melody": ["a", "b"], "samples": ["b", "a"]}, {"melody": [2.0, 0.0], "samples": [3.0, 1.0]})
    assert z == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}


SLOTS_V1 = SceneParams(boost=2.0, pattern="AAS", rank_evidence=3)  # the slots served until 2026-11-09


def test_default_slots_give_every_place_after_the_head_to_the_scene_lane(table):
    unrelated = [f"a{i}" for i in range(8)]
    orders = {"melody": [*unrelated, "uk"], "samples": [], "sonic": ["uk_sonic"]}
    artists = {**{x: 0 for x in unrelated}, "uk": 11, "uk_sonic": 11}
    flat = {"melody": [-2.0] * 9, "samples": []}  # no cover/sample evidence: the head is the first three songs
    keys = [k for k, _, _ in scene_fuse(orders, artists, table.profile(10), table, SceneParams(), scores=flat)]
    # the lane by affinity * (audio + sonic RRF): the sonic-only neighbour, then the melody one; then audio again
    assert keys[:6] == ["a0", "a1", "a2", "uk_sonic", "uk", "a3"]
    top3 = {"melody": [-2.0] * 9, "samples": []}
    orders_top3 = {"melody": [*unrelated, "uk"], "samples": ["s0"], "sonic": ["uk_sonic"]}
    # a top-3 place alone no longer extends the head (rank_evidence 0): only calibrated scores do
    assert [k for k, _, _ in scene_fuse(orders_top3, artists | {"s0": 0}, table.profile(10), table, SceneParams(),
                                        scores=top3 | {"samples": [0.1]})][3:5] == ["uk_sonic", "uk"]


def test_slots_keep_the_audio_head_and_interleave_the_scene_lane(table):
    unrelated = [f"a{i}" for i in range(8)]
    orders = {"melody": [*unrelated, "uk"], "samples": [], "sonic": ["uk_sonic"]}
    artists = {**{x: 0 for x in unrelated}, "uk": 11, "uk_sonic": 11}
    q = table.profile(10)
    flat = {"melody": [-2.0] * 9, "samples": []}  # no cover/sample evidence and no standout
    out = scene_fuse(orders, artists, q, table, SLOTS_V1, scores=flat)
    keys = [k for k, _, _ in out]
    # head of three audio songs; then A: the lifted scene neighbour, A: next audio song, S: the sonic-only neighbour
    assert keys[:6] == ["a0", "a1", "a2", "uk", "a3", "uk_sonic"]
    assert dict((k, r) for k, _, r in out)["uk_sonic"] == {"sonic": 1}
    assert [s for _, s, _ in out] == sorted((s for _, s, _ in out), reverse=True)  # positional scores
    evidence = {"melody": [-2.0] * 5 + [-1.4] + [-2.0] * 3, "samples": []}  # a5 reaches the cover threshold
    assert [k for k, _, _ in scene_fuse(orders, artists, q, table, SLOTS_V1, scores=evidence)][:6] == \
        unrelated[:6]
    plain = [k for k, _, _ in api.fuse({"melody": orders["melody"]})]
    assert [k for k, _, _ in scene_fuse(orders, artists, None, table, SLOTS_V1, scores=flat)] == plain


def test_a_channel_top3_song_extends_the_head(table):
    melody, samples = ["m0", "m1", "m2", "m3", "m4", "uk"], ["s0", "s1", "s2", "s3"]
    orders = {"melody": melody, "samples": samples, "sonic": ["uk_sonic"]}
    artists = {**{x: 0 for x in melody + samples}, "uk": 11, "uk_sonic": 11}
    q = table.profile(10)
    flat = {"melody": [-2.0] * len(melody), "samples": [0.1] * len(samples)}  # below both score thresholds
    baseline = [k for k, _, _ in api.fuse({"melody": melody, "samples": samples})]
    assert baseline[:6] == ["m0", "s0", "m1", "s1", "m2", "s2"]
    # s2 is third in the samples list, so the head reaches it and no scene song moves above it
    assert [k for k, _, _ in scene_fuse(orders, artists, q, table, SLOTS_V1, scores=flat)][:6] == baseline[:6]
    shallow = [k for k, _, _ in scene_fuse(orders, artists, q, table, replace(SLOTS_V1, rank_evidence=0),
                                           scores=flat)]
    assert shallow[:3] == baseline[:3] and shallow[3:6] == ["uk", "s1", "uk_sonic"]


def test_artist_caps_defer_extra_songs_without_dropping_them(table):
    orders = {"melody": ["a1", "a2", "a3", "b1", "b2", "b3", "c1"], "samples": [], "sonic": []}
    artists = {"a1": 10, "a2": 10, "a3": 10, "b1": 11, "b2": 11, "b3": 11, "c1": 12}
    q = table.profile(10)
    params = SceneParams(boost=0.0, sonic=0.0, max_same=2, max_other=1, window=4, pattern="")
    keys = [k for k, _, _ in scene_fuse(orders, artists, q, table, params)]
    assert keys[:4] == ["a1", "a2", "b1", "c1"]
    assert sorted(keys) == sorted(orders["melody"])


def test_table_round_trip_and_validation(table, tmp_path):
    path = tmp_path / "scene.npz"
    table.save(path)
    loaded = SceneTable.load(path)
    assert loaded.artist(1) == 10 and loaded.artist(42) is None
    assert loaded.stored(11) == table.stored(11)
    assert loaded.has_related(10) and not loaded.has_related(20) and not loaded.has_related(None)
    with pytest.raises(ValueError):
        SceneTable({**np.load(path), "track_ids": np.array([3, 1, 2], np.int64)})


def test_table_keeps_each_artists_own_related_list(table, tmp_path):
    assert table.related(10) == [11, 13, 20] and table.related(12) == [30, 31]
    assert table.related(20) is None and table.related(99) is None and table.related(None) is None
    assert table.stored(30).neighbours == {12, 30} and table.related(30) is None  # listed by 12, has no own list
    path = tmp_path / "scene.npz"
    table.save(path)
    assert SceneTable.load(path).related(13) == [10, 11]
    with np.load(path) as z:
        older = {name: z[name] for name in z.files if name not in ("rel_ptr", "rel_idx")}
    assert SceneTable(older).related(10) is None  # tables built before the lists existed still load
    with pytest.raises(ValueError):
        SceneTable({**older, "rel_ptr": np.zeros(3, np.int64), "rel_idx": np.zeros(0, np.int32)})


def test_sources_parse_wikidata_and_build_a_table(tmp_path):
    import json

    from soundalike.anysong import scene_sources
    from soundalike.anysong.scene import load_sources

    def entity(q):
        return {"value": f"http://www.wikidata.org/entity/{q}"}

    def fake(query):
        assert 'VALUES ?dz { "5" "6" }' in query
        return [{"dz": {"value": "5"}, "item": entity("Q1"), "p": {"value": "P27"}, "val": entity("Q145")},
                {"dz": {"value": "5"}, "item": entity("Q1"), "p": {"value": "P136"}, "val": entity("Q9")},
                {"dz": {"value": "5"}, "item": entity("Q1")}]

    facts = scene_sources.artist_facts([5, 6], query=fake)
    assert facts[5]["items"] == ["Q1"] and facts[5]["P27"] == ["Q145"] and facts[5]["P136"] == ["Q9"]
    assert facts[6] == {"dz": 6, "items": [], **{p: [] for p in scene_sources.PROPS}}

    (tmp_path / "deezer_tracks.jsonl").write_text(json.dumps({"track_id": 3, "artist_id": 6}) + "\n")
    (tmp_path / "deezer_related.jsonl").write_text(json.dumps({"artist_id": 5, "related": [[6, None, None]]}) + "\n")
    (tmp_path / "wikidata_artists.jsonl").write_text("\n".join(json.dumps(r) for r in facts.values()) + "\n")
    (tmp_path / "wikidata_values.json").write_text(json.dumps(
        {"countries": {"Q145": {"label": "United Kingdom", "iso": ["GB"], "parent_iso": []}},
         "genres": {"Q9": {"label": "UK drill", "parents": []}}}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"1": 5}))
    table = build(*load_sources(tmp_path, catalog))
    assert table.artist(1) == 5 and table.artist(3) == 6
    assert table.stored(5).country == {"GB": pytest.approx(1.0)} and table.stored(5).genre
    assert table.has_related(5) and not table.has_related(6)
