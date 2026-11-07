import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from soundalike.anysong import crawl, scene
from soundalike.anysong.crawl import (AdaptiveLimiter, CrawlDB, core_title, junk_artist, junk_title, pick_tracks,
                                      song_cap, song_key, work_key)


def song(tid, artist_id, name, title, rank=500_000, duration=200, preview=True, readable=True):
    return {"id": tid, "title": title, "rank": rank, "duration": duration, "readable": readable,
            "preview": f"https://cdn.test/{tid}.mp3" if preview else "",
            "artist": {"id": artist_id, "name": name}, "album": {"id": tid * 10}}


TOP = {
    1: [song(100, 1, "Cat One", "Old Song"), song(110, 1, "Cat One", "Old Song (Live)"),
        song(111, 1, "Cat One", "New One")],
    3: [song(301, 3, "Three", "First"), song(302, 3, "Three", "First (Live)"),
        song(303, 3, "Three", "Second"), song(304, 3, "Three", "Third")],
    4: [song(401, 4, "Four", "Hit"), song(402, 4, "Four", "Hit (Karaoke Version)"),
        song(100, 4, "Four", "Old Song"), song(403, 4, "Four", "Gone")],
    5: [song(501, 5, "Five", "Deep Cut"), song(502, 5, "Five", "Other Cut")],
}
RELATED = {
    1: [{"id": 3, "name": "Three", "nb_fan": 20000}, {"id": 4, "name": "Four", "nb_fan": 300}],
    2: [{"id": 3, "name": "Three", "nb_fan": 20000}],
    3: [{"id": 5, "name": "Five", "nb_fan": 5000}, {"id": 6, "name": "Six", "nb_fan": 10},
        {"id": 1, "name": "Cat One", "nb_fan": 90000}],
    4: [{"id": 5, "name": "Five", "nb_fan": 5000}],
    5: [],
}


def fake_client(top=TOP, related=RELATED, fail_top=(), fail_preview=None, no_preview=(), tracks=None,
                fail_lookup=()):
    calls, fail_top, fail_preview = [], set(fail_top), Counter(fail_preview or {})
    tracks, fail_lookup = tracks or {}, set(fail_lookup)

    class FakeDeezer:
        def __init__(self, cache_dir, **kwargs):
            self.cache = Path(cache_dir)

        def artist_top_tracks(self, artist_id, limit=100):
            calls.append(("top", artist_id))
            if artist_id in fail_top:
                fail_top.discard(artist_id)
                raise RuntimeError("deezer outage")
            return top.get(artist_id, [])

        def related(self, artist_id, limit=50):
            calls.append(("related", artist_id))
            return related.get(artist_id, [])

        def track(self, track_id):
            calls.append(("track", track_id))
            if track_id in fail_lookup:
                fail_lookup.discard(track_id)
                raise RuntimeError("lookup outage")
            return tracks.get(track_id, {})

        def preview_path(self, tid):
            return self.cache / f"{tid % 1000:03d}" / f"{tid}.mp3"

        def fetch_preview(self, tid, meta=None):
            calls.append(("preview", tid, meta is not None))
            if fail_preview[tid] > 0:
                fail_preview[tid] -= 1
                raise RuntimeError("cdn outage")
            if tid in no_preview:
                return None
            path = self.preview_path(tid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 10_001)
            return path

    return FakeDeezer, calls


def setup(tmp_path, **overrides):
    scene.build({100: 1, 101: 2}, {1: [3, 4], 2: [3]}, {}, {}).save(tmp_path / "scene.npz")
    np.savez(tmp_path / "meta.npz", track_ids=np.array([100, 101], dtype=np.int64),
             artists=np.array(["Cat One", "Cat Two"], dtype=object), titles=np.array(["Old Song", "Other"], dtype=object))
    values = dict(scene=tmp_path / "scene.npz", meta=tmp_path / "meta.npz", db=tmp_path / "crawl.sqlite",
                  previews=tmp_path / "previews", target=1000, per_artist=25, min_rank=1000, min_fans=100, rate=1.0,
                  fetchers=1, preview_rate=50.0, workers=2, max_pending=8, max_attempts=3, min_free_gb=0.0,
                  max_artists=0, log_every=3600.0, stop_timeout=5.0, learn_queue=None, learn_every=600.0,
                  keep_learning=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def run(args, client):
    lines = []
    reason = crawl.Crawler(args, client=client, log=lines.append).run()
    return reason, lines


def statuses(db_path, table):
    with CrawlDB(db_path, readonly=True).conn as conn:
        return dict(conn.execute(f"SELECT id, status FROM {table}").fetchall())


def requests(calls):
    return [c for c in calls if c[0] != "preview"]


def test_core_title_collapses_same_recording_variants_but_keeps_remixes():
    same = ["Pale Shelter", "Pale Shelter (Long Version)", "Pale Shelter - Live at Wembley",
            "Pale Shelter [2011 Remaster]", "Pale Shelter (feat. Someone)", "Pale Shelter (Radio Edit)",
            "Pale Shelter (Extended Mix)", "Pale Shelter (Acoustic)", "Pale Shelter (Taylor's Version)",
            "Pale Shelter (En Vivo)", 'Pale Shelter - From "Some Film"']
    assert {core_title(t) for t in same} == {"pale shelter"}
    distinct = ["Pale Shelter (Jos & Eli Remix)", "Pale Shelter (Sex Schön Mix)", "Pale Shelter (Interlude)",
                "Pale Shelter, Pt. 2"]
    assert len({core_title(t) for t in distinct} | {"pale shelter"}) == 5
    assert core_title("Make It High (Sex Schön Mix) [SM Edit] (Live)") == core_title("Make It High (Sex Schön Mix)")
    assert song_key("Tears For Fears", "Pale Shelter (Live)") == song_key("tears for fears", "Pale Shelter")


def test_junk_uploads_and_channels_are_recognised_without_catching_real_titles():
    junk = ["Song (Karaoke Version)", "Song (Sped Up)", "Song - Slowed + Reverb", "Song (Instrumental)",
            "963 Hz Equilibrio Pineal", "Song (8D Audio)", "Song (Made Famous by Someone)", "Song [A Cappella]",
            "Song (Nightcore)", "Song - Tribute to Someone"]
    assert all(junk_title(t) for t in junk)
    assert junk_title("Song", version="(Sped Up Version)")
    real = ["Lullaby", "Tribute", "Instrumental", "Slow Dance", "Reverb Nation", "Song (Remix)", "8 Mile"]
    assert not any(junk_title(t) for t in real)
    assert all(junk_artist(n) for n in ["Karaoke Hits Band", "White Noise Baby Sleep", "Relaxing Music Therapy",
                                        "Lullaby Babies", "432 Hz Healing"])
    assert not any(junk_artist(n) for n in ["Sleep Token", "Tributary", "Rain", "Study Hall", "Mac Miller"])


def test_pick_tracks_keeps_the_artists_own_playable_songs_once_most_popular_first():
    top = [song(1, 10, "A", "Hit"), song(2, 10, "A", "Hit (Live)"), song(3, 99, "B", "Feature"),
           song(4, 10, "A", "No Preview", preview=False), song(5, 10, "A", "Locked", readable=False),
           song(6, 10, "A", "Skit", duration=30), song(7, 10, "A", "Obscure", rank=50),
           song(8, 10, "A", "Hit (Karaoke Version)"), song(9, 10, "A", "Already Here"),
           song(10, 10, "A", "Catalog Id"), song(11, 10, "A", "Second"), song(12, 10, "A", "Third")]
    taken = {song_key("A", "Already Here")}
    picked = pick_tracks(10, top, taken, {10}, per_artist=2, min_rank=1000)
    assert [t["id"] for t in picked] == [1, 11]
    assert picked[0] | {"key": None} == {"id": 1, "artist_id": 10, "artist": "A", "title": "Hit", "key": None,
                                         "rank": 500_000, "duration": 200, "album_id": 10,
                                         "preview": "https://cdn.test/1.mp3"}
    assert {song_key("A", "Hit"), song_key("A", "Second")} <= taken
    assert song_key("A", "Third") not in taken


def test_classical_works_are_kept_for_their_first_performer_only():
    title = "Piano Sonata No. 5 in G Major, K. 283: II. Andante"
    taken = set()
    first = pick_tracks(1, [song(1, 1, "Pianist A", title)], taken, set(), 5, 1000)
    second = pick_tracks(2, [song(2, 2, "Pianist B", title),
                             song(3, 2, "Pianist B", title.replace("II. Andante", "III. Presto"))],
                         taken, set(), 5, 1000)
    assert [t["id"] for t in first] == [1] and [t["id"] for t in second] == [3]
    assert work_key("Cello Suite No. 1 in G Major, BWV 1007: I. Prélude")
    assert work_key("String Quartet in F Major, Op. 3, No. 5, Hob. III:17")
    assert not any(work_key(t) for t in ["Hey Jude", "Op. Cit", "Dr. Feelgood", "K.O.", "Vol. 2"])


def test_song_cap_grows_with_fans():
    fans = (-1, 0, 50, 100, 1_000, 10_000, 100_000, 1_000_000, 50_000_000)
    assert [song_cap(f, 25) for f in fans] == [12, 4, 7, 8, 12, 16, 20, 24, 25]
    assert song_cap(1_000_000, 10) == 10


def test_adaptive_pacing_slows_once_per_burst_and_recovers_when_quiet():
    pacing = AdaptiveLimiter(16.0, floor_per_s=1.0, quiet_s=60.0, burst_s=5.0)
    pacing.limited(100.0)
    pacing.limited(101.0)
    assert pacing.slowdowns == 1 and round(pacing.rate, 2) == 12.8
    pacing.limited(110.0)
    assert pacing.slowdowns == 2 and round(pacing.rate, 2) == 10.24
    pacing.recover(150.0)
    assert round(pacing.rate, 2) == 10.24
    pacing.recover(170.0)
    assert round(pacing.rate, 2) == 11.26
    pacing.recover(200.0)
    assert round(pacing.rate, 2) == 11.26
    for t in range(230, 2000, 60):
        pacing.recover(float(t))
    assert pacing.rate == 16.0
    for t in range(3000, 4000, 10):
        pacing.limited(float(t))
    assert pacing.rate == 1.0


def test_seed_queues_catalog_artists_then_the_artists_their_related_lists_link_to(tmp_path):
    args = setup(tmp_path)
    db = CrawlDB(args.db)
    assert db.seed(scene.SceneTable.load(args.scene)) == (2, 2)
    with db.conn as conn:
        rows = conn.execute("SELECT id, hop, links, fans, status FROM artists ORDER BY id").fetchall()
    assert rows == [(1, 0, 0, -1, "frontier"), (2, 0, 0, -1, "frontier"), (3, 1, 2, -1, "frontier"),
                    (4, 1, 1, -1, "frontier")]
    assert db.next_artist(100) == (1, None, 0, -1, 0)
    assert db.next_artist(100, exclude={1, 2}) == (3, None, 1, -1, 0)


def test_crawl_tops_up_catalog_artists_takes_popular_artists_next_resumes_and_exports(tmp_path):
    args = setup(tmp_path, max_artists=3)
    client, calls = fake_client(fail_preview={303: 2})
    reason, lines = run(args, client)
    assert reason == "artist limit reached"
    assert requests(calls) == [("top", 1), ("related", 1), ("top", 2), ("related", 2), ("top", 3), ("related", 3)]
    assert [c for c in calls if c[:2] == ("preview", 303)] == [("preview", 303, True), ("preview", 303, False)]
    assert statuses(args.db, "tracks") == {111: "ok", 301: "ok", 303: "error", 304: "ok"}
    assert any("fetching 1 failed previews once more" in line for line in lines)
    assert any(line.startswith("done ") for line in lines)

    args.max_artists = 0
    client, calls = fake_client(no_preview={403})
    reason, _ = run(args, client)
    assert reason == "frontier exhausted"
    assert ("preview", 303, False) in calls
    assert requests(calls) == [("top", 5), ("related", 5), ("top", 4), ("related", 4)]
    assert statuses(args.db, "tracks") == {111: "ok", 301: "ok", 303: "ok", 304: "ok", 401: "ok",
                                           403: "no_preview", 501: "ok", 502: "ok"}
    assert statuses(args.db, "artists") == {1: "done", 2: "done", 3: "done", 4: "done", 5: "done", 6: "frontier"}
    assert (tmp_path / "previews" / "501" / "501.mp3").exists()

    report = crawl.status(args.db)
    assert report["tracks"] == {"ok": 7, "no_preview": 1}
    assert report["crawled_by_hop"] == {0: 2, 1: 2, 2: 1}

    out = tmp_path / "export"
    assert crawl.export(args.db, out) == {"tracks": 7, "artists_with_related": 5}
    with np.load(out / "catalog_meta.npz", allow_pickle=True) as z:
        assert z["track_ids"].tolist() == [111, 301, 303, 304, 401, 501, 502]
        assert z["titles"].tolist()[:2] == ["New One", "First"]
    assert (out / "ids.txt").read_text().split() == ["111", "301", "303", "304", "401", "501", "502"]
    tracks = {row["track_id"]: row for row in map(json.loads, (out / "deezer_tracks.jsonl").read_text(
        encoding="utf-8").splitlines())}
    assert tracks[301] == {"track_id": 301, "artist_id": 3, "artist_name": "Three", "album_id": 3010,
                           "rank": 500_000, "duration": 200}
    related = {r["artist_id"]: r["related"] for r in map(json.loads, (out / "deezer_related.jsonl").read_text(
        encoding="utf-8").splitlines())}
    assert related == {1: [[3, "Three", 20000], [4, "Four", 300]], 2: [[3, "Three", 20000]],
                       3: [[5, "Five", 5000], [6, "Six", 10], [1, "Cat One", 90000]], 4: [[5, "Five", 5000]], 5: []}


def test_failed_artist_is_set_aside_then_retried_and_junk_channels_are_skipped(tmp_path):
    top = {**TOP, 4: [song(401, 4, "Karaoke Kings", "Hit")]}
    args = setup(tmp_path)
    client, calls = fake_client(top=top, fail_top={3})
    reason, lines = run(args, client)
    assert reason == "frontier exhausted"
    assert requests(calls) == [("top", 1), ("related", 1), ("top", 2), ("related", 2), ("top", 3), ("top", 4),
                               ("top", 3), ("related", 3), ("top", 5), ("related", 5)]
    assert statuses(args.db, "artists")[4] == "skipped"
    assert statuses(args.db, "tracks") == {111: "ok", 301: "ok", 303: "ok", 304: "ok", 501: "ok", 502: "ok"}
    with CrawlDB(args.db, readonly=True).conn as conn:
        assert conn.execute("SELECT attempts, error FROM artists WHERE id = 3").fetchone() == (1, None)
    assert any("artist 3 failed (1 in a row): deezer outage" in line for line in lines)


def test_crawl_stops_at_the_target_and_counts_only_saved_or_pending_songs(tmp_path):
    args = setup(tmp_path, target=4, max_pending=1)
    client, calls = fake_client(no_preview={301})
    reason, _ = run(args, client)
    assert reason == "target reached"
    assert [c[1] for c in calls if c[0] == "top"] == [1, 2, 3, 5]
    assert statuses(args.db, "tracks")[301] == "no_preview"


def test_parallel_fetchers_crawl_every_artist_once(tmp_path):
    args = setup(tmp_path, fetchers=3)
    client, calls = fake_client()
    assert run(args, client)[0] == "frontier exhausted"
    assert sorted(c[1] for c in calls if c[0] == "top") == [1, 2, 3, 4, 5]
    assert statuses(args.db, "artists") == {1: "done", 2: "done", 3: "done", 4: "done", 5: "done", 6: "frontier"}


def test_song_cap_counts_the_songs_an_artist_already_has_in_the_catalog(tmp_path):
    args = setup(tmp_path, per_artist=2)
    client, _ = fake_client()
    run(args, client)
    tracks = statuses(args.db, "tracks")
    assert 111 not in tracks
    assert {301, 303} <= set(tracks) and 304 not in tracks


LEARNED = {701: song(701, 7, "Seven", "Searched Song", rank=50), 702: song(702, 7, "Seven", "Seven Hit")}
LEARN_TOP = {**TOP, 7: [LEARNED[702]], 8: [song(801, 8, "Eight", "Eight Hit")]}
LEARN_RELATED = {**RELATED, 7: [{"id": 8, "name": "Eight", "nb_fan": 500}, {"id": 9, "name": "Nine", "nb_fan": 10}]}


def write_queue(path, rows, partial=None):
    text = "".join(json.dumps(r) + "\n" for r in rows) + (partial or "")
    path.write_text(text, encoding="utf-8")
    return len(text.encode()) - len((partial or "").encode())


def test_looked_up_songs_and_their_artists_go_first_then_their_neighbours(tmp_path):
    queue = tmp_path / "learn.jsonl"
    complete = "".join(line + "\n" for line in (json.dumps({"id": 701, "artist_id": 7, "day": "2026-11-07"}),
                                                "not json",
                                                json.dumps({"id": 100, "artist_id": 1, "day": "2026-11-07"})))
    queue.write_bytes((complete + '{"id": 70').encode())  # the API is still writing the last line
    args = setup(tmp_path, learn_queue=queue)
    client, calls = fake_client(top=LEARN_TOP, related=LEARN_RELATED, tracks=LEARNED)
    assert run(args, client)[0] == "frontier exhausted"
    assert [c[1] for c in calls if c[0] == "track"] == [701]  # catalog song 100 needs no lookup
    # Seed artists (1 via its catalog song, 7 via the new song), then their neighbours by fans (3, 8, 4; Nine has
    # too few fans), then the bulk crawl (catalog artist 2, then Five).
    assert [c[1] for c in calls if c[0] == "top"] == [1, 7, 3, 8, 4, 2, 5]
    tracks = statuses(args.db, "tracks")
    assert tracks[701] == "ok" and tracks[702] == "ok" and tracks[801] == "ok" and 9 not in tracks
    with CrawlDB(args.db, readonly=True).conn as conn:
        assert conn.execute("SELECT id FROM tracks WHERE learned = 1").fetchall() == [(701,)]
        offset = int(conn.execute("SELECT value FROM settings WHERE key = 'learn_offset'").fetchone()[0])
    assert offset == len(complete.encode())
    report = crawl.status(args.db)
    assert report["learned"]["songs"] == {"ok": 1} and report["learned"]["artists_waiting"] == 0


def test_learning_resumes_after_a_lookup_failure_and_waits_for_partial_lines(tmp_path):
    queue = tmp_path / "learn.jsonl"
    write_queue(queue, [{"id": 701, "artist_id": 7, "day": "2026-11-07"}], partial='{"id": 702, "artist_id": 7}')
    args = setup(tmp_path, learn_queue=queue, max_artists=1)
    client, calls = fake_client(top=LEARN_TOP, related=LEARN_RELATED, tracks=LEARNED, fail_lookup={701})
    run(args, client)
    assert 701 not in statuses(args.db, "tracks")
    with CrawlDB(args.db, readonly=True).conn as conn:
        assert conn.execute("SELECT value FROM settings WHERE key = 'learn_offset'").fetchone() is None
    with queue.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    client, calls = fake_client(top=LEARN_TOP, related=LEARN_RELATED, tracks=LEARNED)
    run(args, client)
    assert [c[1] for c in calls if c[0] == "track"] == [701, 702]
    tracks = statuses(args.db, "tracks")
    assert tracks[701] == "ok" and tracks[702] == "ok"
    assert [c[1] for c in calls if c[0] == "top"] == [7]


def test_keep_learning_keeps_running_after_the_target_for_new_lookups(tmp_path):
    import threading
    import time

    queue = tmp_path / "learn.jsonl"
    queue.write_text("", encoding="utf-8")
    args = setup(tmp_path, learn_queue=queue, learn_every=0.05, keep_learning=True, target=1)
    client, calls = fake_client(top=LEARN_TOP, related=LEARN_RELATED, tracks=LEARNED)
    holder, result = {}, {}

    def go():  # SQLite connections belong to the thread that opened them, as in main()
        holder["crawler"] = crawl.Crawler(args, client=client, log=lambda line: None)
        result["reason"] = holder["crawler"].run()

    worker = threading.Thread(target=go)
    worker.start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not any(c[0] == "top" for c in calls):
            time.sleep(0.02)
        time.sleep(0.3)
        assert [c[1] for c in calls if c[0] == "top"] == [1]  # target reached after one artist; bulk crawl stops
        with queue.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": 701, "artist_id": 7, "day": "2026-11-07"}) + "\n")
        while time.time() < deadline and ("preview", 702, True) not in calls:
            time.sleep(0.02)
        assert ("track", 701) in calls and ("preview", 702, True) in calls
    finally:
        while "crawler" not in holder and worker.is_alive():
            time.sleep(0.01)
        if "crawler" in holder:
            holder["crawler"].stop.set()
        worker.join(30)
    assert not worker.is_alive() and result["reason"] == "stopped"
    assert [c[1] for c in calls if c[0] == "top"] == [1, 7, 8]
