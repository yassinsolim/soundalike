"""Title + artist lookups (``AnySong.resolve``, ``anysong.lookup``) against recorded Deezer answers: no network.

``fixtures/deezer_lookup.json.gz`` holds, for a list of requests in many scripts, every Deezer answer the resolver
read on 2026-11-23 (trimmed to the fields it reads), what the resolver before the wider lookup returned
(``before``) and what it returns now (``after``); both without a catalog, so every request goes through Deezer.
"""

import gzip
import json
from pathlib import Path

import pytest

from soundalike.anysong import api, lookup

FIXTURE = json.loads(gzip.decompress((Path(__file__).parent / "fixtures" / "deezer_lookup.json.gz").read_bytes()))
CASES = FIXTURE["cases"]


class NoEngine:
    device = "cpu"
    indexes = {"melody": ()}

    def indexed(self, tid):
        return False


def _request(path, params, language):
    """A request's key in the recording; the service caches searches without regard to case, so the same search
    recorded for one request answers another's."""
    params = {name: value.casefold() if name == "q" else value for name, value in (params or {}).items()}
    return json.dumps([path, params, language], ensure_ascii=False, sort_keys=True)


DEEZER = {_request(*json.loads(key)): answer for key, answer in FIXTURE["deezer"].items()}


class ReplayDeezer:
    """Deezer's answers as recorded; a request that was not recorded fails the test."""

    def __init__(self):
        self.requests = []

    def _get(self, path, params=None, language=None):
        self.requests.append(path)
        return DEEZER[_request(path, params, language)]

    def search(self, q, limit, language=None):
        return self._get("search", {"q": q, "limit": limit}, language).get("data", [])

    def search_artists(self, name, limit=5):
        return self._get("search/artist", {"q": name, "limit": limit}).get("data", [])

    def track(self, tid):
        return self._get(f"track/{tid}")

    def artist_top_tracks(self, artist_id, limit=100, language=None):
        return self._get(f"artist/{artist_id}/top", {"limit": limit}, language).get("data", [])


def _resolve(case, svc=None):
    svc = svc or api.AnySong(NoEngine(), ReplayDeezer(), {})
    try:
        return svc.resolve(case["artist"], case["title"])
    except api.ApiError as exc:
        assert exc.status == 404
        return None


def test_the_list_covers_scripts_credits_and_slips():
    notes = " ".join(case["note"] for case in CASES)
    assert len(CASES) >= 40 and sum(case["after"] is not None for case in CASES) >= 40
    for kind in ("secondary credit", "feat", "Hangul", "Arabic", "Cyrillic", "Devanagari", "Japanese", "Han ", "typo",
                 "remaster", "NEGATIVE"):
        assert kind in notes


@pytest.mark.parametrize("case", CASES, ids=[f"{case['artist']} - {case['title']}" for case in CASES])
def test_resolve_on_recorded_deezer_answers(case):
    assert _resolve(case) == case["after"]
    if case["before"] is not None:  # nothing that was found before the wider lookup is lost or changed
        assert case["after"] == case["before"]
    if case["note"].startswith("NEGATIVE"):  # never another song in place of "not found"
        assert case["after"] is None


def test_the_two_production_failures_resolve():
    found = {(case["artist"], case["title"]): case for case in CASES}
    peso, aimer = found[("Peso Pluma", "Ella Baila Sola")], found[("Aimer", "残響散歌")]
    assert peso["before"] is None and peso["found"] == ["Eslabon Armado", "Ella Baila Sola"]
    assert aimer["before"] is None and aimer["found"] == ["Aimer", "Zankyosanka"]
    assert sum(case["before"] is not None for case in CASES) < sum(case["after"] is not None for case in CASES)


def test_a_song_found_under_another_credit_answers_from_the_catalog():
    # The catalog holds the song under Deezer's credit and romanised title; the request names neither.
    class Indexed(NoEngine):
        def indexed(self, tid):
            return tid in (7, 8)

    meta = {7: ("Aimer", "Zankyosanka"), 8: ("Eslabon Armado", "Ella Baila Sola"), 9: ("Peso Pluma", "Bye")}
    svc = api.AnySong(Indexed(), ReplayDeezer(), meta)
    assert svc.resolve("Aimer", "残響散歌") == 7
    assert svc.resolve("Peso Pluma", "Ella Baila Sola") == 8  # the catalog's release, not the search hit 2193681047
    assert api.AnySong(NoEngine(), ReplayDeezer(), meta).resolve("Peso Pluma", "Ella Baila Sola") == 2193681047


def test_the_wider_lookup_only_runs_when_the_names_find_nothing_and_is_cached():
    svc = api.AnySong(NoEngine(), ReplayDeezer(), {})
    assert svc.resolve("The Weeknd", "Blinding Lights") == 908604612
    assert svc.deezer.requests == ["search"]  # found under its names: one search, as before
    del svc.deezer.requests[:]
    assert svc.resolve("Peso Pluma", "Ella Baila Sola") == 2193681047
    spent = list(svc.deezer.requests)
    assert spent[0] == "search" and any(path.startswith("track/") for path in spent)  # the credits were read
    assert svc.resolve("Peso Pluma", "Ella Baila Sola") == 2193681047 and svc.deezer.requests == spent
    # The track's Deezer object is kept for the live ranking that follows.
    assert svc.known.get(2193681047)["contributors"]


def test_a_failing_deezer_request_in_the_wider_lookup_is_a_502():
    class Down(ReplayDeezer):
        def track(self, tid):
            raise RuntimeError("down")

    with pytest.raises(api.ApiError) as err:
        api.AnySong(NoEngine(), Down(), {}).resolve("Peso Pluma", "Ella Baila Sola")
    assert err.value.status == 502


KEY = api._lookup_key


def _candidate(artist, title, artist_id=1, rank=100, contributors=None, also=()):
    candidate = lookup.Candidate(1, artist_id, rank, contributors=contributors)
    for names in ((artist, title), *also):
        candidate.named(*names)
    return candidate


def test_script_languages_and_title_forms():
    assert lookup.script_languages("Aimer 残響散歌") == "zh-TW,ja;q=0.9"
    assert lookup.script_languages("うっせぇわ") == "ja" and lookup.script_languages("봄날") == "ko"
    assert lookup.script_languages("Группа крови") == "ru" and lookup.script_languages("تملي معاك") == "ar"
    assert lookup.script_languages("तुम ही हो") == "hi" and lookup.script_languages("Beyoncé – Déjà Vu") is None
    # A bracketed part in another script is the title in another naming; a version or a credit is not.
    assert lookup.title_forms("Gangnam Style (강남스타일)", KEY) == {"gangnam style", "강남스타일"}
    assert lookup.title_forms("Song (Live) [feat. X]", KEY) == {"song"}
    assert lookup.title_forms("Ｓｏｎｇ - Remastered 2011", KEY) == {"song"}
    assert lookup.loose("تملى معاك") == lookup.loose("تملي معاك") and lookup.loose("Plain") == "Plain"
    assert lookup.credit_names("Queen & David Bowie", KEY) == {"queen", "david bowie"}
    assert lookup.credit_names("Mark Ronson feat. Bruno Mars", KEY) == {"mark ronson", "bruno mars"}


def test_title_evidence():
    assert lookup.title_evidence("残響散歌", _candidate("Aimer", "Zankyosanka", also=[("Aimer", "残響散歌")]), KEY) == "same"
    assert lookup.title_evidence("残響散歌", _candidate("Aimer", "Zankyosanka"), KEY) is None
    assert lookup.title_evidence("강남스타일", _candidate("Psy", "Gangnam Style (강남스타일)"), KEY) == "same"
    slip = _candidate("T", "The Less I Know The Better")
    assert lookup.title_evidence("The Less I Know The Beter", slip, KEY) == "close"
    # Short titles and titles that differ in a number are other songs, however alike they are spelled.
    assert lookup.title_evidence("Hello", _candidate("A", "Hallo"), KEY) is None
    assert lookup.title_evidence("Heaven Pt. 1", _candidate("A", "Heaven Pt. 2"), KEY) is None
    assert lookup.title_evidence("Shake It", _candidate("A", "Shake It Off"), KEY) is None


def test_artist_evidence():
    track = _candidate("Eslabon Armado", "Ella Baila Sola", artist_id=5, contributors=[(5, "Eslabon Armado"),
                                                                                  (9, "Peso Pluma")])
    assert lookup.artist_evidence("Eslabon Armado", track, (), KEY) == "name"
    assert lookup.artist_evidence("Peso Pluma & Eslabon Armado", track, (), KEY) == "name"
    assert lookup.artist_evidence("Peso Pluma", track, (), KEY) == "credit"
    assert lookup.artist_evidence("Someone", track, (9,), KEY) == "credit"  # the same Deezer artist, another name
    assert lookup.artist_evidence("周杰倫", _candidate("Jay Chou", "Sunny Day", artist_id=7), (7, 8), KEY) == "id"
    assert lookup.artist_evidence("Pink", _candidate("P!nk", "So What"), (), KEY) == "close"
    assert lookup.artist_evidence("Kesha", _candidate("Ke$ha", "TiK ToK"), (), KEY) == "close"
    assert lookup.artist_evidence("Pink Floyd", _candidate("P!nk", "So What"), (), KEY) is None
    assert lookup.artist_evidence("Eve", _candidate("Ava", "Song"), (), KEY) is None  # too short to be a slip
    assert lookup.artist_evidence("Peso Pluma", _candidate("Eslabon Armado", "Ella Baila Sola"), (), KEY) is None


def test_best_prefers_the_same_title_then_the_evidence_then_popularity():
    def numbered(i, *args, **kw):
        candidate = _candidate(*args, **kw)
        candidate.id = i
        return candidate

    cover = numbered(1, "Cover Band", "Under Pressure", rank=900)
    live = numbered(2, "David Bowie", "Under Pressure (Live)", rank=300)
    credited = numbered(3, "Queen", "Under Pressure", rank=800, contributors=[(4, "David Bowie")])
    slip = numbered(4, "David Bowie", "Under Presure", rank=950)
    order = [c.id for c in lookup.best("David Bowie", "Under Pressure", [cover, slip, credited, live], (), KEY)]
    assert order == [2, 3, 4]  # his own first, then the credit, then the title one slip off; never the cover
    assert [c.id for c in lookup.best("David Bowie", "Under Pressure", [cover, slip, credited, live], (), KEY,
                                      same_title=True)] == [2, 3]
    # A close title needs better artist evidence than a close spelling; two guesses are not a song.
    assert lookup.best("David Bowi", "Under Presure", [numbered(5, "David Bowie", "Under Pressure")], (), KEY) == []
    popular = numbered(6, "Queen", "Under Pressure", rank=999, contributors=[(4, "David Bowie")])
    assert [c.id for c in lookup.best("David Bowie", "Under Pressure", [credited, popular], (), KEY)] == [6, 3]
