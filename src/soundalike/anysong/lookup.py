"""Matching a requested title and artist to Deezer tracks whose credit or naming differs from the request.

``AnySong.resolve`` first asks for the song under the names as given. This module decides what else is the same
song, from evidence only, so that a wrong song is never returned in place of "not found":

- **Title.** The same title key, in any naming Deezer holds for the track: its default one or the one it gives for
  the language of the request's script ("Zankyosanka" is 残響散歌 with ``Accept-Language: ja``), or an alternate
  naming in brackets in another script ("Gangnam Style (강남스타일)"). Arabic letters that are written two ways are
  folded. A title one or two letters off (a typing slip) counts as "close".
- **Artist.** The requested artist, or one of several requested, is the track's credited artist or one of the
  names in that credit ("name"); is the same Deezer artist as one that Deezer's artist search finds for the
  requested name, which knows other scripts ("id"); is among the track's contributors ("credit": "Ella Baila
  Sola" is credited to Eslabon Armado first, Peso Pluma second); or is spelled almost like the credit ("close":
  "Pink" for "P!nk").

A track is the song when its title is the same and there is any artist evidence, or when its title is close and
the artist evidence is better than a close spelling. Among several, the same title wins over a close one, then
the better artist evidence, then the more popular track (Deezer's rank).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

Key = Callable[[str, str], tuple[str, str]]  # ``api._lookup_key``

# The language a title or name in another script is most likely written in, as Deezer's Accept-Language wants it.
# Deezer answers 晴天 for "ja" and "Sunny Day" for every Chinese locale, so Han asks for both.
_SCRIPTS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("cjk", re.compile(r"[\u3040-\u30ff]"), ("ja",)),
    ("cjk", re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"), ("zh-TW", "ja")),
    ("hangul", re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"), ("ko",)),
    ("cyrillic", re.compile(r"[\u0400-\u052f]"), ("ru",)),
    ("arabic", re.compile(r"[\u0600-\u06ff\u0750-\u077f]"), ("ar",)),
    ("hebrew", re.compile(r"[\u0590-\u05ff]"), ("he",)),
    ("greek", re.compile(r"[\u0370-\u03ff]"), ("el",)),
    ("thai", re.compile(r"[\u0e00-\u0e7f]"), ("th",)),
    ("devanagari", re.compile(r"[\u0900-\u097f]"), ("hi",)),
)
_ARTIST_SPLIT = re.compile(r",|&|\s(?:x|feat\.?|ft\.?|featuring|and|with)\s")
_BRACKETED = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_ARABIC_MARKS = re.compile(r"[\u064b-\u065f\u0670\u0640]")  # vowel marks, the hamza and madda marks, the tatweel
_ARABIC_FOLDS = str.maketrans({"\u0671": "\u0627", "\u0649": "\u064a", "\u0629": "\u0647"})
_DIGITS = re.compile(r"\d+")
TITLE_CLOSE = 0.85  # similarity of two title keys that are one typing slip apart
TITLE_CLOSE_MIN = 6  # shorter titles are too easily another song's
ARTIST_CLOSE = 0.75  # "pink" and "p!nk", "kesha" and "ke$ha", "acdc" and "ac/dc"
ARTIST_CLOSE_MIN = 4
EVIDENCE = {"name": 3, "id": 3, "credit": 2, "close": 1}


def script_languages(text: str) -> str | None:
    """Languages for the scripts in ``text`` as an Accept-Language value ("zh-TW,ja;q=0.9"); None for Latin text."""
    languages: list[str] = []
    for _, pattern, names in _SCRIPTS:
        if pattern.search(text):
            languages.extend(name for name in names if name not in languages)
    if not languages:
        return None
    return ",".join(name if i == 0 else f"{name};q={1 - i / 10:.1f}" for i, name in enumerate(languages))


def script(text: str) -> str:
    """The first non-Latin script in ``text`` ("cjk" for Japanese and Chinese), else "latin"."""
    return next((name for name, pattern, _ in _SCRIPTS if pattern.search(text)), "latin")


def loose(text: str) -> str:
    """``text`` with the Arabic letters that are written two ways folded (alef forms, yeh, teh marbuta) and the
    vowel marks dropped; other text unchanged."""
    if not _ARABIC.search(text):
        return text
    folded = _ARABIC_MARKS.sub("", unicodedata.normalize("NFD", text)).translate(_ARABIC_FOLDS)
    return unicodedata.normalize("NFC", folded)


def credit_names(artist: str, key: Key) -> frozenset[str]:
    """The artist names in a credit or a request ("Queen & David Bowie", "A feat. B"), as lookup keys."""
    names = (key(part, "")[0] for part in _ARTIST_SPLIT.split(unicodedata.normalize("NFKC", artist).casefold()))
    return frozenset(loose(name) for name in names if name)


def title_forms(title: str, key: Key) -> frozenset[str]:
    """The names a title stands for, as lookup keys: its core (no version tags) and each bracketed part written in
    another script than the core, which is the title in another naming rather than a version or a credit."""
    title = unicodedata.normalize("NFKC", title)
    core = key("", title)[1]
    forms = {loose(core)} if core else set()
    for part in _BRACKETED.findall(title):
        name = key("", part)[1]
        if name and script(name) != script(core):
            forms.add(loose(name))
    return frozenset(forms)


def _spelling(name: str) -> str:
    return "".join(unicodedata.normalize("NFKC", name).casefold().split())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Candidate:
    """A Deezer track that may be the requested song. ``card``: its search card in Deezer's default naming (None
    while only another naming is known); ``namings``: every (artist, title) Deezer gave for it; ``contributors``:
    (artist id, name) of its credits once its details were read (None: not read)."""

    id: int
    artist_id: int | None
    rank: int
    card: dict | None = None
    namings: list[tuple[str, str]] = field(default_factory=list)
    contributors: list[tuple[int | None, str]] | None = None

    def named(self, artist: str, title: str) -> None:
        if (artist, title) not in self.namings:
            self.namings.append((artist, title))


def title_evidence(title: str, candidate: Candidate, key: Key) -> str | None:
    """"same", "close" or None: how the requested ``title`` matches one of the candidate's namings."""
    wanted = title_forms(title, key)
    core = loose(key("", unicodedata.normalize("NFKC", title))[1])
    close = False
    for _, named in candidate.namings:
        forms = title_forms(named, key)
        if wanted & forms:
            return "same"
        other = loose(key("", unicodedata.normalize("NFKC", named))[1])
        if min(len(core), len(other)) >= TITLE_CLOSE_MIN and _DIGITS.findall(core) == _DIGITS.findall(other) \
                and _ratio(core, other) >= TITLE_CLOSE:
            close = True
    return "close" if close else None


def artist_evidence(artist: str, candidate: Candidate, artist_ids: Iterable[int], key: Key) -> str | None:
    """"name", "id", "credit", "close" or None: the best evidence that the requested ``artist`` (one name or a
    credit of several) made the candidate. ``artist_ids``: the Deezer artists found for the requested name."""
    wanted, ids = credit_names(artist, key), set(artist_ids)
    if any(wanted & credit_names(name, key) for name, _ in candidate.namings):
        return "name"
    if candidate.artist_id in ids:
        return "id"
    for artist_id, name in candidate.contributors or ():
        if artist_id in ids or wanted & credit_names(name, key):
            return "credit"
    spelled = [_spelling(part) for part in _ARTIST_SPLIT.split(artist)]
    for name, _ in candidate.namings:
        for credit in _ARTIST_SPLIT.split(name):
            for want in spelled:
                credit_spelled = _spelling(credit)
                if min(len(want), len(credit_spelled)) >= ARTIST_CLOSE_MIN \
                        and _ratio(want, credit_spelled) >= ARTIST_CLOSE:
                    return "close"
    return None


def judge(artist: str, title: str, candidate: Candidate, artist_ids: Iterable[int], key: Key
          ) -> tuple[int, int, int] | None:
    """The candidate's standing as the requested song, larger is better: (2 for the same title or 1 for a close
    one, the artist evidence's weight, Deezer's rank); None when the evidence does not make it the song."""
    title_match = title_evidence(title, candidate, key)
    if title_match is None:
        return None
    weight = EVIDENCE.get(artist_evidence(artist, candidate, artist_ids, key) or "", 0)
    if weight == 0 or (title_match == "close" and weight < EVIDENCE["credit"]):
        return None
    return (2 if title_match == "same" else 1), weight, candidate.rank


def best(artist: str, title: str, candidates: Sequence[Candidate], artist_ids: Iterable[int], key: Key,
         same_title: bool = False) -> list[Candidate]:
    """The candidates that are the requested song, best first (``judge``; ties keep the given order);
    ``same_title``: only those with the same title and artist evidence better than a close spelling."""
    ids = set(artist_ids)
    judged = []
    for i, candidate in enumerate(candidates):
        standing = judge(artist, title, candidate, ids, key)
        if standing is None or (same_title and (standing[0] < 2 or standing[1] < EVIDENCE["credit"])):
            continue
        judged.append((standing, -i, candidate))
    return [candidate for _, _, candidate in sorted(judged, key=lambda row: row[:2], reverse=True)]
