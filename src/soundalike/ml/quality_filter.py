"""Approach 1 — Track quality filter: remove junk derivatives from candidates.

Many music catalogs (including the Deezer-harvested library) include derivative
tracks that should never appear as recommendations:

  * slowed + reverb TikTok edits of the same song
  * nightcore / sped-up copies
  * karaoke / backing-track versions
  * tribute-band recordings
  * instrumental versions that lose the vocal
  * seed-title mashups (e.g. "Money Trees x Blinding Lights")

These don't improve recommendations — they actively hurt quality by displacing
real alternatives.  This module provides a ``TitleQualityFilter`` that can be
used as a pre-filter on the candidate pool **before** ranking.

The filter uses fast regex matching and is numpy-friendly: ``keep_mask(titles)``
returns a boolean array suitable for index slicing.  It is intentionally
conservative (false negatives are better than false positives: better to miss
one karaoke track than to suppress a real recording).
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------

# Title-level patterns (case-insensitive).  A track is junk if ANY matches.
_TITLE_JUNK_PATTERNS: List[str] = [
    # TikTok audio edits
    r"\bslowed\b",
    r"\breverb\b",
    r"\bsped[- ]up\b",
    r"\bspeed[- ]up\b",
    r"\bnightcore\b",
    # Karaoke / backing tracks
    r"\bkaraoke\b",
    r"\bkaraōke\b",
    r"\bbacking\s+track\b",
    r"\binstrumental\s+(?:version|mix|cover|track)\b",
    r"\ba\s+cappella\b",
    # Cover / tribute copies
    r"^tribute\s+to\b",
    r"\btribute\s+version\b",
    r"\bcover\s+version\b",
    r"\bpiano\s+version\b",
    r"\bstring\s+(?:quartet|version)\b",
    r"\borchestral\s+version\b",
    r"\bremake\b",
    # TikTok / meme edits
    r"\bmarimba\s+remix\b",
    r"\bringtone\b",
    r"\b8\s*bit\s+(?:version|cover)\b",
    # Multi-song mashups with explicit "x" separator (not bands with "x" in name)
    r"\bx\s+\w.*\bx\s+\w",   # "Song A x Song B x Song C" style
    # Medleys
    r"\bmedley\b",
    r"\bmashup\b",
    r"\bsing-?along\b",
    r"\bsing\s+along\b",
    # Lofi re-releases that aren't original
    r"\blofi\s+(?:version|remix|cover|study)\b",
    r"\blo-fi\s+(?:version|remix|cover|study)\b",
]

# Artist-level patterns — if the ARTIST name contains these, all their tracks
# are junk.  We're stricter here because a "Karaoke Universe" style publisher
# never publishes real music.
_ARTIST_JUNK_PATTERNS: List[str] = [
    r"\bkaraoke\b",
    r"\bkaraōke\b",
    r"\btribute\b",
    r"\bcovers?\s+band\b",
    r"\bsound-?alike\b",
    r"\binstrumental\s+all\s+stars?\b",
    r"\bmarimba\s+remix\b",
    r"\bnightcore\b",
    r"\bslowed\b",
]

_TITLE_RE = re.compile("|".join(_TITLE_JUNK_PATTERNS), re.IGNORECASE)
_ARTIST_RE = re.compile("|".join(_ARTIST_JUNK_PATTERNS), re.IGNORECASE)


def _nfkd(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


class TitleQualityFilter:
    """Fast numpy-compatible junk track filter.

    Parameters
    ----------
    extra_title_patterns : list of str, optional
        Additional regex patterns to apply to titles.
    extra_artist_patterns : list of str, optional
        Additional regex patterns to apply to artist names.
    """

    def __init__(
        self,
        extra_title_patterns: Optional[List[str]] = None,
        extra_artist_patterns: Optional[List[str]] = None,
    ):
        tpats = _TITLE_JUNK_PATTERNS + (extra_title_patterns or [])
        apat = _ARTIST_JUNK_PATTERNS + (extra_artist_patterns or [])
        self._title_re = re.compile("|".join(tpats), re.IGNORECASE)
        self._artist_re = re.compile("|".join(apat), re.IGNORECASE)

    def is_junk(self, title: str, artist: str = "") -> bool:
        """Return True if this track should be filtered out."""
        t = _nfkd(str(title))
        a = _nfkd(str(artist))
        return bool(self._title_re.search(t) or self._artist_re.search(a))

    def keep_mask(
        self, titles: Sequence[str], artists: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Boolean mask: True = keep, False = junk.

        Parameters
        ----------
        titles : sequence of str
        artists : sequence of str, optional
            If omitted, only title patterns are applied.
        """
        n = len(titles)
        mask = np.ones(n, dtype=bool)
        artists_seq: Sequence[str] = artists if artists is not None else [""] * n
        for i, (t, a) in enumerate(zip(titles, artists_seq)):
            if self.is_junk(t, a):
                mask[i] = False
        return mask

    def seed_title_in_result(self, seed_title: str, result_title: str) -> bool:
        """True if the seed's title appears inside the result title.

        Catches "Money Trees x Blinding Lights" type mashups where the seed
        artist's own song title re-appears in a candidate.
        """
        def norm_t(s: str) -> str:
            s = _nfkd(s).casefold()
            # Strip parenthetical suffixes
            s = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", s)
            # Strip "- Remaster / 2011 / etc."
            s = re.sub(r"\s+-\s+.*$", "", s)
            return " ".join(s.split())

        st = norm_t(seed_title)
        rt = norm_t(result_title)
        if not st or not rt:
            return False
        if st == rt:
            return True
        # Catch a title embedded in a mashup/cover label and one-character
        # catalogue misspellings such as Ornithology/Orinthology.  The fuzzy
        # check is limited to longer titles to avoid suppressing unrelated
        # songs with generic one-word names.
        if st in rt or rt in st:
            return True
        return min(len(st), len(rt)) >= 8 and SequenceMatcher(None, st, rt).ratio() >= 0.90


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------
_DEFAULT_FILTER: Optional[TitleQualityFilter] = None


def default_filter() -> TitleQualityFilter:
    global _DEFAULT_FILTER
    if _DEFAULT_FILTER is None:
        _DEFAULT_FILTER = TitleQualityFilter()
    return _DEFAULT_FILTER


def keep_mask(titles: Sequence[str], artists: Optional[Sequence[str]] = None) -> np.ndarray:
    """Convenience wrapper using the module default filter."""
    return default_filter().keep_mask(titles, artists)
