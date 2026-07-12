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
from typing import Any, Iterable, List, Mapping, Optional, Sequence

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
    r"\bnightcore(?:d|'d)?\b",
    # Karaoke / backing-track variants.  Keep legitimate originals titled
    # "Karaoke", "Karaoke Bar", etc.; derivative context is required.
    r"\bkara(?:oke|ōke)\s+(?:version|mix|edit|track|instrumental)\b",
    r"(?:\(|\[|\s+-\s*)kara(?:oke|ōke)(?:\s+version)?(?:\)|\]|$)",
    r"\b(?:unreleased|official|instrumental)\s+kara(?:oke|ōke)\b",
    r"\bbacking\s+track\b",
    r"\binstrumental\s+(?:version|mix|cover|track)\b",
    r"\ba\s+cappella\b",
    # Cover / tribute copies
    r"^tribute\s+to\b",
    r"\btribute\s+(?:version|recording)\b",
    r"\bcover\s+(?:version|record)\b",
    r"\(\s*cover(?:\s+of\b[^)]*)?\s*\)",
    r"\s+-\s+cover(?:\s+of\b.*)?$",
    r"\boriginally\s+performed\s+by\b",
    r"\bin\s+the\s+style\s+of\b",
    r"\bas\s+made\s+famous\s+by\b",
    r"\bpiano\s+version\b",
    r"\bstring\s+(?:quartet|version)\b",
    r"\borchestral\s+version\b",
    r"\bremake\b",
    # Mix/remix/version suffixes.  These are intentionally contextual rather
    # than matching the bare word "mix" (legitimate originals include titles
    # such as "Mixed Emotions").  Query-aware callers may allow the same class
    # when the seed itself is a remix/edit; see ``is_eligible_for_query``.
    r"(?:\(|\[)[^)\]]*\b(?:re)?mix(?:es)?\b[^)\]]*(?:\)|\])",
    r"\s+-\s+[^-]*\b(?:re)?mix(?:es)?\b[^-]*$",
    r"\b(?:club|radio|extended|dub|dance|house|vocal)\s+mix\b",
    r"(?:\(|\[|\s+-\s*)[^)\]]*\b(?:rework|bootleg|vip|edit)\b[^)\]]*(?:\)|\]|$)",
    r"\bchopnotslop\b",
    # TikTok / meme edits
    r"\bmarimba\s+remix\b",
    r"\bringtone\b",
    r"\b8\s*bit\s+(?:version|cover)\b",
    # Require at least two words on each side; this catches explicit song-title
    # mashups without suppressing legitimate titles such as "Love X Love".
    r"\b\w[\w']*\s+\w[\w' -]*\s+x\s+(?![^()]*\bremix\b)\w[\w']*\s+\w[\w' -]*\b",
    r"\bx\s+\w.*\bx\s+\w",
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
    r"\btribute\s+(?:to|band|artists?)\b",
    r"\bcovers?\s+band\b",
    r"\boriginally\s+performed\s+by\b",
    r"\bin\s+the\s+style\s+of\b",
    r"\bsound-?alike\b",
    r"\binstrumental\s+all\s+stars?\b",
    r"\bmarimba\s+remix\b",
    r"\bnightcore(?:d|'d)?\b",
    r"\bslowed\b",
]

_TITLE_RE = re.compile("|".join(_TITLE_JUNK_PATTERNS), re.IGNORECASE)
_ARTIST_RE = re.compile("|".join(_ARTIST_JUNK_PATTERNS), re.IGNORECASE)

# Stable labels used by the query-aware exception and canonical preference.
_VERSION_PATTERNS = {
    "slowed": re.compile(r"\bslowed\b", re.IGNORECASE),
    "reverb": re.compile(r"\breverb\b", re.IGNORECASE),
    "nightcore": re.compile(r"\bnightcore(?:d|'d)?\b", re.IGNORECASE),
    "sped_up": re.compile(r"\b(?:sped|speed)[- ]up\b", re.IGNORECASE),
    "karaoke": re.compile(r"\bkara(?:oke|ōke)\b|\bbacking\s+track\b", re.IGNORECASE),
    "tribute": re.compile(
        r"^tribute\s+to\b|\btribute\s+(?:version|recording|band|artists?)\b"
        r"|\boriginally\s+performed\s+by\b|\bas\s+made\s+famous\s+by\b",
        re.IGNORECASE,
    ),
    "cover": re.compile(
        r"\bcover\s+(?:version|record)\b|\(\s*cover(?:\s+of\b[^)]*)?\s*\)"
        r"|\s+-\s+cover(?:\s+of\b.*)?$|\bin\s+the\s+style\s+of\b",
        re.IGNORECASE,
    ),
    "instrumental": re.compile(
        r"\binstrumental\s+(?:version|mix|cover|track)\b"
        r"|\ba\s+cappella\b|\bpiano\s+version\b"
        r"|\bstring\s+(?:quartet|version)\b|\borchestral\s+version\b"
        r"|\b8\s*bit\s+(?:version|cover)\b|\bringtone\b"
        r"|\blo-?fi\s+(?:version|remix|cover|study)\b",
        re.IGNORECASE,
    ),
    "remix": re.compile(
        r"(?:\(|\[)[^)\]]*\b(?:re)?mix(?:es)?\b[^)\]]*(?:\)|\])"
        r"|\s+-\s+[^-]*\b(?:re)?mix(?:es)?\b[^-]*$"
        r"|\b(?:club|radio|extended|dub|dance|house|vocal)\s+mix\b"
        r"|\bchopnotslop\b",
        re.IGNORECASE,
    ),
    "edit": re.compile(
        r"(?:\(|\[|\s+-\s*)[^)\]]*\b(?:rework|bootleg|vip|edit)\b"
        r"[^)\]]*(?:\)|\]|$)",
        re.IGNORECASE,
    ),
    "mashup": re.compile(r"\b(?:mashup|medley)\b|\bx\s+\w.*\bx\s+\w", re.IGNORECASE),
}
_REMASTER_RE = re.compile(
    r"(?:\(|\[|\s+-\s*)\s*(?:\d{4}\s+)?(?:re)?master(?:ed)?"
    r"(?:\s+\d{4})?\s*(?:\)|\]|$)",
    re.IGNORECASE,
)
_LIVE_RE = re.compile(
    r"(?:\(|\[|\s+-\s*)[^)\]]*\blive(?:\s+at|\s+from|\s+\d{4})?\b"
    r"[^)\]]*(?:\)|\]|$)",
    re.IGNORECASE,
)
_VERSION_SUFFIX_RE = re.compile(
    r"\s*(?:\(|\[)[^)\]]*(?:remix|mix|edit|rework|bootleg|vip|instrumental|"
    r"karaoke|cover|slowed|reverb|nightcore|sped[- ]up|live|remaster(?:ed)?)"
    r"[^)\]]*(?:\)|\])\s*$"
    r"|\s+-\s+[^-]*(?:remix|mix|edit|rework|bootleg|vip|instrumental|"
    r"karaoke|cover|slowed|reverb|nightcore|sped[- ]up|live|remaster(?:ed)?).*$",
    re.IGNORECASE,
)


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

    def version_tags(self, title: str, artist: str = "") -> frozenset[str]:
        """Return generic derivative/version labels found in title or artist.

        No artist-specific allow/deny rules are used.  Explicit source metadata
        wins over guesses: an unlabelled cover cannot be identified reliably
        from title/artist strings alone and should be surfaced to human review.
        """
        value = f"{_nfkd(str(title))} {_nfkd(str(artist))}"
        return frozenset(
            name for name, pattern in _VERSION_PATTERNS.items()
            if pattern.search(value)
        )

    def is_eligible_for_query(
        self,
        seed_title: str,
        seed_artist: str,
        result_title: str,
        result_artist: str = "",
    ) -> bool:
        """Apply derivative filtering with the explicit query-version exception.

        A canonical seed receives only canonical/remastered/live recordings.
        When the seed itself is a derivative, candidates carrying only the same
        derivative classes are allowed.  Mashups remain strict: a mashup query
        may match mashups, but a remix query does not implicitly permit mashups.
        """
        result_tags = self.version_tags(result_title, result_artist)
        if not result_tags:
            return not self.is_junk(result_title, result_artist)
        seed_tags = self.version_tags(seed_title, seed_artist)
        if not seed_tags:
            return False
        return result_tags <= seed_tags

    def canonical_title(self, title: str) -> str:
        """Normalize a recording title after removing explicit version suffixes."""
        value = _nfkd(str(title)).casefold().strip()
        previous = None
        while value != previous:
            previous = value
            value = _VERSION_SUFFIX_RE.sub("", value).strip()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())

    def version_priority(self, title: str, artist: str = "") -> int:
        """Lower is a safer canonical choice within one artist/title group."""
        tags = self.version_tags(title, artist)
        if tags:
            return 20 + len(tags)
        value = f"{_nfkd(str(title))} {_nfkd(str(artist))}"
        if _REMASTER_RE.search(value):
            return 1
        if _LIVE_RE.search(value):
            return 3
        return 0

    def prefer_canonical(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        title_field: str = "title",
        artist_field: str = "artist",
    ) -> List[Mapping[str, Any]]:
        """Stable within-artist/title dedup that prefers an original recording.

        The first-ranked item wins between equal-priority variants.  This helper
        never compares different artists, avoiding popularity and artist-specific
        rules while preventing remixes/remasters from displacing an available
        canonical recording by the same artist.
        """
        values = list(candidates)
        winners: dict[tuple[str, str], tuple[int, int]] = {}
        for position, item in enumerate(values):
            title, artist = str(item[title_field]), str(item[artist_field])
            key = (self.canonical_title(title), _nfkd(artist).casefold().strip())
            priority = self.version_priority(title, artist)
            if key not in winners or (priority, position) < winners[key]:
                winners[key] = (priority, position)
        selected = {position for _, position in winners.values()}
        return [item for position, item in enumerate(values) if position in selected]

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
