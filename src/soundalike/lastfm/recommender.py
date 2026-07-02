"""Aggregate Last.fm similar-tracks across a set of seed songs into a ranking.

A song similar to *several* of your seeds should rank above one similar to just
a single seed, so scores are summed across seeds. This turns per-track
similarity into a taste-profile recommendation that works for any catalog.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .client import LastFmClient, SimilarTrack

Seed = Tuple[str, Optional[str]]


@dataclass
class Suggestion:
    title: str
    artist: str
    score: float
    seed_hits: int  # how many of your seeds this song was similar to

    def __str__(self) -> str:
        return f"{self.title} — {self.artist}  ({self.score:.3f}, x{self.seed_hits})"


def _key(title: str, artist: str) -> str:
    return " ".join(str(title).strip().casefold().split()) + " :: " + \
        " ".join(str(artist).strip().casefold().split())


class LastFmRecommender:
    def __init__(self, client: LastFmClient):
        self.client = client

    def recommend(
        self,
        seeds: Sequence[Seed],
        n: int = 25,
        per_seed: int = 50,
        exclude_seeds: bool = True,
    ) -> Tuple[List[Suggestion], List[Seed]]:
        """Return (suggestions, skipped_seeds).

        Seeds without an artist are skipped, since Last.fm similarity needs both
        artist and track.
        """
        scores: Dict[str, float] = defaultdict(float)
        hits: Dict[str, int] = defaultdict(int)
        meta: Dict[str, SimilarTrack] = {}
        seed_keys = {_key(t, a) for t, a in seeds if a}
        skipped: List[Seed] = []

        for title, artist in seeds:
            if not artist:
                skipped.append((title, artist))
                continue
            for cand in self.client.similar_tracks(artist, title, limit=per_seed):
                key = _key(cand.title, cand.artist)
                if exclude_seeds and key in seed_keys:
                    continue
                scores[key] += cand.match
                hits[key] += 1
                # Keep the first-seen display metadata for this candidate.
                meta.setdefault(key, cand)

        suggestions = [
            Suggestion(
                title=meta[key].title,
                artist=meta[key].artist,
                score=score,
                seed_hits=hits[key],
            )
            for key, score in scores.items()
        ]
        # Rank by aggregate score, then by how many seeds it matched.
        suggestions.sort(key=lambda s: (s.score, s.seed_hits), reverse=True)
        return suggestions[:n], skipped
