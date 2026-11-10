"""Scene layer for the Best list: artist neighbourhoods, regions and genres on top of audio fusion.

Audio decides which songs are candidates: the melody, samples and sonic lists. Affinity between the query artist
and a candidate artist combines:

- the same primary artist;
- structural similarity in Deezer's related-artist graph: overlap of closed neighbourhoods, |N(a) & N(b)| /
  sqrt(|N(a)| |N(b)|), where N(a) is the artist, its related artists and the artists that list it;
- the probability that both artists belong to the same country's scene: mostly the stated countries (Wikidata) of
  the artist's graph neighbours, partly its own;
- genre overlap: cosine of IDF-weighted Wikidata genres plus their parents, inferred from neighbours when missing.

Ranking (``scene_fuse``): the audio fusion's first songs stay on top as a head, extended over songs with calibrated
cover/sample evidence or a top-3 place in the melody or samples list; after it, slots alternate between the audio
continuation (audio score lifted by affinity; standout cover/sample matches are protected) and a scene lane ordered
by affinity times audio evidence, followed by per-artist caps. Without metadata for the query's artist the ranking equals ``api.fuse`` (plus caps when
candidate artists are known). The design follows the two-intent slotting recommended by the research report in
docs/ANYSONG_EVAL.md, which also records the selection and safeguards.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FUSE_K = 5
AUDIO_CHANNELS = ("melody", "samples")
INFERRED_WEIGHT = 0.8  # confidence of a country or genre inferred from neighbours instead of stated
STATED_SHARE = 0.3  # share of an artist's own stated country when its neighbours' region is also known


@dataclass(frozen=True)
class SceneParams:
    """Defaults are the configuration selected on development data: docs/ANYSONG_EVAL.md#scene-layer-2026-11-04, then
    a scene lane in every slot after the head, a stronger continuation boost and evidence by score only
    (docs/ANYSONG_EVAL.md, "Ranking for the lists users see", 2026-11-09)."""

    same: float = 0.6  # affinity of another song by the query's primary artist
    graph: float = 4.0  # scale applied to the neighbourhood overlap
    region: float = 0.3
    genre: float = 0.0
    boost: float = 4.0  # continuation: audio RRF score * (1 + boost * affinity)
    sonic: float = 0.0  # continuation: weight of the sonic RRF score, times affinity
    protect: float = 3.0  # z-score in the melody or samples list that marks a standout match (0 = off)
    protect_window: int = 0  # z-scores relative to the list's first this many songs (0 = the whole list)
    pattern: str = "S"  # slots after the head: A = continuation, S = scene lane ("" = no lane, no head)
    head_min: int = 3  # audio-fusion songs always kept on top
    head_scan: int = 10  # the head extends to the last of these with cover/sample evidence
    melody_evidence: float = -1.476  # melody score with ~30 % precision for a labelled cover (development data)
    samples_evidence: float = 0.302  # samples score with ~30 % precision for a labelled sample (development data)
    rank_evidence: int = 0  # a song this high in the melody or samples list also counts as evidence (0 = off)
    max_same: int = 2  # songs by the query's artist allowed in the capped head of the list
    max_other: int = 2  # songs by any other single artist allowed in the capped head
    window: int = 20  # length of the capped head


@dataclass(frozen=True)
class ArtistProfile:
    artist_id: int
    neighbours: frozenset[int]
    country: Mapping[str, float] = field(default_factory=dict)
    genre: Mapping[str, float] = field(default_factory=dict)


def _csr(rows: Sequence[Sequence], dtype) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(r) for r in rows])
    flat = np.fromiter((x for r in rows for x in r), dtype=dtype, count=int(ptr[-1]))
    return ptr, flat


class SceneTable:
    """Per-track primary artist and per-artist neighbourhood, country and genre, loaded from one npz.

    Tables built since 2026-11-08 also hold each crawled artist's own Deezer related list (``related``); older
    tables only have the symmetric neighbourhood."""

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        self.track_ids = np.asarray(arrays["track_ids"], dtype=np.int64)
        self.track_artist = np.asarray(arrays["track_artist"], dtype=np.int64)
        self.artist_ids = np.asarray(arrays["artist_ids"], dtype=np.int64)
        self.nbr_ptr, self.nbr_idx = arrays["nbr_ptr"], arrays["nbr_idx"]
        self.cty_ptr, self.cty_idx, self.cty_p = arrays["cty_ptr"], arrays["cty_idx"], arrays["cty_p"]
        self.gen_ptr, self.gen_idx, self.gen_w = arrays["gen_ptr"], arrays["gen_idx"], arrays["gen_w"]
        self.countries = [str(c) for c in arrays["countries"]]
        self.genres = [str(g) for g in arrays["genres"]]
        self.crawled = np.asarray(arrays["crawled"], dtype=bool) if "crawled" in arrays \
            else np.ones(len(self.artist_ids), dtype=bool)
        self.rel_ptr = np.asarray(arrays["rel_ptr"], dtype=np.int64) if "rel_ptr" in arrays else None
        self.rel_idx = np.asarray(arrays["rel_idx"], dtype=np.int32) if "rel_idx" in arrays else None
        if not (np.all(np.diff(self.track_ids) > 0) and np.all(np.diff(self.artist_ids) > 0)):
            raise ValueError("scene table ids must be sorted and unique")
        if len(self.track_artist) != len(self.track_ids) or len(self.nbr_ptr) != len(self.artist_ids) + 1 \
                or len(self.crawled) != len(self.artist_ids):
            raise ValueError("inconsistent scene table arrays")
        if (self.rel_ptr is None) != (self.rel_idx is None) or (
                self.rel_ptr is not None and (len(self.rel_ptr) != len(self.artist_ids) + 1
                                              or self.rel_ptr[-1] != len(self.rel_idx))):
            raise ValueError("inconsistent related-list arrays")
        self._profiles: dict[int, ArtistProfile] = {}

    @classmethod
    def load(cls, path: str | Path) -> "SceneTable":
        with np.load(path, allow_pickle=False) as z:
            return cls({name: z[name] for name in z.files})

    def save(self, path: str | Path) -> None:
        related = {} if self.rel_ptr is None else {"rel_ptr": self.rel_ptr, "rel_idx": self.rel_idx}
        np.savez_compressed(path, track_ids=self.track_ids, track_artist=self.track_artist,
                            artist_ids=self.artist_ids, nbr_ptr=self.nbr_ptr, nbr_idx=self.nbr_idx,
                            cty_ptr=self.cty_ptr, cty_idx=self.cty_idx, cty_p=self.cty_p,
                            gen_ptr=self.gen_ptr, gen_idx=self.gen_idx, gen_w=self.gen_w,
                            countries=np.array(self.countries), genres=np.array(self.genres),
                            crawled=self.crawled, **related)

    def _row(self, artist_id: int) -> int | None:
        i = int(np.searchsorted(self.artist_ids, artist_id))
        return i if i < len(self.artist_ids) and self.artist_ids[i] == artist_id else None

    def has_related(self, artist_id: int | None) -> bool:
        """Whether the artist's own related-artist list is in the table (else only artists that list it)."""
        row = self._row(int(artist_id)) if artist_id else None
        return row is not None and bool(self.crawled[row])

    def related(self, artist_id: int | None) -> list[int] | None:
        """The artist's own Deezer related list in Deezer's order, or None when the table does not hold it."""
        row = self._row(int(artist_id)) if artist_id else None
        if row is None or self.rel_ptr is None or not self.crawled[row]:
            return None
        return [int(a) for a in self.artist_ids[self.rel_idx[self.rel_ptr[row]:self.rel_ptr[row + 1]]]]

    def artist(self, track_id: int) -> int | None:
        i = int(np.searchsorted(self.track_ids, track_id))
        if i < len(self.track_ids) and self.track_ids[i] == track_id and self.track_artist[i] > 0:
            return int(self.track_artist[i])
        return None

    def stored(self, artist_id: int) -> ArtistProfile | None:
        if artist_id in self._profiles:
            return self._profiles[artist_id]
        row = self._row(artist_id)
        if row is None:
            return None
        a, b = self.nbr_ptr[row], self.nbr_ptr[row + 1]
        c, d = self.cty_ptr[row], self.cty_ptr[row + 1]
        e, f = self.gen_ptr[row], self.gen_ptr[row + 1]
        profile = ArtistProfile(
            artist_id, frozenset(int(x) for x in self.artist_ids[self.nbr_idx[a:b]]),
            {self.countries[i]: float(p) for i, p in zip(self.cty_idx[c:d], self.cty_p[c:d])},
            {self.genres[i]: float(w) for i, w in zip(self.gen_idx[e:f], self.gen_w[e:f])})
        if len(self._profiles) < 200_000:
            self._profiles[artist_id] = profile
        return profile

    def profile(self, artist_id: int | None, related: Iterable[int] | None = None) -> ArtistProfile | None:
        """Query profile: the stored one, extended (or, for an unknown artist, built) from a live related list."""
        if not artist_id:
            return None
        base = self.stored(artist_id)
        related = [int(r) for r in related or () if r and int(r) != artist_id]
        if not related:
            return base
        neighbours = set(base.neighbours if base else {artist_id}) | set(related)
        country, genre = (dict(base.country), dict(base.genre)) if base else inferred(
            [p for p in (self.stored(r) for r in related) if p is not None])
        return ArtistProfile(artist_id, frozenset(neighbours), country, genre)


def inferred(neighbours: Sequence[ArtistProfile], min_support: int = 2) -> tuple[dict[str, float], dict[str, float]]:
    """Country and genre of an artist without stated ones: the mean over its graph neighbours, discounted."""
    out = []
    for attr in ("country", "genre"):
        known = [getattr(p, attr) for p in neighbours if getattr(p, attr)]
        if len(known) < min_support:
            out.append({})
            continue
        total: dict[str, float] = defaultdict(float)
        for vec in known:
            for key, value in vec.items():
                total[key] += value / len(known)
        if attr == "country":
            kept = {k: v for k, v in total.items() if v >= 0.1}
            norm = sum(kept.values())
            out.append({k: INFERRED_WEIGHT * v / norm for k, v in kept.items()} if norm else {})
        else:
            norm = math.sqrt(sum(v * v for v in total.values()))
            out.append({k: INFERRED_WEIGHT * v / norm for k, v in total.items()} if norm else {})
    return out[0], out[1]


def components(query: ArtistProfile, candidate: ArtistProfile) -> tuple[float, float, float]:
    """Raw (graph, region, genre) similarity of two different artists, each in [0, 1] up to rounding."""
    overlap = len(query.neighbours & candidate.neighbours)
    graph = overlap / math.sqrt(len(query.neighbours) * len(candidate.neighbours)) if overlap else 0.0
    region = sum(p * candidate.country.get(k, 0.0) for k, p in query.country.items())
    genre = sum(w * candidate.genre.get(g, 0.0) for g, w in query.genre.items())
    return graph, region, genre


def affinity(query: ArtistProfile | None, candidate: ArtistProfile | None, artist_id: int | None,
             params: SceneParams) -> float:
    if query is None or not artist_id:
        return 0.0
    if artist_id == query.artist_id:
        return params.same
    if candidate is None:
        return 0.0
    graph, region, genre = components(query, candidate)
    return min(1.0, params.graph * graph + params.region * region + params.genre * genre)


def standout(orders: Mapping[str, Sequence[Hashable]], scores: Mapping[str, Sequence[float]] | None,
             window: int = 0) -> dict[Hashable, float]:
    """Each song's best z-score within the melody or samples list it appears in (how far it stands out).

    With ``window`` the statistics cover only the list's first ``window`` songs, so a song must stand out from its
    nearest competitors rather than from the long tail of a 200-song list.
    """
    out: dict[Hashable, float] = {}
    for channel in AUDIO_CHANNELS:
        values = np.asarray((scores or {}).get(channel, ()), dtype=np.float64)
        keys = orders.get(channel, ())
        if len(values) != len(keys):
            continue
        if window:
            values, keys = values[:window], keys[:window]
        if len(values) < 2:
            continue
        sd = values.std()
        if sd <= 0:
            continue
        for key, z in zip(keys, (values - values.mean()) / sd):
            out[key] = max(out.get(key, -np.inf), float(z))
    return out


def _channel_ranks(orders: Mapping[str, Sequence[Hashable]]) -> dict[Hashable, dict[str, int]]:
    ranks: dict[Hashable, dict[str, int]] = {}
    for channel in (*AUDIO_CHANNELS, "sonic"):
        for r, key in enumerate(orders.get(channel, ()), 1):
            ranks.setdefault(key, {})[channel] = r
    return ranks


def audio_head(orders: Mapping[str, Sequence[Hashable]], scores: Mapping[str, Sequence[float]] | None,
               params: SceneParams, k: int = FUSE_K) -> list[Hashable]:
    """The audio fusion's first songs kept on top of a slotted Best list: at least ``head_min``, extended down to the
    last of its first ``head_scan`` songs whose melody or samples score reaches the calibrated cover/sample evidence
    (``scores``: per channel, aligned with ``orders``) or that is among the first ``rank_evidence`` songs of either."""
    ranks = _channel_ranks(orders)
    audio = {key: sum(1.0 / (k + rk[ch]) for ch in AUDIO_CHANNELS if ch in rk) for key, rk in ranks.items()}
    tie = {key: min((r for ch, r in rk.items() if ch in AUDIO_CHANNELS), default=1_000_000 + rk.get("sonic", 0))
           for key, rk in ranks.items()}
    baseline = sorted((key for key in ranks if audio[key] > 0), key=lambda key: (-audio[key], tie[key]))
    evidence = set()
    for channel, threshold in (("melody", params.melody_evidence), ("samples", params.samples_evidence)):
        for key, value in zip(orders.get(channel, ()), (scores or {}).get(channel, ())):
            if value >= threshold:
                evidence.add(key)
        evidence.update(list(orders.get(channel, ()))[:params.rank_evidence])
    hits = [i for i, key in enumerate(baseline[:params.head_scan]) if key in evidence]
    return baseline[:max(params.head_min, hits[-1] + 1 if hits else 0)]


def cap_artists(placed: Sequence, artist_of: Callable[[Any], int | None], query_artist: int | None,
                params: SceneParams) -> list:
    """Per-artist caps over the first ``window`` places: at most ``max_same`` songs by the query's artist and
    ``max_other`` by any other; songs over a cap move behind the window, in order."""
    head_list, tail, counts = [], [], Counter()
    for entry in placed:
        artist_id = artist_of(entry)
        if len(head_list) < params.window and artist_id:
            limit = params.max_same if artist_id == query_artist else params.max_other
            if counts[artist_id] >= limit:
                tail.append(entry)
                continue
            counts[artist_id] += 1
        (head_list if len(head_list) < params.window else tail).append(entry)
    return head_list + tail


def arrange_artists(placed: Sequence, artist_of: Callable[[Any], int | None], query_artist: int | None,
                    params: SceneParams, head: int = 0, gap: int = 1, once: int = 0) -> list:
    """``cap_artists`` plus spacing after the first ``head`` entries (the audio head): an artist's songs are at least
    ``gap`` places apart (2 = never twice in a row), and within the first ``once`` places each artist appears once.

    An entry that breaks a rule waits and takes the first place where it no longer does, ahead of later entries, so
    every artist's first song keeps its order and only later songs move down. Head entries obey the caps only. With
    ``gap`` 1 and ``once`` 0 this is ``cap_artists``. When only waiting entries are left and none fits, the earliest
    goes next."""
    counts: Counter = Counter()  # songs per artist in the first ``window`` places
    out: list[int] = []
    waiting: list[int] = []
    artist = [artist_of(entry) for entry in placed]

    def fits(i: int) -> bool:
        a, p = artist[i], len(out)
        if not a:
            return True
        if p < params.window and counts[a] >= (params.max_same if a == query_artist else params.max_other):
            return False
        if i < head:
            return True
        if gap > 1 and any(artist[j] == a for j in out[max(0, p - gap + 1):]):
            return False
        return not (p < once and counts[a] > 0)

    nxt = 0
    while nxt < len(placed) or waiting:
        k = next((k for k, i in enumerate(waiting) if fits(i)), None)
        if k is not None:
            i = waiting.pop(k)
        elif nxt < len(placed):
            i, nxt = nxt, nxt + 1
            if not fits(i):
                waiting.append(i)
                continue
        else:
            i = waiting.pop(0)
        if artist[i] and len(out) < params.window:
            counts[artist[i]] += 1
        out.append(i)
    return [placed[i] for i in out]


def scene_fuse(orders: Mapping[str, Sequence[Hashable]], artists: Mapping[Hashable, int | None],
               query: ArtistProfile | None, table: SceneTable | None, params: SceneParams,
               k: int = FUSE_K, scores: Mapping[str, Sequence[float]] | None = None
               ) -> list[tuple[Hashable, float, dict[str, int]]]:
    """Best list with the scene layer; same output shape as ``api.fuse`` (key, score, channel ranks).

    Without a query profile this is the audio fusion (plus artist caps). With one:

    1. Head: the first ``head_min`` songs of the audio fusion stay in place, extended down to the last song of its
       first ``head_scan`` whose melody or samples score reaches the calibrated cover/sample evidence thresholds
       (``scores``: per channel, aligned with ``orders``) or that is among the first ``rank_evidence`` songs of the
       melody or samples list.
    2. Continuation: the remaining songs by audio score * (1 + boost * A); a song whose melody or samples score stands
       out by ``protect`` standard deviations is lifted as if A = 1, so scene neighbours cannot pass a likely cover.
    3. Scene lane: songs with A > 0 by A * (audio + sonic RRF); every lane song has audio evidence in some channel.
    4. After the head, slots follow ``pattern`` (A = continuation, S = lane), then per-artist caps.

    Scores of slotted lists are positional, 1 / (k + position), since slots, not sums, decide the order.
    """
    ranks: dict[Hashable, dict[str, int]] = {}
    for channel in (*AUDIO_CHANNELS, "sonic"):
        for r, key in enumerate(orders.get(channel, ()), 1):
            ranks.setdefault(key, {})[channel] = r
    scene_on = query is not None and table is not None
    strong = {key for key, z in standout(orders, scores, params.protect_window).items() if z >= params.protect} \
        if params.protect > 0 and scene_on else set()
    audio, sonic, aff, tie, audio_ranks = {}, {}, {}, {}, {}
    for key, rk in ranks.items():
        audio[key] = sum(1.0 / (k + rk[ch]) for ch in AUDIO_CHANNELS if ch in rk)
        sonic[key] = 1.0 / (k + rk["sonic"]) if "sonic" in rk else 0.0
        artist_id = artists.get(key)
        aff[key] = affinity(query, table.stored(artist_id), artist_id, params) \
            if scene_on and artist_id else 0.0
        audio_ranks[key] = {ch: r for ch, r in rk.items() if ch in AUDIO_CHANNELS}
        # Ties break as in api.fuse: best audio rank, then first appearance; sonic-only songs after those.
        tie[key] = min(audio_ranks[key].values()) if audio_ranks[key] else 1_000_000 + rk["sonic"]

    def shown(key: Hashable, via_sonic: bool) -> dict[str, int]:
        """Channel ranks to show: the audio ranks, plus the sonic rank when the sounds list helped place the song."""
        if "sonic" in ranks[key] and aff[key] > 0 and (via_sonic or not audio_ranks[key]):
            return audio_ranks[key] | {"sonic": ranks[key]["sonic"]}
        return audio_ranks[key]

    def ordered(score: dict) -> list:
        keep = [key for key in ranks if score[key] > 0]
        return sorted(keep, key=lambda key: (-score[key], tie[key]))

    lifted = {key: audio[key] * (1.0 + params.boost * (1.0 if key in strong else aff[key]))
              + params.sonic * aff[key] * sonic[key] for key in ranks}
    if not scene_on or not params.pattern:
        placed = [(key, lifted[key], shown(key, params.sonic > 0)) for key in ordered(lifted)]
    else:
        head = audio_head(orders, scores, params, k)
        used = set(head)
        rest = [key for key in ordered(lifted) if key not in used]
        lane = [key for key in ordered({key: aff[key] * (audio[key] + sonic[key]) for key in ranks})
                if key not in used]
        placed = [(key, 0.0, shown(key, False)) for key in head]
        ia = il = step = 0
        while ia < len(rest) or il < len(lane):
            prefer_lane = params.pattern[step % len(params.pattern)] == "S"
            step += 1
            for from_lane in ((True, False) if prefer_lane else (False, True)):
                source, i = (lane, il) if from_lane else (rest, ia)
                while i < len(source) and source[i] in used:
                    i += 1
                if from_lane:
                    il = i
                else:
                    ia = i
                if i < len(source):
                    used.add(source[i])
                    placed.append((source[i], 0.0, shown(source[i], from_lane or params.sonic > 0)))
                    break
    query_artist = query.artist_id if query is not None else None
    out = cap_artists(placed, lambda entry: artists.get(entry[0]), query_artist, params)
    if scene_on and params.pattern:
        out = [(key, 1.0 / (k + position), rk) for position, (key, _, rk) in enumerate(out, 1)]
    return out


def build(tracks: Mapping[int, int], related: Mapping[int, Sequence[int]], countries: Mapping[int, Sequence[str]],
          genres: Mapping[int, Sequence[tuple[str, float]]], rounds: int = 2) -> SceneTable:
    """Assemble a table: symmetric related-artist graph, stated countries/genres, inference for the rest."""
    nbrs: dict[int, set[int]] = defaultdict(set)
    for a, rel in related.items():
        for b in rel:
            if b and b != a:
                nbrs[a].add(int(b))
                nbrs[int(b)].add(a)
    artist_ids = sorted(set(nbrs) | set(tracks.values()) | set(countries) | set(genres))
    index = {a: i for i, a in enumerate(artist_ids)}
    df = Counter(g for gs in genres.values() for g, _ in gs)
    n_genre = max(1, len(genres))
    stated = {a: {c: 1.0 / len(set(cs)) for c in set(cs)} for a, cs in countries.items() if cs}
    # An artist's region is mostly where its graph neighbours come from: citizenship or birthplace alone would put
    # e.g. a London-born Atlanta rapper in the UK scene. Stated countries keep a minority share.
    country = {}
    for a in artist_ids:
        scene_region, _ = inferred([ArtistProfile(b, frozenset(), stated.get(b, {}), {}) for b in nbrs.get(a, ())])
        own = stated.get(a)
        if own and scene_region:
            mix: dict[str, float] = defaultdict(float)
            for c, v in own.items():
                mix[c] += STATED_SHARE * v
            for c, v in scene_region.items():
                mix[c] += (1 - STATED_SHARE) * v / INFERRED_WEIGHT
            country[a] = dict(mix)
        elif own or scene_region:
            country[a] = own or scene_region
    genre = {}
    for a, gs in genres.items():
        vec: dict[str, float] = defaultdict(float)
        for g, w in gs:
            vec[g] = max(vec[g], w * math.log(1 + n_genre / df[g]))
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm:
            genre[a] = {g: v / norm for g, v in vec.items()}
    for _ in range(rounds):
        add_c, add_g = {}, {}
        for a in artist_ids:
            if a in country and a in genre:
                continue
            around = [ArtistProfile(b, frozenset(), country.get(b, {}), genre.get(b, {})) for b in nbrs.get(a, ())]
            c, g = inferred(around)
            if a not in country and c:
                add_c[a] = c
            if a not in genre and g:
                add_g[a] = g
        country |= add_c
        genre |= add_g
    cvocab = sorted({c for v in country.values() for c in v})
    gvocab = sorted({g for v in genre.values() for g in v})
    ci, gi = {c: i for i, c in enumerate(cvocab)}, {g: i for i, g in enumerate(gvocab)}
    nbr_ptr, nbr_idx = _csr([sorted(index[b] for b in nbrs.get(a, set()) | {a}) for a in artist_ids], np.int32)
    cty_ptr, cty_idx = _csr([[ci[c] for c in sorted(country.get(a, {}))] for a in artist_ids], np.int32)
    _, cty_p = _csr([[country[a][c] for c in sorted(country.get(a, {}))] for a in artist_ids], np.float32)
    gen_ptr, gen_idx = _csr([[gi[g] for g in sorted(genre.get(a, {}))] for a in artist_ids], np.int32)
    _, gen_w = _csr([[genre[a][g] for g in sorted(genre.get(a, {}))] for a in artist_ids], np.float32)
    track_ids = np.array(sorted(tracks), dtype=np.int64)
    crawled = np.array([a in related for a in artist_ids], dtype=bool)
    rel_ptr, rel_idx = _csr([list(dict.fromkeys(index[int(b)] for b in related.get(a, ()) if b and int(b) != a))
                             for a in artist_ids], np.int32)
    return SceneTable({"track_ids": track_ids, "track_artist": np.array([tracks[t] for t in track_ids], np.int64),
                       "artist_ids": np.array(artist_ids, np.int64), "nbr_ptr": nbr_ptr, "nbr_idx": nbr_idx,
                       "cty_ptr": cty_ptr, "cty_idx": cty_idx, "cty_p": cty_p, "gen_ptr": gen_ptr,
                       "gen_idx": gen_idx, "gen_w": gen_w, "countries": np.array(cvocab or [""]),
                       "genres": np.array(gvocab or [""]), "crawled": crawled, "rel_ptr": rel_ptr,
                       "rel_idx": rel_idx})


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def load_sources(data: Path, catalog_artists: Path) -> tuple[dict, dict, dict, dict]:
    """Read collected sources: Deezer track/related JSONL and Wikidata artist/value files."""
    tracks = {int(k): int(v) for k, v in json.loads(catalog_artists.read_text(encoding="utf-8")).items()}
    if (data / "deezer_tracks.jsonl").exists():
        for row in _jsonl(data / "deezer_tracks.jsonl"):
            if row.get("artist_id"):
                tracks.setdefault(int(row["track_id"]), int(row["artist_id"]))
    related = {int(r["artist_id"]): [int(x[0]) for x in r["related"] if x and x[0]]
               for r in _jsonl(data / "deezer_related.jsonl") if "related" in r}
    values = json.loads((data / "wikidata_values.json").read_text(encoding="utf-8"))
    iso = {q: (v["iso"] or v["parent_iso"]) for q, v in values["countries"].items()}
    parents = {q: v.get("parents", []) for q, v in values["genres"].items()}
    countries, genres = {}, {}
    for row in _jsonl(data / "wikidata_artists.jsonl"):
        stated = [q for p in ("P27", "P495", "P740") for q in row.get(p, [])] or row.get("P19", [])
        codes = sorted({c for q in stated for c in iso.get(q, [])[:1]})
        if codes:
            countries[int(row["dz"])] = codes
        gs = [(g, 1.0) for g in row.get("P136", [])]
        gs += [(p, 0.5) for g in row.get("P136", []) for p in parents.get(g, []) if p in parents]
        if gs:
            genres[int(row["dz"])] = gs
    return tracks, related, countries, genres


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="anysong-scene", description="build the Best-list scene table")
    p.add_argument("--data", type=Path, required=True, help="folder with deezer_*.jsonl and wikidata_* files")
    p.add_argument("--catalog-artists", type=Path, required=True, help="JSON track id -> Deezer artist id")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    table = build(*load_sources(args.data, args.catalog_artists))
    table.save(args.out)
    known = int(np.count_nonzero(table.track_artist))
    print(f"{len(table.track_ids)} tracks ({known} with artists), {len(table.artist_ids)} artists, "
          f"{int(np.count_nonzero(np.diff(table.cty_ptr)))} with countries, "
          f"{int(np.count_nonzero(np.diff(table.gen_ptr)))} with genres -> {args.out}")


if __name__ == "__main__":
    main()
