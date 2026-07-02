"""Broad, multi-scene harvesting for the deep-vibe library.

The original harvest seeds ~50 electronic/hyperpop artists, which gives that
scene great depth but leaves everything else (indie, dream-pop, R&B, country,
jazz, metal, mainstream pop/rap...) thin — so a seed from a sparse scene has no
real neighbours to match against. This module fixes coverage by casting a wide
net.

Deezer's public genre endpoints are unusable unauthenticated (``/genre/{id}/
artists`` and ``/chart/{id}/artists`` ignore the id and return the same global
list), so genre-scoped harvesting isn't possible. Instead we lean on the one
signal that *is* genre-coherent — the related-artist graph:

  1. **Roots**: a large, hand-curated multi-scene seed list (below, covering
     every major lane and deliberately over-sampling the indie / dream-pop /
     lo-fi corner the charts miss) plus the global chart for mainstream breadth.
  2. **2-hop related BFS**: fan each root out to its related artists and their
     related artists. Related is genre-coherent (TV Girl -> Alex G, Current
     Joys, Mac DeMarco...), so this reaches deep into every scene the seeds
     touch.

Everything feeds the same harvest-once spec cache, so it's still download-once /
re-embed-forever and fully resumable.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional

import numpy as np
import requests

from ..audio.previews import DeezerClient, DeezerTrack
from ..audio.vibe import vibe_from_file
from .spec_cache import SpecCache, _artist_id
from .spectrogram import SpectrogramConfig, _fit_frames, load_audio, log_mel_full

# Curated artists spanning many scenes. Genre charts skew mainstream and miss
# the niches, so this list deliberately over-samples indie / dream-pop / lo-fi
# (the coverage gap that motivated the broad harvest) while still touching every
# major lane. Related-artist expansion fans each of these out to its neighbours.
BROAD_SEED_ARTISTS: List[str] = [
    # indie / dream-pop / lo-fi / bedroom pop
    "TV Girl", "Mac DeMarco", "Mk.gee", "Beach Fossils", "Cigarettes After Sex",
    "Men I Trust", "Clairo", "Crumb", "Boy Pablo", "Still Woozy", "Rex Orange County",
    "Cuco", "Alvvays", "Beach House", "Slowdive", "DIIV", "Homeshake", "Faye Webster",
    "Current Joys", "Wisp", "The Marías", "Steve Lacy", "Kali Uchis", "Yot Club",
    "Ricky Montgomery", "Wallows", "Peach Pit", "Surf Curse", "Mild High Club",
    "Whitney", "Khruangbin", "Men I Trust", "Japanese Breakfast", "Weyes Blood",
    # mainstream pop
    "The Weeknd", "Taylor Swift", "Dua Lipa", "Billie Eilish", "Olivia Rodrigo",
    "Ariana Grande", "Harry Styles", "Lorde", "Charli xcx", "Lana Del Rey",
    "Sabrina Carpenter", "Chappell Roan", "Troye Sivan",
    # hip-hop / rap
    "Kendrick Lamar", "Drake", "Travis Scott", "Tyler, The Creator", "J. Cole",
    "Kanye West", "A$AP Rocky", "Playboi Carti", "Baby Keem", "MF DOOM",
    "Earl Sweatshirt", "Denzel Curry", "JID", "Mac Miller", "Childish Gambino",
    # R&B / soul / neo-soul
    "Frank Ocean", "Daniel Caesar", "Brent Faiyaz", "SZA", "Solange", "Giveon",
    "Snoh Aalegra", "H.E.R.", "Erykah Badu", "D'Angelo", "Sampha", "Cleo Sol",
    # rock / alternative
    "Arctic Monkeys", "The Strokes", "Radiohead", "Tame Impala", "The 1975",
    "Red Hot Chili Peppers", "Nirvana", "The Smiths", "Cage The Elephant",
    "Paramore", "Foo Fighters", "Pixies", "Interpol", "Phoenix", "MGMT",
    # metal / heavy
    "Deftones", "Metallica", "System of a Down", "Gojira", "Bring Me The Horizon",
    "Sleep Token", "Loathe", "Knocked Loose",
    # electronic / EDM / IDM
    "Fred again..", "Four Tet", "Bonobo", "Jamie xx", "Disclosure", "Aphex Twin",
    "Boards of Canada", "Burial", "Daft Punk", "Justice", "ODESZA", "Flume",
    "Porter Robinson", "Skrillex", "Jai Paul",
    # hyperpop / underground (keep the original strength)
    "100 gecs", "ericdoa", "glaive", "aldn", "brakence", "underscores", "Jane Remover",
    "quannnic", "midwxst", "d0llywood1",
    # jazz / funk / fusion
    "Robert Glasper", "BadBadNotGood", "Thundercat", "Hiatus Kaiyote",
    "Kamasi Washington", "Vulfpeck", "Tom Misch", "FKJ",
    # folk / country / singer-songwriter
    "Bon Iver", "Fleet Foxes", "Sufjan Stevens", "Phoebe Bridgers", "Hozier",
    "Zach Bryan", "Noah Kahan", "Big Thief", "Nick Drake",
    # latin
    "Bad Bunny", "Peso Pluma", "Rauw Alejandro", "Rosalía", "Feid",
]


# A deliberately deep, genre-and-region-spanning list of niches the broad seeds
# above don't reach: whole scenes (world/regional, electronic subgenres, rock /
# metal / punk subgenres, jazz, classical, blues, gospel, reggae, experimental)
# that a listener might seed from and currently find no close neighbours for.
NICHE_SEED_ARTISTS: List[str] = [
    # K-pop / K-R&B / K-hiphop
    "NewJeans", "LE SSERAFIM", "SEVENTEEN", "Stray Kids", "aespa", "IVE", "TWICE",
    "BLACKPINK", "(G)I-DLE", "DEAN", "Crush", "Zico", "Jay Park", "RM",
    # J-pop / city pop / J-rock
    "Fujii Kaze", "YOASOBI", "Kenshi Yonezu", "Vaundy", "King Gnu", "Yorushika",
    "Official HIGE DANdism", "Tatsuro Yamashita", "Mariya Takeuchi", "Ado", "Aimer",
    # Afrobeats / Amapiano / African
    "Burna Boy", "Wizkid", "Davido", "Rema", "Asake", "Tems", "Ayra Starr",
    "Fireboy DML", "Omah Lay", "CKay", "Kabza De Small", "DJ Maphorisa", "Uncle Waffles",
    # French rap / pop
    "PNL", "Ninho", "Aya Nakamura", "Stromae", "Angèle", "Damso", "Orelsan",
    # Latin (reggaeton / trap / regional)
    "J Balvin", "Karol G", "Ozuna", "Anuel AA", "Myke Towers", "Anitta", "Grupo Frontera",
    # Brazilian / bossa / MPB
    "Gilberto Gil", "Caetano Veloso", "João Gilberto", "Tim Maia", "Racionais MC's",
    # reggae / dancehall / dub
    "Bob Marley & The Wailers", "Chronixx", "Koffee", "Popcaan", "Protoje",
    "Lee Scratch Perry", "Toots & The Maytals",
    # techno / house / deep
    "Charlotte de Witte", "Amelie Lens", "Boris Brejcha", "Nina Kraviz", "Peggy Gou",
    "Fisher", "John Summit", "Chris Lake", "Black Coffee", "Kaytranada",
    # trance / DnB / jungle / garage
    "Above & Beyond", "Armin van Buuren", "Netsky", "Sub Focus", "Chase & Status",
    "Pendulum", "Goldie", "Overmono",
    # ambient / modern classical / drone
    "Brian Eno", "Tim Hecker", "William Basinski", "Grouper", "Nils Frahm",
    "Ólafur Arnalds", "Max Richter", "Ludovico Einaudi", "Stars of the Lid",
    # synthwave / vaporwave / phonk
    "The Midnight", "Gunship", "Carpenter Brut", "Perturbator", "Kordhell", "DVRST",
    "Freddie Dredd", "Ghostface Playa",
    # breakcore / hardstyle / experimental electronic
    "Sewerslvt", "Machine Girl", "Headhunterz", "Da Tweekaz", "Oneohtrix Point Never",
    "Arca", "Death Grips", "JPEGMAFIA", "clipping.", "Black Midi",
    # dubstep / riddim / bass
    "Excision", "Virtual Riot", "SVDDEN DEATH", "Subtronics", "Zomboy", "Au5",
    # lofi / chillhop / instrumental hip-hop
    "Nujabes", "J Dilla", "Tomppabeats", "idealism",
    # punk / pop-punk / hardcore
    "Ramones", "The Clash", "Dead Kennedys", "Bad Religion", "Blink-182", "Green Day",
    "Sum 41", "Descendents", "Turnstile",
    # emo / midwest emo / screamo / post-hardcore
    "American Football", "Sunny Day Real Estate", "Modern Baseball", "Mom Jeans.",
    "La Dispute", "Touché Amoré", "The Hotelier",
    # post-rock / math rock
    "Explosions in the Sky", "Godspeed You! Black Emperor", "Mogwai", "Sigur Rós",
    "This Will Destroy You", "toe", "tricot", "CHON", "Covet",
    # shoegaze / dream-pop (deeper)
    "My Bloody Valentine", "Ride", "Whirr", "Nothing", "Cocteau Twins", "Duster",
    # grunge / post-punk / new wave / industrial
    "Soundgarden", "Alice in Chains", "Pearl Jam", "Joy Division", "The Cure",
    "New Order", "Talking Heads", "Nine Inch Nails",
    # psychedelic / garage rock / prog
    "King Gizzard & The Lizard Wizard", "Pond", "The Black Keys", "King Crimson",
    "Tool", "Porcupine Tree",
    # black / death / doom / djent metal
    "Mayhem", "Darkthrone", "Emperor", "Death", "Cannibal Corpse", "Electric Wizard",
    "Sleep", "Periphery", "Meshuggah", "Animals as Leaders", "Lorna Shore",
    "Architects", "Parkway Drive", "Slipknot", "Slayer", "Megadeth",
    # jazz (classic / spiritual / modern)
    "Miles Davis", "John Coltrane", "Bill Evans", "Thelonious Monk", "Charles Mingus",
    "Alice Coltrane", "Pharoah Sanders", "Herbie Hancock", "GoGo Penguin",
    "Snarky Puppy", "Nubya Garcia", "Ezra Collective",
    # classical / film scores
    "Johann Sebastian Bach", "Ludwig van Beethoven", "Frédéric Chopin", "Claude Debussy",
    "Philip Glass", "Steve Reich", "Hans Zimmer", "Ennio Morricone", "Ludwig Göransson",
    # blues / soul / motown / funk
    "B.B. King", "Muddy Waters", "John Lee Hooker", "Gary Clark Jr.", "Marvin Gaye",
    "Stevie Wonder", "Aretha Franklin", "Al Green", "Otis Redding", "James Brown",
    "Parliament", "Sly & The Family Stone",
    # country / americana / bluegrass
    "Johnny Cash", "Willie Nelson", "Dolly Parton", "Sturgill Simpson", "Tyler Childers",
    "Colter Wall", "Chris Stapleton", "Billy Strings", "Nickel Creek",
    # indie folk / freak folk
    "Iron & Wine", "The Tallest Man on Earth", "Sufjan Stevens", "Angel Olsen",
    "Adrianne Lenker",
    # drill / cloud rap / emo rap / underground
    "Central Cee", "Digga D", "Chief Keef", "Lil Durk", "Lil Peep", "Bones",
    "Ghostemane", "$uicideboy$", "Yeat", "Ken Carson", "Destroy Lonely",
    "Westside Gunn", "Griselda", "MIKE",
    # indie / electronic pop
    "CHVRCHES", "Purity Ring", "Grimes", "Caroline Polachek", "Magdalena Bay",
    "Jessie Ware", "Rina Sawayama", "Sylvan Esso", "SOPHIE", "FKA twigs",
    # gospel / christian
    "Kirk Franklin", "Hillsong UNITED",
    # soundtrack / game
    "Toby Fox", "C418", "Nobuo Uematsu",
]


def _global_chart_artists(client: DeezerClient, limit: int,
                          progress: Callable[[str], None]) -> List[int]:
    """Deezer's public genre endpoints ignore the genre id (they all return the
    same global list), so genre-scoped harvesting is impossible unauthenticated.
    We instead grab the global chart once (mainstream breadth across pop/rap/
    latin/etc.) and rely on the related-artist BFS below for genre-coherent
    depth into every scene."""
    try:
        data = client._get("/chart/0/artists", {"limit": limit})
    except Exception as exc:  # noqa: BLE001
        progress(f"global chart failed: {exc}")
        return []
    ids = [int(a["id"]) for a in data.get("data", []) if a.get("id")]
    progress(f"global chart: {len(ids)} mainstream artists")
    return ids


def _gather_artist_ids(
    client: DeezerClient,
    session: requests.Session,
    per_genre_artists: int,
    related_per_seed: int,
    progress: Callable[[str], None],
    hop2_sample: int = 1500,
) -> List[int]:
    """Collect a broad, genre-diverse artist set via a 2-hop related BFS.

    Roots = the curated multi-scene seeds + the global chart. Related-artist
    expansion (which, unlike the genre endpoint, *is* genre-coherent) then fans
    each root out to its neighbours and their neighbours, reaching every scene
    the seeds touch.
    """
    artist_ids: set = set()

    # Roots: curated seeds (resolved) + mainstream global chart.
    roots: set = set(_global_chart_artists(client, per_genre_artists, progress))
    all_seeds = BROAD_SEED_ARTISTS + NICHE_SEED_ARTISTS
    progress(f"Resolving {len(all_seeds)} curated seeds...")
    for nm in all_seeds:
        aid = _artist_id(session, nm)
        if aid:
            roots.add(aid)
        time.sleep(0.03)
    artist_ids |= roots
    progress(f"{len(roots)} root artists")

    # Hop 1: related of every root.
    hop1: set = set()
    for i, aid in enumerate(roots, 1):
        try:
            hop1.update(int(r) for r in client.related_artists(aid, related_per_seed))
        except Exception:  # noqa: BLE001
            pass
        if i % 100 == 0:
            progress(f"  hop1 {i}/{len(roots)} -> {len(hop1)} related")
    artist_ids |= hop1
    progress(f"After hop1: {len(artist_ids)} artists")

    # Hop 2: related of a shuffled sample of hop1 (bounds API calls).
    hop1_new = list(hop1 - roots)
    np.random.default_rng(0).shuffle(hop1_new)
    sample = hop1_new[:hop2_sample]
    for i, aid in enumerate(sample, 1):
        try:
            artist_ids.update(int(r) for r in client.related_artists(aid, related_per_seed))
        except Exception:  # noqa: BLE001
            pass
        if i % 200 == 0:
            progress(f"  hop2 {i}/{len(sample)} -> {len(artist_ids)} total")
    progress(f"Total unique artists: {len(artist_ids)}")
    return list(artist_ids)


def _save_candidates(path: Path, tracks: List[DeezerTrack]) -> None:
    path.write_text(json.dumps(
        [{"id": t.id, "title": t.title, "artist": t.artist,
          "artist_id": t.artist_id, "preview": t.preview_url} for t in tracks]
    ))


def _load_candidates(path: Path) -> List[DeezerTrack]:
    rows = json.loads(path.read_text())
    return [DeezerTrack(id=int(r["id"]), title=r["title"], artist=r["artist"],
                        artist_id=int(r.get("artist_id", 0) or 0),
                        preview_url=r.get("preview", "")) for r in rows]


_API_SESSION = requests.Session()  # thread-safe for concurrent GETs (urllib3 pool)


def _fresh_preview(track_id: int) -> Optional[str]:
    """Deezer preview URLs are signed and expire, so a URL saved during the
    gather phase is often stale (403) by download time. Fetch a fresh one by
    track id right before downloading, backing off on the API rate limit."""
    for attempt in range(5):
        try:
            r = _API_SESSION.get(f"https://api.deezer.com/track/{track_id}", timeout=20)
            data = r.json()
        except Exception:  # noqa: BLE001
            time.sleep(min(2 ** attempt, 15))
            continue
        if isinstance(data, dict) and data.get("error"):
            time.sleep(min(2 ** attempt, 15))
            continue
        return data.get("preview") or None
    return None


def _process_candidate(track: DeezerTrack, cfg: SpectrogramConfig, tmp: Path):
    """Fetch a fresh preview URL, download + analyze it (thread-safe)."""
    preview = _fresh_preview(track.id)
    if not preview:
        return None
    dest = tmp / f"{track.id}.mp3"
    try:
        content = None
        for attempt in range(3):  # tolerate transient CDN hiccups
            try:
                resp = requests.get(preview, timeout=30)
                resp.raise_for_status()
                content = resp.content
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (attempt + 1))
        if not content:
            return None
        dest.write_bytes(content)
        y = load_audio(dest, cfg.sample_rate)
        spec = _fit_frames(log_mel_full(y, cfg), cfg.target_frames)
        vfeat = vibe_from_file(str(dest))
        return track, spec, vfeat.vector()
    except Exception:  # noqa: BLE001
        return None
    finally:
        dest.unlink(missing_ok=True)


def harvest_broad_to_cache(
    cache_path: Path,
    per_artist: int = 15,
    per_genre_artists: int = 100,
    related_per_seed: int = 8,
    hop2_sample: int = 1500,
    max_artists: int = 6000,
    checkpoint_every: int = 200,
    workers: int = 16,
    target: int = 0,
    seed: int = 0,
    progress: Callable[[str], None] = print,
) -> SpecCache:
    """Harvest a broad, multi-scene library into the spec cache.

    Fully resumable: reloads the cache (skips tracks already present) and caches
    the gathered candidate list to a sidecar so a restart never re-does the slow
    API-gathering phase. Download + DSP analysis runs across ``workers`` threads
    (librosa/numpy release the GIL), turning a CPU-idle 0.8/s crawl into a fast
    multi-core harvest.
    """
    cfg = SpectrogramConfig()
    cache = SpecCache.load(cache_path) if Path(cache_path).exists() else SpecCache()
    if len(cache):
        progress(f"Loaded existing cache: {len(cache)} tracks")

    # Candidate pool: reuse the sidecar if present, else gather (slow) and save.
    cand_path = Path(str(cache_path) + ".candidates.json")
    if cand_path.exists():
        candidates = _load_candidates(cand_path)
        progress(f"Loaded {len(candidates)} candidates from sidecar")
    else:
        client = DeezerClient()
        session = requests.Session()
        artist_ids = _gather_artist_ids(client, session, per_genre_artists,
                                        related_per_seed, progress, hop2_sample=hop2_sample)
        rng = np.random.default_rng(seed)
        rng.shuffle(artist_ids)
        if len(artist_ids) > max_artists:
            artist_ids = artist_ids[:max_artists]
            progress(f"Capped to {max_artists} artists")
        progress(f"Gathering top {per_artist} tracks from {len(artist_ids)} artists...")
        pool = {}
        for i, aid in enumerate(artist_ids, 1):
            try:
                for t in client.artist_top_tracks(aid, per_artist):
                    if t.has_preview and t.id not in pool:
                        pool[t.id] = t
            except Exception:  # noqa: BLE001
                continue
            if i % 500 == 0:
                progress(f"  scanned {i}/{len(artist_ids)} artists -> {len(pool)} candidates")
        candidates = list(pool.values())
        np.random.default_rng(seed).shuffle(candidates)  # genre-diverse ordering
        _save_candidates(cand_path, candidates)
        progress(f"Saved {len(candidates)} candidates -> {cand_path}")

    # Only what's not already cached.
    todo = [t for t in candidates if not cache.has(t.id)]
    progress(f"{len(todo)} tracks to harvest ({workers} workers). Target "
             f"{'all' if not target else target} total.")

    t0 = time.time()
    done = 0
    with TemporaryDirectory() as tmp:
        wd = Path(tmp)
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {ex.submit(_process_candidate, t, cfg, wd): t for t in todo}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    track, spec, vibe = res
                    cache.add(track.id, track.title, track.artist, spec, vibe)
                done += 1
                if done % checkpoint_every == 0:
                    cache.save(cache_path)
                    rate = done / (time.time() - t0)
                    progress(f"  {done}/{len(todo)} processed ({rate:.1f}/s) "
                             f"[cache: {len(cache)}]")
                if target and len(cache) >= target:
                    progress(f"Reached target {target}; stopping.")
                    break
        finally:
            # Persist freshly-added tracks before anything can raise, then cancel
            # pending downloads and wait for the few in-flight workers so the temp
            # dir isn't cleaned while they're still reading/writing files in it.
            cache.save(cache_path)
            ex.shutdown(wait=True, cancel_futures=True)
    progress(f"Done. Cache now {len(cache)} tracks -> {cache_path}")
    return cache


def main(argv: Optional[list] = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Broad multi-scene deep-vibe harvest.")
    parser.add_argument("--cache", default="ml_data/spec_cache.npz")
    parser.add_argument("--per-artist", type=int, default=15)
    parser.add_argument("--per-genre-artists", type=int, default=100)
    parser.add_argument("--related-per-seed", type=int, default=8)
    parser.add_argument("--hop2-sample", type=int, default=1500)
    parser.add_argument("--max-artists", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target", type=int, default=0,
                        help="Stop once the cache reaches this many tracks (0 = all).")
    parser.add_argument("--log", default=None, help="Also append progress to this file.")
    args = parser.parse_args(argv)

    log_fh = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None

    def progress(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")

    try:
        harvest_broad_to_cache(
            Path(args.cache), per_artist=args.per_artist,
            per_genre_artists=args.per_genre_artists,
            related_per_seed=args.related_per_seed, hop2_sample=args.hop2_sample,
            max_artists=args.max_artists, workers=args.workers, target=args.target,
            progress=progress,
        )
    finally:
        if log_fh:
            log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
