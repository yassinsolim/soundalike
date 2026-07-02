"""Command-line interface for the soundalike recommender.

Examples:
    soundalike similar --title "Blinding Lights" -n 10
    soundalike similar --title "Believer" --weight energy=2 --weight danceability=1.5
    soundalike profile --seeds "Blinding Lights; Shape of You; One Dance" -n 15
    soundalike profile --file liked_songs.csv -n 25 --exclude-seed-artists
    soundalike stats
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from .dataset import Dataset, load_bundled_dataset
from .features import AUDIO_FEATURES, FeatureConfig, resolve_feature
from .profile import parse_seed_string, read_seed_file
from .recommender import ContentBasedRecommender


def _parse_weights(pairs: Optional[List[str]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"Invalid --weight '{pair}'. Use NAME=VALUE, e.g. energy=2.")
        name, value = pair.split("=", 1)
        try:
            weights[resolve_feature(name)] = float(value)
        except ValueError as exc:
            raise SystemExit(f"Invalid --weight '{pair}': {exc}")
    return weights


def _load_dataset(path: Optional[str]) -> Dataset:
    if path:
        return Dataset.from_csv(path)
    return load_bundled_dataset()


def _build_config(args: argparse.Namespace) -> FeatureConfig:
    return FeatureConfig(
        weights=_parse_weights(getattr(args, "weight", None)),
        metric=getattr(args, "metric", "euclidean"),
    ).validate()


def _print_recommendations(recs, header: str) -> None:
    print(header)
    if not recs:
        print("  (no matches found)")
        return
    width = len(str(len(recs)))
    for i, rec in enumerate(recs, 1):
        print(f"  {i:>{width}}. {rec.title} — {rec.artist}   [{rec.score:.3f}]")


# --------------------------------------------------------------------- commands
def cmd_similar(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    recommender = ContentBasedRecommender(_build_config(args)).fit(dataset)
    try:
        recs = recommender.similar_to(
            args.title,
            artist=args.artist,
            n=args.num,
            exclude_same_artist=args.exclude_artist,
        )
    except LookupError as exc:
        print(str(exc))
        print("Tip: run `soundalike stats` or check the dataset for exact titles.")
        return 1
    seed = args.title + (f" by {args.artist}" if args.artist else "")
    _print_recommendations(recs, f"\nSongs similar to '{seed}':")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    if not args.seeds and not args.file:
        raise SystemExit("Provide seeds with --seeds \"a; b; c\" or --file path.")
    seeds = []
    if args.file:
        seeds.extend(read_seed_file(args.file))
    if args.seeds:
        seeds.extend(parse_seed_string(args.seeds))
    if not seeds:
        raise SystemExit("No seed songs could be parsed.")

    dataset = _load_dataset(args.dataset)
    recommender = ContentBasedRecommender(_build_config(args)).fit(dataset)
    try:
        recs, unmatched = recommender.recommend_for_profile(
            seeds,
            n=args.num,
            exclude_known=not args.keep_known,
            exclude_seed_artists=args.exclude_seed_artists,
        )
    except LookupError as exc:
        print(str(exc))
        return 1

    matched_count = len(seeds) - len(unmatched)
    _print_recommendations(
        recs, f"\nRecommendations for your taste profile ({matched_count}/{len(seeds)} seeds matched):"
    )
    if unmatched:
        print("\nNot found in dataset (ignored):")
        for title, artist in unmatched:
            print(f"  - {title}" + (f" by {artist}" if artist else ""))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    frame = dataset.frame
    print(f"\nDataset: {len(dataset)} songs\n")
    header = f"{'feature':<16}{'min':>8}{'mean':>8}{'max':>8}{'std':>8}   top song"
    print(header)
    print("-" * len(header))
    for feat in AUDIO_FEATURES:
        col = frame[feat]
        top_idx = int(col.idxmax())
        top = f"{frame.at[top_idx, 'title']} — {frame.at[top_idx, 'primary_artist']}"
        print(
            f"{feat:<16}{col.min():>8.1f}{col.mean():>8.1f}{col.max():>8.1f}{col.std():>8.1f}   {top}"
        )
    return 0


# ------------------------------------------------------------- live (Spotify)
def _spotify_client(config):
    from .spotify import SpotifyAuth, SpotifyClient

    return SpotifyClient(SpotifyAuth(config))


def _seeds_from_spotify(client, source: str, limit: int, time_range: str):
    if source == "liked":
        tracks = client.liked_tracks(limit)
    elif source == "recent":
        tracks = client.recently_played(limit)
    else:  # top
        tracks = client.top_tracks(time_range, limit)
    return [(t["title"], t["primary_artist"] or t["artist"]) for t in tracks]


def _create_playlist_from(client, name: str, recs) -> None:
    uris = []
    for rec in recs:
        found = client.search_track(rec.title, rec.artist)
        if found:
            uris.append(found["uri"])
    if not uris:
        print("Could not map any recommendations back to Spotify tracks.")
        return
    playlist = client.create_playlist(
        name, uris, description="Generated by soundalike", public=False
    )
    link = playlist.get("external_urls", {}).get("spotify", "")
    print(f"\nCreated playlist '{name}' with {len(uris)} tracks. {link}")


def cmd_login(args: argparse.Namespace) -> int:
    from .config import Config
    from .spotify import SpotifyAuth, SpotifyClient

    config = Config.from_env()
    auth = SpotifyAuth(config)
    auth.authorize_interactive()
    user = SpotifyClient(auth).current_user()
    print(f"Logged in as {user.get('display_name') or user.get('id')}.")
    print(f"Token cached at {auth.token_path}")
    return 0


def cmd_whoami(args: argparse.Namespace) -> int:
    from .config import Config

    config = Config.from_env()
    user = _spotify_client(config).current_user()
    print(f"{user.get('display_name') or user.get('id')} ({user.get('id')})")
    followers = user.get("followers", {}).get("total")
    if followers is not None:
        print(f"Followers: {followers}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    import csv

    from .config import Config

    config = Config.from_env()
    seeds = _seeds_from_spotify(
        _spotify_client(config), args.source, args.limit, args.time_range
    )
    out = args.out or f"{args.source}_tracks.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["title", "artist"])
        writer.writerows(seeds)
    print(f"Wrote {len(seeds)} tracks from your '{args.source}' to {out}")
    print("Feed it to the offline engine with: "
          f'soundalike profile --file "{out}"')
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from .config import Config

    config = Config.from_env()
    client = _spotify_client(config)
    seeds = _seeds_from_spotify(client, args.source, args.seed_limit, args.time_range)
    if not seeds:
        print(f"No '{args.source}' tracks found on your account.")
        return 1

    if args.engine == "lastfm":
        config.require_lastfm()
        from .lastfm import LastFmClient, LastFmRecommender

        engine = LastFmRecommender(LastFmClient(config.lastfm_api_key))
        recs, skipped = engine.recommend(seeds, n=args.num, per_seed=args.per_seed)
        _print_recommendations(
            recs, f"\nLast.fm recommendations from your '{args.source}' tracks:"
        )
        if skipped:
            print(f"\n({len(skipped)} seed(s) skipped for missing artist info.)")
    else:  # content engine over the bundled dataset
        dataset = _load_dataset(args.dataset)
        recommender = ContentBasedRecommender(_build_config(args)).fit(dataset)
        recs, unmatched = recommender.recommend_for_profile(
            seeds, n=args.num, exclude_known=True
        )
        matched = len(seeds) - len(unmatched)
        _print_recommendations(
            recs,
            f"\nContent-based recommendations "
            f"({matched}/{len(seeds)} of your '{args.source}' tracks are in the dataset):",
        )
        if matched == 0:
            print("None of your tracks are in the bundled dataset — "
                  "use `--engine lastfm` for full coverage.")
            return 1

    if args.playlist:
        _create_playlist_from(client, args.playlist, recs)
    return 0


# ------------------------------------------------------- acoustic (DSP) engine
def _parse_audio_weights(pairs: Optional[List[str]]) -> Dict[str, float]:
    from .audio.features import FEATURE_NAMES

    weights: Dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"Invalid --weight '{pair}'. Use NAME=VALUE, e.g. tempo=2.")
        name, value = pair.split("=", 1)
        name = name.strip()
        if name not in FEATURE_NAMES:
            raise SystemExit(
                f"Unknown acoustic feature '{name}'. Valid: {', '.join(FEATURE_NAMES)}"
            )
        try:
            weights[name] = float(value)
        except ValueError:
            raise SystemExit(f"Invalid weight value in '{pair}'.")
    return weights


def cmd_audio_features(args: argparse.Namespace) -> int:
    from .audio import DeezerClient, features_from_file
    from .audio.features import FEATURE_DESCRIPTIONS
    from tempfile import TemporaryDirectory
    from pathlib import Path

    client = DeezerClient()
    track = client.search_track(args.title, args.artist)
    if track is None or not track.has_preview:
        print(f"No previewable track found for '{args.title}'"
              + (f" by {args.artist}" if args.artist else "") + ".")
        return 1

    print(f"Analyzing: {track.title} — {track.artist}  (Deezer id {track.id})")
    with TemporaryDirectory() as tmp:
        dest = Path(tmp) / f"{track.id}.mp3"
        client.download_preview(track, dest)
        feats = features_from_file(str(dest))

    d = feats.to_dict()
    print("\nMeasured acoustic features (computed from the audio):")
    for name in ["tempo", "rms_energy", "spectral_centroid", "spectral_rolloff",
                 "spectral_bandwidth", "zero_crossing_rate", "spectral_contrast"]:
        desc = FEATURE_DESCRIPTIONS.get(name, "")
        print(f"  {name:<20} {d[name]:>10.3f}   {desc}")
    mfcc = ", ".join(f"{x:.1f}" for x in d["mfcc"])
    print(f"  {'mfcc (timbre)':<20} [{mfcc}]")
    return 0


def cmd_audio_similar(args: argparse.Namespace) -> int:
    from .audio import AudioSimilarityRecommender

    seeds: List = []
    if args.source:
        from .config import Config
        config = Config.from_env()
        seeds.extend(_seeds_from_spotify(_spotify_client(config), args.source,
                                         args.seed_limit, args.time_range))
    if args.file:
        seeds.extend(read_seed_file(args.file))
    if args.seeds:
        seeds.extend(parse_seed_string(args.seeds))
    if args.title:
        seeds.insert(0, (args.title, args.artist))
    if not seeds:
        raise SystemExit(
            "Provide seeds via --title/--seeds, --file, or --source (your Spotify tracks)."
        )

    recommender = AudioSimilarityRecommender(
        weights=_parse_audio_weights(args.weight) or None,
        progress=print,
    )
    try:
        recs, unmatched = recommender.recommend(
            seeds, n=args.num, per_artist=args.per_artist, related_per_seed=args.related
        )
    except (LookupError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    _print_recommendations(recs, "\nSongs acoustically similar to your seeds (ranked by DSP):")
    if unmatched:
        print(f"\n({len(unmatched)} seed(s) had no previewable match and were skipped.)")
    return 0


def cmd_learned_similar(args: argparse.Namespace) -> int:
    from .ml.recommend import TrainedRecommender

    try:
        rec = TrainedRecommender(args.model_dir)
    except FileNotFoundError:
        print(f"No trained model found in '{args.model_dir}'. Train one first "
              "(see soundalike.ml.train_scale).")
        return 1
    try:
        neighbors = rec.neighbors_for_song(args.title, args.artist, n=args.num)
    except LookupError as exc:
        print(str(exc))
        return 1
    seed = args.title + (f" by {args.artist}" if args.artist else "")
    print(f"\nSongs the trained model finds acoustically closest to '{seed}':")
    for i, (title, artist, genre, score) in enumerate(neighbors, 1):
        print(f"  {i:>2}. {title} — {artist}  [{genre}]  ({score:.3f})")
    return 0


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundalike",
        description="Find songs similar to the ones you like (offline, audio-feature based).",
    )
    parser.add_argument(
        "--dataset",
        help="Path to a CSV with audio-feature columns. Defaults to the bundled dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, default_num: int = 10) -> None:
        sp.add_argument("-n", "--num", type=int, default=default_num, help="Number of results.")
        sp.add_argument(
            "--metric", choices=["euclidean", "cosine"], default="euclidean",
            help="Similarity metric (default: euclidean).",
        )
        sp.add_argument(
            "--weight", action="append", metavar="NAME=VALUE",
            help="Weight a feature, e.g. --weight energy=2 (repeatable).",
        )

    p_similar = sub.add_parser("similar", help="Songs similar to one seed song.")
    p_similar.add_argument("--title", required=True, help="Seed song title.")
    p_similar.add_argument("--artist", help="Disambiguate the seed by artist.")
    p_similar.add_argument(
        "--exclude-artist", action="store_true", help="Exclude the seed's own artist."
    )
    add_common(p_similar)
    p_similar.set_defaults(func=cmd_similar)

    p_profile = sub.add_parser("profile", help="Songs for a multi-song taste profile.")
    p_profile.add_argument("--seeds", help='Inline seeds: "Title - Artist; Title2; ...".')
    p_profile.add_argument("--file", help="Path to a .txt or .csv of seed songs.")
    p_profile.add_argument(
        "--keep-known", action="store_true", help="Keep seed songs in the results."
    )
    p_profile.add_argument(
        "--exclude-seed-artists", action="store_true",
        help="Exclude every artist present in your seeds.",
    )
    add_common(p_profile)
    p_profile.set_defaults(func=cmd_profile)

    p_stats = sub.add_parser("stats", help="Summary statistics of the dataset.")
    p_stats.set_defaults(func=cmd_stats)

    # -------------------------------------------------- live commands (Spotify)
    p_login = sub.add_parser("login", help="Authorize with Spotify (OAuth PKCE).")
    p_login.set_defaults(func=cmd_login)

    p_whoami = sub.add_parser("whoami", help="Show the logged-in Spotify user.")
    p_whoami.set_defaults(func=cmd_whoami)

    def add_source(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--source", choices=["top", "liked", "recent"], default="top",
            help="Which of your tracks to use (default: top).",
        )
        sp.add_argument(
            "--time-range", choices=["short_term", "medium_term", "long_term"],
            default="medium_term",
            help="Window for 'top' source (short=4wk, medium=6mo, long=years).",
        )

    p_pull = sub.add_parser("pull", help="Export your Spotify tracks to a seed CSV.")
    add_source(p_pull)
    p_pull.add_argument("--limit", type=int, default=100, help="How many tracks to fetch.")
    p_pull.add_argument("--out", help="Output CSV path (default: <source>_tracks.csv).")
    p_pull.set_defaults(func=cmd_pull)

    p_reco = sub.add_parser(
        "recommend", help="Recommend songs from your live Spotify taste."
    )
    add_source(p_reco)
    p_reco.add_argument(
        "--engine", choices=["content", "lastfm"], default="content",
        help="content = bundled audio-feature dataset; lastfm = full catalog (needs key).",
    )
    p_reco.add_argument(
        "--seed-limit", type=int, default=50, help="How many of your tracks to seed from.",
    )
    p_reco.add_argument(
        "--per-seed", type=int, default=50, help="Last.fm similar tracks per seed.",
    )
    p_reco.add_argument(
        "--playlist", metavar="NAME", help="Also save the results as a new Spotify playlist.",
    )
    add_common(p_reco, default_num=25)
    p_reco.set_defaults(func=cmd_recommend)

    # ------------------------------------------ acoustic (DSP) similarity engine
    p_afeat = sub.add_parser(
        "audio-features",
        help="Measure a song's acoustic features from its actual audio (DSP).",
    )
    p_afeat.add_argument("--title", required=True, help="Song title.")
    p_afeat.add_argument("--artist", help="Artist, to disambiguate.")
    p_afeat.set_defaults(func=cmd_audio_features)

    p_asim = sub.add_parser(
        "audio-similar",
        help="Recommend songs by measured acoustic similarity (the science engine).",
    )
    p_asim.add_argument("--title", help="A single seed song title.")
    p_asim.add_argument("--artist", help="Seed artist, to disambiguate.")
    p_asim.add_argument("--seeds", help='Inline seeds: "Title - Artist; Title2; ...".')
    p_asim.add_argument("--file", help="Path to a .txt/.csv of seed songs.")
    p_asim.add_argument(
        "--source", choices=["top", "liked", "recent"],
        help="Use your live Spotify tracks as seeds (requires login).",
    )
    p_asim.add_argument(
        "--time-range", choices=["short_term", "medium_term", "long_term"],
        default="medium_term", help="Window for the 'top' source.",
    )
    p_asim.add_argument("--seed-limit", type=int, default=10, help="How many Spotify seeds.")
    p_asim.add_argument("-n", "--num", type=int, default=20, help="Number of results.")
    p_asim.add_argument(
        "--per-artist", type=int, default=25, help="Candidate tracks per artist.",
    )
    p_asim.add_argument(
        "--related", type=int, default=6, help="Neighbouring artists per seed (breadth).",
    )
    p_asim.add_argument(
        "--weight", action="append", metavar="NAME=VALUE",
        help="Weight an acoustic feature, e.g. --weight tempo=2 (repeatable).",
    )
    p_asim.set_defaults(func=cmd_audio_similar)

    p_learned = sub.add_parser(
        "learned-similar",
        help="Recommend using a trained neural embedding model (see ml.train_scale).",
    )
    p_learned.add_argument("--title", required=True, help="Seed song title.")
    p_learned.add_argument("--artist", help="Seed artist, to disambiguate.")
    p_learned.add_argument("--model-dir", default="ml_data/model_fma",
                           help="Directory holding encoder.pt + embeddings.npz.")
    p_learned.add_argument("-n", "--num", type=int, default=15, help="Number of results.")
    p_learned.set_defaults(func=cmd_learned_similar)

    return parser


def _configure_stdout() -> None:
    """Best-effort UTF-8 output so accented / non-Latin song titles render."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
