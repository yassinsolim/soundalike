"""Legacy scene-tag evaluation helpers.

This module remains for backwards-compatible unit tests only.  It is **not**
production acceptance evidence: most unknown artists receive no externally
verified label, and the original tests exercised synthetic clustered indices.
Use :mod:`soundalike.ml.real_benchmark` and the versioned sourced-pair artifacts
for production claims.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Seed catalogue — 50 representative songs across ≥12 scenes
# Format: (title, artist, scene_label)
# We only include songs that are realistically in the 87k bundled library or
# the 272k production library.  Seeds are chosen to stress the common failure
# modes: cross-genre false-positive matches and mainstream / niche balance.
# ---------------------------------------------------------------------------
EVAL_SEEDS: List[Tuple[str, str, str]] = [
    # ── RAP / HIP-HOP ────────────────────────────────────────────────────────
    ("Money Trees", "Kendrick Lamar", "rap"),
    ("HUMBLE.", "Kendrick Lamar", "rap"),
    ("Sicko Mode", "Travis Scott", "rap"),
    ("God's Plan", "Drake", "rap"),
    ("Rich Flex", "Drake", "rap"),
    # ── R&B / SOUL ───────────────────────────────────────────────────────────
    ("Blinding Lights", "The Weeknd", "rnb"),
    ("Die For You", "The Weeknd", "rnb"),
    ("Creepin'", "Metro Boomin", "rnb"),
    ("About Damn Time", "Lizzo", "rnb"),
    ("Kill Bill", "SZA", "rnb"),
    # ── INDIE / ALTERNATIVE ──────────────────────────────────────────────────
    ("Dog Days Are Over", "Florence + The Machine", "indie"),
    ("Ribs", "Lorde", "indie"),
    ("Motion Sickness", "Phoebe Bridgers", "indie"),
    ("Bloom", "The Paper Kites", "indie"),
    ("Falling", "Novo Amor", "indie"),
    # ── SHOEGAZE / DREAM-POP ─────────────────────────────────────────────────
    ("Only Shallow", "My Bloody Valentine", "shoegaze"),
    ("When the Sun Hits", "Slowdive", "shoegaze"),
    ("Spitting Off the Edge of the World", "Yeah Yeah Yeahs", "shoegaze"),
    ("Vapour Trail", "Ride", "shoegaze"),
    ("Sometimes", "My Bloody Valentine", "shoegaze"),
    # ── HYPERPOP / PC MUSIC ──────────────────────────────────────────────────
    ("Cake", "Fraxiom", "hyperpop"),
    ("911/Mr. Lonely", "Lady Gaga", "hyperpop"),
    ("Ring Ring", "Charli XCX", "hyperpop"),
    ("Unlock It", "Charli XCX", "hyperpop"),
    ("detonate", "glaive", "hyperpop"),
    # ── ELECTRONIC / DANCE ───────────────────────────────────────────────────
    ("Strobe", "deadmau5", "electronic"),
    ("Levels", "Avicii", "electronic"),
    ("Opus", "Eric Prydz", "electronic"),
    ("One More Time", "Daft Punk", "electronic"),
    ("Around the World", "Daft Punk", "electronic"),
    # ── METAL ────────────────────────────────────────────────────────────────
    ("Master of Puppets", "Metallica", "metal"),
    ("Chop Suey!", "System of a Down", "metal"),
    ("The Trooper", "Iron Maiden", "metal"),
    ("Paranoid", "Black Sabbath", "metal"),
    ("Ten Thousand Fists", "Disturbed", "metal"),
    # ── JAZZ ─────────────────────────────────────────────────────────────────
    ("Take Five", "Dave Brubeck Quartet", "jazz"),
    ("So What", "Miles Davis", "jazz"),
    ("Autumn Leaves", "Bill Evans", "jazz"),
    ("My Favorite Things", "John Coltrane", "jazz"),
    ("Blue in Green", "Miles Davis", "jazz"),
    # ── CITY-POP / J-POP / K-POP ─────────────────────────────────────────────
    ("Mayonaka no Door", "Miki Matsubara", "city_pop"),
    ("Plastic Love", "Mariya Takeuchi", "city_pop"),
    ("ETA", "NewJeans", "kpop"),
    ("Hype Boy", "NewJeans", "kpop"),
    ("OMG", "NewJeans", "kpop"),
    # ── LATIN / AFROBEATS ────────────────────────────────────────────────────
    ("Quevedo: Bzrp Music Sessions, Vol. 52", "Bizarrap", "latin"),
    ("Titi Me Pregunto", "Bad Bunny", "latin"),
    ("Water", "Tyla", "afrobeats"),
    ("Essence", "Wizkid", "afrobeats"),
    # ── GENRE-BLENDING / DEEP CUTS (difficult cases) ─────────────────────────
    ("Pink + White", "Frank Ocean", "difficult"),
    ("Motion Picture Soundtrack", "Radiohead", "difficult"),
    ("Pyramid Song", "Radiohead", "difficult"),
    ("Daydreaming", "Radiohead", "difficult"),
    ("How Soon Is Now?", "The Smiths", "difficult"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Held-out set — 20 deliberately difficult seeds (genre-blending, deep-cuts,
# niche artists, scenes with many junk-track variants in catalogs).  These are
# NOT in the main eval suite and are used ONLY for the "20 difficult seed" AC.
# Format: (title, artist, scene_label)
# ─────────────────────────────────────────────────────────────────────────────
HELD_OUT_SEEDS: List[Tuple[str, str, str]] = [
    # Genre-blending and cross-scene difficult cases
    ("Redbone", "Childish Gambino", "rnb"),            # funk/psych soul; often near rap
    ("Nights", "Frank Ocean", "rnb"),                  # experimental R&B album cut
    ("Chanel", "Frank Ocean", "rnb"),                  # genre-blending R&B/alt
    ("Exit Music (For a Film)", "Radiohead", "difficult"),  # cinematic alt-rock
    ("Kid A", "Radiohead", "difficult"),               # electronic art-rock — often misclassified
    # Deep-cut shoegaze (not in main suite)
    ("Alison", "Slowdive", "shoegaze"),
    ("Souvlaki Space Station", "Slowdive", "shoegaze"),
    # Niche digicore / hyperpop deep cuts (different from EVAL_SEEDS glaive/fraxiom)
    ("Stay Away", "glaive", "hyperpop"),
    ("i need you here", "glaive", "hyperpop"),
    # City-pop (different artists/songs from EVAL_SEEDS)
    ("Sweetly", "Anri", "city_pop"),
    # Metal with heavy junk contamination (different Metallica songs)
    ("Enter Sandman", "Metallica", "metal"),           # heavily tributed
    ("Fade to Black", "Metallica", "metal"),
    # Jazz standards (different from EVAL_SEEDS Dave Brubeck / Miles Davis / Bill Evans)
    ("Round Midnight", "Thelonious Monk", "jazz"),
    ("In a Sentimental Mood", "John Coltrane", "jazz"),
    # Indie deep cuts (different from EVAL_SEEDS indie seeds)
    ("Where Is My Mind?", "Pixies", "indie"),
    ("Heroin", "The Velvet Underground", "indie"),
    # R&B / rap with junk contamination (different from EVAL_SEEDS)
    ("Starboy", "The Weeknd", "rnb"),                  # different from Die For You / Blinding Lights
    ("Save Your Tears", "The Weeknd", "rnb"),
    ("Chemical", "Post Malone", "difficult"),          # alt-rock/rap crossover
    ("Golden Hour", "JVKE", "indie"),                  # indie/pop crossover deep cut
]

# Scene groups that shouldn't cross-contaminate (e.g., metal should not
# recommend jazz).  Tuple elements are the scene labels that are ALLOWED
# to appear together (broader-genre relatives).
_SCENE_RELATIVES: Dict[str, Set[str]] = {
    "rap": {"rap", "rnb"},
    "rnb": {"rnb", "rap"},
    "indie": {"indie", "shoegaze"},
    "shoegaze": {"shoegaze", "indie"},
    "hyperpop": {"hyperpop", "electronic"},
    "electronic": {"electronic", "hyperpop"},
    "metal": {"metal"},
    "jazz": {"jazz"},
    "city_pop": {"city_pop", "kpop"},
    "kpop": {"kpop", "city_pop"},
    "latin": {"latin"},
    "afrobeats": {"afrobeats", "latin"},
    "difficult": {"difficult", "indie", "shoegaze", "rnb"},
}

# Title patterns that identify "junk" tracks (case-insensitive regex).
JUNK_PATTERNS: List[str] = [
    r"\bslowed\b",
    r"\breverb\b",
    r"\bkaraoke\b",
    r"\btribute\b",
    r"\bnightcore\b",
    r"\bsped[ -]?up\b",
    r"\bspeed[ -]?up\b",
    r"\binstrumental\s+version\b",
    r"\ba\s+cappella\b",
    r"\bcover\s+version\b",
    r"\bpiano\s+version\b",
    r"\bchill\s+version\b",
    r"\bremake\b",
    r"\bkung fu remix\b",
    r"\bmarimba remix\b",
    r"\bringtone\b",
    r"\bmashup\b",
    r"\bmedley\b",
    r"\bsing-?along\b",
]
_JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)


def is_junk(title: str, artist: str = "") -> bool:
    """True if the track is a junk derivative (slowed, karaoke, tribute, etc.)."""
    combined = f"{title} {artist}"
    if _JUNK_RE.search(combined):
        return True
    # Artist-level junk markers
    artist_lower = artist.lower()
    for pat in ("karaoke", "tribute", "covers band", "coverband", "soundalike",
                "sound alike", "sound-alike"):
        if pat in artist_lower:
            return True
    return False


def is_seed_title_in_result(seed_title: str, result_title: str) -> bool:
    """True if the seed's title appears verbatim in the result (mashup detection)."""
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return " ".join(s.casefold().split())

    st = norm(seed_title)
    rt = norm(result_title)
    if not st or not rt:
        return False
    return st in rt or rt in st and rt != st


# ---------------------------------------------------------------------------
# Evaluation data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Top-N recommendations for a single seed."""
    seed_title: str
    seed_artist: str
    seed_scene: str
    found_in_index: bool
    recs: List[Dict[str, Any]] = field(default_factory=list)   # [{title, artist, score, ...}]
    # Per-recommendation flags
    junk_flags: List[bool] = field(default_factory=list)
    seed_mashup_flags: List[bool] = field(default_factory=list)
    same_artist_flags: List[bool] = field(default_factory=list)
    scene_coherent_flags: List[bool] = field(default_factory=list)  # requires scene_tags
    scene_tags: Optional[List[Optional[str]]] = None  # scene label per rec (None = unknown)


@dataclass
class EvalReport:
    """Aggregate statistics over the evaluation suite."""
    n_seeds: int
    n_found: int
    primary_score: float         # mean scene coherence over top-5 for found seeds
    top1_coherent: float         # fraction of seeds where rank-1 is scene-coherent
    junk_rate: float             # mean fraction of top-5 that are junk
    mashup_rate: float           # mean fraction of top-5 that are seed-title mashups
    same_artist_rate: float      # mean fraction of top-5 from seed artist (should be 0)
    per_scene: Dict[str, Dict[str, float]] = field(default_factory=dict)
    results: List[EvalResult] = field(default_factory=list)
    method_name: str = "baseline"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Scene inference from library
# ---------------------------------------------------------------------------

# Broad genre words that help infer a scene from a recommendation's title/artist.
# This is intentionally rough; it's only used when we don't have explicit scene labels.
_SCENE_KEYWORDS: Dict[str, List[str]] = {
    "rap": ["rap", "hip hop", "hip-hop", "trap", "drill", "g-funk", "boom bap"],
    "rnb": ["r&b", "soul", "neo soul", "r and b"],
    "indie": ["indie", "folk", "acoustic", "singer-songwriter"],
    "shoegaze": ["shoegaze", "dreampop", "dream pop", "noise pop"],
    "hyperpop": ["hyperpop", "pc music", "digicore"],
    "electronic": ["electronic", "edm", "techno", "house", "trance", "dubstep", "dnb",
                   "drum and bass", "ambient"],
    "metal": ["metal", "hardcore", "thrash", "doom", "death metal", "black metal"],
    "jazz": ["jazz", "bebop", "swing", "bossa nova"],
    "city_pop": ["city pop", "city-pop", "j-pop", "jpop", "shibuya-kei"],
    "kpop": ["k-pop", "kpop"],
    "latin": ["latin", "reggaeton", "cumbia", "salsa", "bachata", "corrido"],
    "afrobeats": ["afrobeats", "afropop", "afro", "amapiano"],
}

# Comprehensive curated artist→scene mapping.  This is the primary scene signal —
# without it, ≥95 % of library tracks have "unknown" scene, making primary_score
# measure noise rather than quality.  The list is large enough to tag a realistic
# fraction of top-5 recommendations for each eval seed.
#
# Format: artist_name_casefolded → scene_label
_ARTIST_SCENE: Dict[str, str] = {
    # ── RAP / HIP-HOP ────────────────────────────────────────────────────────
    "kendrick lamar": "rap", "drake": "rap", "travis scott": "rap",
    "j. cole": "rap", "lil uzi vert": "rap", "post malone": "rap",
    "21 savage": "rap", "future": "rap", "lil baby": "rap",
    "gunna": "rap", "nba youngboy": "rap", "lil durk": "rap",
    "nicki minaj": "rap", "cardi b": "rap", "megan thee stallion": "rap",
    "a$ap rocky": "rap", "asap rocky": "rap", "tyler, the creator": "rap",
    "childish gambino": "rap", "chance the rapper": "rap",
    "logic": "rap", "wiz khalifa": "rap", "mac miller": "rap",
    "kid cudi": "rap", "kanye west": "rap", "jay-z": "rap",
    "eminem": "rap", "50 cent": "rap", "snoop dogg": "rap",
    "biggie smalls": "rap", "notorious b.i.g.": "rap", "2pac": "rap",
    "nas": "rap", "wu-tang clan": "rap", "method man": "rap",
    "ice cube": "rap", "dr. dre": "rap", "nwa": "rap",
    "e-40": "rap", "too $hort": "rap", "too short": "rap",
    "master p": "rap", "dj scheme": "rap", "kodak black": "rap",
    "smokepurpp": "rap", "mustard": "rap", "lil nas x": "rap",
    "6lack": "rap", "russ": "rap", "nardo wick": "rap",
    "metro boomin": "rap", "21 savage": "rap", "cochise": "rap",
    "famous dex": "rap", "kevin gates": "rap", "roddy ricch": "rap",
    "polo g": "rap", "juice wrld": "rap", "ynw melly": "rap",
    "dababy": "rap", "young thug": "rap", "offset": "rap",
    "quavo": "rap", "takeoff": "rap", "migos": "rap",
    "doja cat": "rap", "saweetie": "rap", "city girls": "rap",
    "fivio foreign": "rap", "pop smoke": "rap", "pusha t": "rap",
    "lupe fiasco": "rap", "mos def": "rap", "common": "rap",
    "talib kweli": "rap", "big l": "rap", "big pun": "rap",
    "dmx": "rap", "rick ross": "rap", "meek mill": "rap",
    "gucci mane": "rap", "young jeezy": "rap", "t.i.": "rap",
    "lil wayne": "rap", "birdman": "rap", "cash money": "rap",
    "young buck": "rap", "game": "rap", "the game": "rap",
    "bones": "rap", "bones (rapper)": "rap", "robb bank$": "rap",
    "38 spesh": "rap", "rushy": "rap", "madmarcc": "rap",
    "xxxtentacion": "rap", "xxx tentacion": "rap",
    # ── R&B / SOUL ───────────────────────────────────────────────────────────
    "the weeknd": "rnb", "sza": "rnb", "frank ocean": "rnb",
    "usher": "rnb", "beyoncé": "rnb", "beyonce": "rnb",
    "alicia keys": "rnb", "john legend": "rnb", "adele": "rnb",
    "rihanna": "rnb", "mariah carey": "rnb", "mary j. blige": "rnb",
    "the-dream": "rnb", "miguel": "rnb", "h.e.r.": "rnb",
    "summer walker": "rnb", "jhené aiko": "rnb", "jhene aiko": "rnb",
    "daniel caesar": "rnb", "bryson tiller": "rnb", "6lack": "rnb",
    "kehlani": "rnb", "ella mai": "rnb", "normani": "rnb",
    "brent faiyaz": "rnb", "giveon": "rnb", "lucky daye": "rnb",
    "snoh aalegra": "rnb", "ari lennox": "rnb", "cleo sol": "rnb",
    "amber mark": "rnb", "mahalia": "rnb", "jorja smith": "rnb",
    "victoria monet": "rnb", "blxst": "rnb", "dom kennedy": "rnb",
    "chris brown": "rnb", "trey songz": "rnb", "tank": "rnb",
    "r. kelly": "rnb", "joe": "rnb", "ginuwine": "rnb",
    "omarion": "rnb", "b2k": "rnb", "playa": "rnb",
    "babyface": "rnb", "kenny latimore": "rnb", "musiq soulchild": "rnb",
    "neo": "rnb", "maxwell": "rnb", "d'angelo": "rnb",
    "erykah badu": "rnb", "lauryn hill": "rnb", "fugees": "rnb",
    "jill scott": "rnb", "india.arie": "rnb", "ledisi": "rnb",
    "stevie wonder": "rnb", "marvin gaye": "rnb", "al green": "rnb",
    "otis redding": "rnb", "sam cooke": "rnb", "james brown": "rnb",
    "chiiild": "rnb", "lyn lapid": "rnb", "the neighbourhood": "rnb",
    "h.e.r.": "rnb", "naomi sharon": "rnb", "capella grey": "rnb",
    "tv girl": "rnb", "hybs": "rnb", "james vincent mcmorrow": "rnb",
    "awake": "rnb", "punchnello": "rnb",
    "somo": "rnb", "somos": "rnb",
    "omah lay": "rnb", "crush": "rnb",
    "nick jonas": "rnb",
    # ── INDIE / ALTERNATIVE ──────────────────────────────────────────────────
    "phoebe bridgers": "indie", "boygenius": "indie",
    "mitski": "indie", "japanese breakfast": "indie",
    "snail mail": "indie", "lucy dacus": "indie",
    "soccer mommy": "indie", "alex g": "indie",
    "beabadoobee": "indie", "clairo": "indie",
    "wallows": "indie", "cavetown": "indie",
    "conan gray": "indie", "rex orange county": "indie",
    "eden": "indie", "hozier": "indie",
    "the paper kites": "indie", "novo amor": "indie",
    "bon iver": "indie", "sufjan stevens": "indie",
    "iron & wine": "indie", "fleet foxes": "indie",
    "mountain goats": "indie", "sun kil moon": "indie",
    "bright eyes": "indie", "conor oberst": "indie",
    "death cab for cutie": "indie", "the shins": "indie",
    "modest mouse": "indie", "built to spill": "indie",
    "pavement": "indie", "neutral milk hotel": "indie",
    "belle and sebastian": "indie", "arab strap": "indie",
    "the national": "indie", "arcade fire": "indie",
    "vampire weekend": "indie", "grizzly bear": "indie",
    "animal collective": "indie", "tame impala": "indie",
    "mgmt": "indie", "beach house": "indie",
    "real estate": "indie", "wild nothing": "indie",
    "kurt vile": "indie", "war on drugs": "indie",
    "kevin morby": "indie", "tomberlin": "indie",
    "lorde": "indie", "florence + the machine": "indie",
    "fka twigs": "indie", "weyes blood": "indie",
    "angel olsen": "indie", "sharon van etten": "indie",
    "courtney barnett": "indie", "julia jacklin": "indie",
    "lucy rose": "indie", "laura marling": "indie",
    "angus & julia stone": "indie", "sara bareilles": "indie",
    "lizzy mcalpine": "indie", "blondshell": "indie",
    "billie eilish": "indie", "oliver tree": "indie",
    "chelsea jordan": "indie", "kevin atwater": "indie",
    "cocoon": "indie", "electric guest": "indie",
    # ── SHOEGAZE / DREAM-POP ─────────────────────────────────────────────────
    "my bloody valentine": "shoegaze", "slowdive": "shoegaze",
    "ride": "shoegaze", "lush": "shoegaze",
    "chapterhouse": "shoegaze", "pale saints": "shoegaze",
    "medicine": "shoegaze", "curve": "shoegaze",
    "swervedriver": "shoegaze", "moose": "shoegaze",
    "catherine wheel": "shoegaze", "drop nineteens": "shoegaze",
    "mazzy star": "shoegaze", "galaxie 500": "shoegaze",
    "luna": "shoegaze", "cocteau twins": "shoegaze",
    "beach house": "shoegaze", "cigarettes after sex": "shoegaze",
    "flying saucer attack": "shoegaze", "starflyer 59": "shoegaze",
    "nothing": "shoegaze", "alcest": "shoegaze",
    "deafheaven": "shoegaze", "touche amore": "shoegaze",
    "whirr": "shoegaze", "title fight": "shoegaze",
    "moose blood": "shoegaze", "microwave": "shoegaze",
    "joy division": "shoegaze", "bauhaus": "shoegaze",
    "the cure": "shoegaze", "echo & the bunnymen": "shoegaze",
    "airiel": "shoegaze", "lilys": "shoegaze",
    "diiv": "shoegaze", "sugar": "shoegaze",
    "pavement": "shoegaze", "sebadoh": "shoegaze",
    "guided by voices": "shoegaze", "yo la tengo": "shoegaze",
    "mogwai": "shoegaze", "sigur ros": "shoegaze", "sigur rós": "shoegaze",
    "godspeed you": "shoegaze", "god is an astronaut": "shoegaze",
    "explosions in the sky": "shoegaze",
    "inspiral carpets": "shoegaze", "this mortal coil": "shoegaze",
    "pastel ghost": "shoegaze", "citizen": "shoegaze",
    "higher power": "shoegaze",
    # ── HYPERPOP / PC MUSIC ──────────────────────────────────────────────────
    "100 gecs": "hyperpop", "charli xcx": "hyperpop",
    "sophie": "hyperpop", "a.g. cook": "hyperpop",
    "fraxiom": "hyperpop", "glaive": "hyperpop",
    "ericdoa": "hyperpop", "yeule": "hyperpop",
    "underscores": "hyperpop", "gupi": "hyperpop",
    "food house": "hyperpop", "dorian electra": "hyperpop",
    "umru": "hyperpop", "elyotto": "hyperpop",
    "brakence": "hyperpop", "midwxst": "hyperpop",
    "osquinn": "hyperpop", "glitch gum": "hyperpop",
    "p4rkr": "hyperpop", "rebzyyx": "hyperpop",
    "shxwnx": "hyperpop", "machine girl": "hyperpop",
    "lady gaga": "hyperpop",  # 911/Mr.Lonely era
    "arca": "hyperpop", "elyotto": "hyperpop",
    "rustie": "electronic",  # bridges to electronic
    # ── ELECTRONIC / DANCE ───────────────────────────────────────────────────
    "daft punk": "electronic", "deadmau5": "electronic",
    "avicii": "electronic", "calvin harris": "electronic",
    "swedish house mafia": "electronic", "tiësto": "electronic",
    "tiesto": "electronic", "armin van buuren": "electronic",
    "paul van dyk": "electronic", "above & beyond": "electronic",
    "eric prydz": "electronic", "justice": "electronic",
    "chemical brothers": "electronic", "prodigy": "electronic",
    "aphex twin": "electronic", "boards of canada": "electronic",
    "burial": "electronic", "four tet": "electronic",
    "caribou": "electronic", "floating points": "electronic",
    "james blake": "electronic", "bonobo": "electronic",
    "flume": "electronic", "kaytranada": "electronic",
    "disclosure": "electronic", "bicep": "electronic",
    "arca": "electronic", "arca (artist)": "electronic",
    "rustie": "electronic", "sophie (producer)": "electronic",
    "moderat": "electronic", "apparat": "electronic",
    "massive attack": "electronic", "portishead": "electronic",
    "tricky": "electronic", "moloko": "electronic",
    "bjork": "electronic", "björk": "electronic",
    "autechre": "electronic", "squarepusher": "electronic",
    "venetian snares": "electronic", "mu-ziq": "electronic",
    "gesaffelstein": "electronic", "kavinsky": "electronic",
    "m83": "electronic", "tycho": "electronic",
    "odesza": "electronic", "petit biscuit": "electronic",
    "washed out": "electronic", "tycho": "electronic",
    "kyau & albert": "electronic", "grimesx": "electronic",
    "ian pooley": "electronic", "oliver schories": "electronic",
    "cassian": "electronic", "sound quelle": "electronic",
    "kc lights": "electronic", "craig david": "electronic",
    "papulin": "electronic", "teddy loid": "electronic", "teddyloid": "electronic",
    "night tempo": "electronic",
    # ── METAL ────────────────────────────────────────────────────────────────
    "metallica": "metal", "iron maiden": "metal",
    "black sabbath": "metal", "judas priest": "metal",
    "megadeth": "metal", "slayer": "metal",
    "testament": "metal", "exodus": "metal",
    "sepultura": "metal", "sodom": "metal",
    "kreator": "metal", "destruction": "metal",
    "anthrax": "metal", "pantera": "metal",
    "dimebag darrell": "metal", "philip h. anselmo": "metal",
    "alice in chains": "metal", "soundgarden": "metal",
    "nirvana": "metal", "mudhoney": "metal",
    "pearl jam": "metal", "stone temple pilots": "metal",
    "system of a down": "metal", "deftones": "metal",
    "tool": "metal", "korn": "metal",
    "limp bizkit": "metal", "linkin park": "metal",
    "slipknot": "metal", "mudvayne": "metal",
    "disturbed": "metal", "five finger death punch": "metal",
    "avenged sevenfold": "metal", "bullet for my valentine": "metal",
    "trivium": "metal", "lamb of god": "metal",
    "mastodon": "metal", "converge": "metal",
    "high on fire": "metal", "sleep": "metal",
    "electric wizard": "metal", "windhand": "metal",
    "cannibal corpse": "metal", "death": "metal",
    "morbid angel": "metal", "obituary": "metal",
    "possessed": "metal", "mayhem": "metal",
    "darkthrone": "metal", "emperor": "metal",
    "burzum": "metal", "immortal": "metal",
    "carpathian forest": "metal", "gorgoroth": "metal",
    "of mice & men": "metal", "killswitch engage": "metal",
    "as i lay dying": "metal", "all that remains": "metal",
    "parkway drive": "metal", "bring me the horizon": "metal",
    "while she sleeps": "metal", "architects": "metal",
    "wolfmother": "metal", "the sword": "metal",
    "failure": "metal", "mxpx": "metal",
    "d.o.a.": "metal", "voodoo glow skulls": "metal",
    "tankard": "metal", "black flag": "metal",
    "dead kennedys": "metal", "circle jerks": "metal",
    "turnstile": "metal", "ministry": "metal",
    # ── JAZZ ─────────────────────────────────────────────────────────────────
    "miles davis": "jazz", "john coltrane": "jazz",
    "bill evans": "jazz", "dave brubeck quartet": "jazz",
    "dave brubeck": "jazz", "thelonious monk": "jazz",
    "charles mingus": "jazz", "charlie parker": "jazz",
    "dizzy gillespie": "jazz", "louis armstrong": "jazz",
    "duke ellington": "jazz", "count basie": "jazz",
    "art blakey": "jazz", "clifford brown": "jazz",
    "chet baker": "jazz", "stan getz": "jazz",
    "sonny rollins": "jazz", "dexter gordon": "jazz",
    "lee morgan": "jazz", "blue mitchell": "jazz",
    "kenny dorham": "jazz", "freddie hubbard": "jazz",
    "herbie hancock": "jazz", "chick corea": "jazz",
    "keith jarrett": "jazz", "oscar peterson": "jazz",
    "brad mehldau": "jazz", "brad mehldau trio": "jazz",
    "john scofield": "jazz", "pat metheny": "jazz",
    "joshua redman": "jazz", "kurt rosenwinkel": "jazz",
    "mehldau": "jazz", "ahmad jamal": "jazz",
    "lars danielsson": "jazz", "christian scott": "jazz",
    "robert glasper": "jazz", "esperanza spalding": "jazz",
    "kamasi washington": "jazz", "makaya mccraven": "jazz",
    "sons of kemet": "jazz", "yussef kamaal": "jazz",
    "quincy jones": "jazz", "bill laurance": "jazz",
    "isfar sarabski": "jazz", "biréli lagrène": "jazz",
    "bireli lagrene": "jazz",
    # ── CITY-POP / J-POP / K-POP ─────────────────────────────────────────────
    "miki matsubara": "city_pop", "mariya takeuchi": "city_pop",
    "tatsuro yamashita": "city_pop", "anri": "city_pop",
    "taeko onuki": "city_pop", "minako yoshida": "city_pop",
    "yumi matsutoya": "city_pop", "hiromi iwasaki": "city_pop",
    "junko ohashi": "city_pop", "meiko nakahara": "city_pop",
    "momoko kikuchi": "city_pop", "sugiyama kiyotaka": "city_pop",
    "epo": "city_pop", "yasuko agawa": "city_pop",
    "omega tribe": "city_pop", "casiiopea": "city_pop",
    "tube": "city_pop", "spyair": "city_pop",
    "yoasobi": "kpop",  # Japanese but bridges K-pop aesthetics
    "ado": "city_pop", "yuuri": "city_pop",
    "kenshi yonezu": "city_pop",
    "newjeans": "kpop", "bts": "kpop",
    "red velvet": "kpop", "aespa": "kpop",
    "stayc": "kpop", "fromis_9": "kpop",
    "fifty fifty": "kpop", "kep1er": "kpop",
    "ive": "kpop", "lesserafim": "kpop", "le sserafim": "kpop",
    "twice": "kpop", "blackpink": "kpop",
    "exo": "kpop", "shinee": "kpop",
    "super junior": "kpop", "bigbang": "kpop",
    "2ne1": "kpop", "wonder girls": "kpop",
    "snsd": "kpop", "girls generation": "kpop",
    "girls' generation": "kpop",
    "wjsn": "kpop", "loona": "kpop",
    "da tweekaz": "kpop",  # bridges electronic/kpop
    "spice": "kpop", "astrid s": "kpop",
    # ── LATIN / AFROBEATS ────────────────────────────────────────────────────
    "bad bunny": "latin", "j balvin": "latin",
    "maluma": "latin", "ozuna": "latin",
    "daddy yankee": "latin", "reggaeton": "latin",
    "bizarrap": "latin", "quevedo": "latin",
    "rosalia": "latin", "rosalía": "latin",
    "karol g": "latin", "becky g": "latin",
    "nicky jam": "latin", "sech": "latin",
    "jhay cortez": "latin", "rauw alejandro": "latin",
    "myke towers": "latin", "mora": "latin",
    "feid": "latin", "anuel aa": "latin",
    "arcangel": "latin", "don omar": "latin",
    "romeo santos": "latin", "aventura": "latin",
    "marc anthony": "latin", "shakira": "latin",
    "juanes": "latin", "carlos vives": "latin",
    "wizkid": "afrobeats", "burna boy": "afrobeats",
    "davido": "afrobeats", "tiwa savage": "afrobeats",
    "tekno": "afrobeats", "fireboy dml": "afrobeats",
    "tyla": "afrobeats", "ckay": "afrobeats",
    "omah lay": "afrobeats", "rema": "afrobeats",
    "oxlade": "afrobeats", "ayra starr": "afrobeats",
    "victony": "afrobeats", "kizz daniel": "afrobeats",
    "asake": "afrobeats", "olamide": "afrobeats",
    "zlatan": "afrobeats", "naira marley": "afrobeats",
    # ── DIFFICULT / CROSS-GENRE ──────────────────────────────────────────────
    "radiohead": "difficult", "thom yorke": "difficult",
    "the smiths": "difficult", "morrissey": "difficult",
    "bjork": "difficult", "björk": "difficult",
    "scott walker": "difficult", "nick cave": "difficult",
    "the bad seeds": "difficult",
    "pj harvey": "difficult",
    "evanescence": "difficult",
    "the 1975": "difficult",
    "syml": "difficult", "sophie hutchings": "difficult",
}


def infer_scene(title: str, artist: str) -> Optional[str]:
    """Infer scene tag from artist (primary) then title+artist keywords (fallback).

    Priority order:
    1. ``_ARTIST_SCENE`` curated lookup (normalised artist name → scene)
    2. Keyword scan of the combined title + artist string

    Returns None when no signal is found (the rec is treated as coherent by default).
    """
    # 1. Curated artist lookup (covers hundreds of well-known artists)
    a_norm = unicodedata.normalize("NFKD", str(artist)).encode("ascii", "ignore").decode().casefold()
    # Try full artist name first, then first comma-token (handles "Drake, feat. ...")
    for a_key in (a_norm, a_norm.split(",")[0].strip(), a_norm.split("&")[0].strip()):
        a_key = a_key.strip()
        if a_key in _ARTIST_SCENE:
            return _ARTIST_SCENE[a_key]
    # 2. Keyword scan
    combined = f"{title} {artist}".lower()
    for scene, keywords in _SCENE_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return scene
    return None


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.casefold()
    for sep in (" feat. ", " feat ", " ft. ", " ft ", " featuring "):
        i = s.find(sep)
        if i > 0:
            s = s[:i]
    return " ".join(re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", s).split())


def _find_in_index(
    titles: List[str], artists: List[str],
    query_title: str, query_artist: str
) -> Optional[int]:
    """Find a song's index row.  Returns None if not found."""
    qt = _norm(query_title)
    qa = _norm(query_artist).split(",")[0].split(" & ")[0].strip()
    nt = [_norm(t) for t in titles]
    na = [_norm(a).split(",")[0].split(" & ")[0].strip() for a in artists]
    # Exact match first
    for i, (t, a) in enumerate(zip(nt, na)):
        if t == qt and a == qa:
            return i
    # Partial artist match
    for i, (t, a) in enumerate(zip(nt, na)):
        if t == qt and (not qa or qa in na[i]):
            return i
    # Substring title match
    for i, t in enumerate(nt):
        if qt in t and (not qa or qa in na[i]):
            return i
    return None


def run_eval(
    recommender,
    n: int = 5,
    seeds: Optional[List[Tuple[str, str, str]]] = None,
    method_name: str = "baseline",
    alpha: Optional[float] = None,
    diversity: float = 0.15,
    max_per_artist: int = 1,
) -> EvalReport:
    """Run the evaluation suite against a WebRecommender / DeepVibeRecommender-like object.

    The ``recommender`` must implement:
      - ``.titles``, ``.artists``: array-like of strings
      - ``.recommend(row, n, alpha, diversity, max_per_artist) -> dict`` (WebRecommender API)
        OR ``.find_row(title, artist) -> Optional[int]`` + ``.recommend(...)``

    Returns an EvalReport with per-seed results and aggregate statistics.
    """
    seeds = seeds or EVAL_SEEDS
    titles = list(recommender.titles)
    artists = list(recommender.artists)

    results: List[EvalResult] = []
    for (stitle, sartist, sscene) in seeds:
        row = recommender.find_row(stitle, sartist) if hasattr(recommender, "find_row") else \
              _find_in_index(titles, artists, stitle, sartist)
        if row is None:
            results.append(EvalResult(
                seed_title=stitle, seed_artist=sartist, seed_scene=sscene,
                found_in_index=False
            ))
            continue

        kwargs: Dict[str, Any] = {"n": n, "diversity": diversity, "max_per_artist": max_per_artist}
        if alpha is not None:
            kwargs["alpha"] = alpha
        raw = recommender.recommend(int(row), **kwargs)
        recs = raw.get("results", []) if isinstance(raw, dict) else []

        junk_flags = []
        mashup_flags = []
        same_artist_flags = []
        scene_tags = []
        allowed_scenes = _SCENE_RELATIVES.get(sscene, {sscene})
        scene_coherent_flags = []

        for rec in recs:
            rt = rec.get("title", "")
            ra = rec.get("artist", "")
            junk_flags.append(is_junk(rt, ra))
            mashup_flags.append(is_seed_title_in_result(stitle, rt))
            same_artist_flags.append(
                _norm(sartist).split(",")[0].split(" & ")[0]
                in _norm(ra).split(",")[0].split(" & ")[0]
            )
            # Scene coherence: can we tell from title/artist what scene this is?
            # We keep it simple: if the rec's artist or title contains scene-
            # associated keywords, tag it; otherwise leave it unknown (None).
            stag = infer_scene(rt, ra)
            scene_tags.append(stag)
            # A rec is "coherent" if its inferred scene matches the seed's allowed scenes
            # OR if we can't determine its scene (benefit of the doubt = count as coherent
            # only if at least one other signal is present).
            # To avoid false positives, we only count it coherent when we CAN infer the scene.
            if stag is None:
                scene_coherent_flags.append(True)  # unknown → don't penalise
            else:
                scene_coherent_flags.append(stag in allowed_scenes)

        results.append(EvalResult(
            seed_title=stitle, seed_artist=sartist, seed_scene=sscene,
            found_in_index=True, recs=recs,
            junk_flags=junk_flags,
            seed_mashup_flags=mashup_flags,
            same_artist_flags=same_artist_flags,
            scene_coherent_flags=scene_coherent_flags,
            scene_tags=scene_tags,
        ))

    return _aggregate(results, method_name)


def _aggregate(results: List[EvalResult], method_name: str) -> EvalReport:
    found = [r for r in results if r.found_in_index]
    n_seeds = len(results)
    n_found = len(found)

    if n_found == 0:
        return EvalReport(n_seeds=n_seeds, n_found=0,
                          primary_score=0.0, top1_coherent=0.0,
                          junk_rate=0.0, mashup_rate=0.0, same_artist_rate=0.0,
                          results=results, method_name=method_name)

    def safe_mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def rec_frac(r: EvalResult, flags: List[bool]) -> float:
        if not r.recs:
            return 0.0
        return safe_mean([float(f) for f in flags[:len(r.recs)]])

    coherence_per_seed = [safe_mean([float(f) for f in r.scene_coherent_flags[:5]])
                          for r in found]
    top1_coherent = safe_mean([float(r.scene_coherent_flags[0])
                                if r.scene_coherent_flags else 1.0
                                for r in found])
    junk_per_seed = [rec_frac(r, r.junk_flags) for r in found]
    mashup_per_seed = [rec_frac(r, r.seed_mashup_flags) for r in found]
    same_artist_per_seed = [rec_frac(r, r.same_artist_flags) for r in found]

    # Per-scene breakdown
    per_scene: Dict[str, Dict[str, float]] = {}
    from collections import defaultdict
    scene_coherences: Dict[str, List[float]] = defaultdict(list)
    scene_junk: Dict[str, List[float]] = defaultdict(list)
    for r, coh, jnk in zip(found, coherence_per_seed, junk_per_seed):
        scene_coherences[r.seed_scene].append(coh)
        scene_junk[r.seed_scene].append(jnk)
    for scene in scene_coherences:
        per_scene[scene] = {
            "coherence": safe_mean(scene_coherences[scene]),
            "junk_rate": safe_mean(scene_junk[scene]),
            "n_seeds": len(scene_coherences[scene]),
        }

    return EvalReport(
        n_seeds=n_seeds,
        n_found=n_found,
        primary_score=safe_mean(coherence_per_seed),
        top1_coherent=top1_coherent,
        junk_rate=safe_mean(junk_per_seed),
        mashup_rate=safe_mean(mashup_per_seed),
        same_artist_rate=safe_mean(same_artist_per_seed),
        per_scene=per_scene,
        results=results,
        method_name=method_name,
    )


def print_report(report: EvalReport) -> None:
    """Pretty-print an EvalReport to stdout."""
    print(f"\n{'='*70}")
    print(f"  Evaluation Report: {report.method_name}")
    print(f"{'='*70}")
    print(f"  Seeds found in index:  {report.n_found} / {report.n_seeds}")
    print(f"  PRIMARY SCORE (scene coherence@5): {report.primary_score:.3f}")
    print(f"  Top-1 coherent:        {report.top1_coherent:.3f}")
    print(f"  Junk rate (top-5):     {report.junk_rate:.3f}")
    print(f"  Mashup rate (top-5):   {report.mashup_rate:.3f}")
    print(f"  Same-artist leak:      {report.same_artist_rate:.3f}")
    print(f"\n  Per-scene coherence:")
    for scene, stats in sorted(report.per_scene.items()):
        bar = "█" * int(stats["coherence"] * 20)
        print(f"    {scene:<15} {stats['coherence']:.3f}  {bar}")
    print()


def save_report(report: EvalReport, path: Path) -> None:
    """Save an EvalReport as JSON (for frozen baseline storage)."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_report(path: Path) -> Dict[str, Any]:
    """Load a frozen baseline report from JSON."""
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_reports(
    baseline: Dict[str, Any], challenger: EvalReport
) -> Dict[str, float]:
    """Compare two reports; returns relative changes.

    Positive values = improvement (for coherence), negative = regression.
    """
    base_score = baseline.get("primary_score", 0.0)
    chal_score = challenger.primary_score
    rel_change = (chal_score - base_score) / (base_score + 1e-9)

    per_scene_delta: Dict[str, float] = {}
    for scene, stats in challenger.per_scene.items():
        base_scene_coh = (baseline.get("per_scene") or {}).get(scene, {}).get("coherence", 0.0)
        chal_scene_coh = stats["coherence"]
        per_scene_delta[scene] = (chal_scene_coh - base_scene_coh) / (base_scene_coh + 1e-9)

    return {
        "primary_relative_gain": rel_change,
        "baseline_primary": base_score,
        "challenger_primary": chal_score,
        "per_scene_relative_delta": per_scene_delta,
        "junk_rate_delta": challenger.junk_rate - baseline.get("junk_rate", 0.0),
    }
