"""Build the fresh, source-independent collaborative FINAL benchmark.

The query catalogue below is fixed before collaborative model training.  Targets
come from ListenBrainz session-similarity, while the candidate generator is
trained on Music4All-Onion.  Prior v5 pairs are diagnostics-only development
data; no fresh FINAL target is exposed to model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlencode

import numpy as np
import requests

from .quality_filter import TitleQualityFilter
from .real_benchmark import PairResolver, credited_artists, normalize_text

MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/recording/"
LISTENBRAINZ_URL = "https://labs.api.listenbrainz.org/similar-recordings/json"
LISTENBRAINZ_ALGORITHM = (
    "session_based_days_180_session_300_contribution_5_threshold_15_limit_50_skip_30"
)
USER_AGENT = "soundalike-benchmark/6.0 (https://github.com/yassinsolim/soundalike)"

# One recording per artist.  The list is deliberately independent of the
# Music4All mapping and includes popular, deep-cut, and niche material.
_SEED_ROWS = """
hip-hop	popular	N.Y. State of Mind	Nas
hip-hop	deep_cut	Shook Ones, Pt. II	Mobb Deep
hip-hop	deep_cut	Electric Relaxation	A Tribe Called Quest
hip-hop	popular	99 Problems	JAY-Z
hip-hop	niche	Accordion	Madvillain
hip-hop	popular	See You Again	Tyler, The Creator
hip-hop	deep_cut	1539 N. Calvert	JPEGMAFIA
hip-hop	popular	No Role Modelz	J. Cole
hip-hop	deep_cut	Aquemini	Outkast
hip-hop	niche	Legend Has It	Run The Jewels
r&b-soul	popular	On & On	Erykah Badu
r&b-soul	deep_cut	Untitled (How Does It Feel)	D'Angelo
r&b-soul	popular	Ex-Factor	Lauryn Hill
r&b-soul	popular	Adorn	Miguel
r&b-soul	deep_cut	Dead Man Walking	Brent Faiyaz
r&b-soul	deep_cut	Exchange	Bryson Tiller
r&b-soul	popular	Best Part	Daniel Caesar
r&b-soul	niche	Why Don't You	Cleo Sol
r&b-soul	deep_cut	Shea Butter Baby	Ari Lennox
r&b-soul	popular	What's Going On	Marvin Gaye
indie-rock	popular	Do I Wanna Know?	Arctic Monkeys
indie-rock	deep_cut	Obstacle 1	Interpol
indie-rock	deep_cut	Float On	Modest Mouse
indie-rock	popular	A-Punk	Vampire Weekend
indie-rock	deep_cut	Banquet	Bloc Party
indie-rock	niche	Drunk Drivers/Killer Whales	Car Seat Headrest
indie-rock	deep_cut	New Slang	The Shins
indie-rock	deep_cut	Inside Out	Spoon
indie-rock	niche	Jesus, Etc.	Wilco
indie-rock	deep_cut	Red Eyes	The War on Drugs
shoegaze-dream-pop	deep_cut	Heaven or Las Vegas	Cocteau Twins
shoegaze-dream-pop	niche	For Love	Lush
shoegaze-dream-pop	niche	Doused	DIIV
shoegaze-dream-pop	niche	Leave	Whirr
shoegaze-dream-pop	popular	Space Song	Beach House
shoegaze-dream-pop	deep_cut	Fade Into You	Mazzy Star
shoegaze-dream-pop	niche	Strange	Galaxie 500
shoegaze-dream-pop	niche	Vertigo Flowers	Nothing
shoegaze-dream-pop	niche	Pearl	Chapterhouse
shoegaze-dream-pop	deep_cut	Archie, Marry Me	Alvvays
electronic	popular	Windowlicker	Aphex Twin
electronic	deep_cut	Dayvan Cowboy	Boards of Canada
electronic	niche	Archangel	Burial
electronic	deep_cut	Two Thousand and Seventeen	Four Tet
electronic	niche	Gantz Graf	Autechre
electronic	popular	D.A.N.C.E.	Justice
electronic	popular	Hey Boy Hey Girl	The Chemical Brothers
electronic	deep_cut	Midnight City	M83
electronic	deep_cut	All My Friends	LCD Soundsystem
electronic	deep_cut	Teardrop	Massive Attack
metal	popular	Raining Blood	Slayer
metal	popular	Holy Wars... The Punishment Due	Megadeth
metal	deep_cut	Painkiller	Judas Priest
metal	popular	Walk	Pantera
metal	deep_cut	Schism	Tool
metal	popular	Change (In the House of Flies)	Deftones
metal	popular	Duality	Slipknot
metal	niche	Stranded	Gojira
metal	niche	Bleed	Meshuggah
metal	niche	Ghost of Perdition	Opeth
jazz	deep_cut	Goodbye Pork Pie Hat	Charles Mingus
jazz	popular	Cantaloupe Island	Herbie Hancock
jazz	popular	My Funny Valentine	Chet Baker
jazz	deep_cut	Moanin'	Art Blakey & The Jazz Messengers
jazz	deep_cut	St. Thomas	Sonny Rollins
jazz	niche	West Coast Blues	Wes Montgomery
jazz	popular	The Girl From Ipanema	Stan Getz
jazz	popular	Sinnerman	Nina Simone
jazz	deep_cut	You Look Good to Me	Oscar Peterson Trio
jazz	deep_cut	In a Mellow Tone	Duke Ellington
city-pop-j-pop	deep_cut	Remember Summer Days	Anri
city-pop-j-pop	deep_cut	Stay With Me	Miki Matsubara
city-pop-j-pop	niche	Sparkle	Tatsuro Yamashita
city-pop-j-pop	niche	4:00 A.M.	Taeko Onuki
city-pop-j-pop	niche	Telephone Number	Junko Ohashi
city-pop-j-pop	niche	Bay City	Junko Yagami
city-pop-j-pop	deep_cut	Ride on Time	Tatsuro Yamashita
city-pop-j-pop	niche	Swallowtail Butterfly	Yen Town Band
city-pop-j-pop	popular	Automatic	Hikaru Utada
city-pop-j-pop	niche	For Lovers	Lamp
k-pop	popular	Spring Day	BTS
k-pop	popular	As If It's Your Last	BLACKPINK
k-pop	popular	FANCY	TWICE
k-pop	deep_cut	Bad Boy	Red Velvet
k-pop	deep_cut	View	SHINee
k-pop	popular	Love Shot	EXO
k-pop	popular	God's Menu	Stray Kids
k-pop	deep_cut	WANNABE	ITZY
k-pop	deep_cut	LION	(G)I-DLE
k-pop	niche	MOVE	TAEMIN
latin-reggaeton	popular	Gasolina	Daddy Yankee
latin-reggaeton	popular	Mi Gente	J Balvin
latin-reggaeton	popular	TQG	KAROL G
latin-reggaeton	deep_cut	Dile Que	Tainy
latin-reggaeton	popular	Todo de Ti	Rauw Alejandro
latin-reggaeton	popular	Danza Kuduro	Don Omar
latin-reggaeton	deep_cut	DESPECHÁ	ROSALÍA
latin-reggaeton	popular	Hips Don't Lie	Shakira
latin-reggaeton	deep_cut	A Dios le Pido	Juanes
latin-reggaeton	deep_cut	Como la Flor	Selena
afrobeats	popular	Last Last	Burna Boy
afrobeats	popular	Fall	Davido
afrobeats	deep_cut	Free Mind	Tems
afrobeats	popular	Calm Down	Rema
afrobeats	deep_cut	soso	Omah Lay
afrobeats	deep_cut	Peru	Fireboy DML
afrobeats	niche	Rush	Ayra Starr
afrobeats	niche	Organise	Asake
afrobeats	popular	love nwantiti (ah ah ah)	CKay
afrobeats	niche	Leg Over	Mr Eazi
punk-hardcore	popular	Blitzkrieg Bop	Ramones
punk-hardcore	popular	Anarchy in the U.K.	Sex Pistols
punk-hardcore	popular	Should I Stay or Should I Go	The Clash
punk-hardcore	niche	Banned in D.C.	Bad Brains
punk-hardcore	niche	Minor Threat	Minor Threat
punk-hardcore	deep_cut	Waiting Room	Fugazi
punk-hardcore	niche	HOLIDAY	Turnstile
punk-hardcore	niche	Rise Above	Black Flag
punk-hardcore	niche	Holiday in Cambodia	Dead Kennedys
punk-hardcore	deep_cut	Suburban Home	Descendents
folk-country	deep_cut	A Case of You	Joni Mitchell
folk-country	popular	Like a Rolling Stone	Bob Dylan
folk-country	niche	Pink Moon	Nick Drake
folk-country	deep_cut	Between the Bars	Elliott Smith
folk-country	deep_cut	Chicago	Sufjan Stevens
folk-country	niche	For Emma	Bon Iver
folk-country	niche	Pancho and Lefty	Townes Van Zandt
folk-country	popular	Folsom Prison Blues	Johnny Cash
folk-country	popular	Jolene	Dolly Parton
folk-country	niche	Feathered Indians	Tyler Childers
funk-disco	popular	Good Times	Chic
funk-disco	popular	I Feel Love	Donna Summer
funk-disco	popular	September	Earth, Wind & Fire
funk-disco	deep_cut	Flash Light	Parliament
funk-disco	niche	Maggot Brain	Funkadelic
funk-disco	deep_cut	Move On Up	Curtis Mayfield
funk-disco	deep_cut	Theme From Shaft	Isaac Hayes
funk-disco	popular	Get Down on It	Kool & The Gang
funk-disco	deep_cut	He's the Greatest Dancer	Sister Sledge
funk-disco	deep_cut	Forget Me Nots	Patrice Rushen
classic-psych-rock	popular	Whole Lotta Love	Led Zeppelin
classic-psych-rock	popular	Time	Pink Floyd
classic-psych-rock	popular	All Along the Watchtower	Jimi Hendrix
classic-psych-rock	popular	Riders on the Storm	The Doors
classic-psych-rock	deep_cut	White Room	Cream
classic-psych-rock	popular	Gimme Shelter	The Rolling Stones
classic-psych-rock	popular	Dreams	Fleetwood Mac
classic-psych-rock	deep_cut	Fortunate Son	Creedence Clearwater Revival
classic-psych-rock	deep_cut	White Rabbit	Jefferson Airplane
classic-psych-rock	niche	21st Century Schizoid Man	King Crimson
emo-pop-punk	popular	Welcome to the Black Parade	My Chemical Romance
emo-pop-punk	popular	Sugar, We're Goin Down	Fall Out Boy
emo-pop-punk	popular	What's My Age Again?	blink-182
emo-pop-punk	deep_cut	MakeDamnSure	Taking Back Sunday
emo-pop-punk	niche	Never Meant	American Football
emo-pop-punk	niche	Seven	Sunny Day Real Estate
emo-pop-punk	deep_cut	King for a Day	Pierce the Veil
emo-pop-punk	deep_cut	If You Can't Hang	Sleeping With Sirens
emo-pop-punk	deep_cut	The Anthem	Good Charlotte
emo-pop-punk	deep_cut	Sweetness	Jimmy Eat World
reggae-dub-ska	popular	Three Little Birds	Bob Marley & The Wailers
reggae-dub-ska	deep_cut	Pressure Drop	Toots & The Maytals
reggae-dub-ska	deep_cut	A Message to You Rudy	The Specials
reggae-dub-ska	deep_cut	Our House	Madness
reggae-dub-ska	niche	Disco Devil	Lee "Scratch" Perry
reggae-dub-ska	deep_cut	Legalize It	Peter Tosh
reggae-dub-ska	niche	Night Nurse	Gregory Isaacs
reggae-dub-ska	popular	Santeria	Sublime
reggae-dub-ska	niche	King Tubby Meets Rockers Uptown	Augustus Pablo
reggae-dub-ska	niche	The Harder They Come	Jimmy Cliff
ambient-experimental	deep_cut	An Ending (Ascent)	Brian Eno
ambient-experimental	niche	Virginal II	Tim Hecker
ambient-experimental	niche	Heavy Water/I'd Rather Be Sleeping	Grouper
ambient-experimental	niche	Replica	Oneohtrix Point Never
ambient-experimental	deep_cut	Desafío	Arca
ambient-experimental	deep_cut	Cellophane	FKA twigs
ambient-experimental	popular	Hyperballad	Björk
ambient-experimental	deep_cut	Svefn-g-englar	Sigur Rós
ambient-experimental	niche	Screen Shot	Swans
ambient-experimental	niche	R Plus Seven	Oneohtrix Point Never
hyperpop-digicore	popular	money machine	100 gecs
hyperpop-digicore	deep_cut	Immaterial	SOPHIE
hyperpop-digicore	niche	Beautiful	A. G. Cook
hyperpop-digicore	niche	Flamboyant	Dorian Electra
hyperpop-digicore	niche	Spoiled little brat	underscores
hyperpop-digicore	niche	venus fly trap	brakence
hyperpop-digicore	niche	sad4whattt	ericdoa
hyperpop-digicore	niche	Thos Moser	food house
hyperpop-digicore	niche	Rich Bitch Juice	Alice Longyu Gao
hyperpop-digicore	deep_cut	Claws	Charli xcx
pop	popular	Style	Taylor Swift
pop	popular	Don't Start Now	Dua Lipa
pop	popular	bad guy	Billie Eilish
pop	popular	24K Magic	Bruno Mars
pop	popular	Feather	Sabrina Carpenter
pop	niche	Red Wine Supernova	Chappell Roan
pop	popular	Flowers	Miley Cyrus
pop	popular	As It Was	Harry Styles
pop	popular	Cool for the Summer	Demi Lovato
pop	deep_cut	Into You	Ariana Grande
r&b-soul	popular	Smooth Operator	Sade
r&b-soul	deep_cut	Ascension (Don't Ever Wonder)	Maxwell
r&b-soul	deep_cut	A Long Walk	Jill Scott
r&b-soul	niche	Just Friends (Sunny)	Musiq Soulchild
r&b-soul	deep_cut	Focus	H.E.R.
r&b-soul	deep_cut	Girls Need Love	Summer Walker
r&b-soul	niche	I Want You Around	Snoh Aalegra
r&b-soul	niche	Roll Some Mo	Lucky Daye
indie-rock	deep_cut	Bloodbuzz Ohio	The National
indie-rock	deep_cut	Wolf Like Me	TV on the Radio
indie-rock	niche	Carry the Zero	Built to Spill
indie-rock	niche	Cut Your Hair	Pavement
indie-rock	niche	Feel the Pain	Dinosaur Jr.
indie-rock	niche	Autumn Sweater	Yo La Tengo
indie-rock	niche	Helicopter	Deerhunter
indie-rock	popular	Intro	The xx
shoegaze-dream-pop	niche	Duel	Swervedriver
shoegaze-dream-pop	niche	Black Metallic	Catherine Wheel
shoegaze-dream-pop	niche	Kick the Tragedy	Drop Nineteens
shoegaze-dream-pop	niche	Sight of You	Pale Saints
shoegaze-dream-pop	niche	Never Coming Back	A Place to Bury Strangers
shoegaze-dream-pop	niche	Sure	Hatchie
shoegaze-dream-pop	niche	Chinatown	Wild Nothing
shoegaze-dream-pop	niche	Strange Things Will Happen	The Radio Dept.
city-pop-j-pop	niche	Dress Down	Kaoru Akimoto
city-pop-j-pop	niche	Midnight Pretenders	Tomoko Aran
city-pop-j-pop	niche	Fantasy	Meiko Nakahara
city-pop-j-pop	niche	Summer Suspicion	S. Kiyotaka & Omega Tribe
city-pop-j-pop	niche	Brasilian Skies	Masayoshi Takanaka
city-pop-j-pop	deep_cut	Rouge no Dengon	Yumi Arai
city-pop-j-pop	niche	Adventure	Momoko Kikuchi
city-pop-j-pop	deep_cut	Slow Motion	Akina Nakamori
k-pop	popular	HIP	MAMAMOO
k-pop	popular	Gee	Girls' Generation
k-pop	deep_cut	I Am the Best	2NE1
k-pop	deep_cut	Why So Lonely	Wonder Girls
k-pop	niche	4 Walls	f(x)
k-pop	niche	Butterfly	LOONA
k-pop	niche	Scream	Dreamcatcher
k-pop	deep_cut	Answer	ATEEZ
k-pop	popular	VERY NICE	SEVENTEEN
k-pop	popular	BANG BANG BANG	BIGBANG
jazz	deep_cut	Lullaby of Birdland	Sarah Vaughan
jazz	niche	Poinciana	Ahmad Jamal
jazz	niche	Passion Dance	McCoy Tyner
jazz	deep_cut	Footprints	Wayne Shorter
jazz	deep_cut	Mercy, Mercy, Mercy	Cannonball Adderley
jazz	niche	The Creator Has a Master Plan	Pharoah Sanders
jazz	niche	Journey in Satchidananda	Alice Coltrane
jazz	niche	Street Fighter Mas	Kamasi Washington
jazz	deep_cut	Spain	Chick Corea
jazz	deep_cut	Last Train Home	Pat Metheny Group
electronic	deep_cut	Odessa	Caribou
electronic	niche	Nespole	Floating Points
electronic	deep_cut	Emerald Rush	Jon Hopkins
electronic	niche	Space Is Only Noise If You Can See	Nicolas Jaar
electronic	deep_cut	Bad Kingdom	Moderat
electronic	deep_cut	Kerala	Bonobo
electronic	niche	Glue	Bicep
electronic	deep_cut	Halcyon and On and On	Orbital
electronic	popular	Born Slippy (Nuxx)	Underworld
electronic	niche	La femme d'argent	Air
metal	deep_cut	Redneck	Lamb of God
metal	deep_cut	In Waves	Trivium
metal	deep_cut	My Curse	Killswitch Engage
metal	niche	Doomsday	Architects
metal	niche	Concubine	Converge
metal	niche	43% Burnt	The Dillinger Escape Plan
metal	niche	Dragonaut	Sleep
metal	niche	Funeralopolis	Electric Wizard
metal	niche	Autre temps	Alcest
metal	niche	Dream House	Deafheaven
latin-reggaeton	popular	Rakata	Wisin & Yandel
latin-reggaeton	popular	El Perdón	Nicky Jam
latin-reggaeton	popular	Hawái	Maluma
latin-reggaeton	deep_cut	Normal	Feid
latin-reggaeton	popular	LALA	Myke Towers
latin-reggaeton	deep_cut	Me gustas tú	Manu Chao
latin-reggaeton	deep_cut	De música ligera	Soda Stereo
latin-reggaeton	niche	Eres	Café Tacvba
latin-reggaeton	deep_cut	Hasta la Raíz	Natalia Lafourcade
latin-reggaeton	niche	Matador	Los Fabulosos Cadillacs
afrobeats	deep_cut	Personally	P-Square
afrobeats	deep_cut	Johnny	Yemi Alade
afrobeats	deep_cut	All Over	Tiwa Savage
afrobeats	niche	Buga (Lo Lo Lo)	Kizz Daniel
afrobeats	niche	Baby	Joeboy
afrobeats	niche	High	Adekunle Gold
afrobeats	niche	SAD GIRLZ LUV MONEY	Amaarae
afrobeats	niche	Kwaku the Traveller	Black Sherif
punk-hardcore	niche	New Direction	Gorilla Biscuits
punk-hardcore	niche	Break Down the Walls	Youth of Today
punk-hardcore	niche	Sound System	Operation Ivy
punk-hardcore	deep_cut	Time Bomb	Rancid
punk-hardcore	deep_cut	Linoleum	NOFX
punk-hardcore	deep_cut	American Jesus	Bad Religion
punk-hardcore	niche	Last Caress	Misfits
punk-hardcore	niche	Ever Fallen in Love (With Someone You Shouldn't've)	Buzzcocks
folk-country	niche	Naked as We Came	Iron & Wine
folk-country	niche	Heartbeats	José González
folk-country	niche	Big Black Car	Gregory Alan Isakov
folk-country	niche	anything	Adrianne Lenker
folk-country	niche	Vampire Empire	Big Thief
folk-country	niche	Fire	Waxahatchee
folk-country	niche	Look at Miss Ohio	Gillian Welch
folk-country	deep_cut	Cover Me Up	Jason Isbell
folk-country	niche	Sleeping on the Blacktop	Colter Wall
folk-country	deep_cut	Slow Burn	Kacey Musgraves
funk-disco	deep_cut	Outstanding	The Gap Band
funk-disco	deep_cut	Candy	Cameo
funk-disco	niche	More Bounce to the Ounce	Zapp
funk-disco	deep_cut	Give It to Me Baby	Rick James
funk-disco	niche	Love Come Down	Evelyn "Champagne" King
funk-disco	niche	A Night to Remember	Shalamar
funk-disco	niche	And the Beat Goes On	The Whispers
funk-disco	deep_cut	Boogie Nights	Heatwave
classic-psych-rock	deep_cut	Roundabout	Yes
classic-psych-rock	deep_cut	Firth of Fifth	Genesis
classic-psych-rock	popular	Tom Sawyer	Rush
classic-psych-rock	deep_cut	Highway Star	Deep Purple
classic-psych-rock	popular	(Don't Fear) The Reaper	Blue Öyster Cult
classic-psych-rock	popular	Free Bird	Lynyrd Skynyrd
classic-psych-rock	deep_cut	Ramblin' Man	The Allman Brothers Band
classic-psych-rock	deep_cut	Deacon Blues	Steely Dan
emo-pop-punk	deep_cut	The Taste of Ink	The Used
emo-pop-punk	niche	Hands Down	Dashboard Confessional
emo-pop-punk	niche	Understanding in a Car Crash	Thursday
emo-pop-punk	niche	The Artist in the Ambulance	Thrice
emo-pop-punk	niche	Seven Years	Saosin
emo-pop-punk	niche	Smile in Your Sleep	Silverstein
emo-pop-punk	niche	Writing on the Walls	Underoath
emo-pop-punk	niche	Bite to Break Skin	Senses Fail
reggae-dub-ska	niche	Marcus Garvey	Burning Spear
reggae-dub-ska	niche	Two Sevens Clash	Culture
reggae-dub-ska	niche	Fisherman	The Congos
reggae-dub-ska	niche	Guess Who's Coming to Dinner	Black Uhuru
reggae-dub-ska	deep_cut	Your House	Steel Pulse
reggae-dub-ska	popular	Red Red Wine	UB40
reggae-dub-ska	deep_cut	Israelites	Desmond Dekker
reggae-dub-ska	niche	Guns of Navarone	The Skatalites
ambient-experimental	niche	dlp 1.1	The Disintegration Loops
ambient-experimental	niche	Requiem for Dying Mothers, Pt. 2	Stars of the Lid
ambient-experimental	niche	Says	Nils Frahm
ambient-experimental	deep_cut	On the Nature of Daylight	Max Richter
ambient-experimental	niche	Nepenthe	Julianna Barwick
ambient-experimental	niche	An Intention	Kaitlyn Aurelia Smith
ambient-experimental	niche	Gospel for a New Century	Yves Tumor
ambient-experimental	deep_cut	Hope There's Someone	ANOHNI
pop	deep_cut	Run Away With Me	Carly Rae Jepsen
pop	popular	Rush	Troye Sivan
pop	deep_cut	Dancing on My Own	Robyn
pop	deep_cut	XS	Rina Sawayama
pop	deep_cut	Spotlight	Jessie Ware
pop	popular	Habits (Stay High)	Tove Lo
pop	popular	Lights	Ellie Goulding
pop	popular	Unwritten	Natasha Bedingfield
pop	popular	Say It Right	Nelly Furtado
pop	popular	Can't Get You Out of My Head	Kylie Minogue
hip-hop	popular	In da Club	50 Cent
hip-hop	popular	A Milli	Lil Wayne
hip-hop	popular	Super Bass	Nicki Minaj
hip-hop	popular	Bodak Yellow	Cardi B
hip-hop	popular	Savage	Megan Thee Stallion
hip-hop	deep_cut	Peso	A$AP Rocky
hip-hop	deep_cut	Day 'n' Nite	Kid Cudi
hip-hop	deep_cut	Collard Greens	ScHoolboy Q
r&b-soul	popular	U Got It Bad	Usher
r&b-soul	popular	If I Ain't Got You	Alicia Keys
r&b-soul	popular	Family Affair	Mary J. Blige
r&b-soul	popular	No Scrubs	TLC
r&b-soul	deep_cut	Try Again	Aaliyah
r&b-soul	popular	So Sick	Ne-Yo
r&b-soul	deep_cut	Evergreen (You Didn't Deserve Me at All)	Omar Apollo
r&b-soul	popular	Bad Habit	Steve Lacy
indie-rock	popular	1901	Phoenix
indie-rock	popular	Ain't No Rest for the Wicked	Cage the Elephant
indie-rock	deep_cut	My Number	Foals
indie-rock	deep_cut	Breezeblocks	alt-J
indie-rock	popular	Gooey	Glass Animals
indie-rock	popular	Feel It Still	Portugal. The Man
indie-rock	deep_cut	Young Folks	Peter Bjorn and John
indie-rock	deep_cut	Naive	The Kooks
electronic	popular	Scary Monsters and Nice Sprites	Skrillex
electronic	popular	Feel So Close	Calvin Harris
electronic	deep_cut	Latch	Disclosure
electronic	deep_cut	Never Be Like You	Flume
electronic	deep_cut	A Moment Apart	ODESZA
electronic	niche	Language	Porter Robinson
electronic	niche	All My Friends	Madeon
electronic	popular	Clarity	Zedd
metal	popular	Freak on a Leash	Korn
metal	popular	Break Stuff	Limp Bizkit
metal	deep_cut	Can You Feel My Heart	Bring Me the Horizon
metal	niche	Holy Roller	Spiritbox
metal	niche	The Summoning	Sleep Token
metal	niche	Carrion	Parkway Drive
metal	popular	Du hast	Rammstein
metal	deep_cut	Tears Don't Fall	Bullet for My Valentine
pop	popular	Toxic	Britney Spears
pop	popular	Genie in a Bottle	Christina Aguilera
pop	popular	SexyBack	Justin Timberlake
pop	popular	Umbrella	Rihanna
pop	popular	Rolling in the Deep	Adele
pop	popular	Chandelier	Sia
pop	popular	Just Like a Pill	P!nk
pop	popular	Hollaback Girl	Gwen Stefani
pop	popular	Without Me	Halsey
pop	popular	Beautiful Soul	Jesse McCartney
""".strip()


def seed_catalogue() -> List[Dict[str, str]]:
    """Return the immutable query catalogue in declared order."""
    seeds = []
    for line in _SEED_ROWS.splitlines():
        scene, tier, title, artist = line.split("\t")
        seeds.append({"scene": scene, "catalog_tier": tier, "title": title, "artist": artist})
    return seeds


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_url(mbid: str) -> str:
    return LISTENBRAINZ_URL + "?" + urlencode(
        {"recording_mbids": mbid, "algorithm": LISTENBRAINZ_ALGORITHM}
    )


def _artist_set(pair: Mapping[str, Any]) -> set[str]:
    return {
        normalize_text(artist)
        for side in ("query", "target")
        for artist in credited_artists(pair[side]["artist"])
    }


def _load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"musicbrainz": {}, "listenbrainz": {}}


def _save_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _recording_mbids(
    session: requests.Session,
    title: str,
    artist: str,
    cache: Dict[str, Any],
) -> List[str]:
    key = f"{normalize_text(artist)}\t{normalize_text(title)}"
    cached = cache["musicbrainz"].get(key)
    if isinstance(cached, list):
        return [str(value) for value in cached]
    response = session.get(
        MUSICBRAINZ_URL,
        params={
            "query": f'recording:"{title}" AND artist:"{artist}"',
            "fmt": "json",
            "limit": 25,
        },
        timeout=45,
    )
    response.raise_for_status()
    wanted_title = normalize_text(title)
    wanted_artist = normalize_text(artist)
    mbids = []
    for record in response.json().get("recordings", []):
        names = " ".join(
            credit.get("name", "") for credit in record.get("artist-credit", [])
        )
        if normalize_text(record.get("title", "")) != wanted_title:
            continue
        if wanted_artist not in normalize_text(names):
            continue
        if record.get("id") and record["id"] not in mbids:
            mbids.append(record["id"])
    cache["musicbrainz"][key] = mbids
    time.sleep(1.05)
    return mbids


def _similar_recordings(
    session: requests.Session,
    mbid: str,
    cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if mbid not in cache["listenbrainz"]:
        response = session.get(
            LISTENBRAINZ_URL,
            params={"recording_mbids": mbid, "algorithm": LISTENBRAINZ_ALGORITHM},
            timeout=45,
        )
        response.raise_for_status()
        cache["listenbrainz"][mbid] = response.json()
        time.sleep(0.2)
    unique: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for candidate in cache["listenbrainz"][mbid]:
        key = (
            normalize_text(candidate.get("recording_name", "")),
            normalize_text(candidate.get("artist_credit_name", "")),
        )
        if key not in unique or candidate.get("score", 0) > unique[key].get("score", 0):
            unique[key] = candidate
    return sorted(unique.values(), key=lambda item: -int(item.get("score", 0)))


def _round_robin(
    records: Iterable[Dict[str, Any]],
    per_scene: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        if counts[record["scene"]] >= per_scene:
            continue
        selected.append(record)
        counts[record["scene"]] += 1
    return selected


def build_benchmark(
    index_path: Path,
    prior_path: Path,
    cache_path: Path,
    output_path: Path,
    minimum_final: int = 80,
    per_scene: int = 6,
) -> Dict[str, Any]:
    """Resolve and freeze fresh human-listening pairs without model access."""
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    development = []
    blocked_artists: set[str] = set()
    for pair in prior["pairs"]:
        copied = json.loads(json.dumps(pair))
        copied["id"] = f"DEV-OPENED-{pair['id']}"
        copied["split"] = "development"
        copied["previously_opened_diagnostic"] = True
        development.append(copied)
        blocked_artists |= _artist_set(copied)

    with np.load(index_path, allow_pickle=False) as index:
        titles = np.asarray(index["titles"])
        artists = np.asarray(index["artists"])
    resolver = PairResolver(titles, artists)
    quality = TitleQualityFilter()
    quality_mask = quality.keep_mask(titles, artists)
    cache = _load_cache(cache_path)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    candidates: List[Dict[str, Any]] = []
    used_artists = set(blocked_artists)
    used_tracks: set[Tuple[str, str]] = set()

    for seed in seed_catalogue():
        seed_artists = {
            normalize_text(value) for value in credited_artists(seed["artist"])
        }
        if seed_artists & used_artists:
            continue
        query_row = resolver.query_row(seed)
        if query_row is None or not quality_mask[query_row]:
            continue
        if re.search(
            r"(?i)instrumental|karaoke|tribute|slowed|reverb|sped[ -]?up|"
            r"nightcore|radio edit|cover version|mashup",
            str(titles[query_row]),
        ):
            continue
        mbids = _recording_mbids(
            session, seed["title"], seed["artist"], cache
        )
        _save_cache(cache_path, cache)
        if not mbids:
            continue
        mbid = None
        similar = []
        for candidate_mbid in mbids:
            candidate_similar = _similar_recordings(
                session, candidate_mbid, cache
            )
            _save_cache(cache_path, cache)
            if candidate_similar:
                mbid = candidate_mbid
                similar = candidate_similar
                break
        if mbid is None:
            continue
        chosen = None
        for item in similar:
            target = {
                "title": item.get("recording_name", ""),
                "artist": item.get("artist_credit_name", ""),
            }
            target_artists = {
                normalize_text(value) for value in credited_artists(target["artist"])
            }
            if not target["title"] or not target_artists:
                continue
            if target_artists & (used_artists | seed_artists):
                continue
            target_rows = resolver.target_rows(target)
            if not target_rows:
                continue
            target_row = next(
                (row for row in target_rows if quality_mask[row]), None
            )
            if target_row is None:
                continue
            if quality.seed_title_in_result(seed["title"], str(titles[target_row])):
                continue
            target_key = (
                normalize_text(str(titles[target_row])),
                normalize_text(str(artists[target_row])),
            )
            if target_key in used_tracks:
                continue
            chosen = (item, target_row, target_artists, target_key)
            break
        _save_cache(cache_path, cache)
        if chosen is None:
            continue
        item, target_row, target_artists, target_key = chosen
        pair_id = f"FINAL-FRESH-{len(candidates) + 1:03d}"
        pair = {
            "id": pair_id,
            "split": "final",
            "scene": seed["scene"],
            "query": {
                "title": str(titles[query_row]),
                "artist": str(artists[query_row]),
                "recording_mbid": mbid,
            },
            "target": {
                "title": str(titles[target_row]),
                "artist": str(artists[target_row]),
                "recording_mbid": item.get("recording_mbid"),
            },
            "evidence_mode": "independent-human-songs-like",
            "claim_status": "listenbrainz-session-similarity",
            "sources": [{
                "url": _source_url(mbid),
                "publisher": "ListenBrainz Labs",
                "published_at": None,
                "accessed_at": date.today().isoformat(),
                "source_class": "independent_human_listening_similarity_dataset",
                "excerpt": (
                    "The frozen session-based similar-recordings algorithm returned "
                    f"this target for the query with score {int(item.get('score', 0))}."
                ),
            }],
            "evidence_category": "category_a_human_songs_like",
            "deciding_primary": True,
            "category_reason": (
                "A query-conditioned recommendation derived from independent human "
                "listening sessions; not a sample, cover, remix, legal allegation, "
                "or same-artist relation."
            ),
            "evidence_subtype": "independent_human_songs_like",
            "source_family": "listenbrainz_session_similarity",
            "catalog_tier": seed["catalog_tier"],
            "source_score": int(item.get("score", 0)),
            "source_algorithm": LISTENBRAINZ_ALGORITHM,
        }
        candidates.append(pair)
        used_artists |= seed_artists | target_artists
        used_tracks.add((
            normalize_text(str(titles[query_row])),
            normalize_text(str(artists[query_row])),
        ))
        used_tracks.add(target_key)

    final = _round_robin(candidates, per_scene=per_scene)
    scene_counts = Counter(pair["scene"] for pair in final)
    if len(final) < minimum_final:
        raise RuntimeError(
            f"Only {len(final)} fresh pairs resolved; need {minimum_final}. "
            f"Scenes: {dict(scene_counts)}"
        )
    if len(scene_counts) < 15:
        raise RuntimeError(f"Only {len(scene_counts)} fresh scenes resolved")

    document = {
        "schema_version": 6,
        "benchmark_id": "soundalike-independent-final-v6",
        "benchmark_version": "6.0.0",
        "created_at": _now(),
        "frozen_at": _now(),
        "license_note": (
            "Pair metadata and citations are factual benchmark records. "
            "ListenBrainz data is published by MetaBrainz under its public data terms."
        ),
        "purpose": (
            "Opened v5 pairs are DEV diagnostics only. Fresh FINAL labels are "
            "independent of Music4All-Onion collaborative training."
        ),
        "source_policy": {
            "collaborative_training_source": "Music4All-Onion (Last.fm extraction)",
            "final_source": "ListenBrainz session-based similar recordings",
            "source_independence": True,
            "previous_final_pairs_are_development_only": True,
            "samples_legal_covers_remixes_weak_listicles_excluded": True,
        },
        "split_policy": {
            "development": "all 107 previously opened v5 Category-A pairs",
            "final": (
                f"{len(final)} fresh ListenBrainz human songs-like pairs; "
                "artist-disjoint from DEV and from one another"
            ),
        },
        "metric_policy": {
            "primary": "mean(NDCG@10, MRR, Recall@10)",
            "candidate_recall_at": [100, 500, 1000],
            "success": {
                "minimum_relative_primary_gain": 0.20,
                "minimum_absolute_primary_gain": 0.01,
                "paired_bootstrap_ci95_low_must_exceed": 0.0,
                "minimum_improved_pairs": 10,
                "maximum_scene_relative_regression": -0.10,
                "recall_at_10_must_not_regress": True,
                "mrr_must_not_regress": True,
                "minimum_direct_top5_passes": 16,
            },
        },
        "prohibited_training_features": [
            "ListenBrainz similarity for any benchmark query or target",
            "fresh FINAL labels, target artists, or pair-specific boosts",
            "global Wikipedia or notability priors",
        ],
        "counts": {
            "development": len(development),
            "final": len(final),
            "total": len(development) + len(final),
            "final_scenes": len(scene_counts),
        },
        "pairs": development + final,
    }
    canonical = json.dumps(
        document["pairs"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    document["pairs_sha256"] = hashlib.sha256(canonical).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return document


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-final", type=int, default=80)
    parser.add_argument("--per-scene", type=int, default=6)
    args = parser.parse_args(argv)
    result = build_benchmark(
        args.index,
        args.prior,
        args.cache,
        args.output,
        minimum_final=args.minimum_final,
        per_scene=args.per_scene,
    )
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
